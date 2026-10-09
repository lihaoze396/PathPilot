#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import csv
import hashlib
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import solcx
import random
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from engine.plugin_interfaces import OnTheFlyAnalysis
except ImportError:
    from fuzzer.engine.plugin_interfaces import OnTheFlyAnalysis

from fuzzer.guidance.path_plan import PathPlan, load_path_plan
from fuzzer.guidance.path_progress import PathProgressTracker
from fuzzer.runtime.ksi_runtime import KSIRuntime
from fuzzer.agent.agent_feedback import AgentFeedbackBuilder
from fuzzer.guidance.gei_side_path import GEISidePathRecorder
from fuzzer.guidance.side_path_plan_exporter import SidePathPlanExporter
from fuzzer.guidance.soft_guidance import SoftPathGuidance, SoftGuidanceAnalysis
from fuzzer.agent.target_focus_runtime import TFIRuntime
from fuzzer.agent.target_focus_agent_runtime import TFIAgentRuntime
from fuzzer.agent.target_focus_soft_injection import TFISoftInjectionBuffer
from fuzzer.guidance.checkpoint_guidance import CheckpointGuidedRuntime
from fuzzer.agent.agent_planner import AgentPlanner
from fuzzer.agent.llm_provider import LLMProvider, LLMProviderConfig, LLMProviderError
from fuzzer.guidance.guided_sequence import RepairDiagnosisBuilder, RepairSequenceCandidateBuilder
from fuzzer.guidance.guided_sequence_pilot import RepairSequencePilot, RepairSequencePilotAnalysis
from fuzzer.runtime.phase_cost import (
    build_phase_summary,
    record_phase_event,
    record_phase_span,
    start_resource_sampler,
    stop_resource_sampler,
)
from fuzzer.runtime.local_execution_timing import refresh_local_execution_timing_results, time_local_execution
from fuzzer.runtime.stage_runtime import StageRuntime, StageRuntimeConfig
from fuzzer.runtime.rq1_phase import (
    RQ1_CONFIGS,
    ensure_rq1_phase_state,
    normalize_rq1_config,
    rq1_current_phase,
    rq1_budget_for_config,
    rq1_phase_allows_feature,
)
from fuzzer.runtime.runtime_timing import finalize_timing, refresh_timing_results, start_experiment_timing
from eth_utils import encode_hex, to_canonical_address
from z3 import Solver

from evm import InstrumentedEVM
from detectors import DetectorExecutor
from engine import EvolutionaryFuzzingEngine
from engine.components import Generator, Individual, Population
from engine.analysis import SymbolicTaintAnalyzer
from engine.analysis import ExecutionTraceAnalyzer
from engine.environment import FuzzingEnvironment
from engine.operators import LinearRankingSelection
from engine.operators import DataDependencyLinearRankingSelection
from engine.operators import Crossover
from engine.operators import DataDependencyCrossover
from engine.operators import Mutation
from engine.fitness import fitness_function
from fuzzer.utils.transaction_seq_utils import check_cross_init, gen_trans, init_func

from fuzzer.utils import settings
from utils.source_map import SourceMap
from utils.utils import initialize_logger, compile, get_interface_from_abi, get_pcs_and_jumpis, \
    get_function_signature_mapping, get_function_state_mutability_from_abi
from utils.control_flow_graph import ControlFlowGraph


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _listify(value: Any) -> list:
    return list(value) if isinstance(value, list) else []


def _load_jsonl_dicts(path: Any, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    jsonl_path = _text(path).strip()
    if not jsonl_path or not os.path.exists(jsonl_path):
        return rows
    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if limit > 0 and len(rows) > limit:
        return rows[-limit:]
    return rows


def _compact_llm_event(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
    event = {
        "stage": _text(item.get("stage") or item.get("phase")).strip() or "unknown",
        "duration_sec": round(float(item.get("duration_sec") or 0.0), 6),
        "success": not bool(_text(item.get("error")).strip()),
        "error": _text(item.get("error")).strip() or None,
        "fallback_used": bool(item.get("fallback_used")),
    }
    for key in ("parent_candidate_id", "repair_candidate_id", "target_missing_checkpoint", "runtime_pattern_family"):
        value = item.get(key)
        if value not in (None, ""):
            event[key] = value
    if metadata:
        event["metadata"] = dict(metadata)
    if usage:
        event["usage"] = dict(usage)
    for key in ("model", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "cost_estimation_mode"):
        value = item.get(key)
        if value in (None, "") and isinstance(metadata, dict):
            value = metadata.get(key)
        if value not in (None, ""):
            event[key] = value
    return event


def _checkpoint_llm_events(checkpoint_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(checkpoint_summary, dict):
        return []
    events: List[Dict[str, Any]] = [
        dict(item)
        for item in _listify(checkpoint_summary.get("llm_interaction_events"))
        if isinstance(item, dict)
    ]
    paths = checkpoint_summary.get("paths") if isinstance(checkpoint_summary.get("paths"), dict) else {}
    failure_repair_rows = _load_jsonl_dicts(paths.get("llm_failure_repair_interactions"), limit=256)
    seen = set()
    merged: List[Dict[str, Any]] = []
    for item in events + [_compact_llm_event(row) for row in failure_repair_rows]:
        key = json.dumps(
            {
                "stage": _text(item.get("stage") or item.get("phase")).strip(),
                "duration_sec": round(float(item.get("duration_sec") or 0.0), 6),
                "parent_candidate_id": _text(item.get("parent_candidate_id")).strip(),
                "repair_candidate_id": _text(item.get("repair_candidate_id")).strip(),
                "target_missing_checkpoint": _text(item.get("target_missing_checkpoint")).strip(),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[-256:]


def _dedup_phase_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        key = json.dumps(
            {
                "phase": _text(item.get("phase") or item.get("stage")).strip(),
                "started_at": round(float(item.get("started_at") or 0.0), 6),
                "ended_at": round(float(item.get("ended_at") or 0.0), 6),
                "duration_sec": round(float(item.get("duration_sec") or 0.0), 6),
                "parent_candidate_id": _text(item.get("parent_candidate_id")).strip(),
                "repair_candidate_id": _text(item.get("repair_candidate_id")).strip(),
                "state_obligation_id": _text(item.get("state_obligation_id")).strip(),
                "repair_scope_id": _text(item.get("repair_scope_id")).strip(),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output[-512:]


def _phase_cost_summary_from_events(merged_llm_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    phase_buckets: Dict[str, Dict[str, Any]] = {}
    for item in _dedup_phase_events(merged_llm_events):
        phase = _text(item.get("phase") or item.get("stage")).strip() or "unknown"
        bucket = phase_buckets.setdefault(
            phase,
            {
                "phase": phase,
                "event_count": 0,
                "success_count": 0,
                "error_count": 0,
                "wall_time_sec": 0.0,
                "llm_time_sec": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "models": set(),
            },
        )
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
        if not usage and isinstance(metadata.get("usage"), dict):
            usage = metadata.get("usage")
        bucket["event_count"] += 1
        bucket["success_count"] += int(bool(item.get("success", not bool(_text(item.get("error")).strip()))))
        bucket["error_count"] += int(bool(_text(item.get("error")).strip()))
        duration = float(item.get("duration_sec") or 0.0)
        bucket["wall_time_sec"] += duration
        bucket["llm_time_sec"] += duration
        bucket["prompt_tokens"] += int(item.get("prompt_tokens", usage.get("prompt_tokens", 0)) or 0)
        bucket["completion_tokens"] += int(item.get("completion_tokens", usage.get("completion_tokens", 0)) or 0)
        bucket["total_tokens"] += int(item.get("total_tokens", usage.get("total_tokens", 0)) or 0)
        bucket["cost_usd"] += float(item.get("cost_usd", metadata.get("cost_usd", 0.0)) or 0.0)
        model = _text(item.get("model") or metadata.get("model")).strip()
        if model:
            bucket["models"].add(model)
    phases: Dict[str, Any] = {}
    for phase, bucket in phase_buckets.items():
        phases[phase] = {
            "phase": phase,
            "event_count": int(bucket["event_count"]),
            "success_count": int(bucket["success_count"]),
            "error_count": int(bucket["error_count"]),
            "wall_time_sec": round(float(bucket["wall_time_sec"]), 6),
            "llm_time_sec": round(float(bucket["llm_time_sec"]), 6),
            "prompt_tokens": int(bucket["prompt_tokens"]),
            "completion_tokens": int(bucket["completion_tokens"]),
            "total_tokens": int(bucket["total_tokens"]),
            "cost_usd": round(float(bucket["cost_usd"]), 8),
            "models": sorted(bucket["models"]),
        }
    return {
        "enabled": True,
        "phase_count": len(phases),
        "events_total": sum(int(item["event_count"]) for item in phases.values()),
        "phases": phases,
    }


def _clean_list(value: Any) -> list:
    output = []
    if not isinstance(value, list):
        return output
    for item in value:
        text = _text(item).strip()
        if text:
            output.append(text)
    return output


_RQ1_GEI_RESTRICTED_SENDER_AUXILIARY_ACCOUNTS = [
    "0x1000000000000000000000000000000000000011",
    "0x1000000000000000000000000000000000000022",
]


def _dedup_addresses(values: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in values or []:
        text = _text(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(text)
    return output


HEX_BYTECODE_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]*$")
LINK_PLACEHOLDER_RE = re.compile(r"__(?:\$[0-9a-fA-F]{34}\$|[A-Za-z0-9_./:-]{1,36})__")


def _is_hex_bytecode(value: Any) -> bool:
    text = _text(value).strip()
    return bool(text) and bool(HEX_BYTECODE_RE.fullmatch(text))


def _has_link_references(contract_payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(contract_payload, dict):
        return False
    evm = contract_payload.get("evm") if isinstance(contract_payload.get("evm"), dict) else {}
    for key in ("bytecode", "deployedBytecode"):
        bytecode = evm.get(key) if isinstance(evm.get(key), dict) else {}
        refs = bytecode.get("linkReferences")
        if isinstance(refs, dict) and refs:
            return True
    return False


def _link_reference_items(contract_payload: Dict[str, Any]) -> List[Tuple[str, str, str, int, int]]:
    output: List[Tuple[str, str, str, int, int]] = []
    evm = contract_payload.get("evm") if isinstance(contract_payload.get("evm"), dict) else {}
    for bytecode_kind in ("bytecode", "deployedBytecode"):
        bytecode = evm.get(bytecode_kind) if isinstance(evm.get(bytecode_kind), dict) else {}
        refs = bytecode.get("linkReferences") if isinstance(bytecode.get("linkReferences"), dict) else {}
        for source_name, libraries in refs.items():
            if not isinstance(libraries, dict):
                continue
            for library_name, locations in libraries.items():
                if not isinstance(locations, list):
                    continue
                for location in locations:
                    if not isinstance(location, dict):
                        continue
                    try:
                        output.append(
                            (
                                bytecode_kind,
                                str(source_name),
                                str(library_name),
                                int(location.get("start")),
                                int(location.get("length")),
                            )
                        )
                    except Exception:
                        continue
    return output


def _deterministic_library_address(source_name: str, library_name: str) -> str:
    digest = hashlib.sha256(f"{source_name}:{library_name}".encode("utf-8")).hexdigest()
    return "0x" + digest[-40:]


def _library_candidates(whole_compile_info: Optional[Dict[str, Any]], library_name: str) -> List[Tuple[str, Dict[str, Any]]]:
    if not isinstance(whole_compile_info, dict):
        return []
    wanted = library_name.lower()
    exact: List[Tuple[str, Dict[str, Any]]] = []
    suffix: List[Tuple[str, Dict[str, Any]]] = []
    for name, payload in whole_compile_info.items():
        if not isinstance(payload, dict):
            continue
        lower = str(name).lower()
        if lower == wanted:
            exact.append((str(name), payload))
        elif lower.endswith(":" + wanted) or lower.endswith("." + wanted) or lower.endswith(wanted):
            suffix.append((str(name), payload))
    return exact + suffix


def _flatten_compile_contracts(compiler_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    contracts = compiler_output.get("contracts") if isinstance(compiler_output, dict) else {}
    if not isinstance(contracts, dict):
        return output
    for source_key, per_source in contracts.items():
        if not isinstance(per_source, dict):
            continue
        for contract_name, payload in per_source.items():
            if not isinstance(payload, dict):
                continue
            output.setdefault(str(contract_name), payload)
            output.setdefault(f"{source_key}:{contract_name}", payload)
    return output


def _apply_library_address_to_bytecode(
    bytecode: str,
    refs: Sequence[Tuple[str, str, str, int, int]],
    source_name: str,
    library_name: str,
    address: str,
    placeholder_addresses: Optional[Dict[str, str]] = None,
    single_library_fallback: bool = False,
) -> str:
    text = _text(bytecode).strip()
    has_prefix = text.startswith("0x")
    raw = text[2:] if has_prefix else text
    replacement = address.replace("0x", "").lower().rjust(40, "0")
    replaced = False
    for _, ref_source, ref_library, start, length in sorted(refs, key=lambda item: item[3], reverse=True):
        if ref_source != source_name or ref_library != library_name:
            continue
        if length != 20:
            continue
        begin = start * 2
        end = begin + length * 2
        if begin < 0 or end > len(raw):
            continue
        raw = raw[:begin] + replacement + raw[end:]
        replaced = True
    if not replaced:
        placeholder_addresses = placeholder_addresses or {}
        for placeholder, mapped_address in sorted(placeholder_addresses.items(), key=lambda item: len(item[0]), reverse=True):
            mapped_replacement = mapped_address.replace("0x", "").lower().rjust(40, "0")
            raw = raw.replace(placeholder, mapped_replacement)
        if single_library_fallback and LINK_PLACEHOLDER_RE.search(raw):
            raw = LINK_PLACEHOLDER_RE.sub(replacement, raw)
    return ("0x" + raw) if has_prefix else raw


def _placeholder_address_map(
    linked: Dict[str, Any],
    refs: Sequence[Tuple[str, str, str, int, int]],
    deployed_addresses: Dict[Tuple[str, str], str],
) -> Dict[str, str]:
    output: Dict[str, str] = {}
    evm = linked.get("evm") if isinstance(linked.get("evm"), dict) else {}
    for bytecode_kind in ("bytecode", "deployedBytecode"):
        bytecode = _text((evm.get(bytecode_kind) or {}).get("object")).strip()
        raw = bytecode[2:] if bytecode.startswith("0x") else bytecode
        if not raw:
            continue
        for kind, source_name, library_name, start, length in refs:
            if kind != bytecode_kind or length != 20:
                continue
            address = deployed_addresses.get((source_name, library_name))
            if not address:
                continue
            begin = start * 2
            end = begin + length * 2
            if begin < 0 or end > len(raw):
                continue
            placeholder = raw[begin:end]
            if LINK_PLACEHOLDER_RE.fullmatch(placeholder):
                output[placeholder] = address
    return output


def _link_contract_libraries(
    contract_name: str,
    contract_payload: Dict[str, Any],
    whole_compile_info: Optional[Dict[str, Any]],
    instrumented_evm: InstrumentedEVM,
    logger_obj: Any,
) -> Dict[str, Any]:
    refs = _link_reference_items(contract_payload)
    if not refs:
        return contract_payload
    linked = dict(contract_payload)
    linked["evm"] = dict(contract_payload.get("evm") or {})
    for key in ("bytecode", "deployedBytecode"):
        linked["evm"][key] = dict((contract_payload.get("evm") or {}).get(key) or {})

    deployed_addresses: Dict[Tuple[str, str], str] = {}
    for _, source_name, library_name, _, _ in refs:
        pair = (source_name, library_name)
        if pair in deployed_addresses:
            continue
        address = ""
        for candidate_name, candidate_payload in _library_candidates(whole_compile_info, library_name):
            bytecode = _text(((candidate_payload.get("evm") or {}).get("bytecode") or {}).get("object")).strip()
            runtime = _text(((candidate_payload.get("evm") or {}).get("deployedBytecode") or {}).get("object")).strip()
            if not bytecode or not runtime or not _is_hex_bytecode(bytecode) or not _is_hex_bytecode(runtime):
                continue
            try:
                address = _deterministic_library_address(source_name, library_name)
                runtime_raw = runtime[2:] if runtime.startswith("0x") else runtime
                instrumented_evm.create_fake_account(address, code=bytes.fromhex(runtime_raw))
                logger_obj.info("Linked library %s for %s at %s", candidate_name, contract_name, address)
                break
            except Exception as exc:
                logger_obj.warning("Library code materialization failed for %s used by %s: %s", candidate_name, contract_name, exc)
        if not address:
            logger_obj.error("No deployable library bytecode found for %s used by %s", library_name, contract_name)
            continue
        deployed_addresses[pair] = address

    placeholder_addresses = _placeholder_address_map(linked, refs, deployed_addresses)
    single_library_fallback = len(set(deployed_addresses.values())) == 1
    for bytecode_kind in ("bytecode", "deployedBytecode"):
        bytecode = _text(linked["evm"].get(bytecode_kind, {}).get("object")).strip()
        if not bytecode:
            continue
        kind_refs = [item for item in refs if item[0] == bytecode_kind]
        for source_name, library_name in deployed_addresses:
            bytecode = _apply_library_address_to_bytecode(
                bytecode,
                kind_refs,
                source_name,
                library_name,
                deployed_addresses[(source_name, library_name)],
                placeholder_addresses=placeholder_addresses,
                single_library_fallback=single_library_fallback,
            )
        linked["evm"][bytecode_kind]["object"] = bytecode
    return linked


def _validate_contract_bytecode(contract_name: str, contract_payload: Dict[str, Any]) -> Optional[str]:
    evm = contract_payload.get("evm") if isinstance(contract_payload.get("evm"), dict) else {}
    deployment = _text((evm.get("bytecode") or {}).get("object")).strip()
    runtime = _text((evm.get("deployedBytecode") or {}).get("object")).strip()
    if not deployment or not runtime:
        return f"missing bytecode for {contract_name}"
    if not _is_hex_bytecode(deployment):
        return f"non-hex deployment bytecode for {contract_name}; likely unresolved library placeholder"
    if not _is_hex_bytecode(runtime):
        return f"non-hex runtime bytecode for {contract_name}; likely unresolved library placeholder"
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_json_dump(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return path


def _write_deployment_diagnostic(
    args: Any,
    contract_name: str,
    phase: str,
    diagnostic: Optional[Dict[str, Any]],
    result_error: Any = None,
) -> str:
    run_output_dir = getattr(args, "run_output_dir", None) or _resolve_run_output_dir(getattr(args, "results", None))
    payload = dict(diagnostic or {})
    payload.update(
        {
            "diagnostic_version": payload.get("diagnostic_version") or "deployment_diagnostics_v1",
            "contract_name": contract_name,
            "source_path": getattr(args, "source", ""),
            "solc_version_used": getattr(args, "solc_version", ""),
            "evm_version": getattr(args, "evm_version", "") or getattr(settings, "EVM_VERSION", ""),
            "phase": phase,
            "deployer_account": payload.get("creator", ""),
            "constructor_schema": list(getattr(args, "constructor_args", None) or []),
            "constructor_arguments_materialized": bool(payload.get("constructor_arguments_materialized", False)),
            "constructor_argument_sources": payload.get("constructor_argument_sources", []),
            "dependency_addresses_materialized": payload.get("dependency_addresses_materialized", []),
            "deployment_tx_value": payload.get("deployment_tx_value", 0),
            "deployment_bytecode_length": payload.get("deployment_bytecode_length", 0),
            "deployment_receipt_status": payload.get("deployment_receipt_status", ""),
            "deployment_reverted": payload.get("deployment_reverted", None),
            "exception_signature": payload.get("exception_signature", repr(result_error) if result_error is not None else ""),
            "settings_trans_info_keys": sorted(str(key) for key in getattr(settings, "TRANS_INFO", {}).keys())[:128],
            "settings_deployed_contract_address_keys": sorted(
                str(key) for key in getattr(settings, "DEPLOYED_CONTRACT_ADDRESS", {}).keys()
            )[:128],
        }
    )
    if result_error is not None:
        payload["result_error_repr_from_callsite"] = repr(result_error)
        payload["result_error_text_from_callsite"] = str(result_error)
    path = os.path.join(run_output_dir, "deployment_diagnostics.json")
    try:
        return _safe_json_dump(path, payload)
    except Exception:
        return ""


def _path_within_dir(path: str, root_dir: str) -> bool:
    if not path or not root_dir:
        return False
    try:
        path_abs = os.path.abspath(path)
        root_abs = os.path.abspath(root_dir)
        common = os.path.commonpath([path_abs, root_abs])
        return common == root_abs
    except Exception:
        return False


def _is_current_run_artifact(path: Optional[str], output_dir: str, run_start_time: Optional[float]) -> bool:
    if not path or not output_dir or not os.path.exists(path):
        return False
    if not _path_within_dir(path, output_dir):
        return False
    if run_start_time in (None, ""):
        return True
    try:
        return float(os.path.getmtime(path)) >= float(run_start_time) - 1.0
    except Exception:
        return False


def _load_audit_target_spec_from_args(args: Any) -> Dict[str, Any]:
    spec_path = _text(getattr(args, "audit_target_spec_json", None)).strip()
    if spec_path and os.path.exists(spec_path):
        try:
            with open(spec_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {
        "oracle_family": _text(getattr(args, "audit_oracle_family", "")).strip(),
        "normalized_type": _text(getattr(args, "audit_normalized_type", "")).strip(),
        "finding_title": _text(getattr(args, "audit_finding_title", "")).strip(),
        "affected_function": _text(getattr(args, "audit_affected_function", "")).strip(),
        "affected_variable": _text(getattr(args, "audit_affected_variable", "")).strip(),
        "l1_condition_text": _text(getattr(args, "audit_l1_condition", "")).strip(),
        "l2_condition_text": _text(getattr(args, "audit_l2_condition", "")).strip(),
        "l3_condition_text": _text(getattr(args, "audit_l3_condition", "")).strip(),
        "required_trace_evidence": _text(getattr(args, "audit_required_trace_evidence", "")).strip(),
    }


def _function_name_from_signature(signature: str) -> str:
    text = _text(signature).strip()
    if not text:
        return ""
    if "(" in text:
        return text.split("(", 1)[0].strip()
    return text


def _abi_function_summaries(abi: Any) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    if not isinstance(abi, list):
        return summaries
    for item in abi:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        name = _text(item.get("name")).strip()
        if not name:
            continue
        inputs = []
        for argument in item.get("inputs") or []:
            if not isinstance(argument, dict):
                continue
            inputs.append(
                {
                    "name": _text(argument.get("name")).strip(),
                    "type": _text(argument.get("type")).strip(),
                }
            )
        summaries.append(
            {
                "name": name,
                "signature": "{}({})".format(
                    name,
                    ",".join(_text(argument.get("type")).strip() for argument in item.get("inputs") or []),
                ),
                "state_mutability": _text(item.get("stateMutability")).strip() or "nonpayable",
                "inputs": inputs,
            }
        )
    return summaries


def _read_source_excerpt(source_path: str, max_chars: int = 4000) -> str:
    if not source_path or not os.path.exists(source_path):
        return ""
    try:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(max_chars)
    except Exception:
        return ""


def _read_first_nonempty_line(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                value = str(raw_line or "").strip()
                if value:
                    return value
    except Exception:
        return ""
    return ""


def _resolve_llm_api_key_settings(args: Any) -> None:
    if not hasattr(args, "llm_api_key_loaded"):
        setattr(args, "llm_api_key_loaded", False)
    if not hasattr(args, "llm_api_key_source"):
        setattr(args, "llm_api_key_source", None)

    inline_key = _text(getattr(args, "llm_api_key", None)).strip()
    if inline_key:
        args.llm_api_key_loaded = True
        args.llm_api_key_source = "cli"
        return

    key_file = _text(getattr(args, "llm_api_key_file", None)).strip()
    if key_file:
        loaded_key = _read_first_nonempty_line(key_file)
        if loaded_key:
            args.llm_api_key = loaded_key
            args.llm_api_key_loaded = True
            args.llm_api_key_source = "file"
            return

    env_key = _text(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")).strip()
    if env_key:
        args.llm_api_key_loaded = True
        args.llm_api_key_source = "env"


def _compact_static_summary_for_agent(static_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(static_summary, dict):
        return {}
    return {
        "loaded": bool(static_summary.get("loaded")),
        "error": static_summary.get("error"),
        "contract_name": static_summary.get("contract_name"),
        "function_count": int(static_summary.get("function_count") or 0),
        "functions": list(static_summary.get("functions") or [])[:20],
        "privileged_functions": list(static_summary.get("privileged_functions") or [])[:12],
        "dangerous_operations": list(static_summary.get("dangerous_operations") or [])[:10],
        "storage_dependencies": list(static_summary.get("storage_dependencies") or [])[:12],
    }


def _detect_dangerous_source_signals(source_text: str) -> Dict[str, List[str]]:
    lowered = str(source_text or "").lower()
    signal_map = {
        "external_call": ["call(", ".call(", "call{", "transfer(", ".transfer(", ".send("],
        "delegatecall": ["delegatecall"],
        "selfdestruct": ["selfdestruct", "suicide("],
        "mint": ["mint"],
        "burn": ["burn"],
        "blacklist": ["blacklist"],
        "fee": ["fee", "tax"],
        "owner": ["onlyowner", "owner", "admin"],
    }
    matched: Dict[str, List[str]] = {}
    for label, patterns in signal_map.items():
        hits = [pattern for pattern in patterns if pattern in lowered]
        if hits:
            matched[label] = hits
    return matched


def _build_slither_static_summary(
    source_path: str,
    contract_name: str,
    solc_dir: Optional[str],
    solc_version: Optional[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "loaded": False,
        "error": None,
        "function_count": 0,
        "functions": [],
        "privileged_functions": [],
        "dangerous_operations": [],
        "storage_dependencies": [],
    }
    if not source_path or not os.path.exists(source_path):
        summary["error"] = "missing_source"
        return summary
    try:
        from slither import Slither
    except Exception as exc:
        summary["error"] = "slither_import_failed: {}".format(str(exc))
        return summary

    slither_kwargs: Dict[str, Any] = {}
    solc_path = _resolve_solc_executable(solc_dir, solc_version)
    if solc_path:
        slither_kwargs["solc"] = solc_path
    slither_target = source_path
    slither_cwd = None
    support = _load_forge_compile_support()
    if support is not None and hasattr(support, "slither_compilation_context"):
        try:
            if hasattr(support, "slither_compilation_contexts"):
                contexts = support.slither_compilation_contexts(
                    source_path,
                    solc_path=solc_path,
                    evm_version=getattr(settings, "EVM_VERSION", None),
                )
            else:
                contexts = [
                    support.slither_compilation_context(
                        source_path,
                        solc_path=solc_path,
                        evm_version=getattr(settings, "EVM_VERSION", None),
                    )
                ]
        except Exception:
            contexts = []
    else:
        contexts = []

    old_cwd = os.getcwd()
    try:
        slither = None
        if contexts:
            last_error = None
            for context in contexts:
                try:
                    slither_target = context["target"]
                    slither_kwargs = dict(context["kwargs"])
                    slither_cwd = context["cwd"]
                    if slither_cwd:
                        os.chdir(slither_cwd)
                    slither = Slither(slither_target, **slither_kwargs)
                    break
                except Exception as exc:
                    last_error = exc
                finally:
                    if slither_cwd:
                        os.chdir(old_cwd)
            if slither is None and last_error is not None:
                raise last_error
        else:
            if slither_cwd:
                os.chdir(slither_cwd)
            slither = Slither(slither_target, **slither_kwargs)
    except Exception as exc:
        summary["error"] = "slither_failed: {}".format(str(exc))
        return summary
    finally:
        if slither_cwd:
            os.chdir(old_cwd)

    contract = None
    try:
        contracts = slither.get_contract_from_name(contract_name)
        if contracts:
            contract = contracts[0]
    except Exception:
        contract = None
    if contract is None:
        try:
            contract = slither.contracts[0] if slither.contracts else None
        except Exception:
            contract = None
    if contract is None:
        summary["error"] = "slither_contract_not_found"
        return summary

    privileged_functions: List[str] = []
    dangerous_operations: List[Dict[str, Any]] = []
    storage_dependencies: List[Dict[str, Any]] = []
    function_rows: List[Dict[str, Any]] = []
    for function in getattr(contract, "functions", []) or []:
        if getattr(function, "is_constructor", False) or getattr(function, "is_fallback", False):
            continue
        name = _text(getattr(function, "name", None)).strip()
        if not name:
            continue
        modifiers = [_text(getattr(modifier, "name", None)).strip() for modifier in getattr(function, "modifiers", []) or []]
        modifiers = [item for item in modifiers if item]
        reads = [_text(getattr(variable, "name", None)).strip() for variable in getattr(function, "state_variables_read", []) or []]
        reads = [item for item in reads if item]
        writes = [_text(getattr(variable, "name", None)).strip() for variable in getattr(function, "state_variables_written", []) or []]
        writes = [item for item in writes if item]

        external_calls: List[str] = []
        for call_entry in getattr(function, "high_level_calls", []) or []:
            try:
                target_contract = _text(getattr(call_entry[0], "name", None)).strip()
                target_function = _text(getattr(call_entry[1], "name", None)).strip()
                label = ".".join(item for item in (target_contract, target_function) if item)
                if label:
                    external_calls.append(label)
            except Exception:
                continue

        low_level_calls = []
        for call_entry in getattr(function, "low_level_calls", []) or []:
            try:
                low_level_calls.append(_text(call_entry[0]).strip() or _text(call_entry).strip())
            except Exception:
                continue

        local_dangerous = []
        lowered_name = name.lower()
        if any(token in lowered_name for token in ("owner", "admin", "govern", "minter", "blacklist", "pause", "upgrade")):
            privileged_functions.append(name)
        if external_calls or low_level_calls:
            local_dangerous.append("external_or_low_level_call")
        if any("delegatecall" in item.lower() for item in low_level_calls):
            local_dangerous.append("delegatecall")
        if any("callcode" in item.lower() for item in low_level_calls):
            local_dangerous.append("callcode")
        if any(token in lowered_name for token in ("mint", "burn", "blacklist", "fee", "tax", "transferownership", "setowner")):
            local_dangerous.append("sensitive_function_name")
        if local_dangerous:
            dangerous_operations.append(
                {
                    "function": name,
                    "signals": sorted(set(local_dangerous)),
                    "external_calls": external_calls[:8],
                    "low_level_calls": low_level_calls[:8],
                }
            )
        if reads or writes:
            storage_dependencies.append(
                {
                    "function": name,
                    "reads": reads[:16],
                    "writes": writes[:16],
                }
            )
        function_rows.append(
            {
                "name": name,
                "visibility": _text(getattr(function, "visibility", None)).strip() or "public",
                "modifiers": modifiers[:8],
                "reads": reads[:12],
                "writes": writes[:12],
                "external_calls": external_calls[:8],
                "low_level_calls": low_level_calls[:8],
            }
        )

    summary.update(
        {
            "loaded": True,
            "contract_name": _text(getattr(contract, "name", None)).strip() or contract_name,
            "function_count": len(function_rows),
            "functions": function_rows[:40],
            "privileged_functions": sorted(set(privileged_functions))[:20],
            "dangerous_operations": dangerous_operations[:20],
            "storage_dependencies": storage_dependencies[:30],
        }
    )
    return summary


def _build_llm_provider_from_args(args: Any) -> Optional[LLMProvider]:
    try:
        return LLMProvider(
            LLMProviderConfig.from_sources(
                base_url=getattr(args, "llm_base_url", None),
                api_key=getattr(args, "llm_api_key", None),
                model=getattr(args, "llm_model", None),
                api_style=getattr(args, "llm_api_style", "auto"),
                timeout_sec=getattr(args, "llm_timeout_sec", 60),
                max_retries=getattr(args, "llm_max_retries", 2),
                temperature=getattr(args, "llm_temperature", 0.1),
            )
        )
    except Exception:
        return None


def _record_agent_llm_stats(
    agent_state: Dict[str, Any],
    *,
    llm_called: bool = False,
    llm_success: bool = False,
    error: str = "",
    fallback_used: bool = False,
) -> None:
    if not isinstance(agent_state, dict):
        return
    if llm_called:
        agent_state["llm_called"] = True
    if llm_success:
        agent_state["llm_success_count"] = int(agent_state.get("llm_success_count") or 0) + 1
    if error:
        agent_state["llm_error_count"] = int(agent_state.get("llm_error_count") or 0) + 1
        errors = agent_state.get("llm_errors")
        if not isinstance(errors, list):
            errors = []
        errors.append(error)
        agent_state["llm_errors"] = errors[-20:]
    if fallback_used:
        agent_state["fallback_used"] = True


def _default_agent_runtime_state(args: Any, run_output_dir: str) -> Dict[str, Any]:
    return {
        "enabled": bool(
            getattr(args, "enable_tfi_agent", False)
            or getattr(args, "enable_ksi_llm_feedback", False)
        ),
        "fresh_agent_run": bool(getattr(args, "fresh_agent_run", False)),
        "ignore_precomputed_paths": bool(getattr(args, "ignore_precomputed_paths", False)),
        "ignore_precomputed_tfi_artifacts": bool(getattr(args, "ignore_precomputed_tfi_artifacts", False)),
        "strict_llm_only": bool(getattr(args, "fresh_agent_run", False) and getattr(args, "agent_use_llm", False)),
        "run_start_time": getattr(args, "run_start_time", None),
        "run_output_dir": run_output_dir,
        "llm_called": False,
        "llm_success_count": 0,
        "llm_error_count": 0,
        "llm_errors": [],
        "llm_interaction_events": [],
        "llm_interaction_time_total_sec": 0.0,
        "agent_interaction_total_sec": 0.0,
        "agent_interaction_count": 0,
        "llm_model": getattr(args, "llm_model", None),
        "llm_api_key_loaded": bool(getattr(args, "llm_api_key_loaded", False)),
        "llm_api_key_source": getattr(args, "llm_api_key_source", None),
        "fallback_used": False,
        "precomputed_path_used": False,
        "precomputed_tfi_artifact_used": False,
        "precomputed_artifact_used": False,
        "agent_result_valid": False,
        "initial_hypothesis_count": 0,
        "latest_hypothesis_count": 0,
        "path_plan_source": None,
        "ksi_agent_called_count": 0,
        "ksi_feedback_llm_success_count": 0,
        "ksi_feedback_llm_error_count": 0,
        "tfi_agent_guidance_count": 0,
        "tfi_guidance_llm_success_count": 0,
        "tfi_guidance_llm_error_count": 0,
        "agent_revision_count": 0,
        "current_plan_payload": {"hypotheses": []},
        "current_plan_path": None,
        "function_names": [],
        "plan_dirty": False,
        "_env_ref": None,
    }


def _resolve_gei60_config(args: Any) -> Tuple[bool, int]:
    if args is None:
        return False, 0
    rq1_config = normalize_rq1_config(getattr(args, "rq1_config", ""))
    if rq1_config:
        budget = rq1_budget_for_config(rq1_config)
        return bool(int(budget.get("g", 0)) > 0), int(budget.get("g", 0))
    rq1_g_sec = _safe_int(getattr(args, "rq1_g_sec", 0), 0)
    if rq1_g_sec > 0:
        return True, rq1_g_sec
    legacy_warmup_sec = max(0, _safe_int(getattr(args, "pure_fuzz_warmup_sec", 0), 0))
    if legacy_warmup_sec > 0:
        return True, legacy_warmup_sec
    if not bool(getattr(args, "enable_gei60", False)):
        return False, 0
    gei60_sec = max(1, _safe_int(getattr(args, "gei60_sec", 60), 60))
    return True, gei60_sec


def _default_gei60_state(args: Any) -> Dict[str, Any]:
    enabled, gei60_sec = _resolve_gei60_config(args)
    return {
        "enabled": bool(enabled),
        "gei60_sec": gei60_sec,
        "warmup_sec": gei60_sec,
        "llm_enabled_after_warmup": bool(getattr(args, "agent_use_llm", False)),
        "fresh_agent_requested": bool(getattr(args, "fresh_agent_run", False)),
        "fresh_agent_initialized": False,
        "fresh_agent_initialized_at_fuzz_active_sec": None,
        "fresh_agent_initialization_source": None,
        "gei60_completed": False,
        "gei60_completed_at_fuzz_active_sec": None,
    }


def _sync_gei60_results(env: Any, state: Dict[str, Any]) -> None:
    results = getattr(env, "results", None)
    if isinstance(results, dict):
        payload = dict(state)
        results["gei60"] = payload
        results["pure_fuzz_warmup"] = payload


def _ensure_gei60_state(env: Any, args: Any = None) -> Dict[str, Any]:
    state = getattr(env, "gei60_state", None)
    if not isinstance(state, dict):
        state = getattr(env, "pure_fuzz_warmup_state", None)
    if not isinstance(state, dict):
        state = _default_gei60_state(args or getattr(env, "args", None))
    setattr(env, "gei60_state", state)
    setattr(env, "pure_fuzz_warmup_state", state)
    state.setdefault("enabled", False)
    state.setdefault("warmup_sec", 0)
    state.setdefault("gei60_sec", int(state.get("warmup_sec") or 0))
    state.setdefault("llm_enabled_after_warmup", bool(getattr(args or getattr(env, "args", None), "agent_use_llm", False)))
    state.setdefault("fresh_agent_requested", bool(getattr(args or getattr(env, "args", None), "fresh_agent_run", False)))
    state.setdefault("fresh_agent_initialized", False)
    state.setdefault("fresh_agent_initialized_at_fuzz_active_sec", None)
    state.setdefault("fresh_agent_initialization_source", None)
    state.setdefault("gei60_completed", bool(state.get("warmup_completed", False)))
    state.setdefault("gei60_completed_at_fuzz_active_sec", state.get("warmup_completed_at_fuzz_active_sec"))
    state["warmup_completed"] = bool(state.get("gei60_completed"))
    state["warmup_completed_at_fuzz_active_sec"] = state.get("gei60_completed_at_fuzz_active_sec")
    return state


def _record_agent_llm_timing(
    agent_state: Dict[str, Any],
    *,
    phase: str,
    started_at: Optional[float],
    ended_at: Optional[float],
    success: bool = False,
    error: str = "",
    used_fallback: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if not isinstance(agent_state, dict):
        return
    try:
        start_value = float(started_at) if started_at not in (None, "") else None
        end_value = float(ended_at) if ended_at not in (None, "") else None
    except Exception:
        start_value = None
        end_value = None
    duration = 0.0
    if start_value is not None and end_value is not None and end_value >= start_value:
        duration = float(end_value - start_value)
    events = agent_state.get("llm_interaction_events")
    if not isinstance(events, list):
        events = []
    event_payload = {
        "phase": _text(phase).strip() or "unknown",
        "event_kind": "llm",
        "started_at": start_value,
        "ended_at": end_value,
        "duration_sec": duration,
        "success": bool(success),
        "error": _text(error).strip() or None,
        "fallback_used": bool(used_fallback),
    }
    if isinstance(metadata, dict) and metadata:
        event_payload["metadata"] = dict(metadata)
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        if usage:
            event_payload["usage"] = dict(usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "cost_estimation_mode"):
            if key in metadata:
                event_payload[key] = metadata.get(key)
    events.append(event_payload)
    agent_state["llm_interaction_events"] = events[-256:]
    total_sec = float(agent_state.get("llm_interaction_time_total_sec") or 0.0) + float(duration)
    interaction_count = int(agent_state.get("agent_interaction_count") or 0) + 1
    agent_state["llm_interaction_time_total_sec"] = total_sec
    agent_state["agent_interaction_total_sec"] = total_sec
    agent_state["agent_interaction_count"] = interaction_count
    env_ref = agent_state.get("_env_ref")
    if env_ref is not None:
        try:
            record_phase_event(env_ref, event_payload)
        except Exception:
            pass


def _current_fuzz_active_elapsed_sec(env: Any) -> float:
    try:
        timing = refresh_timing_results(env)
    except Exception:
        timing = {}
    return _safe_float((timing or {}).get("current_fuzz_active_elapsed_sec"), 0.0)


def _runtime_oracle_events(runtime: Any, limit: int = 0) -> List[Dict[str, Any]]:
    return []


def _gei60_active(env: Any) -> bool:
    state = _ensure_gei60_state(env)
    if not bool(state.get("enabled")):
        return False
    gei60_sec = max(0.0, _safe_float(state.get("gei60_sec", state.get("warmup_sec")), 0.0))
    if gei60_sec <= 0.0:
        return False
    current_fuzz_active = _current_fuzz_active_elapsed_sec(env)
    if current_fuzz_active >= gei60_sec:
        if not bool(state.get("gei60_completed")):
            completed_at = round(current_fuzz_active, 6)
            state["gei60_completed"] = True
            state["gei60_completed_at_fuzz_active_sec"] = completed_at
            state["warmup_completed"] = True
            state["warmup_completed_at_fuzz_active_sec"] = completed_at
            _sync_gei60_results(env, state)
        return False
    return True


def _pure_fuzz_warmup_active(env: Any) -> bool:
    return _gei60_active(env)


def _ensure_pure_fuzz_warmup_state(env: Any, args: Any = None) -> Dict[str, Any]:
    return _ensure_gei60_state(env, args)


def _rq1_gei_sender_pool_restriction_enabled(args: Any) -> bool:
    if args is None:
        return False
    if not normalize_rq1_config(getattr(args, "rq1_config", "")):
        return False
    return int(getattr(args, "rq1_g_sec", 0) or 0) > 0


def _sync_rq1_gei_sender_pool_results(env: Any, state: Dict[str, Any]) -> None:
    setattr(env, "rq1_gei_feedback_pool_restricted", bool(state.get("feedback_pool_restricted")))
    results = getattr(env, "results", None)
    if isinstance(results, dict):
        results["rq1_gei_sender_pool"] = dict(state)


def _ensure_rq1_gei_sender_pool_state(env: Any, args: Any = None) -> Dict[str, Any]:
    state = getattr(env, "rq1_gei_sender_pool_state", None)
    if not isinstance(state, dict):
        state = {}
    state.setdefault("enabled", _rq1_gei_sender_pool_restriction_enabled(args or getattr(env, "args", None)))
    state.setdefault("applied", False)
    state.setdefault("restored", False)
    state.setdefault("full_accounts", [])
    state.setdefault("restricted_accounts", [])
    state.setdefault("feedback_pool_restricted", False)
    state.setdefault("applied_at_fuzz_active_sec", None)
    state.setdefault("restored_at_fuzz_active_sec", None)
    setattr(env, "rq1_gei_sender_pool_state", state)
    _sync_rq1_gei_sender_pool_results(env, state)
    return state


def _clear_generator_account_pools(generators: Sequence[Any]) -> None:
    for generator in generators or []:
        if generator is None:
            continue
        try:
            generator.accounts_pool = {}
        except Exception:
            pass


def _build_rq1_gei_restricted_sender_pool(instrumented_evm: Any, full_accounts: Sequence[Any]) -> List[str]:
    if instrumented_evm is None:
        return []
    accounts = _dedup_addresses(full_accounts)
    if not accounts:
        return []
    deployer = _text(accounts[0]).strip().lower()
    attacker_candidates = _dedup_addresses(
        list(getattr(settings, "ATTACKER_ACCOUNTS", []) or [])
        + [item for item in accounts if _text(item).strip().lower() in {_text(addr).strip().lower() for addr in list(getattr(settings, "ATTACKER_ACCOUNTS", []) or [])}]
    )
    for aux_address in _RQ1_GEI_RESTRICTED_SENDER_AUXILIARY_ACCOUNTS:
        try:
            if not instrumented_evm.has_account(aux_address):
                instrumented_evm.create_fake_account(aux_address)
        except Exception:
            pass
    candidate_pool = _dedup_addresses(accounts + list(_RQ1_GEI_RESTRICTED_SENDER_AUXILIARY_ACCOUNTS))
    restricted: List[str] = []
    for address in candidate_pool:
        lowered = _text(address).strip().lower()
        if not lowered or lowered == deployer:
            continue
        code_size = -1
        try:
            code_size = int(instrumented_evm._account_code_size(address))
        except Exception:
            code_size = -1
        if code_size > 0:
            continue
        restricted.append(_text(address).strip())
    restricted = _dedup_addresses(attacker_candidates + restricted)
    if not restricted:
        fallback = [item for item in accounts if _text(item).strip().lower() != deployer]
        restricted = _dedup_addresses(fallback[:2] or accounts[:1])
    return restricted[:3]


def _update_rq1_gei_sender_pool(env: Any, *, logger: Any = None) -> None:
    state = _ensure_rq1_gei_sender_pool_state(env, getattr(env, "args", None))
    if not bool(state.get("enabled")):
        return
    instrumented_evm = getattr(env, "instrumented_evm", None)
    account_ref = getattr(instrumented_evm, "accounts", None)
    if instrumented_evm is None or not isinstance(account_ref, list):
        return
    generators = list(getattr(env, "rq1_sender_pool_generators", []) or [])
    warmup_active = _gei60_active(env)
    if warmup_active and not bool(state.get("applied")):
        full_accounts = _dedup_addresses(list(account_ref))
        restricted_accounts = _build_rq1_gei_restricted_sender_pool(instrumented_evm, full_accounts)
        if restricted_accounts:
            state["full_accounts"] = list(full_accounts)
            state["restricted_accounts"] = list(restricted_accounts)
            account_ref[:] = list(restricted_accounts)
            _clear_generator_account_pools(generators)
            state["applied"] = True
            state["restored"] = False
            state["feedback_pool_restricted"] = True
            state["applied_at_fuzz_active_sec"] = round(_current_fuzz_active_elapsed_sec(env), 6)
            state["restored_at_fuzz_active_sec"] = None
            _sync_rq1_gei_sender_pool_results(env, state)
            if logger is not None:
                try:
                    logger.info(
                        "[RQ1_GEI_SENDER_POOL] status=restricted size=%s accounts=%s",
                        len(restricted_accounts),
                        ",".join(restricted_accounts),
                    )
                except Exception:
                    pass
        return
    if not warmup_active and bool(state.get("applied")) and not bool(state.get("restored")):
        restore_accounts = _dedup_addresses(list(state.get("full_accounts") or []) + list(account_ref))
        if restore_accounts:
            account_ref[:] = list(restore_accounts)
            _clear_generator_account_pools(generators)
        state["restored"] = True
        state["feedback_pool_restricted"] = False
        state["restored_at_fuzz_active_sec"] = round(_current_fuzz_active_elapsed_sec(env), 6)
        _sync_rq1_gei_sender_pool_results(env, state)
        if logger is not None:
            try:
                logger.info(
                    "[RQ1_GEI_SENDER_POOL] status=restored size=%s",
                    len(restore_accounts),
                )
            except Exception:
                pass
    elif not warmup_active and bool(state.get("feedback_pool_restricted")):
        state["feedback_pool_restricted"] = False
        _sync_rq1_gei_sender_pool_results(env, state)


def _rq1_runtime_account_snapshot(env: Any, instrumented_evm: Any) -> List[str]:
    state = _ensure_rq1_gei_sender_pool_state(env, getattr(env, "args", None))
    full_accounts = _dedup_addresses(state.get("full_accounts") or [])
    if full_accounts:
        return full_accounts
    return _dedup_addresses(list(getattr(instrumented_evm, "accounts", []) or []))


def _llm_runtime_allowed(env: Any, args: Any = None) -> bool:
    runtime_args = args or getattr(env, "args", None)
    if runtime_args is not None and not bool(getattr(runtime_args, "agent_use_llm", False)):
        return False
    if not rq1_phase_allows_feature(env, "llm", runtime_args):
        return False
    return not _gei60_active(env)


def _build_agent_init_prompt(
    contract_name: str,
    abi_functions: List[Dict[str, Any]],
    static_summary: Dict[str, Any],
    source_excerpt: str,
    max_hypotheses: int,
) -> str:
    mutable_functions = [
         {
            "name": _text(item.get("name")).strip(),
            "state_mutability": _text(item.get("state_mutability")).strip() or "nonpayable",
        }
        for item in abi_functions
        if isinstance(item, dict)
        and _text(item.get("name")).strip()
        and _text(item.get("state_mutability")).strip() not in ("view", "pure")
    ]
    allowed_functions = sorted(
        {
            _text(item.get("name")).strip()
            for item in abi_functions
            if isinstance(item, dict) and _text(item.get("name")).strip()
        }
    )
    compact_static = _compact_static_summary_for_agent(static_summary)
    dangerous_source_signals = _detect_dangerous_source_signals(source_excerpt)
    payload = {
        "contract_name": contract_name,
        "max_hypotheses": int(max_hypotheses),
        "allowed_functions": allowed_functions[:16],
        "mutable_functions": mutable_functions[:8],
        "privileged_functions": list(compact_static.get("privileged_functions") or [])[:6],
        "dangerous_operations": [
            {
                "function": _text(item.get("function")).strip(),
                "signals": _clean_list(item.get("signals"))[:3],
            }
            for item in list(compact_static.get("dangerous_operations") or [])[:4]
            if isinstance(item, dict)
        ],
        "storage_dependencies": [
            {
                "function": _text(item.get("function")).strip(),
                "reads": _clean_list(item.get("reads"))[:3],
                "writes": _clean_list(item.get("writes"))[:3],
            }
            for item in list(compact_static.get("storage_dependencies") or [])[:4]
            if isinstance(item, dict)
        ],
        "dangerous_source_signals": dangerous_source_signals,
        "source_excerpt": str(source_excerpt or "")[:300],
    }
    instructions = [
        "You are a smart-contract fuzzing hypothesis agent.",
        "Generate cold-start vulnerability path hypotheses only.",
        "Do not generate executable transactions.",
        "Do not invent function names outside allowed_functions.",
        "Prefer short, plausible paths that can be tested by a fuzzer.",
        "Output strict JSON only.",
        "Use exactly this schema:",
        '{ "hypotheses": [ { "hypothesis_id": "AGENT_H1", "vulnerability_type": "permission_or_state_path", "candidate_path": ["funcA","funcB"], "target_sink": "funcB", "expected_preconditions": ["..."], "expected_blockers": ["..."], "sender_hints": ["..."], "value_hints": ["..."], "argument_hints": ["..."], "state_hints": ["..."], "confidence": "high" } ] }',
        "",
        "Input JSON:",
        json.dumps(payload, ensure_ascii=True, indent=2),
    ]
    return "\n".join(instructions)


def _normalize_agent_init_plan(
    payload: Any,
    abi_functions: List[Dict[str, Any]],
    max_hypotheses: int,
) -> Dict[str, Any]:
    allowed_functions = {
        _text(item.get("name")).strip()
        for item in abi_functions
        if isinstance(item, dict) and _text(item.get("name")).strip()
    }
    normalized: List[Dict[str, Any]] = []
    seen = set()
    if isinstance(payload, dict):
        items = payload.get("hypotheses")
    else:
        items = None
    if not isinstance(items, list):
        return {"hypotheses": []}

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        raw_path = _clean_list(item.get("candidate_path")) or _clean_list(item.get("path"))
        path = [name for name in raw_path if name in allowed_functions]
        if not path:
            continue
        signature = tuple(path)
        if signature in seen:
            continue
        seen.add(signature)
        confidence = _text(item.get("confidence")).strip().lower()
        priority = {"high": 0.9, "medium": 0.7, "low": 0.5}.get(confidence, 0.6)
        sink = _text(item.get("target_sink")).strip() or _text(item.get("sink")).strip() or path[-1]
        if sink not in allowed_functions and sink != path[-1]:
            sink = path[-1]
        normalized.append(
            {
                "id": _text(item.get("hypothesis_id")).strip() or _text(item.get("id")).strip() or "AGENT_H{}".format(index),
                "hypothesis_id": _text(item.get("hypothesis_id")).strip() or _text(item.get("id")).strip() or "AGENT_H{}".format(index),
                "vulnerability_type": _text(item.get("vulnerability_type")).strip() or "fresh_agent_cold_start",
                "path": path,
                "sink": sink,
                "priority": priority,
                "expected_preconditions": _clean_list(item.get("expected_preconditions")),
                "expected_blockers": _clean_list(item.get("expected_blockers")),
                "sender_hints": _clean_list(item.get("sender_hints")),
                "value_hints": _clean_list(item.get("value_hints")),
                "argument_hints": _clean_list(item.get("argument_hints")),
                "state_hints": _clean_list(item.get("state_hints")),
                "confidence": confidence or "medium",
                "source": "fresh_agent_cold_start_llm_v1",
            }
        )
        if len(normalized) >= int(max_hypotheses):
            break
    return {"hypotheses": normalized}


def _build_deterministic_agent_init_plan(
    abi_functions: List[Dict[str, Any]],
    static_summary: Dict[str, Any],
    max_hypotheses: int,
) -> Dict[str, Any]:
    mutable_functions = [
        item for item in abi_functions
        if isinstance(item, dict) and _text(item.get("state_mutability")).strip() not in ("view", "pure")
    ]
    privileged = set(_clean_list(static_summary.get("privileged_functions")))
    dangerous = [
        item for item in _listify(static_summary.get("dangerous_operations"))
        if isinstance(item, dict) and _text(item.get("function")).strip()
    ]
    hypotheses: List[Dict[str, Any]] = []
    seen = set()
    for item in dangerous:
        sink = _text(item.get("function")).strip()
        if not sink:
            continue
        path = [sink]
        for privileged_name in sorted(privileged):
            if privileged_name and privileged_name != sink:
                path = [privileged_name, sink]
                break
        marker = tuple(path)
        if marker in seen:
            continue
        seen.add(marker)
        hypotheses.append(
            {
                "id": "AGENT_H{}".format(len(hypotheses) + 1),
                "hypothesis_id": "AGENT_H{}".format(len(hypotheses) + 1),
                "vulnerability_type": "fresh_agent_cold_start_deterministic",
                "path": path,
                "sink": sink,
                "priority": 0.65,
                "expected_preconditions": [],
                "expected_blockers": sorted(set(_clean_list(item.get("signals")))),
                "sender_hints": [],
                "value_hints": [],
                "argument_hints": [],
                "state_hints": [],
                "confidence": "medium",
                "source": "fresh_agent_cold_start_deterministic_v1",
            }
        )
        if len(hypotheses) >= int(max_hypotheses):
            break
    if not hypotheses:
        for function in mutable_functions[: max(1, int(max_hypotheses))]:
            name = _text(function.get("name")).strip()
            if not name:
                continue
            hypotheses.append(
                {
                    "id": "AGENT_H{}".format(len(hypotheses) + 1),
                    "hypothesis_id": "AGENT_H{}".format(len(hypotheses) + 1),
                    "vulnerability_type": "fresh_agent_cold_start_deterministic",
                    "path": [name],
                    "sink": name,
                    "priority": 0.5,
                    "expected_preconditions": [],
                    "expected_blockers": [],
                    "sender_hints": [],
                    "value_hints": [],
                    "argument_hints": [],
                    "state_hints": [],
                    "confidence": "low",
                    "source": "fresh_agent_cold_start_deterministic_v1",
                }
            )
    return {"hypotheses": hypotheses[: int(max_hypotheses)]}


def _write_agent_plan_payload(output_dir: str, plan_payload: Dict[str, Any], filename: str = "agent_path_plan.json") -> str:
    if not output_dir:
        return ""
    path = os.path.join(output_dir, filename)
    return _safe_json_dump(path, plan_payload if isinstance(plan_payload, dict) else {"hypotheses": []})


def _plan_payload_to_path_plan(plan_payload: Dict[str, Any]) -> PathPlan:
    payload = plan_payload if isinstance(plan_payload, dict) else {"hypotheses": []}
    temp_path = os.path.join(os.getcwd(), "__temp_agent_plan_payload.json")
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        return load_path_plan(temp_path)
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _apply_agent_plan_to_runtime(
    env: Any,
    args: Any,
    agent_state: Dict[str, Any],
    plan_payload: Dict[str, Any],
    plan_path: str,
    plan_source: Optional[str] = None,
    is_initial: bool = False,
) -> bool:
    if not isinstance(agent_state, dict):
        return False
    path_plan = _plan_payload_to_path_plan(plan_payload)
    agent_state["current_plan_payload"] = plan_payload if isinstance(plan_payload, dict) else {"hypotheses": []}
    agent_state["current_plan_path"] = plan_path or agent_state.get("current_plan_path")
    latest_hypothesis_count = len(_listify(plan_payload.get("hypotheses")))
    agent_state["latest_hypothesis_count"] = latest_hypothesis_count
    if is_initial or int(agent_state.get("initial_hypothesis_count") or 0) <= 0:
        agent_state["initial_hypothesis_count"] = latest_hypothesis_count
    if plan_source:
        agent_state["path_plan_source"] = plan_source
    setattr(args, "path_plan", plan_path)
    setattr(args, "path_plan_data", path_plan)
    setattr(args, "enable_path_validation", not path_plan.is_empty())
    tracker = getattr(env, "path_progress_tracker", None)
    if tracker is not None:
        try:
            tracker.replace_path_plan(path_plan)
        except Exception:
            pass
    if isinstance(getattr(env, "results", None), dict):
        env.results["agent_path_plan"] = {
            "output_path": plan_path,
            "hypothesis_count": latest_hypothesis_count,
            "fresh_agent_run": bool(agent_state.get("fresh_agent_run")),
            "llm_called": bool(agent_state.get("llm_called")),
            "fallback_used": bool(agent_state.get("fallback_used")),
            "path_plan_source": agent_state.get("path_plan_source"),
        }
    return not path_plan.is_empty()


def _build_ksi_feedback_summary(ksi_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    path_progress = []
    blocked_prefixes = []
    revert_patterns = []
    sender_value_arg_mismatches = []
    sink_or_frontier_status = []
    for record in list(ksi_records or [])[-20:]:
        if not isinstance(record, dict):
            continue
        observations = record.get("observations") if isinstance(record.get("observations"), dict) else {}
        prefix = _clean_list(observations.get("prefix_reached"))
        status = _text(record.get("status")).strip()
        if prefix or status:
            path_progress.append(
                {
                    "hypothesis_id": _text(record.get("hypothesis_id")).strip() or None,
                    "matched_prefix_len": int(record.get("matched_prefix_len") or 0),
                    "prefix_reached": prefix,
                    "status": status or None,
                }
            )
        blockers = _clean_list(record.get("suspected_blockers"))
        if blockers:
            blocked_prefixes.append(
                {
                    "hypothesis_id": _text(record.get("hypothesis_id")).strip() or None,
                    "frontier": _text(record.get("frontier")).strip() or None,
                    "blocked_at": _text(record.get("blocked_at")).strip() or None,
                    "blockers": blockers,
                }
            )
        revert_reason = _text(observations.get("revert_reason")).strip()
        if revert_reason:
            revert_patterns.append(
                {
                    "hypothesis_id": _text(record.get("hypothesis_id")).strip() or None,
                    "revert_reason": revert_reason,
                }
            )
        senders = _clean_list(observations.get("sender_sequence"))
        values = _listify(observations.get("value_sequence"))
        args_list = _clean_list(observations.get("args_sequence"))
        if len(set(senders)) > 1 or len(set(str(item) for item in values if item not in (None, ""))) > 1:
            sender_value_arg_mismatches.append(
                {
                    "hypothesis_id": _text(record.get("hypothesis_id")).strip() or None,
                    "sender_sequence": senders[:8],
                    "value_sequence": values[:8],
                    "args_sequence": args_list[:8],
                }
            )
        sink_or_frontier_status.append(
            {
                "hypothesis_id": _text(record.get("hypothesis_id")).strip() or None,
                "frontier": _text(record.get("frontier")).strip() or None,
                "sink_reached": bool(record.get("sink_reached")),
                "sink_observed": bool(record.get("sink_observed")),
                "sink_function_reached": bool(record.get("sink_function_reached")),
                "sink_executed_successfully": bool(record.get("sink_executed_successfully")),
            }
        )
    return {
        "path_progress": path_progress,
        "blocked_prefixes": blocked_prefixes,
        "revert_patterns": revert_patterns,
        "sender_value_arg_mismatches": sender_value_arg_mismatches,
        "state_dependency_evidence": [],
        "sink_or_frontier_status": sink_or_frontier_status,
    }


def _build_ksi_feedback_prompt(
    current_hypotheses: Sequence[Dict[str, Any]],
    ksi_summary: Dict[str, Any],
    allowed_functions: Sequence[str],
) -> str:
    payload = {
        "current_hypotheses": list(current_hypotheses or [])[:20],
        "ksi_evidence_summary": ksi_summary,
        "allowed_functions": list(allowed_functions or []),
    }
    instructions = [
        "You are a smart-contract fuzzing hypothesis revision agent.",
        "Revise path hypotheses from KSI runtime evidence.",
        "Do not generate executable transactions.",
        "Only revise, keep, drop, or split existing hypotheses.",
        "Do not invent function names outside allowed_functions.",
        "Output strict JSON only.",
        "Use exactly this schema:",
        '{ "agent_feedback_type": "ksi_path_revision", "hypothesis_updates": [ { "hypothesis_id": "AGENT_H1", "action": "revise", "revised_path": ["funcA","funcB"], "reason": "...", "new_expected_blockers": ["..."], "new_binding_hints": ["..."] } ] }',
        "",
        "Input JSON:",
        json.dumps(payload, ensure_ascii=True, indent=2),
    ]
    return "\n".join(instructions)


def _apply_ksi_hypothesis_updates(
    current_plan_payload: Dict[str, Any],
    raw_feedback: Any,
    allowed_functions: Sequence[str],
) -> Tuple[Dict[str, Any], int]:
    if not isinstance(current_plan_payload, dict):
        return {"hypotheses": []}, 0
    if not isinstance(raw_feedback, dict):
        return current_plan_payload, 0
    updates = raw_feedback.get("hypothesis_updates")
    if not isinstance(updates, list):
        return current_plan_payload, 0
    allowed = set(_clean_list(list(allowed_functions or [])))
    hypotheses = [dict(item) for item in _listify(current_plan_payload.get("hypotheses")) if isinstance(item, dict)]
    hypothesis_index = {
        _text(item.get("id")).strip() or _text(item.get("hypothesis_id")).strip(): item
        for item in hypotheses
    }
    revised_count = 0
    next_counter = len(hypotheses) + 1
    for update in updates:
        if not isinstance(update, dict):
            continue
        hypothesis_id = _text(update.get("hypothesis_id")).strip()
        action = _text(update.get("action")).strip().lower() or "keep"
        target = hypothesis_index.get(hypothesis_id)
        if action == "drop" and target is not None:
            hypotheses = [item for item in hypotheses if (_text(item.get("id")).strip() or _text(item.get("hypothesis_id")).strip()) != hypothesis_id]
            hypothesis_index.pop(hypothesis_id, None)
            revised_count += 1
            continue
        raw_path = _clean_list(update.get("revised_path"))
        revised_path = [name for name in raw_path if name in allowed] if raw_path else []
        if action in ("revise", "keep") and target is not None:
            if revised_path:
                target["path"] = revised_path
                target["sink"] = revised_path[-1]
            if _clean_list(update.get("new_expected_blockers")):
                target["expected_blockers"] = _clean_list(update.get("new_expected_blockers"))
            if _clean_list(update.get("new_binding_hints")):
                target["new_binding_hints"] = _clean_list(update.get("new_binding_hints"))
            if _text(update.get("reason")).strip():
                target["revision_reason"] = _text(update.get("reason")).strip()
            target["source"] = "ksi_feedback_agent_revision_v1"
            revised_count += 1
            continue
        if action == "split" and revised_path:
            new_id = "AGENT_H{}".format(next_counter)
            next_counter += 1
            hypotheses.append(
                {
                    "id": new_id,
                    "hypothesis_id": new_id,
                    "vulnerability_type": "ksi_feedback_split_path",
                    "path": revised_path,
                    "sink": revised_path[-1],
                    "priority": 0.75,
                    "expected_blockers": _clean_list(update.get("new_expected_blockers")),
                    "new_binding_hints": _clean_list(update.get("new_binding_hints")),
                    "reason": _text(update.get("reason")).strip() or None,
                    "source": "ksi_feedback_agent_revision_v1",
                }
            )
            revised_count += 1
    return {"hypotheses": hypotheses}, revised_count


def _write_full_agent_summary(run_output_dir: str, contract_name: str, results: Dict[str, Any], args: Any) -> None:
    if not run_output_dir:
        return
    agent_state = results.get("fresh_agent_runtime") if isinstance(results.get("fresh_agent_runtime"), dict) else {}
    tfi_summary = results.get("target_focus_agent_runtime") if isinstance(results.get("target_focus_agent_runtime"), dict) else {}
    path_validation = results.get("path_validation") if isinstance(results.get("path_validation"), dict) else {}
    code_coverage = results.get("code_coverage") if isinstance(results.get("code_coverage"), dict) else {}
    branch_coverage = results.get("branch_coverage") if isinstance(results.get("branch_coverage"), dict) else {}
    function_coverage = results.get("function_coverage") if isinstance(results.get("function_coverage"), dict) else {}
    timing_summary = results.get("experiment_timing") if isinstance(results.get("experiment_timing"), dict) else {}
    local_execution = results.get("local_execution_timing") if isinstance(results.get("local_execution_timing"), dict) else {}
    wall_snapshot = timing_summary.get("wall_snapshot") if isinstance(timing_summary.get("wall_snapshot"), dict) else {}
    final_snapshot = timing_summary.get("final_snapshot") if isinstance(timing_summary.get("final_snapshot"), dict) else {}
    gei60_state = results.get("gei60") if isinstance(results.get("gei60"), dict) else {}
    if not gei60_state and isinstance(results.get("pure_fuzz_warmup"), dict):
        gei60_state = results.get("pure_fuzz_warmup")
    checkpoint_summary = (
        tfi_summary.get("checkpoint_guided_summary")
        if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
        else {}
    )
    agent_llm_events = _listify(agent_state.get("llm_interaction_events"))
    checkpoint_llm_events = _checkpoint_llm_events(checkpoint_summary)
    merged_llm_events = _dedup_phase_events((agent_llm_events + checkpoint_llm_events)[-512:])
    agent_llm_latency = float(
        agent_state.get("agent_interaction_total_sec")
        if agent_state.get("agent_interaction_total_sec") not in (None, "")
        else agent_state.get("llm_interaction_time_total_sec") or 0.0
    )
    agent_llm_count = int(
        agent_state.get("agent_interaction_count")
        if agent_state.get("agent_interaction_count") not in (None, "")
        else len(agent_llm_events)
    )
    checkpoint_llm_count = max(int(checkpoint_summary.get("llm_call_count") or 0), len(checkpoint_llm_events))
    phase_cost_summary = (
        results.get("phase_cost_summary")
        if isinstance(results.get("phase_cost_summary"), dict)
        else _phase_cost_summary_from_events(merged_llm_events)
    )
    summary = {
        "contract": contract_name,
        "time_budget": int(
            getattr(args, "global_timeout", 0)
            or getattr(args, "fuzz_active_budget_sec", 0)
            or 0
        ),
        "budget_mode": getattr(args, "budget_mode", "wall"),
        "budget_fuzz_active_sec": timing_summary.get("budget_fuzz_active_sec"),
        "wall_snapshot_sec": timing_summary.get("wall_snapshot_sec"),
        "wall_snapshot_recorded": bool(timing_summary.get("wall_snapshot_recorded")),
        "fresh_agent_run": bool(agent_state.get("fresh_agent_run")),
        "precomputed_path_used": bool(agent_state.get("precomputed_path_used")),
        "precomputed_tfi_artifact_used": bool(agent_state.get("precomputed_tfi_artifact_used")),
        "precomputed_artifact_used": bool(agent_state.get("precomputed_artifact_used")),
        "gei_core_enabled": bool(getattr(args, "enable_gei_core", False)),
        "ksi_enabled": bool(getattr(args, "enable_ksi_evidence", False)),
        "tfi_enabled": bool(
            getattr(args, "enable_tfi_agent", False)
            or getattr(args, "enable_target_focus_soft_injection", False)
            or getattr(args, "enable_tfi_isolated_pilot", False)
            or getattr(args, "enable_guided_candidate_injection", False)
            or getattr(args, "enable_checkpoint_guided_injection", False)
        ),
        "tfi_planning_only": bool(getattr(args, "tfi_planning_only", False)),
        "candidate_lifecycle_config_status": (
            "planning_only"
            if (
                bool(getattr(args, "tfi_planning_only", False))
                and not bool(getattr(args, "enable_guided_candidate_injection", False))
                and not bool(getattr(args, "enable_checkpoint_guided_injection", False))
            )
            else (
                "injection_enabled"
                if (
                    bool(getattr(args, "enable_guided_candidate_injection", False))
                    or bool(getattr(args, "enable_checkpoint_guided_injection", False))
                )
                else "injection_disabled"
            )
        ),
        "guided_candidate_injection_enabled": bool(getattr(args, "enable_guided_candidate_injection", False)),
        "checkpoint_guided_injection_enabled": bool(getattr(args, "enable_checkpoint_guided_injection", False)),
        "effect_forcing_v3_enabled": bool(getattr(args, "enable_effect_forcing_v3", False)),
        "effect_forcing_v3_1_enabled": bool(getattr(args, "enable_effect_forcing_v3_1", False)),
        "prestate_builder_enabled": bool(getattr(args, "enable_prestate_builder", False)),
        "prefix_replay_suffix_mutation_enabled": bool(getattr(args, "enable_prefix_replay_suffix_mutation", False)),
        "effect_probes_enabled": bool(getattr(args, "enable_effect_probes", False)),
        "agent_effect_repair_enabled": bool(getattr(args, "enable_agent_effect_repair", False)),
        "failure_aware_agent_repair_enabled": bool(getattr(args, "enable_failure_aware_agent_repair", False)),
        "strong_useful_scheduling_enabled": bool(getattr(args, "enable_strong_useful_scheduling", False)),
        "agent_parameter_materializer_enabled": bool(getattr(args, "enable_agent_parameter_materializer", False)),
        "agent_enabled": bool(getattr(args, "enable_tfi_agent", False)),
        "agent_use_llm": bool(getattr(args, "agent_use_llm", False)),
        "rq1_config": normalize_rq1_config(getattr(args, "rq1_config", "")),
        "rq1_g_sec": int(getattr(args, "rq1_g_sec", 0) or 0),
        "rq1_k_sec": int(getattr(args, "rq1_k_sec", 0) or 0),
        "rq1_t_sec": int(getattr(args, "rq1_t_sec", 0) or 0),
        "rq1_total_sec": int(getattr(args, "rq1_total_sec", 0) or 0),
        "gei60_sec": int(gei60_state.get("gei60_sec", gei60_state.get("warmup_sec")) or 0),
        "gei60_enabled": bool(gei60_state.get("enabled")),
        "gei60_completed": bool(gei60_state.get("gei60_completed", gei60_state.get("warmup_completed"))),
        "gei60_completed_at_fuzz_active_sec": gei60_state.get(
            "gei60_completed_at_fuzz_active_sec",
            gei60_state.get("warmup_completed_at_fuzz_active_sec"),
        ),
        "pure_fuzz_warmup_sec": int(gei60_state.get("gei60_sec", gei60_state.get("warmup_sec")) or 0),
        "pure_fuzz_warmup_enabled": bool(gei60_state.get("enabled")),
        "pure_fuzz_warmup_completed": bool(gei60_state.get("gei60_completed", gei60_state.get("warmup_completed"))),
        "pure_fuzz_warmup_completed_at_fuzz_active_sec": gei60_state.get(
            "gei60_completed_at_fuzz_active_sec",
            gei60_state.get("warmup_completed_at_fuzz_active_sec"),
        ),
        "fresh_agent_initialized_after_warmup": bool(gei60_state.get("fresh_agent_initialized")),
        "fresh_agent_initialized_at_fuzz_active_sec": gei60_state.get("fresh_agent_initialized_at_fuzz_active_sec"),
        "fresh_agent_initialization_source": gei60_state.get("fresh_agent_initialization_source"),
        "llm_called": bool(agent_state.get("llm_called")) or bool(checkpoint_summary.get("llm_called")),
        "llm_success_count": int(agent_state.get("llm_success_count") or 0) + int(checkpoint_summary.get("llm_success_count") or 0),
        "llm_error_count": int(agent_state.get("llm_error_count") or 0) + int(checkpoint_summary.get("llm_error_count") or 0),
        "llm_model": agent_state.get("llm_model"),
        "llm_api_key_loaded": bool(agent_state.get("llm_api_key_loaded")),
        "llm_api_key_source": agent_state.get("llm_api_key_source"),
        "llm_interaction_events": merged_llm_events,
        "fallback_used": bool(agent_state.get("fallback_used")),
        "agent_interaction_total_sec": agent_llm_latency,
        "agent_interaction_count": agent_llm_count + checkpoint_llm_count,
        "agent_result_valid": bool(agent_state.get("agent_result_valid")),
        "agent_initial_hypothesis_count": int(agent_state.get("initial_hypothesis_count") or 0),
        "ksi_agent_called_count": int(agent_state.get("ksi_agent_called_count") or 0),
        "ksi_feedback_llm_success_count": int(agent_state.get("ksi_feedback_llm_success_count") or 0),
        "ksi_feedback_llm_error_count": int(agent_state.get("ksi_feedback_llm_error_count") or 0),
        "tfi_agent_guidance_count": int(agent_state.get("tfi_agent_guidance_count") or 0),
        "tfi_guidance_llm_success_count": int(agent_state.get("tfi_guidance_llm_success_count") or 0),
        "tfi_guidance_llm_error_count": int(agent_state.get("tfi_guidance_llm_error_count") or 0),
        "agent_revision_count": int(agent_state.get("agent_revision_count") or 0),
        "candidate_gap_event_count": int(tfi_summary.get("candidate_gap_event_count") or 0),
        "candidate_built_count": int(tfi_summary.get("candidate_built_count") or 0),
        "candidate_injected_count": int(tfi_summary.get("candidate_injected_count") or 0),
        "candidate_executed_count": int(tfi_summary.get("candidate_executed_count") or 0),
        "useful_candidate_count": int(tfi_summary.get("useful_candidate_count") or 0),
        "soft_injected_count": int(tfi_summary.get("soft_injected_count") or 0),
        "guided_candidate_count": int(tfi_summary.get("guided_candidate_count") or 0),
        "agent_guided_candidate_count": int(tfi_summary.get("agent_guided_candidate_count") or 0),
        "tfi_guided_candidate_count": int(tfi_summary.get("tfi_guided_candidate_count") or 0),
        "ksi_guided_candidate_count": int(tfi_summary.get("ksi_guided_candidate_count") or 0),
        "gei_guided_candidate_count": int(tfi_summary.get("gei_guided_candidate_count") or 0),
        "checkpoint_guided_built_count": int(tfi_summary.get("checkpoint_guided_built_count") or 0),
        "checkpoint_guided_injected_count": int(tfi_summary.get("checkpoint_guided_injected_count") or 0),
        "checkpoint_guided_executed_count": int(tfi_summary.get("checkpoint_guided_executed_count") or 0),
        "checkpoint_guided_medium_useful_count": int(tfi_summary.get("checkpoint_guided_medium_useful_count") or 0),
        "checkpoint_guided_strong_useful_count": int(tfi_summary.get("checkpoint_guided_strong_useful_count") or 0),
        "checkpoint_guided_checkpoint_delta_count": int(tfi_summary.get("checkpoint_guided_checkpoint_delta_count") or 0),
        "checkpoint_guided_oracle_level_delta_count": int(tfi_summary.get("checkpoint_guided_oracle_level_delta_count") or 0),
        "prestate_materialized_count": int(
            int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("materialization") or {}).get("prefix_materialized") or 0))
            + int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("materialization") or {}).get("prefix_suffix_materialized") or 0))
            + int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("materialization") or {}).get("suffix_materialized") or 0))
        ),
        "prefix_replay_candidate_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("prefix_replay_suffix_summary") or {}).get("prefix_replay_candidates")) or 0),
        "suffix_mutation_candidate_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("prefix_replay_suffix_summary") or {}).get("suffix_mutation_candidates")) or 0),
        "effect_probe_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("effect_probe_summary") or {}).get("effect_probe_count")) or 0),
        "failure_repair_candidate_built_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("failure_repair_summary") or {}).get("repair_candidate_built")) or 0),
        "failure_repair_candidate_materialized_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("failure_repair_summary") or {}).get("repair_candidate_materialized")) or 0),
        "failure_repair_candidate_executed_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("failure_repair_summary") or {}).get("repair_candidate_executed")) or 0),
        "failure_repair_to_medium_delta_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("failure_repair_summary") or {}).get("repair_to_medium_delta")) or 0),
        "failure_repair_to_strong_delta_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("failure_repair_summary") or {}).get("repair_to_strong_delta")) or 0),
        "failure_repair_to_l2_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("failure_repair_summary") or {}).get("repair_to_l2")) or 0),
        "failure_repair_to_l3_count": int((((tfi_summary.get("checkpoint_guided_summary") or {}).get("failure_repair_summary") or {}).get("repair_to_l3")) or 0),
        "checkpoint_guided_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("checkpoint_guided_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "checkpoint_progress_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("checkpoint_progress")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "checkpoint_guided_candidates_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("checkpoint_guided_candidates")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "checkpoint_guided_candidate_delta_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("checkpoint_guided_candidate_delta")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "parameter_materialization_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("parameter_materialization_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "prestate_builder_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("prestate_builder_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "prefix_replay_suffix_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("prefix_replay_suffix_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "effect_probe_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("effect_probe_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "agent_effect_repair_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("agent_effect_repair_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "failure_signature_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("failure_signature_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "failure_repair_trigger_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("failure_repair_trigger_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "repair_materialization_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("repair_materialization_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "repair_scheduling_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("repair_scheduling_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "failure_repair_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("failure_repair_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "failure_repair_candidate_delta_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("failure_repair_candidate_delta")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "llm_failure_repair_interactions_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("llm_failure_repair_interactions")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "llm_failure_repair_prompt_catalog_json_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("llm_failure_repair_prompt_catalog_json")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "llm_failure_repair_prompt_catalog_md_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("llm_failure_repair_prompt_catalog_md")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "strong_useful_scheduling_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("strong_useful_scheduling_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "module_contribution_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("module_contribution_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "effect_forcing_summary_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("effect_forcing_summary")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "effect_forcing_candidate_delta_path": (
            ((tfi_summary.get("checkpoint_guided_summary") or {}).get("paths") or {}).get("effect_forcing_candidate_delta")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "checkpoint_guided_current_best_level": (
            (tfi_summary.get("checkpoint_guided_summary") or {}).get("current_best_level")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "checkpoint_guided_next_missing_checkpoint": (
            (tfi_summary.get("checkpoint_guided_summary") or {}).get("next_missing_checkpoint")
            if isinstance(tfi_summary.get("checkpoint_guided_summary"), dict)
            else None
        ),
        "main_cov": code_coverage.get("percentage"),
        "total_cov": code_coverage.get("total_cov") if code_coverage.get("total_cov") not in (None, "") else final_snapshot.get("total_cov"),
        "branch_cov": branch_coverage.get("percentage", results.get("branch_cov")),
        "branch_covered": branch_coverage.get("covered", results.get("branch_covered")),
        "branch_total": branch_coverage.get("total", results.get("branch_total")),
        "branch_cov_at_wall_600": wall_snapshot.get("branch_cov"),
        "branch_cov_at_fuzz_600": final_snapshot.get("branch_cov"),
        "function_cov": function_coverage.get("percentage", results.get("function_cov")),
        "function_covered": function_coverage.get("covered", results.get("function_covered")),
        "function_total": function_coverage.get("total", results.get("function_total")),
        "function_cov_at_wall_600": wall_snapshot.get("function_cov"),
        "function_cov_at_fuzz_600": final_snapshot.get("function_cov"),
        "path_progress_count": path_validation.get("path_progress_records"),
        "frontier_touched_count": None,
        "sink_touched_count": path_validation.get("sink_function_reached_count"),
        "revert_count": path_validation.get("blocked_count"),
        "skipped_candidate_injected": False,
        "target_seen_without_progress_injected": False,
        "review_queue_injected": False,
        "target_focus_runtime_error_count": int(tfi_summary.get("target_focus_runtime_error_count") or 0),
        "final_wall_clock_elapsed_sec": timing_summary.get("final_wall_clock_elapsed_sec"),
        "final_fuzz_active_elapsed_sec": timing_summary.get("final_fuzz_active_elapsed_sec"),
        "wall_clock_time_s": timing_summary.get("final_wall_clock_elapsed_sec"),
        "fuzz_active_time_s": timing_summary.get("final_fuzz_active_elapsed_sec"),
        "has_clean_local_tx_execution_timer": bool(local_execution.get("has_clean_local_tx_execution_timer")),
        "clean_local_tx_execution_time_sec": local_execution.get("clean_local_tx_execution_time_sec"),
        "clean_local_tx_execution_count": local_execution.get("clean_local_tx_execution_count"),
        "clean_local_fuzz_tx_execution_time_sec": local_execution.get("clean_local_fuzz_tx_execution_time_sec"),
        "clean_local_fuzz_tx_execution_count": local_execution.get("clean_local_fuzz_tx_execution_count"),
        "clean_local_setup_tx_execution_time_sec": local_execution.get("clean_local_setup_tx_execution_time_sec"),
        "clean_local_setup_tx_execution_count": local_execution.get("clean_local_setup_tx_execution_count"),
        "clean_local_agent_tx_execution_time_sec": local_execution.get("clean_local_agent_tx_execution_time_sec"),
        "clean_local_agent_tx_execution_count": local_execution.get("clean_local_agent_tx_execution_count"),
        "local_execution_timing": local_execution,
        "llm_total_latency_s": agent_llm_latency,
        "phase_cost_summary": phase_cost_summary,
        "active_timer_mode": "wall_minus_agent_latency",
        "active_timer_supported": True,
        "coverage_at_wall_600": wall_snapshot.get("coverage"),
        "main_cov_at_wall_600": wall_snapshot.get("main_cov"),
        "total_cov_at_wall_600": wall_snapshot.get("total_cov"),
        "tx_count_at_wall_600": wall_snapshot.get("fuzz_tx_count"),
        "sequence_count_at_wall_600": wall_snapshot.get("sequence_count"),
        "coverage_at_fuzz_600": final_snapshot.get("coverage"),
        "main_cov_at_fuzz_600": final_snapshot.get("main_cov"),
        "total_cov_at_fuzz_600": final_snapshot.get("total_cov"),
        "tx_count_at_fuzz_600": final_snapshot.get("fuzz_tx_count"),
        "sequence_count_at_fuzz_600": final_snapshot.get("sequence_count"),
    }
    if not bool(summary.get("agent_use_llm")):
        summary["llm_health"] = "not_requested"
    elif not bool(summary.get("llm_called")):
        summary["llm_health"] = "not_called"
    elif int(summary.get("llm_success_count") or 0) > 0:
        summary["llm_health"] = "ok"
    elif int(summary.get("llm_error_count") or 0) > 0:
        summary["llm_health"] = "all_calls_failed"
    else:
        summary["llm_health"] = "no_success_recorded"

    if not bool(summary.get("tfi_enabled")):
        summary["candidate_lifecycle_health"] = "not_enabled"
    elif _text(summary.get("candidate_lifecycle_config_status")).strip() == "planning_only":
        summary["candidate_lifecycle_health"] = "planning_only"
    elif _text(summary.get("candidate_lifecycle_config_status")).strip() != "injection_enabled":
        summary["candidate_lifecycle_health"] = "injection_disabled"
    elif int(summary.get("candidate_built_count") or 0) <= 0:
        summary["candidate_lifecycle_health"] = "injection_configured_no_candidates"
    elif int(summary.get("candidate_injected_count") or 0) <= 0:
        summary["candidate_lifecycle_health"] = "built_not_injected"
    elif int(summary.get("candidate_executed_count") or 0) <= 0:
        summary["candidate_lifecycle_health"] = "injected_not_executed"
    else:
        summary["candidate_lifecycle_health"] = "executed"
    json_path = os.path.join(run_output_dir, "full_agent_600s_summary.json")
    csv_path = os.path.join(run_output_dir, "full_agent_600s_summary.csv")
    _safe_json_dump(json_path, summary)
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
    except Exception:
        pass


def _agent_runtime_result_payload(agent_state: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(agent_state or {})
    payload.pop("_env_ref", None)
    payload["precomputed_artifact_used"] = bool(
        payload.get("precomputed_path_used") or payload.get("precomputed_tfi_artifact_used")
    )
    payload["agent_result_valid"] = bool(
        payload.get("llm_called")
        and int(payload.get("llm_success_count") or 0) > 0
        and not payload.get("precomputed_artifact_used")
        and not payload.get("fallback_used")
    )
    payload["current_plan_path"] = payload.get("current_plan_path")
    payload["initial_hypothesis_count"] = int(payload.get("initial_hypothesis_count") or 0)
    payload["latest_hypothesis_count"] = int(payload.get("latest_hypothesis_count") or 0)
    payload["ksi_agent_called_count"] = int(payload.get("ksi_agent_called_count") or 0)
    payload["ksi_feedback_llm_success_count"] = int(payload.get("ksi_feedback_llm_success_count") or 0)
    payload["ksi_feedback_llm_error_count"] = int(payload.get("ksi_feedback_llm_error_count") or 0)
    payload["tfi_agent_guidance_count"] = int(payload.get("tfi_agent_guidance_count") or 0)
    payload["tfi_guidance_llm_success_count"] = int(payload.get("tfi_guidance_llm_success_count") or 0)
    payload["tfi_guidance_llm_error_count"] = int(payload.get("tfi_guidance_llm_error_count") or 0)
    payload["agent_revision_count"] = int(payload.get("agent_revision_count") or 0)
    payload["llm_interaction_time_total_sec"] = float(payload.get("llm_interaction_time_total_sec") or 0.0)
    payload["agent_interaction_total_sec"] = float(
        payload.get("agent_interaction_total_sec", payload.get("llm_interaction_time_total_sec")) or 0.0
    )
    payload["agent_interaction_count"] = int(
        payload.get("agent_interaction_count")
        if payload.get("agent_interaction_count") not in (None, "")
        else len(_listify(payload.get("llm_interaction_events")))
    )
    if not isinstance(payload.get("llm_interaction_events"), list):
        payload["llm_interaction_events"] = []
    return payload


def _initialize_fresh_agent_plan(
    logger: Any,
    env: Any,
    args: Any,
    contract_name: str,
    abi: Any,
    source_path: str,
    run_output_dir: str,
    llm_provider: Optional[LLMProvider],
) -> Dict[str, Any]:
    agent_state = getattr(env, "fresh_agent_runtime_state", None)
    if not isinstance(agent_state, dict):
        agent_state = _default_agent_runtime_state(args, run_output_dir)
        env.fresh_agent_runtime_state = agent_state

    abi_functions = _abi_function_summaries(abi)
    source_excerpt = _read_source_excerpt(source_path)
    static_summary = _build_slither_static_summary(
        source_path=source_path,
        contract_name=contract_name,
        solc_dir=getattr(args, "solc_dir", None),
        solc_version=getattr(args, "solc_version", None),
    )
    agent_state["function_names"] = [item["name"] for item in abi_functions if isinstance(item, dict) and item.get("name")]

    llm_called = False
    llm_success = False
    fallback_used = False
    error = ""
    plan_payload = {"hypotheses": []}

    if (
        (
            getattr(args, "enable_tfi_agent", False)
            or getattr(args, "enable_ksi_llm_feedback", False)
        )
        and getattr(args, "agent_use_llm", False)
        and llm_provider
    ):
        if llm_provider.is_available() and llm_provider.is_configured():
            llm_started_at = time.time()
            try:
                raw_payload = llm_provider.generate_json(
                    _build_agent_init_prompt(
                        contract_name=contract_name,
                        abi_functions=abi_functions,
                        static_summary=static_summary,
                        source_excerpt=source_excerpt,
                        max_hypotheses=int(getattr(args, "agent_max_hypotheses", 20) or 20),
                    )
                )
                llm_called = True
                plan_payload = _normalize_agent_init_plan(
                    raw_payload,
                    abi_functions=abi_functions,
                    max_hypotheses=int(getattr(args, "agent_max_hypotheses", 20) or 20),
                )
                llm_success = bool(_listify(plan_payload.get("hypotheses")))
                if not llm_success:
                    error = "agent_init_empty_hypotheses"
                _record_agent_llm_timing(
                    agent_state,
                    phase="agent_init",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=llm_success,
                    error=error,
                    used_fallback=False,
                    metadata=llm_provider.get_last_response_metadata(),
                )
            except LLMProviderError as exc:
                llm_called = True
                error = str(exc)
                _record_agent_llm_timing(
                    agent_state,
                    phase="agent_init",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=False,
                    error=error,
                    used_fallback=True,
                )
            except Exception as exc:
                llm_called = True
                error = "agent_init_runtime_error: {}".format(str(exc))
                _record_agent_llm_timing(
                    agent_state,
                    phase="agent_init",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=False,
                    error=error,
                    used_fallback=True,
                )
        else:
            error = "agent_provider_not_configured"

    if not _listify(plan_payload.get("hypotheses")) and not bool(agent_state.get("strict_llm_only")):
        plan_payload = _build_deterministic_agent_init_plan(
            abi_functions=abi_functions,
            static_summary=static_summary,
            max_hypotheses=int(getattr(args, "agent_max_hypotheses", 20) or 20),
        )
        fallback_used = bool(_listify(plan_payload.get("hypotheses")))

    plan_path = ""
    if _listify(plan_payload.get("hypotheses")):
        plan_path = _write_agent_plan_payload(run_output_dir, plan_payload, filename="agent_path_plan.json")
        _apply_agent_plan_to_runtime(
            env,
            args,
            agent_state,
            plan_payload,
            plan_path,
            plan_source="fresh_agent",
            is_initial=True,
        )

    _record_agent_llm_stats(
        agent_state,
        llm_called=llm_called,
        llm_success=llm_success,
        error=error,
        fallback_used=fallback_used,
    )
    agent_state["llm_model"] = getattr(args, "llm_model", None) or agent_state.get("llm_model")
    payload = _agent_runtime_result_payload(agent_state)
    env.results["fresh_agent_runtime"] = payload

    logger.info(
        "[AGENT_INIT] contract=%s source_loaded=%s abi_loaded=%s slither_loaded=%s hypothesis_count=%s llm_called=%s llm_success=%s fallback_used=%s",
        contract_name,
        int(bool(source_excerpt)),
        int(bool(abi_functions)),
        int(bool(static_summary.get("loaded"))),
        len(_listify(plan_payload.get("hypotheses"))),
        int(bool(llm_called)),
        int(bool(llm_success)),
        int(bool(fallback_used)),
    )
    for item in _listify(plan_payload.get("hypotheses")):
        if not isinstance(item, dict):
            continue
        logger.info(
            "[AGENT_PATH] hypothesis_id=%s vulnerability_type=%s path=%s sink=%s expected_blockers=%s",
            _text(item.get("hypothesis_id")).strip() or _text(item.get("id")).strip() or "UNKNOWN",
            _text(item.get("vulnerability_type")).strip() or "unknown",
            json.dumps(_clean_list(item.get("path")), ensure_ascii=True),
            _text(item.get("sink")).strip() or "UNKNOWN",
            json.dumps(_clean_list(item.get("expected_blockers")), ensure_ascii=True),
        )
    return payload


def _ensure_delayed_fresh_agent_plan(
    *,
    logger: Any,
    env: Any,
    args: Any,
    contract_name: str,
    abi: Any,
    source_path: str,
    run_output_dir: str,
    llm_provider: Optional[LLMProvider],
) -> bool:
    warmup_state = _ensure_gei60_state(env, args)
    if not bool(warmup_state.get("fresh_agent_requested")):
        return False
    if bool(warmup_state.get("fresh_agent_initialized")):
        return False
    if _gei60_active(env):
        return False
    _initialize_fresh_agent_plan(
        logger=logger,
        env=env,
        args=args,
        contract_name=contract_name,
        abi=abi,
        source_path=source_path,
        run_output_dir=run_output_dir,
        llm_provider=llm_provider,
    )
    warmup_state["fresh_agent_initialized"] = True
    warmup_state["fresh_agent_initialized_at_fuzz_active_sec"] = round(_current_fuzz_active_elapsed_sec(env), 6)
    warmup_state["fresh_agent_initialization_source"] = "post_warmup"
    results = getattr(env, "results", None)
    if isinstance(results, dict):
        _sync_gei60_results(env, warmup_state)
    return True


class AgentKSIFeedbackAnalysis(OnTheFlyAnalysis):
    interval = 1

    def __init__(self, logger: Any, llm_provider: Optional[LLMProvider], interval: int = 20) -> None:
        self.logger = logger
        self.llm_provider = llm_provider
        self.runtime_interval = max(1, int(interval or 1))
        self.last_record_count = 0

    def setup(self, ng, engine):
        pass

    def register_step(self, g, population, engine):
        round_id = int(g)
        if round_id < 0 or round_id % self.runtime_interval != 0:
            return
        env = getattr(engine, "env", None)
        if env is None:
            return
        if _gei60_active(env):
            return
        if not rq1_phase_allows_feature(env, "agent", getattr(env, "args", None)):
            return
        runtime_args = getattr(env, "args", None)
        try:
            _ensure_delayed_fresh_agent_plan(
                logger=self.logger,
                env=env,
                args=runtime_args,
                contract_name=_text(getattr(env, "contract_name", "")).strip(),
                abi=getattr(env, "abi", None),
                source_path=getattr(runtime_args, "source", None) if runtime_args is not None else None,
                run_output_dir=_text(getattr(runtime_args, "run_output_dir", None)).strip()
                or _resolve_run_output_dir(getattr(runtime_args, "results", None) if runtime_args is not None else None),
                llm_provider=self.llm_provider if _llm_runtime_allowed(env, runtime_args) else None,
            )
        except Exception:
            pass
        agent_state = getattr(env, "fresh_agent_runtime_state", None)
        if not isinstance(agent_state, dict):
            return
        current_payload = agent_state.get("current_plan_payload") if isinstance(agent_state.get("current_plan_payload"), dict) else {}
        current_hypotheses = [item for item in _listify(current_payload.get("hypotheses")) if isinstance(item, dict)]
        if not current_hypotheses:
            return
        ksi_runtime = getattr(env, "ksi_runtime", None)
        ksi_path = getattr(ksi_runtime, "output_path", None)
        if not ksi_path or not os.path.exists(ksi_path):
            return
        records: List[Dict[str, Any]] = []
        try:
            with open(ksi_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        records.append(payload)
        except Exception:
            return
        if len(records) <= self.last_record_count:
            return
        new_records = records[self.last_record_count :]
        self.last_record_count = len(records)
        ksi_summary = _build_ksi_feedback_summary(new_records)
        evidence_count = len(new_records)
        hypotheses_updated = 0
        llm_called = False
        fallback_used = False

        if _llm_runtime_allowed(env, runtime_args) and self.llm_provider and self.llm_provider.is_available() and self.llm_provider.is_configured():
            llm_started_at = time.time()
            try:
                agent_state["ksi_agent_called_count"] = int(agent_state.get("ksi_agent_called_count") or 0) + 1
                raw_feedback = self.llm_provider.generate_json(
                    _build_ksi_feedback_prompt(
                        current_hypotheses=current_hypotheses,
                        ksi_summary=ksi_summary,
                        allowed_functions=agent_state.get("function_names") or [],
                    )
                )
                llm_called = True
                revised_payload, hypotheses_updated = _apply_ksi_hypothesis_updates(
                    current_plan_payload=current_payload,
                    raw_feedback=raw_feedback,
                    allowed_functions=agent_state.get("function_names") or [],
                )
                if hypotheses_updated > 0:
                    plan_path = _write_agent_plan_payload(
                        getattr(agent_state, "run_output_dir", None) if False else agent_state.get("run_output_dir"),
                        revised_payload,
                        filename="agent_path_plan.json",
                    )
                    _apply_agent_plan_to_runtime(
                        env,
                        env.args,
                        agent_state,
                        revised_payload,
                        plan_path,
                        plan_source="ksi_feedback_agent_revision",
                    )
                    agent_state["agent_revision_count"] = int(agent_state.get("agent_revision_count") or 0) + hypotheses_updated
                agent_state["ksi_feedback_llm_success_count"] = int(
                    agent_state.get("ksi_feedback_llm_success_count") or 0
                ) + 1
                _record_agent_llm_stats(agent_state, llm_called=True, llm_success=True)
                _record_agent_llm_timing(
                    agent_state,
                    phase="ksi_feedback",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=True,
                    error="",
                    used_fallback=False,
                    metadata=self.llm_provider.get_last_response_metadata(),
                )
            except LLMProviderError as exc:
                llm_called = True
                fallback_used = True
                _record_agent_llm_stats(agent_state, llm_called=True, error=str(exc), fallback_used=True)
                _record_agent_llm_timing(
                    agent_state,
                    phase="ksi_feedback",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=False,
                    error=str(exc),
                    used_fallback=True,
                )
                agent_state["ksi_feedback_llm_error_count"] = int(
                    agent_state.get("ksi_feedback_llm_error_count") or 0
                ) + 1
            except Exception as exc:
                llm_called = True
                fallback_used = True
                _record_agent_llm_stats(
                    agent_state,
                    llm_called=True,
                    error="ksi_feedback_runtime_error: {}".format(str(exc)),
                    fallback_used=True,
                )
                _record_agent_llm_timing(
                    agent_state,
                    phase="ksi_feedback",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=False,
                    error="ksi_feedback_runtime_error: {}".format(str(exc)),
                    used_fallback=True,
                )
                agent_state["ksi_feedback_llm_error_count"] = int(
                    agent_state.get("ksi_feedback_llm_error_count") or 0
                ) + 1
        else:
            fallback_used = True

        env.results["fresh_agent_runtime"] = _agent_runtime_result_payload(agent_state)
        self.logger.info(
            "[AGENT_KSI_FEEDBACK] round=%s evidence_count=%s hypotheses_updated=%s llm_called=%s fallback_used=%s",
            round_id,
            evidence_count,
            hypotheses_updated,
            int(bool(llm_called)),
            int(bool(fallback_used)),
        )

    def finalize(self, population, engine):
        pass


class RQ1GEISenderPoolRestrictionAnalysis(OnTheFlyAnalysis):
    interval = 1

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def setup(self, ng, engine):
        env = getattr(engine, "env", None) if engine is not None else None
        if env is not None:
            _update_rq1_gei_sender_pool(env, logger=self.logger)

    def register_step(self, g, population, engine):
        env = getattr(engine, "env", None) if engine is not None else None
        if env is not None:
            _update_rq1_gei_sender_pool(env, logger=self.logger)

    def finalize(self, population, engine):
        env = getattr(engine, "env", None) if engine is not None else None
        if env is not None:
            _update_rq1_gei_sender_pool(env, logger=self.logger)


class TFISoftInjectionAnalysis(OnTheFlyAnalysis):
    interval = 1

    def __init__(
        self,
        runtime: TFIAgentRuntime,
        buffer: TFISoftInjectionBuffer,
        logger: Any,
        interval: int = 20,
        max_per_round: int = 3,
        enable_agent: bool = False,
        enable_soft_injection: bool = False,
        enable_guided_candidate_injection: bool = False,
        guided_injection_confidence_threshold: float = 0.0,
        checkpoint_guided_runtime: Optional[CheckpointGuidedRuntime] = None,
        enable_checkpoint_guided_injection: bool = False,
    ) -> None:
        self.runtime = runtime
        self.buffer = buffer
        self.logger = logger
        self.runtime_interval = max(1, int(interval or 1))
        self.max_per_round = max(1, int(max_per_round or 1))
        self.enable_agent = bool(enable_agent)
        self.enable_soft_injection = bool(enable_soft_injection)
        self.enable_guided_candidate_injection = bool(enable_guided_candidate_injection)
        self.checkpoint_guided_runtime = checkpoint_guided_runtime
        self.enable_checkpoint_guided_injection = bool(enable_checkpoint_guided_injection and checkpoint_guided_runtime is not None)
        try:
            self.guided_injection_confidence_threshold = max(
                0.0,
                float(guided_injection_confidence_threshold or 0.0),
            )
        except Exception:
            self.guided_injection_confidence_threshold = 0.0
        self.agent_called_count = 0
        self.agent_error_count = 0
        self.target_focus_runtime_error_count = 0
        self.candidate_built_count = 0
        self.candidate_injected_count = 0
        self.candidate_executed_count = 0
        self.useful_candidate_count = 0
        self.candidate_gap_event_count = 0
        self.candidate_gap_guidance_count = 0
        self.guidance_generated_count = 0
        self.fuzzer_generated_candidate_count = 0
        self.soft_injected_count = 0
        self.duplicate_skipped_count = 0
        self.llm_called_count = 0
        self.fallback_used_count = 0
        self.max_injected_per_round = 0
        self.injection_policy = "append"
        self.bypasses_normal_population_order = False
        self.last_error = ""
        self.force_retry_round = None
        self.tfi_phase_bootstrap_done = False
        self.guided_candidate_count = 0
        self.agent_guided_candidate_count = 0
        self.tfi_guided_candidate_count = 0
        self.ksi_guided_candidate_count = 0
        self.gei_guided_candidate_count = 0
        self.useful_by_reason: Dict[str, int] = {}
        self.checkpoint_guided_built_count = 0
        self.checkpoint_guided_injected_count = 0
        self.checkpoint_guided_executed_count = 0
        self.checkpoint_guided_medium_useful_count = 0
        self.checkpoint_guided_strong_useful_count = 0
        self.checkpoint_guided_checkpoint_delta_count = 0
        self.checkpoint_guided_oracle_level_delta_count = 0
        output_dir = getattr(self.runtime, "output_dir", "")
        self.guided_candidates_path = os.path.join(output_dir, "guided_candidates.jsonl") if output_dir else ""
        self.guided_summary_path = os.path.join(output_dir, "guided_injection_summary.json") if output_dir else ""
        if self.guided_candidates_path:
            try:
                os.makedirs(os.path.dirname(self.guided_candidates_path), exist_ok=True)
                with open(self.guided_candidates_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                self.guided_candidates_path = ""
        self.repair_candidate_drop_reason_histogram: Dict[str, int] = {}

    def setup(self, ng, engine):
        if engine is not None and getattr(engine, "env", None) is not None:
            setattr(engine.env, "target_focus_soft_injection_analysis", self)
            try:
                setattr(self.runtime, "env", engine.env)
            except Exception:
                pass

    def _repair_runtime(self) -> Optional[CheckpointGuidedRuntime]:
        return self.checkpoint_guided_runtime if self.enable_checkpoint_guided_injection else None

    def _is_failure_repair_individual(
        self,
        runtime: Optional[CheckpointGuidedRuntime],
        individual: Any,
        candidate_id: str,
    ) -> bool:
        if runtime is None or not candidate_id:
            return False
        candidate_lookup = getattr(runtime, "candidates_by_id", None)
        if isinstance(candidate_lookup, dict):
            candidate = candidate_lookup.get(candidate_id)
            if isinstance(candidate, dict) and bool(candidate.get("failure_repair_candidate")):
                return True
        metadata = getattr(individual, "repair_pilot_metadata", None)
        if not isinstance(metadata, dict):
            return False
        if _text(metadata.get("source_trigger_id")).strip() or _text(metadata.get("source_parent_id")).strip():
            return True
        source_kind = _text(metadata.get("source_candidate_kind")).strip().lower()
        return "failure_repair" in source_kind

    def record_duplicate_skip(self, individual: Any) -> None:
        runtime = self._repair_runtime()
        if runtime is None or not hasattr(individual, "repair_candidate_id"):
            return
        candidate_id = _text(getattr(individual, "repair_candidate_id", None)).strip()
        if not candidate_id:
            return
        if not self._is_failure_repair_individual(runtime, individual, candidate_id):
            return
        runtime._bump_pipeline_counter("repair_dropped", 1)
        runtime.record_repair_submission_outcome(
            candidate_id,
            submit_result="skipped",
            reason="duplicate_execution_signature",
            individual=individual,
            extra={
                "queue_budget_remaining": None,
                "time_budget_remaining": None,
                "exception_if_any": "",
            },
            terminal=True,
        )
        runtime.repair_lifecycle.mark_dropped(
            candidate_id,
            reason="duplicate_hash_before_execution",
            extra={"sequence_id": _text(getattr(individual, "sequence_id", None)).strip()},
        )

    def record_execution_submission(self, individual: Any) -> None:
        runtime = self._repair_runtime()
        if runtime is None or not hasattr(individual, "repair_candidate_id"):
            return
        candidate_id = _text(getattr(individual, "repair_candidate_id", None)).strip()
        if not candidate_id:
            return
        if not self._is_failure_repair_individual(runtime, individual, candidate_id):
            return
        runtime._bump_pipeline_counter("repair_execution_submitted", 1)
        runtime.record_repair_submission_outcome(
            candidate_id,
            submit_result="submitted",
            individual=individual,
            extra={
                "queue_budget_remaining": None,
                "time_budget_remaining": None,
                "exception_if_any": "",
            },
            terminal=True,
        )
        runtime.repair_lifecycle.mark_stage(
            candidate_id,
            stage="execution_submitted",
            extra={
                "sequence_id": _text(getattr(individual, "sequence_id", None)).strip(),
                "individual_hash": _text(getattr(individual, "hash", None)).strip(),
            },
        )

    def record_execution_completed(self, individual: Any) -> None:
        runtime = self._repair_runtime()
        if runtime is None or not hasattr(individual, "repair_candidate_id"):
            return
        candidate_id = _text(getattr(individual, "repair_candidate_id", None)).strip()
        if not candidate_id:
            return
        if not self._is_failure_repair_individual(runtime, individual, candidate_id):
            return
        runtime._bump_pipeline_counter("repair_executed", 1)
        runtime.failure_repair_summary["repair_candidate_executed"] += 1
        extra = {
            "sequence_id": _text(getattr(individual, "sequence_id", None)).strip(),
            "individual_hash": _text(getattr(individual, "hash", None)).strip(),
        }
        runtime.repair_lifecycle.mark_stage(
            candidate_id,
            stage="executed",
            extra=extra,
        )
        runtime.repair_lifecycle.mark_tx_result(
            candidate_id,
            extra=dict(extra, executed=True),
        )

    def record_execution_tx_result(
        self,
        individual: Any,
        *,
        tx_index: int,
        tx_count: int,
        function_name: str = "",
        success: Optional[bool] = None,
        revert_reason: str = "",
        event_type: str = "tx_result",
    ) -> None:
        runtime = self._repair_runtime()
        if runtime is None or not hasattr(individual, "repair_candidate_id"):
            return
        candidate_id = _text(getattr(individual, "repair_candidate_id", None)).strip()
        if not candidate_id or not self._is_failure_repair_individual(runtime, individual, candidate_id):
            return
        payload = {
            "sequence_id": _text(getattr(individual, "sequence_id", None)).strip(),
            "individual_hash": _text(getattr(individual, "hash", None)).strip(),
            "tx_index": int(tx_index),
            "tx_count": int(tx_count),
            "function_name": _text(function_name).strip(),
            "success": bool(success) if success is not None else None,
            "revert_reason": _text(revert_reason).strip(),
        }
        runtime.repair_lifecycle.mark_tx_result(candidate_id, extra=payload)
        runtime.log_repair_execution_debug(candidate_id, event_type, **payload)

    def register_step(self, g, population, engine):
        raw_generation = int(g)
        if raw_generation < 0:
            return
        round_id = raw_generation
        env = getattr(engine, "env", None)
        if env is not None:
            try:
                _ensure_delayed_fresh_agent_plan(
                    logger=self.logger,
                    env=env,
                    args=getattr(env, "args", None),
                    contract_name=_text(getattr(env, "contract_name", "")).strip(),
                    abi=getattr(env, "abi", None),
                    source_path=getattr(getattr(env, "args", None), "source", None),
                    run_output_dir=_text(getattr(getattr(env, "args", None), "run_output_dir", None)).strip()
                    or _resolve_run_output_dir(getattr(getattr(env, "args", None), "results", None)),
                    llm_provider=self.runtime.llm_provider if _llm_runtime_allowed(env, getattr(env, "args", None)) else None,
                )
            except Exception:
                pass
        if env is not None and _gei60_active(env):
            return
        if env is not None and not rq1_phase_allows_feature(env, "tfi", getattr(env, "args", None)):
            if (
                self.enable_checkpoint_guided_injection
                and self.checkpoint_guided_runtime is not None
                and rq1_phase_allows_feature(env, "ksi", getattr(env, "args", None))
                and round_id % self.runtime_interval == 0
            ):
                self._checkpoint_observe_only(round_id)
            return
        self._checkpoint_observe_only(round_id)
        if self.enable_soft_injection:
            injected = self._inject_from_buffer(round_id, population)
            if injected >= 0:
                self.soft_injected_count += injected
        should_run = (round_id % self.runtime_interval == 0)
        if (
            env is not None
            and not self.tfi_phase_bootstrap_done
            and rq1_current_phase(env, getattr(env, "args", None)) == "T"
        ):
            should_run = True
            self.tfi_phase_bootstrap_done = True
        if self.force_retry_round is not None and round_id >= int(self.force_retry_round):
            should_run = True
            self.force_retry_round = None
        if not should_run:
            return

        result = {}
        try:
            result = self.runtime.run_once()
        except Exception as exc:
            self.agent_error_count += 1
            self.target_focus_runtime_error_count += 1
            self.last_error = "target_focus_runtime_error: {}".format(str(exc))
            self.logger.warning(
                "[TFI_TASK_BUILD] round=%s status=error used_inputs_count=0 missing_inputs_count=0 scope_count=0 request_count=0 plan_count=0 task_count=0 candidate_gap_event_count=0 reason=%s",
                round_id,
                self.last_error,
            )
            self.logger.warning("[TFI_AGENT] round=%s tasks_seen=0 candidate_gap_events_seen=0 guidance_generated=0 candidates_built=0 candidates_executed=0 useful_candidates=0 agent_called=%s llm_called=0 fallback_used=0 skipped_reason=runtime_error errors=%s",
                                round_id, int(self.enable_agent), self.last_error)
            return

        self.agent_called_count += int(result.get("agent_called_count") or 0)
        self.agent_error_count += int(result.get("agent_error_count") or 0)
        self.candidate_gap_event_count += int(result.get("candidate_gap_event_count") or 0)
        self.candidate_gap_guidance_count += int(result.get("candidate_gap_guidance_count") or 0)
        self.guidance_generated_count += int(result.get("guidance_generated_count") or 0)
        self.fuzzer_generated_candidate_count += int(result.get("fuzzer_generated_candidate_count") or 0)
        self.candidate_built_count += int(result.get("candidate_built_count") or 0)
        self.candidate_executed_count += int(result.get("candidate_executed_count") or 0)
        self.useful_candidate_count += int(result.get("useful_candidate_count") or 0)
        self.llm_called_count += int(bool(result.get("llm_called")))
        self.fallback_used_count += int(bool(result.get("fallback_used")))
        self.last_error = ";".join(_clean_list(result.get("agent_errors")))
        env = getattr(engine, "env", None)
        agent_state = getattr(env, "fresh_agent_runtime_state", None) if env is not None else None
        if isinstance(agent_state, dict):
            if result.get("llm_called"):
                agent_state["llm_called"] = True
            agent_state["llm_success_count"] = int(agent_state.get("llm_success_count") or 0) + int(
                result.get("llm_success_count") or 0
            )
            agent_state["llm_error_count"] = int(agent_state.get("llm_error_count") or 0) + int(
                result.get("llm_error_count") or 0
            )
            interaction_events = agent_state.get("llm_interaction_events")
            if not isinstance(interaction_events, list):
                interaction_events = []
            new_events = [item for item in _listify(result.get("llm_interaction_events")) if isinstance(item, dict)]
            if new_events:
                interaction_events.extend(new_events)
                agent_state["llm_interaction_events"] = interaction_events[-256:]
            llm_interaction_total_sec = float(agent_state.get("llm_interaction_time_total_sec") or 0.0) + float(
                result.get("llm_interaction_time_total_sec") or 0.0
            )
            agent_state["llm_interaction_time_total_sec"] = llm_interaction_total_sec
            agent_state["agent_interaction_total_sec"] = float(
                agent_state.get("agent_interaction_total_sec") or 0.0
            ) + float(result.get("llm_interaction_time_total_sec") or 0.0)
            agent_state["agent_interaction_count"] = int(agent_state.get("agent_interaction_count") or 0) + int(
                result.get("llm_interaction_count") or len(new_events)
            )
            if result.get("fallback_used"):
                agent_state["fallback_used"] = True
            env.results["fresh_agent_runtime"] = _agent_runtime_result_payload(agent_state)
        task_build = result.get("task_build") if isinstance(result.get("task_build"), dict) else {}
        self._maybe_enable_runtime_path_tracking(engine, task_build, round_id)
        self.logger.info(
            "[TFI_TASK_BUILD] round=%s status=%s used_inputs_count=%s missing_inputs_count=%s scope_count=%s request_count=%s plan_count=%s task_count=%s candidate_gap_event_count=%s reason=%s",
            round_id,
            _text(task_build.get("status")).strip() or "unknown",
            len(_listify(task_build.get("used_inputs"))),
            len(_listify(task_build.get("missing_inputs"))),
            int(task_build.get("scope_count") or 0),
            int(task_build.get("request_count") or 0),
            int(task_build.get("plan_count") or 0),
            int(task_build.get("task_count") or 0),
            int(task_build.get("candidate_gap_event_count") or 0),
            _text(task_build.get("reason")).strip() or "unknown",
        )

        gap_events = [item for item in _listify(result.get("candidate_gap_events")) if isinstance(item, dict)]
        for item in gap_events:
            self.logger.info(
                "[TFI_GAP] round=%s scope_id=%s target_function=%s frontier=%s sink=%s observed_prefix_len=%s agent_guidance_required=%s fuzzer_generation_required=%s",
                round_id,
                _text(item.get("scope_id")).strip() or "UNKNOWN",
                _text(item.get("target_function")).strip() or "UNKNOWN",
                _text(item.get("frontier")).strip() or "UNKNOWN",
                _text(item.get("sink")).strip() or "UNKNOWN",
                len(_clean_list(item.get("observed_prefix"))),
                int(bool(item.get("agent_guidance_required"))),
                int(bool(item.get("fuzzer_generation_required"))),
            )
        guidance_records = [item for item in _listify(result.get("gap_guidance_records")) if isinstance(item, dict)]
        if isinstance(agent_state, dict) and guidance_records:
            agent_state["tfi_agent_guidance_count"] = int(agent_state.get("tfi_agent_guidance_count") or 0) + len(guidance_records)
            agent_state["tfi_guidance_llm_success_count"] = int(
                agent_state.get("tfi_guidance_llm_success_count") or 0
            ) + int(result.get("gap_llm_success_count") or 0)
            agent_state["tfi_guidance_llm_error_count"] = int(
                agent_state.get("tfi_guidance_llm_error_count") or 0
            ) + int(result.get("gap_llm_error_count") or 0)
            env.results["fresh_agent_runtime"] = _agent_runtime_result_payload(agent_state)
        for item in guidance_records:
            self.logger.info(
                "[AGENT_TFI_GUIDANCE] round=%s scope_id=%s gap_type=%s guidance_generated=%s llm_called=%s fallback_used=%s",
                round_id,
                _text(item.get("scope_id")).strip() or "UNKNOWN",
                _text(item.get("gap_type")).strip() or "local_candidate_gap",
                1,
                int(bool(item.get("llm_called"))),
                int(bool(item.get("fallback_used"))),
            )

        candidate_records = [item for item in _listify(result.get("candidate_records")) if isinstance(item, dict)]
        pilot_results = {}
        pilot_payload = result.get("pilot_result") if isinstance(result.get("pilot_result"), dict) else {}
        for item in _listify(pilot_payload.get("results")):
            if isinstance(item, dict):
                candidate_id = _text(item.get("candidate_id")).strip()
                if candidate_id:
                    pilot_results[candidate_id] = item
        for item in candidate_records:
            candidate_id = _text(item.get("candidate_id")).strip()
            execution = pilot_results.get(candidate_id, {})
            source = self._candidate_source(item, result)
            self._count_guided_source(source)
            self._append_guided_candidate_record(
                round_id=round_id,
                candidate=item,
                source=source,
                executed=bool(execution) and _text(execution.get("execution_status")).strip() != "skipped",
                injected=False,
                execution_result=execution,
                useful=bool(execution.get("useful")) if execution else False,
                stage="built",
            )

        useful_records = [item for item in _listify(result.get("useful_candidate_records")) if isinstance(item, dict)]
        checkpoint_candidates: List[Dict[str, Any]] = []
        if self.enable_checkpoint_guided_injection:
            try:
                events = _runtime_oracle_events(self.runtime)
                checkpoint_result = self.checkpoint_guided_runtime.update_and_build(
                    round_id=round_id,
                    events=events,
                    tfi_candidates=list(candidate_records) + list(useful_records),
                    recent_failed=[],
                )
                checkpoint_candidates = [
                    item for item in _listify(checkpoint_result.get("candidates")) if isinstance(item, dict)
                ]
                self.checkpoint_guided_built_count += len(checkpoint_candidates)
                self.candidate_built_count += len(checkpoint_candidates)
                for item in checkpoint_candidates:
                    source = self._candidate_source(item, result)
                    self._count_guided_source(source)
                    self._append_guided_candidate_record(
                        round_id=round_id,
                        candidate=item,
                        source=source,
                        executed=False,
                        injected=False,
                        execution_result={},
                        useful=bool(item.get("useful")),
                        stage="checkpoint_built",
                    )
                useful_records.extend(checkpoint_candidates)
                if checkpoint_candidates:
                    self.logger.info(
                        "[CHECKPOINT_GUIDED] round=%s built=%s next_missing=%s trigger=%s",
                        round_id,
                        len(checkpoint_candidates),
                        checkpoint_result.get("progress_snapshot", {}).get("next_missing_checkpoint"),
                        checkpoint_result.get("trigger_reason") or "none",
                    )
            except Exception as exc:
                self.last_error = (self.last_error + ";" if self.last_error else "") + "checkpoint_guided_error: {}".format(str(exc))
                self.logger.warning("[CHECKPOINT_GUIDED] round=%s status=error reason=%s", round_id, exc)
        if self.enable_soft_injection:
            self.buffer.add_candidates(useful_records)
            duplicate_count = int(self.buffer.last_pop_duplicate_count or 0)
            self.duplicate_skipped_count += duplicate_count
            for item in useful_records:
                useful_reason = _text(item.get("useful_reason")).strip() or "unknown"
                self.useful_by_reason[useful_reason] = int(self.useful_by_reason.get(useful_reason, 0)) + 1
                self.logger.info(
                    "[TFI_SOFT_CAND] candidate_id=%s scope_id=%s candidate_origin=%s generated_under_tfi_constraint=%s from_local_candidate_gap=%s useful_reason=%s path_progress_delta=%s frontier_touched=%s sink_touched=%s priority=low",
                    _text(item.get("candidate_id")).strip() or "UNKNOWN",
                    _text(item.get("scope_id")).strip() or "UNKNOWN",
                    _text(item.get("candidate_origin")).strip() or _text(item.get("candidate_source_used")).strip() or "unknown",
                    int(bool(item.get("generated_under_tfi_constraint"))),
                    int(bool(item.get("metadata", {}).get("from_local_candidate_gap"))) if isinstance(item.get("metadata"), dict) else 0,
                    _text(item.get("useful_reason")).strip() or "unknown",
                    _text(item.get("path_progress_delta")).strip() or "0",
                    int(bool(item.get("frontier_touched"))),
                    int(bool(item.get("sink_touched"))),
                )
            self.logger.info(
                "[TFI_AGENT] round=%s tasks_seen=%s candidate_gap_events_seen=%s guidance_generated=%s candidates_built=%s candidates_executed=%s useful_candidates=%s agent_called=%s llm_called=%s fallback_used=%s skipped_reason=%s errors=%s",
                round_id,
                int(result.get("tasks_seen") or 0),
                int(result.get("candidate_gap_events_seen") or 0),
                int(result.get("guidance_generated_count") or result.get("candidate_gap_guidance_count") or 0),
                int(result.get("candidate_built_count") or 0),
                int(result.get("candidate_executed_count") or 0),
                len(useful_records),
                int(bool(result.get("agent_called"))),
                int(bool(result.get("llm_called"))),
                int(bool(result.get("fallback_used"))),
                _text(result.get("skip_reason")).strip() or "none",
                self.last_error or "none",
            )
            self.logger.info(
                "[TFI_SOFT_INJECT] round=%s buffer_size=%s injected_count=%s skipped_duplicate_count=%s reason=%s",
                round_id,
                len(self.buffer),
                0,
                duplicate_count,
                "buffer_add",
            )
            injected = self._inject_from_buffer(round_id, population)
            if injected >= 0:
                self.soft_injected_count += injected
            self._write_guided_summary()
        else:
            self.logger.info(
                "[TFI_AGENT] round=%s tasks_seen=%s candidate_gap_events_seen=%s guidance_generated=%s candidates_built=%s candidates_executed=%s useful_candidates=%s agent_called=%s llm_called=%s fallback_used=%s skipped_reason=%s errors=%s",
                round_id,
                int(result.get("tasks_seen") or 0),
                int(result.get("candidate_gap_events_seen") or 0),
                int(result.get("guidance_generated_count") or result.get("candidate_gap_guidance_count") or 0),
                int(result.get("candidate_built_count") or 0),
                int(result.get("candidate_executed_count") or 0),
                len(useful_records),
                int(bool(result.get("agent_called"))),
                int(bool(result.get("llm_called"))),
                int(bool(result.get("fallback_used"))),
                _text(result.get("skip_reason")).strip() or "none",
                self.last_error or "none",
            )

    def finalize(self, population, engine):
        try:
            summary = self.build_summary()
            summary_path = getattr(self.runtime, "summary_path", "")
            if summary_path:
                with open(summary_path, "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, ensure_ascii=True, indent=2)
            env = getattr(engine, "env", None)
            if env is not None and isinstance(getattr(env, "results", None), dict):
                env.results["target_focus_agent_runtime"] = summary
            self._write_guided_summary(summary)
        except Exception:
            pass

    def build_summary(self) -> Dict[str, Any]:
        planning_only = bool(getattr(self.runtime, "tfi_planning_only", False))
        injection_enabled = bool(self.enable_guided_candidate_injection or self.enable_checkpoint_guided_injection)
        lifecycle_status = (
            "planning_only"
            if planning_only and not injection_enabled
            else (
                "injection_enabled"
                if injection_enabled
                else "injection_disabled"
            )
        )
        return {
            "enabled": True,
            "agent_enabled": bool(self.enable_agent),
            "tfi_planning_only": bool(planning_only),
            "candidate_lifecycle_config_status": lifecycle_status,
            "soft_injection_enabled": bool(self.enable_soft_injection),
            "guided_candidate_injection_enabled": bool(self.enable_guided_candidate_injection),
            "checkpoint_guided_injection_enabled": bool(self.enable_checkpoint_guided_injection),
            "guided_injection_confidence_threshold": float(self.guided_injection_confidence_threshold),
            "isolated_pilot_enabled": bool(getattr(self.runtime, "enable_isolated_pilot", False)),
            "agent_called_count": int(self.agent_called_count),
            "agent_error_count": int(self.agent_error_count),
            "candidate_gap_event_count": int(self.candidate_gap_event_count),
            "candidate_gap_guidance_count": int(self.candidate_gap_guidance_count),
            "guidance_generated_count": int(self.guidance_generated_count),
            "fuzzer_generated_candidate_count": int(self.fuzzer_generated_candidate_count),
            "candidate_built_count": int(self.candidate_built_count),
            "candidate_injected_count": int(self.candidate_injected_count),
            "candidate_executed_count": int(self.candidate_executed_count),
            "useful_candidate_count": int(self.useful_candidate_count),
            "useful_by_reason": dict(sorted(self.useful_by_reason.items())),
            "soft_injected_count": int(self.soft_injected_count),
            "duplicate_skipped_count": int(self.duplicate_skipped_count),
            "guided_candidate_count": int(self.guided_candidate_count),
            "agent_guided_candidate_count": int(self.agent_guided_candidate_count),
            "tfi_guided_candidate_count": int(self.tfi_guided_candidate_count),
            "ksi_guided_candidate_count": int(self.ksi_guided_candidate_count),
            "gei_guided_candidate_count": int(self.gei_guided_candidate_count),
            "checkpoint_guided_built_count": int(self.checkpoint_guided_built_count),
            "checkpoint_guided_injected_count": int(self.checkpoint_guided_injected_count),
            "checkpoint_guided_executed_count": int(self.checkpoint_guided_executed_count),
            "checkpoint_guided_medium_useful_count": int(self.checkpoint_guided_medium_useful_count),
            "checkpoint_guided_strong_useful_count": int(self.checkpoint_guided_strong_useful_count),
            "checkpoint_guided_checkpoint_delta_count": int(self.checkpoint_guided_checkpoint_delta_count),
            "checkpoint_guided_oracle_level_delta_count": int(self.checkpoint_guided_oracle_level_delta_count),
            "checkpoint_guided_summary": (
                self.checkpoint_guided_runtime.build_summary()
                if self.enable_checkpoint_guided_injection and self.checkpoint_guided_runtime is not None
                else None
            ),
            "fallback_used": bool(self.fallback_used_count > 0),
            "max_injected_per_round": int(self.max_injected_per_round),
            "injection_policy": self.injection_policy,
            "bypasses_normal_population_order": bool(self.bypasses_normal_population_order),
            "target_focus_runtime_error_count": int(self.agent_error_count),
            "task_build_attempted": bool(getattr(self.runtime, "last_task_build_result", {})),
            "task_build_status": _text(getattr(self.runtime, "last_task_build_result", {}).get("status")).strip() or None,
            "task_count": int(getattr(self.runtime, "last_task_build_result", {}).get("task_count") or 0),
            "task_build_reason": _text(getattr(self.runtime, "last_task_build_result", {}).get("reason")).strip() or None,
            "no_main_search_replacement": True,
            "no_generator_mutation_crossover_selection_change": True,
            "llm_called": bool(self.llm_called_count > 0),
            "rag_used": False,
            "guided_candidates_path": self.guided_candidates_path or None,
            "guided_injection_summary_path": self.guided_summary_path or None,
        }

    def _candidate_source(self, candidate: Dict[str, Any], result: Dict[str, Any]) -> str:
        source_text = " ".join(
            _text(value).lower()
            for value in (
                candidate.get("source"),
                candidate.get("candidate_origin"),
                candidate.get("candidate_source_used"),
                result.get("candidate_origin"),
            )
        )
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        if "gei" in source_text:
            return "gei_side_path"
        if "checkpoint_guided" in source_text and "agent" in source_text:
            return "agent_initial_path"
        if "checkpoint_guided" in source_text:
            return "tfi_guidance"
        if "ksi" in source_text or _clean_list(candidate.get("expected_blockers")):
            return "ksi_blocker_feedback"
        if "revision" in source_text:
            return "agent_revision"
        if (
            "llm" in source_text
            or bool(metadata.get("agent_guided"))
            or (_text(candidate.get("candidate_source_used")).strip() == "tfi_agent_llm")
        ):
            return "agent_initial_path"
        return "tfi_guidance"

    def _count_guided_source(self, source: str) -> None:
        self.guided_candidate_count += 1
        if source == "agent_initial_path" or source == "agent_revision":
            self.agent_guided_candidate_count += 1
        elif source == "ksi_blocker_feedback":
            self.ksi_guided_candidate_count += 1
        elif source == "gei_side_path":
            self.gei_guided_candidate_count += 1
        else:
            self.tfi_guided_candidate_count += 1

    def _candidate_sequence(self, candidate: Dict[str, Any]) -> List[str]:
        for field in ("sequence_skeleton", "candidate_sequence", "guided_sequence", "required_prefix"):
            value = candidate.get(field)
            cleaned = _clean_list(value)
            if cleaned:
                return cleaned
        return []

    def _append_guided_candidate_record(
        self,
        *,
        round_id: int,
        candidate: Dict[str, Any],
        source: str,
        executed: bool,
        injected: bool,
        execution_result: Optional[Dict[str, Any]],
        useful: bool,
        stage: str,
        injected_rank: Optional[int] = None,
        injection_policy: Optional[str] = None,
        bypasses_normal_population_order: bool = False,
    ) -> None:
        if not self.guided_candidates_path:
            return
        binding_choices = candidate.get("binding_choices") if isinstance(candidate.get("binding_choices"), dict) else {}
        execution_result = execution_result if isinstance(execution_result, dict) else {}
        payload = {
            "candidate_id": _text(candidate.get("candidate_id")).strip(),
            "source": source,
            "target_oracle_family": candidate.get("target_oracle_family"),
            "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
            "candidate_goal": candidate.get("candidate_goal"),
            "target_function": _text(candidate.get("target_function")).strip(),
            "sequence_skeleton": self._candidate_sequence(candidate),
            "sender_constraints": (
                _text(binding_choices.get("sender_policy_used")).strip()
                or _text(candidate.get("sender_policy")).strip()
                or None
            ),
            "value_constraints": (
                _text(binding_choices.get("value_policy_used")).strip()
                or _text(candidate.get("value_policy")).strip()
                or None
            ),
            "args_constraints": (
                _text(binding_choices.get("args_policy_used")).strip()
                or _text(candidate.get("argument_policy")).strip()
                or None
            ),
            "injected_round": int(round_id) if injected else None,
            "injected_rank": injected_rank,
            "injection_policy": injection_policy,
            "bypasses_normal_population_order": bool(bypasses_normal_population_order),
            "executed": bool(executed),
            "injected": bool(injected),
            "execution_result": {
                "execution_status": _text(execution_result.get("execution_status") or execution_result.get("status")).strip(),
                "useful_reason": _text(execution_result.get("useful_reason")).strip(),
                "frontier_touched": bool(execution_result.get("frontier_touched")),
                "sink_touched": bool(execution_result.get("sink_touched")),
                "sink_reverted": bool(execution_result.get("sink_reverted")),
            },
            "oracle_level_delta": execution_result.get("oracle_level_delta"),
            "vspc_delta": execution_result.get("vspc_delta"),
            "path_progress_delta": execution_result.get("path_progress_delta", candidate.get("path_progress_delta")),
            "materialization_status": candidate.get("materialization_status"),
            "param_sources": candidate.get("param_sources"),
            "rejected_reason": candidate.get("rejected_reason"),
            "llm_hallucination_detected": candidate.get("llm_hallucination_detected"),
            "function_not_in_abi": candidate.get("function_not_in_abi"),
            "arg_type_valid": candidate.get("arg_type_valid"),
            "prestate_supported": candidate.get("prestate_supported"),
            "useful": bool(useful),
            "stage": stage,
        }
        try:
            with open(self.guided_candidates_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def _write_guided_summary(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if not self.guided_summary_path:
            return
        payload = dict(summary) if isinstance(summary, dict) else self.build_summary()
        try:
            os.makedirs(os.path.dirname(self.guided_summary_path), exist_ok=True)
            with open(self.guided_summary_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def _maybe_enable_runtime_path_tracking(self, engine: Any, task_build: Dict[str, Any], round_id: int) -> None:
        env = getattr(engine, "env", None)
        if env is None:
            return
        tracker = getattr(env, "path_progress_tracker", None)
        if tracker is not None and bool(getattr(tracker, "enabled", False)):
            return
        path_plan_path = _text(task_build.get("path_plan_path")).strip()
        if not path_plan_path or not os.path.exists(path_plan_path):
            return
        try:
            path_plan = load_path_plan(path_plan_path)
            if path_plan.is_empty():
                return
            new_tracker = PathProgressTracker(
                path_plan=path_plan,
                output_dir=getattr(self.runtime, "output_dir", ""),
                enabled=True,
            )
            if not getattr(new_tracker, "enabled", False):
                return
            env.path_progress_tracker = new_tracker
            if hasattr(env, "args"):
                try:
                    env.args.path_plan = path_plan_path
                    env.args.path_plan_data = path_plan
                    env.args.enable_path_validation = True
                except Exception:
                    pass
            self.force_retry_round = int(round_id) + 1
        except Exception:
            return

    def _inject_from_buffer(self, round_id: int, population: Any) -> int:
        candidates = self.buffer.pop_for_round(limit=self.max_per_round)
        if not candidates:
            self.logger.info(
                "[TFI_SOFT_INJECT] round=%s buffer_size=%s injected_count=0 skipped_duplicate_count=0 reason=buffer_empty",
                round_id,
                len(self.buffer),
            )
            return 0
        injector = getattr(self.runtime, "repair_pilot_injector", None)
        if injector is None:
            self.logger.info(
                "[TFI_SOFT_INJECT] round=%s buffer_size=%s injected_count=0 skipped_duplicate_count=0 reason=soft_injection_pending_no_seed_queue_api",
                round_id,
                len(self.buffer),
            )
            return 0
        try:
            inject_result = injector.inject_candidates(candidates, population=population, generation=round_id)
        except Exception as exc:
            self.logger.warning(
                "[TFI_SOFT_INJECT] round=%s buffer_size=%s injected_count=0 skipped_duplicate_count=0 reason=%s",
                round_id,
                len(self.buffer),
                "soft_injection_error: {}".format(str(exc)),
            )
            return 0
        if isinstance(inject_result, dict):
            injected_count = int(inject_result.get("injected_count") or 0)
            injected_candidate_ids = [
                _text(item).strip()
                for item in list(inject_result.get("injected_candidate_ids") or [])
                if _text(item).strip()
            ]
            insertion_policy = _text(inject_result.get("insertion_policy")).strip() or "append"
            bypasses_normal_order = bool(inject_result.get("bypasses_normal_population_order"))
        else:
            injected_count = int(inject_result or 0)
            injected_candidate_ids = []
            insertion_policy = "append"
            bypasses_normal_order = False
        self.injection_policy = insertion_policy
        self.bypasses_normal_population_order = bool(bypasses_normal_order)
        injected_candidate_id_set = set(injected_candidate_ids)
        injected_rank_map: Dict[str, int] = {}
        for rank, candidate_id in enumerate(injected_candidate_ids, start=1):
            if candidate_id not in injected_rank_map:
                injected_rank_map[candidate_id] = rank
        self.logger.info(
            "[TFI_SOFT_INJECT] round=%s buffer_size=%s injected_count=%s skipped_duplicate_count=0 reason=%s bypasses_normal_population_order=%s",
            round_id,
            len(self.buffer),
            injected_count,
            "population_{}".format(insertion_policy) if injected_count > 0 else "no_eligible_individuals",
            bool(bypasses_normal_order),
        )
        self.candidate_injected_count += max(0, int(injected_count))
        checkpoint_candidates = [
            item for item in candidates
            if isinstance(item, dict) and _text(item.get("candidate_origin")).strip() == "checkpoint_guided_injection_v2"
        ]
        if injected_candidate_id_set:
            checkpoint_injected_ids = [
                _text(item.get("candidate_id")).strip()
                for item in checkpoint_candidates
                if _text(item.get("candidate_id")).strip() in injected_candidate_id_set
            ]
            checkpoint_injected = len(checkpoint_injected_ids)
        else:
            checkpoint_injected_ids = []
            checkpoint_injected = min(len(checkpoint_candidates), max(0, int(injected_count)))
        if self.enable_checkpoint_guided_injection and self.checkpoint_guided_runtime is not None:
            self.checkpoint_guided_injected_count += checkpoint_injected
            self.checkpoint_guided_executed_count += checkpoint_injected
            try:
                self.checkpoint_guided_runtime.record_injection(
                    checkpoint_candidates,
                    injected_count=checkpoint_injected,
                    round_id=round_id,
                    injected_candidate_ids=checkpoint_injected_ids,
                )
                checkpoint_summary = self.checkpoint_guided_runtime.build_summary()
                self.checkpoint_guided_medium_useful_count = int(
                    checkpoint_summary.get("medium_useful_candidate_count") or 0
                )
                self.checkpoint_guided_strong_useful_count = int(
                    checkpoint_summary.get("strong_useful_candidate_count") or 0
                )
                self.checkpoint_guided_checkpoint_delta_count = int(
                    checkpoint_summary.get("checkpoint_delta_count") or 0
                )
                self.checkpoint_guided_oracle_level_delta_count = int(
                    checkpoint_summary.get("oracle_level_delta_count") or 0
                )
            except Exception:
                pass
        for index, candidate in enumerate(candidates, start=1):
            candidate_id = _text(candidate.get("candidate_id")).strip()
            injected = candidate_id in injected_candidate_id_set if injected_candidate_id_set else index <= int(injected_count)
            injected_rank = injected_rank_map.get(candidate_id) if injected_candidate_id_set else (index if injected else None)
            source = self._candidate_source(candidate, {})
            self._append_guided_candidate_record(
                round_id=round_id,
                candidate=candidate,
                source=source,
                executed=bool(candidate.get("useful")),
                injected=injected,
                execution_result=candidate,
                useful=bool(candidate.get("useful")),
                stage="injected" if injected else "injection_skipped",
                injected_rank=injected_rank,
                injection_policy=insertion_policy if injected else None,
                bypasses_normal_population_order=bool(bypasses_normal_order) if injected else False,
            )
        self.max_injected_per_round = max(self.max_injected_per_round, int(injected_count))
        return injected_count

    def _checkpoint_observe_only(self, round_id: int) -> None:
        if not self.enable_checkpoint_guided_injection or self.checkpoint_guided_runtime is None:
            return
        try:
            events = _runtime_oracle_events(self.runtime)
            observe_result = self.checkpoint_guided_runtime.observe(
                round_id=round_id,
                events=events,
            )
            if (
                isinstance(observe_result, dict)
                and int(observe_result.get("new_failure_signature_count") or 0) > 0
                and int(observe_result.get("pending_failure_repair_signature_count") or 0) > 0
            ):
                self.force_retry_round = int(round_id) + 1
        except Exception:
            pass


def _resolve_solc_path_from_dir(solc_dir: Optional[str]) -> Optional[str]:
    if not solc_dir:
        return None
    root = Path(solc_dir)
    if not root.exists():
        return None
    direct = root / "solc.exe"
    if direct.exists():
        return str(direct)
    versions = []
    for version_dir in root.glob("solc-v*"):
        candidate = version_dir / "solc.exe"
        if candidate.exists():
            versions.append((version_dir.name, candidate))
    if not versions:
        return None
    versions.sort(key=lambda item: item[0])
    return str(versions[-1][1])


def _load_forge_compile_support():
    support_path = Path(BASE_DIR) / "forge_compile_support.py"
    if not support_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("forge_compile_support", support_path)
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        return support
    except Exception:
        return None


def _version_tuple(value) -> Tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(value))
    if not match:
        return (0, 0, 0)
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def _compare_versions(left, right) -> int:
    left_tuple = _version_tuple(left)
    right_tuple = _version_tuple(right)
    if left_tuple < right_tuple:
        return -1
    if left_tuple > right_tuple:
        return 1
    return 0


def _caret_upper_bound(version_text: str) -> Tuple[int, int, int]:
    major, minor, patch = _version_tuple(version_text)
    if major > 0:
        return (major + 1, 0, 0)
    if minor > 0:
        return (0, minor + 1, 0)
    return (0, 0, patch + 1)


def _matches_single_constraint(version_value, token: str) -> bool:
    token = token.strip()
    if not token:
        return True
    for operator in (">=", "<=", ">", "<", "^", "="):
        if token.startswith(operator):
            target = token[len(operator):].strip()
            break
    else:
        operator = "="
        target = token

    if operator == "^":
        lower_ok = _compare_versions(version_value, target) >= 0
        upper_ok = _version_tuple(version_value) < _caret_upper_bound(target)
        return lower_ok and upper_ok
    if operator == ">=":
        return _compare_versions(version_value, target) >= 0
    if operator == "<=":
        return _compare_versions(version_value, target) <= 0
    if operator == ">":
        return _compare_versions(version_value, target) > 0
    if operator == "<":
        return _compare_versions(version_value, target) < 0
    return _compare_versions(version_value, target) == 0


def _pragma_matches(version_value, pragma_text: str) -> bool:
    clauses = [clause.strip() for clause in str(pragma_text or "").split("||") if clause.strip()]
    if not clauses:
        return True
    for clause in clauses:
        tokens = [token for token in re.split(r"[,\s]+", clause) if token]
        if all(_matches_single_constraint(version_value, token) for token in tokens):
            return True
    return False


def _select_solc_version_for_source(source_path: str) -> Optional[str]:
    try:
        source_text = Path(source_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    support = _load_forge_compile_support()
    if support is not None and hasattr(support, "candidate_solc_versions"):
        try:
            candidates = support.candidate_solc_versions(source_text)
            if candidates:
                return "v" + str(candidates[0])
        except Exception:
            pass
    match = re.search(r"pragma\s+solidity\s+([^;]+);", source_text)
    if not match:
        return None
    pragma_text = match.group(1).strip()
    try:
        installed_versions = list(solcx.get_installed_solc_versions())
    except Exception:
        installed_versions = []
    installed_versions.sort(key=_version_tuple, reverse=True)
    for version_value in installed_versions:
        if _pragma_matches(version_value, pragma_text):
            return "v" + str(version_value)
    return None


def _resolve_solc_executable(solc_dir: Optional[str], solc_version: Optional[str]) -> Optional[str]:
    if solc_dir:
        root = Path(solc_dir)
    else:
        try:
            root = Path(solcx.install.get_solcx_install_folder())
        except Exception:
            root = Path.home() / ".solcx"
    if not root.exists():
        return None
    if solc_version:
        normalized = str(solc_version)
        if not normalized.startswith("v"):
            normalized = "v" + normalized
        normalized = normalized.split("+", 1)[0]
        candidate = root / ("solc-" + normalized) / "solc.exe"
        if candidate.exists():
            return str(candidate)
    return _resolve_solc_path_from_dir(str(root))


def _infer_evm_version_for_solc(solc_version: Optional[str]) -> str:
    major, minor, patch = _version_tuple(solc_version or "")
    if (major, minor, patch) >= (0, 5, 14):
        return "istanbul"
    return "petersburg"


def _apply_disabled_stage_summary(results_path: Optional[str], contract_name: str) -> None:
    if not results_path:
        return
    summary = StageRuntime.disabled_summary()
    try:
        if results_path.lower().endswith(".json"):
            if not os.path.exists(results_path):
                return
            with open(results_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if contract_name in payload and isinstance(payload[contract_name], dict):
                payload[contract_name].update(summary)
            with open(results_path, "w", encoding="utf-8") as file:
                json.dump(payload, file)
            return

        result_file = os.path.join(results_path, os.path.splitext(os.path.basename(contract_name))[0] + ".json")
        if not os.path.exists(result_file):
            return
        with open(result_file, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if contract_name in payload and isinstance(payload[contract_name], dict):
            payload[contract_name].update(summary)
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(payload, file)
    except Exception:
        pass


def _refresh_result_file(results_path: Optional[str], contract_name: str, results_payload: Optional[Dict[str, Any]]) -> None:
    if not results_path or not isinstance(results_payload, dict):
        return
    payload = {}
    if results_path.lower().endswith(".json"):
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        payload[contract_name] = results_payload
        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        return
    result_file = os.path.join(results_path, os.path.splitext(os.path.basename(contract_name))[0] + ".json")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as file:
            payload = json.load(file)
    payload[contract_name] = results_payload
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def _resolve_run_output_dir(results_path: Optional[str]) -> str:
    if results_path:
        if results_path.lower().endswith(".json"):
            output_dir = os.path.dirname(os.path.abspath(results_path))
        else:
            output_dir = os.path.abspath(results_path)
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    if not output_dir:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _should_write_path_validation_sidecar(results_path: Optional[str]) -> bool:
    return not (results_path and results_path.lower().endswith(".json"))


def _write_path_validation_summary(output_dir: str, contract_name: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["path_validation"] = summary
        contracts = payload.get("contracts")
        if not isinstance(contracts, dict):
            contracts = {}
        contracts[contract_name] = {"path_validation": summary}
        payload["contracts"] = contracts
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _write_ksi_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["ksi"] = summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _write_agent_feedback_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["agent_feedback"] = summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _write_gei_side_path_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["gei_side_path"] = summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _write_side_path_plan_export_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["side_path_plan_export"] = summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _write_soft_guidance_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["soft_guidance"] = summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _write_tfi_focus_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["tfi_focus"] = summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _write_agent_planner_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["agent_planner"] = summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _write_guided_sequence_pilot_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    if not output_dir:
        return
    summary_path = os.path.join(output_dir, "path_validation_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        payload["guided_sequence_pilot"] = summary
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _merge_ksi_summary(path_validation_summary: Optional[Dict[str, Any]], ksi_summary: Optional[Dict[str, Any]]) -> None:
    if not isinstance(path_validation_summary, dict) or not isinstance(ksi_summary, dict):
        return
    path_validation_summary["ksi_enabled"] = bool(ksi_summary.get("enabled", False))
    path_validation_summary["ksi_evidence_records"] = int(ksi_summary.get("ksi_evidence_records") or 0)
    path_validation_summary["ksi_evidence_path"] = ksi_summary.get("ksi_evidence_path")


def _merge_agent_feedback_summary(
    path_validation_summary: Optional[Dict[str, Any]],
    agent_feedback_summary: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(path_validation_summary, dict) or not isinstance(agent_feedback_summary, dict):
        return
    path_validation_summary["agent_feedback_enabled"] = bool(agent_feedback_summary.get("enabled", False))
    path_validation_summary["blocker_evidence_path"] = agent_feedback_summary.get("blocker_evidence_path")


def _merge_gei_side_path_summary(
    path_validation_summary: Optional[Dict[str, Any]],
    gei_side_path_summary: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(path_validation_summary, dict) or not isinstance(gei_side_path_summary, dict):
        return
    path_validation_summary["gei_side_path"] = gei_side_path_summary
    path_validation_summary["gei_core"] = gei_side_path_summary


def _merge_side_path_plan_export_summary(
    path_validation_summary: Optional[Dict[str, Any]],
    export_summary: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(path_validation_summary, dict) or not isinstance(export_summary, dict):
        return
    path_validation_summary["side_path_plan_export"] = export_summary


def _merge_soft_guidance_summary(
    path_validation_summary: Optional[Dict[str, Any]],
    soft_guidance_summary: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(path_validation_summary, dict) or not isinstance(soft_guidance_summary, dict):
        return
    path_validation_summary["soft_guidance"] = soft_guidance_summary


def _merge_tfi_focus_summary(
    path_validation_summary: Optional[Dict[str, Any]],
    tfi_focus_summary: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(path_validation_summary, dict) or not isinstance(tfi_focus_summary, dict):
        return
    path_validation_summary["tfi_focus"] = tfi_focus_summary


def _merge_agent_planner_summary(
    path_validation_summary: Optional[Dict[str, Any]],
    agent_planner_summary: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(path_validation_summary, dict) or not isinstance(agent_planner_summary, dict):
        return
    path_validation_summary["agent_planner"] = agent_planner_summary


def _merge_guided_sequence_pilot_summary(
    path_validation_summary: Optional[Dict[str, Any]],
    repair_pilot_summary: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(path_validation_summary, dict) or not isinstance(repair_pilot_summary, dict):
        return
    path_validation_summary["guided_sequence_pilot"] = repair_pilot_summary


class Fuzzer:
    def __init__(self, contract_name, abi, deployment_bytecode, runtime_bytecode, test_instrumented_evm,
                 blockchain_state, solver, args, seed, source_map=None, whole_compile_info=None):
        global logger

        logger = initialize_logger("Fuzzer  ")
        logger.title("Fuzzing contract %s", contract_name)

        cfg = ControlFlowGraph()
        cfg.build(runtime_bytecode, settings.EVM_VERSION)

        self.contract_name = contract_name
        self.abi = abi
        self.interface, self.interface_mapper = get_interface_from_abi(abi)
        self.function_state_mutability = get_function_state_mutability_from_abi(abi)
        self.deployement_bytecode = deployment_bytecode
        self.blockchain_state = blockchain_state
        self.instrumented_evm = test_instrumented_evm
        self.solver = solver
        self.args = args
        # 璇ュ悎绾︿緷璧栫殑鍏朵粬鍚堢害
        self.depend_contracts = args.depend_contracts
        self.whole_compile_info = whole_compile_info

        # Get some overall metric on the code
        self.overall_pcs, self.overall_jumpis = get_pcs_and_jumpis(runtime_bytecode)

        # Initialize results
        self.results = {"errors": {}}

        # Initialize fuzzing environment
        self.env = FuzzingEnvironment(instrumented_evm=self.instrumented_evm,
                                      contract_name=self.contract_name,
                                      solver=self.solver,
                                      results=self.results,
                                      symbolic_taint_analyzer=SymbolicTaintAnalyzer(),
                                      detector_executor=DetectorExecutor(source_map,
                                                                         get_function_signature_mapping(abi)),
                                      interface=self.interface,
                                      overall_pcs=self.overall_pcs,
                                      overall_jumpis=self.overall_jumpis,
                                      len_overall_pcs_with_children=0,
                                      other_contracts=list(),
                                      args=args,
                                      seed=seed,
                                      cfg=cfg,
                                      abi=abi)
        self.env.gei60_state = _ensure_gei60_state(self.env, args)
        self.env.pure_fuzz_warmup_state = self.env.gei60_state
        _sync_gei60_results(self.env, self.env.gei60_state)
        init_func(args.source)  # 鍒濆鍖栧垎鏋愯法鍚堢害搴忓垪
        assert check_cross_init(), "璺ㄥ悎绾﹀垵濮嬪寲澶辫触"  # 妫€鏌ユ槸鍚﹀垵濮嬪寲鎴愬姛
        print("璺ㄥ悎绾﹀垵濮嬪寲鎴愬姛......")

    def deploy_depend_contracts(self):
        generators = []
        if self.whole_compile_info is None:
            logger.error("娌℃湁鎵惧埌缂栬瘧淇℃伅, 閫€鍑虹▼搴?")
            sys.exit(-1)
        for depend_contract in self.depend_contracts:
            # 寰楀埌杩欎釜鍚堢害鐨刬nterface
            depend_abi = self.whole_compile_info[depend_contract]['abi']
            interface, interface_mapper = get_interface_from_abi(depend_abi)
            function_state_mutability = get_function_state_mutability_from_abi(depend_abi)
            deployement_bytecode = self.whole_compile_info[depend_contract]['evm']['bytecode']['object']
            if "constructor" in interface:
                del interface['constructor']
            if "constructor" not in interface:
                result = time_local_execution(
                    self.env,
                    phase="setup_deploy",
                    tx_kind="dependency_contract_deploy",
                    func=lambda: self.instrumented_evm.deploy_contract(
                        self.instrumented_evm.accounts[0],
                        deployement_bytecode,
                    ),
                    metadata={"contract_name": depend_contract},
                )  # Deploy dependency contract.
                if result.is_error:
                    _write_deployment_diagnostic(
                        self.args,
                        depend_contract,
                        "dependency_contract_deploy",
                        getattr(self.instrumented_evm, "last_deploy_diagnostic", None),
                        getattr(result, "_error", None),
                    )
                    logger.error("Problem while deploying contract %s using account %s. Error message: %s",
                                 depend_contract, self.instrumented_evm.accounts[0], result._error)
                    # 杩欓噷鍘熸湰鏄€€鍑虹▼搴? 浣嗕緷璧栧悎绾﹂儴缃插け璐ュ彲浠ヤ笉閫€鍑? 涓嶅奖鍝嶄富绋嬪簭鐨勫ぇ閮ㄥ垎fuzz
                else:
                    contract_address = encode_hex(result.msg.storage_address)
                    self.instrumented_evm.accounts.append(contract_address)
                    self.env.nr_of_transactions += 1
                    logger.info(
                        "Dependency contract %s deployed at\t%s, created by %s",
                        depend_contract,
                        contract_address,
                        self.instrumented_evm.accounts[0],
                    )
                    # 瀛樺偍閮ㄧ讲淇℃伅
                    settings.TRANS_INFO[depend_contract] = contract_address
                    settings.DEPLOYED_CONTRACT_ADDRESS[depend_contract] = contract_address
                    generator = Generator(interface=interface, bytecode=deployement_bytecode,
                                          accounts=self.instrumented_evm.accounts, contract=contract_address,
                                          interface_mapper=interface_mapper, contract_name=depend_contract,
                                          sol_path=self.args.source,
                                          function_state_mutability=function_state_mutability)
                    generators.append(generator)
        return generators

    def run(self):
        contract_address = None
        run_started_at = time.time()
        fuzz_core_started_at = None
        fuzz_core_finished_at = None
        self.instrumented_evm.create_fake_accounts()
        try:
            start_resource_sampler(self.env, sample_interval_sec=0.25)
        except Exception:
            pass
        start_experiment_timing(self.env, self.args, start_time=run_started_at)
        refresh_timing_results(self.env)
        if self.args.cross_contract == 1:
            generators = self.deploy_depend_contracts()
        else:
            generators = []
        if self.args.source:
            for transaction in self.blockchain_state:  # 濡傛灉鍧楀唴鏈夊垵濮嬩簨鍔? 鎵ц浠栦滑
                if transaction['from'].lower() not in self.instrumented_evm.accounts:
                    self.instrumented_evm.accounts.append(
                        self.instrumented_evm.create_fake_account(transaction['from']))

                if not transaction['to']:
                    result = time_local_execution(
                        self.env,
                        phase="setup_deploy",
                        tx_kind="blockchain_state_constructor_deploy",
                        func=lambda transaction=transaction: self.instrumented_evm.deploy_contract(
                            transaction['from'],
                            transaction['input'],
                            int(transaction['value']),
                            int(transaction['gas']),
                            int(transaction['gasPrice']),
                        ),
                        metadata={"contract_name": self.contract_name},
                    )
                    if result.is_error:
                        _write_deployment_diagnostic(
                            self.args,
                            self.contract_name,
                            "blockchain_state_constructor_deploy",
                            getattr(self.instrumented_evm, "last_deploy_diagnostic", None),
                            getattr(result, "_error", None),
                        )
                        logger.error("Problem while deploying contract %s using account %s. Error message: %s",
                                     self.contract_name, transaction['from'], result._error)
                        sys.exit(-2)
                    else:
                        contract_address = encode_hex(result.msg.storage_address)
                        self.instrumented_evm.accounts.append(contract_address)
                        self.env.nr_of_transactions += 1
                        logger.debug("Contract deployed at %s", contract_address)
                        self.env.other_contracts.append(to_canonical_address(contract_address))
                        cc, _ = get_pcs_and_jumpis(
                            self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex())
                        self.env.len_overall_pcs_with_children += len(cc)
                else:
                    input = {}
                    input["block"] = {}
                    input["transaction"] = {
                        "from": transaction["from"],
                        "to": transaction["to"],
                        "gaslimit": int(transaction["gas"]),
                        "value": int(transaction["value"]),
                        "data": transaction["input"]
                    }
                    input["global_state"] = {}
                    out = time_local_execution(
                        self.env,
                        phase="setup_deploy",
                        tx_kind="blockchain_state_transaction",
                        func=lambda input=input, transaction=transaction: self.instrumented_evm.deploy_transaction(
                            input,
                            int(transaction["gasPrice"]),
                        ),
                        metadata={"contract_name": self.contract_name},
                    )

            if "constructor" in self.interface:
                del self.interface["constructor"]

            if not contract_address:
                if "constructor" not in self.interface:
                    result = time_local_execution(
                        self.env,
                        phase="setup_deploy",
                        tx_kind="target_contract_deploy",
                        func=lambda: self.instrumented_evm.deploy_contract(
                            self.instrumented_evm.accounts[0],
                            self.deployement_bytecode,
                            deploy_args=self.args.constructor_args,
                            deploy_mode=settings.CROSS_INIT_MODE,
                        ),
                        metadata={"contract_name": self.contract_name},
                    )
                    if result.is_error:
                        _write_deployment_diagnostic(
                            self.args,
                            self.contract_name,
                            "target_contract_deploy",
                            getattr(self.instrumented_evm, "last_deploy_diagnostic", None),
                            getattr(result, "_error", None),
                        )
                        logger.error("Problem while deploying contract %s using account %s. Error message: %s",
                                     self.contract_name, self.instrumented_evm.accounts[0], result._error)
                        sys.exit(-2)
                    else:
                        contract_address = encode_hex(result.msg.storage_address)
                        self.instrumented_evm.accounts.append(contract_address)
                        self.env.nr_of_transactions += 1
                        logger.info("涓籆ontract deployed at %s", contract_address)
                        # 灏嗗悎绾︾殑閮ㄧ讲鎯呭喌瀛樺偍璧锋潵
                        settings.TRANS_INFO[self.contract_name] = contract_address
                        settings.DEPLOYED_CONTRACT_ADDRESS[self.contract_name] = contract_address

            if contract_address in self.instrumented_evm.accounts:
                self.instrumented_evm.accounts.remove(contract_address)

            self.env.overall_pcs, self.env.overall_jumpis = get_pcs_and_jumpis(
                self.instrumented_evm.get_code(to_canonical_address(contract_address)).hex())

        if self.args.abi:
            contract_address = self.args.contract

        self.instrumented_evm.create_snapshot()  # 閮ㄧ讲鎵€鏈夊悎绾﹀悗, 鍒涘缓蹇収

        generator = Generator(interface=self.interface,
                              bytecode=self.deployement_bytecode,
                              accounts=self.instrumented_evm.accounts,
                              contract=contract_address,
                              other_generators=generators,
                              interface_mapper=self.interface_mapper,
                              contract_name=self.contract_name,
                              sol_path=self.args.source,
                              function_state_mutability=self.function_state_mutability)

        # update the generator with the interface of the other contracts
        all_generators = [generator] + generators
        for gen in generators:
            gen.update_other_generators(all_generators, generator.total_interface_mapper)
        self.env.rq1_sender_pool_generators = all_generators
        _update_rq1_gei_sender_pool(self.env, logger=logger)

        # Create initial population
        size = 2 * len(self.interface)
        population = Population(indv_template=Individual(generator=generator, other_generators=generators),
                                indv_generator=generator,
                                size=settings.POPULATION_SIZE if settings.POPULATION_SIZE else size,
                                other_generators=generators).init(init_seed=False)

        # Create genetic operators
        if self.args.data_dependency:
            selection = DataDependencyLinearRankingSelection(env=self.env)  # 鍩轰簬Read After Write鍏崇郴鐨勭瀛愰€夋嫨
            crossover = DataDependencyCrossover(pc=settings.PROBABILITY_CROSSOVER, env=self.env)  # 鍩轰簬鏁版嵁娴佺殑浜ゅ弶绛栫暐
            mutation = Mutation(pm=settings.PROBABILITY_MUTATION)
        else:
            selection = LinearRankingSelection()
            crossover = Crossover(pc=settings.PROBABILITY_CROSSOVER)
            mutation = Mutation(pm=settings.PROBABILITY_MUTATION)

# Create and run our evolutionary fuzzing engine
        engine = EvolutionaryFuzzingEngine(population=population, selection=selection, crossover=crossover,
                                           mutation=mutation,
                                           mapping=get_function_signature_mapping(self.env.abi))
        engine.env = self.env
        engine.analysis.append(RQ1GEISenderPoolRestrictionAnalysis(logger))
        engine.fitness_register(lambda x: fitness_function(x, self.env))  # 璁＄畻x鐨勯€傚簲搴? x鏄痠ndividual
        soft_guidance = None
        guided_sequence_pilot = None
        target_focus_soft_injection_analysis = None
        run_output_dir = getattr(self.args, "run_output_dir", None) or _resolve_run_output_dir(self.args.results)
        runtime_account_snapshot = _rq1_runtime_account_snapshot(self.env, self.instrumented_evm)
        runtime_deployer_account = _text((runtime_account_snapshot or [None])[0]).strip()
        runtime_llm_provider = None
        if getattr(self.args, "agent_use_llm", False) and (
            getattr(self.args, "enable_tfi_agent", False)
            or getattr(self.args, "enable_ksi_llm_feedback", False)
            or getattr(self.args, "enable_agent_effect_repair", False)
            or getattr(self.args, "enable_failure_aware_agent_repair", False)
            or getattr(self.args, "failure_repair_use_agent", False)
        ):
            runtime_llm_provider = _build_llm_provider_from_args(self.args)
        if (
            getattr(self.args, "fresh_agent_run", False)
            or getattr(self.args, "enable_tfi_agent", False)
            or getattr(self.args, "enable_ksi_llm_feedback", False)
            or getattr(self.args, "agent_use_llm", False)
        ):
            self.env.fresh_agent_runtime_state = _default_agent_runtime_state(self.args, run_output_dir)
            self.env.fresh_agent_runtime_state["_env_ref"] = self.env
        if getattr(self.args, "ignore_precomputed_paths", False):
            stale_path_plan = _text(getattr(self.args, "path_plan", None)).strip()
            if stale_path_plan and not _is_current_run_artifact(
                stale_path_plan,
                run_output_dir,
                getattr(self.args, "run_start_time", None),
            ):
                self.args.path_plan = None
                self.args.path_plan_data = PathPlan()
                self.args.enable_path_validation = False
        if getattr(self.args, "ignore_precomputed_tfi_artifacts", False):
            for field_name in ("tfi_materialization_tasks", "target_focus_candidate_scope", "repair_selected_sequences"):
                field_value = _text(getattr(self.args, field_name, None)).strip()
                if field_value and not _is_current_run_artifact(
                    field_value,
                    run_output_dir,
                    getattr(self.args, "run_start_time", None),
                ):
                    setattr(self.args, field_name, None)
        if getattr(self.args, "enable_soft_path_guidance", False):
            soft_guidance = SoftPathGuidance(
                path_plan=getattr(self.args, "path_plan_data", PathPlan()),
                output_dir=run_output_dir,
                enabled=True,
                ratio=min(float(getattr(self.args, "path_guidance_ratio", 0.10) or 0.0), 0.10),
                max_per_generation=int(getattr(self.args, "path_guidance_max_per_gen", 5) or 5),
                blocker_evidence_path=getattr(self.args, "soft_guidance_blocker_evidence", None),
                enable_side_path_guidance=bool(
                    getattr(self.args, "enable_ksi_evidence", False)
                    or getattr(self.args, "enable_gei_core", False)
                    or getattr(self.args, "enable_gei_side_path", False)
                ),
                dry_run=bool(getattr(self.args, "soft_guidance_dry_run", False)),
            )
            self.env.soft_guidance = soft_guidance
            engine.analysis.append(SoftGuidanceAnalysis(soft_guidance))
        if getattr(self.args, "enable_guided_sequence_pilot", False):
            selected_sequences_path = getattr(self.args, "repair_selected_sequences", None)
            if selected_sequences_path and os.path.exists(selected_sequences_path):
                guided_sequence_pilot = RepairSequencePilot(
                    selected_guided_sequences_path=selected_sequences_path,
                    max_injected=int(getattr(self.args, "repair_pilot_top_k", 5) or 5),
                    dry_run=bool(getattr(self.args, "repair_pilot_dry_run", False)),
                    output_dir=getattr(self.args, "repair_pilot_output_dir", None) or run_output_dir,
                    concretization_version=getattr(self.args, "repair_pilot_concretization", "v1"),
                    insertion_policy=getattr(self.args, "repair_pilot_insertion_policy", "append"),
                )
                self.env.guided_sequence_pilot = guided_sequence_pilot
                engine.analysis.append(RepairSequencePilotAnalysis(guided_sequence_pilot))
            else:
                self.env.results["guided_sequence_pilot"] = {
                    "enabled": False,
                    "dry_run": bool(getattr(self.args, "repair_pilot_dry_run", False)),
                    "selected_path": selected_sequences_path,
                    "reason": "repair_selected_sequences_missing",
                }
        if (
            getattr(self.args, "enable_tfi_agent", False)
            or getattr(self.args, "enable_target_focus_soft_injection", False)
            or getattr(self.args, "enable_tfi_isolated_pilot", False)
            or getattr(self.args, "enable_guided_candidate_injection", False)
            or getattr(self.args, "enable_checkpoint_guided_injection", False)
        ):
            try:
                runtime_static_summary = _compact_static_summary_for_agent(
                    _build_slither_static_summary(
                        source_path=getattr(self.args, "source", None),
                        contract_name=self.contract_name,
                        solc_dir=getattr(self.args, "solc_dir", None),
                        solc_version=getattr(self.args, "solc_version", None),
                    )
                )
                checkpoint_guided_enabled = bool(getattr(self.args, "enable_checkpoint_guided_injection", False))
                guided_injection_enabled = bool(
                    getattr(self.args, "enable_guided_candidate_injection", False)
                    or checkpoint_guided_enabled
                )
                active_agent_llm = bool(
                    getattr(self.args, "agent_use_llm", False)
                    and (
                        getattr(self.args, "enable_tfi_agent", False)
                        or getattr(self.args, "enable_ksi_llm_feedback", False)
                        or getattr(self.args, "enable_agent_effect_repair", False)
                        or getattr(self.args, "enable_failure_aware_agent_repair", False)
                        or getattr(self.args, "failure_repair_use_agent", False)
                    )
                )
                active_soft_injection = bool(
                    getattr(self.args, "enable_target_focus_soft_injection", False)
                    or guided_injection_enabled
                    or checkpoint_guided_enabled
                )
                active_isolated_pilot = bool(
                    getattr(self.args, "enable_tfi_isolated_pilot", False)
                    or guided_injection_enabled
                    or checkpoint_guided_enabled
                )
                guided_budget = int(
                    (getattr(self.args, "checkpoint_guided_max_per_round", 0) if checkpoint_guided_enabled else 0)
                    or getattr(self.args, "guided_injection_budget_per_round", 0)
                    or getattr(self.args, "target_focus_soft_injection_max_per_round", 3)
                    or 3
                )
                guided_max_total = int(
                    (max(1, guided_budget) * max(1, int(getattr(self.args, "checkpoint_guided_dedup_window", 100) or 100))
                    if checkpoint_guided_enabled else 0)
                    or getattr(self.args, "guided_injection_max_total", 0)
                    or getattr(self.args, "target_focus_soft_injection_buffer_size", 64)
                    or 64
                )
                target_focus_runtime = TFIAgentRuntime(
                    output_dir=run_output_dir,
                    materialization_tasks_path=getattr(self.args, "tfi_materialization_tasks", None),
                    candidate_scope_path=getattr(self.args, "target_focus_candidate_scope", None),
                    contract_name=self.contract_name,
                    source_path=getattr(self.args, "source", None),
                    gas_limit=getattr(self.args, "gas_limit", None),
                    max_tasks_per_round=int(getattr(self.args, "tfi_agent_max_tasks_per_round", 4) or 4),
                    max_candidates_per_task=int(getattr(self.args, "tfi_agent_max_candidates_per_task", 2) or 2),
                    max_total_candidates=max(
                        guided_max_total if guided_injection_enabled else (
                            int(getattr(self.args, "tfi_agent_max_tasks_per_round", 4) or 4)
                            * int(getattr(self.args, "tfi_agent_max_candidates_per_task", 2) or 2)
                        ),
                        1,
                    ),
                    enable_agent=active_agent_llm,
                    enable_isolated_pilot=active_isolated_pilot,
                    llm_provider=runtime_llm_provider,
                    debug_artifacts=bool(getattr(self.args, "enable_tfi_debug_artifacts", False)),
                    fresh_run=bool(getattr(self.args, "fresh_agent_run", False)),
                    ignore_precomputed_paths=bool(getattr(self.args, "ignore_precomputed_paths", False)),
                    ignore_precomputed_tfi_artifacts=bool(getattr(self.args, "ignore_precomputed_tfi_artifacts", False)),
                    run_start_time=getattr(self.args, "run_start_time", None),
                    strict_llm_only=bool(
                        getattr(self.args, "fresh_agent_run", False)
                        and getattr(self.args, "enable_tfi_agent", False)
                        and getattr(self.args, "agent_use_llm", False)
                    ),
                    phase_cost_env=self.env,
                )
                try:
                    setattr(target_focus_runtime, "env", self.env)
                    setattr(target_focus_runtime, "tfi_planning_only", bool(getattr(self.args, "tfi_planning_only", False)))
                except Exception:
                    pass
                checkpoint_runtime = None
                if checkpoint_guided_enabled:
                    audit_spec = _load_audit_target_spec_from_args(self.args)
                    trigger_levels = [
                        item.strip()
                        for item in _text(getattr(self.args, "checkpoint_guided_trigger_levels", "L1,L2")).split(",")
                        if item.strip()
                    ]
                    checkpoint_runtime = CheckpointGuidedRuntime(
                        output_dir=run_output_dir,
                        oracle_family=_text(audit_spec.get("oracle_family")).strip(),
                        normalized_type=_text(audit_spec.get("normalized_type")).strip(),
                        finding_title=_text(audit_spec.get("finding_title")).strip(),
                        affected_function=_text(audit_spec.get("affected_function")).strip(),
                        affected_variable=_text(audit_spec.get("affected_variable")).strip(),
                        l1_condition=_text(audit_spec.get("l1_condition_text") or audit_spec.get("l1_condition")).strip(),
                        l2_condition=_text(audit_spec.get("l2_condition_text") or audit_spec.get("l2_condition")).strip(),
                        l3_condition=_text(audit_spec.get("l3_condition_text") or audit_spec.get("l3_condition")).strip(),
                        required_trace_evidence=_text(audit_spec.get("required_trace_evidence")).strip(),
                        abi=self.env.abi,
                        interface_mapper=self.interface_mapper,
                        interface=self.interface,
                        accounts=list(runtime_account_snapshot),
                        guidance_profile=_text(getattr(self.args, "checkpoint_guidance_profile", "default")).strip() or "default",
                        deployer_account=runtime_deployer_account,
                        contract_address=_text(contract_address).strip(),
                        budget_ratio=float(getattr(self.args, "checkpoint_guided_budget_ratio", 0.2) or 0.2),
                        max_per_round=int(getattr(self.args, "checkpoint_guided_max_per_round", guided_budget) or guided_budget),
                        trigger_stagnation=int(getattr(self.args, "checkpoint_guided_trigger_vspc_stagnation", 3) or 3),
                        trigger_levels=trigger_levels or ["L1", "L2"],
                        dedup_window=int(getattr(self.args, "checkpoint_guided_dedup_window", 100) or 100),
                        enable_parameter_materializer=bool(getattr(self.args, "enable_agent_parameter_materializer", False)),
                        confidence_threshold=float(getattr(self.args, "guided_injection_confidence_threshold", 0.0) or 0.0),
                        enable_agent=active_agent_llm,
                        llm_provider=runtime_llm_provider if active_agent_llm else None,
                        enable_effect_forcing_v3=bool(getattr(self.args, "enable_effect_forcing_v3", False)),
                        enable_effect_forcing_v3_1=bool(getattr(self.args, "enable_effect_forcing_v3_1", False)),
                        enable_prestate_builder=bool(getattr(self.args, "enable_prestate_builder", False)),
                        enable_prefix_replay_suffix_mutation=bool(getattr(self.args, "enable_prefix_replay_suffix_mutation", False)),
                        enable_effect_probes=bool(getattr(self.args, "enable_effect_probes", False)),
                        enable_agent_effect_repair=bool(getattr(self.args, "enable_agent_effect_repair", False)),
                        enable_failure_aware_agent_repair=bool(getattr(self.args, "enable_failure_aware_agent_repair", False)),
                        enable_failure_repair_pipeline=bool(getattr(self.args, "enable_failure_repair_pipeline", False)),
                        enable_strong_useful_scheduling=bool(getattr(self.args, "enable_strong_useful_scheduling", False)),
                        failure_repair_max_retries=int(getattr(self.args, "failure_repair_max_retries", 2) or 2),
                        failure_repair_use_agent=bool(getattr(self.args, "failure_repair_use_agent", False)),
                        failure_repair_window_global_tx=int(getattr(self.args, "failure_repair_window_global_tx", 20) or 20),
                        failure_repair_priority_ratio=float(getattr(self.args, "failure_repair_priority_ratio", 0.5) or 0.5),
                        dump_failure_repair_prompts=bool(getattr(self.args, "dump_failure_repair_prompts", False)),
                        enable_runtime_logging=bool(getattr(self.args, "enable_vspc_runtime_logging", False)),
                        bench_id=_text(audit_spec.get("bench_contract_id") or audit_spec.get("bench_id")).strip(),
                        contract_name=self.contract_name,
                        static_summary=runtime_static_summary,
                        phase_cost_env=self.env,
                        rq1_rescue_mode=bool(int(getattr(self.args, "rq1_t_sec", 0) or 0) > 0),
                    )
                tfi_soft_buffer = TFISoftInjectionBuffer(
                    max_size=guided_max_total if guided_injection_enabled else int(getattr(self.args, "target_focus_soft_injection_buffer_size", 64) or 64),
                    max_per_round=guided_budget if guided_injection_enabled else int(getattr(self.args, "target_focus_soft_injection_max_per_round", 3) or 3),
                    min_useful_score=1,
                    confidence_threshold=float(getattr(self.args, "guided_injection_confidence_threshold", 0.0) or 0.0),
                )
                target_focus_soft_injection_analysis = TFISoftInjectionAnalysis(
                    runtime=target_focus_runtime,
                    buffer=tfi_soft_buffer,
                    logger=logger,
                    interval=int(getattr(self.args, "tfi_agent_interval", 20) or 20),
                    max_per_round=guided_budget if guided_injection_enabled else int(getattr(self.args, "target_focus_soft_injection_max_per_round", 3) or 3),
                    enable_agent=active_agent_llm,
                    enable_soft_injection=active_soft_injection,
                    enable_guided_candidate_injection=guided_injection_enabled,
                    guided_injection_confidence_threshold=float(
                        getattr(self.args, "guided_injection_confidence_threshold", 0.0) or 0.0
                    ),
                    checkpoint_guided_runtime=checkpoint_runtime,
                    enable_checkpoint_guided_injection=checkpoint_guided_enabled,
                )
                self.env.target_focus_agent_runtime = target_focus_runtime
                self.env.checkpoint_guided_runtime = checkpoint_runtime
                self.env.target_focus_soft_injection_buffer = tfi_soft_buffer
                engine.analysis.append(target_focus_soft_injection_analysis)
            except Exception as exc:
                self.env.results["target_focus_agent_runtime"] = {
                    "enabled": False,
                    "agent_enabled": bool(getattr(self.args, "enable_tfi_agent", False)),
                    "tfi_planning_only": bool(getattr(self.args, "tfi_planning_only", False)),
                    "candidate_lifecycle_config_status": (
                        "planning_only"
                        if (
                            bool(getattr(self.args, "tfi_planning_only", False))
                            and not bool(getattr(self.args, "enable_guided_candidate_injection", False))
                            and not bool(getattr(self.args, "enable_checkpoint_guided_injection", False))
                        )
                        else (
                            "injection_enabled"
                            if (
                                bool(getattr(self.args, "enable_guided_candidate_injection", False))
                                or bool(getattr(self.args, "enable_checkpoint_guided_injection", False))
                            )
                            else "injection_disabled"
                        )
                    ),
                    "soft_injection_enabled": bool(
                        getattr(self.args, "enable_target_focus_soft_injection", False)
                        or getattr(self.args, "enable_guided_candidate_injection", False)
                        or getattr(self.args, "enable_checkpoint_guided_injection", False)
                    ),
                    "guided_candidate_injection_enabled": bool(getattr(self.args, "enable_guided_candidate_injection", False)),
                    "checkpoint_guided_injection_enabled": bool(getattr(self.args, "enable_checkpoint_guided_injection", False)),
                    "isolated_pilot_enabled": bool(
                        getattr(self.args, "enable_tfi_isolated_pilot", False)
                        or getattr(self.args, "enable_guided_candidate_injection", False)
                        or getattr(self.args, "enable_checkpoint_guided_injection", False)
                    ),
                    "agent_error_count": 1,
                    "candidate_built_count": 0,
                    "candidate_executed_count": 0,
                    "useful_candidate_count": 0,
                    "soft_injected_count": 0,
                    "duplicate_skipped_count": 0,
                    "no_main_search_replacement": True,
                    "no_generator_mutation_crossover_selection_change": True,
                    "llm_called": False,
                    "rag_used": False,
                    "error": "target_focus_runtime_init_failed: {}".format(str(exc)),
                }
        engine.analysis.append(ExecutionTraceAnalyzer(self.env))  # 娉ㄥ唽浜嗘墽琛屽櫒

        if getattr(self.args, "enable_stage", False):
            engine.stage_runtime = StageRuntime(
                StageRuntimeConfig(
                    enable_stage=True,
                    force_gei_only=bool(self.args.force_gei_only),
                    max_stage=self.args.max_stage,
                    stage_log_path=self.args.stage_log,
                    min_gei_generations=self.args.min_gei_generations,
                    stagnation_window=self.args.stagnation_window,
                    enable_snapshot_pool=bool(self.args.enable_snapshot_pool),
                    snapshot_pool_capacity=self.args.snapshot_pool_capacity,
                )
            )

        run_output_dir = getattr(self.args, "run_output_dir", None) or _resolve_run_output_dir(self.args.results)
        if (
            getattr(self.args, "fresh_agent_run", False)
            and (
                getattr(self.args, "enable_tfi_agent", False)
                or getattr(self.args, "enable_ksi_llm_feedback", False)
            )
            and getattr(self.args, "agent_use_llm", False)
        ):
            warmup_state = _ensure_gei60_state(self.env, self.args)
            if _llm_runtime_allowed(self.env, self.args):
                try:
                    _initialize_fresh_agent_plan(
                        logger=logger,
                        env=self.env,
                        args=self.args,
                        contract_name=self.contract_name,
                        abi=self.env.abi,
                        source_path=getattr(self.args, "source", None),
                        run_output_dir=run_output_dir,
                        llm_provider=runtime_llm_provider,
                    )
                    warmup_state["fresh_agent_initialized"] = True
                    warmup_state["fresh_agent_initialized_at_fuzz_active_sec"] = round(
                        _current_fuzz_active_elapsed_sec(self.env),
                        6,
                    )
                    warmup_state["fresh_agent_initialization_source"] = "startup"
                    _sync_gei60_results(self.env, warmup_state)
                except Exception as exc:
                    agent_state = getattr(self.env, "fresh_agent_runtime_state", None)
                    if isinstance(agent_state, dict):
                        _record_agent_llm_stats(
                            agent_state,
                            error="agent_init_runtime_error: {}".format(str(exc)),
                        )
                        self.env.results["fresh_agent_runtime"] = _agent_runtime_result_payload(agent_state)

        path_progress_tracker = None
        if getattr(self.args, "path_plan", None) or getattr(self.args, "enable_path_validation", False):
            path_plan = getattr(self.args, "path_plan_data", PathPlan())
            path_progress_tracker = PathProgressTracker(
                path_plan=path_plan,
                output_dir=run_output_dir,
                enabled=bool(getattr(self.args, "enable_path_validation", False)),
            )
            self.env.path_progress_tracker = path_progress_tracker

        ksi_runtime = None
        if getattr(self.args, "enable_ksi_evidence", False):
            ksi_runtime = KSIRuntime(
                output_dir=run_output_dir,
                enabled=bool(getattr(self.args, "enable_ksi_evidence", False)),
            )
            self.env.ksi_runtime = ksi_runtime

        gei_side_path_recorder = None
        if getattr(self.args, "enable_gei_core", False):
            gei_side_path_recorder = GEISidePathRecorder(
                output_dir=run_output_dir,
                enabled=bool(getattr(self.args, "enable_gei_core", False)),
            )
            self.env.gei_side_path_recorder = gei_side_path_recorder

        self.env.setup_transactions = self.env.nr_of_transactions
        self.env.current_generation = 0
        refresh_timing_results(self.env)
        self.env.population = population
        settings.GLOBAL_ENV = self.env
        fuzz_core_started_at = time.time()
        try:
            record_phase_span(
                self.env,
                phase="setup_deploy",
                started_at=run_started_at,
                ended_at=fuzz_core_started_at,
                metadata={"source": "main.run"},
            )
        except Exception:
            pass

        if getattr(self.args, "enable_tfi_agent", False) or getattr(self.args, "enable_ksi_llm_feedback", False):
            engine.analysis.append(
                AgentKSIFeedbackAnalysis(
                    logger=logger,
                    llm_provider=runtime_llm_provider,
                    interval=int(getattr(self.args, "tfi_agent_interval", 20) or 20),
                )
            )

        _sync_gei60_results(self.env, _ensure_gei60_state(self.env, self.args))
        _sync_rq1_gei_sender_pool_results(self.env, _ensure_rq1_gei_sender_pool_state(self.env, self.args))
        engine.run(ng=settings.GENERATIONS)
        fuzz_core_finished_at = time.time()
        try:
            record_phase_span(
                self.env,
                phase="fuzz_core",
                started_at=fuzz_core_started_at,
                ended_at=fuzz_core_finished_at,
                metadata={"source": "main.run"},
            )
        except Exception:
            pass
        _sync_gei60_results(self.env, _ensure_gei60_state(self.env, self.args))
        _sync_rq1_gei_sender_pool_results(self.env, _ensure_rq1_gei_sender_pool_state(self.env, self.args))
        if not getattr(self.args, "enable_stage", False):
            self.env.results.update(StageRuntime.disabled_summary())

        gei_side_path_summary = None
        if getattr(self.args, "enable_gei_core", False):
            if gei_side_path_recorder is not None:
                gei_side_path_summary = gei_side_path_recorder.get_summary()
            else:
                gei_side_path_summary = {
                    "enabled": False,
                    "observation_count": 0,
                    "side_path_candidate_count": 0,
                    "input_source": "disabled",
                    "ksi_required": False,
                    "ksi_enrichment_used": False,
                    "coverage_delta_sum": 0.0,
                    "revert_count": 0,
                    "reason": "gei_core_init_failed",
                }
            self.env.results["gei_side_path"] = gei_side_path_summary
            self.env.results["gei_core"] = gei_side_path_summary

        soft_guidance_summary = None
        if soft_guidance is not None:
            try:
                soft_guidance_summary = soft_guidance.get_summary()
                self.env.results["soft_guidance"] = soft_guidance_summary
            except Exception:
                pass

        guided_sequence_pilot_summary = None
        if guided_sequence_pilot is not None:
            try:
                guided_sequence_pilot_summary = guided_sequence_pilot.finalize()
                self.env.results["guided_sequence_pilot"] = guided_sequence_pilot_summary
            except Exception:
                pass
        if target_focus_soft_injection_analysis is not None:
            try:
                self.env.results["target_focus_agent_runtime"] = target_focus_soft_injection_analysis.build_summary()
            except Exception:
                pass

        agent_feedback_summary = None
        path_plan = getattr(self.args, "path_plan_data", PathPlan())
        if not path_plan.is_empty():
            try:
                builder = AgentFeedbackBuilder(
                    path_plan=path_plan,
                    output_dir=run_output_dir,
                )
                blocker_evidence_path = builder.build_and_write()
                if blocker_evidence_path:
                    agent_feedback_summary = {
                        "enabled": True,
                        "blocker_evidence_path": blocker_evidence_path,
                    }
                    self.env.results["agent_feedback"] = agent_feedback_summary
            except Exception:
                pass

        if path_progress_tracker is not None:
            try:
                path_validation_summary = path_progress_tracker.get_summary()
                if ksi_runtime is not None:
                    _merge_ksi_summary(path_validation_summary, ksi_runtime.get_summary())
                if agent_feedback_summary is not None:
                    _merge_agent_feedback_summary(path_validation_summary, agent_feedback_summary)
                if gei_side_path_summary is not None:
                    _merge_gei_side_path_summary(path_validation_summary, gei_side_path_summary)
                if soft_guidance_summary is not None:
                    _merge_soft_guidance_summary(path_validation_summary, soft_guidance_summary)
                if guided_sequence_pilot_summary is not None:
                    _merge_guided_sequence_pilot_summary(path_validation_summary, guided_sequence_pilot_summary)
                self.env.results["path_validation"] = path_validation_summary
                if _should_write_path_validation_sidecar(self.args.results):
                    _write_path_validation_summary(
                        getattr(self.args, "run_output_dir", None) or _resolve_run_output_dir(self.args.results),
                        self.contract_name,
                        path_validation_summary,
                    )
            except Exception:
                pass
        if ksi_runtime is not None and path_progress_tracker is None:
            try:
                ksi_summary = ksi_runtime.get_summary()
                self.env.results["ksi"] = ksi_summary
                if _should_write_path_validation_sidecar(self.args.results):
                    _write_ksi_summary(run_output_dir, ksi_summary)
            except Exception:
                pass
        if gei_side_path_summary is not None and path_progress_tracker is None:
            try:
                if _should_write_path_validation_sidecar(self.args.results):
                    _write_gei_side_path_summary(run_output_dir, gei_side_path_summary)
            except Exception:
                pass
        if agent_feedback_summary is not None and path_progress_tracker is None:
            try:
                if _should_write_path_validation_sidecar(self.args.results):
                    _write_agent_feedback_summary(run_output_dir, agent_feedback_summary)
            except Exception:
                pass
        if soft_guidance_summary is not None and path_progress_tracker is None:
            try:
                if _should_write_path_validation_sidecar(self.args.results):
                    _write_soft_guidance_summary(run_output_dir, soft_guidance_summary)
            except Exception:
                pass
        if guided_sequence_pilot_summary is not None and path_progress_tracker is None:
            try:
                if _should_write_path_validation_sidecar(self.args.results):
                    _write_guided_sequence_pilot_summary(run_output_dir, guided_sequence_pilot_summary)
            except Exception:
                pass
        tfi_focus_summary = None
        if getattr(self.args, "enable_tfi_focus", False):
            try:
                default_blocker_path = None
                if isinstance(agent_feedback_summary, dict):
                    default_blocker_path = agent_feedback_summary.get("blocker_evidence_path")
                target_focus_runtime = TFIRuntime(
                    output_dir=run_output_dir,
                    enabled=True,
                    blocker_evidence_path=getattr(self.args, "tfi_blocker_evidence", None) or default_blocker_path,
                    min_blocked_count=int(getattr(self.args, "tfi_min_blocked_count", 3) or 3),
                    min_sink_reverted_count=int(getattr(self.args, "tfi_min_sink_reverted_count", 1) or 1),
                )
                tfi_focus_summary = target_focus_runtime.run()
                self.env.results["tfi_focus"] = tfi_focus_summary
                if isinstance(self.env.results.get("path_validation"), dict):
                    _merge_tfi_focus_summary(
                        self.env.results["path_validation"],
                        tfi_focus_summary,
                    )
                    if _should_write_path_validation_sidecar(self.args.results):
                        _write_path_validation_summary(
                            getattr(self.args, "run_output_dir", None) or _resolve_run_output_dir(self.args.results),
                            self.contract_name,
                            self.env.results["path_validation"],
                        )
                elif _should_write_path_validation_sidecar(self.args.results):
                    _write_tfi_focus_summary(run_output_dir, tfi_focus_summary)
            except Exception:
                tfi_focus_summary = {
                    "enabled": True,
                    "target_count": 0,
                    "priority_counts": {"high": 0, "medium": 0, "low": 0},
                    "focus_type_counts": {},
                    "top_focus_functions": {},
                    "output_path": os.path.join(run_output_dir, "tfi_focus_targets.jsonl") if run_output_dir else None,
                    "summary_path": os.path.join(run_output_dir, "tfi_focus_summary.json") if run_output_dir else None,
                    "blocker_evidence_path": getattr(self.args, "tfi_blocker_evidence", None) or (
                        agent_feedback_summary.get("blocker_evidence_path") if isinstance(agent_feedback_summary, dict) else None
                    ),
                    "blocker_evidence_loaded": False,
                }
                self.env.results["tfi_focus"] = tfi_focus_summary

        side_path_plan_export_summary = None
        if getattr(self.args, "export_side_path_plan", False):
            try:
                exporter = SidePathPlanExporter(output_dir=run_output_dir)
                proposed_path_plan_path = exporter.export(max_hypotheses=20)
                hypothesis_count = 0
                if proposed_path_plan_path and os.path.exists(proposed_path_plan_path):
                    with open(proposed_path_plan_path, "r", encoding="utf-8") as handle:
                        exported_payload = json.load(handle)
                    if isinstance(exported_payload, dict) and isinstance(exported_payload.get("hypotheses"), list):
                        hypothesis_count = len(exported_payload.get("hypotheses"))
                side_path_plan_export_summary = {
                    "enabled": True,
                    "output_path": proposed_path_plan_path if proposed_path_plan_path else os.path.join(run_output_dir, "proposed_path_plan.json"),
                    "hypothesis_count": int(hypothesis_count),
                }
                self.env.results["side_path_plan_export"] = side_path_plan_export_summary
                if isinstance(self.env.results.get("path_validation"), dict):
                    _merge_side_path_plan_export_summary(
                        self.env.results["path_validation"],
                        side_path_plan_export_summary,
                    )
                    if _should_write_path_validation_sidecar(self.args.results):
                        _write_path_validation_summary(
                            getattr(self.args, "run_output_dir", None) or _resolve_run_output_dir(self.args.results),
                            self.contract_name,
                            self.env.results["path_validation"],
                        )
                elif _should_write_path_validation_sidecar(self.args.results):
                    _write_side_path_plan_export_summary(run_output_dir, side_path_plan_export_summary)
            except Exception:
                pass

        agent_planner_summary = None
        if getattr(self.args, "enable_agent_planner", False):
            try:
                default_blocker_path = getattr(self.args, "agent_blocker_evidence", None) or (
                    agent_feedback_summary.get("blocker_evidence_path") if isinstance(agent_feedback_summary, dict) else None
                )
                default_tfi_path = getattr(self.args, "agent_tfi_focus", None) or (
                    tfi_focus_summary.get("output_path") if isinstance(tfi_focus_summary, dict) else None
                )
                default_shortlist_path = getattr(self.args, "agent_side_path_shortlist", None) or (
                    os.path.join(run_output_dir, "side_path_shortlist.json") if run_output_dir else None
                )
                default_proposed_plan_path = getattr(self.args, "agent_proposed_path_plan", None) or (
                    side_path_plan_export_summary.get("output_path") if isinstance(side_path_plan_export_summary, dict) else None
                )
                planner_mode = "llm" if getattr(self.args, "agent_use_llm", False) else getattr(
                    self.args,
                    "agent_planner_mode",
                    "deterministic",
                )
                llm_provider = None
                if planner_mode == "llm":
                    llm_provider = LLMProvider(
                        LLMProviderConfig.from_sources(
                            base_url=getattr(self.args, "llm_base_url", None),
                            api_key=getattr(self.args, "llm_api_key", None),
                            model=getattr(self.args, "llm_model", None),
                            api_style=getattr(self.args, "llm_api_style", "auto"),
                            timeout_sec=getattr(self.args, "llm_timeout_sec", 60),
                            max_retries=getattr(self.args, "llm_max_retries", 2),
                            temperature=getattr(self.args, "llm_temperature", 0.1),
                        )
                    )
                planner = AgentPlanner(
                    output_dir=run_output_dir,
                    blocker_evidence_path=default_blocker_path,
                    tfi_focus_path=default_tfi_path,
                    side_path_shortlist_path=default_shortlist_path,
                    proposed_path_plan_path=default_proposed_plan_path,
                    max_hypotheses=int(getattr(self.args, "agent_max_hypotheses", 20) or 20),
                    enabled=True,
                    planner_mode=planner_mode,
                    llm_provider=llm_provider,
                    fallback_to_deterministic=not bool(getattr(self.args, "llm_no_fallback", False)),
                )
                planner_result = planner.run()
                agent_planner_summary = {
                    "enabled": True,
                    "agent_type": planner_result.get("agent_type"),
                    "planner_mode": planner_result.get("planner_mode"),
                    "output_path": planner_result.get("output_path"),
                    "summary_path": planner_result.get("summary_path"),
                    "hypothesis_count": int(planner_result.get("hypothesis_count") or 0),
                    "source_counts": planner_result.get("source_counts", {}),
                    "loaded_inputs": planner_result.get("loaded_inputs", {}),
                    "llm_available": planner_result.get("llm_available"),
                    "llm_configured": planner_result.get("llm_configured"),
                    "llm_used": planner_result.get("llm_used"),
                    "fallback_used": planner_result.get("fallback_used"),
                    "fallback_reason": planner_result.get("fallback_reason"),
                    "model": planner_result.get("model"),
                    "base_url": planner_result.get("base_url"),
                    "llm_api_style": planner_result.get("llm_api_style"),
                }
                self.env.results["agent_planner"] = agent_planner_summary
                if isinstance(self.env.results.get("path_validation"), dict):
                    _merge_agent_planner_summary(
                        self.env.results["path_validation"],
                        agent_planner_summary,
                    )
                    if _should_write_path_validation_sidecar(self.args.results):
                        _write_path_validation_summary(
                            getattr(self.args, "run_output_dir", None) or _resolve_run_output_dir(self.args.results),
                            self.contract_name,
                            self.env.results["path_validation"],
                        )
                elif _should_write_path_validation_sidecar(self.args.results):
                    _write_agent_planner_summary(run_output_dir, agent_planner_summary)
            except Exception:
                agent_planner_summary = {
                    "enabled": True,
                    "agent_type": "deterministic_agent_planner_v1",
                    "planner_mode": "llm" if getattr(self.args, "agent_use_llm", False) else getattr(
                        self.args,
                        "agent_planner_mode",
                        "deterministic",
                    ),
                    "output_path": os.path.join(run_output_dir, "agent_path_plan.json") if run_output_dir else None,
                    "summary_path": os.path.join(run_output_dir, "agent_plan_summary.json") if run_output_dir else None,
                    "hypothesis_count": 0,
                    "source_counts": {
                        "tfi_focus": 0,
                        "blocker_evidence": 0,
                        "side_path_shortlist": 0,
                    },
                    "loaded_inputs": {
                        "blocker_evidence": False,
                        "tfi_focus": False,
                        "side_path_shortlist": False,
                        "proposed_path_plan": False,
                    },
                    "llm_available": False,
                    "llm_configured": False,
                    "llm_used": False,
                    "fallback_used": False,
                    "fallback_reason": None,
                    "model": None,
                    "base_url": None,
                    "llm_api_style": None,
                }
                self.env.results["agent_planner"] = agent_planner_summary

        guided_sequence_summary = None
        if getattr(self.args, "enable_guided_sequence_generation", False):
            try:
                default_repair_blocker_path = getattr(self.args, "repair_blocker_evidence", None) or (
                    agent_feedback_summary.get("blocker_evidence_path") if isinstance(agent_feedback_summary, dict) else None
                )
                default_repair_tfi_path = getattr(self.args, "repair_tfi_focus", None) or (
                    tfi_focus_summary.get("output_path") if isinstance(tfi_focus_summary, dict) else None
                )
                default_repair_path_plan = getattr(self.args, "repair_path_plan", None) or (
                    agent_planner_summary.get("output_path") if isinstance(agent_planner_summary, dict) else None
                ) or getattr(self.args, "path_plan", None)

                diagnosis_builder = RepairDiagnosisBuilder(
                    output_dir=run_output_dir,
                    blocker_evidence_path=default_repair_blocker_path,
                    ksi_evidence_path=getattr(self.args, "repair_ksi_evidence", None),
                    tfi_focus_path=default_repair_tfi_path,
                    agent_path_plan_path=default_repair_path_plan,
                )
                diagnoses = diagnosis_builder.build_diagnoses()
                diagnosis_path = diagnosis_builder.write_diagnoses(diagnoses, filename="repair_diagnosis.json")

                candidate_builder = RepairSequenceCandidateBuilder(
                    output_dir=run_output_dir,
                    diagnoses=diagnoses,
                    max_candidates_per_hypothesis=int(
                        getattr(self.args, "repair_max_candidates_per_hypothesis", 5) or 5
                    ),
                )
                candidate_builder.loaded_inputs = dict(diagnosis_builder.loaded_inputs)
                candidate_builder.diagnosis_output_path = diagnosis_path
                candidates = candidate_builder.build_candidates()
                candidate_path = candidate_builder.write_candidates(
                    candidates,
                    filename="guided_sequence_candidates.json",
                )
                candidate_builder.candidates_output_path = candidate_path
                summary_path = candidate_builder.write_summary(
                    candidates,
                    filename="guided_sequence_summary.json",
                )
                guided_sequence_summary = dict(candidate_builder.last_summary_payload)
                guided_sequence_summary["output_paths"] = dict(guided_sequence_summary.get("output_paths") or {})
                guided_sequence_summary["output_paths"]["guided_sequence_summary"] = summary_path
                self.env.results["guided_sequence_generation"] = guided_sequence_summary
            except Exception:
                guided_sequence_summary = {
                    "enabled": True,
                    "diagnosis_count": 0,
                    "candidate_count": 0,
                    "diagnosis_type_counts": {},
                    "sequence_strategy_counts": {},
                    "top_target_functions": {},
                    "loaded_inputs": {
                        "blocker_evidence": False,
                        "ksi_evidence": False,
                        "tfi_focus": False,
                        "path_plan": False,
                    },
                    "output_paths": {
                        "repair_diagnosis": os.path.join(run_output_dir, "repair_diagnosis.json") if run_output_dir else None,
                        "guided_sequence_candidates": os.path.join(run_output_dir, "guided_sequence_candidates.json") if run_output_dir else None,
                        "guided_sequence_summary": os.path.join(run_output_dir, "guided_sequence_summary.json") if run_output_dir else None,
                    },
                }
                self.env.results["guided_sequence_generation"] = guided_sequence_summary

        agent_state = getattr(self.env, "fresh_agent_runtime_state", None)
        if isinstance(agent_state, dict):
            self.env.results["fresh_agent_runtime"] = _agent_runtime_result_payload(agent_state)
        postprocess_started_at = fuzz_core_finished_at or time.time()
        try:
            stop_resource_sampler(self.env)
        except Exception:
            pass
        try:
            record_phase_span(
                self.env,
                phase="postprocess",
                started_at=postprocess_started_at,
                ended_at=time.time(),
                metadata={"source": "main.run"},
            )
        except Exception:
            pass
        try:
            self.env.results["phase_cost_summary"] = build_phase_summary(self.env)
        except Exception:
            pass
        try:
            finalize_timing(self.env)
        except Exception:
            pass
        try:
            refresh_timing_results(self.env)
        except Exception:
            pass
        try:
            refresh_local_execution_timing_results(self.env)
        except Exception:
            pass
        if isinstance(agent_state, dict):
            self.env.results["fresh_agent_runtime"] = _agent_runtime_result_payload(agent_state)
        try:
            self.env.results["phase_cost_summary"] = build_phase_summary(self.env)
        except Exception:
            pass
        try:
            _write_full_agent_summary(run_output_dir, self.contract_name, self.env.results, self.args)
        except Exception:
            pass

        if self.env.args.cfg:
            if self.env.args.source:
                self.env.cfg.save_control_flow_graph(
                    os.path.splitext(self.env.args.source)[0] + '-' + self.contract_name, 'pdf')
            elif self.env.args.abi:
                self.env.cfg.save_control_flow_graph(
                    os.path.join(os.path.dirname(self.env.args.abi), self.contract_name), 'pdf')

        self.instrumented_evm.reset()
        settings.TRANS_INFO["end_time"] = str(datetime.now())
        try:
            _refresh_result_file(self.args.results, self.contract_name, self.env.results)
        except Exception:
            pass
        if not getattr(self.args, "enable_stage", False):
            _apply_disabled_stage_summary(self.args.results, self.contract_name)


def main():
    args = launch_argument_parser()
    _resolve_llm_api_key_settings(args)
    args.run_start_time = time.time()
    if getattr(args, "fresh_agent_run", False):
        if not getattr(args, "results", None):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.results = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "out_fresh_agent_runtime_{}".format(timestamp),
            )
        args.ignore_precomputed_paths = True
        args.ignore_precomputed_tfi_artifacts = True

    gei_core_requested = bool(
        getattr(args, "enable_gei_core", False)
        or getattr(args, "enable_gei_side_path", False)
    )
    if getattr(args, "disable_gei_core", False):
        gei_core_requested = False
    args.enable_gei_core = gei_core_requested
    args.enable_gei_side_path = gei_core_requested

    target_focus_runtime_requested = bool(
        getattr(args, "enable_tfi_agent", False)
        or getattr(args, "enable_target_focus_soft_injection", False)
        or getattr(args, "enable_tfi_isolated_pilot", False)
        or getattr(args, "enable_guided_candidate_injection", False)
        or getattr(args, "enable_checkpoint_guided_injection", False)
    )
    if target_focus_runtime_requested:
        args.enable_ksi_evidence = True
        args.enable_gei_core = True
        args.enable_gei_side_path = True
        run_output_dir = _resolve_run_output_dir(args.results)
        args.run_output_dir = run_output_dir
        if not getattr(args, "path_plan", None) and not getattr(args, "ignore_precomputed_paths", False):
            for candidate_name in ("agent_path_plan.json", "proposed_path_plan.json"):
                candidate_path = os.path.join(run_output_dir, candidate_name)
                if os.path.exists(candidate_path):
                    args.path_plan = candidate_path
                    break
        if getattr(args, "path_plan", None):
            args.enable_path_validation = True

    if (
        args.path_plan
        or args.enable_path_validation
        or args.enable_ksi_evidence
        or args.enable_gei_core
        or args.enable_soft_path_guidance
        or args.enable_guided_sequence_generation
        or args.enable_guided_sequence_pilot
        or args.enable_agent_planner
        or target_focus_runtime_requested
    ):
        args.run_output_dir = _resolve_run_output_dir(args.results)
        args.path_plan_data = load_path_plan(args.path_plan) if getattr(args, "path_plan", None) else PathPlan()
        if _should_write_path_validation_sidecar(args.results):
            summary_path = os.path.join(args.run_output_dir, "path_validation_summary.json")
            try:
                if os.path.exists(summary_path):
                    os.remove(summary_path)
            except Exception:
                pass
    logger = initialize_logger("Main    ")

    # Check if contract has already been analyzed
    if args.results and os.path.exists(args.results):
        if os.path.isfile(args.results):
            os.remove(args.results)
        elif os.path.isdir(args.results):
            logger.info("Results directory exists and will be reused: " + str(args.results))
        logger.info("Contract " + str(args.source) + " has already been analyzed: " + str(args.results))
        logger.info("Original test output path was cleared or reused: %s", args.results)

    # Initializing random
    if args.seed:
        seed = args.seed
        if not "PYTHONHASHSEED" in os.environ:
            logger.debug("Please set PYTHONHASHSEED to '1' for Python's hash function to behave deterministically.")
    else:
        seed = random.random()
    random.seed(seed)
    logger.title("Initializing seed to %s", seed)

    # Initialize EVM
    instrumented_evm = InstrumentedEVM(settings.RPC_HOST, settings.RPC_PORT)
    instrumented_evm.set_vm_by_name(settings.EVM_VERSION)

    # Create Z3 solver instance
    solver = Solver()
    solver.set("timeout", settings.SOLVER_TIMEOUT)

    # Parse blockchain state if provided
    blockchain_state = []
    if args.blockchain_state:
        if args.blockchain_state.endswith(".json"):
            with open(args.blockchain_state) as json_file:
                for line in json_file.readlines():
                    blockchain_state.append(json.loads(line))
        elif args.blockchain_state.isnumeric():
            settings.BLOCK_HEIGHT = int(args.blockchain_state)
            instrumented_evm.set_vm(settings.BLOCK_HEIGHT)
        else:
            logger.error("Unsupported input file: " + args.blockchain_state)
            sys.exit(-1)

    # Compile source code to get deployment bytecode, runtime bytecode and ABI
    if args.source:
        if args.source.endswith(".sol"):
            compiler_output = compile(args.solc_version, settings.EVM_VERSION, args.source)
            if not compiler_output:
                logger.error("No compiler output for: " + args.source)
                sys.exit(-1)
            any_contract_run = False
            all_compile_contracts = _flatten_compile_contracts(compiler_output)
            for contract_name, contract in compiler_output['contracts'][args.source].items():
                if args.contract and contract_name != args.contract:
                    continue
                contract = _link_contract_libraries(
                    contract_name,
                    contract,
                    all_compile_contracts,
                    instrumented_evm,
                    logger,
                )
                bytecode_error = _validate_contract_bytecode(contract_name, contract)
                if bytecode_error:
                    logger.error(bytecode_error)
                    continue
                if contract['abi'] and contract['evm']['bytecode']['object'] and contract['evm']['deployedBytecode'][
                    'object']:
                    source_map = SourceMap(':'.join([args.source, contract_name]), compiler_output)
                    compiler_output['contracts'][args.source][contract_name] = contract
                    any_contract_run = True
                    Fuzzer(contract_name, contract["abi"], contract['evm']['bytecode']['object'],
                           contract['evm']['deployedBytecode']['object'], instrumented_evm, blockchain_state, solver,
                           args, seed, source_map, compiler_output['contracts'][args.source]).run()
            if args.contract and not any_contract_run:
                logger.error("Requested contract did not run after compile/link validation: " + str(args.contract))
                sys.exit(-1)
        else:
            logger.error("Unsupported input file: " + args.source)
            sys.exit(-1)

    if args.abi:
        with open(args.abi) as json_file:
            abi = json.load(json_file)
            runtime_bytecode = instrumented_evm.get_code(to_canonical_address(args.contract)).hex()
            Fuzzer(args.contract, abi, None, runtime_bytecode, instrumented_evm, blockchain_state, solver, args,
                   seed).run()


def launch_argument_parser():
    parser = argparse.ArgumentParser()
    try:
        solc_version_text = str(solcx.get_solc_version())
    except Exception:
        solc_version_text = "not-installed"
    try:
        installed_solc_text = str(solcx.get_installed_solc_versions())
    except Exception:
        installed_solc_text = "[]"

    # Contract parameters
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("positional_source", nargs="?",
                        help="Solidity smart contract source code file (.sol).")
    group1.add_argument("-s", "--source", type=str,
                        help="Solidity smart contract source code file (.sol).")
    group1.add_argument("-a", "--abi", type=str,
                        help="Smart contract ABI file (.json).")

    # group2 = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-c", "--contract", type=str,
                        help="Contract name to be fuzzed (if Solidity source code file provided) or blockchain contract address (if ABI file provided).")

    parser.add_argument("-b", "--blockchain-state", type=str,
                        help="Initialize fuzzer with a blockchain state by providing a JSON file (if Solidity source code file provided) or a block number (if ABI file provided).")

    # Compiler parameters
    parser.add_argument("--solc", help="Solidity compiler version (default '" + solc_version_text +
                                       "'). Installed compiler versions: " + installed_solc_text + ".",
                        action="store", dest="solc_version", type=str)
    parser.add_argument("--solc-dir", type=str, dest="solc_dir",
                        help="Directory containing solc binaries for py-solc-x.")
    parser.add_argument("--evm", help="Ethereum VM (default '" + str(
        settings.EVM_VERSION) + "'). Available VM's: 'homestead', 'byzantium' or 'petersburg'.", action="store",
                        dest="evm_version", type=str)

    # Evolutionary parameters
    group3 = parser.add_mutually_exclusive_group(required=False)
    group3.add_argument("-g", "--generations",
                        help="Number of generations (default " + str(settings.GENERATIONS) + ").", action="store",
                        dest="generations", type=int)
    group3.add_argument("-t", "--timeout",
                        help="Number of seconds for fuzzer to stop.", action="store",
                        dest="global_timeout", type=int)
    group3.add_argument("--time-budget",
                        help="Alias for --timeout.", action="store",
                        dest="global_timeout", type=int)
    parser.add_argument(
        "--budget-mode",
        choices=["wall", "fuzz-active"],
        default="wall",
        dest="budget_mode",
        help="Choose whether the runtime budget is wall-clock or fuzz-active time.",
    )
    parser.add_argument(
        "--fuzz-active-budget-sec",
        type=int,
        dest="fuzz_active_budget_sec",
        help="Active fuzzing budget in seconds when --budget-mode fuzz-active is enabled.",
    )
    parser.add_argument(
        "--enable-gei60",
        action="store_true",
        dest="enable_gei60",
        help="Enable the GEI60 warm-up phase before API/LLM-driven guidance resumes.",
    )
    parser.add_argument(
        "--gei60-sec",
        type=int,
        dest="gei60_sec",
        default=60,
        help="GEI60 warm-up duration in fuzz-active seconds when --enable-gei60 is set.",
    )
    parser.add_argument(
        "--pure-fuzz-warmup-sec",
        type=int,
        dest="pure_fuzz_warmup_sec",
        default=0,
        help="Deprecated alias for --enable-gei60 --gei60-sec N.",
    )
    parser.add_argument(
        "--wall-snapshot-sec",
        type=int,
        dest="wall_snapshot_sec",
        help="Record a wall-clock snapshot when elapsed real time first reaches this many seconds.",
    )
    parser.add_argument("-n", "--population-size",
                        help="Size of the population.", action="store",
                        dest="population_size", type=int)
    parser.add_argument("-pc", "--probability-crossover",
                        help="Size of the population.", action="store",
                        dest="probability_crossover", type=float)
    parser.add_argument("-pm", "--probability-mutation",
                        help="Size of the population.", action="store",
                        dest="probability_mutation", type=float)

    # Miscellaneous parameters
    parser.add_argument("-r", "--results", type=str, help="Folder or JSON file where results should be stored.")
    parser.add_argument("--out", type=str, dest="results", help="Alias for --results.")
    parser.add_argument("--seed", type=float, help="Initialize the random number generator with a given seed.")
    parser.add_argument("--cfg", help="Build control-flow graph and highlight code coverage.", action="store_true")
    parser.add_argument("--rpc-host", help="Ethereum client RPC hostname.", action="store", dest="rpc_host", type=str)
    parser.add_argument("--rpc-port", help="Ethereum client RPC port.", action="store", dest="rpc_port", type=int)

    parser.add_argument("--data-dependency",
                        help="Disable/Enable data dependency analysis: 0 - Disable, 1 - Enable (default: 1)",
                        action="store",
                        dest="data_dependency", type=int)
    parser.add_argument("--constraint-solving",
                        help="Disable/Enable constraint solving: 0 - Disable, 1 - Enable (default: 1)", action="store",
                        dest="constraint_solving", type=int)
    parser.add_argument("--environmental-instrumentation",
                        help="Disable/Enable environmental instrumentation: 0 - Disable, 1 - Enable (default: 1)",
                        action="store",
                        dest="environmental_instrumentation", type=int)
    parser.add_argument("--max-individual-length",
                        help="Maximal length of an individual (default: " + str(settings.MAX_INDIVIDUAL_LENGTH) + ")",
                        action="store",
                        dest="max_individual_length", type=int)
    parser.add_argument("--max-symbolic-execution",
                        help="Maximum number of symbolic execution calls before restting population (default: " + str(
                            settings.MAX_SYMBOLIC_EXECUTION) + ")", action="store",
                        dest="max_symbolic_execution", type=int)
    parser.add_argument("--gas-limit",
                        help="Gas limit used for deployment and generated transactions (default: " +
                             str(settings.GAS_LIMIT) + ").",
                        action="store",
                        dest="gas_limit", type=int)
    parser.add_argument("--path-plan", type=str, dest="path_plan",
                        help="JSON file containing passive vulnerability path hypotheses.")
    parser.add_argument("--enable-path-validation", action="store_true", dest="enable_path_validation")
    parser.add_argument("--enable-ksi-evidence", action="store_true", dest="enable_ksi_evidence")
    parser.add_argument("--enable-ksi-llm-feedback", action="store_true", dest="enable_ksi_llm_feedback")
    parser.add_argument("--enable-gei-core", action="store_true", dest="enable_gei_core")
    parser.add_argument("--disable-gei-core", action="store_true", dest="disable_gei_core")
    parser.add_argument("--enable-gei-side-path", action="store_true", dest="enable_gei_side_path")
    parser.add_argument("--export-side-path-plan", action="store_true", dest="export_side_path_plan")
    parser.add_argument("--enable-soft-path-guidance", action="store_true", dest="enable_soft_path_guidance")
    parser.add_argument("--path-guidance-ratio", type=float, default=0.10, dest="path_guidance_ratio")
    parser.add_argument("--path-guidance-max-per-gen", type=int, default=5, dest="path_guidance_max_per_gen")
    parser.add_argument("--soft-guidance-dry-run", action="store_true", dest="soft_guidance_dry_run")
    parser.add_argument("--soft-guidance-blocker-evidence", type=str, dest="soft_guidance_blocker_evidence")
    parser.add_argument("--enable-tfi-focus", action="store_true", dest="enable_tfi_focus")
    parser.add_argument("--tfi-min-blocked-count", type=int, default=3, dest="tfi_min_blocked_count")
    parser.add_argument("--tfi-min-sink-reverted-count", type=int, default=1, dest="tfi_min_sink_reverted_count")
    parser.add_argument("--tfi-blocker-evidence", type=str, dest="tfi_blocker_evidence")
    parser.add_argument("--enable-agent-planner", action="store_true", dest="enable_agent_planner")
    parser.add_argument("--agent-blocker-evidence", type=str, dest="agent_blocker_evidence")
    parser.add_argument("--agent-tfi-focus", type=str, dest="agent_tfi_focus")
    parser.add_argument("--agent-side-path-shortlist", type=str, dest="agent_side_path_shortlist")
    parser.add_argument("--agent-proposed-path-plan", type=str, dest="agent_proposed_path_plan")
    parser.add_argument("--agent-max-hypotheses", type=int, default=20, dest="agent_max_hypotheses")
    parser.add_argument(
        "--agent-planner-mode",
        choices=["deterministic", "llm"],
        default="deterministic",
        dest="agent_planner_mode",
    )
    parser.add_argument("--agent-use-llm", action="store_true", dest="agent_use_llm")
    parser.add_argument("--llm-base-url", type=str, dest="llm_base_url")
    parser.add_argument("--llm-api-key", type=str, dest="llm_api_key")
    parser.add_argument("--llm-api-key-file", type=str, dest="llm_api_key_file")
    parser.add_argument("--llm-model", type=str, dest="llm_model")
    parser.add_argument(
        "--llm-api-style",
        choices=["responses", "chat_completions", "auto"],
        default="auto",
        dest="llm_api_style",
    )
    parser.add_argument("--llm-timeout-sec", type=int, default=60, dest="llm_timeout_sec")
    parser.add_argument("--llm-max-retries", type=int, default=2, dest="llm_max_retries")
    parser.add_argument("--llm-temperature", type=float, default=0.1, dest="llm_temperature")
    parser.add_argument("--llm-no-fallback", action="store_true", dest="llm_no_fallback")
    parser.add_argument(
        "--enable-repair-sequence-generation",
        action="store_true",
        dest="enable_guided_sequence_generation",
    )
    parser.add_argument("--repair-blocker-evidence", type=str, dest="repair_blocker_evidence")
    parser.add_argument("--repair-ksi-evidence", type=str, dest="repair_ksi_evidence")
    parser.add_argument("--repair-tfi-focus", type=str, dest="repair_tfi_focus")
    parser.add_argument("--repair-path-plan", type=str, dest="repair_path_plan")
    parser.add_argument(
        "--repair-max-candidates-per-hypothesis",
        type=int,
        default=5,
        dest="repair_max_candidates_per_hypothesis",
    )
    parser.add_argument(
        "--enable-repair-sequence-pilot",
        action="store_true",
        dest="enable_guided_sequence_pilot",
    )
    parser.add_argument("--repair-selected-sequences", type=str, dest="repair_selected_sequences")
    parser.add_argument("--repair-pilot-top-k", type=int, default=5, dest="repair_pilot_top_k")
    parser.add_argument("--repair-pilot-dry-run", action="store_true", dest="repair_pilot_dry_run")
    parser.add_argument("--repair-pilot-output-dir", type=str, dest="repair_pilot_output_dir")
    parser.add_argument(
        "--repair-pilot-insertion-policy",
        choices=["append", "front"],
        default="append",
        dest="repair_pilot_insertion_policy",
        help="append keeps guided candidates behind the normal population; front is an explicit priority mode.",
    )
    parser.add_argument(
        "--repair-pilot-concretization",
        choices=["v1", "v2"],
        default="v1",
        dest="repair_pilot_concretization",
    )
    parser.add_argument("--enable-tfi-agent", action="store_true", dest="enable_tfi_agent")
    parser.add_argument("--fresh-agent-run", action="store_true", dest="fresh_agent_run")
    parser.add_argument(
        "--tfi-planning-only",
        action="store_true",
        dest="tfi_planning_only",
        help="Allow TFI to emit planning/gap artifacts without candidate injection or execution.",
    )
    parser.add_argument("--ignore-precomputed-paths", action="store_true", dest="ignore_precomputed_paths")
    parser.add_argument(
        "--ignore-precomputed-tfi-artifacts",
        action="store_true",
        dest="ignore_precomputed_tfi_artifacts",
    )
    parser.add_argument("--enable-tfi-soft-injection", action="store_true", dest="enable_target_focus_soft_injection")
    parser.add_argument("--enable-tfi-isolated-pilot", action="store_true", dest="enable_tfi_isolated_pilot")
    parser.add_argument(
        "--enable-guided-candidate-injection",
        action="store_true",
        dest="enable_guided_candidate_injection",
        help="Enable pathpilot guided candidates to enter the existing soft-injection path.",
    )
    parser.add_argument(
        "--guided-injection-budget-per-round",
        type=int,
        default=3,
        dest="guided_injection_budget_per_round",
        help="Maximum guided candidates softly injected per runtime feedback round.",
    )
    parser.add_argument(
        "--guided-injection-max-total",
        type=int,
        default=64,
        dest="guided_injection_max_total",
        help="Maximum guided candidates retained in the soft-injection buffer.",
    )
    parser.add_argument(
        "--guided-injection-confidence-threshold",
        type=float,
        default=0.0,
        dest="guided_injection_confidence_threshold",
        help="Minimum candidate confidence for guided soft injection when confidence is available.",
    )
    parser.add_argument("--enable-checkpoint-guided-injection", action="store_true", dest="enable_checkpoint_guided_injection")
    parser.add_argument(
        "--checkpoint-guidance-profile",
        type=str,
        default="default",
        choices=["default", "access_control_state_push"],
        dest="checkpoint_guidance_profile",
    )
    parser.add_argument("--checkpoint-guided-budget-ratio", type=float, default=0.2, dest="checkpoint_guided_budget_ratio")
    parser.add_argument("--checkpoint-guided-max-per-round", type=int, default=3, dest="checkpoint_guided_max_per_round")
    parser.add_argument("--checkpoint-guided-trigger-vspc-stagnation", type=int, default=3, dest="checkpoint_guided_trigger_vspc_stagnation")
    parser.add_argument("--checkpoint-guided-trigger-levels", type=str, default="L1,L2", dest="checkpoint_guided_trigger_levels")
    parser.add_argument("--checkpoint-guided-dedup-window", type=int, default=100, dest="checkpoint_guided_dedup_window")
    parser.add_argument("--enable-vspc-runtime-logging", action="store_true", dest="enable_vspc_runtime_logging")
    parser.add_argument("--enable-agent-parameter-materializer", action="store_true", dest="enable_agent_parameter_materializer")
    parser.add_argument("--enable-effect-forcing-v3", action="store_true", dest="enable_effect_forcing_v3")
    parser.add_argument("--enable-effect-forcing-v3-1", action="store_true", dest="enable_effect_forcing_v3_1")
    parser.add_argument("--enable-prestate-builder", action="store_true", dest="enable_prestate_builder")
    parser.add_argument("--enable-prefix-replay-suffix-mutation", action="store_true", dest="enable_prefix_replay_suffix_mutation")
    parser.add_argument("--enable-effect-probes", action="store_true", dest="enable_effect_probes")
    parser.add_argument("--enable-agent-effect-repair", action="store_true", dest="enable_agent_effect_repair")
    parser.add_argument("--enable-failure-aware-agent-repair", action="store_true", dest="enable_failure_aware_agent_repair")
    parser.add_argument("--enable-failure-repair-pipeline", action="store_true", dest="enable_failure_repair_pipeline")
    parser.add_argument("--enable-strong-useful-scheduling", action="store_true", dest="enable_strong_useful_scheduling")
    parser.add_argument("--failure-repair-max-retries", type=int, default=2, dest="failure_repair_max_retries")
    parser.add_argument("--failure-repair-window-global-tx", type=int, default=20, dest="failure_repair_window_global_tx")
    parser.add_argument("--failure-repair-priority-ratio", type=float, default=0.5, dest="failure_repair_priority_ratio")
    parser.add_argument(
        "--failure-repair-use-agent",
        nargs="?",
        const=True,
        default=False,
        dest="failure_repair_use_agent",
        type=lambda value: str(value).strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--dump-failure-repair-prompts",
        nargs="?",
        const=True,
        default=False,
        dest="dump_failure_repair_prompts",
        type=lambda value: str(value).strip().lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument("--audit-target-spec-json", type=str, dest="audit_target_spec_json")
    parser.add_argument("--audit-oracle-family", type=str, dest="audit_oracle_family")
    parser.add_argument("--audit-normalized-type", type=str, dest="audit_normalized_type")
    parser.add_argument("--audit-finding-title", type=str, dest="audit_finding_title")
    parser.add_argument("--audit-affected-function", type=str, dest="audit_affected_function")
    parser.add_argument("--audit-affected-variable", type=str, dest="audit_affected_variable")
    parser.add_argument("--audit-l1-condition", type=str, dest="audit_l1_condition")
    parser.add_argument("--audit-l2-condition", type=str, dest="audit_l2_condition")
    parser.add_argument("--audit-l3-condition", type=str, dest="audit_l3_condition")
    parser.add_argument("--audit-required-trace-evidence", type=str, dest="audit_required_trace_evidence")
    parser.add_argument("--tfi-agent-interval", type=int, default=20, dest="tfi_agent_interval")
    parser.add_argument("--tfi-agent-max-tasks-per-round", type=int, default=4, dest="tfi_agent_max_tasks_per_round")
    parser.add_argument(
        "--tfi-agent-max-candidates-per-task",
        type=int,
        default=2,
        dest="tfi_agent_max_candidates_per_task",
    )
    parser.add_argument(
        "--tfi-soft-injection-max-per-round",
        type=int,
        default=3,
        dest="target_focus_soft_injection_max_per_round",
    )
    parser.add_argument(
        "--tfi-soft-injection-buffer-size",
        type=int,
        default=64,
        dest="target_focus_soft_injection_buffer_size",
    )
    parser.add_argument("--tfi-materialization-tasks", type=str, dest="tfi_materialization_tasks")
    parser.add_argument("--tfi-candidate-scope", type=str, dest="target_focus_candidate_scope")
    parser.add_argument("--enable-tfi-debug-artifacts", action="store_true", dest="enable_tfi_debug_artifacts")
    parser.add_argument("--enable-stage", action="store_true", dest="enable_stage")
    parser.add_argument("--force-gei-only", action="store_true", dest="force_gei_only")
    parser.add_argument("--max-stage", choices=["gei", "ksi"], default="ksi", dest="max_stage")
    parser.add_argument("--stage-log", type=str, dest="stage_log")
    parser.add_argument("--min-gei-generations", type=int, default=2, dest="min_gei_generations")
    parser.add_argument("--stagnation-window", type=int, default=2, dest="stagnation_window")
    parser.add_argument("--enable-snapshot-pool", action="store_true", dest="enable_snapshot_pool")
    parser.add_argument("--snapshot-pool-capacity", type=int, default=32, dest="snapshot_pool_capacity")
    parser.add_argument("--rq1-config", choices=sorted(RQ1_CONFIGS.keys()), default="", dest="rq1_config")
    parser.add_argument("--rq1-g-sec", type=int, default=0, dest="rq1_g_sec")
    parser.add_argument("--rq1-k-sec", type=int, default=0, dest="rq1_k_sec")
    parser.add_argument("--rq1-t-sec", type=int, default=0, dest="rq1_t_sec")
    parser.add_argument("--rq1-total-sec", type=int, default=0, dest="rq1_total_sec")
    # cross contract fuzz parameters
    parser.add_argument("--cross-contract", type=int, help="open cross contract mode, open -- 1, close -- 2 (default)",
                        action="store", dest="cross_contract", default=2)
    parser.add_argument("--depend-contracts", type=str, nargs="*",
                        help="main fuzzed contract depend those contracts, you should give some names.",
                        dest="depend_contracts")
    parser.add_argument("--trans-json-path", type=str, help="location to save trans info to json",
                        dest="trans_json_path")
    parser.add_argument("--solc-path-cross", type=str, help="solc path, used by cross-slither", dest="solc_path_cross")
    parser.add_argument("--constructor-args", type=str, nargs="*",
                        help="constructor args, like: [address, uint, .....]", dest="constructor_args")
    parser.add_argument("--open-trans-comp", type=int, help="open cross trans mode, open -- 1 (default), close -- 2",
                        action="store", dest="trans_comp", default=1)
    parser.add_argument("--trans-mode", type=int, help="trans support mode, open other -- 1, no exec other -- 2",
                        default=1, dest="trans_mode")
    parser.add_argument("--p-open-cross", type=int, help="use cross trans probability: (1~8)", default=5,
                        dest="p_open_cross")
    parser.add_argument("--cross-init-mode", type=int, help="cross init mode: 1 -- specify, 2 -- random, 3 -- close",
                        default=1, dest="cross_init_mode")
    parser.add_argument("--duplication", type=str, help="duplication mode: 0 -- close, 1 -- open", default='0',
                        dest="duplication")

    version = "ConFuzzius - Version 0.0.2 - "
    version += "\"By three methods we may learn wisdom:\n"
    version += "First, by reflection, which is noblest;\n"
    version += "Second, by imitation, which is easiest;\n"
    version += "And third by experience, which is the bitterest.\"\n"
    parser.add_argument("-v", "--version", action="version", version=version)

    args = parser.parse_args()

    if hasattr(args, "path_guidance_ratio"):
        try:
            args.path_guidance_ratio = max(0.0, min(float(args.path_guidance_ratio), 0.3))
        except Exception:
            args.path_guidance_ratio = 0.10
    if hasattr(args, "path_guidance_max_per_gen"):
        try:
            args.path_guidance_max_per_gen = max(0, int(args.path_guidance_max_per_gen))
        except Exception:
            args.path_guidance_max_per_gen = 5
    if hasattr(args, "repair_pilot_top_k"):
        try:
            args.repair_pilot_top_k = max(0, int(args.repair_pilot_top_k))
        except Exception:
            args.repair_pilot_top_k = 5
    if hasattr(args, "repair_pilot_insertion_policy"):
        policy = _text(args.repair_pilot_insertion_policy).strip().lower().replace("-", "_")
        args.repair_pilot_insertion_policy = policy if policy in ("append", "front") else "append"
    for field_name, default_value in (
        ("tfi_agent_interval", 20),
        ("tfi_agent_max_tasks_per_round", 4),
        ("tfi_agent_max_candidates_per_task", 2),
        ("target_focus_soft_injection_max_per_round", 3),
        ("target_focus_soft_injection_buffer_size", 64),
        ("checkpoint_guided_max_per_round", 3),
        ("checkpoint_guided_trigger_vspc_stagnation", 3),
        ("checkpoint_guided_dedup_window", 100),
    ):
        if hasattr(args, field_name):
            try:
                setattr(args, field_name, max(1, int(getattr(args, field_name) or default_value)))
            except Exception:
                setattr(args, field_name, default_value)
    if hasattr(args, "checkpoint_guided_budget_ratio"):
        try:
            args.checkpoint_guided_budget_ratio = max(0.0, min(float(args.checkpoint_guided_budget_ratio), 1.0))
        except Exception:
            args.checkpoint_guided_budget_ratio = 0.2
    if getattr(args, "enable_checkpoint_guided_injection", False):
        args.enable_guided_candidate_injection = True
        args.enable_target_focus_soft_injection = True
        args.enable_tfi_isolated_pilot = True
    args.rq1_config = normalize_rq1_config(getattr(args, "rq1_config", ""))
    if args.rq1_config:
        rq1_budget = rq1_budget_for_config(args.rq1_config)
        args.rq1_g_sec = int(rq1_budget.get("g", 0))
        args.rq1_k_sec = int(rq1_budget.get("k", 0))
        args.rq1_t_sec = int(rq1_budget.get("t", 0))
        args.rq1_total_sec = int(rq1_budget.get("total", 0))
        args.budget_mode = "fuzz-active"
        args.global_timeout = int(args.rq1_total_sec or args.global_timeout or 1)
        args.fuzz_active_budget_sec = int(args.rq1_total_sec or args.fuzz_active_budget_sec or args.global_timeout or 1)
        args.wall_snapshot_sec = int(args.rq1_total_sec or args.wall_snapshot_sec or args.global_timeout or 1)
        args.enable_gei60 = bool(args.rq1_g_sec > 0)
        args.gei60_sec = max(1, int(args.rq1_g_sec or 1))
        args.pure_fuzz_warmup_sec = int(args.rq1_g_sec or 0)
        # Keep pure-G settings as low-overhead exploration only. The stronger
        # GEI core / side-path guidance is reserved for settings that actually
        # allocate time to KSI/TFI.
        pure_gei_only = bool(args.rq1_g_sec > 0 and args.rq1_k_sec <= 0 and args.rq1_t_sec <= 0)
        args.enable_gei_core = bool((args.rq1_k_sec > 0 or args.rq1_t_sec > 0) and not pure_gei_only)
        args.enable_gei_side_path = args.enable_gei_core
        if args.rq1_k_sec > 0 or args.rq1_t_sec > 0:
            args.enable_ksi_evidence = True
        if args.rq1_t_sec > 0:
            args.enable_tfi_agent = True
            args.enable_target_focus_soft_injection = True
            args.enable_tfi_isolated_pilot = True
            args.enable_guided_candidate_injection = True
            args.enable_checkpoint_guided_injection = True
            args.enable_agent_parameter_materializer = True
            args.enable_effect_forcing_v3 = True
            args.enable_effect_forcing_v3_1 = True
            args.enable_prestate_builder = True
            args.enable_prefix_replay_suffix_mutation = True
            args.enable_effect_probes = True
            args.enable_agent_effect_repair = True
            args.enable_failure_aware_agent_repair = True
            args.enable_failure_repair_pipeline = True
            args.enable_strong_useful_scheduling = True
            args.failure_repair_use_agent = True
            args.guided_injection_budget_per_round = max(8, int(getattr(args, "guided_injection_budget_per_round", 3) or 3))
            args.guided_injection_max_total = max(192, int(getattr(args, "guided_injection_max_total", 64) or 64))
            args.target_focus_soft_injection_max_per_round = max(
                8,
                int(getattr(args, "target_focus_soft_injection_max_per_round", 3) or 3),
            )
            args.checkpoint_guided_budget_ratio = max(0.5, float(getattr(args, "checkpoint_guided_budget_ratio", 0.2) or 0.2))
            args.checkpoint_guided_max_per_round = max(8, int(getattr(args, "checkpoint_guided_max_per_round", 3) or 3))
            args.checkpoint_guided_trigger_vspc_stagnation = min(
                1,
                int(getattr(args, "checkpoint_guided_trigger_vspc_stagnation", 3) or 3),
            )
            trigger_levels = {
                item.strip()
                for item in _text(getattr(args, "checkpoint_guided_trigger_levels", "L1,L2")).split(",")
                if item.strip()
            }
            trigger_levels.update({"none", "L1", "L2"})
            args.checkpoint_guided_trigger_levels = ",".join(sorted(trigger_levels))
            args.tfi_agent_interval = min(5, int(getattr(args, "tfi_agent_interval", 20) or 20))
        args.fresh_agent_run = bool(args.rq1_t_sec > 0 or getattr(args, "fresh_agent_run", False))
        args.ignore_precomputed_paths = True
        args.ignore_precomputed_tfi_artifacts = True
        args.enable_stage = True
        args.force_gei_only = bool(args.rq1_k_sec == 0 and args.rq1_t_sec == 0)
        args.max_stage = "gei" if args.force_gei_only else "ksi"
    if (
        getattr(args, "enable_tfi_agent", False)
        and not getattr(args, "tfi_planning_only", False)
        and not getattr(args, "enable_guided_candidate_injection", False)
        and not getattr(args, "enable_checkpoint_guided_injection", False)
    ):
        args.enable_guided_candidate_injection = True
        args.enable_checkpoint_guided_injection = True
    if getattr(args, "enable_guided_candidate_injection", False):
        args.enable_target_focus_soft_injection = True
        args.enable_tfi_isolated_pilot = True
    if hasattr(args, "fuzz_active_budget_sec"):
        try:
            args.fuzz_active_budget_sec = (
                max(1, int(args.fuzz_active_budget_sec))
                if args.fuzz_active_budget_sec not in (None, "")
                else None
            )
        except Exception:
            args.fuzz_active_budget_sec = None
    if hasattr(args, "pure_fuzz_warmup_sec"):
        try:
            args.pure_fuzz_warmup_sec = max(0, int(args.pure_fuzz_warmup_sec or 0))
        except Exception:
            args.pure_fuzz_warmup_sec = 0
    if hasattr(args, "gei60_sec"):
        try:
            args.gei60_sec = max(1, int(args.gei60_sec or 60))
        except Exception:
            args.gei60_sec = 60
    if getattr(args, "pure_fuzz_warmup_sec", 0):
        args.enable_gei60 = True
        args.gei60_sec = int(args.pure_fuzz_warmup_sec)
    elif not getattr(args, "enable_gei60", False):
        args.pure_fuzz_warmup_sec = 0
    if hasattr(args, "wall_snapshot_sec"):
        try:
            args.wall_snapshot_sec = (
                max(1, int(args.wall_snapshot_sec))
                if args.wall_snapshot_sec not in (None, "")
                else None
            )
        except Exception:
            args.wall_snapshot_sec = None

    if args.positional_source:
        args.source = args.positional_source
    if args.solc_dir:
        os.environ["SOLCX_BINARY_PATH"] = args.solc_dir

    if not args.contract:
        args.contract = ""

    if args.source and args.contract.startswith("0x"):
        parser.error("--source requires --contract to be a name, not an address.")
    if args.source and args.blockchain_state and args.blockchain_state.isnumeric():
        parser.error("--source requires --blockchain-state to be a file, not a number.")

    if args.abi and not args.contract.startswith("0x"):
        parser.error("--abi requires --contract to be an address, not a name.")
    if args.abi and args.blockchain_state and not args.blockchain_state.isnumeric():
        parser.error("--abi requires --blockchain-state to be a number, not a file.")

    if args.evm_version:
        settings.EVM_VERSION = args.evm_version
    if not args.solc_version:
        selected_version = None
        if args.source:
            selected_version = _select_solc_version_for_source(args.source)
        if selected_version:
            args.solc_version = selected_version
        else:
            try:
                args.solc_version = "v" + str(solcx.get_solc_version())
            except Exception:
                args.solc_version = "v0.6.12"
    if not args.evm_version:
        settings.EVM_VERSION = _infer_evm_version_for_solc(args.solc_version)
    if args.generations:
        settings.GENERATIONS = args.generations
    if args.global_timeout:
        settings.GLOBAL_TIMEOUT = args.global_timeout
    if args.population_size:
        settings.POPULATION_SIZE = args.population_size
    if args.probability_crossover:
        settings.PROBABILITY_CROSSOVER = args.probability_crossover
    if args.probability_mutation:
        settings.PROBABILITY_MUTATION = args.probability_mutation

    if args.data_dependency is None:
        args.data_dependency = 1
    if args.constraint_solving is None:
        args.constraint_solving = 1
    if args.environmental_instrumentation is None:
        args.environmental_instrumentation = 1

    if args.environmental_instrumentation == 1:
        settings.ENVIRONMENTAL_INSTRUMENTATION = True
    elif args.environmental_instrumentation == 0:
        settings.ENVIRONMENTAL_INSTRUMENTATION = False

    if args.max_individual_length:
        settings.MAX_INDIVIDUAL_LENGTH = args.max_individual_length
    if args.max_symbolic_execution:
        settings.MAX_SYMBOLIC_EXECUTION = args.max_symbolic_execution
    if args.gas_limit:
        settings.GAS_LIMIT = args.gas_limit

    if args.abi:
        settings.REMOTE_FUZZING = True

    if args.rpc_host:
        settings.RPC_HOST = args.rpc_host
    if args.rpc_port:
        settings.RPC_PORT = args.rpc_port

    # cross contract
    if args.contract is None or args.contract == "" or args.cross_contract == 2:
        args.cross_contract = 2  # close
        args.depend_contracts = []
        args.trans_json_path = None
    else:
        if args.contract is None or args.contract == "":
            print(
                '\033[42;31m!!!!!!if open cross contract mode, you need specify a main contract which will be fuzzed!!!!!!\033[0m')
            print('\033[42;31m!!!!!!use --contract [Example]!!!!!!\033[0m')
            sys.exit(-1)
        if args.depend_contracts is None:
            print(
                '\033[42;31m!!!!!!if open cross contract mode, you need specify some contract names which depended by main contract!!!!!!\033[0m')
            print('\033[42;31m!!!!!!use --depend-contracts [A B C]!!!!!!\033[0m')
            sys.exit(-1)
        if args.constructor_args is None:
            print('\033[42;31m!!!!!!if open cross contract mode, you need specify some constructor args!!!!!!\033[0m')
            print('\033[42;31m!!!!!!use --constructor-args [address, uint, .....]!!!!!!\033[0m')
            sys.exit(-1)
    if args.trans_json_path is not None:
        settings.TRANS_INFO_JSON_PATH = args.trans_json_path
        print(f'\033[42;31m!!!!!!璁剧疆鐢ㄤ簬瀛樺偍浜嬪姟搴忓垪淇℃伅鐨刯son鍦板潃{settings.TRANS_INFO_JSON_PATH}!!!!!!\033[0m')
        if os.path.exists(settings.TRANS_INFO_JSON_PATH):
            print(
                f'\033[42;31m!!!!!!鐢ㄤ簬瀛樺偍浜嬪姟搴忓垪淇℃伅鐨刯son鍦板潃{settings.TRANS_INFO_JSON_PATH}宸茬粡瀛樺湪浜? 鐜板凡瑕嗙洊!!!!!!\033[0m')
    if args.trans_comp == 1:
        settings.TRANS_COMP_OPEN = True
    elif args.trans_comp == 2:
        settings.TRANS_COMP_OPEN = False
    settings.MAIN_CONTRACT_NAME = args.contract
    settings.SOLC_PATH_CROSS = args.solc_path_cross or _resolve_solc_executable(args.solc_dir, args.solc_version)
    settings.P_OPEN_CROSS = args.p_open_cross
    settings.CROSS_INIT_MODE = args.cross_init_mode
    settings.TRANS_SUPPORT_MODE = args.trans_mode
    if args.duplication == '0':
        settings.DUPLICATION = True
    else:
        settings.DUPLICATION = False
    if settings.SOLC_PATH_CROSS is None:
        print('\033[42;31m!!!!!!you need specify a solc path!!!!!!\033[0m')
        sys.exit(-1)

    return args


if '__main__' == __name__:
    main()

