# -*- coding: utf-8 -*-

"""TFI candidate pilot builder and isolated executor.

This module turns passed TFI materialization tasks into a small number of
scope-local pilot candidates. It can optionally execute ready candidates in an
isolated environment without injecting them into the main fuzzing population or
changing the main search loop.
"""

import copy
import json
import os
import re
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import solcx

try:
    from z3 import Solver
except ImportError:
    from z3.z3 import Solver

try:
    from eth_utils import encode_hex, to_canonical_address
except ImportError:
    from eth_utils import encode_hex, to_canonical_address

try:
    from detectors import DetectorExecutor
    from engine.analysis import ExecutionTraceAnalyzer, SymbolicTaintAnalyzer
    from engine.components import Generator, Individual
    from engine.environment import FuzzingEnvironment
    from evm import InstrumentedEVM
    from fuzzer.runtime.ksi_runtime import KSIRuntime
    from fuzzer.guidance.path_plan import PathHypothesis, PathPlan
    from fuzzer.guidance.path_progress import PathProgressTracker
    from utils.control_flow_graph import ControlFlowGraph
    from utils.utils import compile, get_function_signature_mapping, get_function_state_mutability_from_abi, get_interface_from_abi, get_pcs_and_jumpis
    from fuzzer.utils import settings
except ImportError:
    from fuzzer.detectors import DetectorExecutor
    from fuzzer.engine.analysis import ExecutionTraceAnalyzer, SymbolicTaintAnalyzer
    from fuzzer.engine.components import Generator, Individual
    from fuzzer.engine.environment import FuzzingEnvironment
    from fuzzer.evm import InstrumentedEVM
    from fuzzer.runtime.ksi_runtime import KSIRuntime
    from fuzzer.guidance.path_plan import PathHypothesis, PathPlan
    from fuzzer.guidance.path_progress import PathProgressTracker
    from fuzzer.utils import settings
    from fuzzer.utils.control_flow_graph import ControlFlowGraph
    from fuzzer.utils.utils import compile, get_function_signature_mapping, get_function_state_mutability_from_abi, get_interface_from_abi, get_pcs_and_jumpis


RESTRICTED_TERMS = [
    "APEBOYS",
    "DoKENV2",
    "setswap",
    "setFeR",
    "setWME",
    "setstaT",
    "setTargetLiquidity",
    "setFeeReceivers",
    "setIsTxLimitExempt",
    "approve_transfer",
    "approve_transferFrom",
    "preserve_approve_transfer",
]

SOURCE_ROOT = os.environ.get("FUZZ_SOURCE_ROOT", "")
DEFAULT_SOLC_DIR = os.environ.get("SOLC_DIR", "")


def _read_json(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _read_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
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
        return []
    return records


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _listify(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _clean_list(value: Any) -> List[str]:
    output: List[str] = []
    if not isinstance(value, list):
        return output
    for item in value:
        text = _text(item).strip()
        if text:
            output.append(text)
    return output


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_token(value: Any) -> str:
    text = _text(value).strip()
    if not text:
        return "NONE"
    output: List[str] = []
    for char in text:
        output.append(char if char.isalnum() else "_")
    sanitized = "".join(output).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or "NONE"


def _unique_preserve(values: Sequence[Any]) -> List[Any]:
    output: List[Any] = []
    seen = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, (dict, list)) else repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def _version_tuple(value: Any) -> Tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", _text(value))
    if not match:
        return (0, 0, 0)
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def _compare_versions(left: Any, right: Any) -> int:
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


def _matches_single_constraint(version_value: Any, token: str) -> bool:
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


def _pragma_matches(version_value: Any, pragma_text: str) -> bool:
    clauses = [clause.strip() for clause in _text(pragma_text).split("||") if clause.strip()]
    if not clauses:
        return True
    for clause in clauses:
        tokens = [token for token in re.split(r"[,\s]+", clause) if token]
        if all(_matches_single_constraint(version_value, token) for token in tokens):
            return True
    return False


def _select_solc_version_for_source(source_path: str) -> Optional[str]:
    try:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as handle:
            source_text = handle.read()
    except Exception:
        return None
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


def _extract_scope_hypothesis(scope_payload: Dict[str, Any]) -> PathPlan:
    hypotheses: List[PathHypothesis] = []
    seen = set()
    for item in _listify(scope_payload.get("scopes")):
        if not isinstance(item, dict):
            continue
        hypothesis_id = _text(item.get("hypothesis_id")).strip()
        path = _clean_list(item.get("path"))
        sink = _text(item.get("sink")).strip() or (path[-1] if path else None)
        marker = (hypothesis_id, tuple(path), sink)
        if not hypothesis_id or not path or marker in seen:
            continue
        seen.add(marker)
        hypotheses.append(
            PathHypothesis(
                id=hypothesis_id,
                vulnerability_type=_text(item.get("focus_type")).strip() or "target_focus_candidate_pilot",
                path=path,
                sink=sink,
                priority=1.0,
                expected_blockers=_clean_list(item.get("secondary_focus_types")),
                raw=dict(item),
            )
        )
    return PathPlan(hypotheses=hypotheses)


class TFICandidatePilot:
    def __init__(
        self,
        output_dir: str,
        materialization_tasks_path: Optional[str] = None,
        candidate_scope_path: Optional[str] = None,
        path_progress_path: Optional[str] = None,
        ksi_evidence_path: Optional[str] = None,
        selected_guided_sequences_path: Optional[str] = None,
        max_candidates_per_task: int = 3,
        max_total_candidates: Optional[int] = None,
        execute: bool = False,
        gas_limit: Optional[int] = None,
        contract_name: Optional[str] = None,
        source_path: Optional[str] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.materialization_tasks_path = materialization_tasks_path or os.path.join(
            self.output_dir, "target_focus_candidate_materialization_tasks.json"
        )
        self.candidate_scope_path = candidate_scope_path or os.path.join(self.output_dir, "target_focus_candidate_scope.json")
        self.path_progress_path = path_progress_path or os.path.join(self.output_dir, "path_progress.jsonl")
        self.ksi_evidence_path = ksi_evidence_path or os.path.join(self.output_dir, "ksi_evidence.jsonl")
        self.selected_guided_sequences_path = selected_guided_sequences_path or os.path.join(
            self.output_dir, "selected_guided_sequences.json"
        )
        self.max_candidates_per_task = max(1, int(max_candidates_per_task or 1))
        self.max_total_candidates = int(max_total_candidates) if max_total_candidates not in (None, "") else None
        self.execute = bool(execute)
        self.gas_limit = int(gas_limit) if gas_limit not in (None, "") else None
        self.contract_name = _text(contract_name).strip()
        self.source_path = os.path.abspath(source_path) if source_path else ""
        self._repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    def build_candidates(self) -> Dict[str, Any]:
        loaded_inputs = {
            "materialization_tasks": bool(self.materialization_tasks_path and os.path.exists(self.materialization_tasks_path)),
            "candidate_scope": bool(self.candidate_scope_path and os.path.exists(self.candidate_scope_path)),
            "path_progress": bool(self.path_progress_path and os.path.exists(self.path_progress_path)),
            "ksi_evidence": bool(self.ksi_evidence_path and os.path.exists(self.ksi_evidence_path)),
            "selected_guided_sequences": bool(
                self.selected_guided_sequences_path and os.path.exists(self.selected_guided_sequences_path)
            ),
        }
        missing_inputs = [name for name, loaded in loaded_inputs.items() if not loaded]

        tasks_payload = _read_json(self.materialization_tasks_path)
        scope_payload = _read_json(self.candidate_scope_path)

        if not loaded_inputs["materialization_tasks"]:
            return self._base_result(
                loaded_inputs=loaded_inputs,
                missing_inputs=missing_inputs,
                task_count=0,
                candidates=[],
            )

        scope_index = self._index_scopes(scope_payload)
        task_candidates: List[List[Dict[str, Any]]] = []
        tasks = [item for item in _listify(tasks_payload.get("tasks")) if isinstance(item, dict)]

        for task_index, task in enumerate(tasks, start=1):
            scope = scope_index.get(_text(task.get("scope_id")).strip(), {})
            if _text(task.get("task_type")).strip() == "binding_materialization_task":
                built = self._build_binding_candidates(task, scope, task_index)
            elif _text(task.get("task_type")).strip() == "scoped_candidate_materialization_task":
                built = self._build_scoped_candidates(task, scope, task_index)
            else:
                built = []
            if built:
                task_candidates.append(built[: self.max_candidates_per_task])

        selected_candidates = self._round_robin_candidates(task_candidates)
        return self._base_result(
            loaded_inputs=loaded_inputs,
            missing_inputs=missing_inputs,
            task_count=len(tasks),
            candidates=selected_candidates,
        )

    def run_candidates(
        self,
        candidates: List[Dict[str, Any]],
        task_count: int = 0,
        loaded_inputs: Optional[Dict[str, bool]] = None,
        missing_inputs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        base_result = self._base_result(
            loaded_inputs=loaded_inputs or {
                "materialization_tasks": bool(self.materialization_tasks_path and os.path.exists(self.materialization_tasks_path)),
                "candidate_scope": bool(self.candidate_scope_path and os.path.exists(self.candidate_scope_path)),
                "path_progress": bool(self.path_progress_path and os.path.exists(self.path_progress_path)),
                "ksi_evidence": bool(self.ksi_evidence_path and os.path.exists(self.ksi_evidence_path)),
                "selected_guided_sequences": bool(
                    self.selected_guided_sequences_path and os.path.exists(self.selected_guided_sequences_path)
                ),
            },
            missing_inputs=list(missing_inputs or []),
            task_count=int(task_count or 0),
            candidates=[item for item in candidates if isinstance(item, dict)],
        )
        if not self.execute:
            return base_result

        results = dict(base_result)
        results["pilot_mode"] = "isolated_execution"
        results["executed"] = True

        candidates = [item for item in _listify(base_result.get("candidates")) if isinstance(item, dict)]
        ready_candidates = [
            item
            for item in candidates
            if _text(item.get("materialization_status")).strip() == "ready"
            and bool(item.get("is_executable_candidate"))
        ]

        execution_items: List[Dict[str, Any]] = []
        execution_context = self._build_execution_context()
        deploy_check = self._build_deployment_context_check(
            execution_context=execution_context,
            deploy_attempted=False,
            deploy_success=False,
            deploy_error=None,
        )
        original_gas_limit = getattr(settings, "GAS_LIMIT", None)
        resolved_gas_limit = execution_context.get("gas_limit")
        if isinstance(resolved_gas_limit, int) and resolved_gas_limit > 0:
            settings.GAS_LIMIT = resolved_gas_limit

        try:
            env_context, env_error, deploy_check = self._prepare_execution_context()
            if env_context is None:
                skip_reason = self._skip_reason_from_env_error(env_error)
                for candidate in ready_candidates:
                    execution_items.append(
                        self._build_skipped_execution_result(candidate, skip_reason, env_error)
                    )
                results.update(self._execution_stats_from_results(execution_items, candidates, len(ready_candidates)))
                results["results"] = execution_items
                results["execution_environment_error"] = env_error
                results["execution_context"] = execution_context
                results["deployment_context_check"] = deploy_check
                return results

            historical_best_prefix = self._baseline_best_prefix_by_hypothesis(_read_jsonl(self.path_progress_path))
            historical_revert_index = self._baseline_revert_index(_read_jsonl(self.ksi_evidence_path))

            for candidate in ready_candidates:
                execution_items.append(
                    self._execute_candidate(
                        candidate=candidate,
                        env_context=env_context,
                        historical_best_prefix=historical_best_prefix,
                        historical_revert_index=historical_revert_index,
                    )
                )
        finally:
            if original_gas_limit is not None:
                settings.GAS_LIMIT = original_gas_limit
            runtime_dir = env_context.get("runtime_dir") if isinstance(locals().get("env_context"), dict) else None
            if runtime_dir and os.path.isdir(runtime_dir):
                try:
                    for name in os.listdir(runtime_dir):
                        try:
                            os.remove(os.path.join(runtime_dir, name))
                        except Exception:
                            pass
                    os.rmdir(runtime_dir)
                except Exception:
                    pass

        results.update(self._execution_stats_from_results(execution_items, candidates, len(ready_candidates)))
        results["results"] = execution_items
        results["execution_context"] = execution_context
        results["deployment_context_check"] = deploy_check
        return results

    def run_pilot(self) -> Dict[str, Any]:
        build_result = self.build_candidates()
        if not self.execute:
            return build_result

        return self.run_candidates(
            candidates=[item for item in _listify(build_result.get("candidates")) if isinstance(item, dict)],
            task_count=int(build_result.get("task_count") or 0),
            loaded_inputs=build_result.get("loaded_inputs") if isinstance(build_result.get("loaded_inputs"), dict) else None,
            missing_inputs=_clean_list(build_result.get("missing_inputs")),
        )

    def write_results(self, results: Dict[str, Any], filename: str = "target_focus_candidate_pilot_results.json") -> str:
        path = os.path.join(self.output_dir, filename)
        _write_json(path, results)
        return path

    def write_summary(self, results: Dict[str, Any], filename: str = "target_focus_candidate_pilot_summary.json") -> str:
        payload = self._build_summary(results)
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        return path

    def _base_result(
        self,
        loaded_inputs: Dict[str, bool],
        missing_inputs: List[str],
        task_count: int,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ready_count = 0
        partial_count = 0
        failed_count = 0
        binding_count = 0
        scoped_count = 0
        for candidate in candidates:
            status = _text(candidate.get("materialization_status")).strip()
            if status == "ready":
                ready_count += 1
            elif status == "partial":
                partial_count += 1
            elif status == "failed":
                failed_count += 1
            if _text(candidate.get("candidate_type")).strip() == "binding_pilot_candidate":
                binding_count += 1
            elif _text(candidate.get("candidate_type")).strip() == "scoped_selection_pilot_candidate":
                scoped_count += 1

        execution_context = self._build_execution_context()
        return {
            "enabled": True,
            "source": "target_focus_candidate_pilot_v1",
            "pilot_mode": "candidate_build_only",
            "executed": False,
            "loaded_inputs": loaded_inputs,
            "missing_inputs": missing_inputs,
            "task_count": int(task_count),
            "candidate_count": int(len(candidates)),
            "ready_candidate_count": int(ready_count),
            "partial_candidate_count": int(partial_count),
            "failed_candidate_count": int(failed_count),
            "binding_candidate_count": int(binding_count),
            "scoped_selection_candidate_count": int(scoped_count),
            "attempted_candidate_count": 0,
            "executed_candidate_count": 0,
            "skipped_count": 0,
            "skip_reason_counts": {
                "deploy_failed": 0,
                "materialization_not_ready": 0,
                "missing_execution_context": 0,
            },
            "candidates": candidates,
            "review_queue_ignored": True,
            "rejected_queue_ignored": True,
            "no_population_injection": True,
            "no_main_search_modification": True,
            "llm_called": False,
            "rag_used": False,
            "execution_context": execution_context,
            "deployment_context_check": self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=False,
                deploy_success=False,
                deploy_error=None,
            ),
            "global_constraints": {
                "no_fuzz_execution": not self.execute,
                "no_main_search_modification": True,
                "no_population_injection": True,
                "no_llm_call": True,
                "no_rag": True,
                "pilot_isolated_only": True,
            },
        }

    def _index_scopes(self, scope_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in _listify(scope_payload.get("scopes")):
            if not isinstance(item, dict):
                continue
            scope_id = _text(item.get("scope_id")).strip()
            if scope_id:
                indexed[scope_id] = item
        return indexed

    def _build_binding_candidates(
        self,
        task: Dict[str, Any],
        scope: Dict[str, Any],
        task_index: int,
    ) -> List[Dict[str, Any]]:
        skeleton = _clean_list(task.get("fixed_sequence_shape", {}).get("sequence_skeleton"))
        if not skeleton:
            return [
                self._build_binding_candidate_record(
                    task=task,
                    task_index=task_index,
                    variant_index=1,
                    skeleton=[],
                    sender_policy="none",
                    value_policy="none",
                    args_policy="none",
                    prestate_policy="none",
                    status="failed",
                    warnings=["missing_sequence_skeleton"],
                    reason="binding task missing sequence skeleton",
                )
            ]

        fields = task.get("allowed_materialization_fields") if isinstance(task.get("allowed_materialization_fields"), dict) else {}
        sender_options = _clean_list(fields.get("sender", {}).get("candidate_policies")) or ["keep_same_sender_across_sequence"]
        value_options = _clean_list(fields.get("value", {}).get("candidate_policies")) or ["keep_observed_success_value"]
        args_options = _clean_list(fields.get("args", {}).get("candidate_policies")) or ["abi_default_or_boundary_mix"]
        prestate_options = _clean_list(fields.get("prestate", {}).get("candidate_setup_hints")) or ["none"]

        variants: List[Tuple[str, str, str, str]] = []
        base = (sender_options[0], value_options[0], args_options[0], prestate_options[0])
        variants.append(base)
        for sender_policy in sender_options[1:]:
            variants.append((sender_policy, base[1], base[2], base[3]))
        for value_policy in value_options[1:]:
            variants.append((base[0], value_policy, base[2], base[3]))
        for args_policy in args_options[1:]:
            variants.append((base[0], base[1], args_policy, base[3]))
        for prestate_policy in prestate_options[1:]:
            variants.append((base[0], base[1], base[2], prestate_policy))

        unique_variants = _unique_preserve(variants)[: self.max_candidates_per_task]
        built: List[Dict[str, Any]] = []
        for variant_index, (sender_policy, value_policy, args_policy, prestate_policy) in enumerate(unique_variants, start=1):
            warnings: List[str] = []
            if prestate_policy != "none":
                warnings.append("prestate_policy_not_materialized_in_pilot_v1")
            built.append(
                self._build_binding_candidate_record(
                    task=task,
                    task_index=task_index,
                    variant_index=variant_index,
                    skeleton=skeleton,
                    sender_policy=sender_policy,
                    value_policy=value_policy,
                    args_policy=args_policy,
                    prestate_policy=prestate_policy,
                    status="ready",
                    warnings=warnings,
                    reason="binding pilot candidate preserves task skeleton and varies only binding fields",
                )
            )
        return built

    def _build_binding_candidate_record(
        self,
        task: Dict[str, Any],
        task_index: int,
        variant_index: int,
        skeleton: List[str],
        sender_policy: str,
        value_policy: str,
        args_policy: str,
        prestate_policy: str,
        status: str,
        warnings: List[str],
        reason: str,
    ) -> Dict[str, Any]:
        task_token = _safe_token(task.get("task_id"))
        scope_token = _safe_token(task.get("scope_id"))
        return {
            "candidate_id": "TFI_CAND_{:04d}_{}_{}_V{:02d}".format(task_index, scope_token, task_token, variant_index),
            "candidate_type": "binding_pilot_candidate",
            "source_task_id": task.get("task_id"),
            "scope_id": task.get("scope_id"),
            "hypothesis_id": task.get("hypothesis_id"),
            "target_function": task.get("target_function"),
            "frontier": task.get("frontier"),
            "sink": task.get("sink"),
            "sequence_skeleton": list(skeleton),
            "guided_sequence": list(skeleton),
            "preserve_function_order": True,
            "binding_choices": {
                "sender_policy_used": sender_policy,
                "value_policy_used": value_policy,
                "args_policy_used": args_policy,
                "prestate_policy_used": prestate_policy,
            },
            "candidate_constraints": {
                "do_not_change_function_order": True,
                "do_not_add_function": True,
                "do_not_drop_function": True,
                "only_bind_sender_value_args_prestate": True,
            },
            "is_executable_candidate": status == "ready" and bool(skeleton),
            "materialization_status": status,
            "materialization_warnings": warnings,
            "reason": reason,
        }

    def _build_scoped_candidates(
        self,
        task: Dict[str, Any],
        scope: Dict[str, Any],
        task_index: int,
    ) -> List[Dict[str, Any]]:
        required_prefix = _clean_list(task.get("required_prefix", {}).get("observed_matched_prefix"))
        frontier = _text(task.get("required_progress", {}).get("frontier")).strip() or _text(task.get("frontier")).strip()
        sink = _text(task.get("required_progress", {}).get("sink")).strip() or _text(task.get("sink")).strip()
        target_function = _text(task.get("target_function")).strip()
        constraints = task.get("materialization_constraints") if isinstance(task.get("materialization_constraints"), dict) else {}
        allow_scope_local_composition = (
            _text(task.get("diagnosis_type")).strip() == "possible_local_candidate_gap"
            or bool(constraints.get("allow_scope_local_action_composition"))
        )
        nonlocal_examples = self._nonlocal_skeleton_tuples(task)

        if not required_prefix and not allow_scope_local_composition:
            return [
                self._build_scoped_candidate_record(
                    task=task,
                    task_index=task_index,
                    variant_index=1,
                    sequence=[],
                    status="partial",
                    warnings=["observed_prefix_missing"],
                    source_used="scope_local_path_missing_prefix",
                    reason="scoped task missing observed prefix for pilot candidate construction",
                )
            ]

        candidate_sequences: List[Tuple[List[str], str]] = []

        if not required_prefix and allow_scope_local_composition:
            composed = self._scope_local_composed_sequences(
                target_function=target_function,
                frontier=frontier,
                sink=sink,
            )
            candidate_sequences.extend(composed)

        if frontier:
            seq = list(required_prefix)
            if not seq or seq[-1] != frontier:
                seq.append(frontier)
            candidate_sequences.append((seq, "required_prefix_plus_frontier"))
        if sink and sink != frontier:
            seq = list(required_prefix)
            if not seq or seq[-1] != frontier:
                if frontier:
                    seq.append(frontier)
            if not seq or seq[-1] != sink:
                seq.append(sink)
            candidate_sequences.append((seq, "required_prefix_plus_frontier_plus_sink"))

        for partial in _listify(task.get("partial_candidates")):
            if not isinstance(partial, dict):
                continue
            sequence = _clean_list(partial.get("guided_sequence"))
            if not sequence:
                continue
            if sequence[: len(required_prefix)] != required_prefix:
                continue
            if frontier and frontier not in sequence and sink and sink not in sequence:
                continue
            marker = tuple(sequence)
            if marker in nonlocal_examples:
                continue
            candidate_sequences.append((sequence, "validated_partial_candidate"))

        built: List[Dict[str, Any]] = []
        unique_sequences = _unique_preserve(candidate_sequences)[: self.max_candidates_per_task]
        for variant_index, item in enumerate(unique_sequences, start=1):
            sequence, source_used = item
            warnings: List[str] = []
            status = "ready"
            if not sequence:
                status = "failed"
                warnings.append("empty_candidate_sequence")
            elif sequence[: len(required_prefix)] != required_prefix:
                status = "failed"
                warnings.append("observed_prefix_not_preserved")
            elif frontier and frontier not in sequence and sink and sink not in sequence:
                status = "failed"
                warnings.append("missing_frontier_or_sink")
            elif tuple(sequence) in nonlocal_examples:
                status = "failed"
                warnings.append("nonlocal_generic_skeleton_detected")
            built.append(
                self._build_scoped_candidate_record(
                    task=task,
                    task_index=task_index,
                    variant_index=variant_index,
                    sequence=sequence,
                    status=status,
                    warnings=warnings,
                    source_used=source_used,
                    reason="scoped pilot candidate preserves observed prefix and stays target-local",
                )
            )
        return built

    def _scope_local_composed_sequences(
        self,
        *,
        target_function: str,
        frontier: str,
        sink: str,
    ) -> List[Tuple[List[str], str]]:
        combos = [
            ([target_function, frontier, sink], "scope_local_target_frontier_sink"),
            ([frontier, sink], "scope_local_frontier_sink"),
            ([target_function, sink], "scope_local_target_sink"),
            ([target_function, frontier], "scope_local_target_frontier"),
            ([frontier], "scope_local_frontier"),
            ([sink], "scope_local_sink"),
            ([target_function], "scope_local_target"),
        ]
        output: List[Tuple[List[str], str]] = []
        seen = set()
        for raw_sequence, source_used in combos:
            sequence: List[str] = []
            for name in raw_sequence:
                text = _text(name).strip()
                if text and text not in sequence:
                    sequence.append(text)
            marker = tuple(sequence)
            if not marker or marker in seen:
                continue
            seen.add(marker)
            output.append((sequence, source_used))
        return output

    def _build_scoped_candidate_record(
        self,
        task: Dict[str, Any],
        task_index: int,
        variant_index: int,
        sequence: List[str],
        status: str,
        warnings: List[str],
        source_used: str,
        reason: str,
    ) -> Dict[str, Any]:
        task_token = _safe_token(task.get("task_id"))
        scope_token = _safe_token(task.get("scope_id"))
        return {
            "candidate_id": "TFI_CAND_{:04d}_{}_{}_V{:02d}".format(task_index, scope_token, task_token, variant_index),
            "candidate_type": "scoped_selection_pilot_candidate",
            "source_task_id": task.get("task_id"),
            "scope_id": task.get("scope_id"),
            "hypothesis_id": task.get("hypothesis_id"),
            "target_function": task.get("target_function"),
            "frontier": task.get("frontier"),
            "sink": task.get("sink"),
            "required_prefix": _clean_list(task.get("required_prefix", {}).get("observed_matched_prefix")),
            "candidate_sequence": list(sequence),
            "guided_sequence": list(sequence),
            "candidate_shape": {
                "preserve_observed_prefix": bool(_clean_list(task.get("required_prefix", {}).get("observed_matched_prefix"))),
                "must_touch_frontier_or_sink": True,
                "frontier": task.get("frontier"),
                "sink": task.get("sink"),
            },
            "candidate_constraints": {
                "must_preserve_observed_prefix": bool(_clean_list(task.get("required_prefix", {}).get("observed_matched_prefix"))),
                "must_touch_frontier_or_sink": True,
                "must_not_use_nonlocal_generic_skeleton": True,
                "must_not_select_target_seen_without_path_progress": True,
                "scope_local_action_composition": _text(task.get("diagnosis_type")).strip() == "possible_local_candidate_gap",
            },
            "candidate_source_used": source_used,
            "is_executable_candidate": status == "ready" and bool(sequence),
            "materialization_status": status,
            "materialization_warnings": warnings,
            "reason": reason,
        }

    def _nonlocal_skeleton_tuples(self, task: Dict[str, Any]) -> set:
        tuples = set()
        policy = task.get("generic_skeleton_policy") if isinstance(task.get("generic_skeleton_policy"), dict) else {}
        for item in _listify(policy.get("nonlocal_skeleton_examples")):
            if not isinstance(item, dict):
                continue
            sequence = _clean_list(item.get("binding_skeleton")) or _clean_list(item.get("generic_skeleton_functions"))
            if sequence:
                tuples.add(tuple(sequence))
        return tuples

    def _round_robin_candidates(self, task_candidates: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        if not task_candidates:
            return selected
        max_depth = max(len(item) for item in task_candidates)
        for depth in range(max_depth):
            for candidates in task_candidates:
                if depth >= len(candidates):
                    continue
                selected.append(candidates[depth])
                if self.max_total_candidates is not None and len(selected) >= self.max_total_candidates:
                    return selected
        return selected

    def _prepare_execution_context(
        self,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        execution_context = self._build_execution_context()
        gas_limit = execution_context.get("gas_limit")
        contract_name = self._discover_contract_name()
        if not contract_name:
            return None, "contract_name_not_resolved", self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=False,
                deploy_success=False,
                deploy_error="contract_name_not_resolved",
            )

        source_path = self._discover_source_path(contract_name)
        if not source_path or not os.path.exists(source_path):
            return None, "contract_source_not_found", self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=False,
                deploy_success=False,
                deploy_error="contract_source_not_found",
            )

        solc_dir = DEFAULT_SOLC_DIR if os.path.isdir(DEFAULT_SOLC_DIR) else None
        if solc_dir:
            os.environ["SOLCX_BINARY_PATH"] = solc_dir
        solc_version = _select_solc_version_for_source(source_path) or "v0.6.12"

        try:
            compiler_output = compile(solc_version, settings.EVM_VERSION, source_path)
        except Exception as exc:
            return None, "compile_failed: {}".format(exc), self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=False,
                deploy_success=False,
                deploy_error="compile_failed: {}".format(exc),
            )
        if not compiler_output:
            return None, "compile_failed", self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=False,
                deploy_success=False,
                deploy_error="compile_failed",
            )

        contract_payload = {}
        for name, payload in (compiler_output.get("contracts", {}).get(source_path, {}) or {}).items():
            if name == contract_name:
                contract_payload = payload
                break
        if not contract_payload:
            return None, "contract_not_found_in_compiler_output", self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=False,
                deploy_success=False,
                deploy_error="contract_not_found_in_compiler_output",
            )

        abi = contract_payload.get("abi") or []
        deployment_bytecode = _text(contract_payload.get("evm", {}).get("bytecode", {}).get("object")).strip()
        runtime_bytecode = _text(contract_payload.get("evm", {}).get("deployedBytecode", {}).get("object")).strip()
        if not abi or not deployment_bytecode or not runtime_bytecode:
            return None, "abi_or_bytecode_missing", self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=False,
                deploy_success=False,
                deploy_error="abi_or_bytecode_missing",
            )

        try:
            instrumented_evm = InstrumentedEVM(settings.RPC_HOST, settings.RPC_PORT)
            instrumented_evm.set_vm_by_name(settings.EVM_VERSION)
            instrumented_evm.create_fake_accounts()
        except Exception as exc:
            return None, "instrumented_evm_init_failed: {}".format(exc), self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=False,
                deploy_success=False,
                deploy_error="instrumented_evm_init_failed: {}".format(exc),
            )

        settings.MAIN_CONTRACT_NAME = contract_name
        settings.TRANS_INFO = {"start_time": str(time.time())}
        settings.DEPLOYED_CONTRACT_ADDRESS = {}

        try:
            deploy_result = instrumented_evm.deploy_contract(
                instrumented_evm.accounts[0],
                deployment_bytecode,
                gas=gas_limit if isinstance(gas_limit, int) and gas_limit > 0 else None,
            )
        except Exception as exc:
            error_text = "deploy_failed: {}".format(exc)
            return None, error_text, self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=True,
                deploy_success=False,
                deploy_error=error_text,
            )
        if getattr(deploy_result, "is_error", False):
            error_text = "deploy_failed: {}".format(_text(getattr(deploy_result, "_error", "")) or "unknown_error")
            return None, error_text, self._build_deployment_context_check(
                execution_context=execution_context,
                deploy_attempted=True,
                deploy_success=False,
                deploy_error=error_text,
            )

        contract_address = encode_hex(deploy_result.msg.storage_address)
        instrumented_evm.accounts.append(contract_address)
        settings.TRANS_INFO[contract_name] = contract_address
        settings.DEPLOYED_CONTRACT_ADDRESS[contract_name] = contract_address

        interface, interface_mapper = get_interface_from_abi(abi)
        function_state_mutability = get_function_state_mutability_from_abi(abi)
        if "constructor" in interface:
            del interface["constructor"]
        if contract_address in instrumented_evm.accounts:
            instrumented_evm.accounts.remove(contract_address)

        instrumented_evm.create_snapshot()
        deployed_runtime = instrumented_evm.get_code(to_canonical_address(contract_address)).hex()
        overall_pcs, overall_jumpis = get_pcs_and_jumpis(deployed_runtime or runtime_bytecode)

        cfg = ControlFlowGraph()
        try:
            cfg.build(deployed_runtime or runtime_bytecode, settings.EVM_VERSION)
        except Exception:
            cfg = None

        solver = Solver()
        try:
            solver.set("timeout", settings.SOLVER_TIMEOUT)
        except Exception:
            pass

        args = SimpleNamespace(
            constraint_solving=1,
            data_dependency=1,
            environmental_instrumentation=1,
            enable_path_validation=True,
            enable_ksi_evidence=True,
        )

        runtime_dir = tempfile.mkdtemp(prefix="target_focus_candidate_pilot_", dir=self.output_dir or None)
        scope_payload = _read_json(self.candidate_scope_path)
        path_plan = _extract_scope_hypothesis(scope_payload)
        path_progress_tracker = PathProgressTracker(path_plan=path_plan, output_dir=runtime_dir, enabled=not path_plan.is_empty())
        ksi_runtime = KSIRuntime(output_dir=runtime_dir, enabled=True)

        env = FuzzingEnvironment(
            instrumented_evm=instrumented_evm,
            contract_name=contract_name,
            solver=solver,
            results={"errors": {}},
            symbolic_taint_analyzer=SymbolicTaintAnalyzer(),
            detector_executor=DetectorExecutor(source_map=None, function_signature_mapping=get_function_signature_mapping(abi)),
            interface=interface,
            overall_pcs=overall_pcs,
            overall_jumpis=overall_jumpis,
            len_overall_pcs_with_children=0,
            other_contracts=[],
            args=args,
            seed=0,
            cfg=cfg,
            abi=abi,
            path_progress_tracker=path_progress_tracker,
            ksi_runtime=ksi_runtime,
        )
        env.current_generation = 0
        env.execution_begin = time.time()

        generator = Generator(
            interface=interface,
            bytecode=deployment_bytecode,
            accounts=instrumented_evm.accounts,
            contract=contract_address,
            other_generators=[],
            interface_mapper=interface_mapper,
            contract_name=contract_name,
            sol_path=source_path,
            function_state_mutability=function_state_mutability,
        )
        analyzer = ExecutionTraceAnalyzer(env)
        return {
            "contract_name": contract_name,
            "source_path": source_path,
            "generator": generator,
            "env": env,
            "analyzer": analyzer,
            "runtime_dir": runtime_dir,
            "runtime_path_progress": os.path.join(runtime_dir, "path_progress.jsonl"),
            "runtime_ksi": os.path.join(runtime_dir, "ksi_evidence.jsonl"),
        }, None, self._build_deployment_context_check(
            execution_context=execution_context,
            deploy_attempted=True,
            deploy_success=True,
            deploy_error=None,
        )

    def _discover_contract_name(self) -> str:
        if self.contract_name:
            return self.contract_name
        if self.output_dir and os.path.isdir(self.output_dir):
            preferred = []
            for name in os.listdir(self.output_dir):
                if not name.lower().endswith(".json"):
                    continue
                if name.startswith("tfi_") or name.startswith("path_") or name.startswith("repair_") or name.startswith("side_"):
                    continue
                preferred.append(name)
            preferred.sort()
            if preferred:
                return os.path.splitext(preferred[0])[0]
        basename = os.path.basename(os.path.dirname(self.output_dir))
        return _text(basename).strip()

    def _discover_source_path(self, contract_name: str) -> Optional[str]:
        if self.source_path and os.path.exists(self.source_path):
            return self.source_path
        if not contract_name or not os.path.isdir(SOURCE_ROOT):
            return None
        candidates = sorted(
            os.path.join(SOURCE_ROOT, name)
            for name in os.listdir(SOURCE_ROOT)
            if name.startswith(contract_name + "__") and name.endswith(".sol")
        )
        return candidates[0] if candidates else None

    def _execute_candidate(
        self,
        candidate: Dict[str, Any],
        env_context: Dict[str, Any],
        historical_best_prefix: Dict[str, int],
        historical_revert_index: Dict[Tuple[str, ...], bool],
    ) -> Dict[str, Any]:
        generator = env_context["generator"]
        env = env_context["env"]
        analyzer = env_context["analyzer"]
        sequence = _clean_list(candidate.get("guided_sequence")) or _clean_list(candidate.get("sequence_skeleton")) or _clean_list(candidate.get("candidate_sequence"))

        materialized = self._materialize_candidate_to_individual(candidate, generator, sequence)
        if materialized[0] is None:
            return self._build_skipped_execution_result(
                candidate,
                "materialization_failed",
                _text(materialized[1].get("skip_reason")).strip() or "materialization_failed",
            )

        individual = materialized[0]
        sequence_id = _text(candidate.get("candidate_id")).strip() or "TFI_CAND_UNKNOWN"
        setattr(individual, "sequence_id", sequence_id)
        setattr(individual, "individual_id", sequence_id)
        setattr(individual, "generation_id", sequence_id)

        try:
            analyzer.execution_function(individual, env)
        except Exception as exc:
            return {
                "candidate_id": candidate.get("candidate_id"),
                "source_task_id": candidate.get("source_task_id"),
                "candidate_type": candidate.get("candidate_type"),
                "execution_status": "error",
                "revert": False,
                "revert_reason": str(exc),
                "path_progress_delta": None,
                "coverage_delta": None,
                "frontier_touched": None,
                "sink_touched": None,
                "sink_reverted": None,
                "target_seen": None,
                "useful": False,
                "useful_reason": "execution_error",
            }

        ksi_records = [
            item
            for item in _read_jsonl(env_context["runtime_ksi"])
            if _text(item.get("sequence_id")).strip() == sequence_id
        ]
        progress_records = [
            item
            for item in _read_jsonl(env_context["runtime_path_progress"])
            if _text(item.get("sequence_id")).strip() == sequence_id
        ]

        observations = {}
        for item in ksi_records:
            observed = item.get("observations") if isinstance(item.get("observations"), dict) else {}
            if observed:
                observations = observed
                break

        observed_functions = _clean_list(observations.get("functions"))
        hypothesis_id = _text(candidate.get("hypothesis_id")).strip()
        baseline_best = historical_best_prefix.get(hypothesis_id, 0)
        new_best = 0
        for record in progress_records:
            if _text(record.get("matched_hypothesis")).strip() == hypothesis_id:
                new_best = max(new_best, _safe_int(record.get("matched_prefix_len"), 0))
        path_progress_delta = max(0, new_best - baseline_best) if hypothesis_id else None

        frontier = _text(candidate.get("frontier")).strip()
        sink = _text(candidate.get("sink")).strip()
        target_function = _text(candidate.get("target_function")).strip()
        frontier_touched = frontier in observed_functions if frontier else False
        sink_touched = sink in observed_functions if sink else False
        target_seen = target_function in observed_functions if target_function else False
        sink_reverted = any(_text(item.get("status")).strip() == "sink_reverted" for item in progress_records)

        revert = bool(observations.get("revert", False))
        revert_reason = _text(observations.get("revert_reason")).strip() or None
        coverage_delta_raw = observations.get("coverage_delta")
        coverage_delta = int(coverage_delta_raw) if isinstance(coverage_delta_raw, (int, float)) else None

        execution_status = "success"
        if revert:
            execution_status = "revert"
        if not ksi_records and not progress_records:
            execution_status = "skipped"

        useful = False
        useful_reason = "no_frontier_or_sink_touch"
        if path_progress_delta is not None and path_progress_delta > 0:
            useful = True
            useful_reason = "path_progress_delta"
        elif frontier_touched:
            useful = True
            useful_reason = "frontier_touched"
        elif sink_touched and not sink_reverted and not revert:
            useful = True
            useful_reason = "sink_touched_without_revert"
        else:
            history_reverted = historical_revert_index.get(tuple(sequence), False)
            if history_reverted and not revert:
                useful = True
                useful_reason = "revert_reduced"
            elif coverage_delta is not None and coverage_delta > 0 and (frontier_touched or sink_touched or target_seen):
                useful = True
                useful_reason = "target_related_coverage_delta"
            elif target_seen and not frontier_touched and not sink_touched and not (path_progress_delta or 0) > 0:
                useful_reason = "target_seen_without_path_progress"
            elif revert:
                useful_reason = "reverted"

        return {
            "candidate_id": candidate.get("candidate_id"),
            "source_task_id": candidate.get("source_task_id"),
            "candidate_type": candidate.get("candidate_type"),
            "execution_status": execution_status,
            "executed": True,
            "skipped": False,
            "skip_reason": None,
            "revert": revert,
            "revert_reason": revert_reason,
            "path_progress_delta": path_progress_delta,
            "coverage_delta": coverage_delta,
            "frontier_touched": frontier_touched,
            "sink_touched": sink_touched,
            "sink_reverted": sink_reverted,
            "target_seen": target_seen,
            "useful": useful,
            "useful_reason": useful_reason,
        }

    def _build_skipped_execution_result(
        self,
        candidate: Dict[str, Any],
        reason: str,
        detail: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "candidate_id": candidate.get("candidate_id"),
            "source_task_id": candidate.get("source_task_id"),
            "candidate_type": candidate.get("candidate_type"),
            "execution_status": "skipped",
            "executed": False,
            "skipped": True,
            "skip_reason": reason,
            "revert": False,
            "revert_reason": _text(detail).strip() or None,
            "path_progress_delta": None,
            "coverage_delta": None,
            "frontier_touched": None,
            "sink_touched": None,
            "sink_reverted": None,
            "target_seen": None,
            "useful": False,
            "useful_reason": reason,
        }

    def _baseline_best_prefix_by_hypothesis(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        indexed: Dict[str, int] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            hypothesis_id = _text(record.get("matched_hypothesis")).strip() or _text(record.get("hypothesis_id")).strip()
            if not hypothesis_id:
                continue
            indexed[hypothesis_id] = max(indexed.get(hypothesis_id, 0), _safe_int(record.get("matched_prefix_len"), 0))
        return indexed

    def _baseline_revert_index(self, ksi_records: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], bool]:
        indexed: Dict[Tuple[str, ...], bool] = {}
        for record in ksi_records:
            if not isinstance(record, dict):
                continue
            observations = record.get("observations") if isinstance(record.get("observations"), dict) else {}
            functions = tuple(_clean_list(observations.get("functions")))
            if not functions:
                continue
            indexed[functions] = indexed.get(functions, False) or bool(observations.get("revert", False))
        return indexed

    def _materialize_candidate_to_individual(
        self,
        candidate: Dict[str, Any],
        generator: Any,
        sequence: List[str],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        if not sequence:
            return None, {"skip_reason": "empty_guided_sequence"}

        chromosome: List[Dict[str, Any]] = []
        binding_choices = candidate.get("binding_choices") if isinstance(candidate.get("binding_choices"), dict) else {}
        sender_policy = _text(binding_choices.get("sender_policy_used")).strip() or "keep_same_sender_across_sequence"
        value_policy = _text(binding_choices.get("value_policy_used")).strip() or "keep_observed_success_value"
        args_policy = _text(binding_choices.get("args_policy_used")).strip() or "abi_default_or_boundary_mix"
        account_pool = list(getattr(generator, "accounts", []) or [])
        if not account_pool:
            return None, {"skip_reason": "account_pool_empty"}

        address_memo: Dict[int, Any] = {}
        numeric_memo: Dict[int, Any] = {}

        for gene_index, function_name in enumerate(sequence):
            resolved = self._resolve_function(generator, function_name)
            if resolved is None:
                return None, {"skip_reason": "function_not_resolved"}
            function_selector, argument_types, state_mutability = resolved
            temp_generator = self._clone_generator(generator)
            generated = temp_generator.generate_individual(function_selector, argument_types)
            if not generated:
                return None, {"skip_reason": "generator_failed"}
            gene = copy.deepcopy(generated[0])
            gene["account"] = self._pick_sender(account_pool, sender_policy, gene_index)
            gene["contract"] = getattr(generator, "contract", gene.get("contract"))
            gene["amount"] = self._pick_value(
                value_policy,
                gene.get("amount"),
                gene_index,
                function_name,
                state_mutability,
            )
            gene["arguments"] = self._build_generic_arguments(
                function_selector=function_selector,
                argument_types=argument_types,
                base_arguments=list(gene.get("arguments") or [])[1:],
                account_pool=account_pool,
                sender=gene["account"],
                policy=args_policy,
                gene_index=gene_index,
                address_memo=address_memo,
                numeric_memo=numeric_memo,
            )
            if gene["arguments"] is None:
                return None, {"skip_reason": "argument_construction_failed"}
            chromosome.append(gene)

        individual = Individual(generator=generator, other_generators=[]).init(chromosome=chromosome)
        return individual, {"skip_reason": None}

    def _resolve_function(self, generator: Any, function_name: str) -> Optional[Tuple[str, List[str], str]]:
        normalized = _text(function_name).strip()
        if not normalized:
            return None
        if normalized in ("fallback", "receive") and "fallback" in getattr(generator, "interface", {}):
            state_mutability = self._resolve_state_mutability(
                generator,
                selector="fallback",
                function_name=normalized,
            )
            return "fallback", list(getattr(generator, "interface", {}).get("fallback") or []), state_mutability
        interface_mapper = getattr(generator, "interface_mapper", {}) or {}
        matches: List[Tuple[str, str]] = []
        for signature, selector in interface_mapper.items():
            if _text(signature).split("(", 1)[0] == normalized:
                matches.append((_text(signature), _text(selector)))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0])
        signature, selector = matches[0]
        interface = getattr(generator, "interface", {}) or {}
        state_mutability = self._resolve_state_mutability(
            generator,
            selector=selector,
            signature=signature,
            function_name=normalized,
        )
        return selector, list(interface.get(selector) or []), state_mutability

    def _clone_generator(self, generator: Any) -> Any:
        cloned = generator.__class__(
            interface=copy.deepcopy(getattr(generator, "interface", {})),
            bytecode=getattr(generator, "bytecode", None),
            accounts=list(getattr(generator, "accounts", []) or []),
            contract=getattr(generator, "contract", None),
            other_generators=list(getattr(generator, "other_generators", []) or []),
            interface_mapper=copy.deepcopy(getattr(generator, "interface_mapper", {})),
            contract_name=getattr(generator, "contract_name", None),
            sol_path=getattr(generator, "sol_path", None),
            function_state_mutability=copy.deepcopy(getattr(generator, "function_state_mutability", {})),
        )
        try:
            cloned.update_other_generators(
                list(getattr(generator, "other_generators", []) or []),
                copy.deepcopy(getattr(generator, "total_interface_mapper", {})),
            )
        except Exception:
            pass
        return cloned

    @staticmethod
    def _resolve_state_mutability(
        generator: Any,
        selector: Optional[str] = None,
        signature: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> str:
        resolver = getattr(generator, "resolve_state_mutability", None)
        if callable(resolver):
            try:
                return _text(
                    resolver(selector=selector, signature=signature, function_name=function_name)
                ).strip().lower() or "nonpayable"
            except Exception:
                pass
        mapping = getattr(generator, "function_state_mutability", {}) or {}
        for key in (signature, function_name, selector):
            text = _text(key).strip()
            if not text:
                continue
            state_mutability = _text(mapping.get(text)).strip().lower()
            if state_mutability:
                return state_mutability
        return "nonpayable"

    def _pick_sender(self, account_pool: Sequence[str], policy: str, gene_index: int) -> str:
        if not account_pool:
            return ""
        normalized = _text(policy).strip()
        if normalized == "rotate_existing_senders":
            return _text(account_pool[gene_index % len(account_pool)]).strip()
        return _text(account_pool[0]).strip()

    def _pick_value(self, policy: str, base_value: Any, gene_index: int, function_name: str, state_mutability: Optional[str] = None) -> int:
        normalized = _text(policy).strip()
        normalized_mutability = _text(state_mutability).strip().lower() or "nonpayable"
        accepts_value = _text(function_name).strip() in ("fallback", "receive") or normalized_mutability == "payable"
        if normalized == "force_zero_value":
            return 0
        if not accepts_value:
            return 0
        if normalized == "try_small_nonzero_value":
            return 1
        if normalized == "conservative_zero_or_small_value":
            return 1 if gene_index == 0 else 0
        if normalized == "keep_observed_success_value":
            if isinstance(base_value, int):
                return base_value
            return 0
        if _text(function_name).strip() in ("fallback", "receive"):
            return 1
        return base_value if isinstance(base_value, int) else 0

    def _build_generic_arguments(
        self,
        function_selector: str,
        argument_types: Sequence[str],
        base_arguments: List[Any],
        account_pool: Sequence[str],
        sender: str,
        policy: str,
        gene_index: int,
        address_memo: Dict[int, Any],
        numeric_memo: Dict[int, Any],
    ) -> Optional[List[Any]]:
        while len(base_arguments) < len(argument_types):
            base_arguments.append(None)
        values: List[Any] = []
        lowered_policy = _text(policy).strip().lower()
        for arg_index, arg_type in enumerate(argument_types):
            normalized_type = _text(arg_type).replace(" storage", "").replace(" memory", "").strip().lower()
            base_value = base_arguments[arg_index]
            if "[" in normalized_type and "]" in normalized_type:
                values.append(base_value if base_value is not None else [])
                continue
            if normalized_type.startswith("address"):
                if "preserve_cross_call_argument_consistency" in lowered_policy:
                    if arg_index not in address_memo:
                        address_memo[arg_index] = account_pool[(arg_index + 1) % len(account_pool)] if account_pool else sender
                    values.append(address_memo[arg_index])
                else:
                    values.append(account_pool[(gene_index + arg_index + 1) % len(account_pool)] if account_pool else sender)
                continue
            if normalized_type.startswith("bool"):
                if "boolean_true_false" in lowered_policy:
                    values.append(bool((gene_index + arg_index) % 2 == 0))
                elif isinstance(base_value, bool):
                    values.append(base_value)
                else:
                    values.append(False)
                continue
            if normalized_type.startswith("uint") or normalized_type.startswith("int"):
                if "preserve_cross_call_argument_consistency" in lowered_policy and arg_index in numeric_memo:
                    values.append(numeric_memo[arg_index])
                    continue
                value = 0
                if "amount_boundary" in lowered_policy or "mutate_boundary_arguments" in lowered_policy:
                    value = self._boundary_numeric_value(normalized_type, gene_index + arg_index)
                elif isinstance(base_value, int):
                    value = base_value
                elif "abi_default_or_boundary_mix" in lowered_policy:
                    value = 1 if (gene_index + arg_index) % 2 == 0 else 0
                if "preserve_cross_call_argument_consistency" in lowered_policy:
                    numeric_memo[arg_index] = value
                values.append(value)
                continue
            if normalized_type.startswith("bytes"):
                if base_value is not None:
                    values.append(base_value)
                else:
                    values.append(bytearray())
                continue
            if normalized_type == "string":
                values.append(base_value if isinstance(base_value, str) else "")
                continue
            values.append(base_value if base_value is not None else 0)
        return [function_selector] + values

    def _boundary_numeric_value(self, arg_type: str, ordinal: int) -> int:
        normalized = _text(arg_type).strip().lower()
        if normalized.startswith("uint"):
            bit_text = normalized.replace("uint", "") or "256"
            try:
                bits = max(8, int(bit_text))
            except Exception:
                bits = 256
            max_value = (2 ** bits) - 1
            values = [0, 1, max_value]
            return values[ordinal % len(values)]
        if normalized.startswith("int"):
            bit_text = normalized.replace("int", "") or "256"
            try:
                bits = max(8, int(bit_text))
            except Exception:
                bits = 256
            max_value = (2 ** (bits - 1)) - 1
            values = [0, 1, max_value]
            return values[ordinal % len(values)]
        return 1

    def _execution_stats_from_results(
        self,
        execution_items: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        attempted_candidate_count: int,
    ) -> Dict[str, Any]:
        success_count = 0
        revert_count = 0
        error_count = 0
        skipped_count = 0
        executed_count = 0
        useful_count = 0
        useful_by_reason = {
            "path_progress_delta": 0,
            "frontier_touched": 0,
            "sink_touched_without_revert": 0,
            "revert_reduced": 0,
            "target_related_coverage_delta": 0,
        }
        skip_reason_counts = {
            "deploy_failed": 0,
            "materialization_not_ready": 0,
            "missing_execution_context": 0,
        }
        not_useful_reasons = {
            "target_seen_without_path_progress": 0,
            "reverted": 0,
            "no_frontier_or_sink_touch": 0,
            "materialization_failed": 0,
            "execution_error": 0,
        }

        for item in execution_items:
            status = _text(item.get("execution_status")).strip()
            if bool(item.get("executed")):
                executed_count += 1
            if status == "success":
                success_count += 1
            elif status == "revert":
                revert_count += 1
            elif status == "error":
                error_count += 1
            if bool(item.get("skipped")):
                skipped_count += 1
                skip_reason = _text(item.get("skip_reason")).strip()
                if skip_reason in skip_reason_counts:
                    skip_reason_counts[skip_reason] += 1

            reason = _text(item.get("useful_reason")).strip() or "no_frontier_or_sink_touch"
            if bool(item.get("useful")):
                useful_count += 1
                if reason in useful_by_reason:
                    useful_by_reason[reason] += 1
            else:
                if reason in not_useful_reasons:
                    not_useful_reasons[reason] += 1
                elif status == "revert":
                    not_useful_reasons["reverted"] += 1
                elif status == "error":
                    not_useful_reasons["execution_error"] += 1

        for candidate in candidates:
            materialization_status = _text(candidate.get("materialization_status")).strip()
            if materialization_status == "failed":
                not_useful_reasons["materialization_failed"] += 1

        return {
            "attempted_candidate_count": int(attempted_candidate_count),
            "executed_candidate_count": int(executed_count),
            "success_count": int(success_count),
            "revert_count": int(revert_count),
            "error_count": int(error_count),
            "skipped_count": int(skipped_count),
            "skip_reason_counts": skip_reason_counts,
            "useful_candidate_count": int(useful_count),
            "useful_by_reason": useful_by_reason,
            "not_useful_reasons": not_useful_reasons,
        }

    def _build_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        candidates = [item for item in _listify(results.get("candidates")) if isinstance(item, dict)]
        execution_items = [item for item in _listify(results.get("results")) if isinstance(item, dict)]
        ready_count = 0
        partial_count = 0
        failed_count = 0
        binding_count = 0
        scoped_count = 0
        for candidate in candidates:
            status = _text(candidate.get("materialization_status")).strip()
            if status == "ready":
                ready_count += 1
            elif status == "partial":
                partial_count += 1
            elif status == "failed":
                failed_count += 1
            if _text(candidate.get("candidate_type")).strip() == "binding_pilot_candidate":
                binding_count += 1
            elif _text(candidate.get("candidate_type")).strip() == "scoped_selection_pilot_candidate":
                scoped_count += 1

        stats = self._execution_stats_from_results(
            execution_items,
            candidates,
            int(results.get("attempted_candidate_count") or len(execution_items)),
        )
        return {
            "enabled": True,
            "source": "target_focus_candidate_pilot_summary_v1",
            "pilot_mode": _text(results.get("pilot_mode")).strip() or "candidate_build_only",
            "executed": bool(results.get("executed", False)),
            "task_count": int(results.get("task_count") or 0),
            "candidate_count": int(len(candidates)),
            "ready_candidate_count": int(ready_count),
            "partial_candidate_count": int(partial_count),
            "failed_candidate_count": int(failed_count),
            "binding_candidate_count": int(binding_count),
            "scoped_selection_candidate_count": int(scoped_count),
            "attempted_candidate_count": int(stats.get("attempted_candidate_count") or 0),
            "executed_candidate_count": int(stats.get("executed_candidate_count") or 0),
            "success_count": int(stats.get("success_count") or 0),
            "revert_count": int(stats.get("revert_count") or 0),
            "error_count": int(stats.get("error_count") or 0),
            "skipped_count": int(stats.get("skipped_count") or 0),
            "skip_reason_counts": stats.get("skip_reason_counts") or {},
            "useful_candidate_count": int(stats.get("useful_candidate_count") or 0),
            "useful_by_reason": stats.get("useful_by_reason") or {},
            "not_useful_reasons": stats.get("not_useful_reasons") or {},
            "review_queue_ignored": True,
            "rejected_queue_ignored": True,
            "no_population_injection": True,
            "no_main_search_modification": True,
            "llm_called": False,
            "rag_used": False,
            "loaded_inputs": results.get("loaded_inputs") if isinstance(results.get("loaded_inputs"), dict) else {},
            "missing_inputs": _clean_list(results.get("missing_inputs")),
            "execution_context": results.get("execution_context") if isinstance(results.get("execution_context"), dict) else self._build_execution_context(),
            "deployment_context_check": self._build_deployment_context_check(
                execution_context=results.get("execution_context") if isinstance(results.get("execution_context"), dict) else self._build_execution_context(),
                deploy_attempted=bool(results.get("deployment_context_check", {}).get("deploy_attempted")) if isinstance(results.get("deployment_context_check"), dict) else False,
                deploy_success=bool(results.get("deployment_context_check", {}).get("deploy_success")) if isinstance(results.get("deployment_context_check"), dict) else False,
                deploy_error=_text(results.get("deployment_context_check", {}).get("deploy_error")).strip() or None if isinstance(results.get("deployment_context_check"), dict) else None,
            ),
            "generality_audit": self._build_generality_audit(results),
            "output_path": os.path.join(self.output_dir, "target_focus_candidate_pilot_results.json"),
        }

    def _build_execution_context(self) -> Dict[str, Any]:
        gas_limit, gas_limit_source = self._resolve_effective_gas_limit()
        return {
            "gas_limit": int(gas_limit),
            "gas_limit_source": gas_limit_source,
            "deployment_entry": "target_focus_candidate_pilot_lightweight",
            "repair_pilot_entry_reused": False,
            "deployment_parameter_parity_checked": True,
        }

    def _resolve_effective_gas_limit(self) -> Tuple[int, str]:
        if self.gas_limit is not None and self.gas_limit > 0:
            return int(self.gas_limit), "cli"
        try:
            settings_value = int(getattr(settings, "GAS_LIMIT", 0))
        except Exception:
            settings_value = 0
        if settings_value > 0:
            return settings_value, "settings"
        return 0, "unknown"

    def _load_known_repair_pilot_gas_limit(self) -> Optional[int]:
        script_path = os.path.join(self._repo_root, "run_guided_sequence_pilot_2x5.py")
        if not os.path.exists(script_path):
            return None
        try:
            with open(script_path, "r", encoding="utf-8", errors="ignore") as handle:
                source_text = handle.read()
        except Exception:
            return None
        match = re.search(r'--gas-limit"\s*,\s*"(\d+)"', source_text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _build_deployment_context_check(
        self,
        execution_context: Dict[str, Any],
        deploy_attempted: bool,
        deploy_success: bool,
        deploy_error: Optional[str],
    ) -> Dict[str, Any]:
        known_repair_pilot_gas_limit = self._load_known_repair_pilot_gas_limit()
        gas_limit = execution_context.get("gas_limit")
        if known_repair_pilot_gas_limit is None or not isinstance(gas_limit, int) or gas_limit <= 0:
            parity = None
        else:
            parity = bool(gas_limit == known_repair_pilot_gas_limit)
        return {
            "deploy_attempted": bool(deploy_attempted),
            "deploy_success": bool(deploy_success),
            "deploy_error": _text(deploy_error).strip() or None,
            "gas_limit": int(gas_limit) if isinstance(gas_limit, int) else 0,
            "gas_limit_source": _text(execution_context.get("gas_limit_source")).strip() or "unknown",
            "known_repair_pilot_gas_limit": known_repair_pilot_gas_limit,
            "gas_limit_matches_repair_pilot": parity,
            "recommend_reuse_repair_pilot_entry": not bool(execution_context.get("repair_pilot_entry_reused")),
        }

    def _skip_reason_from_env_error(self, env_error: Optional[str]) -> str:
        error_text = _text(env_error).strip().lower()
        if error_text.startswith("deploy_failed"):
            return "deploy_failed"
        return "missing_execution_context"

    def _build_generality_audit(self, results: Dict[str, Any]) -> Dict[str, Any]:
        contract_terms: List[str] = []
        function_specific_examples: List[str] = []
        scan_items = _listify(results.get("candidates")) + _listify(results.get("results"))
        for item in scan_items:
            if not isinstance(item, dict):
                continue
            for field in (
                "reason",
                "candidate_source_used",
                "materialization_status",
                "useful_reason",
            ):
                text = _text(item.get(field))
                if not text:
                    continue
                for term in RESTRICTED_TERMS:
                    if term in text and term not in contract_terms:
                        contract_terms.append(term)
                lowered = text.lower()
                if any(token in lowered for token in ("approve_transfer", "preserve_approve_transfer", "transferfrom")):
                    function_specific_examples.append(text)
            binding_choices = item.get("binding_choices") if isinstance(item.get("binding_choices"), dict) else {}
            for value in binding_choices.values():
                text = _text(value)
                for term in RESTRICTED_TERMS:
                    if term in text and term not in contract_terms:
                        contract_terms.append(term)
                lowered = text.lower()
                if any(token in lowered for token in ("approve_transfer", "preserve_approve_transfer", "transferfrom")):
                    function_specific_examples.append(text)

        function_specific_examples = _unique_preserve([item for item in function_specific_examples if item])[:10]
        return {
            "contract_specific_rule_count": int(len(contract_terms)),
            "contract_specific_terms_found": contract_terms,
            "function_specific_strategy_count": int(len(function_specific_examples)),
            "function_specific_strategy_examples": function_specific_examples,
            "generic_strategy_names_only": not bool(contract_terms or function_specific_examples),
            "uses_task_type_dispatch_only": True,
            "uses_scope_fields_only": True,
        }

