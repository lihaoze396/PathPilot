# -*- coding: utf-8 -*-

import copy
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from engine.plugin_interfaces import OnTheFlyAnalysis
except ImportError:
    from fuzzer.engine.plugin_interfaces import OnTheFlyAnalysis

try:
    from fuzzer.utils import settings
except ImportError:
    from utils import settings

try:
    from fuzzer.guidance.guided_sequence_concretizer import RepairSequenceConcretizer
except ImportError:
    from fuzzer.guidance.guided_sequence_concretizer import RepairSequenceConcretizer


STATUS_PRIORITY = {
    "mappable": 0,
    "partial": 1,
}

ADDRESS_POLICIES = {
    "any_existing_sender",
    "keep_same_sender_across_sequence",
    "rotate_existing_senders",
    "try_owner_or_deployer",
    "try_owner_or_deployer_or_previous_success_sender",
}


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
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
    except Exception:
        pass


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _normalize_path(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
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


class RepairSequencePilot:
    def __init__(
        self,
        selected_guided_sequences_path: str,
        max_injected: int = 5,
        dry_run: bool = False,
        output_dir: Optional[str] = None,
        concretization_version: str = "v1",
        insertion_policy: str = "append",
    ):
        self.selected_guided_sequences_path = (
            os.path.abspath(selected_guided_sequences_path) if selected_guided_sequences_path else ""
        )
        inferred_output_dir = os.path.dirname(self.selected_guided_sequences_path) if self.selected_guided_sequences_path else ""
        self.output_dir = os.path.abspath(output_dir or inferred_output_dir) if (output_dir or inferred_output_dir) else ""
        self.max_injected = max(0, int(max_injected or 0))
        self.dry_run = bool(dry_run)
        self.concretization_version = _text(concretization_version).strip().lower() or "v1"
        self.insertion_policy = self._normalize_insertion_policy(insertion_policy)
        self.enabled = bool(self.output_dir)

        self.log_path = os.path.join(self.output_dir, "guided_sequence_pilot.jsonl") if self.output_dir else ""
        self.summary_path = (
            os.path.join(self.output_dir, "guided_sequence_pilot_summary.json") if self.output_dir else ""
        )
        self.results_path = (
            os.path.join(self.output_dir, "guided_sequence_pilot_results.json") if self.output_dir else ""
        )
        self.concretization_debug_path = (
            os.path.join(self.output_dir, "repair_concretization_debug.jsonl") if self.output_dir else ""
        )

        self.loaded_selected = False
        self._selected_cache: Optional[List[Dict[str, Any]]] = None
        self._registered_once = False
        self._last_generation = 0

        self.selected_count = 0
        self.attempted_count = 0
        self.constructed_count = 0
        self.injected_count = 0
        self.skipped_count = 0

        self._skip_reason_counts: Counter = Counter()
        self.notes: List[str] = []
        self.records: List[Dict[str, Any]] = []
        self._pending_records: List[Dict[str, Any]] = []
        self.injected_mappings: List[Dict[str, Any]] = []
        self.summary_payload: Dict[str, Any] = {}
        self.results_payload: Dict[str, Any] = {}

        if self.enabled:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(self.log_path, "w", encoding="utf-8"):
                    pass
                if self.concretization_debug_path:
                    with open(self.concretization_debug_path, "w", encoding="utf-8"):
                        pass
            except Exception:
                self.enabled = False

    @staticmethod
    def _normalize_insertion_policy(value: Any) -> str:
        policy = _text(value).strip().lower().replace("-", "_")
        aliases = {
            "tail": "append",
            "back": "append",
            "normal": "append",
            "normal_order": "append",
            "front": "front",
            "head": "front",
            "priority": "front",
        }
        return aliases.get(policy, policy if policy in ("append", "front") else "append")

    def load_selected(self) -> List[Dict[str, Any]]:
        if self._selected_cache is not None:
            return list(self._selected_cache)

        payload = _read_json(self.selected_guided_sequences_path)
        selected = payload.get("selected")
        if not isinstance(selected, list):
            selected = []

        normalized: List[Dict[str, Any]] = []
        for item in selected:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "candidate_id": _text(item.get("candidate_id")).strip(),
                    "selected_rank": _safe_int(item.get("selected_rank"), 10 ** 9),
                    "selection_score": float(item.get("selection_score") or 0.0),
                    "hypothesis_id": _text(item.get("hypothesis_id")).strip() or None,
                    "diagnosis_type": _text(item.get("diagnosis_type")).strip() or None,
                    "sequence_strategy": (
                        _text(item.get("sequence_strategy")).strip()
                        or _text(item.get("raw", {}).get("sequence_strategy")).strip()
                        or None
                    ),
                    "target_function": _text(item.get("target_function")).strip() or None,
                    "guided_sequence": _normalize_path(item.get("guided_sequence")),
                    "sender_policy": _text(item.get("sender_policy")).strip() or "any_existing_sender",
                    "value_policy": _text(item.get("value_policy")).strip() or "keep_original",
                    "argument_policy": _text(item.get("argument_policy")).strip() or "unknown",
                    "prestate_policy": _text(item.get("prestate_policy")).strip() or "none",
                    "dryrun_status": _text(item.get("dryrun_status")).strip().lower() or "unknown",
                    "runtime_pattern_family": (
                        _text(item.get("runtime_pattern_family")).strip()
                        or _text(item.get("target_runtime_pattern_family")).strip()
                        or _text(item.get("raw", {}).get("runtime_pattern_family")).strip()
                        or _text(item.get("raw", {}).get("target_runtime_pattern_family")).strip()
                        or None
                    ),
                    "target_runtime_pattern_family": (
                        _text(item.get("target_runtime_pattern_family")).strip()
                        or _text(item.get("runtime_pattern_family")).strip()
                        or _text(item.get("raw", {}).get("target_runtime_pattern_family")).strip()
                        or _text(item.get("raw", {}).get("runtime_pattern_family")).strip()
                        or None
                    ),
                    "selection_reason": list(item.get("selection_reason") or []),
                    "access_control_address_pool": list(item.get("access_control_address_pool") or []),
                    "access_control_binding_hints": dict(item.get("access_control_binding_hints") or {}),
                    "guidance_profile": _text(item.get("guidance_profile")).strip() or None,
                    "raw": dict(item),
                }
            )

        self._selected_cache = normalized
        self.loaded_selected = bool(normalized)
        self.selected_count = len(normalized)
        if not self.selected_guided_sequences_path:
            self.notes.append("selected_guided_sequences_path_missing")
        elif not os.path.exists(self.selected_guided_sequences_path):
            self.notes.append("selected_guided_sequences_file_missing")
        elif not normalized:
            self.notes.append("selected_guided_sequences_empty")
        return list(normalized)

    def build_repair_individuals(
        self,
        generator: Any,
        population: Any,
        contract_name: Optional[str] = None,
    ) -> List[Any]:
        selected = self.load_selected()
        candidates = self._choose_candidates(selected)

        self.attempted_count = 0
        self.constructed_count = 0
        self.injected_count = 0
        self.skipped_count = 0
        self._skip_reason_counts = Counter()
        self.records = []
        self._pending_records = []
        self.injected_mappings = []

        individuals: List[Any] = []
        existing_hashes = set()
        try:
            existing_hashes = {str(getattr(indv, "hash", "")) for indv in list(population.individuals)}
        except Exception:
            existing_hashes = set()
        built_hashes = set()

        for candidate in candidates:
            self.attempted_count += 1
            record = {
                "generation": int(self._last_generation),
                "candidate_id": candidate.get("candidate_id"),
                "hypothesis_id": candidate.get("hypothesis_id"),
                "guided_sequence": list(candidate.get("guided_sequence") or []),
                "sequence_strategy": candidate.get("sequence_strategy"),
                "runtime_pattern_family": candidate.get("runtime_pattern_family"),
                "target_runtime_pattern_family": candidate.get("target_runtime_pattern_family"),
                "adapter_status": candidate.get("dryrun_status") or "unknown",
                "repair_concretization_version": self.concretization_version,
                "concretization_status": None,
                "sender_binding_reason": None,
                "value_binding_reason": None,
                "argument_binding_reason": None,
                "prestate_binding_reason": None,
                "preserve_prefix": False,
                "concretization_notes": [],
                "constructed": False,
                "injected": False,
                "individual_hash": None,
                "sequence_id": None,
                "skip_reason": None,
            }

            if not candidate.get("candidate_id"):
                record["skip_reason"] = "missing_candidate_id"
                self._finalize_record(record)
                continue
            if not candidate.get("guided_sequence"):
                record["skip_reason"] = "empty_guided_sequence"
                self._finalize_record(record)
                continue

            individual = None
            concretization_metadata: Dict[str, Any] = {}
            try:
                individual, concretization_metadata = self._candidate_to_individual(
                    candidate=candidate,
                    generator=generator,
                    population=population,
                    contract_name=contract_name,
                )
            except Exception:
                individual = None
                concretization_metadata = {}

            for key in (
                "concretization_status",
                "sender_binding_reason",
                "value_binding_reason",
                "argument_binding_reason",
                "prestate_binding_reason",
                "preserve_prefix",
                "concretization_notes",
            ):
                if key in concretization_metadata:
                    record[key] = concretization_metadata.get(key)

            if individual is None:
                record["skip_reason"] = _text(concretization_metadata.get("skip_reason")).strip() or "construction_failed"
                self._finalize_record(record)
                continue

            record["constructed"] = True
            self.constructed_count += 1
            individual_hash = _text(getattr(individual, "hash", "")).strip()
            record["individual_hash"] = individual_hash or None
            record["sequence_id"] = _text(getattr(individual, "sequence_id", "")).strip() or None
            self._write_concretization_debug(record, concretization_metadata)

            if individual_hash and (individual_hash in existing_hashes or individual_hash in built_hashes):
                record["skip_reason"] = "duplicate_hash_in_population"
                self._finalize_record(record)
                continue

            built_hashes.add(individual_hash)
            record["_individual"] = individual
            individuals.append(individual)
            self._pending_records.append(record)

        return individuals

    def register_injection(
        self,
        population: Any,
        generator: Any,
        generation: int,
    ) -> Dict[str, Any]:
        if self._registered_once:
            return dict(self.summary_payload)

        self._registered_once = True
        self._last_generation = max(int(generation or 0), 0)

        if not self.enabled:
            self.summary_payload = self._build_summary()
            self._write_summary()
            self._write_results([])
            return dict(self.summary_payload)

        individuals = self.build_repair_individuals(
            generator=generator,
            population=population,
            contract_name=getattr(generator, "contract_name", None),
        )

        injection_failed = False
        if individuals and not self.dry_run:
            try:
                if self.insertion_policy == "front":
                    population.individuals[0:0] = individuals
                else:
                    population.individuals.extend(individuals)
                if hasattr(population, "update_flag"):
                    population.update_flag()
            except Exception:
                injection_failed = True

        for record in list(self._pending_records):
            individual = record.pop("_individual", None)
            if individual is None:
                self._finalize_record(record)
                continue

            if self.dry_run:
                record["injected"] = False
            elif injection_failed:
                record["skip_reason"] = "population_injection_failed"
            else:
                record["injected"] = True
                record["insertion_policy"] = self.insertion_policy
                record["bypasses_normal_population_order"] = self.insertion_policy == "front"
                self.injected_count += 1
                self.injected_mappings.append(
                    {
                        "candidate_id": record.get("candidate_id"),
                        "hypothesis_id": record.get("hypothesis_id"),
                        "guided_sequence": list(record.get("guided_sequence") or []),
                        "sequence_strategy": record.get("sequence_strategy"),
                        "sequence_id": record.get("sequence_id"),
                        "individual_hash": record.get("individual_hash"),
                        "insertion_policy": self.insertion_policy,
                        "bypasses_normal_population_order": self.insertion_policy == "front",
                    }
                )
            self._finalize_record(record)
        self._pending_records = []

        self.summary_payload = self._build_summary()
        self._write_summary()
        self._write_results([])
        return dict(self.summary_payload)

    def write_log_record(self, record: Dict[str, Any]) -> None:
        if not self.enabled or not self.log_path:
            return
        try:
            payload = dict(record)
            payload.pop("_individual", None)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def finalize(self) -> Dict[str, Any]:
        if not self.enabled:
            self.summary_payload = self._build_summary()
            self._write_summary()
            self._write_results([])
            return dict(self.summary_payload)

        results = self._build_pilot_results()
        self.results_payload = {
            "enabled": True,
            "dry_run": bool(self.dry_run),
            "repair_pilot_concretization": self.concretization_version,
            "selected_count": int(self.selected_count),
            "attempted_count": int(self.attempted_count),
            "constructed_count": int(self.constructed_count),
            "injected_count": int(self.injected_count),
            "results": results,
            "source": "guided_sequence_execution_pilot_v1",
        }
        self._write_results(results)
        self.summary_payload = self._build_summary(results=results)
        self._write_summary()
        return dict(self.summary_payload)

    def get_summary(self) -> Dict[str, Any]:
        if not self.summary_payload:
            self.summary_payload = self._build_summary()
        return dict(self.summary_payload)

    def _choose_candidates(self, selected: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        eligible: List[Dict[str, Any]] = []
        for item in selected:
            if not isinstance(item, dict):
                continue
            if not item.get("candidate_id"):
                continue
            if not item.get("guided_sequence"):
                continue
            status = _text(item.get("dryrun_status")).strip().lower()
            if status not in ("mappable", "partial"):
                continue
            eligible.append(dict(item))

        eligible.sort(
            key=lambda item: (
                STATUS_PRIORITY.get(_text(item.get("dryrun_status")).lower(), 99),
                _safe_int(item.get("selected_rank"), 10 ** 9),
                -float(item.get("selection_score") or 0.0),
            )
        )
        if self.max_injected <= 0:
            return []
        return eligible[: self.max_injected]

    def _candidate_to_individual(
        self,
        candidate: Dict[str, Any],
        generator: Any,
        population: Any,
        contract_name: Optional[str] = None,
    ):
        chromosome, concretization_metadata = self._build_chromosome(candidate, generator)
        if not chromosome:
            return None, dict(concretization_metadata or {})

        indv_type = population.indv_template.__class__
        individual = indv_type(
            generator=population.indv_template.generator,
            other_generators=getattr(population.indv_template, "other_generators", []),
        ).init(chromosome=chromosome)

        sequence_id = "REPAIR_PILOT:{}:G{}".format(candidate.get("candidate_id"), int(self._last_generation))
        metadata = {
            "candidate_id": candidate.get("candidate_id"),
            "hypothesis_id": candidate.get("hypothesis_id"),
            "target_function": candidate.get("target_function"),
            "guided_sequence": list(candidate.get("guided_sequence") or []),
            "sequence_strategy": candidate.get("sequence_strategy"),
            "dryrun_status": candidate.get("dryrun_status"),
            "selected_rank": candidate.get("selected_rank"),
            "selection_score": candidate.get("selection_score"),
            "contract_name": contract_name or getattr(generator, "contract_name", None),
            "candidate_origin": candidate.get("candidate_origin"),
            "candidate_source_used": candidate.get("candidate_source_used"),
            "candidate_source": candidate.get("candidate_source"),
            "source": candidate.get("source"),
            "modules_used": list(candidate.get("modules_used") or []),
            "module_source": candidate.get("module_source"),
            "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
            "prefix_replay": bool(candidate.get("prefix_replay")),
            "suffix_mutation_type": candidate.get("suffix_mutation_type"),
            "materialization_status": candidate.get("materialization_status"),
            "materialization_status_final": candidate.get("materialization_status_final") or candidate.get("materialization_status"),
            "has_concrete_prefix": bool(candidate.get("has_concrete_prefix")),
            "has_concrete_suffix": bool(candidate.get("has_concrete_suffix")),
            "sequence_materialized": bool(candidate.get("sequence_materialized")),
            "effect_probe_appended": bool(candidate.get("effect_probe_appended") or candidate.get("probe_appended")),
            "parent_candidate_id": candidate.get("parent_candidate_id"),
            "source_trigger_id": candidate.get("source_trigger_id"),
            "source_parent_id": candidate.get("source_parent_id") or candidate.get("parent_candidate_id"),
            "source_scope_id": candidate.get("source_scope_id") or candidate.get("scope_id") or candidate.get("source_task_id"),
            "source_candidate_kind": candidate.get("source_candidate_kind") or candidate.get("candidate_type"),
            "bench_id": candidate.get("bench_id"),
            "do_not_bypass_normal_selection": bool(candidate.get("do_not_bypass_normal_selection")),
            "insertion_policy": self.insertion_policy,
            "bypasses_normal_population_order": self.insertion_policy == "front",
        }
        setattr(individual, "is_guided_sequence_pilot", True)
        setattr(individual, "repair_candidate_id", candidate.get("candidate_id"))
        setattr(individual, "guided_sequence", list(candidate.get("guided_sequence") or []))
        setattr(individual, "repair_strategy", candidate.get("sequence_strategy"))
        setattr(individual, "sequence_strategy", candidate.get("sequence_strategy"))
        setattr(individual, "repair_hypothesis_id", candidate.get("hypothesis_id"))
        setattr(individual, "repair_target_function", candidate.get("target_function"))
        setattr(individual, "repair_adapter_status", candidate.get("dryrun_status"))
        setattr(individual, "repair_concretization_version", self.concretization_version)
        setattr(individual, "concretization_status", concretization_metadata.get("concretization_status"))
        setattr(individual, "sender_binding_reason", concretization_metadata.get("sender_binding_reason"))
        setattr(individual, "value_binding_reason", concretization_metadata.get("value_binding_reason"))
        setattr(individual, "argument_binding_reason", concretization_metadata.get("argument_binding_reason"))
        setattr(individual, "prestate_binding_reason", concretization_metadata.get("prestate_binding_reason"))
        setattr(individual, "preserve_prefix", bool(concretization_metadata.get("preserve_prefix")))
        setattr(individual, "concretization_notes", list(concretization_metadata.get("concretization_notes") or []))
        setattr(individual, "repair_pilot_metadata", metadata)
        setattr(individual, "sequence_id", sequence_id)
        setattr(individual, "individual_id", sequence_id)
        setattr(individual, "generation_id", sequence_id)
        return individual, dict(concretization_metadata or {})

    def _write_concretization_debug(
        self,
        record: Dict[str, Any],
        concretization_metadata: Dict[str, Any],
    ) -> None:
        if not self.concretization_debug_path:
            return
        payload = {
            "generation": int(record.get("generation") or 0),
            "candidate_id": record.get("candidate_id"),
            "hypothesis_id": record.get("hypothesis_id"),
            "guided_sequence": list(record.get("guided_sequence") or []),
            "sequence_strategy": record.get("sequence_strategy"),
            "runtime_pattern_family": record.get("runtime_pattern_family"),
            "target_runtime_pattern_family": record.get("target_runtime_pattern_family"),
            "sequence_id": record.get("sequence_id"),
            "individual_hash": record.get("individual_hash"),
            "concretization_status": concretization_metadata.get("concretization_status"),
            "skip_reason": concretization_metadata.get("skip_reason"),
            "concretization_notes": list(concretization_metadata.get("concretization_notes") or []),
            "binding_plan": copy.deepcopy(concretization_metadata.get("binding_plan") or {}),
            "gene_debug": copy.deepcopy(concretization_metadata.get("gene_debug") or []),
        }
        try:
            with open(self.concretization_debug_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def _build_chromosome(self, candidate: Dict[str, Any], main_generator: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.concretization_version == "v2":
            concretizer = RepairSequenceConcretizer(
                generator=main_generator,
                contract_name=getattr(main_generator, "contract_name", None),
                strategy_version="v2",
            )
            chromosome, metadata = concretizer.build_chromosome(candidate)
            return list(chromosome or []), dict(metadata or {})

        guided_sequence = _normalize_path(candidate.get("guided_sequence"))
        if not guided_sequence:
            return [], {"concretization_status": "failed", "skip_reason": "empty_guided_sequence", "concretization_notes": []}

        generators = [main_generator] + list(getattr(main_generator, "other_generators", []) or [])
        account_pool = list(getattr(main_generator, "accounts", []) or [])
        if not account_pool:
            return [], {"concretization_status": "failed", "skip_reason": "account_pool_empty", "concretization_notes": []}

        sender_policy = _text(candidate.get("sender_policy")).strip() or "any_existing_sender"
        value_policy = _text(candidate.get("value_policy")).strip() or "keep_original"
        argument_policy = _text(candidate.get("argument_policy")).strip() or "unknown"

        same_sender = self._pick_owner_or_default_sender(account_pool)
        chromosome: List[Dict[str, Any]] = []

        for gene_index, function_name in enumerate(guided_sequence):
            resolved = self._resolve_function(generators, function_name)
            if resolved is None:
                return [], {"concretization_status": "failed", "skip_reason": "function_not_resolved", "concretization_notes": []}
            target_generator, function_selector, argument_types, state_mutability = resolved
            temp_generator = self._clone_generator(target_generator)
            generated = temp_generator.generate_individual(function_selector, argument_types)
            if not generated:
                return [], {"concretization_status": "failed", "skip_reason": "generator_failed", "concretization_notes": []}
            gene = copy.deepcopy(generated[0])
            sender = self._pick_sender_for_gene(
                sender_policy=sender_policy,
                accounts=account_pool,
                gene_index=gene_index,
                same_sender=same_sender,
            )
            if not sender:
                return [], {"concretization_status": "failed", "skip_reason": "sender_unavailable", "concretization_notes": []}

            gene["account"] = sender
            gene["contract"] = getattr(target_generator, "contract", gene.get("contract"))
            gene["amount"] = self._pick_value_for_gene(
                value_policy=value_policy,
                gene=gene,
                gene_index=gene_index,
                function_name=function_name,
                state_mutability=state_mutability,
            )

            arguments = self._build_arguments(
                function_name=function_name,
                function_selector=function_selector,
                argument_types=argument_types,
                generated_gene=gene,
                account_pool=account_pool,
                argument_policy=argument_policy,
                sender=sender,
                gene_index=gene_index,
            )
            if arguments is None:
                return [], {"concretization_status": "failed", "skip_reason": "argument_construction_failed", "concretization_notes": []}
            gene["arguments"] = arguments
            gene.setdefault("blocknumber", None)
            gene.setdefault("timestamp", None)
            gene.setdefault("gaslimit", settings.GAS_LIMIT)
            gene.setdefault("call_return", {})
            gene.setdefault("extcodesize", {})
            gene.setdefault("returndatasize", {})
            chromosome.append(gene)

        return chromosome, {
            "concretization_status": "mappable",
            "skip_reason": None,
            "sender_binding_reason": "v1 sender policy mapping",
            "value_binding_reason": "v1 value policy mapping",
            "argument_binding_reason": "v1 argument mutation policy",
            "prestate_binding_reason": None,
            "preserve_prefix": False,
            "concretization_notes": [],
        }

    def _resolve_function(self, generators: Sequence[Any], function_name: str):
        normalized_name = _text(function_name).strip()
        if not normalized_name:
            return None

        for generator in generators:
            if normalized_name in ("fallback", "receive") and "fallback" in getattr(generator, "interface", {}):
                state_mutability = self._resolve_state_mutability(
                    generator,
                    selector="fallback",
                    function_name=normalized_name,
                )
                return generator, "fallback", list(getattr(generator, "interface", {}).get("fallback") or []), state_mutability

            interface_mapper = getattr(generator, "interface_mapper", {}) or {}
            matches: List[Tuple[str, str]] = []
            for signature, selector in interface_mapper.items():
                if _text(signature).split("(", 1)[0] == normalized_name:
                    matches.append((_text(signature), _text(selector)))
            if not matches:
                continue
            matches.sort(key=lambda item: item[0])
            signature, selector = matches[0]
            interface = getattr(generator, "interface", {}) or {}
            argument_types = list(interface.get(selector) or [])
            state_mutability = self._resolve_state_mutability(
                generator,
                selector=selector,
                signature=signature,
                function_name=normalized_name,
            )
            return generator, selector, argument_types, state_mutability
        return None

    def _clone_generator(self, generator: Any):
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

    @staticmethod
    def _pick_owner_or_default_sender(accounts: Sequence[str]) -> Optional[str]:
        if not accounts:
            return None
        return _text(accounts[0]).strip() or None

    def _pick_sender_for_gene(
        self,
        sender_policy: str,
        accounts: Sequence[str],
        gene_index: int,
        same_sender: Optional[str],
    ) -> Optional[str]:
        normalized = _text(sender_policy).strip() or "any_existing_sender"
        if not accounts:
            return None
        if normalized == "rotate_existing_senders":
            return _text(accounts[gene_index % len(accounts)]).strip() or None
        if normalized == "keep_same_sender_across_sequence":
            return same_sender
        if normalized in (
            "try_owner_or_deployer",
            "try_owner_or_deployer_or_previous_success_sender",
            "any_existing_sender",
        ):
            return same_sender
        return same_sender

    @staticmethod
    def _pick_value_for_gene(
        value_policy: str,
        gene: Dict[str, Any],
        gene_index: int,
        function_name: str,
        state_mutability: Optional[str] = None,
    ) -> int:
        normalized = _text(value_policy).strip() or "keep_original"
        normalized_mutability = _text(state_mutability).strip().lower() or "nonpayable"
        accepts_value = _text(function_name).strip() in ("fallback", "receive") or normalized_mutability == "payable"
        if normalized == "force_zero_value":
            return 0
        if not accepts_value:
            value = gene.get("amount")
            if isinstance(value, int) and value == 0:
                return 0
            return 0
        if normalized == "try_small_nonzero_value":
            return 1
        if normalized == "try_zero_and_small_nonzero":
            return 1 if gene_index == 0 else 0
        value = gene.get("amount")
        if isinstance(value, int):
            return value
        return 0

    def _build_arguments(
        self,
        function_name: str,
        function_selector: str,
        argument_types: Sequence[str],
        generated_gene: Dict[str, Any],
        account_pool: Sequence[str],
        argument_policy: str,
        sender: str,
        gene_index: int,
    ) -> Optional[List[Any]]:
        generated_arguments = list(generated_gene.get("arguments") or [])
        generated_values = list(generated_arguments[1:]) if generated_arguments else []
        policy = _text(argument_policy).strip().lower()
        normalized_name = _text(function_name).strip()

        if len(generated_values) < len(argument_types):
            generated_values.extend([None] * (len(argument_types) - len(generated_values)))

        if "transfer_arguments" in policy:
            overridden = self._build_transfer_arguments(
                function_name=normalized_name,
                argument_types=argument_types,
                account_pool=account_pool,
            )
            if overridden is not None:
                return [function_selector] + overridden

        values: List[Any] = []
        for arg_index, arg_type in enumerate(argument_types):
            base_value = generated_values[arg_index] if arg_index < len(generated_values) else None
            mutated = self._mutate_argument_value(
                arg_type=arg_type,
                base_value=base_value,
                argument_policy=policy,
                account_pool=account_pool,
                sender=sender,
                arg_index=arg_index,
                gene_index=gene_index,
            )
            values.append(mutated)
        return [function_selector] + values

    def _build_transfer_arguments(
        self,
        function_name: str,
        argument_types: Sequence[str],
        account_pool: Sequence[str],
    ) -> Optional[List[Any]]:
        if not account_pool:
            return None
        addr_a = account_pool[0]
        addr_b = account_pool[1] if len(account_pool) > 1 else account_pool[0]
        amount = 1
        if function_name == "approve" and len(argument_types) == 2:
            return [addr_b, amount]
        if function_name == "transfer" and len(argument_types) == 2:
            return [addr_b, amount]
        if function_name == "transferFrom" and len(argument_types) == 3:
            return [addr_a, addr_b, amount]
        return None

    def _mutate_argument_value(
        self,
        arg_type: str,
        base_value: Any,
        argument_policy: str,
        account_pool: Sequence[str],
        sender: str,
        arg_index: int,
        gene_index: int,
    ) -> Any:
        normalized_type = _text(arg_type).replace(" storage", "").replace(" memory", "").strip()
        lowered_policy = _text(argument_policy).strip().lower()

        if "[" in normalized_type and "]" in normalized_type:
            return base_value

        if normalized_type.startswith("address"):
            if account_pool:
                if arg_index == 0 and lowered_policy in ("mutate_permission_and_sink_args",):
                    return sender
                return account_pool[(arg_index + gene_index + 1) % len(account_pool)]
            return base_value

        if normalized_type.startswith("bool"):
            if "boolean" in lowered_policy:
                return bool((arg_index + gene_index) % 2 == 0)
            if "mutate" in lowered_policy:
                return True
            if isinstance(base_value, bool):
                return base_value
            return False

        if normalized_type.startswith("uint") or normalized_type.startswith("int"):
            if "boundary" in lowered_policy:
                return self._boundary_numeric_value(normalized_type, arg_index + gene_index)
            if "transfer_arguments" in lowered_policy:
                return 1
            if "mutate" in lowered_policy:
                return max(1, arg_index + 1)
            if isinstance(base_value, int):
                return base_value
            return 0

        if normalized_type.startswith("bytes") or normalized_type == "string":
            if isinstance(base_value, (bytes, bytearray, str)):
                return base_value
            return base_value if base_value is not None else ""

        if base_value is not None:
            return base_value
        return 0

    @staticmethod
    def _boundary_numeric_value(arg_type: str, ordinal: int) -> int:
        normalized_type = _text(arg_type).strip().lower()
        if normalized_type.startswith("uint"):
            bit_text = normalized_type.replace("uint", "") or "256"
            try:
                bits = max(8, int(bit_text))
            except Exception:
                bits = 256
            values = [0, 1, (2 ** bits) - 1]
            return values[ordinal % len(values)]
        if normalized_type.startswith("int"):
            bit_text = normalized_type.replace("int", "") or "256"
            try:
                bits = max(8, int(bit_text))
            except Exception:
                bits = 256
            max_value = (2 ** (bits - 1)) - 1
            values = [0, 1, max_value]
            return values[ordinal % len(values)]
        return 1

    def _finalize_record(self, record: Dict[str, Any]) -> None:
        skip_reason = _text(record.get("skip_reason")).strip() or None
        if skip_reason:
            self.skipped_count += 1
            self._skip_reason_counts[skip_reason] += 1
        payload = dict(record)
        payload.pop("_individual", None)
        self.records.append(payload)
        self.write_log_record(payload)

    def _build_pilot_results(self) -> List[Dict[str, Any]]:
        injected_by_sequence = {
            _text(item.get("sequence_id")).strip(): item
            for item in self.injected_mappings
            if _text(item.get("sequence_id")).strip()
        }

        path_progress_path = os.path.join(self.output_dir, "path_progress.jsonl") if self.output_dir else ""
        ksi_path = os.path.join(self.output_dir, "ksi_evidence.jsonl") if self.output_dir else ""
        path_records = _read_jsonl(path_progress_path)
        ksi_records = _read_jsonl(ksi_path)

        progress_by_sequence: Dict[str, List[Dict[str, Any]]] = {}
        for record in path_records:
            sequence_id = _text(record.get("sequence_id")).strip()
            if not sequence_id:
                continue
            progress_by_sequence.setdefault(sequence_id, []).append(record)

        ksi_by_sequence: Dict[str, List[Dict[str, Any]]] = {}
        for record in ksi_records:
            sequence_id = _text(record.get("sequence_id")).strip()
            if not sequence_id:
                continue
            ksi_by_sequence.setdefault(sequence_id, []).append(record)

        results: List[Dict[str, Any]] = []
        for sequence_id, mapping in injected_by_sequence.items():
            progress_items = progress_by_sequence.get(sequence_id, [])
            ksi_items = ksi_by_sequence.get(sequence_id, [])
            observed_functions = []
            seen_functions = set()
            for ksi_item in ksi_items:
                observations = ksi_item.get("observations") if isinstance(ksi_item, dict) else {}
                for function_name in observations.get("functions") or []:
                    function_text = _text(function_name).strip()
                    if function_text and function_text not in seen_functions:
                        seen_functions.add(function_text)
                        observed_functions.append(function_text)

            matched_hypotheses = []
            seen_hypotheses = set()
            for progress_item in progress_items:
                hypothesis_id = _text(progress_item.get("matched_hypothesis")).strip()
                if hypothesis_id and hypothesis_id not in seen_hypotheses:
                    seen_hypotheses.add(hypothesis_id)
                    matched_hypotheses.append(hypothesis_id)

            sink_function_reached = any(bool(item.get("sink_function_reached")) for item in progress_items)
            sink_reverted = any(_text(item.get("status")) == "sink_reverted" for item in progress_items)
            blocked = any(_text(item.get("status")) in ("blocked", "sink_reverted") for item in progress_items)
            revert = False
            coverage_delta = None
            for ksi_item in ksi_items:
                observations = ksi_item.get("observations") if isinstance(ksi_item, dict) else {}
                if observations.get("revert") is True:
                    revert = True
                observed_delta = observations.get("coverage_delta")
                if isinstance(observed_delta, int):
                    if coverage_delta is None or observed_delta > coverage_delta:
                        coverage_delta = observed_delta

            executed = bool(progress_items or ksi_items)
            results.append(
                {
                    "candidate_id": mapping.get("candidate_id"),
                    "individual_hash": mapping.get("individual_hash"),
                    "sequence_id": sequence_id,
                    "guided_sequence": list(mapping.get("guided_sequence") or []),
                    "executed": executed,
                    "matched_hypothesis": matched_hypotheses[0] if matched_hypotheses else None,
                    "matched_hypotheses": matched_hypotheses,
                    "path_progress_records": progress_items,
                    "sink_function_reached": sink_function_reached,
                    "sink_reverted": sink_reverted,
                    "blocked": blocked,
                    "revert": revert,
                    "coverage_delta": coverage_delta,
                    "observed_functions": observed_functions,
                    "status": "executed" if executed else "not_observed",
                    "source": "guided_sequence_execution_pilot_v1",
                }
            )
        results.sort(key=lambda item: _text(item.get("candidate_id")))
        return results

    def _build_summary(self, results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        total_path_records = _read_jsonl(os.path.join(self.output_dir, "path_progress.jsonl")) if self.output_dir else []
        total_ksi_records = _read_jsonl(os.path.join(self.output_dir, "ksi_evidence.jsonl")) if self.output_dir else []

        sink_function_reached_count = sum(1 for item in total_path_records if bool(item.get("sink_function_reached")))
        sink_reverted_count = sum(1 for item in total_path_records if _text(item.get("status")) == "sink_reverted")
        blocked_count = sum(
            1 for item in total_path_records if _text(item.get("status")) in ("blocked", "sink_reverted")
        )

        results = results if isinstance(results, list) else list(self.results_payload.get("results") or [])
        repair_pilot_executed_count = sum(1 for item in results if bool(item.get("executed")))
        repair_pilot_sink_function_reached = sum(1 for item in results if bool(item.get("sink_function_reached")))
        repair_pilot_sink_reverted = sum(1 for item in results if bool(item.get("sink_reverted")))
        concretization_partial_count = sum(
            1 for item in self.records if _text(item.get("concretization_status")).strip().lower() == "partial"
        )
        concretization_mappable_count = sum(
            1 for item in self.records if _text(item.get("concretization_status")).strip().lower() == "mappable"
        )

        summary = {
            "enabled": bool(self.enabled),
            "dry_run": bool(self.dry_run),
            "repair_pilot_concretization": self.concretization_version,
            "insertion_policy": self.insertion_policy,
            "bypasses_normal_population_order": self.insertion_policy == "front",
            "selected_path": self.selected_guided_sequences_path or None,
            "selected_count": int(self.selected_count),
            "attempted_count": int(self.attempted_count),
            "constructed_count": int(self.constructed_count),
            "injected_count": int(self.injected_count),
            "skipped_count": int(self.skipped_count),
            "path_progress_records": len(total_path_records),
            "sink_function_reached_count": int(sink_function_reached_count),
            "sink_reverted_count": int(sink_reverted_count),
            "blocked_count": int(blocked_count),
            "ksi_evidence_records": len(total_ksi_records),
            "repair_pilot_executed_count": int(repair_pilot_executed_count),
            "repair_pilot_sink_function_reached": int(repair_pilot_sink_function_reached),
            "repair_pilot_sink_reverted": int(repair_pilot_sink_reverted),
            "repair_pilot_concretization_partial_count": int(concretization_partial_count),
            "repair_pilot_concretization_mappable_count": int(concretization_mappable_count),
            "top_skip_reasons": {str(key): int(value) for key, value in self._skip_reason_counts.items()},
            "notes": list(dict.fromkeys(self.notes)),
            "output_path": self.log_path or None,
            "results_path": self.results_path or None,
            "summary_path": self.summary_path or None,
        }
        return summary

    def _write_summary(self) -> None:
        if self.summary_path:
            _write_json(self.summary_path, self.summary_payload)

    def _write_results(self, results: List[Dict[str, Any]]) -> None:
        if not self.results_path:
            return
        payload = dict(self.results_payload) if self.results_payload else {}
        payload.setdefault("enabled", bool(self.enabled))
        payload.setdefault("dry_run", bool(self.dry_run))
        payload["results"] = list(results)
        _write_json(self.results_path, payload)


class RepairSequencePilotAnalysis(OnTheFlyAnalysis):
    def __init__(self, pilot: RepairSequencePilot) -> None:
        self.pilot = pilot

    def setup(self, ng, engine):
        pass

    def register_step(self, g, population, engine):
        if self.pilot is None:
            return
        try:
            generation = 0 if int(g) < 0 else int(g)
            self.pilot.register_injection(
                population=population,
                generator=population.indv_generator,
                generation=generation,
            )
        except Exception:
            pass

    def finalize(self, population, engine):
        if self.pilot is None:
            return
        try:
            self.pilot.finalize()
        except Exception:
            pass

