# -*- coding: utf-8 -*-

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


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


SENDER_POLICY_TEMPLATES = {
    "any_existing_sender": "ANY_EXISTING_SENDER",
    "keep_same_sender_across_sequence": "SAME_SENDER_FOR_SEQUENCE",
    "rotate_existing_senders": "ROTATE_EXISTING_SENDERS",
    "try_owner_or_deployer": "OWNER_OR_DEPLOYER_CANDIDATE",
    "try_owner_or_deployer_or_previous_success_sender": "OWNER_DEPLOYER_OR_PREVIOUS_SUCCESS",
    "keep_original_sender": "KEEP_ORIGINAL_SENDER",
    "keep_prefix_sender_stable": "KEEP_PREFIX_SENDER_STABLE",
}

VALUE_POLICY_TEMPLATES = {
    "keep_original": "KEEP_ORIGINAL",
    "force_zero_value": "ZERO_VALUE",
    "try_small_nonzero_value": "SMALL_NONZERO_VALUE",
    "try_zero_and_small_nonzero": "ZERO_OR_SMALL_NONZERO",
}


class RepairSequenceExecutionAdapterDryRun:
    def __init__(
        self,
        output_dir: str,
        blueprint_path: Optional[str] = None,
        selected_guided_sequences_path: Optional[str] = None,
        source_path: Optional[str] = None,
        contract_name: Optional[str] = None,
        path_plan_path: Optional[str] = None,
        ksi_evidence_path: Optional[str] = None,
        blocker_evidence_path: Optional[str] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.contract_name = contract_name

        self._requested_blueprint_path = os.path.abspath(blueprint_path) if blueprint_path else None
        self._requested_selected_path = (
            os.path.abspath(selected_guided_sequences_path) if selected_guided_sequences_path else None
        )
        self._requested_source_path = os.path.abspath(source_path) if source_path else None
        self._requested_path_plan_path = os.path.abspath(path_plan_path) if path_plan_path else None
        self._requested_ksi_evidence_path = os.path.abspath(ksi_evidence_path) if ksi_evidence_path else None
        self._requested_blocker_evidence_path = (
            os.path.abspath(blocker_evidence_path) if blocker_evidence_path else None
        )

        self.default_blueprint_path = (
            os.path.join(self.output_dir, "guided_sequence_blueprint_dryrun.json") if self.output_dir else ""
        )
        self.default_selected_path = (
            os.path.join(self.output_dir, "selected_guided_sequences.json") if self.output_dir else ""
        )
        self.default_agent_path_plan_path = (
            os.path.join(self.output_dir, "agent_path_plan.json") if self.output_dir else ""
        )
        self.default_proposed_path_plan_path = (
            os.path.join(self.output_dir, "proposed_path_plan.json") if self.output_dir else ""
        )
        self.default_ksi_evidence_path = (
            os.path.join(self.output_dir, "ksi_evidence.jsonl") if self.output_dir else ""
        )
        self.default_blocker_evidence_path = (
            os.path.join(self.output_dir, "blocker_evidence.json") if self.output_dir else ""
        )

        self.loaded_inputs: Dict[str, bool] = {
            "blueprint": False,
            "selected_guided_sequences": False,
            "source": False,
            "path_plan": False,
            "ksi_evidence": False,
            "blocker_evidence": False,
        }
        self.input_paths: Dict[str, Optional[str]] = {
            "blueprint": self._requested_blueprint_path or self.default_blueprint_path or None,
            "selected_guided_sequences": self._requested_selected_path or self.default_selected_path or None,
            "source": self._requested_source_path,
            "path_plan": self._requested_path_plan_path or self.default_agent_path_plan_path or None,
            "ksi_evidence": self._requested_ksi_evidence_path or self.default_ksi_evidence_path or None,
            "blocker_evidence": self._requested_blocker_evidence_path or self.default_blocker_evidence_path or None,
        }

        self.adapter_mode = "pseudo_template"
        self.last_templates_payload: Dict[str, Any] = {}
        self.last_summary_payload: Dict[str, Any] = {}

    def build_templates(self) -> List[Dict[str, Any]]:
        blueprint_payload = self._load_json_with_fallback(
            self._requested_blueprint_path,
            self.default_blueprint_path,
            "blueprint",
        )
        selected_payload = self._load_json_with_fallback(
            self._requested_selected_path,
            self.default_selected_path,
            "selected_guided_sequences",
        )
        _ = self._load_path_plan_with_fallback()
        _ = self._load_jsonl_with_fallback(
            self._requested_ksi_evidence_path,
            self.default_ksi_evidence_path,
            "ksi_evidence",
        )
        _ = self._load_json_with_fallback(
            self._requested_blocker_evidence_path,
            self.default_blocker_evidence_path,
            "blocker_evidence",
        )
        self._mark_source_loaded(self._requested_source_path)

        blueprints = blueprint_payload.get("blueprints")
        if not isinstance(blueprints, list):
            blueprints = []

        selected = selected_payload.get("selected")
        if not isinstance(selected, list):
            selected = []
        selected_by_id = {
            _text(item.get("candidate_id")): item
            for item in selected
            if isinstance(item, dict) and _text(item.get("candidate_id"))
        }

        templates: List[Dict[str, Any]] = []
        for blueprint in blueprints:
            selected_item = selected_by_id.get(_text(blueprint.get("candidate_id"))) or {}
            templates.append(self._build_single_template(blueprint, selected_item))

        templates.sort(key=lambda item: int(item.get("selected_rank") or 10 ** 9))
        return templates

    def write_templates(
        self,
        templates: List[Dict[str, Any]],
        filename: str = "guided_sequence_individual_dryrun.json",
    ) -> str:
        if not self.output_dir:
            return ""
        output_path = os.path.join(self.output_dir, filename)
        payload = {
            "loaded_inputs": dict(self.loaded_inputs),
            "adapter_mode": self.adapter_mode,
            "templates": templates,
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        self.last_templates_payload = payload
        return output_path

    def write_summary(
        self,
        templates: List[Dict[str, Any]],
        filename: str = "guided_sequence_individual_summary.json",
    ) -> str:
        if not self.output_dir:
            return ""

        status_counts = Counter()
        adapter_mode_counts = Counter()
        strategy_status_counts: Dict[str, Counter] = defaultdict(Counter)
        target_function_counts = Counter()
        bad_function_count = 0
        for item in templates:
            status = _text(item.get("adapter_status")) or "skipped"
            adapter_mode = _text(item.get("adapter_mode")) or "pseudo_template"
            strategy = _text(item.get("sequence_strategy")) or "unknown"
            target_function = _text(item.get("target_function")) or "unknown"
            status_counts[status] += 1
            adapter_mode_counts[adapter_mode] += 1
            strategy_status_counts[strategy][status] += 1
            target_function_counts[target_function] += 1
            if item.get("all_functions_exist") is False:
                bad_function_count += 1

        payload = {
            "enabled": True,
            "selected_count": len(templates),
            "template_count": len(templates),
            "mappable_count": int(status_counts.get("mappable", 0)),
            "partial_count": int(status_counts.get("partial", 0)),
            "unmappable_count": int(status_counts.get("unmappable", 0)),
            "skipped_count": int(status_counts.get("skipped", 0)),
            "bad_function_count": int(bad_function_count),
            "adapter_mode_counts": {str(key): int(value) for key, value in adapter_mode_counts.items()},
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "sequence_strategy_status_counts": {
                str(key): {str(inner_key): int(inner_value) for inner_key, inner_value in counter.items()}
                for key, counter in strategy_status_counts.items()
            },
            "top_target_functions": {str(key): int(value) for key, value in target_function_counts.items()},
            "loaded_inputs": dict(self.loaded_inputs),
            "output_path": os.path.join(self.output_dir, "guided_sequence_individual_dryrun.json"),
        }

        output_path = os.path.join(self.output_dir, filename)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""

        self.last_summary_payload = payload
        return output_path

    def _load_json_with_fallback(
        self,
        requested_path: Optional[str],
        default_path: Optional[str],
        key: str,
    ) -> Dict[str, Any]:
        candidate_paths: List[str] = []
        if requested_path:
            candidate_paths.append(requested_path)
        if default_path and default_path not in candidate_paths:
            candidate_paths.append(default_path)

        self.loaded_inputs[key] = False
        self.input_paths[key] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_json(path)
            if payload:
                self.loaded_inputs[key] = True
                self.input_paths[key] = path
                return payload
        return {}

    def _load_jsonl_with_fallback(
        self,
        requested_path: Optional[str],
        default_path: Optional[str],
        key: str,
    ) -> List[Dict[str, Any]]:
        candidate_paths: List[str] = []
        if requested_path:
            candidate_paths.append(requested_path)
        if default_path and default_path not in candidate_paths:
            candidate_paths.append(default_path)

        self.loaded_inputs[key] = False
        self.input_paths[key] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_jsonl(path)
            if payload:
                self.loaded_inputs[key] = True
                self.input_paths[key] = path
                return payload
        return []

    def _load_path_plan_with_fallback(self) -> Dict[str, Any]:
        candidate_paths: List[str] = []
        if self._requested_path_plan_path:
            candidate_paths.append(self._requested_path_plan_path)
        if self.default_agent_path_plan_path and self.default_agent_path_plan_path not in candidate_paths:
            candidate_paths.append(self.default_agent_path_plan_path)
        if self.default_proposed_path_plan_path and self.default_proposed_path_plan_path not in candidate_paths:
            candidate_paths.append(self.default_proposed_path_plan_path)

        self.loaded_inputs["path_plan"] = False
        self.input_paths["path_plan"] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_json(path)
            hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
            if isinstance(hypotheses, list):
                self.loaded_inputs["path_plan"] = True
                self.input_paths["path_plan"] = path
                return payload
        return {}

    def _mark_source_loaded(self, requested_path: Optional[str]) -> None:
        self.loaded_inputs["source"] = False
        self.input_paths["source"] = requested_path
        if requested_path and os.path.exists(requested_path):
            self.loaded_inputs["source"] = True

    def _build_single_template(self, blueprint: Any, selected_item: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(blueprint, dict):
            return {
                "candidate_id": None,
                "selected_rank": None,
                "hypothesis_id": None,
                "guided_sequence": [],
                "adapter_status": "skipped",
                "adapter_mode": self.adapter_mode,
                "all_functions_exist": False,
                "missing_functions": [],
                "chromosome_template": {"length": 0, "genes": []},
                "skip_reason": "blueprint_not_dict",
                "source": "guided_sequence_execution_adapter_dryrun_v1",
            }

        candidate_id = _text(blueprint.get("candidate_id"))
        guided_sequence = _normalize_path(blueprint.get("guided_sequence"))
        if not candidate_id:
            return {
                "candidate_id": None,
                "selected_rank": blueprint.get("selected_rank"),
                "hypothesis_id": _text(blueprint.get("hypothesis_id")) or None,
                "guided_sequence": guided_sequence,
                "adapter_status": "skipped",
                "adapter_mode": self.adapter_mode,
                "all_functions_exist": False,
                "missing_functions": [],
                "chromosome_template": {"length": 0, "genes": []},
                "skip_reason": "missing_candidate_id",
                "source": "guided_sequence_execution_adapter_dryrun_v1",
            }
        if not guided_sequence:
            return {
                "candidate_id": candidate_id,
                "selected_rank": blueprint.get("selected_rank"),
                "hypothesis_id": _text(blueprint.get("hypothesis_id")) or None,
                "guided_sequence": [],
                "adapter_status": "skipped",
                "adapter_mode": self.adapter_mode,
                "all_functions_exist": False,
                "missing_functions": [],
                "chromosome_template": {"length": 0, "genes": []},
                "skip_reason": "empty_guided_sequence",
                "source": "guided_sequence_execution_adapter_dryrun_v1",
            }

        tx_blueprint = blueprint.get("tx_blueprint")
        if not isinstance(tx_blueprint, list):
            tx_blueprint = []

        genes = []
        gene_statuses: List[str] = []
        for index, tx_item in enumerate(tx_blueprint):
            gene = self._build_gene_template(tx_item, selected_item, index)
            genes.append(gene)
            gene_statuses.append(_text(gene.get("call_template_status")) or "partial")

        missing_functions = list(blueprint.get("missing_functions") or [])
        blueprint_status = _text(blueprint.get("blueprint_status")) or "partial"
        all_functions_exist = blueprint.get("all_functions_exist") is not False and not missing_functions

        if missing_functions or not genes:
            adapter_status = "unmappable"
        elif blueprint_status == "mappable" and all(status == "mappable" for status in gene_statuses):
            adapter_status = "mappable"
        elif any(status == "unmappable" for status in gene_statuses):
            adapter_status = "unmappable"
        else:
            adapter_status = "partial"

        return {
            "candidate_id": candidate_id,
            "selected_rank": blueprint.get("selected_rank"),
            "hypothesis_id": _text(blueprint.get("hypothesis_id")) or None,
            "origin_hypothesis_id": _text(blueprint.get("origin_hypothesis_id")) or None,
            "diagnosis_type": _text(blueprint.get("diagnosis_type")) or None,
            "sequence_strategy": _text(blueprint.get("sequence_strategy")) or None,
            "target_function": _text(blueprint.get("target_function")) or None,
            "guided_sequence": guided_sequence,
            "adapter_status": adapter_status,
            "adapter_mode": self.adapter_mode,
            "all_functions_exist": bool(all_functions_exist),
            "missing_functions": missing_functions,
            "chromosome_template": {
                "length": len(genes),
                "genes": genes,
            },
            "skip_reason": None,
            "source": "guided_sequence_execution_adapter_dryrun_v1",
        }

    def _build_gene_template(
        self,
        tx_item: Any,
        selected_item: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        if not isinstance(tx_item, dict):
            return {
                "index": int(index),
                "function": None,
                "arity": None,
                "sender_policy": _text(selected_item.get("sender_policy")) or None,
                "value_policy": _text(selected_item.get("value_policy")) or None,
                "argument_policy": _text(selected_item.get("argument_policy")) or None,
                "args_template": [],
                "value_template": "UNKNOWN_VALUE_POLICY",
                "sender_template": "UNKNOWN_SENDER_POLICY",
                "call_template_status": "unmappable",
                "reason": "tx blueprint item is not a dictionary",
            }

        function_name = _text(tx_item.get("function"))
        arity = tx_item.get("arity")
        sender_policy = _text(tx_item.get("sender_policy")) or _text(selected_item.get("sender_policy"))
        value_policy = _text(tx_item.get("value_policy")) or _text(selected_item.get("value_policy"))
        argument_policy = _text(tx_item.get("argument_policy")) or _text(selected_item.get("argument_policy"))
        base_status = _text(tx_item.get("status")) or "partial"

        sender_template = self._map_sender_policy(sender_policy)
        value_template = self._map_value_policy(value_policy)
        args_template = self._map_argument_policy(argument_policy, function_name, arity)

        if not function_name:
            call_template_status = "unmappable"
            reason = "function name is missing"
        elif not args_template:
            call_template_status = "unmappable"
            reason = "argument template could not be generated"
        elif base_status == "mappable" and sender_template != "UNKNOWN_SENDER_POLICY" and value_template != "UNKNOWN_VALUE_POLICY":
            call_template_status = "mappable"
            reason = "function and template bindings can map to a chromosome gene preview"
        elif base_status == "unmappable":
            call_template_status = "unmappable"
            reason = _text(tx_item.get("reason")) or "blueprint already marked this call as unmappable"
        else:
            call_template_status = "partial"
            reason = _text(tx_item.get("reason")) or "function can map conceptually, but some bindings remain approximate"

        return {
            "index": int(index),
            "function": function_name or None,
            "arity": arity if isinstance(arity, int) else None,
            "sender_policy": sender_policy or None,
            "value_policy": value_policy or None,
            "argument_policy": argument_policy or None,
            "args_template": args_template,
            "value_template": value_template,
            "sender_template": sender_template,
            "call_template_status": call_template_status,
            "reason": reason,
            "chromosome_gene_shape": {
                "account": sender_template,
                "contract": self.contract_name or "CURRENT_CONTRACT_INSTANCE",
                "amount": value_template,
                "arguments": ["FUNCTION_SELECTOR"] + args_template,
                "blocknumber": "REUSE_OR_UNSPECIFIED",
                "timestamp": "REUSE_OR_UNSPECIFIED",
                "gaslimit": "REUSE_OR_UNSPECIFIED",
                "call_return": "OPTIONAL_EMPTY_DICT",
                "extcodesize": "OPTIONAL_EMPTY_DICT",
                "returndatasize": "OPTIONAL_EMPTY_DICT",
            },
        }

    @staticmethod
    def _map_sender_policy(sender_policy: str) -> str:
        return SENDER_POLICY_TEMPLATES.get(sender_policy, "UNKNOWN_SENDER_POLICY")

    @staticmethod
    def _map_value_policy(value_policy: str) -> str:
        return VALUE_POLICY_TEMPLATES.get(value_policy, "UNKNOWN_VALUE_POLICY")

    @staticmethod
    def _map_argument_policy(argument_policy: str, function_name: str, arity: Any) -> List[str]:
        normalized_arity = arity if isinstance(arity, int) and arity >= 0 else None
        lowered_function = _text(function_name).lower()

        if argument_policy == "mutate_transfer_arguments":
            if lowered_function == "approve" and normalized_arity == 2:
                return ["ARG_SPENDER_OR_ADDRESS", "ARG_AMOUNT"]
            if lowered_function == "transferfrom" and normalized_arity == 3:
                return ["ARG_FROM", "ARG_TO", "ARG_AMOUNT"]
            if lowered_function == "transfer" and normalized_arity == 2:
                return ["ARG_TO", "ARG_AMOUNT"]
            return RepairSequenceExecutionAdapterDryRun._repeat_template("ARG_MUTATED", normalized_arity)

        if argument_policy in ("mutate_sink_arguments", "mutate_frontier_arguments", "mutate_permission_and_sink_args"):
            return RepairSequenceExecutionAdapterDryRun._repeat_template("ARG_MUTATED", normalized_arity)

        if argument_policy == "try_boolean_true_false":
            return RepairSequenceExecutionAdapterDryRun._repeat_template("BOOL_TRUE_FALSE", normalized_arity)

        if argument_policy == "try_zero_one_max_boundary_values":
            return RepairSequenceExecutionAdapterDryRun._repeat_template("ZERO_ONE_MAX_BOUNDARY", normalized_arity)

        return RepairSequenceExecutionAdapterDryRun._repeat_template("ARG_UNKNOWN", normalized_arity)

    @staticmethod
    def _repeat_template(prefix: str, arity: Optional[int]) -> List[str]:
        if arity is None or arity <= 0:
            return ["{}_0".format(prefix)]
        return ["{}_{}".format(prefix, index) for index in range(arity)]

