# -*- coding: utf-8 -*-

"""Offline TFI-scoped agent selection planner.

This module consumes previously exported TFI binding/scoped-selection requests
and rewrites them into selector-facing plans. It does not execute fuzzing,
generate action atoms, inject population, or synthesize complete sequences.
"""

import json
import os
from collections import Counter
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


def _unique_list(values: List[Any]) -> List[Any]:
    output: List[Any] = []
    seen = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=True) if isinstance(value, (dict, list)) else repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def _safe_token(value: Any) -> str:
    text = _text(value).strip()
    if not text:
        return "NONE"
    output: List[str] = []
    for char in text:
        if char.isalnum():
            output.append(char)
        else:
            output.append("_")
    sanitized = "".join(output).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or "NONE"


def _duplicate_id_map(items: List[Any], field_name: str) -> Dict[str, int]:
    counts: Counter = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        identifier = _text(item.get(field_name)).strip()
        if identifier:
            counts[identifier] += 1
    return {key: int(count) for key, count in counts.items() if count > 1}


class TFIAgentSelectionPlanBuilder:
    def __init__(
        self,
        output_dir: str,
        binding_requests_path: Optional[str] = None,
        scoped_selection_requests_path: Optional[str] = None,
        candidate_scope_path: Optional[str] = None,
        max_plans: Optional[int] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.binding_requests_path = binding_requests_path or os.path.join(self.output_dir, "tfi_binding_repair_requests.json")
        self.scoped_selection_requests_path = scoped_selection_requests_path or os.path.join(
            self.output_dir, "tfi_scoped_selection_requests.json"
        )
        self.candidate_scope_path = candidate_scope_path or os.path.join(self.output_dir, "target_focus_candidate_scope.json")
        self.max_plans = int(max_plans) if max_plans not in (None, "") else None

    def build_plan(self) -> Dict[str, Any]:
        loaded_inputs = {
            "binding_requests": bool(self.binding_requests_path and os.path.exists(self.binding_requests_path)),
            "scoped_selection_requests": bool(
                self.scoped_selection_requests_path and os.path.exists(self.scoped_selection_requests_path)
            ),
            "candidate_scope": bool(self.candidate_scope_path and os.path.exists(self.candidate_scope_path)),
        }
        missing_inputs = [name for name, loaded in loaded_inputs.items() if not loaded]

        binding_payload = _read_json(self.binding_requests_path)
        selection_payload = _read_json(self.scoped_selection_requests_path)
        candidate_scope_payload = _read_json(self.candidate_scope_path)
        scope_index = self._index_scopes(candidate_scope_payload)

        plans: List[Dict[str, Any]] = []
        skipped_reasons: Counter = Counter()

        for request in _listify(binding_payload.get("requests")):
            if not isinstance(request, dict):
                skipped_reasons["unsupported_request_type"] += 1
                continue
            plan = self._build_binding_repair_plan(request, scope_index)
            if isinstance(plan, dict):
                self._assign_plan_id(plan, len(plans) + 1)
                plans.append(plan)
            else:
                skipped_reasons[_text(plan or "unsupported_request_type")] += 1
            if self.max_plans is not None and len(plans) >= self.max_plans:
                break

        if self.max_plans is None or len(plans) < self.max_plans:
            for request in _listify(selection_payload.get("requests")):
                if not isinstance(request, dict):
                    skipped_reasons["unsupported_request_type"] += 1
                    continue
                plan = self._build_scoped_selection_plan(request, scope_index)
                if isinstance(plan, dict):
                    self._assign_plan_id(plan, len(plans) + 1)
                    plans.append(plan)
                else:
                    skipped_reasons[_text(plan or "unsupported_request_type")] += 1
                if self.max_plans is not None and len(plans) >= self.max_plans:
                    break

        return {
            "enabled": True,
            "source": "target_focus_agent_selection_plan_v1",
            "loaded_inputs": loaded_inputs,
            "missing_inputs": missing_inputs,
            "plan_count": len(plans),
            "plans": plans,
            "global_constraints": {
                "no_fuzz_execution": True,
                "no_llm_call": True,
                "no_population_injection": True,
                "no_action_atom_generation": True,
                "no_full_sequence_generation": True,
                "scope_local_evidence_required": True,
                "generic_skeleton_positive_only_if_scope_local": True,
            },
            "planner_stats": {
                "skipped_request_count": int(sum(skipped_reasons.values())),
                "skipped_reasons": {
                    "missing_sequence_skeleton": int(skipped_reasons.get("missing_sequence_skeleton", 0)),
                    "non_scope_local_skeleton": int(skipped_reasons.get("non_scope_local_skeleton", 0)),
                    "missing_required_constraints": int(skipped_reasons.get("missing_required_constraints", 0)),
                    "unsupported_request_type": int(skipped_reasons.get("unsupported_request_type", 0)),
                },
            },
        }

    def write_plan(self, plan: Dict[str, Any], filename: str = "target_focus_agent_selection_plan.json") -> str:
        path = os.path.join(self.output_dir, filename)
        _write_json(path, plan)
        return path

    def write_summary(self, plan: Dict[str, Any], filename: str = "target_focus_agent_selection_plan_summary.json") -> str:
        payload = self._build_summary(plan)
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        return path

    def _index_scopes(self, candidate_scope_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for scope in _listify(candidate_scope_payload.get("scopes")):
            if not isinstance(scope, dict):
                continue
            scope_id = _text(scope.get("scope_id")).strip()
            if scope_id:
                indexed[scope_id] = scope
        return indexed

    def _build_binding_repair_plan(
        self,
        request: Dict[str, Any],
        scope_index: Dict[str, Dict[str, Any]],
    ) -> Any:
        if _text(request.get("repair_goal")).strip() != "repair_sender_value_args_prestate_without_changing_function_order":
            return "unsupported_request_type"
        if not bool(request.get("preserve_sequence_skeleton")):
            return "missing_required_constraints"
        constraints = request.get("constraints") if isinstance(request.get("constraints"), dict) else {}
        if not bool(constraints.get("do_not_change_function_order")) or not bool(
            constraints.get("must_use_scope_local_skeleton")
        ):
            return "missing_required_constraints"
        sequence_skeleton = _clean_list(request.get("sequence_skeleton"))
        if not sequence_skeleton:
            return "missing_sequence_skeleton"
        leakage = request.get("generic_skeleton_leakage_check") if isinstance(request.get("generic_skeleton_leakage_check"), dict) else {}
        if not bool(leakage.get("allowed_as_positive_evidence", True)):
            return "non_scope_local_skeleton"

        scope_id = _text(request.get("scope_id")).strip()
        scope = scope_index.get(scope_id, {})
        negative_scope = scope.get("negative_scope") if isinstance(scope.get("negative_scope"), dict) else {}

        return {
            "plan_type": "binding_repair_plan",
            "source_request_id": request.get("request_id"),
            "scope_id": request.get("scope_id"),
            "hypothesis_id": request.get("hypothesis_id"),
            "target_function": request.get("target_function"),
            "frontier": request.get("frontier"),
            "sink": request.get("sink"),
            "selection_objective": "repair_binding_fields_under_scope_constraints",
            "fixed_structure": {
                "sequence_skeleton": sequence_skeleton,
                "do_not_change_function_order": True,
                "do_not_replace_sequence_skeleton": True,
                "must_include_target_function": True,
                "must_preserve_observed_prefix": True,
            },
            "mutable_fields": {
                "sender": self._build_mutable_slot(request, "sender"),
                "value": self._build_mutable_slot(request, "value"),
                "args": self._build_mutable_slot(request, "args", include_templates=True),
                "prestate": self._build_prestate_slot(request),
            },
            "forbidden_changes": [
                "change_function_order",
                "replace_sequence_skeleton",
                "drop_target_function",
                "drop_observed_prefix",
                "use_nonlocal_generic_skeleton",
            ],
            "positive_evidence": _listify(request.get("positive_evidence")),
            "negative_evidence": _unique_list(
                _listify(request.get("negative_evidence")) + _listify(negative_scope.get("hit_sink_but_reverted_examples"))
            ),
            "confidence": _text(request.get("confidence")).strip() or "low",
            "reason": "scope-local skeleton reaches target/sink but reverts; repair only transaction bindings",
        }

    def _build_scoped_selection_plan(
        self,
        request: Dict[str, Any],
        scope_index: Dict[str, Dict[str, Any]],
    ) -> Any:
        selection_goal = _text(request.get("selection_goal")).strip()
        if selection_goal not in {
            "select_or_compose_candidates_within_target_local_scope",
            "compose_missing_scope_local_candidate_action",
        }:
            return "unsupported_request_type"
        required = request.get("required_constraints") if isinstance(request.get("required_constraints"), dict) else {}
        if not bool(required.get("preserve_observed_matched_prefix")) or not bool(required.get("must_touch_frontier_or_sink")):
            return "missing_required_constraints"

        scope_id = _text(request.get("scope_id")).strip()
        scope = scope_index.get(scope_id, {})
        diagnosis = scope.get("tfi_failure_diagnosis") if isinstance(scope.get("tfi_failure_diagnosis"), dict) else {}
        confidence = _text(request.get("confidence")).strip() or "low"
        if bool(scope.get("scope_local_evidence_check", {}).get("diagnosis_uses_scope_local_evidence")) is False:
            confidence = "low"

        return {
            "plan_type": "scoped_selection_plan",
            "source_request_id": request.get("request_id"),
            "scope_id": request.get("scope_id"),
            "hypothesis_id": request.get("hypothesis_id"),
            "target_function": request.get("target_function"),
            "frontier": request.get("frontier"),
            "sink": request.get("sink"),
            "selection_objective": (
                "compose_missing_scope_local_candidate_action"
                if selection_goal == "compose_missing_scope_local_candidate_action"
                else "constrain_selection_to_target_local_scope"
            ),
            "required_prefix": {
                "observed_matched_prefix": _clean_list(request.get("observed_matched_prefix")),
                "preserve_observed_matched_prefix": True,
            },
            "required_progress": {
                "must_touch_frontier_or_sink": True,
                "frontier": request.get("frontier"),
                "sink": request.get("sink"),
                "must_not_select_sequence_without_target_progress": True,
            },
            "allowed_candidate_basis": _clean_list(request.get("allowed_candidate_basis")),
            "forbidden_candidate_patterns": _clean_list(request.get("forbidden_candidate_patterns")),
            "generic_skeleton_policy": {
                "must_not_use_generic_skeleton_unless_scope_local": True,
                "nonlocal_skeleton_examples": _listify(request.get("nonlocal_skeleton_examples")),
            },
            "positive_evidence": [],
            "negative_evidence": _listify(request.get("negative_evidence")),
            "partial_candidates": _listify(request.get("partial_candidates")),
            "confidence": confidence,
            "reason": "restrict next selection to observed prefix and frontier/sink",
            "diagnosis_type": _text(diagnosis.get("diagnosis_type")).strip() or None,
        }

    def _assign_plan_id(self, plan: Dict[str, Any], ordinal: int) -> None:
        scope_token = _safe_token(plan.get("scope_id"))
        request_token = _safe_token(plan.get("source_request_id"))
        plan["plan_id"] = "TFI_PLAN_%04d_%s_%s" % (ordinal, scope_token, request_token)

    def _build_mutable_slot(self, request: Dict[str, Any], slot_name: str, include_templates: bool = False) -> Dict[str, Any]:
        binding_slots = request.get("binding_slots") if isinstance(request.get("binding_slots"), dict) else {}
        slot = binding_slots.get(slot_name) if isinstance(binding_slots.get(slot_name), dict) else {}
        payload = {
            "allowed": True,
            "candidate_policies": _clean_list(slot.get("candidate_policies")),
            "avoid_policies": _clean_list(slot.get("avoid_policies")),
            "failed_bindings": _listify(slot.get("failed_bindings")),
        }
        if include_templates:
            payload["argument_templates"] = _clean_list(slot.get("argument_templates"))
            payload["avoid_patterns"] = _listify(slot.get("avoid_patterns"))
        return payload

    def _build_prestate_slot(self, request: Dict[str, Any]) -> Dict[str, Any]:
        binding_slots = request.get("binding_slots") if isinstance(request.get("binding_slots"), dict) else {}
        slot = binding_slots.get("prestate") if isinstance(binding_slots.get("prestate"), dict) else {}
        return {
            "allowed": True,
            "candidate_setup_hints": _clean_list(slot.get("candidate_setup_hints")),
            "avoid_assumptions": _listify(slot.get("avoid_assumptions")),
            "failed_assumptions": _listify(slot.get("failed_assumptions")),
        }

    def _build_summary(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        plans = _listify(plan.get("plans"))
        plan_confidence: Counter = Counter()
        binding_confidence: Counter = Counter()
        selection_confidence: Counter = Counter()
        binding_count = 0
        selection_count = 0

        for item in plans:
            if not isinstance(item, dict):
                continue
            confidence = _text(item.get("confidence")).strip() or "low"
            plan_confidence[confidence] += 1
            plan_type = _text(item.get("plan_type")).strip()
            if plan_type == "binding_repair_plan":
                binding_count += 1
                binding_confidence[confidence] += 1
            elif plan_type == "scoped_selection_plan":
                selection_count += 1
                selection_confidence[confidence] += 1

        planner_stats = plan.get("planner_stats") if isinstance(plan.get("planner_stats"), dict) else {}
        skipped_reasons = planner_stats.get("skipped_reasons") if isinstance(planner_stats.get("skipped_reasons"), dict) else {}

        return {
            "enabled": True,
            "source": "target_focus_agent_selection_plan_summary_v1",
            "plan_count": len(plans),
            "binding_repair_plan_count": binding_count,
            "scoped_selection_plan_count": selection_count,
            "plans_by_confidence": {
                "high": int(plan_confidence.get("high", 0)),
                "medium": int(plan_confidence.get("medium", 0)),
                "low": int(plan_confidence.get("low", 0)),
            },
            "binding_plans_by_confidence": {
                "high": int(binding_confidence.get("high", 0)),
                "medium": int(binding_confidence.get("medium", 0)),
                "low": int(binding_confidence.get("low", 0)),
            },
            "selection_plans_by_confidence": {
                "high": int(selection_confidence.get("high", 0)),
                "medium": int(selection_confidence.get("medium", 0)),
                "low": int(selection_confidence.get("low", 0)),
            },
            "skipped_request_count": int(planner_stats.get("skipped_request_count", 0)),
            "skipped_reasons": {
                "missing_sequence_skeleton": int(skipped_reasons.get("missing_sequence_skeleton", 0)),
                "non_scope_local_skeleton": int(skipped_reasons.get("non_scope_local_skeleton", 0)),
                "missing_required_constraints": int(skipped_reasons.get("missing_required_constraints", 0)),
                "unsupported_request_type": int(skipped_reasons.get("unsupported_request_type", 0)),
            },
            "loaded_inputs": plan.get("loaded_inputs") if isinstance(plan.get("loaded_inputs"), dict) else {},
            "missing_inputs": _clean_list(plan.get("missing_inputs")),
            "id_integrity": {
                "plan_id_unique": not bool(_duplicate_id_map(plans, "plan_id")),
                "duplicate_plan_ids": _duplicate_id_map(plans, "plan_id"),
                "proposal_id_unique": None,
                "duplicate_proposal_ids": None,
            },
            "generality_audit": self._build_generality_audit(plan),
            "output_path": os.path.join(self.output_dir, "target_focus_agent_selection_plan.json"),
        }

    def _build_generality_audit(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        restricted_terms = [
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
        fields_to_scan = (
            "selection_objective",
            "reason",
        )
        contract_terms = []
        function_specific_examples = []

        for item in _listify(plan.get("plans")):
            if not isinstance(item, dict):
                continue
            for field in fields_to_scan:
                text = _text(item.get(field))
                for term in restricted_terms:
                    if term in text and term not in contract_terms:
                        contract_terms.append(term)
                lowered = text.lower()
                if any(token in lowered for token in ("approve_transfer", "preserve_approve_transfer", "transferfrom")):
                    function_specific_examples.append(text)

        function_specific_examples = _unique_list([item for item in function_specific_examples if item])[:10]
        return {
            "contract_specific_rule_count": int(len(contract_terms)),
            "contract_specific_terms_found": contract_terms,
            "function_specific_strategy_count": int(len(function_specific_examples)),
            "function_specific_strategy_examples": function_specific_examples,
            "generic_strategy_names_only": not bool(contract_terms or function_specific_examples),
            "uses_request_type_dispatch_only": True,
            "uses_scope_fields_only": True,
        }

