# -*- coding: utf-8 -*-

"""Offline TFI candidate proposal builder.

This module converts TFI agent selection plans into non-executable candidate
proposals. It does not run fuzzing, generate action atoms, inject population,
or materialize concrete transactions.
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


class TFICandidateProposalBuilder:
    def __init__(
        self,
        output_dir: str,
        agent_selection_plan_path: Optional[str] = None,
        binding_requests_path: Optional[str] = None,
        scoped_selection_requests_path: Optional[str] = None,
        candidate_scope_path: Optional[str] = None,
        max_proposals: Optional[int] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.agent_selection_plan_path = agent_selection_plan_path or os.path.join(self.output_dir, "target_focus_agent_selection_plan.json")
        self.binding_requests_path = binding_requests_path or os.path.join(self.output_dir, "tfi_binding_repair_requests.json")
        self.scoped_selection_requests_path = scoped_selection_requests_path or os.path.join(
            self.output_dir, "tfi_scoped_selection_requests.json"
        )
        self.candidate_scope_path = candidate_scope_path or os.path.join(self.output_dir, "target_focus_candidate_scope.json")
        self.max_proposals = int(max_proposals) if max_proposals not in (None, "") else None

    def build_proposals(self) -> Dict[str, Any]:
        loaded_inputs = {
            "agent_selection_plan": bool(self.agent_selection_plan_path and os.path.exists(self.agent_selection_plan_path)),
            "binding_requests": bool(self.binding_requests_path and os.path.exists(self.binding_requests_path)),
            "scoped_selection_requests": bool(
                self.scoped_selection_requests_path and os.path.exists(self.scoped_selection_requests_path)
            ),
            "candidate_scope": bool(self.candidate_scope_path and os.path.exists(self.candidate_scope_path)),
        }
        missing_inputs = [name for name, loaded in loaded_inputs.items() if not loaded]

        plan_payload = _read_json(self.agent_selection_plan_path)
        binding_payload = _read_json(self.binding_requests_path)
        selection_payload = _read_json(self.scoped_selection_requests_path)
        candidate_scope_payload = _read_json(self.candidate_scope_path)

        request_index = self._index_requests(binding_payload, selection_payload)
        scope_index = self._index_scopes(candidate_scope_payload)

        proposals: List[Dict[str, Any]] = []
        skipped_reasons: Counter = Counter()
        possible_missing_action_atom_seen = 0

        for plan in _listify(plan_payload.get("plans")):
            if not isinstance(plan, dict):
                skipped_reasons["unsupported_plan_type"] += 1
                continue
            if _text(plan.get("diagnosis_type")).strip() == "possible_missing_action_atom":
                possible_missing_action_atom_seen += 1
                continue
            proposal = self._build_proposal(plan, request_index, scope_index)
            if isinstance(proposal, dict):
                self._assign_proposal_id(proposal, len(proposals) + 1)
                proposals.append(proposal)
            else:
                skipped_reasons[_text(proposal or "unsupported_plan_type")] += 1
            if self.max_proposals is not None and len(proposals) >= self.max_proposals:
                break

        return {
            "enabled": True,
            "source": "target_focus_candidate_proposal_v1",
            "loaded_inputs": loaded_inputs,
            "missing_inputs": missing_inputs,
            "proposal_count": len(proposals),
            "proposals": proposals,
            "global_constraints": {
                "no_fuzz_execution": True,
                "no_llm_call": True,
                "no_population_injection": True,
                "no_action_atom_generation": True,
                "no_full_sequence_generation": True,
                "proposal_is_not_executable_sequence": True,
                "scope_local_evidence_required": True,
                "generic_skeleton_positive_only_if_scope_local": True,
            },
            "proposal_stats": {
                "skipped_plan_count": int(sum(skipped_reasons.values())),
                "skipped_reasons": {
                    "missing_sequence_skeleton": int(skipped_reasons.get("missing_sequence_skeleton", 0)),
                    "non_scope_local_skeleton": int(skipped_reasons.get("non_scope_local_skeleton", 0)),
                    "missing_required_constraints": int(skipped_reasons.get("missing_required_constraints", 0)),
                    "unsupported_plan_type": int(skipped_reasons.get("unsupported_plan_type", 0)),
                },
                "possible_missing_action_atom_seen": int(possible_missing_action_atom_seen),
                "action_atom_generated": False,
                "full_sequence_generated": False,
            },
        }

    def write_proposals(self, proposals: Dict[str, Any], filename: str = "target_focus_candidate_proposals.json") -> str:
        path = os.path.join(self.output_dir, filename)
        _write_json(path, proposals)
        return path

    def write_summary(self, proposals: Dict[str, Any], filename: str = "target_focus_candidate_proposal_summary.json") -> str:
        payload = self._build_summary(proposals)
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        return path

    def _index_requests(self, binding_payload: Dict[str, Any], selection_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for bucket in (_listify(binding_payload.get("requests")), _listify(selection_payload.get("requests"))):
            for item in bucket:
                if not isinstance(item, dict):
                    continue
                request_id = _text(item.get("request_id")).strip()
                if request_id:
                    indexed[request_id] = item
        return indexed

    def _index_scopes(self, candidate_scope_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for scope in _listify(candidate_scope_payload.get("scopes")):
            if not isinstance(scope, dict):
                continue
            scope_id = _text(scope.get("scope_id")).strip()
            if scope_id:
                indexed[scope_id] = scope
        return indexed

    def _build_proposal(
        self,
        plan: Dict[str, Any],
        request_index: Dict[str, Dict[str, Any]],
        scope_index: Dict[str, Dict[str, Any]],
    ) -> Any:
        plan_type = _text(plan.get("plan_type")).strip()
        if plan_type == "binding_repair_plan":
            return self._build_binding_repair_proposal(plan, request_index, scope_index)
        if plan_type == "scoped_selection_plan":
            return self._build_scoped_selection_proposal(plan, request_index, scope_index)
        return "unsupported_plan_type"

    def _build_binding_repair_proposal(
        self,
        plan: Dict[str, Any],
        request_index: Dict[str, Dict[str, Any]],
        scope_index: Dict[str, Dict[str, Any]],
    ) -> Any:
        fixed = plan.get("fixed_structure") if isinstance(plan.get("fixed_structure"), dict) else {}
        mutable = plan.get("mutable_fields") if isinstance(plan.get("mutable_fields"), dict) else {}
        skeleton = _clean_list(fixed.get("sequence_skeleton"))
        if not skeleton:
            return "missing_sequence_skeleton"
        if not bool(fixed.get("do_not_change_function_order")) or not bool(fixed.get("do_not_replace_sequence_skeleton")):
            return "missing_required_constraints"
        if set(mutable.keys()) != {"sender", "value", "args", "prestate"}:
            return "missing_required_constraints"

        request = request_index.get(_text(plan.get("source_request_id")).strip(), {})
        leakage = request.get("generic_skeleton_leakage_check") if isinstance(request.get("generic_skeleton_leakage_check"), dict) else {}
        if leakage and not bool(leakage.get("allowed_as_positive_evidence", True)):
            return "non_scope_local_skeleton"

        return {
            "proposal_type": "binding_repair_proposal",
            "source_plan_id": plan.get("plan_id"),
            "source_request_id": plan.get("source_request_id"),
            "scope_id": plan.get("scope_id"),
            "hypothesis_id": plan.get("hypothesis_id"),
            "target_function": plan.get("target_function"),
            "frontier": plan.get("frontier"),
            "sink": plan.get("sink"),
            "skeleton_source_used": "source_plan.fixed_structure.sequence_skeleton",
            "proposal_goal": "preserve_scope_local_skeleton_and_repair_transaction_bindings",
            "fixed_sequence_shape": {
                "sequence_skeleton": skeleton,
                "do_not_change_function_order": True,
                "do_not_replace_sequence_skeleton": True,
                "must_include_target_function": bool(fixed.get("must_include_target_function")),
                "must_preserve_observed_prefix": bool(fixed.get("must_preserve_observed_prefix")),
            },
            "binding_mutation_space": {
                "sender": self._copy_slot(mutable, "sender"),
                "value": self._copy_slot(mutable, "value"),
                "args": self._copy_slot(mutable, "args", include_templates=True),
                "prestate": self._copy_prestate_slot(mutable),
            },
            "proposal_constraints": {
                "do_not_generate_new_function_order": True,
                "do_not_add_new_function": True,
                "do_not_drop_existing_function": True,
                "do_not_replace_skeleton": True,
                "only_repair_transaction_fields": True,
                "not_executable_without_later_concretization": True,
            },
            "positive_evidence": _listify(plan.get("positive_evidence")),
            "negative_evidence": _listify(plan.get("negative_evidence")),
            "forbidden_changes": _clean_list(plan.get("forbidden_changes")),
            "confidence": _text(plan.get("confidence")).strip() or "low",
            "reason": "scope-local skeleton is preserved; proposal only defines binding mutation space",
        }

    def _build_scoped_selection_proposal(
        self,
        plan: Dict[str, Any],
        request_index: Dict[str, Dict[str, Any]],
        scope_index: Dict[str, Dict[str, Any]],
    ) -> Any:
        required_prefix = plan.get("required_prefix") if isinstance(plan.get("required_prefix"), dict) else {}
        required_progress = plan.get("required_progress") if isinstance(plan.get("required_progress"), dict) else {}
        if not bool(required_prefix.get("preserve_observed_matched_prefix")) or not bool(
            required_progress.get("must_touch_frontier_or_sink")
        ):
            return "missing_required_constraints"

        request = request_index.get(_text(plan.get("source_request_id")).strip(), {})
        scope = scope_index.get(_text(plan.get("scope_id")).strip(), {})
        scope_diagnosis = (
            scope.get("tfi_failure_diagnosis")
            if isinstance(scope.get("tfi_failure_diagnosis"), dict)
            else {}
        )
        local_check = scope.get("scope_local_evidence_check") if isinstance(scope.get("scope_local_evidence_check"), dict) else {}
        confidence = _text(plan.get("confidence")).strip() or "low"
        if local_check and not bool(local_check.get("diagnosis_uses_scope_local_evidence")):
            confidence = "low"

        return {
            "proposal_type": "scoped_selection_proposal",
            "source_plan_id": plan.get("plan_id"),
            "source_request_id": plan.get("source_request_id"),
            "scope_id": plan.get("scope_id"),
            "hypothesis_id": plan.get("hypothesis_id"),
            "diagnosis_type": (
                _text(plan.get("diagnosis_type")).strip()
                or _text(request.get("diagnosis_type")).strip()
                or _text(scope_diagnosis.get("diagnosis_type")).strip()
                or None
            ),
            "target_function": plan.get("target_function"),
            "frontier": plan.get("frontier"),
            "sink": plan.get("sink"),
            "skeleton_source_used": None,
            "proposal_goal": "select_or_compose_within_target_local_constraints",
            "required_prefix": {
                "observed_matched_prefix": _clean_list(required_prefix.get("observed_matched_prefix")),
                "preserve_observed_matched_prefix": True,
            },
            "required_progress": {
                "must_touch_frontier_or_sink": True,
                "frontier": plan.get("frontier"),
                "sink": plan.get("sink"),
                "must_not_select_sequence_without_target_progress": bool(
                    required_progress.get("must_not_select_sequence_without_target_progress")
                ),
            },
            "allowed_candidate_basis": _clean_list(plan.get("allowed_candidate_basis")),
            "proposal_constraints": {
                "must_preserve_observed_prefix": True,
                "must_touch_frontier_or_sink": True,
                "must_not_use_nonlocal_best_repair": True,
                "must_not_use_generic_skeleton_unless_scope_local": True,
                "must_not_select_target_seen_without_path_progress": True,
                "not_executable_without_later_candidate_materialization": True,
            },
            "forbidden_candidate_patterns": _clean_list(plan.get("forbidden_candidate_patterns")),
            "generic_skeleton_policy": {
                "must_not_use_generic_skeleton_unless_scope_local": True,
                "nonlocal_skeleton_examples": _listify(plan.get("generic_skeleton_policy", {}).get("nonlocal_skeleton_examples")),
            },
            "positive_evidence": [],
            "negative_evidence": _listify(plan.get("negative_evidence")),
            "partial_candidates": _listify(plan.get("partial_candidates")),
            "confidence": confidence,
            "reason": "proposal constrains later candidate materialization to observed prefix and frontier/sink",
        }

    def _assign_proposal_id(self, proposal: Dict[str, Any], ordinal: int) -> None:
        scope_token = _safe_token(proposal.get("scope_id"))
        plan_token = _safe_token(proposal.get("source_plan_id"))
        proposal["proposal_id"] = "TFI_PROP_%04d_%s_%s" % (ordinal, scope_token, plan_token)

    def _copy_slot(self, mutable: Dict[str, Any], slot_name: str, include_templates: bool = False) -> Dict[str, Any]:
        slot = mutable.get(slot_name) if isinstance(mutable.get(slot_name), dict) else {}
        payload = {
            "enabled": bool(slot.get("allowed")),
            "candidate_policies": _clean_list(slot.get("candidate_policies")),
            "avoid_policies": _clean_list(slot.get("avoid_policies")),
            "failed_bindings": _listify(slot.get("failed_bindings")),
        }
        if include_templates:
            payload["argument_templates"] = _clean_list(slot.get("argument_templates"))
            payload["avoid_patterns"] = _listify(slot.get("avoid_patterns"))
        return payload

    def _copy_prestate_slot(self, mutable: Dict[str, Any]) -> Dict[str, Any]:
        slot = mutable.get("prestate") if isinstance(mutable.get("prestate"), dict) else {}
        return {
            "enabled": bool(slot.get("allowed")),
            "candidate_setup_hints": _clean_list(slot.get("candidate_setup_hints")),
            "avoid_assumptions": _listify(slot.get("avoid_assumptions")),
            "failed_assumptions": _listify(slot.get("failed_assumptions")),
        }

    def _build_summary(self, proposals: Dict[str, Any]) -> Dict[str, Any]:
        items = _listify(proposals.get("proposals"))
        confidence_counts: Counter = Counter()
        binding_confidence: Counter = Counter()
        selection_confidence: Counter = Counter()
        binding_count = 0
        selection_count = 0

        for item in items:
            if not isinstance(item, dict):
                continue
            confidence = _text(item.get("confidence")).strip() or "low"
            confidence_counts[confidence] += 1
            proposal_type = _text(item.get("proposal_type")).strip()
            if proposal_type == "binding_repair_proposal":
                binding_count += 1
                binding_confidence[confidence] += 1
            elif proposal_type == "scoped_selection_proposal":
                selection_count += 1
                selection_confidence[confidence] += 1

        proposal_stats = proposals.get("proposal_stats") if isinstance(proposals.get("proposal_stats"), dict) else {}
        skipped_reasons = proposal_stats.get("skipped_reasons") if isinstance(proposal_stats.get("skipped_reasons"), dict) else {}
        duplicate_proposal_ids = _duplicate_id_map(items, "proposal_id")
        source_plan_reference_fields_present = True
        for item in items:
            if not isinstance(item, dict):
                continue
            if not _text(item.get("source_plan_id")).strip():
                source_plan_reference_fields_present = False
                break
            if not _text(item.get("source_request_id")).strip():
                source_plan_reference_fields_present = False
                break
            if not _text(item.get("scope_id")).strip():
                source_plan_reference_fields_present = False
                break
            if not _text(item.get("hypothesis_id")).strip():
                source_plan_reference_fields_present = False
                break
            if "skeleton_source_used" not in item:
                source_plan_reference_fields_present = False
                break

        return {
            "enabled": True,
            "source": "target_focus_candidate_proposal_summary_v1",
            "proposal_count": len(items),
            "binding_repair_proposal_count": binding_count,
            "scoped_selection_proposal_count": selection_count,
            "proposals_by_confidence": {
                "high": int(confidence_counts.get("high", 0)),
                "medium": int(confidence_counts.get("medium", 0)),
                "low": int(confidence_counts.get("low", 0)),
            },
            "binding_proposals_by_confidence": {
                "high": int(binding_confidence.get("high", 0)),
                "medium": int(binding_confidence.get("medium", 0)),
                "low": int(binding_confidence.get("low", 0)),
            },
            "selection_proposals_by_confidence": {
                "high": int(selection_confidence.get("high", 0)),
                "medium": int(selection_confidence.get("medium", 0)),
                "low": int(selection_confidence.get("low", 0)),
            },
            "skipped_plan_count": int(proposal_stats.get("skipped_plan_count", 0)),
            "skipped_reasons": {
                "missing_sequence_skeleton": int(skipped_reasons.get("missing_sequence_skeleton", 0)),
                "non_scope_local_skeleton": int(skipped_reasons.get("non_scope_local_skeleton", 0)),
                "missing_required_constraints": int(skipped_reasons.get("missing_required_constraints", 0)),
                "unsupported_plan_type": int(skipped_reasons.get("unsupported_plan_type", 0)),
            },
            "possible_missing_action_atom_seen": int(proposal_stats.get("possible_missing_action_atom_seen", 0)),
            "action_atom_generated": False,
            "full_sequence_generated": False,
            "loaded_inputs": proposals.get("loaded_inputs") if isinstance(proposals.get("loaded_inputs"), dict) else {},
            "missing_inputs": _clean_list(proposals.get("missing_inputs")),
            "id_integrity": {
                "proposal_id_unique": not bool(duplicate_proposal_ids),
                "duplicate_proposal_ids": duplicate_proposal_ids,
                "source_plan_reference_fields_present": source_plan_reference_fields_present,
            },
            "generality_audit": self._build_generality_audit(proposals),
            "output_path": os.path.join(self.output_dir, "target_focus_candidate_proposals.json"),
        }

    def _build_generality_audit(self, proposals: Dict[str, Any]) -> Dict[str, Any]:
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
        contract_terms = []
        function_specific_examples = []

        for item in _listify(proposals.get("proposals")):
            if not isinstance(item, dict):
                continue
            for field in ("proposal_goal", "reason"):
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
            "uses_plan_type_dispatch_only": True,
            "uses_scope_fields_only": True,
        }

