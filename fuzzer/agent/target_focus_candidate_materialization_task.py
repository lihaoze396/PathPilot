# -*- coding: utf-8 -*-

"""Offline TFI candidate materialization task builder.

This module converts passed TFI candidate proposals into non-executable
materialization tasks. It does not execute fuzzing, run EVM transactions,
generate action atoms, inject population, or materialize concrete
transactions.
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


class TFICandidateMaterializationTaskBuilder:
    def __init__(
        self,
        output_dir: str,
        proposal_dryrun_path: Optional[str] = None,
        proposals_path: Optional[str] = None,
        agent_selection_plan_path: Optional[str] = None,
        candidate_scope_path: Optional[str] = None,
        max_tasks: Optional[int] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.proposal_dryrun_path = proposal_dryrun_path or os.path.join(
            self.output_dir, "target_focus_candidate_proposal_dryrun.json"
        )
        self.proposals_path = proposals_path or os.path.join(self.output_dir, "target_focus_candidate_proposals.json")
        self.agent_selection_plan_path = agent_selection_plan_path or os.path.join(
            self.output_dir, "target_focus_agent_selection_plan.json"
        )
        self.candidate_scope_path = candidate_scope_path or os.path.join(self.output_dir, "target_focus_candidate_scope.json")
        self.max_tasks = int(max_tasks) if max_tasks not in (None, "") else None

    def build_tasks(self) -> Dict[str, Any]:
        loaded_inputs = {
            "proposal_dryrun": bool(self.proposal_dryrun_path and os.path.exists(self.proposal_dryrun_path)),
            "candidate_proposals": bool(self.proposals_path and os.path.exists(self.proposals_path)),
            "agent_selection_plan": bool(self.agent_selection_plan_path and os.path.exists(self.agent_selection_plan_path)),
            "candidate_scope": bool(self.candidate_scope_path and os.path.exists(self.candidate_scope_path)),
        }
        missing_inputs = [name for name, loaded in loaded_inputs.items() if not loaded]

        dryrun_payload = _read_json(self.proposal_dryrun_path)
        proposal_payload = _read_json(self.proposals_path)
        plan_payload = _read_json(self.agent_selection_plan_path)
        scope_payload = _read_json(self.candidate_scope_path)

        if not loaded_inputs["proposal_dryrun"] or not loaded_inputs["candidate_proposals"]:
            return {
                "enabled": True,
                "source": "target_focus_candidate_materialization_task_v1",
                "loaded_inputs": loaded_inputs,
                "missing_inputs": missing_inputs,
                "task_count": 0,
                "tasks": [],
                "review_queue": [],
                "rejected_queue": [],
                "global_constraints": {
                    "only_passed_proposals": True,
                    "no_fuzz_execution": True,
                    "no_evm_execution": True,
                    "no_llm_call": True,
                    "no_population_injection": True,
                    "no_action_atom_generation": True,
                    "no_full_sequence_generation": True,
                    "no_concrete_tx_generation": True,
                    "task_is_not_executable_tx": True,
                },
            }

        proposal_index = self._index_proposals(proposal_payload)
        plan_index = self._index_plan_metadata(plan_payload)
        scope_index = self._index_scopes(scope_payload)

        tasks: List[Dict[str, Any]] = []
        review_queue: List[Dict[str, Any]] = []
        rejected_queue: List[Dict[str, Any]] = []

        for dryrun in _listify(dryrun_payload.get("dryruns")):
            if not isinstance(dryrun, dict):
                continue
            proposal_id = _text(dryrun.get("proposal_id")).strip()
            proposal = proposal_index.get(proposal_id, {})
            if _text(dryrun.get("dryrun_status")).strip() == "pass":
                if not isinstance(proposal, dict) or not proposal:
                    review_queue.append(
                        {
                            "proposal_id": proposal_id,
                            "proposal_type": dryrun.get("proposal_type"),
                            "scope_id": dryrun.get("scope_id"),
                            "dryrun_status": "partial",
                            "warnings": ["missing_proposal_reference"],
                            "review_reason": "incomplete_candidate_scope",
                        }
                    )
                    continue
                task = self._build_task(dryrun, proposal, plan_index, scope_index, len(tasks) + 1)
                if isinstance(task, dict):
                    tasks.append(task)
                    if self.max_tasks is not None and len(tasks) >= self.max_tasks:
                        break
                else:
                    review_queue.append(
                        {
                            "proposal_id": proposal_id,
                            "proposal_type": dryrun.get("proposal_type"),
                            "scope_id": dryrun.get("scope_id"),
                            "dryrun_status": "partial",
                            "warnings": [_text(task or "incomplete_candidate_scope")],
                            "review_reason": self._normalize_review_reason(_text(task or "incomplete_candidate_scope")),
                        }
                    )
            elif _text(dryrun.get("dryrun_status")).strip() == "partial":
                review_queue.append(
                    {
                        "proposal_id": proposal_id,
                        "proposal_type": dryrun.get("proposal_type"),
                        "scope_id": dryrun.get("scope_id"),
                        "dryrun_status": "partial",
                        "warnings": _clean_list(dryrun.get("warnings")),
                        "review_reason": self._pick_review_reason(dryrun),
                    }
                )
            elif _text(dryrun.get("dryrun_status")).strip() == "fail":
                rejected_queue.append(
                    {
                        "proposal_id": proposal_id,
                        "proposal_type": dryrun.get("proposal_type"),
                        "scope_id": dryrun.get("scope_id"),
                        "dryrun_status": "fail",
                        "violations": _clean_list(dryrun.get("violations")),
                        "reject_reason": self._pick_reject_reason(dryrun),
                    }
                )

        return {
            "enabled": True,
            "source": "target_focus_candidate_materialization_task_v1",
            "loaded_inputs": loaded_inputs,
            "missing_inputs": missing_inputs,
            "task_count": len(tasks),
            "tasks": tasks,
            "review_queue": review_queue,
            "rejected_queue": rejected_queue,
            "global_constraints": {
                "only_passed_proposals": True,
                "no_fuzz_execution": True,
                "no_evm_execution": True,
                "no_llm_call": True,
                "no_population_injection": True,
                "no_action_atom_generation": True,
                "no_full_sequence_generation": True,
                "no_concrete_tx_generation": True,
                "task_is_not_executable_tx": True,
            },
        }

    def write_tasks(self, tasks: Dict[str, Any], filename: str = "target_focus_candidate_materialization_tasks.json") -> str:
        path = os.path.join(self.output_dir, filename)
        _write_json(path, tasks)
        return path

    def write_summary(
        self,
        tasks: Dict[str, Any],
        filename: str = "target_focus_candidate_materialization_task_summary.json",
    ) -> str:
        payload = self._build_summary(tasks)
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        return path

    def _index_proposals(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in _listify(payload.get("proposals")):
            if not isinstance(item, dict):
                continue
            proposal_id = _text(item.get("proposal_id")).strip()
            if proposal_id:
                indexed[proposal_id] = item
        return indexed

    def _index_plan_metadata(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in _listify(payload.get("plans")):
            if not isinstance(item, dict):
                continue
            plan_id = _text(item.get("plan_id")).strip()
            if plan_id:
                indexed[plan_id] = item
        return indexed

    def _index_scopes(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in _listify(payload.get("scopes")):
            if not isinstance(item, dict):
                continue
            scope_id = _text(item.get("scope_id")).strip()
            if scope_id:
                indexed[scope_id] = item
        return indexed

    def _build_task(
        self,
        dryrun: Dict[str, Any],
        proposal: Dict[str, Any],
        plan_index: Dict[str, Dict[str, Any]],
        scope_index: Dict[str, Dict[str, Any]],
        ordinal: int,
    ) -> Any:
        del plan_index
        del scope_index
        proposal_type = _text(proposal.get("proposal_type")).strip()
        if proposal_type == "binding_repair_proposal":
            return self._build_binding_task(dryrun, proposal, ordinal)
        if proposal_type == "scoped_selection_proposal":
            return self._build_scoped_task(dryrun, proposal, ordinal)
        return "unsupported_proposal_type"

    def _build_binding_task(self, dryrun: Dict[str, Any], proposal: Dict[str, Any], ordinal: int) -> Any:
        fixed = proposal.get("fixed_sequence_shape") if isinstance(proposal.get("fixed_sequence_shape"), dict) else {}
        mutation = proposal.get("binding_mutation_space") if isinstance(proposal.get("binding_mutation_space"), dict) else {}
        if _text(dryrun.get("dryrun_status")).strip() != "pass":
            return "non_passed_proposal"
        skeleton = _clean_list(fixed.get("sequence_skeleton"))
        if not skeleton:
            return "incomplete_candidate_scope"

        return {
            "task_id": self._make_task_id(ordinal, proposal.get("scope_id"), proposal.get("source_proposal_id") or proposal.get("proposal_id")),
            "task_type": "binding_materialization_task",
            "source_proposal_id": proposal.get("proposal_id"),
            "source_plan_id": proposal.get("source_plan_id"),
            "source_request_id": proposal.get("source_request_id"),
            "scope_id": proposal.get("scope_id"),
            "hypothesis_id": proposal.get("hypothesis_id"),
            "target_function": proposal.get("target_function"),
            "frontier": proposal.get("frontier"),
            "sink": proposal.get("sink"),
            "materialization_goal": "materialize_transaction_bindings_without_changing_function_order",
            "fixed_sequence_shape": {
                "sequence_skeleton": skeleton,
                "do_not_change_function_order": bool(fixed.get("do_not_change_function_order")),
                "do_not_replace_sequence_skeleton": bool(fixed.get("do_not_replace_sequence_skeleton")),
                "do_not_add_new_function": bool(proposal.get("proposal_constraints", {}).get("do_not_add_new_function")),
                "do_not_drop_existing_function": bool(proposal.get("proposal_constraints", {}).get("do_not_drop_existing_function")),
            },
            "allowed_materialization_fields": {
                "sender": self._copy_slot(mutation, "sender"),
                "value": self._copy_slot(mutation, "value"),
                "args": self._copy_slot(mutation, "args", include_templates=True),
                "prestate": self._copy_prestate_slot(mutation),
            },
            "materialization_constraints": {
                "preserve_sequence_skeleton": True,
                "only_bind_transaction_fields": True,
                "do_not_generate_new_sequence": True,
                "do_not_generate_action_atom": True,
                "do_not_generate_concrete_tx_now": True,
                "requires_later_materializer": True,
            },
            "positive_evidence": _listify(dryrun.get("positive_evidence")),
            "negative_evidence": _listify(dryrun.get("negative_evidence")),
            "confidence": _text(proposal.get("confidence")).strip() or _text(dryrun.get("confidence")).strip() or "low",
            "next_stage": "binding_materializer",
            "reason": "dryrun passed; task is ready for later transaction binding materialization",
        }

    def _build_scoped_task(self, dryrun: Dict[str, Any], proposal: Dict[str, Any], ordinal: int) -> Any:
        required_prefix = proposal.get("required_prefix") if isinstance(proposal.get("required_prefix"), dict) else {}
        required_progress = proposal.get("required_progress") if isinstance(proposal.get("required_progress"), dict) else {}
        diagnosis_type = _text(proposal.get("diagnosis_type")).strip()
        if _text(dryrun.get("dryrun_status")).strip() != "pass":
            return "non_passed_proposal"

        return {
            "task_id": self._make_task_id(ordinal, proposal.get("scope_id"), proposal.get("source_proposal_id") or proposal.get("proposal_id")),
            "task_type": "scoped_candidate_materialization_task",
            "source_proposal_id": proposal.get("proposal_id"),
            "source_plan_id": proposal.get("source_plan_id"),
            "source_request_id": proposal.get("source_request_id"),
            "scope_id": proposal.get("scope_id"),
            "hypothesis_id": proposal.get("hypothesis_id"),
            "diagnosis_type": diagnosis_type or None,
            "target_function": proposal.get("target_function"),
            "frontier": proposal.get("frontier"),
            "sink": proposal.get("sink"),
            "materialization_goal": "materialize_candidate_under_target_local_constraints",
            "required_prefix": {
                "observed_matched_prefix": _clean_list(required_prefix.get("observed_matched_prefix")),
                "preserve_observed_matched_prefix": bool(required_prefix.get("preserve_observed_matched_prefix")),
            },
            "required_progress": {
                "must_touch_frontier_or_sink": bool(required_progress.get("must_touch_frontier_or_sink")),
                "frontier": proposal.get("frontier"),
                "sink": proposal.get("sink"),
                "must_not_select_sequence_without_target_progress": bool(
                    required_progress.get("must_not_select_sequence_without_target_progress")
                ),
            },
            "allowed_candidate_basis": _clean_list(proposal.get("allowed_candidate_basis")),
            "forbidden_candidate_patterns": _clean_list(proposal.get("forbidden_candidate_patterns")),
            "materialization_constraints": {
                "must_preserve_observed_prefix": bool(proposal.get("proposal_constraints", {}).get("must_preserve_observed_prefix")),
                "must_touch_frontier_or_sink": bool(proposal.get("proposal_constraints", {}).get("must_touch_frontier_or_sink")),
                "must_not_use_nonlocal_best_repair": bool(
                    proposal.get("proposal_constraints", {}).get("must_not_use_nonlocal_best_repair")
                ),
                "must_not_use_generic_skeleton_unless_scope_local": bool(
                    proposal.get("proposal_constraints", {}).get("must_not_use_generic_skeleton_unless_scope_local")
                ),
                "do_not_generate_concrete_tx_now": True,
                "requires_later_materializer": True,
                "allow_scope_local_action_composition": diagnosis_type == "possible_local_candidate_gap",
                "do_not_generate_unscoped_action_atom": True,
            },
            "generic_skeleton_policy": {
                "must_not_use_generic_skeleton_unless_scope_local": bool(
                    proposal.get("generic_skeleton_policy", {}).get("must_not_use_generic_skeleton_unless_scope_local")
                ),
                "nonlocal_skeleton_examples": _listify(proposal.get("generic_skeleton_policy", {}).get("nonlocal_skeleton_examples")),
            },
            "positive_evidence": _listify(dryrun.get("positive_evidence")),
            "negative_evidence": _listify(dryrun.get("negative_evidence")),
            "partial_candidates": _listify(proposal.get("partial_candidates")),
            "confidence": _text(proposal.get("confidence")).strip() or _text(dryrun.get("confidence")).strip() or "low",
            "next_stage": "scoped_candidate_materializer",
            "reason": "dryrun passed; task is ready for later candidate materialization under local constraints",
        }

    def _copy_slot(self, mutation: Dict[str, Any], slot_name: str, include_templates: bool = False) -> Dict[str, Any]:
        slot = mutation.get(slot_name) if isinstance(mutation.get(slot_name), dict) else {}
        payload = {
            "enabled": bool(slot.get("enabled")),
            "candidate_policies": _clean_list(slot.get("candidate_policies")),
            "avoid_policies": _clean_list(slot.get("avoid_policies")),
            "failed_bindings": _listify(slot.get("failed_bindings")),
        }
        if include_templates:
            payload["argument_templates"] = _clean_list(slot.get("argument_templates"))
            payload["avoid_patterns"] = _listify(slot.get("avoid_patterns"))
        return payload

    def _copy_prestate_slot(self, mutation: Dict[str, Any]) -> Dict[str, Any]:
        slot = mutation.get("prestate") if isinstance(mutation.get("prestate"), dict) else {}
        return {
            "enabled": bool(slot.get("enabled")),
            "candidate_setup_hints": _clean_list(slot.get("candidate_setup_hints")),
            "avoid_assumptions": _listify(slot.get("avoid_assumptions")),
            "failed_assumptions": _listify(slot.get("failed_assumptions")),
        }

    def _make_task_id(self, ordinal: int, scope_id: Any, proposal_id: Any) -> str:
        return "TFI_MAT_TASK_%04d_%s_%s" % (ordinal, _safe_token(scope_id), _safe_token(proposal_id))

    def _pick_review_reason(self, dryrun: Dict[str, Any]) -> str:
        warnings = _clean_list(dryrun.get("warnings"))
        confidence = _text(dryrun.get("confidence")).strip() or "low"
        if "observed_prefix_not_preserved" in warnings:
            return "observed_prefix_not_preserved"
        if "candidate_scope_missing" in warnings or "agent_selection_plan_missing" in warnings:
            return "incomplete_candidate_scope"
        if confidence == "low":
            return "low_confidence"
        return "missing_optional_scope_evidence"

    def _normalize_review_reason(self, reason: str) -> str:
        if reason in (
            "observed_prefix_not_preserved",
            "missing_optional_scope_evidence",
            "incomplete_candidate_scope",
            "low_confidence",
        ):
            return reason
        return "incomplete_candidate_scope"

    def _pick_reject_reason(self, dryrun: Dict[str, Any]) -> str:
        violations = _clean_list(dryrun.get("violations"))
        if "concrete_tx_generated" in violations:
            return "concrete_tx_generated"
        if "full_sequence_generated" in violations:
            return "full_sequence_generated"
        if "action_atom_generated" in violations:
            return "action_atom_generated"
        if "binding_skeleton_mismatch" in violations:
            return "skeleton_mismatch"
        return "constraint_violation"

    def _build_summary(self, tasks: Dict[str, Any]) -> Dict[str, Any]:
        task_items = _listify(tasks.get("tasks"))
        review_queue = _listify(tasks.get("review_queue"))
        rejected_queue = _listify(tasks.get("rejected_queue"))
        confidence_counts: Counter = Counter()
        binding_count = 0
        scoped_count = 0
        review_reasons: Counter = Counter()
        rejected_reasons: Counter = Counter()

        for item in task_items:
            if not isinstance(item, dict):
                continue
            confidence = _text(item.get("confidence")).strip() or "low"
            confidence_counts[confidence] += 1
            task_type = _text(item.get("task_type")).strip()
            if task_type == "binding_materialization_task":
                binding_count += 1
            elif task_type == "scoped_candidate_materialization_task":
                scoped_count += 1

        for item in review_queue:
            if not isinstance(item, dict):
                continue
            review_reasons[_text(item.get("review_reason")).strip() or "missing_optional_scope_evidence"] += 1

        for item in rejected_queue:
            if not isinstance(item, dict):
                continue
            rejected_reasons[_text(item.get("reject_reason")).strip() or "constraint_violation"] += 1

        source_status_counts = {
            "pass": len(task_items),
            "partial": len(review_queue),
            "fail": len(rejected_queue),
        }

        return {
            "enabled": True,
            "source": "target_focus_candidate_materialization_task_summary_v1",
            "task_count": len(task_items),
            "binding_materialization_task_count": binding_count,
            "scoped_candidate_materialization_task_count": scoped_count,
            "tasks_by_confidence": {
                "high": int(confidence_counts.get("high", 0)),
                "medium": int(confidence_counts.get("medium", 0)),
                "low": int(confidence_counts.get("low", 0)),
            },
            "review_queue_count": len(review_queue),
            "rejected_queue_count": len(rejected_queue),
            "review_reasons": {
                "observed_prefix_not_preserved": int(review_reasons.get("observed_prefix_not_preserved", 0)),
                "missing_optional_scope_evidence": int(review_reasons.get("missing_optional_scope_evidence", 0)),
                "incomplete_candidate_scope": int(review_reasons.get("incomplete_candidate_scope", 0)),
                "low_confidence": int(review_reasons.get("low_confidence", 0)),
            },
            "rejected_reasons": {
                "constraint_violation": int(rejected_reasons.get("constraint_violation", 0)),
                "skeleton_mismatch": int(rejected_reasons.get("skeleton_mismatch", 0)),
                "concrete_tx_generated": int(rejected_reasons.get("concrete_tx_generated", 0)),
                "full_sequence_generated": int(rejected_reasons.get("full_sequence_generated", 0)),
                "action_atom_generated": int(rejected_reasons.get("action_atom_generated", 0)),
            },
            "source_dryrun_status_counts": source_status_counts,
            "action_atom_generated": False,
            "full_sequence_generated": False,
            "concrete_tx_generated": False,
            "fuzz_executed": False,
            "evm_executed": False,
            "llm_called": False,
            "loaded_inputs": tasks.get("loaded_inputs") if isinstance(tasks.get("loaded_inputs"), dict) else {},
            "missing_inputs": _clean_list(tasks.get("missing_inputs")),
            "generality_audit": self._build_generality_audit(tasks),
            "output_path": os.path.join(self.output_dir, "target_focus_candidate_materialization_tasks.json"),
        }

    def _build_generality_audit(self, tasks: Dict[str, Any]) -> Dict[str, Any]:
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
        contract_terms: List[str] = []
        function_specific_examples: List[str] = []

        for item in _listify(tasks.get("tasks")):
            if not isinstance(item, dict):
                continue
            for field in ("materialization_goal", "reason", "next_stage"):
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
            "uses_task_type_dispatch_only": True,
            "uses_scope_fields_only": True,
        }

