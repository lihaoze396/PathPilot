# -*- coding: utf-8 -*-

"""State-obligation construction for grounded repair.

This module only rewrites existing failure/runtime/static evidence into a
compact obligation record. It does not call the LLM, generate transactions, or
change the search policy.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = _text(item).strip()
        if text:
            output.append(text)
    return output


def _dedup_text(items: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in items:
        text = _text(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _safe_token(value: Any) -> str:
    text = _text(value).strip()
    if not text:
        return "NONE"
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return text or "NONE"


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


class StateObligationBuilder:
    def __init__(self, output_dir: str = "") -> None:
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.jsonl_path = os.path.join(self.output_dir, "state_obligations.jsonl") if self.output_dir else ""
        self.summary_path = os.path.join(self.output_dir, "state_obligation_summary.json") if self.output_dir else ""
        self.counts = Counter()
        self.kind_counts = Counter()
        self.repair_location_counts = Counter()
        self.grounding_counts = Counter()
        if self.jsonl_path:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.jsonl_path, "w", encoding="utf-8"):
                pass

    def build(
        self,
        *,
        bench_id: Any,
        oracle_family: Any,
        runtime_pattern_family: Any = "",
        runtime_pattern_family_provenance: Any = "",
        best_level_before: Any,
        target_missing_checkpoint: Any,
        missing_checkpoint_explanation: Any,
        failure_signature: Any,
        covered_checkpoints: Any,
        parent_candidate: Optional[Dict[str, Any]] = None,
        execution_feedback: Optional[Dict[str, Any]] = None,
        runtime_observations: Optional[Dict[str, Any]] = None,
        static_hints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parent_candidate = parent_candidate if isinstance(parent_candidate, dict) else {}
        execution_feedback = execution_feedback if isinstance(execution_feedback, dict) else {}
        runtime_observations = runtime_observations if isinstance(runtime_observations, dict) else {}
        static_hints = static_hints if isinstance(static_hints, dict) else {}

        checkpoint = _text(target_missing_checkpoint).strip()
        best_level = _text(best_level_before).strip() or "none"
        failure_sig = _text(failure_signature).strip()
        explanation = _text(missing_checkpoint_explanation).strip()
        tx_success = bool(execution_feedback.get("tx_success"))
        reverted = bool(execution_feedback.get("reverted"))
        state_delta = bool(execution_feedback.get("state_delta_observed") or runtime_observations.get("state_delta_observed"))
        balance_delta = bool(execution_feedback.get("balance_delta_observed") or runtime_observations.get("balance_delta_observed"))
        token_transfer = bool(execution_feedback.get("token_transfer_observed") or runtime_observations.get("token_transfer_observed"))
        external_call = bool(execution_feedback.get("external_call_result_observed") or runtime_observations.get("external_call_result_observed"))
        repeated_action = bool(runtime_observations.get("repeated_function_observed") or runtime_observations.get("candidate_repeat_action_available"))
        target_suffix_reached = bool(execution_feedback.get("target_suffix_reached") or runtime_observations.get("target_suffix_reached"))
        state_writer_functions = _clean_list(runtime_observations.get("state_writer_functions"))
        sink_candidate_functions = _clean_list(runtime_observations.get("sink_candidate_functions"))
        guard_related_functions = _clean_list(runtime_observations.get("guard_related_functions"))
        repeat_candidate_functions = _clean_list(runtime_observations.get("repeat_candidate_functions"))
        observed_obligations = _clean_list(runtime_observations.get("observed_obligation_hints"))
        runtime_effect_signals = _clean_list(runtime_observations.get("runtime_effect_signals"))
        static_loaded = bool(static_hints.get("loaded"))
        static_present = static_loaded and bool(
            static_hints.get("storage_dependencies")
            or static_hints.get("dangerous_operations")
            or static_hints.get("privileged_functions")
        )

        kinds: List[str] = []
        required_effect_evidence: List[str] = []
        already_satisfied_facts: List[str] = []
        unmet_state_obligations: List[str] = []
        preserved_prefix_facts: List[str] = []
        covered = _clean_list(covered_checkpoints)

        if not tx_success or reverted:
            kinds.extend(["enable_successful_transition", "satisfy_guard_or_precondition"])
            unmet_state_obligations.append("repair_failed_transition")
        if state_delta or state_writer_functions:
            kinds.append("produce_target_state_write")
        if balance_delta or token_transfer or external_call or sink_candidate_functions:
            kinds.append("produce_asset_or_sink_effect")
        if repeated_action or repeat_candidate_functions:
            kinds.append("preserve_or_repeat_relation")
        if state_delta:
            already_satisfied_facts.append("state_delta_observed")
        if balance_delta or token_transfer:
            already_satisfied_facts.append("asset_delta_observed")
        if external_call:
            already_satisfied_facts.append("external_call_observed")
        if repeated_action:
            already_satisfied_facts.append("repeated_action_observed")
        if target_suffix_reached:
            already_satisfied_facts.append("target_suffix_reached")
        if not tx_success:
            required_effect_evidence.append("tx_end_success")
        if not state_delta and (state_writer_functions or "state_delta_expected" in observed_obligations):
            required_effect_evidence.append("state_write_or_storage_delta")
            unmet_state_obligations.append("produce_target_state_write")
        if not (balance_delta or token_transfer or external_call) and (
            sink_candidate_functions or "asset_or_sink_effect_expected" in observed_obligations
        ):
            required_effect_evidence.append("asset_or_sink_effect")
            unmet_state_obligations.append("produce_asset_or_sink_effect")
        if not repeated_action and repeat_candidate_functions:
            required_effect_evidence.append("repeat_action")
            unmet_state_obligations.append("repeat_relation_not_realized")
        if guard_related_functions:
            kinds.append("satisfy_guard_or_precondition")
            if reverted or not tx_success:
                unmet_state_obligations.append("satisfy_guard_or_precondition")
        for hint in observed_obligations:
            if hint == "initialization_or_lifecycle_prestate_needed":
                kinds.extend(["construct_initialization_or_lifecycle_prestate", "satisfy_guard_or_precondition"])
                unmet_state_obligations.append("construct_initialization_or_lifecycle_prestate")
            elif hint == "balance_or_allowance_prestate_needed":
                kinds.extend(["construct_balance_or_allowance_prestate", "satisfy_guard_or_precondition"])
                unmet_state_obligations.append("construct_balance_or_allowance_prestate")
            elif hint == "dependency_local_reordering_needed":
                kinds.append("reorder_dependency_related_suffix")
                unmet_state_obligations.append("reorder_dependency_related_suffix")
            elif hint == "parameter_binding_repair_needed":
                kinds.append("repair_cross_transaction_parameter_binding")
                unmet_state_obligations.append("repair_cross_transaction_parameter_binding")
            elif hint == "sender_or_role_context_variation_needed":
                kinds.append("vary_sender_or_role_context_around_observed_revert")
                unmet_state_obligations.append("vary_sender_or_role_context_around_observed_revert")

        if failure_sig in {"wrong_sender_binding", "wrong_beneficiary_binding", "wrong_receiver_binding", "wrong_amount_binding"}:
            kinds.append("satisfy_guard_or_precondition")
            kinds.append("repair_cross_transaction_parameter_binding")
        if failure_sig in {"missing_prestate_prefix", "effect_function_not_reached", "suffix_not_effective"}:
            kinds.append("enable_successful_transition")
        if failure_sig == "missing_prestate_prefix":
            kinds.append("construct_initialization_or_lifecycle_prestate")
        if failure_sig == "effect_reached_but_no_state_delta":
            kinds.append("produce_target_state_write")
        if failure_sig == "effect_probe_missing_or_uninformative" and (state_delta or balance_delta or token_transfer or external_call):
            kinds.append("observe_already_produced_effect")
            already_satisfied_facts.append("effect_without_sufficient_probe_observed")
        elif failure_sig == "effect_probe_missing_or_uninformative":
            unmet_state_obligations.append("probe_only_repair_is_insufficient_without_effect_evidence")

        if checkpoint in covered:
            already_satisfied_facts.append("checkpoint_already_marked_covered")

        functions_reached = _clean_list(execution_feedback.get("functions_reached") or runtime_observations.get("functions_reached"))
        required_prefix = _clean_list(parent_candidate.get("required_prefix"))
        if required_prefix:
            preserved_prefix_facts.append("required_prefix:" + " -> ".join(required_prefix))
        if functions_reached:
            preserved_prefix_facts.append("functions_reached:" + " -> ".join(functions_reached[:8]))
        if bool(runtime_observations.get("same_sender_relation_observed")):
            preserved_prefix_facts.append("same_sender_relation_observed")
        if bool(runtime_observations.get("same_amount_relation_observed")):
            preserved_prefix_facts.append("same_amount_relation_observed")
        if bool(runtime_observations.get("same_beneficiary_relation_observed")):
            preserved_prefix_facts.append("same_beneficiary_relation_observed")

        repair_location = self._suspected_repair_location(
            checkpoint=checkpoint,
            failure_signature=failure_sig,
            tx_success=tx_success,
            reverted=reverted,
            state_delta=state_delta,
            balance_delta=balance_delta,
            observation_available=("observe_already_produced_effect" in kinds),
        )
        location_reason = self._repair_location_reason(
            checkpoint=checkpoint,
            failure_signature=failure_sig,
            repair_location=repair_location,
            tx_success=tx_success,
            reverted=reverted,
            state_delta=state_delta,
            balance_delta=balance_delta,
        )

        kinds = _dedup_text(kinds)
        required_effect_evidence = _dedup_text(required_effect_evidence)
        already_satisfied_facts = _dedup_text(already_satisfied_facts)
        unmet_state_obligations = _dedup_text(unmet_state_obligations)
        preserved_prefix_facts = _dedup_text(preserved_prefix_facts)

        if not kinds and already_satisfied_facts:
            kinds = ["observe_already_produced_effect"]
        elif not kinds:
            kinds = ["satisfy_guard_or_precondition"]

        grounding_status = "insufficient_grounding"
        if static_present and (functions_reached or state_delta or balance_delta or token_transfer or external_call):
            grounding_status = "static_and_runtime_grounded"
        elif functions_reached or state_delta or balance_delta or token_transfer or external_call or failure_sig:
            grounding_status = "runtime_grounded_only"

        confidence = self._confidence(
            failure_signature=failure_sig,
            grounding_status=grounding_status,
            unmet_state_obligations=unmet_state_obligations,
            static_present=static_present,
        )

        source_evidence = _dedup_text(
            list(execution_feedback.get("evidence") or [])
            + list(runtime_observations.get("source_evidence") or [])
            + required_effect_evidence
            + already_satisfied_facts
            + runtime_effect_signals
        )[:16]

        obligation_id = "OBL_{}_{}_{}_{}".format(
            _safe_token(bench_id),
            _safe_token(parent_candidate.get("candidate_id")),
            _safe_token(checkpoint),
            max(1, int(self.counts.get("state_obligation_created", 0) + 1)),
        )
        payload = {
            "obligation_id": obligation_id,
            "bench_id": _text(bench_id).strip(),
            "parent_candidate_id": _text(parent_candidate.get("candidate_id")).strip() or None,
            "runtime_pattern_family": _text(runtime_pattern_family).strip() or None,
            "runtime_pattern_family_provenance": _text(runtime_pattern_family_provenance).strip() or None,
            "target_missing_checkpoint": checkpoint,
            "best_level_before": best_level,
            "obligation_kinds": kinds,
            "required_effect_evidence": required_effect_evidence,
            "already_satisfied_facts": already_satisfied_facts,
            "unmet_state_obligations": unmet_state_obligations,
            "preserved_prefix_facts": preserved_prefix_facts,
            "suspected_repair_location": repair_location,
            "repair_location_reason": location_reason,
            "source_evidence": source_evidence,
            "confidence": confidence,
            "grounding_status": grounding_status,
            "missing_checkpoint_explanation": explanation if explanation and explanation.startswith("runtime:") else "",
            "failure_signature": failure_sig,
            "obligation_provenance": _dedup_text(
                ["KSI_RUNTIME_EVIDENCE_DERIVED"]
                + (["STATIC_ANALYSIS_DERIVED"] if static_present else [])
                + (["SOURCE_DERIVED"] if functions_reached else [])
            ),
            "runtime_soft_obligation": True,
        }
        self._record(payload)
        return payload

    def build_summary(self) -> Dict[str, Any]:
        return {
            "state_obligation_created": int(self.counts.get("state_obligation_created", 0)),
            "state_obligation_by_kind": dict(self.kind_counts),
            "state_obligation_static_and_runtime_grounded": int(self.grounding_counts.get("static_and_runtime_grounded", 0)),
            "state_obligation_runtime_grounded_only": int(self.grounding_counts.get("runtime_grounded_only", 0)),
            "state_obligation_insufficient_grounding": int(self.grounding_counts.get("insufficient_grounding", 0)),
            "state_obligation_repair_location_prefix": int(self.repair_location_counts.get("prefix", 0)),
            "state_obligation_repair_location_binding": int(self.repair_location_counts.get("binding", 0)),
            "state_obligation_repair_location_suffix": int(self.repair_location_counts.get("suffix", 0)),
            "state_obligation_repair_location_probe": int(self.repair_location_counts.get("probe", 0)),
            "state_obligation_repair_location_mixed": int(self.repair_location_counts.get("mixed", 0)),
            "state_obligation_repair_location_unknown": int(self.repair_location_counts.get("unknown", 0)),
            "state_obligations_jsonl": self.jsonl_path or None,
        }

    def write_summary(self) -> Dict[str, Any]:
        summary = self.build_summary()
        _write_json(self.summary_path, summary)
        return summary

    def _record(self, payload: Dict[str, Any]) -> None:
        self.counts["state_obligation_created"] += 1
        for kind in _clean_list(payload.get("obligation_kinds")):
            self.kind_counts[kind] += 1
        self.repair_location_counts[_text(payload.get("suspected_repair_location")).strip() or "unknown"] += 1
        self.grounding_counts[_text(payload.get("grounding_status")).strip() or "insufficient_grounding"] += 1
        _append_jsonl(self.jsonl_path, payload)
        self.write_summary()

    @staticmethod
    def _suspected_repair_location(
        *,
        checkpoint: str,
        failure_signature: str,
        tx_success: bool,
        reverted: bool,
        state_delta: bool,
        balance_delta: bool,
        observation_available: bool,
    ) -> str:
        if failure_signature in {"wrong_sender_binding", "wrong_beneficiary_binding", "wrong_receiver_binding", "wrong_amount_binding"}:
            return "binding"
        if failure_signature == "missing_prestate_prefix":
            return "prefix"
        if failure_signature in {"effect_function_not_reached", "suffix_not_effective"}:
            return "suffix"
        if checkpoint == "first_claim_success":
            if not tx_success or reverted:
                return "mixed"
            if not state_delta:
                return "mixed" if balance_delta else "suffix"
            if observation_available and state_delta and balance_delta:
                return "mixed"
            return "suffix"
        if failure_signature == "effect_probe_missing_or_uninformative" and observation_available:
            if checkpoint == "abnormal_balance_or_state_no_reset_observed":
                return "suffix"
            if checkpoint == "privileged_or_asset_effect_observed":
                return "mixed"
            return "probe"
        if checkpoint == "abnormal_balance_or_state_no_reset_observed":
            return "suffix"
        if checkpoint == "privileged_or_asset_effect_observed" and not tx_success:
            return "mixed"
        if reverted and not tx_success:
            return "mixed"
        if checkpoint == "privileged_or_asset_effect_observed":
            return "mixed" if (state_delta or balance_delta) else "suffix"
        if state_delta or balance_delta:
            return "probe" if observation_available else "suffix"
        return "unknown"

    @staticmethod
    def _repair_location_reason(
        *,
        checkpoint: str,
        failure_signature: str,
        repair_location: str,
        tx_success: bool,
        reverted: bool,
        state_delta: bool,
        balance_delta: bool,
    ) -> str:
        if repair_location == "binding":
            return "failure signature points to caller/beneficiary/receiver/amount relations"
        if repair_location == "prefix":
            return "failure signature points to missing prestate or setup facts"
        if repair_location == "suffix":
            if checkpoint == "abnormal_balance_or_state_no_reset_observed":
                return "missing checkpoint requires a repeated or downstream state-changing action"
            return "current path needs a state-changing target action or repeat action"
        if repair_location == "probe":
            return "runtime evidence suggests the effect may already exist but observation is incomplete"
        if repair_location == "mixed":
            if checkpoint == "first_claim_success":
                return "the path shows partial asset/state evidence, but a successful first transition still needs executable state progress"
            if checkpoint == "privileged_or_asset_effect_observed":
                return "the path needs both privileged-state reachability and an observable state or asset effect"
            if reverted and not tx_success:
                return "both precondition repair and state-changing action repair remain plausible"
            if state_delta or balance_delta:
                return "state/asset signals exist but transition success or relation facts are still incomplete"
        return "evidence is weak or split across binding, prefix, and suffix hypotheses"

    @staticmethod
    def _confidence(
        *,
        failure_signature: str,
        grounding_status: str,
        unmet_state_obligations: Sequence[str],
        static_present: bool,
    ) -> str:
        if grounding_status == "static_and_runtime_grounded" and failure_signature in {
            "missing_prestate_prefix",
            "effect_function_not_reached",
            "effect_reached_but_no_state_delta",
        }:
            return "high"
        if grounding_status != "insufficient_grounding" and unmet_state_obligations:
            return "medium"
        if static_present:
            return "medium"
        return "low"
