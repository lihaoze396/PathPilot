# -*- coding: utf-8 -*-

"""Dependency-grounded repair scope construction.

This module does not generate concrete transactions. It derives a constrained
repair scope from obligation/runtime/static evidence and preserves a safe
fallback when no grounded semantic action exists.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _is_unpaired_owner_accept(action: str, prior_actions: Sequence[str]) -> bool:
    lowered = _text(action).strip().lower()
    if lowered not in {"acceptownership", "claimownership"}:
        return False
    prior_text = " ".join(_text(item).strip().lower() for item in prior_actions)
    return not any(token in prior_text for token in ("transferownership", "nominateowner", "setpendingowner"))


def _safe_token(value: Any) -> str:
    text = _text(value).strip()
    if not text:
        return "NONE"
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_") or "NONE"


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


class DependencyGroundedRepairScopeBuilder:
    def __init__(self, output_dir: str = "") -> None:
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.jsonl_path = os.path.join(self.output_dir, "dependency_grounded_scopes.jsonl") if self.output_dir else ""
        self.summary_path = os.path.join(self.output_dir, "dependency_grounded_scope_summary.json") if self.output_dir else ""
        self.counts = Counter()
        self.grounding_counts = Counter()
        if self.jsonl_path:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.jsonl_path, "w", encoding="utf-8"):
                pass

    def build(
        self,
        *,
        state_obligation: Dict[str, Any],
        parent_candidate: Optional[Dict[str, Any]] = None,
        runtime_observations: Optional[Dict[str, Any]] = None,
        successful_prefix_evidence: Optional[Dict[str, Any]] = None,
        static_hints: Optional[Dict[str, Any]] = None,
        abi_functions: Optional[Sequence[str]] = None,
        abi_view_functions: Optional[Sequence[str]] = None,
        prior_repair_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state_obligation = state_obligation if isinstance(state_obligation, dict) else {}
        parent_candidate = parent_candidate if isinstance(parent_candidate, dict) else {}
        runtime_observations = runtime_observations if isinstance(runtime_observations, dict) else {}
        successful_prefix_evidence = successful_prefix_evidence if isinstance(successful_prefix_evidence, dict) else {}
        static_hints = static_hints if isinstance(static_hints, dict) else {}
        prior_repair_evidence = prior_repair_evidence if isinstance(prior_repair_evidence, dict) else {}
        abi_names = _dedup_text(list(abi_functions or []))
        view_names = set(_clean_list(list(abi_view_functions or [])))

        parent_id = _text(parent_candidate.get("candidate_id")).strip()
        obligation_kinds = set(_clean_list(state_obligation.get("obligation_kinds")))
        required_prefix = _clean_list(parent_candidate.get("required_prefix"))
        suffix_actions = _clean_list(parent_candidate.get("suffix_actions"))
        parent_sequence = _clean_list(parent_candidate.get("guided_sequence") or parent_candidate.get("function_sequence"))
        reached_functions = _dedup_text(
            _clean_list(runtime_observations.get("functions_reached"))
            + _clean_list(successful_prefix_evidence.get("functions_reached"))
        )
        state_effect_functions = _dedup_text(runtime_observations.get("state_effect_functions") or [])
        sink_functions = _dedup_text(runtime_observations.get("sink_functions") or [])
        successful_prefix_functions = _dedup_text(
            _clean_list(successful_prefix_evidence.get("required_prefix"))
            + _clean_list(successful_prefix_evidence.get("successful_prefix_functions"))
            + required_prefix
        )
        repeated_functions = _dedup_text(runtime_observations.get("repeated_functions") or [])
        static_functions = static_hints.get("functions") if isinstance(static_hints.get("functions"), list) else []
        static_storage = static_hints.get("storage_dependencies") if isinstance(static_hints.get("storage_dependencies"), list) else []
        static_dangerous = static_hints.get("dangerous_operations") if isinstance(static_hints.get("dangerous_operations"), list) else []
        static_privileged = set(_clean_list(static_hints.get("privileged_functions")))

        must_preserve_prefix = _dedup_text(required_prefix or successful_prefix_functions)
        must_preserve_relations = self._must_preserve_relations(state_obligation, parent_candidate, runtime_observations)
        mutable_binding_fields = self._mutable_binding_fields(state_obligation, runtime_observations, parent_candidate)

        candidate_prefix_actions = self._candidate_prefix_actions(
            must_preserve_prefix=must_preserve_prefix,
            reached_functions=reached_functions,
            state_effect_functions=state_effect_functions,
            abi_names=abi_names,
            view_names=view_names,
        )
        candidate_state_writer_actions = self._candidate_state_writers(
            parent_sequence=parent_sequence,
            reached_functions=reached_functions,
            state_effect_functions=state_effect_functions,
            static_storage=static_storage,
            static_functions=static_functions,
            static_privileged=static_privileged,
            obligation_kinds=obligation_kinds,
            abi_names=abi_names,
            view_names=view_names,
        )
        candidate_sink_actions = self._candidate_sink_actions(
            parent_sequence=parent_sequence,
            reached_functions=reached_functions,
            sink_functions=sink_functions,
            static_dangerous=static_dangerous,
            static_privileged=static_privileged,
            obligation_kinds=obligation_kinds,
            abi_names=abi_names,
            view_names=view_names,
        )
        candidate_repeat_actions = self._candidate_repeat_actions(
            suffix_actions=suffix_actions,
            parent_sequence=parent_sequence,
            repeated_functions=repeated_functions,
            state_effect_functions=state_effect_functions,
            sink_functions=sink_functions,
            obligation_kinds=obligation_kinds,
            view_names=view_names,
        )
        candidate_observation_actions = self._candidate_observation_actions(
            parent_candidate=parent_candidate,
            view_names=view_names,
            runtime_observations=runtime_observations,
        )

        has_static = bool(static_hints.get("loaded")) and bool(static_storage or static_dangerous or static_privileged)
        has_runtime = bool(reached_functions or state_effect_functions or sink_functions)
        grounding_status = "insufficient_grounding"
        if has_static and has_runtime:
            grounding_status = "static_and_runtime_grounded"
        elif has_runtime:
            grounding_status = "runtime_grounded_only"

        no_safe_semantic_scope = False
        no_safe_scope_reason = None
        if "observe_already_produced_effect" not in obligation_kinds:
            semantic_lists = [
                candidate_state_writer_actions,
                candidate_sink_actions,
                candidate_repeat_actions,
            ]
            if not any(semantic_lists):
                no_safe_semantic_scope = True
                if not has_runtime and not has_static:
                    no_safe_scope_reason = "no_runtime_or_static_dependency_evidence"
                elif prior_repair_evidence.get("observation_only_repair"):
                    no_safe_scope_reason = "only_observation_scope_available"
                else:
                    no_safe_scope_reason = "no_state_changing_or_sink_action_grounded"

        verification_evidence_required = _dedup_text(
            list(state_obligation.get("required_effect_evidence") or [])
            + list(runtime_observations.get("required_verification_evidence") or [])
        )
        scope_confidence = self._scope_confidence(
            grounding_status=grounding_status,
            no_safe_semantic_scope=no_safe_semantic_scope,
            semantic_action_count=(
                len(candidate_state_writer_actions)
                + len(candidate_sink_actions)
                + len(candidate_repeat_actions)
            ),
            observation_action_count=len(candidate_observation_actions),
        )

        scope_id = "SCOPE_{}_{}_{}".format(
            _safe_token(state_obligation.get("bench_id")),
            _safe_token(parent_id or state_obligation.get("obligation_id")),
            max(1, int(self.counts.get("repair_scope_created", 0) + 1)),
        )
        payload = {
            "repair_scope_id": scope_id,
            "obligation_id": _text(state_obligation.get("obligation_id")).strip(),
            "parent_candidate_id": parent_id or None,
            "grounding_status": grounding_status,
            "must_preserve_prefix": must_preserve_prefix,
            "must_preserve_relations": must_preserve_relations,
            "mutable_binding_fields": mutable_binding_fields,
            "candidate_prefix_actions": candidate_prefix_actions,
            "candidate_state_writer_actions": candidate_state_writer_actions,
            "candidate_sink_actions": candidate_sink_actions,
            "candidate_repeat_actions": candidate_repeat_actions,
            "candidate_observation_actions": candidate_observation_actions,
            "forbidden_unscoped_actions": True,
            "verification_evidence_required": verification_evidence_required,
            "scope_confidence": scope_confidence,
            "no_safe_semantic_scope": bool(no_safe_semantic_scope),
            "no_safe_scope_reason": no_safe_scope_reason,
        }
        self._record(payload)
        return payload

    def build_summary(self) -> Dict[str, Any]:
        return {
            "repair_scope_created": int(self.counts.get("repair_scope_created", 0)),
            "repair_scope_static_and_runtime_grounded": int(self.grounding_counts.get("static_and_runtime_grounded", 0)),
            "repair_scope_runtime_grounded_only": int(self.grounding_counts.get("runtime_grounded_only", 0)),
            "repair_scope_insufficient_grounding": int(self.grounding_counts.get("insufficient_grounding", 0)),
            "repair_scope_no_safe_semantic_scope": int(self.counts.get("repair_scope_no_safe_semantic_scope", 0)),
            "repair_scope_candidate_state_writer_count": int(self.counts.get("repair_scope_candidate_state_writer_count", 0)),
            "repair_scope_candidate_sink_action_count": int(self.counts.get("repair_scope_candidate_sink_action_count", 0)),
            "repair_scope_candidate_repeat_action_count": int(self.counts.get("repair_scope_candidate_repeat_action_count", 0)),
            "repair_scope_candidate_observation_action_count": int(self.counts.get("repair_scope_candidate_observation_action_count", 0)),
            "dependency_grounded_scopes_jsonl": self.jsonl_path or None,
        }

    def write_summary(self) -> Dict[str, Any]:
        summary = self.build_summary()
        _write_json(self.summary_path, summary)
        return summary

    def _record(self, payload: Dict[str, Any]) -> None:
        self.counts["repair_scope_created"] += 1
        grounding = _text(payload.get("grounding_status")).strip() or "insufficient_grounding"
        self.grounding_counts[grounding] += 1
        if bool(payload.get("no_safe_semantic_scope")):
            self.counts["repair_scope_no_safe_semantic_scope"] += 1
        self.counts["repair_scope_candidate_state_writer_count"] += len(payload.get("candidate_state_writer_actions") or [])
        self.counts["repair_scope_candidate_sink_action_count"] += len(payload.get("candidate_sink_actions") or [])
        self.counts["repair_scope_candidate_repeat_action_count"] += len(payload.get("candidate_repeat_actions") or [])
        self.counts["repair_scope_candidate_observation_action_count"] += len(payload.get("candidate_observation_actions") or [])
        _append_jsonl(self.jsonl_path, payload)
        self.write_summary()

    @staticmethod
    def _action_row(
        action: str,
        role: str,
        grounding_sources: Sequence[Any],
        related_state_or_sink: Sequence[Any],
        related_binding_requirements: Sequence[Any],
        confidence: str,
    ) -> Dict[str, Any]:
        return {
            "action": _text(action).strip(),
            "action_role": role,
            "grounding_sources": _dedup_text(grounding_sources),
            "related_state_or_sink": _dedup_text(related_state_or_sink),
            "related_binding_requirements": _dedup_text(related_binding_requirements),
            "confidence": confidence,
        }

    @staticmethod
    def _must_preserve_relations(
        obligation: Dict[str, Any],
        parent_candidate: Dict[str, Any],
        runtime_observations: Dict[str, Any],
    ) -> List[str]:
        output: List[str] = []
        if "preserve_or_repeat_relation" in set(_clean_list(obligation.get("obligation_kinds"))):
            output.extend(["same_sender_if_already_successful", "same_amount_if_repeated_effect_is_required"])
        if _text(parent_candidate.get("beneficiary_binding")).strip():
            output.append("beneficiary_binding_if_already_observed")
        if bool(runtime_observations.get("same_sender_relation_observed")):
            output.append("same_sender_relation_observed")
        if bool(runtime_observations.get("same_beneficiary_relation_observed")):
            output.append("same_beneficiary_relation_observed")
        if bool(runtime_observations.get("same_amount_relation_observed")):
            output.append("same_amount_relation_observed")
        return _dedup_text(output)

    @staticmethod
    def _mutable_binding_fields(
        obligation: Dict[str, Any],
        runtime_observations: Dict[str, Any],
        parent_candidate: Dict[str, Any],
    ) -> List[str]:
        output: List[str] = []
        location = _text(obligation.get("suspected_repair_location")).strip()
        if location in {"binding", "mixed", "unknown"}:
            output.extend(["sender_binding", "beneficiary_binding", "receiver_binding", "amount_binding", "arg_binding"])
        if location in {"prefix", "mixed"}:
            output.append("prefix")
        if location in {"suffix", "mixed"}:
            output.append("suffix_actions")
        if "observe_already_produced_effect" in set(_clean_list(obligation.get("obligation_kinds"))):
            output.append("effect_probe")
        if runtime_observations.get("target_suffix_reached") and not _text(parent_candidate.get("receiver_binding")).strip():
            output.append("receiver_binding")
        return _dedup_text(output)

    def _candidate_prefix_actions(
        self,
        *,
        must_preserve_prefix: Sequence[str],
        reached_functions: Sequence[str],
        state_effect_functions: Sequence[str],
        abi_names: Sequence[str],
        view_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        accepted_prefix: List[str] = []
        for action in _dedup_text(list(must_preserve_prefix) + list(reached_functions)):
            if action not in abi_names or action in view_names:
                continue
            if _is_unpaired_owner_accept(action, accepted_prefix):
                continue
            confidence = "high" if action in must_preserve_prefix else "medium"
            actions.append(
                self._action_row(
                    action,
                    "prefix_builder",
                    ["runtime_successful_prefix"],
                    ["prefix_preservation"],
                    ["preserve_prefix"],
                    confidence,
                )
            )
            accepted_prefix.append(action)
            if action in state_effect_functions:
                actions[-1]["grounding_sources"] = _dedup_text(actions[-1]["grounding_sources"] + ["runtime_related_state_action"])
        return actions[:8]

    def _candidate_state_writers(
        self,
        *,
        parent_sequence: Sequence[str],
        reached_functions: Sequence[str],
        state_effect_functions: Sequence[str],
        static_storage: Sequence[Any],
        static_functions: Sequence[Any],
        static_privileged: Sequence[str],
        obligation_kinds: Sequence[str],
        abi_names: Sequence[str],
        view_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        seen = set()
        relevant_runtime = _dedup_text(list(state_effect_functions) + list(parent_sequence) + list(reached_functions))
        for action in relevant_runtime:
            if action not in abi_names or action in view_names or action in seen:
                continue
            grounding = []
            if action in state_effect_functions:
                grounding.append("runtime_related_state_action")
            if action in static_privileged:
                grounding.append("static_permission_dependency")
            storage_targets = self._storage_targets_for_action(action, static_storage)
            if storage_targets:
                grounding.append("static_write_dependency")
            if "runtime_related_state_action" not in grounding and "static_write_dependency" not in grounding:
                if not (
                    "produce_target_state_write" in obligation_kinds
                    and "static_permission_dependency" in grounding
                ):
                    continue
            if not grounding:
                continue
            seen.add(action)
            actions.append(
                self._action_row(
                    action,
                    "state_writer",
                    grounding,
                    storage_targets or ["state_write"],
                    ["preserve_prefix"] if action in parent_sequence else [],
                    "high" if "runtime_related_state_action" in grounding else "medium",
                )
            )

        if not actions and "produce_target_state_write" in obligation_kinds:
            for item in static_storage:
                if not isinstance(item, dict):
                    continue
                action = _text(item.get("function")).strip()
                if not action or action in seen or action not in abi_names or action in view_names:
                    continue
                if action not in static_privileged and "produce_target_state_write" not in obligation_kinds:
                    continue
                seen.add(action)
                actions.append(
                    self._action_row(
                        action,
                        "state_writer",
                        ["static_write_dependency", "static_permission_dependency"] if action in static_privileged else ["static_write_dependency"],
                        _clean_list(item.get("writes")) or ["state_write"],
                        ["unexpected_sender_path"],
                        "medium",
                    )
                )
                if len(actions) >= 6:
                    break
        return actions[:10]

    def _candidate_sink_actions(
        self,
        *,
        parent_sequence: Sequence[str],
        reached_functions: Sequence[str],
        sink_functions: Sequence[str],
        static_dangerous: Sequence[Any],
        static_privileged: Sequence[str],
        obligation_kinds: Sequence[str],
        abi_names: Sequence[str],
        view_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        seen = set()
        for action in _dedup_text(list(sink_functions)):
            if action not in abi_names or action in view_names or action in seen:
                continue
            sources = []
            if action in sink_functions:
                sources.append("runtime_related_state_action")
            dangerous = self._dangerous_targets_for_action(action, static_dangerous)
            if dangerous:
                sources.append("static_sink_dependency")
            if action in static_privileged:
                sources.append("static_permission_dependency")
            if not sources:
                continue
            seen.add(action)
            actions.append(
                self._action_row(
                    action,
                    "sink_action",
                    sources,
                    dangerous or ["asset_or_external_sink"],
                    ["unexpected_sender_path"] if action in static_privileged else [],
                    "high" if action in sink_functions else "medium",
                )
            )
        if not actions and "produce_asset_or_sink_effect" in obligation_kinds:
            for item in static_dangerous:
                if not isinstance(item, dict):
                    continue
                action = _text(item.get("function")).strip()
                if not action or action in seen or action not in abi_names or action in view_names:
                    continue
                seen.add(action)
                actions.append(
                    self._action_row(
                        action,
                        "sink_action",
                        ["static_sink_dependency"] + (["static_permission_dependency"] if action in static_privileged else []),
                        _clean_list(item.get("signals")) + _clean_list(item.get("external_calls")),
                        ["unexpected_sender_path"] if action in static_privileged else [],
                        "medium",
                    )
                )
                if len(actions) >= 6:
                    break
        return actions[:10]

    def _candidate_repeat_actions(
        self,
        *,
        suffix_actions: Sequence[str],
        parent_sequence: Sequence[str],
        repeated_functions: Sequence[str],
        state_effect_functions: Sequence[str],
        sink_functions: Sequence[str],
        obligation_kinds: Sequence[str],
        view_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        if "preserve_or_repeat_relation" not in obligation_kinds:
            return actions
        seen = set()
        effectful_suffix = [
            action
            for action in _dedup_text(list(suffix_actions) + list(parent_sequence[-2:]))
            if action in set(_dedup_text(list(state_effect_functions) + list(sink_functions)))
        ]
        repeat_candidates = _dedup_text(list(repeated_functions) + effectful_suffix)
        for action in repeat_candidates:
            if action in seen or action in view_names:
                continue
            sources = []
            if action in repeated_functions:
                sources.append("runtime_related_state_action")
            if action in state_effect_functions:
                sources.append("runtime_related_state_action")
            if action in suffix_actions or action in parent_sequence:
                sources.append("runtime_successful_prefix")
            if action in sink_functions:
                sources.append("static_sink_dependency")
            if not sources:
                continue
            seen.add(action)
            actions.append(
                self._action_row(
                    action,
                    "repeat_action",
                    sources,
                    ["repeat_relation"],
                    ["same_sender_if_already_successful", "same_amount_if_repeated_effect_is_required"],
                    "high" if action in repeated_functions else "medium",
                )
            )
        return actions[:8]

    def _candidate_observation_actions(
        self,
        *,
        parent_candidate: Dict[str, Any],
        view_names: Sequence[str],
        runtime_observations: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        seen = set()
        for probe in parent_candidate.get("effect_probe_plan") or []:
            if not isinstance(probe, dict):
                continue
            function = _text(probe.get("function")).strip()
            if not function or function in seen:
                continue
            if function not in view_names and function != "native_balance":
                continue
            seen.add(function)
            actions.append(
                self._action_row(
                    function,
                    "observation",
                    ["runtime_successful_prefix" if function in view_names else "runtime_related_state_action"],
                    [_text(probe.get("probe_type")).strip() or "effect_probe"],
                    [],
                    "medium",
                )
            )
        for function in _clean_list(runtime_observations.get("observation_only_functions")):
            if function in seen:
                continue
            seen.add(function)
            actions.append(
                self._action_row(
                    function,
                    "observation",
                    ["runtime_related_state_action"],
                    ["observation_only"],
                    [],
                    "low",
                )
            )
        return actions[:10]

    @staticmethod
    def _storage_targets_for_action(action: str, static_storage: Sequence[Any]) -> List[str]:
        for item in static_storage:
            if not isinstance(item, dict):
                continue
            if _text(item.get("function")).strip() == action:
                return _dedup_text(list(item.get("writes") or []) + list(item.get("reads") or []))[:8]
        return []

    @staticmethod
    def _dangerous_targets_for_action(action: str, static_dangerous: Sequence[Any]) -> List[str]:
        for item in static_dangerous:
            if not isinstance(item, dict):
                continue
            if _text(item.get("function")).strip() == action:
                return _dedup_text(list(item.get("signals") or []) + list(item.get("external_calls") or []))[:8]
        return []

    @staticmethod
    def _scope_confidence(
        *,
        grounding_status: str,
        no_safe_semantic_scope: bool,
        semantic_action_count: int,
        observation_action_count: int,
    ) -> str:
        if no_safe_semantic_scope and observation_action_count <= 0:
            return "low"
        if grounding_status == "static_and_runtime_grounded" and semantic_action_count > 0:
            return "high"
        if grounding_status != "insufficient_grounding" and (semantic_action_count > 0 or observation_action_count > 0):
            return "medium"
        return "low"

