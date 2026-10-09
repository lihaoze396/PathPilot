# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _clean_list(value: Any) -> List[str]:
    output: List[str] = []
    if not isinstance(value, list):
        return output
    for item in value:
        text = _text(item).strip()
        if text:
            output.append(text)
    return output


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


class TFICandidateGapRepair:
    def __init__(
        self,
        abi: Optional[Any] = None,
        static_hints: Optional[Dict[str, Any]] = None,
        max_candidates_per_gap: int = 3,
    ):
        self.abi = abi
        self.static_hints = static_hints if isinstance(static_hints, dict) else {}
        self.max_candidates_per_gap = max(1, int(max_candidates_per_gap or 1))

    def build_guidance(
        self,
        gap_event: Dict[str, Any],
        agent_guidance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        guidance = {
            "guidance_type": "candidate_gap_repair_guidance",
            "scope_id": _text(gap_event.get("scope_id")).strip(),
            "gap_event_id": _text(gap_event.get("gap_event_id")).strip(),
            "hypothesis_id": _text(gap_event.get("hypothesis_id")).strip(),
            "target_function": _text(gap_event.get("target_function")).strip(),
            "frontier": _text(gap_event.get("frontier")).strip(),
            "sink": _text(gap_event.get("sink")).strip(),
            "preferred_candidate_roles": [
                "prefix_preserving_candidate",
                "frontier_touching_candidate",
                "state_setup_candidate",
                "binding_repair_candidate",
            ],
            "binding_hints": {
                "sender_policy_used": "keep_same_sender_across_sequence",
                "value_policy_used": "keep_observed_success_value",
                "args_policy_used": "abi_default_or_boundary_mix",
                "prestate_policy_used": "none",
            },
            "avoid_patterns": _clean_list(gap_event.get("forbidden_patterns")),
            "negative_evidence": _clean_list(gap_event.get("negative_evidence")),
            "do_not_generate_final_candidate": True,
            "source": "deterministic_fallback",
            "from_local_candidate_gap": True,
        }
        if isinstance(agent_guidance, dict):
            merged = dict(guidance)
            for key in ("preferred_candidate_roles", "avoid_patterns", "negative_evidence"):
                merged[key] = self._merge_unique_lists(merged.get(key), agent_guidance.get(key))
            binding_hints = dict(merged.get("binding_hints") or {})
            if isinstance(agent_guidance.get("binding_hints"), dict):
                for key, value in agent_guidance["binding_hints"].items():
                    text = _text(value).strip()
                    if text:
                        binding_hints[key] = text
            merged["binding_hints"] = binding_hints
            merged["source"] = _text(agent_guidance.get("source")).strip() or "agent_guidance"
            merged["agent_guidance"] = dict(agent_guidance)
            return merged
        return guidance

    def generate_candidates_from_gap(
        self,
        gap_event: Dict[str, Any],
        guidance: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(gap_event, dict):
            return []
        guidance_payload = guidance if isinstance(guidance, dict) else self.build_guidance(gap_event)

        observed_prefix = _clean_list(gap_event.get("observed_prefix"))
        target_function = _text(gap_event.get("target_function")).strip()
        frontier = _text(gap_event.get("frontier")).strip()
        sink = _text(gap_event.get("sink")).strip()
        scope_id = _text(gap_event.get("scope_id")).strip()
        hypothesis_id = _text(gap_event.get("hypothesis_id")).strip()
        gap_event_id = _text(gap_event.get("gap_event_id")).strip() or "TFI_GAP_EVT_{}_{}".format(
            _safe_token(scope_id),
            _safe_token(hypothesis_id),
        )

        candidate_sequences: List[Tuple[List[str], str]] = []
        candidate_sequences.extend(self._candidate_sequences(observed_prefix, frontier, sink, target_function))
        if not candidate_sequences and observed_prefix:
            candidate_sequences.append((list(observed_prefix), "observed_prefix_only"))

        unique_sequences = self._unique_sequences(candidate_sequences)
        candidates: List[Dict[str, Any]] = []
        for variant_index, (sequence, source_used) in enumerate(unique_sequences[: self.max_candidates_per_gap], start=1):
            if not sequence:
                continue
            candidate_id = "TFI_GAP_CAND_{:04d}_{}_{}_V{:02d}".format(
                variant_index,
                _safe_token(scope_id),
                _safe_token(gap_event_id),
                variant_index,
            )
            binding_hints = guidance_payload.get("binding_hints") if isinstance(guidance_payload.get("binding_hints"), dict) else {}
            candidate = {
                "candidate_id": candidate_id,
                "candidate_type": "tfi_guided_candidate_gap_repair",
                "candidate_origin": "tfi_guided_candidate_gap_repair",
                "generated_under_tfi_constraint": True,
                "candidate_source": "fuzzer_existing_pool_generation",
                "candidate_source_used": source_used,
                "source_task_id": gap_event_id,
                "scope_id": scope_id,
                "hypothesis_id": hypothesis_id,
                "target_function": target_function,
                "frontier": frontier,
                "sink": sink,
                "observed_prefix": list(observed_prefix),
                "candidate_sequence": list(sequence),
                "guided_sequence": list(sequence),
                "priority": "soft",
                "do_not_bypass_normal_selection": True,
                "gap_event_id": gap_event_id,
                "gap_type": _text(gap_event.get("gap_type")).strip() or "local_candidate_gap",
                "materialization_status": "ready",
                "is_executable_candidate": True,
                "materialization_warnings": [],
                "binding_choices": {
                    "sender_policy_used": _text(binding_hints.get("sender_policy_used")).strip() or "keep_same_sender_across_sequence",
                    "value_policy_used": _text(binding_hints.get("value_policy_used")).strip() or "keep_observed_success_value",
                    "args_policy_used": _text(binding_hints.get("args_policy_used")).strip() or "abi_default_or_boundary_mix",
                    "prestate_policy_used": _text(binding_hints.get("prestate_policy_used")).strip() or "none",
                },
                "candidate_constraints": {
                    "must_preserve_observed_prefix": bool(observed_prefix),
                    "must_touch_frontier_or_sink": True,
                    "must_not_use_nonlocal_generic_skeleton": True,
                    "must_not_select_target_seen_without_path_progress": True,
                },
                "metadata": {
                    "from_local_candidate_gap": True,
                    "agent_guided": bool(isinstance(guidance_payload, dict) and _text(guidance_payload.get("source")).strip() == "agent_guidance"),
                    "uses_existing_pool": True,
                    "new_pool_created": False,
                    "candidate_origin": "tfi_guided_candidate_gap_repair",
                    "scope_id": scope_id,
                    "target_function": target_function,
                    "frontier": frontier,
                    "sink": sink,
                    "do_not_bypass_normal_selection": True,
                    "guidance_type": _text(guidance_payload.get("guidance_type")).strip() or "candidate_gap_repair_guidance",
                },
                "candidate_gap_guidance": guidance_payload,
                "reason": "tfi_guided_candidate_gap_repair_from_local_candidate_gap",
            }
            candidates.append(candidate)
        return candidates

    def _candidate_sequences(
        self,
        observed_prefix: Sequence[str],
        frontier: str,
        sink: str,
        target_function: str,
    ) -> List[Tuple[List[str], str]]:
        sequences: List[Tuple[List[str], str]] = []
        base = list(observed_prefix)

        for anchors, label in (
            ([target_function, frontier, sink], "target_frontier_sink"),
            ([frontier, sink], "frontier_sink"),
            ([target_function, sink], "target_sink"),
            ([target_function, frontier], "target_frontier"),
        ):
            seq = list(base)
            for anchor in anchors:
                if anchor and anchor not in seq:
                    seq.append(anchor)
            if seq and seq != base:
                sequences.append((seq, "observed_prefix_plus_{}".format(label)))

        for anchor, label in ((frontier, "frontier"), (sink, "sink"), (target_function, "target_function")):
            if not anchor:
                continue
            seq = list(base)
            if anchor not in seq:
                seq.append(anchor)
            if seq:
                sequences.append((seq, "observed_prefix_plus_{}".format(label)))

        if not sequences and base:
            sequences.append((list(base), "observed_prefix_only"))
        return sequences

    def _unique_sequences(self, sequences: List[Tuple[List[str], str]]) -> List[Tuple[List[str], str]]:
        seen = set()
        output: List[Tuple[List[str], str]] = []
        for sequence, source_used in sequences:
            marker = json.dumps({"sequence": sequence}, ensure_ascii=True, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            output.append((sequence, source_used))
        return output

    def _merge_unique_lists(self, left: Any, right: Any) -> List[str]:
        values: List[str] = []
        for item in _clean_list(left if isinstance(left, list) else []):
            if item not in values:
                values.append(item)
        for item in _clean_list(right if isinstance(right, list) else []):
            if item not in values:
                values.append(item)
        return values

