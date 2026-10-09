# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _listify(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _sequence_of_candidate(candidate: Dict[str, Any]) -> List[str]:
    for field in ("sequence_skeleton", "candidate_sequence", "guided_sequence", "required_prefix"):
        value = candidate.get(field)
        if isinstance(value, list):
            output = []
            for item in value:
                text = _text(item).strip()
                if text:
                    output.append(text)
            if output:
                return output
    return []


def _is_checkpoint_guided_candidate(candidate: Dict[str, Any]) -> bool:
    if _text(candidate.get("candidate_origin")).strip() == "checkpoint_guided_injection_v2":
        return True
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    return bool(metadata.get("checkpoint_guided"))


def _is_failure_repair_candidate(candidate: Dict[str, Any]) -> bool:
    if bool(candidate.get("failure_repair_candidate")):
        return True
    candidate_type = _text(candidate.get("candidate_type")).strip().lower()
    source_kind = _text(candidate.get("source_candidate_kind")).strip().lower()
    repair_action = _text(candidate.get("repair_action")).strip().lower()
    return "failure_repair" in candidate_type or "failure_repair" in source_kind or bool(repair_action)


class TFISoftInjectionBuffer:
    def __init__(
        self,
        max_size: int = 64,
        max_per_round: int = 3,
        min_useful_score: int = 1,
        confidence_threshold: float = 0.0,
    ):
        self.max_size = max(1, int(max_size or 1))
        self.max_per_round = max(1, int(max_per_round or 1))
        self.min_useful_score = max(1, int(min_useful_score or 1))
        try:
            self.confidence_threshold = max(0.0, float(confidence_threshold or 0.0))
        except Exception:
            self.confidence_threshold = 0.0
        self._items: List[Dict[str, Any]] = []
        self._signatures = set()
        self.duplicate_skipped_count = 0
        self.last_pop_duplicate_count = 0

    def add_candidates(self, candidates: List[Dict[str, Any]]) -> int:
        added = 0
        duplicate_count = 0
        for item in candidates:
            if not isinstance(item, dict):
                continue
            prepared = self._prepare_candidate(item)
            if prepared is None:
                continue
            signature = prepared.pop("_signature", None)
            if not signature:
                continue
            if signature in self._signatures:
                duplicate_count += 1
                continue
            self._signatures.add(signature)
            self._items.append(prepared)
            added += 1
        self.duplicate_skipped_count += duplicate_count
        self.last_pop_duplicate_count = duplicate_count
        self._items.sort(key=self._sort_key)
        self._trim()
        return added

    def pop_for_round(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        count = max(0, int(limit if limit is not None else self.max_per_round))
        if count <= 0 or not self._items:
            self.last_pop_duplicate_count = 0
            return []
        selected = self._items[:count]
        self._items = self._items[count:]
        for item in selected:
            signature = _text(item.get("injection_signature")).strip()
            if signature and signature in self._signatures:
                self._signatures.remove(signature)
        self.last_pop_duplicate_count = 0
        return [dict(item) for item in selected]

    def __len__(self) -> int:
        return len(self._items)

    def _prepare_candidate(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not bool(candidate.get("useful")):
            return None
        if _text(candidate.get("execution_status")).strip() == "skipped":
            return None
        if _text(candidate.get("useful_reason")).strip() == "target_seen_without_path_progress":
            return None
        if _is_checkpoint_guided_candidate(candidate):
            status = _text(candidate.get("materialization_status")).strip()
            if status == "rejected_as_unresolved":
                return None
        if not self._passes_useful_gate(candidate):
            return None
        if not self._passes_confidence_gate(candidate):
            return None
        useful_score = self._useful_score(candidate)
        if useful_score < self.min_useful_score:
            return None
        candidate_id = _text(candidate.get("candidate_id")).strip()
        source_task_id = _text(candidate.get("source_task_id")).strip()
        scope_id = _text(candidate.get("scope_id")).strip()
        if not candidate_id or not source_task_id or not scope_id:
            return None
        signature = self._build_signature(candidate)
        if not signature:
            return None

        prepared = dict(candidate)
        prepared["_signature"] = signature
        prepared["injection_signature"] = signature
        prepared["injection_source"] = "tfi_isolated_pilot"
        prepared["injection_mode"] = "soft"
        prepared["priority"] = "checkpoint" if _is_checkpoint_guided_candidate(candidate) else "low"
        prepared["do_not_bypass_normal_selection"] = True
        prepared["useful_score"] = useful_score
        if "generated_under_tfi_constraint" not in prepared:
            prepared["generated_under_tfi_constraint"] = bool(
                prepared.get("metadata", {}).get("from_local_candidate_gap")
            ) if isinstance(prepared.get("metadata"), dict) else False
        return prepared

    def _passes_confidence_gate(self, candidate: Dict[str, Any]) -> bool:
        if self.confidence_threshold <= 0:
            return True
        for field in ("confidence", "candidate_confidence", "selection_confidence"):
            value = candidate.get(field)
            if value in (None, ""):
                continue
            try:
                return float(value) >= self.confidence_threshold
            except Exception:
                return True
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        value = metadata.get("confidence") if isinstance(metadata, dict) else None
        if value in (None, ""):
            return True
        try:
            return float(value) >= self.confidence_threshold
        except Exception:
            return True

    def _passes_useful_gate(self, candidate: Dict[str, Any]) -> bool:
        if _is_checkpoint_guided_candidate(candidate):
            return bool(_sequence_of_candidate(candidate)) and bool(candidate.get("target_missing_checkpoint"))
        path_progress_delta = _safe_int(candidate.get("path_progress_delta"), 0)
        frontier_touched = bool(candidate.get("frontier_touched"))
        sink_touched = bool(candidate.get("sink_touched"))
        sink_reverted = bool(candidate.get("sink_reverted"))
        coverage_delta = _safe_int(candidate.get("coverage_delta"), 0)
        useful_reason = _text(candidate.get("useful_reason")).strip()
        if path_progress_delta > 0:
            return True
        if frontier_touched:
            return True
        if sink_touched and not sink_reverted:
            return True
        if useful_reason == "target_related_coverage_delta" and coverage_delta > 0:
            return True
        return False

    def _useful_score(self, candidate: Dict[str, Any]) -> int:
        if _is_checkpoint_guided_candidate(candidate):
            score = 3
            if _is_failure_repair_candidate(candidate):
                score += 3
            if bool(candidate.get("sink_touched")):
                score += 1
            modules_used = _listify(candidate.get("modules_used"))
            if "prefix_replay" in modules_used:
                score += 1
            if "effect_probe" in modules_used:
                score += 1
            if "strong_useful_scheduling" in modules_used:
                score += 1
            return score
        score = 0
        if _safe_int(candidate.get("path_progress_delta"), 0) > 0:
            score += 2
        if bool(candidate.get("frontier_touched")):
            score += 1
        if bool(candidate.get("sink_touched")) and not bool(candidate.get("sink_reverted")):
            score += 1
        if _text(candidate.get("useful_reason")).strip() == "target_related_coverage_delta" and _safe_int(candidate.get("coverage_delta"), 0) > 0:
            score += 1
        return score

    def _build_signature(self, candidate: Dict[str, Any]) -> str:
        sequence = _sequence_of_candidate(candidate)
        payload = {
            "candidate_type": _text(candidate.get("candidate_type")).strip(),
            "scope_id": _text(candidate.get("scope_id")).strip(),
            "source_task_id": _text(candidate.get("source_task_id")).strip(),
            "sequence": sequence,
            "target_function": _text(candidate.get("target_function")).strip(),
            "frontier": _text(candidate.get("frontier")).strip(),
            "sink": _text(candidate.get("sink")).strip(),
        }
        if not payload["candidate_type"] or not payload["scope_id"] or not payload["source_task_id"]:
            return ""
        if not sequence:
            return ""
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)

    def _sort_key(self, candidate: Dict[str, Any]) -> Tuple[int, int, int, str]:
        useful_score = _safe_int(candidate.get("useful_score"), 0)
        path_progress = _safe_int(candidate.get("path_progress_delta"), 0)
        repair_rank = 0 if _is_failure_repair_candidate(candidate) else 1
        candidate_id = _text(candidate.get("candidate_id")).strip()
        return (-useful_score, repair_rank, -path_progress, candidate_id)

    def _trim(self) -> None:
        if len(self._items) <= self.max_size:
            return
        overflow = self._items[self.max_size :]
        self._items = self._items[: self.max_size]
        for item in overflow:
            signature = _text(item.get("injection_signature")).strip()
            if signature and signature in self._signatures:
                self._signatures.remove(signature)

