#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional


class StageType(str, Enum):
    GEI = "GEI"
    KSI = "KSI"


@dataclass
class StageRuntimeConfig:
    enable_stage: bool = False
    force_gei_only: bool = False
    max_stage: str = "ksi"
    stage_log_path: Optional[str] = None
    min_gei_generations: int = 2
    stagnation_window: int = 2
    min_hard_gain_delta: int = 1
    enable_snapshot_pool: bool = False
    snapshot_pool_capacity: int = 32


@dataclass
class SnapshotCandidate:
    snapshot_id: str
    generation: int
    individual_id: str
    sequence_length: int
    code_coverage_count: int
    code_delta: int
    revert_count: int
    invalid_count: int
    stage: str
    reason: str = "hard_gain"


class StageRuntime:
    def __init__(self, config: Optional[StageRuntimeConfig] = None) -> None:
        self.config = config or StageRuntimeConfig()
        self.stage = StageType.GEI
        self.stage_counts: Dict[str, int] = {StageType.GEI.value: 0, StageType.KSI.value: 0}
        self.recent_feedback: Deque[Dict[str, Any]] = deque(
            maxlen=max(int(self.config.stagnation_window or 1), 1)
        )
        self.transitions: List[Dict[str, Any]] = []
        self.transition_reasons: List[str] = []
        self.log_records: List[Dict[str, Any]] = []
        self.snapshot_candidates: Deque[SnapshotCandidate] = deque(
            maxlen=max(int(self.config.snapshot_pool_capacity or 1), 1)
        )
        self.snapshot_reasons: Dict[str, int] = {}
        self.snapshot_sequence = 0
        self.ksi_entry_generation: Optional[int] = None
        self.ksi_transition_reason: Optional[str] = None
        self.latest_ksi_observation: Dict[str, Any] = {}
        self._log_path: Optional[str] = None
        self._jsonl_mode = False
        self._prepare_log()

    def before_individual(self, generation: int, individual_id: Optional[str] = None) -> str:
        if self.config.force_gei_only:
            self.stage = StageType.GEI
        return self.stage.value

    def after_individual(self, generation: int, individual_id: Any, feedback: Dict[str, Any]) -> None:
        current_stage = self.stage
        normalized = self._normalize_feedback(feedback)
        normalized["hard_gain"] = bool(normalized["code_delta"] > 0)
        self.stage_counts[current_stage.value] = self.stage_counts.get(current_stage.value, 0) + 1
        self.recent_feedback.append(normalized)

        snapshot = self._maybe_record_snapshot(
            generation=generation,
            individual_id=str(individual_id),
            stage=current_stage.value,
            feedback=normalized,
        )
        transition, transition_reason = self._maybe_transition(generation)

        if current_stage is StageType.KSI:
            self.latest_ksi_observation = {
                "generation": int(generation),
                "transition_reason": self.ksi_transition_reason,
                "recent_code_delta": self._recent_sum("code_delta"),
                "recent_revert_count": self._recent_sum("revert_count"),
                "recent_invalid_count": self._recent_sum("invalid_count"),
                "snapshot_candidate_count": len(self.snapshot_candidates),
                "current_hard_gain": normalized["hard_gain"],
            }

        record = {
            "generation": int(generation),
            "individual_id": str(individual_id),
            "stage": current_stage.value,
            "code_coverage_count": normalized["code_coverage_count"],
            "code_delta": normalized["code_delta"],
            "hard_gain": normalized["hard_gain"],
            "revert_count": normalized["revert_count"],
            "invalid_count": normalized["invalid_count"],
            "tx_count": normalized["tx_count"],
            "transition": transition,
            "transition_reason": transition_reason,
            "snapshot_saved": snapshot is not None,
            "snapshot_id": snapshot.snapshot_id if snapshot else None,
            "snapshot_reason": snapshot.reason if snapshot else None,
            "recent_code_delta": self._recent_sum("code_delta"),
            "snapshot_candidate_count": len(self.snapshot_candidates),
        }
        self._write_log_record(record)

    def summary(self) -> Dict[str, Any]:
        return {
            "engine": "search-stage",
            "stage_enabled": bool(self.config.enable_stage),
            "stage_counts": dict(self.stage_counts),
            "gei_rounds": int(self.stage_counts.get(StageType.GEI.value, 0)),
            "ksi_rounds": int(self.stage_counts.get(StageType.KSI.value, 0)),
            "tfi_rounds": 0,
            "force_gei_only": bool(self.config.force_gei_only),
            "stage_transition_count": len(self.transitions),
            "stage_transition_reasons": list(self.transition_reasons),
            "ksi_entry_generation": self.ksi_entry_generation,
            "ksi_transition_reason": self.ksi_transition_reason,
            "ksi_observation": dict(self.latest_ksi_observation),
            "snapshot_pool_enabled": bool(self.config.enable_snapshot_pool),
            "snapshot_candidate_count": len(self.snapshot_candidates),
            "snapshot_pool_size": len(self.snapshot_candidates),
            "snapshot_reasons": dict(self.snapshot_reasons),
            "snapshot_candidates": [asdict(candidate) for candidate in self.snapshot_candidates],
            "business_template_enabled": False,
            "agent_enabled": False,
            "rag_enabled": False,
        }

    @staticmethod
    def disabled_summary() -> Dict[str, Any]:
        return {
            "engine": "search-stage",
            "stage_enabled": False,
            "stage_counts": {},
            "gei_rounds": 0,
            "ksi_rounds": 0,
            "tfi_rounds": 0,
            "force_gei_only": False,
            "stage_transition_count": 0,
            "stage_transition_reasons": [],
            "snapshot_pool_enabled": False,
            "snapshot_candidate_count": 0,
            "snapshot_pool_size": 0,
            "snapshot_reasons": {},
            "business_template_enabled": False,
            "agent_enabled": False,
            "rag_enabled": False,
        }

    def _normalize_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "code_coverage_count": self._as_int(feedback.get("code_coverage_count")),
            "code_coverage_total": self._as_int(feedback.get("code_coverage_total")),
            "code_delta": self._as_int(feedback.get("code_delta")),
            "hard_gain": bool(feedback.get("hard_gain", False)),
            "revert_count": self._as_int(feedback.get("revert_count")),
            "invalid_count": self._as_int(feedback.get("invalid_count")),
            "tx_count": self._as_int(feedback.get("tx_count")),
            "sequence_length": self._as_int(feedback.get("sequence_length")),
        }

    def _maybe_transition(self, generation: int) -> tuple[Optional[str], Optional[str]]:
        if self.stage is not StageType.GEI:
            return None, None
        if self.config.force_gei_only:
            return None, None
        if str(self.config.max_stage or "ksi").lower() == "gei":
            return None, None
        if int(generation) < int(self.config.min_gei_generations):
            return None, None
        if len(self.recent_feedback) < max(int(self.config.stagnation_window or 1), 1):
            return None, None

        hard_delta = self._recent_sum("code_delta")
        if hard_delta >= int(self.config.min_hard_gain_delta):
            return None, None

        reason = "gei_stagnation"
        transition = f"{StageType.GEI.value}->{StageType.KSI.value}"
        self.stage = StageType.KSI
        self.ksi_entry_generation = int(generation)
        self.ksi_transition_reason = reason
        self.transition_reasons.append(reason)
        self.transitions.append(
            {
                "generation": int(generation),
                "transition": transition,
                "reason": reason,
                "recent_code_delta": self._recent_sum("code_delta"),
            }
        )
        return transition, reason

    def _maybe_record_snapshot(
        self,
        generation: int,
        individual_id: str,
        stage: str,
        feedback: Dict[str, Any],
    ) -> Optional[SnapshotCandidate]:
        if not self.config.enable_snapshot_pool:
            return None
        if not feedback.get("hard_gain"):
            return None

        self.snapshot_sequence += 1
        snapshot = SnapshotCandidate(
            snapshot_id=f"snap-{self.snapshot_sequence:06d}",
            generation=int(generation),
            individual_id=str(individual_id),
            sequence_length=int(feedback.get("sequence_length") or feedback.get("tx_count") or 0),
            code_coverage_count=int(feedback.get("code_coverage_count") or 0),
            code_delta=int(feedback.get("code_delta") or 0),
            revert_count=int(feedback.get("revert_count") or 0),
            invalid_count=int(feedback.get("invalid_count") or 0),
            stage=str(stage),
            reason="hard_gain",
        )
        self.snapshot_candidates.append(snapshot)
        self.snapshot_reasons[snapshot.reason] = self.snapshot_reasons.get(snapshot.reason, 0) + 1
        return snapshot

    def _prepare_log(self) -> None:
        if not self.config.stage_log_path:
            return
        self._log_path = os.path.abspath(self.config.stage_log_path)
        self._jsonl_mode = self._log_path.lower().endswith(".jsonl")
        parent = os.path.dirname(self._log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self._jsonl_mode:
            with open(self._log_path, "w", encoding="utf-8"):
                pass
        else:
            with open(self._log_path, "w", encoding="utf-8") as log_file:
                json.dump([], log_file)

    def _write_log_record(self, record: Dict[str, Any]) -> None:
        if not self._log_path:
            return
        if self._jsonl_mode:
            with open(self._log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            return

        self.log_records.append(record)

    def flush_logs(self) -> None:
        if not self._log_path or self._jsonl_mode:
            return
        with open(self._log_path, "w", encoding="utf-8") as log_file:
            json.dump(self.log_records, log_file, ensure_ascii=False, indent=2)

    def _recent_sum(self, key: str) -> int:
        return sum(self._as_int(item.get(key)) for item in self.recent_feedback)

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            if value in (None, ""):
                return 0
            return int(value)
        except Exception:
            return 0
