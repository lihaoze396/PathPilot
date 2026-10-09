# -*- coding: utf-8 -*-

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


PRIORITY_RANK = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


class TFIRuntime:
    def __init__(
        self,
        output_dir: str,
        enabled: bool = False,
        blocker_evidence_path: Optional[str] = None,
        min_blocked_count: int = 3,
        min_sink_reverted_count: int = 1,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.targets_path = os.path.join(self.output_dir, "tfi_focus_targets.jsonl") if self.output_dir else ""
        self.summary_path = os.path.join(self.output_dir, "tfi_focus_summary.json") if self.output_dir else ""
        self._requested_blocker_evidence_path = os.path.abspath(blocker_evidence_path) if blocker_evidence_path else None
        self._default_blocker_evidence_path = os.path.join(self.output_dir, "blocker_evidence.json") if self.output_dir else ""
        self.blocker_evidence_path = self._requested_blocker_evidence_path or self._default_blocker_evidence_path or None
        self.blocker_evidence_loaded = False
        self.enabled = bool(enabled and self.output_dir)
        self.min_blocked_count = max(0, int(min_blocked_count or 0))
        self.min_sink_reverted_count = max(0, int(min_sink_reverted_count or 0))

    def build_focus_targets(self) -> List[Dict[str, Any]]:
        blocker_evidence = self._load_blocker_evidence()
        hypotheses = blocker_evidence.get("hypotheses")
        if not isinstance(hypotheses, list):
            return []

        deduped: Dict[Tuple[Any, Any, Any, Any], Dict[str, Any]] = {}
        for entry in hypotheses:
            if not isinstance(entry, dict):
                continue

            hypothesis_id = self._as_text(entry.get("hypothesis_id"))
            if not hypothesis_id:
                continue

            path = [self._as_text(item) for item in entry.get("path", []) if self._as_text(item)]
            sink = self._as_text(entry.get("sink"))
            progress = entry.get("progress_summary") if isinstance(entry.get("progress_summary"), dict) else {}
            runtime = entry.get("runtime_evidence") if isinstance(entry.get("runtime_evidence"), dict) else {}
            repair = entry.get("repair_objective") if isinstance(entry.get("repair_objective"), dict) else {}

            best_prefix_len = self._as_int(progress.get("best_prefix_len"))
            sink_reached_count = self._as_int(progress.get("sink_reached_count"))
            sink_function_reached_count = self._as_int(progress.get("sink_function_reached_count"))
            sink_reverted_count = self._as_int(progress.get("sink_reverted_count"))
            blocked_count = self._as_int(progress.get("blocked_count"))
            prefix_progress_count = self._as_int(progress.get("prefix_progress_count"))
            dominant_blocked_at = self._as_text(progress.get("dominant_blocked_at"))
            frontier = self._as_text(progress.get("frontier")) or sink
            repair_action = self._as_text(repair.get("action"))
            revert_at_sink_count = self._as_int(runtime.get("revert_at_sink_count"))
            sender_change_count = self._as_int(runtime.get("sender_change_count"))
            related_sequences = self._extract_related_sequences(entry, runtime)
            top_suspected_blockers = self._extract_text_list(runtime.get("top_suspected_blockers"))
            recent_revert_reasons = self._extract_text_list(runtime.get("recent_revert_reasons"))
            hints = self._extract_text_list(repair.get("hints"))
            evidence_lines = self._extract_text_list(entry.get("evidence"))

            permission_signals = self._collect_permission_signals(
                top_suspected_blockers=top_suspected_blockers,
                recent_revert_reasons=recent_revert_reasons,
                hints=hints,
                evidence_lines=evidence_lines,
            )

            if sink_reverted_count >= self.min_sink_reverted_count or repair_action == "repair_sink_execution":
                self._add_target(
                    deduped,
                    {
                        "hypothesis_id": hypothesis_id,
                        "path": path,
                        "sink": sink,
                        "focus_type": "sink_reverted",
                        "focus_function": sink or frontier,
                        "focus_frontier": frontier or sink,
                        "priority": "high",
                        "reason": "sink function is reached but execution reverts",
                        "related_sequences": related_sequences,
                        "evidence": {
                            "sink_function_reached_count": sink_function_reached_count,
                            "sink_reverted_count": sink_reverted_count,
                            "revert_at_sink_count": revert_at_sink_count,
                            "blocked_count": blocked_count,
                        },
                        "suggested_next_step": "localize caller/value/argument/pre-state constraints around the sink",
                        "source": "blocker_evidence",
                    },
                )

            if blocked_count >= self.min_blocked_count and dominant_blocked_at:
                self._add_target(
                    deduped,
                    {
                        "hypothesis_id": hypothesis_id,
                        "path": path,
                        "sink": sink,
                        "focus_type": "repeated_frontier_block",
                        "focus_function": dominant_blocked_at,
                        "focus_frontier": frontier,
                        "priority": "medium",
                        "reason": "the same frontier or blocked function repeatedly stops path progress",
                        "related_sequences": related_sequences,
                        "evidence": {
                            "blocked_count": blocked_count,
                            "dominant_blocked_at": dominant_blocked_at,
                            "frontier": frontier,
                            "best_prefix_len": best_prefix_len,
                        },
                        "suggested_next_step": "focus local observation around the blocked function and frontier",
                        "source": "blocker_evidence",
                    },
                )

            if (
                "sender_change" in top_suspected_blockers
                or "revert_at_sink" in top_suspected_blockers
                or permission_signals
            ):
                self._add_target(
                    deduped,
                    {
                        "hypothesis_id": hypothesis_id,
                        "path": path,
                        "sink": sink,
                        "focus_type": "caller_or_permission_condition",
                        "focus_function": dominant_blocked_at or sink or frontier,
                        "focus_frontier": frontier or sink,
                        "priority": "medium",
                        "reason": "execution evidence suggests caller or permission condition",
                        "related_sequences": related_sequences,
                        "evidence": {
                            "blocked_count": blocked_count,
                            "sender_change_count": sender_change_count,
                            "revert_at_sink_count": revert_at_sink_count,
                            "permission_signals": permission_signals,
                        },
                        "suggested_next_step": "focus caller, permission, and access-control related conditions around the frontier",
                        "source": "blocker_evidence",
                    },
                )

            if (
                best_prefix_len > 0
                and sink_reached_count == 0
                and sink_reverted_count == 0
                and blocked_count == 0
            ):
                self._add_target(
                    deduped,
                    {
                        "hypothesis_id": hypothesis_id,
                        "path": path,
                        "sink": sink,
                        "focus_type": "prefix_ready_frontier_unknown",
                        "focus_function": frontier or sink,
                        "focus_frontier": frontier or sink,
                        "priority": "low",
                        "reason": "a stable prefix is reachable but no dominant blocker has been localized yet",
                        "related_sequences": related_sequences,
                        "evidence": {
                            "best_prefix_len": best_prefix_len,
                            "sink_reached_count": sink_reached_count,
                            "sink_reverted_count": sink_reverted_count,
                            "blocked_count": blocked_count,
                            "prefix_progress_count": prefix_progress_count,
                        },
                        "suggested_next_step": "focus local observation around the next frontier after the reached prefix",
                        "source": "blocker_evidence",
                    },
                )

        return sorted(
            deduped.values(),
            key=lambda item: (
                PRIORITY_RANK.get(str(item.get("priority") or "low"), 99),
                str(item.get("hypothesis_id") or ""),
                str(item.get("focus_type") or ""),
                str(item.get("focus_function") or ""),
                str(item.get("focus_frontier") or ""),
            ),
        )

    def write_focus_targets(self, targets: List[Dict[str, Any]]) -> str:
        if not self.enabled or not self.targets_path:
            return ""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.targets_path, "w", encoding="utf-8") as handle:
            for target in targets:
                handle.write(json.dumps(target, ensure_ascii=True) + "\n")
        return self.targets_path

    def write_summary(self, targets: List[Dict[str, Any]]) -> str:
        if not self.enabled or not self.summary_path:
            return ""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(self._build_summary_payload(targets), handle, ensure_ascii=True)
        return self.summary_path

    def run(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        try:
            targets = self.build_focus_targets()
            self.write_focus_targets(targets)
            self.write_summary(targets)
            return self._build_summary_payload(targets)
        except Exception:
            summary = self._build_summary_payload([])
            try:
                self.write_summary([])
            except Exception:
                pass
            return summary

    def _build_summary_payload(self, targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        priority_counts = Counter()
        focus_type_counts = Counter()
        function_counts = Counter()
        for target in targets:
            priority = self._as_text(target.get("priority")) or "low"
            focus_type = self._as_text(target.get("focus_type")) or "unknown"
            focus_function = self._as_text(target.get("focus_function"))
            priority_counts[priority] += 1
            focus_type_counts[focus_type] += 1
            if focus_function:
                function_counts[focus_function] += 1
        return {
            "enabled": True,
            "target_count": len(targets),
            "priority_counts": {
                "high": int(priority_counts.get("high", 0)),
                "medium": int(priority_counts.get("medium", 0)),
                "low": int(priority_counts.get("low", 0)),
            },
            "focus_type_counts": dict(focus_type_counts),
            "top_focus_functions": dict(function_counts.most_common(10)),
            "output_path": self.targets_path if self.targets_path else None,
            "summary_path": self.summary_path if self.summary_path else None,
            "blocker_evidence_path": self.blocker_evidence_path,
            "blocker_evidence_loaded": bool(self.blocker_evidence_loaded),
        }

    def _load_blocker_evidence(self) -> Dict[str, Any]:
        candidate_paths = []
        if self._requested_blocker_evidence_path:
            candidate_paths.append(self._requested_blocker_evidence_path)
        if self._default_blocker_evidence_path and self._default_blocker_evidence_path not in candidate_paths:
            candidate_paths.append(self._default_blocker_evidence_path)

        self.blocker_evidence_loaded = False
        self.blocker_evidence_path = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = self._read_json(path)
            if payload:
                self.blocker_evidence_loaded = True
                self.blocker_evidence_path = path
                return payload
        return {}

    @staticmethod
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

    @staticmethod
    def _as_text(value: Any) -> str:
        if value in (None, ""):
            return ""
        return str(value)

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0

    @staticmethod
    def _extract_text_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        output = []
        for item in value:
            if item in (None, ""):
                continue
            output.append(str(item))
        return output

    @staticmethod
    def _extract_related_sequences(entry: Dict[str, Any], runtime: Dict[str, Any]) -> List[str]:
        candidates = []
        for item in entry.get("recent_sequences", []):
            if item not in (None, ""):
                candidates.append(str(item))
        for item in runtime.get("recent_sequences", []):
            if item not in (None, ""):
                candidates.append(str(item))
        seen = set()
        ordered = []
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered[:10]

    @staticmethod
    def _collect_permission_signals(
        top_suspected_blockers: List[str],
        recent_revert_reasons: List[str],
        hints: List[str],
        evidence_lines: List[str],
    ) -> List[str]:
        keywords = ("owner", "authorized", "permission")
        signals = []

        for blocker in top_suspected_blockers:
            lowered = blocker.lower()
            if any(keyword in lowered for keyword in keywords):
                signals.append(blocker)

        for text in recent_revert_reasons + hints + evidence_lines:
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                signals.append(text)

        seen = set()
        ordered = []
        for item in signals:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered[:10]

    @staticmethod
    def _merge_target(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        if PRIORITY_RANK.get(str(incoming.get("priority") or "low"), 99) < PRIORITY_RANK.get(
            str(existing.get("priority") or "low"),
            99,
        ):
            winner = dict(incoming)
            loser = existing
        else:
            winner = dict(existing)
            loser = incoming

        merged_sequences = []
        seen = set()
        for item in winner.get("related_sequences", []) + loser.get("related_sequences", []):
            if item in seen:
                continue
            seen.add(item)
            merged_sequences.append(item)
        winner["related_sequences"] = merged_sequences[:10]
        return winner

    def _add_target(self, deduped: Dict[Tuple[Any, Any, Any, Any], Dict[str, Any]], target: Dict[str, Any]) -> None:
        signature = (
            target.get("hypothesis_id"),
            target.get("focus_type"),
            target.get("focus_function"),
            target.get("focus_frontier"),
        )
        existing = deduped.get(signature)
        if existing is None:
            deduped[signature] = target
        else:
            deduped[signature] = self._merge_target(existing, target)
