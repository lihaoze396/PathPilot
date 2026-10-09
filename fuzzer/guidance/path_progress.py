# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List, Optional

try:
    from fuzzer.guidance.path_plan import PathPlan
except ImportError:
    from fuzzer.guidance.path_plan import PathPlan


class PathProgressTracker:
    def __init__(self, path_plan: PathPlan, output_dir: str, enabled: bool = False):
        self.path_plan = path_plan or PathPlan()
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.output_path = os.path.join(self.output_dir, "path_progress.jsonl") if self.output_dir else ""
        self.enabled = bool(enabled and not self.path_plan.is_empty() and self.output_path)
        self.sequence_counter = 0
        self.path_progress_records = 0
        self.best_prefix_len = {
            hypothesis.id: 0 for hypothesis in self.path_plan.hypotheses
        }
        self.sink_reached_count = 0
        self.sink_function_reached_count = 0
        self.sink_reverted_count = 0
        self.blocked_count = 0
        self.first_prefix_progress_time_sec = None
        self.first_prefix_progress_tx_index = None
        self.first_sink_function_reached_time_sec = None
        self.first_sink_function_reached_tx_index = None
        self.first_sink_reached_time_sec = None
        self.first_sink_reached_tx_index = None

        if self.enabled:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(self.output_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                self.enabled = False

    def record_sequence(
        self,
        sequence_id: str,
        function_names: List[str],
        tx_summaries: Optional[List[Dict[str, Any]]] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        elapsed_time_sec: Optional[float] = None,
        global_tx_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or self.path_plan.is_empty():
            return []

        try:
            records = self._build_records(
                sequence_id=sequence_id,
                function_names=function_names,
                tx_summaries=tx_summaries,
                execution_result=execution_result,
                elapsed_time_sec=elapsed_time_sec,
                global_tx_index=global_tx_index,
            )
            for record in records:
                matched_prefix_len = int(record.get("matched_prefix_len") or 0)
                hypothesis_id = record.get("matched_hypothesis")
                self.best_prefix_len[hypothesis_id] = max(
                    self.best_prefix_len.get(hypothesis_id, 0),
                    matched_prefix_len,
                )
                if matched_prefix_len > 0 and self.first_prefix_progress_time_sec is None:
                    self.first_prefix_progress_time_sec = elapsed_time_sec
                    self.first_prefix_progress_tx_index = global_tx_index
                if bool(record.get("sink_function_reached")):
                    self.sink_function_reached_count += 1
                    if self.first_sink_function_reached_time_sec is None:
                        self.first_sink_function_reached_time_sec = elapsed_time_sec
                        self.first_sink_function_reached_tx_index = global_tx_index
                if bool(record.get("sink_reached")):
                    self.sink_reached_count += 1
                    if self.first_sink_reached_time_sec is None:
                        self.first_sink_reached_time_sec = elapsed_time_sec
                        self.first_sink_reached_tx_index = global_tx_index
                elif record.get("status") == "sink_reverted":
                    self.sink_reverted_count += 1
                    self.blocked_count += 1
                elif record.get("status") == "blocked":
                    self.blocked_count += 1

            self._append_records(records)
            self.path_progress_records += len(records)
            return records
        except Exception:
            return []

    def build_progress_hint(
        self,
        function_names: List[str],
        tx_summaries: Optional[List[Dict[str, Any]]] = None,
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.path_plan.is_empty():
            return {}
        try:
            records = self._build_records(
                sequence_id="oracle_progress_hint",
                function_names=function_names,
                tx_summaries=tx_summaries,
                execution_result=execution_result,
                elapsed_time_sec=None,
                global_tx_index=None,
            )
        except Exception:
            return {}
        best_record = None
        best_key = None
        for record in records:
            if not isinstance(record, dict):
                continue
            matched_prefix_len = int(record.get("matched_prefix_len") or 0)
            sink_function_reached = 1 if bool(record.get("sink_function_reached")) else 0
            sink_reached = 1 if bool(record.get("sink_reached")) else 0
            candidate_key = (matched_prefix_len, sink_function_reached, sink_reached)
            if best_record is None or candidate_key > best_key:
                best_record = record
                best_key = candidate_key
        return dict(best_record) if isinstance(best_record, dict) else {}

    def get_summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "hypothesis_count": len(self.path_plan.hypotheses),
            "path_progress_records": int(self.path_progress_records),
            "best_prefix_len": dict(self.best_prefix_len),
            "sink_reached_count": int(self.sink_reached_count),
            "sink_function_reached_count": int(self.sink_function_reached_count),
            "sink_reverted_count": int(self.sink_reverted_count),
            "blocked_count": int(self.blocked_count),
            "first_prefix_progress_time_sec": self.first_prefix_progress_time_sec,
            "first_prefix_progress_tx_index": self.first_prefix_progress_tx_index,
            "first_sink_function_reached_time_sec": self.first_sink_function_reached_time_sec,
            "first_sink_function_reached_tx_index": self.first_sink_function_reached_tx_index,
            "first_sink_reached_time_sec": self.first_sink_reached_time_sec,
            "first_sink_reached_tx_index": self.first_sink_reached_tx_index,
        }

    def replace_path_plan(self, path_plan: PathPlan) -> bool:
        if not isinstance(path_plan, PathPlan):
            return False
        self.path_plan = path_plan
        self.enabled = bool(self.output_path and not self.path_plan.is_empty())
        self.best_prefix_len = {
            hypothesis.id: int(self.best_prefix_len.get(hypothesis.id, 0))
            for hypothesis in self.path_plan.hypotheses
        }
        return True

    def _normalize_sequence_id(self, sequence_id: str) -> str:
        if sequence_id:
            return str(sequence_id)
        self.sequence_counter += 1
        return "S{}".format(self.sequence_counter)

    @staticmethod
    def _normalize_function_names(function_names: List[str]) -> List[str]:
        if not isinstance(function_names, list):
            return []
        normalized = []
        for item in function_names:
            text = str(item or "").strip()
            normalized.append(text or "unknown")
        return normalized

    @staticmethod
    def _matched_prefix_len(path: List[str], function_names: List[str]) -> int:
        matched = 0
        if not path:
            return matched
        for actual in function_names:
            if matched >= len(path):
                break
            expected = path[matched]
            if str(expected) != str(actual):
                continue
            matched += 1
        return matched

    @staticmethod
    def _sink_function_reached(hypothesis: Any, matched_prefix_len: int) -> bool:
        return bool(hypothesis.path) and matched_prefix_len >= len(hypothesis.path)

    @staticmethod
    def _has_revert(execution_result: Dict[str, Any], tx_summaries: List[Dict[str, Any]]) -> bool:
        if execution_result.get("revert") is True:
            return True
        for summary in tx_summaries:
            if isinstance(summary, dict) and summary.get("revert") is True:
                return True
        return False

    @staticmethod
    def _extract_revert_reason(
        execution_result: Dict[str, Any],
        tx_summaries: List[Dict[str, Any]],
    ) -> Optional[str]:
        reason = execution_result.get("revert_reason")
        if reason not in (None, ""):
            return str(reason)
        for summary in reversed(tx_summaries):
            if not isinstance(summary, dict):
                continue
            reason = summary.get("revert_reason")
            if reason not in (None, ""):
                return str(reason)
        return None

    def _build_records(
        self,
        *,
        sequence_id: str,
        function_names: List[str],
        tx_summaries: Optional[List[Dict[str, Any]]] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        elapsed_time_sec: Optional[float] = None,
        global_tx_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        tx_summaries = tx_summaries if isinstance(tx_summaries, list) else []
        execution_result = execution_result if isinstance(execution_result, dict) else {}
        sequence_id = self._normalize_sequence_id(sequence_id)
        function_names = self._normalize_function_names(function_names)
        has_revert = self._has_revert(execution_result, tx_summaries)
        revert_reason = self._extract_revert_reason(execution_result, tx_summaries)

        records: List[Dict[str, Any]] = []
        observed_names = {str(name) for name in function_names}
        for hypothesis in self.path_plan.hypotheses:
            matched_prefix_len = self._matched_prefix_len(hypothesis.path, function_names)
            if matched_prefix_len <= 0:
                continue
            hypothesis_path_length = len(hypothesis.path)
            sink_observed = bool(hypothesis.sink and str(hypothesis.sink) in observed_names)
            sink_function_reached = self._sink_function_reached(hypothesis, matched_prefix_len)
            sink_executed_successfully = bool(sink_function_reached and not has_revert)
            sink_reached = sink_executed_successfully

            status = "sink_reached" if sink_reached else "prefix_progress"
            if sink_function_reached and has_revert:
                status = "sink_reverted"
            elif not sink_reached and has_revert:
                status = "blocked"

            frontier = None
            if matched_prefix_len < hypothesis_path_length:
                frontier = hypothesis.path[matched_prefix_len]
            elif hypothesis.sink:
                frontier = hypothesis.sink

            bug_path_coverage = None
            if hypothesis_path_length > 0:
                bug_path_coverage = min(
                    1.0,
                    max(0.0, float(matched_prefix_len) / float(hypothesis_path_length)),
                )

            record = {
                "sequence_id": sequence_id,
                "elapsed_time_sec": elapsed_time_sec,
                "global_tx_index": global_tx_index,
                "matched_hypothesis": hypothesis.id,
                "matched_prefix_len": matched_prefix_len,
                "hypothesis_path_length": hypothesis_path_length,
                "bug_path_coverage": bug_path_coverage,
                "best_prefix": hypothesis.path[:matched_prefix_len],
                "frontier": frontier,
                "sink_reached": sink_reached,
                "sink_observed": sink_observed,
                "sink_function_reached": sink_function_reached,
                "sink_executed_successfully": sink_executed_successfully,
                "status": status,
            }
            if status == "sink_reverted":
                record["blocked_at"] = hypothesis.sink
                record["frontier"] = hypothesis.sink
                record["revert_reason"] = revert_reason
            elif status == "blocked":
                record["blocked_at"] = hypothesis.path[matched_prefix_len - 1]
                record["revert_reason"] = revert_reason

            records.append(record)
        return records

    def _append_records(self, records: List[Dict[str, Any]]) -> None:
        if not records or not self.enabled or not self.output_path:
            return
        try:
            with open(self.output_path, "a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            pass
