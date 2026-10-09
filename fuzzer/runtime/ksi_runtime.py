# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List, Optional


class KSIRuntime:
    def __init__(self, output_dir: str, enabled: bool = False):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.output_path = os.path.join(self.output_dir, "ksi_evidence.jsonl") if self.output_dir else ""
        self.enabled = bool(enabled and self.output_path)
        self.ksi_evidence_records = 0

        if self.enabled:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(self.output_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                self.enabled = False

    def record(
        self,
        sequence_id: str,
        function_names: List[str],
        tx_summaries: Optional[List[Dict[str, Any]]],
        progress_records: Optional[List[Dict[str, Any]]],
        execution_result: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        try:
            sequence_id = str(sequence_id or "")
            function_names = self._normalize_list(function_names)
            tx_summaries = tx_summaries if isinstance(tx_summaries, list) else []
            progress_records = progress_records if isinstance(progress_records, list) else []
            execution_result = execution_result if isinstance(execution_result, dict) else {}

            observations = self._build_observations(function_names, tx_summaries, execution_result)
            checkpoint_snapshot = self._extract_checkpoint_snapshot(execution_result)
            if checkpoint_snapshot:
                observations["checkpoint_current_best_level"] = checkpoint_snapshot.get("best_level")
                observations["checkpoint_strict_vspc"] = checkpoint_snapshot.get("strict_vspc")
                observations["checkpoint_next_missing_checkpoint"] = checkpoint_snapshot.get("next_missing_checkpoint")
                observations["checkpoint_runtime_pattern_family"] = checkpoint_snapshot.get("runtime_pattern_family")
                covered = checkpoint_snapshot.get("covered_checkpoints")
                observations["checkpoint_covered_count"] = len(covered) if isinstance(covered, list) else checkpoint_snapshot.get("covered_count")
            rq1_phase = execution_result.get("rq1_phase")
            records = []
            if progress_records:
                for progress_record in progress_records:
                    if not isinstance(progress_record, dict):
                        continue
                    record = self._build_hypothesis_record(sequence_id, progress_record, observations)
                    record["rq1_phase"] = rq1_phase
                    record["checkpoint_progress"] = checkpoint_snapshot
                    records.append(record)
            else:
                records.append(
                    {
                        "sequence_id": sequence_id,
                        "rq1_phase": rq1_phase,
                        "checkpoint_progress": checkpoint_snapshot,
                        "hypothesis_id": None,
                        "matched_prefix_len": 0,
                        "blocked_at": None,
                        "frontier": None,
                        "sink_reached": False,
                        "sink_observed": False,
                        "sink_function_reached": False,
                        "sink_executed_successfully": False,
                        "status": "non_hypothesis_sequence",
                        "observations": observations,
                        "suspected_blockers": [],
                    }
                )

            self._append_records(records)
            self.ksi_evidence_records += len(records)
            return records
        except Exception:
            return []

    def get_summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "ksi_evidence_records": int(self.ksi_evidence_records),
            "ksi_evidence_path": self.output_path if self.output_path else None,
        }

    @staticmethod
    def _normalize_list(value: Any) -> List[Any]:
        if not isinstance(value, list):
            return []
        return list(value)

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

    @staticmethod
    def _has_revert(execution_result: Dict[str, Any], tx_summaries: List[Dict[str, Any]]) -> bool:
        if execution_result.get("revert") is True:
            return True
        for summary in tx_summaries:
            if isinstance(summary, dict) and summary.get("revert") is True:
                return True
        return False

    def _build_observations(
        self,
        function_names: List[str],
        tx_summaries: List[Dict[str, Any]],
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        sender_sequence = []
        value_sequence = []
        args_sequence = []
        for summary in tx_summaries:
            if not isinstance(summary, dict):
                continue
            sender_sequence.append(summary.get("sender"))
            value_sequence.append(summary.get("value", 0))
            args_sequence.append(summary.get("args", ""))

        return {
            "functions": list(function_names),
            "sender_sequence": sender_sequence,
            "value_sequence": value_sequence,
            "args_sequence": args_sequence,
            "revert": self._has_revert(execution_result, tx_summaries),
            "revert_reason": self._extract_revert_reason(execution_result, tx_summaries),
            "coverage_delta": execution_result.get("coverage_delta"),
            "rq1_phase": execution_result.get("rq1_phase"),
            "checkpoint_current_best_level": execution_result.get("checkpoint_current_best_level"),
            "checkpoint_strict_vspc": execution_result.get("checkpoint_strict_vspc"),
            "checkpoint_next_missing_checkpoint": execution_result.get("checkpoint_next_missing_checkpoint"),
            "checkpoint_runtime_pattern_family": execution_result.get("checkpoint_runtime_pattern_family"),
            "checkpoint_covered_count": execution_result.get("checkpoint_covered_count"),
            "prefix_reached": [],
            "storage_reads": [],
            "storage_writes": [],
            "calls": [],
        }

    @staticmethod
    def _extract_checkpoint_snapshot(execution_result: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = execution_result.get("checkpoint_progress_snapshot")
        if isinstance(snapshot, dict):
            return dict(snapshot)
        return {
            "best_level": execution_result.get("checkpoint_current_best_level"),
            "strict_vspc": execution_result.get("checkpoint_strict_vspc"),
            "next_missing_checkpoint": execution_result.get("checkpoint_next_missing_checkpoint"),
            "runtime_pattern_family": execution_result.get("checkpoint_runtime_pattern_family"),
            "covered_count": execution_result.get("checkpoint_covered_count"),
        }

    def _build_hypothesis_record(
        self,
        sequence_id: str,
        progress_record: Dict[str, Any],
        observations: Dict[str, Any],
    ) -> Dict[str, Any]:
        record_observations = dict(observations)
        record_observations["prefix_reached"] = list(progress_record.get("best_prefix") or [])
        record_observations["sink_function_reached"] = bool(progress_record.get("sink_function_reached", False))
        record_observations["sink_executed_successfully"] = bool(
            progress_record.get("sink_executed_successfully", False)
        )
        status = progress_record.get("status")
        sink_reached = bool(progress_record.get("sink_reached", False))
        sink_observed = bool(progress_record.get("sink_observed", False))
        sink_function_reached = bool(progress_record.get("sink_function_reached", False))
        sink_executed_successfully = bool(progress_record.get("sink_executed_successfully", False))
        suspected_blockers = self._suspected_blockers(progress_record, record_observations)

        return {
            "sequence_id": sequence_id,
            "hypothesis_id": progress_record.get("matched_hypothesis"),
            "matched_prefix_len": progress_record.get("matched_prefix_len", 0),
            "blocked_at": progress_record.get("blocked_at"),
            "frontier": progress_record.get("frontier"),
            "sink_reached": sink_reached,
            "sink_observed": sink_observed,
            "sink_function_reached": sink_function_reached,
            "sink_executed_successfully": sink_executed_successfully,
            "status": status,
            "observations": record_observations,
            "suspected_blockers": suspected_blockers,
        }

    @staticmethod
    def _suspected_blockers(
        progress_record: Dict[str, Any],
        observations: Dict[str, Any],
    ) -> List[str]:
        blockers = []
        if progress_record.get("status") == "blocked" and observations.get("revert") is True:
            blockers.append("revert_before_frontier")
        if (
            progress_record.get("status") == "sink_reverted"
            or (
                progress_record.get("sink_function_reached") is True
                and observations.get("revert") is True
                and progress_record.get("sink_executed_successfully") is not True
            )
        ):
            blockers.append("revert_at_sink")

        sender_sequence = observations.get("sender_sequence") or []
        normalized_senders = [sender for sender in sender_sequence if sender not in (None, "")]
        if normalized_senders and len(set(normalized_senders)) > 1:
            blockers.append("sender_change")

        if progress_record.get("sink_observed") is True and progress_record.get("sink_reached") is not True:
            blockers.append("sink_observed_without_valid_prefix")
        return blockers

    def _append_records(self, records: List[Dict[str, Any]]) -> None:
        if not records or not self.enabled or not self.output_path:
            return
        try:
            with open(self.output_path, "a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            pass
