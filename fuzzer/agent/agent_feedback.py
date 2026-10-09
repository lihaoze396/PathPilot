# -*- coding: utf-8 -*-

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

try:
    from fuzzer.guidance.path_plan import PathPlan
except ImportError:
    from fuzzer.guidance.path_plan import PathPlan


class AgentFeedbackBuilder:
    def __init__(
        self,
        path_plan: PathPlan,
        output_dir: str,
        path_progress_path: Optional[str] = None,
        ksi_evidence_path: Optional[str] = None,
        side_path_candidates_path: Optional[str] = None,
    ):
        self.path_plan = path_plan or PathPlan()
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.path_progress_path = path_progress_path or os.path.join(self.output_dir, "path_progress.jsonl")
        self.ksi_evidence_path = ksi_evidence_path or os.path.join(self.output_dir, "ksi_evidence.jsonl")
        self.side_path_candidates_path = (
            side_path_candidates_path or os.path.join(self.output_dir, "side_path_candidates.jsonl")
        )
        self.side_path_shortlist_path = os.path.join(self.output_dir, "side_path_shortlist.json")

    def build(self) -> Dict[str, Any]:
        try:
            path_progress_records = self._read_jsonl(self.path_progress_path)
            ksi_evidence_records = self._read_jsonl(self.ksi_evidence_path)
            side_path_records = self._read_jsonl(self.side_path_candidates_path)
            side_path_shortlist = self._read_json(self.side_path_shortlist_path)
            hypotheses = []
            sink_reached_total = 0
            sink_function_reached_total = 0
            sink_reverted_total = 0
            blocked_total = 0

            for hypothesis in self.path_plan.hypotheses:
                progress_records = [
                    record
                    for record in path_progress_records
                    if record.get("matched_hypothesis") == hypothesis.id
                ]
                ksi_records = [
                    record
                    for record in ksi_evidence_records
                    if record.get("hypothesis_id") == hypothesis.id
                ]
                progress_summary = self._build_progress_summary(hypothesis, progress_records)
                runtime_evidence = self._build_runtime_evidence(ksi_records)
                sink_reached_total += progress_summary["sink_reached_count"]
                sink_function_reached_total += progress_summary["sink_function_reached_count"]
                sink_reverted_total += progress_summary["sink_reverted_count"]
                blocked_total += progress_summary["blocked_count"]
                hypotheses.append(
                    {
                        "hypothesis_id": hypothesis.id,
                        "vulnerability_type": hypothesis.vulnerability_type,
                        "path": list(hypothesis.path),
                        "sink": hypothesis.sink,
                        "priority": hypothesis.priority,
                        "expected_blockers_from_plan": list(hypothesis.expected_blockers),
                        "progress_summary": progress_summary,
                        "runtime_evidence": runtime_evidence,
                        "evidence": self._build_evidence_sentences(progress_summary, runtime_evidence),
                        "repair_objective": self._build_repair_objective(
                            hypothesis,
                            progress_summary,
                            runtime_evidence,
                        ),
                    }
                )

            return {
                "hypotheses": hypotheses,
                "side_observations": self._build_side_observations(ksi_evidence_records),
                "global_summary": {
                    "hypothesis_count": len(self.path_plan.hypotheses),
                    "total_path_progress_records": len(path_progress_records),
                    "total_ksi_evidence_records": len(ksi_evidence_records),
                    "sink_reached_total": sink_reached_total,
                    "sink_function_reached_total": sink_function_reached_total,
                    "sink_reverted_total": sink_reverted_total,
                    "blocked_total": blocked_total,
                },
                "side_path_candidates": self._build_side_path_candidates(side_path_records, side_path_shortlist),
            }
        except Exception:
            return {
                "hypotheses": [],
                "side_observations": {
                    "non_hypothesis_sequence_count": 0,
                    "sink_like_sequence_count": 0,
                },
                "global_summary": {
                    "hypothesis_count": 0,
                    "total_path_progress_records": 0,
                    "total_ksi_evidence_records": 0,
                    "sink_reached_total": 0,
                    "sink_function_reached_total": 0,
                    "sink_reverted_total": 0,
                    "blocked_total": 0,
                },
                "side_path_candidates": {
                    "count": 0,
                    "top_reasons": {},
                    "suggested_new_hypotheses": [],
                    "shortlist_count": 0,
                    "shortlist": [],
                },
            }

    def write(
        self,
        feedback: Dict[str, Any],
        filename: str = "blocker_evidence.json",
    ) -> str:
        if not self.output_dir:
            return ""
        output_path = os.path.join(self.output_dir, filename)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(feedback, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        return output_path

    def build_and_write(self) -> str:
        return self.write(self.build())

    @staticmethod
    def _read_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except Exception:
            return []
        return records

    @staticmethod
    def _read_json(path: Optional[str]) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    @staticmethod
    def _build_progress_summary(hypothesis: Any, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        best_prefix_len = 0
        best_prefix = []
        frontier = hypothesis.path[0] if hypothesis.path else hypothesis.sink
        blocked_at_counter = Counter()
        sink_reached_count = 0
        sink_function_reached_count = 0
        sink_reverted_count = 0
        blocked_count = 0
        prefix_progress_count = 0
        last_status = None

        for record in records:
            status = record.get("status")
            last_status = status
            if record.get("sink_function_reached") is True:
                sink_function_reached_count += 1
            if status == "sink_reached":
                sink_reached_count += 1
            elif status == "sink_reverted":
                sink_reverted_count += 1
                blocked_count += 1
                blocked_at = record.get("blocked_at")
                if blocked_at not in (None, ""):
                    blocked_at_counter[str(blocked_at)] += 1
            elif status == "blocked":
                blocked_count += 1
                blocked_at = record.get("blocked_at")
                if blocked_at not in (None, ""):
                    blocked_at_counter[str(blocked_at)] += 1
            elif status == "prefix_progress":
                prefix_progress_count += 1

            prefix_len = int(record.get("matched_prefix_len") or 0)
            if prefix_len >= best_prefix_len:
                best_prefix_len = prefix_len
                best_prefix = list(record.get("best_prefix") or [])
                frontier = record.get("frontier") or frontier

        if not frontier and best_prefix_len < len(hypothesis.path):
            frontier = hypothesis.path[best_prefix_len]
        elif not frontier:
            frontier = hypothesis.sink

        dominant_blocked_at = None
        if blocked_at_counter:
            dominant_blocked_at = blocked_at_counter.most_common(1)[0][0]

        return {
            "best_prefix_len": best_prefix_len,
            "best_prefix": best_prefix,
            "frontier": frontier,
            "sink_reached_count": sink_reached_count,
            "sink_function_reached_count": sink_function_reached_count,
            "sink_reverted_count": sink_reverted_count,
            "blocked_count": blocked_count,
            "prefix_progress_count": prefix_progress_count,
            "last_status": last_status,
            "dominant_blocked_at": dominant_blocked_at,
        }

    @staticmethod
    def _build_runtime_evidence(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        blocker_counter = Counter()
        recent_revert_reasons = []
        recent_sequences = []
        revert_count = 0
        sender_change_count = 0
        sink_observed_without_valid_prefix_count = 0
        revert_at_sink_count = 0

        for record in records:
            observations = record.get("observations") if isinstance(record.get("observations"), dict) else {}
            if observations.get("revert") is True:
                revert_count += 1
            reason = observations.get("revert_reason")
            if reason not in (None, ""):
                recent_revert_reasons.append(str(reason))
            sequence_id = record.get("sequence_id")
            if sequence_id not in (None, ""):
                recent_sequences.append(str(sequence_id))

            suspected_blockers = record.get("suspected_blockers")
            if not isinstance(suspected_blockers, list):
                suspected_blockers = []
            for blocker in suspected_blockers:
                blocker_text = str(blocker)
                blocker_counter[blocker_text] += 1
                if blocker_text == "sender_change":
                    sender_change_count += 1
                elif blocker_text == "sink_observed_without_valid_prefix":
                    sink_observed_without_valid_prefix_count += 1
                elif blocker_text == "revert_at_sink":
                    revert_at_sink_count += 1

        return {
            "revert_count": revert_count,
            "sender_change_count": sender_change_count,
            "sink_observed_without_valid_prefix_count": sink_observed_without_valid_prefix_count,
            "revert_at_sink_count": revert_at_sink_count,
            "top_suspected_blockers": [item[0] for item in blocker_counter.most_common(5)],
            "recent_revert_reasons": AgentFeedbackBuilder._recent_unique(recent_revert_reasons, 5),
            "recent_sequences": AgentFeedbackBuilder._recent_unique(recent_sequences, 5),
        }

    @staticmethod
    def _recent_unique(values: List[str], limit: int) -> List[str]:
        seen = set()
        recent = []
        for value in reversed(values):
            if value in seen:
                continue
            seen.add(value)
            recent.append(value)
            if len(recent) >= limit:
                break
        recent.reverse()
        return recent

    @staticmethod
    def _build_side_observations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        non_hypothesis_sequence_count = 0
        sink_like_sequence_count = 0
        for record in records:
            if record.get("hypothesis_id") is None or record.get("status") == "non_hypothesis_sequence":
                non_hypothesis_sequence_count += 1
            if record.get("sink_observed") is True and record.get("sink_reached") is not True:
                sink_like_sequence_count += 1
        return {
            "non_hypothesis_sequence_count": non_hypothesis_sequence_count,
            "sink_like_sequence_count": sink_like_sequence_count,
        }

    @staticmethod
    def _build_side_path_candidates(
        records: List[Dict[str, Any]],
        shortlist_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reason_counter = Counter()
        suggestions = []
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            reason = str(record.get("reason") or "unknown")
            reason_counter[reason] += 1
            suggested_path = record.get("suggested_path")
            if not isinstance(suggested_path, list):
                suggested_path = []
            signature = (tuple(str(item) for item in suggested_path), reason)
            if signature in seen:
                continue
            seen.add(signature)
            suggestions.append(
                {
                    "suggested_path": [str(item) for item in suggested_path],
                    "reason": reason,
                    "example_sequence_id": record.get("sequence_id"),
                }
            )

        shortlist = AgentFeedbackBuilder._build_side_path_shortlist(shortlist_payload)
        return {
            "count": len(records),
            "top_reasons": dict(reason_counter.most_common(5)),
            "suggested_new_hypotheses": suggestions[:20],
            "shortlist_count": len(shortlist),
            "shortlist": shortlist,
        }

    @staticmethod
    def _build_side_path_shortlist(shortlist_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(shortlist_payload, dict):
            return []
        shortlist = shortlist_payload.get("shortlist")
        if not isinstance(shortlist, list):
            return []

        result = []
        for record in shortlist:
            if not isinstance(record, dict):
                continue
            tier = str(record.get("tier") or "low")
            if tier == "low":
                continue
            suggested_path = record.get("suggested_path")
            if not isinstance(suggested_path, list):
                suggested_path = []
            result.append(
                {
                    "suggested_path": [str(item) for item in suggested_path],
                    "reason": str(record.get("reason") or "unknown"),
                    "quality_score": int(record.get("quality_score") or 0),
                    "tier": tier,
                    "example_sequence_id": record.get("sequence_id"),
                }
            )
            if len(result) >= 20:
                break
        return result

    @staticmethod
    def _build_evidence_sentences(
        progress_summary: Dict[str, Any],
        runtime_evidence: Dict[str, Any],
    ) -> List[str]:
        evidence = []
        best_prefix = progress_summary.get("best_prefix") or []
        frontier = progress_summary.get("frontier")
        if best_prefix:
            evidence.append(
                "The prefix {} has been reached.".format(" -> ".join(str(item) for item in best_prefix))
            )
        if progress_summary.get("blocked_count", 0) > 0 and frontier:
            evidence.append(
                "The sequence is blocked before frontier {}.".format(frontier)
            )
        if progress_summary.get("sink_reverted_count", 0) > 0:
            evidence.append(
                "The sink function is reached but execution reverts in {} executions.".format(
                    progress_summary["sink_reverted_count"]
                )
            )
        if runtime_evidence.get("revert_count", 0) > 0:
            evidence.append(
                "KSI observed revert before or around the next frontier in {} executions.".format(
                    runtime_evidence["revert_count"]
                )
            )
        if runtime_evidence.get("sender_change_count", 0) > 0:
            evidence.append(
                "KSI observed sender changes in {} executions.".format(
                    runtime_evidence["sender_change_count"]
                )
            )
        if runtime_evidence.get("sink_observed_without_valid_prefix_count", 0) > 0:
            evidence.append(
                "KSI observed sink-like calls without a validated prefix in {} executions.".format(
                    runtime_evidence["sink_observed_without_valid_prefix_count"]
                )
            )
        return evidence

    @staticmethod
    def _build_repair_objective(
        hypothesis: Any,
        progress_summary: Dict[str, Any],
        runtime_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        best_prefix = list(progress_summary.get("best_prefix") or [])
        frontier = progress_summary.get("frontier")
        top_blockers = list(runtime_evidence.get("top_suspected_blockers") or [])

        if progress_summary.get("sink_reached_count", 0) > 0:
            return {
                "action": "validate_oracle_at_sink",
                "affected_prefix": best_prefix,
                "target_frontier": hypothesis.sink,
                "hints": [
                    "sink has been reached",
                    "validate vulnerability oracle or post-state",
                ],
            }
        if progress_summary.get("sink_reverted_count", 0) > 0:
            return {
                "action": "repair_sink_execution",
                "affected_prefix": best_prefix,
                "target_frontier": hypothesis.sink,
                "hints": [
                    "sink function is reached but execution reverts",
                    "inspect revert_at_sink",
                    "adjust caller/value/arguments or required pre-state",
                ] + top_blockers,
            }
        if progress_summary.get("blocked_count", 0) > 0 and progress_summary.get("best_prefix_len", 0) > 0:
            return {
                "action": "preserve_prefix_and_mutate_frontier",
                "affected_prefix": best_prefix,
                "target_frontier": frontier,
                "hints": [
                    "preserve the reached prefix",
                    "mutate transactions around the frontier",
                ] + top_blockers,
            }
        if progress_summary.get("best_prefix_len", 0) > 0:
            return {
                "action": "extend_prefix_towards_frontier",
                "affected_prefix": best_prefix,
                "target_frontier": frontier,
                "hints": [
                    "preserve the reached prefix",
                    "extend the sequence toward the frontier",
                ],
            }
        return {
            "action": "search_for_initial_prefix",
            "affected_prefix": [],
            "target_frontier": hypothesis.path[0] if hypothesis.path else hypothesis.sink,
            "hints": [
                "search for the initial path prefix",
            ],
        }
