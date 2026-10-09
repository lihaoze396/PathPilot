# -*- coding: utf-8 -*-

import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
    except Exception:
        pass


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


class RepairCandidateLifecycleTracker:
    STAGE_ORDER = {
        "materialized": 1,
        "queued": 2,
        "admitted": 3,
        "selected": 4,
        "execution_submitted": 5,
        "execution_started": 6,
        "execution_exception": 7,
        "executed": 8,
        "attributed": 9,
    }

    STALE_DROP_REASON_MIN_STAGE = {
        "queued_but_not_admitted_this_round": 3,
        "dropped_before_scheduler_admission": 3,
        "admitted_but_never_selected": 4,
        "selected_but_never_submitted": 5,
        "execution_submitted_but_not_started": 6,
        "execution_submitted_but_not_completed": 8,
        "execution_started_but_not_completed": 8,
        "execution_exception": 8,
        "executed_but_not_attributed": 9,
    }

    STAGE_TO_FIELD = {
        "materialized": "materialized_at",
        "queued": "queued_at",
        "admitted": "admitted_at",
        "selected": "selected_at",
        "execution_submitted": "execution_submitted_at",
        "execution_started": "execution_started_at",
        "execution_exception": "execution_exception_at",
        "executed": "executed_at",
        "attributed": "attributed_at",
    }

    STAGE_TO_COUNTER = {
        "materialized": "repair_candidate_materialized",
        "queued": "repair_candidate_queued",
        "admitted": "repair_candidate_admitted",
        "selected": "repair_candidate_selected",
        "execution_submitted": "repair_candidate_execution_submitted",
        "execution_started": "repair_candidate_execution_started",
        "execution_exception": "repair_candidate_execution_exception",
        "executed": "repair_candidate_executed",
        "attributed": "repair_candidate_attributed",
    }

    BASE_FIELDS = [
        "repair_candidate_id",
        "bench_id",
        "contract",
        "source_trigger_id",
        "source_parent_id",
        "source_scope_id",
        "source_candidate_kind",
        "materialized_at",
        "queued_at",
        "admitted_at",
        "selected_at",
        "execution_submitted_at",
        "execution_started_at",
        "execution_exception_at",
        "executed_at",
        "attributed_at",
        "drop_reason",
    ]

    def __init__(self, *, output_dir: str, bench_id: str = "", contract_name: str = "") -> None:
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.bench_id = _text(bench_id).strip()
        self.contract_name = _text(contract_name).strip()
        self.enabled = bool(self.output_dir)
        self.events_path = os.path.join(self.output_dir, "candidate_lifecycle.jsonl") if self.output_dir else ""
        self.summary_path = os.path.join(self.output_dir, "candidate_lifecycle_summary.json") if self.output_dir else ""
        self.records: Dict[str, Dict[str, Any]] = {}
        self.counters: Counter = Counter()
        self.drop_reason_histogram: Counter = Counter()
        if self.enabled:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(self.events_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                self.enabled = False

    def _seed_record(self, repair_candidate_id: Any, **seed: Any) -> Optional[Dict[str, Any]]:
        candidate_id = _text(repair_candidate_id).strip()
        if not candidate_id:
            return None
        record = self.records.get(candidate_id)
        if not isinstance(record, dict):
            record = {field: None for field in self.BASE_FIELDS}
            record["repair_candidate_id"] = candidate_id
            record["bench_id"] = self.bench_id or None
            record["contract"] = self.contract_name or None
            record["_seen_stages"] = []
            record["_event_count"] = 0
        for key, value in seed.items():
            if key not in record:
                record[key] = value
                continue
            if record.get(key) in (None, "", []) and value not in (None, "", []):
                record[key] = value
        self.records[candidate_id] = record
        return record

    def seed_candidate(self, candidate: Optional[Dict[str, Any]] = None, **seed: Any) -> Optional[Dict[str, Any]]:
        candidate = candidate if isinstance(candidate, dict) else {}
        candidate_id = _text(seed.get("repair_candidate_id") or candidate.get("candidate_id") or candidate.get("repair_candidate_id")).strip()
        if not candidate_id:
            return None
        seed.setdefault("repair_candidate_id", candidate_id)
        seed.setdefault("bench_id", _text(candidate.get("bench_id")).strip() or self.bench_id)
        seed.setdefault("contract", _text(candidate.get("contract_name") or candidate.get("contract")).strip() or self.contract_name)
        seed.setdefault("source_trigger_id", _text(candidate.get("source_trigger_id") or candidate.get("failure_signature_id") or candidate.get("failure_signature")).strip())
        seed.setdefault("source_parent_id", _text(candidate.get("source_parent_id") or candidate.get("parent_candidate_id")).strip())
        seed.setdefault("source_scope_id", _text(candidate.get("source_scope_id") or candidate.get("scope_id") or candidate.get("source_task_id")).strip())
        seed.setdefault("source_candidate_kind", _text(candidate.get("source_candidate_kind") or candidate.get("candidate_type") or candidate.get("repair_action")).strip())
        return self._seed_record(candidate_id, **seed)

    def _log_event(self, record: Dict[str, Any], *, stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "stage": _text(stage).strip() or "unknown",
            "event_time": time.time(),
        }
        for field in self.BASE_FIELDS:
            payload[field] = record.get(field)
        if isinstance(extra, dict):
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value
        _append_jsonl(self.events_path, payload)
        record["_event_count"] = int(record.get("_event_count") or 0) + 1

    def _clear_stale_drop_reason(self, record: Dict[str, Any], stage: str) -> None:
        reason_text = _text(record.get("drop_reason")).strip()
        if not reason_text:
            return
        min_stage = self.STALE_DROP_REASON_MIN_STAGE.get(reason_text)
        current_stage = self.STAGE_ORDER.get(_text(stage).strip(), 0)
        if min_stage is None or current_stage < min_stage:
            return
        record["drop_reason"] = None
        self.counters["repair_candidate_dropped"] = max(0, int(self.counters.get("repair_candidate_dropped", 0)) - 1)
        if int(self.drop_reason_histogram.get(reason_text, 0)) > 0:
            self.drop_reason_histogram[reason_text] -= 1
            if int(self.drop_reason_histogram.get(reason_text, 0)) <= 0:
                self.drop_reason_histogram.pop(reason_text, None)

    def mark_stage(
        self,
        repair_candidate_id: Any,
        *,
        stage: str,
        stage_time: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        seed: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        record = self._seed_record(repair_candidate_id, **(seed or {}))
        if record is None:
            return None
        self._clear_stale_drop_reason(record, stage)
        field = self.STAGE_TO_FIELD.get(stage)
        if field and record.get(field) in (None, ""):
            record[field] = float(stage_time) if stage_time not in (None, "") else time.time()
        seen = set(record.get("_seen_stages") or [])
        if stage not in seen:
            counter_name = self.STAGE_TO_COUNTER.get(stage)
            if counter_name:
                self.counters[counter_name] += 1
            seen.add(stage)
            record["_seen_stages"] = sorted(seen)
        if isinstance(extra, dict):
            for key, value in extra.items():
                if value is None:
                    continue
                if key not in record or record.get(key) in (None, "", []):
                    record[key] = value
        self.records[_text(repair_candidate_id).strip()] = record
        self._log_event(record, stage=stage, extra=extra)
        return record

    def mark_materialized(self, candidate: Dict[str, Any], **extra: Any) -> Optional[Dict[str, Any]]:
        seed = dict(extra)
        return self.mark_stage(
            candidate.get("candidate_id"),
            stage="materialized",
            seed={
                "bench_id": _text(candidate.get("bench_id")).strip() or self.bench_id,
                "contract": _text(candidate.get("contract_name") or candidate.get("contract")).strip() or self.contract_name,
                "source_trigger_id": _text(candidate.get("source_trigger_id") or candidate.get("failure_signature_id") or candidate.get("failure_signature")).strip(),
                "source_parent_id": _text(candidate.get("source_parent_id") or candidate.get("parent_candidate_id")).strip(),
                "source_scope_id": _text(candidate.get("source_scope_id") or candidate.get("scope_id") or candidate.get("source_task_id")).strip(),
                "source_candidate_kind": _text(candidate.get("source_candidate_kind") or candidate.get("candidate_type") or candidate.get("repair_action")).strip(),
            },
            extra=seed,
        )

    def mark_dropped(self, repair_candidate_id: Any, *, reason: str, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        record = self._seed_record(repair_candidate_id)
        if record is None:
            return None
        reason_text = _text(reason).strip() or "unknown"
        if not _text(record.get("drop_reason")).strip():
            record["drop_reason"] = reason_text
            self.counters["repair_candidate_dropped"] += 1
            self.drop_reason_histogram[reason_text] += 1
        self.records[_text(repair_candidate_id).strip()] = record
        self._log_event(record, stage="dropped", extra=dict(extra or {}, drop_reason=record.get("drop_reason")))
        return record

    def mark_tx_result(self, repair_candidate_id: Any, *, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        record = self._seed_record(repair_candidate_id)
        if record is None:
            return None
        self._log_event(record, stage="tx_result_recorded", extra=extra)
        return record

    def mark_oracle_event(self, repair_candidate_id: Any, *, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        record = self._seed_record(repair_candidate_id)
        if record is None:
            return None
        self._log_event(record, stage="oracle_event_recorded", extra=extra)
        return record

    def get_record(self, repair_candidate_id: Any) -> Optional[Dict[str, Any]]:
        candidate_id = _text(repair_candidate_id).strip()
        record = self.records.get(candidate_id)
        return dict(record) if isinstance(record, dict) else None

    def finalize(self) -> Dict[str, Any]:
        for candidate_id, record in list(self.records.items()):
            if _text(record.get("drop_reason")).strip():
                continue
            if record.get("queued_at") and not record.get("admitted_at"):
                self.mark_dropped(candidate_id, reason="dropped_before_scheduler_admission")
            elif record.get("admitted_at") and not record.get("selected_at"):
                self.mark_dropped(candidate_id, reason="admitted_but_never_selected")
            elif record.get("selected_at") and not record.get("execution_submitted_at"):
                self.mark_dropped(candidate_id, reason="selected_but_never_submitted")
            elif record.get("execution_submitted_at") and not record.get("execution_started_at"):
                self.mark_dropped(candidate_id, reason="execution_submitted_but_not_started")
            elif record.get("execution_exception_at") and not record.get("executed_at"):
                self.mark_dropped(candidate_id, reason="execution_exception")
            elif record.get("execution_started_at") and not record.get("executed_at"):
                self.mark_dropped(candidate_id, reason="execution_started_but_not_completed")
            elif record.get("executed_at") and not record.get("attributed_at"):
                self.mark_dropped(candidate_id, reason="executed_but_not_attributed")
        rows: List[Dict[str, Any]] = []
        for candidate_id in sorted(self.records):
            record = dict(self.records[candidate_id])
            record.pop("_seen_stages", None)
            record.pop("_event_count", None)
            rows.append(record)
        summary = {
            "bench_id": self.bench_id,
            "contract": self.contract_name,
            "repair_candidate_materialized": int(self.counters.get("repair_candidate_materialized", 0)),
            "repair_candidate_queued": int(self.counters.get("repair_candidate_queued", 0)),
            "repair_candidate_admitted": int(self.counters.get("repair_candidate_admitted", 0)),
            "repair_candidate_selected": int(self.counters.get("repair_candidate_selected", 0)),
            "repair_candidate_execution_submitted": int(self.counters.get("repair_candidate_execution_submitted", 0)),
            "repair_candidate_execution_started": int(self.counters.get("repair_candidate_execution_started", 0)),
            "repair_candidate_execution_exception": int(self.counters.get("repair_candidate_execution_exception", 0)),
            "repair_candidate_executed": int(self.counters.get("repair_candidate_executed", 0)),
            "repair_candidate_attributed": int(self.counters.get("repair_candidate_attributed", 0)),
            "repair_candidate_dropped": int(self.counters.get("repair_candidate_dropped", 0)),
            "repair_candidate_drop_reason_histogram": dict(self.drop_reason_histogram),
            "rows": rows,
        }
        _write_json(self.summary_path, summary)
        return summary

