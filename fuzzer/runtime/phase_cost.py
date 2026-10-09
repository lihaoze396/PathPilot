from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _event_usage(item: Dict[str, Any]) -> Dict[str, int]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
    if not usage and isinstance(metadata.get("usage"), dict):
        usage = metadata.get("usage")
    return {
        "prompt_tokens": _safe_int(item.get("prompt_tokens", usage.get("prompt_tokens")), 0),
        "completion_tokens": _safe_int(item.get("completion_tokens", usage.get("completion_tokens")), 0),
        "total_tokens": _safe_int(item.get("total_tokens", usage.get("total_tokens")), 0),
    }


def _event_cost(item: Dict[str, Any]) -> float:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return _safe_float(item.get("cost_usd", metadata.get("cost_usd")), 0.0)


def _event_phase(item: Dict[str, Any]) -> str:
    phase = str(item.get("phase") or item.get("stage") or "").strip()
    return phase or "unknown"


def _event_kind(item: Dict[str, Any]) -> str:
    kind = str(item.get("event_kind") or "").strip().lower()
    if kind:
        return kind
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
    if usage or isinstance(metadata.get("usage"), dict):
        return "llm"
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "model"):
        if item.get(key) not in (None, ""):
            return "llm"
    return "generic"


def _ensure_state(env: Any) -> Dict[str, Any]:
    state = getattr(env, "phase_cost_state", None)
    if not isinstance(state, dict):
        state = {
            "started_at": float(time.time()),
            "resource_samples": [],
            "phase_events": [],
            "phase_summary": {},
            "resource_monitor": {},
        }
        setattr(env, "phase_cost_state", state)
    state.setdefault("started_at", float(time.time()))
    state.setdefault("resource_samples", [])
    state.setdefault("phase_events", [])
    state.setdefault("phase_summary", {})
    state.setdefault("resource_monitor", {})
    return state


def _process_cpu_total(proc: Any) -> Optional[float]:
    try:
        cpu_times = proc.cpu_times()
        return float(getattr(cpu_times, "user", 0.0)) + float(getattr(cpu_times, "system", 0.0))
    except Exception:
        return None


def _iter_process_tree(root_pid: int) -> List[Any]:
    if psutil is None:
        return []
    try:
        root = psutil.Process(root_pid)
    except Exception:
        return []
    try:
        procs = [root]
        procs.extend(root.children(recursive=True))
    except Exception:
        procs = [root]
    unique: List[Any] = []
    seen = set()
    for proc in procs:
        try:
            pid = int(proc.pid)
        except Exception:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(proc)
    return unique


def _collect_resource_sample(env: Any) -> Optional[Dict[str, float]]:
    if psutil is None:
        return None
    state = _ensure_state(env)
    monitor = state.get("resource_monitor") if isinstance(state.get("resource_monitor"), dict) else {}
    root_pid = _safe_int(monitor.get("root_pid"), os.getpid())
    procs = _iter_process_tree(root_pid)
    if not procs:
        return None

    now_wall = time.time()
    now_mono = time.monotonic()
    last_cpu_times = monitor.get("last_cpu_times") if isinstance(monitor.get("last_cpu_times"), dict) else {}
    monitor["last_cpu_times"] = last_cpu_times
    state["resource_monitor"] = monitor

    total_cpu = 0.0
    total_rss_mb = 0.0
    saw_cpu = False
    saw_rss = False
    live_pids = set()

    for proc in procs:
        try:
            pid = int(proc.pid)
        except Exception:
            continue
        live_pids.add(pid)

        total = _process_cpu_total(proc)
        previous = last_cpu_times.get(pid)
        if total is not None:
            last_cpu_times[pid] = (now_mono, total)
            if previous is not None:
                prev_ts, prev_total = previous
                elapsed = now_mono - _safe_float(prev_ts, now_mono)
                delta = total - _safe_float(prev_total, total)
                if elapsed > 0.0 and delta >= 0.0:
                    total_cpu += max((delta / elapsed) * 100.0, 0.0)
                    saw_cpu = True
        try:
            total_rss_mb += float(proc.memory_info().rss) / 1024.0 / 1024.0
            saw_rss = True
        except Exception:
            pass

    for pid in list(last_cpu_times.keys()):
        if pid not in live_pids:
            last_cpu_times.pop(pid, None)

    sample = {
        "timestamp": now_wall,
        "cpu_percent": total_cpu if saw_cpu else 0.0,
        "rss_mb": total_rss_mb if saw_rss else 0.0,
    }
    record_resource_sample(
        env,
        timestamp=sample["timestamp"],
        cpu_percent=sample["cpu_percent"],
        rss_mb=sample["rss_mb"],
    )
    return sample


def start_resource_sampler(env: Any, sample_interval_sec: float = 0.25) -> bool:
    state = _ensure_state(env)
    monitor = state.get("resource_monitor") if isinstance(state.get("resource_monitor"), dict) else {}
    state["resource_monitor"] = monitor
    if psutil is None:
        monitor["enabled"] = False
        return False

    thread = monitor.get("thread")
    if thread is not None and getattr(thread, "is_alive", lambda: False)():
        return True

    stop_event = threading.Event()
    interval_sec = max(0.1, _safe_float(sample_interval_sec, 0.25))
    monitor["enabled"] = True
    monitor["root_pid"] = int(os.getpid())
    monitor["interval_sec"] = interval_sec
    monitor["started_at"] = float(time.time())
    monitor["stop_event"] = stop_event
    monitor["last_cpu_times"] = {}
    state["started_at"] = float(monitor["started_at"])

    def _run() -> None:
        _collect_resource_sample(env)
        while not stop_event.wait(interval_sec):
            _collect_resource_sample(env)
        _collect_resource_sample(env)

    thread = threading.Thread(target=_run, name="phase-cost-resource-sampler", daemon=True)
    monitor["thread"] = thread
    thread.start()
    return True


def stop_resource_sampler(env: Any, join_timeout_sec: float = 2.0) -> bool:
    state = _ensure_state(env)
    monitor = state.get("resource_monitor") if isinstance(state.get("resource_monitor"), dict) else {}
    thread = monitor.get("thread")
    stop_event = monitor.get("stop_event")
    if stop_event is not None and hasattr(stop_event, "set"):
        try:
            stop_event.set()
        except Exception:
            pass
    if thread is not None and hasattr(thread, "join"):
        try:
            thread.join(max(0.1, _safe_float(join_timeout_sec, 2.0)))
        except Exception:
            pass
    _collect_resource_sample(env)
    stopped_at = float(time.time())
    monitor["stopped_at"] = stopped_at
    state["finished_at"] = stopped_at
    monitor["thread"] = None
    monitor["stop_event"] = None
    return True


def record_resource_sample(
    env: Any,
    *,
    timestamp: Optional[float] = None,
    cpu_percent: Any = None,
    rss_mb: Any = None,
) -> None:
    state = _ensure_state(env)
    sample = {
        "timestamp": _safe_float(timestamp, time.time()),
        "cpu_percent": _safe_float(cpu_percent, 0.0),
        "rss_mb": _safe_float(rss_mb, 0.0),
    }
    state["resource_samples"].append(sample)
    if len(state["resource_samples"]) > 20000:
        state["resource_samples"] = state["resource_samples"][-20000:]


def record_phase_event(env: Any, event: Dict[str, Any]) -> None:
    if not isinstance(event, dict):
        return
    state = _ensure_state(env)
    phase = _event_phase(event)
    started_at = _safe_float(event.get("started_at"), 0.0)
    ended_at = _safe_float(event.get("ended_at"), started_at)
    if ended_at < started_at:
        ended_at = started_at
    usage = _event_usage(event)
    payload = {
        "phase": phase,
        "event_kind": _event_kind(event),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": max(0.0, _safe_float(event.get("duration_sec"), ended_at - started_at)),
        "success": bool(event.get("success", not bool(str(event.get("error") or "").strip()))),
        "error": str(event.get("error") or "").strip(),
        "fallback_used": bool(event.get("fallback_used")),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost_usd": _event_cost(event),
        "model": str(event.get("model") or ((event.get("metadata") or {}).get("model") if isinstance(event.get("metadata"), dict) else "") or "").strip(),
        "metadata": dict(event.get("metadata")) if isinstance(event.get("metadata"), dict) else {},
    }
    state["phase_events"].append(payload)
    if len(state["phase_events"]) > 2000:
        state["phase_events"] = state["phase_events"][-2000:]


def record_phase_span(
    env: Any,
    *,
    phase: str,
    started_at: Optional[float],
    ended_at: Optional[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    record_phase_event(
        env,
        {
            "phase": phase,
            "event_kind": "phase_span",
            "started_at": started_at,
            "ended_at": ended_at,
            "success": True,
            "metadata": dict(metadata) if isinstance(metadata, dict) else {},
        },
    )


def _resource_slice(samples: List[Dict[str, Any]], start_ts: float, end_ts: float) -> List[Dict[str, Any]]:
    if end_ts < start_ts:
        end_ts = start_ts
    return [
        item for item in samples
        if isinstance(item, dict)
        and _safe_float(item.get("timestamp"), -1.0) >= start_ts
        and _safe_float(item.get("timestamp"), -1.0) <= end_ts
    ]


def _normalize_intervals(intervals: List[Any]) -> List[List[float]]:
    normalized: List[List[float]] = []
    for item in intervals:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start_ts = _safe_float(item[0], 0.0)
        end_ts = _safe_float(item[1], start_ts)
        if end_ts < start_ts:
            end_ts = start_ts
        normalized.append([start_ts, end_ts])
    normalized.sort(key=lambda pair: (pair[0], pair[1]))
    merged: List[List[float]] = []
    for start_ts, end_ts in normalized:
        if not merged or start_ts > merged[-1][1]:
            merged.append([start_ts, end_ts])
            continue
        merged[-1][1] = max(merged[-1][1], end_ts)
    return merged


def _interval_duration(intervals: List[List[float]]) -> float:
    return sum(max(0.0, end_ts - start_ts) for start_ts, end_ts in intervals)


def _overlap_duration(start_ts: float, end_ts: float, intervals: List[List[float]]) -> float:
    if end_ts <= start_ts:
        return 0.0
    total = 0.0
    for left, right in intervals:
        if right <= start_ts:
            continue
        if left >= end_ts:
            break
        total += max(0.0, min(end_ts, right) - max(start_ts, left))
    return total


def _subtract_intervals(base_intervals: List[List[float]], exclude_intervals: List[List[float]]) -> List[List[float]]:
    if not base_intervals:
        return []
    if not exclude_intervals:
        return [list(item) for item in base_intervals]
    remaining: List[List[float]] = []
    for start_ts, end_ts in base_intervals:
        cursor = start_ts
        for ex_start, ex_end in exclude_intervals:
            if ex_end <= cursor:
                continue
            if ex_start >= end_ts:
                break
            if ex_start > cursor:
                remaining.append([cursor, min(ex_start, end_ts)])
            cursor = max(cursor, ex_end)
            if cursor >= end_ts:
                break
        if cursor < end_ts:
            remaining.append([cursor, end_ts])
    return _normalize_intervals(remaining)


def _samples_for_intervals(samples: List[Dict[str, Any]], intervals: List[List[float]]) -> List[Dict[str, Any]]:
    if not intervals:
        return []
    selected: List[Dict[str, Any]] = []
    for item in samples:
        timestamp = _safe_float(item.get("timestamp"), -1.0)
        for start_ts, end_ts in intervals:
            if timestamp < start_ts:
                continue
            if timestamp > end_ts:
                continue
            selected.append(item)
            break
    selected.sort(key=lambda item: _safe_float(item.get("timestamp"), 0.0))
    return selected


def _estimate_cpu_time(samples: List[Dict[str, Any]], intervals: List[List[float]]) -> float:
    if len(samples) < 2 or not intervals:
        return 0.0
    ordered = sorted(samples, key=lambda item: _safe_float(item.get("timestamp"), 0.0))
    total_cpu_time = 0.0
    for prev, curr in zip(ordered, ordered[1:]):
        seg_start = _safe_float(prev.get("timestamp"), 0.0)
        seg_end = _safe_float(curr.get("timestamp"), seg_start)
        overlap = _overlap_duration(seg_start, seg_end, intervals)
        if overlap <= 0.0:
            continue
        total_cpu_time += overlap * max(0.0, _safe_float(prev.get("cpu_percent"), 0.0) / 100.0)
    return total_cpu_time


def _summarize_resources(samples: List[Dict[str, Any]], intervals: List[List[float]]) -> Dict[str, Any]:
    selected = _samples_for_intervals(samples, intervals)
    cpu_values = [_safe_float(item.get("cpu_percent"), 0.0) for item in selected]
    rss_values = [_safe_float(item.get("rss_mb"), 0.0) for item in selected]
    return {
        "wall_time_sec": round(_interval_duration(intervals), 6),
        "cpu_time_sec_est": round(_estimate_cpu_time(samples, intervals), 6),
        "cpu_avg_percent": round(sum(cpu_values) / len(cpu_values), 6) if cpu_values else 0.0,
        "cpu_peak_percent": round(max(cpu_values), 6) if cpu_values else 0.0,
        "memory_avg_rss_mb": round(sum(rss_values) / len(rss_values), 6) if rss_values else 0.0,
        "memory_peak_rss_mb": round(max(rss_values), 6) if rss_values else 0.0,
        "memory_final_rss_mb": round(rss_values[-1], 6) if rss_values else 0.0,
        "sample_count": len(selected),
        "interval_count": len(intervals),
    }


def _aggregate_llm_stats(llm_events: List[Dict[str, Any]], intervals: List[List[float]]) -> Dict[str, Any]:
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    total_cost = 0.0
    llm_time_sec = 0.0
    llm_event_count = 0
    llm_success_count = 0
    llm_error_count = 0
    models = set()
    for item in llm_events:
        started_at = _safe_float(item.get("started_at"), 0.0)
        ended_at = _safe_float(item.get("ended_at"), started_at)
        duration_sec = max(0.0, _safe_float(item.get("duration_sec"), ended_at - started_at))
        overlap = _overlap_duration(started_at, ended_at, intervals)
        if duration_sec <= 0.0:
            overlap = 1.0 if _overlap_duration(started_at, started_at + 1e-9, intervals) > 0.0 else 0.0
        if overlap <= 0.0:
            continue
        ratio = 1.0 if duration_sec <= 0.0 else min(overlap / duration_sec, 1.0)
        llm_event_count += 1
        llm_success_count += int(bool(item.get("success")))
        llm_error_count += int(bool(str(item.get("error") or "").strip()))
        llm_time_sec += overlap if duration_sec > 0.0 else 0.0
        prompt_tokens += _safe_int(item.get("prompt_tokens"), 0) * ratio
        completion_tokens += _safe_int(item.get("completion_tokens"), 0) * ratio
        total_tokens += _safe_int(item.get("total_tokens"), 0) * ratio
        total_cost += _safe_float(item.get("cost_usd"), 0.0) * ratio
        model = str(item.get("model") or "").strip()
        if model:
            models.add(model)
    return {
        "llm_event_count": llm_event_count,
        "llm_success_count": llm_success_count,
        "llm_error_count": llm_error_count,
        "llm_time_sec": round(llm_time_sec, 6),
        "prompt_tokens": int(round(prompt_tokens)),
        "completion_tokens": int(round(completion_tokens)),
        "total_tokens": int(round(total_tokens)),
        "cost_usd": round(total_cost, 8),
        "models": sorted(models),
    }


def _summarize_llm_phase(phase: str, events: List[Dict[str, Any]], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    intervals = _normalize_intervals(
        [(_safe_float(item.get("started_at"), 0.0), _safe_float(item.get("ended_at"), _safe_float(item.get("started_at"), 0.0))) for item in events]
    )
    resource_summary = _summarize_resources(samples, intervals)
    total_wall = sum(_safe_float(item.get("duration_sec"), 0.0) for item in events)
    total_prompt = sum(_safe_int(item.get("prompt_tokens"), 0) for item in events)
    total_completion = sum(_safe_int(item.get("completion_tokens"), 0) for item in events)
    total_tokens = sum(_safe_int(item.get("total_tokens"), 0) for item in events)
    total_cost = sum(_safe_float(item.get("cost_usd"), 0.0) for item in events)
    return {
        "phase": phase,
        "phase_kind": "llm",
        "event_count": len(events),
        "success_count": sum(1 for item in events if bool(item.get("success"))),
        "error_count": sum(1 for item in events if str(item.get("error") or "").strip()),
        "wall_time_sec": round(total_wall, 6),
        "llm_time_sec": round(total_wall, 6),
        "llm_event_count": len(events),
        "llm_success_count": sum(1 for item in events if bool(item.get("success"))),
        "llm_error_count": sum(1 for item in events if str(item.get("error") or "").strip()),
        "cpu_time_sec_est": resource_summary["cpu_time_sec_est"],
        "cpu_avg_percent": resource_summary["cpu_avg_percent"],
        "cpu_peak_percent": resource_summary["cpu_peak_percent"],
        "memory_avg_rss_mb": resource_summary["memory_avg_rss_mb"],
        "memory_peak_rss_mb": resource_summary["memory_peak_rss_mb"],
        "memory_final_rss_mb": resource_summary["memory_final_rss_mb"],
        "sample_count": resource_summary["sample_count"],
        "interval_count": resource_summary["interval_count"],
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "cost_usd": round(total_cost, 8),
        "models": sorted({str(item.get("model") or "").strip() for item in events if str(item.get("model") or "").strip()}),
    }


def _summarize_phase_spans(
    phase: str,
    events: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    llm_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    intervals = _normalize_intervals(
        [(_safe_float(item.get("started_at"), 0.0), _safe_float(item.get("ended_at"), _safe_float(item.get("started_at"), 0.0))) for item in events]
    )
    resource_summary = _summarize_resources(samples, intervals)
    llm_summary = _aggregate_llm_stats(llm_events, intervals)
    return {
        "phase": phase,
        "phase_kind": "coarse",
        "event_count": len(events),
        "success_count": sum(1 for item in events if bool(item.get("success"))),
        "error_count": sum(1 for item in events if str(item.get("error") or "").strip()),
        "wall_time_sec": resource_summary["wall_time_sec"],
        "llm_time_sec": llm_summary["llm_time_sec"],
        "llm_event_count": llm_summary["llm_event_count"],
        "llm_success_count": llm_summary["llm_success_count"],
        "llm_error_count": llm_summary["llm_error_count"],
        "cpu_time_sec_est": resource_summary["cpu_time_sec_est"],
        "cpu_avg_percent": resource_summary["cpu_avg_percent"],
        "cpu_peak_percent": resource_summary["cpu_peak_percent"],
        "memory_avg_rss_mb": resource_summary["memory_avg_rss_mb"],
        "memory_peak_rss_mb": resource_summary["memory_peak_rss_mb"],
        "memory_final_rss_mb": resource_summary["memory_final_rss_mb"],
        "sample_count": resource_summary["sample_count"],
        "interval_count": resource_summary["interval_count"],
        "prompt_tokens": llm_summary["prompt_tokens"],
        "completion_tokens": llm_summary["completion_tokens"],
        "total_tokens": llm_summary["total_tokens"],
        "cost_usd": llm_summary["cost_usd"],
        "models": llm_summary["models"],
    }


def _summarize_derived_phase(
    phase: str,
    phase_kind: str,
    intervals: List[List[float]],
    samples: List[Dict[str, Any]],
    llm_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    resource_summary = _summarize_resources(samples, intervals)
    llm_summary = _aggregate_llm_stats(llm_events, intervals)
    return {
        "phase": phase,
        "phase_kind": phase_kind,
        "event_count": 1 if intervals else 0,
        "success_count": 1 if intervals else 0,
        "error_count": 0,
        "wall_time_sec": resource_summary["wall_time_sec"],
        "llm_time_sec": llm_summary["llm_time_sec"],
        "llm_event_count": llm_summary["llm_event_count"],
        "llm_success_count": llm_summary["llm_success_count"],
        "llm_error_count": llm_summary["llm_error_count"],
        "cpu_time_sec_est": resource_summary["cpu_time_sec_est"],
        "cpu_avg_percent": resource_summary["cpu_avg_percent"],
        "cpu_peak_percent": resource_summary["cpu_peak_percent"],
        "memory_avg_rss_mb": resource_summary["memory_avg_rss_mb"],
        "memory_peak_rss_mb": resource_summary["memory_peak_rss_mb"],
        "memory_final_rss_mb": resource_summary["memory_final_rss_mb"],
        "sample_count": resource_summary["sample_count"],
        "interval_count": resource_summary["interval_count"],
        "prompt_tokens": llm_summary["prompt_tokens"],
        "completion_tokens": llm_summary["completion_tokens"],
        "total_tokens": llm_summary["total_tokens"],
        "cost_usd": llm_summary["cost_usd"],
        "models": llm_summary["models"],
    }


def build_phase_summary(env: Any) -> Dict[str, Any]:
    state = _ensure_state(env)
    events = [item for item in list(state.get("phase_events") or []) if isinstance(item, dict)]
    samples = [item for item in list(state.get("resource_samples") or []) if isinstance(item, dict)]
    monitor = state.get("resource_monitor") if isinstance(state.get("resource_monitor"), dict) else {}
    llm_events = [item for item in events if _event_kind(item) == "llm"]
    span_events = [item for item in events if _event_kind(item) == "phase_span"]
    grouped_llm: Dict[str, List[Dict[str, Any]]] = {}
    grouped_spans: Dict[str, List[Dict[str, Any]]] = {}
    for item in llm_events:
        grouped_llm.setdefault(_event_phase(item), []).append(item)
    for item in span_events:
        grouped_spans.setdefault(_event_phase(item), []).append(item)
    phases: Dict[str, Any] = {}
    for phase in sorted(grouped_spans):
        phases[phase] = _summarize_phase_spans(phase, grouped_spans[phase], samples, llm_events)
    for phase in sorted(grouped_llm):
        phases[phase] = _summarize_llm_phase(phase, grouped_llm[phase], samples)

    overall_start = _safe_float(state.get("started_at"), 0.0)
    overall_end = _safe_float(
        state.get("finished_at"),
        _safe_float(samples[-1].get("timestamp"), overall_start) if samples else overall_start,
    )
    if overall_end < overall_start:
        overall_end = overall_start
    overall_intervals = _normalize_intervals([(overall_start, overall_end)]) if overall_end >= overall_start else []
    overall_summary = _summarize_derived_phase("total_run", "overall", overall_intervals, samples, llm_events)
    phases["total_run"] = overall_summary

    fuzz_core_phase = phases.get("fuzz_core") if isinstance(phases.get("fuzz_core"), dict) else {}
    fuzz_core_intervals = _normalize_intervals(
        [(_safe_float(item.get("started_at"), 0.0), _safe_float(item.get("ended_at"), _safe_float(item.get("started_at"), 0.0))) for item in grouped_spans.get("fuzz_core", [])]
    )
    llm_intervals = _normalize_intervals(
        [(_safe_float(item.get("started_at"), 0.0), _safe_float(item.get("ended_at"), _safe_float(item.get("started_at"), 0.0))) for item in llm_events]
    )
    pure_fuzz_intervals = _subtract_intervals(fuzz_core_intervals, llm_intervals)
    pure_fuzz_summary = _summarize_derived_phase("pure_fuzz", "derived_pure_fuzz", pure_fuzz_intervals, samples, [])
    if fuzz_core_phase:
        pure_fuzz_summary["llm_time_sec"] = 0.0
        pure_fuzz_summary["llm_event_count"] = 0
        pure_fuzz_summary["llm_success_count"] = 0
        pure_fuzz_summary["llm_error_count"] = 0
        pure_fuzz_summary["prompt_tokens"] = 0
        pure_fuzz_summary["completion_tokens"] = 0
        pure_fuzz_summary["total_tokens"] = 0
        pure_fuzz_summary["cost_usd"] = 0.0
        pure_fuzz_summary["models"] = []
        phases["pure_fuzz"] = pure_fuzz_summary
    summary = {
        "enabled": True,
        "resource_sampling_enabled": bool(monitor.get("enabled", False)),
        "resource_sample_count": len(samples),
        "resource_sample_interval_sec": round(_safe_float(monitor.get("interval_sec"), 0.0), 6),
        "phase_count": len(phases),
        "events_total": len(events),
        "overall": overall_summary,
        "pure_fuzz": pure_fuzz_summary if fuzz_core_phase else {},
        "phases": phases,
    }
    state["phase_summary"] = summary
    return summary
