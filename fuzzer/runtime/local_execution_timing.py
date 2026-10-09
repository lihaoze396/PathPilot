"""Clean local EVM/transaction execution timing for pathpilot runs.

This timer records only wall time spent inside local EVM transaction/deploy
calls. It is intentionally separate from configured fuzz budgets, legacy active
runtime, and non-LLM residual wall time.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional, TypeVar


T = TypeVar("T")


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


def _phase_name(value: Any) -> str:
    text = str(value or "").strip()
    return text or "unknown"


def ensure_local_execution_timing_state(env: Any) -> Dict[str, Any]:
    state = getattr(env, "local_execution_timing_state", None)
    if not isinstance(state, dict):
        state = {
            "events": [],
            "by_phase": {},
            "total_time_sec": 0.0,
            "total_count": 0,
            "success_count": 0,
            "error_count": 0,
            "sidecar_last_flush_monotonic": 0.0,
            "sidecar_last_flush_count": 0,
        }
        setattr(env, "local_execution_timing_state", state)
    state.setdefault("events", [])
    state.setdefault("by_phase", {})
    state.setdefault("total_time_sec", 0.0)
    state.setdefault("total_count", 0)
    state.setdefault("success_count", 0)
    state.setdefault("error_count", 0)
    state.setdefault("sidecar_last_flush_monotonic", 0.0)
    state.setdefault("sidecar_last_flush_count", 0)
    return state


def record_local_tx_execution(
    env: Any,
    *,
    phase: str,
    tx_kind: str,
    duration_sec: Any,
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    success: bool = True,
    error: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = ensure_local_execution_timing_state(env)
    phase = _phase_name(phase)
    duration = max(0.0, _safe_float(duration_sec, 0.0))
    ended = _safe_float(ended_at, time.time())
    started = _safe_float(started_at, ended - duration)
    event = {
        "phase": phase,
        "tx_kind": str(tx_kind or "transaction"),
        "started_at": started,
        "ended_at": ended,
        "duration_sec": duration,
        "success": bool(success),
        "error": str(error or ""),
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
    }
    state["events"].append(event)
    if len(state["events"]) > 5000:
        state["events"] = state["events"][-5000:]

    state["total_time_sec"] = _safe_float(state.get("total_time_sec"), 0.0) + duration
    state["total_count"] = int(state.get("total_count") or 0) + 1
    if success:
        state["success_count"] = int(state.get("success_count") or 0) + 1
    else:
        state["error_count"] = int(state.get("error_count") or 0) + 1

    by_phase = state.get("by_phase") if isinstance(state.get("by_phase"), dict) else {}
    phase_payload = by_phase.setdefault(
        phase,
        {
            "phase": phase,
            "time_sec": 0.0,
            "count": 0,
            "success_count": 0,
            "error_count": 0,
        },
    )
    phase_payload["time_sec"] = _safe_float(phase_payload.get("time_sec"), 0.0) + duration
    phase_payload["count"] = int(phase_payload.get("count") or 0) + 1
    if success:
        phase_payload["success_count"] = int(phase_payload.get("success_count") or 0) + 1
    else:
        phase_payload["error_count"] = int(phase_payload.get("error_count") or 0) + 1
    state["by_phase"] = by_phase
    try:
        persist_local_execution_timing_summary(env, force=False)
    except Exception:
        pass
    return event


def time_local_execution(
    env: Any,
    *,
    phase: str,
    tx_kind: str,
    func: Callable[[], T],
    metadata: Optional[Dict[str, Any]] = None,
) -> T:
    started_at = time.time()
    started_perf = time.perf_counter()
    success = False
    error = ""
    result: Any = None
    try:
        result = func()
        success = not bool(getattr(result, "is_error", False))
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ended_at = time.time()
        duration = time.perf_counter() - started_perf
        record_local_tx_execution(
            env,
            phase=phase,
            tx_kind=tx_kind,
            duration_sec=duration,
            started_at=started_at,
            ended_at=ended_at,
            success=success,
            error=error,
            metadata=metadata,
        )


def _phase_payload(state: Dict[str, Any], phase: str) -> Dict[str, Any]:
    by_phase = state.get("by_phase") if isinstance(state.get("by_phase"), dict) else {}
    payload = by_phase.get(phase) if isinstance(by_phase.get(phase), dict) else {}
    return payload


def _phase_time(state: Dict[str, Any], phase: str) -> float:
    return _safe_float(_phase_payload(state, phase).get("time_sec"), 0.0)


def _phase_count(state: Dict[str, Any], phase: str) -> int:
    try:
        return int(_phase_payload(state, phase).get("count") or 0)
    except Exception:
        return 0


def build_local_execution_timing_summary(env: Any) -> Dict[str, Any]:
    state = ensure_local_execution_timing_state(env)
    total_count = int(state.get("total_count") or 0)
    by_phase = state.get("by_phase") if isinstance(state.get("by_phase"), dict) else {}
    phase_breakdown = {
        phase: {
            "phase": phase,
            "time_sec": round(_safe_float(payload.get("time_sec"), 0.0), 6),
            "count": int(payload.get("count") or 0),
            "success_count": int(payload.get("success_count") or 0),
            "error_count": int(payload.get("error_count") or 0),
        }
        for phase, payload in sorted(by_phase.items())
        if isinstance(payload, dict)
    }
    agent_time = sum(
        _safe_float(payload.get("time_sec"), 0.0)
        for phase, payload in by_phase.items()
        if isinstance(payload, dict) and (str(phase).startswith("agent_") or "pilot" in str(phase))
    )
    agent_count = sum(
        int(payload.get("count") or 0)
        for phase, payload in by_phase.items()
        if isinstance(payload, dict) and (str(phase).startswith("agent_") or "pilot" in str(phase))
    )
    total_time = _safe_float(state.get("total_time_sec"), 0.0)
    has_timer = total_count > 0
    return {
        "has_clean_local_tx_execution_timer": bool(has_timer),
        "clean_local_tx_execution_time_sec": round(total_time, 6) if has_timer else None,
        "clean_local_tx_execution_count": total_count,
        "clean_local_fuzz_tx_execution_time_sec": round(_phase_time(state, "fuzz_core"), 6) if has_timer else None,
        "clean_local_fuzz_tx_execution_count": _phase_count(state, "fuzz_core"),
        "clean_local_setup_tx_execution_time_sec": round(_phase_time(state, "setup_deploy"), 6) if has_timer else None,
        "clean_local_setup_tx_execution_count": _phase_count(state, "setup_deploy"),
        "clean_local_agent_tx_execution_time_sec": round(agent_time, 6) if has_timer else None,
        "clean_local_agent_tx_execution_count": agent_count,
        "success_count": int(state.get("success_count") or 0),
        "error_count": int(state.get("error_count") or 0),
        "phase_breakdown": phase_breakdown,
        "events_sample": list(state.get("events") or [])[-128:],
    }


def _resolve_local_execution_timing_output_dir(env: Any) -> str:
    args = getattr(env, "args", None)
    for candidate in (
        getattr(args, "run_output_dir", None),
    ):
        if candidate:
            try:
                return os.path.abspath(str(candidate))
            except Exception:
                continue
    results_path = getattr(args, "results", None)
    if results_path:
        try:
            results_text = os.path.abspath(str(results_path))
            if os.path.splitext(results_text)[1].lower() in {".json", ".csv"}:
                return os.path.dirname(results_text)
            return results_text
        except Exception:
            pass
    runtime = getattr(env, "runtime", None)
    output_dir = getattr(runtime, "output_dir", None)
    if output_dir:
        try:
            return os.path.abspath(str(output_dir))
        except Exception:
            return ""
    return ""


def _local_execution_timing_sidecar_path(env: Any) -> str:
    output_dir = _resolve_local_execution_timing_output_dir(env)
    if not output_dir:
        return ""
    return os.path.join(output_dir, "local_execution_timing_summary.json")


def persist_local_execution_timing_summary(env: Any, force: bool = False) -> str:
    state = ensure_local_execution_timing_state(env)
    path = _local_execution_timing_sidecar_path(env)
    if not path:
        return ""
    total_count = _safe_int(state.get("total_count"), 0)
    if total_count <= 0 and not force:
        return ""

    now_monotonic = time.monotonic()
    last_flush_monotonic = _safe_float(state.get("sidecar_last_flush_monotonic"), 0.0)
    last_flush_count = _safe_int(state.get("sidecar_last_flush_count"), 0)
    if not force and (total_count - last_flush_count) < 5 and (now_monotonic - last_flush_monotonic) < 1.0:
        return path

    payload = build_local_execution_timing_summary(env)
    payload["updated_at_unix"] = round(time.time(), 6)
    payload["sidecar_version"] = 1
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        os.replace(temp_path, path)
        state["sidecar_last_flush_monotonic"] = now_monotonic
        state["sidecar_last_flush_count"] = total_count
        return path
    except Exception:
        return ""


def refresh_local_execution_timing_results(env: Any) -> Dict[str, Any]:
    summary = build_local_execution_timing_summary(env)
    results = getattr(env, "results", None)
    if isinstance(results, dict):
        results["local_execution_timing"] = summary
        for key in (
            "has_clean_local_tx_execution_timer",
            "clean_local_tx_execution_time_sec",
            "clean_local_tx_execution_count",
            "clean_local_fuzz_tx_execution_time_sec",
            "clean_local_fuzz_tx_execution_count",
            "clean_local_setup_tx_execution_time_sec",
            "clean_local_setup_tx_execution_count",
            "clean_local_agent_tx_execution_time_sec",
            "clean_local_agent_tx_execution_count",
        ):
            results[key] = summary.get(key)
    sidecar_path = persist_local_execution_timing_summary(env, force=True)
    if isinstance(results, dict) and sidecar_path:
        results["local_execution_timing_summary_path"] = sidecar_path
    return summary
