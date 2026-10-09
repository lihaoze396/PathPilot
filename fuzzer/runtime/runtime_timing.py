import time
from typing import Any, Dict, Optional


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


def _clean_budget_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "fuzz-active":
        return "fuzz-active"
    return "wall"


def _agent_state(env: Any) -> Dict[str, Any]:
    state = getattr(env, "fresh_agent_runtime_state", None)
    return state if isinstance(state, dict) else {}


def _local_execution_summary(env: Any) -> Dict[str, Any]:
    state = getattr(env, "local_execution_timing_state", None)
    if not isinstance(state, dict):
        return {}
    try:
        from fuzzer.runtime.local_execution_timing import build_local_execution_timing_summary

        payload = build_local_execution_timing_summary(env)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _path_validation_live_summary(env: Any) -> Dict[str, Any]:
    tracker = getattr(env, "path_progress_tracker", None)
    if tracker is not None and hasattr(tracker, "get_summary"):
        try:
            payload = tracker.get_summary()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    results = getattr(env, "results", None)
    if isinstance(results, dict):
        payload = results.get("path_validation")
        if isinstance(payload, dict):
            return payload
    return {}


def _code_coverage_live_summary(env: Any) -> Dict[str, Any]:
    overall_pcs = list(getattr(env, "overall_pcs", []) or [])
    code_coverage = set(getattr(env, "code_coverage", set()) or set())
    covered = len(code_coverage)
    total = len(overall_pcs)
    percentage = (float(covered) / float(total) * 100.0) if total > 0 else 0.0
    payload: Dict[str, Any] = {
        "percentage": percentage,
        "covered": covered,
        "total": total,
    }
    children_code_coverage = dict(getattr(env, "children_code_coverage", {}) or {})
    if children_code_coverage:
        covered_with_children = covered
        for child_cov in children_code_coverage.values():
            try:
                covered_with_children += len(child_cov or [])
            except Exception:
                continue
        payload["covered_with_children"] = covered_with_children
    total_with_children = getattr(env, "len_overall_pcs_with_children", None)
    if total_with_children not in (None, ""):
        payload["total_with_children"] = _safe_int(total_with_children, 0)
    results = getattr(env, "results", None)
    if isinstance(results, dict):
        existing = results.get("code_coverage")
        if isinstance(existing, dict):
            if existing.get("covered_with_children") not in (None, ""):
                payload["covered_with_children"] = existing.get("covered_with_children")
            if existing.get("total_with_children") not in (None, ""):
                payload["total_with_children"] = existing.get("total_with_children")
            if existing.get("total_cov") not in (None, ""):
                payload["total_cov"] = existing.get("total_cov")
    total_cov = _snapshot_total_cov(payload)
    if total_cov is not None:
        payload["total_cov"] = total_cov
    return payload


def _snapshot_total_cov(code_coverage: Dict[str, Any]) -> Optional[float]:
    if not isinstance(code_coverage, dict):
        return None
    if code_coverage.get("total_cov") not in (None, ""):
        return _safe_float(code_coverage.get("total_cov"), 0.0)
    covered = code_coverage.get("covered_with_children")
    total = code_coverage.get("total_with_children")
    if covered in (None, "") or total in (None, ""):
        return None
    total_value = _safe_float(total, 0.0)
    if total_value <= 0.0:
        return None
    return (_safe_float(covered, 0.0) / total_value) * 100.0


def _branch_coverage_live_summary(env: Any) -> Dict[str, Any]:
    covered = len(set(getattr(env, "branch_coverage", set()) or set()))
    total = len(list(getattr(env, "overall_jumpis", []) or [])) * 2
    percentage = (float(covered) / float(total) * 100.0) if total > 0 else 0.0
    return {"percentage": percentage, "covered": covered, "total": total}


def _function_coverage_live_summary(env: Any) -> Dict[str, Any]:
    interface = getattr(env, "interface", {}) or {}
    total_selectors = {
        str(selector).lower().replace("0x", "")
        for selector in interface.keys()
        if selector not in ("constructor", "fallback")
    }
    covered_selectors = {
        str(selector).lower().replace("0x", "")
        for selector in set(getattr(env, "function_coverage", set()) or set())
        if str(selector).lower().replace("0x", "") in total_selectors
    }
    total = len(total_selectors)
    covered = len(covered_selectors)
    percentage = (float(covered) / float(total) * 100.0) if total > 0 else 0.0
    return {"percentage": percentage, "covered": covered, "total": total}


def ensure_timing_state(env: Any, args: Any = None) -> Dict[str, Any]:
    state = getattr(env, "experiment_timing_state", None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, "experiment_timing_state", state)

    state.setdefault("budget_mode", "wall")
    state.setdefault("budget_limit_sec", None)
    state.setdefault("budget_fuzz_active_sec", None)
    state.setdefault("wall_snapshot_sec", None)
    state.setdefault("wall_clock_start_time", None)
    state.setdefault("wall_snapshot_recorded", False)
    state.setdefault("wall_snapshot", None)
    state.setdefault("final_snapshot", None)
    state.setdefault("final_wall_clock_elapsed_sec", None)
    state.setdefault("final_fuzz_active_elapsed_sec", None)

    if args is not None:
        budget_mode = _clean_budget_mode(getattr(args, "budget_mode", None))
        global_timeout = getattr(args, "global_timeout", None)
        fuzz_active_budget = getattr(args, "fuzz_active_budget_sec", None)
        wall_snapshot_sec = getattr(args, "wall_snapshot_sec", None)

        budget_limit_sec = None
        if budget_mode == "fuzz-active":
            budget_limit_sec = fuzz_active_budget if fuzz_active_budget not in (None, "") else global_timeout
        else:
            budget_limit_sec = global_timeout

        state["budget_mode"] = budget_mode
        state["budget_limit_sec"] = (
            _safe_float(budget_limit_sec, 0.0) if budget_limit_sec not in (None, "") else None
        )
        state["budget_fuzz_active_sec"] = (
            _safe_float(
                fuzz_active_budget if fuzz_active_budget not in (None, "") else budget_limit_sec,
                0.0,
            )
            if budget_mode == "fuzz-active" and budget_limit_sec not in (None, "")
            else None
        )
        state["wall_snapshot_sec"] = (
            _safe_float(wall_snapshot_sec, 0.0) if wall_snapshot_sec not in (None, "") else None
        )

    return state


def start_experiment_timing(env: Any, args: Any = None, start_time: Optional[float] = None) -> Dict[str, Any]:
    state = ensure_timing_state(env, args)
    started_at = _safe_float(start_time, 0.0) if start_time not in (None, "") else time.time()
    if state.get("wall_clock_start_time") in (None, ""):
        state["wall_clock_start_time"] = float(started_at)
    if getattr(env, "execution_begin", None) in (None, ""):
        env.execution_begin = float(state["wall_clock_start_time"])
    return state


def get_current_time_metrics(env: Any, now: Optional[float] = None) -> Dict[str, Any]:
    state = ensure_timing_state(env)
    current_time = _safe_float(now, 0.0) if now not in (None, "") else time.time()
    wall_start_time = state.get("wall_clock_start_time")
    if wall_start_time in (None, ""):
        wall_start_time = getattr(env, "execution_begin", None)
    wall_start_value = _safe_float(wall_start_time, current_time)
    wall_elapsed_sec = max(0.0, float(current_time) - wall_start_value)

    agent_state = _agent_state(env)
    base_agent_interaction_total_sec = _safe_float(
        agent_state.get("agent_interaction_total_sec", agent_state.get("llm_interaction_time_total_sec")),
        0.0,
    )
    base_agent_interaction_count = _safe_int(
        agent_state.get("agent_interaction_count", len(agent_state.get("llm_interaction_events") or [])),
        0,
    )
    agent_interaction_total_sec = base_agent_interaction_total_sec
    agent_interaction_count = base_agent_interaction_count
    fuzz_active_elapsed_sec = max(0.0, wall_elapsed_sec - agent_interaction_total_sec)
    local_execution = _local_execution_summary(env)
    clean_local_fuzz_elapsed_sec = _safe_float(
        local_execution.get("clean_local_fuzz_tx_execution_time_sec"),
        0.0,
    )
    return {
        "wall_clock_start_time": wall_start_value,
        "wall_elapsed_sec": wall_elapsed_sec,
        "agent_interaction_total_sec": agent_interaction_total_sec,
        "agent_interaction_count": agent_interaction_count,
        "fuzz_active_elapsed_sec": fuzz_active_elapsed_sec,
        "clean_local_fuzz_elapsed_sec": clean_local_fuzz_elapsed_sec,
        "budget_mode": state.get("budget_mode", "wall"),
        "budget_limit_sec": state.get("budget_limit_sec"),
        "budget_fuzz_active_sec": state.get("budget_fuzz_active_sec"),
        "wall_snapshot_sec": state.get("wall_snapshot_sec"),
    }


def has_time_budget(env: Any, fallback_timeout: Any = None) -> bool:
    metrics = get_current_time_metrics(env)
    if metrics.get("budget_limit_sec") not in (None, "", 0, 0.0):
        return True
    return fallback_timeout not in (None, "", 0, 0.0)


def build_live_snapshot(env: Any, now: Optional[float] = None) -> Dict[str, Any]:
    metrics = get_current_time_metrics(env, now=now)
    code_coverage = _code_coverage_live_summary(env)
    branch_coverage = _branch_coverage_live_summary(env)
    function_coverage = _function_coverage_live_summary(env)
    path_validation = _path_validation_live_summary(env)
    total_tx_count = _safe_int(getattr(env, "nr_of_transactions", 0), 0)
    setup_tx_count = _safe_int(getattr(env, "setup_transactions", 0), 0)
    sequence_count = getattr(env, "_path_progress_sequence_counter", None)
    if sequence_count in (None, ""):
        tracker = getattr(env, "path_progress_tracker", None)
        sequence_count = getattr(tracker, "sequence_counter", None) if tracker is not None else None

    snapshot: Dict[str, Any] = {
        "wall_elapsed_sec": metrics["wall_elapsed_sec"],
        "fuzz_active_elapsed_sec": metrics["fuzz_active_elapsed_sec"],
        "clean_local_fuzz_elapsed_sec": metrics.get("clean_local_fuzz_elapsed_sec"),
        "agent_interaction_total_sec": metrics["agent_interaction_total_sec"],
        "agent_interaction_count": metrics["agent_interaction_count"],
        "coverage": code_coverage.get("percentage"),
        "main_cov": code_coverage.get("percentage"),
        "branch_cov": branch_coverage.get("percentage"),
        "branch_covered": branch_coverage.get("covered"),
        "branch_total": branch_coverage.get("total"),
        "function_cov": function_coverage.get("percentage"),
        "function_covered": function_coverage.get("covered"),
        "function_total": function_coverage.get("total"),
        "path_progress_count": path_validation.get("path_progress_records"),
        "sink_touched_count": path_validation.get("sink_function_reached_count"),
        "revert_count": path_validation.get("blocked_count"),
        "total_tx_count": total_tx_count,
        "fuzz_tx_count": max(total_tx_count - setup_tx_count, 0),
        "sequence_count": _safe_int(sequence_count, 0) if sequence_count not in (None, "") else None,
    }
    total_cov = _snapshot_total_cov(code_coverage)
    if total_cov is not None:
        snapshot["total_cov"] = total_cov
    return snapshot


def record_wall_snapshot_if_needed(env: Any, now: Optional[float] = None) -> bool:
    state = ensure_timing_state(env)
    snapshot_sec = state.get("wall_snapshot_sec")
    if snapshot_sec in (None, "", 0, 0.0) or bool(state.get("wall_snapshot_recorded")):
        return False
    metrics = get_current_time_metrics(env, now=now)
    if float(metrics.get("wall_elapsed_sec") or 0.0) < float(snapshot_sec):
        return False
    snapshot = build_live_snapshot(env, now=now)
    state["wall_snapshot_recorded"] = True
    state["wall_snapshot"] = snapshot
    refresh_timing_results(env, now=now)
    return True


def budget_exhausted(env: Any, now: Optional[float] = None, fallback_timeout: Any = None) -> bool:
    record_wall_snapshot_if_needed(env, now=now)
    metrics = get_current_time_metrics(env, now=now)
    budget_limit_sec = metrics.get("budget_limit_sec")
    if budget_limit_sec in (None, "", 0, 0.0):
        budget_limit_sec = _safe_float(fallback_timeout, 0.0) if fallback_timeout not in (None, "") else None
    if budget_limit_sec in (None, "", 0, 0.0):
        return False
    if metrics.get("budget_mode") == "fuzz-active":
        return float(metrics.get("fuzz_active_elapsed_sec") or 0.0) >= float(budget_limit_sec)
    return float(metrics.get("wall_elapsed_sec") or 0.0) >= float(budget_limit_sec)


def refresh_timing_results(env: Any, now: Optional[float] = None) -> Dict[str, Any]:
    state = ensure_timing_state(env)
    metrics = get_current_time_metrics(env, now=now)
    payload = {
        "budget_mode": state.get("budget_mode", "wall"),
        "budget_limit_sec": state.get("budget_limit_sec"),
        "budget_fuzz_active_sec": state.get("budget_fuzz_active_sec"),
        "wall_snapshot_sec": state.get("wall_snapshot_sec"),
        "wall_snapshot_recorded": bool(state.get("wall_snapshot_recorded")),
        "current_wall_clock_elapsed_sec": metrics.get("wall_elapsed_sec"),
        "current_fuzz_active_elapsed_sec": metrics.get("fuzz_active_elapsed_sec"),
        "agent_interaction_total_sec": metrics.get("agent_interaction_total_sec"),
        "agent_interaction_count": metrics.get("agent_interaction_count"),
        "wall_snapshot": state.get("wall_snapshot"),
        "final_snapshot": state.get("final_snapshot"),
        "final_wall_clock_elapsed_sec": state.get("final_wall_clock_elapsed_sec"),
        "final_fuzz_active_elapsed_sec": state.get("final_fuzz_active_elapsed_sec"),
    }
    results = getattr(env, "results", None)
    if isinstance(results, dict):
        results["experiment_timing"] = payload
    return payload


def finalize_timing(env: Any, now: Optional[float] = None) -> Dict[str, Any]:
    state = ensure_timing_state(env)
    record_wall_snapshot_if_needed(env, now=now)
    metrics = get_current_time_metrics(env, now=now)
    state["final_snapshot"] = build_live_snapshot(env, now=now)
    state["final_wall_clock_elapsed_sec"] = metrics.get("wall_elapsed_sec")
    state["final_fuzz_active_elapsed_sec"] = metrics.get("fuzz_active_elapsed_sec")
    return refresh_timing_results(env, now=now)
