from __future__ import annotations

from typing import Any, Dict, Optional


RQ1_CONFIGS: Dict[str, Dict[str, int]] = {
    "G50": {"g": 50, "k": 0, "t": 0, "total": 50},
    "G100": {"g": 100, "k": 0, "t": 0, "total": 100},
    "G30": {"g": 30, "k": 0, "t": 0, "total": 30},
    "G180": {"g": 180, "k": 0, "t": 0, "total": 180},
    "G600": {"g": 600, "k": 0, "t": 0, "total": 600},
    "G60K120": {"g": 60, "k": 120, "t": 0, "total": 180},
    "G60K60T60": {"g": 60, "k": 60, "t": 60, "total": 180},
    "G120K30T30": {"g": 120, "k": 30, "t": 30, "total": 180},
    "G30K120T30": {"g": 30, "k": 120, "t": 30, "total": 180},
    "G30K30T120": {"g": 30, "k": 30, "t": 120, "total": 180},
    "G200K400": {"g": 200, "k": 400, "t": 0, "total": 600},
    "G200K200T200": {"g": 200, "k": 200, "t": 200, "total": 600},
    "G400K100T100": {"g": 400, "k": 100, "t": 100, "total": 600},
    "G100K400T100": {"g": 100, "k": 400, "t": 100, "total": 600},
    "G100K100T400": {"g": 100, "k": 100, "t": 400, "total": 600},
}


def normalize_rq1_config(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in RQ1_CONFIGS else ""


def rq1_budget_for_config(value: Any) -> Dict[str, int]:
    config = normalize_rq1_config(value)
    return dict(RQ1_CONFIGS.get(config, {}))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _current_fuzz_active_elapsed_sec(env: Any) -> float:
    try:
        from fuzzer.runtime.runtime_timing import refresh_timing_results

        timing = refresh_timing_results(env) or {}
        return float(timing.get("current_fuzz_active_elapsed_sec") or 0.0)
    except Exception:
        return 0.0


def build_rq1_phase_state(args: Any) -> Dict[str, Any]:
    config = normalize_rq1_config(getattr(args, "rq1_config", ""))
    budget = rq1_budget_for_config(config)
    if not budget:
        g_sec = _safe_int(getattr(args, "rq1_g_sec", 0), 0)
        k_sec = _safe_int(getattr(args, "rq1_k_sec", 0), 0)
        t_sec = _safe_int(getattr(args, "rq1_t_sec", 0), 0)
        total_sec = _safe_int(getattr(args, "rq1_total_sec", 0), g_sec + k_sec + t_sec)
        enabled = bool(g_sec or k_sec or t_sec or total_sec)
    else:
        g_sec = int(budget["g"])
        k_sec = int(budget["k"])
        t_sec = int(budget["t"])
        total_sec = int(budget["total"])
        enabled = True
    if total_sec <= 0:
        total_sec = g_sec + k_sec + t_sec
    return {
        "enabled": bool(enabled),
        "config": config,
        "g_sec": max(0, int(g_sec)),
        "k_sec": max(0, int(k_sec)),
        "t_sec": max(0, int(t_sec)),
        "total_sec": max(0, int(total_sec)),
        "k_start_sec": max(0, int(g_sec)),
        "t_start_sec": max(0, int(g_sec) + int(k_sec)),
    }


def ensure_rq1_phase_state(env: Any, args: Optional[Any] = None) -> Dict[str, Any]:
    state = getattr(env, "rq1_phase_state", None)
    runtime_args = args or getattr(env, "args", None)
    if not isinstance(state, dict):
        state = build_rq1_phase_state(runtime_args)
        setattr(env, "rq1_phase_state", state)
    state.setdefault("enabled", False)
    state.setdefault("config", "")
    state.setdefault("g_sec", 0)
    state.setdefault("k_sec", 0)
    state.setdefault("t_sec", 0)
    state.setdefault("total_sec", 0)
    state.setdefault("k_start_sec", int(state.get("g_sec") or 0))
    state.setdefault("t_start_sec", int(state.get("g_sec") or 0) + int(state.get("k_sec") or 0))
    return state


def rq1_current_phase(env: Any, args: Optional[Any] = None, elapsed_sec: Optional[float] = None) -> str:
    state = ensure_rq1_phase_state(env, args)
    if not bool(state.get("enabled")):
        return ""
    elapsed = float(elapsed_sec if elapsed_sec is not None else _current_fuzz_active_elapsed_sec(env))
    g_sec = float(state.get("g_sec") or 0)
    k_sec = float(state.get("k_sec") or 0)
    t_sec = float(state.get("t_sec") or 0)
    total_sec = float(state.get("total_sec") or (g_sec + k_sec + t_sec))
    if elapsed < g_sec:
        return "G"
    if elapsed < g_sec + k_sec:
        return "K" if k_sec > 0 else ("T" if t_sec > 0 else "DONE")
    if elapsed < total_sec:
        return "T" if t_sec > 0 else "DONE"
    return "DONE"


def rq1_phase_allows_feature(env: Any, feature: str, args: Optional[Any] = None) -> bool:
    state = ensure_rq1_phase_state(env, args)
    if not bool(state.get("enabled")):
        return True
    feature_name = str(feature or "").strip().lower()
    elapsed = _current_fuzz_active_elapsed_sec(env)
    g_sec = float(state.get("g_sec") or 0)
    k_sec = float(state.get("k_sec") or 0)
    t_sec = float(state.get("t_sec") or 0)
    total_sec = float(state.get("total_sec") or (g_sec + k_sec + t_sec))
    if total_sec > 0 and elapsed >= total_sec:
        return False
    if feature_name == "gei":
        return elapsed < g_sec if g_sec > 0 else False
    if feature_name == "ksi":
        # Keep collecting KSI evidence during TFI, because TFI can consume the
        # evidence stream while it builds and repairs candidates.
        return k_sec > 0 and elapsed >= g_sec
    if feature_name in {"tfi", "llm", "agent"}:
        return t_sec > 0 and elapsed >= g_sec + k_sec
    return True
