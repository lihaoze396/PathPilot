"""Canonical timer metric extraction for pathpilot result exports.

This module intentionally keeps compatibility with historical result fields,
but it exposes a single paper-facing timer vocabulary.  Old ambiguous names
such as ``llm_total_latency_s`` and ``pure_fuzz_wall_time_sec`` should be
treated as legacy input aliases only.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def first_float(*values: Any, default: Optional[float] = None) -> Optional[float]:
    for value in values:
        parsed = safe_float(value)
        if parsed is not None:
            return parsed
    return default


def first_int(*values: Any, default: Optional[int] = None) -> Optional[int]:
    for value in values:
        parsed = safe_int(value)
        if parsed is not None:
            return parsed
    return default


def safe_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def parse_json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_phase_summary(value: Any) -> Dict[str, Any]:
    return parse_json_dict(value)


def phase_payload(phase_summary: Dict[str, Any], phase_name: str) -> Dict[str, Any]:
    if not isinstance(phase_summary, dict):
        return {}
    phases = phase_summary.get("phases") if isinstance(phase_summary.get("phases"), dict) else {}
    payload = phases.get(phase_name) if isinstance(phases.get(phase_name), dict) else {}
    return payload if isinstance(payload, dict) else {}


def overall_phase_payload(phase_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(phase_summary, dict):
        return {}
    overall = phase_summary.get("overall") if isinstance(phase_summary.get("overall"), dict) else {}
    if not overall:
        overall = phase_payload(phase_summary, "total_run")
    return overall if isinstance(overall, dict) else {}


def non_llm_residual_payload(phase_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(phase_summary, dict):
        return {}
    residual = phase_summary.get("pure_fuzz") if isinstance(phase_summary.get("pure_fuzz"), dict) else {}
    if not residual:
        residual = phase_payload(phase_summary, "pure_fuzz")
    return residual if isinstance(residual, dict) else {}


def canonical_timer_metrics(row: Dict[str, Any], phase_summary: Any = None) -> Dict[str, Any]:
    """Return the single normalized timer vocabulary for one run/contract.

    Canonical fields:
    - ``legacy_active_runtime_sec``: historical active timer
      (wall minus old top-level agent latency).
    - ``complete_llm_event_time_sec``: phase-cost LLM event total.
    - ``legacy_top_level_llm_latency_sec``: old top-level latency alias.
    - ``non_llm_residual_sec``: fuzz-core wall residual after subtracting LLM
      intervals. This is not a clean local EVM/transaction execution timer.
    """

    if not isinstance(row, dict):
        row = {}
    phase_summary = parse_phase_summary(
        phase_summary
        if phase_summary is not None
        else row.get("phase_cost_summary_json", row.get("phase_cost_summary"))
    )
    overall = overall_phase_payload(phase_summary)
    residual = non_llm_residual_payload(phase_summary)
    local_execution = parse_json_dict(row.get("local_execution_timing"))

    complete_llm = first_float(
        row.get("complete_llm_event_time_sec"),
        row.get("total_run_llm_time_sec"),
        overall.get("llm_time_sec"),
    )
    legacy_llm = first_float(
        row.get("legacy_top_level_llm_latency_sec"),
        row.get("llm_interaction_time_total_sec"),
        row.get("llm_total_latency_s"),
        row.get("agent_interaction_total_sec"),
    )
    legacy_active = first_float(
        row.get("legacy_active_runtime_sec"),
        row.get("final_fuzz_active_elapsed_sec"),
        row.get("fuzz_active_sec"),
        row.get("fuzz_active_time_s"),
        row.get("active_runtime_sec"),
    )
    wall_clock = first_float(
        row.get("wall_clock_sec"),
        row.get("wall_clock_time_s"),
        row.get("final_wall_clock_elapsed_sec"),
        overall.get("wall_time_sec"),
    )
    non_llm_residual = first_float(
        row.get("non_llm_residual_sec"),
        residual.get("wall_time_sec"),
        row.get("pure_fuzz_wall_time_sec"),
    )

    llm_call_count = first_int(
        row.get("llm_call_count"),
        row.get("llm_interaction_count"),
        row.get("agent_interaction_count"),
        overall.get("llm_event_count"),
    )
    total_tokens = first_int(
        row.get("total_tokens"),
        row.get("llm_total_tokens"),
        overall.get("total_tokens"),
    )
    cost_usd = first_float(
        row.get("cost_usd"),
        row.get("api_cost"),
        row.get("llm_cost_usd"),
        overall.get("cost_usd"),
    )
    clean_local_total = first_float(
        row.get("clean_local_tx_execution_time_sec"),
        local_execution.get("clean_local_tx_execution_time_sec"),
    )
    clean_local_total_count = first_int(
        row.get("clean_local_tx_execution_count"),
        local_execution.get("clean_local_tx_execution_count"),
    )
    clean_local_fuzz = first_float(
        row.get("clean_local_fuzz_tx_execution_time_sec"),
        local_execution.get("clean_local_fuzz_tx_execution_time_sec"),
    )
    clean_local_fuzz_count = first_int(
        row.get("clean_local_fuzz_tx_execution_count"),
        local_execution.get("clean_local_fuzz_tx_execution_count"),
    )
    clean_local_setup = first_float(
        row.get("clean_local_setup_tx_execution_time_sec"),
        local_execution.get("clean_local_setup_tx_execution_time_sec"),
    )
    clean_local_setup_count = first_int(
        row.get("clean_local_setup_tx_execution_count"),
        local_execution.get("clean_local_setup_tx_execution_count"),
    )
    clean_local_agent = first_float(
        row.get("clean_local_agent_tx_execution_time_sec"),
        local_execution.get("clean_local_agent_tx_execution_time_sec"),
    )
    clean_local_agent_count = first_int(
        row.get("clean_local_agent_tx_execution_count"),
        local_execution.get("clean_local_agent_tx_execution_count"),
    )
    has_clean_local_timer = safe_bool(local_execution.get("has_clean_local_tx_execution_timer"))
    if has_clean_local_timer is None:
        has_clean_local_timer = safe_bool(row.get("has_clean_local_tx_execution_timer"))
    if has_clean_local_timer is None:
        has_clean_local_timer = bool(
            (clean_local_total is not None and clean_local_total_count not in (None, 0))
            or (clean_local_fuzz is not None and clean_local_fuzz_count not in (None, 0))
        )

    return {
        "configured_budget_sec": first_float(
            row.get("configured_budget_sec"),
            row.get("budget_fuzz_active_sec"),
            row.get("time_budget_sec"),
            row.get("time_budget"),
        ),
        "wall_clock_sec": wall_clock,
        "legacy_active_runtime_sec": legacy_active,
        "legacy_active_timer_mode": row.get("legacy_active_timer_mode")
        or row.get("active_timer_mode")
        or "wall_minus_legacy_agent_latency",
        "complete_llm_event_time_sec": complete_llm,
        "legacy_top_level_llm_latency_sec": legacy_llm,
        "llm_call_count": llm_call_count,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "non_llm_residual_sec": non_llm_residual,
        "has_complete_phase_cost": complete_llm is not None,
        "has_clean_local_tx_execution_timer": bool(has_clean_local_timer),
        "clean_local_tx_execution_time_sec": clean_local_total,
        "clean_local_tx_execution_count": clean_local_total_count,
        "clean_local_fuzz_tx_execution_time_sec": clean_local_fuzz,
        "clean_local_fuzz_tx_execution_count": clean_local_fuzz_count,
        "clean_local_setup_tx_execution_time_sec": clean_local_setup,
        "clean_local_setup_tx_execution_count": clean_local_setup_count,
        "clean_local_agent_tx_execution_time_sec": clean_local_agent,
        "clean_local_agent_tx_execution_count": clean_local_agent_count,
    }
