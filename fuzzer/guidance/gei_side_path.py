# -*- coding: utf-8 -*-

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


READ_ONLY_PREFIXES = ("get", "is", "has")
READ_ONLY_EXACT = {
    "allowance",
    "balanceof",
    "totalsupply",
    "name",
    "symbol",
    "decimals",
    "owner",
    "pair",
    "router",
    "factory",
}
READ_ONLY_TOTAL_PREFIX = "total"
READ_ONLY_TOTAL_EXCLUDE = ("set", "update", "change", "config", "configure")

CONFIG_KEYWORDS = (
    "set",
    "update",
    "change",
    "configure",
    "config",
    "enable",
    "disable",
    "authorize",
    "unauthorize",
    "approveowner",
    "owner",
    "admin",
    "role",
    "fee",
    "tax",
    "wallet",
    "router",
    "pair",
    "factory",
    "list",
    "pause",
    "unpause",
    "trading",
    "limit",
    "max",
    "manual",
    "exclude",
    "include",
    "exempt",
    "permission",
    "distributor",
    "marketing",
    "treasury",
    "reward",
)

ASSET_KEYWORDS = (
    "withdraw",
    "claim",
    "transfer",
    "approve",
    "mint",
    "burn",
    "swap",
    "sell",
    "buy",
    "liquidity",
    "rescue",
    "recover",
    "skim",
    "sync",
)

ASSET_WEAK_EXACT = {
    "increaseallowance",
    "decreaseallowance",
}

EXECUTOR_KEYWORDS = (
    "execute",
    "multicall",
    "batch",
)

FALLBACK_RECEIVE = {"fallback", "receive"}
COMMON_INTERFACE_PENALTIES = {
    "transfer": 1,
    "approve": 1,
    "allowance": 4,
    "balanceof": 4,
    "totalsupply": 4,
    "name": 4,
    "symbol": 4,
    "decimals": 4,
}
REASON_PRIORITY = {
    "executor_like_sequence": 0,
    "config_or_state_update_sequence": 2,
    "asset_interaction_sequence": 3,
    "multi_tx_semantic_prefix": 4,
    "semantic_high_coverage_sequence": 5,
    "path_progress_sequence": 6,
}


def is_read_only_like(function_name: str) -> bool:
    name = _normalize_name(function_name)
    if not name:
        return False
    if name in READ_ONLY_EXACT:
        return True
    if name.startswith(READ_ONLY_PREFIXES):
        return True
    if name.startswith(READ_ONLY_TOTAL_PREFIX) and not any(token in name for token in READ_ONLY_TOTAL_EXCLUDE):
        return True
    return False


def is_config_like(function_name: str) -> bool:
    name = _normalize_name(function_name)
    if not name or is_read_only_like(name):
        return False
    return any(keyword in name for keyword in CONFIG_KEYWORDS)


def is_asset_interaction_like(function_name: str) -> bool:
    name = _normalize_name(function_name)
    if not name:
        return False
    if name == "allowance":
        return False
    if name in ASSET_WEAK_EXACT:
        return False
    return any(keyword in name for keyword in ASSET_KEYWORDS)


def is_executor_like(function_name: str) -> bool:
    name = _normalize_name(function_name)
    if not name:
        return False
    return any(keyword in name for keyword in EXECUTOR_KEYWORDS)


def classify_side_path_candidate(observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _classify_side_path_candidate(observation)[0]


def classify_side_path_candidate_with_meta(
    observation: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    return _classify_side_path_candidate(observation)


class GEISidePathRecorder:
    def __init__(self, output_dir: str, enabled: bool = False):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.output_path = os.path.join(self.output_dir, "side_path_candidates.jsonl") if self.output_dir else ""
        self.shortlist_path = os.path.join(self.output_dir, "side_path_shortlist.json") if self.output_dir else ""
        self.enabled = bool(enabled and self.output_path)
        self.signature_counts: Counter = Counter()
        self.reason_counts: Counter = Counter()
        self.function_counts: Counter = Counter()
        self.filtered_function_counts: Counter = Counter()
        self.tier_counts: Counter = Counter()
        self.total_candidates = 0
        self.filtered_read_only_like_count = 0
        self.filtered_high_coverage_only_count = 0
        self.observation_count = 0
        self.coverage_delta_sum = 0.0
        self.revert_count = 0
        self.ksi_enrichment_used = False
        self.source_counts: Counter = Counter()
        self.default_input_source = "gei_core_observation"
        self._candidates: List[Dict[str, Any]] = []

        if self.enabled:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(self.output_path, "w", encoding="utf-8"):
                    pass
                if self.shortlist_path and os.path.exists(self.shortlist_path):
                    os.remove(self.shortlist_path)
            except Exception:
                self.enabled = False

    def record_from_ksi(self, ksi_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.record_from_observation(
            ksi_record,
            source="legacy_ksi_record",
            ksi_enrichment_used=True,
        )

    def record_from_observation(
        self,
        observation: Dict[str, Any],
        source: str = "gei_core_observation",
        ksi_enrichment_used: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            if not isinstance(observation, dict):
                return None
            self.observation_count += 1
            self.source_counts[str(source or self.default_input_source)] += 1
            self.coverage_delta_sum += _to_number(observation.get("coverage_delta"))
            if observation.get("revert") is True:
                self.revert_count += 1
            if ksi_enrichment_used:
                self.ksi_enrichment_used = True

            candidate, meta = classify_side_path_candidate_with_meta(observation)
            self._record_filtered(meta)
            if not candidate:
                return None

            signature = self._signature(candidate)
            if self.signature_counts[signature] >= 3:
                return None

            self.signature_counts[signature] += 1
            self.reason_counts[str(candidate.get("reason") or "unknown")] += 1
            self.tier_counts[str(candidate.get("tier") or "low")] += 1
            for function in candidate.get("suggested_path") or candidate.get("functions") or []:
                self.function_counts[str(function)] += 1

            with open(self.output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(candidate, ensure_ascii=True) + "\n")
            self.total_candidates += 1
            self._candidates.append(candidate)
            return candidate
        except Exception:
            return None

    def get_summary(self) -> Dict[str, Any]:
        shortlist_payload = self._write_shortlist()
        shortlist_summary = shortlist_payload.get("summary") if isinstance(shortlist_payload, dict) else {}
        input_source = "disabled"
        if self.enabled:
            if self.source_counts.get("legacy_ksi_record", 0) > 0 and self.source_counts.get("gei_core_observation", 0) == 0:
                input_source = "legacy_ksi_record"
            else:
                input_source = self.default_input_source
        return {
            "enabled": bool(self.enabled),
            "observation_count": int(self.observation_count),
            "side_path_candidate_count": int(self.total_candidates),
            "input_source": input_source,
            "ksi_required": False,
            "ksi_enrichment_used": bool(self.ksi_enrichment_used),
            "coverage_delta_sum": float(self.coverage_delta_sum),
            "revert_count": int(self.revert_count),
            "reason_counts": dict(self.reason_counts.most_common()),
            "top_functions": dict(self.function_counts.most_common(10)),
            "filtered_read_only_like_count": int(self.filtered_read_only_like_count),
            "filtered_high_coverage_only_count": int(self.filtered_high_coverage_only_count),
            "filtered_top_functions": dict(self.filtered_function_counts.most_common(10)),
            "shortlist_path": self.shortlist_path if self.shortlist_path else None,
            "shortlist_count": int(shortlist_summary.get("shortlist_count") or 0),
            "tier_counts": {
                "high": int(self.tier_counts.get("high", 0)),
                "medium": int(self.tier_counts.get("medium", 0)),
                "low": int(self.tier_counts.get("low", 0)),
            },
            "source_counts": dict(self.source_counts.most_common()),
            "output_path": self.output_path if self.output_path else None,
        }

    @staticmethod
    def _signature(candidate: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
        path = tuple(str(item) for item in candidate.get("suggested_path") or [])
        reason = str(candidate.get("reason") or "")
        return path, reason

    def _record_filtered(self, meta: Dict[str, Any]) -> None:
        if not isinstance(meta, dict):
            return
        function_name = str(meta.get("primary_function") or "")
        if meta.get("filtered_read_only_like"):
            self.filtered_read_only_like_count += 1
            if function_name:
                self.filtered_function_counts[function_name] += 1
        elif meta.get("filtered_high_coverage_only"):
            self.filtered_high_coverage_only_count += 1
            if function_name:
                self.filtered_function_counts[function_name] += 1

    def _write_shortlist(self) -> Dict[str, Any]:
        payload = self._build_shortlist_payload()
        if not self.shortlist_path:
            return payload
        try:
            with open(self.shortlist_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            pass
        return payload

    def _build_shortlist_payload(self) -> Dict[str, Any]:
        best_by_path: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for candidate in self._candidates:
            if str(candidate.get("tier") or "low") not in ("high", "medium"):
                continue
            signature = tuple(str(item) for item in candidate.get("suggested_path") or [])
            current = best_by_path.get(signature)
            if current is None or _shortlist_sort_key(candidate) > _shortlist_sort_key(current):
                best_by_path[signature] = candidate

        shortlist = sorted(
            best_by_path.values(),
            key=_shortlist_sort_key,
            reverse=True,
        )[:20]

        return {
            "summary": {
                "candidate_count": int(self.total_candidates),
                "shortlist_count": len(shortlist),
                "high_tier_count": int(self.tier_counts.get("high", 0)),
                "medium_tier_count": int(self.tier_counts.get("medium", 0)),
                "low_tier_count": int(self.tier_counts.get("low", 0)),
            },
            "shortlist": [
                {
                    "sequence_id": str(candidate.get("sequence_id") or ""),
                    "suggested_path": [str(item) for item in candidate.get("suggested_path") or []],
                    "reason": str(candidate.get("reason") or "unknown"),
                    "quality_score": int(candidate.get("quality_score") or 0),
                    "tier": str(candidate.get("tier") or "low"),
                    "signals": dict(candidate.get("signals") or {}),
                }
                for candidate in shortlist
            ],
        }


def build_gei_core_observation_from_legacy_ksi(ksi_record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ksi_record, dict):
        return {}

    observations = ksi_record.get("observations")
    if not isinstance(observations, dict):
        observations = {}

    function_names = _normalize_string_list(observations.get("functions"))
    matched_prefix_len = _to_optional_int(ksi_record.get("matched_prefix_len"))
    frontier = ksi_record.get("frontier")
    sink_reached = _to_optional_bool(ksi_record.get("sink_reached"))
    sink_function_reached = _to_optional_bool(ksi_record.get("sink_function_reached"))
    sink_observed = _to_optional_bool(ksi_record.get("sink_observed"))

    return {
        "source": "legacy_ksi_record",
        "sequence_id": str(ksi_record.get("sequence_id") or ""),
        "function_names": function_names,
        "coverage_delta": _to_number(observations.get("coverage_delta")),
        "revert": observations.get("revert") is True,
        "success": observations.get("revert") is not True,
        "tx_count": len(function_names),
        "matched_prefix_len": matched_prefix_len,
        "frontier": frontier if frontier not in ("", None) else None,
        "sink_reached": sink_reached,
        "sink_function_reached": sink_function_reached,
        "target_seen": bool(
            sink_observed is True
            or sink_function_reached is True
            or sink_reached is True
        ),
    }


def _classify_side_path_candidate(
    observation: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "primary_function": "",
        "filtered_read_only_like": False,
        "filtered_high_coverage_only": False,
    }
    try:
        if not isinstance(observation, dict):
            return None, meta
        if observation.get("source") == "legacy_ksi_record" or "observations" in observation:
            observation = build_gei_core_observation_from_legacy_ksi(observation)

        functions = _normalize_string_list(observation.get("function_names"))
        if not functions:
            return None, meta

        coverage_delta = _to_number(observation.get("coverage_delta"))
        revert = observation.get("revert") is True
        success = observation.get("success") is True
        matched_prefix_len = _to_optional_int(observation.get("matched_prefix_len")) or 0
        frontier = observation.get("frontier")
        sink_reached = observation.get("sink_reached") is True
        sink_function_reached = observation.get("sink_function_reached") is True
        target_seen = observation.get("target_seen") is True
        tx_count = _to_optional_int(observation.get("tx_count")) or len(functions)

        normalized_functions = [_normalize_name(function) for function in functions]
        meta["primary_function"] = functions[0]

        read_only_hits = [function for function in normalized_functions if is_read_only_like(function)]
        config_hits = [function for function in normalized_functions if is_config_like(function)]
        asset_hits = [function for function in normalized_functions if is_asset_interaction_like(function)]
        executor_hits = [function for function in normalized_functions if is_executor_like(function)]
        fallback_or_receive = any(function in FALLBACK_RECEIVE for function in normalized_functions)

        has_read_only = bool(read_only_hits)
        has_config = bool(config_hits)
        has_asset = bool(asset_hits)
        has_executor = bool(executor_hits)
        semantic_risk = bool(has_config or has_asset or has_executor)
        new_coverage_sequence = coverage_delta > 0
        high_coverage_delta = coverage_delta >= 10
        multi_tx = tx_count >= 2
        multi_tx_semantic = bool(multi_tx and new_coverage_sequence and semantic_risk)
        prefix_progress = matched_prefix_len > 0
        target_related = bool(
            target_seen
            or sink_function_reached
            or sink_reached
            or frontier not in (None, "")
        )
        high_coverage_allowed = bool(
            high_coverage_delta
            and (
                has_config
                or has_asset
                or has_executor
                or prefix_progress
                or (revert and not has_read_only)
            )
        )

        only_high_coverage = bool(
            high_coverage_delta
            and not semantic_risk
            and not prefix_progress
            and not (revert and not has_read_only)
        )
        if has_read_only and not semantic_risk and not prefix_progress and not (revert and not has_read_only):
            meta["filtered_read_only_like"] = True
            return None, meta
        if only_high_coverage:
            meta["filtered_high_coverage_only"] = True
            return None, meta

        strong_candidate = any(
            (
                has_executor,
                (has_config and new_coverage_sequence),
                (has_config and revert),
                (has_config and prefix_progress),
                (has_asset and new_coverage_sequence),
                (has_asset and revert),
                (has_asset and prefix_progress),
                multi_tx_semantic,
                (prefix_progress and target_related),
                high_coverage_allowed and semantic_risk,
                high_coverage_allowed and revert and not has_read_only,
            )
        )
        if not strong_candidate:
            return None, meta

        reason = _select_reason(
            has_executor=has_executor,
            has_config=has_config,
            has_asset=has_asset,
            multi_tx_semantic=multi_tx_semantic,
            high_coverage_allowed=high_coverage_allowed,
            prefix_progress=prefix_progress,
        )
        if not reason:
            return None, meta

        suggested_path = _build_semantic_suggested_path(functions)
        if not suggested_path:
            suggested_path = _dedupe_preserve_order(functions)
        quality_score = _compute_quality_score(
            functions=functions,
            suggested_path=suggested_path,
            has_read_only=has_read_only,
            has_config=has_config,
            has_asset=has_asset,
            has_executor=has_executor,
            multi_tx_semantic=multi_tx_semantic,
            high_coverage_delta=high_coverage_delta,
            new_coverage_sequence=new_coverage_sequence,
            revert=revert,
            prefix_progress=prefix_progress,
            target_related=target_related,
        )
        tier = _score_to_tier(quality_score)
        return {
            "type": "side_path_candidate",
            "sequence_id": str(observation.get("sequence_id") or ""),
            "functions": functions,
            "reason": reason,
            "signals": {
                "coverage_delta": coverage_delta,
                "revert": revert,
                "success": success,
                "matched_prefix_len": matched_prefix_len,
                "frontier": frontier if frontier not in ("", None) else None,
                "sink_reached": sink_reached,
                "sink_function_reached": sink_function_reached,
                "target_seen": target_seen,
                "tx_count": tx_count,
                "read_only_like": has_read_only,
                "config_like": has_config,
                "asset_interaction_like": has_asset,
                "executor_like": has_executor,
                "high_coverage_delta": high_coverage_delta,
                "multi_tx_semantic": multi_tx_semantic,
                "path_progress": prefix_progress,
            },
            "suggested_new_hypothesis": True,
            "suggested_path": suggested_path,
            "quality_score": quality_score,
            "tier": tier,
            "source": str(observation.get("source") or "gei_core_observation"),
        }, meta
    except Exception:
        return None, meta


def _normalize_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _normalize_name(function_name: Any) -> str:
    return str(function_name or "").strip().lower()


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return 0.0
        if text.lower().startswith("0x"):
            return float(int(text, 16))
        return float(text)
    except Exception:
        return 0.0


def _select_reason(
    has_executor: bool,
    has_config: bool,
    has_asset: bool,
    multi_tx_semantic: bool,
    high_coverage_allowed: bool,
    prefix_progress: bool,
) -> Optional[str]:
    if has_executor:
        return "executor_like_sequence"
    if has_config:
        return "config_or_state_update_sequence"
    if has_asset:
        return "asset_interaction_sequence"
    if multi_tx_semantic:
        return "multi_tx_semantic_prefix"
    if prefix_progress:
        return "path_progress_sequence"
    if high_coverage_allowed:
        return "semantic_high_coverage_sequence"
    return None


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _build_semantic_suggested_path(functions: List[str]) -> List[str]:
    semantic = []
    for function in functions:
        normalized = _normalize_name(function)
        if not normalized:
            continue
        if normalized in FALLBACK_RECEIVE:
            semantic.append(function)
            continue
        if is_config_like(normalized) or is_asset_interaction_like(normalized) or is_executor_like(normalized):
            semantic.append(function)
    return _dedupe_preserve_order(semantic)


def _compute_quality_score(
    functions: List[str],
    suggested_path: List[str],
    has_read_only: bool,
    has_config: bool,
    has_asset: bool,
    has_executor: bool,
    multi_tx_semantic: bool,
    high_coverage_delta: bool,
    new_coverage_sequence: bool,
    revert: bool,
    prefix_progress: bool,
    target_related: bool,
) -> int:
    score = 0
    if has_config:
        score += 3
    if has_asset:
        score += 3
    if has_executor:
        score += 4
    if multi_tx_semantic:
        score += 2
    if prefix_progress:
        score += 2
    if target_related:
        score += 1
    if high_coverage_delta:
        score += 1
    if new_coverage_sequence:
        score += 1
    if revert:
        score += 1
    if has_read_only:
        score -= 4
    if not suggested_path:
        score -= 5

    penalty_source = suggested_path or functions
    for normalized in {_normalize_name(function) for function in penalty_source}:
        score -= COMMON_INTERFACE_PENALTIES.get(normalized, 0)
    return int(score)


def _score_to_tier(quality_score: int) -> str:
    if quality_score >= 5:
        return "high"
    if quality_score >= 3:
        return "medium"
    return "low"


def _shortlist_sort_key(candidate: Dict[str, Any]) -> Tuple[int, float]:
    signals = candidate.get("signals") if isinstance(candidate.get("signals"), dict) else {}
    coverage_delta = _to_number(signals.get("coverage_delta"))
    return int(candidate.get("quality_score") or 0), coverage_delta


def _to_optional_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _to_optional_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    return bool(value)
