# -*- coding: utf-8 -*-

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


DIAGNOSIS_SCORE_WEIGHTS = {
    "caller_or_permission_block": 3,
    "prestate_dependency_block": 3,
    "argument_constraint_block": 2,
    "value_condition_block": 2,
    "repeated_frontier_block": 2,
    "sink_reverted_unknown": 1,
}

STRATEGY_SCORE_WEIGHTS = {
    "approve_before_transferFrom": 4,
    "permission_setup_then_sink": 4,
    "config_before_sink": 3,
    "preserve_prefix_and_retry_sink": 3,
    "preserve_reached_prefix_and_mutate_frontier": 3,
    "retry_with_nonzero_value": 2,
    "mutate_sink_args": 2,
    "conservative_retry": 1,
}

STATUS_PRIORITY = {
    "mappable": 2,
    "partial": 1,
}


def _read_json(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _normalize_path(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = _text(item).strip()
        if text:
            output.append(text)
    return output


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


class RepairSequenceSelector:
    def __init__(
        self,
        output_dir: str,
        repair_candidates_path: Optional[str] = None,
        dryrun_path: Optional[str] = None,
        diagnosis_path: Optional[str] = None,
        max_selected: int = 20,
        selection_mode: str = "deterministic",
        llm_provider: Optional[Any] = None,
        fallback_to_deterministic: bool = True,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.max_selected = max(1, int(max_selected or 1))

        normalized_mode = _text(selection_mode).strip().lower()
        if normalized_mode not in ("deterministic", "llm"):
            normalized_mode = "deterministic"
        self.selection_mode = normalized_mode

        self.llm_provider = llm_provider
        self.fallback_to_deterministic = bool(fallback_to_deterministic)

        self._requested_repair_candidates_path = (
            os.path.abspath(repair_candidates_path) if repair_candidates_path else None
        )
        self._requested_dryrun_path = os.path.abspath(dryrun_path) if dryrun_path else None
        self._requested_diagnosis_path = os.path.abspath(diagnosis_path) if diagnosis_path else None

        self.default_repair_candidates_path = (
            os.path.join(self.output_dir, "guided_sequence_candidates.json") if self.output_dir else ""
        )
        self.default_dryrun_path = (
            os.path.join(self.output_dir, "guided_sequence_dryrun.json") if self.output_dir else ""
        )
        self.default_diagnosis_path = (
            os.path.join(self.output_dir, "repair_diagnosis.json") if self.output_dir else ""
        )

        self.loaded_inputs: Dict[str, bool] = {
            "repair_candidates": False,
            "repair_dryrun": False,
            "repair_diagnosis": False,
        }
        self.input_paths: Dict[str, Optional[str]] = {
            "repair_candidates": self._requested_repair_candidates_path or self.default_repair_candidates_path or None,
            "repair_dryrun": self._requested_dryrun_path or self.default_dryrun_path or None,
            "repair_diagnosis": self._requested_diagnosis_path or self.default_diagnosis_path or None,
        }

        self.last_selected_payload: Dict[str, Any] = {}
        self.last_summary_payload: Dict[str, Any] = {}

    def select(self) -> Dict[str, Any]:
        candidates_payload = self._load_json_with_fallback(
            self._requested_repair_candidates_path,
            self.default_repair_candidates_path,
            "repair_candidates",
        )
        dryrun_payload = self._load_json_with_fallback(
            self._requested_dryrun_path,
            self.default_dryrun_path,
            "repair_dryrun",
        )
        diagnosis_payload = self._load_json_with_fallback(
            self._requested_diagnosis_path,
            self.default_diagnosis_path,
            "repair_diagnosis",
        )

        candidates = candidates_payload.get("candidates")
        if not isinstance(candidates, list):
            candidates = []

        dryrun_results = dryrun_payload.get("results")
        if not isinstance(dryrun_results, list):
            dryrun_results = []

        diagnoses = diagnosis_payload.get("diagnoses")
        if not isinstance(diagnoses, list):
            diagnoses = []

        diagnosis_by_hypothesis: Dict[str, Dict[str, Any]] = {}
        for diagnosis in diagnoses:
            if not isinstance(diagnosis, dict):
                continue
            diagnosis_id = _text(diagnosis.get("hypothesis_id"))
            if diagnosis_id and diagnosis_id not in diagnosis_by_hypothesis:
                diagnosis_by_hypothesis[diagnosis_id] = diagnosis

        dryrun_by_candidate: Dict[str, Dict[str, Any]] = {}
        for result in dryrun_results:
            if not isinstance(result, dict):
                continue
            candidate_id = _text(result.get("candidate_id"))
            if candidate_id and candidate_id not in dryrun_by_candidate:
                dryrun_by_candidate[candidate_id] = result

        merged_candidates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = _text(candidate.get("candidate_id"))
            if not candidate_id:
                continue

            merged = dict(candidate)
            dryrun_result = dryrun_by_candidate.get(candidate_id) or {}
            diagnosis = diagnosis_by_hypothesis.get(_text(candidate.get("hypothesis_id"))) or {}

            merged["guided_sequence"] = _normalize_path(merged.get("guided_sequence"))
            merged["base_path"] = _normalize_path(merged.get("base_path"))
            merged["dryrun"] = dryrun_result
            merged["diagnosis"] = diagnosis
            merged["dryrun_status"] = _text(dryrun_result.get("dryrun_status")) or "missing"
            merged["missing_functions"] = list(dryrun_result.get("missing_functions") or [])
            merged["dryrun_checks"] = dryrun_result.get("checks") if isinstance(dryrun_result.get("checks"), dict) else {}
            merged["all_functions_exist"] = dryrun_result.get("all_functions_exist")
            merged["target_function"] = _text(merged.get("target_function")) or _text(diagnosis.get("target_function"))
            if not merged["target_function"] and merged["guided_sequence"]:
                merged["target_function"] = merged["guided_sequence"][-1]
            merged_candidates.append(merged)

        eligible_candidates = [item for item in merged_candidates if self._is_eligible(item)]
        llm_metadata = self._build_llm_metadata(used=False, fallback_used=False, fallback_reason=None)

        actual_mode = "deterministic"
        selected = []
        if self.selection_mode == "llm":
            actual_mode = "llm"
            try:
                selected = self._select_with_llm(eligible_candidates)
                llm_metadata = self._build_llm_metadata(used=True, fallback_used=False, fallback_reason=None)
            except Exception as exc:
                reason = _text(exc) or "llm_selection_failed"
                llm_metadata = self._build_llm_metadata(
                    used=False,
                    fallback_used=self.fallback_to_deterministic,
                    fallback_reason=reason,
                )
                if self.fallback_to_deterministic:
                    selected = self._select_deterministic(eligible_candidates)
                    actual_mode = "deterministic"
                else:
                    selected = []
        else:
            selected = self._select_deterministic(eligible_candidates)

        payload = {
            "requested_selection_mode": self.selection_mode,
            "selection_mode": actual_mode,
            "selected_count": len(selected),
            "candidate_count": len(merged_candidates),
            "eligible_count": len(eligible_candidates),
            "selected": selected,
            "loaded_inputs": dict(self.loaded_inputs),
            "llm": llm_metadata,
            "source": "guided_sequence_selection_v1",
        }
        self.last_selected_payload = payload
        return payload

    def write_selected(
        self,
        selected_plan: Dict[str, Any],
        filename: str = "selected_guided_sequences.json",
    ) -> str:
        if not self.output_dir:
            return ""
        output_path = os.path.join(self.output_dir, filename)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(selected_plan, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        self.last_selected_payload = dict(selected_plan)
        return output_path

    def write_summary(
        self,
        selected_plan: Dict[str, Any],
        filename: str = "selected_guided_sequences_summary.json",
    ) -> str:
        if not self.output_dir:
            return ""

        selected = selected_plan.get("selected")
        if not isinstance(selected, list):
            selected = []

        diagnosis_type_counts = Counter()
        sequence_strategy_counts = Counter()
        target_function_counts = Counter()
        status_counts = Counter()
        for item in selected:
            if not isinstance(item, dict):
                continue
            diagnosis_type_counts[_text(item.get("diagnosis_type")) or "unknown"] += 1
            sequence_strategy_counts[_text(item.get("sequence_strategy")) or "unknown"] += 1
            target_function_counts[_text(item.get("target_function")) or "unknown"] += 1
            status_counts[_text(item.get("dryrun_status")) or "unknown"] += 1

        payload = {
            "enabled": True,
            "selection_mode": _text(selected_plan.get("selection_mode")) or "deterministic",
            "candidate_count": int(selected_plan.get("candidate_count") or 0),
            "eligible_count": int(selected_plan.get("eligible_count") or 0),
            "selected_count": len(selected),
            "mappable_selected_count": int(status_counts.get("mappable", 0)),
            "partial_selected_count": int(status_counts.get("partial", 0)),
            "unmappable_selected_count": int(status_counts.get("unmappable", 0)),
            "diagnosis_type_counts": _safe_counter_to_dict(diagnosis_type_counts),
            "sequence_strategy_counts": _safe_counter_to_dict(sequence_strategy_counts),
            "top_target_functions": _safe_counter_to_dict(target_function_counts),
            "loaded_inputs": dict(self.loaded_inputs),
            "llm": dict(selected_plan.get("llm") or {}),
            "output_path": os.path.join(self.output_dir, "selected_guided_sequences.json"),
        }

        output_path = os.path.join(self.output_dir, filename)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""

        self.last_summary_payload = payload
        return output_path

    def _load_json_with_fallback(
        self,
        requested_path: Optional[str],
        default_path: Optional[str],
        key: str,
    ) -> Dict[str, Any]:
        candidate_paths: List[str] = []
        if requested_path:
            candidate_paths.append(requested_path)
        if default_path and default_path not in candidate_paths:
            candidate_paths.append(default_path)

        self.loaded_inputs[key] = False
        self.input_paths[key] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_json(path)
            if payload:
                self.loaded_inputs[key] = True
                self.input_paths[key] = path
                return payload
        return {}

    @staticmethod
    def _is_eligible(candidate: Dict[str, Any]) -> bool:
        if not isinstance(candidate, dict):
            return False
        guided_sequence = _normalize_path(candidate.get("guided_sequence"))
        if not guided_sequence:
            return False
        missing_functions = candidate.get("missing_functions")
        if isinstance(missing_functions, list) and any(_text(item).strip() for item in missing_functions):
            return False
        dryrun_status = _text(candidate.get("dryrun_status")).strip().lower()
        if dryrun_status not in ("mappable", "partial"):
            return False
        return True

    def _score_candidate(self, candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        dryrun_status = _text(candidate.get("dryrun_status")).strip().lower()
        if dryrun_status == "mappable":
            score += 5.0
            reasons.append("dryrun_status=mappable")
        elif dryrun_status == "partial":
            score += 3.0
            reasons.append("dryrun_status=partial")

        diagnosis_type = _text(candidate.get("diagnosis_type")).strip()
        diagnosis_weight = DIAGNOSIS_SCORE_WEIGHTS.get(diagnosis_type, 0)
        if diagnosis_weight:
            score += float(diagnosis_weight)
            reasons.append("diagnosis_type={}".format(diagnosis_type))

        sequence_strategy = _text(candidate.get("sequence_strategy")).strip()
        strategy_weight = STRATEGY_SCORE_WEIGHTS.get(sequence_strategy, 0)
        if strategy_weight:
            score += float(strategy_weight)
            reasons.append("strategy={}".format(sequence_strategy))

        sender_policy = _text(candidate.get("sender_policy")).lower()
        if "owner" in sender_policy or "deployer" in sender_policy:
            score += 2.0
            reasons.append("sender_policy={}".format(_text(candidate.get("sender_policy"))))

        value_policy = _text(candidate.get("value_policy")).lower()
        if "nonzero" in value_policy:
            score += 1.0
            reasons.append("value_policy={}".format(_text(candidate.get("value_policy"))))

        argument_policy = _text(candidate.get("argument_policy")).lower()
        if any(keyword in argument_policy for keyword in ("boolean", "boundary", "mutate")):
            score += 1.0
            reasons.append("argument_policy={}".format(_text(candidate.get("argument_policy"))))

        prestate_policy = _text(candidate.get("prestate_policy")).strip()
        if prestate_policy and prestate_policy.lower() != "none":
            score += 2.0
            reasons.append("prestate_policy={}".format(prestate_policy))

        sequence_length = len(_normalize_path(candidate.get("guided_sequence")))
        if 2 <= sequence_length <= 4:
            score += 2.0
            reasons.append("sequence_length=2_to_4")
        elif 5 <= sequence_length <= 6:
            score += 1.0
            reasons.append("sequence_length=5_to_6")
        elif sequence_length > 6:
            score -= 2.0
            reasons.append("sequence_length_gt_6")

        priority = _as_float(candidate.get("priority"), default=0.0)
        if priority > 0:
            score += round(min(priority, 1.0), 3)
            reasons.append("candidate_priority={:.2f}".format(min(priority, 1.0)))

        return round(score, 3), reasons

    def _sorted_eligible_candidates(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        decorated = []
        for candidate in candidates:
            score, reasons = self._score_candidate(candidate)
            item = dict(candidate)
            item["_selection_score"] = score
            item["_selection_reason"] = reasons
            decorated.append(item)

        decorated.sort(
            key=lambda item: (
                -float(item.get("_selection_score", 0.0)),
                -STATUS_PRIORITY.get(_text(item.get("dryrun_status")).lower(), 0),
                -_as_float(item.get("priority"), default=0.0),
                len(_normalize_path(item.get("guided_sequence"))),
                _text(item.get("candidate_id")),
            )
        )
        return decorated

    def _select_deterministic(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordered = self._sorted_eligible_candidates(candidates)
        if not ordered:
            return []

        strategy_cap = max(2, int(math.ceil(self.max_selected * 0.4)))
        selected = self._apply_constraints(ordered, strategy_cap=strategy_cap)
        if len(selected) < self.max_selected:
            selected = self._apply_constraints(ordered, strategy_cap=None)

        return self._finalize_selected(selected, mode="deterministic")

    def _apply_constraints(
        self,
        ordered_candidates: Sequence[Dict[str, Any]],
        strategy_cap: Optional[int],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        sequence_signatures = set()
        hypothesis_counts = Counter()
        target_counts = Counter()
        strategy_counts = Counter()

        for candidate in ordered_candidates:
            if len(selected) >= self.max_selected:
                break

            candidate_id = _text(candidate.get("candidate_id"))
            if not candidate_id or candidate_id in selected_ids:
                continue

            guided_sequence = tuple(_normalize_path(candidate.get("guided_sequence")))
            if not guided_sequence or guided_sequence in sequence_signatures:
                continue

            hypothesis_id = _text(candidate.get("hypothesis_id")) or "unknown"
            target_function = _text(candidate.get("target_function")) or "unknown"
            sequence_strategy = _text(candidate.get("sequence_strategy")) or "unknown"

            if hypothesis_counts[hypothesis_id] >= 3:
                continue
            if target_counts[target_function] >= 5:
                continue
            if strategy_cap is not None and strategy_counts[sequence_strategy] >= strategy_cap:
                continue

            selected.append(candidate)
            selected_ids.add(candidate_id)
            sequence_signatures.add(guided_sequence)
            hypothesis_counts[hypothesis_id] += 1
            target_counts[target_function] += 1
            strategy_counts[sequence_strategy] += 1

        return selected

    def _finalize_selected(self, selected: Sequence[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
        if not selected:
            return []

        max_score = max(float(item.get("_selection_score", 0.0)) for item in selected) or 1.0
        finalized = []
        for index, item in enumerate(selected, start=1):
            candidate_priority = _as_float(item.get("priority"), default=-1.0)
            if 0.0 <= candidate_priority <= 1.0:
                selection_priority = round(candidate_priority, 3)
            else:
                selection_priority = round(min(0.99, max(0.01, float(item.get("_selection_score", 0.0)) / max_score)), 3)

            finalized.append(
                {
                    "candidate_id": _text(item.get("candidate_id")) or None,
                    "selected_rank": index,
                    "selection_score": round(float(item.get("_selection_score", 0.0)), 3),
                    "selection_priority": selection_priority,
                    "hypothesis_id": _text(item.get("hypothesis_id")) or None,
                    "origin_hypothesis_id": _text(item.get("origin_hypothesis_id")) or None,
                    "diagnosis_type": _text(item.get("diagnosis_type")) or None,
                    "sequence_strategy": _text(item.get("sequence_strategy")) or None,
                    "target_function": _text(item.get("target_function")) or None,
                    "guided_sequence": _normalize_path(item.get("guided_sequence")),
                    "sender_policy": _text(item.get("sender_policy")) or None,
                    "value_policy": _text(item.get("value_policy")) or None,
                    "argument_policy": _text(item.get("argument_policy")) or None,
                    "prestate_policy": _text(item.get("prestate_policy")) or None,
                    "dryrun_status": _text(item.get("dryrun_status")) or None,
                    "selection_reason": list(item.get("_selection_reason") or []),
                    "source": "guided_sequence_selection_v1",
                    "selection_origin": mode,
                }
            )
        return finalized

    def _select_with_llm(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.llm_provider is None:
            raise RuntimeError("llm_selector_unavailable")

        ordered = self._sorted_eligible_candidates(candidates)
        if not ordered:
            return []

        compact_candidates = [self._compact_candidate_for_llm(item) for item in ordered[:50]]
        prompt = self._build_llm_prompt(compact_candidates)
        payload = self.llm_provider.generate_json(prompt)
        selected_items = payload.get("selected")
        if not isinstance(selected_items, list):
            raise RuntimeError("llm_selector_invalid_response")

        eligible_by_id = {item["candidate_id"]: item for item in ordered if _text(item.get("candidate_id"))}
        validated = []
        seen_ids = set()
        for item in selected_items:
            if len(validated) >= self.max_selected:
                break
            if not isinstance(item, dict):
                continue
            candidate_id = _text(item.get("candidate_id"))
            if not candidate_id or candidate_id in seen_ids:
                continue
            if candidate_id not in eligible_by_id:
                continue

            base_candidate = dict(eligible_by_id[candidate_id])
            reason_text = _text(item.get("selection_reason")).strip()
            llm_priority = _as_float(item.get("selection_priority"), default=-1.0)
            if llm_priority < 0.0 or llm_priority > 1.0:
                llm_priority = _as_float(base_candidate.get("priority"), default=0.5)
            base_candidate["_selection_reason"] = [reason_text] if reason_text else list(base_candidate.get("_selection_reason") or [])
            base_candidate["_llm_selection_priority"] = round(min(max(llm_priority, 0.0), 1.0), 3)
            validated.append(base_candidate)
            seen_ids.add(candidate_id)

        if not validated:
            raise RuntimeError("llm_selector_empty_selection")

        validated.sort(
            key=lambda item: (
                -_as_float(item.get("_llm_selection_priority"), default=0.0),
                -_as_float(item.get("_selection_score"), default=0.0),
                _text(item.get("candidate_id")),
            )
        )

        finalized = self._finalize_selected(validated[: self.max_selected], mode="llm")
        for item, validated_item in zip(finalized, validated):
            item["selection_priority"] = round(_as_float(validated_item.get("_llm_selection_priority"), default=item["selection_priority"]), 3)
        return finalized

    @staticmethod
    def _compact_candidate_for_llm(candidate: Dict[str, Any]) -> Dict[str, Any]:
        check_summary = {}
        checks = candidate.get("dryrun_checks")
        if isinstance(checks, dict):
            for key, value in checks.items():
                if isinstance(value, dict):
                    check_summary[key] = {
                        "status": _text(value.get("status")) or "unknown",
                        "reason": _text(value.get("reason"))[:160],
                    }

        return {
            "candidate_id": _text(candidate.get("candidate_id")),
            "hypothesis_id": _text(candidate.get("hypothesis_id")),
            "diagnosis_type": _text(candidate.get("diagnosis_type")),
            "sequence_strategy": _text(candidate.get("sequence_strategy")),
            "target_function": _text(candidate.get("target_function")),
            "guided_sequence": _normalize_path(candidate.get("guided_sequence")),
            "dryrun_status": _text(candidate.get("dryrun_status")),
            "checks": check_summary,
            "sender_policy": _text(candidate.get("sender_policy")),
            "value_policy": _text(candidate.get("value_policy")),
            "argument_policy": _text(candidate.get("argument_policy")),
            "prestate_policy": _text(candidate.get("prestate_policy")),
            "priority": _as_float(candidate.get("priority"), default=0.0),
            "reason": _text(candidate.get("reason"))[:240],
        }

    def _build_llm_prompt(self, compact_candidates: Sequence[Dict[str, Any]]) -> str:
        prompt_payload = {
            "task": "Select the best repair sequence candidates for later non-executing validation.",
            "rules": [
                "Return only valid JSON.",
                "Choose at most {} candidates.".format(self.max_selected),
                "Only use candidate_id values that appear in the input.",
                "Do not invent new candidates, sequences, functions, or policies.",
                "Do not select unmappable or skipped candidates.",
                "Prefer mappable candidates over partial candidates.",
                "Prefer strong pre-state and permission repair sequences when they are concrete and compact.",
            ],
            "output_schema": {
                "selected": [
                    {
                        "candidate_id": "RSC_1",
                        "selection_priority": 0.95,
                        "selection_reason": "brief reason",
                    }
                ]
            },
            "candidates": list(compact_candidates),
        }
        return json.dumps(prompt_payload, ensure_ascii=True, indent=2)

    def _build_llm_metadata(
        self,
        used: bool,
        fallback_used: bool,
        fallback_reason: Optional[str],
    ) -> Dict[str, Any]:
        model = None
        api_style = None
        if self.llm_provider is not None:
            config = getattr(self.llm_provider, "config", None)
            model = _text(getattr(config, "model", None)) or None
            try:
                resolved_api_style = self.llm_provider.resolved_api_style()
            except Exception:
                resolved_api_style = _text(getattr(config, "api_style", None))
            api_style = _text(resolved_api_style) or None
        return {
            "used": bool(used),
            "fallback_used": bool(fallback_used),
            "fallback_reason": _text(fallback_reason) or None,
            "model": model,
            "api_style": api_style,
        }

