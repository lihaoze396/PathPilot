# -*- coding: utf-8 -*-

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


PERMISSION_KEYWORDS = (
    "owner",
    "onlyowner",
    "admin",
    "authorized",
    "authorize",
    "permission",
    "role",
    "whitelist",
    "blacklist",
    "trading",
    "pause",
)
VALUE_KEYWORDS = ("value", "payable", "fallback", "receive")
ARGUMENT_KEYWORDS = (
    "argument",
    "parameter",
    "consistency",
    "bool",
    "amount",
    "threshold",
    "limit",
    "max",
    "min",
    "fee",
    "tax",
)
BOOLEAN_FUNCTION_KEYWORDS = (
    "set",
    "update",
    "enable",
    "disable",
    "pause",
    "unpause",
    "trading",
    "blacklist",
    "whitelist",
    "exclude",
    "include",
)
AMOUNT_FUNCTION_KEYWORDS = (
    "fee",
    "tax",
    "limit",
    "max",
    "threshold",
    "amount",
    "liquidity",
    "swap",
)
PRESTATE_KEYWORDS = (
    "trading",
    "router",
    "pair",
    "approve",
    "whitelist",
    "blacklist",
    "pause",
    "enable",
    "set",
    "update",
    "allowance",
)
PERMISSION_SETUP_POSITIVE = (
    "authorize",
    "transferownership",
    "setowner",
    "setadmin",
    "owner",
    "admin",
    "whitelist",
    "include",
    "exclude",
    "role",
)
PERMISSION_SETUP_NEGATIVE = (
    "unauthorize",
    "renounce",
    "remove",
    "disable",
    "blacklist",
)
CONFIG_KEYWORDS = (
    "set",
    "update",
    "config",
    "configure",
    "router",
    "pair",
    "whitelist",
    "blacklist",
    "pause",
    "unpause",
    "enable",
    "trading",
    "tax",
    "fee",
    "limit",
)
TRADING_KEYWORDS = ("trading",)


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


def _read_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except Exception:
        return []
    return records


def _normalize_path(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            output.append(text)
    return output


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            output.append(text)
    return output


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _path_signature(path: Sequence[str], sink: str = "") -> Tuple[str, ...]:
    parts = [str(item).strip() for item in path if str(item).strip()]
    if sink and (not parts or parts[-1] != sink):
        parts.append(str(sink))
    return tuple(parts)


def _flatten_texts(values: Sequence[Any]) -> List[str]:
    output: List[str] = []
    for value in values:
        if isinstance(value, list):
            output.extend(_flatten_texts(value))
        elif value not in (None, ""):
            output.append(str(value))
    return output


def _contains_any(texts: Sequence[str], keywords: Sequence[str]) -> bool:
    joined = "\n".join(str(item).lower() for item in texts if item not in (None, ""))
    return any(keyword in joined for keyword in keywords)


def _function_matches(function_name: str, keywords: Sequence[str]) -> bool:
    lowered = str(function_name or "").strip().lower()
    return bool(lowered and any(keyword in lowered for keyword in keywords))


def _recent_unique(values: Sequence[str], limit: int) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
        if len(ordered) >= limit:
            break
    return ordered


class RepairDiagnosisBuilder:
    def __init__(
        self,
        output_dir: str,
        blocker_evidence_path: Optional[str] = None,
        ksi_evidence_path: Optional[str] = None,
        tfi_focus_path: Optional[str] = None,
        agent_path_plan_path: Optional[str] = None,
        max_recent_sequences: int = 20,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.max_recent_sequences = max(1, int(max_recent_sequences or 1))

        self._requested_blocker_evidence_path = (
            os.path.abspath(blocker_evidence_path) if blocker_evidence_path else None
        )
        self._requested_ksi_evidence_path = os.path.abspath(ksi_evidence_path) if ksi_evidence_path else None
        self._requested_tfi_focus_path = os.path.abspath(tfi_focus_path) if tfi_focus_path else None
        self._requested_path_plan_path = os.path.abspath(agent_path_plan_path) if agent_path_plan_path else None

        self.default_blocker_evidence_path = (
            os.path.join(self.output_dir, "blocker_evidence.json") if self.output_dir else ""
        )
        self.default_ksi_evidence_path = (
            os.path.join(self.output_dir, "ksi_evidence.jsonl") if self.output_dir else ""
        )
        self.default_tfi_focus_path = (
            os.path.join(self.output_dir, "tfi_focus_targets.jsonl") if self.output_dir else ""
        )
        self.default_agent_path_plan_path = (
            os.path.join(self.output_dir, "agent_path_plan.json") if self.output_dir else ""
        )
        self.default_proposed_path_plan_path = (
            os.path.join(self.output_dir, "proposed_path_plan.json") if self.output_dir else ""
        )

        self.loaded_inputs: Dict[str, bool] = {
            "blocker_evidence": False,
            "ksi_evidence": False,
            "tfi_focus": False,
            "path_plan": False,
        }
        self.input_paths: Dict[str, Optional[str]] = {
            "blocker_evidence": self._requested_blocker_evidence_path or self.default_blocker_evidence_path or None,
            "ksi_evidence": self._requested_ksi_evidence_path or self.default_ksi_evidence_path or None,
            "tfi_focus": self._requested_tfi_focus_path or self.default_tfi_focus_path or None,
            "path_plan": self._requested_path_plan_path or self.default_agent_path_plan_path or None,
        }

    def build_diagnoses(self) -> List[Dict[str, Any]]:
        blocker_payload = self._load_json_with_fallback(
            self._requested_blocker_evidence_path,
            self.default_blocker_evidence_path,
            "blocker_evidence",
        )
        ksi_records = self._load_jsonl_with_fallback(
            self._requested_ksi_evidence_path,
            self.default_ksi_evidence_path,
            "ksi_evidence",
        )
        tfi_records = self._load_jsonl_with_fallback(
            self._requested_tfi_focus_path,
            self.default_tfi_focus_path,
            "tfi_focus",
        )
        path_plan_payload = self._load_path_plan_with_fallback()

        blocker_hypotheses = blocker_payload.get("hypotheses")
        if not isinstance(blocker_hypotheses, list):
            return []

        ksi_by_hypothesis: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in ksi_records:
            hypothesis_id = _text(record.get("hypothesis_id"))
            if hypothesis_id:
                ksi_by_hypothesis[hypothesis_id].append(record)

        tfi_by_hypothesis, tfi_by_signature = self._index_tfi_records(tfi_records)
        path_plan_by_id, path_plan_by_signature = self._index_path_plan(path_plan_payload)
        function_pool = self._collect_function_pool(
            blocker_hypotheses=blocker_hypotheses,
            ksi_records=ksi_records,
            tfi_records=tfi_records,
            path_plan_payload=path_plan_payload,
        )

        support_functions = self._build_support_function_groups(function_pool)
        diagnoses: List[Dict[str, Any]] = []
        for entry in blocker_hypotheses:
            if not isinstance(entry, dict):
                continue

            hypothesis_id = _text(entry.get("hypothesis_id"))
            path = _normalize_path(entry.get("path"))
            sink = _text(entry.get("sink")) or (path[-1] if path else "")
            if not hypothesis_id or not path:
                continue

            signature = _path_signature(path, sink)
            plan_entry = path_plan_by_id.get(hypothesis_id) or path_plan_by_signature.get(signature) or {}
            origin_hypothesis_id = _text(plan_entry.get("origin_hypothesis_id"))
            matching_tfi = list(tfi_by_hypothesis.get(hypothesis_id, []))
            if origin_hypothesis_id:
                matching_tfi.extend(tfi_by_hypothesis.get(origin_hypothesis_id, []))
            matching_tfi.extend(tfi_by_signature.get(signature, []))
            matching_tfi = self._dedupe_dict_records(matching_tfi)

            hypothesis_ksi = list(ksi_by_hypothesis.get(hypothesis_id, []))
            diagnosis = self._build_single_diagnosis(
                entry=entry,
                hypothesis_id=hypothesis_id,
                origin_hypothesis_id=origin_hypothesis_id or None,
                path=path,
                sink=sink,
                ksi_records=hypothesis_ksi,
                tfi_records=matching_tfi,
                support_functions=support_functions,
            )
            if diagnosis:
                diagnoses.append(diagnosis)

        diagnoses.sort(
            key=lambda item: (
                -_as_float(item.get("confidence")),
                _text(item.get("hypothesis_id")),
                _text(item.get("diagnosis_type")),
            )
        )
        return diagnoses

    def write_diagnoses(
        self,
        diagnoses: List[Dict[str, Any]],
        filename: str = "repair_diagnosis.json",
    ) -> str:
        if not self.output_dir:
            return ""
        output_path = os.path.join(self.output_dir, filename)
        payload = {
            "loaded_inputs": dict(self.loaded_inputs),
            "diagnoses": [self._sanitize_diagnosis(item) for item in diagnoses],
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
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

    def _load_jsonl_with_fallback(
        self,
        requested_path: Optional[str],
        default_path: Optional[str],
        key: str,
    ) -> List[Dict[str, Any]]:
        candidate_paths: List[str] = []
        if requested_path:
            candidate_paths.append(requested_path)
        if default_path and default_path not in candidate_paths:
            candidate_paths.append(default_path)
        self.loaded_inputs[key] = False
        self.input_paths[key] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_jsonl(path)
            if payload:
                self.loaded_inputs[key] = True
                self.input_paths[key] = path
                return payload
        return []

    def _load_path_plan_with_fallback(self) -> Dict[str, Any]:
        candidate_paths: List[str] = []
        if self._requested_path_plan_path:
            candidate_paths.append(self._requested_path_plan_path)
        if self.default_agent_path_plan_path and self.default_agent_path_plan_path not in candidate_paths:
            candidate_paths.append(self.default_agent_path_plan_path)
        if self.default_proposed_path_plan_path and self.default_proposed_path_plan_path not in candidate_paths:
            candidate_paths.append(self.default_proposed_path_plan_path)

        self.loaded_inputs["path_plan"] = False
        self.input_paths["path_plan"] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_json(path)
            hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
            if isinstance(hypotheses, list):
                self.loaded_inputs["path_plan"] = True
                self.input_paths["path_plan"] = path
                return payload
        return {}

    @staticmethod
    def _index_tfi_records(
        tfi_records: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, ...], List[Dict[str, Any]]]]:
        by_hypothesis: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_signature: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        for record in tfi_records:
            if not isinstance(record, dict):
                continue
            hypothesis_id = _text(record.get("hypothesis_id"))
            if hypothesis_id:
                by_hypothesis[hypothesis_id].append(record)
            path = _normalize_path(record.get("path"))
            sink = _text(record.get("sink")) or (_text(record.get("focus_frontier")) or _text(record.get("focus_function")))
            signature = _path_signature(path, sink)
            if signature:
                by_signature[signature].append(record)
        return by_hypothesis, by_signature

    @staticmethod
    def _index_path_plan(
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, ...], Dict[str, Any]]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        by_signature: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
        if not isinstance(hypotheses, list):
            return by_id, by_signature
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            hypothesis_id = _text(item.get("id"))
            path = _normalize_path(item.get("path"))
            sink = _text(item.get("sink")) or (path[-1] if path else "")
            signature = _path_signature(path, sink)
            if hypothesis_id:
                by_id[hypothesis_id] = item
            if signature:
                by_signature[signature] = item
        return by_id, by_signature

    @staticmethod
    def _collect_function_pool(
        blocker_hypotheses: Sequence[Dict[str, Any]],
        ksi_records: Sequence[Dict[str, Any]],
        tfi_records: Sequence[Dict[str, Any]],
        path_plan_payload: Dict[str, Any],
    ) -> Set[str]:
        function_pool: Set[str] = set()
        for entry in blocker_hypotheses:
            if not isinstance(entry, dict):
                continue
            function_pool.update(_normalize_path(entry.get("path")))
            sink = _text(entry.get("sink"))
            if sink:
                function_pool.add(sink)
        for record in ksi_records:
            observations = record.get("observations") if isinstance(record.get("observations"), dict) else {}
            function_pool.update(_normalize_path(observations.get("functions")))
        for record in tfi_records:
            if not isinstance(record, dict):
                continue
            function_pool.update(_normalize_path(record.get("path")))
            for field in ("sink", "focus_function", "focus_frontier"):
                value = _text(record.get(field))
                if value:
                    function_pool.add(value)
        hypotheses = path_plan_payload.get("hypotheses") if isinstance(path_plan_payload, dict) else None
        if isinstance(hypotheses, list):
            for item in hypotheses:
                if not isinstance(item, dict):
                    continue
                function_pool.update(_normalize_path(item.get("path")))
                sink = _text(item.get("sink"))
                if sink:
                    function_pool.add(sink)
        function_pool.discard("")
        return function_pool

    @staticmethod
    def _build_support_function_groups(function_pool: Set[str]) -> Dict[str, List[str]]:
        functions = sorted(function_pool)

        permission_like = [
            function_name
            for function_name in functions
            if _function_matches(function_name, PERMISSION_SETUP_POSITIVE)
            and not _function_matches(function_name, PERMISSION_SETUP_NEGATIVE)
        ]
        config_like = [function_name for function_name in functions if _function_matches(function_name, CONFIG_KEYWORDS)]
        trading_like = [function_name for function_name in functions if _function_matches(function_name, TRADING_KEYWORDS)]
        approve_like = [function_name for function_name in functions if "approve" in function_name.lower()]
        fallback_like = [
            function_name
            for function_name in functions
            if function_name.lower() in ("fallback", "receive")
        ]
        return {
            "permission_like": permission_like[:20],
            "config_like": config_like[:20],
            "trading_like": trading_like[:20],
            "approve_like": approve_like[:20],
            "fallback_like": fallback_like[:20],
        }

    def _build_single_diagnosis(
        self,
        entry: Dict[str, Any],
        hypothesis_id: str,
        origin_hypothesis_id: Optional[str],
        path: List[str],
        sink: str,
        ksi_records: Sequence[Dict[str, Any]],
        tfi_records: Sequence[Dict[str, Any]],
        support_functions: Dict[str, List[str]],
    ) -> Optional[Dict[str, Any]]:
        progress = entry.get("progress_summary") if isinstance(entry.get("progress_summary"), dict) else {}
        runtime = entry.get("runtime_evidence") if isinstance(entry.get("runtime_evidence"), dict) else {}
        repair = entry.get("repair_objective") if isinstance(entry.get("repair_objective"), dict) else {}

        sink_function_reached_count = _as_int(progress.get("sink_function_reached_count"))
        sink_reverted_count = _as_int(progress.get("sink_reverted_count"))
        blocked_count = _as_int(progress.get("blocked_count"))
        revert_at_sink_count = _as_int(runtime.get("revert_at_sink_count"))
        sender_change_count = _as_int(runtime.get("sender_change_count"))
        top_suspected_blockers = _normalize_string_list(runtime.get("top_suspected_blockers"))
        recent_revert_reasons = _normalize_string_list(runtime.get("recent_revert_reasons"))
        evidence_lines = _normalize_string_list(entry.get("evidence"))
        hints = _normalize_string_list(repair.get("hints"))
        repair_action = _text(repair.get("action"))
        dominant_blocked_at = _text(progress.get("dominant_blocked_at"))
        frontier = _text(progress.get("frontier")) or sink
        best_prefix = _normalize_path(progress.get("best_prefix"))

        tfi_focus_types = [_text(record.get("focus_type")) for record in tfi_records if isinstance(record, dict)]
        tfi_focus_types = [item for item in tfi_focus_types if item]
        tfi_reasons = [_text(record.get("reason")) for record in tfi_records if isinstance(record, dict)]
        tfi_reasons = [item for item in tfi_reasons if item]

        value_sequence_numbers: List[float] = []
        value_has_zero = False
        value_has_nonzero = False
        args_nonempty = False
        ksi_related_sequences: List[str] = []
        observed_functions: List[str] = []
        for record in ksi_records:
            observations = record.get("observations") if isinstance(record.get("observations"), dict) else {}
            observed_functions.extend(_normalize_path(observations.get("functions")))
            ksi_related_sequences.append(_text(record.get("sequence_id")))
            for value in observations.get("value_sequence", []) if isinstance(observations.get("value_sequence"), list) else []:
                numeric_value = _as_float(value)
                value_sequence_numbers.append(numeric_value)
                if numeric_value == 0:
                    value_has_zero = True
                if numeric_value != 0:
                    value_has_nonzero = True
            for arg in observations.get("args_sequence", []) if isinstance(observations.get("args_sequence"), list) else []:
                text = _text(arg)
                if text not in ("", "[]"):
                    args_nonempty = True

        combined_texts = _flatten_texts(
            [
                path,
                [sink, frontier, dominant_blocked_at],
                top_suspected_blockers,
                recent_revert_reasons,
                evidence_lines,
                hints,
                tfi_focus_types,
                tfi_reasons,
                observed_functions,
            ]
        )
        path_texts = _flatten_texts([path, [sink, frontier, dominant_blocked_at]])

        caller_score = 0
        if "caller_or_permission_condition" in tfi_focus_types:
            caller_score += 2
        if sender_change_count > 0 or "sender_change" in top_suspected_blockers:
            caller_score += 1
        if revert_at_sink_count > 0 or "revert_at_sink" in top_suspected_blockers:
            caller_score += 1
        if repair_action == "repair_sink_execution":
            caller_score += 1
        if _contains_any(combined_texts, PERMISSION_KEYWORDS):
            caller_score += 2

        value_score = 0
        target_is_fallback = sink.lower() in ("fallback", "receive") or frontier.lower() in ("fallback", "receive")
        if target_is_fallback:
            value_score += 2
        if value_has_zero and value_has_nonzero and (sink_reverted_count > 0 or blocked_count > 0):
            value_score += 2
        if _contains_any(combined_texts, VALUE_KEYWORDS):
            value_score += 1

        argument_score = 0
        if sink_reverted_count >= 2:
            argument_score += 2
        if args_nonempty:
            argument_score += 1
        if _contains_any(combined_texts, ARGUMENT_KEYWORDS):
            argument_score += 2
        if _function_matches(sink, BOOLEAN_FUNCTION_KEYWORDS) or _function_matches(sink, AMOUNT_FUNCTION_KEYWORDS):
            argument_score += 1

        prestate_score = 0
        if any(function_name.lower() in ("transferfrom", "approve", "allowance") for function_name in path_texts):
            prestate_score += 2
        if "repeated_frontier_block" in tfi_focus_types or "frontier_block" in top_suspected_blockers:
            prestate_score += 1
        if _contains_any(path_texts, PRESTATE_KEYWORDS):
            prestate_score += 1
        if "state_or_configuration_dependency" in top_suspected_blockers:
            prestate_score += 1
        if _contains_any(recent_revert_reasons, ("allowance", "balance", "approve")):
            prestate_score += 2
        if support_functions.get("approve_like") and any("transferfrom" == item.lower() for item in path_texts):
            prestate_score += 1

        repeated_frontier_score = 0
        if blocked_count >= 3:
            repeated_frontier_score += 1
        if dominant_blocked_at:
            repeated_frontier_score += 1
        if "repeated_frontier_block" in tfi_focus_types:
            repeated_frontier_score += 2

        selected_type = ""
        selected_confidence = 0.0
        selected_reason = ""
        target_function = sink

        def choose(diagnosis_type: str, score: int, confidence: float, reason: str, target: str) -> None:
            nonlocal selected_type, selected_confidence, selected_reason, target_function
            if score <= 0:
                return
            if confidence > selected_confidence:
                selected_type = diagnosis_type
                selected_confidence = confidence
                selected_reason = reason
                target_function = target

        if value_score >= 3:
            confidence = min(0.95, 0.60 + 0.10 * min(value_score, 3))
            choose(
                "value_condition_block",
                value_score,
                confidence,
                "value usage patterns and fallback/value signals suggest a value-dependent block around the sink",
                sink or frontier,
            )

        if prestate_score >= 3:
            confidence = min(0.95, 0.60 + 0.08 * min(prestate_score, 4))
            choose(
                "prestate_dependency_block",
                prestate_score,
                confidence,
                "path shape and runtime evidence suggest missing pre-state or configuration dependencies",
                sink or frontier,
            )

        if caller_score >= 3:
            confidence = min(0.95, 0.60 + 0.08 * min(caller_score, 4))
            choose(
                "caller_or_permission_block",
                caller_score,
                confidence,
                "sink function is reached or approached with repeated reverts; evidence suggests caller or permission condition",
                dominant_blocked_at or sink or frontier,
            )

        if argument_score >= 3:
            confidence = min(0.90, 0.58 + 0.08 * min(argument_score, 4))
            choose(
                "argument_constraint_block",
                argument_score,
                confidence,
                "repeated reverts with argument activity suggest sink or frontier argument constraints",
                sink or frontier,
            )

        if not selected_type and repeated_frontier_score >= 3:
            confidence = min(0.80, 0.58 + 0.08 * min(repeated_frontier_score, 3))
            choose(
                "repeated_frontier_block",
                repeated_frontier_score,
                confidence,
                "the same frontier repeatedly blocks progress before the sink",
                dominant_blocked_at or frontier or sink,
            )

        if not selected_type and sink_reverted_count > 0:
            selected_type = "sink_reverted_unknown"
            selected_confidence = 0.45 if revert_at_sink_count <= 1 else 0.50
            selected_reason = "sink function is reached but the underlying failure pattern is still unclear"
            target_function = sink or frontier
        elif not selected_type and blocked_count > 0:
            selected_type = "repeated_frontier_block"
            selected_confidence = 0.55
            selected_reason = "path progress repeatedly stops around the same frontier"
            target_function = dominant_blocked_at or frontier or sink

        if not selected_type:
            return None

        related_sequences = _recent_unique(
            _flatten_texts(
                [
                    runtime.get("recent_sequences"),
                    ksi_related_sequences,
                    [record.get("sequence_id") for record in tfi_records if isinstance(record, dict)],
                    [record.get("related_sequences") for record in tfi_records if isinstance(record, dict)],
                ]
            ),
            self.max_recent_sequences,
        )

        return {
            "hypothesis_id": hypothesis_id,
            "origin_hypothesis_id": origin_hypothesis_id,
            "path": list(path),
            "sink": sink,
            "target_function": target_function or sink or frontier,
            "diagnosis_type": selected_type,
            "confidence": round(selected_confidence, 3),
            "evidence": {
                "sink_function_reached_count": sink_function_reached_count,
                "sink_reverted_count": sink_reverted_count,
                "blocked_count": blocked_count,
                "revert_at_sink_count": revert_at_sink_count,
                "sender_change_count": sender_change_count,
                "top_suspected_blockers": list(top_suspected_blockers),
            },
            "related_sequences": related_sequences,
            "reason": selected_reason,
            "frontier": frontier,
            "dominant_blocked_at": dominant_blocked_at,
            "best_prefix": list(best_prefix),
            "support_functions": {
                "permission_like": list(support_functions.get("permission_like", [])),
                "config_like": list(support_functions.get("config_like", [])),
                "trading_like": list(support_functions.get("trading_like", [])),
                "approve_like": list(support_functions.get("approve_like", [])),
                "fallback_like": list(support_functions.get("fallback_like", [])),
            },
        }

    @staticmethod
    def _dedupe_dict_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for record in records:
            try:
                signature = json.dumps(record, sort_keys=True, ensure_ascii=True)
            except Exception:
                signature = repr(record)
            if signature in seen:
                continue
            seen.add(signature)
            output.append(record)
        return output

    @staticmethod
    def _sanitize_diagnosis(item: Dict[str, Any]) -> Dict[str, Any]:
        return dict(item)


class RepairSequenceCandidateBuilder:
    def __init__(
        self,
        output_dir: str,
        diagnoses: List[Dict[str, Any]],
        max_candidates_per_hypothesis: int = 5,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.diagnoses = [item for item in diagnoses if isinstance(item, dict)]
        self.max_candidates_per_hypothesis = max(1, int(max_candidates_per_hypothesis or 1))
        self.loaded_inputs: Dict[str, bool] = {}
        self.diagnosis_output_path: Optional[str] = None
        self.candidates_output_path: Optional[str] = None
        self.last_summary_payload: Dict[str, Any] = {}

    def build_candidates(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for diagnosis in self.diagnoses:
            candidates = self._build_candidates_for_diagnosis(diagnosis)
            for candidate in candidates:
                hypothesis_id = _text(candidate.get("hypothesis_id"))
                if hypothesis_id:
                    grouped[hypothesis_id].append(candidate)

        final_candidates: List[Dict[str, Any]] = []
        for hypothesis_id, candidates in grouped.items():
            del hypothesis_id
            deduped = self._dedupe_candidates(candidates)
            deduped.sort(
                key=lambda item: (
                    -_as_float(item.get("priority")),
                    _text(item.get("hypothesis_id")),
                    _text(item.get("sequence_strategy")),
                )
            )
            final_candidates.extend(deduped[: self.max_candidates_per_hypothesis])

        final_candidates.sort(
            key=lambda item: (
                -_as_float(item.get("priority")),
                _text(item.get("hypothesis_id")),
                _text(item.get("sequence_strategy")),
            )
        )
        for index, candidate in enumerate(final_candidates, start=1):
            candidate["candidate_id"] = "RSC_{}".format(index)
        return final_candidates

    def write_candidates(
        self,
        candidates: List[Dict[str, Any]],
        filename: str = "guided_sequence_candidates.json",
    ) -> str:
        if not self.output_dir:
            return ""
        output_path = os.path.join(self.output_dir, filename)
        payload = {"candidates": candidates}
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        self.candidates_output_path = output_path
        return output_path

    def write_summary(
        self,
        candidates: List[Dict[str, Any]],
        filename: str = "guided_sequence_summary.json",
    ) -> str:
        if not self.output_dir:
            return ""
        diagnosis_type_counts = Counter()
        sequence_strategy_counts = Counter()
        target_function_counts = Counter()
        for diagnosis in self.diagnoses:
            diagnosis_type = _text(diagnosis.get("diagnosis_type"))
            if diagnosis_type:
                diagnosis_type_counts[diagnosis_type] += 1
            target_function = _text(diagnosis.get("target_function"))
            if target_function:
                target_function_counts[target_function] += 1
        for candidate in candidates:
            strategy = _text(candidate.get("sequence_strategy"))
            if strategy:
                sequence_strategy_counts[strategy] += 1

        payload = {
            "enabled": True,
            "diagnosis_count": len(self.diagnoses),
            "candidate_count": len(candidates),
            "diagnosis_type_counts": dict(diagnosis_type_counts),
            "sequence_strategy_counts": dict(sequence_strategy_counts),
            "top_target_functions": dict(target_function_counts.most_common(10)),
            "loaded_inputs": dict(self.loaded_inputs),
            "output_paths": {
                "repair_diagnosis": self.diagnosis_output_path,
                "guided_sequence_candidates": self.candidates_output_path,
            },
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

    def _build_candidates_for_diagnosis(self, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        diagnosis_type = _text(diagnosis.get("diagnosis_type"))
        if not diagnosis_type:
            return []

        path = _normalize_path(diagnosis.get("path"))
        target_function = _text(diagnosis.get("target_function")) or (_text(diagnosis.get("sink")) or (path[-1] if path else ""))
        support_functions = diagnosis.get("support_functions") if isinstance(diagnosis.get("support_functions"), dict) else {}
        permission_like = _normalize_string_list(support_functions.get("permission_like"))
        config_like = _normalize_string_list(support_functions.get("config_like"))
        trading_like = _normalize_string_list(support_functions.get("trading_like"))
        approve_like = _normalize_string_list(support_functions.get("approve_like"))
        fallback_like = _normalize_string_list(support_functions.get("fallback_like"))

        base_priority = _as_float(diagnosis.get("confidence"))
        candidates: List[Dict[str, Any]] = []

        def add_candidate(
            guided_sequence: List[str],
            sequence_strategy: str,
            sender_policy: str,
            value_policy: str,
            argument_policy: str,
            prestate_policy: str,
            priority_bonus: float,
            reason: str,
        ) -> None:
            sequence = _normalize_path(guided_sequence)
            if not sequence:
                return
            priority = max(0.0, min(base_priority + priority_bonus, 0.99))
            candidates.append(
                {
                    "candidate_id": None,
                    "hypothesis_id": _text(diagnosis.get("hypothesis_id")),
                    "origin_hypothesis_id": diagnosis.get("origin_hypothesis_id"),
                    "diagnosis_type": diagnosis_type,
                    "target_function": target_function,
                    "base_path": list(path),
                    "guided_sequence": sequence,
                    "sequence_strategy": sequence_strategy,
                    "sender_policy": sender_policy,
                    "value_policy": value_policy,
                    "argument_policy": argument_policy,
                    "prestate_policy": prestate_policy,
                    "priority": round(priority, 3),
                    "reason": reason,
                    "source": "ksi_guided_guided_sequence_v1",
                }
            )

        if diagnosis_type == "caller_or_permission_block":
            add_candidate(
                guided_sequence=path,
                sequence_strategy="preserve_prefix_and_retry_sink",
                sender_policy="try_owner_or_deployer_or_previous_success_sender",
                value_policy="keep_original",
                argument_policy="mutate_sink_arguments",
                prestate_policy="none",
                priority_bonus=0.10,
                reason="repair sink revert by varying caller and sink arguments",
            )
            permission_function = self._choose_support_function(permission_like, exclude={target_function})
            if permission_function:
                add_candidate(
                    guided_sequence=[permission_function, target_function],
                    sequence_strategy="permission_setup_then_sink",
                    sender_policy="try_owner_or_deployer",
                    value_policy="keep_original",
                    argument_policy="mutate_permission_and_sink_args",
                    prestate_policy="permission_setup",
                    priority_bonus=0.07,
                    reason="repair permission-sensitive sink by attempting an observed permission-like setup before the target",
                )
            add_candidate(
                guided_sequence=path,
                sequence_strategy="same_sender_prefix_retry",
                sender_policy="keep_same_sender_across_sequence",
                value_policy="keep_original",
                argument_policy="mutate_sink_arguments",
                prestate_policy="none",
                priority_bonus=0.03,
                reason="repair caller-dependent sink by preserving a stable sender across the full sequence",
            )

        elif diagnosis_type == "value_condition_block":
            add_candidate(
                guided_sequence=path or [target_function],
                sequence_strategy="retry_with_nonzero_value",
                sender_policy="any_existing_sender",
                value_policy="try_small_nonzero_value",
                argument_policy="keep_original",
                prestate_policy="none",
                priority_bonus=0.08,
                reason="probe value-dependent behavior with a small non-zero call value",
            )
            add_candidate(
                guided_sequence=path or [target_function],
                sequence_strategy="retry_with_zero_value",
                sender_policy="any_existing_sender",
                value_policy="force_zero_value",
                argument_policy="keep_original",
                prestate_policy="none",
                priority_bonus=0.05,
                reason="probe whether the sink only succeeds with zero call value",
            )
            if target_function.lower() in ("fallback", "receive") or target_function in fallback_like:
                add_candidate(
                    guided_sequence=[target_function],
                    sequence_strategy="fallback_value_probe",
                    sender_policy="any_existing_sender",
                    value_policy="try_small_nonzero_value",
                    argument_policy="keep_original",
                    prestate_policy="none",
                    priority_bonus=0.06,
                    reason="probe fallback or receive behavior with a minimal value-carrying call",
                )

        elif diagnosis_type == "argument_constraint_block":
            add_candidate(
                guided_sequence=path,
                sequence_strategy="mutate_sink_args",
                sender_policy="keep_original_sender",
                value_policy="keep_original",
                argument_policy="mutate_sink_arguments",
                prestate_policy="none",
                priority_bonus=0.08,
                reason="repair repeated sink revert by mutating sink-facing arguments",
            )
            if _function_matches(target_function, BOOLEAN_FUNCTION_KEYWORDS):
                add_candidate(
                    guided_sequence=path,
                    sequence_strategy="boolean_flip_args",
                    sender_policy="keep_original_sender",
                    value_policy="keep_original",
                    argument_policy="try_boolean_true_false",
                    prestate_policy="none",
                    priority_bonus=0.05,
                    reason="probe boolean-style configuration toggles around the target function",
                )
            if _function_matches(target_function, AMOUNT_FUNCTION_KEYWORDS):
                add_candidate(
                    guided_sequence=path,
                    sequence_strategy="amount_boundary_args",
                    sender_policy="keep_original_sender",
                    value_policy="keep_original",
                    argument_policy="try_zero_one_max_boundary_values",
                    prestate_policy="none",
                    priority_bonus=0.05,
                    reason="probe amount and threshold boundaries around the target function",
                )

        elif diagnosis_type == "prestate_dependency_block":
            transfer_from_index = self._find_function_index(path, "transferFrom")
            if transfer_from_index >= 0:
                approve_function = self._choose_support_function(approve_like)
                if approve_function:
                    prefix = path[:transfer_from_index]
                    suffix = path[transfer_from_index:]
                    sequence = list(prefix)
                    if approve_function not in sequence:
                        sequence.append(approve_function)
                    sequence.extend(suffix)
                    add_candidate(
                        guided_sequence=sequence,
                        sequence_strategy="approve_before_transferFrom",
                        sender_policy="any_existing_sender",
                        value_policy="keep_original",
                        argument_policy="mutate_transfer_arguments",
                        prestate_policy="insert_approve_before_transferFrom",
                        priority_bonus=0.10,
                        reason="insert an observed approve-like function before transferFrom to satisfy allowance pre-state",
                    )
            trading_function = self._choose_support_function(trading_like, exclude={target_function})
            if trading_function:
                add_candidate(
                    guided_sequence=[trading_function, target_function],
                    sequence_strategy="enable_trading_before_transfer",
                    sender_policy="try_owner_or_deployer",
                    value_policy="keep_original",
                    argument_policy="mutate_sink_arguments",
                    prestate_policy="insert_trading_enable",
                    priority_bonus=0.07,
                    reason="attempt an observed trading-related setup before retrying the target function",
                )
            config_function = self._choose_support_function(config_like, exclude={target_function})
            if config_function:
                add_candidate(
                    guided_sequence=[config_function, target_function],
                    sequence_strategy="config_before_sink",
                    sender_policy="try_owner_or_deployer_or_previous_success_sender",
                    value_policy="keep_original",
                    argument_policy="mutate_sink_arguments",
                    prestate_policy="insert_config_prestate",
                    priority_bonus=0.06,
                    reason="attempt an observed configuration-like function before retrying the blocked sink",
                )

        elif diagnosis_type == "sink_reverted_unknown":
            add_candidate(
                guided_sequence=path,
                sequence_strategy="conservative_retry",
                sender_policy="rotate_existing_senders",
                value_policy="try_zero_and_small_nonzero",
                argument_policy="mutate_sink_arguments",
                prestate_policy="none",
                priority_bonus=0.05,
                reason="retry conservatively by varying sender, value, and sink arguments without changing the path shape",
            )

        elif diagnosis_type == "repeated_frontier_block":
            best_prefix = _normalize_path(diagnosis.get("best_prefix"))
            frontier = _text(diagnosis.get("frontier")) or target_function
            sequence = list(best_prefix)
            if frontier and (not sequence or sequence[-1] != frontier):
                sequence.append(frontier)
            if not sequence and target_function:
                sequence = [target_function]
            add_candidate(
                guided_sequence=sequence,
                sequence_strategy="preserve_reached_prefix_and_mutate_frontier",
                sender_policy="keep_prefix_sender_stable",
                value_policy="keep_original",
                argument_policy="mutate_frontier_arguments",
                prestate_policy="none",
                priority_bonus=0.07,
                reason="preserve the reached prefix and mutate frontier-facing arguments around the repeated block",
            )

        return candidates

    @staticmethod
    def _find_function_index(path: Sequence[str], function_name: str) -> int:
        target = str(function_name or "").lower()
        for index, item in enumerate(path):
            if str(item or "").lower() == target:
                return index
        return -1

    @staticmethod
    def _choose_support_function(functions: Sequence[str], exclude: Optional[Set[str]] = None) -> Optional[str]:
        exclude = exclude or set()
        for function_name in functions:
            text = _text(function_name)
            if text and text not in exclude:
                return text
        return None

    @staticmethod
    def _dedupe_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, Tuple[str, ...]]] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            guided_sequence = _normalize_path(candidate.get("guided_sequence"))
            if not guided_sequence:
                continue
            signature = (
                _text(candidate.get("hypothesis_id")),
                _text(candidate.get("diagnosis_type")),
                tuple(guided_sequence),
            )
            if signature in seen:
                continue
            seen.add(signature)
            normalized = dict(candidate)
            normalized["guided_sequence"] = guided_sequence
            output.append(normalized)
        return output

