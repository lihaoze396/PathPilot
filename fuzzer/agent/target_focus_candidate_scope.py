# -*- coding: utf-8 -*-

"""TFI stage-3 target-local focusing with repair-local context.

This module does not generate or execute new repair sequences. It only consumes
existing KSI / blocker / repair artifacts and rewrites them into a local scope
that can constrain later agent-side repair selection.
"""

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


PERMISSION_KEYWORDS = (
    "owner",
    "admin",
    "auth",
    "authorize",
    "role",
    "whitelist",
    "blacklist",
    "exclude",
    "include",
    "pause",
    "unpause",
)

GENERIC_TOKEN_KEYWORDS = (
    "approve",
    "transfer",
    "transferfrom",
    "allowance",
    "balance",
)

CONFIG_KEYWORDS = (
    "set",
    "update",
    "config",
    "router",
    "pair",
    "fee",
    "tax",
    "limit",
    "max",
    "threshold",
    "enable",
    "disable",
)

ASSET_KEYWORDS = (
    "approve",
    "allowance",
    "transferfrom",
    "transfer",
    "balance",
)

VALUE_ENTRY_KEYWORDS = (
    "fallback",
    "receive",
    "payable",
)

FOCUS_PRIORITY = {
    "sink_reverted": 0,
    "repeated_frontier_block": 1,
    "caller_or_permission_condition": 2,
    "prefix_ready_frontier_unknown": 3,
    "target_seen_but_not_path_progress": 4,
    "target_only": 5,
    "unknown": 9,
}

REPAIR_FAILURE_MODE_MAP = {
    "hit_sink_but_reverted": "sink_reached_but_reverted",
    "target_not_seen": "target_seen_but_sink_not_reached",
    "prefix_not_preserved": "sequence_order_misaligned",
    "sequence_mismatch": "sequence_order_misaligned",
    "sender_or_permission_likely": "sender_binding_failed",
    "value_likely": "value_binding_failed",
    "args_or_state_likely": "argument_binding_failed",
    "construction_failed": "unknown",
    "execution_not_observed": "unknown",
    "unknown": "unknown",
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


def _read_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
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
                    rows.append(payload)
    except Exception:
        return []
    return rows


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _listify(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _clean_list(value: Any) -> List[str]:
    output: List[str] = []
    if not isinstance(value, list):
        return output
    for item in value:
        text = _text(item).strip()
        if text:
            output.append(text)
    return output


def _unique_list(values: Sequence[Any]) -> List[Any]:
    output: List[Any] = []
    seen = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=True) if isinstance(value, (dict, list)) else repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def _prefix_len(left: Sequence[str], right: Sequence[str]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def _ordered_overlap_score(left: Sequence[str], right: Sequence[str]) -> float:
    left_seq = [item for item in left if _text(item).strip()]
    right_seq = [item for item in right if _text(item).strip()]
    if not left_seq or not right_seq:
        return 0.0
    right_index = 0
    matches = 0
    for item in left_seq:
        while right_index < len(right_seq) and right_seq[right_index] != item:
            right_index += 1
        if right_index >= len(right_seq):
            break
        matches += 1
        right_index += 1
    return round(float(matches) / float(max(len(left_seq), len(right_seq), 1)), 4)


def _contains_keyword(name: str, keywords: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in keywords)


def _looks_permission_like(name: str) -> bool:
    return _contains_keyword(name, PERMISSION_KEYWORDS)


def _looks_config_like(name: str) -> bool:
    return _contains_keyword(name, CONFIG_KEYWORDS)


def _looks_asset_like(name: str) -> bool:
    return _contains_keyword(name, ASSET_KEYWORDS)


def _looks_value_entry(name: str) -> bool:
    return _contains_keyword(name, VALUE_ENTRY_KEYWORDS)


def _truncate_samples(items: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    if max_count <= 0:
        return []
    return items[:max_count]


def _sample_sort_key(item: Dict[str, Any]) -> Tuple[int, int, float, str]:
    return (
        -_to_int(item.get("matched_prefix_len")),
        -int(bool(item.get("sink_function_reached"))),
        -_to_float(item.get("coverage_delta")),
        _text(item.get("sequence_id")),
    )


class TFICandidateScopeBuilder:
    def __init__(
        self,
        output_dir: str,
        path_progress_path: Optional[str] = None,
        ksi_evidence_path: Optional[str] = None,
        blocker_evidence_path: Optional[str] = None,
        tfi_focus_path: Optional[str] = None,
        path_plan_path: Optional[str] = None,
        selected_guided_sequences_path: Optional[str] = None,
        repair_pilot_failure_analysis_path: Optional[str] = None,
        max_samples_per_scope: int = 10,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.path_progress_path = path_progress_path or os.path.join(self.output_dir, "path_progress.jsonl")
        self.ksi_evidence_path = ksi_evidence_path or os.path.join(self.output_dir, "ksi_evidence.jsonl")
        self.blocker_evidence_path = blocker_evidence_path or os.path.join(self.output_dir, "blocker_evidence.json")
        self.tfi_focus_path = tfi_focus_path or os.path.join(self.output_dir, "tfi_focus_targets.jsonl")
        self.path_plan_path = path_plan_path or self._default_path_plan_path()
        self.selected_guided_sequences_path = (
            selected_guided_sequences_path or os.path.join(self.output_dir, "selected_guided_sequences.json")
        )
        self.repair_pilot_failure_analysis_path = (
            repair_pilot_failure_analysis_path or os.path.join(self.output_dir, "repair_pilot_failure_analysis.json")
        )
        self.max_samples_per_scope = max(1, int(max_samples_per_scope or 10))

    def _default_path_plan_path(self) -> Optional[str]:
        candidate_paths = [
            os.path.join(self.output_dir, "agent_path_plan.json"),
            os.path.join(self.output_dir, "proposed_path_plan.json"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return path
        return candidate_paths[0]

    def build_scopes(self) -> Dict[str, Any]:
        loaded_inputs = {
            "path_progress": bool(self.path_progress_path and os.path.exists(self.path_progress_path)),
            "ksi_evidence": bool(self.ksi_evidence_path and os.path.exists(self.ksi_evidence_path)),
            "blocker_evidence": bool(self.blocker_evidence_path and os.path.exists(self.blocker_evidence_path)),
            "tfi_focus": bool(self.tfi_focus_path and os.path.exists(self.tfi_focus_path)),
            "path_plan": bool(self.path_plan_path and os.path.exists(self.path_plan_path)),
            "selected_guided_sequences": bool(
                self.selected_guided_sequences_path and os.path.exists(self.selected_guided_sequences_path)
            ),
            "repair_pilot_failure_analysis": bool(
                self.repair_pilot_failure_analysis_path and os.path.exists(self.repair_pilot_failure_analysis_path)
            ),
        }

        path_progress_rows = _read_jsonl(self.path_progress_path)
        ksi_rows = _read_jsonl(self.ksi_evidence_path)
        blocker_payload = _read_json(self.blocker_evidence_path)
        blocker_hypotheses = _listify(blocker_payload.get("hypotheses"))
        tfi_focus_rows = _read_jsonl(self.tfi_focus_path)
        path_plan_payload = _read_json(self.path_plan_path)
        path_plan_hypotheses = _listify(path_plan_payload.get("hypotheses"))
        selected_payload = _read_json(self.selected_guided_sequences_path)
        selected_repairs = _listify(selected_payload.get("selected"))
        failure_payload = _read_json(self.repair_pilot_failure_analysis_path)
        repair_failures = _listify(failure_payload.get("analysis"))

        plan_by_id = self._index_path_plan(path_plan_hypotheses)
        origin_to_plan_ids = self._build_origin_index(plan_by_id)
        blocker_by_id = self._index_blockers(blocker_hypotheses)
        focus_by_key = self._index_tfi_focus(tfi_focus_rows)
        samples = self._build_samples(path_progress_rows, ksi_rows)
        samples_by_hypothesis = defaultdict(list)
        non_hypothesis_samples: List[Dict[str, Any]] = []
        for sample in samples:
            hypothesis_id = _text(sample.get("hypothesis_id")).strip()
            if hypothesis_id:
                samples_by_hypothesis[hypothesis_id].append(sample)
            else:
                non_hypothesis_samples.append(sample)

        repair_by_candidate = self._index_repair_failures(repair_failures)
        selected_by_candidate = self._index_selected_repairs(selected_repairs)

        seeds = self._build_scope_seeds(
            plan_by_id=plan_by_id,
            blocker_by_id=blocker_by_id,
            focus_by_key=focus_by_key,
            samples_by_hypothesis=samples_by_hypothesis,
            selected_repairs=selected_repairs,
            origin_to_plan_ids=origin_to_plan_ids,
        )

        scopes: List[Dict[str, Any]] = []
        for index, seed in enumerate(seeds, start=1):
            scope = self._materialize_scope(
                index=index,
                seed=seed,
                plan_by_id=plan_by_id,
                origin_to_plan_ids=origin_to_plan_ids,
                blocker_by_id=blocker_by_id,
                samples_by_hypothesis=samples_by_hypothesis,
                non_hypothesis_samples=non_hypothesis_samples,
                selected_repairs=selected_repairs,
                selected_by_candidate=selected_by_candidate,
                repair_by_candidate=repair_by_candidate,
            )
            scopes.append(scope)

        return {
            "loaded_inputs": loaded_inputs,
            "scope_count": len(scopes),
            "design_role": "tfi_stage3_target_local_focusing_with_repair_local_context",
            "scopes": scopes,
        }

    def write_scopes(self, scopes: Dict[str, Any], filename: str = "target_focus_candidate_scope.json") -> str:
        path = os.path.join(self.output_dir, filename)
        _write_json(path, scopes)
        return path

    def write_summary(self, scopes: Dict[str, Any], filename: str = "target_focus_candidate_scope_summary.json") -> str:
        payload = self._build_summary(scopes)
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        return path

    def write_binding_repair_requests(
        self,
        scopes: Dict[str, Any],
        filename: str = "tfi_binding_repair_requests.json",
    ) -> str:
        payload = self._build_request_exports(scopes).get("binding_payload", {})
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        return path

    def write_scoped_selection_requests(
        self,
        scopes: Dict[str, Any],
        filename: str = "tfi_scoped_selection_requests.json",
    ) -> str:
        payload = self._build_request_exports(scopes).get("selection_payload", {})
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        return path

    def _index_path_plan(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            hypothesis_id = _text(item.get("id") or item.get("hypothesis_id")).strip()
            if not hypothesis_id:
                continue
            indexed[hypothesis_id] = {
                "hypothesis_id": hypothesis_id,
                "origin_hypothesis_id": _text(item.get("origin_hypothesis_id")).strip() or None,
                "path": _clean_list(item.get("path")),
                "sink": _text(item.get("sink")).strip() or None,
                "focus_type": _text(item.get("focus_type")).strip() or None,
                "reason": _text(item.get("reason")).strip() or None,
                "priority": item.get("priority"),
                "raw": dict(item),
            }
        return indexed

    def _build_origin_index(self, plan_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        output: Dict[str, List[str]] = defaultdict(list)
        for hypothesis_id, item in plan_by_id.items():
            origin_id = _text(item.get("origin_hypothesis_id")).strip()
            if origin_id:
                output[origin_id].append(hypothesis_id)
        return output

    def _index_blockers(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            hypothesis_id = _text(item.get("hypothesis_id") or item.get("id")).strip()
            if not hypothesis_id:
                continue
            indexed[hypothesis_id] = dict(item)
        return indexed

    def _index_tfi_focus(self, rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        output: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            hypothesis_id = _text(row.get("hypothesis_id")).strip()
            target_function = _text(row.get("focus_function") or row.get("sink")).strip()
            key = (hypothesis_id, target_function)
            output[key] = dict(row)
        return output

    def _index_selected_repairs(self, selected_repairs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in selected_repairs:
            if not isinstance(item, dict):
                continue
            candidate_id = _text(item.get("candidate_id")).strip()
            if candidate_id:
                indexed[candidate_id] = dict(item)
        return indexed

    def _index_repair_failures(self, failures: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in failures:
            if not isinstance(item, dict):
                continue
            candidate_id = _text(item.get("candidate_id")).strip()
            if candidate_id:
                indexed[candidate_id] = dict(item)
        return indexed

    def _build_samples(self, path_rows: List[Dict[str, Any]], ksi_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ksi_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        ksi_by_sequence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in ksi_rows:
            if not isinstance(row, dict):
                continue
            sequence_id = _text(row.get("sequence_id")).strip()
            hypothesis_id = _text(row.get("hypothesis_id")).strip()
            if sequence_id:
                ksi_by_sequence[sequence_id].append(row)
                ksi_by_pair[(sequence_id, hypothesis_id)].append(row)

        samples: List[Dict[str, Any]] = []
        seen = set()

        for row in path_rows:
            if not isinstance(row, dict):
                continue
            sequence_id = _text(row.get("sequence_id")).strip()
            hypothesis_id = _text(row.get("matched_hypothesis") or row.get("hypothesis_id")).strip()
            match_rows = ksi_by_pair.get((sequence_id, hypothesis_id)) or ksi_by_sequence.get(sequence_id) or []
            observations = {}
            if match_rows:
                observations = match_rows[0].get("observations") if isinstance(match_rows[0].get("observations"), dict) else {}
            sample = self._build_sample_from_rows(
                sequence_id=sequence_id,
                hypothesis_id=hypothesis_id or None,
                path_row=row,
                observations=observations,
            )
            marker = (
                sample.get("sequence_id"),
                sample.get("hypothesis_id"),
                sample.get("status"),
                tuple(sample.get("functions") or []),
            )
            if marker in seen:
                continue
            seen.add(marker)
            samples.append(sample)

        for row in ksi_rows:
            if not isinstance(row, dict):
                continue
            sequence_id = _text(row.get("sequence_id")).strip()
            hypothesis_id = _text(row.get("hypothesis_id")).strip() or None
            observations = row.get("observations") if isinstance(row.get("observations"), dict) else {}
            path_stub = {
                "sequence_id": sequence_id,
                "matched_hypothesis": hypothesis_id,
                "matched_prefix_len": row.get("matched_prefix_len"),
                "blocked_at": row.get("blocked_at"),
                "frontier": row.get("frontier"),
                "sink_function_reached": row.get("sink_function_reached"),
                "status": row.get("status"),
            }
            sample = self._build_sample_from_rows(
                sequence_id=sequence_id,
                hypothesis_id=hypothesis_id,
                path_row=path_stub,
                observations=observations,
            )
            marker = (
                sample.get("sequence_id"),
                sample.get("hypothesis_id"),
                sample.get("status"),
                tuple(sample.get("functions") or []),
            )
            if marker in seen:
                continue
            seen.add(marker)
            samples.append(sample)

        samples.sort(key=_sample_sort_key)
        return samples

    def _build_sample_from_rows(
        self,
        sequence_id: str,
        hypothesis_id: Optional[str],
        path_row: Dict[str, Any],
        observations: Dict[str, Any],
    ) -> Dict[str, Any]:
        functions = _clean_list(observations.get("functions"))
        sender_sequence = _clean_list(observations.get("sender_sequence"))
        value_sequence = _listify(observations.get("value_sequence"))
        args_sequence = _listify(observations.get("args_sequence"))
        observed_prefix = _clean_list(path_row.get("best_prefix"))
        observed_prefix_source = "path_progress" if observed_prefix else "unknown"
        ksi_prefix = _clean_list(observations.get("prefix_reached"))
        if not observed_prefix and ksi_prefix:
            observed_prefix = ksi_prefix
            observed_prefix_source = "ksi_evidence"
        status = _text(path_row.get("status")).strip() or "unknown"
        sink_reverted = bool(status == "sink_reverted" or path_row.get("sink_reverted"))
        sink_function_reached = bool(
            path_row.get("sink_function_reached")
            or observations.get("sink_function_reached")
            or path_row.get("sink_reached")
        )
        return {
            "sequence_id": sequence_id or None,
            "hypothesis_id": hypothesis_id,
            "functions": functions,
            "sender_sequence": sender_sequence,
            "value_sequence": value_sequence,
            "args_sequence": args_sequence,
            "revert": bool(observations.get("revert")),
            "revert_reason": observations.get("revert_reason"),
            "matched_prefix_len": _to_int(path_row.get("matched_prefix_len")),
            "blocked_at": _text(path_row.get("blocked_at")).strip() or None,
            "frontier": _text(path_row.get("frontier")).strip() or None,
            "sink_reverted": sink_reverted,
            "sink_function_reached": sink_function_reached,
            "coverage_delta": _to_float(observations.get("coverage_delta")),
            "status": status,
            "prefix_reached": ksi_prefix,
            "observed_prefix": observed_prefix,
            "observed_prefix_source": observed_prefix_source,
        }

    def _build_scope_seeds(
        self,
        plan_by_id: Dict[str, Dict[str, Any]],
        blocker_by_id: Dict[str, Dict[str, Any]],
        focus_by_key: Dict[Tuple[str, str], Dict[str, Any]],
        samples_by_hypothesis: Dict[str, List[Dict[str, Any]]],
        selected_repairs: List[Dict[str, Any]],
        origin_to_plan_ids: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def upsert(seed: Dict[str, Any]) -> None:
            hypothesis_id = _text(seed.get("hypothesis_id")).strip()
            target_function = _text(seed.get("target_function")).strip()
            if not target_function:
                return
            key = (hypothesis_id, target_function)
            if key not in merged:
                merged[key] = dict(seed)
                return
            current = merged[key]
            current_focus = _text(current.get("focus_type")).strip() or "unknown"
            next_focus = _text(seed.get("focus_type")).strip() or "unknown"
            if FOCUS_PRIORITY.get(next_focus, 99) < FOCUS_PRIORITY.get(current_focus, 99):
                current["focus_type"] = next_focus
            current_secondary = set(_clean_list(current.get("secondary_focus_types")))
            current_secondary.update(_clean_list(seed.get("secondary_focus_types")))
            if current_focus and current_focus != current.get("focus_type"):
                current_secondary.add(current_focus)
            if next_focus and next_focus != current.get("focus_type"):
                current_secondary.add(next_focus)
            current["secondary_focus_types"] = sorted(current_secondary)
            for field in ("path", "sink", "frontier", "origin_hypothesis_id", "matched_prefix", "best_prefix_len"):
                if current.get(field) in (None, "", [], 0) and seed.get(field) not in (None, "", [], 0):
                    current[field] = seed.get(field)

        for hypothesis_id, plan_item in plan_by_id.items():
            blocker = blocker_by_id.get(hypothesis_id, {})
            progress = blocker.get("progress_summary") if isinstance(blocker.get("progress_summary"), dict) else {}
            path = _clean_list(plan_item.get("path")) or _clean_list(blocker.get("path"))
            sink = _text(plan_item.get("sink") or blocker.get("sink")).strip() or None
            frontier = _text(progress.get("frontier")).strip() or (path[-1] if path else sink)
            best_prefix = _clean_list(progress.get("best_prefix"))
            best_prefix_len = _to_int(progress.get("best_prefix_len"))
            if not best_prefix and best_prefix_len > 0 and path:
                best_prefix = path[:best_prefix_len]
            focus_type = self._derive_focus_type(
                blocker=blocker,
                fallback_focus=_text(plan_item.get("focus_type")).strip(),
                target_function=sink or frontier,
            )
            upsert(
                {
                    "hypothesis_id": hypothesis_id,
                    "origin_hypothesis_id": plan_item.get("origin_hypothesis_id"),
                    "focus_type": focus_type,
                    "target_function": sink or frontier,
                    "path": path,
                    "sink": sink,
                    "frontier": frontier,
                    "matched_prefix": best_prefix,
                    "best_prefix_len": best_prefix_len,
                    "scope_type": "hypothesis_target",
                    "secondary_focus_types": [],
                }
            )

        if not plan_by_id:
            for hypothesis_id, blocker in blocker_by_id.items():
                progress = blocker.get("progress_summary") if isinstance(blocker.get("progress_summary"), dict) else {}
                path = _clean_list(blocker.get("path"))
                sink = _text(blocker.get("sink")).strip() or None
                frontier = _text(progress.get("frontier")).strip() or (path[-1] if path else sink)
                best_prefix = _clean_list(progress.get("best_prefix"))
                best_prefix_len = _to_int(progress.get("best_prefix_len"))
                focus_type = self._derive_focus_type(blocker=blocker, fallback_focus="", target_function=sink or frontier)
                upsert(
                    {
                        "hypothesis_id": hypothesis_id,
                        "origin_hypothesis_id": _text(blocker.get("origin_hypothesis_id")).strip() or None,
                        "focus_type": focus_type,
                        "target_function": sink or frontier,
                        "path": path,
                        "sink": sink,
                        "frontier": frontier,
                        "matched_prefix": best_prefix,
                        "best_prefix_len": best_prefix_len,
                        "scope_type": "hypothesis_target",
                        "secondary_focus_types": [],
                    }
                )

        for focus in focus_by_key.values():
            if not isinstance(focus, dict):
                continue
            hypothesis_id = _text(focus.get("hypothesis_id")).strip()
            target_function = _text(focus.get("focus_function") or focus.get("sink")).strip()
            if not target_function:
                continue
            linked_plan_ids = origin_to_plan_ids.get(hypothesis_id, [])
            if linked_plan_ids:
                for plan_id in linked_plan_ids:
                    upsert(
                        {
                            "hypothesis_id": plan_id,
                            "origin_hypothesis_id": hypothesis_id,
                            "focus_type": _text(focus.get("focus_type")).strip() or "unknown",
                            "target_function": target_function,
                            "path": _clean_list(focus.get("path")),
                            "sink": _text(focus.get("sink")).strip() or None,
                            "frontier": _text(focus.get("focus_frontier")).strip() or None,
                            "matched_prefix": [],
                            "best_prefix_len": 0,
                            "scope_type": "hypothesis_target",
                            "secondary_focus_types": [],
                        }
                    )
            else:
                upsert(
                    {
                        "hypothesis_id": hypothesis_id,
                        "origin_hypothesis_id": None,
                        "focus_type": _text(focus.get("focus_type")).strip() or "unknown",
                        "target_function": target_function,
                        "path": _clean_list(focus.get("path")),
                        "sink": _text(focus.get("sink")).strip() or None,
                        "frontier": _text(focus.get("focus_frontier")).strip() or None,
                        "matched_prefix": [],
                        "best_prefix_len": 0,
                        "scope_type": "hypothesis_target",
                        "secondary_focus_types": [],
                    }
                )

        for hypothesis_id, sample_items in samples_by_hypothesis.items():
            if any(key[0] == hypothesis_id for key in merged):
                continue
            sample_items = sorted(sample_items, key=_sample_sort_key)
            best = sample_items[0] if sample_items else {}
            functions = _clean_list(best.get("functions"))
            if not functions:
                continue
            matched_prefix_len = _to_int(best.get("matched_prefix_len"))
            upsert(
                {
                    "hypothesis_id": hypothesis_id,
                    "origin_hypothesis_id": None,
                    "focus_type": "target_only" if matched_prefix_len <= 0 else "repeated_frontier_block",
                    "target_function": _text(best.get("frontier") or functions[-1]).strip(),
                    "path": functions,
                    "sink": functions[-1] if functions else None,
                    "frontier": _text(best.get("frontier")).strip() or (functions[-1] if functions else None),
                    "matched_prefix": functions[:matched_prefix_len],
                    "best_prefix_len": matched_prefix_len,
                    "scope_type": "hypothesis_target",
                    "secondary_focus_types": [],
                }
            )

        covered_hypothesis_ids = {key[0] for key in merged}
        covered_targets = {key[1] for key in merged}
        for repair in selected_repairs:
            if not isinstance(repair, dict):
                continue
            hypothesis_id = _text(repair.get("hypothesis_id")).strip()
            target_function = _text(repair.get("target_function")).strip()
            sequence = _clean_list(repair.get("guided_sequence"))
            if not target_function and sequence:
                target_function = sequence[-1]
            if not target_function:
                continue
            if hypothesis_id and hypothesis_id in covered_hypothesis_ids:
                continue
            if hypothesis_id or target_function not in covered_targets:
                upsert(
                    {
                        "hypothesis_id": hypothesis_id,
                        "origin_hypothesis_id": _text(repair.get("origin_hypothesis_id")).strip() or None,
                        "focus_type": "target_only",
                        "target_function": target_function,
                        "path": sequence,
                        "sink": target_function,
                        "frontier": target_function,
                        "matched_prefix": [],
                        "best_prefix_len": 0,
                        "scope_type": "target_only" if not hypothesis_id else "hypothesis_target",
                        "secondary_focus_types": [],
                    }
                )

        seeds = list(merged.values())
        seeds.sort(
            key=lambda item: (
                FOCUS_PRIORITY.get(_text(item.get("focus_type")).strip() or "unknown", 99),
                _text(item.get("hypothesis_id")),
                _text(item.get("target_function")),
            )
        )
        return seeds

    def _derive_focus_type(self, blocker: Dict[str, Any], fallback_focus: str, target_function: Optional[str]) -> str:
        progress = blocker.get("progress_summary") if isinstance(blocker.get("progress_summary"), dict) else {}
        sink_reverted_count = _to_int(progress.get("sink_reverted_count"))
        blocked_count = _to_int(progress.get("blocked_count"))
        best_prefix_len = _to_int(progress.get("best_prefix_len"))
        runtime = blocker.get("runtime_evidence") if isinstance(blocker.get("runtime_evidence"), dict) else {}
        sink_without_prefix = _to_int(runtime.get("sink_observed_without_valid_prefix_count"))
        if sink_reverted_count > 0:
            return "sink_reverted"
        if blocked_count > 0 and _text(progress.get("dominant_blocked_at")).strip():
            return "repeated_frontier_block"
        if sink_without_prefix > 0 and target_function:
            return "target_seen_but_not_path_progress"
        if fallback_focus:
            return fallback_focus
        if best_prefix_len > 0:
            return "prefix_ready_frontier_unknown"
        return "unknown"

    def _materialize_scope(
        self,
        index: int,
        seed: Dict[str, Any],
        plan_by_id: Dict[str, Dict[str, Any]],
        origin_to_plan_ids: Dict[str, List[str]],
        blocker_by_id: Dict[str, Dict[str, Any]],
        samples_by_hypothesis: Dict[str, List[Dict[str, Any]]],
        non_hypothesis_samples: List[Dict[str, Any]],
        selected_repairs: List[Dict[str, Any]],
        selected_by_candidate: Dict[str, Dict[str, Any]],
        repair_by_candidate: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        hypothesis_id = _text(seed.get("hypothesis_id")).strip() or None
        origin_hypothesis_id = _text(seed.get("origin_hypothesis_id")).strip() or None
        target_function = _text(seed.get("target_function")).strip()
        blocker = blocker_by_id.get(hypothesis_id or "", {})
        progress = blocker.get("progress_summary") if isinstance(blocker.get("progress_summary"), dict) else {}

        plan_item = plan_by_id.get(hypothesis_id or "", {})
        scope_path = _clean_list(seed.get("path")) or _clean_list(plan_item.get("path")) or _clean_list(blocker.get("path"))
        planned_path = list(scope_path)
        sink = _text(seed.get("sink") or plan_item.get("sink") or blocker.get("sink")).strip() or (scope_path[-1] if scope_path else None)
        frontier = _text(seed.get("frontier") or progress.get("frontier")).strip() or sink or target_function

        lineage_hypotheses = [item for item in [hypothesis_id] if item]
        if origin_hypothesis_id and origin_hypothesis_id in origin_to_plan_ids:
            lineage_hypotheses.extend(origin_to_plan_ids[origin_hypothesis_id])
        lineage_hypotheses = _unique_list(lineage_hypotheses)

        hypothesis_samples: List[Dict[str, Any]] = []
        for lineage_id in lineage_hypotheses:
            hypothesis_samples.extend(samples_by_hypothesis.get(lineage_id, []))
        hypothesis_samples = _unique_list(hypothesis_samples)
        hypothesis_samples.sort(key=_sample_sort_key)
        observed_matched_prefix, matched_prefix_source = self._derive_observed_matched_prefix(
            hypothesis_samples=hypothesis_samples,
            target_function=target_function,
            frontier=frontier,
            sink=sink,
        )
        matched_prefix = list(observed_matched_prefix)
        matched_prefix_is_observed = bool(observed_matched_prefix)
        best_prefix_len = len(observed_matched_prefix)
        preserve_prefix_required = bool(
            matched_prefix_is_observed and best_prefix_len > 0
        ) or _text(blocker.get("repair_objective", {}).get("action")).startswith("preserve_prefix")

        execution_samples = self._build_execution_samples(
            scope_path=scope_path,
            matched_prefix=matched_prefix,
            target_function=target_function,
            frontier=frontier,
            sink=sink,
            hypothesis_samples=hypothesis_samples,
            non_hypothesis_samples=non_hypothesis_samples,
        )

        repair_alignment = self._build_repair_alignment_scope(
            hypothesis_id=hypothesis_id,
            origin_hypothesis_id=origin_hypothesis_id,
            target_function=target_function,
            frontier=frontier,
            sink=sink,
            scope_path=scope_path,
            matched_prefix=matched_prefix,
            selected_repairs=selected_repairs,
            repair_by_candidate=repair_by_candidate,
        )

        repair_context = self._build_repair_context(
            repair_alignment=repair_alignment,
            repair_by_candidate=repair_by_candidate,
            selected_by_candidate=selected_by_candidate,
        )

        sender_scope = self._build_sender_scope(execution_samples, blocker)
        value_scope = self._build_value_scope(execution_samples, target_function)
        argument_scope = self._build_argument_scope(
            execution_samples=execution_samples,
            repair_alignment=repair_alignment,
            target_function=target_function,
            scope_path=scope_path,
            frontier=frontier,
            sink=sink,
        )
        setup_function_scope = self._build_setup_function_scope(
            scope_path=scope_path,
            execution_samples=execution_samples,
            repair_alignment=repair_alignment,
        )
        negative_scope = self._build_negative_scope(execution_samples, repair_alignment, repair_by_candidate)
        repair_negative_scope = self._build_repair_negative_scope(repair_alignment, repair_by_candidate)
        repair_positive_scope = self._build_repair_positive_scope(
            execution_samples=execution_samples,
            repair_alignment=repair_alignment,
            repair_by_candidate=repair_by_candidate,
            setup_function_scope=setup_function_scope,
            scope_path=scope_path,
            target_function=target_function,
        )
        repair_candidate_scope = self._build_repair_candidate_scope(repair_alignment)
        scope_local_evidence_check = self._build_scope_local_evidence_check(
            planned_path=planned_path,
            observed_matched_prefix=observed_matched_prefix,
            matched_prefix_source=matched_prefix_source,
            target_function=target_function,
            frontier=frontier,
            sink=sink,
            repair_context=repair_context,
            repair_alignment=repair_alignment,
        )
        tfi_failure_diagnosis = self._build_tfi_failure_diagnosis(
            target_function=target_function,
            frontier=frontier,
            sink=sink,
            matched_prefix=matched_prefix,
            matched_prefix_is_observed=matched_prefix_is_observed,
            path_context_seed={
                "best_prefix_len": best_prefix_len,
                "preserve_prefix_required": preserve_prefix_required,
            },
            repair_context=repair_context,
            repair_alignment=repair_alignment,
            repair_negative_scope=repair_negative_scope,
            repair_positive_scope=repair_positive_scope,
            scope_local_evidence_check=scope_local_evidence_check,
        )
        binding_repair_context = self._build_binding_repair_context(
            diagnosis=tfi_failure_diagnosis,
            target_function=target_function,
            sink=sink,
            repair_context=repair_context,
            repair_alignment=repair_alignment,
            repair_negative_scope=repair_negative_scope,
            sender_scope=sender_scope,
            value_scope=value_scope,
            argument_scope=argument_scope,
            scope_local_evidence_check=scope_local_evidence_check,
        )
        generic_skeleton_leakage_check = self._build_generic_skeleton_leakage_check(
            scope_local_evidence_check=scope_local_evidence_check,
            binding_repair_context=binding_repair_context,
            target_function=target_function,
            frontier=frontier,
            sink=sink,
            observed_matched_prefix=observed_matched_prefix,
        )
        repair_positive_scope = self._apply_generic_skeleton_leakage_guard(
            repair_positive_scope=repair_positive_scope,
            generic_skeleton_leakage_check=generic_skeleton_leakage_check,
        )
        scope_restriction_context = self._build_scope_restriction_context(
            diagnosis=tfi_failure_diagnosis,
            matched_prefix=matched_prefix,
            frontier=frontier,
            sink=sink,
            target_function=target_function,
            repair_alignment=repair_alignment,
        )

        path_context = {
            "path": planned_path,
            "planned_path": planned_path,
            "matched_prefix": observed_matched_prefix,
            "observed_matched_prefix": observed_matched_prefix,
            "best_prefix_len": best_prefix_len,
            "frontier": frontier,
            "sink": sink,
            "target_function": target_function,
            "preserve_prefix_required": preserve_prefix_required,
            "ordered_overlap_score": self._path_overlap_score(scope_path, repair_alignment),
            "matched_prefix_source": matched_prefix_source,
            "matched_prefix_is_observed": matched_prefix_is_observed,
        }
        context_completeness = self._build_context_completeness(
            path_context=path_context,
            execution_samples=execution_samples,
            sender_scope=sender_scope,
            value_scope=value_scope,
            argument_scope=argument_scope,
            repair_candidate_scope=repair_candidate_scope,
        )

        return {
            "scope_id": "TFI_SCOPE_%d" % index,
            "scope_type": seed.get("scope_type") or "hypothesis_target",
            "hypothesis_id": hypothesis_id,
            "origin_hypothesis_id": origin_hypothesis_id,
            "focus_type": _text(seed.get("focus_type")).strip() or "unknown",
            "secondary_focus_types": _clean_list(seed.get("secondary_focus_types")),
            "target_function": target_function,
            "frontier": frontier,
            "sink": sink,
            "path": scope_path,
            "matched_prefix": matched_prefix,
            "best_prefix_len": best_prefix_len,
            "path_context": path_context,
            "execution_samples": execution_samples,
            "sender_scope": sender_scope,
            "value_scope": value_scope,
            "argument_scope": argument_scope,
            "setup_function_scope": setup_function_scope,
            "repair_context": repair_context,
            "repair_alignment_scope": repair_alignment,
            "repair_candidate_scope": repair_candidate_scope,
            "negative_scope": negative_scope,
            "repair_negative_scope": repair_negative_scope,
            "repair_positive_scope": repair_positive_scope,
            "scope_local_evidence_check": scope_local_evidence_check,
            "generic_skeleton_leakage_check": generic_skeleton_leakage_check,
            "tfi_failure_diagnosis": tfi_failure_diagnosis,
            "binding_repair_context": binding_repair_context,
            "scope_restriction_context": scope_restriction_context,
            "context_completeness": context_completeness,
            "source": "target_focus_candidate_scope_v1",
        }

    def _path_overlap_score(self, scope_path: List[str], repair_alignment: Dict[str, Any]) -> float:
        scores = []
        for item in repair_alignment.get("aligned_guided_sequences") or []:
            scores.append(_to_float(item.get("ordered_overlap_score")))
        for item in repair_alignment.get("partially_aligned_guided_sequences") or []:
            scores.append(_to_float(item.get("ordered_overlap_score")))
        if scores:
            return round(max(scores), 4)
        return 0.0

    def _build_execution_samples(
        self,
        scope_path: List[str],
        matched_prefix: List[str],
        target_function: str,
        frontier: Optional[str],
        sink: Optional[str],
        hypothesis_samples: List[Dict[str, Any]],
        non_hypothesis_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prefix_success = []
        sink_failure = []
        frontier_block = []
        nearby = []

        for sample in hypothesis_samples:
            candidate = self._format_sample(sample)
            if _to_int(sample.get("matched_prefix_len")) > 0 and not bool(sample.get("sink_reverted")):
                prefix_success.append(candidate)
            if bool(sample.get("sink_function_reached")) and bool(sample.get("sink_reverted")):
                sink_failure.append(candidate)
            if _text(sample.get("blocked_at")).strip() and (
                _text(sample.get("blocked_at")).strip() == _text(frontier).strip()
                or _text(sample.get("frontier")).strip() == _text(frontier).strip()
                or _text(sample.get("blocked_at")).strip() == _text(sink).strip()
            ):
                frontier_block.append(candidate)

        nearby_targets = {item for item in [target_function, frontier, sink] if _text(item).strip()}
        for sample in non_hypothesis_samples:
            functions = _clean_list(sample.get("functions"))
            if not functions:
                continue
            if nearby_targets.intersection(functions):
                nearby.append(self._format_sample(sample))
                continue
            if scope_path and _ordered_overlap_score(scope_path, functions) > 0:
                nearby.append(self._format_sample(sample))

        prefix_success.sort(key=_sample_sort_key)
        sink_failure.sort(key=_sample_sort_key)
        frontier_block.sort(key=_sample_sort_key)
        nearby.sort(key=_sample_sort_key)

        return {
            "prefix_success_samples": _truncate_samples(prefix_success, self.max_samples_per_scope),
            "sink_failure_samples": _truncate_samples(sink_failure, self.max_samples_per_scope),
            "frontier_block_samples": _truncate_samples(frontier_block, self.max_samples_per_scope),
            "non_hypothesis_nearby_samples": _truncate_samples(_unique_list(nearby), self.max_samples_per_scope),
        }

    def _format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sequence_id": sample.get("sequence_id"),
            "hypothesis_id": sample.get("hypothesis_id"),
            "functions": _clean_list(sample.get("functions")),
            "sender_sequence": _clean_list(sample.get("sender_sequence")),
            "value_sequence": _listify(sample.get("value_sequence")),
            "args_sequence": _listify(sample.get("args_sequence")),
            "revert": bool(sample.get("revert")),
            "revert_reason": sample.get("revert_reason"),
            "matched_prefix_len": _to_int(sample.get("matched_prefix_len")),
            "blocked_at": sample.get("blocked_at"),
            "sink_reverted": bool(sample.get("sink_reverted")),
            "sink_function_reached": bool(sample.get("sink_function_reached")),
            "coverage_delta": _to_float(sample.get("coverage_delta")),
            "frontier": sample.get("frontier"),
            "status": sample.get("status"),
        }

    def _build_sender_scope(self, execution_samples: Dict[str, Any], blocker: Dict[str, Any]) -> Dict[str, Any]:
        prefix_success = execution_samples.get("prefix_success_samples") or []
        sink_failure = execution_samples.get("sink_failure_samples") or []

        prefix_senders = _unique_list([item.get("sender_sequence") for item in prefix_success if item.get("sender_sequence")])
        sink_senders = _unique_list([item.get("sender_sequence") for item in sink_failure if item.get("sender_sequence")])
        stable_same_sender = []
        for sender_sequence in prefix_senders:
            if len(set(_clean_list(sender_sequence))) <= 1:
                stable_same_sender.append(_clean_list(sender_sequence)[0] if _clean_list(sender_sequence) else None)
        stable_same_sender = [item for item in stable_same_sender if item]

        sender_change_failed = []
        for item in sink_failure:
            sequence = _clean_list(item.get("sender_sequence"))
            if len(set(sequence)) > 1:
                sender_change_failed.append(
                    {
                        "sequence_id": item.get("sequence_id"),
                        "sender_sequence": sequence,
                        "revert_reason": item.get("revert_reason"),
                    }
                )

        permission_signals = blocker.get("runtime_evidence", {}).get("top_suspected_blockers") if isinstance(blocker.get("runtime_evidence"), dict) else []
        owner_like = []
        recommended = "unknown"
        if stable_same_sender:
            recommended = "keep_same_sender_across_sequence"
        elif prefix_senders:
            recommended = "try_prefix_success_sender"
        elif sender_change_failed:
            recommended = "avoid_sink_revert_sender"
        elif any(_text(item).strip() == "sender_change" for item in _listify(permission_signals)):
            recommended = "try_owner_or_deployer_if_available"

        return {
            "prefix_success_senders": prefix_senders,
            "sink_revert_senders": sink_senders,
            "stable_same_sender_candidates": _unique_list(stable_same_sender),
            "sender_change_failed_examples": sender_change_failed[: self.max_samples_per_scope],
            "owner_or_deployer_like_candidates": owner_like,
            "recommended_sender_policy": recommended,
        }

    def _build_value_scope(self, execution_samples: Dict[str, Any], target_function: str) -> Dict[str, Any]:
        prefix_success = execution_samples.get("prefix_success_samples") or []
        sink_failure = execution_samples.get("sink_failure_samples") or []

        success_values = _unique_list([item.get("value_sequence") for item in prefix_success if item.get("value_sequence")])
        sink_values = _unique_list([item.get("value_sequence") for item in sink_failure if item.get("value_sequence")])
        flat_success = [value for seq in success_values for value in _listify(seq)]
        flat_sink = [value for seq in sink_values for value in _listify(seq)]
        zero_values = [value for value in flat_success + flat_sink if value in (0, 0.0)]
        nonzero_values = [value for value in flat_success + flat_sink if value not in (0, 0.0)]

        recommended = "unknown"
        if flat_success and all(value in (0, 0.0) for value in flat_success):
            recommended = "force_zero_value"
        elif flat_success:
            recommended = "keep_observed_success_value"
        elif nonzero_values and _text(target_function).lower() in VALUE_ENTRY_KEYWORDS:
            recommended = "try_small_nonzero_value"
        elif flat_sink:
            recommended = "avoid_reverted_value"

        return {
            "observed_zero_values": _unique_list(zero_values)[: self.max_samples_per_scope],
            "observed_nonzero_values": _unique_list(nonzero_values)[: self.max_samples_per_scope],
            "value_used_in_success_prefix": success_values[: self.max_samples_per_scope],
            "value_used_in_reverted_sink": sink_values[: self.max_samples_per_scope],
            "recommended_value_policy": recommended,
        }

    def _build_argument_scope(
        self,
        execution_samples: Dict[str, Any],
        repair_alignment: Dict[str, Any],
        target_function: str,
        scope_path: List[str],
        frontier: Optional[str],
        sink: Optional[str],
    ) -> Dict[str, Any]:
        prefix_success = execution_samples.get("prefix_success_samples") or []
        sink_failure = execution_samples.get("sink_failure_samples") or []
        prefix_args = _unique_list([item.get("args_sequence") for item in prefix_success if item.get("args_sequence")])
        sink_args = _unique_list([item.get("args_sequence") for item in sink_failure if item.get("args_sequence")])

        failed_counter: Counter = Counter()
        for item in sink_args:
            failed_counter[json.dumps(item, ensure_ascii=True)] += 1
        repeated_failed = []
        for raw, count in failed_counter.most_common(self.max_samples_per_scope):
            repeated_failed.append({"args_sequence": json.loads(raw), "count": int(count)})

        argument_templates: List[str] = []
        asset_like_count = sum(1 for item in scope_path if _looks_asset_like(item))
        if asset_like_count >= 2:
            argument_templates.append("preserve_cross_call_argument_consistency")
        if any(_looks_permission_like(item) for item in scope_path):
            argument_templates.append("address_from_existing_accounts")
        if any("true" in json.dumps(args).lower() or "false" in json.dumps(args).lower() for args in prefix_args + sink_args):
            argument_templates.append("boolean_true_false")
        if asset_like_count > 0:
            argument_templates.append("amount_boundary")
        if sink or frontier or target_function:
            argument_templates.append("mutate_boundary_arguments")
        argument_templates = _unique_list(argument_templates)

        recommended = "unknown"
        if "preserve_cross_call_argument_consistency" in argument_templates:
            recommended = "repair_binding_fields_under_scope_constraints"
        elif sink_failure:
            recommended = "mutate_boundary_arguments"
        elif frontier and prefix_success:
            recommended = "mutate_scope_local_arguments_under_prefix_constraints"

        return {
            "prefix_success_args": prefix_args[: self.max_samples_per_scope],
            "sink_revert_args": sink_args[: self.max_samples_per_scope],
            "repeated_failed_arg_patterns": repeated_failed,
            "argument_templates": argument_templates,
            "recommended_argument_policy": recommended,
        }

    def _build_setup_function_scope(
        self,
        scope_path: List[str],
        execution_samples: Dict[str, Any],
        repair_alignment: Dict[str, Any],
    ) -> Dict[str, Any]:
        observed_functions: List[str] = []
        for bucket in execution_samples.values():
            for item in bucket or []:
                observed_functions.extend(_clean_list(item.get("functions")))
        for repair in repair_alignment.get("aligned_guided_sequences") or []:
            observed_functions.extend(_clean_list(repair.get("guided_sequence")))
        for repair in repair_alignment.get("partially_aligned_guided_sequences") or []:
            observed_functions.extend(_clean_list(repair.get("guided_sequence")))
        observed_functions.extend(scope_path)
        observed_functions = _unique_list([item for item in observed_functions if _text(item).strip()])

        permission_like = [item for item in observed_functions if _looks_permission_like(item)]
        config_like = [item for item in observed_functions if _looks_config_like(item)]
        approve_transfer = [item for item in observed_functions if _looks_asset_like(item)]
        value_entry = [item for item in observed_functions if _looks_value_entry(item)]

        revert_counter: Counter = Counter()
        for item in execution_samples.get("sink_failure_samples") or []:
            for function_name in _clean_list(item.get("functions")):
                revert_counter[function_name] += 1

        candidate_setup = _unique_list(permission_like + config_like + approve_transfer + value_entry)
        avoid_functions = [name for name, count in revert_counter.most_common(self.max_samples_per_scope) if count >= 2]

        return {
            "permission_like_functions": permission_like[: self.max_samples_per_scope],
            "config_like_functions": config_like[: self.max_samples_per_scope],
            "approve_transfer_functions": approve_transfer[: self.max_samples_per_scope],
            "value_entry_functions": value_entry[: self.max_samples_per_scope],
            "candidate_setup_functions": candidate_setup[: self.max_samples_per_scope],
            "functions_to_avoid_if_repeatedly_reverted": avoid_functions,
        }

    def _build_repair_alignment_scope(
        self,
        hypothesis_id: Optional[str],
        origin_hypothesis_id: Optional[str],
        target_function: str,
        frontier: Optional[str],
        sink: Optional[str],
        scope_path: List[str],
        matched_prefix: List[str],
        selected_repairs: List[Dict[str, Any]],
        repair_by_candidate: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        aligned = []
        partial = []
        misaligned = []

        for item in selected_repairs:
            if not isinstance(item, dict):
                continue
            guided_sequence = _clean_list(item.get("guided_sequence"))
            if not guided_sequence:
                continue
            candidate_id = _text(item.get("candidate_id")).strip() or None
            candidate_hypothesis = _text(item.get("hypothesis_id")).strip() or None
            candidate_origin = _text(item.get("origin_hypothesis_id")).strip() or None
            candidate_target = _text(item.get("target_function")).strip() or (guided_sequence[-1] if guided_sequence else None)
            failure = repair_by_candidate.get(candidate_id or "", {})

            reasons = []
            same_hypothesis = bool(hypothesis_id and candidate_hypothesis == hypothesis_id)
            same_origin = bool(origin_hypothesis_id and (candidate_hypothesis == origin_hypothesis_id or candidate_origin == origin_hypothesis_id))
            if same_hypothesis:
                reasons.append("same_hypothesis_id")
            elif same_origin:
                reasons.append("same_origin_hypothesis")

            contains_target = bool(target_function and target_function in guided_sequence)
            if contains_target:
                reasons.append("contains_target_function")

            overlap = _ordered_overlap_score(scope_path, guided_sequence)
            if overlap > 0:
                reasons.append("ordered_overlap_with_path")

            preserves_prefix = bool(matched_prefix and _prefix_len(matched_prefix, guided_sequence) == len(matched_prefix))
            if preserves_prefix:
                reasons.append("preserves_matched_prefix")

            touches_frontier_or_sink = bool(
                (frontier and frontier in guided_sequence) or (sink and sink in guided_sequence) or (candidate_target in [frontier, sink])
            )
            if touches_frontier_or_sink:
                reasons.append("touches_frontier_or_sink")

            shares_pattern = False
            scope_has_asset_pattern = any(_looks_asset_like(name) for name in scope_path)
            scope_has_permission_pattern = any(_looks_permission_like(name) for name in scope_path)
            scope_has_config_pattern = any(_looks_config_like(name) for name in scope_path)
            sequence_has_asset_pattern = any(_looks_asset_like(name) for name in guided_sequence)
            sequence_has_permission_pattern = any(_looks_permission_like(name) for name in guided_sequence)
            sequence_has_config_pattern = any(_looks_config_like(name) for name in guided_sequence)
            if scope_has_asset_pattern and sequence_has_asset_pattern:
                shares_pattern = True
            if scope_has_permission_pattern and sequence_has_permission_pattern:
                shares_pattern = True
            if scope_has_config_pattern and sequence_has_config_pattern:
                shares_pattern = True
            if shares_pattern:
                reasons.append("shares_sender_or_argument_pattern")

            failure_prefix = _to_int(failure.get("matched_prefix_len"))
            failure_target_seen = bool(failure.get("target_function_seen"))
            failure_sink_reached = bool(failure.get("sink_function_reached"))
            if failure_prefix > 0 or failure_target_seen or failure_sink_reached:
                reasons.append("repair_history_supports_alignment")

            aligned_score = 0
            if same_hypothesis:
                aligned_score += 5
            if same_origin:
                aligned_score += 3
            if preserves_prefix:
                aligned_score += 4
            if contains_target:
                aligned_score += 2
            if touches_frontier_or_sink:
                aligned_score += 2
            if overlap >= 0.5:
                aligned_score += 2
            elif overlap > 0:
                aligned_score += 1
            if shares_pattern:
                aligned_score += 1
            if failure_sink_reached:
                aligned_score += 1

            enriched = {
                "candidate_id": candidate_id,
                "hypothesis_id": candidate_hypothesis,
                "origin_hypothesis_id": candidate_origin,
                "target_function": candidate_target,
                "guided_sequence": guided_sequence,
                "selected_rank": _to_int(item.get("selected_rank"), 10 ** 9),
                "dryrun_status": _text(item.get("dryrun_status")).strip() or None,
                "ordered_overlap_score": overlap,
                "alignment_reasons": reasons,
                "matched_hypothesis": _text(failure.get("matched_hypothesis")).strip() or None,
                "target_seen": failure_target_seen,
                "sink_function_reached": failure_sink_reached,
                "sink_reverted": bool(failure.get("sink_reverted")),
                "matched_prefix_len": failure_prefix,
                "prefix_preserved": bool(failure.get("prefix_preserved")),
                "exact_match": bool(failure.get("exact_match")),
            }

            if (same_hypothesis or same_origin or preserves_prefix) and (
                contains_target or touches_frontier_or_sink or overlap > 0
            ):
                aligned.append(enriched)
            elif aligned_score >= 5:
                aligned.append(enriched)
            elif same_origin or contains_target or overlap > 0 or failure_target_seen:
                partial.append(enriched)
            else:
                misaligned.append(enriched)

        aligned.sort(key=lambda item: (item.get("selected_rank"), -_to_float(item.get("ordered_overlap_score"))))
        partial.sort(key=lambda item: (item.get("selected_rank"), -_to_float(item.get("ordered_overlap_score"))))
        misaligned.sort(key=lambda item: (item.get("selected_rank"), -_to_float(item.get("ordered_overlap_score"))))

        if aligned:
            policy = "prefer_aligned_sequences_that_preserve_prefix_and_touch_frontier_or_sink"
        elif partial:
            policy = "prefer_partial_overlap_sequences_then_bind_to_current_prefix"
        else:
            policy = "restrict_to_target_local_scope_and_avoid_global_sequence_drift"

        return {
            "aligned_guided_sequences": aligned[:10],
            "partially_aligned_guided_sequences": partial[:10],
            "misaligned_guided_sequences": misaligned[:10],
            "alignment_rules": [
                "same_hypothesis_id",
                "contains_target_function",
                "ordered_overlap_with_path",
                "preserves_matched_prefix",
                "touches_frontier_or_sink",
                "shares_sender_or_argument_pattern",
            ],
            "recommended_alignment_policy": policy,
        }

    def _build_repair_context(
        self,
        repair_alignment: Dict[str, Any],
        repair_by_candidate: Dict[str, Dict[str, Any]],
        selected_by_candidate: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        aligned = _listify(repair_alignment.get("aligned_guided_sequences"))
        partial = _listify(repair_alignment.get("partially_aligned_guided_sequences"))
        all_candidates = aligned + partial + _listify(repair_alignment.get("misaligned_guided_sequences"))
        pilot_items = []
        dryrun_attempt_count = len(all_candidates)
        for item in all_candidates:
            candidate_id = _text(item.get("candidate_id")).strip()
            failure = repair_by_candidate.get(candidate_id, {})
            if failure:
                pilot_items.append(failure)
        target_seen_count = sum(1 for item in pilot_items if bool(item.get("target_function_seen")))
        sink_reached_count = sum(1 for item in pilot_items if bool(item.get("sink_function_reached")))
        sink_reverted_count = sum(1 for item in pilot_items if bool(item.get("sink_reverted")))
        path_progress_count = sum(1 for item in pilot_items if _to_int(item.get("matched_prefix_len")) > 0)

        best_item = None
        best_key = None
        for item in aligned + partial:
            candidate_id = _text(item.get("candidate_id")).strip()
            failure = repair_by_candidate.get(candidate_id, {})
            key = (
                -int(bool(failure.get("sink_function_reached"))),
                -int(bool(failure.get("target_function_seen"))),
                -_to_int(failure.get("matched_prefix_len")),
                item.get("selected_rank"),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_item = item

        best_sequence = _clean_list(best_item.get("guided_sequence")) if isinstance(best_item, dict) else []
        best_reason = "unknown"
        if best_item:
            candidate_id = _text(best_item.get("candidate_id")).strip()
            failure = repair_by_candidate.get(candidate_id, {})
            if bool(failure.get("sink_function_reached")):
                best_reason = "highest aligned repair that reached the sink"
            elif bool(failure.get("target_function_seen")):
                best_reason = "highest aligned repair that observed the target function"
            elif _to_int(failure.get("matched_prefix_len")) > 0:
                best_reason = "highest aligned repair that preserved prefix progress"
            else:
                best_reason = "best overlap among aligned or partially aligned repairs"

        failure_counter: Counter = Counter()
        for item in pilot_items:
            failure_counter[REPAIR_FAILURE_MODE_MAP.get(_text(item.get("failure_type")).strip(), "unknown")] += 1
        dominant_failure_mode = "unknown"
        if failure_counter:
            dominant_failure_mode = failure_counter.most_common(1)[0][0]

        return {
            "has_repair_history": bool(all_candidates),
            "repair_attempt_count": len(all_candidates),
            "dryrun_attempt_count": dryrun_attempt_count,
            "pilot_attempt_count": len(pilot_items),
            "target_seen_attempt_count": target_seen_count,
            "sink_reached_attempt_count": sink_reached_count,
            "sink_reverted_attempt_count": sink_reverted_count,
            "path_progress_attempt_count": path_progress_count,
            "best_repair_prefix_len": max([_to_int(item.get("matched_prefix_len")) for item in pilot_items] or [0]),
            "best_guided_sequence": best_sequence,
            "best_repair_reason": best_reason,
            "dominant_failure_mode": dominant_failure_mode,
            "repair_context_role": "tfi_local_scope_evidence",
        }

    def _build_repair_candidate_scope(self, repair_alignment: Dict[str, Any]) -> Dict[str, Any]:
        aligned = _listify(repair_alignment.get("aligned_guided_sequences"))
        partial = _listify(repair_alignment.get("partially_aligned_guided_sequences"))
        recommended = sorted(
            aligned,
            key=lambda item: (
                -int(bool(item.get("sink_function_reached"))),
                -int(bool(item.get("target_seen"))),
                -_to_int(item.get("matched_prefix_len")),
                item.get("selected_rank"),
            ),
        )[:10]
        return {
            "aligned_repair_candidates": aligned[:10],
            "partially_aligned_repair_candidates": partial[:10],
            "misaligned_repair_candidates": _listify(repair_alignment.get("misaligned_guided_sequences"))[:10],
            "recommended_candidates": recommended,
        }

    def _derive_observed_matched_prefix(
        self,
        hypothesis_samples: List[Dict[str, Any]],
        target_function: str,
        frontier: Optional[str],
        sink: Optional[str],
    ) -> Tuple[List[str], str]:
        best_prefix: List[str] = []
        best_source = "unknown"
        best_len = -1
        boundary_functions = {item for item in [target_function, frontier, sink] if _text(item).strip()}

        for sample in hypothesis_samples:
            prefix = _clean_list(sample.get("observed_prefix"))
            prefix_source = _text(sample.get("observed_prefix_source")).strip() or "unknown"
            if not prefix:
                prefix = _clean_list(sample.get("prefix_reached"))
                if prefix:
                    prefix_source = "ksi_evidence"
            if not prefix:
                continue
            sample_functions = _clean_list(sample.get("functions"))
            if boundary_functions and prefix and any(item in boundary_functions for item in prefix):
                if not bool(sample.get("sink_function_reached")):
                    # Observed prefix should not claim an unreached frontier/sink.
                    continue
                if sample_functions and _prefix_len(prefix, sample_functions) != len(prefix):
                    continue
            current_len = len(prefix)
            if current_len > best_len:
                best_len = current_len
                best_prefix = prefix
                best_source = prefix_source
        return best_prefix, best_source

    def _build_scope_local_evidence_check(
        self,
        planned_path: List[str],
        observed_matched_prefix: List[str],
        matched_prefix_source: str,
        target_function: str,
        frontier: Optional[str],
        sink: Optional[str],
        repair_context: Dict[str, Any],
        repair_alignment: Dict[str, Any],
    ) -> Dict[str, Any]:
        matched_prefix_is_observed = bool(observed_matched_prefix)
        boundary_functions = {item for item in [target_function, frontier, sink] if _text(item).strip()}
        best_sequence = _clean_list(repair_context.get("best_guided_sequence"))
        aligned = _listify(repair_alignment.get("aligned_guided_sequences"))
        partial = _listify(repair_alignment.get("partially_aligned_guided_sequences"))

        binding_sequence = list(best_sequence)
        binding_sequence_source = "repair_context_best"
        chosen_item: Optional[Dict[str, Any]] = None
        chosen_key = None
        for item in aligned + partial:
            sequence = _clean_list(item.get("guided_sequence"))
            if not sequence:
                continue
            contains_target = bool(target_function and target_function in sequence)
            contains_frontier = bool(frontier and frontier in sequence)
            contains_sink = bool(sink and sink in sequence)
            preserves_observed_prefix = bool(
                matched_prefix_is_observed
                and observed_matched_prefix
                and _prefix_len(observed_matched_prefix, sequence) == len(observed_matched_prefix)
            )
            exact_match = bool(item.get("exact_match"))
            sink_reached = bool(item.get("sink_function_reached"))
            target_seen = bool(item.get("target_seen"))
            if not (contains_target or contains_frontier or contains_sink or preserves_observed_prefix or exact_match or sink_reached):
                continue
            candidate_key = (
                -int(sink_reached),
                -int(exact_match),
                -int(preserves_observed_prefix),
                -int(contains_sink),
                -int(contains_frontier),
                -int(contains_target),
                -int(target_seen),
                -_to_int(item.get("matched_prefix_len")),
                _to_int(item.get("selected_rank"), 10 ** 9),
            )
            if chosen_key is None or candidate_key < chosen_key:
                chosen_key = candidate_key
                binding_sequence = sequence
                chosen_item = item
                binding_sequence_source = "aligned_guided_sequence" if item in aligned else "partially_aligned_guided_sequence"

        scope_local_reasons: List[str] = []
        scope_local_failures: List[str] = []
        contains_target = bool(target_function and target_function in binding_sequence)
        contains_frontier = bool(frontier and frontier in binding_sequence)
        contains_sink = bool(sink and sink in binding_sequence)
        candidate_target_matches_boundary = bool(
            chosen_item
            and _text(chosen_item.get("target_function")).strip()
            and _text(chosen_item.get("target_function")).strip() in boundary_functions
        )
        touches_scope_boundary = bool(
            contains_target or contains_frontier or contains_sink or candidate_target_matches_boundary
        )
        preserves_observed_prefix = bool(
            matched_prefix_is_observed and observed_matched_prefix and _prefix_len(observed_matched_prefix, binding_sequence) == len(observed_matched_prefix)
        )
        exact_match_for_scope = bool(
            chosen_item
            and bool(chosen_item.get("exact_match"))
            and touches_scope_boundary
        )
        sink_reached_for_scope = bool(
            chosen_item
            and bool(chosen_item.get("sink_function_reached"))
            and touches_scope_boundary
        )
        target_seen_is_scope_local = bool(
            chosen_item
            and bool(chosen_item.get("target_seen"))
            and touches_scope_boundary
        )

        if contains_target:
            scope_local_reasons.append("contains_target_function")
        else:
            scope_local_failures.append("missing_target_function")
        if contains_frontier:
            scope_local_reasons.append("contains_frontier")
        elif frontier:
            scope_local_failures.append("missing_frontier")
        if contains_sink:
            scope_local_reasons.append("contains_sink")
        elif sink:
            scope_local_failures.append("missing_sink")
        if preserves_observed_prefix:
            scope_local_reasons.append("preserves_observed_matched_prefix")
        elif matched_prefix_is_observed and observed_matched_prefix:
            scope_local_failures.append("does_not_preserve_observed_matched_prefix")
        if sink_reached_for_scope:
            scope_local_reasons.append("sink_reached_for_this_scope")
        if exact_match_for_scope:
            scope_local_reasons.append("exact_match_for_this_scope")
        if chosen_item:
            scope_local_reasons.append("selected_from_scope_aligned_repairs")
        else:
            scope_local_failures.append("binding_skeleton_selected_from_nonlocal_best_repair")

        binding_skeleton_is_scope_local = bool(
            binding_sequence and (
                contains_target
                or contains_frontier
                or contains_sink
                or preserves_observed_prefix
                or exact_match_for_scope
                or sink_reached_for_scope
            )
        )
        sink_reached_is_scope_local = bool(binding_skeleton_is_scope_local and sink_reached_for_scope)

        diagnosis_uses_scope_local_evidence = bool(
            binding_skeleton_is_scope_local
            or matched_prefix_is_observed
            or target_seen_is_scope_local
            or sink_reached_is_scope_local
        )

        return {
            "planned_path": planned_path,
            "observed_matched_prefix": observed_matched_prefix,
            "matched_prefix_is_observed": matched_prefix_is_observed,
            "matched_prefix_source": matched_prefix_source if matched_prefix_is_observed else "unknown",
            "binding_skeleton": binding_sequence,
            "binding_skeleton_source": binding_sequence_source,
            "binding_sequence_skeleton": binding_sequence,
            "binding_sequence_skeleton_source": binding_sequence_source,
            "binding_skeleton_is_scope_local": binding_skeleton_is_scope_local,
            "binding_skeleton_scope_local_reasons": scope_local_reasons,
            "binding_skeleton_scope_local_failures": scope_local_failures,
            "target_seen_is_scope_local": target_seen_is_scope_local,
            "sink_reached_is_scope_local": sink_reached_is_scope_local,
            "exact_match_is_scope_local": bool(binding_skeleton_is_scope_local and exact_match_for_scope),
            "diagnosis_uses_scope_local_evidence": diagnosis_uses_scope_local_evidence,
        }

    def _build_tfi_failure_diagnosis(
        self,
        target_function: str,
        frontier: Optional[str],
        sink: Optional[str],
        matched_prefix: List[str],
        matched_prefix_is_observed: bool,
        path_context_seed: Dict[str, Any],
        repair_context: Dict[str, Any],
        repair_alignment: Dict[str, Any],
        repair_negative_scope: Dict[str, Any],
        repair_positive_scope: Dict[str, Any],
        scope_local_evidence_check: Dict[str, Any],
    ) -> Dict[str, Any]:
        aligned = _listify(repair_alignment.get("aligned_guided_sequences"))
        partial = _listify(repair_alignment.get("partially_aligned_guided_sequences"))
        sink_reached_count = _to_int(repair_context.get("sink_reached_attempt_count"))
        sink_reverted_count = _to_int(repair_context.get("sink_reverted_attempt_count"))
        target_seen_count = _to_int(repair_context.get("target_seen_attempt_count"))
        path_progress_count = _to_int(repair_context.get("path_progress_attempt_count"))
        dominant_failure_mode = _text(repair_context.get("dominant_failure_mode")).strip() or "unknown"
        best_repair_prefix_len = _to_int(repair_context.get("best_repair_prefix_len"))
        preserve_prefix = bool(matched_prefix_is_observed and (matched_prefix or path_context_seed.get("preserve_prefix_required")))
        sink_reached_but_reverted = _listify(repair_negative_scope.get("sink_reached_but_reverted"))
        prefix_improving = _listify(repair_positive_scope.get("prefix_improving_repairs"))
        binding_skeleton_is_scope_local = bool(scope_local_evidence_check.get("binding_skeleton_is_scope_local"))
        sink_reached_is_scope_local = bool(scope_local_evidence_check.get("sink_reached_is_scope_local"))
        target_seen_is_scope_local = bool(scope_local_evidence_check.get("target_seen_is_scope_local"))
        exact_match_is_scope_local = bool(scope_local_evidence_check.get("exact_match_is_scope_local"))
        binding_like_nonlocal_skeleton = bool(
            target_seen_count > 0
            and sink_reached_count > 0
            and sink_reverted_count > 0
            and (preserve_prefix or best_repair_prefix_len > 0 or aligned or partial)
            and not binding_skeleton_is_scope_local
        )
        binding_like_nonlocal_observation = bool(
            target_seen_count > 0
            and sink_reached_count > 0
            and sink_reverted_count > 0
            and (preserve_prefix or best_repair_prefix_len > 0 or aligned or partial)
            and binding_skeleton_is_scope_local
            and (not sink_reached_is_scope_local or not target_seen_is_scope_local)
        )

        evidence: List[str] = []
        diagnosis_type = "unknown"
        legacy_diagnosis_type = None
        gap_type = None
        recommended_next_step = "continue_target_local_diagnosis"
        reason = "insufficient evidence to distinguish binding vs scope alignment gap"
        diagnosis_downgrade_reason = None
        agent_guidance_required = False
        should_mark_existing_pool_candidates = False
        should_generate_action_atom_now = False

        if target_seen_count > 0:
            evidence.append("target_seen_attempt_count>0")
        if sink_reached_count > 0:
            evidence.append("sink_reached_attempt_count>0")
        if sink_reverted_count > 0:
            evidence.append("sink_reverted_attempt_count>0")
        if preserve_prefix:
            evidence.append("preserves_matched_prefix")
        if best_repair_prefix_len > 0:
            evidence.append("best_repair_prefix_len>0")
        if aligned:
            evidence.append("aligned_guided_sequences_available")
        if partial:
            evidence.append("partially_aligned_guided_sequences_available")
        if dominant_failure_mode != "unknown":
            evidence.append("dominant_failure_mode=%s" % dominant_failure_mode)
        if binding_skeleton_is_scope_local:
            evidence.append("binding_skeleton_is_scope_local")
        if exact_match_is_scope_local:
            evidence.append("exact_match_for_this_scope")
        if sink_reached_is_scope_local:
            evidence.append("sink_reached_for_this_scope")
        if target_seen_is_scope_local:
            evidence.append("target_seen_is_scope_local")
        if not matched_prefix_is_observed:
            evidence.append("matched_prefix_is_not_observed")
        if binding_like_nonlocal_skeleton:
            evidence.append("binding_like_but_nonlocal_skeleton")
        if binding_like_nonlocal_observation:
            evidence.append("binding_like_but_nonlocal_observation")

        if (
            target_seen_is_scope_local
            and binding_skeleton_is_scope_local
            and sink_reached_is_scope_local
            and sink_reverted_count > 0
            and (preserve_prefix or exact_match_is_scope_local or sink_reached_is_scope_local)
        ):
            diagnosis_type = "binding_gap"
            recommended_next_step = "preserve_sequence_skeleton_and_repair_sender_value_args_prestate"
            reason = "repair skeleton reaches target/sink but still reverts, so binding is more suspect than missing actions"
        elif binding_like_nonlocal_skeleton:
            diagnosis_type = "scope_alignment_gap" if (target_seen_count > 0 or partial or aligned) else "unknown"
            recommended_next_step = (
                "apply_tfi_scoped_candidate_restriction_before_action_generation"
                if diagnosis_type == "scope_alignment_gap"
                else "continue_target_local_diagnosis"
            )
            reason = "binding-shaped evidence exists, but the skeleton is not scope-local to the current target/frontier/sink"
            diagnosis_downgrade_reason = "binding_skeleton_not_scope_local"
        elif binding_like_nonlocal_observation:
            diagnosis_type = "scope_alignment_gap"
            recommended_next_step = "apply_tfi_scoped_candidate_restriction_before_action_generation"
            reason = "repair skeleton preserves some local structure, but target/sink reach evidence is not scope-local enough to support binding repair"
        elif (
            target_seen_count > 0
            and not sink_reached_is_scope_local
            and (
                path_progress_count == 0
                or not matched_prefix_is_observed
                or best_repair_prefix_len <= max(0, len(matched_prefix))
                or dominant_failure_mode == "sequence_order_misaligned"
                or not binding_skeleton_is_scope_local
                or not target_seen_is_scope_local
                or (not aligned and (partial or _listify(repair_alignment.get("misaligned_guided_sequences"))))
            )
        ):
            diagnosis_type = "scope_alignment_gap"
            recommended_next_step = "apply_tfi_scoped_candidate_restriction_before_action_generation"
            reason = "repair candidates touch the target but do not preserve or advance the current local path context"
        elif (
            not aligned
            and not partial
            and bool(target_function or frontier or sink)
        ):
            diagnosis_type = "possible_local_candidate_gap"
            legacy_diagnosis_type = "possible_missing_action_atom"
            gap_type = "local_candidate_gap"
            recommended_next_step = "emit_local_candidate_gap_feedback_for_existing_pool_generation"
            reason = "no aligned or partially aligned local repair candidates remain after scope construction"
            agent_guidance_required = True
            should_mark_existing_pool_candidates = True
            should_generate_action_atom_now = True

        return {
            "diagnosis_type": diagnosis_type,
            "legacy_diagnosis_type": legacy_diagnosis_type,
            "gap_type": gap_type,
            "evidence": evidence,
            "recommended_next_step": recommended_next_step,
            "should_generate_action_atom_now": bool(should_generate_action_atom_now),
            "should_create_new_pool": False,
            "should_mark_existing_pool_candidates": bool(should_mark_existing_pool_candidates),
            "agent_guidance_required": bool(agent_guidance_required),
            "reason": reason,
            "diagnosis_downgrade_reason": diagnosis_downgrade_reason,
        }

    def _build_binding_repair_context(
        self,
        diagnosis: Dict[str, Any],
        target_function: str,
        sink: Optional[str],
        repair_context: Dict[str, Any],
        repair_alignment: Dict[str, Any],
        repair_negative_scope: Dict[str, Any],
        sender_scope: Dict[str, Any],
        value_scope: Dict[str, Any],
        argument_scope: Dict[str, Any],
        scope_local_evidence_check: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if _text(diagnosis.get("diagnosis_type")).strip() != "binding_gap":
            return None

        if not bool(scope_local_evidence_check.get("binding_skeleton_is_scope_local")):
            return None

        sequence_skeleton = _clean_list(scope_local_evidence_check.get("binding_sequence_skeleton"))
        if not sequence_skeleton:
            return None

        candidate_sender_policies: List[str] = []
        recommended_sender = _text(sender_scope.get("recommended_sender_policy")).strip()
        if recommended_sender and recommended_sender != "unknown":
            candidate_sender_policies.append(recommended_sender)
        if sender_scope.get("stable_same_sender_candidates"):
            candidate_sender_policies.append("keep_same_sender_across_sequence")
        if sender_scope.get("prefix_success_senders"):
            candidate_sender_policies.append("try_prefix_success_sender")
        if sender_scope.get("owner_or_deployer_like_candidates"):
            candidate_sender_policies.append("try_owner_or_deployer_if_available")

        candidate_value_policies: List[str] = []
        recommended_value = _text(value_scope.get("recommended_value_policy")).strip()
        if recommended_value and recommended_value != "unknown":
            candidate_value_policies.append(recommended_value)
        if value_scope.get("value_used_in_success_prefix"):
            candidate_value_policies.append("keep_observed_success_value")
        if value_scope.get("observed_zero_values"):
            candidate_value_policies.append("force_zero_value")
        if value_scope.get("observed_nonzero_values"):
            candidate_value_policies.append("try_small_nonzero_value")

        candidate_argument_policies: List[str] = []
        recommended_args = _text(argument_scope.get("recommended_argument_policy")).strip()
        if recommended_args and recommended_args != "unknown":
            candidate_argument_policies.append(recommended_args)
        for template_name in _clean_list(argument_scope.get("argument_templates")):
            candidate_argument_policies.append(template_name)

        strategy = "preserve_scope_local_skeleton_and_rebind_transaction_fields"

        return {
            "preserve_sequence_skeleton": True,
            "sequence_skeleton": sequence_skeleton,
            "sink_function": sink or target_function or None,
            "failed_sender_bindings": _listify(repair_negative_scope.get("failed_sender_bindings"))[: self.max_samples_per_scope],
            "failed_value_bindings": _listify(repair_negative_scope.get("failed_value_bindings"))[: self.max_samples_per_scope],
            "failed_argument_bindings": _listify(repair_negative_scope.get("failed_argument_bindings"))[: self.max_samples_per_scope],
            "failed_prestate_assumptions": _listify(repair_negative_scope.get("failed_prestate_assumptions"))[: self.max_samples_per_scope],
            "candidate_sender_policies": _unique_list(candidate_sender_policies),
            "candidate_value_policies": _unique_list(candidate_value_policies),
            "candidate_argument_policies": _unique_list(candidate_argument_policies),
            "recommended_binding_strategy": strategy,
        }

    def _extract_generic_skeleton_functions(self, skeleton: List[str]) -> List[str]:
        generic_hits = []
        for name in skeleton:
            lowered = _text(name).strip().lower()
            if not lowered:
                continue
            if _contains_keyword(lowered, GENERIC_TOKEN_KEYWORDS):
                generic_hits.append(name)
                continue
            if _looks_config_like(lowered) or _looks_permission_like(lowered):
                generic_hits.append(name)
        return _unique_list(generic_hits)

    def _build_generic_skeleton_leakage_check(
        self,
        scope_local_evidence_check: Dict[str, Any],
        binding_repair_context: Optional[Dict[str, Any]],
        target_function: str,
        frontier: Optional[str],
        sink: Optional[str],
        observed_matched_prefix: List[str],
    ) -> Dict[str, Any]:
        skeleton = _clean_list(
            (binding_repair_context or {}).get("sequence_skeleton")
            or scope_local_evidence_check.get("binding_skeleton")
        )
        generic_functions = self._extract_generic_skeleton_functions(skeleton)
        has_generic = bool(generic_functions)
        is_scope_local = bool(scope_local_evidence_check.get("binding_skeleton_is_scope_local"))
        reasons = _clean_list(scope_local_evidence_check.get("binding_skeleton_scope_local_reasons"))
        failures = _clean_list(scope_local_evidence_check.get("binding_skeleton_scope_local_failures"))
        exact_match_for_scope = bool(scope_local_evidence_check.get("exact_match_is_scope_local"))
        sink_reached_for_scope = bool(scope_local_evidence_check.get("sink_reached_is_scope_local"))

        contains_target = bool(target_function and target_function in skeleton)
        contains_frontier = bool(frontier and frontier in skeleton)
        contains_sink = bool(sink and sink in skeleton)
        preserves_prefix = bool(
            observed_matched_prefix
            and _prefix_len(observed_matched_prefix, skeleton) == len(observed_matched_prefix)
        )

        leakage_detected = bool(
            has_generic
            and not (
                contains_target
                or contains_frontier
                or contains_sink
                or preserves_prefix
                or exact_match_for_scope
                or sink_reached_for_scope
            )
        )

        leakage_reasons = []
        if has_generic:
            leakage_reasons.append("generic_defi_token_skeleton_detected")
        if has_generic and not contains_target:
            leakage_reasons.append("generic_skeleton_missing_target_function")
        if has_generic and frontier and not contains_frontier:
            leakage_reasons.append("generic_skeleton_missing_frontier")
        if has_generic and sink and not contains_sink:
            leakage_reasons.append("generic_skeleton_missing_sink")
        if has_generic and observed_matched_prefix and not preserves_prefix:
            leakage_reasons.append("generic_skeleton_does_not_preserve_observed_prefix")
        if has_generic and not exact_match_for_scope:
            leakage_reasons.append("generic_skeleton_not_exact_match_for_scope")
        if has_generic and not sink_reached_for_scope:
            leakage_reasons.append("generic_skeleton_not_sink_reached_for_scope")
        leakage_reasons.extend(failures)

        allowed_as_positive_evidence = bool(
            has_generic is False
            or (
                is_scope_local
                and (
                    contains_target
                    or contains_frontier
                    or contains_sink
                    or preserves_prefix
                    or exact_match_for_scope
                    or sink_reached_for_scope
                )
                and not leakage_detected
            )
        )

        return {
            "has_generic_defi_token_skeleton": has_generic,
            "generic_skeleton_functions": generic_functions,
            "is_scope_local": is_scope_local,
            "leakage_detected": leakage_detected,
            "leakage_reasons": _unique_list(leakage_reasons),
            "allowed_as_positive_evidence": allowed_as_positive_evidence,
        }

    def _apply_generic_skeleton_leakage_guard(
        self,
        repair_positive_scope: Dict[str, Any],
        generic_skeleton_leakage_check: Dict[str, Any],
    ) -> Dict[str, Any]:
        guarded = dict(repair_positive_scope)
        if bool(generic_skeleton_leakage_check.get("allowed_as_positive_evidence")):
            return guarded
        guarded["positive_evidence_blocked_due_to_generic_skeleton_leakage"] = True
        guarded["blocked_positive_evidence_reason"] = "generic_defi_token_skeleton_leakage"
        guarded["prefix_improving_repairs"] = []
        guarded["target_seen_repairs"] = []
        guarded["sink_reaching_repairs"] = []
        guarded["non_reverting_prefix_repairs"] = []
        guarded["candidate_repair_templates"] = []
        guarded["recommended_repair_direction"] = "do_not_use_nonlocal_generic_skeleton_as_positive_evidence"
        return guarded

    def _build_scope_restriction_context(
        self,
        diagnosis: Dict[str, Any],
        matched_prefix: List[str],
        frontier: Optional[str],
        sink: Optional[str],
        target_function: str,
        repair_alignment: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        diagnosis_type = _text(diagnosis.get("diagnosis_type")).strip()
        if diagnosis_type not in {"scope_alignment_gap", "possible_local_candidate_gap"}:
            return None

        required_frontier_or_sink = frontier or sink or target_function or None
        recommended_scope_policy = "preserve_prefix_and_require_frontier_or_sink_touch"
        if not matched_prefix:
            recommended_scope_policy = "require_frontier_or_sink_touch_and_reject_target_only_drift"
        if not _listify(repair_alignment.get("aligned_guided_sequences")) and _listify(repair_alignment.get("partially_aligned_guided_sequences")):
            recommended_scope_policy = "promote_partially_aligned_repairs_only_if_they_preserve_prefix_or_touch_frontier"
        if diagnosis_type == "possible_local_candidate_gap":
            recommended_scope_policy = "compose_scope_local_candidate_from_frontier_sink_and_observed_prefix"

        return {
            "preserve_matched_prefix_required": True,
            "matched_prefix": matched_prefix,
            "required_frontier_or_sink": required_frontier_or_sink,
            "allowed_candidate_basis": [
                "matched_prefix",
                "frontier",
                "sink",
                "target_function",
                "partially_aligned_repair",
            ],
            "forbidden_candidate_patterns": [
                "target_seen_but_no_path_progress",
                "missing_frontier_or_sink",
                "does_not_preserve_prefix",
                "sequence_order_misaligned",
            ],
            "recommended_scope_policy": recommended_scope_policy,
            "diagnosis_type": diagnosis_type,
        }

    def _build_negative_scope(
        self,
        execution_samples: Dict[str, Any],
        repair_alignment: Dict[str, Any],
        repair_by_candidate: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        avoid_bindings = []
        repeated_revert_patterns: Counter = Counter()
        target_seen_no_progress = []
        hit_sink_reverted = []

        for item in execution_samples.get("sink_failure_samples") or []:
            avoid_bindings.append(
                {
                    "sequence_id": item.get("sequence_id"),
                    "sender_sequence": item.get("sender_sequence"),
                    "value_sequence": item.get("value_sequence"),
                    "args_sequence": item.get("args_sequence"),
                    "revert_reason": item.get("revert_reason"),
                }
            )
            repeated_revert_patterns[_text(item.get("revert_reason")).strip() or "b''"] += 1

        misaligned_sequences = []
        for item in _listify(repair_alignment.get("misaligned_guided_sequences")):
            misaligned_sequences.append(
                {
                    "candidate_id": item.get("candidate_id"),
                    "guided_sequence": item.get("guided_sequence"),
                    "alignment_reasons": item.get("alignment_reasons"),
                }
            )

        for bucket in (
            _listify(repair_alignment.get("aligned_guided_sequences"))
            + _listify(repair_alignment.get("partially_aligned_guided_sequences"))
        ):
            candidate_id = _text(bucket.get("candidate_id")).strip()
            failure = repair_by_candidate.get(candidate_id, {})
            if bool(failure.get("target_function_seen")) and _to_int(failure.get("matched_prefix_len")) <= 0:
                target_seen_no_progress.append(
                    {
                        "candidate_id": candidate_id,
                        "guided_sequence": bucket.get("guided_sequence"),
                        "observed_functions": failure.get("observed_functions"),
                    }
                )
            if bool(failure.get("sink_function_reached")) and bool(failure.get("sink_reverted")):
                hit_sink_reverted.append(
                    {
                        "candidate_id": candidate_id,
                        "guided_sequence": bucket.get("guided_sequence"),
                        "revert_reason": failure.get("revert_reason"),
                    }
                )

        return {
            "avoid_sender_value_arg_bindings": avoid_bindings[: self.max_samples_per_scope],
            "repeated_revert_patterns": [
                {"revert_reason": key, "count": int(value)}
                for key, value in repeated_revert_patterns.most_common(self.max_samples_per_scope)
            ],
            "misaligned_sequences": misaligned_sequences[: self.max_samples_per_scope],
            "target_seen_but_not_path_progress_examples": target_seen_no_progress[: self.max_samples_per_scope],
            "hit_sink_but_reverted_examples": hit_sink_reverted[: self.max_samples_per_scope],
        }

    def _build_repair_negative_scope(
        self,
        repair_alignment: Dict[str, Any],
        repair_by_candidate: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        failed_sender = []
        failed_value = []
        failed_args = []
        failed_order = []
        failed_prestate = []
        repeated_patterns: Counter = Counter()
        target_seen_no_progress = []
        sink_reached_reverted = []
        avoid_candidates = []

        for bucket_name in ("aligned_guided_sequences", "partially_aligned_guided_sequences", "misaligned_guided_sequences"):
            for item in _listify(repair_alignment.get(bucket_name)):
                candidate_id = _text(item.get("candidate_id")).strip()
                failure = repair_by_candidate.get(candidate_id, {})
                failure_type = _text(failure.get("failure_type")).strip() or "unknown"
                repeated_patterns[failure_type] += 1

                record = {
                    "candidate_id": candidate_id,
                    "guided_sequence": item.get("guided_sequence"),
                    "failure_type": failure_type,
                    "sender_sequence": failure.get("sender_sequence"),
                    "value_sequence": failure.get("value_sequence"),
                    "args_sequence": failure.get("args_sequence"),
                    "revert_reason": failure.get("revert_reason"),
                }

                if failure_type == "sender_or_permission_likely":
                    failed_sender.append(record)
                if failure_type == "value_likely":
                    failed_value.append(record)
                if failure_type in ("args_or_state_likely",):
                    failed_args.append(record)
                    failed_prestate.append(record)
                if failure_type in ("prefix_not_preserved", "sequence_mismatch"):
                    failed_order.append(record)
                if bool(failure.get("target_function_seen")) and _to_int(failure.get("matched_prefix_len")) <= 0:
                    target_seen_no_progress.append(record)
                if bool(failure.get("sink_function_reached")) and bool(failure.get("sink_reverted")):
                    sink_reached_reverted.append(record)
                if failure_type not in ("unknown", ""):
                    avoid_candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "failure_type": failure_type,
                            "guided_sequence": item.get("guided_sequence"),
                        }
                    )

        return {
            "failed_sender_bindings": failed_sender[: self.max_samples_per_scope],
            "failed_value_bindings": failed_value[: self.max_samples_per_scope],
            "failed_argument_bindings": failed_args[: self.max_samples_per_scope],
            "failed_sequence_orders": failed_order[: self.max_samples_per_scope],
            "failed_prestate_assumptions": failed_prestate[: self.max_samples_per_scope],
            "repeated_failed_repair_patterns": [
                {"failure_type": key, "count": int(value)}
                for key, value in repeated_patterns.most_common(self.max_samples_per_scope)
            ],
            "target_seen_but_no_path_progress": target_seen_no_progress[: self.max_samples_per_scope],
            "sink_reached_but_reverted": sink_reached_reverted[: self.max_samples_per_scope],
            "avoid_repair_candidates": _unique_list(avoid_candidates)[: self.max_samples_per_scope],
        }

    def _build_repair_positive_scope(
        self,
        execution_samples: Dict[str, Any],
        repair_alignment: Dict[str, Any],
        repair_by_candidate: Dict[str, Dict[str, Any]],
        setup_function_scope: Dict[str, Any],
        scope_path: List[str],
        target_function: str,
    ) -> Dict[str, Any]:
        prefix_improving = []
        target_seen = []
        sink_reaching = []
        non_reverting_prefix = []

        for item in _listify(repair_alignment.get("aligned_guided_sequences")) + _listify(
            repair_alignment.get("partially_aligned_guided_sequences")
        ):
            candidate_id = _text(item.get("candidate_id")).strip()
            failure = repair_by_candidate.get(candidate_id, {})
            record = {
                "candidate_id": candidate_id,
                "guided_sequence": item.get("guided_sequence"),
                "matched_prefix_len": failure.get("matched_prefix_len"),
                "target_function_seen": failure.get("target_function_seen"),
                "sink_function_reached": failure.get("sink_function_reached"),
                "sink_reverted": failure.get("sink_reverted"),
            }
            if _to_int(failure.get("matched_prefix_len")) > 0:
                prefix_improving.append(record)
            if bool(failure.get("target_function_seen")):
                target_seen.append(record)
            if bool(failure.get("sink_function_reached")):
                sink_reaching.append(record)
            if _to_int(failure.get("matched_prefix_len")) > 0 and not bool(failure.get("revert")):
                non_reverting_prefix.append(record)

        candidate_templates = []
        aligned_sequences = _listify(repair_alignment.get("aligned_guided_sequences"))
        if any(sum(1 for name in _clean_list(item.get("guided_sequence")) if _looks_asset_like(name)) >= 2 for item in aligned_sequences):
            candidate_templates.append("preserve_cross_call_asset_state_consistency")
        if prefix_improving and execution_samples.get("sink_failure_samples"):
            candidate_templates.append("preserve_observed_prefix_and_mutate_boundary_arguments")
        if prefix_improving:
            candidate_templates.append("reuse_scope_local_progress_with_additional_prestate_binding")
        if setup_function_scope.get("permission_like_functions"):
            candidate_templates.append("insert_scope_local_permission_or_prestate_setup")
        if execution_samples.get("sink_failure_samples"):
            candidate_templates.append("avoid_repeating_failed_binding_fields_after_revert")
        if execution_samples.get("prefix_success_samples"):
            candidate_templates.append("preserve_successful_prefix_binding_then_mutate_boundary_fields")
        candidate_templates = _unique_list(candidate_templates)

        recommended_direction = "stay_with_current_target_local_scope"
        if "preserve_cross_call_asset_state_consistency" in candidate_templates:
            recommended_direction = "preserve_scope_local_progress_and_repair_cross_call_state_binding"
        elif "insert_scope_local_permission_or_prestate_setup" in candidate_templates:
            recommended_direction = "preserve_scope_local_progress_and_bind_permission_or_prestate"
        elif "preserve_observed_prefix_and_mutate_boundary_arguments" in candidate_templates:
            recommended_direction = "reuse_scope_local_progress_and_mutate_boundary_fields"

        return {
            "prefix_improving_repairs": prefix_improving[: self.max_samples_per_scope],
            "target_seen_repairs": target_seen[: self.max_samples_per_scope],
            "sink_reaching_repairs": sink_reaching[: self.max_samples_per_scope],
            "non_reverting_prefix_repairs": non_reverting_prefix[: self.max_samples_per_scope],
            "candidate_repair_templates": candidate_templates,
            "recommended_repair_direction": recommended_direction,
        }

    def _build_context_completeness(
        self,
        path_context: Dict[str, Any],
        execution_samples: Dict[str, Any],
        sender_scope: Dict[str, Any],
        value_scope: Dict[str, Any],
        argument_scope: Dict[str, Any],
        repair_candidate_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        flags = {
            "has_path_context": bool(path_context.get("path") or path_context.get("target_function")),
            "has_prefix_success_sample": bool(execution_samples.get("prefix_success_samples")),
            "has_sink_failure_sample": bool(execution_samples.get("sink_failure_samples")),
            "has_sender_scope": bool(sender_scope.get("recommended_sender_policy") != "unknown" or sender_scope.get("prefix_success_senders")),
            "has_value_scope": bool(value_scope.get("recommended_value_policy") != "unknown" or value_scope.get("value_used_in_success_prefix") or value_scope.get("value_used_in_reverted_sink")),
            "has_argument_scope": bool(argument_scope.get("recommended_argument_policy") != "unknown" or argument_scope.get("argument_templates")),
            "has_repair_candidates": bool(repair_candidate_scope.get("aligned_repair_candidates") or repair_candidate_scope.get("partially_aligned_repair_candidates")),
        }
        score = round(sum(1 for value in flags.values() if value) / float(len(flags) or 1), 4)
        missing = []
        if not flags["has_path_context"]:
            missing.append("path_context")
        if not flags["has_prefix_success_sample"]:
            missing.append("prefix_success_sample")
        if not flags["has_sink_failure_sample"]:
            missing.append("sink_failure_sample")
        if not flags["has_sender_scope"]:
            missing.append("sender_scope")
        if not flags["has_value_scope"]:
            missing.append("value_scope")
        if not flags["has_argument_scope"]:
            missing.append("argument_scope")
        if not flags["has_repair_candidates"]:
            missing.append("repair_candidates")
        return {
            "has_path_context": flags["has_path_context"],
            "has_prefix_success_sample": flags["has_prefix_success_sample"],
            "has_sink_failure_sample": flags["has_sink_failure_sample"],
            "has_sender_scope": flags["has_sender_scope"],
            "has_value_scope": flags["has_value_scope"],
            "has_argument_scope": flags["has_argument_scope"],
            "has_repair_candidates": flags["has_repair_candidates"],
            "completeness_score": score,
            "missing_context_fields": missing,
        }

    def _build_request_exports(self, scopes_payload: Dict[str, Any]) -> Dict[str, Any]:
        scopes = _listify(scopes_payload.get("scopes"))
        binding_requests = []
        selection_requests = []
        binding_confidence: Counter = Counter()
        selection_confidence: Counter = Counter()
        skipped_nonlocal = 0
        skipped_generic = 0
        generic_scope_count = 0
        generic_positive_blocked = 0

        for scope in scopes:
            if not isinstance(scope, dict):
                continue
            diagnosis = scope.get("tfi_failure_diagnosis") if isinstance(scope.get("tfi_failure_diagnosis"), dict) else {}
            diagnosis_type = _text(diagnosis.get("diagnosis_type")).strip() or "unknown"
            path_context = scope.get("path_context") if isinstance(scope.get("path_context"), dict) else {}
            local_check = scope.get("scope_local_evidence_check") if isinstance(scope.get("scope_local_evidence_check"), dict) else {}
            leakage = scope.get("generic_skeleton_leakage_check") if isinstance(scope.get("generic_skeleton_leakage_check"), dict) else {}
            binding_ctx = scope.get("binding_repair_context") if isinstance(scope.get("binding_repair_context"), dict) else None
            restriction_ctx = scope.get("scope_restriction_context") if isinstance(scope.get("scope_restriction_context"), dict) else None
            negative_scope = scope.get("negative_scope") if isinstance(scope.get("negative_scope"), dict) else {}
            repair_negative = scope.get("repair_negative_scope") if isinstance(scope.get("repair_negative_scope"), dict) else {}
            repair_positive = scope.get("repair_positive_scope") if isinstance(scope.get("repair_positive_scope"), dict) else {}
            setup_scope = scope.get("setup_function_scope") if isinstance(scope.get("setup_function_scope"), dict) else {}

            if bool(leakage.get("has_generic_defi_token_skeleton")):
                generic_scope_count += 1
            if bool(leakage.get("has_generic_defi_token_skeleton")) and not bool(leakage.get("allowed_as_positive_evidence")):
                generic_positive_blocked += 1

            if diagnosis_type == "binding_gap":
                binding_allowed = bool(
                    bool(local_check.get("diagnosis_uses_scope_local_evidence"))
                    and bool(local_check.get("binding_skeleton_is_scope_local"))
                    and binding_ctx
                    and bool(binding_ctx.get("preserve_sequence_skeleton"))
                    and _clean_list(binding_ctx.get("sequence_skeleton"))
                    and bool(leakage.get("allowed_as_positive_evidence"))
                )
                if not binding_allowed:
                    if not bool(local_check.get("diagnosis_uses_scope_local_evidence")) or not bool(local_check.get("binding_skeleton_is_scope_local")):
                        skipped_nonlocal += 1
                    elif bool(leakage.get("has_generic_defi_token_skeleton")) and not bool(leakage.get("allowed_as_positive_evidence")):
                        skipped_generic += 1
                    continue
                request = self._build_binding_repair_request(
                    request_index=len(binding_requests) + 1,
                    scope=scope,
                    path_context=path_context,
                    binding_ctx=binding_ctx,
                    repair_negative=repair_negative,
                    repair_positive=repair_positive,
                    setup_scope=setup_scope,
                    leakage=leakage,
                )
                binding_requests.append(request)
                binding_confidence[request.get("confidence") or "low"] += 1
                continue

            if diagnosis_type in {"scope_alignment_gap", "possible_local_candidate_gap"}:
                if not restriction_ctx or not (path_context.get("planned_path") or scope.get("target_function") or scope.get("frontier") or scope.get("sink")):
                    skipped_nonlocal += 1
                    continue
                request = self._build_scoped_selection_request(
                    request_index=len(selection_requests) + 1,
                    scope=scope,
                    path_context=path_context,
                    restriction_ctx=restriction_ctx,
                    negative_scope=negative_scope,
                    repair_negative=repair_negative,
                    leakage=leakage,
                )
                selection_requests.append(request)
                selection_confidence[request.get("confidence") or "low"] += 1

        binding_payload = {
            "enabled": True,
            "source": "tfi_binding_repair_request_v1",
            "request_count": len(binding_requests),
            "requests": binding_requests,
        }
        selection_payload = {
            "enabled": True,
            "source": "tfi_scoped_selection_request_v1",
            "request_count": len(selection_requests),
            "requests": selection_requests,
        }
        stats = {
            "binding_repair_request_count": len(binding_requests),
            "scoped_selection_request_count": len(selection_requests),
            "binding_requests_by_confidence": {
                "high": int(binding_confidence.get("high", 0)),
                "medium": int(binding_confidence.get("medium", 0)),
                "low": int(binding_confidence.get("low", 0)),
            },
            "selection_requests_by_confidence": {
                "high": int(selection_confidence.get("high", 0)),
                "medium": int(selection_confidence.get("medium", 0)),
                "low": int(selection_confidence.get("low", 0)),
            },
            "requests_skipped_due_to_nonlocal_evidence": int(skipped_nonlocal),
            "requests_skipped_due_to_generic_skeleton_leakage": int(skipped_generic),
            "generic_skeleton_leakage_scope_count": int(generic_scope_count),
            "generic_skeleton_positive_evidence_blocked_count": int(generic_positive_blocked),
        }
        return {
            "binding_payload": binding_payload,
            "selection_payload": selection_payload,
            "stats": stats,
        }

    def _build_binding_repair_request(
        self,
        request_index: int,
        scope: Dict[str, Any],
        path_context: Dict[str, Any],
        binding_ctx: Dict[str, Any],
        repair_negative: Dict[str, Any],
        repair_positive: Dict[str, Any],
        setup_scope: Dict[str, Any],
        leakage: Dict[str, Any],
    ) -> Dict[str, Any]:
        observed_prefix = _clean_list(path_context.get("observed_matched_prefix"))
        skeleton = _clean_list(binding_ctx.get("sequence_skeleton"))
        local_positive = []
        for item in _listify(repair_positive.get("sink_reaching_repairs")):
            sequence = _clean_list(item.get("guided_sequence"))
            if not sequence:
                continue
            if sequence == skeleton:
                local_positive.append(item)
                continue
            if observed_prefix and _prefix_len(observed_prefix, sequence) == len(observed_prefix):
                local_positive.append(item)
                continue
            if scope.get("target_function") and scope.get("target_function") in sequence:
                local_positive.append(item)

        sender_failed = _listify(binding_ctx.get("failed_sender_bindings"))[: self.max_samples_per_scope]
        value_failed = _listify(binding_ctx.get("failed_value_bindings"))[: self.max_samples_per_scope]
        args_failed = _listify(binding_ctx.get("failed_argument_bindings"))[: self.max_samples_per_scope]
        prestate_failed = _listify(binding_ctx.get("failed_prestate_assumptions"))[: self.max_samples_per_scope]
        confidence = "high"
        if len(_clean_list(path_context.get("observed_matched_prefix"))) < len(_clean_list(binding_ctx.get("sequence_skeleton"))):
            confidence = "medium"
        if not sender_failed and not value_failed and not args_failed and not prestate_failed:
            confidence = "medium"

        return {
            "request_id": "TFI_BIND_REQ_%d" % request_index,
            "scope_id": scope.get("scope_id"),
            "hypothesis_id": scope.get("hypothesis_id"),
            "target_function": scope.get("target_function"),
            "frontier": scope.get("frontier"),
            "sink": scope.get("sink"),
            "planned_path": _clean_list(path_context.get("planned_path")),
            "observed_matched_prefix": _clean_list(path_context.get("observed_matched_prefix")),
            "sequence_skeleton": _clean_list(binding_ctx.get("sequence_skeleton")),
            "preserve_sequence_skeleton": True,
            "repair_goal": "repair_sender_value_args_prestate_without_changing_function_order",
            "binding_slots": {
                "sender": {
                    "failed_bindings": sender_failed,
                    "candidate_policies": _clean_list(binding_ctx.get("candidate_sender_policies")),
                    "avoid_policies": ["avoid_sink_revert_sender"] if repair_negative.get("failed_sender_bindings") else [],
                },
                "value": {
                    "failed_bindings": value_failed,
                    "candidate_policies": _clean_list(binding_ctx.get("candidate_value_policies")),
                    "avoid_policies": ["avoid_reverted_value"] if repair_negative.get("failed_value_bindings") else [],
                },
                "args": {
                    "failed_bindings": args_failed,
                    "candidate_policies": _clean_list(binding_ctx.get("candidate_argument_policies")),
                    "argument_templates": _clean_list(scope.get("argument_scope", {}).get("argument_templates")),
                    "avoid_patterns": _listify(scope.get("argument_scope", {}).get("repeated_failed_arg_patterns"))[: self.max_samples_per_scope],
                },
                "prestate": {
                    "failed_assumptions": prestate_failed,
                    "candidate_setup_hints": _clean_list(setup_scope.get("candidate_setup_functions")),
                    "avoid_assumptions": _listify(repair_negative.get("failed_prestate_assumptions"))[: self.max_samples_per_scope],
                },
            },
            "positive_evidence": _unique_list(local_positive)[: self.max_samples_per_scope],
            "negative_evidence": _listify(repair_negative.get("sink_reached_but_reverted"))[: self.max_samples_per_scope],
            "generic_skeleton_leakage_check": leakage,
            "constraints": {
                "do_not_change_function_order": True,
                "do_not_replace_sequence_skeleton": True,
                "must_include_target_function": True,
                "must_preserve_observed_prefix": True,
                "must_use_scope_local_skeleton": True,
            },
            "confidence": confidence,
            "reason": "scope-local skeleton reaches target/sink but reverts; binding repair is preferred over action generation",
        }

    def _build_scoped_selection_request(
        self,
        request_index: int,
        scope: Dict[str, Any],
        path_context: Dict[str, Any],
        restriction_ctx: Dict[str, Any],
        negative_scope: Dict[str, Any],
        repair_negative: Dict[str, Any],
        leakage: Dict[str, Any],
    ) -> Dict[str, Any]:
        diagnosis = scope.get("tfi_failure_diagnosis") if isinstance(scope.get("tfi_failure_diagnosis"), dict) else {}
        diagnosis_type = _text(diagnosis.get("diagnosis_type")).strip() or "unknown"
        local_check = scope.get("scope_local_evidence_check") if isinstance(scope.get("scope_local_evidence_check"), dict) else {}
        confidence = "medium"
        if not bool(local_check.get("diagnosis_uses_scope_local_evidence")):
            confidence = "low"
        elif _clean_list(path_context.get("observed_matched_prefix")):
            confidence = "high"

        nonlocal_examples = []
        if bool(leakage.get("has_generic_defi_token_skeleton")) and not bool(leakage.get("allowed_as_positive_evidence")):
            nonlocal_examples.append(
                {
                    "binding_skeleton": _clean_list(local_check.get("binding_skeleton")),
                    "generic_skeleton_functions": _clean_list(leakage.get("generic_skeleton_functions")),
                    "leakage_reasons": _clean_list(leakage.get("leakage_reasons")),
                }
            )

        selection_goal = "select_or_compose_candidates_within_target_local_scope"
        reason = "current repair candidates touch target but do not advance the local hypothesis path"
        if diagnosis_type == "possible_local_candidate_gap":
            selection_goal = "compose_missing_scope_local_candidate_action"
            reason = "no aligned local candidate exists, so compose a candidate constrained by the current frontier/sink scope"

        return {
            "request_id": "TFI_SCOPE_SEL_REQ_%d" % request_index,
            "scope_id": scope.get("scope_id"),
            "hypothesis_id": scope.get("hypothesis_id"),
            "target_function": scope.get("target_function"),
            "frontier": scope.get("frontier"),
            "sink": scope.get("sink"),
            "planned_path": _clean_list(path_context.get("planned_path")),
            "observed_matched_prefix": _clean_list(path_context.get("observed_matched_prefix")),
            "selection_goal": selection_goal,
            "diagnosis_type": diagnosis_type,
            "required_constraints": {
                "preserve_observed_matched_prefix": True,
                "must_touch_frontier_or_sink": True,
                "must_not_use_nonlocal_best_repair": True,
                "must_not_select_sequence_without_target_progress": True,
                "must_not_use_generic_skeleton_unless_scope_local": True,
            },
            "allowed_candidate_basis": [
                "observed_matched_prefix",
                "frontier",
                "sink",
                "target_function",
                "partially_aligned_repair",
                "prefix_success_samples",
            ],
            "forbidden_candidate_patterns": _unique_list(
                _clean_list(restriction_ctx.get("forbidden_candidate_patterns"))
                + [
                    "nonlocal_binding_skeleton",
                    "generic_defi_token_skeleton_leakage",
                ]
            ),
            "negative_evidence": _unique_list(
                _listify(negative_scope.get("target_seen_but_not_path_progress_examples"))[: self.max_samples_per_scope]
                + _listify(repair_negative.get("target_seen_but_no_path_progress"))[: self.max_samples_per_scope]
            )[: self.max_samples_per_scope],
            "partial_candidates": _listify(scope.get("repair_candidate_scope", {}).get("partially_aligned_repair_candidates"))[: self.max_samples_per_scope],
            "nonlocal_skeleton_examples": nonlocal_examples[: self.max_samples_per_scope],
            "generic_skeleton_leakage_check": leakage,
            "confidence": confidence,
            "reason": reason,
        }

    def _build_summary(self, scopes_payload: Dict[str, Any]) -> Dict[str, Any]:
        scopes = _listify(scopes_payload.get("scopes"))
        request_exports = self._build_request_exports(scopes_payload)
        request_stats = request_exports.get("stats") if isinstance(request_exports.get("stats"), dict) else {}
        focus_counts: Counter = Counter()
        target_counts: Counter = Counter()
        missing_counts: Counter = Counter()
        repair_failure_modes: Counter = Counter()
        tfi_failure_diagnosis_counts: Counter = Counter()
        legacy_tfi_failure_diagnosis_counts: Counter = Counter()

        complete_scope_count = 0
        partial_scope_count = 0
        total_score = 0.0
        prefix_success_count = 0
        sink_failure_count = 0
        aligned_count = 0
        repair_history_count = 0
        repair_aligned_count = 0
        repair_sink_reverted_count = 0
        repair_target_seen_no_progress_count = 0
        repair_negative_count = 0
        repair_positive_count = 0
        scopes_should_generate_action_atom_now = 0
        scopes_with_binding_repair_context = 0
        scopes_with_scope_restriction_context = 0
        scopes_with_nonlocal_binding_skeleton = 0
        binding_gap_downgraded_due_to_nonlocal_skeleton = 0
        scopes_with_unobserved_matched_prefix = 0
        scopes_where_matched_prefix_contains_unreached_sink = 0
        diagnosis_not_scope_local_count = 0

        for scope in scopes:
            if not isinstance(scope, dict):
                continue
            focus_type = _text(scope.get("focus_type")).strip() or "unknown"
            target_function = _text(scope.get("target_function")).strip() or "unknown"
            focus_counts[focus_type] += 1
            target_counts[target_function] += 1

            completeness = scope.get("context_completeness") if isinstance(scope.get("context_completeness"), dict) else {}
            score = _to_float(completeness.get("completeness_score"))
            total_score += score
            if score >= 0.75:
                complete_scope_count += 1
            else:
                partial_scope_count += 1

            if bool(completeness.get("has_prefix_success_sample")):
                prefix_success_count += 1
            if bool(completeness.get("has_sink_failure_sample")):
                sink_failure_count += 1
            if bool(completeness.get("has_repair_candidates")):
                aligned_count += 1
            for field in _clean_list(completeness.get("missing_context_fields")):
                missing_counts[field] += 1

            repair_context = scope.get("repair_context") if isinstance(scope.get("repair_context"), dict) else {}
            if bool(repair_context.get("has_repair_history")):
                repair_history_count += 1
            dominant_mode = _text(repair_context.get("dominant_failure_mode")).strip() or "unknown"
            repair_failure_modes[dominant_mode] += 1

            repair_alignment = scope.get("repair_alignment_scope") if isinstance(scope.get("repair_alignment_scope"), dict) else {}
            if _listify(repair_alignment.get("aligned_guided_sequences")):
                repair_aligned_count += 1

            repair_negative = scope.get("repair_negative_scope") if isinstance(scope.get("repair_negative_scope"), dict) else {}
            if _listify(repair_negative.get("sink_reached_but_reverted")):
                repair_sink_reverted_count += 1
            if _listify(repair_negative.get("target_seen_but_no_path_progress")):
                repair_target_seen_no_progress_count += 1
            if any(_listify(repair_negative.get(key)) for key in (
                "failed_sender_bindings",
                "failed_value_bindings",
                "failed_argument_bindings",
                "failed_sequence_orders",
                "failed_prestate_assumptions",
            )):
                repair_negative_count += 1

            repair_positive = scope.get("repair_positive_scope") if isinstance(scope.get("repair_positive_scope"), dict) else {}
            if any(_listify(repair_positive.get(key)) for key in (
                "prefix_improving_repairs",
                "target_seen_repairs",
                "sink_reaching_repairs",
                "non_reverting_prefix_repairs",
            )):
                repair_positive_count += 1

            diagnosis = scope.get("tfi_failure_diagnosis") if isinstance(scope.get("tfi_failure_diagnosis"), dict) else {}
            diagnosis_type = _text(diagnosis.get("diagnosis_type")).strip() or "unknown"
            tfi_failure_diagnosis_counts[diagnosis_type] += 1
            legacy_diagnosis_type = _text(diagnosis.get("legacy_diagnosis_type")).strip()
            if legacy_diagnosis_type:
                legacy_tfi_failure_diagnosis_counts[legacy_diagnosis_type] += 1
            if bool(diagnosis.get("should_generate_action_atom_now")):
                scopes_should_generate_action_atom_now += 1

            if isinstance(scope.get("binding_repair_context"), dict):
                scopes_with_binding_repair_context += 1
            if isinstance(scope.get("scope_restriction_context"), dict):
                scopes_with_scope_restriction_context += 1

            scope_local_check = scope.get("scope_local_evidence_check") if isinstance(scope.get("scope_local_evidence_check"), dict) else {}
            if not bool(scope_local_check.get("binding_skeleton_is_scope_local")):
                scopes_with_nonlocal_binding_skeleton += 1
            if not bool(scope_local_check.get("matched_prefix_is_observed")):
                scopes_with_unobserved_matched_prefix += 1
            observed_prefix = _clean_list(scope_local_check.get("observed_matched_prefix"))
            sink_name = _text(scope.get("sink")).strip()
            sink_is_local = bool(scope_local_check.get("sink_reached_is_scope_local"))
            if observed_prefix and sink_name and sink_name in observed_prefix and not sink_is_local:
                scopes_where_matched_prefix_contains_unreached_sink += 1
            if not bool(scope_local_check.get("diagnosis_uses_scope_local_evidence")):
                diagnosis_not_scope_local_count += 1
            if _text(diagnosis.get("diagnosis_downgrade_reason")).strip() == "binding_skeleton_not_scope_local":
                binding_gap_downgraded_due_to_nonlocal_skeleton += 1

        return {
            "enabled": True,
            "design_role": "tfi_stage3_target_local_focusing_with_repair_local_context",
            "scope_count": len(scopes),
            "focus_type_counts": dict(focus_counts),
            "target_function_counts": dict(target_counts.most_common(10)),
            "complete_scope_count": complete_scope_count,
            "partial_scope_count": partial_scope_count,
            "avg_completeness_score": round(total_score / float(len(scopes) or 1), 4),
            "scopes_with_prefix_success_sample": prefix_success_count,
            "scopes_with_sink_failure_sample": sink_failure_count,
            "scopes_with_aligned_repair_candidates": aligned_count,
            "top_missing_context_fields": dict(missing_counts.most_common(10)),
            "scopes_with_repair_history": repair_history_count,
            "scopes_with_aligned_guided_sequences": repair_aligned_count,
            "scopes_with_sink_reached_but_reverted": repair_sink_reverted_count,
            "scopes_with_target_seen_but_no_path_progress": repair_target_seen_no_progress_count,
            "scopes_with_repair_negative_scope": repair_negative_count,
            "scopes_with_repair_positive_scope": repair_positive_count,
            "dominant_repair_failure_modes": dict(repair_failure_modes),
            "tfi_failure_diagnosis_counts": {
                "binding_gap": int(tfi_failure_diagnosis_counts.get("binding_gap", 0)),
                "scope_alignment_gap": int(tfi_failure_diagnosis_counts.get("scope_alignment_gap", 0)),
                "possible_local_candidate_gap": int(tfi_failure_diagnosis_counts.get("possible_local_candidate_gap", 0)),
                "possible_missing_action_atom": int(tfi_failure_diagnosis_counts.get("possible_missing_action_atom", 0)),
                "unknown": int(tfi_failure_diagnosis_counts.get("unknown", 0)),
            },
            "legacy_tfi_failure_diagnosis_counts": {
                "possible_missing_action_atom": int(legacy_tfi_failure_diagnosis_counts.get("possible_missing_action_atom", 0)),
            },
            "scopes_should_generate_action_atom_now": int(scopes_should_generate_action_atom_now),
            "scopes_with_binding_repair_context": int(scopes_with_binding_repair_context),
            "scopes_with_scope_restriction_context": int(scopes_with_scope_restriction_context),
            "scope_local_evidence_audit": {
                "scopes_with_nonlocal_binding_skeleton": int(scopes_with_nonlocal_binding_skeleton),
                "binding_gap_downgraded_due_to_nonlocal_skeleton": int(binding_gap_downgraded_due_to_nonlocal_skeleton),
                "scopes_with_unobserved_matched_prefix": int(scopes_with_unobserved_matched_prefix),
                "scopes_where_matched_prefix_contains_unreached_sink": int(scopes_where_matched_prefix_contains_unreached_sink),
                "diagnosis_not_scope_local_count": int(diagnosis_not_scope_local_count),
            },
            "binding_repair_request_count": int(request_stats.get("binding_repair_request_count", 0)),
            "scoped_selection_request_count": int(request_stats.get("scoped_selection_request_count", 0)),
            "binding_requests_by_confidence": request_stats.get("binding_requests_by_confidence", {"high": 0, "medium": 0, "low": 0}),
            "selection_requests_by_confidence": request_stats.get("selection_requests_by_confidence", {"high": 0, "medium": 0, "low": 0}),
            "requests_skipped_due_to_nonlocal_evidence": int(request_stats.get("requests_skipped_due_to_nonlocal_evidence", 0)),
            "requests_skipped_due_to_generic_skeleton_leakage": int(request_stats.get("requests_skipped_due_to_generic_skeleton_leakage", 0)),
            "generic_skeleton_leakage_scope_count": int(request_stats.get("generic_skeleton_leakage_scope_count", 0)),
            "generic_skeleton_positive_evidence_blocked_count": int(request_stats.get("generic_skeleton_positive_evidence_blocked_count", 0)),
            "generality_audit": self._build_generality_audit(scopes_payload, request_exports),
            "loaded_inputs": scopes_payload.get("loaded_inputs") if isinstance(scopes_payload.get("loaded_inputs"), dict) else {},
            "output_path": os.path.join(self.output_dir, "target_focus_candidate_scope.json"),
        }

    def _build_generality_audit(
        self,
        scopes_payload: Dict[str, Any],
        request_exports: Dict[str, Any],
    ) -> Dict[str, Any]:
        restricted_terms = [
            "APEBOYS",
            "DoKENV2",
            "setswap",
            "setFeR",
            "setWME",
            "setstaT",
            "setTargetLiquidity",
            "setFeeReceivers",
            "setIsTxLimitExempt",
            "approve_transfer",
            "approve_transferFrom",
            "preserve_approve_transfer",
        ]
        strategy_fields = (
            "repair_goal",
            "selection_goal",
            "reason",
            "recommended_binding_strategy",
            "recommended_repair_direction",
            "recommended_scope_policy",
        )

        binding_payload = request_exports.get("binding_payload") if isinstance(request_exports.get("binding_payload"), dict) else {}
        selection_payload = request_exports.get("selection_payload") if isinstance(request_exports.get("selection_payload"), dict) else {}
        contract_specific_terms_found = []
        function_specific_examples = []

        def _scan_text(value: Any) -> None:
            text = _text(value)
            if not text:
                return
            for term in restricted_terms:
                if term in text and term not in contract_specific_terms_found:
                    contract_specific_terms_found.append(term)

        def _scan_record(record: Dict[str, Any]) -> None:
            for field in strategy_fields:
                _scan_text(record.get(field))
            binding_slots = record.get("binding_slots") if isinstance(record.get("binding_slots"), dict) else {}
            for slot in binding_slots.values():
                if isinstance(slot, dict):
                    for key in ("candidate_policies", "avoid_policies", "argument_templates"):
                        for item in _clean_list(slot.get(key)):
                            _scan_text(item)
                            lowered = item.lower()
                            if any(token in lowered for token in ("approve", "transferfrom", "router", "update")):
                                function_specific_examples.append(item)
            for key in ("allowed_candidate_basis", "forbidden_candidate_patterns"):
                for item in _clean_list(record.get(key)):
                    _scan_text(item)
                    lowered = item.lower()
                    if any(token in lowered for token in ("approve", "transferfrom", "router", "update")):
                        function_specific_examples.append(item)

        for record in _listify(binding_payload.get("requests")):
            if isinstance(record, dict):
                _scan_record(record)
        for record in _listify(selection_payload.get("requests")):
            if isinstance(record, dict):
                _scan_record(record)

        for scope in _listify(scopes_payload.get("scopes")):
            if not isinstance(scope, dict):
                continue
            for section_name in ("binding_repair_context", "scope_restriction_context", "repair_positive_scope", "tfi_failure_diagnosis"):
                section = scope.get(section_name)
                if not isinstance(section, dict):
                    continue
                for field in strategy_fields:
                    _scan_text(section.get(field))
                for key in ("candidate_repair_templates", "argument_templates", "forbidden_candidate_patterns"):
                    for item in _clean_list(section.get(key)):
                        _scan_text(item)
                        lowered = item.lower()
                        if any(token in lowered for token in ("approve", "transferfrom", "router", "update")):
                            function_specific_examples.append(item)

        function_specific_examples = _unique_list(function_specific_examples)[:10]
        generic_strategy_names_only = not bool(contract_specific_terms_found or function_specific_examples)

        return {
            "contract_specific_rule_count": int(len(contract_specific_terms_found)),
            "contract_specific_terms_found": contract_specific_terms_found,
            "function_specific_strategy_count": int(len(function_specific_examples)),
            "function_specific_strategy_examples": function_specific_examples,
            "generic_strategy_names_only": bool(generic_strategy_names_only),
            "uses_failure_type_dispatch_only": True,
            "uses_scope_fields_only": True,
        }

