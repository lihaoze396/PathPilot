# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from fuzzer.agent.llm_provider import LLMProvider, LLMProviderError


class AgentPlanner:
    def __init__(
        self,
        output_dir: str,
        blocker_evidence_path: Optional[str] = None,
        tfi_focus_path: Optional[str] = None,
        side_path_shortlist_path: Optional[str] = None,
        proposed_path_plan_path: Optional[str] = None,
        max_hypotheses: int = 20,
        enabled: bool = False,
        planner_mode: str = "deterministic",
        llm_provider: Optional[LLMProvider] = None,
        fallback_to_deterministic: bool = True,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.enabled = bool(enabled and self.output_dir)
        self.max_hypotheses = max(0, int(max_hypotheses or 0))
        requested_mode = str(planner_mode or "deterministic").strip().lower()
        self.planner_mode = requested_mode if requested_mode in ("deterministic", "llm") else "deterministic"
        self.llm_provider = llm_provider
        self.fallback_to_deterministic = bool(fallback_to_deterministic)

        self._requested_blocker_evidence_path = (
            os.path.abspath(blocker_evidence_path) if blocker_evidence_path else None
        )
        self._requested_tfi_focus_path = os.path.abspath(tfi_focus_path) if tfi_focus_path else None
        self._requested_side_path_shortlist_path = (
            os.path.abspath(side_path_shortlist_path) if side_path_shortlist_path else None
        )
        self._requested_proposed_path_plan_path = (
            os.path.abspath(proposed_path_plan_path) if proposed_path_plan_path else None
        )

        self.default_blocker_evidence_path = (
            os.path.join(self.output_dir, "blocker_evidence.json") if self.output_dir else ""
        )
        self.default_tfi_focus_path = (
            os.path.join(self.output_dir, "tfi_focus_targets.jsonl") if self.output_dir else ""
        )
        self.default_side_path_shortlist_path = (
            os.path.join(self.output_dir, "side_path_shortlist.json") if self.output_dir else ""
        )
        self.default_proposed_path_plan_path = (
            os.path.join(self.output_dir, "proposed_path_plan.json") if self.output_dir else ""
        )

        self.agent_path_plan_path = (
            os.path.join(self.output_dir, "agent_path_plan.json") if self.output_dir else ""
        )
        self.summary_path = os.path.join(self.output_dir, "agent_plan_summary.json") if self.output_dir else ""

        self.loaded_inputs: Dict[str, bool] = {
            "blocker_evidence": False,
            "tfi_focus": False,
            "side_path_shortlist": False,
            "proposed_path_plan": False,
        }
        self.input_paths: Dict[str, Optional[str]] = {
            "blocker_evidence": self._requested_blocker_evidence_path or self.default_blocker_evidence_path or None,
            "tfi_focus": self._requested_tfi_focus_path or self.default_tfi_focus_path or None,
            "side_path_shortlist": self._requested_side_path_shortlist_path or self.default_side_path_shortlist_path or None,
            "proposed_path_plan": self._requested_proposed_path_plan_path or self.default_proposed_path_plan_path or None,
        }
        self.source_counts: Dict[str, int] = {
            "tfi_focus": 0,
            "blocker_evidence": 0,
            "side_path_shortlist": 0,
        }
        self._last_summary: Dict[str, Any] = self._default_summary()

    def build_plan(self) -> Dict[str, Any]:
        inputs = self._load_inputs()
        if self.planner_mode == "llm":
            return self._build_plan_with_llm(inputs)
        return self._build_plan_deterministic(inputs)

    def write_plan(self, plan: Dict[str, Any], filename: str = "agent_path_plan.json") -> str:
        if not self.output_dir:
            return ""
        output_path = os.path.join(self.output_dir, filename)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(plan, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        return output_path

    def write_summary(self, summary: Dict[str, Any], filename: str = "agent_plan_summary.json") -> str:
        if not self.output_dir:
            return ""
        output_path = os.path.join(self.output_dir, filename)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        return output_path

    def run(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._default_summary()

        summary = self._default_summary()
        summary["enabled"] = True
        summary["planner_mode"] = self.planner_mode
        summary["llm_available"] = bool(self.llm_provider.is_available()) if self.llm_provider else False
        summary["llm_configured"] = bool(self.llm_provider.is_configured()) if self.llm_provider else False
        summary["model"] = self._provider_model()
        summary["base_url"] = self._provider_base_url()
        summary["llm_api_style"] = self._provider_api_style()

        try:
            plan = self.build_plan()
            output_path = self.write_plan(plan, filename="agent_path_plan.json")
            summary.update(self._last_summary)
            summary["enabled"] = True
            summary["output_path"] = output_path if output_path else self.agent_path_plan_path
            summary["summary_path"] = self.summary_path
            summary["input_paths"] = dict(self.input_paths)
            summary_path = self.write_summary(summary, filename="agent_plan_summary.json")
            summary["summary_path"] = summary_path if summary_path else self.summary_path
            return summary
        except Exception as exc:
            fallback_summary = dict(summary)
            fallback_summary.update(
                {
                    "enabled": True,
                    "agent_type": "deterministic_agent_planner_v1" if self.planner_mode == "llm" else "deterministic_agent_planner_v1",
                    "output_path": self.agent_path_plan_path if self.agent_path_plan_path else None,
                    "summary_path": self.summary_path,
                    "hypothesis_count": 0,
                    "source_counts": {
                        "tfi_focus": 0,
                        "blocker_evidence": 0,
                        "side_path_shortlist": 0,
                    },
                    "loaded_inputs": dict(self.loaded_inputs),
                    "input_paths": dict(self.input_paths),
                    "llm_used": False,
                    "fallback_used": bool(self.planner_mode == "llm"),
                    "fallback_reason": str(exc),
                    "llm_api_style": self._provider_api_style(),
                }
            )
            summary_path = self.write_summary(fallback_summary, filename="agent_plan_summary.json")
            fallback_summary["summary_path"] = summary_path if summary_path else self.summary_path
            return fallback_summary

    def _load_inputs(self) -> Dict[str, Any]:
        blocker_payload = self._load_json_with_fallback(
            self._requested_blocker_evidence_path,
            self.default_blocker_evidence_path,
            "blocker_evidence",
        )
        tfi_records = self._load_jsonl_with_fallback(
            self._requested_tfi_focus_path,
            self.default_tfi_focus_path,
            "tfi_focus",
        )
        shortlist_payload = self._load_json_with_fallback(
            self._requested_side_path_shortlist_path,
            self.default_side_path_shortlist_path,
            "side_path_shortlist",
        )
        proposed_payload = self._load_json_with_fallback(
            self._requested_proposed_path_plan_path,
            self.default_proposed_path_plan_path,
            "proposed_path_plan",
        )
        return {
            "blocker_payload": blocker_payload,
            "tfi_records": tfi_records,
            "shortlist_payload": shortlist_payload,
            "proposed_payload": proposed_payload,
        }

    def _build_plan_deterministic(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        blocker_payload = inputs.get("blocker_payload", {})
        tfi_records = inputs.get("tfi_records", [])
        shortlist_payload = inputs.get("shortlist_payload", {})
        proposed_payload = inputs.get("proposed_payload", {})

        blocker_hypotheses = self._index_blocker_hypotheses(blocker_payload)
        proposed_hypotheses = self._index_proposed_hypotheses(proposed_payload)

        selected: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        order_seen = 0
        self.source_counts = {
            "tfi_focus": 0,
            "blocker_evidence": 0,
            "side_path_shortlist": 0,
        }

        for record in tfi_records:
            if str(record.get("priority") or "").lower() != "high":
                continue
            candidate = self._build_from_tfi(record, blocker_hypotheses, proposed_hypotheses)
            if candidate:
                inserted = self._upsert_candidate(selected, candidate, rank=1, order=order_seen)
                order_seen += 1
                if inserted:
                    self.source_counts["tfi_focus"] += 1

        for record in blocker_hypotheses.values():
            candidate = self._build_from_blocker(record)
            if candidate:
                inserted = self._upsert_candidate(selected, candidate, rank=2, order=order_seen)
                order_seen += 1
                if inserted:
                    self.source_counts["blocker_evidence"] += 1

        shortlist = shortlist_payload.get("shortlist") if isinstance(shortlist_payload, dict) else None
        if isinstance(shortlist, list):
            for record in shortlist:
                candidate = self._build_from_shortlist(record)
                if candidate:
                    inserted = self._upsert_candidate(selected, candidate, rank=3, order=order_seen)
                    order_seen += 1
                    if inserted:
                        self.source_counts["side_path_shortlist"] += 1

        hypotheses = sorted(
            selected.values(),
            key=lambda item: (
                -float(item.get("priority") or 0.0),
                int(item.get("_source_rank", 99)),
                int(item.get("_insert_order", 10**9)),
            ),
        )
        hypotheses = hypotheses[: self.max_hypotheses]

        self.source_counts = {
            "tfi_focus": 0,
            "blocker_evidence": 0,
            "side_path_shortlist": 0,
        }
        for item in hypotheses:
            rank = int(item.get("_source_rank", 99))
            if rank == 1:
                self.source_counts["tfi_focus"] += 1
            elif rank == 2:
                self.source_counts["blocker_evidence"] += 1
            elif rank == 3:
                self.source_counts["side_path_shortlist"] += 1

        normalized = []
        for index, item in enumerate(hypotheses, start=1):
            normalized.append(self._finalize_deterministic_hypothesis(item, index))

        plan = {"hypotheses": normalized}
        self._last_summary = {
            "agent_type": "deterministic_agent_planner_v1",
            "planner_mode": self.planner_mode,
            "llm_available": bool(self.llm_provider.is_available()) if self.llm_provider else False,
            "llm_configured": bool(self.llm_provider.is_configured()) if self.llm_provider else False,
            "llm_used": False,
            "fallback_used": False,
            "fallback_reason": None,
            "model": self._provider_model(),
            "base_url": self._provider_base_url(),
            "llm_api_style": self._provider_api_style(),
            "hypothesis_count": len(normalized),
            "source_counts": dict(self.source_counts),
            "loaded_inputs": dict(self.loaded_inputs),
        }
        return plan

    def _build_plan_with_llm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        llm_available = bool(self.llm_provider.is_available()) if self.llm_provider else False
        llm_configured = bool(self.llm_provider.is_configured()) if self.llm_provider else False

        if not self.llm_provider or not llm_available:
            return self._fallback_or_raise(inputs, "llm_provider_unavailable", llm_available, llm_configured)
        if not llm_configured:
            return self._fallback_or_raise(inputs, "missing_api_key", llm_available, llm_configured)

        prompt = self._build_llm_prompt(inputs)
        try:
            raw_plan = self.llm_provider.generate_json(prompt)
        except LLMProviderError as exc:
            return self._fallback_or_raise(inputs, str(exc), llm_available, llm_configured)

        normalized_plan = self._normalize_llm_plan(raw_plan)
        hypotheses = normalized_plan.get("hypotheses", [])
        if not hypotheses:
            return self._fallback_or_raise(inputs, "empty_llm_hypotheses", llm_available, llm_configured)

        self.source_counts = {"llm_agent": len(hypotheses)}
        self._last_summary = {
            "agent_type": "llm_agent_planner_v1",
            "planner_mode": "llm",
            "llm_available": llm_available,
            "llm_configured": llm_configured,
            "llm_used": True,
            "fallback_used": False,
            "fallback_reason": None,
            "model": self._provider_model(),
            "base_url": self._provider_base_url(),
            "llm_api_style": self._provider_api_style(),
            "hypothesis_count": len(hypotheses),
            "source_counts": {"llm_agent": len(hypotheses)},
            "loaded_inputs": dict(self.loaded_inputs),
        }
        return normalized_plan

    def _fallback_or_raise(
        self,
        inputs: Dict[str, Any],
        reason: str,
        llm_available: bool,
        llm_configured: bool,
    ) -> Dict[str, Any]:
        if not self.fallback_to_deterministic:
            raise RuntimeError(reason)

        original_mode = self.planner_mode
        self.planner_mode = "deterministic"
        plan = self._build_plan_deterministic(inputs)
        self.planner_mode = original_mode
        self._last_summary.update(
            {
                "planner_mode": "llm",
                "llm_available": llm_available,
                "llm_configured": llm_configured,
                "llm_used": False,
                "fallback_used": True,
                "fallback_reason": reason,
                "model": self._provider_model(),
                "base_url": self._provider_base_url(),
                "llm_api_style": self._provider_api_style(),
            }
        )
        return plan

    def _build_llm_prompt(self, inputs: Dict[str, Any]) -> str:
        blocker_payload = inputs.get("blocker_payload", {})
        tfi_records = list(inputs.get("tfi_records", []) or [])
        shortlist_payload = inputs.get("shortlist_payload", {})
        proposed_payload = inputs.get("proposed_payload", {})

        blocker_hypotheses = blocker_payload.get("hypotheses") if isinstance(blocker_payload, dict) else []
        shortlist = shortlist_payload.get("shortlist") if isinstance(shortlist_payload, dict) else []
        proposed_hypotheses = proposed_payload.get("hypotheses") if isinstance(proposed_payload, dict) else []

        sorted_tfi = sorted(
            [item for item in tfi_records if isinstance(item, dict)],
            key=lambda item: (
                0 if str(item.get("priority") or "").lower() == "high" else 1,
                str(item.get("hypothesis_id") or ""),
            ),
        )

        compact_blockers = []
        observed_functions: Set[str] = set()
        for item in blocker_hypotheses[:20] if isinstance(blocker_hypotheses, list) else []:
            if not isinstance(item, dict):
                continue
            path = self._normalize_path(item.get("path"))
            observed_functions.update(path)
            compact_blockers.append(
                {
                    "hypothesis_id": self._text(item.get("hypothesis_id")),
                    "path": path,
                    "sink": self._text(item.get("sink")) or (path[-1] if path else ""),
                    "repair_objective": item.get("repair_objective") if isinstance(item.get("repair_objective"), dict) else {},
                    "progress_summary": item.get("progress_summary") if isinstance(item.get("progress_summary"), dict) else {},
                    "top_suspected_blockers": self._normalize_blockers(
                        (item.get("runtime_evidence") or {}).get("top_suspected_blockers")
                        if isinstance(item.get("runtime_evidence"), dict)
                        else []
                    ),
                }
            )

        compact_tfi = []
        for item in sorted_tfi[:30]:
            path = self._normalize_path(item.get("path"))
            observed_functions.update(path)
            focus_function = self._text(item.get("focus_function"))
            if focus_function:
                observed_functions.add(focus_function)
            compact_tfi.append(
                {
                    "hypothesis_id": self._text(item.get("hypothesis_id")),
                    "focus_type": self._text(item.get("focus_type")),
                    "priority": self._text(item.get("priority")),
                    "focus_function": focus_function,
                    "focus_frontier": self._text(item.get("focus_frontier")),
                    "path": path,
                    "reason": self._text(item.get("reason")),
                }
            )

        compact_shortlist = []
        for item in shortlist[:20] if isinstance(shortlist, list) else []:
            if not isinstance(item, dict):
                continue
            path = self._normalize_path(item.get("suggested_path"))
            observed_functions.update(path)
            compact_shortlist.append(
                {
                    "hypothesis_id": self._text(item.get("hypothesis_id")),
                    "suggested_path": path,
                    "sink": self._text(item.get("sink")) or (path[-1] if path else ""),
                    "tier": self._text(item.get("tier")),
                    "priority": item.get("priority"),
                    "reason": self._text(item.get("reason")),
                }
            )

        compact_proposed = []
        for item in proposed_hypotheses[:20] if isinstance(proposed_hypotheses, list) else []:
            if not isinstance(item, dict):
                continue
            path = self._normalize_path(item.get("path"))
            observed_functions.update(path)
            compact_proposed.append(
                {
                    "id": self._text(item.get("id")),
                    "path": path,
                    "sink": self._text(item.get("sink")) or (path[-1] if path else ""),
                    "priority": item.get("priority"),
                    "expected_blockers": self._normalize_blockers(item.get("expected_blockers")),
                }
            )

        prompt_payload = {
            "task": "Generate the next fuzzing path plan from runtime evidence.",
            "max_hypotheses": self.max_hypotheses,
            "observed_functions": sorted(observed_functions),
            "blocker_evidence": compact_blockers,
            "tfi_focus_targets": compact_tfi,
            "side_path_shortlist": compact_shortlist,
            "proposed_path_plan": compact_proposed,
        }

        instructions = [
            "You are an intelligent smart contract testing path planning agent.",
            "Your job is not to confirm whether a vulnerability is real.",
            "Your job is to generate the next fuzzing path_plan from runtime evidence.",
            "Output strict JSON only.",
            "Use this schema exactly:",
            '{ "hypotheses": [ { "id": "AGENT_H1", "vulnerability_type": "agent_refined_path", "path": ["funcA","funcB"], "sink": "funcB", "priority": 0.9, "expected_blockers": ["revert_at_sink"], "source": "llm_agent_planner_v1", "origin_hypothesis_id": "GEI_H1", "repair_action": "repair_sink_execution", "focus_type": "sink_reverted", "reason": "..." } ] }',
            "Constraints:",
            "1. Output at most max_hypotheses items.",
            "2. path must be a non-empty list of function names.",
            "3. sink should default to the last function in path.",
            "4. priority must be between 0.0 and 1.0.",
            "5. Prioritize TFI high priority sink_reverted, repeated_frontier_block, caller_or_permission_condition.",
            "6. Prefer shorter and more executable paths.",
            "7. Only use function names that appear in observed_functions or the evidence payload.",
            "8. Do not invent new function names.",
            "9. Do not output markdown.",
            "10. Do not output explanatory prose.",
            "11. Output JSON only.",
            "",
            "Input evidence JSON:",
            json.dumps(prompt_payload, ensure_ascii=True, indent=2),
        ]
        return "\n".join(instructions)

    def _normalize_llm_plan(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise LLMProviderError("llm_plan_not_dict")

        items = payload.get("hypotheses")
        if not isinstance(items, list):
            raise LLMProviderError("llm_plan_missing_hypotheses")

        selected: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            path = self._normalize_path(item.get("path"))
            if not path:
                continue
            signature = tuple(path)
            sink = self._text(item.get("sink")) or path[-1]
            try:
                priority = float(item.get("priority", 0.75))
            except Exception:
                priority = 0.75
            priority = max(0.0, min(priority, 1.0))

            normalized = {
                "vulnerability_type": self._text(item.get("vulnerability_type")) or "agent_refined_path",
                "path": path,
                "sink": sink,
                "priority": priority,
                "expected_blockers": self._normalize_blockers(item.get("expected_blockers")),
                "source": "llm_agent_planner_v1",
                "origin_hypothesis_id": self._text(item.get("origin_hypothesis_id")) or None,
                "repair_action": self._text(item.get("repair_action")) or None,
                "focus_type": self._text(item.get("focus_type")) or None,
                "focus_function": self._text(item.get("focus_function")) or None,
                "reason": self._text(item.get("reason")) or None,
            }

            existing = selected.get(signature)
            if existing is None or priority > float(existing.get("priority") or 0.0):
                selected[signature] = normalized

        hypotheses = sorted(
            selected.values(),
            key=lambda item: (-float(item.get("priority") or 0.0), len(item.get("path") or [])),
        )[: self.max_hypotheses]

        output = []
        for index, item in enumerate(hypotheses, start=1):
            hypothesis = {
                "id": "AGENT_H{}".format(index),
                "vulnerability_type": item.get("vulnerability_type") or "agent_refined_path",
                "path": item.get("path") or [],
                "sink": item.get("sink") or ((item.get("path") or [None])[-1]),
                "priority": float(item.get("priority") or 0.75),
                "expected_blockers": self._normalize_blockers(item.get("expected_blockers")),
                "source": "llm_agent_planner_v1",
            }
            for key in (
                "origin_hypothesis_id",
                "repair_action",
                "focus_type",
                "focus_function",
                "reason",
            ):
                value = item.get(key)
                if value not in (None, ""):
                    hypothesis[key] = value
            output.append(hypothesis)

        return {"hypotheses": output}

    def _default_summary(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "agent_type": "deterministic_agent_planner_v1",
            "planner_mode": self.planner_mode,
            "output_path": self.agent_path_plan_path if self.agent_path_plan_path else None,
            "summary_path": self.summary_path if self.summary_path else None,
            "hypothesis_count": 0,
            "source_counts": {
                "tfi_focus": 0,
                "blocker_evidence": 0,
                "side_path_shortlist": 0,
            },
            "loaded_inputs": dict(self.loaded_inputs),
            "input_paths": dict(self.input_paths),
            "llm_available": bool(self.llm_provider.is_available()) if self.llm_provider else False,
            "llm_configured": bool(self.llm_provider.is_configured()) if self.llm_provider else False,
            "llm_used": False,
            "fallback_used": False,
            "fallback_reason": None,
            "model": self._provider_model(),
            "base_url": self._provider_base_url(),
            "llm_api_style": self._provider_api_style(),
        }

    def _provider_model(self) -> Optional[str]:
        if self.llm_provider and getattr(self.llm_provider, "config", None):
            return getattr(self.llm_provider.config, "model", None)
        return None

    def _provider_base_url(self) -> Optional[str]:
        if self.llm_provider and getattr(self.llm_provider, "config", None):
            return getattr(self.llm_provider.config, "base_url", None)
        return None

    def _provider_api_style(self) -> Optional[str]:
        if self.llm_provider:
            try:
                return self.llm_provider.resolved_api_style()
            except Exception:
                config = getattr(self.llm_provider, "config", None)
                if config is not None:
                    return getattr(config, "api_style", None)
        return None

    def _load_json_with_fallback(
        self,
        requested_path: Optional[str],
        default_path: Optional[str],
        key: str,
    ) -> Dict[str, Any]:
        candidate_paths = []
        if requested_path:
            candidate_paths.append(requested_path)
        if default_path and default_path not in candidate_paths:
            candidate_paths.append(default_path)
        self.loaded_inputs[key] = False
        self.input_paths[key] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = self._read_json(path)
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
        candidate_paths = []
        if requested_path:
            candidate_paths.append(requested_path)
        if default_path and default_path not in candidate_paths:
            candidate_paths.append(default_path)
        self.loaded_inputs[key] = False
        self.input_paths[key] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = self._read_jsonl(path)
            if payload:
                self.loaded_inputs[key] = True
                self.input_paths[key] = path
                return payload
        return []

    @staticmethod
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

    @staticmethod
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

    def _index_blocker_hypotheses(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
        if not isinstance(hypotheses, list):
            return output
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            hypothesis_id = self._text(item.get("hypothesis_id"))
            if hypothesis_id:
                output[hypothesis_id] = item
        return output

    def _index_proposed_hypotheses(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
        if not isinstance(hypotheses, list):
            return output
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            hypothesis_id = self._text(item.get("id"))
            if hypothesis_id:
                output[hypothesis_id] = item
        return output

    def _build_from_tfi(
        self,
        target: Dict[str, Any],
        blocker_hypotheses: Dict[str, Dict[str, Any]],
        proposed_hypotheses: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(target, dict):
            return None
        origin_id = self._text(target.get("hypothesis_id"))
        path = self._normalize_path(target.get("path"))
        if not path and origin_id:
            blocker_entry = blocker_hypotheses.get(origin_id) or {}
            path = self._normalize_path(blocker_entry.get("path"))
        if not path and origin_id:
            proposed_entry = proposed_hypotheses.get(origin_id) or {}
            path = self._normalize_path(proposed_entry.get("path"))
        if not path:
            return None

        focus_type = self._text(target.get("focus_type"))
        focus_function = self._text(target.get("focus_function"))
        sink = focus_function or self._text(target.get("sink")) or path[-1]
        reason = self._text(target.get("reason")) or "tfi high priority focus target"

        expected_blockers = []
        if focus_type == "sink_reverted":
            expected_blockers = [
                "revert_at_sink",
                "sink_execution_constraint",
                "caller_value_argument_or_prestate",
            ]
        elif focus_type == "caller_or_permission_condition":
            expected_blockers = [
                "permission_or_caller_constraint",
                "sender_condition",
            ]
        elif focus_type == "repeated_frontier_block":
            expected_blockers = [
                "frontier_block",
                "state_or_configuration_dependency",
            ]
        else:
            expected_blockers = ["state_or_configuration_dependency"]

        return {
            "vulnerability_type": "agent_refined_path",
            "path": path,
            "sink": sink,
            "priority": 0.95,
            "expected_blockers": expected_blockers,
            "source": "agent_planner_v1",
            "origin_hypothesis_id": origin_id or None,
            "repair_action": "repair_sink_execution" if focus_type == "sink_reverted" else None,
            "focus_type": focus_type or None,
            "focus_function": focus_function or None,
            "reason": reason,
            "_source_rank": 1,
        }

    def _build_from_blocker(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None
        origin_id = self._text(entry.get("hypothesis_id"))
        path = self._normalize_path(entry.get("path"))
        if not path:
            return None
        repair = entry.get("repair_objective") if isinstance(entry.get("repair_objective"), dict) else {}
        action = self._text(repair.get("action"))
        sink = self._text(entry.get("sink")) or path[-1]
        focus_function = self._text(repair.get("target_frontier")) or sink

        if action == "repair_sink_execution":
            priority = 0.90
            expected_blockers = ["revert_at_sink", "sink_execution_constraint"]
        elif action == "preserve_prefix_and_mutate_frontier":
            priority = 0.85
            expected_blockers = ["frontier_block", "argument_or_state_dependency"]
        elif action == "extend_prefix_towards_frontier":
            priority = 0.75
            expected_blockers = ["path_extension_needed"]
        else:
            priority = 0.60
            expected_blockers = ["initial_prefix_not_reached"]

        return {
            "vulnerability_type": "agent_refined_path",
            "path": path,
            "sink": sink,
            "priority": priority,
            "expected_blockers": expected_blockers,
            "source": "agent_planner_v1",
            "origin_hypothesis_id": origin_id or None,
            "repair_action": action or None,
            "focus_type": None,
            "focus_function": focus_function or None,
            "reason": action or "blocker evidence repair objective",
            "_source_rank": 2,
        }

    def _build_from_shortlist(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(record, dict):
            return None
        tier = self._text(record.get("tier")).lower()
        if tier not in ("high", "medium"):
            return None
        path = self._normalize_path(record.get("suggested_path"))
        if not path:
            return None
        priority = 0.85 if tier == "high" else 0.75
        return {
            "vulnerability_type": "gei_side_path_candidate",
            "path": path,
            "sink": path[-1],
            "priority": priority,
            "expected_blockers": [
                "state_or_configuration_dependency",
                "side_path_validation_needed",
            ],
            "source": "agent_planner_v1",
            "origin_hypothesis_id": None,
            "repair_action": None,
            "focus_type": None,
            "focus_function": None,
            "reason": self._text(record.get("reason")) or "side path shortlist candidate",
            "_source_rank": 3,
        }

    def _upsert_candidate(
        self,
        selected: Dict[Tuple[str, ...], Dict[str, Any]],
        candidate: Dict[str, Any],
        rank: int,
        order: int,
    ) -> bool:
        path = self._normalize_path(candidate.get("path"))
        if not path:
            return False
        signature = tuple(path)
        existing = selected.get(signature)
        payload = dict(candidate)
        payload["path"] = path
        payload["_source_rank"] = int(rank)
        payload["_insert_order"] = int(order)

        if existing is None:
            selected[signature] = payload
            return True

        existing_priority = float(existing.get("priority") or 0.0)
        new_priority = float(payload.get("priority") or 0.0)
        existing_rank = int(existing.get("_source_rank", 99))
        if new_priority > existing_priority:
            selected[signature] = payload
            return True
        if new_priority == existing_priority and rank < existing_rank:
            selected[signature] = payload
            return True
        return False

    def _finalize_deterministic_hypothesis(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        normalized_path = self._normalize_path(item.get("path"))
        output = {
            "id": "AGENT_H{}".format(index),
            "vulnerability_type": self._text(item.get("vulnerability_type")) or "agent_refined_path",
            "path": normalized_path,
            "sink": self._text(item.get("sink")) or (normalized_path[-1] if normalized_path else None),
            "priority": float(item.get("priority") or 0.0),
            "expected_blockers": self._normalize_blockers(item.get("expected_blockers")),
            "source": "agent_planner_v1",
        }
        optional_fields = [
            "origin_hypothesis_id",
            "repair_action",
            "focus_type",
            "focus_function",
            "reason",
        ]
        for key in optional_fields:
            value = item.get(key)
            if value not in (None, ""):
                output[key] = value
        return output

    @staticmethod
    def _normalize_path(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        output = []
        for item in value:
            text = str(item or "").strip()
            if text:
                output.append(text)
        return output

    @staticmethod
    def _normalize_blockers(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        output = []
        seen = set()
        for item in value:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            output.append(text)
        return output

    @staticmethod
    def _text(value: Any) -> str:
        if value in (None, ""):
            return ""
        return str(value)
