# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class SidePathPlanExporter:
    def __init__(
        self,
        output_dir: str,
        shortlist_path: Optional[str] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.shortlist_path = shortlist_path or os.path.join(self.output_dir, "side_path_shortlist.json")

    def export(
        self,
        max_hypotheses: int = 20,
        filename: str = "proposed_path_plan.json",
    ) -> str:
        if not self.output_dir:
            return ""

        output_path = os.path.join(self.output_dir, filename)
        shortlist_payload = self._read_shortlist()
        hypotheses = self._build_hypotheses(shortlist_payload, max_hypotheses=max_hypotheses)
        payload = {"hypotheses": hypotheses}

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        return output_path

    def _read_shortlist(self) -> Dict[str, Any]:
        if not self.shortlist_path or not os.path.exists(self.shortlist_path):
            return {}
        try:
            with open(self.shortlist_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    def _build_hypotheses(
        self,
        shortlist_payload: Dict[str, Any],
        max_hypotheses: int,
    ) -> List[Dict[str, Any]]:
        shortlist = shortlist_payload.get("shortlist")
        if not isinstance(shortlist, list):
            return []

        hypotheses = []
        seen_paths = set()

        for record in shortlist:
            if not isinstance(record, dict):
                continue
            tier = str(record.get("tier") or "low")
            if tier not in ("high", "medium"):
                continue

            path = self._normalize_path(record.get("suggested_path"))
            if not path:
                continue

            signature = tuple(path)
            if signature in seen_paths:
                continue
            seen_paths.add(signature)

            reason = str(record.get("reason") or "")
            quality_score = self._to_int(record.get("quality_score"))
            hypotheses.append(
                {
                    "id": "GEI_H{}".format(len(hypotheses) + 1),
                    "vulnerability_type": "gei_side_path",
                    "path": path,
                    "sink": path[-1],
                    "priority": self._map_priority(quality_score),
                    "expected_blockers": self._build_expected_blockers(reason),
                    "source": "gei_side_path_shortlist",
                    "reason": reason,
                    "quality_score": quality_score,
                    "tier": tier,
                    "example_sequence_id": str(record.get("sequence_id") or ""),
                }
            )
            if len(hypotheses) >= max(0, int(max_hypotheses or 0)):
                break

        return hypotheses

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
    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _map_priority(score: int) -> float:
        if score >= 10:
            return 0.95
        if score >= 8:
            return 0.90
        if score >= 6:
            return 0.85
        if score >= 4:
            return 0.75
        return 0.60

    @staticmethod
    def _build_expected_blockers(reason: str) -> List[str]:
        normalized = str(reason or "").strip().lower()
        blockers = ["state_or_config_dependency"]
        if "fallback" in normalized or "value" in normalized:
            blockers.append("value_condition")
        if "asset" in normalized:
            blockers.append("asset_state_dependency")
        if any(token in normalized for token in ("config", "state", "update")):
            blockers.append("permission_or_configuration_dependency")
        return blockers


def export_side_path_plan(output_dir: str, max_hypotheses: int = 20) -> str:
    return SidePathPlanExporter(output_dir=output_dir).export(max_hypotheses=max_hypotheses)
