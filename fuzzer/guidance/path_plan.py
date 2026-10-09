# -*- coding: utf-8 -*-

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PathHypothesis:
    id: str
    vulnerability_type: str
    path: List[str]
    sink: Optional[str]
    priority: float
    expected_blockers: List[str]
    raw: Dict[str, Any]


@dataclass
class PathPlan:
    hypotheses: List[PathHypothesis] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.hypotheses) == 0

    def to_summary(self) -> Dict[str, Any]:
        return {
            "hypothesis_count": len(self.hypotheses),
            "hypotheses": [
                {
                    "id": hypothesis.id,
                    "vulnerability_type": hypothesis.vulnerability_type,
                    "path": list(hypothesis.path),
                    "sink": hypothesis.sink,
                    "priority": hypothesis.priority,
                    "expected_blockers": list(hypothesis.expected_blockers),
                }
                for hypothesis in self.hypotheses
            ],
        }


def _normalize_path(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_blockers(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    blockers = []
    for item in value:
        if item is None:
            continue
        blockers.append(str(item))
    return blockers


def _normalize_priority(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.5


def load_path_plan(path: Optional[str]) -> PathPlan:
    if not path:
        return PathPlan()
    if not os.path.exists(path):
        return PathPlan()

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return PathPlan()

    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list):
        return PathPlan()

    normalized_hypotheses = []
    for index, item in enumerate(hypotheses, start=1):
        if not isinstance(item, dict):
            continue
        hypothesis_path = _normalize_path(item.get("path"))
        sink = item.get("sink")
        if sink in (None, ""):
            sink = hypothesis_path[-1] if hypothesis_path else None
        else:
            sink = str(sink)
        normalized_hypotheses.append(
            PathHypothesis(
                id=str(item.get("id") or "H{}".format(index)),
                vulnerability_type=str(item.get("vulnerability_type") or ""),
                path=hypothesis_path,
                sink=sink,
                priority=_normalize_priority(item.get("priority", 0.5)),
                expected_blockers=_normalize_blockers(item.get("expected_blockers", [])),
                raw=dict(item),
            )
        )

    return PathPlan(hypotheses=normalized_hypotheses)
