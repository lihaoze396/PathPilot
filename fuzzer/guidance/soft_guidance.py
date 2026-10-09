# -*- coding: utf-8 -*-

import json
import math
import os
from typing import Any, Dict, List, Optional

try:
    from engine.plugin_interfaces import OnTheFlyAnalysis
except ImportError:
    from fuzzer.engine.plugin_interfaces import OnTheFlyAnalysis

try:
    from fuzzer.guidance.path_plan import PathPlan
except ImportError:
    from fuzzer.guidance.path_plan import PathPlan


ACTION_PRIORITY = {
    "repair_sink_execution": 0,
    "preserve_prefix_and_mutate_frontier": 1,
    "explore_side_path_candidate": 2,
    "extend_prefix_towards_frontier": 2,
    "search_for_initial_prefix": 3,
}


class SoftPathGuidance:
    def __init__(
        self,
        path_plan,
        output_dir: str,
        enabled: bool = False,
        ratio: float = 0.10,
        max_per_generation: int = 5,
        blocker_evidence_path: Optional[str] = None,
        side_path_shortlist_path: Optional[str] = None,
        side_path_candidates_path: Optional[str] = None,
        enable_side_path_guidance: bool = False,
        dry_run: bool = False,
    ):
        self.path_plan = path_plan or PathPlan()
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.output_path = os.path.join(self.output_dir, "soft_guidance.jsonl") if self.output_dir else ""
        self._requested_blocker_evidence_path = os.path.abspath(blocker_evidence_path) if blocker_evidence_path else None
        self._default_blocker_evidence_path = os.path.join(self.output_dir, "blocker_evidence.json") if self.output_dir else ""
        self.blocker_evidence_path = self._requested_blocker_evidence_path or self._default_blocker_evidence_path or None
        self.blocker_evidence_loaded = False
        self._requested_side_path_shortlist_path = (
            os.path.abspath(side_path_shortlist_path) if side_path_shortlist_path else None
        )
        self._default_side_path_shortlist_path = os.path.join(self.output_dir, "side_path_shortlist.json") if self.output_dir else ""
        self.side_path_shortlist_path = (
            self._requested_side_path_shortlist_path or self._default_side_path_shortlist_path or None
        )
        self._requested_side_path_candidates_path = (
            os.path.abspath(side_path_candidates_path) if side_path_candidates_path else None
        )
        self._default_side_path_candidates_path = os.path.join(self.output_dir, "side_path_candidates.jsonl") if self.output_dir else ""
        self.side_path_candidates_path = (
            self._requested_side_path_candidates_path or self._default_side_path_candidates_path or None
        )
        self.enable_side_path_guidance = bool(enable_side_path_guidance)
        self.side_path_guidance_loaded = False
        self.side_path_guidance_count = 0
        self.enabled = bool(
            enabled
            and self.output_path
            and (
                not self.path_plan.is_empty()
                or self.enable_side_path_guidance
            )
        )
        self.dry_run = bool(dry_run)
        self.ratio = max(0.0, min(float(ratio or 0.0), 0.3))
        self.max_per_generation = max(0, int(max_per_generation or 0))
        self.guidance_pool: List[Dict[str, Any]] = []
        self._cursor = 0
        self.injected_total = 0
        self.skipped_total = 0
        self.planned_total = 0
        self._skip_reason_counts: Dict[str, int] = {}

        if self.enabled:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(self.output_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                self.enabled = False

        if self.enabled:
            self.guidance_pool = self.build_guidance_pool()

    def build_guidance_pool(self) -> List[Dict[str, Any]]:
        blocker_evidence = self._load_blocker_evidence()
        hypotheses = blocker_evidence.get("hypotheses")
        if not isinstance(hypotheses, list):
            hypotheses = []
        evidence_by_id = {
            item.get("hypothesis_id"): item
            for item in hypotheses
            if isinstance(item, dict) and item.get("hypothesis_id")
        }

        pool = []
        for hypothesis in self.path_plan.hypotheses:
            evidence = evidence_by_id.get(hypothesis.id, {})
            repair = evidence.get("repair_objective") if isinstance(evidence.get("repair_objective"), dict) else {}
            progress = evidence.get("progress_summary") if isinstance(evidence.get("progress_summary"), dict) else {}
            action = str(repair.get("action") or "search_for_initial_prefix")
            target_frontier = repair.get("target_frontier") or progress.get("frontier") or hypothesis.sink
            prefix = repair.get("affected_prefix")
            if not isinstance(prefix, list):
                prefix = list(progress.get("best_prefix") or [])
            pool.append(
                {
                    "hypothesis_id": hypothesis.id,
                    "path": [str(item) for item in hypothesis.path],
                    "prefix": [str(item) for item in prefix],
                    "frontier": str(target_frontier) if target_frontier not in (None, "") else None,
                    "sink": str(hypothesis.sink) if hypothesis.sink not in (None, "") else None,
                    "priority": float(hypothesis.priority),
                    "repair_action": action,
                    "hints": [str(item) for item in repair.get("hints", []) if item not in (None, "")],
                    "source": "path_plan_and_blocker_evidence",
                }
            )

        if self.enable_side_path_guidance:
            pool.extend(self._build_side_path_guidance_pool())

        pool = self._dedupe_guidance_pool(pool)
        pool.sort(key=self._pool_sort_key)
        return pool

    def refresh_guidance_pool(self) -> None:
        if not self.enabled or not self.enable_side_path_guidance:
            return
        self.guidance_pool = self.build_guidance_pool()

    def select_for_generation(
        self,
        generation: int,
        normal_population_size: int,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not self.guidance_pool:
            return []
        ratio_limit = int(math.floor(max(0, normal_population_size) * self.ratio))
        if self.ratio > 0 and ratio_limit == 0 and normal_population_size > 0:
            ratio_limit = 1
        limit = min(self.max_per_generation, ratio_limit)
        if limit <= 0:
            return []

        selected = []
        seen = set()
        start = self._cursor
        attempts = 0
        pool_size = len(self.guidance_pool)
        while len(selected) < limit and attempts < pool_size:
            candidate = self.guidance_pool[(start + attempts) % pool_size]
            attempts += 1
            signature = tuple(candidate.get("path") or [])
            if signature in seen:
                continue
            seen.add(signature)
            selected.append(dict(candidate))
        self._cursor = (start + attempts) % pool_size if pool_size else 0
        self.planned_total += len(selected)
        return selected

    def build_individuals(
        self,
        generation: int,
        normal_population_size: int,
        indv_template,
    ) -> List[Any]:
        self.refresh_guidance_pool()
        selected = self.select_for_generation(generation, normal_population_size)
        if not selected:
            self._append_log(
                {
                    "generation": generation,
                    "dry_run": self.dry_run,
                    "blocker_evidence_path": self.blocker_evidence_path,
                    "blocker_evidence_loaded": self.blocker_evidence_loaded,
                    "side_path_shortlist_path": self.side_path_shortlist_path,
                    "side_path_candidates_path": self.side_path_candidates_path,
                    "side_path_guidance_loaded": self.side_path_guidance_loaded,
                    "side_path_guidance_count": self.side_path_guidance_count,
                    "selected_count": 0,
                    "injected_count": 0,
                    "ratio": self.ratio,
                    "source": "path_plan_blocker_and_side_path_evidence",
                    "candidates": [],
                }
            )
            return []

        records = []
        individuals = []
        for candidate in selected:
            indv_or_none, skip_reason = self._candidate_to_individual(candidate, indv_template)
            record = {
                "hypothesis_id": candidate.get("hypothesis_id"),
                "path": list(candidate.get("path") or []),
                "frontier": candidate.get("frontier"),
                "repair_action": candidate.get("repair_action"),
                "priority": candidate.get("priority"),
                "injected": False,
            }
            if skip_reason:
                record["skip_reason"] = skip_reason
                self.skipped_total += 1
                self._skip_reason_counts[skip_reason] = self._skip_reason_counts.get(skip_reason, 0) + 1
            elif indv_or_none is not None:
                record["skip_reason"] = None
                if not self.dry_run:
                    self._mark_guided_individual(indv_or_none, candidate, generation)
                    individuals.append(indv_or_none)
                    self.injected_total += 1
                    record["injected"] = True
            records.append(record)

        injected_count = sum(1 for record in records if record.get("injected"))
        self._append_log(
            {
                "generation": generation,
                "dry_run": self.dry_run,
                "blocker_evidence_path": self.blocker_evidence_path,
                "blocker_evidence_loaded": self.blocker_evidence_loaded,
                "side_path_shortlist_path": self.side_path_shortlist_path,
                "side_path_candidates_path": self.side_path_candidates_path,
                "side_path_guidance_loaded": self.side_path_guidance_loaded,
                "side_path_guidance_count": self.side_path_guidance_count,
                "selected_count": len(records),
                "injected_count": injected_count,
                "ratio": self.ratio,
                "source": "path_plan_blocker_and_side_path_evidence",
                "candidates": records,
            }
        )
        return [] if self.dry_run else individuals

    def get_summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "dry_run": bool(self.dry_run),
            "ratio": float(self.ratio),
            "max_per_generation": int(self.max_per_generation),
            "guidance_pool_size": len(self.guidance_pool),
            "injected_total": int(self.injected_total),
            "skipped_total": int(self.skipped_total),
            "planned_total": int(self.planned_total),
            "blocker_evidence_path": self.blocker_evidence_path,
            "blocker_evidence_loaded": bool(self.blocker_evidence_loaded),
            "side_path_guidance_enabled": bool(self.enable_side_path_guidance),
            "side_path_guidance_loaded": bool(self.side_path_guidance_loaded),
            "side_path_guidance_count": int(self.side_path_guidance_count),
            "side_path_shortlist_path": self.side_path_shortlist_path,
            "side_path_candidates_path": self.side_path_candidates_path,
            "output_path": self.output_path if self.output_path else None,
        }

    @staticmethod
    def _pool_sort_key(candidate: Dict[str, Any]):
        action = str(candidate.get("repair_action") or "search_for_initial_prefix")
        priority = float(candidate.get("priority") or 0.0)
        path = candidate.get("path") or []
        low_priority_search = 1 if action == "search_for_initial_prefix" and priority < 0.8 else 0
        return (low_priority_search, ACTION_PRIORITY.get(action, 99), -priority, -len(path))

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

    def _load_blocker_evidence(self) -> Dict[str, Any]:
        candidate_paths = []
        if self._requested_blocker_evidence_path:
            candidate_paths.append(self._requested_blocker_evidence_path)
        if self._default_blocker_evidence_path and self._default_blocker_evidence_path not in candidate_paths:
            candidate_paths.append(self._default_blocker_evidence_path)

        if candidate_paths:
            self.blocker_evidence_path = candidate_paths[0]
        else:
            self.blocker_evidence_path = None

        self.blocker_evidence_loaded = False
        for path in candidate_paths:
            payload = self._read_json(path)
            if payload:
                self.blocker_evidence_path = path
                self.blocker_evidence_loaded = True
                return payload
        return {}

    def _build_side_path_guidance_pool(self) -> List[Dict[str, Any]]:
        records = []
        shortlist_payload = self._load_side_path_shortlist()
        shortlist = shortlist_payload.get("shortlist") if isinstance(shortlist_payload, dict) else None
        if isinstance(shortlist, list):
            for record in shortlist:
                if isinstance(record, dict):
                    item = self._side_path_record_to_guidance(record, source="side_path_shortlist")
                    if item:
                        records.append(item)

        for record in self._load_side_path_candidates():
            item = self._side_path_record_to_guidance(record, source="side_path_candidates")
            if item:
                records.append(item)

        records = self._dedupe_guidance_pool(records)
        self.side_path_guidance_count = len(records)
        self.side_path_guidance_loaded = bool(records)
        return records

    def _load_side_path_shortlist(self) -> Dict[str, Any]:
        candidate_paths = []
        if self._requested_side_path_shortlist_path:
            candidate_paths.append(self._requested_side_path_shortlist_path)
        if self._default_side_path_shortlist_path and self._default_side_path_shortlist_path not in candidate_paths:
            candidate_paths.append(self._default_side_path_shortlist_path)

        if candidate_paths:
            self.side_path_shortlist_path = candidate_paths[0]
        else:
            self.side_path_shortlist_path = None

        for path in candidate_paths:
            payload = self._read_json(path)
            if payload:
                self.side_path_shortlist_path = path
                return payload
        return {}

    def _load_side_path_candidates(self) -> List[Dict[str, Any]]:
        candidate_paths = []
        if self._requested_side_path_candidates_path:
            candidate_paths.append(self._requested_side_path_candidates_path)
        if self._default_side_path_candidates_path and self._default_side_path_candidates_path not in candidate_paths:
            candidate_paths.append(self._default_side_path_candidates_path)

        if candidate_paths:
            self.side_path_candidates_path = candidate_paths[0]
        else:
            self.side_path_candidates_path = None

        for path in candidate_paths:
            records = self._read_jsonl(path)
            if records:
                self.side_path_candidates_path = path
                return records
        return []

    @staticmethod
    def _read_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return []
        records = []
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

    @staticmethod
    def _side_path_record_to_guidance(record: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        if not isinstance(record, dict):
            return None
        path = record.get("suggested_path")
        if not isinstance(path, list):
            path = record.get("path")
        if not isinstance(path, list):
            return None
        normalized_path = [str(item).strip() for item in path if item not in (None, "") and str(item).strip()]
        if not normalized_path:
            return None
        tier = str(record.get("tier") or "low").lower()
        try:
            quality = float(record.get("quality_score") or 0.0)
        except Exception:
            quality = 0.0
        tier_bonus = {"high": 0.35, "medium": 0.20, "low": 0.05}.get(tier, 0.05)
        priority = max(0.10, min(0.95, 0.45 + tier_bonus + min(quality, 10.0) / 50.0))
        return {
            "hypothesis_id": str(record.get("hypothesis_id") or record.get("sequence_id") or source),
            "path": normalized_path,
            "prefix": [],
            "frontier": normalized_path[-1],
            "sink": normalized_path[-1],
            "priority": priority,
            "repair_action": "explore_side_path_candidate",
            "hints": [str(record.get("reason") or "side_path_candidate")],
            "source": source,
            "tier": tier,
            "quality_score": quality,
        }

    @staticmethod
    def _dedupe_guidance_pool(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_path: Dict[tuple, Dict[str, Any]] = {}
        for candidate in pool:
            if not isinstance(candidate, dict):
                continue
            path = candidate.get("path")
            if not isinstance(path, list) or not path:
                continue
            signature = tuple(str(item) for item in path)
            current = best_by_path.get(signature)
            if current is None or SoftPathGuidance._pool_sort_key(candidate) < SoftPathGuidance._pool_sort_key(current):
                best_by_path[signature] = candidate
        return list(best_by_path.values())

    def _candidate_to_individual(self, candidate: Dict[str, Any], indv_template) -> tuple:
        try:
            chromosome = self._build_chromosome(candidate, indv_template.generator)
            if not chromosome:
                return None, "no_valid_transactions"
            indv = indv_template.__class__(
                generator=indv_template.generator,
                other_generators=getattr(indv_template, "other_generators", []),
            ).init(chromosome=chromosome)
            return indv, None
        except Exception:
            return None, "individual_build_failed"

    @staticmethod
    def _mark_guided_individual(indv, candidate: Dict[str, Any], generation: int) -> None:
        setattr(indv, "is_soft_guided", True)
        setattr(indv, "guidance_hypothesis_id", candidate.get("hypothesis_id"))
        setattr(indv, "guidance_generation", int(generation))
        setattr(indv, "guidance_frontier", candidate.get("frontier"))
        setattr(indv, "guidance_repair_action", candidate.get("repair_action"))
        setattr(indv, "guidance_priority", candidate.get("priority"))
        setattr(indv, "guidance_path", list(candidate.get("path") or []))
        setattr(
            indv,
            "soft_guidance_metadata",
            {
                "hypothesis_id": candidate.get("hypothesis_id"),
                "generation": int(generation),
                "frontier": candidate.get("frontier"),
                "repair_action": candidate.get("repair_action"),
                "priority": candidate.get("priority"),
                "path": list(candidate.get("path") or []),
            },
        )

    def _build_chromosome(self, candidate: Dict[str, Any], generator) -> List[Dict[str, Any]]:
        path = candidate.get("path") or []
        if not isinstance(path, list) or not path:
            return []

        chromosome = []
        if "constructor" in generator.interface and generator.bytecode:
            chromosome.extend(generator.generate_constructor())

        try:
            from utils import settings
        except ImportError:
            from fuzzer.utils import settings
        max_len = getattr(settings, "MAX_INDIVIDUAL_LENGTH", 5)
        for function_name in path[:max_len]:
            function, argument_types = generator.get_specific_function_with_argument_types_without_arg_support(function_name)
            if not function or argument_types is None:
                return []
            generated = generator.generate_individual(function, argument_types)
            if not generated:
                return []
            chromosome.extend(generated)
        return chromosome

    def _append_log(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or not self.output_path:
            return
        try:
            with open(self.output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            pass


class SoftGuidanceAnalysis(OnTheFlyAnalysis):
    def __init__(self, guidance: SoftPathGuidance) -> None:
        self.guidance = guidance

    def setup(self, ng, engine):
        pass

    def register_step(self, g, population, engine):
        if self.guidance is None or not getattr(self.guidance, "enabled", False):
            return
        try:
            generation = max(int(g), 0)
            normal_population_size = len(population.individuals)
            guided_individuals = self.guidance.build_individuals(
                generation=generation,
                normal_population_size=normal_population_size,
                indv_template=population.indv_template,
            )
            if guided_individuals:
                population.individuals.extend(guided_individuals)
        except Exception:
            pass

    def finalize(self, population, engine):
        try:
            env = getattr(engine, "population", None)
            _ = env
        except Exception:
            pass
