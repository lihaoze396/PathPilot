# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from fuzzer.agent.agent_feedback import AgentFeedbackBuilder
from fuzzer.guidance.gei_side_path import GEISidePathRecorder
from fuzzer.guidance.path_plan import PathPlan, load_path_plan
from fuzzer.guidance.side_path_plan_exporter import SidePathPlanExporter
from fuzzer.agent.target_focus_agent_selection_plan import TFIAgentSelectionPlanBuilder
from fuzzer.agent.target_focus_candidate_materialization_task import TFICandidateMaterializationTaskBuilder
from fuzzer.agent.target_focus_candidate_proposal import TFICandidateProposalBuilder
from fuzzer.agent.target_focus_candidate_proposal_dryrun import TFICandidateProposalDryrun
from fuzzer.agent.target_focus_candidate_scope import TFICandidateScopeBuilder
from fuzzer.agent.target_focus_runtime import TFIRuntime


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
            for raw_line in handle:
                line = raw_line.strip()
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


def _write_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return path


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


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


def _safe_token(value: Any) -> str:
    text = _text(value).strip()
    if not text:
        return "NONE"
    output: List[str] = []
    for char in text:
        output.append(char if char.isalnum() else "_")
    sanitized = "".join(output).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or "NONE"


def _unique_preserve(values: List[Any]) -> List[Any]:
    output: List[Any] = []
    seen = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, (dict, list)) else repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def _path_within_dir(path: str, root_dir: str) -> bool:
    if not path or not root_dir:
        return False
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root_dir)]) == os.path.abspath(root_dir)
    except Exception:
        return False


def _is_current_run_artifact(path: Optional[str], output_dir: str, run_start_time: Optional[float]) -> bool:
    if not path or not output_dir or not os.path.exists(path):
        return False
    if not _path_within_dir(path, output_dir):
        return False
    if run_start_time in (None, ""):
        return True
    try:
        return float(os.path.getmtime(path)) >= float(run_start_time) - 1.0
    except Exception:
        return False


class TFIRuntimeTaskBuilder:
    def __init__(
        self,
        output_dir: str,
        max_tasks: Optional[int] = None,
        debug_artifacts: bool = False,
        fresh_run: bool = False,
        ignore_precomputed_paths: bool = False,
        ignore_precomputed_tfi_artifacts: bool = False,
        run_start_time: Optional[float] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.max_tasks = int(max_tasks) if max_tasks not in (None, "") else None
        self.debug_artifacts = bool(debug_artifacts)
        self.fresh_run = bool(fresh_run)
        self.ignore_precomputed_paths = bool(ignore_precomputed_paths or fresh_run)
        self.ignore_precomputed_tfi_artifacts = bool(ignore_precomputed_tfi_artifacts or fresh_run)
        self.run_start_time = float(run_start_time) if run_start_time not in (None, "") else None
        self._temp_paths: List[str] = []

    def build_runtime_tasks(self) -> Dict[str, Any]:
        if not self.output_dir:
            return self._result(
                status="no_evidence",
                reason="missing_output_dir",
                tasks_payload={},
                used_inputs=[],
                missing_inputs=["output_dir"],
            )

        input_paths = self._discover_input_paths()
        initial_missing = [name for name, path in input_paths.items() if not path or not os.path.exists(path)]
        used_inputs = [name for name, path in input_paths.items() if path and os.path.exists(path)]

        if not input_paths.get("path_progress") and not input_paths.get("ksi_evidence") and not input_paths.get("candidate_scope"):
            return self._result(
                status="no_evidence",
                reason="missing_runtime_evidence",
                tasks_payload={},
                used_inputs=used_inputs,
                missing_inputs=initial_missing,
            )

        try:
            self._ensure_side_path_shortlist(input_paths)
            self._ensure_proposed_path_plan(input_paths)
            self._ensure_blocker_evidence(input_paths)
            self._ensure_tfi_focus(input_paths)

            scopes_payload, scope_path, scope_summary = self._build_scopes(input_paths)
            candidate_gap_events = self._build_candidate_gap_events(scopes_payload)
            request_exports = self._build_request_exports(scopes_payload)

            binding_count = int(request_exports.get("stats", {}).get("binding_repair_request_count", 0) or 0)
            selection_count = int(request_exports.get("stats", {}).get("scoped_selection_request_count", 0) or 0)
            request_count = binding_count + selection_count

            if request_count <= 0:
                return self._result(
                    status="candidate_gap_only" if candidate_gap_events else "no_tasks",
                    reason=self._reason_from_scope_summary(scope_summary, candidate_gap_events),
                    tasks_payload={},
                    used_inputs=self._used_inputs(input_paths),
                    missing_inputs=self._missing_inputs(input_paths),
                    scope_count=int(scopes_payload.get("scope_count") or 0),
                    request_count=request_count,
                    plan_count=0,
                    scope_path=scope_path,
                    candidate_gap_events=candidate_gap_events,
                )

            binding_requests_path, scoped_requests_path = self._write_request_payloads(request_exports)
            plan_builder = TFIAgentSelectionPlanBuilder(
                output_dir=self.output_dir,
                binding_requests_path=binding_requests_path,
                scoped_selection_requests_path=scoped_requests_path,
                candidate_scope_path=scope_path,
                max_plans=self.max_tasks,
            )
            plan_payload = plan_builder.build_plan()
            plan_count = int(plan_payload.get("plan_count") or 0)
            plan_path = self._write_payload(
                plan_payload,
                canonical_name="target_focus_agent_selection_plan.json",
                temp_name="__target_focus_runtime_agent_selection_plan.json",
            )

            if plan_count <= 0:
                return self._result(
                    status="no_tasks",
                    reason="no_agent_selection_plans",
                    tasks_payload={},
                    used_inputs=self._used_inputs(input_paths),
                    missing_inputs=self._missing_inputs(input_paths),
                    scope_count=int(scopes_payload.get("scope_count") or 0),
                    request_count=request_count,
                    plan_count=0,
                    scope_path=scope_path,
                    candidate_gap_events=candidate_gap_events,
                )

            proposal_builder = TFICandidateProposalBuilder(
                output_dir=self.output_dir,
                agent_selection_plan_path=plan_path,
                binding_requests_path=binding_requests_path,
                scoped_selection_requests_path=scoped_requests_path,
                candidate_scope_path=scope_path,
                max_proposals=self.max_tasks,
            )
            proposal_payload = proposal_builder.build_proposals()
            proposal_path = self._write_payload(
                proposal_payload,
                canonical_name="target_focus_candidate_proposals.json",
                temp_name="__target_focus_runtime_candidate_proposals.json",
            )

            dryrun_builder = TFICandidateProposalDryrun(
                output_dir=self.output_dir,
                proposals_path=proposal_path,
                candidate_scope_path=scope_path,
                agent_selection_plan_path=plan_path,
                path_progress_path=input_paths.get("path_progress"),
                ksi_evidence_path=input_paths.get("ksi_evidence"),
                selected_guided_sequences_path=input_paths.get("selected_guided_sequences"),
                max_items=self.max_tasks,
            )
            dryrun_payload = dryrun_builder.run_dryrun()
            dryrun_path = self._write_payload(
                dryrun_payload,
                canonical_name="target_focus_candidate_proposal_dryrun.json",
                temp_name="__target_focus_runtime_candidate_proposal_dryrun.json",
            )

            task_builder = TFICandidateMaterializationTaskBuilder(
                output_dir=self.output_dir,
                proposal_dryrun_path=dryrun_path,
                proposals_path=proposal_path,
                agent_selection_plan_path=plan_path,
                candidate_scope_path=scope_path,
                max_tasks=self.max_tasks,
            )
            tasks_payload = task_builder.build_tasks()
            task_count = int(tasks_payload.get("task_count") or 0)
            if task_count <= 0:
                return self._result(
                    status="no_tasks",
                    reason=self._reason_from_tasks_payload(tasks_payload, scope_summary),
                    tasks_payload=tasks_payload,
                    used_inputs=self._used_inputs(input_paths),
                    missing_inputs=self._missing_inputs(input_paths),
                    scope_count=int(scopes_payload.get("scope_count") or 0),
                    request_count=request_count,
                    plan_count=plan_count,
                    scope_path=scope_path,
                    candidate_gap_events=candidate_gap_events,
                )

            task_path = self._write_payload(
                tasks_payload,
                canonical_name="target_focus_candidate_materialization_tasks.json",
                temp_name="__target_focus_runtime_candidate_materialization_tasks.json",
            )
            return self._result(
                status="ok",
                reason="tasks_ready",
                tasks_payload=tasks_payload,
                used_inputs=self._used_inputs(input_paths),
                missing_inputs=self._missing_inputs(input_paths),
                scope_count=int(scopes_payload.get("scope_count") or 0),
                request_count=request_count,
                plan_count=plan_count,
                scope_path=scope_path,
                tasks_path=task_path,
                candidate_gap_events=candidate_gap_events,
            )
        except Exception as exc:
            return self._result(
                status="error",
                reason="runtime_task_builder_error: {}".format(str(exc)),
                tasks_payload={},
                used_inputs=self._used_inputs(input_paths),
                missing_inputs=self._missing_inputs(input_paths),
                candidate_gap_events=[],
            )

    def _discover_input_paths(self) -> Dict[str, Optional[str]]:
        def choose(*names: str) -> Optional[str]:
            for name in names:
                path = os.path.join(self.output_dir, name)
                if os.path.exists(path):
                    return path
            return os.path.join(self.output_dir, names[0]) if names else None

        paths = {
            "path_progress": choose("path_progress.jsonl"),
            "ksi_evidence": choose("ksi_evidence.jsonl"),
            "blocker_evidence": choose("blocker_evidence.json"),
            "tfi_focus_targets": choose("tfi_focus_targets.jsonl"),
            "selected_guided_sequences": choose("selected_guided_sequences.json"),
            "repair_pilot_failure_analysis": choose("repair_pilot_failure_analysis.json"),
            "guided_sequence_pilot_results": choose("guided_sequence_pilot_results.json"),
            "guided_sequence_pilot_summary": choose("guided_sequence_pilot_summary.json"),
            "agent_path_plan": choose("agent_path_plan.json"),
            "proposed_path_plan": choose("proposed_path_plan.json"),
            "side_path_candidates": choose("side_path_candidates.jsonl"),
            "side_path_shortlist": choose("side_path_shortlist.json"),
            "candidate_scope": choose("target_focus_candidate_scope.json"),
        }
        return self._filter_precomputed_inputs(paths)

    def _filter_precomputed_inputs(self, input_paths: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        filtered = dict(input_paths)
        path_keys = {"agent_path_plan", "proposed_path_plan"}
        tfi_artifact_keys = {
            "blocker_evidence",
            "tfi_focus_targets",
            "selected_guided_sequences",
            "repair_pilot_failure_analysis",
            "guided_sequence_pilot_results",
            "guided_sequence_pilot_summary",
            "side_path_shortlist",
            "candidate_scope",
        }
        for key, path in list(filtered.items()):
            if not path or not os.path.exists(path):
                continue
            if key in ("path_progress", "ksi_evidence", "side_path_candidates"):
                continue
            if key in path_keys and self.ignore_precomputed_paths:
                if not _is_current_run_artifact(path, self.output_dir, self.run_start_time):
                    filtered[key] = os.path.join(self.output_dir, os.path.basename(path))
            if key in tfi_artifact_keys and self.ignore_precomputed_tfi_artifacts:
                if not _is_current_run_artifact(path, self.output_dir, self.run_start_time):
                    filtered[key] = os.path.join(self.output_dir, os.path.basename(path))
        return filtered

    def _ensure_side_path_shortlist(self, input_paths: Dict[str, Optional[str]]) -> None:
        shortlist_path = input_paths.get("side_path_shortlist")
        if shortlist_path and os.path.exists(shortlist_path):
            return
        side_path_candidates_path = input_paths.get("side_path_candidates")
        if not side_path_candidates_path or not os.path.exists(side_path_candidates_path):
            return
        try:
            recorder = GEISidePathRecorder(output_dir=self.output_dir, enabled=False)
            recorder._candidates = _read_jsonl(side_path_candidates_path)
            shortlist_payload = recorder._build_shortlist_payload()
            shortlist_target = os.path.join(self.output_dir, "side_path_shortlist.json")
            _write_json(shortlist_target, shortlist_payload)
            input_paths["side_path_shortlist"] = shortlist_target
        except Exception:
            return

    def _ensure_proposed_path_plan(self, input_paths: Dict[str, Optional[str]]) -> None:
        if input_paths.get("agent_path_plan") and os.path.exists(input_paths["agent_path_plan"]):
            return
        proposed_path = input_paths.get("proposed_path_plan")
        if proposed_path and os.path.exists(proposed_path):
            return
        shortlist_path = input_paths.get("side_path_shortlist")
        if not shortlist_path or not os.path.exists(shortlist_path):
            return
        exporter = SidePathPlanExporter(output_dir=self.output_dir, shortlist_path=shortlist_path)
        exported_path = exporter.export(max_hypotheses=20, filename="proposed_path_plan.json")
        if exported_path:
            input_paths["proposed_path_plan"] = exported_path

    def _ensure_blocker_evidence(self, input_paths: Dict[str, Optional[str]]) -> None:
        blocker_path = input_paths.get("blocker_evidence")
        if blocker_path and os.path.exists(blocker_path):
            return
        path_plan_path = self._path_plan_path(input_paths)
        if not path_plan_path or not os.path.exists(path_plan_path):
            return
        path_plan = load_path_plan(path_plan_path)
        if not isinstance(path_plan, PathPlan) or path_plan.is_empty():
            return
        builder = AgentFeedbackBuilder(
            path_plan=path_plan,
            output_dir=self.output_dir,
            path_progress_path=input_paths.get("path_progress"),
            ksi_evidence_path=input_paths.get("ksi_evidence"),
            side_path_candidates_path=input_paths.get("side_path_candidates"),
        )
        blocker_output = builder.build_and_write()
        if blocker_output:
            input_paths["blocker_evidence"] = blocker_output

    def _ensure_tfi_focus(self, input_paths: Dict[str, Optional[str]]) -> None:
        focus_path = input_paths.get("tfi_focus_targets")
        blocker_path = input_paths.get("blocker_evidence")
        if not blocker_path or not os.path.exists(blocker_path):
            return
        if focus_path and os.path.exists(focus_path):
            try:
                focus_nonempty = os.path.getsize(focus_path) > 0
                focus_fresh = os.path.getmtime(focus_path) >= os.path.getmtime(blocker_path)
                if focus_nonempty and focus_fresh:
                    return
            except Exception:
                pass
        runtime = TFIRuntime(
            output_dir=self.output_dir,
            enabled=True,
            blocker_evidence_path=blocker_path,
        )
        summary = runtime.run()
        output_path = _text(summary.get("output_path")).strip()
        if output_path and os.path.exists(output_path):
            input_paths["tfi_focus_targets"] = output_path

    def _build_scopes(self, input_paths: Dict[str, Optional[str]]) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        existing_scope_path = input_paths.get("candidate_scope")
        existing_scope_payload = _read_json(existing_scope_path)
        if isinstance(existing_scope_payload, dict) and _listify(existing_scope_payload.get("scopes")):
            builder = TFICandidateScopeBuilder(output_dir=self.output_dir)
            scope_summary = builder._build_summary(existing_scope_payload)
            return existing_scope_payload, existing_scope_path or "", scope_summary

        scope_builder = TFICandidateScopeBuilder(
            output_dir=self.output_dir,
            path_progress_path=input_paths.get("path_progress"),
            ksi_evidence_path=input_paths.get("ksi_evidence"),
            blocker_evidence_path=input_paths.get("blocker_evidence"),
            tfi_focus_path=input_paths.get("tfi_focus_targets"),
            path_plan_path=self._path_plan_path(input_paths),
            selected_guided_sequences_path=input_paths.get("selected_guided_sequences"),
            repair_pilot_failure_analysis_path=input_paths.get("repair_pilot_failure_analysis"),
        )
        scopes_payload = scope_builder.build_scopes()
        scope_summary = scope_builder._build_summary(scopes_payload)
        scope_path = self._write_payload(
            scopes_payload,
            canonical_name="target_focus_candidate_scope.json",
            temp_name="__target_focus_runtime_candidate_scope.json",
        )
        if self.debug_artifacts:
            scope_builder.write_summary(scopes_payload, filename="target_focus_candidate_scope_summary.json")
            scope_builder.write_binding_repair_requests(scopes_payload, filename="tfi_binding_repair_requests.json")
            scope_builder.write_scoped_selection_requests(
                scopes_payload,
                filename="tfi_scoped_selection_requests.json",
            )
        return scopes_payload, scope_path, scope_summary

    def _build_request_exports(self, scopes_payload: Dict[str, Any]) -> Dict[str, Any]:
        builder = TFICandidateScopeBuilder(output_dir=self.output_dir)
        return builder._build_request_exports(scopes_payload)

    def _write_request_payloads(self, request_exports: Dict[str, Any]) -> Tuple[str, str]:
        binding_payload = request_exports.get("binding_payload") if isinstance(request_exports.get("binding_payload"), dict) else {}
        selection_payload = request_exports.get("selection_payload") if isinstance(request_exports.get("selection_payload"), dict) else {}
        binding_path = self._write_payload(
            binding_payload,
            canonical_name="tfi_binding_repair_requests.json",
            temp_name="__target_focus_runtime_binding_repair_requests.json",
        )
        selection_path = self._write_payload(
            selection_payload,
            canonical_name="tfi_scoped_selection_requests.json",
            temp_name="__target_focus_runtime_scoped_selection_requests.json",
        )
        return binding_path, selection_path

    def _write_payload(self, payload: Dict[str, Any], canonical_name: str, temp_name: str) -> str:
        filename = canonical_name if self.debug_artifacts else temp_name
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        if not self.debug_artifacts:
            self._temp_paths.append(path)
        return path

    def _path_plan_path(self, input_paths: Dict[str, Optional[str]]) -> Optional[str]:
        for key in ("agent_path_plan", "proposed_path_plan"):
            path = input_paths.get(key)
            if path and os.path.exists(path):
                return path
        return input_paths.get("agent_path_plan") or input_paths.get("proposed_path_plan")

    def _reason_from_scope_summary(self, scope_summary: Dict[str, Any], candidate_gap_events: Optional[List[Dict[str, Any]]] = None) -> str:
        diagnosis_counts = scope_summary.get("tfi_failure_diagnosis_counts") if isinstance(scope_summary.get("tfi_failure_diagnosis_counts"), dict) else {}
        legacy_counts = scope_summary.get("legacy_tfi_failure_diagnosis_counts") if isinstance(scope_summary.get("legacy_tfi_failure_diagnosis_counts"), dict) else {}
        binding_request_count = int(scope_summary.get("binding_repair_request_count") or 0)
        selection_request_count = int(scope_summary.get("scoped_selection_request_count") or 0)
        if candidate_gap_events:
            return "local_candidate_gap_only"
        if binding_request_count <= 0 and selection_request_count <= 0:
            if int(diagnosis_counts.get("possible_local_candidate_gap", 0) or 0) > 0:
                return "only_possible_local_candidate_gap"
            if int(diagnosis_counts.get("possible_missing_action_atom", 0) or 0) > 0 or int(legacy_counts.get("possible_missing_action_atom", 0) or 0) > 0:
                return "only_possible_missing_action_atom"
            local_audit = scope_summary.get("scope_local_evidence_audit") if isinstance(scope_summary.get("scope_local_evidence_audit"), dict) else {}
            if int(local_audit.get("diagnosis_not_scope_local_count", 0) or 0) > 0:
                return "insufficient_scope_local_evidence"
            if int(local_audit.get("scopes_with_unobserved_matched_prefix", 0) or 0) > 0:
                return "no_observed_prefix"
        return "no_binding_or_scoped_selection_requests"

    def _reason_from_tasks_payload(self, tasks_payload: Dict[str, Any], scope_summary: Dict[str, Any]) -> str:
        task_count = int(tasks_payload.get("task_count") or 0)
        if task_count > 0:
            return "tasks_ready"
        review_queue = _listify(tasks_payload.get("review_queue"))
        rejected_queue = _listify(tasks_payload.get("rejected_queue"))
        if review_queue and not rejected_queue:
            return "no_passed_materialization_tasks"
        if rejected_queue and not review_queue:
            return "all_candidates_rejected"
        return self._reason_from_scope_summary(scope_summary)

    def _used_inputs(self, input_paths: Dict[str, Optional[str]]) -> List[str]:
        return [name for name, path in input_paths.items() if path and os.path.exists(path)]

    def _missing_inputs(self, input_paths: Dict[str, Optional[str]]) -> List[str]:
        return [name for name, path in input_paths.items() if not path or not os.path.exists(path)]

    def _result(
        self,
        status: str,
        reason: str,
        tasks_payload: Dict[str, Any],
        used_inputs: List[str],
        missing_inputs: List[str],
        scope_count: int = 0,
        request_count: int = 0,
        plan_count: int = 0,
        scope_path: Optional[str] = None,
        tasks_path: Optional[str] = None,
        candidate_gap_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        tasks = [item for item in _listify(tasks_payload.get("tasks")) if isinstance(item, dict)]
        review_queue = [item for item in _listify(tasks_payload.get("review_queue")) if isinstance(item, dict)]
        rejected_queue = [item for item in _listify(tasks_payload.get("rejected_queue")) if isinstance(item, dict)]
        candidate_gap_events = [item for item in _listify(candidate_gap_events) if isinstance(item, dict)]
        return {
            "enabled": True,
            "source": "target_focus_runtime_task_builder_v1",
            "task_count": len(tasks),
            "tasks": tasks,
            "candidate_gap_event_count": len(candidate_gap_events),
            "candidate_gap_events": candidate_gap_events,
            "review_queue_count": len(review_queue),
            "rejected_queue_count": len(rejected_queue),
            "status": status,
            "reason": reason,
            "used_inputs": used_inputs,
            "missing_inputs": missing_inputs,
            "debug_artifacts_written": bool(self.debug_artifacts),
            "scope_count": int(scope_count),
            "request_count": int(request_count),
            "plan_count": int(plan_count),
            "path_plan_path": self._path_plan_path(self._discover_input_paths()),
            "scope_path": scope_path,
            "tasks_path": tasks_path,
            "tasks_payload": tasks_payload if isinstance(tasks_payload, dict) else {},
        }

    def _build_candidate_gap_events(self, scopes_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for scope in _listify(scopes_payload.get("scopes")):
            if not isinstance(scope, dict):
                continue
            diagnosis = scope.get("tfi_failure_diagnosis") if isinstance(scope.get("tfi_failure_diagnosis"), dict) else {}
            if _text(diagnosis.get("diagnosis_type")).strip() != "possible_local_candidate_gap":
                continue
            event = {
                "gap_event_id": "TFI_GAP_EVT_{}_{}".format(_safe_token(scope.get("scope_id")), _safe_token(scope.get("hypothesis_id"))),
                "event_type": "candidate_gap_repair_event",
                "scope_id": scope.get("scope_id"),
                "hypothesis_id": scope.get("hypothesis_id"),
                "target_function": scope.get("target_function"),
                "frontier": scope.get("frontier"),
                "sink": scope.get("sink"),
                "observed_prefix": _clean_list(scope.get("scope_local_evidence_check", {}).get("observed_matched_prefix") if isinstance(scope.get("scope_local_evidence_check"), dict) else []),
                "gap_type": "local_candidate_gap",
                "recommended_search_direction": _clean_list(scope.get("scope_local_evidence_check", {}).get("recommended_search_direction") if isinstance(scope.get("scope_local_evidence_check"), dict) else []),
                "negative_evidence": _unique_preserve(
                    _clean_list(scope.get("repair_negative_scope", {}).get("target_seen_but_no_path_progress") if isinstance(scope.get("repair_negative_scope"), dict) else [])
                ),
                "forbidden_patterns": _unique_preserve(
                    _clean_list(scope.get("scope_restriction_context", {}).get("forbidden_candidate_patterns") if isinstance(scope.get("scope_restriction_context"), dict) else [])
                ),
                "agent_guidance_required": True,
                "fuzzer_generation_required": True,
                "do_not_create_new_pool": True,
                "do_not_generate_action_atom_in_tfi": False,
                "do_not_generate_unscoped_action_atom_in_tfi": True,
                "allow_scope_local_action_composition": True,
                "diagnosis_type": "possible_local_candidate_gap",
                "legacy_diagnosis_type": "possible_missing_action_atom",
                "should_generate_action_atom_now": True,
                "should_create_new_pool": False,
                "should_mark_existing_pool_candidates": True,
                "reason": _text(diagnosis.get("reason")).strip() or "local_candidate_gap_detected",
            }
            events.append(event)
        return events

    def _cleanup_temp_paths(self) -> None:
        for path in reversed(self._temp_paths):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        self._temp_paths = []

    def cleanup(self) -> None:
        if not self.debug_artifacts:
            self._cleanup_temp_paths()

