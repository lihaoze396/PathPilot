# -*- coding: utf-8 -*-

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fuzzer.agent.llm_provider import LLMProvider, LLMProviderError
from fuzzer.agent.target_focus_candidate_gap import TFICandidateGapRepair
from fuzzer.agent.target_focus_candidate_pilot import TFICandidatePilot
from fuzzer.agent.target_focus_runtime_task_builder import TFIRuntimeTaskBuilder
from fuzzer.guidance.guided_sequence_pilot import RepairSequencePilot
from fuzzer.runtime.phase_cost import record_phase_event


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


def _build_llm_timing_event(
    *,
    phase: str,
    started_at: Optional[float],
    ended_at: Optional[float],
    success: bool,
    error: str = "",
    fallback_used: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        started_value = float(started_at) if started_at not in (None, "") else None
        ended_value = float(ended_at) if ended_at not in (None, "") else None
    except Exception:
        started_value = None
        ended_value = None
    duration = 0.0
    if started_value is not None and ended_value is not None and ended_value >= started_value:
        duration = float(ended_value - started_value)
    payload: Dict[str, Any] = {
        "phase": _text(phase).strip() or "unknown",
        "event_kind": "llm",
        "started_at": started_value,
        "ended_at": ended_value,
        "duration_sec": duration,
        "success": bool(success),
        "error": _text(error).strip() or None,
        "fallback_used": bool(fallback_used),
    }
    if isinstance(metadata, dict) and metadata:
        payload["metadata"] = dict(metadata)
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        if usage:
            payload["usage"] = dict(usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "cost_estimation_mode"):
            if key in metadata:
                payload[key] = metadata.get(key)
    return payload


class TFIAgentRuntime:
    def __init__(
        self,
        output_dir: str,
        materialization_tasks_path: Optional[str] = None,
        candidate_scope_path: Optional[str] = None,
        contract_name: Optional[str] = None,
        source_path: Optional[str] = None,
        gas_limit: Optional[int] = None,
        max_tasks_per_round: int = 4,
        max_candidates_per_task: int = 2,
        max_total_candidates: int = 20,
        enable_agent: bool = False,
        enable_isolated_pilot: bool = True,
        llm_provider: Optional[LLMProvider] = None,
        debug_artifacts: bool = False,
        fresh_run: bool = False,
        ignore_precomputed_paths: bool = False,
        ignore_precomputed_tfi_artifacts: bool = False,
        run_start_time: Optional[float] = None,
        strict_llm_only: bool = False,
        phase_cost_env: Any = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.fresh_run = bool(fresh_run)
        self.ignore_precomputed_paths = bool(ignore_precomputed_paths or fresh_run)
        self.ignore_precomputed_tfi_artifacts = bool(ignore_precomputed_tfi_artifacts or fresh_run)
        self.run_start_time = float(run_start_time) if run_start_time not in (None, "") else None
        self.strict_llm_only = bool(strict_llm_only)
        self.phase_cost_env = phase_cost_env
        self.explicit_materialization_tasks_path = (
            os.path.abspath(materialization_tasks_path) if materialization_tasks_path else None
        )
        if self.explicit_materialization_tasks_path and self.ignore_precomputed_tfi_artifacts:
            if not _is_current_run_artifact(self.explicit_materialization_tasks_path, self.output_dir, self.run_start_time):
                self.explicit_materialization_tasks_path = None
        self.materialization_tasks_path = self.explicit_materialization_tasks_path
        resolved_scope_path = candidate_scope_path or os.path.join(self.output_dir, "target_focus_candidate_scope.json")
        if resolved_scope_path and self.ignore_precomputed_tfi_artifacts:
            if os.path.exists(resolved_scope_path) and not _is_current_run_artifact(
                resolved_scope_path,
                self.output_dir,
                self.run_start_time,
            ):
                resolved_scope_path = os.path.join(self.output_dir, "target_focus_candidate_scope.json")
        self.candidate_scope_path = resolved_scope_path
        self.contract_name = _text(contract_name).strip()
        self.source_path = os.path.abspath(source_path) if source_path else ""
        self.gas_limit = int(gas_limit) if gas_limit not in (None, "") else None
        self.max_tasks_per_round = max(1, int(max_tasks_per_round or 1))
        self.max_candidates_per_task = max(1, int(max_candidates_per_task or 1))
        self.max_total_candidates = max(1, int(max_total_candidates or 1))
        self.enable_agent = bool(enable_agent)
        self.enable_isolated_pilot = bool(enable_isolated_pilot)
        self.llm_provider = llm_provider
        self.debug_artifacts = bool(debug_artifacts)
        self.summary_path = os.path.join(self.output_dir, "target_focus_runtime_agent_summary.json") if self.output_dir else ""
        self._last_error = ""
        self._task_cursor = 0
        self.repair_pilot_injector = _TFIRepairPilotInjector(self.output_dir)
        self.gap_repair = TFICandidateGapRepair(
            abi=None,
            static_hints={
                "contract_name": self.contract_name,
                "source_path": self.source_path,
            },
            max_candidates_per_gap=self.max_candidates_per_task,
        )
        self._agent_candidate_counter = 0
        self._gap_candidate_counter = 0
        self._active_tasks_path: Optional[str] = None
        self._active_scope_path: Optional[str] = None
        self.last_task_build_result: Dict[str, Any] = {}
        self.last_fallback_used = False
        self.last_gap_events: List[Dict[str, Any]] = []
        self.last_gap_guidances: List[Dict[str, Any]] = []
        self.last_llm_success_count = 0
        self.last_llm_error_count = 0
        self.last_gap_llm_success_count = 0
        self.last_gap_llm_error_count = 0
        reference_dir = self._reference_input_dir()
        self.path_progress_path = os.path.join(reference_dir, "path_progress.jsonl") if reference_dir else None
        self.ksi_evidence_path = os.path.join(reference_dir, "ksi_evidence.jsonl") if reference_dir else None
        self.selected_guided_sequences_path = (
            os.path.join(reference_dir, "selected_guided_sequences.json") if reference_dir else None
        )
        if self.selected_guided_sequences_path and self.ignore_precomputed_tfi_artifacts:
            if not _is_current_run_artifact(self.selected_guided_sequences_path, self.output_dir, self.run_start_time):
                self.selected_guided_sequences_path = None

    def run_once(self) -> Dict[str, Any]:
        task_source = self._load_tasks_for_round()
        cleanup_builder = task_source.get("cleanup_builder")
        try:
            self.last_llm_success_count = 0
            self.last_llm_error_count = 0
            self._active_tasks_path = task_source.get("tasks_path")
            self._active_scope_path = task_source.get("scope_path") or self.candidate_scope_path
            self.last_task_build_result = task_source.get("task_build") if isinstance(task_source.get("task_build"), dict) else {}
            tasks = [item for item in _listify(task_source.get("tasks")) if isinstance(item, dict)]
            gap_events = [item for item in _listify(task_source.get("candidate_gap_events")) if isinstance(item, dict)]
            scope_payload = _read_json(self._active_scope_path)
            selected_tasks = self._select_tasks(tasks)
            selected_gap_events = self._select_gap_events(gap_events)
            if not selected_tasks and not selected_gap_events:
                result = self._build_skip_result(
                    scope_payload=scope_payload,
                    task_build=self.last_task_build_result,
                    skip_reason="no_target_focus_runtime_tasks",
                )
                self._write_summary(result)
                return result

            candidate_records: List[Dict[str, Any]] = []
            rejected_candidates: List[Dict[str, Any]] = []
            agent_called = False
            llm_called = False
            errors: List[str] = []
            self.last_fallback_used = False

            guidance_records: List[Dict[str, Any]] = []
            candidate_origin = "unknown"
            candidate_gap_guidance_count = 0
            fuzzer_generated_candidate_count = 0
            llm_interaction_events: List[Dict[str, Any]] = []

            if selected_tasks:
                candidate_origin = "tfi_materialization_task"
                if self.enable_agent:
                    agent_called = True
                    llm_result = self._build_candidates_with_agent(selected_tasks)
                    candidate_records.extend(llm_result["accepted"])
                    rejected_candidates.extend(llm_result["rejected"])
                    llm_called = bool(llm_result["llm_called"])
                    llm_interaction_events.extend(
                        [item for item in _listify(llm_result.get("llm_interaction_events")) if isinstance(item, dict)]
                    )
                    self.last_llm_success_count += int(llm_result.get("llm_success_count") or 0)
                    self.last_llm_error_count += int(llm_result.get("llm_error_count") or 0)
                    if llm_result["error"] and llm_result["error"] not in (
                        "agent_provider_unavailable",
                        "agent_provider_not_configured",
                    ):
                        errors.append(llm_result["error"])

                if not candidate_records and not self.strict_llm_only:
                    fallback = self._build_candidates_with_fallback(selected_tasks)
                    candidate_records.extend(fallback)
                    self.last_fallback_used = bool(candidate_records)
            else:
                gap_result = self._build_candidates_from_gap_events(selected_gap_events)
                candidate_records.extend(gap_result.get("candidates", []))
                rejected_candidates.extend(gap_result.get("rejected", []))
                guidance_records.extend(gap_result.get("guidances", []))
                candidate_gap_guidance_count = int(gap_result.get("guidance_generated_count") or 0)
                fuzzer_generated_candidate_count = int(gap_result.get("fuzzer_generated_candidate_count") or 0)
                candidate_origin = _text(gap_result.get("candidate_origin")).strip() or "tfi_guided_candidate_gap_repair"
                agent_called = bool(gap_result.get("agent_called"))
                llm_called = bool(gap_result.get("llm_called"))
                self.last_fallback_used = bool(gap_result.get("fallback_used"))
                llm_interaction_events.extend(
                    [item for item in _listify(gap_result.get("llm_interaction_events")) if isinstance(item, dict)]
                )
                self.last_llm_success_count += int(gap_result.get("llm_success_count") or 0)
                self.last_llm_error_count += int(gap_result.get("llm_error_count") or 0)
                self.last_gap_events = list(selected_gap_events)
                self.last_gap_guidances = list(guidance_records)
                if gap_result.get("error"):
                    errors.append(_text(gap_result.get("error")).strip())

            if len(candidate_records) > self.max_total_candidates:
                candidate_records = candidate_records[: self.max_total_candidates]

            pilot_result = {
                "enabled": True,
                "agent_enabled": bool(self.enable_agent),
                "soft_injection_enabled": False,
                "isolated_pilot_enabled": bool(self.enable_isolated_pilot),
                "agent_called": agent_called,
                "agent_called_count": 1 if agent_called else 0,
                "agent_error_count": len(errors),
                "candidate_gap_event_count": len(gap_events),
                "candidate_gap_guidance_count": int(candidate_gap_guidance_count),
                "guidance_generated_count": int(candidate_gap_guidance_count),
                "fuzzer_generated_candidate_count": int(fuzzer_generated_candidate_count),
                "candidate_origin": candidate_origin,
                "candidate_built_count": len(candidate_records),
                "candidate_executed_count": 0,
                "useful_candidate_count": 0,
                "soft_injected_count": 0,
                "duplicate_skipped_count": 0,
                "no_main_search_replacement": True,
                "no_generator_mutation_crossover_selection_change": True,
                "llm_called": llm_called,
                "llm_success_count": int(self.last_llm_success_count),
                "llm_error_count": int(self.last_llm_error_count),
                "fallback_used": bool(self.last_fallback_used),
                "rag_used": False,
                "tasks_seen": len(selected_tasks),
                "candidate_gap_events_seen": len(selected_gap_events),
                "task_count": int(self.last_task_build_result.get("task_count") or len(tasks)),
                "task_build_attempted": bool(self.last_task_build_result),
                "task_build": self.last_task_build_result,
                "candidates_built": len(candidate_records),
                "candidate_records": candidate_records,
                "candidates_executed": 0,
                "useful_candidates": 0,
                "useful_candidate_records": [],
                "rejected_candidates": rejected_candidates,
                "agent_errors": errors,
                "scope_loaded": bool(scope_payload),
                "skip_reason": "",
                "candidate_gap_events": selected_gap_events,
                "gap_guidance_records": guidance_records,
                "llm_interaction_events": llm_interaction_events[-256:],
                "llm_interaction_count": len(llm_interaction_events),
                "llm_interaction_time_total_sec": sum(
                    float(item.get("duration_sec") or 0.0)
                    for item in llm_interaction_events
                    if isinstance(item, dict)
                ),
            }

            if self.enable_isolated_pilot and candidate_records:
                pilot = TFICandidatePilot(
                    output_dir=self.output_dir,
                    candidate_scope_path=self._active_scope_path,
                    path_progress_path=self.path_progress_path,
                    ksi_evidence_path=self.ksi_evidence_path,
                    selected_guided_sequences_path=self.selected_guided_sequences_path,
                    max_candidates_per_task=self.max_candidates_per_task,
                    max_total_candidates=self.max_total_candidates,
                    execute=True,
                    gas_limit=self.gas_limit,
                    contract_name=self.contract_name,
                    source_path=self.source_path,
                )
                result = pilot.run_candidates(
                    candidates=candidate_records,
                    task_count=len(selected_tasks),
                    loaded_inputs={
                        "materialization_tasks": bool(self._active_tasks_path and os.path.exists(self._active_tasks_path)),
                        "candidate_scope": bool(self._active_scope_path and os.path.exists(self._active_scope_path)),
                        "path_progress": bool(pilot.path_progress_path and os.path.exists(pilot.path_progress_path)),
                        "ksi_evidence": bool(pilot.ksi_evidence_path and os.path.exists(pilot.ksi_evidence_path)),
                        "selected_guided_sequences": bool(
                            pilot.selected_guided_sequences_path and os.path.exists(pilot.selected_guided_sequences_path)
                        ),
                    },
                    missing_inputs=[],
                )
                result_by_id = {
                    _text(item.get("candidate_id")).strip(): item
                    for item in _listify(result.get("results"))
                    if isinstance(item, dict)
                }
                useful_records: List[Dict[str, Any]] = []
                for candidate in candidate_records:
                    candidate_id = _text(candidate.get("candidate_id")).strip()
                    execution = result_by_id.get(candidate_id)
                    if not execution:
                        continue
                    merged = dict(candidate)
                    merged.update(execution)
                    if bool(execution.get("useful")) and _text(execution.get("execution_status")).strip() != "skipped":
                        useful_records.append(merged)
                pilot_result.update(
                    {
                        "candidate_executed_count": int(result.get("executed_candidate_count") or 0),
                        "useful_candidate_count": len(useful_records),
                        "candidates_executed": int(result.get("executed_candidate_count") or 0),
                        "useful_candidates": len(useful_records),
                        "useful_candidate_records": useful_records,
                        "pilot_result": result,
                    }
                )
            else:
                pilot_result["pilot_result"] = {
                    "enabled": True,
                    "executed": False,
                    "reason": "isolated_pilot_disabled_or_no_candidates",
                }

            self._write_summary(pilot_result)
            return pilot_result
        finally:
            if cleanup_builder is not None:
                try:
                    cleanup_builder.cleanup()
                except Exception:
                    pass

    def _build_skip_result(
        self,
        scope_payload: Dict[str, Any],
        task_build: Dict[str, Any],
        skip_reason: str,
    ) -> Dict[str, Any]:
        return {
            "enabled": True,
            "agent_enabled": bool(self.enable_agent),
            "soft_injection_enabled": False,
            "isolated_pilot_enabled": bool(self.enable_isolated_pilot),
            "agent_called": False,
            "agent_called_count": 0,
            "agent_error_count": 0,
            "candidate_built_count": 0,
            "candidate_executed_count": 0,
            "useful_candidate_count": 0,
            "soft_injected_count": 0,
            "duplicate_skipped_count": 0,
            "no_main_search_replacement": True,
            "no_generator_mutation_crossover_selection_change": True,
            "llm_called": False,
            "fallback_used": False,
            "rag_used": False,
            "tasks_seen": 0,
            "candidate_gap_event_count": int(task_build.get("candidate_gap_event_count") or 0),
            "candidate_gap_guidance_count": 0,
            "guidance_generated_count": 0,
            "fuzzer_generated_candidate_count": 0,
            "candidate_origin": "",
            "candidate_gap_events_seen": 0,
            "task_count": int(task_build.get("task_count") or 0),
            "task_build_attempted": bool(task_build),
            "task_build": task_build,
            "candidates_built": 0,
            "candidates_executed": 0,
            "useful_candidates": 0,
            "useful_candidate_records": [],
            "rejected_candidates": [],
            "agent_errors": [],
            "scope_loaded": bool(scope_payload),
            "scope_path": self._active_scope_path,
            "skip_reason": skip_reason,
            "candidate_gap_events": _listify(task_build.get("candidate_gap_events")),
            "gap_guidance_records": [],
            "llm_interaction_events": [],
            "llm_interaction_count": 0,
            "llm_interaction_time_total_sec": 0.0,
            "pilot_result": {
                "enabled": True,
                "executed": False,
                "reason": skip_reason,
            },
        }

    def _load_tasks_for_round(self) -> Dict[str, Any]:
        if self.explicit_materialization_tasks_path:
            tasks_payload = _read_json(self.explicit_materialization_tasks_path)
            tasks = [item for item in _listify(tasks_payload.get("tasks")) if isinstance(item, dict)]
            if tasks:
                task_build = {
                    "enabled": True,
                    "source": "target_focus_runtime_task_builder_v1",
                    "status": "ok",
                    "reason": "explicit_materialization_tasks",
                    "task_count": len(tasks),
                    "tasks": tasks,
                    "review_queue_count": len(_listify(tasks_payload.get("review_queue"))),
                    "rejected_queue_count": len(_listify(tasks_payload.get("rejected_queue"))),
                    "used_inputs": ["explicit_materialization_tasks"],
                    "missing_inputs": [] if os.path.exists(self.explicit_materialization_tasks_path) else ["explicit_materialization_tasks"],
                    "debug_artifacts_written": True,
                    "scope_count": 0,
                    "request_count": 0,
                    "plan_count": 0,
                    "scope_path": self.candidate_scope_path,
                    "tasks_path": self.explicit_materialization_tasks_path,
                    "candidate_gap_event_count": 0,
                    "candidate_gap_events": [],
                }
                return {
                    "tasks": tasks if os.path.exists(self.explicit_materialization_tasks_path) else [],
                    "candidate_gap_events": [],
                    "tasks_path": self.explicit_materialization_tasks_path,
                    "scope_path": self.candidate_scope_path,
                    "task_build": task_build,
                    "cleanup_builder": None,
                }

        builder = TFIRuntimeTaskBuilder(
            output_dir=self.output_dir,
            max_tasks=self.max_total_candidates,
            debug_artifacts=self.debug_artifacts,
            fresh_run=self.fresh_run,
            ignore_precomputed_paths=self.ignore_precomputed_paths,
            ignore_precomputed_tfi_artifacts=self.ignore_precomputed_tfi_artifacts,
            run_start_time=self.run_start_time,
        )
        result = builder.build_runtime_tasks()
        tasks = [item for item in _listify(result.get("tasks")) if isinstance(item, dict)]
        candidate_gap_events = [item for item in _listify(result.get("candidate_gap_events")) if isinstance(item, dict)]
        scope_path = _text(result.get("scope_path")).strip() or self.candidate_scope_path
        tasks_path = _text(result.get("tasks_path")).strip() or None
        return {
            "tasks": tasks,
            "candidate_gap_events": candidate_gap_events,
            "tasks_path": tasks_path,
            "scope_path": scope_path,
            "task_build": result,
            "cleanup_builder": builder,
        }

    def _build_candidates_with_fallback(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        temp_path = os.path.join(self.output_dir, "__target_focus_agent_runtime_tasks.json")
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump({"tasks": tasks}, handle, ensure_ascii=True, indent=2)
            pilot = TFICandidatePilot(
                output_dir=self.output_dir,
                materialization_tasks_path=temp_path,
                candidate_scope_path=self._active_scope_path or self.candidate_scope_path,
                path_progress_path=self.path_progress_path,
                ksi_evidence_path=self.ksi_evidence_path,
                selected_guided_sequences_path=self.selected_guided_sequences_path,
                max_candidates_per_task=self.max_candidates_per_task,
                max_total_candidates=self.max_total_candidates,
                execute=False,
                gas_limit=self.gas_limit,
                contract_name=self.contract_name,
                source_path=self.source_path,
            )
            built = pilot.build_candidates()
            return [item for item in _listify(built.get("candidates")) if isinstance(item, dict)]
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    def _build_candidates_with_agent(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        llm_called = False
        llm_success_count = 0
        llm_error_count = 0
        error = ""
        llm_interaction_events: List[Dict[str, Any]] = []

        if self.phase_cost_env is not None:
            try:
                from fuzzer.main import _llm_runtime_allowed  # lazy import to avoid import cycle at module load time

                if not _llm_runtime_allowed(self.phase_cost_env, getattr(self.phase_cost_env, "args", None)):
                    return {
                        "accepted": [],
                        "rejected": [],
                        "llm_called": False,
                        "llm_success_count": 0,
                        "llm_error_count": 0,
                        "error": "llm_warmup_disabled",
                        "llm_interaction_events": [],
                    }
            except Exception:
                pass

        if not self.llm_provider:
            return {
                "accepted": [],
                "rejected": [],
                "llm_called": False,
                "llm_success_count": 0,
                "llm_error_count": 1,
                "error": "agent_provider_unavailable",
                "llm_interaction_events": [],
            }
        if not self.llm_provider.is_available():
            return {
                "accepted": [],
                "rejected": [],
                "llm_called": False,
                "llm_success_count": 0,
                "llm_error_count": 1,
                "error": "agent_provider_unavailable",
                "llm_interaction_events": [],
            }
        if not self.llm_provider.is_configured():
            return {
                "accepted": [],
                "rejected": [],
                "llm_called": False,
                "llm_success_count": 0,
                "llm_error_count": 1,
                "error": "agent_provider_not_configured",
                "llm_interaction_events": [],
            }

        for task in tasks:
            llm_started_at = time.time()
            llm_called = True
            try:
                payload = self.llm_provider.generate_json(self._build_agent_prompt(task))
                llm_success_count += 1
                llm_interaction_events.append(
                    _build_llm_timing_event(
                        phase="tfi_materialization",
                        started_at=llm_started_at,
                        ended_at=time.time(),
                        success=True,
                        metadata={
                            "task_id": _text(task.get("task_id")).strip(),
                            "task_type": _text(task.get("task_type")).strip(),
                            **self.llm_provider.get_last_response_metadata(),
                        },
                    )
                )
                if self.phase_cost_env is not None:
                    try:
                        record_phase_event(self.phase_cost_env, llm_interaction_events[-1])
                    except Exception:
                        pass
            except LLMProviderError as exc:
                llm_error_count += 1
                error = str(exc)
                llm_interaction_events.append(
                    _build_llm_timing_event(
                        phase="tfi_materialization",
                        started_at=llm_started_at,
                        ended_at=time.time(),
                        success=False,
                        error=error,
                        fallback_used=False,
                        metadata={
                            "task_id": _text(task.get("task_id")).strip(),
                            "task_type": _text(task.get("task_type")).strip(),
                        },
                    )
                )
                if self.phase_cost_env is not None:
                    try:
                        record_phase_event(self.phase_cost_env, llm_interaction_events[-1])
                    except Exception:
                        pass
                break
            except Exception as exc:
                llm_error_count += 1
                error = "agent_runtime_error: {}".format(str(exc))
                llm_interaction_events.append(
                    _build_llm_timing_event(
                        phase="tfi_materialization",
                        started_at=llm_started_at,
                        ended_at=time.time(),
                        success=False,
                        error=error,
                        fallback_used=False,
                        metadata={
                            "task_id": _text(task.get("task_id")).strip(),
                            "task_type": _text(task.get("task_type")).strip(),
                        },
                    )
                )
                if self.phase_cost_env is not None:
                    try:
                        record_phase_event(self.phase_cost_env, llm_interaction_events[-1])
                    except Exception:
                        pass
                break

            for raw_candidate in _listify(payload.get("candidates")):
                normalized = self._normalize_agent_candidate(task, raw_candidate)
                if normalized is None:
                    rejected.append(
                        {
                            "source_task_id": task.get("task_id"),
                            "reject_reason": self._last_error or "agent_output_invalid",
                            "raw_candidate": raw_candidate,
                        }
                    )
                    continue
                accepted.append(normalized)

        return {
            "accepted": accepted[: self.max_total_candidates],
            "rejected": rejected,
            "llm_called": llm_called,
            "llm_success_count": llm_success_count,
            "llm_error_count": llm_error_count,
            "error": error,
            "llm_interaction_events": llm_interaction_events,
        }

    def _reference_input_dir(self) -> str:
        for path in (self.explicit_materialization_tasks_path, self.candidate_scope_path, self.output_dir):
            if not path:
                continue
            if os.path.isfile(path):
                return os.path.dirname(os.path.abspath(path))
            if os.path.isdir(path):
                return os.path.abspath(path)
        return ""

    def _select_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not tasks:
            return []
        if len(tasks) <= self.max_tasks_per_round:
            self._task_cursor = 0
            return list(tasks)
        selected: List[Dict[str, Any]] = []
        start = self._task_cursor % len(tasks)
        for offset in range(self.max_tasks_per_round):
            selected.append(tasks[(start + offset) % len(tasks)])
        self._task_cursor = (start + self.max_tasks_per_round) % len(tasks)
        return selected

    def _select_gap_events(self, gap_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not gap_events:
            return []
        if len(gap_events) <= self.max_tasks_per_round:
            return list(gap_events)
        return list(gap_events[: self.max_tasks_per_round])

    def _build_gap_guidance_prompt(self, gap_event: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "guidance_type": "candidate_gap_repair_guidance",
                "goal": "Provide repair guidance only; do not output a final executable candidate.",
                "constraints": {
                    "do_not_generate_final_candidate": True,
                    "do_not_create_new_pool": True,
                    "do_not_use_nonlocal_generic_skeleton": True,
                    "must_preserve_observed_prefix": True,
                    "must_touch_frontier_or_sink": True,
                },
                "gap_event": gap_event,
                "output_schema": {
                    "guidance_type": "candidate_gap_repair_guidance",
                    "preferred_candidate_roles": [
                        "prefix_preserving_candidate",
                        "frontier_touching_candidate",
                        "state_setup_candidate",
                        "binding_repair_candidate",
                    ],
                    "binding_hints": {
                        "sender_policy_used": "string",
                        "value_policy_used": "string",
                        "args_policy_used": "string",
                        "prestate_policy_used": "string",
                    },
                    "avoid_patterns": ["string"],
                    "negative_evidence": ["string"],
                    "do_not_generate_final_candidate": True,
                },
            },
            ensure_ascii=True,
        )

    def _resolve_gap_guidance(
        self,
        gap_event: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, bool, str, bool, bool, Optional[Dict[str, Any]]]:
        fallback_guidance = self.gap_repair.build_guidance(gap_event)
        if self.phase_cost_env is not None:
            try:
                from fuzzer.main import _llm_runtime_allowed  # lazy import to avoid import cycle at module load time

                if not _llm_runtime_allowed(self.phase_cost_env, getattr(self.phase_cost_env, "args", None)):
                    return fallback_guidance, False, True, "llm_warmup_disabled", False, False, None
            except Exception:
                pass
        if not self.enable_agent:
            return fallback_guidance, False, True, "", False, False, None

        if not self.llm_provider or not self.llm_provider.is_available() or not self.llm_provider.is_configured():
            return fallback_guidance, True, True, "agent_provider_not_configured", False, True, None

        llm_started_at = time.time()
        try:
            payload = self.llm_provider.generate_json(self._build_gap_guidance_prompt(gap_event))
            guidance = self.gap_repair.build_guidance(
                gap_event,
                agent_guidance=payload if isinstance(payload, dict) else None,
            )
            if isinstance(payload, dict):
                guidance["agent_guidance_payload"] = payload
            return (
                guidance,
                True,
                False,
                "",
                True,
                False,
                _build_llm_timing_event(
                    phase="tfi_gap_guidance",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=True,
                    metadata={
                        "scope_id": _text(gap_event.get("scope_id")).strip(),
                        "gap_event_id": _text(gap_event.get("gap_event_id")).strip(),
                        "gap_type": _text(gap_event.get("gap_type")).strip(),
                        **self.llm_provider.get_last_response_metadata(),
                    },
                ),
            )
        except LLMProviderError as exc:
            error = str(exc)
            return (
                fallback_guidance,
                True,
                True,
                error,
                False,
                True,
                _build_llm_timing_event(
                    phase="tfi_gap_guidance",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=False,
                    error=error,
                    fallback_used=True,
                    metadata={
                        "scope_id": _text(gap_event.get("scope_id")).strip(),
                        "gap_event_id": _text(gap_event.get("gap_event_id")).strip(),
                        "gap_type": _text(gap_event.get("gap_type")).strip(),
                    },
                ),
            )
        except Exception as exc:
            error = "agent_runtime_error: {}".format(str(exc))
            return (
                fallback_guidance,
                True,
                True,
                error,
                False,
                True,
                _build_llm_timing_event(
                    phase="tfi_gap_guidance",
                    started_at=llm_started_at,
                    ended_at=time.time(),
                    success=False,
                    error=error,
                    fallback_used=True,
                    metadata={
                        "scope_id": _text(gap_event.get("scope_id")).strip(),
                        "gap_event_id": _text(gap_event.get("gap_event_id")).strip(),
                        "gap_type": _text(gap_event.get("gap_type")).strip(),
                    },
                ),
            )

    def _build_candidates_from_gap_events(self, gap_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        guidances: List[Dict[str, Any]] = []
        errors: List[str] = []
        agent_called = False
        llm_called = False
        fallback_used = False
        llm_success_count = 0
        llm_error_count = 0
        llm_interaction_events: List[Dict[str, Any]] = []

        for gap_event in gap_events:
            guidance, gap_agent_called, gap_fallback_used, error, gap_llm_success, gap_llm_error, llm_event = self._resolve_gap_guidance(gap_event)
            agent_called = agent_called or gap_agent_called
            fallback_used = fallback_used or gap_fallback_used
            if gap_llm_success:
                llm_success_count += 1
            if gap_llm_error:
                llm_error_count += 1
            if isinstance(llm_event, dict):
                llm_interaction_events.append(llm_event)
                if self.phase_cost_env is not None:
                    try:
                        record_phase_event(self.phase_cost_env, llm_event)
                    except Exception:
                        pass
            if error:
                errors.append(error)
            if self.strict_llm_only and gap_fallback_used:
                rejected.append(
                    {
                        "source_task_id": _text(gap_event.get("gap_event_id")).strip() or _text(gap_event.get("scope_id")).strip(),
                        "reject_reason": "strict_llm_only_gap_guidance_unavailable",
                        "raw_candidate": gap_event,
                    }
                )
                continue
            if not isinstance(guidance, dict):
                rejected.append(
                    {
                        "source_task_id": _text(gap_event.get("gap_event_id")).strip() or _text(gap_event.get("scope_id")).strip(),
                        "reject_reason": "gap_guidance_invalid",
                        "raw_candidate": gap_event,
                    }
                )
                continue
            if isinstance(llm_event, dict):
                llm_called = True
            guidances.append(
                {
                    "gap_event_id": _text(gap_event.get("gap_event_id")).strip(),
                    "scope_id": _text(gap_event.get("scope_id")).strip(),
                    "gap_type": _text(gap_event.get("gap_type")).strip() or "local_candidate_gap",
                    "llm_called": bool(gap_agent_called and not gap_fallback_used),
                    "fallback_used": bool(gap_fallback_used),
                    "guidance": guidance,
                }
            )
            built = self.gap_repair.generate_candidates_from_gap(gap_event, guidance=guidance)
            if not built:
                rejected.append(
                    {
                        "source_task_id": _text(gap_event.get("gap_event_id")).strip() or _text(gap_event.get("scope_id")).strip(),
                        "reject_reason": "gap_candidate_generation_failed",
                        "raw_candidate": gap_event,
                    }
                )
                continue
            candidates.extend(built)

        return {
            "agent_called": bool(agent_called),
            "llm_called": bool(llm_called),
            "llm_success_count": int(llm_success_count),
            "llm_error_count": int(llm_error_count),
            "gap_llm_success_count": int(llm_success_count),
            "gap_llm_error_count": int(llm_error_count),
            "fallback_used": bool(fallback_used or not self.enable_agent),
            "guidance_generated_count": len(guidances),
            "fuzzer_generated_candidate_count": len(candidates),
            "candidate_origin": "tfi_guided_candidate_gap_repair" if candidates else "candidate_gap_only",
            "candidates": candidates,
            "rejected": rejected,
            "guidances": guidances,
            "error": ";".join(errors),
            "llm_interaction_events": llm_interaction_events,
        }

    def _build_agent_prompt(self, task: Dict[str, Any]) -> str:
        task_type = _text(task.get("task_type")).strip()
        if task_type == "binding_materialization_task":
            return json.dumps(
                {
                    "task_type": task_type,
                    "goal": "Preserve the exact function skeleton and suggest only sender/value/args/prestate materialization choices.",
                    "constraints": {
                        "sequence_skeleton": _clean_list(task.get("fixed_sequence_shape", {}).get("sequence_skeleton")),
                        "do_not_change_function_order": True,
                        "do_not_add_function": True,
                        "do_not_drop_function": True,
                        "allowed_fields": ["sender", "value", "args", "prestate"],
                    },
                    "task": task,
                    "output_schema": {
                        "candidates": [
                            {
                                "candidate_type": "binding_pilot_candidate",
                                "sequence_skeleton": ["..."],
                                "binding_choices": {
                                    "sender_policy_used": "string",
                                    "value_policy_used": "string",
                                    "args_policy_used": "string",
                                    "prestate_policy_used": "string",
                                },
                            }
                        ]
                    },
                },
                ensure_ascii=True,
            )
        return json.dumps(
            {
                "task_type": task_type,
                "goal": "Preserve the observed prefix and propose only scope-local candidates that touch frontier or sink.",
                "constraints": {
                    "required_prefix": _clean_list(task.get("required_prefix", {}).get("observed_matched_prefix")),
                    "must_touch_frontier_or_sink": True,
                    "frontier": _text(task.get("frontier")).strip(),
                    "sink": _text(task.get("sink")).strip(),
                    "must_not_use_nonlocal_generic_skeleton": True,
                    "must_not_select_target_seen_without_path_progress": True,
                },
                "task": task,
                "output_schema": {
                    "candidates": [
                        {
                            "candidate_type": "scoped_selection_pilot_candidate",
                            "candidate_sequence": ["..."],
                        }
                    ]
                },
            },
            ensure_ascii=True,
        )

    def _normalize_agent_candidate(self, task: Dict[str, Any], raw_candidate: Any) -> Optional[Dict[str, Any]]:
        self._last_error = ""
        if not isinstance(raw_candidate, dict):
            self._last_error = "agent_output_not_dict"
            return None
        task_type = _text(task.get("task_type")).strip()
        if task_type == "binding_materialization_task":
            return self._normalize_binding_agent_candidate(task, raw_candidate)
        if task_type == "scoped_candidate_materialization_task":
            return self._normalize_scoped_agent_candidate(task, raw_candidate)
        self._last_error = "agent_task_type_unsupported"
        return None

    def _normalize_binding_agent_candidate(self, task: Dict[str, Any], raw_candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sequence_skeleton = _clean_list(task.get("fixed_sequence_shape", {}).get("sequence_skeleton"))
        candidate_sequence = _clean_list(raw_candidate.get("sequence_skeleton")) or _clean_list(raw_candidate.get("guided_sequence"))
        if candidate_sequence != sequence_skeleton:
            self._last_error = "agent_violated_binding_constraints"
            return None
        candidate = {
            "candidate_id": self._next_agent_candidate_id(task, prefix="BIND"),
            "candidate_type": "binding_pilot_candidate",
            "source_task_id": task.get("task_id"),
            "scope_id": task.get("scope_id"),
            "hypothesis_id": task.get("hypothesis_id"),
            "target_function": task.get("target_function"),
            "frontier": task.get("frontier"),
            "sink": task.get("sink"),
            "sequence_skeleton": list(sequence_skeleton),
            "guided_sequence": list(sequence_skeleton),
            "preserve_function_order": True,
            "binding_choices": {
                "sender_policy_used": _text(raw_candidate.get("binding_choices", {}).get("sender_policy_used")).strip() or "keep_same_sender_across_sequence",
                "value_policy_used": _text(raw_candidate.get("binding_choices", {}).get("value_policy_used")).strip() or "keep_observed_success_value",
                "args_policy_used": _text(raw_candidate.get("binding_choices", {}).get("args_policy_used")).strip() or "abi_default_or_boundary_mix",
                "prestate_policy_used": _text(raw_candidate.get("binding_choices", {}).get("prestate_policy_used")).strip() or "none",
            },
            "candidate_constraints": {
                "do_not_change_function_order": True,
                "do_not_add_function": True,
                "do_not_drop_function": True,
                "only_bind_sender_value_args_prestate": True,
            },
            "is_executable_candidate": bool(sequence_skeleton),
            "materialization_status": "ready" if sequence_skeleton else "failed",
            "materialization_warnings": [],
            "reason": "agent_candidate_preserved_binding_constraints",
            "candidate_source_used": "tfi_agent_llm",
        }
        return candidate

    def _normalize_scoped_agent_candidate(self, task: Dict[str, Any], raw_candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        required_prefix = _clean_list(task.get("required_prefix", {}).get("observed_matched_prefix"))
        candidate_sequence = _clean_list(raw_candidate.get("candidate_sequence")) or _clean_list(raw_candidate.get("guided_sequence"))
        frontier = _text(task.get("frontier")).strip()
        sink = _text(task.get("sink")).strip()
        nonlocal_tuples = self._nonlocal_tuples(task)
        if not candidate_sequence or candidate_sequence[: len(required_prefix)] != required_prefix:
            self._last_error = "agent_violated_scope_constraints"
            return None
        if frontier and frontier not in candidate_sequence and sink and sink not in candidate_sequence:
            self._last_error = "agent_violated_scope_constraints"
            return None
        if tuple(candidate_sequence) in nonlocal_tuples:
            self._last_error = "agent_violated_scope_constraints"
            return None
        candidate = {
            "candidate_id": self._next_agent_candidate_id(task, prefix="SCOPED"),
            "candidate_type": "scoped_selection_pilot_candidate",
            "source_task_id": task.get("task_id"),
            "scope_id": task.get("scope_id"),
            "hypothesis_id": task.get("hypothesis_id"),
            "diagnosis_type": task.get("diagnosis_type"),
            "target_function": task.get("target_function"),
            "frontier": task.get("frontier"),
            "sink": task.get("sink"),
            "required_prefix": list(required_prefix),
            "candidate_sequence": list(candidate_sequence),
            "guided_sequence": list(candidate_sequence),
            "candidate_shape": {
                "preserve_observed_prefix": True,
                "must_touch_frontier_or_sink": True,
                "frontier": task.get("frontier"),
                "sink": task.get("sink"),
            },
            "candidate_constraints": {
                "must_preserve_observed_prefix": bool(required_prefix),
                "must_touch_frontier_or_sink": True,
                "must_not_use_nonlocal_generic_skeleton": True,
                "must_not_select_target_seen_without_path_progress": True,
                "scope_local_action_composition": _text(task.get("diagnosis_type")).strip() == "possible_local_candidate_gap",
            },
            "candidate_source_used": "tfi_agent_llm",
            "is_executable_candidate": bool(candidate_sequence),
            "materialization_status": "ready" if candidate_sequence else "failed",
            "materialization_warnings": [],
            "reason": "agent_candidate_preserved_scope_constraints",
        }
        return candidate

    def _next_agent_candidate_id(self, task: Dict[str, Any], prefix: str) -> str:
        self._agent_candidate_counter += 1
        scope_id = _text(task.get("scope_id")).strip() or "NONE"
        task_id = _text(task.get("task_id")).strip() or "TFI_TASK"
        return "TFI_AGENT_CAND_{:04d}_{}_{}_{}".format(
            self._agent_candidate_counter,
            prefix,
            self._safe_token(scope_id),
            self._safe_token(task_id),
        )

    @staticmethod
    def _safe_token(value: Any) -> str:
        text = _text(value).strip()
        if not text:
            return "NONE"
        output = []
        for char in text:
            output.append(char if char.isalnum() else "_")
        sanitized = "".join(output).strip("_")
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized or "NONE"

    def _nonlocal_tuples(self, task: Dict[str, Any]) -> set:
        tuples = set()
        policy = task.get("generic_skeleton_policy") if isinstance(task.get("generic_skeleton_policy"), dict) else {}
        for item in _listify(policy.get("nonlocal_skeleton_examples")):
            if not isinstance(item, dict):
                continue
            sequence = _clean_list(item.get("binding_skeleton")) or _clean_list(item.get("generic_skeleton_functions"))
            if sequence:
                tuples.add(tuple(sequence))
        return tuples

    def _write_summary(self, result: Dict[str, Any]) -> None:
        if not self.summary_path:
            return
        payload = {
            "enabled": True,
            "agent_enabled": bool(result.get("agent_enabled")),
            "soft_injection_enabled": bool(result.get("soft_injection_enabled")),
            "isolated_pilot_enabled": bool(result.get("isolated_pilot_enabled")),
            "agent_called_count": int(result.get("agent_called_count") or 0),
            "agent_error_count": int(result.get("agent_error_count") or 0),
            "candidate_gap_event_count": int(result.get("candidate_gap_event_count") or 0),
            "candidate_gap_guidance_count": int(result.get("candidate_gap_guidance_count") or 0),
            "guidance_generated_count": int(result.get("guidance_generated_count") or 0),
            "fuzzer_generated_candidate_count": int(result.get("fuzzer_generated_candidate_count") or 0),
            "candidate_origin": _text(result.get("candidate_origin")).strip() or None,
            "candidate_built_count": int(result.get("candidate_built_count") or 0),
            "candidate_executed_count": int(result.get("candidate_executed_count") or 0),
            "useful_candidate_count": int(result.get("useful_candidate_count") or 0),
            "soft_injected_count": int(result.get("soft_injected_count") or 0),
            "duplicate_skipped_count": int(result.get("duplicate_skipped_count") or 0),
            "task_build_attempted": bool(result.get("task_build_attempted")),
            "task_build_status": _text(result.get("task_build", {}).get("status")).strip() if isinstance(result.get("task_build"), dict) else None,
            "task_count": int(result.get("task_count") or 0),
            "task_build_reason": _text(result.get("task_build", {}).get("reason")).strip() if isinstance(result.get("task_build"), dict) else _text(result.get("skip_reason")).strip() or None,
            "fallback_used": bool(result.get("fallback_used")),
            "skip_reason": _text(result.get("skip_reason")).strip() or None,
            "tasks_seen": int(result.get("tasks_seen") or 0),
            "candidate_gap_events_seen": int(result.get("candidate_gap_events_seen") or 0),
            "llm_success_count": int(result.get("llm_success_count") or 0),
            "llm_error_count": int(result.get("llm_error_count") or 0),
            "no_main_search_replacement": True,
            "no_generator_mutation_crossover_selection_change": True,
            "llm_called": bool(result.get("llm_called")),
            "rag_used": False,
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.summary_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            pass


class _TFIRepairPilotInjector:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def inject_candidates(self, candidates: List[Dict[str, Any]], population: Any, generation: int) -> Dict[str, Any]:
        if not candidates or population is None:
            return {"injected_count": 0, "injected_candidate_ids": []}
        temp_path = os.path.join(self.output_dir, "__target_focus_soft_injection_selected.json")
        payload = {"selected": []}
        for index, candidate in enumerate(candidates, start=1):
            payload["selected"].append(
                {
                    "candidate_id": _text(candidate.get("candidate_id")).strip(),
                    "selected_rank": index,
                    "selection_score": float(index),
                    "hypothesis_id": _text(candidate.get("hypothesis_id")).strip() or None,
                    "diagnosis_type": "target_focus_soft_injection",
                    "sequence_strategy": (
                        _text(candidate.get("sequence_strategy")).strip()
                        or "target_focus_soft_injection"
                    ),
                    "target_function": _text(candidate.get("target_function")).strip() or None,
                    "guided_sequence": _clean_list(candidate.get("guided_sequence")),
                    "sender_policy": _text(candidate.get("binding_choices", {}).get("sender_policy_used")).strip() or "any_existing_sender",
                    "value_policy": _text(candidate.get("binding_choices", {}).get("value_policy_used")).strip() or "keep_original",
                    "argument_policy": _text(candidate.get("binding_choices", {}).get("args_policy_used")).strip() or "unknown",
                    "prestate_policy": _text(candidate.get("binding_choices", {}).get("prestate_policy_used")).strip() or "none",
                    "dryrun_status": "mappable",
                    "runtime_pattern_family": (
                        _text(candidate.get("runtime_pattern_family")).strip()
                        or _text(candidate.get("target_runtime_pattern_family")).strip()
                        or None
                    ),
                    "target_runtime_pattern_family": (
                        _text(candidate.get("target_runtime_pattern_family")).strip()
                        or _text(candidate.get("runtime_pattern_family")).strip()
                        or None
                    ),
                    "source_trigger_id": _text(candidate.get("source_trigger_id") or candidate.get("failure_signature_id") or candidate.get("failure_signature")).strip() or None,
                    "source_parent_id": _text(candidate.get("source_parent_id") or candidate.get("parent_candidate_id")).strip() or None,
                    "source_scope_id": _text(candidate.get("source_scope_id") or candidate.get("scope_id") or candidate.get("source_task_id")).strip() or None,
                    "source_candidate_kind": _text(candidate.get("source_candidate_kind") or candidate.get("candidate_type") or candidate.get("repair_action")).strip() or None,
                    "candidate_origin": _text(candidate.get("candidate_origin")).strip() or "checkpoint_guided_injection_v2",
                    "candidate_source_used": _text(candidate.get("candidate_source_used")).strip() or _text(candidate.get("candidate_source")).strip() or None,
                    "candidate_source": _text(candidate.get("candidate_source")).strip() or _text(candidate.get("source")).strip() or None,
                    "source": _text(candidate.get("source")).strip() or None,
                    "do_not_bypass_normal_selection": bool(
                        candidate.get("do_not_bypass_normal_selection", True)
                    ),
                    "injection_policy": "append",
                    "modules_used": list(candidate.get("modules_used") or []),
                    "module_source": candidate.get("module_source"),
                    "target_missing_checkpoint": _text(candidate.get("target_missing_checkpoint")).strip() or None,
                    "prefix_replay": bool(candidate.get("prefix_replay")),
                    "suffix_mutation_type": _text(candidate.get("suffix_mutation_type")).strip() or None,
                    "materialization_status": candidate.get("materialization_status"),
                    "materialization_status_final": candidate.get("materialization_status_final"),
                    "has_concrete_prefix": bool(candidate.get("has_concrete_prefix")),
                    "has_concrete_suffix": bool(candidate.get("has_concrete_suffix")),
                    "sequence_materialized": bool(candidate.get("sequence_materialized")),
                    "effect_probe_appended": bool(candidate.get("effect_probe_appended") or candidate.get("probe_appended")),
                    "parent_candidate_id": _text(candidate.get("parent_candidate_id")).strip() or None,
                    "bench_id": _text(candidate.get("bench_id")).strip() or None,
                    "contract_name": _text(candidate.get("contract_name")).strip() or None,
                    "access_control_address_pool": list(candidate.get("access_control_address_pool") or []),
                    "access_control_binding_hints": dict(candidate.get("access_control_binding_hints") or {}),
                    "guidance_profile": _text(candidate.get("guidance_profile")).strip() or None,
                    "selection_reason": [
                        "soft_injection",
                        _text(candidate.get("useful_reason")).strip() or "unknown",
                    ],
                    "raw": dict(candidate),
                }
            )
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
            pilot = RepairSequencePilot(
                selected_guided_sequences_path=temp_path,
                max_injected=len(payload["selected"]),
                dry_run=False,
                output_dir=self.output_dir,
                concretization_version="v2",
                insertion_policy="append",
            )
            summary = pilot.register_injection(
                population=population,
                generator=population.indv_generator,
                generation=int(generation or 0),
            )
            injected_candidate_ids = [
                _text(item.get("candidate_id")).strip()
                for item in list(getattr(pilot, "injected_mappings", []) or [])
                if isinstance(item, dict) and _text(item.get("candidate_id")).strip()
            ]
            return {
                "injected_count": int(summary.get("injected_count") or 0),
                "injected_candidate_ids": injected_candidate_ids,
                "insertion_policy": _text(summary.get("insertion_policy")).strip() or "append",
                "bypasses_normal_population_order": bool(
                    summary.get("bypasses_normal_population_order")
                ),
            }
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

