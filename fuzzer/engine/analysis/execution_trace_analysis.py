#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import json
import psutil

from fuzzer.engine.environment import FuzzingEnvironment
from fuzzer.engine.plugin_interfaces import OnTheFlyAnalysis

from fuzzer.utils.utils import initialize_logger, convert_stack_value_to_int, convert_stack_value_to_hex, \
    normalize_32_byte_hex_address, get_function_signature_mapping
from eth._utils.address import force_bytes_to_address
from eth_utils import to_hex, to_int, int_to_big_endian, encode_hex, ValidationError, to_canonical_address, \
    to_normalized_address

from z3 import simplify, BitVec, BitVecVal, Not, Optimize, sat, unsat, unknown, is_expr
from z3.z3util import get_vars

from fuzzer.utils import settings
try:
    from fuzzer.runtime.rq1_phase import rq1_current_phase, rq1_phase_allows_feature
except ImportError:
    from fuzzer.runtime.rq1_phase import rq1_current_phase, rq1_phase_allows_feature
try:
    from fuzzer.runtime.runtime_timing import (
        budget_exhausted,
        get_current_time_metrics,
        record_wall_snapshot_if_needed,
        refresh_timing_results,
    )
except ImportError:
    from fuzzer.runtime.runtime_timing import (
        budget_exhausted,
        get_current_time_metrics,
        record_wall_snapshot_if_needed,
        refresh_timing_results,
    )
try:
    from fuzzer.runtime.local_execution_timing import (
        refresh_local_execution_timing_results,
        time_local_execution,
    )
except ImportError:
    from fuzzer.runtime.local_execution_timing import (
        refresh_local_execution_timing_results,
        time_local_execution,
    )
import json


def _text(value):
    if value in (None, ""):
        return ""
    return str(value)


def _coverage_percentage(covered, total):
    try:
        total_value = int(total or 0)
        if total_value <= 0:
            return 0.0
        return (float(int(covered or 0)) / float(total_value)) * 100.0
    except Exception:
        return 0.0


def build_branch_coverage_summary(env):
    covered = len(getattr(env, "branch_coverage", set()) or set())
    total = len(getattr(env, "overall_jumpis", []) or []) * 2
    return {
        "percentage": _coverage_percentage(covered, total),
        "covered": covered,
        "total": total,
        "covered_edges": sorted(getattr(env, "branch_coverage", set()) or set()),
        "total_jumpis": len(getattr(env, "overall_jumpis", []) or []),
    }


def _main_function_selectors(env):
    interface = getattr(env, "interface", {}) or {}
    return {
        str(selector).lower().replace("0x", "")
        for selector in interface.keys()
        if selector not in ("constructor", "fallback")
    }


def build_function_coverage_summary(env):
    main_selectors = _main_function_selectors(env)
    covered_selectors = {
        str(selector).lower().replace("0x", "")
        for selector in (getattr(env, "function_coverage", set()) or set())
        if str(selector).lower().replace("0x", "") in main_selectors
    }
    covered_names = sorted(getattr(env, "function_coverage_names", set()) or set())
    mapping = {}
    detector_executor = getattr(env, "detector_executor", None)
    if detector_executor is not None:
        mapping = getattr(detector_executor, "function_signature_mapping", {}) or {}
    covered_signatures = []
    for selector in sorted(covered_selectors):
        signature = mapping.get(selector) or mapping.get("0x" + selector)
        covered_signatures.append(signature or selector)
    return {
        "percentage": _coverage_percentage(len(covered_selectors), len(main_selectors)),
        "covered": len(covered_selectors),
        "total": len(main_selectors),
        "covered_selectors": sorted(covered_selectors),
        "covered_signatures": covered_signatures,
        "covered_names": covered_names,
    }


def attach_coverage_summaries(env):
    branch_summary = build_branch_coverage_summary(env)
    function_summary = build_function_coverage_summary(env)
    env.results["branch_coverage"] = branch_summary
    env.results["function_coverage"] = function_summary
    env.results["branch_cov"] = branch_summary.get("percentage")
    env.results["function_cov"] = function_summary.get("percentage")
    env.results["branch_covered"] = branch_summary.get("covered")
    env.results["branch_total"] = branch_summary.get("total")
    env.results["function_covered"] = function_summary.get("covered")
    env.results["function_total"] = function_summary.get("total")
    return branch_summary, function_summary


def _rq1_gei_feedback_pool_restricted(env):
    state = getattr(env, "rq1_gei_sender_pool_state", None)
    if isinstance(state, dict) and bool(state.get("enabled")):
        return bool(state.get("feedback_pool_restricted"))
    return bool(getattr(env, "rq1_gei_feedback_pool_restricted", False))


def _safe_model_value(model, variable):
    if model is None or variable is None:
        return None
    try:
        value = model[variable]
    except Exception:
        value = None
    if value is not None:
        return value
    try:
        if hasattr(variable, "arity") and variable.arity() == 0:
            expr = variable()
            value = model.eval(expr, model_completion=False)
            if value is not None and is_expr(value) and str(value) != str(expr):
                return value
    except Exception:
        pass
    return None


class ExecutionTraceAnalyzer(OnTheFlyAnalysis):
    def __init__(self, fuzzing_environment: FuzzingEnvironment) -> None:
        self.logger = initialize_logger("Analysis")
        self.env = fuzzing_environment
        self.symbolic_execution_count = 0
        self._active_repair_grace_candidate_id = None

    def setup(self, ng, engine):
        pass

    def _timeout_reached(self):
        return bool(budget_exhausted(self.env, fallback_timeout=settings.GLOBAL_TIMEOUT))

    def _repair_timeout_grace_state(self):
        tracker = getattr(self.env, "target_focus_soft_injection_analysis", None)
        runtime_getter = getattr(tracker, "_repair_runtime", None) if tracker is not None else None
        runtime = runtime_getter() if callable(runtime_getter) else None
        if runtime is None:
            return tracker, runtime, set(), False
        candidate_ids = set(getattr(runtime, "repair_execution_grace_candidate_ids", set()) or set())
        grace_active = bool(getattr(runtime, "repair_execution_grace_active", False) and candidate_ids)
        return tracker, runtime, candidate_ids, grace_active

    def _grace_candidate_id(self, individual):
        return getattr(individual, "repair_candidate_id", None) if hasattr(individual, "repair_candidate_id") else None

    def _repair_grace_execution_active(self, individual=None):
        candidate_id = _text(self._active_repair_grace_candidate_id).strip()
        if not candidate_id:
            return False
        if individual is None:
            return True
        return candidate_id == _text(self._grace_candidate_id(individual)).strip()

    def execute(self, population, engine):
        self.env.memoized_fitness.clear()
        self.env.memoized_storage.clear()
        self.env.memoized_symbolic_execution.clear()
        self.env.individual_branches.clear()

        tracker, _, grace_candidate_ids, grace_active = self._repair_timeout_grace_state()
        executed_individuals = dict()
        executed_population = []
        for individual in list(population.individuals):
            record_wall_snapshot_if_needed(self.env)
            timeout_now = self._timeout_reached()
            grace_candidate_id = self._grace_candidate_id(individual)
            grace_eligible = bool(grace_active and grace_candidate_id in grace_candidate_ids)
            if timeout_now and not grace_eligible:
                if grace_active:
                    continue
                break
            if individual.hash in executed_individuals:
                if tracker is not None:
                    try:
                        tracker.record_duplicate_skip(individual)
                    except Exception:
                        pass
                executed_population.append(executed_individuals[individual.hash])
                continue
            if tracker is not None:
                try:
                    tracker.record_execution_submission(individual)
                except Exception:
                    pass
            self._active_repair_grace_candidate_id = grace_candidate_id if grace_eligible else None
            try:
                self.execution_function(individual, self.env)
            except Exception as exc:
                if tracker is not None:
                    try:
                        tracker.record_execution_exception(individual, exc, phase="execution_function")
                    except Exception:
                        pass
                raise
            finally:
                self._active_repair_grace_candidate_id = None
            if tracker is not None:
                try:
                    tracker.record_execution_completed(individual)
                except Exception:
                    pass
            executed_individuals[individual.hash] = individual
            executed_population.append(individual)
        population.individuals = executed_population
        executed_individuals.clear()

        # Update statistic variables.
        if population.individuals:
            engine._update_statvars()

    def register_step(self, g, population, engine):
        self.env.current_generation = max(g + 1, 0)
        record_wall_snapshot_if_needed(self.env)
        self.execute(population, engine)
        refresh_timing_results(self.env)

        time_metrics = get_current_time_metrics(self.env)
        code_coverage_percentage = 0
        if len(self.env.overall_pcs) > 0:
            code_coverage_percentage = (len(self.env.code_coverage) / len(self.env.overall_pcs)) * 100
        branch_coverage_summary = build_branch_coverage_summary(self.env)
        function_coverage_summary = build_function_coverage_summary(self.env)

        msg = 'Generation number {} \t Code coverage: {:.2f}% ({}/{}) \t ' \
              'Branch coverage: {:.2f}% ({}/{}) \t Function coverage: {:.2f}% ({}/{}) \t ' \
              'Transactions: {} ({} unique, {} from cross)   \t Time: {}'.format(
            g + 1, code_coverage_percentage, len(self.env.code_coverage), len(self.env.overall_pcs),
            float(branch_coverage_summary.get("percentage") or 0.0),
            int(branch_coverage_summary.get("covered") or 0),
            int(branch_coverage_summary.get("total") or 0),
            float(function_coverage_summary.get("percentage") or 0.0),
            int(function_coverage_summary.get("covered") or 0),
            int(function_coverage_summary.get("total") or 0),
            self.env.nr_of_transactions,
            len(self.env.unique_individuals), settings.CROSS_TRANS_EXEC_COUNT,
            time_metrics.get("wall_elapsed_sec"))
        self.logger.title(msg)

        # Save to results
        if "generations" not in self.env.results:
            self.env.results["generations"] = []

        self.env.results["generations"].append({
            "generation": g + 1,
            "time": time_metrics.get("wall_elapsed_sec"),
            "fuzz_active_time": time_metrics.get("fuzz_active_elapsed_sec"),
            "total_transactions": self.env.nr_of_transactions,
            "unique_transactions": len(self.env.unique_individuals),
            "code_coverage": code_coverage_percentage,
            "branch_coverage": branch_coverage_summary.get("percentage"),
            "branch_covered": branch_coverage_summary.get("covered"),
            "branch_total": branch_coverage_summary.get("total"),
            "function_coverage": function_coverage_summary.get("percentage"),
            "function_covered": function_coverage_summary.get("covered"),
            "function_total": function_coverage_summary.get("total"),
            "cross_transactions": settings.CROSS_TRANS_EXEC_COUNT
        })
        gei_recorder = getattr(self.env, "gei_side_path_recorder", None)
        if gei_recorder is not None and bool(getattr(gei_recorder, "enabled", False)):
            try:
                gei_summary = gei_recorder.get_summary()
                self.logger.info(
                    "[GEI_CORE] round=%s enabled=%s observation_count=%s side_path_candidate_count=%s ksi_enrichment_used=%s reason=%s",
                    g + 1,
                    int(bool(gei_summary.get("enabled", False))),
                    int(gei_summary.get("observation_count") or 0),
                    int(gei_summary.get("side_path_candidate_count") or 0),
                    int(bool(gei_summary.get("ksi_enrichment_used", False))),
                    "active" if gei_summary.get("enabled", False) else "disabled",
                )
                self.logger.info(
                    "[GEI_SIDE_PATH] round=%s candidate_count=%s source=%s coverage_delta_sum=%s revert_count=%s",
                    g + 1,
                    int(gei_summary.get("side_path_candidate_count") or 0),
                    str(gei_summary.get("input_source") or "unknown"),
                    gei_summary.get("coverage_delta_sum", 0),
                    int(gei_summary.get("revert_count") or 0),
                )
            except Exception as exc:
                self.logger.warning("GEI logging failed: %s", exc)
        if self._timeout_reached():
            self.env.previous_code_coverage_length = len(self.env.code_coverage)
            return

        if len(self.env.code_coverage) == self.env.previous_code_coverage_length:  # 濡傛灉杩欎竴娆¤鐩栫巼娌℃湁澧炲姞, 鍚姩绗﹀彿鎵ц
            self.symbolic_execution(population.indv_generator, population.other_generators)
            if self.symbolic_execution_count == settings.MAX_SYMBOLIC_EXECUTION:
                del population.individuals[:]
                population.init(no_cross=True)
                self.logger.debug("Resetting population...")
                self.execute(population, engine)
                self.symbolic_execution_count = 0
            self.symbolic_execution_count += 1
        else:
            self.symbolic_execution_count = 0

        self.env.previous_code_coverage_length = len(self.env.code_coverage)

    def _stringify_value(self, value):
        if value is None:
            return ""
        try:
            return json.dumps(value, ensure_ascii=True, default=str)
        except TypeError:
            return repr(value)

    def _resolve_function_name_from_selector(self, selector):
        if selector in (None, ""):
            return "unknown"
        selector_text = str(selector).strip()
        if not selector_text:
            return "unknown"
        if selector_text in {"fallback", "constructor"}:
            return selector_text
        if "(" in selector_text and not selector_text.startswith("0x"):
            return selector_text.split("(", 1)[0]

        mapping = {}
        detector_executor = getattr(self.env, "detector_executor", None)
        if detector_executor is not None:
            mapping = getattr(detector_executor, "function_signature_mapping", {}) or {}

        signature = mapping.get(selector_text)
        if signature is None and selector_text.startswith("0x"):
            signature = mapping.get(selector_text[2:])
        if signature is None and not selector_text.startswith("0x") and len(selector_text) == 8:
            signature = mapping.get("0x" + selector_text) or mapping.get(selector_text)
        if signature:
            return str(signature).split("(", 1)[0]

        lowered = selector_text.lower().replace("0x", "")
        if len(lowered) == 8 and all(ch in "0123456789abcdef" for ch in lowered):
            return "unknown"
        return selector_text

    @staticmethod
    def _selector_key(selector):
        return str(selector or "").strip().lower().replace("0x", "")

    def _resolve_function_name_from_generators(self, selector, individual, contract_address=None):
        selector_key = self._selector_key(selector)
        if not selector_key:
            return None

        generators = []
        main_generator = getattr(individual, "generator", None)
        if main_generator is not None:
            generators.append(main_generator)
        generators.extend(getattr(individual, "other_generators", []) or [])
        if contract_address:
            generators.sort(key=lambda gen: 0 if getattr(gen, "contract", None) == contract_address else 1)

        for generator in generators:
            interface_mapper = getattr(generator, "interface_mapper", {}) or {}
            for signature, mapped_selector in interface_mapper.items():
                if self._selector_key(mapped_selector) == selector_key:
                    return str(signature).split("(", 1)[0]

            total_mapper = getattr(generator, "total_interface_mapper", {}) or {}
            for interface_mapper in total_mapper.values():
                if not isinstance(interface_mapper, dict):
                    continue
                for signature, mapped_selector in interface_mapper.items():
                    if self._selector_key(mapped_selector) == selector_key:
                        return str(signature).split("(", 1)[0]
        return None

    def _extract_transaction_function_name(self, individual, transaction_index, test):
        transaction = test.get("transaction", {}) if isinstance(test, dict) else {}

        for key in ("function", "function_name", "method", "name", "abi name", "abi_name"):
            value = None
            if isinstance(transaction, dict):
                value = transaction.get(key)
            if value in (None, "") and isinstance(test, dict):
                value = test.get(key)
            if value not in (None, ""):
                return self._resolve_function_name_from_selector(value)

        chromosome = getattr(individual, "chromosome", []) or []
        if transaction_index < len(chromosome):
            arguments = chromosome[transaction_index].get("arguments", [])
            if isinstance(arguments, list) and arguments:
                generator_name = self._resolve_function_name_from_generators(
                    arguments[0],
                    individual,
                    chromosome[transaction_index].get("contract"),
                )
                if generator_name:
                    return generator_name
                return self._resolve_function_name_from_selector(arguments[0])

        data = transaction.get("data")
        if isinstance(data, str):
            selector = data[2:10] if data.startswith("0x") else data[:8]
            if not selector:
                return "fallback"
            generator_name = self._resolve_function_name_from_generators(
                selector,
                individual,
                transaction.get("to"),
            )
            if generator_name:
                return generator_name
            return self._resolve_function_name_from_selector(selector)

        return "unknown"

    def _extract_transaction_args(self, individual, transaction_index, test):
        transaction = test.get("transaction", {}) if isinstance(test, dict) else {}

        for key in ("args", "arguments", "params", "parameters"):
            value = None
            if isinstance(transaction, dict):
                value = transaction.get(key)
            if value in (None, "") and isinstance(test, dict):
                value = test.get(key)
            if value not in (None, ""):
                return self._stringify_value(value)

        chromosome = getattr(individual, "chromosome", []) or []
        if transaction_index < len(chromosome):
            arguments = chromosome[transaction_index].get("arguments", [])
            if isinstance(arguments, list) and len(arguments) > 1:
                return self._stringify_value(arguments[1:])
        return ""

    def _extract_transaction_args_list_for_trace(self, individual, transaction_index, test):
        transaction = test.get("transaction", {}) if isinstance(test, dict) else {}
        for key in ("args", "arguments", "params", "parameters"):
            value = None
            if isinstance(transaction, dict):
                value = transaction.get(key)
            if value in (None, "") and isinstance(test, dict):
                value = test.get(key)
            if value not in (None, ""):
                return value if isinstance(value, list) else [value]
        return self._extract_tx_args_list(individual, transaction_index)

    @staticmethod
    def _hex_selector_from_transaction(transaction):
        data = transaction.get("data") if isinstance(transaction, dict) else None
        if not isinstance(data, str):
            return None
        text = data[2:] if data.startswith("0x") else data
        if not text:
            return "fallback"
        return text[:8].lower()

    @staticmethod
    def _hex_address_from_bytes(value):
        if value in (None, ""):
            return None
        try:
            if isinstance(value, str):
                return value
            return encode_hex(value)
        except Exception:
            try:
                return to_hex(value)
            except Exception:
                return str(value)

    @staticmethod
    def _safe_stack_int(instruction, offset_from_top):
        try:
            stack = instruction.get("stack") or []
            if not stack:
                return None
            return convert_stack_value_to_int(stack[offset_from_top])
        except Exception:
            return None

    @staticmethod
    def _safe_stack_hex(instruction, offset_from_top):
        try:
            stack = instruction.get("stack") or []
            if not stack:
                return None
            return "0x" + convert_stack_value_to_hex(stack[offset_from_top])
        except Exception:
            return None

    def _safe_balance(self, address):
        if address in (None, "", b""):
            return None
        try:
            if isinstance(address, str):
                target = to_canonical_address(address)
            else:
                target = address
            return self.env.instrumented_evm.get_balance(target)
        except Exception:
            return None

    def _next_audit_sequence_index(self):
        current = int(getattr(self.env, "_audit_global_sequence_counter", 0) or 0)
        setattr(self.env, "_audit_global_sequence_counter", current + 1)
        return current

    def _next_audit_tx_index(self):
        current = int(getattr(self.env, "_audit_global_tx_counter", 0) or 0)
        setattr(self.env, "_audit_global_tx_counter", current + 1)
        return current

    def _build_function_meta(self, function_name):
        metadata = {
            "name": str(function_name or "").strip(),
            "payable": False,
            "state_mutability": "nonpayable",
        }
        abi = getattr(self.env, "abi", None)
        if not isinstance(abi, list):
            return metadata
        target = str(function_name or "").strip()
        if not target:
            return metadata
        for item in abi:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function":
                continue
            name = str(item.get("name") or "").strip()
            if name != target:
                continue
            state_mutability = str(item.get("stateMutability") or "").strip() or "nonpayable"
            metadata["state_mutability"] = state_mutability
            metadata["payable"] = state_mutability == "payable"
            return metadata
        return metadata

    @staticmethod
    def _extract_result_status(result):
        error_text = None
        for attr in ("_error", "error"):
            value = getattr(result, attr, None)
            if value not in (None, ""):
                error_text = str(value)
                break

        trace = getattr(result, "trace", []) or []
        last_op = None
        if trace and isinstance(trace[-1], dict):
            last_op = trace[-1].get("op")

        revert = False
        if last_op == "REVERT":
            revert = True
        elif error_text and "revert" in error_text.lower():
            revert = True

        return {
            "success": not bool(getattr(result, "is_error", False)),
            "revert": revert,
            "revert_reason": error_text,
        }

    def _build_execution_result(self, tx_summaries, coverage_delta):
        success = True
        revert = False
        revert_reason = None
        for tx_summary in tx_summaries:
            if tx_summary.get("success") is False:
                success = False
            if tx_summary.get("revert") is True:
                revert = True
            if tx_summary.get("revert_reason") not in (None, ""):
                revert_reason = tx_summary.get("revert_reason")
        return {
            "success": success,
            "revert": revert,
            "revert_reason": revert_reason,
            "coverage_delta": coverage_delta,
        }

    def _enrich_execution_result_context(self, env, execution_result):
        if not isinstance(execution_result, dict):
            return execution_result
        try:
            execution_result["rq1_phase"] = rq1_current_phase(env, getattr(env, "args", None))
        except Exception:
            execution_result["rq1_phase"] = ""
        checkpoint_snapshot = self._checkpoint_progress_snapshot(env)
        if checkpoint_snapshot:
            execution_result["checkpoint_progress_snapshot"] = checkpoint_snapshot
            execution_result["checkpoint_current_best_level"] = checkpoint_snapshot.get("best_level")
            execution_result["checkpoint_strict_vspc"] = checkpoint_snapshot.get("strict_vspc")
            execution_result["checkpoint_next_missing_checkpoint"] = checkpoint_snapshot.get("next_missing_checkpoint")
            execution_result["checkpoint_runtime_pattern_family"] = checkpoint_snapshot.get("runtime_pattern_family")
            covered = checkpoint_snapshot.get("covered_checkpoints")
            execution_result["checkpoint_covered_count"] = len(covered) if isinstance(covered, list) else 0
        return execution_result

    def _checkpoint_progress_snapshot(self, env):
        runtime = getattr(env, "checkpoint_guided_runtime", None)
        progress = getattr(runtime, "progress", None)
        if progress is None or not hasattr(progress, "snapshot"):
            return {}
        try:
            round_id = int(getattr(env, "current_generation", 0) or 0)
        except Exception:
            round_id = 0
        try:
            snapshot = progress.snapshot(round_id, {}, "")
        except Exception:
            return {}
        if not isinstance(snapshot, dict):
            return {}
        snapshot = dict(snapshot)
        snapshot["runtime_pattern_family"] = getattr(runtime, "runtime_pattern_family", None)
        snapshot["runtime_pattern_family_provenance"] = getattr(runtime, "runtime_pattern_family_provenance", None)
        return snapshot

    def _extract_sequence_id(self, individual):
        for attr in ("id", "sequence_id", "individual_id", "generation_id"):
            value = getattr(individual, attr, None)
            if value not in (None, ""):
                return str(value)

        generation = getattr(self.env, "current_generation", None)
        individual_hash = getattr(individual, "hash", None)
        if individual_hash not in (None, ""):
            if generation not in (None, ""):
                return "G{}:{}".format(generation, individual_hash)
            return str(individual_hash)

        counter = getattr(self.env, "_path_progress_sequence_counter", 0) + 1
        self.env._path_progress_sequence_counter = counter
        return "S{}".format(counter)

    def _record_path_progress(self, sequence_id, env, function_names, tx_summaries, execution_result):
        tracker = getattr(env, "path_progress_tracker", None)
        if tracker is None or not getattr(tracker, "enabled", False):
            return []
        try:
            return tracker.record_sequence(
                sequence_id=sequence_id,
                function_names=function_names,
                tx_summaries=tx_summaries,
                execution_result=execution_result,
                elapsed_time_sec=time.time() - env.execution_begin,
                global_tx_index=int(getattr(env, "nr_of_transactions", 0)),
            )
        except Exception as exc:
            self.logger.warning("Path progress tracking failed: %s", exc)
        return []

    def _record_ksi_evidence(
        self,
        sequence_id,
        env,
        function_names,
        tx_summaries,
        progress_records,
        execution_result,
    ):
        ksi_runtime = getattr(env, "ksi_runtime", None)
        if ksi_runtime is None or not getattr(ksi_runtime, "enabled", False):
            return []
        try:
            return ksi_runtime.record(
                sequence_id=sequence_id,
                function_names=function_names,
                tx_summaries=tx_summaries,
                progress_records=progress_records,
                execution_result=execution_result,
            )
        except Exception as exc:
            self.logger.warning("KSI evidence tracking failed: %s", exc)
        return []

    def _build_gei_core_observation(
        self,
        sequence_id,
        function_names,
        progress_records,
        execution_result,
    ):
        progress_records = progress_records if isinstance(progress_records, list) else []
        execution_result = execution_result if isinstance(execution_result, dict) else {}
        best_record = None
        best_prefix_len = None

        for record in progress_records:
            if not isinstance(record, dict):
                continue
            prefix_len = record.get("matched_prefix_len")
            try:
                prefix_len = int(prefix_len or 0)
            except Exception:
                prefix_len = 0
            if best_record is None or prefix_len > best_prefix_len:
                best_record = record
                best_prefix_len = prefix_len

        target_seen = None
        sink_reached = None
        sink_function_reached = None
        frontier = None
        matched_prefix_len = None
        if best_record is not None:
            matched_prefix_len = int(best_record.get("matched_prefix_len") or 0)
            frontier = best_record.get("frontier")
            sink_reached = bool(best_record.get("sink_reached", False))
            sink_function_reached = bool(best_record.get("sink_function_reached", False))
            target_seen = bool(
                best_record.get("sink_observed", False)
                or best_record.get("sink_function_reached", False)
                or best_record.get("sink_reached", False)
            )

        return {
            "source": "gei_core_observation",
            "sequence_id": str(sequence_id or ""),
            "function_names": list(function_names or []),
            "coverage_delta": execution_result.get("coverage_delta", 0),
            "revert": execution_result.get("revert") is True,
            "success": execution_result.get("success") is True,
            "tx_count": len(function_names or []),
            "matched_prefix_len": matched_prefix_len,
            "frontier": frontier if frontier not in (None, "") else None,
            "sink_reached": sink_reached,
            "sink_function_reached": sink_function_reached,
            "target_seen": target_seen,
        }

    def _record_gei_side_paths(self, env, gei_observations, ksi_records=None):
        recorder = getattr(env, "gei_side_path_recorder", None)
        if recorder is None or not getattr(recorder, "enabled", False):
            return []
        records = []
        try:
            for observation in gei_observations or []:
                candidate = recorder.record_from_observation(
                    observation,
                    source=str((observation or {}).get("source") or "gei_core_observation"),
                    ksi_enrichment_used=False,
                )
                if candidate:
                    records.append(candidate)
            for ksi_record in ksi_records or []:
                if not isinstance(ksi_record, dict):
                    continue
                candidate = recorder.record_from_ksi(ksi_record)
                if candidate:
                    records.append(candidate)
        except Exception as exc:
            self.logger.warning("GEI side-path tracking failed: %s", exc)
        return records

    @staticmethod
    def _extract_tx_args_list(individual, transaction_index):
        chromosome = getattr(individual, "chromosome", []) or []
        if transaction_index >= len(chromosome):
            return []
        arguments = chromosome[transaction_index].get("arguments", [])
        if isinstance(arguments, list) and len(arguments) > 1:
            return list(arguments[1:])
        return []

    @staticmethod
    def _build_candidate_meta(indv):
        meta = {
            "candidate_origin": None,
            "candidate_source": "original",
            "generated_under_tfi_constraint": None,
            "soft_injected_candidate_id": None,
            "parent_candidate_id": None,
            "modules_used": [],
            "module_source": None,
            "target_missing_checkpoint": None,
            "prefix_replay": False,
            "suffix_mutation_type": None,
            "materialization_status": None,
            "materialization_status_final": None,
            "has_concrete_prefix": False,
            "has_concrete_suffix": False,
            "sequence_materialized": False,
            "effect_probe_appended": False,
        }
        if hasattr(indv, "repair_candidate_id"):
            meta["soft_injected_candidate_id"] = getattr(indv, "repair_candidate_id", None)
            meta["candidate_origin"] = "target_focus_soft_injection"
            meta["candidate_source"] = "tfi"
            meta["generated_under_tfi_constraint"] = True
        if hasattr(indv, "repair_pilot_metadata"):
            metadata = getattr(indv, "repair_pilot_metadata", {}) or {}
            if isinstance(metadata, dict):
                origin = (
                    metadata.get("candidate_source_used")
                    or metadata.get("candidate_origin")
                    or metadata.get("source")
                    or metadata.get("diagnosis_type")
                )
                meta["candidate_origin"] = meta["candidate_origin"] or origin
                source_text = str(origin or "").lower()
                if "agent" in source_text or "llm" in source_text:
                    meta["candidate_source"] = "agent"
                elif "ksi" in source_text:
                    meta["candidate_source"] = "ksi"
                elif "gei" in source_text:
                    meta["candidate_source"] = "gei"
                elif "tfi" in source_text:
                    meta["candidate_source"] = "tfi"
                modules_used = metadata.get("modules_used")
                if isinstance(modules_used, list):
                    meta["modules_used"] = list(modules_used)
                meta["module_source"] = metadata.get("module_source")
                meta["target_missing_checkpoint"] = metadata.get("target_missing_checkpoint")
                meta["prefix_replay"] = bool(metadata.get("prefix_replay"))
                meta["suffix_mutation_type"] = metadata.get("suffix_mutation_type")
                meta["materialization_status"] = metadata.get("materialization_status")
                meta["materialization_status_final"] = metadata.get("materialization_status_final") or metadata.get("materialization_status")
                meta["has_concrete_prefix"] = bool(metadata.get("has_concrete_prefix"))
                meta["has_concrete_suffix"] = bool(metadata.get("has_concrete_suffix"))
                meta["sequence_materialized"] = bool(metadata.get("sequence_materialized"))
                meta["effect_probe_appended"] = bool(metadata.get("effect_probe_appended"))
                meta["parent_candidate_id"] = metadata.get("parent_candidate_id")
        if hasattr(indv, "soft_guidance_metadata"):
            meta["candidate_origin"] = meta["candidate_origin"] or "soft_guidance"
            meta["candidate_source"] = meta["candidate_source"] or "tfi"
        return meta

    @staticmethod
    def _best_progress_record(progress_records, tx_count):
        best = None
        for record in progress_records or []:
            if not isinstance(record, dict):
                continue
            if int(record.get("matched_prefix_len") or 0) > int((best or {}).get("matched_prefix_len") or 0):
                best = record
        if best is None:
            return {}
        return best

    @staticmethod
    def _current_progress_hint(env, function_names, tx_summaries, execution_result):
        tracker = getattr(env, "path_progress_tracker", None)
        if tracker is None:
            return {}
        try:
            return tracker.build_progress_hint(
                function_names=function_names,
                tx_summaries=tx_summaries,
                execution_result=execution_result,
            )
        except Exception:
            return {}

    def execution_function(self, indv, env: FuzzingEnvironment):
        env.unique_individuals.add(indv.hash)

        # Initialize metric
        branches = {}
        indv.data_dependencies = []
        contract_address = None
        path_progress_tracker = getattr(env, "path_progress_tracker", None)
        ksi_runtime = getattr(env, "ksi_runtime", None)
        gei_side_path_recorder = getattr(env, "gei_side_path_recorder", None)
        path_progress_active = bool(getattr(path_progress_tracker, "enabled", False))
        ksi_active = bool(getattr(ksi_runtime, "enabled", False)) and rq1_phase_allows_feature(env, "ksi")
        gei_active = bool(getattr(gei_side_path_recorder, "enabled", False)) and rq1_phase_allows_feature(env, "gei")
        evidence_active = path_progress_active or ksi_active or gei_active
        starting_coverage = len(env.code_coverage) if evidence_active else 0
        function_names = []
        tx_summaries = []
        sequence_id = self._extract_sequence_id(indv)
        global_sequence_index = self._next_audit_sequence_index()
        candidate_meta = self._build_candidate_meta(indv)
        tracker = getattr(env, "target_focus_soft_injection_analysis", None)
        repair_started_logged = False
        repair_tx_count = len(getattr(indv, "solution", []) or [])

        env.detector_executor.initialize_detectors()

        for transaction_index, test in enumerate(indv.solution):
            if self._timeout_reached() and not self._repair_grace_execution_active(indv):
                break
            if not repair_started_logged and tracker is not None and hasattr(indv, "repair_candidate_id"):
                try:
                    tracker.record_execution_started(
                        indv,
                        tx_count=repair_tx_count,
                        tx_index=transaction_index,
                    )
                    repair_started_logged = True
                except Exception:
                    pass

            this_error_cross_check = True

            transaction = test["transaction"]
            tx_summary = None
            function_name = "unknown"
            tx_args_for_trace = []
            global_tx_index = self._next_audit_tx_index()
            if evidence_active:
                try:
                    function_name = self._extract_transaction_function_name(indv, transaction_index, test)
                    tx_args_for_trace = self._extract_transaction_args_list_for_trace(indv, transaction_index, test)
                    function_names.append(function_name)
                    tx_summary = {
                        "function": function_name,
                        "sender": transaction.get("from"),
                        "value": transaction.get("value", 0),
                        "args": self._extract_transaction_args(indv, transaction_index, test),
                        "global_tx_index": global_tx_index,
                        "global_sequence_index": global_sequence_index,
                        "local_tx_index": transaction_index,
                        "sequence_id": sequence_id,
                        "success": None,
                        "revert": False,
                        "revert_reason": None,
                    }
                except Exception as exc:
                    self.logger.warning("Path progress transaction summary failed: %s", exc)
                    function_names.append("unknown")
                    tx_args_for_trace = []
                    tx_summary = {
                        "function": "unknown",
                        "sender": None,
                        "value": 0,
                        "args": "",
                        "global_tx_index": global_tx_index,
                        "global_sequence_index": global_sequence_index,
                        "local_tx_index": transaction_index,
                        "sequence_id": sequence_id,
                        "success": None,
                        "revert": False,
                        "revert_reason": None,
                    }

            _function_hash = transaction["data"][:10] if transaction["data"].startswith("0x") else transaction["data"][
                                                                                                   :8]
            _function_hash = "fallback" if _function_hash == '' else _function_hash
            _array_size_indexes = dict()

            if transaction["to"] is None and contract_address is not None:
                transaction["to"] = contract_address

            if transaction["to"] is None:
                if evidence_active and tx_summary is not None:
                    tx_summary["success"] = False
                    tx_summary["revert_reason"] = "missing_target"
                    tx_summaries.append(tx_summary)
                continue


            main_contract_call = transaction.get("to") == getattr(indv.generator, "contract", None)
            normalized_function_hash = str(_function_hash or "").lower().replace("0x", "")
            if (
                main_contract_call
                and normalized_function_hash
                and normalized_function_hash not in {"constructor", "fallback"}
                and normalized_function_hash in _main_function_selectors(env)
            ):
                env.function_coverage.add(normalized_function_hash)
                resolved_function_name = self._resolve_function_name_from_selector(normalized_function_hash)
                if resolved_function_name and resolved_function_name != "unknown":
                    env.function_coverage_names.add(resolved_function_name)



            local_execution_phase = getattr(env, "local_execution_phase", "fuzz_core") or "fuzz_core"
            try:
                result = time_local_execution(
                    env,
                    phase=local_execution_phase,
                    tx_kind="transaction",
                    func=lambda: env.instrumented_evm.deploy_transaction(test),
                    metadata={
                        "sequence_id": sequence_id,
                        "global_tx_index": global_tx_index,
                        "local_tx_index": transaction_index,
                        "function_name": function_name,
                    },
                )  # Execute one local EVM transaction.
            except ValidationError as e:
                self.logger.error("Validation error in %s : %s (ignoring for now)", indv.hash, e)
                if tracker is not None and hasattr(indv, "repair_candidate_id"):
                    try:
                        tracker.record_execution_tx_result(
                            indv,
                            tx_index=transaction_index,
                            tx_count=repair_tx_count,
                            function_name=function_name,
                            success=False,
                            revert_reason=str(e),
                            event_type="validation_error",
                        )
                    except Exception:
                        pass
                if evidence_active and tx_summary is not None:
                    tx_summary["success"] = False
                    tx_summary["revert_reason"] = str(e)
                    tx_summaries.append(tx_summary)
                continue

            result_status = {}
            if evidence_active and tx_summary is not None:
                try:
                    result_status = self._extract_result_status(result)
                    tx_summary.update(result_status)
                    tx_summaries.append(tx_summary)
                except Exception as exc:
                    self.logger.warning("Path progress result summary failed: %s", exc)
            if tracker is not None and hasattr(indv, "repair_candidate_id"):
                try:
                    tracker.record_execution_tx_result(
                        indv,
                        tx_index=transaction_index,
                        tx_count=repair_tx_count,
                        function_name=function_name,
                        success=result_status.get("success"),
                        revert_reason=_text(result_status.get("revert_reason")).strip(),
                    )
                except Exception:
                    pass

            if not result.is_error and transaction["to"] == b'':
                contract_address = encode_hex(result.msg.storage_address)
                self.logger.debug("(%s - %d) Contract deployed at %s", indv.hash, transaction_index, contract_address)

            for child_computation in result.children:
                if child_computation.msg.to not in env.other_contracts:
                    continue
                if child_computation.msg.to not in env.children_code_coverage:
                    env.children_code_coverage[child_computation.msg.to] = set()
                env.children_code_coverage[child_computation.msg.to].update([x["pc"] for x in child_computation.trace])

            env.nr_of_transactions += 1

            previous_instruction = None
            previous_branch = []
            previous_branch_expression = None
            previous_branch_address = None
            previous_call_address = None
            sha3 = {}

            for i, instruction in enumerate(result.trace):
                if self._timeout_reached() and not self._repair_grace_execution_active(indv):
                    break
                main_contract_target = None
                if settings.MAIN_CONTRACT_NAME != "":
                    try:
                        main_contract_target = settings.TRANS_INFO.get(settings.MAIN_CONTRACT_NAME)
                    except Exception:
                        main_contract_target = None
                if main_contract_target not in (None, "") and main_contract_target != test["transaction"]["to"]:
                    # 瀵逛簬璺ㄥ悎绾︾殑鎯呭喌, 鏆傛椂涓嶇粺璁″叾浠栧悎绾?
                    continue

                env.symbolic_taint_analyzer.propagate_taint(instruction, contract_address)

                env.detector_executor.run_detectors(previous_instruction, instruction, env.results["errors"],
                                                    env.symbolic_taint_analyzer.get_tainted_record(index=-2), indv, env,
                                                    previous_branch,
                                                    transaction_index)


                # If constructor, we don't have to take into account the constructor inputs because they will be part of the
                # state. We don't have to compute the code coverage, because the code is not the deployed one. We don't need
                # to compute the cfg because we are on a different code. We actually don't need analyzing its traces.
                if indv.chromosome[transaction_index]["arguments"][0] == "constructor":
                    continue

                # Code coverage, 鍒ゆ柇鏄惁trace灞炰簬涓诲悎绾? 濡傛灉涓嶅睘浜? 閭ｄ箞涓嶆坊鍔犲埌code_coverage閲?
                env.code_coverage.add(hex(instruction["pc"]))

                # Dynamically build control flow graph
                if env.cfg:
                    env.cfg.execute(instruction["pc"], instruction["stack"], instruction["op"], env.visited_branches,
                                    env.results["errors"].keys())

                if previous_instruction and previous_instruction["op"] == "SHA3":
                    sha3[instruction["stack"][-1][1]] = instruction["memory"]

                elif previous_instruction and previous_instruction["op"] == "ADD":
                    if previous_instruction["stack"][-1][1] in sha3:
                        sha3[instruction["stack"][-1][1]] = sha3[previous_instruction["stack"][-1][1]]
                    if previous_instruction["stack"][-2][1] in sha3:
                        sha3[instruction["stack"][-1][1]] = sha3[previous_instruction["stack"][-2][1]]

                if instruction["op"] == "JUMPI":
                    jumpi_pc = hex(instruction["pc"])
                    if jumpi_pc not in env.visited_branches:
                        env.visited_branches[jumpi_pc] = {}
                    if jumpi_pc not in branches:
                        branches[jumpi_pc] = dict()

                    destination = convert_stack_value_to_int(instruction["stack"][-1])
                    jumpi_condition = convert_stack_value_to_int(instruction["stack"][-2])

                    if jumpi_condition == 0:
                        # don't jump, but increase pc
                        branches[jumpi_pc][hex(destination)] = False
                        branches[jumpi_pc][hex(instruction["pc"] + 1)] = True
                        env.branch_coverage.add("{}:fallthrough".format(jumpi_pc))
                    else:
                        # jump to destination
                        branches[jumpi_pc][hex(destination)] = True
                        branches[jumpi_pc][hex(instruction["pc"] + 1)] = False
                        env.branch_coverage.add("{}:taken".format(jumpi_pc))

                    env.visited_branches[jumpi_pc][jumpi_condition] = {}
                    env.visited_branches[jumpi_pc][jumpi_condition]["indv_hash"] = indv.hash
                    env.visited_branches[jumpi_pc][jumpi_condition]["chromosome"] = indv.chromosome
                    env.visited_branches[jumpi_pc][jumpi_condition]["transaction_index"] = transaction_index

                    tainted_record = env.symbolic_taint_analyzer.check_taint(instruction=instruction)
                    if tainted_record and tainted_record.stack and tainted_record.stack[-2]:
                        if jumpi_condition != 0:
                            previous_branch.append(tainted_record.stack[-2][0] != 0)
                        else:
                            previous_branch.append(tainted_record.stack[-2][0] == 0)
                        previous_branch_expression = previous_branch[-1]
                        env.visited_branches[jumpi_pc][jumpi_condition]["expression"] = previous_branch.copy()
                    else:
                        env.visited_branches[jumpi_pc][jumpi_condition]["expression"] = None
                        previous_branch_expression = None

                    previous_branch_address = jumpi_pc

                # Extract data dependencies (read-after-write)
                elif instruction["op"] == "SLOAD":
                    if instruction["stack"][-1][1] in sha3:
                        hash = instruction["stack"][-1][1]
                        while hash in sha3:
                            if len(sha3[hash]) == 64:
                                hash = sha3[hash][32:64]
                            else:
                                hash = sha3[hash]
                        storage_slot = int.from_bytes(hash, byteorder='big')
                    else:
                        storage_slot = convert_stack_value_to_int(instruction["stack"][-1])

                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                    if _function_hash not in self.env.data_dependencies:
                        self.env.data_dependencies[_function_hash] = {"read": set(), "write": set()}
                    self.env.data_dependencies[_function_hash]["read"].add(storage_slot)
                    temp_dict = settings.GLOBAL_DATA_INFO.get(indv.hash, {})
                    temp_dict_2 = temp_dict.get(transaction_index, {})
                    temp_dict_3 = temp_dict_2.get("read", set())
                    temp_dict_3.add(storage_slot)
                    temp_dict_2["read"] = temp_dict_3
                    temp_dict[transaction_index] = temp_dict_2
                    settings.GLOBAL_DATA_INFO[indv.hash] = temp_dict


                elif instruction["op"] == "SSTORE":
                    if instruction["stack"][-1][1] in sha3:
                        hash = instruction["stack"][-1][1]
                        while hash in sha3:
                            if len(sha3[hash]) == 64:
                                hash = sha3[hash][32:64]
                            else:
                                hash = sha3[hash]
                        storage_slot = int.from_bytes(hash, byteorder='big')
                    else:
                        storage_slot = convert_stack_value_to_int(instruction["stack"][-1])

                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                    if _function_hash not in self.env.data_dependencies:
                        self.env.data_dependencies[_function_hash] = {"read": set(), "write": set()}
                    self.env.data_dependencies[_function_hash]["write"].add(storage_slot)
                    temp_dict = settings.GLOBAL_DATA_INFO.get(indv.hash, {})
                    temp_dict_2 = temp_dict.get(transaction_index, {})
                    temp_dict_3 = temp_dict_2.get("write", set())
                    temp_dict_3.add(storage_slot)
                    temp_dict_2["write"] = temp_dict_3
                    temp_dict[transaction_index] = temp_dict_2
                    settings.GLOBAL_DATA_INFO[indv.hash] = temp_dict


                # If something goes wrong, we need to clean some pools
                elif instruction["op"] in ["REVERT", "INVALID", "ASSERTFAIL"]:
                    if (
                        this_error_cross_check
                        and settings.TRANS_COMP_OPEN
                        and not _rq1_gei_feedback_pool_restricted(self.env)
                        and random.randint(0, 100) <= settings.P_OPEN_CROSS
                    ):
                        hash_4_chromosome = count_hash_4_chromosome(indv.chromosome)
                        if hash_4_chromosome not in settings.TRANS_CROSS_BAD_INDVS_HASH:
                            settings.TRANS_MODE = "cross"  # 鍚姩浜ゅ弶妯″紡
                            indv_success = (
                                indv,
                                transaction_index)  # 浠?鍒皌ransaction_index-1鐨勪簨鍔￠兘鎵ц鎴愬姛浜? transaction_index鏄綋鍓嶄簨鍔?杩欎釜鎵ц澶辫触浜?                            settings.TRANS_CROSS_BAD_INDVS.append(indv_success)
                            settings.TRANS_CROSS_BAD_INDVS_HASH.add(hash_4_chromosome)
                            this_error_cross_check = False
                    if previous_branch_expression is not None and is_expr(previous_branch_expression):
                        # Only remove from pool when you are sure which variable caused the exception
                        if len(get_vars(previous_branch_expression)) == 1:
                            for var in get_vars(previous_branch_expression):
                                _str_var = str(var)
                                if _str_var.startswith("calldataload_") or str(var).startswith("calldatacopy_"):
                                    _parameter_index = int(str(var).split("_")[-1])
                                    _transaction_index = int(str(var).split("_")[-2])
                                    _function_hash = indv.chromosome[_transaction_index]["arguments"][0]
                                    _argument = indv.chromosome[_transaction_index]["arguments"][_parameter_index + 1]
                                    indv.generator.remove_argument_from_pool(_function_hash, _parameter_index,
                                                                             _argument)

                                elif _str_var.startswith("callvalue_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _amount = transaction["value"]
                                    if _amount == 0 or _amount == 1:
                                        indv.generator.remove_amount_from_pool(_function_hash, _amount)

                                elif _str_var.startswith("caller_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _caller = transaction["from"]
                                    indv.generator.remove_account_from_pool(_function_hash, _caller)

                                elif _str_var.startswith("gas_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _gas_limit = indv.chromosome[transaction_index]["gaslimit"]
                                    indv.generator.remove_gaslimit_from_pool(_function_hash, _gas_limit)

                                elif _str_var.startswith("blocknumber_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _blocknumber = indv.chromosome[transaction_index]["blocknumber"]
                                    indv.generator.remove_blocknumber_from_pool(_function_hash, _blocknumber)

                                elif _str_var.startswith("timestamp_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _timestamp = indv.chromosome[transaction_index]["timestamp"]
                                    indv.generator.remove_timestamp_from_pool(_function_hash, _timestamp)

                                elif _str_var.startswith("call_"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _var_split = str(var).split("_")
                                    _address = to_normalized_address(_var_split[2])
                                    _result = int(_var_split[3], 16)
                                    indv.generator.remove_callresult_from_pool(_function_hash, _address, _result)

                                elif _str_var.startswith("extcodesize"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _var_split = str(var).split("_")
                                    _address = to_normalized_address(_var_split[2])
                                    _size = int(_var_split[3], 16)
                                    indv.generator.remove_extcodesize_from_pool(_function_hash, _address, _size)
                                elif _str_var.startswith("returndatasize"):
                                    _function_hash = indv.chromosome[transaction_index]["arguments"][0]
                                    _var_split = str(var).split("_")
                                    _address = to_normalized_address(_var_split[2])
                                    _size = int(_var_split[3], 16)
                                    indv.generator.remove_returndatasize_from_pool(_function_hash, _address, _size)

                elif instruction["op"] == "BALANCE":
                    taint = BitVec("_".join(["balance", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] in ["CALL", "STATICCALL"]:
                    _address_as_hex = to_hex(force_bytes_to_address(
                        int_to_big_endian(convert_stack_value_to_int(result.trace[i]["stack"][-2]))))
                    if i + 1 < len(result.trace):
                        _result_as_hex = convert_stack_value_to_hex(result.trace[i + 1]["stack"][-1])
                    else:
                        _result_as_hex = ""
                    previous_call_address = _address_as_hex
                    call_type = "call"
                    if instruction["op"] == "STATICCALL":
                        call_type = "staticcall"
                    taint = BitVec("_".join(
                        [call_type, str(transaction_index), str(_address_as_hex), str(_result_as_hex),
                         str(instruction["pc"])]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "CALLER":
                    taint = BitVec("_".join(["caller", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "CALLDATALOAD":
                    input_index = convert_stack_value_to_int(instruction["stack"][-1])
                    if input_index > 0 and _function_hash in env.interface:
                        input_index = int((input_index - 4) / 32)
                        if input_index < len(env.interface[_function_hash]):
                            parameter_type = env.interface[_function_hash][input_index]
                            if '[' in parameter_type:
                                array_size_index = convert_stack_value_to_int(result.trace[i + 1]["stack"][-1]) / 32
                                _array_size_indexes[array_size_index] = input_index
                            elif "bytes" in parameter_type:
                                pass
                            else:
                                taint = BitVec("_".join(["calldataload",
                                                         str(transaction_index),
                                                         str(input_index)
                                                         ]), 256)
                                env.symbolic_taint_analyzer.introduce_taint(taint, instruction)
                        else:
                            if input_index in _array_size_indexes:
                                array_size = convert_stack_value_to_int(result.trace[i + 1]["stack"][-1])
                                taint = BitVec("_".join(["inputarraysize",
                                                         str(transaction_index),
                                                         str(_array_size_indexes[input_index])
                                                         ]), 256)
                                env.symbolic_taint_analyzer.introduce_taint(taint, instruction)
                            else:
                                pass

                elif instruction["op"] == "CALLDATACOPY":
                    destOffset = convert_stack_value_to_int(instruction["stack"][-1])
                    offset = convert_stack_value_to_int(instruction["stack"][-2])
                    array_start_index = (offset - 4) / 32
                    lenght = convert_stack_value_to_int(instruction["stack"][-3])

                    if array_start_index - 1 in _array_size_indexes:
                        taint = BitVec("_".join(["calldatacopy",
                                                 str(transaction_index),
                                                 str(_array_size_indexes[array_start_index - 1])
                                                 ]), 256)
                        env.symbolic_taint_analyzer.introduce_taint(taint, instruction)
                    else:
                        pass

                elif instruction["op"] == "CALLDATASIZE":
                    taint = BitVec("_".join(["calldatasize", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "CALLVALUE":
                    taint = BitVec("_".join(["callvalue", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "GAS":
                    taint = BitVec("_".join(["gas", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                # BLOCK Opcodes
                elif instruction["op"] == "BLOCKHASH":
                    taint = BitVec("_".join(["blockhash", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "COINBASE":
                    taint = BitVec("_".join(["coinbase", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "TIMESTAMP":
                    taint = BitVec("_".join(["timestamp", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "NUMBER":
                    taint = BitVec("_".join(["blocknumber", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "DIFFICULTY":
                    taint = BitVec("_".join(["difficulty", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "GASLIMIT":
                    taint = BitVec("_".join(["gaslimit", str(transaction_index)]), 256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "EXTCODESIZE":
                    _address_as_hex = to_hex(
                        force_bytes_to_address(
                            int_to_big_endian(convert_stack_value_to_int(result.trace[i]["stack"][-1]))))
                    if i + 1 < len(result.trace):
                        _result_as_hex = convert_stack_value_to_hex(result.trace[i + 1]["stack"][-1])
                    else:
                        _result_as_hex = ""
                    taint = BitVec(
                        "_".join(["extcodesize", str(transaction_index), str(_address_as_hex), str(_result_as_hex)]),
                        256)
                    env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                elif instruction["op"] == "RETURNDATASIZE":
                    if previous_call_address:
                        if i + 1 < len(result.trace):
                            _size = convert_stack_value_to_int(result.trace[i + 1]["stack"][-1])
                        else:
                            _size = 0
                        taint = BitVec(
                            "_".join(["returndatasize", str(transaction_index), previous_call_address, str(_size)]),
                            256)
                        env.symbolic_taint_analyzer.introduce_taint(taint, instruction)

                previous_instruction = instruction


            env.symbolic_taint_analyzer.clear_callstack()

            if self._timeout_reached() and not self._repair_grace_execution_active(indv):
                break

            try:
                env.detector_executor.analyze_transaction(
                    transaction=transaction,
                    result=result,
                    errors=env.results["errors"],
                    individual=indv,
                    mfe=env,
                    transaction_index=transaction_index,
                )
            except Exception as exc:
                self.logger.warning("Detector analyze_transaction failed: %s", exc)


            if not result.is_error and not transaction["to"]:
                contract_address = encode_hex(result.msg.storage_address)

        env.individual_branches[indv.hash] = branches
        if evidence_active:
            try:
                execution_result = self._build_execution_result(
                    tx_summaries,
                    len(env.code_coverage) - starting_coverage,
                )
                execution_result = self._enrich_execution_result_context(env, execution_result)
                sequence_id = self._extract_sequence_id(indv)
                progress_records = self._record_path_progress(
                    sequence_id,
                    env,
                    function_names,
                    tx_summaries,
                    execution_result,
                )
                ksi_records = self._record_ksi_evidence(
                    sequence_id,
                    env,
                    function_names,
                    tx_summaries,
                    progress_records,
                    execution_result,
                )
                gei_observations = []
                if gei_active:
                    gei_observations.append(
                        self._build_gei_core_observation(
                            sequence_id,
                            function_names,
                            progress_records,
                            execution_result,
                        )
                    )
                self._record_gei_side_paths(env, gei_observations, ksi_records=ksi_records)
            except Exception as exc:
                self.logger.warning("Path progress tracking failed: %s", exc)

        env.symbolic_taint_analyzer.clear_storage()
        env.instrumented_evm.restore_from_snapshot()

    def get_coverage_with_children(self, children_code_coverage, code_coverage):
        code_coverage = len(code_coverage)

        for child_cc in children_code_coverage:
            code_coverage += len(child_cc)
        return code_coverage

    def symbolic_execution(self, indv_generator, other_generators):
        if not self.env.args.constraint_solving:  # 鏄惁鍚敤绗﹀彿鎵ц妯″潡
            return

        for index, pc in enumerate(self.env.visited_branches):
            self.logger.debug("b(%d) pc : %s - visited branches : %s", index, pc, self.env.visited_branches[pc].keys())

            if len(self.env.visited_branches[pc]) != 1:  # 濡傛灉杩欎釜鍒嗘敮, 宸茬粡鏈?鏉¤矾寰勫彲浠ュ嚭鍙戜簡, 閭ｄ箞璺宠繃, 娌″繀瑕佷负鍏剁敓鎴愪簡
                continue

            branch, _d = next(iter(self.env.visited_branches[pc].items()))

            if not _d["expression"]:
                self.logger.debug("No expression for b(%d) pc : %s", index, pc)
                continue

            negated_branch = simplify(Not(_d["expression"][-1]))  # 鍙嶈浆鏈€鍚庝竴涓潯浠?
            if negated_branch in self.env.memoized_symbolic_execution:
                continue

            self.env.solver.reset()
            for expression_index in range(len(_d["expression"]) - 1):  # 闄や簡鏈€鍚庝竴涓潯浠? 鍏朵粬鐨勬潯浠跺姞鍏ュ埌绾︽潫姹傝В鍣ㄩ噷
                expression = simplify(_d["expression"][expression_index])
                self.env.solver.add(expression)
            self.env.solver.add(negated_branch)  # 灏嗗弽杞潯浠跺姞鍏ュ埌绾︽潫姹傝В鍣ㄩ噷

            check = self.env.solver.check()  # 鍒ゆ柇鏄惁婊¤冻, 濡傛灉婊¤冻浜? 璇存槑鍙嶈浆鏉′欢鍦ㄧ害鏉熸眰瑙ｅ櫒閲屽彲浠ユ弧瓒? 瀛樺湪杩欎箞涓€涓€? 璁╁彟涓€涓垎鏀垚绔? 浣嗘槸! 涓嶄竴瀹氭弧瓒崇湡瀹炴儏鍐?
            if check == sat:
                model = self.env.solver.model()

                self.logger.debug("(%s) Symbolic Solution to branch %s: %s ", _d["indv_hash"], pc,
                                  "; ".join([str(x) + " (" + str(model[x]) + ")" for x in model]))

                feedback_pool_restricted = _rq1_gei_feedback_pool_restricted(self.env)
                for variable in model:
                    variable_name = str(variable)
                    if variable_name.startswith("underflow"):
                        continue
                    if (
                        feedback_pool_restricted
                        and (
                            variable_name.startswith("balance")
                            or variable_name.startswith("blocknumber")
                            or variable_name.startswith("caller_")
                            or variable_name.startswith("calldatacopy_")
                            or variable_name.startswith("calldataload_")
                            or variable_name.startswith("callvalue_")
                            or variable_name.startswith("gas_")
                            or variable_name.startswith("inputarraysize")
                            or variable_name.startswith("timestamp")
                        )
                    ):
                        continue
                    model_value = _safe_model_value(model, variable)
                    if model_value is None:
                        self.logger.debug("Skip unconcretized symbolic variable: %s", variable)
                        continue

                    var_split = str(variable).split("_")
                    transaction_index = int(var_split[1])

                    if str(variable).startswith("balance"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        opt = Optimize()
                        for expression_index in range(len(_d["expression"]) - 1):
                            opt.add(_d["expression"][expression_index])
                        opt.add(negated_branch)
                        check = opt.check()
                        if check == sat:
                            opt_model = opt.model()
                            opt_model_value = _safe_model_value(opt_model, variable)
                            if opt_model_value is None:
                                continue
                            balance = int(opt_model_value.as_long())
                            if _d["chromosome"][transaction_index]["contract"]:
                                indv_generator.add_balance_to_pool(_function_hash,
                                                                   self.env.instrumented_evm.get_balance(
                                                                       to_canonical_address(
                                                                           _d["chromosome"][transaction_index][
                                                                               "contract"])))
                            indv_generator.add_balance_to_pool(_function_hash, balance)

                    elif str(variable).startswith("blocknumber"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        blocknumber = int(model_value.as_long())
                        indv_generator.add_blocknumber_to_pool(_function_hash,
                                                               self.env.instrumented_evm.vm.state.block_number)
                        indv_generator.add_blocknumber_to_pool(_function_hash, blocknumber)

                    elif str(variable).startswith("call_") or str(variable).startswith("staticcall_"):
                        address = to_normalized_address(var_split[2])
                        old_result = int(var_split[3], 16)
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        new_result = 1 - old_result
                        # indv_generator.add_callresult_to_pool(_function_hash, address, old_result)
                        # indv_generator.add_callresult_to_pool(_function_hash, address, new_result)

                    elif str(variable).startswith("caller_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        if model_value.as_long() > 8 and model_value.as_long() < 2 ** 160:
                            account_address = normalize_32_byte_hex_address(
                                "0x" + hex(model_value.as_long()).replace("0x", "").zfill(40))
                            if not self.env.instrumented_evm.has_account(account_address):
                                self.env.instrumented_evm.restore_from_snapshot()
                                self.env.instrumented_evm.accounts.append(
                                    self.env.instrumented_evm.create_fake_account(account_address))
                                self.env.instrumented_evm.create_snapshot()
                            indv_generator.add_account_to_pool(_function_hash,
                                                               _d["chromosome"][transaction_index]["account"])
                            indv_generator.add_account_to_pool(_function_hash, account_address)

                    elif str(variable).startswith("calldatacopy_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        parameter_index = int(var_split[2])
                        if "[" in indv_generator.interface[_function_hash][parameter_index]:
                            if indv_generator.interface[_function_hash][parameter_index].startswith("int"):
                                argument = model_value.as_signed_long()
                            elif indv_generator.interface[_function_hash][parameter_index].startswith("address"):
                                try:
                                    _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                                    argument = normalize_32_byte_hex_address(hex(model_value.as_long()))
                                    if not self.env.instrumented_evm.has_account(argument):
                                        self.env.instrumented_evm.restore_from_snapshot()
                                        self.env.instrumented_evm.accounts.append(
                                            self.env.instrumented_evm.create_fake_account(argument))
                                        self.env.instrumented_evm.create_snapshot()
                                except Exception as e:
                                    self.logger.error("(%s) [symbolic execution : calldatacopy ] %s", _function_hash,
                                                      e)
                                    continue
                            else:
                                argument = model_value.as_long()
                            indv_generator.add_argument_to_pool(_function_hash, parameter_index,
                                                                _d["chromosome"][transaction_index]["arguments"][
                                                                    parameter_index + 1])
                            indv_generator.add_argument_to_pool(_function_hash, parameter_index, argument)

                    elif str(variable).startswith("calldataload_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        parameter_index = int(var_split[2])
                        # TODO: THE SOLVER DOES NOT CONSIDER THE MAX SIZE OF THE VARIABLE
                        #   GENERATING LATER A eth_abi.exceptions.ValueOutOfBounds
                        if "[" in indv_generator.interface[_function_hash][parameter_index]:  # 濡傛灉鏄暟缁?
                            if indv_generator.interface[_function_hash][parameter_index].startswith("int"):
                                argument = model_value.as_signed_long()
                            elif indv_generator.interface[_function_hash][parameter_index].startswith("address"):
                                try:
                                    _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                                    argument = normalize_32_byte_hex_address(hex(model_value.as_long()))
                                    if not self.env.instrumented_evm.has_account(argument):
                                        self.env.instrumented_evm.restore_from_snapshot()
                                        self.env.instrumented_evm.accounts.append(
                                            self.env.instrumented_evm.create_fake_account(argument))
                                        self.env.instrumented_evm.create_snapshot()
                                except Exception as e:
                                    self.logger.error("(%s) [symbolic execution : calldataload ] %s", _function_hash, e)
                                    continue

                        elif indv_generator.interface[_function_hash][parameter_index].startswith("int"):
                            argument = model_value.as_signed_long()

                        elif indv_generator.interface[_function_hash][parameter_index] == "address":
                            try:
                                _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                                argument = to_hex(
                                    force_bytes_to_address(int_to_big_endian(int(model_value.as_long()))))
                                if not self.env.instrumented_evm.has_account(argument):
                                    self.env.instrumented_evm.restore_from_snapshot()
                                    self.env.instrumented_evm.accounts.append(
                                        self.env.instrumented_evm.create_fake_account(argument))
                                    self.env.instrumented_evm.create_snapshot()
                            except Exception as e:
                                self.logger.error("(%s) [symbolic execution : calldataload ] %s", _function_hash, e)
                                continue

                        elif indv_generator.interface[_function_hash][parameter_index] == "string":
                            argument = _d["chromosome"][transaction_index]["arguments"][parameter_index + 1]
                        elif indv_generator.interface[_function_hash][parameter_index].startswith("uint"):
                            argument = model_value.as_long()
                            bits = 256
                            if indv_generator.interface[_function_hash][parameter_index] != "uint":
                                bits = int(
                                    indv_generator.interface[_function_hash][parameter_index].replace("uint", ""))
                            base = 1 << bits
                            argument %= base
                        else:
                            argument = model_value.as_long()
                            self.env.solver.add(BitVec(str(variable), 256) != BitVecVal(0, 256))
                            for variable_2 in model:
                                if variable_2 != variable and str(variable_2).startswith("callvalue"):
                                    callvalue_index = int(str(variable_2).split("_")[1])
                                    self.env.solver.add(BitVec(str(variable_2), 256) == BitVecVal(
                                        int(_d["chromosome"][callvalue_index]["amount"]), 256))
                            check = self.env.solver.check()
                            if check == sat:
                                model = self.env.solver.model()
                                refined_model_value = _safe_model_value(model, variable)
                                if refined_model_value is not None:
                                    argument = refined_model_value.as_long()

                        indv_generator.add_argument_to_pool(_function_hash, parameter_index,
                                                            _d["chromosome"][transaction_index]["arguments"][
                                                                parameter_index + 1])
                        indv_generator.add_argument_to_pool(_function_hash, parameter_index, argument)

                    elif str(variable).startswith("callvalue_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        amount = model_value.as_long()
                        if amount > settings.ACCOUNT_BALANCE:
                            amount = settings.ACCOUNT_BALANCE
                        indv_generator.remove_amount_from_pool(_function_hash, 0)
                        indv_generator.remove_amount_from_pool(_function_hash, 1)
                        indv_generator.add_amount_to_pool(_function_hash, _d["chromosome"][transaction_index]["amount"])
                        indv_generator.add_amount_to_pool(_function_hash, amount)

                    elif str(variable).startswith("gas_"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        indv_generator.add_gaslimit_to_pool(_function_hash,
                                                            _d["chromosome"][transaction_index]["gaslimit"])
                        indv_generator.add_gaslimit_to_pool(_function_hash, model_value.as_long())

                    elif str(variable).startswith("inputarraysize"):
                        opt = Optimize()
                        for expression_index in range(len(_d["expression"]) - 1):
                            opt.add(_d["expression"][expression_index])
                        opt.add(negated_branch)
                        check = opt.check()
                        if check == sat:
                            opt_model = opt.model()
                            opt_model_value = _safe_model_value(opt_model, variable)
                            if opt_model_value is None:
                                continue
                            array_size = opt_model_value.as_long()
                            _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                            parameter_index = int(var_split[2])
                            indv_generator.add_parameter_array_size(_function_hash, parameter_index, len(
                                _d["chromosome"][transaction_index]["arguments"][parameter_index + 1]))
                            indv_generator.add_parameter_array_size(_function_hash, parameter_index, array_size)

                    elif str(variable).startswith("timestamp"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        timestamp = int(model_value.as_long())
                        indv_generator.add_timestamp_to_pool(_function_hash,
                                                             self.env.instrumented_evm.vm.state.timestamp)
                        indv_generator.add_timestamp_to_pool(_function_hash, timestamp)

                    elif str(variable).startswith("calldatasize"):
                        pass

                    elif str(variable).startswith("extcodesize"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        _address = to_normalized_address(var_split[2])
                        # indv_generator.add_extcodesize_to_pool(_function_hash, _address, int(var_split[3], 16))
                        # indv_generator.add_extcodesize_to_pool(_function_hash, _address, int(model[variable].as_long()))

                    elif str(variable).startswith("returndatasize"):
                        _function_hash = _d["chromosome"][transaction_index]["arguments"][0]
                        _address = to_normalized_address(var_split[2])
                        _size = int(var_split[3], 16)
                        # indv_generator.add_returndatasize_to_pool(_function_hash, _address, int(var_split[3], 16))
                        # indv_generator.add_returndatasize_to_pool(_function_hash, _address, int(model[variable].as_long()))

                    else:
                        self.logger.warning("Unknown symbolic variable: %s ", str(variable))

            self.env.memoized_symbolic_execution[negated_branch] = True

    def finalize(self, population, engine):
        execution_end = time.time()
        execution_delta = execution_end - self.env.execution_begin
        time_metrics = get_current_time_metrics(self.env, now=execution_end)

        self.logger.title("-----------------------------------------------------")
        msg = 'Number of generations: \t {}'.format(engine.current_generation + 1)
        self.logger.info(msg)
        msg = 'Number of transactions: \t {} ({} unique)'.format(self.env.nr_of_transactions,
                                                                 len(self.env.unique_individuals))
        self.logger.info(msg)
        wall_delta = max(float(time_metrics.get("wall_elapsed_sec") or 0.0), 1e-9)
        msg = 'Transactions per second: \t {:.0f}'.format(self.env.nr_of_transactions / wall_delta)
        self.logger.info(msg)
        code_coverage_percentage = 0
        if len(self.env.overall_pcs) > 0:
            code_coverage_percentage = (len(self.env.code_coverage) / len(self.env.overall_pcs)) * 100
        branch_coverage_summary, function_coverage_summary = attach_coverage_summaries(self.env)
        msg = 'Total code coverage: \t {:.2f}% ({}/{})'.format(code_coverage_percentage,
                                                               len(self.env.code_coverage),
                                                               len(self.env.overall_pcs))
        self.logger.info(msg)
        msg = 'Total branch coverage: \t {:.2f}% ({}/{})'.format(
            float(branch_coverage_summary.get("percentage") or 0.0),
            int(branch_coverage_summary.get("covered") or 0),
            int(branch_coverage_summary.get("total") or 0),
        )
        self.logger.info(msg)
        msg = 'Total function coverage: \t {:.2f}% ({}/{})'.format(
            float(function_coverage_summary.get("percentage") or 0.0),
            int(function_coverage_summary.get("covered") or 0),
            int(function_coverage_summary.get("total") or 0),
        )
        self.logger.info(msg)
        msg = 'Total execution time: \t {:.2f} seconds'.format(execution_delta)
        self.logger.info(msg)
        msg = 'Total memory consumption: \t {:.2f} MB'.format(
            psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        self.logger.info(msg)

        # Save to results
        self.env.results["transactions"] = {"total": self.env.nr_of_transactions,
                                            "per_second": self.env.nr_of_transactions / wall_delta}
        self.env.results["setup_transactions"] = int(getattr(self.env, "setup_transactions", 0) or 0)
        self.env.results["fuzz_transactions"] = max(
            int(self.env.nr_of_transactions or 0) - int(getattr(self.env, "setup_transactions", 0) or 0),
            0,
        )
        self.env.results["code_coverage"] = {"percentage": code_coverage_percentage,
                                             "covered": len(self.env.code_coverage),
                                             "total": len(self.env.overall_pcs),
                                             "covered_with_children": self.get_coverage_with_children(
                                                 self.env.children_code_coverage,
                                                 self.env.code_coverage),
                                             "total_with_children": self.env.len_overall_pcs_with_children,
                                             "branch_cov": branch_coverage_summary.get("percentage"),
                                             "function_cov": function_coverage_summary.get("percentage"),
                                             }
        self.env.results["execution_time"] = execution_delta
        self.env.results["execution_time_wall_sec"] = time_metrics.get("wall_elapsed_sec")
        self.env.results["execution_time_fuzz_active_sec"] = time_metrics.get("fuzz_active_elapsed_sec")
        refresh_local_execution_timing_results(self.env)
        self.env.results["memory_consumption"] = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.env.results["address_under_test"] = self.env.population.indv_generator.contract
        self.env.results["seed"] = self.env.seed

        self.env.results["cross_trans_count"] = settings.CROSS_TRANS_EXEC_COUNT
        self.env.results["taint_tracking_disabled"] = bool(getattr(self.env.symbolic_taint_analyzer, "disabled", False))
        refresh_timing_results(self.env, now=execution_end)
        try:
            tracker = getattr(self.env, "path_progress_tracker", None)
            if tracker is not None:
                path_validation_summary = tracker.get_summary()
                ksi_runtime = getattr(self.env, "ksi_runtime", None)
                if ksi_runtime is not None:
                    ksi_summary = ksi_runtime.get_summary()
                    path_validation_summary["ksi_enabled"] = bool(ksi_summary.get("enabled", False))
                    path_validation_summary["ksi_evidence_records"] = int(
                        ksi_summary.get("ksi_evidence_records") or 0
                    )
                    path_validation_summary["ksi_evidence_path"] = ksi_summary.get("ksi_evidence_path")
                self.env.results["path_validation"] = path_validation_summary
            else:
                ksi_runtime = getattr(self.env, "ksi_runtime", None)
                if ksi_runtime is not None:
                    self.env.results["ksi"] = ksi_runtime.get_summary()
        except Exception as exc:
            self.logger.warning("Path validation summary failed: %s", exc)

        self.env.results["total_op"] = list(self.env.overall_pcs)
        self.env.results["coverage_op"] = list(self.env.code_coverage)

        # Write results to file
        if self.env.args.results:
            results = {}
            if self.env.args.results.lower().endswith(".json"):
                if os.path.exists(self.env.args.results):
                    with open(self.env.args.results, 'r') as file:
                        results = json.load(file)
                results[self.env.contract_name] = self.env.results
                with open(self.env.args.results, 'w') as file:
                    json.dump(results, file)
            else:
                if os.path.exists(
                        self.env.args.results + '/' + os.path.splitext(os.path.basename(self.env.contract_name))[
                            0] + '.json'):
                    with open(self.env.args.results + '/' + os.path.splitext(os.path.basename(self.env.contract_name))[
                        0] + '.json', 'r') as file:
                        results = json.load(file)
                results[self.env.contract_name] = self.env.results
                with open(self.env.args.results + '/' + os.path.splitext(os.path.basename(self.env.contract_name))[
                    0] + '.json', 'w') as file:
                    json.dump(results, file)

        try:
            diff = list(set(self.env.code_coverage).symmetric_difference(set([hex(x) for x in self.env.overall_pcs])))
            self.logger.debug("Instructions not executed: %s", sorted(diff))
        except MemoryError:
            self.logger.warning("Skipping final uncovered-instruction diff because memory is exhausted.")


def count_hash_4_chromosome(_chromosome: list):
    value = 0
    for ch in _chromosome:
        value += hash(ch["account"])
        value += hash(ch["contract"])
        for arg in ch["arguments"]:
            try:
                value += hash(arg)
            except TypeError:
                pass
    return value




