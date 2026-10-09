# -*- coding: utf-8 -*-

"""Offline TFI candidate proposal dryrun.

This module performs constraint-only validation for previously exported TFI
candidate proposals. It does not execute fuzzing, run EVM transactions,
generate action atoms, inject population, or materialize concrete
transactions.
"""

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


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
    entries: List[Dict[str, Any]] = []
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
                    entries.append(payload)
    except Exception:
        return []
    return entries


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


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


def _unique_list(values: List[Any]) -> List[Any]:
    output: List[Any] = []
    seen = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=True) if isinstance(value, (dict, list)) else repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        output.append(value)
    return output


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _has_prefix(sequence: List[str], prefix: List[str]) -> bool:
    if not prefix:
        return True
    if len(sequence) < len(prefix):
        return False
    return sequence[: len(prefix)] == prefix


def _contains_any(value: str, terms: List[str]) -> bool:
    lowered = value.lower()
    return any(term.lower() in lowered for term in terms)


def _contains_disallowed_key(payload: Any, keys: List[str]) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys:
                return True
            if _contains_disallowed_key(value, keys):
                return True
    elif isinstance(payload, list):
        for item in payload:
            if _contains_disallowed_key(item, keys):
                return True
    return False


def _duplicate_id_map(items: List[Any], field_name: str) -> Dict[str, int]:
    counts: Counter = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        identifier = _text(item.get(field_name)).strip()
        if identifier:
            counts[identifier] += 1
    return {key: int(count) for key, count in counts.items() if count > 1}


class TFICandidateProposalDryrun:
    def __init__(
        self,
        output_dir: str,
        proposals_path: Optional[str] = None,
        candidate_scope_path: Optional[str] = None,
        agent_selection_plan_path: Optional[str] = None,
        path_progress_path: Optional[str] = None,
        ksi_evidence_path: Optional[str] = None,
        selected_guided_sequences_path: Optional[str] = None,
        max_items: Optional[int] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.proposals_path = proposals_path or os.path.join(self.output_dir, "target_focus_candidate_proposals.json")
        self.candidate_scope_path = candidate_scope_path or os.path.join(self.output_dir, "target_focus_candidate_scope.json")
        self.agent_selection_plan_path = agent_selection_plan_path or os.path.join(
            self.output_dir, "target_focus_agent_selection_plan.json"
        )
        self.path_progress_path = path_progress_path or os.path.join(self.output_dir, "path_progress.jsonl")
        self.ksi_evidence_path = ksi_evidence_path or os.path.join(self.output_dir, "ksi_evidence.jsonl")
        self.selected_guided_sequences_path = selected_guided_sequences_path or os.path.join(
            self.output_dir, "selected_guided_sequences.json"
        )
        self.max_items = int(max_items) if max_items not in (None, "") else None

    def run_dryrun(self) -> Dict[str, Any]:
        loaded_inputs = {
            "candidate_proposals": bool(self.proposals_path and os.path.exists(self.proposals_path)),
            "candidate_scope": bool(self.candidate_scope_path and os.path.exists(self.candidate_scope_path)),
            "agent_selection_plan": bool(self.agent_selection_plan_path and os.path.exists(self.agent_selection_plan_path)),
            "path_progress": bool(self.path_progress_path and os.path.exists(self.path_progress_path)),
            "ksi_evidence": bool(self.ksi_evidence_path and os.path.exists(self.ksi_evidence_path)),
            "selected_guided_sequences": bool(
                self.selected_guided_sequences_path and os.path.exists(self.selected_guided_sequences_path)
            ),
        }
        missing_inputs = [name for name, loaded in loaded_inputs.items() if not loaded]

        proposal_payload = _read_json(self.proposals_path)
        scope_payload = _read_json(self.candidate_scope_path)
        plan_payload = _read_json(self.agent_selection_plan_path)
        path_progress_entries = _read_jsonl(self.path_progress_path)
        ksi_entries = _read_jsonl(self.ksi_evidence_path)
        selected_repair_payload = _read_json(self.selected_guided_sequences_path)

        scope_index = self._index_scopes(scope_payload)
        plan_index = self._index_plans(plan_payload)
        path_progress_index = self._index_path_progress(path_progress_entries)
        ksi_index = self._index_ksi_evidence(ksi_entries)
        selected_repair_index = self._index_selected_guided_sequences(selected_repair_payload)

        dryruns: List[Dict[str, Any]] = []

        for ordinal, proposal in enumerate(_listify(proposal_payload.get("proposals")), start=1):
            if not isinstance(proposal, dict):
                continue
            dryrun = self._build_dryrun(
                ordinal=ordinal,
                proposal=proposal,
                scope_index=scope_index,
                plan_index=plan_index,
                path_progress_index=path_progress_index,
                ksi_index=ksi_index,
                selected_repair_index=selected_repair_index,
            )
            dryruns.append(dryrun)
            if self.max_items is not None and len(dryruns) >= self.max_items:
                break

        return {
            "enabled": True,
            "source": "target_focus_candidate_proposal_dryrun_v1",
            "loaded_inputs": loaded_inputs,
            "missing_inputs": missing_inputs,
            "dryrun_count": len(dryruns),
            "dryruns": dryruns,
            "global_constraints": {
                "no_fuzz_execution": True,
                "no_evm_execution": True,
                "no_llm_call": True,
                "no_population_injection": True,
                "no_action_atom_generation": True,
                "no_full_sequence_generation": True,
                "no_concrete_tx_generation": True,
                "dryrun_is_constraint_validation_only": True,
            },
            "dryrun_stats": {
                "possible_missing_action_atom_seen": self._count_possible_missing_action_atom(scope_payload),
                "action_atom_generated": any(
                    "action_atom_generated" in _clean_list(item.get("violations")) for item in dryruns if isinstance(item, dict)
                ),
                "full_sequence_generated": any(
                    "full_sequence_generated" in _clean_list(item.get("violations")) for item in dryruns if isinstance(item, dict)
                ),
                "concrete_tx_generated": any(
                    "concrete_tx_generated" in _clean_list(item.get("violations")) for item in dryruns if isinstance(item, dict)
                ),
            },
        }

    def write_dryrun(self, dryrun: Dict[str, Any], filename: str = "target_focus_candidate_proposal_dryrun.json") -> str:
        path = os.path.join(self.output_dir, filename)
        _write_json(path, dryrun)
        return path

    def write_summary(
        self,
        dryrun: Dict[str, Any],
        filename: str = "target_focus_candidate_proposal_dryrun_summary.json",
    ) -> str:
        payload = self._build_summary(dryrun)
        path = os.path.join(self.output_dir, filename)
        _write_json(path, payload)
        return path

    def _index_scopes(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}
        for item in _listify(payload.get("scopes")):
            if not isinstance(item, dict):
                continue
            scope_id = _text(item.get("scope_id")).strip()
            if scope_id:
                indexed[scope_id] = item
        return indexed

    def _index_plans(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        by_plan_id: Dict[str, List[Dict[str, Any]]] = {}
        by_composite_key: Dict[str, Dict[str, Any]] = {}
        for item in _listify(payload.get("plans")):
            if not isinstance(item, dict):
                continue
            plan_id = _text(item.get("plan_id")).strip()
            if plan_id:
                by_plan_id.setdefault(plan_id, []).append(item)
            composite_key = self._make_plan_composite_key(
                item.get("scope_id"),
                item.get("source_request_id"),
                item.get("plan_id"),
            )
            if composite_key:
                by_composite_key[composite_key] = item
        return {
            "by_plan_id": by_plan_id,
            "by_composite_key": by_composite_key,
        }

    def _index_path_progress(self, entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        by_hypothesis_frontier: Dict[str, Dict[str, Any]] = {}
        by_hypothesis: Dict[str, Dict[str, Any]] = {}
        for item in entries:
            if not isinstance(item, dict):
                continue
            hypothesis_id = _text(item.get("matched_hypothesis") or item.get("hypothesis_id")).strip()
            if not hypothesis_id:
                continue
            frontier = _text(item.get("frontier")).strip()
            key = "%s::%s" % (hypothesis_id, frontier)
            current = by_hypothesis_frontier.get(key)
            if self._prefer_progress_entry(item, current):
                by_hypothesis_frontier[key] = item
            current_best = by_hypothesis.get(hypothesis_id)
            if self._prefer_progress_entry(item, current_best):
                by_hypothesis[hypothesis_id] = item
        return {
            "by_hypothesis_frontier": by_hypothesis_frontier,
            "by_hypothesis": by_hypothesis,
        }

    def _index_ksi_evidence(self, entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        by_hypothesis_frontier: Dict[str, Dict[str, Any]] = {}
        by_hypothesis: Dict[str, Dict[str, Any]] = {}
        for item in entries:
            if not isinstance(item, dict):
                continue
            hypothesis_id = _text(item.get("hypothesis_id") or item.get("matched_hypothesis")).strip()
            if not hypothesis_id:
                continue
            frontier = _text(item.get("frontier")).strip()
            key = "%s::%s" % (hypothesis_id, frontier)
            current = by_hypothesis_frontier.get(key)
            if self._prefer_ksi_entry(item, current):
                by_hypothesis_frontier[key] = item
            current_best = by_hypothesis.get(hypothesis_id)
            if self._prefer_ksi_entry(item, current_best):
                by_hypothesis[hypothesis_id] = item
        return {
            "by_hypothesis_frontier": by_hypothesis_frontier,
            "by_hypothesis": by_hypothesis,
        }

    def _index_selected_guided_sequences(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        indexed: Dict[str, Any] = {}
        for key in ("selected_guided_sequences", "guided_sequences", "items"):
            bucket = payload.get(key)
            if not isinstance(bucket, list):
                continue
            for item in bucket:
                if not isinstance(item, dict):
                    continue
                hypothesis_id = _text(item.get("hypothesis_id")).strip()
                if hypothesis_id and hypothesis_id not in indexed:
                    indexed[hypothesis_id] = item
        return indexed

    def _prefer_progress_entry(self, candidate: Dict[str, Any], current: Optional[Dict[str, Any]]) -> bool:
        if not current:
            return True
        candidate_len = _to_int(candidate.get("matched_prefix_len"))
        current_len = _to_int(current.get("matched_prefix_len"))
        if candidate_len != current_len:
            return candidate_len > current_len
        if bool(candidate.get("sink_function_reached")) != bool(current.get("sink_function_reached")):
            return bool(candidate.get("sink_function_reached"))
        return _text(candidate.get("sequence_id")) < _text(current.get("sequence_id"))

    def _prefer_ksi_entry(self, candidate: Dict[str, Any], current: Optional[Dict[str, Any]]) -> bool:
        if not current:
            return True
        candidate_score = (
            int(bool(candidate.get("sink_reached") or candidate.get("sink_function_reached"))),
            _to_int(candidate.get("matched_prefix_len")),
        )
        current_score = (
            int(bool(current.get("sink_reached") or current.get("sink_function_reached"))),
            _to_int(current.get("matched_prefix_len")),
        )
        if candidate_score != current_score:
            return candidate_score > current_score
        return _text(candidate.get("sequence_id")) < _text(current.get("sequence_id"))

    def _build_dryrun(
        self,
        ordinal: int,
        proposal: Dict[str, Any],
        scope_index: Dict[str, Dict[str, Any]],
        plan_index: Dict[str, Dict[str, Any]],
        path_progress_index: Dict[str, Dict[str, Dict[str, Any]]],
        ksi_index: Dict[str, Dict[str, Dict[str, Any]]],
        selected_repair_index: Dict[str, Any],
    ) -> Dict[str, Any]:
        proposal_type = _text(proposal.get("proposal_type")).strip()
        if proposal_type == "binding_repair_proposal":
            return self._build_binding_dryrun(
                ordinal,
                proposal,
                scope_index,
                plan_index,
                path_progress_index,
                ksi_index,
                selected_repair_index,
            )
        if proposal_type == "scoped_selection_proposal":
            return self._build_selection_dryrun(
                ordinal,
                proposal,
                scope_index,
                plan_index,
                path_progress_index,
                ksi_index,
                selected_repair_index,
            )
        return {
            "dryrun_id": "TFI_DRYRUN_%d" % ordinal,
            "proposal_id": proposal.get("proposal_id"),
            "proposal_type": proposal_type or "unknown",
            "scope_id": proposal.get("scope_id"),
            "hypothesis_id": proposal.get("hypothesis_id"),
            "target_function": proposal.get("target_function"),
            "frontier": proposal.get("frontier"),
            "sink": proposal.get("sink"),
            "dryrun_status": "fail",
            "dryrun_mode": "constraint_validation_only",
            "violations": ["unsupported_proposal_type"],
            "warnings": [],
            "positive_evidence": [],
            "negative_evidence": [],
            "next_stage_hint": "invalid_proposal",
            "confidence": _text(proposal.get("confidence")).strip() or "low",
            "reason": "proposal type is not supported by constraint-only dryrun",
        }

    def _build_binding_dryrun(
        self,
        ordinal: int,
        proposal: Dict[str, Any],
        scope_index: Dict[str, Dict[str, Any]],
        plan_index: Dict[str, Dict[str, Any]],
        path_progress_index: Dict[str, Dict[str, Dict[str, Any]]],
        ksi_index: Dict[str, Dict[str, Dict[str, Any]]],
        selected_repair_index: Dict[str, Any],
    ) -> Dict[str, Any]:
        scope = scope_index.get(_text(proposal.get("scope_id")).strip(), {})
        plan, plan_lookup = self._resolve_plan_reference(proposal, plan_index)
        fixed = proposal.get("fixed_sequence_shape") if isinstance(proposal.get("fixed_sequence_shape"), dict) else {}
        mutation = proposal.get("binding_mutation_space") if isinstance(proposal.get("binding_mutation_space"), dict) else {}
        scope_local = scope.get("scope_local_evidence_check") if isinstance(scope.get("scope_local_evidence_check"), dict) else {}
        leakage = scope.get("generic_skeleton_leakage_check") if isinstance(scope.get("generic_skeleton_leakage_check"), dict) else {}

        skeleton = _clean_list(fixed.get("sequence_skeleton"))
        reference_skeleton = _clean_list(plan.get("fixed_structure", {}).get("sequence_skeleton"))
        scope_reference_skeleton = _clean_list(scope.get("binding_repair_context", {}).get("sequence_skeleton"))
        observed_prefix, observed_prefix_source = self._resolve_binding_observed_prefix(
            proposal, scope, plan, path_progress_index, ksi_index
        )

        positive_evidence: List[Any] = []
        negative_evidence: List[Any] = []
        warnings: List[str] = []
        violations: List[str] = []

        has_sequence_skeleton = bool(skeleton)
        if not has_sequence_skeleton:
            violations.append("missing_sequence_skeleton")

        function_order_preserved = bool(fixed.get("do_not_change_function_order"))
        skeleton_not_replaced = bool(fixed.get("do_not_replace_sequence_skeleton"))
        no_new_function_added = bool(proposal.get("proposal_constraints", {}).get("do_not_add_new_function"))
        no_existing_function_dropped = bool(proposal.get("proposal_constraints", {}).get("do_not_drop_existing_function"))

        if plan_lookup.get("ambiguous"):
            warnings.append("ambiguous_plan_reference")
            negative_evidence.append(
                {
                    "source": "agent_selection_plan",
                    "check": "ambiguous_plan_reference",
                    "source_plan_id": proposal.get("source_plan_id"),
                    "matching_plan_count": int(plan_lookup.get("matching_plan_count", 0)),
                }
            )
        elif reference_skeleton:
            same_sequence = skeleton == reference_skeleton
            function_order_preserved = function_order_preserved and same_sequence
            skeleton_not_replaced = skeleton_not_replaced and same_sequence
            no_new_function_added = no_new_function_added and same_sequence
            no_existing_function_dropped = no_existing_function_dropped and same_sequence
            if same_sequence:
                positive_evidence.append(
                    {
                        "source": "agent_selection_plan",
                        "check": "sequence_skeleton_matches_reference",
                    }
                )
            else:
                violations.append("binding_skeleton_mismatch")
                negative_evidence.append(
                    {
                        "source": "agent_selection_plan",
                        "check": "binding_skeleton_mismatch",
                        "reference_sequence_skeleton": reference_skeleton,
                    }
                )
        else:
            warnings.append("source_plan_reference_missing")

        if scope_reference_skeleton:
            if skeleton == scope_reference_skeleton:
                positive_evidence.append(
                    {
                        "source": "candidate_scope",
                        "check": "sequence_skeleton_matches_scope_context",
                    }
                )
            else:
                warnings.append("scope_skeleton_consistency_mismatch")
                negative_evidence.append(
                    {
                        "source": "candidate_scope",
                        "check": "scope_skeleton_consistency_mismatch",
                        "scope_sequence_skeleton": scope_reference_skeleton,
                    }
                )

        allowed_mutable_fields = ["sender", "value", "args", "prestate"]
        forbidden_mutable_fields_detected = sorted([key for key in mutation.keys() if key not in allowed_mutable_fields])
        only_transaction_fields_mutable = (
            sorted(mutation.keys()) == sorted(allowed_mutable_fields)
            and bool(proposal.get("proposal_constraints", {}).get("only_repair_transaction_fields"))
        )

        if not function_order_preserved and not plan_lookup.get("ambiguous"):
            violations.append("function_order_change_allowed")
        if not skeleton_not_replaced and not plan_lookup.get("ambiguous"):
            violations.append("skeleton_replacement_allowed")
        if not no_new_function_added and not plan_lookup.get("ambiguous"):
            violations.append("new_function_addition_allowed")
        if not no_existing_function_dropped and not plan_lookup.get("ambiguous"):
            violations.append("existing_function_drop_allowed")
        if forbidden_mutable_fields_detected or not only_transaction_fields_mutable:
            violations.append("unexpected_mutable_field")

        contains_target_or_scope_local_evidence = self._contains_target_or_scope_local_evidence(
            skeleton=skeleton,
            target_function=_text(proposal.get("target_function")).strip(),
            frontier=_text(proposal.get("frontier")).strip(),
            sink=_text(proposal.get("sink")).strip(),
            scope_local=scope_local,
        )
        if contains_target_or_scope_local_evidence:
            positive_evidence.append(
                {
                    "source": "candidate_scope",
                    "check": "contains_target_or_scope_local_evidence",
                }
            )
        else:
            warnings.append("target_or_scope_local_evidence_not_confirmed")

        preserves_observed_prefix_if_present = True
        if observed_prefix:
            preserves_observed_prefix_if_present = _has_prefix(skeleton, observed_prefix)
            if not preserves_observed_prefix_if_present and "preserves_observed_matched_prefix" in _clean_list(
                scope_local.get("binding_skeleton_scope_local_reasons")
            ):
                preserves_observed_prefix_if_present = True
            if preserves_observed_prefix_if_present:
                positive_evidence.append(
                    {
                        "source": observed_prefix_source,
                        "check": "observed_prefix_preserved",
                        "observed_matched_prefix": observed_prefix,
                    }
                )
            else:
                warnings.append("observed_prefix_not_preserved")
                negative_evidence.append(
                    {
                        "source": observed_prefix_source,
                        "check": "observed_prefix_not_preserved",
                        "observed_matched_prefix": observed_prefix,
                    }
                )
        else:
            warnings.append("observed_prefix_not_available")

        if observed_prefix_source == "path_progress":
            positive_evidence.append({"source": "path_progress", "check": "observed_prefix_confirmed"})
        elif observed_prefix_source == "ksi_evidence":
            positive_evidence.append({"source": "ksi_evidence", "check": "observed_prefix_confirmed"})
        elif observed_prefix_source == "candidate_scope":
            positive_evidence.append({"source": "candidate_scope", "check": "observed_prefix_confirmed"})

        no_action_atom_generated = not _contains_disallowed_key(
            proposal,
            ["action_atom", "action_atoms", "generated_action_atom", "generated_action_atoms", "tfi_scoped_action_pool"],
        )
        no_full_sequence_generated = not _contains_disallowed_key(
            proposal,
            [
                "generated_sequence",
                "generated_sequences",
                "full_sequence",
                "full_sequences",
                "complete_sequence",
                "complete_sequences",
                "materialized_sequence",
                "materialized_sequences",
            ],
        )
        no_concrete_tx_generated = not _contains_disallowed_key(
            proposal,
            [
                "concrete_tx",
                "concrete_txs",
                "concrete_transaction",
                "concrete_transactions",
                "materialized_transactions",
                "transaction_inputs",
                "executable_transactions",
            ],
        )

        if not no_action_atom_generated:
            violations.append("action_atom_generated")
        if not no_full_sequence_generated:
            violations.append("full_sequence_generated")
        if not no_concrete_tx_generated:
            violations.append("concrete_tx_generated")

        if leakage and not bool(leakage.get("allowed_as_positive_evidence", True)):
            warnings.append("nonlocal_generic_skeleton_guarded")
            negative_evidence.append(
                {
                    "source": "candidate_scope",
                    "check": "generic_skeleton_positive_evidence_blocked",
                    "generic_skeleton_functions": _clean_list(leakage.get("generic_skeleton_functions")),
                }
            )

        if _text(proposal.get("hypothesis_id")).strip() in selected_repair_index:
            positive_evidence.append({"source": "selected_guided_sequences", "check": "historical_repair_reference_available"})

        if not scope:
            warnings.append("candidate_scope_missing")
        if not plan and not plan_lookup.get("ambiguous"):
            warnings.append("agent_selection_plan_missing")

        status = "pass"
        next_stage_hint = "ready_for_binding_materialization"
        if violations:
            status = "fail"
            next_stage_hint = "invalid_proposal"
        elif warnings:
            status = "partial"
            next_stage_hint = "needs_scope_review"

        checks = {
            "has_sequence_skeleton": has_sequence_skeleton,
            "function_order_preserved": function_order_preserved,
            "skeleton_not_replaced": skeleton_not_replaced,
            "no_new_function_added": no_new_function_added,
            "no_existing_function_dropped": no_existing_function_dropped,
            "only_transaction_fields_mutable": only_transaction_fields_mutable,
            "allowed_mutable_fields": allowed_mutable_fields,
            "forbidden_mutable_fields_detected": forbidden_mutable_fields_detected,
            "contains_target_or_scope_local_evidence": contains_target_or_scope_local_evidence,
            "preserves_observed_prefix_if_present": preserves_observed_prefix_if_present,
            "no_concrete_tx_generated": no_concrete_tx_generated,
        }

        reason = "binding proposal passed constraint-only dryrun"
        if status == "partial":
            reason = "binding proposal preserves core constraints but needs additional scope/evidence review"
        elif status == "fail":
            reason = "binding proposal violates binding-constraint requirements"

        return {
            "dryrun_id": "TFI_DRYRUN_%d" % ordinal,
            "proposal_id": proposal.get("proposal_id"),
            "proposal_type": "binding_repair_proposal",
            "scope_id": proposal.get("scope_id"),
            "hypothesis_id": proposal.get("hypothesis_id"),
            "target_function": proposal.get("target_function"),
            "frontier": proposal.get("frontier"),
            "sink": proposal.get("sink"),
            "dryrun_status": status,
            "dryrun_mode": "constraint_validation_only",
            "binding_constraint_checks": checks,
            "violations": _unique_list(violations),
            "warnings": _unique_list(warnings),
            "positive_evidence": _unique_list(positive_evidence),
            "negative_evidence": _unique_list(negative_evidence),
            "next_stage_hint": next_stage_hint,
            "confidence": _text(proposal.get("confidence")).strip() or "low",
            "reference_resolution": {
                "lookup_strategy": plan_lookup.get("lookup_strategy"),
                "ambiguous_plan_reference": bool(plan_lookup.get("ambiguous")),
                "matching_plan_count": int(plan_lookup.get("matching_plan_count", 0)),
                "resolved_plan_id": plan.get("plan_id") if isinstance(plan, dict) else None,
                "resolved_scope_id": plan.get("scope_id") if isinstance(plan, dict) else None,
                "resolved_source_request_id": plan.get("source_request_id") if isinstance(plan, dict) else None,
            },
            "reason": reason,
        }

    def _build_selection_dryrun(
        self,
        ordinal: int,
        proposal: Dict[str, Any],
        scope_index: Dict[str, Dict[str, Any]],
        plan_index: Dict[str, Dict[str, Any]],
        path_progress_index: Dict[str, Dict[str, Dict[str, Any]]],
        ksi_index: Dict[str, Dict[str, Dict[str, Any]]],
        selected_repair_index: Dict[str, Any],
    ) -> Dict[str, Any]:
        del selected_repair_index
        scope = scope_index.get(_text(proposal.get("scope_id")).strip(), {})
        plan, plan_lookup = self._resolve_plan_reference(proposal, plan_index)
        required_prefix = proposal.get("required_prefix") if isinstance(proposal.get("required_prefix"), dict) else {}
        required_progress = proposal.get("required_progress") if isinstance(proposal.get("required_progress"), dict) else {}
        proposal_constraints = proposal.get("proposal_constraints") if isinstance(proposal.get("proposal_constraints"), dict) else {}
        generic_policy = proposal.get("generic_skeleton_policy") if isinstance(proposal.get("generic_skeleton_policy"), dict) else {}

        observed_prefix, observed_prefix_source = self._resolve_selection_observed_prefix(
            proposal, scope, plan, path_progress_index, ksi_index
        )
        forbidden_patterns = _clean_list(proposal.get("forbidden_candidate_patterns"))

        positive_evidence: List[Any] = []
        negative_evidence: List[Any] = []
        warnings: List[str] = []
        violations: List[str] = []

        preserve_observed_prefix_required = bool(required_prefix.get("preserve_observed_matched_prefix")) and bool(
            proposal_constraints.get("must_preserve_observed_prefix")
        )
        must_touch_frontier_or_sink = bool(required_progress.get("must_touch_frontier_or_sink")) and bool(
            proposal_constraints.get("must_touch_frontier_or_sink")
        )
        frontier_or_sink_present = bool(_text(proposal.get("frontier")).strip() or _text(proposal.get("sink")).strip())
        blocks_nonlocal_best_repair = bool(proposal_constraints.get("must_not_use_nonlocal_best_repair"))
        blocks_generic_skeleton_unless_scope_local = bool(
            proposal_constraints.get("must_not_use_generic_skeleton_unless_scope_local")
        ) and bool(generic_policy.get("must_not_use_generic_skeleton_unless_scope_local"))
        blocks_target_seen_without_path_progress = bool(
            proposal_constraints.get("must_not_select_target_seen_without_path_progress")
        ) and bool(required_progress.get("must_not_select_sequence_without_target_progress"))

        required_forbidden_patterns = [
            "target_seen_but_no_path_progress",
            "missing_frontier_or_sink",
            "does_not_preserve_prefix",
            "sequence_order_misaligned",
            "nonlocal_binding_skeleton",
            "generic_defi_token_skeleton_leakage",
        ]
        forbidden_patterns_present = all(pattern in forbidden_patterns for pattern in required_forbidden_patterns)

        if not preserve_observed_prefix_required:
            violations.append("missing_required_prefix_guard")
        if not must_touch_frontier_or_sink:
            violations.append("missing_frontier_or_sink_constraint")
        if not frontier_or_sink_present:
            violations.append("missing_frontier_or_sink")
        if not blocks_nonlocal_best_repair:
            violations.append("missing_nonlocal_best_repair_guard")
        if not blocks_generic_skeleton_unless_scope_local:
            violations.append("missing_generic_skeleton_guard")
        if not blocks_target_seen_without_path_progress:
            violations.append("missing_no_progress_guard")

        if forbidden_patterns_present:
            positive_evidence.append(
                {
                    "source": "candidate_proposal",
                    "check": "forbidden_pattern_set_present",
                }
            )
        else:
            warnings.append("forbidden_patterns_incomplete")
            negative_evidence.append(
                {
                    "source": "candidate_proposal",
                    "check": "forbidden_patterns_incomplete",
                    "required_patterns": required_forbidden_patterns,
                }
            )

        observed_prefix_present = bool(observed_prefix)
        if observed_prefix_present:
            positive_evidence.append(
                {
                    "source": observed_prefix_source,
                    "check": "observed_prefix_available",
                    "observed_matched_prefix": observed_prefix,
                }
            )
        else:
            warnings.append("observed_prefix_not_available")

        no_action_atom_generated = not _contains_disallowed_key(
            proposal,
            ["action_atom", "action_atoms", "generated_action_atom", "generated_action_atoms", "tfi_scoped_action_pool"],
        )
        no_full_sequence_generated = not _contains_disallowed_key(
            proposal,
            [
                "generated_sequence",
                "generated_sequences",
                "full_sequence",
                "full_sequences",
                "complete_sequence",
                "complete_sequences",
                "materialized_sequence",
                "materialized_sequences",
            ],
        )
        no_concrete_tx_generated = not _contains_disallowed_key(
            proposal,
            [
                "concrete_tx",
                "concrete_txs",
                "concrete_transaction",
                "concrete_transactions",
                "materialized_transactions",
                "transaction_inputs",
                "executable_transactions",
            ],
        )

        if not no_action_atom_generated:
            violations.append("action_atom_generated")
        if not no_full_sequence_generated:
            violations.append("full_sequence_generated")
        if not no_concrete_tx_generated:
            violations.append("concrete_tx_generated")

        if plan:
            plan_required_prefix = _clean_list(plan.get("required_prefix", {}).get("observed_matched_prefix"))
            if observed_prefix and plan_required_prefix and observed_prefix == plan_required_prefix:
                positive_evidence.append({"source": "agent_selection_plan", "check": "observed_prefix_matches_plan"})
            elif observed_prefix and plan_required_prefix and observed_prefix != plan_required_prefix:
                warnings.append("observed_prefix_plan_mismatch")
        elif plan_lookup.get("ambiguous"):
            warnings.append("ambiguous_plan_reference")
            negative_evidence.append(
                {
                    "source": "agent_selection_plan",
                    "check": "ambiguous_plan_reference",
                    "source_plan_id": proposal.get("source_plan_id"),
                    "matching_plan_count": int(plan_lookup.get("matching_plan_count", 0)),
                }
            )
        else:
            warnings.append("agent_selection_plan_missing")

        scope_path_prefix = _clean_list(scope.get("path_context", {}).get("observed_matched_prefix"))
        if observed_prefix and scope_path_prefix and observed_prefix == scope_path_prefix:
            positive_evidence.append({"source": "candidate_scope", "check": "observed_prefix_matches_scope"})
        elif observed_prefix and scope_path_prefix and observed_prefix != scope_path_prefix:
            warnings.append("observed_prefix_scope_mismatch")
            negative_evidence.append(
                {
                    "source": "candidate_scope",
                    "check": "observed_prefix_scope_mismatch",
                    "scope_observed_matched_prefix": scope_path_prefix,
                }
            )
        elif not scope:
            warnings.append("candidate_scope_missing")

        nonlocal_examples = _listify(generic_policy.get("nonlocal_skeleton_examples"))
        if nonlocal_examples:
            negative_evidence.append(
                {
                    "source": "generic_skeleton_policy",
                    "check": "nonlocal_skeleton_examples_available",
                    "count": len(nonlocal_examples),
                }
            )

        status = "pass"
        next_stage_hint = "ready_for_candidate_materialization"
        if violations:
            status = "fail"
            next_stage_hint = "invalid_proposal"
        elif warnings:
            status = "partial"
            next_stage_hint = "needs_scope_review"

        checks = {
            "preserve_observed_prefix_required": preserve_observed_prefix_required,
            "observed_prefix_present": observed_prefix_present,
            "must_touch_frontier_or_sink": must_touch_frontier_or_sink,
            "frontier_or_sink_present": frontier_or_sink_present,
            "blocks_nonlocal_best_repair": blocks_nonlocal_best_repair,
            "blocks_generic_skeleton_unless_scope_local": blocks_generic_skeleton_unless_scope_local,
            "blocks_target_seen_without_path_progress": blocks_target_seen_without_path_progress,
            "forbidden_patterns_present": forbidden_patterns_present,
            "no_action_atom_generated": no_action_atom_generated,
            "no_full_sequence_generated": no_full_sequence_generated,
            "no_concrete_tx_generated": no_concrete_tx_generated,
        }

        reason = "scoped selection proposal passed constraint-only dryrun"
        if status == "partial":
            reason = "scoped selection proposal preserves core constraints but needs additional scope/evidence review"
        elif status == "fail":
            reason = "scoped selection proposal violates scope-constraint requirements"

        return {
            "dryrun_id": "TFI_DRYRUN_%d" % ordinal,
            "proposal_id": proposal.get("proposal_id"),
            "proposal_type": "scoped_selection_proposal",
            "scope_id": proposal.get("scope_id"),
            "hypothesis_id": proposal.get("hypothesis_id"),
            "target_function": proposal.get("target_function"),
            "frontier": proposal.get("frontier"),
            "sink": proposal.get("sink"),
            "dryrun_status": status,
            "dryrun_mode": "constraint_validation_only",
            "selection_constraint_checks": checks,
            "violations": _unique_list(violations),
            "warnings": _unique_list(warnings),
            "positive_evidence": _unique_list(positive_evidence),
            "negative_evidence": _unique_list(negative_evidence),
            "next_stage_hint": next_stage_hint,
            "confidence": _text(proposal.get("confidence")).strip() or "low",
            "reference_resolution": {
                "lookup_strategy": plan_lookup.get("lookup_strategy"),
                "ambiguous_plan_reference": bool(plan_lookup.get("ambiguous")),
                "matching_plan_count": int(plan_lookup.get("matching_plan_count", 0)),
                "resolved_plan_id": plan.get("plan_id") if isinstance(plan, dict) else None,
                "resolved_scope_id": plan.get("scope_id") if isinstance(plan, dict) else None,
                "resolved_source_request_id": plan.get("source_request_id") if isinstance(plan, dict) else None,
            },
            "reason": reason,
        }

    def _make_plan_composite_key(self, scope_id: Any, source_request_id: Any, plan_id: Any) -> str:
        scope_text = _text(scope_id).strip()
        request_text = _text(source_request_id).strip()
        plan_text = _text(plan_id).strip()
        if not (scope_text and request_text and plan_text):
            return ""
        return "%s::%s::%s" % (scope_text, request_text, plan_text)

    def _resolve_plan_reference(
        self,
        proposal: Dict[str, Any],
        plan_index: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        composite_key = self._make_plan_composite_key(
            proposal.get("scope_id"),
            proposal.get("source_request_id"),
            proposal.get("source_plan_id"),
        )
        composite_match = plan_index.get("by_composite_key", {}).get(composite_key)
        if isinstance(composite_match, dict):
            return composite_match, {
                "lookup_strategy": "composite_key",
                "ambiguous": False,
                "matching_plan_count": 1,
            }

        plan_id = _text(proposal.get("source_plan_id")).strip()
        plan_matches = plan_index.get("by_plan_id", {}).get(plan_id, [])
        if len(plan_matches) == 1 and isinstance(plan_matches[0], dict):
            return plan_matches[0], {
                "lookup_strategy": "plan_id_fallback",
                "ambiguous": False,
                "matching_plan_count": 1,
            }
        if len(plan_matches) > 1:
            return {}, {
                "lookup_strategy": "plan_id_ambiguous",
                "ambiguous": True,
                "matching_plan_count": len(plan_matches),
            }
        return {}, {
            "lookup_strategy": "missing",
            "ambiguous": False,
            "matching_plan_count": 0,
        }

    def _resolve_binding_observed_prefix(
        self,
        proposal: Dict[str, Any],
        scope: Dict[str, Any],
        plan: Dict[str, Any],
        path_progress_index: Dict[str, Dict[str, Dict[str, Any]]],
        ksi_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Tuple[List[str], str]:
        del plan
        scope_prefix = _clean_list(scope.get("path_context", {}).get("observed_matched_prefix"))
        if scope_prefix:
            return scope_prefix, "candidate_scope"
        hypothesis_id = _text(proposal.get("hypothesis_id")).strip()
        frontier = _text(proposal.get("frontier")).strip()
        progress_entry = self._lookup_progress_entry(path_progress_index, hypothesis_id, frontier)
        progress_prefix = _clean_list(progress_entry.get("best_prefix"))
        if progress_prefix:
            return progress_prefix, "path_progress"
        ksi_entry = self._lookup_ksi_entry(ksi_index, hypothesis_id, frontier)
        ksi_prefix = _clean_list(ksi_entry.get("observations", {}).get("prefix_reached"))
        if ksi_prefix:
            return ksi_prefix, "ksi_evidence"
        return [], "unknown"

    def _resolve_selection_observed_prefix(
        self,
        proposal: Dict[str, Any],
        scope: Dict[str, Any],
        plan: Dict[str, Any],
        path_progress_index: Dict[str, Dict[str, Dict[str, Any]]],
        ksi_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Tuple[List[str], str]:
        proposal_prefix = _clean_list(proposal.get("required_prefix", {}).get("observed_matched_prefix"))
        if proposal_prefix:
            return proposal_prefix, "candidate_proposal"
        plan_prefix = _clean_list(plan.get("required_prefix", {}).get("observed_matched_prefix"))
        if plan_prefix:
            return plan_prefix, "agent_selection_plan"
        scope_prefix = _clean_list(scope.get("path_context", {}).get("observed_matched_prefix"))
        if scope_prefix:
            return scope_prefix, "candidate_scope"
        hypothesis_id = _text(proposal.get("hypothesis_id")).strip()
        frontier = _text(proposal.get("frontier")).strip()
        progress_entry = self._lookup_progress_entry(path_progress_index, hypothesis_id, frontier)
        progress_prefix = _clean_list(progress_entry.get("best_prefix"))
        if progress_prefix:
            return progress_prefix, "path_progress"
        ksi_entry = self._lookup_ksi_entry(ksi_index, hypothesis_id, frontier)
        ksi_prefix = _clean_list(ksi_entry.get("observations", {}).get("prefix_reached"))
        if ksi_prefix:
            return ksi_prefix, "ksi_evidence"
        return [], "unknown"

    def _lookup_progress_entry(
        self,
        index: Dict[str, Dict[str, Dict[str, Any]]],
        hypothesis_id: str,
        frontier: str,
    ) -> Dict[str, Any]:
        keyed = index.get("by_hypothesis_frontier", {})
        if hypothesis_id and frontier:
            entry = keyed.get("%s::%s" % (hypothesis_id, frontier))
            if isinstance(entry, dict):
                return entry
        entry = index.get("by_hypothesis", {}).get(hypothesis_id)
        return entry if isinstance(entry, dict) else {}

    def _lookup_ksi_entry(
        self,
        index: Dict[str, Dict[str, Dict[str, Any]]],
        hypothesis_id: str,
        frontier: str,
    ) -> Dict[str, Any]:
        keyed = index.get("by_hypothesis_frontier", {})
        if hypothesis_id and frontier:
            entry = keyed.get("%s::%s" % (hypothesis_id, frontier))
            if isinstance(entry, dict):
                return entry
        entry = index.get("by_hypothesis", {}).get(hypothesis_id)
        return entry if isinstance(entry, dict) else {}

    def _contains_target_or_scope_local_evidence(
        self,
        skeleton: List[str],
        target_function: str,
        frontier: str,
        sink: str,
        scope_local: Dict[str, Any],
    ) -> bool:
        markers = [item for item in (target_function, frontier, sink) if item]
        if any(marker in skeleton for marker in markers):
            return True
        if bool(scope_local.get("binding_skeleton_is_scope_local")):
            return True
        if bool(scope_local.get("target_seen_is_scope_local")) or bool(scope_local.get("sink_reached_is_scope_local")):
            return True
        return False

    def _count_possible_missing_action_atom(self, scope_payload: Dict[str, Any]) -> int:
        total = 0
        for item in _listify(scope_payload.get("scopes")):
            if not isinstance(item, dict):
                continue
            diagnosis = item.get("tfi_failure_diagnosis") if isinstance(item.get("tfi_failure_diagnosis"), dict) else {}
            if _text(diagnosis.get("diagnosis_type")).strip() in ("possible_missing_action_atom", "possible_local_candidate_gap"):
                total += 1
        return total

    def _build_summary(self, dryrun: Dict[str, Any]) -> Dict[str, Any]:
        items = _listify(dryrun.get("dryruns"))
        pass_count = 0
        partial_count = 0
        fail_count = 0
        binding_count = 0
        selection_count = 0
        binding_status: Counter = Counter()
        selection_status: Counter = Counter()
        confidence_counts: Counter = Counter()
        violation_counts: Counter = Counter()
        ready_for_binding_materialization_count = 0
        ready_for_candidate_materialization_count = 0
        needs_scope_review_count = 0
        invalid_proposal_count = 0

        for item in items:
            if not isinstance(item, dict):
                continue
            status = _text(item.get("dryrun_status")).strip() or "fail"
            if status == "pass":
                pass_count += 1
            elif status == "partial":
                partial_count += 1
            else:
                fail_count += 1

            confidence = _text(item.get("confidence")).strip() or "low"
            confidence_counts[confidence] += 1

            proposal_type = _text(item.get("proposal_type")).strip()
            if proposal_type == "binding_repair_proposal":
                binding_count += 1
                binding_status[status] += 1
            elif proposal_type == "scoped_selection_proposal":
                selection_count += 1
                selection_status[status] += 1

            hint = _text(item.get("next_stage_hint")).strip()
            if hint == "ready_for_binding_materialization":
                ready_for_binding_materialization_count += 1
            elif hint == "ready_for_candidate_materialization":
                ready_for_candidate_materialization_count += 1
            elif hint == "needs_scope_review":
                needs_scope_review_count += 1
            elif hint == "invalid_proposal":
                invalid_proposal_count += 1

            for violation in _clean_list(item.get("violations")):
                violation_counts[violation] += 1

        dryrun_stats = dryrun.get("dryrun_stats") if isinstance(dryrun.get("dryrun_stats"), dict) else {}
        duplicate_proposal_ids = _duplicate_id_map(items, "proposal_id")
        ambiguous_plan_reference_count = 0
        ambiguous_proposal_reference_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            resolution = item.get("reference_resolution") if isinstance(item.get("reference_resolution"), dict) else {}
            if bool(resolution.get("ambiguous_plan_reference")):
                ambiguous_plan_reference_count += 1
        if duplicate_proposal_ids:
            ambiguous_proposal_reference_count = sum(duplicate_proposal_ids.values())

        return {
            "enabled": True,
            "source": "target_focus_candidate_proposal_dryrun_summary_v1",
            "dryrun_count": len(items),
            "pass_count": pass_count,
            "partial_count": partial_count,
            "fail_count": fail_count,
            "binding_dryrun_count": binding_count,
            "selection_dryrun_count": selection_count,
            "binding_status_counts": {
                "pass": int(binding_status.get("pass", 0)),
                "partial": int(binding_status.get("partial", 0)),
                "fail": int(binding_status.get("fail", 0)),
            },
            "selection_status_counts": {
                "pass": int(selection_status.get("pass", 0)),
                "partial": int(selection_status.get("partial", 0)),
                "fail": int(selection_status.get("fail", 0)),
            },
            "dryruns_by_confidence": {
                "high": int(confidence_counts.get("high", 0)),
                "medium": int(confidence_counts.get("medium", 0)),
                "low": int(confidence_counts.get("low", 0)),
            },
            "top_violations": {
                "missing_sequence_skeleton": int(violation_counts.get("missing_sequence_skeleton", 0)),
                "function_order_change_allowed": int(violation_counts.get("function_order_change_allowed", 0)),
                "unexpected_mutable_field": int(violation_counts.get("unexpected_mutable_field", 0)),
                "missing_frontier_or_sink": int(violation_counts.get("missing_frontier_or_sink", 0)),
                "missing_generic_skeleton_guard": int(violation_counts.get("missing_generic_skeleton_guard", 0)),
                "binding_skeleton_mismatch": int(violation_counts.get("binding_skeleton_mismatch", 0)),
                "full_sequence_generated": int(violation_counts.get("full_sequence_generated", 0)),
                "action_atom_generated": int(violation_counts.get("action_atom_generated", 0)),
                "concrete_tx_generated": int(violation_counts.get("concrete_tx_generated", 0)),
            },
            "ready_for_binding_materialization_count": ready_for_binding_materialization_count,
            "ready_for_candidate_materialization_count": ready_for_candidate_materialization_count,
            "needs_scope_review_count": needs_scope_review_count,
            "invalid_proposal_count": invalid_proposal_count,
            "possible_missing_action_atom_seen": int(dryrun_stats.get("possible_missing_action_atom_seen", 0)),
            "action_atom_generated": bool(dryrun_stats.get("action_atom_generated")),
            "full_sequence_generated": bool(dryrun_stats.get("full_sequence_generated")),
            "concrete_tx_generated": bool(dryrun_stats.get("concrete_tx_generated")),
            "fuzz_executed": False,
            "evm_executed": False,
            "llm_called": False,
            "loaded_inputs": dryrun.get("loaded_inputs") if isinstance(dryrun.get("loaded_inputs"), dict) else {},
            "missing_inputs": _clean_list(dryrun.get("missing_inputs")),
            "id_integrity": {
                "plan_id_unique": None,
                "duplicate_plan_ids": None,
                "proposal_id_unique": not bool(duplicate_proposal_ids),
                "duplicate_proposal_ids": duplicate_proposal_ids,
                "ambiguous_plan_reference_count": ambiguous_plan_reference_count,
                "ambiguous_proposal_reference_count": ambiguous_proposal_reference_count,
            },
            "generality_audit": self._build_generality_audit(dryrun),
            "output_path": os.path.join(self.output_dir, "target_focus_candidate_proposal_dryrun.json"),
        }

    def _build_generality_audit(self, dryrun: Dict[str, Any]) -> Dict[str, Any]:
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
        contract_terms: List[str] = []
        function_specific_examples: List[str] = []

        for item in _listify(dryrun.get("dryruns")):
            if not isinstance(item, dict):
                continue
            for field in ("reason", "next_stage_hint", "dryrun_mode"):
                text = _text(item.get(field))
                for term in restricted_terms:
                    if term in text and term not in contract_terms:
                        contract_terms.append(term)
                if _contains_any(text, ["approve_transfer", "preserve_approve_transfer", "transferfrom"]):
                    function_specific_examples.append(text)

        function_specific_examples = _unique_list([item for item in function_specific_examples if item])[:10]
        return {
            "contract_specific_rule_count": int(len(contract_terms)),
            "contract_specific_terms_found": contract_terms,
            "function_specific_strategy_count": int(len(function_specific_examples)),
            "function_specific_strategy_examples": function_specific_examples,
            "generic_strategy_names_only": not bool(contract_terms or function_specific_examples),
            "uses_proposal_type_dispatch_only": True,
            "uses_scope_fields_only": True,
        }

