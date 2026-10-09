# -*- coding: utf-8 -*-

"""Checkpoint-targeted guided candidate injection.

This module is deliberately a guidance layer. It does not change oracle
semantics or the base generator/mutation/crossover/selection logic.
"""

import csv
import gzip
import json
import os
import re
import time
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from fuzzer.runtime.phase_cost import record_phase_event

try:
    from fuzzer.guidance.candidate_lifecycle import RepairCandidateLifecycleTracker
except ImportError:
    from fuzzer.guidance.candidate_lifecycle import RepairCandidateLifecycleTracker

try:
    from fuzzer.guidance.state_obligation import StateObligationBuilder
except ImportError:
    from fuzzer.guidance.state_obligation import StateObligationBuilder

try:
    from fuzzer.guidance.dependency_grounded_scope import DependencyGroundedRepairScopeBuilder
except ImportError:
    from fuzzer.guidance.dependency_grounded_scope import DependencyGroundedRepairScopeBuilder


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = _text(item).strip()
        if text:
            out.append(text)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_token(value: Any) -> str:
    text = _text(value).strip()
    if not text:
        return "NONE"
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "NONE"


def _dedup_text(items: Sequence[Any], limit: Optional[int] = None) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in items:
        text = _text(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if limit is not None and len(output) >= int(limit):
            break
    return output


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clean_any_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    return []


def _normalize_text_list(value: Any) -> List[str]:
    items = _clean_any_list(value)
    out: List[str] = []
    for item in items:
        if isinstance(item, (dict, list)):
            out.append(_json_compact(item))
            continue
        text = _text(item).strip()
        if text:
            out.append(text)
    return out


def _normalize_probe_plan(value: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return output
    for item in value:
        if isinstance(item, dict):
            payload = dict(item)
            targets = payload.get("targets")
            if isinstance(targets, list):
                payload["targets"] = [entry for entry in targets if entry not in (None, "")]
            output.append(payload)
        else:
            text = _text(item).strip()
            if text:
                output.append({"probe_type": "state_view", "function": text, "targets": []})
    return output


PROFILE_EFFECT_FIELDS = [
    "profile_enabled",
    "profile_name",
    "access_control_heuristic_matched",
    "matched_access_control_functions_count",
    "matched_sensitive_functions_count",
    "caller_pool_size_before",
    "caller_pool_size_after",
    "observed_owner_admin_address_count",
    "generated_permission_setup_candidates",
    "generated_caller_permutation_candidates",
    "unauthorized_revert_seen_count",
    "unauthorized_revert_repair_candidates",
    "setup_then_sensitive_success_count",
    "ownership_or_role_event_seen_count",
    "access_control_storage_write_seen_count",
    "profile_boosted_candidate_count",
]


def _compact_probe_plan(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    normalized = []
    for item in _normalize_probe_plan(value):
        normalized.append(
            {
                "probe_type": _text(item.get("probe_type")).strip(),
                "function": _text(item.get("function")).strip(),
                "targets": [_text(entry).strip() for entry in list(item.get("targets") or []) if _text(entry).strip()],
            }
        )
    return _json_compact(normalized)


def _json_compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except Exception:
        return _text(value).strip()


def _write_single_row_csv(path: str, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    except Exception:
        pass


def _semantic_field_alias(field: Any) -> str:
    text = _text(field).strip().lower()
    aliases = {
        "effect_probe_fix": "effect_probe",
        "effect_probe_plan": "effect_probe",
        "probe": "effect_probe",
        "additional_prefix": "prefix",
        "required_prefix": "prefix",
        "prefix_updates": "prefix",
        "sender": "sender_binding",
        "sender_binding_fix": "sender_binding",
        "beneficiary": "beneficiary_binding",
        "beneficiary_binding_fix": "beneficiary_binding",
        "receiver": "receiver_binding",
        "receiver_binding_fix": "receiver_binding",
        "amount": "amount_binding",
        "amount_binding_fix": "amount_binding",
        "arg": "arg_binding",
        "args": "arg_binding",
        "argument": "arg_binding",
        "arg_constraints": "arg_binding",
        "argument_policy": "arg_binding",
        "suffix": "suffix_actions",
        "suffix_action": "suffix_actions",
        "suffix_updates": "suffix_actions",
    }
    return aliases.get(text, text)


def _normalize_changed_fields(value: Any) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in _normalize_text_list(value):
        canonical = _semantic_field_alias(item)
        if canonical and canonical not in seen:
            output.append(canonical)
            seen.add(canonical)
    return output


def _canonical_action_name(value: Any) -> str:
    text = _text(value).strip()
    if not text:
        return ""
    if text == "native_balance":
        return text
    direct = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", text)
    if direct:
        return direct.group(1)
    call_like = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if call_like:
        return call_like.group(1)
    return text


def _action_lookup_key(value: Any) -> str:
    return _canonical_action_name(value).strip().lower()


def _level_rank(level: Any) -> int:
    return {"none": 0, "L1": 1, "L2": 2, "L3": 3}.get(_text(level).strip(), 0)


_ADDRESS_WORD_MASK = (1 << 160) - 1


def _normalized_address_candidates(value: Any) -> List[str]:
    output: List[str] = []
    seen = set()

    def remember(number: int) -> None:
        try:
            parsed = int(number)
        except Exception:
            return
        if parsed < 0:
            return
        normalized = "0x{:040x}".format(parsed & _ADDRESS_WORD_MASK)
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)

    def visit(item: Any) -> None:
        if item in (None, "", [], {}):
            return
        if isinstance(item, dict):
            for key in ("address", "to", "from", "from_addr", "contract", "value", "old_value", "new_value"):
                if key in item:
                    visit(item.get(key))
            return
        if isinstance(item, (list, tuple, set)):
            for entry in item:
                visit(entry)
            return
        text = _text(item).strip()
        if not text:
            return
        lowered = text.lower()
        if re.fullmatch(r"0x[a-f0-9]{40}", lowered):
            if lowered not in seen:
                seen.add(lowered)
                output.append(lowered)
        if re.fullmatch(r"0x[a-f0-9]{1,64}", lowered):
            try:
                remember(int(lowered, 16))
            except Exception:
                pass
            return
        if re.fullmatch(r"[0-9]+", lowered):
            try:
                remember(int(lowered, 10))
            except Exception:
                pass
            return
        for match in re.findall(r"0x[a-fA-F0-9]{40,64}", text):
            try:
                remember(int(match, 16))
            except Exception:
                continue

    visit(value)
    return output


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _disabled_runtime_log_path(path: str, enabled: bool) -> str:
    return path if enabled else ""


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
    except Exception:
        pass


CHECKPOINT_TEMPLATES: Dict[str, List[str]] = {
    "initialization_takeover": [
        "init_function_reached",
        "unexpected_sender_observed",
        "init_tx_success",
        "owner_admin_role_state_written",
        "privileged_effect_after_takeover",
    ],
    "access_control_state_change": [
        "protected_function_reached",
        "unexpected_sender_observed",
        "guarded_region_or_tx_success",
        "critical_state_written",
        "privileged_or_asset_effect_observed",
    ],
    "asset_flow_redirection": [
        "critical_setter_reached",
        "receiver_or_fee_state_written",
        "dependent_business_path_executed",
        "asset_flow_sink_touched",
        "transfer_or_balance_change_observed",
        "funds_flow_to_changed_or_unexpected_receiver",
    ],
    "repeat_claim_no_reset": [
        "claim_like_function_reached",
        "setup_or_balance_preparation_observed",
        "first_claim_success",
        "second_claim_attempted",
        "second_claim_success_or_no_expected_revert",
        "abnormal_balance_or_state_no_reset_observed",
    ],
    "lock_pause_blacklist_bypass": [
        "restricted_state_observed",
        "restricted_operation_called",
        "restriction_active",
        "restricted_operation_success",
        "balance_storage_or_asset_change_under_restriction",
    ],
    "critical_parameter_abuse": [
        "parameter_setter_reached",
        "abnormal_parameter_value_observed",
        "dependent_business_path_executed",
        "sink_or_critical_state_update_touched",
        "abnormal_effect_observed",
    ],
    "unchecked_external_call_effect": [
        "external_call_reached",
        "external_call_failed",
        "parent_tx_did_not_revert",
        "state_or_asset_effect_after_failed_call",
        "inconsistent_state_observed",
    ],
    "reentrancy_state_inconsistency": [
        "withdraw_claim_or_external_call_path_reached",
        "external_call_before_state_update_observed",
        "nested_or_reentrant_call_observed",
        "sensitive_state_not_updated_before_reentry",
        "repeated_withdraw_or_state_inconsistency_observed",
    ],
    "arithmetic_accounting_invariant": [
        "accounting_path_reached",
        "boundary_or_precision_operation_observed",
        "accounting_state_change_observed",
        "invariant_check_possible",
        "invariant_violation_observed",
    ],
}


ASSET_FLOW_SETTER_HINTS: Tuple[str, ...] = (
    "set",
    "wallet",
    "receiver",
    "treasury",
    "fee",
    "router",
    "recipient",
    "collector",
    "beneficiary",
    "payout",
    "commission",
    "controller",
    "supplycontroller",
    "devaddress",
    "devwallet",
    "marketing",
    "tax",
    "burnaddress",
)
ASSET_FLOW_STATE_HINTS: Tuple[str, ...] = (
    "wallet",
    "receiver",
    "treasury",
    "fee",
    "router",
    "recipient",
    "collector",
    "beneficiary",
    "payout",
    "commission",
    "controller",
    "supplycontroller",
    "devaddress",
    "devwallet",
    "marketing",
    "tax",
    "burnaddress",
)
ASSET_FLOW_BUSINESS_HINTS: Tuple[str, ...] = (
    "transfer",
    "withdraw",
    "distribute",
    "claim",
    "swap",
    "decrease",
    "redeem",
    "collect",
    "harvest",
    "payout",
    "sell",
    "buy",
    "send",
)


ACTION_MAPPING: Dict[str, Dict[str, Dict[str, Any]]] = {
    "asset_flow_redirection": {
        "critical_setter_reached": {
            "goal": "reach receiver/fee/router setter before asset-flow sink",
            "sequence_roles": ["setter"],
            "function_keywords": list(ASSET_FLOW_SETTER_HINTS),
            "sender_policy": "try_owner_or_deployer_or_previous_success_sender",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["function_call"],
        },
        "receiver_or_fee_state_written": {
            "goal": "call setter with attacker or nonzero known address",
            "sequence_roles": ["setter"],
            "function_keywords": list(ASSET_FLOW_SETTER_HINTS),
            "sender_policy": "try_owner_or_deployer_or_previous_success_sender",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["sstore", "receiver_or_fee_state_written"],
        },
        "dependent_business_path_executed": {
            "goal": "setter prefix followed by transfer/withdraw/distribute/swap-like function",
            "sequence_roles": ["setter", "business_sink"],
            "function_keywords": ["transfer", "withdraw", "distribute", "swap", "claim", "decrease", "redeem"],
            "argument_policy": "abi_default_or_boundary_mix",
            "expected_evidence": ["function_call", "tx_end"],
        },
        "asset_flow_sink_touched": {
            "goal": "trigger fee/withdraw/transfer path and monitor sink evidence",
            "sequence_roles": ["business_sink"],
            "function_keywords": ["transfer", "withdraw", "claim", "swap", "redeem", "send"],
            "argument_policy": "transfer_arguments_or_boundary_mix",
            "expected_evidence": ["sink_touch", "external_call", "balance_change"],
        },
        "transfer_or_balance_change_observed": {
            "goal": "trigger asset path with nonzero amount and watch transfer/balance_change",
            "sequence_roles": ["business_sink"],
            "function_keywords": ["transfer", "withdraw", "claim", "swap", "redeem"],
            "argument_policy": "transfer_arguments_or_boundary_mix",
            "value_policy": "try_zero_and_small_nonzero",
            "expected_evidence": ["token_transfer", "balance_change"],
        },
        "funds_flow_to_changed_or_unexpected_receiver": {
            "goal": "trigger dependent business path after receiver changed",
            "sequence_roles": ["setter", "business_sink"],
            "function_keywords": list(ASSET_FLOW_SETTER_HINTS + ASSET_FLOW_BUSINESS_HINTS),
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["balance_change", "unexpected_receiver"],
        },
    },
    "repeat_claim_no_reset": {
        "claim_like_function_reached": {
            "goal": "reach a claim/reward/harvest path",
            "sequence_roles": ["claim"],
            "function_keywords": ["claim", "reward", "harvest", "withdraw"],
            "argument_policy": "abi_default_or_boundary_mix",
            "expected_evidence": ["function_call"],
        },
        "setup_or_balance_preparation_observed": {
            "goal": "approve/deposit/stake/mint before claim",
            "sequence_roles": ["setup", "claim"],
            "function_keywords": ["approve", "deposit", "stake", "mint", "buy", "claim", "reward"],
            "argument_policy": "transfer_arguments_or_boundary_mix",
            "expected_evidence": ["sstore", "balance_change"],
        },
        "first_claim_success": {
            "goal": "setup followed by claim",
            "sequence_roles": ["setup", "claim"],
            "function_keywords": ["deposit", "stake", "mint", "claim", "reward", "harvest"],
            "argument_policy": "abi_default_or_boundary_mix",
            "expected_evidence": ["tx_end_success", "balance_change"],
        },
        "second_claim_attempted": {
            "goal": "setup -> claim -> claim",
            "sequence_roles": ["setup", "claim", "claim"],
            "function_keywords": ["deposit", "stake", "claim", "reward", "harvest"],
            "argument_policy": "abi_default_or_boundary_mix",
            "expected_evidence": ["repeat_function_call"],
        },
        "abnormal_balance_or_state_no_reset_observed": {
            "goal": "repeat claim with same sender/beneficiary and compare reward state",
            "sequence_roles": ["setup", "claim", "claim"],
            "function_keywords": ["deposit", "stake", "claim", "reward", "harvest"],
            "sender_policy": "keep_same_sender_across_sequence",
            "argument_policy": "abi_default_or_boundary_mix",
            "expected_evidence": ["balance_change", "sstore"],
        },
    },
    "initialization_takeover": {
        "init_function_reached": {
            "goal": "reach initialize/setup entrypoint",
            "sequence_roles": ["initializer"],
            "function_keywords": ["init", "initialize", "setup"],
            "sender_policy": "rotate_existing_senders",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["function_call"],
        },
        "unexpected_sender_observed": {
            "goal": "attacker or non-owner calls initialize/setup",
            "sequence_roles": ["initializer"],
            "function_keywords": ["init", "initialize", "setup"],
            "sender_policy": "rotate_existing_senders",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["function_call"],
        },
        "owner_admin_role_state_written": {
            "goal": "initialize with attacker sender and owner/admin-like args",
            "sequence_roles": ["initializer"],
            "function_keywords": ["init", "initialize", "setup"],
            "sender_policy": "rotate_existing_senders",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["sstore"],
        },
        "privileged_effect_after_takeover": {
            "goal": "attacker initialize followed by onlyOwner/privileged function",
            "sequence_roles": ["initializer", "privileged"],
            "function_keywords": ["init", "owner", "admin", "set", "upgrade", "withdraw", "mint"],
            "sender_policy": "keep_same_sender_across_sequence",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["sstore", "balance_change"],
        },
    },
    "access_control_state_change": {
        "protected_function_reached": {
            "goal": "reach a protected or privileged state-changing function",
            "sequence_roles": ["protected"],
            "function_keywords": ["set", "admin", "owner", "mint", "withdraw", "upgrade", "pause", "grant", "role"],
            "sender_policy": "rotate_existing_senders",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["function_call"],
        },
        "unexpected_sender_observed": {
            "goal": "non-owner/non-admin calls protected function",
            "sequence_roles": ["protected"],
            "function_keywords": ["set", "admin", "owner", "mint", "withdraw", "upgrade", "pause"],
            "sender_policy": "rotate_existing_senders",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["function_call"],
        },
        "critical_state_written": {
            "goal": "unauthorized call with boundary args causing state write",
            "sequence_roles": ["protected"],
            "function_keywords": ["set", "admin", "owner", "mint", "withdraw", "upgrade", "pause"],
            "sender_policy": "rotate_existing_senders",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["sstore"],
        },
        "privileged_or_asset_effect_observed": {
            "goal": "unauthorized state change followed by dependent privileged operation",
            "sequence_roles": ["protected", "privileged"],
            "function_keywords": ["set", "mint", "withdraw", "transfer", "upgrade"],
            "sender_policy": "keep_same_sender_across_sequence",
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["sstore", "balance_change"],
        },
    },
    "critical_parameter_abuse": {
        "parameter_setter_reached": {
            "goal": "reach a parameter setter/configuration function",
            "sequence_roles": ["setter"],
            "function_keywords": ["set", "update", "config", "fee", "rate", "limit", "price", "slippage"],
            "argument_policy": "boundary_numeric_values",
            "expected_evidence": ["function_call"],
        },
        "abnormal_parameter_value_observed": {
            "goal": "setter with 0/1/max/threshold boundary args",
            "sequence_roles": ["setter"],
            "function_keywords": ["set", "update", "config", "fee", "rate", "limit", "price", "slippage"],
            "argument_policy": "boundary_numeric_values",
            "expected_evidence": ["function_call", "sstore"],
        },
        "dependent_business_path_executed": {
            "goal": "setter(boundary) followed by transfer/swap/claim/withdraw",
            "sequence_roles": ["setter", "business_sink"],
            "function_keywords": ["set", "swap", "claim", "withdraw", "transfer", "buy", "sell"],
            "argument_policy": "boundary_numeric_values",
            "expected_evidence": ["tx_end_success"],
        },
        "abnormal_effect_observed": {
            "goal": "trigger downstream business effect after abnormal parameter",
            "sequence_roles": ["setter", "business_sink"],
            "function_keywords": ["set", "swap", "claim", "withdraw", "transfer", "sell"],
            "argument_policy": "boundary_numeric_values",
            "expected_evidence": ["balance_change", "sstore"],
        },
    },
    "lock_pause_blacklist_bypass": {
        "restricted_state_observed": {
            "goal": "set pause/blacklist/lock active if possible",
            "sequence_roles": ["restriction_setter"],
            "function_keywords": ["pause", "blacklist", "lock", "ban", "block", "set"],
            "argument_policy": "mutate_boolean_and_address_args",
            "expected_evidence": ["sstore"],
        },
        "restricted_operation_called": {
            "goal": "restricted transfer/sell/withdraw under active restriction",
            "sequence_roles": ["restriction_setter", "restricted_operation"],
            "function_keywords": ["pause", "blacklist", "transfer", "sell", "withdraw"],
            "argument_policy": "transfer_arguments_or_boundary_mix",
            "expected_evidence": ["function_call"],
        },
        "restriction_active": {
            "goal": "activate pause/blacklist/lock state before restricted operation",
            "sequence_roles": ["restriction_setter"],
            "function_keywords": ["pause", "blacklist", "lock", "ban", "block", "set", "enable", "disable"],
            "sender_policy": "try_owner_or_deployer_or_previous_success_sender",
            "argument_policy": "mutate_boolean_and_address_args",
            "expected_evidence": ["sstore", "restriction_active"],
        },
        "restricted_operation_success": {
            "goal": "attempt restricted operation with affected sender",
            "sequence_roles": ["restriction_setter", "restricted_operation"],
            "function_keywords": ["pause", "blacklist", "transfer", "sell", "withdraw"],
            "sender_policy": "keep_same_sender_across_sequence",
            "argument_policy": "transfer_arguments_or_boundary_mix",
            "expected_evidence": ["tx_end_success", "balance_change"],
        },
        "balance_storage_or_asset_change_under_restriction": {
            "goal": "activate restriction, then observe asset/storage effect from restricted operation",
            "sequence_roles": ["restriction_setter", "restricted_operation"],
            "function_keywords": ["pause", "blacklist", "lock", "transfer", "transferFrom", "sell", "withdraw", "burn"],
            "sender_policy": "keep_same_sender_across_sequence",
            "argument_policy": "transfer_arguments_or_boundary_mix",
            "value_policy": "try_zero_and_small_nonzero",
            "expected_evidence": ["sstore", "token_transfer", "balance_change"],
        },
    },
    "unchecked_external_call_effect": {
        "external_call_reached": {
            "goal": "reach an external call path",
            "sequence_roles": ["external_call_path"],
            "function_keywords": ["call", "transfer", "withdraw", "execute", "send"],
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["external_call"],
        },
        "external_call_failed": {
            "goal": "make callee fail/revert if controllable",
            "sequence_roles": ["external_call_path"],
            "function_keywords": ["call", "transfer", "withdraw", "execute", "send"],
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["external_call_failed"],
        },
        "parent_tx_did_not_revert": {
            "goal": "failed external call path continues parent execution",
            "sequence_roles": ["external_call_path"],
            "function_keywords": ["call", "transfer", "withdraw", "execute", "send"],
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["tx_end_success"],
        },
        "state_or_asset_effect_after_failed_call": {
            "goal": "observe sstore or asset effect after failed call",
            "sequence_roles": ["external_call_path"],
            "function_keywords": ["call", "transfer", "withdraw", "execute", "send"],
            "argument_policy": "mutate_permission_and_sink_args",
            "expected_evidence": ["sstore", "balance_change"],
        },
    },
    "reentrancy_state_inconsistency": {
        "withdraw_claim_or_external_call_path_reached": {
            "goal": "reach withdraw/claim/redeem external-call path",
            "sequence_roles": ["withdraw_or_claim"],
            "function_keywords": ["withdraw", "claim", "redeem", "collect"],
            "argument_policy": "abi_default_or_boundary_mix",
            "expected_evidence": ["function_call", "external_call"],
        },
        "external_call_before_state_update_observed": {
            "goal": "reach withdraw/claim external call path",
            "sequence_roles": ["withdraw_or_claim"],
            "function_keywords": ["withdraw", "claim", "redeem", "collect"],
            "argument_policy": "abi_default_or_boundary_mix",
            "expected_evidence": ["external_call", "sstore"],
        },
        "nested_or_reentrant_call_observed": {
            "goal": "use attacker/receiver contract if available",
            "sequence_roles": ["withdraw_or_claim"],
            "function_keywords": ["withdraw", "claim", "redeem", "collect"],
            "requires_attacker_contract": True,
            "no_candidate_reason": "attacker receiver contract is not available in current local fuzzer context",
            "expected_evidence": ["call_depth"],
        },
        "repeated_withdraw_or_state_inconsistency_observed": {
            "goal": "repeat withdraw/claim around external call evidence",
            "sequence_roles": ["withdraw_or_claim", "withdraw_or_claim"],
            "function_keywords": ["withdraw", "claim", "redeem", "collect"],
            "argument_policy": "abi_default_or_boundary_mix",
            "expected_evidence": ["balance_change", "sstore"],
        },
    },
    "arithmetic_accounting_invariant": {
        "accounting_path_reached": {
            "goal": "reach mint/burn/deposit/withdraw/share/reward accounting path",
            "sequence_roles": ["accounting"],
            "function_keywords": ["mint", "burn", "deposit", "withdraw", "share", "reward", "convert", "calculate"],
            "argument_policy": "boundary_numeric_values",
            "expected_evidence": ["function_call"],
        },
        "boundary_or_precision_operation_observed": {
            "goal": "use 0/1/small-large ratio/rounding boundary/max-like values",
            "sequence_roles": ["accounting"],
            "function_keywords": ["mint", "burn", "deposit", "withdraw", "share", "reward", "convert", "calculate"],
            "argument_policy": "boundary_numeric_values",
            "expected_evidence": ["function_call"],
        },
        "accounting_state_change_observed": {
            "goal": "mint/burn/deposit/withdraw/share/reward state path",
            "sequence_roles": ["accounting"],
            "function_keywords": ["mint", "burn", "deposit", "withdraw", "share", "reward"],
            "argument_policy": "boundary_numeric_values",
            "expected_evidence": ["sstore"],
        },
        "invariant_check_possible": {
            "goal": "preserve an accounting-changing path and attach invariant-relevant observations",
            "sequence_roles": ["accounting", "accounting_followup"],
            "function_keywords": ["mint", "burn", "deposit", "withdraw", "share", "reward", "harvest", "claim", "update", "pool"],
            "argument_policy": "boundary_numeric_values",
            "expected_evidence": ["function_call", "sstore"],
        },
        "invariant_violation_observed": {
            "goal": "compare supply/share/assets/reward balances if invariant is computable",
            "sequence_roles": ["accounting", "accounting"],
            "function_keywords": ["mint", "burn", "deposit", "withdraw", "share", "reward"],
            "argument_policy": "boundary_numeric_values",
            "no_candidate_reason": "invariant comparator is not available online; candidate can only target accounting state changes",
            "expected_evidence": ["balance_change", "sstore"],
        },
    },
}


EFFECT_FORCING_MODULES = [
    "prestate_builder",
    "prefix_replay",
    "suffix_mutation",
    "effect_probe",
    "agent_repair",
    "strong_useful_scheduling",
    "failure_aware_agent_repair",
    "parameter_materializer",
]

RQ1_RESCUE_HARD_CHECKPOINTS = {
    "balance_storage_or_asset_change_under_restriction",
    "funds_flow_to_changed_or_unexpected_receiver",
    "abnormal_effect_observed",
    "invariant_check_possible",
    "privileged_or_asset_effect_observed",
    "invariant_violation_observed",
    "state_or_asset_effect_after_failed_call",
    "privileged_effect_after_takeover",
}

FAILURE_REPAIR_PIPELINE_COUNTER_FIELDS = [
    "effect_forcing_candidate_executed",
    "no_delta_candidate_detected",
    "failure_signature_attempted",
    "failure_signature_created",
    "repairable_failure_signature_count",
    "repair_trigger_evaluated",
    "repair_trigger_true",
    "repair_trigger_false",
    "llm_repair_call_attempted",
    "llm_repair_call_success",
    "llm_repair_call_error",
    "llm_repair_no_repair",
    "repair_plan_parsed",
    "repair_materialization_attempted",
    "repair_attempt_key_created",
    "repair_attempt_key_skipped_cached",
    "repair_attempt_budget_exceeded",
    "repair_materialization_budget_exceeded",
    "repair_rejected_repair_repeats_parent",
    "repair_rejected_no_changed_fields",
    "repair_rejected_duplicate_signature",
    "repair_storm_guard_triggered",
    "repair_subsystem_error",
    "repair_materialized",
    "repair_rejected",
    "repair_queued",
    "repair_admitted",
    "repair_selected",
    "repair_execution_submitted",
    "repair_execution_started",
    "repair_execution_exception",
    "repair_executed",
    "repair_attributed",
    "repair_dropped",
    "repair_oracle_events_linked",
    "repair_direct_delta",
    "repair_delayed_delta",
    "repair_to_l2",
    "repair_to_l3",
    "repair_probe_only_count",
    "repair_observation_only_count",
    "repair_semantic_delta_weak",
    "repair_semantic_delta_medium",
    "repair_semantic_delta_strong",
    "state_obligation_created",
    "state_obligation_static_and_runtime_grounded",
    "state_obligation_runtime_grounded_only",
    "state_obligation_insufficient_grounding",
    "state_obligation_repair_location_prefix",
    "state_obligation_repair_location_binding",
    "state_obligation_repair_location_suffix",
    "state_obligation_repair_location_probe",
    "state_obligation_repair_location_mixed",
    "repair_scope_created",
    "repair_scope_static_and_runtime_grounded",
    "repair_scope_runtime_grounded_only",
    "repair_scope_insufficient_grounding",
    "repair_scope_no_safe_semantic_scope",
    "repair_scope_candidate_state_writer_count",
    "repair_scope_candidate_sink_action_count",
    "repair_scope_candidate_repeat_action_count",
    "repair_scope_candidate_observation_action_count",
    "state_obligation_repair_requested",
    "grounded_semantic_repair_proposed",
    "grounded_semantic_repair_materialized",
    "grounded_semantic_repair_executed",
    "grounded_semantic_repair_attributed",
    "grounded_semantic_repair_to_checkpoint_delta",
    "grounded_semantic_repair_to_vspc_delta",
    "grounded_semantic_repair_to_L1",
    "grounded_semantic_repair_to_L2",
    "grounded_semantic_repair_to_L3",
    "observation_only_misaligned_repair_count",
    "ungrounded_semantic_change_rejected_count",
    "no_safe_semantic_scope_count",
    "observation_only_repair_dropped_or_deprioritized",
    "observation_only_repair_deprioritized",
    "semantic_repair_materialized",
    "semantic_repair_executed",
    "semantic_repair_to_checkpoint_delta",
    "semantic_repair_to_l1",
    "semantic_repair_to_l2",
    "semantic_repair_to_l3",
    "semantic_repair_to_L1",
    "semantic_repair_to_L2",
    "semantic_repair_to_L3",
]


def checkpoint_action_mapping_summary() -> Dict[str, Any]:
    return {
        "families": ACTION_MAPPING,
        "template_checkpoints": CHECKPOINT_TEMPLATES,
        "conservative_policy": {
            "no_generic_candidate_when_mapping_missing": True,
            "unsupported_reentrancy_attacker_contract": "record_no_candidate_reason",
            "invariant_violation_requires_computable_invariant": True,
        },
    }


class ABIIndex:
    def __init__(self, abi: Any = None, interface_mapper: Optional[Dict[str, str]] = None, interface: Optional[Dict[str, Any]] = None):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.by_lower: Dict[str, str] = {}
        if isinstance(abi, list):
            for item in abi:
                if not isinstance(item, dict) or item.get("type") != "function":
                    continue
                name = _text(item.get("name")).strip()
                if not name:
                    continue
                inputs = []
                for arg in item.get("inputs") or []:
                    if isinstance(arg, dict):
                        inputs.append(_text(arg.get("type")).strip() or "unknown")
                self.functions.setdefault(
                    name,
                    {
                        "name": name,
                        "inputs": inputs,
                        "state_mutability": _text(item.get("stateMutability")).strip().lower(),
                    },
                )
        if isinstance(interface_mapper, dict):
            for signature in interface_mapper:
                name = _text(signature).split("(", 1)[0].strip()
                if not name:
                    continue
                selector = interface_mapper.get(signature)
                inputs = []
                if isinstance(interface, dict) and selector in interface:
                    inputs = [_text(item).strip() for item in (interface.get(selector) or [])]
                self.functions.setdefault(name, {"name": name, "inputs": inputs, "state_mutability": ""})
        for name in self.functions:
            self.by_lower[name.lower()] = name

    def has(self, name: str) -> bool:
        return _text(name).strip().lower() in self.by_lower

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self.functions.get(self.by_lower.get(_text(name).strip().lower(), ""))

    def names(self) -> List[str]:
        return sorted(self.functions)

    def is_view(self, name: str) -> bool:
        meta = self.get(name)
        if not isinstance(meta, dict):
            return False
        return _text(meta.get("state_mutability")).strip().lower() in {"view", "pure"}

    def select_by_keywords(self, keywords: Sequence[str], preferred: Sequence[str] = ()) -> List[str]:
        names = self.names()
        output: List[str] = []
        lowered_keywords = [_text(key).lower() for key in keywords if _text(key).strip()]

        def matches(name: str) -> bool:
            if not lowered_keywords:
                return True
            low = name.lower()
            return any(key in low for key in lowered_keywords)

        for item in preferred:
            clean = _text(item).strip()
            if not clean or not self.has(clean):
                continue
            canonical = self.by_lower[clean.lower()]
            if canonical not in output and matches(canonical):
                output.append(canonical)
        for name in names:
            if matches(name) and name not in output:
                output.append(name)
        return output


class CheckpointProgressMirror:
    def __init__(self, runtime_pattern_family: str, affected_function: str = "", affected_variable: str = "") -> None:
        self.runtime_pattern_family = _text(runtime_pattern_family).strip()
        if self.runtime_pattern_family not in CHECKPOINT_TEMPLATES:
            self.runtime_pattern_family = "critical_parameter_abuse"
        self.affected_function = _text(affected_function).strip()
        self.affected_variable = _text(affected_variable).strip()
        self.covered = set()
        self.last_event_count = 0
        self.best_level = "none"
        self.last_vspc_improvement_round = 0
        self.vspc_stagnation_rounds = 0
        self.best_prefix_sequence: List[str] = []
        self.best_prefix_state_evidence: Dict[str, Any] = {}
        self._lock_pause_restricted_state_orders: List[int] = []
        self._lock_pause_effectful_call_orders: List[int] = []

    @staticmethod
    def _event_type(event: Dict[str, Any]) -> str:
        return _text(event.get("event_type")).strip()

    @staticmethod
    def _event_function(event: Dict[str, Any]) -> str:
        return _text(event.get("function")).strip().lower()

    @staticmethod
    def _event_successish(event: Dict[str, Any]) -> bool:
        return event.get("reverted") is not True and event.get("success") is not False

    @staticmethod
    def _event_tx_key(event: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
        global_tx = event.get("global_tx_index")
        if global_tx not in (None, ""):
            try:
                return ("global", int(global_tx))
            except Exception:
                return ("global", global_tx)
        sequence_id = _text(event.get("sequence_id")).strip()
        local_tx = event.get("local_tx_index", event.get("tx_index"))
        if sequence_id and local_tx not in (None, ""):
            try:
                return ("sequence", sequence_id, int(local_tx))
            except Exception:
                return ("sequence", sequence_id, local_tx)
        tx_index = event.get("tx_index")
        if tx_index not in (None, ""):
            try:
                return ("tx", int(tx_index))
            except Exception:
                return ("tx", tx_index)
        return None

    @staticmethod
    def _event_tx_order(event: Dict[str, Any]) -> Optional[int]:
        for key in ("global_tx_index", "tx_index", "local_tx_index"):
            value = event.get(key)
            if value in (None, ""):
                continue
            try:
                return int(value)
            except Exception:
                continue
        return None

    @staticmethod
    def _event_text(event: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in (
            "event_type",
            "contract",
            "function",
            "selector",
            "sender",
            "value",
            "args",
            "pc",
            "opcode",
            "slot",
            "variable_name",
            "old_value",
            "new_value",
            "from",
            "from_addr",
            "to",
            "amount",
            "error",
            "sink_type",
            "address",
        ):
            value = event.get(key)
            if value not in (None, "", [], {}):
                parts.append(_text(value).lower())
        raw = event.get("raw")
        if isinstance(raw, dict):
            for key in ("source", "label", "type", "kind", "reason", "error", "guard_type", "related_slot", "related_value", "sink_type"):
                value = raw.get(key)
                if value not in (None, "", [], {}):
                    parts.append(_text(value).lower())
        elif raw not in (None, "", [], {}):
            parts.append(_text(raw).lower())
        return " ".join(parts)

    @staticmethod
    def _value_changed(event: Dict[str, Any]) -> bool:
        old_value = event.get("old_value")
        new_value = event.get("new_value")
        if old_value in (None, "") and new_value in (None, ""):
            return False
        return _text(old_value).strip().lower() != _text(new_value).strip().lower()

    @staticmethod
    def _unexpected_sender_value(sender: Any) -> bool:
        sender_low = _text(sender).strip().lower()
        if not sender_low:
            return False
        return sender_low not in {
            "0x0000000000000000000000000000000000000000",
            "deployer",
            "owner",
            "admin",
            "governance",
        }

    @classmethod
    def _success_tx_keys(cls, events: Sequence[Dict[str, Any]]) -> set:
        keys = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            if cls._event_type(event) != "tx_end":
                continue
            if event.get("success") is True or event.get("reverted") is False:
                key = cls._event_tx_key(event)
                if key is not None:
                    keys.add(key)
        return keys

    @classmethod
    def _event_successish_with_tx(cls, event: Dict[str, Any], success_tx_keys: set) -> bool:
        if event.get("reverted") is True or event.get("success") is False:
            return False
        if event.get("success") is True or event.get("reverted") is False:
            return True
        key = cls._event_tx_key(event)
        return key in success_tx_keys if key is not None else cls._event_successish(event)

    @staticmethod
    def _normalized_function_name(event: Dict[str, Any]) -> str:
        return _text(event.get("function")).strip().lower().replace("_", "")

    def _function_matches_affected(self, event: Dict[str, Any]) -> bool:
        affected = _text(self.affected_function).strip().lower()
        function = _text(event.get("function")).strip().lower()
        return bool(affected and function and (function == affected or affected in function or function in affected))

    def _access_control_target_call(self, event: Dict[str, Any]) -> bool:
        if self._event_type(event) != "function_call":
            return False
        function = self._normalized_function_name(event)
        if not function:
            return False
        view_names = {
            "allowance",
            "balanceof",
            "decimals",
            "name",
            "owner",
            "paused",
            "symbol",
            "totalsupply",
            "totalburn",
            "minter",
        }
        if function in view_names or function.startswith("is") or function.startswith("get"):
            return False
        if self._function_matches_affected(event):
            return True
        mutating_tokens = (
            "accept",
            "add",
            "blacklist",
            "burn",
            "destroy",
            "exclude",
            "freeze",
            "grant",
            "include",
            "issue",
            "lock",
            "mint",
            "pause",
            "remove",
            "renounce",
            "redeem",
            "revoke",
            "set",
            "transfer",
            "unfreeze",
            "unlock",
            "unpause",
            "update",
            "upgrade",
            "whitelist",
            "wipe",
        )
        return any(token in function for token in mutating_tokens)

    @classmethod
    def _access_control_effect_call(cls, event: Dict[str, Any]) -> bool:
        function = cls._normalized_function_name(event)
        effect_tokens = (
            "mint",
            "burn",
            "withdraw",
            "sweep",
            "rescue",
            "upgrade",
            "transferownership",
            "setowner",
            "setadmin",
            "grantrole",
            "revokerole",
        )
        return any(token in function for token in effect_tokens)

    @classmethod
    def _access_control_l3_state_write(cls, event: Dict[str, Any]) -> bool:
        if cls._event_type(event) != "sstore" or not cls._value_changed(event):
            return False
        text = cls._event_text(event)
        if any(token in text for token in ("owner", "admin", "role", "implementation", "governance", "operator", "minter")):
            return True
        function = cls._normalized_function_name(event)
        if any(token in function for token in ("mint", "burn", "issue", "redeem", "destroy", "wipe")):
            return "totalsupply" in text or "total supply" in text or "supply" in text
        return False

    def _access_control_unexpected_success_calls_by_tx(self, events: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
        success_tx_keys = self._success_tx_keys(events)
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            if not self._access_control_target_call(event):
                continue
            if not self._unexpected_sender_value(event.get("sender")):
                continue
            if not self._event_successish_with_tx(event, success_tx_keys):
                continue
            key = self._event_tx_key(event)
            if key is not None:
                grouped.setdefault(key, []).append(event)
        return grouped

    def _access_control_l3_effect_seen(self, events: Sequence[Dict[str, Any]]) -> bool:
        unexpected_by_tx = self._access_control_unexpected_success_calls_by_tx(events)
        if not unexpected_by_tx:
            return False
        for event in events:
            if not isinstance(event, dict):
                continue
            key = self._event_tx_key(event)
            if key not in unexpected_by_tx:
                continue
            if self._access_control_l3_state_write(event):
                return True
            if (
                self._event_type(event) in {"token_transfer", "native_transfer", "external_call"}
                and self._event_successish(event)
                and any(self._access_control_effect_call(call) for call in unexpected_by_tx.get(key, []))
            ):
                return True
        return False

    def _receiver_or_fee_state_write(self, event: Dict[str, Any]) -> bool:
        if self._event_type(event) != "sstore" or not self._value_changed(event):
            return False
        text = self._event_text(event)
        affected = _text(self.affected_variable).strip().lower()
        if not (
            (affected and affected in text)
            or any(token in text for token in ASSET_FLOW_STATE_HINTS)
        ):
            return False
        old_addresses = set(_normalized_address_candidates(event.get("old_value")))
        new_addresses = set(_normalized_address_candidates(event.get("new_value")))
        if old_addresses and new_addresses and old_addresses == new_addresses:
            return False
        return True

    @staticmethod
    def _event_address_values(event: Dict[str, Any]) -> List[str]:
        values = []
        for key in ("to", "from", "from_addr", "contract", "address"):
            text = _text(event.get(key)).strip().lower()
            if text:
                values.append(text)
            values.extend(_normalized_address_candidates(event.get(key)))
        values.extend(_normalized_address_candidates(event.get("args")))
        return _dedup_text(values, limit=12)

    @classmethod
    def _same_or_later_event(cls, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        left_order = cls._event_tx_order(left)
        right_order = cls._event_tx_order(right)
        if left_order is not None and right_order is not None:
            return left_order >= right_order
        left_key = cls._event_tx_key(left)
        right_key = cls._event_tx_key(right)
        return bool(left_key is not None and left_key == right_key)

    def _asset_flow_redirect_seen(self, events: Sequence[Dict[str, Any]]) -> bool:
        writes = [event for event in events if isinstance(event, dict) and self._receiver_or_fee_state_write(event)]
        if not writes:
            return False
        transfers = [
            event
            for event in events
            if isinstance(event, dict)
            and self._event_type(event) in {"token_transfer", "native_transfer", "balance_change"}
            and self._event_successish(event)
        ]
        for write in writes:
            new_values = set(_normalized_address_candidates(write.get("new_value")))
            if not new_values:
                continue
            for transfer in transfers:
                transfer_values = set(self._event_address_values(transfer))
                if new_values.intersection(transfer_values) and self._same_or_later_event(transfer, write):
                    return True
        return False

    @classmethod
    def _lock_pause_restricted_state_active(cls, event: Dict[str, Any]) -> bool:
        text = cls._event_text(event)
        if not any(
            token in text
            for token in (
                "paused",
                "tradingopen",
                "locked",
                "blacklisted",
                "sellingdisabled",
                "blacklist",
                "pause",
                "lock",
                "freeze",
                "frozen",
                "restrict",
                "restricted",
                "deny",
                "denied",
                "forbid",
                "forbidden",
                "disable",
                "disabled",
                "ban",
                "banned",
                "block",
                "blocked",
            )
        ):
            return False
        value_text = _text(event.get("new_value")).strip().lower()
        false_markers = {"0", "false", "0x0"}
        true_markers = {"1", "true", "0x1"}
        if "tradingopen" in text:
            return value_text in false_markers
        if any(
            token in text
            for token in (
                "paused",
                "locked",
                "blacklisted",
                "sellingdisabled",
                "blacklist",
                "pause",
                "lock",
                "freeze",
                "frozen",
                "restrict",
                "restricted",
                "deny",
                "denied",
                "forbid",
                "forbidden",
                "disable",
                "disabled",
                "ban",
                "banned",
                "block",
                "blocked",
            )
        ):
            return value_text in true_markers
        return value_text in true_markers

    @classmethod
    def _lock_pause_meaningful_effect(cls, event: Dict[str, Any]) -> bool:
        event_type = cls._event_type(event)
        if event_type in {"token_transfer", "native_transfer"}:
            return cls._event_successish(event)
        if event_type != "balance_change" or not cls._event_successish(event):
            return False
        raw = event.get("raw")
        label = _text(raw.get("label")).strip().lower() if isinstance(raw, dict) else ""
        address = _text(event.get("address")).strip().lower()
        sender = _text(event.get("sender")).strip().lower()
        if label == "sender" and address and sender and address == sender:
            return False
        delta = _text(event.get("delta")).strip().lower()
        return delta not in {"", "0", "0.0", "-0", "+0"}

    @classmethod
    def _lock_pause_revert_reason_triggered(cls, events: Sequence[Dict[str, Any]]) -> bool:
        tokens = (
            "please wait",
            "paused",
            "pause",
            "blacklist",
            "blacklisted",
            "lock",
            "locked",
            "blocked",
            "block",
            "freeze",
            "frozen",
            "disabled",
            "forbidden",
            "banned",
            "tradingopen",
            "trading not open",
            "sellingdisabled",
        )
        for event in events:
            if not isinstance(event, dict):
                continue
            lowered = cls._event_text(event)
            if any(token in lowered for token in tokens):
                return True
        return False

    def _lock_pause_bypass_pair_seen(self, events: Sequence[Dict[str, Any]]) -> bool:
        success_tx_keys = self._success_tx_keys(events)
        effects_by_tx: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        restricted_calls: List[Dict[str, Any]] = []
        events_by_tx: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        guard_revert_orders: List[int] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = self._event_type(event)
            order = self._event_tx_order(event)
            key = self._event_tx_key(event)
            if key is not None:
                events_by_tx.setdefault(key, []).append(event)
            if (
                event_type == "sstore"
                and order is not None
                and self._lock_pause_restricted_state_active(event)
            ):
                self._lock_pause_restricted_state_orders.append(order)
            if self._lock_pause_meaningful_effect(event):
                if key is not None:
                    effects_by_tx.setdefault(key, []).append(event)
            if event_type == "function_call" and self._event_successish(event):
                function = self._event_function(event)
                if self._function_matches_affected(event) or any(token in function for token in ("transfer", "sell", "withdraw", "trade", "claim")):
                    restricted_calls.append(event)
            if (
                event_type in {"tx_end", "revert"}
                and event.get("success") is False
                and order is not None
                and self._lock_pause_revert_reason_triggered([event])
            ):
                guard_revert_orders.append(order)

        if len(self._lock_pause_restricted_state_orders) > 256:
            self._lock_pause_restricted_state_orders = self._lock_pause_restricted_state_orders[-256:]
        for call in restricted_calls:
            key = self._event_tx_key(call)
            order = self._event_tx_order(call)
            if key is not None and order is not None and effects_by_tx.get(key):
                self._lock_pause_effectful_call_orders.append(order)
        if len(self._lock_pause_effectful_call_orders) > 256:
            self._lock_pause_effectful_call_orders = self._lock_pause_effectful_call_orders[-256:]
        same_tx_effect_orders: List[int] = []
        for key, tx_events in events_by_tx.items():
            tx_orders = [self._event_tx_order(item) for item in tx_events if self._event_tx_order(item) is not None]
            if not tx_orders:
                continue
            order = min(tx_orders)
            has_meaningful_effect = any(self._lock_pause_meaningful_effect(item) for item in tx_events)
            has_successful_restricted_call = any(
                self._event_type(item) == "function_call"
                and self._event_successish_with_tx(item, success_tx_keys)
                and (
                    self._function_matches_affected(item)
                    or any(token in self._event_function(item) for token in ("transfer", "sell", "withdraw", "trade", "claim"))
                )
                for item in tx_events
            )
            has_state_write = any(
                self._event_type(item) == "sstore"
                and self._event_successish_with_tx(item, success_tx_keys)
                and (
                    self._value_changed(item)
                    or self._function_matches_affected(item)
                )
                for item in tx_events
            )
            if has_meaningful_effect and has_successful_restricted_call and has_state_write:
                same_tx_effect_orders.append(order)
        if same_tx_effect_orders and guard_revert_orders:
            earliest_guard = min(guard_revert_orders)
            if any(order < earliest_guard for order in same_tx_effect_orders):
                return True
        if not self._lock_pause_restricted_state_orders or not self._lock_pause_effectful_call_orders:
            return False
        earliest_state = min(self._lock_pause_restricted_state_orders)
        return any(order >= earliest_state for order in self._lock_pause_effectful_call_orders)

    @staticmethod
    def _arithmetic_invariant_probe_seen(functions: Sequence[str]) -> bool:
        tokens = (
            "balance",
            "allowance",
            "pending",
            "supply",
            "price",
            "ratio",
            "preview",
            "debt",
            "pool",
            "exchange",
            "totalasset",
            "userinfo",
        )
        for name in functions:
            lowered = _text(name).strip().lower().replace("_", "")
            if lowered and any(token in lowered for token in tokens):
                return True
        return False

    def update_from_events(self, events: Sequence[Dict[str, Any]], round_id: int) -> Dict[str, Any]:
        before = set(self.covered)
        relevant_functions = []
        senders = []
        event_counts = Counter()
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = _text(event.get("event_type")).strip()
            event_counts[event_type] += 1
            function = _text(event.get("function")).strip()
            if function:
                relevant_functions.append(function)
            sender = _text(event.get("sender")).strip()
            if sender:
                senders.append(sender)

        functions_text = " ".join(relevant_functions).lower()
        has_function_call = event_counts.get("function_call", 0) > 0
        has_tx_success = any((e.get("event_type") == "tx_end" and e.get("success") is True) for e in events if isinstance(e, dict))
        has_sstore = event_counts.get("sstore", 0) > 0
        has_external_call = event_counts.get("external_call", 0) > 0
        has_external_fail = any((e.get("event_type") == "external_call" and e.get("success") is False) for e in events if isinstance(e, dict))
        has_balance = event_counts.get("balance_change", 0) > 0
        has_transfer = event_counts.get("token_transfer", 0) > 0
        has_asset_effect = bool(has_transfer or has_balance or event_counts.get("native_transfer", 0) > 0)
        repeated_functions = any(count > 1 for count in Counter(relevant_functions).values())
        affected_function = _text(self.affected_function).strip().lower()
        affected_reached = bool(
            affected_function
            and any(affected_function == _text(name).strip().lower() for name in relevant_functions)
        )
        unexpected_sender = len(set(senders)) >= 2
        asset_sink_call_seen = any(key in functions_text for key in ("transfer", "withdraw", "claim", "swap", "decrease", "redeem", "send"))
        privileged_effect_call_seen = any(key in functions_text for key in ("mint", "withdraw", "transfer", "upgrade", "execute", "grant", "admin", "owner"))
        business_effect_call_seen = any(key in functions_text for key in ("swap", "claim", "withdraw", "transfer", "buy", "sell", "redeem"))

        family = self.runtime_pattern_family
        if family == "asset_flow_redirection":
            if affected_reached or any(key in functions_text for key in ASSET_FLOW_SETTER_HINTS):
                self.covered.add("critical_setter_reached")
            if any(self._receiver_or_fee_state_write(e) for e in events if isinstance(e, dict)):
                self.covered.add("receiver_or_fee_state_written")
            if any(key in functions_text for key in ASSET_FLOW_BUSINESS_HINTS) and has_tx_success:
                self.covered.add("dependent_business_path_executed")
            if has_external_call or any(key in functions_text for key in ASSET_FLOW_BUSINESS_HINTS):
                self.covered.add("asset_flow_sink_touched")
            if has_asset_effect:
                self.covered.add("transfer_or_balance_change_observed")
            if self._asset_flow_redirect_seen(events):
                self.covered.add("funds_flow_to_changed_or_unexpected_receiver")
            else:
                self.covered.discard("funds_flow_to_changed_or_unexpected_receiver")
        elif family == "repeat_claim_no_reset":
            if any(key in functions_text for key in ("claim", "reward", "harvest", "withdraw")):
                self.covered.add("claim_like_function_reached")
            if any(key in functions_text for key in ("approve", "deposit", "stake", "mint")) or has_sstore:
                self.covered.add("setup_or_balance_preparation_observed")
            if has_tx_success and any(key in functions_text for key in ("claim", "reward", "harvest")):
                self.covered.add("first_claim_success")
            if repeated_functions and any(key in functions_text for key in ("claim", "reward", "harvest", "withdraw")):
                self.covered.add("second_claim_attempted")
            if repeated_functions and has_tx_success:
                self.covered.add("second_claim_success_or_no_expected_revert")
        elif family == "initialization_takeover":
            if any(key in functions_text for key in ("init", "initialize", "setup")):
                self.covered.add("init_function_reached")
            if unexpected_sender:
                self.covered.add("unexpected_sender_observed")
            if has_tx_success:
                self.covered.add("init_tx_success")
            if has_sstore:
                self.covered.add("owner_admin_role_state_written")
        elif family == "access_control_state_change":
            unexpected_success_calls_by_tx = self._access_control_unexpected_success_calls_by_tx(events)
            if any(self._access_control_target_call(e) for e in events if isinstance(e, dict)):
                self.covered.add("protected_function_reached")
            if unexpected_success_calls_by_tx:
                self.covered.add("unexpected_sender_observed")
            if unexpected_success_calls_by_tx:
                self.covered.add("guarded_region_or_tx_success")
            if any(self._access_control_l3_state_write(e) for e in events if isinstance(e, dict)):
                self.covered.add("critical_state_written")
            if self._access_control_l3_effect_seen(events):
                self.covered.add("privileged_or_asset_effect_observed")
            else:
                self.covered.discard("privileged_or_asset_effect_observed")
        elif family == "critical_parameter_abuse":
            if affected_reached or any(key in functions_text for key in ("set", "update", "config", "fee", "rate", "limit", "price")):
                self.covered.add("parameter_setter_reached")
            if has_function_call:
                self.covered.add("abnormal_parameter_value_observed")
            if has_tx_success and any(key in functions_text for key in ("swap", "claim", "withdraw", "transfer", "buy", "sell")):
                self.covered.add("dependent_business_path_executed")
            if has_sstore or has_external_call:
                self.covered.add("sink_or_critical_state_update_touched")
            if (
                "dependent_business_path_executed" in self.covered
                and (
                    has_asset_effect
                    or has_external_call
                    or (business_effect_call_seen and event_counts.get("sstore", 0) >= 2)
                )
            ):
                self.covered.add("abnormal_effect_observed")
        elif family == "lock_pause_blacklist_bypass":
            if any(key in functions_text for key in ("pause", "blacklist", "lock", "ban", "block")) or has_sstore:
                self.covered.add("restricted_state_observed")
            if any(key in functions_text for key in ("transfer", "sell", "withdraw")):
                self.covered.add("restricted_operation_called")
            if has_sstore:
                self.covered.add("restriction_active")
            if has_tx_success and any(key in functions_text for key in ("transfer", "sell", "withdraw")):
                self.covered.add("restricted_operation_success")
            if self._lock_pause_bypass_pair_seen(events):
                self.covered.add("balance_storage_or_asset_change_under_restriction")
            else:
                self.covered.discard("balance_storage_or_asset_change_under_restriction")
        elif family == "unchecked_external_call_effect":
            if has_external_call:
                self.covered.add("external_call_reached")
            if has_external_fail:
                self.covered.add("external_call_failed")
            if has_external_fail and has_tx_success:
                self.covered.add("parent_tx_did_not_revert")
            if has_external_fail and (has_sstore or has_balance):
                self.covered.add("state_or_asset_effect_after_failed_call")
        elif family == "reentrancy_state_inconsistency":
            if any(key in functions_text for key in ("withdraw", "claim", "redeem", "collect")) or has_external_call:
                self.covered.add("withdraw_claim_or_external_call_path_reached")
            if has_external_call and has_sstore:
                self.covered.add("external_call_before_state_update_observed")
            if repeated_functions:
                self.covered.add("nested_or_reentrant_call_observed")
            if repeated_functions and has_sstore:
                self.covered.add("sensitive_state_not_updated_before_reentry")
        elif family == "arithmetic_accounting_invariant":
            invariant_probe_seen = self._arithmetic_invariant_probe_seen(relevant_functions)
            if affected_reached or any(key in functions_text for key in ("mint", "burn", "deposit", "withdraw", "share", "reward", "convert")):
                self.covered.add("accounting_path_reached")
            if has_function_call:
                self.covered.add("boundary_or_precision_operation_observed")
            if has_sstore or has_balance:
                self.covered.add("accounting_state_change_observed")
            if (
                "accounting_state_change_observed" in self.covered
                and invariant_probe_seen
                and event_counts.get("sload", 0) > 0
            ):
                self.covered.add("invariant_check_possible")
            if (
                "accounting_state_change_observed" in self.covered
                and "invariant_check_possible" in self.covered
                and repeated_functions
                and (has_balance or event_counts.get("sstore", 0) >= 2)
            ):
                self.covered.add("invariant_violation_observed")

        improved = len(self.covered) > len(before)
        if improved:
            self.last_vspc_improvement_round = int(round_id)
            self.vspc_stagnation_rounds = 0
        else:
            self.vspc_stagnation_rounds += 1
        self.last_event_count = len(events)
        self.best_level = self._best_level()
        return {
            "covered_changed": improved,
            "newly_covered": sorted(self.covered - before),
            "event_counts": dict(event_counts),
            "relevant_functions": relevant_functions[-12:],
            "observed_senders": sorted(set(senders))[:8],
        }

    def _best_level(self) -> str:
        prefix = self.strict_prefix_len()
        total = len(CHECKPOINT_TEMPLATES.get(self.runtime_pattern_family, []))
        if prefix <= 0:
            return "none"
        if prefix >= total:
            return "L3"
        if prefix >= max(2, total - 2):
            return "L2"
        return "L1"

    def strict_prefix_len(self) -> int:
        count = 0
        for checkpoint in CHECKPOINT_TEMPLATES.get(self.runtime_pattern_family, []):
            if checkpoint not in self.covered:
                break
            count += 1
        return count

    def strict_vspc(self) -> float:
        total = len(CHECKPOINT_TEMPLATES.get(self.runtime_pattern_family, []))
        return float(self.strict_prefix_len()) / float(total) if total else 0.0

    def level_vspc(self) -> float:
        return {"none": 0.0, "L1": 1.0 / 3.0, "L2": 2.0 / 3.0, "L3": 1.0}.get(self.best_level, 0.0)

    def next_missing(self) -> Optional[str]:
        for checkpoint in CHECKPOINT_TEMPLATES.get(self.runtime_pattern_family, []):
            if checkpoint not in self.covered:
                return checkpoint
        return None

    def snapshot(self, round_id: int, event_summary: Optional[Dict[str, Any]] = None, trigger_reason: str = "") -> Dict[str, Any]:
        event_summary = event_summary if isinstance(event_summary, dict) else {}
        return {
            "round": int(round_id),
            "global_tx_index": event_summary.get("global_tx_index"),
            "global_sequence_index": event_summary.get("global_sequence_index"),
            "best_level": self.best_level,
            "level_vspc": round(self.level_vspc(), 6),
            "strict_vspc": round(self.strict_vspc(), 6),
            "covered_checkpoints": sorted(self.covered),
            "next_missing_checkpoint": self.next_missing(),
            "last_vspc_improvement_round": int(self.last_vspc_improvement_round),
            "vspc_stagnation_rounds": int(self.vspc_stagnation_rounds),
            "best_prefix_sequence": list(self.best_prefix_sequence),
            "best_prefix_state_evidence": dict(self.best_prefix_state_evidence),
            "evidence_summary": event_summary,
            "trigger_reason": trigger_reason,
        }


class ParameterMaterializer:
    def __init__(self, abi_index: ABIIndex, accounts: Sequence[Any], output_dir: str = "") -> None:
        self.abi_index = abi_index
        self.accounts = [_text(item).strip() for item in accounts if _text(item).strip()]
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, "parameter_materialization_log.jsonl") if output_dir else ""
        self.summary_path = os.path.join(output_dir, "parameter_materialization_summary.json") if output_dir else ""
        self.prestate_log_path = os.path.join(output_dir, "prestate_materialization_log.jsonl") if output_dir else ""
        self.prestate_summary_path = os.path.join(output_dir, "prestate_materialization_summary.json") if output_dir else ""
        self.counts = Counter()
        self.rejected_reason_counts = Counter()
        self.llm_hallucination_count = 0
        self.function_not_in_abi_count = 0
        self.records_by_candidate: Dict[str, Dict[str, Any]] = {}
        for path in (self.log_path, self.prestate_log_path):
            try:
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8"):
                        pass
            except Exception:
                if path == self.log_path:
                    self.log_path = ""
                elif path == self.prestate_log_path:
                    self.prestate_log_path = ""

    def materialize(self, candidate: Dict[str, Any], *, observed: Dict[str, Any], allow_llm_hypothesis: bool) -> Dict[str, Any]:
        self._normalize_candidate_policies(candidate)
        sequence = _clean_list(candidate.get("guided_sequence")) or _clean_list(candidate.get("candidate_sequence")) or _clean_list(candidate.get("function_sequence"))
        required_prefix = _clean_list(candidate.get("required_prefix"))
        suffix_actions = _clean_list(candidate.get("suffix_actions"))
        if not suffix_actions and sequence and not required_prefix:
            suffix_actions = list(sequence)
        prefix_tx_count = len(required_prefix)
        suffix_tx_count = len(suffix_actions)
        materialized_tx_count = len(sequence)
        modules_used = set(_clean_list(candidate.get("modules_used")))
        probe_plan = _clean_list(candidate.get("effect_probe_plan"))
        has_concrete_prefix = bool(required_prefix) and all(self.abi_index.has(function_name) for function_name in required_prefix)
        has_concrete_suffix = bool(suffix_actions) and all(self.abi_index.has(function_name) for function_name in suffix_actions)
        sequence_materialized = bool(sequence) and all(self.abi_index.has(function_name) for function_name in sequence)
        effect_probe_appended = bool(candidate.get("probe_appended")) and bool(probe_plan)
        is_failure_repair_candidate = bool(candidate.get("failure_repair_candidate")) or bool(candidate.get("parent_candidate_id"))
        function_not_in_abi = False
        for function_name in sequence:
            if not self.abi_index.has(function_name):
                function_not_in_abi = True
                break
        llm_hallucination = bool(candidate.get("source") == "agent" and function_not_in_abi)
        repair_status, repair_reason = self._repair_candidate_status(
            candidate=candidate,
            observed=observed,
            sequence=sequence,
            required_prefix=required_prefix,
            suffix_actions=suffix_actions,
            has_concrete_prefix=has_concrete_prefix,
            has_concrete_suffix=has_concrete_suffix,
            sequence_materialized=sequence_materialized,
            effect_probe_appended=effect_probe_appended,
        ) if is_failure_repair_candidate else ("", "")
        if repair_status:
            status = repair_status
            rejected_reason = repair_reason
        elif function_not_in_abi:
            status = "rejected_as_unresolved"
            rejected_reason = "function_not_in_abi"
            self.function_not_in_abi_count += 1
            if llm_hallucination:
                self.llm_hallucination_count += 1
        elif not sequence:
            status = "rejected_as_unresolved"
            rejected_reason = "empty_sequence"
        elif candidate.get("uses_llm_hypothesis") and not allow_llm_hypothesis:
            status = "rejected_as_unresolved"
            rejected_reason = "llm_hypothesis_materializer_disabled"
        else:
            policy = _text(candidate.get("argument_policy")).strip().lower()
            parameter_plan = candidate.get("parameter_plan") if isinstance(candidate.get("parameter_plan"), list) else []
            has_llm_parameter_source = any(
                isinstance(item, dict) and _text(item.get("source_type")).strip() == "llm_hypothesis"
                for item in parameter_plan
            )
            if has_concrete_prefix and has_concrete_suffix and sequence_materialized:
                status = "prefix_suffix_materialized"
            elif has_concrete_prefix and sequence_materialized:
                status = "prefix_materialized"
            elif has_concrete_suffix and sequence_materialized:
                status = "suffix_materialized"
            elif candidate.get("uses_llm_hypothesis") and has_llm_parameter_source:
                status = "materialized_from_llm_hypothesis"
            elif "boundary" in policy:
                status = "materialized_from_boundary"
            elif observed.get("observed_args") or observed.get("observed_senders"):
                status = "materialized_from_context"
            else:
                status = "pool_matched"
            rejected_reason = ""
        candidate["materialization_status"] = status
        candidate["materialization_status_final"] = status
        candidate["param_sources"] = self._param_sources(candidate, status)
        candidate["rejected_reason"] = rejected_reason
        candidate["llm_hallucination_detected"] = bool(llm_hallucination)
        candidate["function_not_in_abi"] = bool(function_not_in_abi)
        candidate["arg_type_valid"] = not function_not_in_abi and bool(sequence)
        candidate["prestate_supported"] = True
        candidate["has_concrete_prefix"] = bool(has_concrete_prefix)
        candidate["has_concrete_suffix"] = bool(has_concrete_suffix)
        candidate["sequence_materialized"] = bool(sequence_materialized)
        candidate["materialized_tx_count"] = int(materialized_tx_count)
        candidate["prefix_action_count"] = int(prefix_tx_count)
        candidate["suffix_action_count"] = int(suffix_tx_count)
        candidate["effect_probe_appended"] = bool(effect_probe_appended)
        reason_bits: List[str] = []
        if has_concrete_prefix:
            reason_bits.append("concrete_prefix")
        if has_concrete_suffix:
            reason_bits.append("concrete_suffix")
        if effect_probe_appended:
            reason_bits.append("effect_probe_appended")
        if sequence_materialized:
            reason_bits.append("sequence_materialized")
        if rejected_reason:
            reason_bits.append(rejected_reason)
        candidate["materialization_reason"] = ",".join(reason_bits) or status
        log_payload = {
            "time": time.time(),
            "candidate_id": candidate.get("candidate_id"),
            "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
            "function_sequence": list(sequence),
            "required_prefix": list(required_prefix),
            "suffix_actions": list(suffix_actions),
            "modules_used": list(candidate.get("modules_used") or []),
            "materialization_status": status,
            "materialization_status_final": status,
            "param_sources": candidate.get("param_sources"),
            "rejected_reason": rejected_reason,
            "llm_hallucination_detected": bool(llm_hallucination),
            "function_not_in_abi": bool(function_not_in_abi),
            "arg_type_valid": candidate["arg_type_valid"],
            "prestate_supported": candidate["prestate_supported"],
            "has_concrete_prefix": bool(has_concrete_prefix),
            "has_concrete_suffix": bool(has_concrete_suffix),
            "sequence_materialized": bool(sequence_materialized),
            "materialized_tx_count": int(materialized_tx_count),
            "prefix_action_count": int(prefix_tx_count),
            "suffix_action_count": int(suffix_tx_count),
            "effect_probe_appended": bool(effect_probe_appended),
            "materialization_reason": candidate["materialization_reason"],
        }
        candidate_id = _text(candidate.get("candidate_id")).strip() or "UNKNOWN_CANDIDATE"
        self.records_by_candidate[candidate_id] = dict(log_payload)
        for path in (self.log_path, self.prestate_log_path):
            _append_jsonl(path, log_payload)
        return candidate

    @staticmethod
    def _normalize_candidate_policies(candidate: Dict[str, Any]) -> None:
        binding_choices = candidate.get("binding_choices") if isinstance(candidate.get("binding_choices"), dict) else {}
        if not _text(candidate.get("sender_policy")).strip():
            candidate["sender_policy"] = (
                _text(binding_choices.get("sender_policy_used")).strip()
                or _text(candidate.get("sender_constraints")).strip()
                or "any_existing_sender"
            )
        if not _text(candidate.get("value_policy")).strip():
            candidate["value_policy"] = (
                _text(binding_choices.get("value_policy_used")).strip()
                or _text(candidate.get("value_constraints")).strip()
                or "keep_original"
            )
        if not _text(candidate.get("argument_policy")).strip():
            candidate["argument_policy"] = (
                _text(binding_choices.get("args_policy_used")).strip()
                or _text(candidate.get("arg_constraints")).strip()
                or "abi_default_or_boundary_mix"
            )

    def _repair_candidate_status(
        self,
        *,
        candidate: Dict[str, Any],
        observed: Dict[str, Any],
        sequence: Sequence[str],
        required_prefix: Sequence[str],
        suffix_actions: Sequence[str],
        has_concrete_prefix: bool,
        has_concrete_suffix: bool,
        sequence_materialized: bool,
        effect_probe_appended: bool,
    ) -> Tuple[str, str]:
        parent_candidate_id = _text(candidate.get("parent_candidate_id")).strip()
        if not parent_candidate_id:
            return "rejected_repair_target_mismatch", "missing_parent_candidate_id"
        if bool(candidate.get("repair_no_changed_fields")):
            return "rejected_repair_no_changed_fields", "no_changed_fields"
        if bool(candidate.get("repair_repeats_parent")):
            return "rejected_repair_repeats_parent", "repair_repeats_parent"
        if bool(candidate.get("repair_target_mismatch")):
            return "rejected_repair_target_mismatch", "target_missing_checkpoint_changed"
        if not sequence:
            return "rejected_repair_unresolved_binding", "empty_guided_sequence"
        if bool(candidate.get("function_not_in_abi")):
            return "rejected_repair_function_not_in_abi", "function_not_in_abi"
        unresolved = bool(candidate.get("repair_unresolved_binding"))
        if unresolved:
            return "rejected_repair_binding_unresolved", "repair_unresolved_binding"
        binding_only = bool(candidate.get("repair_binding_only"))
        sender_changed = bool(candidate.get("repair_sender_binding_changed"))
        beneficiary_changed = bool(candidate.get("repair_beneficiary_binding_changed"))
        receiver_changed = bool(candidate.get("repair_receiver_binding_changed"))
        amount_changed = bool(candidate.get("repair_amount_binding_changed"))
        probe_changed = bool(candidate.get("repair_probe_binding_changed"))
        if binding_only and sequence_materialized:
            if sender_changed:
                return "repair_sender_binding_materialized", "repair_sender_binding_materialized"
            if beneficiary_changed:
                return "repair_beneficiary_binding_materialized", "repair_beneficiary_binding_materialized"
            if receiver_changed:
                return "repair_receiver_binding_materialized", "repair_receiver_binding_materialized"
            if amount_changed:
                return "repair_amount_binding_materialized", "repair_amount_binding_materialized"
            if probe_changed:
                return "repair_probe_binding_materialized", "repair_probe_binding_materialized"
            return "repair_binding_only_materialized", "repair_binding_only_materialized"
        if required_prefix and has_concrete_prefix and has_concrete_suffix and sequence_materialized:
            if effect_probe_appended:
                return "repair_prefix_suffix_materialized", "repair_prefix_suffix_with_probe"
            return "repair_prefix_suffix_materialized", "repair_prefix_suffix_materialized"
        if required_prefix and has_concrete_prefix and sequence_materialized:
            return "repair_prefix_materialized", "repair_prefix_materialized"
        if suffix_actions and has_concrete_suffix and sequence_materialized:
            if effect_probe_appended:
                return "repair_probe_materialized", "repair_suffix_with_probe"
            return "repair_suffix_materialized", "repair_suffix_materialized"
        if effect_probe_appended and sequence_materialized:
            return "repair_probe_materialized", "repair_probe_materialized"
        if observed.get("observed_args") or observed.get("observed_senders"):
            return "repair_materialized", "repair_context_materialized"
        return "rejected_repair_binding_unresolved", "repair_not_materialized"

    def finalize_candidate(self, candidate: Dict[str, Any], *, observed: Dict[str, Any], allow_llm_hypothesis: bool = True) -> Dict[str, Any]:
        candidate = self.materialize(
            candidate,
            observed=observed,
            allow_llm_hypothesis=allow_llm_hypothesis,
        )
        final_status = _text(candidate.get("materialization_status_final") or candidate.get("materialization_status")).strip()
        if bool(candidate.get("effect_probe_appended")) and final_status and not final_status.startswith("rejected_"):
            reason = _text(candidate.get("materialization_reason")).strip()
            if "effect_probe_appended" not in reason:
                candidate["materialization_reason"] = ",".join(
                    [bit for bit in [reason, "effect_probe_appended"] if bit]
                )
            candidate["effect_probe_appended"] = True
            candidate_id = _text(candidate.get("candidate_id")).strip() or "UNKNOWN_CANDIDATE"
            record = self.records_by_candidate.get(candidate_id)
            if isinstance(record, dict):
                record["effect_probe_appended"] = True
                record["materialization_reason"] = candidate.get("materialization_reason")
                self.records_by_candidate[candidate_id] = record
        return candidate

    def _param_sources(self, candidate: Dict[str, Any], status: str) -> Dict[str, str]:
        if status == "pool_matched":
            source = "existing_pool"
        elif status == "materialized_from_context":
            source = "observed_trace"
        elif status == "materialized_from_boundary":
            source = "boundary_domain"
        elif status == "materialized_from_llm_hypothesis":
            source = "llm_hypothesis"
        elif status == "prefix_materialized":
            source = "observed_trace"
        elif status == "prefix_suffix_materialized":
            source = "observed_trace"
        elif status == "suffix_materialized":
            source = "observed_trace"
        elif status == "effect_probe_appended":
            source = "static_context"
        else:
            source = "unresolved"
        sequence = _clean_list(candidate.get("guided_sequence")) or _clean_list(candidate.get("function_sequence"))
        return {function_name: source for function_name in sequence}

    def build_summary(self) -> Dict[str, Any]:
        counts = Counter()
        rejected_reason_counts = Counter()
        llm_hallucination_count = 0
        function_not_in_abi_count = 0
        has_concrete_prefix_count = 0
        has_concrete_suffix_count = 0
        sequence_materialized_count = 0
        effect_probe_appended_count = 0
        suffix_materialized_count = 0
        for record in self.records_by_candidate.values():
            if not isinstance(record, dict):
                continue
            status = _text(record.get("materialization_status_final") or record.get("materialization_status")).strip()
            if status:
                counts[status] += 1
            rejected_reason = _text(record.get("rejected_reason")).strip()
            if rejected_reason:
                rejected_reason_counts[rejected_reason] += 1
            if bool(record.get("llm_hallucination_detected")):
                llm_hallucination_count += 1
            if bool(record.get("function_not_in_abi")):
                function_not_in_abi_count += 1
            if bool(record.get("has_concrete_prefix")):
                has_concrete_prefix_count += 1
            if bool(record.get("has_concrete_suffix")):
                has_concrete_suffix_count += 1
            if bool(record.get("sequence_materialized")):
                sequence_materialized_count += 1
            if bool(record.get("effect_probe_appended")):
                effect_probe_appended_count += 1
            if status == "suffix_materialized":
                suffix_materialized_count += 1
        return {
            "enabled": True,
            "counts": dict(counts),
            "pool_matched": int(counts.get("pool_matched", 0)),
            "materialized_from_context": int(counts.get("materialized_from_context", 0)),
            "materialized_from_boundary": int(counts.get("materialized_from_boundary", 0)),
            "materialized_from_llm_hypothesis": int(counts.get("materialized_from_llm_hypothesis", 0)),
            "prefix_materialized": int(counts.get("prefix_materialized", 0)),
            "prefix_suffix_materialized": int(counts.get("prefix_suffix_materialized", 0)),
            "suffix_materialized": int(suffix_materialized_count),
            "effect_probe_appended": int(effect_probe_appended_count),
            "rejected_missing_prefix_function": int(counts.get("rejected_missing_prefix_function", 0)),
            "rejected_unresolved_binding": int(counts.get("rejected_unresolved_binding", 0)),
            "rejected_prestate_unsupported": int(counts.get("rejected_prestate_unsupported", 0)),
            "rejected_as_unresolved": int(counts.get("rejected_as_unresolved", 0)),
            "rejected_reason_counts": dict(rejected_reason_counts),
            "llm_hallucination_detected_count": int(llm_hallucination_count),
            "function_not_in_abi_count": int(function_not_in_abi_count),
            "has_concrete_prefix_count": int(has_concrete_prefix_count),
            "has_concrete_suffix_count": int(has_concrete_suffix_count),
            "sequence_materialized_count": int(sequence_materialized_count),
            "candidate_record_count": len(self.records_by_candidate),
            "log_path": self.log_path or None,
            "prestate_log_path": self.prestate_log_path or None,
        }

    def write_summary(self) -> Dict[str, Any]:
        summary = self.build_summary()
        _write_json(self.summary_path, summary)
        _write_json(self.prestate_summary_path, summary)
        return summary


class CheckpointGuidedRuntime:
    def __init__(
        self,
        *,
        output_dir: str,
        oracle_family: str,
        normalized_type: str = "",
        finding_title: str = "",
        affected_function: str = "",
        affected_variable: str = "",
        l1_condition: str = "",
        l2_condition: str = "",
        l3_condition: str = "",
        required_trace_evidence: str = "",
        abi: Any = None,
        interface_mapper: Optional[Dict[str, str]] = None,
        interface: Optional[Dict[str, Any]] = None,
        accounts: Sequence[Any] = (),
        guidance_profile: str = "default",
        deployer_account: str = "",
        contract_address: str = "",
        budget_ratio: float = 0.2,
        max_per_round: int = 3,
        trigger_stagnation: int = 3,
        trigger_levels: Sequence[str] = ("L1", "L2"),
        dedup_window: int = 100,
        enable_parameter_materializer: bool = True,
        confidence_threshold: float = 0.0,
        enable_agent: bool = False,
        llm_provider: Any = None,
        enable_effect_forcing_v3: bool = False,
        enable_effect_forcing_v3_1: bool = False,
        enable_prestate_builder: bool = False,
        enable_prefix_replay_suffix_mutation: bool = False,
        enable_effect_probes: bool = False,
        enable_agent_effect_repair: bool = False,
        enable_failure_aware_agent_repair: bool = False,
        enable_failure_repair_pipeline: bool = False,
        enable_strong_useful_scheduling: bool = False,
        failure_repair_max_retries: int = 2,
        failure_repair_use_agent: bool = False,
        failure_repair_window_global_tx: int = 20,
        failure_repair_priority_ratio: float = 0.5,
        dump_failure_repair_prompts: bool = False,
        enable_runtime_logging: bool = False,
        bench_id: str = "",
        contract_name: str = "",
        static_summary: Optional[Dict[str, Any]] = None,
        phase_cost_env: Any = None,
        rq1_rescue_mode: bool = False,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.oracle_family = _text(oracle_family).strip()
        if self.oracle_family in CHECKPOINT_TEMPLATES:
            self.runtime_pattern_family = self.oracle_family
            self.runtime_pattern_family_provenance = "ORACLE_SPEC_FAMILY"
        else:
            self.runtime_pattern_family, self.runtime_pattern_family_provenance = self._infer_runtime_pattern_family(
                static_summary=static_summary,
                abi=abi,
            )
        if self.runtime_pattern_family not in CHECKPOINT_TEMPLATES:
            self.runtime_pattern_family = "critical_parameter_abuse"
            if not self.runtime_pattern_family_provenance:
                self.runtime_pattern_family_provenance = "STATIC_ANALYSIS_DERIVED_FALLBACK"
        self.normalized_type = _text(normalized_type).strip()
        self.finding_title = _text(finding_title).strip()
        self.bench_id = _text(bench_id).strip()
        self.contract_name = _text(contract_name).strip()
        self.static_summary = static_summary if isinstance(static_summary, dict) else {}
        self.phase_cost_env = phase_cost_env
        self.rq1_rescue_mode = bool(rq1_rescue_mode)
        self.affected_function = _text(affected_function).strip()
        self.affected_variable = _text(affected_variable).strip()
        self.l1_condition = _text(l1_condition).strip()
        self.l2_condition = _text(l2_condition).strip()
        self.l3_condition = _text(l3_condition).strip()
        self.required_trace_evidence = _text(required_trace_evidence).strip()
        self.abi_index = ABIIndex(abi=abi, interface_mapper=interface_mapper, interface=interface)
        self.accounts = [_text(item).strip() for item in accounts if _text(item).strip()]
        self.guidance_profile = _text(guidance_profile).strip().lower() or "default"
        self.deployer_account = _text(deployer_account).strip()
        self.contract_address = _text(contract_address).strip()
        self.budget_ratio = max(0.0, min(float(budget_ratio or 0.0), 1.0))
        self.max_per_round = max(1, int(max_per_round or 1))
        self.trigger_stagnation = max(0, int(trigger_stagnation or 0))
        self.trigger_levels = {_text(item).strip() for item in trigger_levels if _text(item).strip()}
        if self.rq1_rescue_mode:
            self.budget_ratio = max(self.budget_ratio, 0.5)
            self.max_per_round = max(self.max_per_round, 8)
            self.trigger_stagnation = min(self.trigger_stagnation, 1)
            self.trigger_levels.update({"none", "L1", "L2"})
        self.dedup_window = max(1, int(dedup_window or 1))
        self.enable_parameter_materializer = bool(enable_parameter_materializer)
        self.confidence_threshold = max(0.0, float(confidence_threshold or 0.0))
        self.enable_agent = bool(enable_agent)
        self.llm_provider = llm_provider
        self.enable_effect_forcing_v3 = bool(enable_effect_forcing_v3)
        self.enable_effect_forcing_v3_1 = bool(self.enable_effect_forcing_v3 and enable_effect_forcing_v3_1)
        self.enable_prestate_builder = bool(self.enable_effect_forcing_v3 and enable_prestate_builder)
        self.enable_prefix_replay_suffix_mutation = bool(self.enable_effect_forcing_v3 and enable_prefix_replay_suffix_mutation)
        self.enable_effect_probes = bool(self.enable_effect_forcing_v3 and enable_effect_probes)
        self.enable_agent_effect_repair = bool(self.enable_effect_forcing_v3 and enable_agent_effect_repair and enable_agent)
        self.enable_failure_repair_pipeline = bool(self.enable_effect_forcing_v3_1 and enable_failure_repair_pipeline)
        self.enable_failure_aware_agent_repair = bool(
            self.enable_failure_repair_pipeline and enable_failure_aware_agent_repair
        )
        self.enable_strong_useful_scheduling = bool(self.enable_effect_forcing_v3 and enable_strong_useful_scheduling)
        self.failure_repair_max_retries = max(0, int(failure_repair_max_retries or 0))
        self.failure_repair_use_agent = bool(self.enable_failure_repair_pipeline and failure_repair_use_agent and enable_agent)
        self.failure_repair_window_global_tx = max(1, int(failure_repair_window_global_tx or 1))
        self.failure_repair_priority_ratio = max(0.0, min(float(failure_repair_priority_ratio or 0.0), 1.0))
        self.dump_failure_repair_prompts = bool(self.enable_failure_repair_pipeline and dump_failure_repair_prompts)
        self.enable_runtime_logging = bool(enable_runtime_logging)
        self.progress = CheckpointProgressMirror(self.runtime_pattern_family, self.affected_function, self.affected_variable)
        self.materializer = ParameterMaterializer(self.abi_index, self.accounts, self.output_dir)
        self.progress_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "checkpoint_progress.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.candidates_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "checkpoint_guided_candidates.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.filter_summary_path = os.path.join(self.output_dir, "checkpoint_guided_filter_summary.json") if self.output_dir else ""
        self.delta_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "checkpoint_guided_candidate_delta.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.summary_path = os.path.join(self.output_dir, "checkpoint_guided_summary.json") if self.output_dir else ""
        self.mapping_summary_path = os.path.join(self.output_dir, "checkpoint_action_mapping_summary.json") if self.output_dir else ""
        self.prestate_candidates_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "prestate_builder_candidates.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.prestate_summary_path = os.path.join(self.output_dir, "prestate_builder_summary.json") if self.output_dir else ""
        self.prefix_candidates_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "prefix_replay_suffix_candidates.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.prefix_summary_path = os.path.join(self.output_dir, "prefix_replay_suffix_summary.json") if self.output_dir else ""
        self.effect_probe_events_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "effect_probe_events.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.effect_probe_summary_path = os.path.join(self.output_dir, "effect_probe_summary.json") if self.output_dir else ""
        self.agent_effect_repair_summary_path = os.path.join(self.output_dir, "agent_effect_repair_summary.json") if self.output_dir else ""
        self.strong_scheduling_summary_path = os.path.join(self.output_dir, "strong_useful_scheduling_summary.json") if self.output_dir else ""
        self.module_contribution_summary_path = os.path.join(self.output_dir, "module_contribution_summary.json") if self.output_dir else ""
        self.effect_forcing_delta_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "effect_forcing_candidate_delta.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.effect_forcing_summary_path = os.path.join(self.output_dir, "effect_forcing_summary.json") if self.output_dir else ""
        self.failure_signature_log_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "failure_signature_log.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.failure_signature_summary_path = os.path.join(self.output_dir, "failure_signature_summary.json") if self.output_dir else ""
        self.failure_repair_trigger_log_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "failure_repair_trigger_log.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.failure_repair_trigger_summary_path = os.path.join(self.output_dir, "failure_repair_trigger_summary.json") if self.output_dir else ""
        self.repair_materialization_log_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "repair_materialization_log.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.repair_materialization_summary_path = os.path.join(self.output_dir, "repair_materialization_summary.json") if self.output_dir else ""
        self.repair_payload_normalization_log_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "repair_payload_normalization_log.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.repair_payload_normalization_summary_path = os.path.join(self.output_dir, "repair_payload_normalization_summary.json") if self.output_dir else ""
        self.changed_fields_inference_log_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "changed_fields_inference_log.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.changed_fields_inference_summary_path = os.path.join(self.output_dir, "changed_fields_inference_summary.json") if self.output_dir else ""
        self.repair_queue_events_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "repair_queue_events.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.repair_scheduling_summary_path = os.path.join(self.output_dir, "repair_scheduling_summary.json") if self.output_dir else ""
        self.repair_scheduler_decisions_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "repair_scheduler_decisions.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.repair_selected_not_submitted_reasons_path = os.path.join(self.output_dir, "repair_selected_not_submitted_reasons.json") if self.output_dir else ""
        self.repair_execution_submission_summary_path = os.path.join(self.output_dir, "repair_execution_submission_summary.json") if self.output_dir else ""
        self.repair_execution_debug_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "repair_execution_debug.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.repair_fallback_log_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "repair_fallback_log.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.repair_fallback_summary_path = os.path.join(self.output_dir, "repair_fallback_summary.json") if self.output_dir else ""
        self.failure_repair_delta_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "failure_repair_candidate_delta.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.failure_repair_summary_path = os.path.join(self.output_dir, "failure_repair_summary.json") if self.output_dir else ""
        self.failure_repair_interactions_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "llm_failure_repair_interactions.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.failure_repair_prompt_catalog_json_path = os.path.join(self.output_dir, "llm_failure_repair_prompt_catalog.json") if self.output_dir else ""
        self.failure_repair_prompt_catalog_md_path = os.path.join(self.output_dir, "llm_failure_repair_prompt_catalog.md") if self.output_dir else ""
        self.failure_repair_pipeline_counters_path = os.path.join(self.output_dir, "failure_repair_pipeline_counters.json") if self.output_dir else ""
        self.state_obligation_summary_path = os.path.join(self.output_dir, "state_obligation_summary.json") if self.output_dir else ""
        self.dependency_grounded_scope_summary_path = os.path.join(self.output_dir, "dependency_grounded_scope_summary.json") if self.output_dir else ""
        self.rejected_repair_keys_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "rejected_repair_keys.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.repair_attempt_cache_summary_path = os.path.join(self.output_dir, "repair_attempt_cache_summary.json") if self.output_dir else ""
        self.repair_subsystem_error_log_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "repair_subsystem_error_log.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.candidate_lifecycle_path = _disabled_runtime_log_path(os.path.join(self.output_dir, "candidate_lifecycle.jsonl") if self.output_dir else "", self.enable_runtime_logging)
        self.candidate_lifecycle_summary_path = os.path.join(self.output_dir, "candidate_lifecycle_summary.json") if self.output_dir else ""
        for path in (
            self.progress_path,
            self.candidates_path,
            self.delta_path,
            self.prestate_candidates_path,
            self.prefix_candidates_path,
            self.effect_probe_events_path,
            self.effect_forcing_delta_path,
            self.failure_signature_log_path,
            self.failure_repair_trigger_log_path,
            self.repair_materialization_log_path,
            self.repair_payload_normalization_log_path,
            self.changed_fields_inference_log_path,
            self.repair_queue_events_path,
            self.repair_scheduler_decisions_path,
            self.repair_fallback_log_path,
            self.repair_execution_debug_path,
            self.failure_repair_delta_path,
            self.failure_repair_interactions_path,
            self.rejected_repair_keys_path,
            self.repair_subsystem_error_log_path,
            self.candidate_lifecycle_path,
        ):
            if path:
                try:
                    with open(path, "w", encoding="utf-8"):
                        pass
                except Exception:
                    pass
        _write_json(self.mapping_summary_path, checkpoint_action_mapping_summary())
        self.built_count = 0
        self.injected_count = 0
        self.executed_count = 0
        self.candidate_budget_used = 0
        self.candidate_executed_before_l2 = 0
        self.candidate_executed_before_l3 = 0
        self.candidate_delayed_after_l3 = 0
        self.medium_useful_count = 0
        self.strong_useful_count = 0
        self.checkpoint_delta_count = 0
        self.oracle_level_delta_count = 0
        self.no_candidate_reasons = Counter()
        self.filter_counts = Counter()
        self.source_counts = Counter()
        self.recent_signatures = deque(maxlen=self.dedup_window)
        self.recent_failed_checkpoint_candidates = deque(maxlen=self.dedup_window)
        self.inflight_candidates: Dict[str, Dict[str, Any]] = {}
        self.observed_execution_ids = set()
        self.last_event_offset = 0
        self.last_candidate_id = 0
        self.last_candidates: List[Dict[str, Any]] = []
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_error_count = 0
        self.llm_interaction_events: List[Dict[str, Any]] = []
        self.llm_interaction_time_total_sec = 0.0
        self.prestate_summary = Counter()
        self.prefix_summary = Counter()
        self.effect_probe_summary = Counter()
        self.agent_repair_summary = Counter()
        self.strong_scheduling_summary = Counter()
        self.failure_signature_summary = Counter()
        self.pending_failure_repair_signature_ids = set()
        self.failure_repair_trigger_summary = Counter()
        self.repair_materialization_summary = Counter()
        self.repair_payload_normalization_summary = Counter()
        self.changed_fields_inference_summary = Counter()
        self.repair_scheduling_summary = Counter()
        self.repair_fallback_summary = Counter()
        self.failure_repair_summary = Counter()
        self.repair_submission_summary = Counter()
        self.repair_submit_false_by_reason = Counter()
        self.module_contribution: Dict[str, Counter] = {
            module: Counter() for module in EFFECT_FORCING_MODULES
        }
        self.module_vspc_delta_sum = Counter()
        self.failure_signatures_by_candidate: Dict[str, Dict[str, Any]] = {}
        self.candidates_by_id: Dict[str, Dict[str, Any]] = {}
        self.repair_attempts_by_parent: Counter = Counter()
        self.repair_attempts_by_parent_signature: Counter = Counter()
        self.repair_cooldown_signatures = deque(maxlen=self.dedup_window)
        self.repair_recent_signatures = deque(maxlen=self.dedup_window)
        self.rejected_repair_attempt_keys = set()
        self.rejected_repair_attempt_reasons: Counter = Counter()
        self.repair_attempt_cache_summary = Counter()
        self.max_repair_attempts_per_parent_signature = 1
        self.max_repair_materialization_attempts_per_run = 100
        self.repair_interaction_records: List[Dict[str, Any]] = []
        self.repair_interaction_by_candidate_id: Dict[str, Dict[str, Any]] = {}
        self.failure_repair_pipeline_counters: Counter = Counter()
        self.failure_repair_pipeline_zero_reasons: Dict[str, str] = {
            field: "not_attempted_yet" for field in FAILURE_REPAIR_PIPELINE_COUNTER_FIELDS
        }
        self.pending_selected_repair_candidates: Dict[str, Dict[str, Any]] = {}
        self.repair_execution_grace_candidate_ids = set()
        self.repair_execution_grace_active = False
        self.repair_execution_grace_generation: Optional[int] = None
        self.repair_execution_grace_reason = ""
        self.state_obligation_builder = StateObligationBuilder(self.output_dir)
        self.dependency_grounded_scope_builder = DependencyGroundedRepairScopeBuilder(self.output_dir)
        self.repair_lifecycle = RepairCandidateLifecycleTracker(
            output_dir=self.output_dir,
            bench_id=self._runtime_bench_label(),
            contract_name=self._runtime_contract_label(),
        )
        self.profile_effect_summary_path = os.path.join(self.output_dir, "profile_effect_summary.json") if self.output_dir else ""
        self.profile_effect_csv_path = os.path.join(self.output_dir, "profile_effect_summary.csv") if self.output_dir else ""
        self.profile_effect_summary: Dict[str, Any] = {
            "profile_enabled": bool(self.guidance_profile != "default"),
            "profile_name": self.guidance_profile,
            "access_control_heuristic_matched": False,
            "matched_access_control_functions_count": 0,
            "matched_sensitive_functions_count": 0,
            "caller_pool_size_before": int(len(self.accounts)),
            "caller_pool_size_after": int(len(self.accounts)),
            "observed_owner_admin_address_count": 0,
            "generated_permission_setup_candidates": 0,
            "generated_caller_permutation_candidates": 0,
            "unauthorized_revert_seen_count": 0,
            "unauthorized_revert_repair_candidates": 0,
            "setup_then_sensitive_success_count": 0,
            "ownership_or_role_event_seen_count": 0,
            "access_control_storage_write_seen_count": 0,
            "profile_boosted_candidate_count": 0,
            "access_control_profile_budget_cap": int(self.max_per_round),
            "access_control_profile_fallback_active": False,
            "runtime_pattern_family_source": _text(self.static_summary.get("summary_source")).strip() or "ABI/static_summary derived heuristic",
            "runtime_pattern_family": self.runtime_pattern_family,
            "runtime_pattern_family_provenance": self.runtime_pattern_family_provenance,
        }

    def _runtime_bench_label(self) -> str:
        return _text(self.bench_id).strip() or "runtime_checkpoint_guided"

    def _runtime_contract_label(self) -> str:
        return _text(self.contract_name).strip() or "runtime_target_contract"

    def update_and_build(
        self,
        *,
        round_id: int,
        events: Sequence[Dict[str, Any]],
        tfi_candidates: Sequence[Dict[str, Any]],
        recent_failed: Sequence[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
        observed, snapshot, trigger_reason, _ = self._observe_progress(round_id=round_id, events=events)
        if not trigger_reason:
            self.last_candidates = []
            return {"candidates": [], "trigger_reason": "", "progress_snapshot": snapshot}
        next_missing = snapshot.get("next_missing_checkpoint")
        if not next_missing:
            self.no_candidate_reasons["all_checkpoints_covered"] += 1
            self.last_candidates = []
            return {"candidates": [], "trigger_reason": trigger_reason, "progress_snapshot": snapshot}
        built: List[Dict[str, Any]] = []
        if self.rq1_rescue_mode:
            built.extend(self._build_mapping_candidates(next_missing, observed, tfi_candidates))
            if self.enable_effect_forcing_v3:
                built.extend(self._build_effect_forcing_candidates(next_missing, observed, tfi_candidates, recent_failed))
            if self.enable_agent and self._rq1_rescue_agent_needed(next_missing, built):
                built.extend(self._build_agent_candidates(next_missing, observed, tfi_candidates, recent_failed))
        else:
            if self.enable_agent:
                built.extend(self._build_agent_candidates(next_missing, observed, tfi_candidates, recent_failed))
            built.extend(self._build_mapping_candidates(next_missing, observed, tfi_candidates))
            if self.enable_effect_forcing_v3:
                built.extend(self._build_effect_forcing_candidates(next_missing, observed, tfi_candidates, recent_failed))
        if self.enable_failure_repair_pipeline:
            built = self._apply_failure_repair_priority(built)
        if self.rq1_rescue_mode:
            built = self._rq1_rescue_prioritize_for_next_missing(built, next_missing)
        filtered = self._filter_candidates(built, next_missing)
        self.built_count += len(filtered)
        self.last_candidates = filtered
        for candidate in filtered:
            candidate_id = _text(candidate.get("candidate_id")).strip()
            if candidate_id:
                self.candidates_by_id[candidate_id] = dict(candidate)
            for module in _clean_list(candidate.get("modules_used")):
                self.module_contribution.setdefault(module, Counter())["candidates_built_by_module"] += 1
            _append_jsonl(self.candidates_path, candidate)
        self._write_filter_summary()
        self._write_summary()
        return {"candidates": filtered, "trigger_reason": trigger_reason, "progress_snapshot": snapshot}

    def observe(self, *, round_id: int, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        before_pending = len(self.pending_failure_repair_signature_ids)
        _, snapshot, trigger_reason, new_signature_count = self._observe_progress(round_id=round_id, events=events)
        self._write_summary()
        pending_count = len(self.pending_failure_repair_signature_ids)
        return {
            "trigger_reason": trigger_reason,
            "progress_snapshot": snapshot,
            "new_failure_signature_count": int(new_signature_count),
            "pending_failure_repair_signature_count": int(pending_count),
            "pending_failure_repair_signature_delta": int(max(0, pending_count - before_pending)),
        }

    def _observe_progress(
        self,
        *,
        round_id: int,
        events: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str, int]:
        new_events = list(events)[self.last_event_offset :]
        self.last_event_offset = len(events)
        observed = self._observed_context(events)
        progress_delta = self.progress.update_from_events(events, round_id)
        new_failure_signature_count = self._observe_inflight_execution(new_events, round_id)
        trigger_reason = self._trigger_reason()
        snapshot = self.progress.snapshot(round_id, self._progress_event_summary(events, progress_delta), trigger_reason)
        _append_jsonl(self.progress_path, snapshot)
        return observed, snapshot, trigger_reason, int(new_failure_signature_count or 0)

    def _trigger_reason(self) -> str:
        if self.enable_failure_repair_pipeline:
            next_missing = _text(self.progress.next_missing()).strip()
            for candidate_id in list(self.pending_failure_repair_signature_ids):
                parent = self.candidates_by_id.get(candidate_id)
                if not isinstance(parent, dict):
                    self.pending_failure_repair_signature_ids.discard(candidate_id)
                    continue
                if _text(parent.get("target_missing_checkpoint")).strip() == next_missing:
                    return "pending_failure_repair_signature"
            if self.pending_failure_repair_signature_ids:
                return "pending_failure_repair_signature"
        if self.progress.best_level in self.trigger_levels:
            return "current_best_level_in_trigger_levels"
        if self.progress.vspc_stagnation_rounds >= self.trigger_stagnation:
            return "vspc_stagnation"
        return ""

    def _rq1_rescue_agent_needed(self, next_missing: str, built: Sequence[Dict[str, Any]]) -> bool:
        if not self.rq1_rescue_mode:
            return True
        materialized = [
            item for item in built
            if isinstance(item, dict)
            and _text(item.get("materialization_status")).strip() != "rejected_as_unresolved"
            and bool(item.get("is_executable_candidate", True))
        ]
        if not materialized:
            return True
        hard_checkpoint = _text(next_missing).strip() in RQ1_RESCUE_HARD_CHECKPOINTS
        stalled = self.progress.vspc_stagnation_rounds >= 3
        if hard_checkpoint and stalled and len(materialized) < max(2, int(self.max_per_round // 2)):
            return True
        return False

    def _rq1_rescue_prioritize_for_next_missing(
        self,
        candidates: Sequence[Dict[str, Any]],
        next_missing: str,
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        target = _text(next_missing).strip()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            item = dict(candidate)
            sequence_text = " ".join(_clean_list(item.get("guided_sequence") or item.get("candidate_sequence") or item.get("function_sequence"))).lower()
            score = _safe_int(item.get("rq1_rescue_priority"), 0)
            if _text(item.get("target_missing_checkpoint")).strip() == target:
                score += 20
            if target in RQ1_RESCUE_HARD_CHECKPOINTS:
                score += 5
            if bool(item.get("prefix_replay")):
                score += 4
            if bool(item.get("has_concrete_prefix")):
                score += 3
            if bool(item.get("has_concrete_suffix")):
                score += 3
            if bool(item.get("effect_probe_appended")) or bool(item.get("probe_appended")):
                score += 2
            if bool(item.get("access_control_rescue_candidate")):
                score += 3
            if item.get("source") == "agent":
                score += 1
            if target == "restriction_active" and any(token in sequence_text for token in ("pause", "blacklist", "lock", "ban", "block")):
                score += 6
            if target == "balance_storage_or_asset_change_under_restriction" and any(token in sequence_text for token in ("transfer", "sell", "withdraw", "burn")):
                score += 6
            if target == "privileged_or_asset_effect_observed" and any(token in sequence_text for token in ("withdraw", "mint", "transfer", "upgrade", "execute")):
                score += 5
            if target in {"invariant_check_possible", "invariant_violation_observed"} and any(
                token in sequence_text for token in ("deposit", "withdraw", "mint", "burn", "harvest", "claim", "updatepool")
            ):
                score += 5
            item["rq1_rescue_mode"] = True
            item["rq1_rescue_priority"] = score
            output.append(item)
        output.sort(
            key=lambda item: (
                -_safe_int(item.get("rq1_rescue_priority"), 0),
                -_safe_float(item.get("confidence"), 0.0),
                _text(item.get("candidate_id")).strip(),
            )
        )
        return output

    def _bump_pipeline_counter(self, field: str, amount: int = 1, reason: str = "") -> None:
        if field not in FAILURE_REPAIR_PIPELINE_COUNTER_FIELDS:
            return
        self.failure_repair_pipeline_counters[field] += max(0, int(amount or 0))
        if self.failure_repair_pipeline_counters[field] > 0:
            self.failure_repair_pipeline_zero_reasons[field] = ""
        elif reason:
            self.failure_repair_pipeline_zero_reasons[field] = str(reason)

    def _set_zero_reason(self, field: str, reason: str) -> None:
        if field not in FAILURE_REPAIR_PIPELINE_COUNTER_FIELDS:
            return
        if int(self.failure_repair_pipeline_counters.get(field, 0) or 0) <= 0:
            self.failure_repair_pipeline_zero_reasons[field] = str(reason or "not_triggered")

    def _pipeline_counters_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for field in FAILURE_REPAIR_PIPELINE_COUNTER_FIELDS:
            value = int(self.failure_repair_pipeline_counters.get(field, 0) or 0)
            payload[field] = value
            payload["{}_zero_reason".format(field)] = "" if value > 0 else self.failure_repair_pipeline_zero_reasons.get(field, "not_triggered")
        return payload

    def _repair_submission_candidate_record(
        self,
        repair_candidate_id: Any,
        candidate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        candidate_id = _text(repair_candidate_id).strip()
        stored = self.pending_selected_repair_candidates.get(candidate_id)
        if isinstance(candidate, dict):
            base = dict(candidate)
        elif candidate_id and isinstance(self.candidates_by_id.get(candidate_id), dict):
            base = dict(self.candidates_by_id.get(candidate_id) or {})
        else:
            base = {}
        if isinstance(stored, dict):
            for key, value in stored.items():
                if base.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                    base[key] = value
        binding_choices = base.get("binding_choices") if isinstance(base.get("binding_choices"), dict) else {}
        guided_sequence = _clean_list(base.get("guided_sequence"))
        parameter_plan = list(base.get("parameter_plan") or []) if isinstance(base.get("parameter_plan"), list) else []
        effect_probe_plan = list(base.get("effect_probe_plan") or []) if isinstance(base.get("effect_probe_plan"), list) else []
        source_kind = _text(base.get("source_candidate_kind") or base.get("candidate_type") or base.get("repair_action")).strip()
        has_sender = bool(_text(binding_choices.get("sender_policy_used") or base.get("sender_policy") or base.get("sender_binding_fix")).strip())
        has_args = bool(parameter_plan or _text(binding_choices.get("args_policy_used") or base.get("argument_policy")).strip())
        has_value = bool(_text(binding_choices.get("value_policy_used") or base.get("value_policy")).strip())
        has_function_sequence = bool(guided_sequence)
        has_concrete_tx_sequence = bool(base.get("sequence_materialized") or base.get("has_concrete_prefix") or base.get("has_concrete_suffix") or guided_sequence)
        return {
            "repair_candidate_id": candidate_id,
            "bench_id": _text(base.get("bench_id") or self.bench_id).strip() or None,
            "contract": _text(base.get("contract_name") or base.get("contract") or self.contract_name).strip() or None,
            "source_trigger_id": _text(base.get("source_trigger_id") or base.get("failure_signature_id") or base.get("failure_signature")).strip() or None,
            "source_parent_id": _text(base.get("source_parent_id") or base.get("parent_candidate_id")).strip() or None,
            "source_scope_id": _text(base.get("source_scope_id") or base.get("scope_id") or base.get("source_task_id")).strip() or None,
            "source_candidate_kind": source_kind or None,
            "repair_type": _text(base.get("repair_action")).strip() or None,
            "has_concrete_tx_sequence": bool(has_concrete_tx_sequence),
            "has_valid_tx_count": len(guided_sequence),
            "has_function_sequence": bool(has_function_sequence),
            "has_sender": bool(has_sender),
            "has_args": bool(has_args),
            "has_value": bool(has_value),
            "has_probe": bool(effect_probe_plan or base.get("effect_probe_appended") or base.get("probe_appended")),
            "duplicate_signature": False,
            "source_allowed": bool(base.get("failure_repair_candidate") or ("failure_repair" in source_kind.lower())),
            "schema_valid": bool(has_function_sequence and has_concrete_tx_sequence),
            "remaining_budget_at_selection": base.get("remaining_budget_at_selection"),
            "execution_queue_size_at_selection": base.get("execution_queue_size_at_selection"),
        }

    def _log_repair_scheduler_decision(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        _append_jsonl(self.repair_scheduler_decisions_path, payload)

    def log_repair_execution_debug(self, repair_candidate_id: Any, event: str, **extra: Any) -> None:
        candidate_id = _text(repair_candidate_id).strip()
        if not candidate_id or not self.repair_execution_debug_path:
            return
        record = self._repair_submission_candidate_record(candidate_id)
        payload = {
            "event": _text(event).strip() or "unknown",
            "event_time": time.time(),
            "repair_candidate_id": candidate_id,
            "bench_id": record.get("bench_id"),
            "contract": record.get("contract"),
            "source_trigger_id": record.get("source_trigger_id"),
            "source_parent_id": record.get("source_parent_id"),
            "source_scope_id": record.get("source_scope_id"),
            "source_candidate_kind": record.get("source_candidate_kind"),
            "repair_type": record.get("repair_type"),
        }
        for key, value in extra.items():
            if value is not None:
                payload[key] = value
        _append_jsonl(self.repair_execution_debug_path, payload)

    def register_selected_repair_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        round_id: int,
        selected_rank: int,
        injected_rank: int,
    ) -> None:
        candidate_id = _text(candidate.get("candidate_id")).strip()
        if not candidate_id:
            return
        record = self._repair_submission_candidate_record(candidate_id, candidate)
        record.update(
            {
                "selected_round": int(round_id),
                "selected_rank": int(selected_rank),
                "injected_rank": int(injected_rank),
                "selected_timestamp": time.time(),
                "remaining_budget_at_selection": max(0, int(self.max_per_round) - int(selected_rank)),
                "execution_queue_size_at_selection": None,
                "attempted_submit": False,
                "submit_result": "pending",
                "submit_false_reason": "",
            }
        )
        self.pending_selected_repair_candidates[candidate_id] = record
        self._log_repair_scheduler_decision(
            dict(
                record,
                event="selected",
                selected=True,
                attempted_submit=False,
                submit_result="pending",
                submit_false_reason="",
                queue_budget_remaining=record.get("remaining_budget_at_selection"),
                time_budget_remaining=None,
                exception_if_any="",
            )
        )

    def record_repair_submission_outcome(
        self,
        repair_candidate_id: Any,
        *,
        submit_result: str,
        reason: str = "",
        individual: Any = None,
        extra: Optional[Dict[str, Any]] = None,
        terminal: bool = True,
    ) -> None:
        candidate_id = _text(repair_candidate_id).strip()
        if not candidate_id:
            return
        record = self._repair_submission_candidate_record(candidate_id)
        pending = self.pending_selected_repair_candidates.get(candidate_id, {})
        if isinstance(pending, dict):
            for key, value in pending.items():
                if record.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                    record[key] = value
        if individual is not None:
            record["sequence_id"] = _text(getattr(individual, "sequence_id", None)).strip() or record.get("sequence_id")
            record["individual_hash"] = _text(getattr(individual, "hash", None)).strip() or record.get("individual_hash")
        reason_text = _text(reason).strip()
        submit_result = _text(submit_result).strip().lower() or "unknown"
        if pending.get("submission_terminal") and submit_result in {"skipped", "failed", "submitted"}:
            return
        if not pending.get("submission_attempted"):
            self.repair_submission_summary["repair_candidate_submission_attempted"] += 1
            pending["submission_attempted"] = True
        if submit_result == "submitted":
            self.repair_submission_summary["repair_candidate_submission_succeeded"] += 1
            pending["submission_terminal"] = True
        elif submit_result == "skipped":
            self.repair_submission_summary["repair_candidate_submission_skipped"] += 1
            self.repair_submit_false_by_reason[reason_text or "unknown"] += 1
            pending["submission_terminal"] = bool(terminal)
        elif submit_result == "failed":
            self.repair_submission_summary["repair_candidate_submission_failed"] += 1
            self.repair_submit_false_by_reason[reason_text or "unknown"] += 1
            pending["submission_terminal"] = bool(terminal)
        else:
            pending["submission_terminal"] = bool(terminal)
        pending["attempted_submit"] = True
        pending["submit_result"] = submit_result
        pending["submit_false_reason"] = reason_text
        pending["last_submit_event_at"] = time.time()
        self.pending_selected_repair_candidates[candidate_id] = pending
        payload = dict(
            record,
            event="submission",
            selected=True,
            attempted_submit=True,
            submit_result=submit_result,
            submit_false_reason=reason_text,
            queue_budget_remaining=None,
            time_budget_remaining=None,
            exception_if_any="",
        )
        if isinstance(extra, dict):
            payload.update(extra)
        self._log_repair_scheduler_decision(payload)
        if submit_result == "submitted" or (terminal and submit_result in {"skipped", "failed"}):
            self.pending_selected_repair_candidates.pop(candidate_id, None)
            self.repair_execution_grace_candidate_ids.discard(candidate_id)

    def record_execution_started(self, individual: Any, *, tx_count: int = 0, tx_index: Optional[int] = None) -> None:
        runtime = self._repair_runtime()
        if runtime is None or not hasattr(individual, "repair_candidate_id"):
            return
        candidate_id = _text(getattr(individual, "repair_candidate_id", None)).strip()
        if not candidate_id or not self._is_failure_repair_individual(runtime, individual, candidate_id):
            return
        runtime._bump_pipeline_counter("repair_execution_started", 1)
        extra = {
            "sequence_id": _text(getattr(individual, "sequence_id", None)).strip(),
            "individual_hash": _text(getattr(individual, "hash", None)).strip(),
            "tx_count": int(tx_count or 0),
        }
        if tx_index not in (None, ""):
            extra["first_tx_index"] = int(tx_index)
        runtime.repair_lifecycle.mark_stage(
            candidate_id,
            stage="execution_started",
            extra=extra,
        )
        runtime.log_repair_execution_debug(
            candidate_id,
            "execution_started",
            sequence_id=extra["sequence_id"],
            individual_hash=extra["individual_hash"],
            tx_count=int(tx_count or 0),
            first_tx_index=extra.get("first_tx_index"),
        )

    def record_execution_exception(self, individual: Any, exc: Any, *, phase: str = "", tx_index: Optional[int] = None) -> None:
        runtime = self._repair_runtime()
        if runtime is None or not hasattr(individual, "repair_candidate_id"):
            return
        candidate_id = _text(getattr(individual, "repair_candidate_id", None)).strip()
        if not candidate_id or not self._is_failure_repair_individual(runtime, individual, candidate_id):
            return
        runtime._bump_pipeline_counter("repair_execution_exception", 1)
        message = _text(exc).strip() or exc.__class__.__name__
        extra = {
            "sequence_id": _text(getattr(individual, "sequence_id", None)).strip(),
            "individual_hash": _text(getattr(individual, "hash", None)).strip(),
            "exception_type": exc.__class__.__name__ if exc is not None else "",
            "exception_message": message,
            "phase": _text(phase).strip(),
        }
        if tx_index not in (None, ""):
            extra["tx_index"] = int(tx_index)
        runtime.failure_repair_summary["repair_execution_exception"] += 1
        runtime.repair_lifecycle.mark_stage(
            candidate_id,
            stage="execution_exception",
            extra=extra,
        )
        runtime.repair_lifecycle.mark_dropped(
            candidate_id,
            reason="execution_exception",
            extra=extra,
        )
        runtime.log_repair_execution_debug(
            candidate_id,
            "execution_exception",
            **extra,
        )

    def activate_repair_execution_grace(self, *, generation: int, reason: str) -> List[str]:
        pending_ids = [
            candidate_id
            for candidate_id, record in self.pending_selected_repair_candidates.items()
            if isinstance(record, dict) and record.get("submit_result") != "submitted"
        ]
        if not pending_ids:
            self.repair_execution_grace_candidate_ids = set()
            self.repair_execution_grace_active = False
            self.repair_execution_grace_generation = None
            self.repair_execution_grace_reason = ""
            return []
        self.repair_execution_grace_candidate_ids = set(pending_ids)
        self.repair_execution_grace_active = True
        self.repair_execution_grace_generation = int(generation)
        self.repair_execution_grace_reason = _text(reason).strip() or "selected_after_budget_end"
        return list(pending_ids)

    def finalize_repair_execution_grace(self, default_reason: str = "selected_after_budget_end") -> List[str]:
        unresolved: List[str] = []
        active_ids = list(self.repair_execution_grace_candidate_ids)
        for candidate_id in active_ids:
            if candidate_id not in self.pending_selected_repair_candidates:
                continue
            unresolved.append(candidate_id)
            self.record_repair_submission_outcome(
                candidate_id,
                submit_result="skipped",
                reason=default_reason,
                extra={
                    "queue_budget_remaining": 0,
                    "time_budget_remaining": 0.0,
                },
                terminal=True,
            )
        self.repair_execution_grace_candidate_ids = set()
        self.repair_execution_grace_active = False
        self.repair_execution_grace_generation = None
        self.repair_execution_grace_reason = ""
        return unresolved

    def _repair_submission_summary_payload(self) -> Dict[str, Any]:
        selected = int(self.failure_repair_summary.get("repair_candidate_selected", 0) or 0)
        submitted = int(self.failure_repair_summary.get("repair_candidate_execution_submitted", 0) or 0)
        attempted = int(self.repair_submission_summary.get("repair_candidate_submission_attempted", 0) or 0)
        succeeded = int(self.repair_submission_summary.get("repair_candidate_submission_succeeded", 0) or 0)
        skipped = int(self.repair_submission_summary.get("repair_candidate_submission_skipped", 0) or 0)
        failed = int(self.repair_submission_summary.get("repair_candidate_submission_failed", 0) or 0)
        selected_not_submitted = max(0, selected - submitted)
        return {
            "repair_candidate_submission_attempted": attempted,
            "repair_candidate_submission_succeeded": succeeded,
            "repair_candidate_submission_skipped": skipped,
            "repair_candidate_submission_failed": failed,
            "repair_selected_not_submitted": selected_not_submitted,
            "repair_selected_submitted_ratio": round(float(submitted) / float(selected), 6) if selected else 0.0,
            "repair_candidate_drop_reason_histogram": dict(self.repair_submit_false_by_reason),
        }

    def _observed_context(self, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        functions: List[str] = []
        senders: List[str] = []
        args: List[Any] = []
        values: List[Any] = []
        receivers: List[str] = []
        privileged_addresses: List[str] = []
        asset_flow_state_addresses: List[str] = []
        asset_flow_noop_state_addresses: List[str] = []
        asset_flow_arg_addresses: List[str] = []
        asset_flow_success_senders: List[str] = []
        asset_flow_success_sender_counts: Counter = Counter()
        asset_flow_success_sender_last_seen: Dict[str, int] = {}
        asset_flow_success_sender_order = 0
        ownership_or_role_event_seen_count = 0
        access_control_storage_write_seen_count = 0
        access_control_revert_reason_seen = False
        for event in events:
            if not isinstance(event, dict):
                continue
            function = _text(event.get("function")).strip()
            if function:
                functions.append(function)
            sender = _text(event.get("sender")).strip()
            if sender:
                senders.append(sender)
            if isinstance(event.get("args"), list):
                args.extend(event.get("args"))
            if event.get("value") not in (None, ""):
                values.append(event.get("value"))
            for key in ("to", "from", "callee"):
                value = _text(event.get(key)).strip()
                if value:
                    receivers.append(value)
            if self.runtime_pattern_family == "asset_flow_redirection":
                asset_flow_addresses = self._asset_flow_observed_addresses(event)
                receivers.extend(asset_flow_addresses)
                privileged_addresses.extend(asset_flow_addresses)
                asset_flow_state_addresses.extend(self._asset_flow_state_addresses(event))
                asset_flow_noop_state_addresses.extend(self._asset_flow_noop_state_addresses(event))
                asset_flow_arg_addresses.extend(self._asset_flow_setter_arg_addresses(event))
            lowered_function = function.lower()
            if any(token in lowered_function for token in self._access_control_runtime_keywords()):
                for value in list(event.get("args") or []):
                    if self._looks_like_address(value):
                        privileged_addresses.append(_text(value).strip())
            lowered_event = _text(event.get("event_type")).strip().lower()
            if lowered_event == "sstore":
                storage_text = " ".join(
                    [
                        _text(event.get("function")).strip().lower(),
                        _text(event.get("storage_slot")).strip().lower(),
                        _text(event.get("raw")).strip().lower(),
                    ]
                )
                if any(token in storage_text for token in self._access_control_runtime_keywords()):
                    access_control_storage_write_seen_count += 1
            if lowered_event in {"function_call", "external_call", "tx_end"}:
                if any(token in lowered_function for token in ("transferownership", "acceptownership", "grantrole", "revoke", "setadmin", "setoperator", "whitelist", "allowlist", "pause", "unpause")):
                    ownership_or_role_event_seen_count += 1
            if (
                self.runtime_pattern_family == "asset_flow_redirection"
                and lowered_event == "tx_end"
                and bool(event.get("success"))
                and sender
                and any(token in lowered_function for token in ASSET_FLOW_SETTER_HINTS)
            ):
                asset_flow_success_senders.append(sender)
                normalized_sender = sender.lower()
                asset_flow_success_sender_counts[normalized_sender] += 1
                asset_flow_success_sender_order += 1
                asset_flow_success_sender_last_seen[normalized_sender] = asset_flow_success_sender_order
            if self._access_control_revert_reason_triggered([event]):
                access_control_revert_reason_seen = True
        ranked_asset_flow_success_senders = [
            sender
            for sender, _ in sorted(
                asset_flow_success_sender_counts.items(),
                key=lambda item: (
                    -int(item[1]),
                    -int(asset_flow_success_sender_last_seen.get(item[0], 0)),
                    item[0],
                ),
            )
        ]
        if not ranked_asset_flow_success_senders:
            ranked_asset_flow_success_senders = _dedup_text(list(reversed(asset_flow_success_senders)), limit=8)
        return {
            "observed_functions": list(dict.fromkeys(functions))[-32:],
            "observed_senders": list(dict.fromkeys(senders))[-16:],
            "observed_args": args[-64:],
            "observed_values": values[-32:],
            "observed_receivers": list(dict.fromkeys(receivers))[-16:],
            "access_control_privileged_addresses": _dedup_text(privileged_addresses, limit=16),
            "asset_flow_state_addresses": _dedup_text(list(reversed(asset_flow_state_addresses)), limit=8),
            "asset_flow_noop_state_addresses": _dedup_text(list(reversed(asset_flow_noop_state_addresses)), limit=8),
            "asset_flow_arg_addresses": _dedup_text(list(reversed(asset_flow_arg_addresses)), limit=8),
            "asset_flow_success_senders": _dedup_text(ranked_asset_flow_success_senders, limit=8),
            "asset_flow_success_sender_counts": {
                sender: int(count)
                for sender, count in sorted(
                    asset_flow_success_sender_counts.items(),
                    key=lambda item: (
                        -int(item[1]),
                        -int(asset_flow_success_sender_last_seen.get(item[0], 0)),
                        item[0],
                    ),
                )
            },
            "ownership_or_role_event_seen_count": int(ownership_or_role_event_seen_count),
            "access_control_storage_write_seen_count": int(access_control_storage_write_seen_count),
            "access_control_revert_reason_seen": bool(access_control_revert_reason_seen),
        }

    def _progress_event_summary(self, events: Sequence[Dict[str, Any]], progress_delta: Dict[str, Any]) -> Dict[str, Any]:
        last = events[-1] if events else {}
        return {
            "global_tx_index": last.get("global_tx_index") if isinstance(last, dict) else None,
            "global_sequence_index": last.get("global_sequence_index") if isinstance(last, dict) else None,
            "event_count": len(events),
            "newly_covered": progress_delta.get("newly_covered") or [],
            "event_counts": progress_delta.get("event_counts") or {},
            "relevant_functions": progress_delta.get("relevant_functions") or [],
            "observed_senders": progress_delta.get("observed_senders") or [],
        }

    def _build_mapping_candidates(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        tfi_candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        mapping = ACTION_MAPPING.get(self.runtime_pattern_family, {}).get(next_missing)
        if not mapping:
            self.no_candidate_reasons["mapping_missing:{}".format(next_missing)] += 1
            return []
        if mapping.get("requires_attacker_contract"):
            self.no_candidate_reasons[_text(mapping.get("no_candidate_reason")).strip() or "unsupported_prestate"] += 1
            return []
        keywords = mapping.get("function_keywords") or []
        preferred = list(observed.get("observed_functions") or [])
        selected_functions = self.abi_index.select_by_keywords(keywords, preferred=preferred)
        if not selected_functions:
            self.no_candidate_reasons["no_abi_function_matches_checkpoint_mapping"] += 1
            return []
        source_candidates = [item for item in tfi_candidates if isinstance(item, dict)]
        built: List[Dict[str, Any]] = []
        if self._access_control_profile_active_for_candidate(observed):
            base_sequences = self._unique_sequences(
                self._access_control_state_sequences(observed, mapping, source_candidates)
                + self._sequence_variants(selected_functions, mapping, source_candidates)
            )
        else:
            base_sequences = self._sequence_variants(selected_functions, mapping, source_candidates)
        use_asset_flow_sender_hints = self._asset_flow_sender_context_available(observed)
        for sequence in base_sequences[: self.max_per_round * 2]:
            mapping_variants = [mapping]
            if self._access_control_profile_active_for_candidate(observed):
                mapping_variants = []
                for sender_policy in self._access_control_sender_policies(observed):
                    variant = dict(mapping)
                    variant["sender_policy"] = sender_policy
                    mapping_variants.append(variant)
            elif use_asset_flow_sender_hints:
                mapping_variants = []
                for sender_policy in self._access_control_sender_policies(observed):
                    variant = dict(mapping)
                    variant["sender_policy"] = sender_policy
                    mapping_variants.append(variant)
            for variant_index, mapping_variant in enumerate(mapping_variants[:3], start=1):
                candidate = self._new_candidate(
                    source="tfi",
                    next_missing=next_missing,
                    mapping=mapping_variant,
                    sequence=sequence,
                    observed=observed,
                    build_reason="checkpoint_action_mapping",
                )
                if self._access_control_profile_active_for_candidate(observed):
                    self._annotate_access_control_rescue_candidate(
                        candidate,
                        observed=observed,
                        variant="mapping_sender_variant_{}".format(variant_index),
                    )
                elif use_asset_flow_sender_hints:
                    self._attach_asset_flow_sender_hints(
                        candidate,
                        observed=observed,
                        variant="asset_flow_sender_variant_{}".format(variant_index),
                    )
                candidate = self._finalize_candidate_materialization(candidate, observed)
                built.append(candidate)
        return built

    def _build_agent_candidates(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        tfi_candidates: Sequence[Dict[str, Any]],
        recent_failed: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.llm_provider or not getattr(self.llm_provider, "is_available", lambda: False)():
            self.no_candidate_reasons["agent_provider_unavailable"] += 1
            return []
        if not getattr(self.llm_provider, "is_configured", lambda: False)():
            self.no_candidate_reasons["agent_provider_not_configured"] += 1
            return []
        prompt = self._agent_prompt(next_missing, observed, tfi_candidates, recent_failed)
        started = time.time()
        self.llm_call_count += 1
        try:
            payload = self.llm_provider.generate_json(prompt)
        except Exception as exc:
            ended = time.time()
            duration = max(0.0, ended - started)
            self.llm_error_count += 1
            self.llm_interaction_time_total_sec += duration
            self.llm_interaction_events.append(
                {
                    "stage": "checkpoint_guided_next_missing",
                    "started_at": started,
                    "ended_at": ended,
                    "error": str(exc),
                    "target_missing_checkpoint": next_missing,
                    "runtime_pattern_family": self.runtime_pattern_family,
                    "duration_sec": round(duration, 6),
                }
            )
            if self.phase_cost_env is not None:
                try:
                    record_phase_event(self.phase_cost_env, self.llm_interaction_events[-1])
                except Exception:
                    pass
            self.no_candidate_reasons["agent_llm_error"] += 1
            return []
        ended = time.time()
        duration = max(0.0, ended - started)
        self.llm_success_count += 1
        self.llm_interaction_time_total_sec += duration
        metadata = {}
        try:
            metadata = self.llm_provider.get_last_response_metadata()
        except Exception:
            metadata = {}
        event_payload = {
            "stage": "checkpoint_guided_next_missing",
            "started_at": started,
            "ended_at": ended,
            "target_missing_checkpoint": next_missing,
            "runtime_pattern_family": self.runtime_pattern_family,
            "duration_sec": round(duration, 6),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
        if isinstance(metadata, dict):
            event_payload.update(
                {
                    "model": metadata.get("model"),
                    "usage": metadata.get("usage"),
                    "cost_usd": metadata.get("cost_usd"),
                    "cost_estimation_mode": metadata.get("cost_estimation_mode"),
                }
            )
        self.llm_interaction_events.append(event_payload)
        self.llm_interaction_events = self.llm_interaction_events[-256:]
        if self.phase_cost_env is not None:
            try:
                record_phase_event(self.phase_cost_env, event_payload)
            except Exception:
                pass
        if not isinstance(payload, dict):
            self.no_candidate_reasons["agent_output_not_json_object"] += 1
            return []
        if payload.get("no_candidate"):
            reason = _text(payload.get("reason")).strip() or "agent_no_candidate"
            self.no_candidate_reasons["agent_no_candidate:{}".format(reason[:80])] += 1
            return []
        raw_sequence = payload.get("function_sequence") or payload.get("sequence_skeleton") or payload.get("guided_sequence")
        sequence = _clean_list(raw_sequence)
        mapping = ACTION_MAPPING.get(self.runtime_pattern_family, {}).get(next_missing, {"goal": "agent target checkpoint"})
        candidate = self._new_candidate(
            source="agent",
            next_missing=next_missing,
            mapping=mapping,
            sequence=sequence,
            observed=observed,
            build_reason="next_missing_checkpoint_agent",
            raw_agent_payload=payload,
        )
        candidate["confidence"] = _safe_float(payload.get("confidence"), candidate.get("confidence", 0.6))
        candidate = self._finalize_candidate_materialization(candidate, observed)
        return [candidate]

    def _agent_prompt(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        tfi_candidates: Sequence[Dict[str, Any]],
        recent_failed: Sequence[Dict[str, Any]],
    ) -> str:
        runtime_goal = self._runtime_goal_for_missing_checkpoint(next_missing, observed)
        return json.dumps(
            {
                "task": "runtime_soft_guidance_candidate",
                "current_best_level": self.progress.best_level,
                "covered_checkpoints": sorted(self.progress.covered),
                "runtime_focus_id": "RTF_{}_{}".format(_safe_token(self.runtime_pattern_family), _safe_token(next_missing)),
                "runtime_goal": runtime_goal,
                "runtime_pattern_family": self.runtime_pattern_family,
                "runtime_pattern_family_provenance": self.runtime_pattern_family_provenance,
                "recent_failed_candidates": list(recent_failed)[-8:],
                "abi_function_list": self.abi_index.names(),
                "mutable_abi_functions": [name for name in self.abi_index.names() if not self.abi_index.is_view(name)],
                "static_summary": self._static_summary_hints(),
                "available_sender_value_args_hints": observed,
                "allowed_parameter_source_types": [
                    "existing_pool",
                    "observed_trace",
                    "static_context",
                    "boundary_domain",
                    "llm_hypothesis",
                ],
                "constraints": {
                    "must_follow_runtime_goal": True,
                    "must_use_existing_abi_function": True,
                    "no_generic_advice": True,
                    "materializer_will_validate_before_execution": True,
                    "forbidden_pre_execution_sources": [
                        "audit_report_semantics",
                        "prepared_oracle_semantics",
                        "post_execution_baseline_results",
                    ],
                },
                "output_schema": {
                    "runtime_goal_id": "string",
                    "sequence_skeleton": ["functionName"],
                    "sender_constraints": "string",
                    "arg_constraints": "string",
                    "value_constraints": "string",
                    "prestate_requirements": "string",
                    "parameter_plan": [
                        {
                            "function": "string",
                            "arg": "string",
                            "source_type": "existing_pool|observed_trace|static_context|boundary_domain|llm_hypothesis",
                            "source_hint": "string",
                        }
                    ],
                    "confidence": 0.0,
                    "rationale": "string",
                    "no_candidate": False,
                    "reason": "string if no_candidate",
                },
            },
            ensure_ascii=True,
        )

    def _build_effect_forcing_candidates(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        tfi_candidates: Sequence[Dict[str, Any]],
        recent_failed: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        built: List[Dict[str, Any]] = []
        prestate_plans: List[Dict[str, Any]] = []
        if self.enable_prestate_builder:
            prestate_plans = self._build_prestate_plans(next_missing, observed, tfi_candidates)
            for plan in prestate_plans:
                _append_jsonl(self.prestate_candidates_path, plan)
        if self.enable_prefix_replay_suffix_mutation:
            built.extend(self._build_prefix_replay_candidates(next_missing, observed, prestate_plans, tfi_candidates))
        elif self.enable_prestate_builder:
            built.extend(self._build_prestate_candidates(next_missing, observed, prestate_plans))
        if self.enable_agent_effect_repair:
            built.extend(self._build_agent_effect_repair_candidates(next_missing, observed, tfi_candidates, recent_failed, prestate_plans))
        if self.enable_failure_repair_pipeline:
            built.extend(self._build_failure_repair_candidates(next_missing, observed))
        return built

    def _apply_failure_repair_priority(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        repair_candidates: List[Dict[str, Any]] = []
        observation_only_candidates: List[Dict[str, Any]] = []
        regular_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            if bool(candidate.get("failure_repair_candidate")):
                if bool(candidate.get("observation_only_repair")):
                    observation_only_candidates.append(candidate)
                else:
                    repair_candidates.append(candidate)
            else:
                regular_candidates.append(candidate)
        if not repair_candidates and not observation_only_candidates:
            return list(candidates)
        repair_limit = max(1, int(round(float(self.max_per_round) * float(self.failure_repair_priority_ratio or 0.0))))
        prioritized = repair_candidates[:repair_limit] + regular_candidates + observation_only_candidates
        self.repair_scheduling_summary["repair_candidates_prioritized"] += len(repair_candidates[:repair_limit])
        self.repair_scheduling_summary["regular_candidates_deferred"] += max(0, len(repair_candidates) - repair_limit)
        self.repair_scheduling_summary["observation_only_candidates_seen"] += len(observation_only_candidates)
        if observation_only_candidates:
            self._bump_pipeline_counter("observation_only_repair_dropped_or_deprioritized", len(observation_only_candidates))
            self._bump_pipeline_counter("observation_only_repair_deprioritized", len(observation_only_candidates))
        return prioritized

    def _build_prestate_plans(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        tfi_candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        mapping = ACTION_MAPPING.get(self.runtime_pattern_family, {}).get(next_missing, {})
        selected = self._select_effect_functions(next_missing, observed, mapping)
        if not selected:
            self.prestate_summary["no_candidate"] += 1
            self.no_candidate_reasons["prestate_builder_no_abi_function"] += 1
            return []
        source_prefix = self._best_prefix_from_context(observed, tfi_candidates)
        plans: List[Dict[str, Any]] = []
        for index, sequence in enumerate(selected[: max(1, self.max_per_round * 2)], start=1):
            required_prefix = list(sequence[:-1]) if len(sequence) > 1 else []
            suffix_actions = [sequence[-1]] if sequence else []
            plan = {
                "prestate_plan_id": "PRESTATE_{:04d}_{}_{}".format(
                    self.last_candidate_id + index,
                    _safe_token(self.runtime_pattern_family),
                    _safe_token(next_missing),
                ),
                "runtime_pattern_family": self.runtime_pattern_family,
                "next_missing_checkpoint": next_missing,
                "required_prefix": required_prefix,
                "suffix_actions": suffix_actions,
                "function_sequence": list(sequence),
                "source_prefix": source_prefix,
                "sender_binding": self._sender_binding_for_family(),
                "beneficiary_binding": self._beneficiary_binding_for_family(),
                "amount_binding": self._amount_binding_for_family(next_missing),
                "arg_binding": mapping.get("argument_policy") or "abi_default_or_boundary_mix",
                "expected_effect": self._expected_effect_for_checkpoint(next_missing, mapping),
                "build_status": "built",
                "no_candidate_reason": "",
                "module_source": "prestate_builder",
            }
            plans.append(plan)
            self.prestate_summary["plans_built"] += 1
        return plans

    def _select_effect_functions(self, next_missing: str, observed: Dict[str, Any], mapping: Dict[str, Any]) -> List[List[str]]:
        observed_functions = observed.get("observed_functions") or []
        affected: List[str] = []

        def pick(keywords: Sequence[str], preferred: Sequence[str] = ()) -> List[str]:
            return self.abi_index.select_by_keywords(keywords, preferred=list(preferred) + affected + list(observed_functions))

        family = self.runtime_pattern_family
        sequences: List[List[str]] = []
        if family == "repeat_claim_no_reset":
            setup = pick(["approve", "deposit", "stake", "mint", "buy"])
            claim = pick(["claim", "reward", "harvest", "withdraw"])
            if "second" in next_missing or "abnormal" in next_missing:
                if claim:
                    prefix = setup[0] if setup else ""
                    sequences.append(([prefix] if prefix and prefix != claim[0] else []) + [claim[0], claim[0]])
            elif setup and claim:
                sequences.append([setup[0], claim[0]])
            elif claim:
                sequences.append([claim[0]])
        elif family == "asset_flow_redirection":
            setters = self._asset_flow_state_writer_candidates(observed)
            sinks = self._asset_flow_business_effect_candidates(observed)
            if setters and sinks and setters[0] != sinks[0]:
                sequences.append([setters[0], sinks[0]])
            if setters:
                sequences.append([setters[0]])
            if sinks:
                sequences.append([sinks[0]])
        elif family == "initialization_takeover":
            init = pick(["init", "initialize", "setup"])
            privileged = pick(["owner", "admin", "set", "upgrade", "withdraw", "mint"])
            if init and privileged and init[0] != privileged[0]:
                sequences.append([init[0], privileged[0]])
            elif init:
                sequences.append([init[0]])
        elif family == "critical_parameter_abuse":
            setter = pick(["set", "update", "config", "fee", "rate", "limit", "price", "slippage"])
            downstream = pick(["transfer", "swap", "claim", "withdraw", "deposit", "harvest", "update", "collect", "buy", "sell"])
            if setter and downstream and setter[0] != downstream[0]:
                sequences.append([setter[0], downstream[0]])
            elif setter:
                sequences.append([setter[0]])
        elif family == "access_control_state_change":
            sequences.extend(self._access_control_state_sequences(observed, mapping, []))
            if not sequences:
                protected = pick(["set", "admin", "owner", "mint", "withdraw", "upgrade", "pause"])
                effect = pick(["withdraw", "mint", "update", "transfer", "set"])
                if protected and effect and protected[0] != effect[0]:
                    sequences.append([protected[0], effect[0]])
                elif protected:
                    sequences.append([protected[0]])
        elif family == "arithmetic_accounting_invariant":
            prefix = pick(["add", "update", "deposit", "withdraw", "mint", "burn", "share", "reward", "convert"])
            suffix_keywords = ["updatePool", "harvest", "withdraw", "claim", "deposit", "mint", "burn", "massupdate", "update"]
            if _text(next_missing).strip() != "invariant_check_possible":
                suffix_keywords.extend(["balance", "pending", "reward"])
            suffix = pick(suffix_keywords)
            if prefix and suffix and prefix[0] != suffix[0]:
                sequences.append([prefix[0], suffix[0]])
            elif prefix:
                sequences.append([prefix[0]])
        else:
            selected = pick(mapping.get("function_keywords") or [])
            if selected:
                sequences.append([selected[0]])
        extra_function_names = pick(mapping.get("function_keywords") or [])
        if family == "asset_flow_redirection":
            extra_function_names = self._asset_flow_state_writer_candidates(observed)
        for function_name in extra_function_names:
            sequences.append([function_name])
        return self._unique_sequences(sequences)

    def _access_control_static_function_rows(self) -> List[Dict[str, Any]]:
        rows = self._static_summary_hints().get("functions")
        return [dict(item) for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []

    def _access_control_profile_requested(self) -> bool:
        return self.guidance_profile == "access_control_state_push"

    def _access_control_profile_active_for_candidate(self, observed: Optional[Dict[str, Any]] = None) -> bool:
        observed = observed if isinstance(observed, dict) else {}
        if not self._access_control_profile_requested():
            return False
        if self._access_control_heuristic_match(observed):
            return True
        return self.runtime_pattern_family == "access_control_state_change" and self._access_control_runtime_evidence_seen(observed)

    def _access_control_heuristic_match(self, observed: Optional[Dict[str, Any]] = None) -> bool:
        observed = observed if isinstance(observed, dict) else {}
        privileged = self._access_control_role_writer_candidates(observed)
        sensitive = self._access_control_sensitive_suffix_candidates(observed)
        sender_count = len(_clean_list(observed.get("observed_senders")))
        revert_hint = bool(observed.get("access_control_revert_reason_seen"))
        matched = bool(privileged and sensitive) or bool(privileged and sender_count >= 2) or revert_hint
        self.profile_effect_summary["access_control_heuristic_matched"] = bool(matched)
        self.profile_effect_summary["matched_access_control_functions_count"] = int(len(privileged))
        self.profile_effect_summary["matched_sensitive_functions_count"] = int(len(sensitive))
        return bool(matched)

    def _access_control_state_push_enabled(self, observed: Optional[Dict[str, Any]] = None) -> bool:
        return self._access_control_profile_active_for_candidate(observed)

    def _is_access_control_profile_candidate(self, candidate: Dict[str, Any]) -> bool:
        if not isinstance(candidate, dict):
            return False
        if bool(candidate.get("access_control_rescue_candidate")):
            return True
        if bool(candidate.get("access_control_profile_active")):
            return True
        return _text(candidate.get("guidance_profile")).strip() == "access_control_state_push"

    def _access_control_runtime_evidence_seen(self, observed: Optional[Dict[str, Any]] = None) -> bool:
        observed = observed if isinstance(observed, dict) else {}
        return bool(
            observed.get("access_control_revert_reason_seen")
            or _safe_int(observed.get("ownership_or_role_event_seen_count"), 0) > 0
            or _safe_int(observed.get("access_control_storage_write_seen_count"), 0) > 0
            or _safe_int(self.profile_effect_summary.get("unauthorized_revert_seen_count"), 0) > 0
            or _safe_int(self.profile_effect_summary.get("ownership_or_role_event_seen_count"), 0) > 0
            or _safe_int(self.profile_effect_summary.get("access_control_storage_write_seen_count"), 0) > 0
        )

    def _access_control_productive_signal_seen(self, observed: Optional[Dict[str, Any]] = None) -> bool:
        observed = observed if isinstance(observed, dict) else {}
        return bool(
            _safe_int(observed.get("ownership_or_role_event_seen_count"), 0) > 0
            or _safe_int(observed.get("access_control_storage_write_seen_count"), 0) > 0
            or _safe_int(self.profile_effect_summary.get("setup_then_sensitive_success_count"), 0) > 0
            or _safe_int(self.profile_effect_summary.get("ownership_or_role_event_seen_count"), 0) > 0
            or _safe_int(self.profile_effect_summary.get("access_control_storage_write_seen_count"), 0) > 0
        )

    def _access_control_profile_budget_limit(self) -> int:
        if not self._access_control_profile_requested():
            return int(self.max_per_round)
        heuristic_matched = bool(self.profile_effect_summary.get("access_control_heuristic_matched"))
        runtime_evidence = self._access_control_runtime_evidence_seen()
        productive = self._access_control_productive_signal_seen()
        stagnation_floor = max(1, int(self.trigger_stagnation or 0))
        hard_stagnation = self.progress.vspc_stagnation_rounds >= max(4, stagnation_floor * 2)
        very_hard_stagnation = self.progress.vspc_stagnation_rounds >= max(6, stagnation_floor * 3)
        base_cap = 1 if self.max_per_round <= 4 else max(1, int(round(float(self.max_per_round) * 0.3)))
        cap = base_cap
        if productive and not hard_stagnation:
            cap = min(int(self.max_per_round), cap + 1)
        if not heuristic_matched and not runtime_evidence:
            cap = 0
        elif hard_stagnation and not productive:
            cap = min(cap, 1)
        if very_hard_stagnation and not productive and not runtime_evidence:
            cap = 0
        fallback_active = bool((not heuristic_matched and not runtime_evidence) or (hard_stagnation and not productive))
        self.profile_effect_summary["access_control_profile_budget_cap"] = int(max(0, cap))
        self.profile_effect_summary["access_control_profile_fallback_active"] = fallback_active
        return max(0, min(int(self.max_per_round), int(cap)))

    def _access_control_profile_boost_allowed(self, candidate: Dict[str, Any]) -> bool:
        if not self._access_control_profile_requested() or not self._is_access_control_profile_candidate(candidate):
            return False
        heuristic_matched = bool(self.profile_effect_summary.get("access_control_heuristic_matched"))
        runtime_evidence = self._access_control_runtime_evidence_seen()
        productive = self._access_control_productive_signal_seen()
        stagnation_floor = max(1, int(self.trigger_stagnation or 0))
        hard_stagnation = self.progress.vspc_stagnation_rounds >= max(4, stagnation_floor * 2)
        if not heuristic_matched and not runtime_evidence:
            return False
        if hard_stagnation and not productive and not bool(candidate.get("failure_repair_candidate")):
            return False
        return True

    def _apply_access_control_profile_budget(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        working = [dict(item) for item in candidates if isinstance(item, dict)]
        if not self._access_control_profile_requested() or not working:
            return working
        profile_candidates: List[Dict[str, Any]] = []
        regular_candidates: List[Dict[str, Any]] = []
        for candidate in working:
            if self._is_access_control_profile_candidate(candidate):
                profile_candidates.append(candidate)
            else:
                regular_candidates.append(candidate)
        if not profile_candidates:
            self.profile_effect_summary["access_control_profile_fallback_active"] = False
            return working
        cap = self._access_control_profile_budget_limit()
        self.strong_scheduling_summary["access_control_profile_candidates_considered"] += len(profile_candidates)
        self.strong_scheduling_summary["access_control_profile_candidate_cap_sum"] += int(cap)
        if cap <= 0:
            self.strong_scheduling_summary["access_control_profile_fallback_rounds"] += 1
            self.strong_scheduling_summary["access_control_profile_candidates_deferred"] += len(profile_candidates)
            return regular_candidates
        kept = profile_candidates[:cap]
        deferred = profile_candidates[cap:]
        self.strong_scheduling_summary["access_control_profile_candidates_kept"] += len(kept)
        if deferred:
            self.strong_scheduling_summary["access_control_profile_fallback_rounds"] += 1
            self.strong_scheduling_summary["access_control_profile_candidates_deferred"] += len(deferred)
        if regular_candidates:
            return kept[:1] + regular_candidates + kept[1:]
        return kept

    def _access_control_runtime_keywords(self) -> Tuple[str, ...]:
        return (
            "owner",
            "admin",
            "role",
            "minter",
            "operator",
            "manager",
            "govern",
            "auth",
            "permission",
            "allow",
            "whitelist",
            "allowlist",
            "pause",
            "paused",
            "pendingowner",
        )

    def _asset_flow_observed_addresses(self, event: Dict[str, Any]) -> List[str]:
        if self.runtime_pattern_family != "asset_flow_redirection" or not isinstance(event, dict):
            return []
        function_name = _text(event.get("function")).strip().lower()
        text = CheckpointProgressMirror._event_text(event)
        if not (
            any(token in function_name for token in ASSET_FLOW_SETTER_HINTS)
            or any(token in text for token in ASSET_FLOW_STATE_HINTS)
        ):
            return []
        values: List[str] = []
        for key in ("old_value", "new_value", "value", "args", "to", "from", "from_addr", "address"):
            values.extend(_normalized_address_candidates(event.get(key)))
        return _dedup_text(values, limit=8)

    def _asset_flow_state_addresses(self, event: Dict[str, Any]) -> List[str]:
        if self.runtime_pattern_family != "asset_flow_redirection" or not isinstance(event, dict):
            return []
        if _text(event.get("event_type")).strip().lower() != "sstore":
            return []
        function_name = _text(event.get("function")).strip().lower()
        text = CheckpointProgressMirror._event_text(event)
        if not (
            any(token in function_name for token in ASSET_FLOW_SETTER_HINTS)
            or any(token in text for token in ASSET_FLOW_STATE_HINTS)
        ):
            return []
        values: List[str] = []
        # `_observed_context()` reverses the aggregated list so the most recent post-write
        # value becomes the first current-state candidate.
        for key in ("old_value", "new_value"):
            values.extend(_normalized_address_candidates(event.get(key)))
        return _dedup_text(values, limit=4)

    def _asset_flow_noop_state_addresses(self, event: Dict[str, Any]) -> List[str]:
        if self.runtime_pattern_family != "asset_flow_redirection" or not isinstance(event, dict):
            return []
        if _text(event.get("event_type")).strip().lower() != "sstore":
            return []
        function_name = _text(event.get("function")).strip().lower()
        text = CheckpointProgressMirror._event_text(event)
        if not (
            any(token in function_name for token in ASSET_FLOW_SETTER_HINTS)
            or any(token in text for token in ASSET_FLOW_STATE_HINTS)
        ):
            return []
        old_values = _dedup_text(_normalized_address_candidates(event.get("old_value")), limit=4)
        new_values = _dedup_text(_normalized_address_candidates(event.get("new_value")), limit=4)
        if not old_values or not new_values:
            return []
        if {item.lower() for item in old_values} != {item.lower() for item in new_values}:
            return []
        return list(new_values)

    def _asset_flow_setter_arg_addresses(self, event: Dict[str, Any]) -> List[str]:
        if self.runtime_pattern_family != "asset_flow_redirection" or not isinstance(event, dict):
            return []
        function_name = _text(event.get("function")).strip().lower()
        if not any(token in function_name for token in ASSET_FLOW_SETTER_HINTS):
            return []
        values = _normalized_address_candidates(event.get("args"))
        return _dedup_text(values, limit=4)

    def _asset_flow_function_context_text(self, name: str) -> str:
        canonical = _text(name).strip()
        if not canonical:
            return ""
        lowered = canonical.lower()
        parts: List[str] = [lowered]
        for row in self._access_control_static_function_rows():
            if _text(row.get("name")).strip().lower() != lowered:
                continue
            parts.extend(_normalize_text_list(row.get("modifiers")))
            parts.extend(_normalize_text_list(row.get("reads")))
            parts.extend(_normalize_text_list(row.get("writes")))
            parts.extend(_normalize_text_list(row.get("external_calls")))
            parts.extend(_normalize_text_list(row.get("low_level_calls")))
            break
        static_summary = self._static_summary_hints()
        for row in list(static_summary.get("storage_dependencies") or []):
            if not isinstance(row, dict):
                continue
            row_name = _text(row.get("function")).strip().lower()
            if row_name != lowered:
                continue
            parts.extend(_normalize_text_list(row.get("reads")))
            parts.extend(_normalize_text_list(row.get("writes")))
            break
        for row in list(static_summary.get("dangerous_operations") or []):
            if not isinstance(row, dict):
                continue
            row_name = _text(row.get("function")).strip().lower()
            if row_name != lowered:
                continue
            parts.extend(_normalize_text_list(row.get("effects")))
            break
        return " ".join(_text(item).strip().lower() for item in parts if _text(item).strip())

    def _asset_flow_relevant_state_tokens(self, observed: Optional[Dict[str, Any]] = None) -> List[str]:
        observed = observed if isinstance(observed, dict) else {}
        contexts: List[str] = []
        for name in self._asset_flow_state_writer_candidates(observed)[:3]:
            context = self._asset_flow_function_context_text(name)
            if context:
                contexts.append(context)
        for name in _clean_list(observed.get("observed_functions")):
            lowered = _text(name).strip().lower()
            if any(token in lowered for token in ASSET_FLOW_SETTER_HINTS):
                contexts.append(self._asset_flow_function_context_text(name) or lowered)
        joined = " ".join(contexts).lower()
        return _dedup_text([token for token in ASSET_FLOW_STATE_HINTS if token in joined], limit=6)

    def _asset_flow_state_writer_candidates(self, observed: Optional[Dict[str, Any]] = None) -> List[str]:
        observed = observed if isinstance(observed, dict) else {}
        static_summary = self._static_summary_hints()
        preferred = _clean_list(observed.get("observed_functions"))
        candidates: List[str] = []
        candidates.extend(preferred)
        for row in list(static_summary.get("storage_dependencies") or []):
            if not isinstance(row, dict):
                continue
            name = _text(row.get("function")).strip()
            if name:
                candidates.append(name)
        candidates.extend(
            self.abi_index.select_by_keywords(
                list(ASSET_FLOW_SETTER_HINTS) + ["update", "config", "change", "assign"],
                preferred=preferred,
            )
        )

        ranked: List[Tuple[int, int, str]] = []
        for order, name in enumerate(_dedup_text(candidates)):
            if not self.abi_index.has(name) or self.abi_index.is_view(name):
                continue
            canonical = self.abi_index.by_lower[name.lower()]
            lowered = canonical.lower()
            if self._is_observation_only_action(canonical):
                continue
            context = self._asset_flow_function_context_text(canonical)
            meta = self.abi_index.get(canonical) or {}
            inputs = [_text(item).strip().lower() for item in list(meta.get("inputs") or [])]
            has_asset_state_hint = any(token in context for token in ASSET_FLOW_STATE_HINTS)
            if not has_asset_state_hint and not any(token in lowered for token in ASSET_FLOW_STATE_HINTS):
                continue
            if any(token in lowered for token in ("transferownership", "acceptownership", "pendingowner", "newowner")):
                continue
            score = 0
            if any(token in lowered for token in ("set", "update", "change", "config", "assign")):
                score += 4
            if has_asset_state_hint:
                score += 4
            if any("address" in item for item in inputs):
                score += 3
            if any(token in lowered for token in ("owner", "admin", "role", "pause", "unpause", "blacklist", "whitelist")):
                score -= 3
            ranked.append((-score, order, canonical))
        ranked.sort()
        return _dedup_text([name for _, _, name in ranked], limit=8)

    def _asset_flow_business_effect_candidates(self, observed: Optional[Dict[str, Any]] = None) -> List[str]:
        observed = observed if isinstance(observed, dict) else {}
        static_summary = self._static_summary_hints()
        preferred = _clean_list(observed.get("observed_functions"))
        relevant_state_tokens = self._asset_flow_relevant_state_tokens(observed)
        restrictive_prefix_seen = any(
            token in _text(name).strip().lower()
            for name in preferred
            for token in ("pause", "freeze", "blacklist", "lock", "ban", "block", "deny", "forbid")
        )
        candidates: List[str] = []
        candidates.extend(preferred)
        for row in list(static_summary.get("dangerous_operations") or []):
            if not isinstance(row, dict):
                continue
            name = _text(row.get("function")).strip()
            if name:
                candidates.append(name)
        candidates.extend(
            self.abi_index.select_by_keywords(
                list(ASSET_FLOW_BUSINESS_HINTS) + ["transferfrom", "mint", "burn", "increase", "decrease"],
                preferred=preferred,
            )
        )

        ranked: List[Tuple[int, int, str]] = []
        for order, name in enumerate(_dedup_text(candidates)):
            if not self.abi_index.has(name) or self.abi_index.is_view(name):
                continue
            canonical = self.abi_index.by_lower[name.lower()]
            lowered = canonical.lower()
            if self._is_observation_only_action(canonical):
                continue
            if any(token in lowered for token in ("transferownership", "acceptownership", "pendingowner", "newowner")):
                continue
            if any(token in lowered for token in ("owner", "admin", "role", "pause", "unpause", "whitelist", "blacklist", "setlawenforcementrole")):
                continue
            if any(token in lowered for token in ASSET_FLOW_STATE_HINTS) and any(token in lowered for token in ("set", "update", "change", "config", "controller", "receiver", "treasury", "wallet", "beneficiary", "collector", "router", "tax", "marketing", "dev")):
                continue
            score = 0
            context = self._asset_flow_function_context_text(canonical)
            if any(token in lowered for token in ("transferfrom", "transfer", "withdraw", "decrease", "increase", "claim", "redeem", "collect", "harvest", "sell", "buy", "swap", "mint", "burn", "send")):
                score += 5
            if relevant_state_tokens and any(token in context for token in relevant_state_tokens):
                score += 5
            if any(token in relevant_state_tokens for token in ("controller", "supplycontroller")):
                if "onlysupplycontroller" in context:
                    score += 6
                if any(token in lowered for token in ("increase", "mint")):
                    score += 4
                elif any(token in lowered for token in ("decrease", "burn")):
                    score += 2
            if "whennotpaused" in context:
                score -= 2
                if restrictive_prefix_seen:
                    score -= 4
            if any(token in lowered for token in ("pause", "owner", "admin", "role")):
                score -= 4
            ranked.append((-score, order, canonical))
        ranked.sort()
        return _dedup_text([name for _, _, name in ranked], limit=8)

    def _asset_flow_sender_context_available(self, observed: Optional[Dict[str, Any]] = None) -> bool:
        observed = observed if isinstance(observed, dict) else {}
        if self.runtime_pattern_family != "asset_flow_redirection":
            return False
        return bool(
            _normalize_text_list(observed.get("access_control_privileged_addresses"))
            or _normalize_text_list(observed.get("observed_receivers"))
            or observed.get("access_control_revert_reason_seen")
        )

    def _looks_like_address(self, value: Any) -> bool:
        text = _text(value).strip()
        return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", text))

    def _access_control_revert_reason_tokens(self, events: Sequence[Dict[str, Any]]) -> List[str]:
        reasons: List[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            for key in ("revert_reason", "error"):
                text = _text(event.get(key)).strip()
                if text:
                    reasons.append(text)
        return reasons

    def _access_control_revert_reason_triggered(self, events: Sequence[Dict[str, Any]]) -> bool:
        tokens = (
            "onlyowner",
            "only owner",
            "onlyadmin",
            "only admin",
            "onlyrole",
            "missing role",
            "not authorized",
            "unauthorized",
            "access denied",
            "forbidden",
            "not owner",
            "not admin",
            "onlysupplycontroller",
            "only supplycontroller",
            "only controller",
        )
        for reason in self._access_control_revert_reason_tokens(events):
            lowered = reason.lower()
            if any(token in lowered for token in tokens):
                return True
        return False

    def _access_control_candidate_address_pool(self, observed: Optional[Dict[str, Any]] = None) -> List[str]:
        observed = observed if isinstance(observed, dict) else {}
        addresses: List[str] = []
        if self.deployer_account:
            addresses.append(self.deployer_account)
        addresses.extend(self.accounts[:6])
        for key in (
            "access_control_owner_addresses",
            "access_control_admin_addresses",
            "access_control_pending_owner_addresses",
            "access_control_privileged_addresses",
        ):
            addresses.extend(_normalize_text_list(observed.get(key)))
        addresses.extend(_normalize_text_list(observed.get("observed_senders")))
        addresses.extend(_normalize_text_list(observed.get("observed_receivers")))
        for value in _clean_any_list(observed.get("observed_args")):
            if self._looks_like_address(value):
                addresses.append(_text(value).strip())
        if self.contract_address:
            addresses.append(self.contract_address)
        return _dedup_text([addr for addr in addresses if self._looks_like_address(addr)], limit=12)

    def _access_control_binding_hints(self, observed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        observed = observed if isinstance(observed, dict) else {}
        pool = self._access_control_candidate_address_pool(observed)
        owner_candidates = _dedup_text(
            _normalize_text_list(observed.get("access_control_owner_addresses"))
            + ([self.deployer_account] if self.deployer_account else [])
            + pool[:1],
            limit=4,
        )
        admin_candidates = _dedup_text(
            _normalize_text_list(observed.get("access_control_admin_addresses"))
            + owner_candidates
            + pool[1:3],
            limit=4,
        )
        pending_owner_candidates = _dedup_text(
            _normalize_text_list(observed.get("access_control_pending_owner_addresses"))
            + pool[1:4],
            limit=4,
        )
        attacker_candidates = [addr for addr in pool if addr not in set(owner_candidates + admin_candidates + pending_owner_candidates)]
        if not attacker_candidates:
            attacker_candidates = pool[-2:] or pool[:1]
        return {
            "deployer": self.deployer_account or (pool[0] if pool else ""),
            "owner_candidates": owner_candidates,
            "admin_candidates": admin_candidates,
            "pending_owner_candidates": pending_owner_candidates,
            "attacker_candidates": _dedup_text(attacker_candidates, limit=4),
            "contract_self": self.contract_address or "",
            "address_pool": pool,
        }

    def _asset_flow_binding_hints(self, observed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        observed = observed if isinstance(observed, dict) else {}

        def _clean_addresses(values: Sequence[Any], *, allow_zero: bool = False, allow_contract_self: bool = True) -> List[str]:
            output: List[str] = []
            seen = set()
            zero_address = "0x" + ("0" * 40)
            contract_self = _text(self.contract_address).strip().lower()
            for value in values:
                text = _text(value).strip()
                lowered = text.lower()
                if not self._looks_like_address(text):
                    continue
                if not allow_zero and lowered == zero_address:
                    continue
                if not allow_contract_self and contract_self and lowered == contract_self:
                    continue
                if lowered in seen:
                    continue
                seen.add(lowered)
                output.append(text)
            return output

        observed_state_candidates = _clean_addresses(
            _normalize_text_list(observed.get("asset_flow_noop_state_addresses"))
            + _normalize_text_list(observed.get("asset_flow_state_addresses"))
        )
        if not observed_state_candidates:
            observed_state_candidates = _clean_addresses(
                _normalize_text_list(observed.get("access_control_privileged_addresses"))
            )
        observed_sender_candidates = _clean_addresses(
            _normalize_text_list(observed.get("observed_senders"))
        )
        observed_receiver_candidates = _clean_addresses(
            _normalize_text_list(observed.get("observed_receivers"))
        )
        observed_success_sender_candidates = _clean_addresses(
            _normalize_text_list(observed.get("asset_flow_success_senders"))
        )
        observed_setter_arg_candidates = _clean_addresses(
            _normalize_text_list(observed.get("asset_flow_arg_addresses"))
        )
        observed_arg_candidates: List[str] = []
        for value in _clean_any_list(observed.get("observed_args")):
            observed_arg_candidates.extend(_normalized_address_candidates(value))
        observed_arg_candidates = _clean_addresses(observed_arg_candidates)
        preferred_sender_candidates = _clean_addresses(
            observed_success_sender_candidates
            + observed_sender_candidates
            + self.accounts[:6]
            + ([self.deployer_account] if self.deployer_account else []),
        )
        state_change_candidates = _clean_addresses(
            observed_setter_arg_candidates
            + observed_receiver_candidates
            + observed_arg_candidates
            + observed_sender_candidates
            + self.accounts[:6]
            + preferred_sender_candidates,
            allow_contract_self=False,
        )
        current_state_set = {item.lower() for item in observed_state_candidates}
        state_change_candidates = [
            addr for addr in state_change_candidates
            if addr.lower() not in current_state_set
        ]
        preferred_sender_set = {item.lower() for item in preferred_sender_candidates}
        state_change_candidates = [
            addr for addr in state_change_candidates
            if addr.lower() not in preferred_sender_set
        ] + [
            addr for addr in state_change_candidates
            if addr.lower() in preferred_sender_set
        ]

        addresses: List[str] = list(preferred_sender_candidates)
        addresses.extend(observed_state_candidates)
        addresses.extend(state_change_candidates)
        addresses.extend(observed_receiver_candidates)
        addresses.extend(observed_arg_candidates)
        if self.deployer_account:
            addresses.append(self.deployer_account)
        if self.contract_address:
            addresses.append(self.contract_address)
        pool = _dedup_text([addr for addr in addresses if self._looks_like_address(addr)], limit=12)
        owner_candidates = _dedup_text(
            preferred_sender_candidates
            + pool[:2],
            limit=4,
        )
        admin_candidates = _dedup_text(
            preferred_sender_candidates[1:]
            + owner_candidates
            + pool[1:3],
            limit=4,
        )
        pending_owner_candidates = _dedup_text(state_change_candidates[:4] + pool[1:4], limit=4)
        attacker_candidates = [
            addr for addr in state_change_candidates
            if addr not in set(owner_candidates + admin_candidates + pending_owner_candidates)
        ]
        if not attacker_candidates:
            attacker_candidates = [
                addr for addr in pool
                if addr not in set(owner_candidates + admin_candidates + pending_owner_candidates)
            ]
        if not attacker_candidates:
            attacker_candidates = state_change_candidates[:2] or pool[-2:] or pool[:1]
        return {
            "deployer": self.deployer_account or (pool[0] if pool else ""),
            "owner_candidates": owner_candidates,
            "admin_candidates": admin_candidates,
            "pending_owner_candidates": pending_owner_candidates,
            "attacker_candidates": _dedup_text(attacker_candidates, limit=4),
            "preferred_sender_candidates": _dedup_text(preferred_sender_candidates, limit=6),
            "current_state_candidates": _dedup_text(observed_state_candidates, limit=4),
            "state_change_candidates": _dedup_text(state_change_candidates, limit=6),
            "contract_self": self.contract_address or "",
            "address_pool": pool,
        }

    def _asset_flow_replay_prefix(self, source_prefix: Sequence[str], plan_required_prefix: Sequence[str]) -> List[str]:
        combined = _dedup_text(
            _clean_list(source_prefix) + _clean_list(plan_required_prefix),
            limit=6,
        )
        if not combined:
            return []

        useful: List[str] = []
        for name in combined:
            lowered = _text(name).strip().lower()
            if not lowered or self._is_observation_only_action(name):
                continue
            if any(token in lowered for token in ("pause", "freeze", "blacklist", "lock", "ban", "block", "deny", "forbid")):
                continue
            if any(token in lowered for token in ASSET_FLOW_BUSINESS_HINTS) and not any(token in lowered for token in ASSET_FLOW_SETTER_HINTS):
                continue
            if (
                any(token in lowered for token in ("init", "initialize", "setup", "bootstrap", "owner", "admin", "auth", "authorize", "role", "whitelist", "allowlist"))
                or any(token in lowered for token in ASSET_FLOW_SETTER_HINTS)
            ):
                useful.append(name)

        if not useful:
            useful = [
                name
                for name in combined
                if _text(name).strip()
                and not self._is_observation_only_action(name)
                and not any(
                    token in _text(name).strip().lower()
                    for token in ("pause", "freeze", "blacklist", "lock", "ban", "block", "deny", "forbid")
                )
            ]

        setter_names = [
            name for name in useful
            if any(token in _text(name).strip().lower() for token in ASSET_FLOW_SETTER_HINTS)
        ]
        if setter_names:
            useful = [
                name for name in useful
                if not any(token in _text(name).strip().lower() for token in ASSET_FLOW_SETTER_HINTS)
            ][-2:] + setter_names[-1:]
        return _dedup_text(useful, limit=3)

    def _attach_access_control_profile_metadata(self, candidate: Dict[str, Any], observed: Optional[Dict[str, Any]] = None) -> None:
        if not self._access_control_state_push_enabled(observed):
            return
        binding_hints = self._access_control_binding_hints(observed)
        existing_hints = candidate.get("access_control_binding_hints") if isinstance(candidate.get("access_control_binding_hints"), dict) else {}

        def _merged_hint_list(key: str, limit: Optional[int] = None) -> List[str]:
            return _dedup_text(
                _normalize_text_list(existing_hints.get(key)) + _normalize_text_list(binding_hints.get(key)),
                limit=limit,
            )

        merged_hints = dict(binding_hints)
        if existing_hints:
            merged_hints.update(
                {
                    "deployer": _text(existing_hints.get("deployer")).strip() or _text(binding_hints.get("deployer")).strip(),
                    "owner_candidates": _merged_hint_list("owner_candidates", limit=4),
                    "admin_candidates": _merged_hint_list("admin_candidates", limit=4),
                    "pending_owner_candidates": _merged_hint_list("pending_owner_candidates", limit=4),
                    "attacker_candidates": _merged_hint_list("attacker_candidates", limit=4),
                    "preferred_sender_candidates": _merged_hint_list("preferred_sender_candidates", limit=6),
                    "current_state_candidates": _merged_hint_list("current_state_candidates", limit=4),
                    "state_change_candidates": _merged_hint_list("state_change_candidates", limit=6),
                    "contract_self": _text(existing_hints.get("contract_self")).strip() or _text(binding_hints.get("contract_self")).strip(),
                    "address_pool": _merged_hint_list("address_pool", limit=12),
                }
            )

        candidate["guidance_profile"] = _text(candidate.get("guidance_profile")).strip() or "access_control_state_push"
        candidate["access_control_address_pool"] = list(merged_hints.get("address_pool") or [])
        candidate["access_control_binding_hints"] = dict(merged_hints)
        candidate["access_control_profile_active"] = True
        candidate["access_control_profile_tag"] = "generic_access_control_state_push"
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        metadata["guidance_profile"] = _text(metadata.get("guidance_profile")).strip() or candidate["guidance_profile"]
        runtime_sources = _dedup_text(
            _normalize_text_list(metadata.get("access_control_runtime_sources"))
            + ["abi", "static_summary", "runtime_observed_context", "effect_probe"]
        )
        metadata["access_control_runtime_sources"] = runtime_sources
        candidate["metadata"] = metadata
        owner_admin_count = len(
            _dedup_text(
                list(binding_hints.get("owner_candidates") or [])
                + list(binding_hints.get("admin_candidates") or [])
                + list(binding_hints.get("pending_owner_candidates") or [])
            )
        )
        self.profile_effect_summary["caller_pool_size_before"] = int(len(self.accounts))
        self.profile_effect_summary["caller_pool_size_after"] = int(len(binding_hints.get("address_pool") or []))
        self.profile_effect_summary["observed_owner_admin_address_count"] = int(owner_admin_count)
        caller_pool_size = int(len(binding_hints.get("address_pool") or []))
        self.profile_effect_summary["generated_caller_permutation_candidates"] = max(
            int(self.profile_effect_summary.get("generated_caller_permutation_candidates", 0)),
            max(0, caller_pool_size - 1),
        )

    def _attach_asset_flow_sender_hints(
        self,
        candidate: Dict[str, Any],
        observed: Optional[Dict[str, Any]] = None,
        *,
        variant: str,
    ) -> None:
        if not self._asset_flow_sender_context_available(observed):
            return
        binding_hints = self._asset_flow_binding_hints(observed)
        candidate["targeted_patch_id"] = "TARGETED_ASSET_FLOW_SENDER_HINTS_PATCH"
        candidate["asset_flow_sender_hint_candidate"] = True
        candidate["asset_flow_sender_hint_variant"] = _text(variant).strip() or "asset_flow_sender_hint"
        candidate["sender_policy"] = _text(candidate.get("sender_policy")).strip() or "try_owner_or_deployer_or_previous_success_sender"
        candidate["sender_constraints"] = candidate["sender_policy"]
        candidate["argument_policy"] = _text(candidate.get("argument_policy")).strip() or "mutate_permission_and_sink_args"
        candidate["arg_constraints"] = candidate["argument_policy"]
        candidate["access_control_address_pool"] = list(binding_hints.get("address_pool") or [])
        candidate["access_control_binding_hints"] = dict(binding_hints)
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        metadata["targeted_patch_id"] = "TARGETED_ASSET_FLOW_SENDER_HINTS_PATCH"
        metadata["asset_flow_sender_hint_variant"] = candidate["asset_flow_sender_hint_variant"]
        metadata["asset_flow_runtime_sources"] = ["runtime_observed_context", "asset_flow_address_normalization"]
        candidate["metadata"] = metadata

    def _access_control_function_text(self, name: str) -> str:
        row_text: List[str] = [name]
        lowered = _text(name).strip().lower()
        for row in self._access_control_static_function_rows():
            if _text(row.get("name")).strip().lower() != lowered:
                continue
            row_text.extend(_normalize_text_list(row.get("modifiers")))
            row_text.extend(_normalize_text_list(row.get("reads")))
            row_text.extend(_normalize_text_list(row.get("writes")))
            break
        return " ".join(row_text).lower()

    def _is_access_control_initializer(self, name: str) -> bool:
        lowered = _text(name).strip().lower()
        return any(token in lowered for token in ("init", "initialize", "setup", "configure", "bootstrap"))

    def _access_control_role_writer_candidates(self, observed: Optional[Dict[str, Any]] = None) -> List[str]:
        observed = observed if isinstance(observed, dict) else {}
        static_summary = self._static_summary_hints()
        candidates: List[str] = []
        for name in _clean_list(static_summary.get("privileged_functions")):
            candidates.append(name)
        for item in list(static_summary.get("storage_dependencies") or []):
            if not isinstance(item, dict):
                continue
            name = _text(item.get("function")).strip()
            haystack = " ".join(
                [name]
                + _normalize_text_list(item.get("reads"))
                + _normalize_text_list(item.get("writes"))
            ).lower()
            if any(token in haystack for token in ("owner", "admin", "role", "minter", "operator", "manager", "govern", "auth", "allow", "permission", "blacklist", "whitelist", "pause")):
                candidates.append(name)
        for row in self._access_control_static_function_rows():
            name = _text(row.get("name")).strip()
            if not name:
                continue
            haystack = self._access_control_function_text(name)
            if any(token in haystack for token in ("onlyowner", "onlyadmin", "onlyrole", "owner", "admin", "role", "minter", "operator", "manager", "govern", "auth", "permission", "blacklist", "whitelist", "pause")):
                candidates.append(name)
        candidates.extend(
            self.abi_index.select_by_keywords(
                [
                    "init",
                    "initialize",
                    "setup",
                    "owner",
                    "admin",
                    "role",
                    "grant",
                    "revoke",
                    "set",
                    "approve",
                    "allow",
                    "minter",
                    "operator",
                    "manager",
                    "govern",
                    "auth",
                    "permission",
                    "blacklist",
                    "whitelist",
                    "pause",
                    "unpause",
                ],
                preferred=_clean_list(observed.get("observed_functions")),
            )
        )
        output: List[str] = []
        for name in _dedup_text(candidates):
            if not self.abi_index.has(name) or self.abi_index.is_view(name):
                continue
            output.append(self.abi_index.by_lower[name.lower()])
            if len(output) >= 12:
                break
        return output

    def _access_control_sensitive_suffix_candidates(self, observed: Optional[Dict[str, Any]] = None) -> List[str]:
        observed = observed if isinstance(observed, dict) else {}
        static_summary = self._static_summary_hints()
        candidates: List[str] = []
        for name in _clean_list(observed.get("observed_functions")):
            candidates.append(name)
        for item in list(static_summary.get("dangerous_operations") or []):
            if isinstance(item, dict) and _text(item.get("function")).strip():
                candidates.append(_text(item.get("function")).strip())
        for name in _clean_list(static_summary.get("privileged_functions")):
            candidates.append(name)
        candidates.extend(
            self.abi_index.select_by_keywords(
                [
                    "withdraw",
                    "mint",
                    "burn",
                    "transfer",
                    "transferfrom",
                    "upgrade",
                    "pause",
                    "unpause",
                    "blacklist",
                    "whitelist",
                    "claim",
                    "rescue",
                    "sweep",
                    "recover",
                    "drain",
                    "set",
                    "update",
                    "execute",
                ],
                preferred=_clean_list(observed.get("observed_functions")),
            )
        )
        output: List[str] = []
        for name in _dedup_text(candidates):
            if not self.abi_index.has(name) or self.abi_index.is_view(name):
                continue
            output.append(self.abi_index.by_lower[name.lower()])
            if len(output) >= 12:
                break
        non_initializer = [name for name in output if not self._is_access_control_initializer(name)]
        return non_initializer or output

    def _access_control_initializer_candidates(self) -> List[str]:
        output: List[str] = []
        for name in self.abi_index.select_by_keywords(["init", "initialize", "setup", "configure", "bootstrap"]):
            if self.abi_index.has(name) and not self.abi_index.is_view(name):
                output.append(name)
        return _dedup_text(output, limit=4)

    def _access_control_sender_policies(self, observed: Optional[Dict[str, Any]] = None) -> List[str]:
        observed = observed if isinstance(observed, dict) else {}
        policies = [
            "try_owner_or_deployer_or_previous_success_sender",
            "keep_same_sender_across_sequence",
        ]
        if len(_clean_list(observed.get("observed_senders"))) >= 2 or len(self.accounts) >= 2:
            policies.insert(1, "rotate_existing_senders")
        if self._access_control_state_push_enabled():
            policies = [
                "try_owner_or_deployer_or_previous_success_sender",
                "rotate_existing_senders",
                "keep_same_sender_across_sequence",
            ]
        return _dedup_text(policies, limit=3)

    def _access_control_state_sequences(
        self,
        observed: Dict[str, Any],
        mapping: Dict[str, Any],
        source_candidates: Sequence[Dict[str, Any]],
    ) -> List[List[str]]:
        del mapping
        role_writers = self._access_control_role_writer_candidates(observed)
        suffixes = self._access_control_sensitive_suffix_candidates(observed)
        initializers = self._access_control_initializer_candidates()
        observed_prefix = self._access_control_preserved_prefix(self._best_prefix_from_context(observed, source_candidates))
        sequences: List[List[str]] = []
        for writer in role_writers[:4]:
            for suffix in suffixes[:4]:
                if suffix == writer:
                    continue
                sequences.append([writer, suffix])
                if observed_prefix:
                    sequences.append(self._merge_prefix_suffix(observed_prefix + [writer], [suffix]))
                if initializers and initializers[0] != writer:
                    sequences.append([initializers[0], writer, suffix])
        if self._access_control_state_push_enabled():
            transfer_names = self.abi_index.select_by_keywords(["transferownership"])
            accept_names = self.abi_index.select_by_keywords(["acceptownership"])
            role_setup_names = self.abi_index.select_by_keywords(["grantrole", "addrole", "addminter", "setoperator", "setadmin"])
            allowlist_names = self.abi_index.select_by_keywords(["whitelist", "allowlist", "allow", "include"])
            pause_names = self.abi_index.select_by_keywords(["pause"])
            unpause_names = self.abi_index.select_by_keywords(["unpause"])
            for suffix in suffixes[:4]:
                if transfer_names:
                    if accept_names:
                        sequences.append([transfer_names[0], accept_names[0], suffix])
                    sequences.append([transfer_names[0], suffix])
                if role_setup_names:
                    sequences.append([role_setup_names[0], suffix])
                if allowlist_names:
                    sequences.append([allowlist_names[0], suffix])
                if pause_names:
                    sequences.append([pause_names[0], suffix])
                if unpause_names:
                    sequences.append([unpause_names[0], suffix])
        for suffix in suffixes[:4]:
            sequences.append([suffix])
        if self._access_control_state_push_enabled(observed):
            self.profile_effect_summary["generated_permission_setup_candidates"] = int(
                self.profile_effect_summary.get("generated_permission_setup_candidates", 0) + len(sequences)
            )
        return self._unique_sequences(sequences)

    def _access_control_preserved_prefix(self, prefix: Sequence[str]) -> List[str]:
        preserved: List[str] = []
        role_writer_set = set(self._access_control_role_writer_candidates({}))
        for name in _clean_list(prefix):
            if not self.abi_index.has(name) or self.abi_index.is_view(name):
                continue
            canonical = self.abi_index.by_lower[name.lower()]
            if canonical in role_writer_set or self._is_access_control_initializer(canonical):
                preserved.append(canonical)
            elif len(preserved) < 2:
                preserved.append(canonical)
            if len(preserved) >= 3:
                break
        return _dedup_text(preserved, limit=3)

    def _access_control_replay_prefix(self, source_prefix: Sequence[str], plan_required_prefix: Sequence[str]) -> List[str]:
        if self.runtime_pattern_family == "asset_flow_redirection":
            return self._asset_flow_replay_prefix(source_prefix, plan_required_prefix)
        if not self._access_control_profile_requested():
            return _clean_list(source_prefix) or _clean_list(plan_required_prefix)
        return _dedup_text(
            self._access_control_preserved_prefix(source_prefix)
            + _clean_list(plan_required_prefix),
            limit=4,
        )

    def _annotate_access_control_rescue_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        observed: Dict[str, Any],
        variant: str,
    ) -> None:
        if not self._access_control_profile_active_for_candidate(observed):
            return
        candidate["targeted_patch_id"] = "TARGETED_ACCESS_CONTROL_NONE34_RESCUE_PATCH"
        candidate["access_control_rescue_candidate"] = True
        candidate["access_control_rescue_variant"] = _text(variant).strip() or "generic_access_control_rescue"
        candidate["sender_policy"] = _text(candidate.get("sender_policy")).strip() or "try_owner_or_deployer_or_previous_success_sender"
        candidate["sender_constraints"] = candidate["sender_policy"]
        candidate["argument_policy"] = _text(candidate.get("argument_policy")).strip() or "mutate_permission_and_sink_args"
        candidate["arg_constraints"] = candidate["argument_policy"]
        candidate["access_control_role_writer_candidates"] = self._access_control_role_writer_candidates(observed)[:8]
        candidate["access_control_sensitive_suffix_candidates"] = self._access_control_sensitive_suffix_candidates(observed)[:8]
        candidate["access_control_sender_policy_variants"] = self._access_control_sender_policies(observed)
        candidate["expected_evidence"] = _dedup_text(
            list(candidate.get("expected_evidence") or [])
            + ["access_control_state_writer_prefix", "sender_context_variation", "privileged_suffix_effect"]
        )
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        metadata["targeted_patch_id"] = "TARGETED_ACCESS_CONTROL_NONE34_RESCUE_PATCH"
        metadata["access_control_runtime_sources"] = ["abi", "static_summary", "runtime_observed_context"]
        metadata["access_control_rescue_variant"] = candidate["access_control_rescue_variant"]
        candidate["metadata"] = metadata
        self._attach_access_control_profile_metadata(candidate, observed)
        candidate["novelty_signature"] = self._novelty_signature(
            _clean_list(candidate.get("guided_sequence") or candidate.get("function_sequence")),
            _text(candidate.get("target_missing_checkpoint")).strip(),
            {
                "sender_policy": candidate.get("sender_policy"),
                "value_policy": candidate.get("value_policy"),
                "argument_policy": candidate.get("argument_policy"),
            },
        )

    def _build_prestate_candidates(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        prestate_plans: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        built: List[Dict[str, Any]] = []
        mapping = ACTION_MAPPING.get(self.runtime_pattern_family, {}).get(next_missing, {"goal": "effect forcing prestate"})
        for plan in prestate_plans:
            sequence = _clean_list(plan.get("function_sequence"))
            if not sequence:
                continue
            candidate = self._new_candidate(
                source="tfi",
                next_missing=next_missing,
                mapping=mapping,
                sequence=sequence,
                observed=observed,
                build_reason="effect_forcing_prestate_builder",
            )
            self._mark_modules(candidate, ["prestate_builder"], "prestate_builder")
            candidate["required_prefix"] = _clean_list(plan.get("required_prefix"))
            candidate["suffix_actions"] = _clean_list(plan.get("suffix_actions"))
            candidate["prestate_plan_id"] = plan.get("prestate_plan_id")
            candidate["expected_effect"] = plan.get("expected_effect")
            if self._access_control_profile_active_for_candidate(observed):
                self._annotate_access_control_rescue_candidate(
                    candidate,
                    observed=observed,
                    variant="prestate_role_writer_prefix",
                )
            candidate = self._maybe_append_effect_probe(candidate, observed)
            candidate = self._finalize_candidate_materialization(candidate, observed)
            built.append(candidate)
        return built

    def _build_prefix_replay_candidates(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        prestate_plans: Sequence[Dict[str, Any]],
        tfi_candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        built: List[Dict[str, Any]] = []
        mapping = ACTION_MAPPING.get(self.runtime_pattern_family, {}).get(next_missing, {"goal": "effect forcing prefix replay"})
        if not prestate_plans and self.enable_prestate_builder:
            self.prefix_summary["missing_prestate_plan"] += 1
        source_prefix = self._best_prefix_from_context(observed, tfi_candidates)
        for plan in prestate_plans:
            plan_sequence = _clean_list(plan.get("function_sequence"))
            suffixes = self._suffix_mutations(next_missing, plan_sequence, observed)
            for suffix_sequence, mutation_type in suffixes[: max(1, self.max_per_round)]:
                replay_prefix = self._access_control_replay_prefix(source_prefix, _clean_list(plan.get("required_prefix")))
                sequence = self._merge_prefix_suffix(replay_prefix, suffix_sequence)
                if not sequence:
                    sequence = plan_sequence
                if not sequence:
                    continue
                candidate = self._new_candidate(
                    source="tfi",
                    next_missing=next_missing,
                    mapping=mapping,
                    sequence=sequence,
                    observed=observed,
                    build_reason="effect_forcing_prefix_replay_suffix_mutation",
                )
                self._mark_modules(candidate, ["prestate_builder", "prefix_replay", "suffix_mutation"], "prefix_replay_suffix_mutation")
                candidate["required_prefix"] = replay_prefix
                candidate["suffix_actions"] = list(suffix_sequence)
                candidate["source_prefix_id"] = plan.get("prestate_plan_id") or "observed_best_prefix"
                candidate["prefix_replay"] = True
                candidate["suffix_mutation_type"] = mutation_type
                candidate["prestate_plan_id"] = plan.get("prestate_plan_id")
                candidate["expected_effect"] = plan.get("expected_effect")
                if self._access_control_profile_active_for_candidate(observed):
                    candidate["access_control_prefix_preserving"] = bool(source_prefix)
                    self._annotate_access_control_rescue_candidate(
                        candidate,
                        observed=observed,
                        variant="prefix_preserving_role_writer_suffix",
                    )
                candidate = self._maybe_append_effect_probe(candidate, observed)
                candidate = self._finalize_candidate_materialization(candidate, observed)
                built.append(candidate)
                _append_jsonl(self.prefix_candidates_path, candidate)
                self.prefix_summary["prefix_replay_candidates"] += 1
                self.prefix_summary["suffix_mutation_candidates"] += 1
        return built

    def _suffix_mutations(self, next_missing: str, base_sequence: Sequence[str], observed: Dict[str, Any]) -> List[Tuple[List[str], str]]:
        sequence = _clean_list(base_sequence)
        mutations: List[Tuple[List[str], str]] = []
        if sequence:
            last = sequence[-1]
            if any(key in last.lower() for key in ("claim", "reward", "harvest", "withdraw")):
                mutations.append(([last, last], "repeat_last_claim_like_action"))
            if not (self._access_control_profile_active_for_candidate(observed) and self._is_access_control_initializer(last)):
                mutations.append(([last], "replay_target_suffix"))
        family = self.runtime_pattern_family
        if family == "asset_flow_redirection":
            for name in self._asset_flow_business_effect_candidates(observed):
                mutations.append(([name], "append_downstream_asset_effect_function"))
        elif family == "initialization_takeover":
            for name in self.abi_index.select_by_keywords(["set", "owner", "admin", "withdraw", "mint", "upgrade"]):
                mutations.append(([name], "append_privileged_effect_function"))
        elif family == "critical_parameter_abuse":
            for name in self.abi_index.select_by_keywords(["transfer", "swap", "claim", "withdraw", "deposit", "harvest", "collect", "sell"]):
                mutations.append(([name], "append_downstream_boundary_effect_function"))
        elif family == "access_control_state_change" or self._access_control_profile_active_for_candidate(observed):
            for name in self._access_control_sensitive_suffix_candidates(observed):
                mutations.append(([name], "append_dependent_privileged_effect"))
        elif family == "arithmetic_accounting_invariant":
            for name in self.abi_index.select_by_keywords(["update", "harvest", "withdraw", "claim", "pending", "reward", "balance"]):
                mutations.append(([name], "append_accounting_observation_or_effect"))
        elif family == "repeat_claim_no_reset":
            for name in self.abi_index.select_by_keywords(["claim", "reward", "harvest", "withdraw"]):
                mutations.append(([name, name], "repeat_claim_same_sender_beneficiary"))
        return self._unique_suffixes(mutations)

    def _sequence_variants(
        self,
        selected_functions: Sequence[str],
        mapping: Dict[str, Any],
        source_candidates: Sequence[Dict[str, Any]],
    ) -> List[List[str]]:
        sequences: List[List[str]] = []
        selected_functions = self._semantic_tx_actions(selected_functions, allow_owner_accept_pair=True)
        roles = mapping.get("sequence_roles") or []
        if len(roles) >= 2 and len(selected_functions) >= 2:
            sequences.append([selected_functions[0], selected_functions[-1]])
        if len(roles) >= 3 and len(selected_functions) >= 2:
            sequences.append([selected_functions[0], selected_functions[-1], selected_functions[-1]])
        if selected_functions:
            sequences.append([selected_functions[0]])
        for candidate in source_candidates:
            raw_sequence = _clean_list(candidate.get("guided_sequence")) or _clean_list(candidate.get("candidate_sequence"))
            sequence = self._semantic_tx_actions(raw_sequence, allow_owner_accept_pair=True)
            if sequence:
                anchor = selected_functions[-1] if selected_functions else ""
                if anchor and anchor not in sequence:
                    sequence = list(sequence) + [anchor]
                sequences.append(sequence)
        unique: List[List[str]] = []
        seen = set()
        for sequence in sequences:
            marker = tuple(sequence)
            if not sequence or marker in seen:
                continue
            seen.add(marker)
            unique.append(sequence)
        return unique

    def _new_candidate(
        self,
        *,
        source: str,
        next_missing: str,
        mapping: Dict[str, Any],
        sequence: Sequence[str],
        observed: Dict[str, Any],
        build_reason: str,
        raw_agent_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.last_candidate_id += 1
        raw_sequence = [_text(item).strip() for item in sequence if _text(item).strip()]
        clean_sequence = self._semantic_tx_actions(raw_sequence, allow_owner_accept_pair=True)
        sequence_sanitization_notes: List[str] = []
        if clean_sequence != raw_sequence:
            sequence_sanitization_notes.append("dropped_view_or_unpaired_owner_accept_from_checkpoint_sequence")
        source_task_id = "CKPT_{}_{}".format(_safe_token(self.runtime_pattern_family), _safe_token(next_missing))
        sender_policy = (
            _text(mapping.get("sender_policy")).strip()
            or _text(mapping.get("sender_constraints")).strip()
            or "any_existing_sender"
        )
        value_policy = (
            _text(mapping.get("value_policy")).strip()
            or _text(mapping.get("value_constraints")).strip()
            or "keep_original"
        )
        argument_policy = (
            _text(mapping.get("argument_policy")).strip()
            or _text(mapping.get("arg_constraints")).strip()
            or "abi_default_or_boundary_mix"
        )
        candidate = {
            "candidate_id": "CKPT_GUIDED_{:04d}_{}_{}".format(self.last_candidate_id, _safe_token(self.runtime_pattern_family), _safe_token(next_missing)),
            "candidate_type": "checkpoint_guided_candidate_v2",
            "candidate_origin": "checkpoint_guided_injection_v2",
            "group_name": "",
            "candidate_source": source,
            "candidate_source_used": "checkpoint_guided_{}".format(source),
            "source_task_id": source_task_id,
            "scope_id": source_task_id,
            "hypothesis_id": source_task_id,
            "source": source,
            "target_runtime_pattern_family": self.runtime_pattern_family,
            "runtime_pattern_family": self.runtime_pattern_family,
            "target_missing_checkpoint": next_missing,
            "candidate_goal": _text(mapping.get("goal")).strip(),
            "sequence_skeleton": list(clean_sequence),
            "function_sequence": list(clean_sequence),
            "candidate_sequence": list(clean_sequence),
            "guided_sequence": list(clean_sequence),
            "target_function": clean_sequence[-1] if clean_sequence else self._runtime_contract_label(),
            "frontier": clean_sequence[0] if clean_sequence else self._runtime_contract_label(),
            "sink": clean_sequence[-1] if clean_sequence else self._runtime_contract_label(),
            "sender_constraints": sender_policy,
            "value_constraints": value_policy,
            "arg_constraints": argument_policy,
            "sender_policy": sender_policy,
            "value_policy": value_policy,
            "argument_policy": argument_policy,
            "prestate_requirements": mapping.get("prestate_requirements") or "none",
            "expected_evidence": list(mapping.get("expected_evidence") or []),
            "binding_choices": {
                "sender_policy_used": sender_policy,
                "value_policy_used": value_policy,
                "args_policy_used": argument_policy,
                "prestate_policy_used": mapping.get("prestate_policy") or "none",
            },
            "candidate_constraints": {
                "must_target_missing_checkpoint": True,
                "must_use_existing_abi_function": True,
                "must_not_bypass_materializer": True,
                "do_not_bypass_normal_selection": True,
            },
            "novelty_signature": self._novelty_signature(clean_sequence, next_missing, mapping),
            "confidence": 0.75 if source != "agent" else 0.65,
            "build_reason": build_reason,
            "materialization_status": "pending",
            "sequence_sanitization_notes": sequence_sanitization_notes,
            "uses_llm_hypothesis": bool(source == "agent"),
            "parameter_plan": (
                raw_agent_payload.get("parameter_plan")
                if isinstance(raw_agent_payload, dict) and isinstance(raw_agent_payload.get("parameter_plan"), list)
                else []
            ),
            "is_executable_candidate": bool(clean_sequence),
            "useful": True,
            "useful_reason": "checkpoint_targeted_candidate",
            "path_progress_delta": 1,
            "frontier_touched": True,
            "sink_touched": bool(len(clean_sequence) > 1),
            "generated_under_tfi_constraint": True,
            "priority": "checkpoint",
            "do_not_bypass_normal_selection": True,
            "metadata": {
                "checkpoint_guided": True,
                "target_missing_checkpoint": next_missing,
                "runtime_pattern_family": self.runtime_pattern_family,
                "runtime_pattern_family_provenance": self.runtime_pattern_family_provenance,
                "source": source,
                "observed_context_used": True,
                "raw_agent_payload": raw_agent_payload if isinstance(raw_agent_payload, dict) else None,
            },
            "modules_used": [],
            "module_source": "checkpoint_guided_v2",
            "before_level": None,
            "after_level": None,
            "before_vspc": None,
            "after_vspc": None,
            "checkpoint_delta": 0,
            "oracle_level_delta": 0,
            "useful_strength": "weak",
            "l3_confirmed_by_candidate": False,
        }
        self._attach_access_control_profile_metadata(candidate, observed)
        if self.runtime_pattern_family == "asset_flow_redirection":
            self._attach_asset_flow_sender_hints(
                candidate,
                observed=observed,
                variant="candidate_{}".format(_safe_token(build_reason)),
            )
        return candidate

    def _finalize_candidate_materialization(self, candidate: Dict[str, Any], observed: Dict[str, Any]) -> Dict[str, Any]:
        self._attach_access_control_profile_metadata(candidate, observed)
        if (
            self.runtime_pattern_family == "asset_flow_redirection"
            and not isinstance(candidate.get("access_control_binding_hints"), dict)
        ):
            self._attach_asset_flow_sender_hints(
                candidate,
                observed=observed,
                variant="materialization_fallback",
            )
        if not self.enable_parameter_materializer:
            sequence = _clean_list(candidate.get("guided_sequence")) or _clean_list(candidate.get("candidate_sequence")) or _clean_list(candidate.get("function_sequence"))
            required_prefix = _clean_list(candidate.get("required_prefix"))
            suffix_actions = _clean_list(candidate.get("suffix_actions"))
            if not suffix_actions and sequence and not required_prefix:
                suffix_actions = list(sequence)
            candidate["has_concrete_prefix"] = bool(required_prefix)
            candidate["has_concrete_suffix"] = bool(suffix_actions)
            candidate["sequence_materialized"] = bool(sequence)
            candidate["materialized_tx_count"] = int(len(sequence))
            candidate["prefix_action_count"] = int(len(required_prefix))
            candidate["suffix_action_count"] = int(len(suffix_actions))
            candidate["effect_probe_appended"] = bool(candidate.get("probe_appended")) and bool(_clean_list(candidate.get("effect_probe_plan")))
            candidate["materialization_status_final"] = _text(candidate.get("materialization_status")).strip() or "pending"
            candidate["materialization_reason"] = _text(candidate.get("materialization_reason")).strip() or candidate["materialization_status_final"]
            return candidate
        return self.materializer.finalize_candidate(
            candidate,
            observed=observed,
            allow_llm_hypothesis=True,
        )

    def _best_prefix_from_context(self, observed: Dict[str, Any], tfi_candidates: Sequence[Dict[str, Any]]) -> List[str]:
        best = _clean_list(self.progress.best_prefix_sequence)
        if best:
            return best
        for candidate in tfi_candidates:
            if not isinstance(candidate, dict):
                continue
            sequence = _clean_list(candidate.get("guided_sequence")) or _clean_list(candidate.get("candidate_sequence"))
            if sequence:
                return sequence[:3]
        return _clean_list(observed.get("observed_functions"))[-3:]

    def _runtime_goal_for_missing_checkpoint(self, next_missing: str, observed: Dict[str, Any]) -> Dict[str, Any]:
        del next_missing
        static_summary = self._static_summary_hints()
        observed_functions = _clean_list(observed.get("observed_functions"))
        sender_count = len(_clean_list(observed.get("observed_senders")))
        value_count = len(_clean_list(observed.get("observed_values")))
        goal_kinds: List[str] = []
        if static_summary.get("storage_dependencies"):
            goal_kinds.append("prefer_state_writer_before_frontier")
        if static_summary.get("privileged_functions") or sender_count >= 2:
            goal_kinds.append("prefer_sender_context_variation")
        if static_summary.get("dangerous_operations"):
            goal_kinds.append("prefer_dependency_local_reordering")
        if value_count > 0:
            goal_kinds.append("prefer_parameter_binding_repair")
        if not goal_kinds:
            goal_kinds.append("diagnostic_probe_only")
        return {
            "goal_kinds": _dedup_text(goal_kinds),
            "observed_functions": observed_functions[-8:],
            "static_privileged_functions": list(static_summary.get("privileged_functions") or [])[:8],
            "static_sink_functions": [
                _text(item.get("function")).strip()
                for item in list(static_summary.get("dangerous_operations") or [])[:8]
                if isinstance(item, dict) and _text(item.get("function")).strip()
            ],
            "state_writer_functions": [
                _text(item.get("function")).strip()
                for item in list(static_summary.get("storage_dependencies") or [])[:8]
                if isinstance(item, dict) and _text(item.get("function")).strip()
            ],
        }

    def _merge_prefix_suffix(self, prefix: Sequence[str], suffix: Sequence[str]) -> List[str]:
        return (_clean_list(prefix) + _clean_list(suffix))[:6]

    def _is_observation_only_action(self, name: str) -> bool:
        lowered = _text(name).strip().lower()
        if not lowered:
            return True
        if lowered == "native_balance":
            return True
        if lowered in {
            "allowance",
            "balanceof",
            "decimals",
            "name",
            "symbol",
            "owner",
            "pendingowner",
            "newowner",
            "admin",
            "paused",
            "receiver",
            "treasury",
            "feereceiver",
            "totalsupply",
            "totalshares",
        }:
            return True
        return bool(self.abi_index.is_view(name))

    @staticmethod
    def _is_unpaired_owner_accept(action: str, prior_actions: Sequence[str]) -> bool:
        lowered = _text(action).strip().lower()
        if lowered not in {"acceptownership", "claimownership"}:
            return False
        prior_text = " ".join(_text(item).strip().lower() for item in prior_actions)
        return not any(token in prior_text for token in ("transferownership", "nominateowner", "setpendingowner"))

    def _semantic_tx_actions(self, actions: Sequence[str], *, allow_owner_accept_pair: bool = True) -> List[str]:
        cleaned: List[str] = []
        for action in _normalize_text_list(actions):
            if not self.abi_index.has(action):
                continue
            canonical = self.abi_index.by_lower.get(action.lower(), action)
            if self._is_observation_only_action(canonical):
                continue
            if allow_owner_accept_pair and self._is_unpaired_owner_accept(canonical, cleaned):
                continue
            cleaned.append(canonical)
        return cleaned

    def _sanitize_failure_repair_sequence(
        self,
        *,
        required_prefix: Sequence[str],
        suffix_actions: Sequence[str],
        parent_candidate: Dict[str, Any],
        target_missing: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        notes: List[str] = []
        original_prefix = _normalize_text_list(required_prefix)
        original_suffix = _normalize_text_list(suffix_actions)
        prefix = self._semantic_tx_actions(original_prefix, allow_owner_accept_pair=True)
        suffix = self._semantic_tx_actions(original_suffix, allow_owner_accept_pair=True)
        if prefix != original_prefix:
            notes.append("dropped_view_or_unpaired_owner_accept_from_prefix")
        if suffix != original_suffix:
            notes.append("dropped_view_or_unpaired_owner_accept_from_suffix")
        if (
            any(_text(action).strip().lower() == "transferfrom" for action in suffix)
            and self.abi_index.has("approve")
            and "approve" not in {item.lower() for item in prefix + suffix}
        ):
            prefix = _dedup_text(prefix + [self.abi_index.by_lower.get("approve", "approve")])
            notes.append("added_approve_prefix_for_transferFrom")
        if not suffix:
            fallback_keywords: List[str] = []
            lowered_target = _text(target_missing).strip().lower()
            if any(token in lowered_target for token in ("balance", "transfer", "funds", "asset")):
                fallback_keywords = ["transferFrom", "transfer", "withdraw", "sell", "burn", "claim", "redeem"]
            elif any(token in lowered_target for token in ("privileged", "state", "owner", "admin")):
                fallback_keywords = ["transferFrom", "transfer", "withdraw", "mint", "set", "grant", "update"]
            if fallback_keywords:
                suffix = self._semantic_tx_actions(
                    self.abi_index.select_by_keywords(fallback_keywords),
                    allow_owner_accept_pair=True,
                )[:2]
                if suffix:
                    notes.append("filled_empty_suffix_from_semantic_abi_keywords")
        if not suffix:
            parent_suffix = self._semantic_tx_actions(parent_candidate.get("suffix_actions") or [], allow_owner_accept_pair=True)
            parent_sequence_suffix = self._semantic_tx_actions(_normalize_text_list(parent_candidate.get("guided_sequence"))[-2:], allow_owner_accept_pair=True)
            suffix = _dedup_text(parent_suffix + parent_sequence_suffix, limit=2)
            if suffix:
                notes.append("filled_empty_suffix_from_parent_semantic_tail")
        return prefix[:3], suffix[:3], _dedup_text(notes)

    def _unique_sequences(self, sequences: Sequence[Sequence[str]]) -> List[List[str]]:
        unique: List[List[str]] = []
        seen = set()
        for sequence in sequences:
            cleaned = _clean_list(sequence)
            if not cleaned:
                continue
            marker = tuple(cleaned)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(cleaned)
        return unique

    def _unique_suffixes(self, suffixes: Sequence[Tuple[Sequence[str], str]]) -> List[Tuple[List[str], str]]:
        unique: List[Tuple[List[str], str]] = []
        seen = set()
        for sequence, mutation_type in suffixes:
            cleaned = _clean_list(sequence)
            if not cleaned:
                continue
            marker = (tuple(cleaned), _text(mutation_type).strip())
            if marker in seen:
                continue
            seen.add(marker)
            unique.append((cleaned, _text(mutation_type).strip() or "unknown_suffix_mutation"))
        return unique

    def _sender_binding_for_family(self) -> str:
        if self.runtime_pattern_family in {"repeat_claim_no_reset", "initialization_takeover"}:
            return "keep_same_sender_or_attacker"
        if self._access_control_profile_requested():
            return "try_owner_or_deployer_or_previous_success_sender"
        if self.runtime_pattern_family == "asset_flow_redirection":
            return "attacker_or_previous_success_sender"
        return "previous_success_sender_or_any_existing_sender"

    def _beneficiary_binding_for_family(self) -> str:
        if self.runtime_pattern_family == "repeat_claim_no_reset":
            return "same_beneficiary_as_first_claim"
        if self.runtime_pattern_family == "asset_flow_redirection":
            return "attacker_or_changed_receiver"
        return "observed_sender_or_attacker"

    def _amount_binding_for_family(self, next_missing: str) -> str:
        if self.runtime_pattern_family in {"critical_parameter_abuse", "arithmetic_accounting_invariant"}:
            return "boundary_0_1_observed_plus_minus_1"
        if self.runtime_pattern_family in {"repeat_claim_no_reset", "asset_flow_redirection"}:
            return "observed_balance_or_amount_plus_minus_1"
        if "balance" in next_missing or "transfer" in next_missing:
            return "small_nonzero_or_observed_amount"
        return "zero_or_small_nonzero"

    def _expected_effect_for_checkpoint(self, next_missing: str, mapping: Dict[str, Any]) -> str:
        evidence = _clean_list(mapping.get("expected_evidence"))
        if evidence:
            return ",".join(evidence)
        if any(token in next_missing for token in ("balance", "transfer", "funds")):
            return "balance_change_or_transfer"
        if any(token in next_missing for token in ("state", "owner", "admin", "privileged")):
            return "state_write_or_privileged_effect"
        return "checkpoint_delta_or_l3_effect"

    def _mark_modules(self, candidate: Dict[str, Any], modules: Sequence[str], module_source: str) -> None:
        candidate["modules_used"] = _dedup_text(_clean_list(candidate.get("modules_used")) + list(modules))
        module_source_existing = candidate.get("module_source")
        if isinstance(module_source_existing, list):
            candidate["module_source"] = _dedup_text(module_source_existing + [module_source])
        else:
            source_list = _dedup_text([module_source_existing, module_source])
            candidate["module_source"] = source_list[0] if len(source_list) == 1 else source_list
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        metadata["effect_forcing_v3"] = bool(self.enable_effect_forcing_v3)
        metadata["modules_used"] = list(candidate.get("modules_used") or [])
        candidate["metadata"] = metadata

    def _probeable_view_functions(self) -> List[str]:
        names: List[str] = []
        for name, meta in sorted(self.abi_index.functions.items()):
            inputs = meta.get("inputs") if isinstance(meta, dict) else []
            lowered = name.lower()
            known_view_name = lowered in {
                "balanceof",
                "allowance",
                "owner",
                "admin",
                "pendingowner",
                "newowner",
                "paused",
                "treasury",
                "feereceiver",
                "receiver",
                "totalsupply",
                "decimals",
                "symbol",
                "name",
            }
            if not known_view_name and not self.abi_index.is_view(name):
                continue
            if known_view_name:
                names.append(name)
            elif any(key in lowered for key in ("owner", "admin", "pendingowner", "paused", "treasury", "receiver", "fee", "claimed", "reward", "userinfo", "pending", "balance", "supply", "allowlist", "whitelist", "role")):
                names.append(name)
            elif len(inputs or []) <= 1 and any(key in lowered for key in ("claim", "reward", "debt", "allowance")):
                names.append(name)
        return _dedup_text(names, limit=12)

    def _probe_targets(self, observed: Dict[str, Any]) -> List[str]:
        targets: List[str] = []
        targets.extend(self.accounts[:4])
        targets.extend(observed.get("observed_senders") or [])
        targets.extend(observed.get("observed_receivers") or [])
        return _dedup_text(targets, limit=8)

    def _build_effect_probe_plan(self, observed: Dict[str, Any], probe_plan: Any = None) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for item in _normalize_probe_plan(probe_plan):
            function_name = _canonical_action_name(item.get("function"))
            if not function_name:
                continue
            probe_type = _text(item.get("probe_type")).strip() or "agent_view_hint"
            targets = [target for target in _normalize_text_list(item.get("targets")) if target]
            output.append(
                {
                    "probe_type": probe_type,
                    "function": function_name,
                    "targets": targets or self._probe_targets(observed),
                }
            )
        for name in self._probeable_view_functions()[:4]:
            lowered = name.lower()
            if lowered in {"balanceof", "allowance"}:
                output.append({"probe_type": "erc20_{}".format(lowered), "function": name, "targets": self._probe_targets(observed)})
            else:
                output.append({"probe_type": "state_view", "function": name, "targets": self._probe_targets(observed)})
        output.append({"probe_type": "native_balance_snapshot", "function": "native_balance", "targets": self._probe_targets(observed)})
        unique: List[Dict[str, Any]] = []
        seen = set()
        for probe in output:
            marker = json.dumps(probe, ensure_ascii=True, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(probe)
            if len(unique) >= 6:
                break
        return unique

    def _maybe_append_effect_probe(
        self,
        candidate: Dict[str, Any],
        observed: Dict[str, Any],
        probe_plan: Any = None,
    ) -> Dict[str, Any]:
        if not self.enable_effect_probes:
            return candidate
        probes = self._build_effect_probe_plan(observed, probe_plan)
        if not probes:
            self.effect_probe_summary["no_probe_available"] += 1
            return candidate
        self._mark_modules(candidate, ["effect_probe"], "effect_probe")
        candidate["effect_probe_plan"] = probes
        candidate["probe_appended"] = True
        candidate["expected_evidence"] = _dedup_text(list(candidate.get("expected_evidence") or []) + ["effect_probe"])
        for probe in probes:
            _append_jsonl(
                self.effect_probe_events_path,
                {
                    "stage": "plan",
                    "candidate_id": candidate.get("candidate_id"),
                    "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
                    "probe": probe,
                    "probe_success": None,
                },
            )
        self.effect_probe_summary["effect_probe_count"] += len(probes)
        self.effect_probe_summary["candidates_with_probe"] += 1
        return candidate

    def _agent_effect_repair_prompt(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        tfi_candidates: Sequence[Dict[str, Any]],
        recent_failed: Sequence[Dict[str, Any]],
        prestate_plans: Sequence[Dict[str, Any]],
    ) -> str:
        return json.dumps(
            {
                "task": "effect_forcing_repair_candidate",
                "current_best_level": self.progress.best_level,
                "current_best_prefix": self._best_prefix_from_context(observed, tfi_candidates),
                "covered_checkpoints": sorted(self.progress.covered),
                "next_missing_checkpoint": next_missing,
                "runtime_pattern_family": self.runtime_pattern_family,
                "runtime_pattern_family_provenance": self.runtime_pattern_family_provenance,
                "prestate_plans": list(prestate_plans)[:4],
                "recent_failed_candidates": list(recent_failed)[-8:],
                "recent_no_delta_candidates": list(self.recent_failed_checkpoint_candidates)[-8:],
                "abi_function_list": self.abi_index.names(),
                "probeable_view_functions": self._probeable_view_functions(),
                "available_sender_value_args_hints": observed,
                "allowed_binding_sources": [
                    "existing_pool",
                    "observed_trace",
                    "static_context",
                    "boundary_domain",
                    "llm_hypothesis",
                ],
                "constraints": {
                    "must_target_missing_checkpoint": True,
                    "must_use_existing_abi_function": True,
                    "reuse_prefix_when_it_preserves_L1_or_L2": True,
                    "no_generic_advice": True,
                    "materializer_will_validate_before_execution": True,
                },
                "output_schema": {
                    "target_missing_checkpoint": next_missing,
                    "reuse_prefix": True,
                    "required_prefix": ["functionName"],
                    "suffix_actions": ["functionName"],
                    "sender_binding": "string",
                    "arg_binding": "string",
                    "effect_probe_plan": ["viewFunctionName(address)"],
                    "parameter_plan": [
                        {
                            "function": "string",
                            "arg": "string",
                            "source_type": "existing_pool|observed_trace|static_context|boundary_domain|llm_hypothesis",
                            "source_hint": "string",
                        }
                    ],
                    "repair_reason": "string",
                    "confidence": 0.0,
                    "no_candidate": False,
                    "reason": "string if no_candidate",
                },
            },
            ensure_ascii=True,
        )

    def _failure_repair_system_prompt(self) -> str:
        return (
            "You are a smart-contract fuzzing failure-repair agent.\n"
            "You must repair one failed or no-delta parent candidate.\n"
            "You are not generating a new unrestricted path; you are repairing the parent inside a grounded local scope.\n"
            "Do not invent functions outside ABI.\n"
            "Do not ignore the failure_signature, state obligation, or dependency-grounded scope.\n"
            "Only modify allowed repair fields.\n"
            "You must choose executable actions only from dependency_grounded_scope candidate actions.\n"
            "If the obligation requires state progress, a probe-only repair is not a semantic repair.\n"
            "If no safe grounded semantic repair exists, return no_repair=true.\n"
            "Return only valid JSON.\n"
            "Every output must be materializable by ABI, observed trace values, known accounts, or boundary domains.\n"
            "If no safe repair exists, return {\"no_repair\": true, \"reason\": \"...\"}."
        )

    def _failure_signature_for_candidate(
        self,
        candidate: Dict[str, Any],
        related_events: Sequence[Dict[str, Any]],
        *,
        before_level: str,
        after_level: str,
        before_strict_vspc: float,
        after_strict_vspc: float,
    ) -> Dict[str, Any]:
        event_types = Counter(
            _text(event.get("event_type")).strip()
            for event in related_events
            if isinstance(event, dict)
        )
        reverted = any(
            bool(event.get("reverted")) or _text(event.get("event_type")).strip() == "revert"
            for event in related_events
            if isinstance(event, dict)
        )
        functions_reached = _dedup_text(
            [_text(event.get("function")).strip() for event in related_events if isinstance(event, dict)]
        )
        target_function = _text(candidate.get("target_function")).strip()
        target_suffix_reached = bool(target_function and target_function in functions_reached)
        state_delta_observed = bool(event_types.get("sstore", 0) > 0)
        balance_delta_observed = bool(event_types.get("balance_change", 0) > 0 or event_types.get("native_transfer", 0) > 0)
        token_transfer_observed = bool(event_types.get("token_transfer", 0) > 0)
        external_call_result_observed = bool(event_types.get("external_call", 0) > 0)
        strict_delta = round(float(after_strict_vspc) - float(before_strict_vspc), 6)
        checkpoint_advanced = bool(candidate.get("target_missing_checkpoint") in self.progress.covered)
        probe_plan = candidate.get("effect_probe_plan") if isinstance(candidate.get("effect_probe_plan"), list) else []
        sender_policy = _text(candidate.get("sender_policy") or candidate.get("sender_constraints")).strip().lower()
        amount_policy = _text(candidate.get("value_policy") or candidate.get("value_constraints")).strip().lower()
        arg_policy = _text(candidate.get("argument_policy") or candidate.get("arg_constraints")).strip().lower()
        suspected_wrong_fields: List[str] = []
        failure_signature = "unknown"
        confidence = "low"
        evidence: List[str] = []
        access_control_reason_triggered = self._access_control_revert_reason_triggered(related_events)
        if self._access_control_profile_requested() and reverted and access_control_reason_triggered:
            self.profile_effect_summary["unauthorized_revert_seen_count"] = int(
                self.profile_effect_summary.get("unauthorized_revert_seen_count", 0) + 1
            )
            failure_signature = "wrong_sender_binding"
            confidence = "high"
            suspected_wrong_fields.extend(["sender", "prestate"])
            evidence.append("access_control_revert_reason_detected")
        elif reverted and "owner" in sender_policy:
            failure_signature = "wrong_sender_binding"
            confidence = "medium"
            suspected_wrong_fields.append("sender")
            evidence.append("revert_with_owner_or_deployer_sender_policy")
        elif reverted and ("same_beneficiary" in arg_policy or "beneficiary" in arg_policy):
            failure_signature = "wrong_beneficiary_binding"
            confidence = "medium"
            suspected_wrong_fields.append("beneficiary")
            evidence.append("revert_with_beneficiary_binding_policy")
        elif reverted and ("receiver" in arg_policy or "receiver" in _text(candidate.get("target_missing_checkpoint")).lower()):
            failure_signature = "wrong_receiver_binding"
            confidence = "medium"
            suspected_wrong_fields.append("receiver")
            evidence.append("revert_with_receiver_binding_policy")
        elif reverted and ("boundary" in amount_policy or "nonzero" in amount_policy or "small" in amount_policy):
            failure_signature = "wrong_amount_binding"
            confidence = "medium"
            suspected_wrong_fields.append("amount")
            evidence.append("revert_with_boundary_or_nonzero_amount_policy")
        elif not target_suffix_reached:
            failure_signature = "effect_function_not_reached"
            confidence = "high"
            suspected_wrong_fields.append("suffix")
            evidence.append("target_function_not_seen_in_events")
        elif target_suffix_reached and not (state_delta_observed or balance_delta_observed or token_transfer_observed or external_call_result_observed):
            failure_signature = "effect_reached_but_no_state_delta"
            confidence = "high"
            suspected_wrong_fields.append("suffix")
            evidence.append("suffix_reached_but_no_state_or_balance_delta")
        elif strict_delta <= 0 and not checkpoint_advanced and probe_plan and not event_types.get("effect_probe", 0):
            failure_signature = "effect_probe_missing_or_uninformative"
            confidence = "medium"
            suspected_wrong_fields.append("probe")
            evidence.append("probe_planned_but_no_effect_probe_event")
        elif strict_delta <= 0 and not checkpoint_advanced and _clean_list(candidate.get("required_prefix")):
            failure_signature = "missing_prestate_prefix"
            confidence = "medium"
            suspected_wrong_fields.append("prestate")
            evidence.append("required_prefix_present_but_no_checkpoint_or_vspc_delta")
        elif strict_delta <= 0 and not checkpoint_advanced:
            failure_signature = "suffix_not_effective"
            confidence = "medium"
            suspected_wrong_fields.append("suffix")
            evidence.append("no_checkpoint_or_vspc_delta_after_execution")
        final_status = _text(candidate.get("materialization_status_final") or candidate.get("materialization_status")).strip()
        if final_status.startswith("rejected_"):
            failure_signature = "materialization_too_weak"
            confidence = "high"
            suspected_wrong_fields = ["prestate", "amount", "sender"]
            evidence.append(final_status)
        if not evidence and reverted:
            evidence.append("revert_observed")
        if not evidence:
            evidence.append("no_immediate_progress_delta")
        allowed_repair_actions = self._allowed_repair_actions_for_signature(failure_signature)
        payload = {
            "candidate_id": candidate.get("candidate_id"),
            "bench_contract_id": self._runtime_bench_label(),
            "group_name": _text(candidate.get("group_name")).strip(),
            "runtime_pattern_family": self.runtime_pattern_family,
            "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
            "parent_sequence": list(candidate.get("guided_sequence") or []),
            "execution_status": "revert" if reverted else ("success" if related_events else "unknown"),
            "tx_success": not reverted and bool(related_events),
            "reverted": bool(reverted),
            "failure_signature": failure_signature,
            "confidence": confidence,
            "evidence": evidence,
            "suspected_wrong_fields": suspected_wrong_fields,
            "allowed_repair_actions": allowed_repair_actions,
            "functions_reached": functions_reached,
            "target_suffix_reached": target_suffix_reached,
            "state_delta_observed": state_delta_observed,
            "balance_delta_observed": balance_delta_observed,
            "token_transfer_observed": token_transfer_observed,
            "external_call_result_observed": external_call_result_observed,
            "probe_before_present": bool(candidate.get("probe_before")),
            "probe_after_present": bool(candidate.get("probe_after")),
            "probe_delta_present": bool(candidate.get("probe_delta")),
            "access_control_revert_reason_triggered": bool(access_control_reason_triggered),
            "modules_used": list(candidate.get("modules_used") or []),
            "evidence_summary": ",".join(evidence[:4]),
            "before_best_level": before_level,
            "after_best_level": after_level,
            "before_strict_vspc": round(before_strict_vspc, 6),
            "after_strict_vspc": round(after_strict_vspc, 6),
        }
        return payload

    def _allowed_repair_actions_for_signature(self, failure_signature: str) -> List[str]:
        if failure_signature == "wrong_sender_binding":
            return ["fix_sender_binding", "reuse_same_sender", "add_missing_prestate_prefix"]
        if failure_signature in {"wrong_beneficiary_binding", "wrong_receiver_binding"}:
            return ["fix_beneficiary_binding", "fix_receiver_binding", "reuse_same_beneficiary"]
        if failure_signature == "wrong_amount_binding":
            return ["fix_amount_binding", "use_observed_balance_or_allowance", "use_boundary_amount"]
        if failure_signature == "missing_prestate_prefix":
            return ["add_missing_prestate_prefix", "reuse_parent_prefix", "fix_sender_binding"]
        if failure_signature in {"suffix_not_effective", "effect_function_not_reached", "effect_reached_but_no_state_delta"}:
            return ["strengthen_suffix", "fix_sender_binding", "fix_amount_binding", "change_effect_probe"]
        if failure_signature == "effect_probe_missing_or_uninformative":
            return ["change_effect_probe", "fix_receiver_binding", "fix_beneficiary_binding"]
        if failure_signature == "materialization_too_weak":
            return ["fix_sender_binding", "fix_amount_binding", "add_missing_prestate_prefix"]
        return []

    def _should_trigger_failure_repair(self, signature: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        candidate_id = _text(candidate.get("candidate_id")).strip()
        failure_signature = _text(signature.get("failure_signature")).strip()
        parent_sequence = tuple(_clean_list(candidate.get("guided_sequence")))
        triggerable = {
            "wrong_sender_binding",
            "wrong_beneficiary_binding",
            "wrong_receiver_binding",
            "wrong_amount_binding",
            "missing_prestate_prefix",
            "suffix_not_effective",
            "effect_reached_but_no_state_delta",
            "effect_probe_missing_or_uninformative",
            "materialization_too_weak",
            "effect_function_not_reached",
        }
        skipped_reason = ""
        should_trigger = False
        self._bump_pipeline_counter("repair_trigger_evaluated", 1)
        if self.progress.best_level == "L3":
            skipped_reason = "no_headroom_already_l3"
        elif failure_signature not in triggerable:
            skipped_reason = "failure_signature_not_triggerable"
        elif parent_sequence and _json_compact(parent_sequence) in self.repair_cooldown_signatures:
            skipped_reason = "parent_sequence_in_cooldown"
        elif self.repair_attempts_by_parent[candidate_id] >= self.failure_repair_max_retries:
            skipped_reason = "repair_retry_budget_exhausted"
        elif not self.enable_failure_repair_pipeline:
            skipped_reason = "failure_repair_pipeline_disabled"
        else:
            should_trigger = True
        payload = {
            "candidate_id": candidate_id,
            "failure_signature": failure_signature,
            "repairable": bool(failure_signature in triggerable),
            "trigger_rule": failure_signature if failure_signature in triggerable else "",
            "trigger_agent_repair": should_trigger,
            "trigger_rule_repair": bool(should_trigger and not self.failure_repair_use_agent),
            "trigger_reason": failure_signature if should_trigger else "",
            "skipped_reason": skipped_reason,
            "retry_count_for_parent": int(self.repair_attempts_by_parent[candidate_id] or 0),
            "skeleton_cooldown_hit": bool(parent_sequence and _json_compact(parent_sequence) in self.repair_cooldown_signatures),
        }
        if should_trigger:
            self.repair_scheduling_summary["repair_triggered"] += 1
        else:
            self.repair_scheduling_summary["repair_skipped"] += 1
        _append_jsonl(self.failure_repair_trigger_log_path, payload)
        self.failure_repair_trigger_summary["triggered" if should_trigger else "skipped"] += 1
        if skipped_reason:
            self.failure_repair_trigger_summary["skip:{}".format(skipped_reason)] += 1
            self._bump_pipeline_counter("repair_trigger_false", 1)
        else:
            self._bump_pipeline_counter("repair_trigger_true", 1)
        if failure_signature in triggerable:
            self._bump_pipeline_counter("repairable_failure_signature_count", 1)
        return payload

    def _build_failure_repair_prompt(
        self,
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
        observed: Dict[str, Any],
        *,
        state_obligation: Optional[Dict[str, Any]] = None,
        repair_scope: Optional[Dict[str, Any]] = None,
    ) -> str:
        functions_reached = list(signature.get("functions_reached") or [])
        state_obligation = state_obligation if isinstance(state_obligation, dict) and state_obligation else self._build_state_obligation(parent_candidate, signature, observed)
        repair_scope = repair_scope if isinstance(repair_scope, dict) and repair_scope else self._build_dependency_grounded_scope(parent_candidate, signature, observed, state_obligation)
        return json.dumps(
            {
                "task": "failure_aware_repair",
                "bench_contract_id": self._runtime_bench_label(),
                "contract_name": self._runtime_contract_label(),
                "runtime_pattern_family": self.runtime_pattern_family,
                "runtime_pattern_family_provenance": self.runtime_pattern_family_provenance,
                "parent_candidate": {
                    "candidate_id": parent_candidate.get("candidate_id"),
                    "source": parent_candidate.get("source"),
                    "modules_used": list(parent_candidate.get("modules_used") or []),
                    "target_missing_checkpoint": parent_candidate.get("target_missing_checkpoint"),
                    "sequence_skeleton": list(parent_candidate.get("sequence_skeleton") or []),
                    "function_sequence": list(parent_candidate.get("function_sequence") or []),
                    "required_prefix": list(parent_candidate.get("required_prefix") or []),
                    "suffix_actions": list(parent_candidate.get("suffix_actions") or []),
                    "sender_binding": parent_candidate.get("sender_policy") or parent_candidate.get("sender_constraints"),
                    "arg_binding": parent_candidate.get("argument_policy") or parent_candidate.get("arg_constraints"),
                    "receiver_binding": parent_candidate.get("receiver_binding") or parent_candidate.get("beneficiary_binding"),
                    "amount_binding": parent_candidate.get("value_policy") or parent_candidate.get("value_constraints"),
                    "effect_probe_plan": list(parent_candidate.get("effect_probe_plan") or []),
                    "materialization_status": parent_candidate.get("materialization_status_final") or parent_candidate.get("materialization_status"),
                    "param_sources": parent_candidate.get("param_sources") or {},
                },
                "execution_feedback": {
                    "execution_status": signature.get("execution_status"),
                    "reverted_steps": [],
                    "functions_reached": functions_reached,
                    "target_suffix_reached": bool(signature.get("target_suffix_reached")),
                    "state_delta_observed": bool(signature.get("state_delta_observed")),
                    "balance_delta_observed": bool(signature.get("balance_delta_observed")),
                    "token_transfer_observed": bool(signature.get("token_transfer_observed")),
                    "external_call_result_observed": bool(signature.get("external_call_result_observed")),
                    "probe_before": list(parent_candidate.get("probe_before") or []),
                    "probe_after": list(parent_candidate.get("probe_after") or []),
                    "probe_delta": list(parent_candidate.get("probe_delta") or []),
                    "no_delta_reason": ",".join(list(signature.get("evidence") or [])[:4]),
                    "revert_or_error_summary": ",".join(list(signature.get("evidence") or [])[:4]),
                },
                "progress_state": {
                    "before_best_level": signature.get("before_best_level"),
                    "after_best_level": signature.get("after_best_level"),
                    "before_strict_vspc": signature.get("before_strict_vspc"),
                    "after_strict_vspc": signature.get("after_strict_vspc"),
                    "covered_checkpoints_before": sorted(self.progress.covered),
                    "covered_checkpoints_after": sorted(self.progress.covered),
                    "runtime_frontier": _clean_list(parent_candidate.get("guided_sequence") or parent_candidate.get("function_sequence"))[-2:],
                },
                "failure_diagnosis": {
                    "failure_signature": signature.get("failure_signature"),
                    "confidence": signature.get("confidence"),
                    "suspected_wrong_fields": list(signature.get("suspected_wrong_fields") or []),
                    "evidence": list(signature.get("evidence") or []),
                },
                "state_obligation": state_obligation,
                "dependency_grounded_scope": repair_scope,
                "repair_policy": {
                    "must_address_obligation": True,
                    "must_choose_actions_from_scope": True,
                    "must_preserve_required_prefix_facts": True,
                    "must_explain_expected_runtime_evidence": True,
                    "observation_only_is_not_semantic_repair": True,
                    "return_no_repair_if_no_grounded_action_exists": True,
                },
                "repair_constraints": {
                    "must_preserve": [
                        "parent_candidate_id",
                        "target_missing_checkpoint",
                        "working_prefix_if_any",
                    ],
                    "allowed_repair_actions": list(signature.get("allowed_repair_actions") or []),
                    "forbidden_actions": [
                        "invent_non_abi_function",
                        "generate_unbounded_new_path",
                        "drop_target_missing_checkpoint",
                        "ignore_materialization_constraints",
                    ],
                    "max_added_transactions": 2,
                    "max_suffix_actions": 2,
                    "missing_checkpoint_requirements": self._missing_checkpoint_prompt_requirements(
                        _text(parent_candidate.get("target_missing_checkpoint")).strip()
                    ),
                },
                "available_context": {
                    "abi_function_list": self.abi_index.names(),
                    "mutable_functions": self.abi_index.names(),
                    "view_functions": self._probeable_view_functions(),
                    "static_summary": self._static_summary_hints(),
                    "known_accounts": list(self.accounts),
                    "observed_senders": list(observed.get("observed_senders") or []),
                    "observed_receivers": list(observed.get("observed_receivers") or []),
                    "observed_amounts": list(observed.get("observed_values") or []),
                    "observed_args_summary": list(observed.get("observed_args") or [])[-16:],
                    "probeable_view_functions": self._probeable_view_functions(),
                    "recent_failed_skeletons": list(self.repair_cooldown_signatures)[-8:],
                    "forbidden_pre_execution_sources": [
                        "audit_report_semantics",
                        "prepared_oracle_semantics",
                        "post_execution_baseline_results",
                    ],
                },
                "output_schema": {
                    "no_repair": False,
                    "reason": "string if no_repair",
                    "obligation_id": state_obligation.get("obligation_id"),
                    "repair_scope_id": repair_scope.get("repair_scope_id"),
                    "soft_hint_type": "prefer_state_writer_before_frontier|prefer_sender_context_variation|prefer_initialization_or_lifecycle_prefix|prefer_balance_or_allowance_setup|prefer_repeat_suffix_preserving_prefix|prefer_parameter_binding_repair|prefer_dependency_local_reordering|diagnostic_probe_only",
                    "addressed_obligation_kinds": ["string"],
                    "selected_scope_actions": [
                        {
                            "action": "functionName",
                            "action_role": "prefix_builder|state_writer|sink_action|repeat_action|observation",
                            "grounding_source": "string",
                        }
                    ],
                    "preserved_prefix_facts": ["string"],
                    "changed_fields": ["string"],
                    "binding_updates": {
                        "sender_binding": "string",
                        "beneficiary_binding": "string",
                        "receiver_binding": "string",
                        "amount_binding": "string",
                        "arg_binding": "string",
                    },
                    "prefix_updates": ["functionName"],
                    "suffix_updates": ["functionName"],
                    "effect_probe_plan": ["probe description"],
                    "expected_runtime_evidence": ["string"],
                    "semantic_repair_claim": "prefix|binding|suffix|mixed|observation_only",
                },
            },
            ensure_ascii=True,
        )

    def _infer_repair_action_from_signature(self, failure_signature: str, payload: Optional[Dict[str, Any]] = None) -> str:
        action = _text(_as_dict(payload).get("repair_action")).strip()
        if action:
            return action
        mapping = {
            "wrong_sender_binding": "fix_sender_binding",
            "wrong_beneficiary_binding": "fix_beneficiary_binding",
            "wrong_receiver_binding": "fix_receiver_binding",
            "wrong_amount_binding": "fix_amount_binding",
            "missing_prestate_prefix": "add_missing_prestate_prefix",
            "suffix_not_effective": "strengthen_suffix",
            "effect_function_not_reached": "strengthen_suffix",
            "effect_reached_but_no_state_delta": "combined_repair",
            "effect_probe_missing_or_uninformative": "change_effect_probe",
            "materialization_too_weak": "combined_repair",
        }
        return mapping.get(_text(failure_signature).strip(), "combined_repair")

    def _repair_intent_from_reason(self, reason: str) -> Dict[str, Any]:
        lowered = _text(reason).strip().lower()
        inferred_fields: List[str] = []
        warnings: List[str] = []
        inferred: Dict[str, Any] = {}
        if not lowered:
            return {"fields": inferred_fields, "warnings": warnings, "values": inferred}
        if "same sender" in lowered or "reuse sender" in lowered or "keep same sender" in lowered:
            inferred_fields.append("sender_binding")
            inferred["sender_binding_fix"] = "keep_same_sender_across_sequence"
        if "same beneficiary" in lowered or "reuse beneficiary" in lowered:
            inferred_fields.append("beneficiary_binding")
            inferred["beneficiary_binding_fix"] = "reuse_same_beneficiary"
        if "receiver" in lowered or "attacker receiver" in lowered or "change receiver" in lowered:
            inferred_fields.append("receiver_binding")
        if "observed balance" in lowered or "allowance" in lowered or "boundary amount" in lowered or "observed amount" in lowered:
            inferred_fields.append("amount_binding")
            inferred["amount_binding_fix"] = "use_observed_balance_or_allowance"
        if "probe" in lowered:
            inferred_fields.append("effect_probe")
        if "prefix" in lowered or "prestate" in lowered or "setup" in lowered:
            inferred_fields.append("prefix")
        if "suffix" in lowered or "repeat claim" in lowered or "follow-up" in lowered or "downstream" in lowered:
            inferred_fields.append("suffix_actions")
        if inferred_fields:
            warnings.append("low_confidence_reason_text_inference")
        return {
            "fields": _dedup_text(inferred_fields),
            "warnings": warnings,
            "values": inferred,
        }

    def _static_summary_hints(self) -> Dict[str, Any]:
        static_summary = getattr(self, "static_summary", None)
        if isinstance(static_summary, dict):
            return static_summary
        return {}

    def _infer_runtime_pattern_family(self, static_summary: Optional[Dict[str, Any]], abi: Any) -> Tuple[str, str]:
        oracle_family = _text(getattr(self, "oracle_family", "")).strip()
        if oracle_family in CHECKPOINT_TEMPLATES:
            return oracle_family, "ORACLE_SPEC_FAMILY"
        # Important: this family inference is intentionally derived only from
        # ABI names and non-report static-summary hints such as modifiers,
        # dangerous operations, and storage dependencies. It must not depend
        # on audit-report fields or oracle-spec semantic labels.
        summary = static_summary if isinstance(static_summary, dict) else {}
        privileged = {_text(item).strip().lower() for item in list(summary.get("privileged_functions") or []) if _text(item).strip()}
        dangerous = [
            item for item in list(summary.get("dangerous_operations") or [])
            if isinstance(item, dict)
        ]
        storage = [
            item for item in list(summary.get("storage_dependencies") or [])
            if isinstance(item, dict)
        ]
        abi_names = []
        if isinstance(abi, list):
            for item in abi:
                if isinstance(item, dict) and item.get("type") == "function":
                    name = _text(item.get("name")).strip().lower()
                    if name:
                        abi_names.append(name)
        abi_text = " ".join(sorted(set(abi_names)))
        dangerous_text = " ".join(
            _text(item.get("function")).strip().lower() + " " + _text(item.get("op")).strip().lower()
            for item in dangerous
        )
        storage_text = " ".join(
            _text(item.get("function")).strip().lower() + " " + _text(item.get("slot")).strip().lower()
            for item in storage
        )

        has_initializer = any(token in abi_text for token in ("initialize", "init", "setup"))
        non_initializer_privileged = any(
            not any(token in name for token in ("initialize", "init", "setup"))
            for name in privileged
        )
        permission_storage = any(
            token in storage_text
            for token in ("owner", "admin", "role", "minter", "operator", "manager", "govern", "auth", "permission", "blacklist", "whitelist", "pause")
        )
        if privileged and storage and (non_initializer_privileged or permission_storage):
            return "access_control_state_change", "STATIC_ANALYSIS_DERIVED"
        if has_initializer and privileged:
            return "initialization_takeover", "STATIC_ANALYSIS_DERIVED"
        if "claim" in abi_text and any(token in abi_text for token in ("stake", "harvest", "reward", "deposit")):
            return "repeat_claim_no_reset", "STATIC_ANALYSIS_DERIVED"
        if any(token in dangerous_text for token in ("call", "transfer", "send")) and any(
            token in abi_text for token in (ASSET_FLOW_SETTER_HINTS + ("update",))
        ):
            return "asset_flow_redirection", "STATIC_ANALYSIS_DERIVED"
        if privileged and storage:
            return "access_control_state_change", "STATIC_ANALYSIS_DERIVED"
        if any(token in dangerous_text for token in ("call", "delegatecall", "staticcall")):
            return "unchecked_external_call_effect", "STATIC_ANALYSIS_DERIVED"
        if any(token in abi_text for token in ("withdraw", "claim", "redeem")) and any(
            token in dangerous_text for token in ("call", "transfer", "send")
        ):
            return "reentrancy_state_inconsistency", "STATIC_ANALYSIS_DERIVED"
        if any(token in abi_text for token in ("mint", "burn", "share", "convert", "reward")) or "balance" in storage_text:
            return "arithmetic_accounting_invariant", "STATIC_ANALYSIS_DERIVED"
        if any(token in abi_text for token in ("pause", "blacklist", "lock", "ban", "block")):
            return "lock_pause_blacklist_bypass", "STATIC_ANALYSIS_DERIVED"
        return "critical_parameter_abuse", "STATIC_ANALYSIS_DERIVED_FALLBACK"

    def _runtime_obligation_observations(
        self,
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
        observed: Dict[str, Any],
    ) -> Dict[str, Any]:
        sequence = _normalize_text_list(parent_candidate.get("guided_sequence") or parent_candidate.get("function_sequence"))
        suffix = _normalize_text_list(parent_candidate.get("suffix_actions"))
        functions_reached = _normalize_text_list(signature.get("functions_reached"))
        repeated = []
        counts = Counter(functions_reached)
        for name, count in counts.items():
            if count > 1:
                repeated.append(name)
        return {
            "functions_reached": functions_reached,
            "target_suffix_reached": bool(signature.get("target_suffix_reached")),
            "state_delta_observed": bool(signature.get("state_delta_observed")),
            "balance_delta_observed": bool(signature.get("balance_delta_observed")),
            "token_transfer_observed": bool(signature.get("token_transfer_observed")),
            "external_call_result_observed": bool(signature.get("external_call_result_observed")),
            "same_sender_relation_observed": "keep_same_sender" in _text(parent_candidate.get("sender_policy") or parent_candidate.get("sender_constraints")).lower(),
            "same_beneficiary_relation_observed": "beneficiary" in _text(parent_candidate.get("beneficiary_binding")).lower(),
            "same_amount_relation_observed": "amount" in _text(parent_candidate.get("value_policy") or parent_candidate.get("value_constraints")).lower(),
            "repeated_function_observed": bool(repeated),
            "repeated_functions": repeated,
            "state_effect_functions": list(sequence[-2:]) if bool(signature.get("state_delta_observed")) else [],
            "sink_functions": list(suffix[-2:] or sequence[-1:]),
            "observation_only_functions": [item.get("function") for item in _normalize_probe_plan(parent_candidate.get("effect_probe_plan")) if isinstance(item, dict)],
            "source_evidence": list(signature.get("evidence") or []),
            "observed_senders": list(observed.get("observed_senders") or []),
            "observed_receivers": list(observed.get("observed_receivers") or []),
            "observed_values": list(observed.get("observed_values") or []),
            "candidate_repeat_action_available": len(suffix) >= 2 and (suffix[-1:] == suffix[-2:-1] or suffix[-1:] == sequence[-1:]),
        }

    def _build_state_obligation(
        self,
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
        observed: Dict[str, Any],
    ) -> Dict[str, Any]:
        next_missing = _text(parent_candidate.get("target_missing_checkpoint")).strip()
        runtime_observations = self._runtime_obligation_observations(parent_candidate, signature, observed)
        static_hints = self._static_summary_hints()
        runtime_observations["guard_related_functions"] = [
            _text(item.get("function")).strip()
            for item in list(static_hints.get("storage_dependencies") or [])
            if isinstance(item, dict) and _text(item.get("function")).strip() in set(_clean_list(runtime_observations.get("functions_reached")))
        ][:8]
        runtime_observations["state_writer_functions"] = [
            _text(item.get("function")).strip()
            for item in list(static_hints.get("storage_dependencies") or [])
            if isinstance(item, dict) and _text(item.get("function")).strip()
        ][:12]
        runtime_observations["sink_candidate_functions"] = [
            _text(item.get("function")).strip()
            for item in list(static_hints.get("dangerous_operations") or [])
            if isinstance(item, dict) and _text(item.get("function")).strip()
        ][:12]
        runtime_observations["repeat_candidate_functions"] = list(runtime_observations.get("repeated_functions") or [])
        runtime_observations["runtime_effect_signals"] = _dedup_text(
            (["tx_end_success"] if bool(signature.get("tx_success")) else [])
            + (["state_delta_observed"] if bool(signature.get("state_delta_observed")) else [])
            + (["balance_delta_observed"] if bool(signature.get("balance_delta_observed")) else [])
            + (["token_transfer_observed"] if bool(signature.get("token_transfer_observed")) else [])
            + (["external_call_result_observed"] if bool(signature.get("external_call_result_observed")) else [])
        )
        observed_hints: List[str] = []
        if bool(signature.get("reverted")):
            observed_hints.append("sender_or_role_context_variation_needed")
        if not bool(signature.get("state_delta_observed")) and runtime_observations["state_writer_functions"]:
            observed_hints.append("state_delta_expected")
        if not (
            bool(signature.get("balance_delta_observed"))
            or bool(signature.get("token_transfer_observed"))
            or bool(signature.get("external_call_result_observed"))
        ) and runtime_observations["sink_candidate_functions"]:
            observed_hints.append("asset_or_sink_effect_expected")
        if _text(signature.get("failure_signature")).strip() == "missing_prestate_prefix":
            observed_hints.append("initialization_or_lifecycle_prestate_needed")
        runtime_observations["observed_obligation_hints"] = _dedup_text(observed_hints)
        payload = self.state_obligation_builder.build(
            bench_id=self._runtime_bench_label(),
            oracle_family="",
            runtime_pattern_family=self.runtime_pattern_family,
            runtime_pattern_family_provenance=self.runtime_pattern_family_provenance,
            best_level_before=signature.get("before_best_level") or self.progress.best_level,
            target_missing_checkpoint=next_missing,
            missing_checkpoint_explanation="runtime:{}".format(next_missing),
            failure_signature=signature.get("failure_signature"),
            covered_checkpoints=sorted(self.progress.covered),
            parent_candidate=parent_candidate,
            execution_feedback={
                "tx_success": bool(signature.get("tx_success")),
                "reverted": bool(signature.get("reverted")),
                "functions_reached": list(signature.get("functions_reached") or []),
                "target_suffix_reached": bool(signature.get("target_suffix_reached")),
                "state_delta_observed": bool(signature.get("state_delta_observed")),
                "balance_delta_observed": bool(signature.get("balance_delta_observed")),
                "token_transfer_observed": bool(signature.get("token_transfer_observed")),
                "external_call_result_observed": bool(signature.get("external_call_result_observed")),
                "evidence": list(signature.get("evidence") or []),
            },
            runtime_observations=runtime_observations,
            static_hints=static_hints,
        )
        summary = self.state_obligation_builder.build_summary()
        for key, value in summary.items():
            if key.startswith("state_obligation_") and isinstance(value, int):
                self.failure_repair_pipeline_counters[key] = value
                if value > 0:
                    self.failure_repair_pipeline_zero_reasons[key] = ""
        self._bump_pipeline_counter("state_obligation_repair_requested", 1)
        return payload

    def _build_dependency_grounded_scope(
        self,
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
        observed: Dict[str, Any],
        state_obligation: Dict[str, Any],
    ) -> Dict[str, Any]:
        runtime_observations = self._runtime_obligation_observations(parent_candidate, signature, observed)
        static_hints = self._static_summary_hints()
        scope = self.dependency_grounded_scope_builder.build(
            state_obligation=state_obligation,
            parent_candidate=parent_candidate,
            runtime_observations=runtime_observations,
            successful_prefix_evidence={
                "required_prefix": list(parent_candidate.get("required_prefix") or []),
                "functions_reached": list(signature.get("functions_reached") or []),
            },
            static_hints=static_hints,
            abi_functions=self.abi_index.names(),
            abi_view_functions=[name for name in self.abi_index.names() if self.abi_index.is_view(name)],
            prior_repair_evidence={
                "observation_only_repair": bool(parent_candidate.get("observation_only_repair")),
                "semantic_delta_type": _text(parent_candidate.get("semantic_delta_type")).strip(),
            },
        )
        summary = self.dependency_grounded_scope_builder.build_summary()
        for key, value in summary.items():
            if key.startswith("repair_scope_") and isinstance(value, int):
                self.failure_repair_pipeline_counters[key] = value
                if value > 0:
                    self.failure_repair_pipeline_zero_reasons[key] = ""
        if bool(scope.get("no_safe_semantic_scope")):
            self._bump_pipeline_counter("no_safe_semantic_scope_count", 1)
        return scope

    def _parent_binding_snapshot(self, parent_candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sender_binding": _text(parent_candidate.get("sender_policy") or parent_candidate.get("sender_constraints")).strip(),
            "beneficiary_binding": _text(parent_candidate.get("beneficiary_binding")).strip(),
            "receiver_binding": _text(parent_candidate.get("receiver_binding")).strip(),
            "amount_binding": _text(parent_candidate.get("value_policy") or parent_candidate.get("value_constraints")).strip(),
            "effect_probe_plan": _normalize_probe_plan(parent_candidate.get("effect_probe_plan")),
            "required_prefix": _normalize_text_list(parent_candidate.get("required_prefix")),
            "suffix_actions": _normalize_text_list(parent_candidate.get("suffix_actions")),
            "parameter_plan": _clean_any_list(parent_candidate.get("parameter_plan")),
        }

    def _missing_checkpoint_requires_semantic_repair(self, checkpoint: str) -> bool:
        lowered = _text(checkpoint).strip().lower()
        if lowered in {
            "first_claim_success",
            "abnormal_balance_or_state_no_reset_observed",
            "privileged_or_asset_effect_observed",
            "privileged_effect_after_takeover",
            "funds_flow_to_changed_or_unexpected_receiver",
            "restriction_active",
            "balance_storage_or_asset_change_under_restriction",
            "abnormal_effect_observed",
            "state_or_asset_effect_after_failed_call",
            "invariant_check_possible",
            "invariant_violation_observed",
        }:
            return True
        return False

    def _repair_checkpoint_compatible(
        self,
        parent_checkpoint: str,
        current_checkpoint: str,
        signature: Optional[Dict[str, Any]] = None,
    ) -> bool:
        parent = _text(parent_checkpoint).strip()
        current = _text(current_checkpoint).strip()
        if not parent or not current:
            return False
        if parent == current:
            return True
        if parent in self.progress.covered:
            return False
        semantic_parent = self._missing_checkpoint_requires_semantic_repair(parent)
        semantic_current = self._missing_checkpoint_requires_semantic_repair(current)
        if not (semantic_parent and semantic_current):
            return False
        family = CHECKPOINT_TEMPLATES.get(self.runtime_pattern_family, [])
        if parent not in family or current not in family:
            return False
        failure_signature = _text((signature or {}).get("failure_signature")).strip()
        return failure_signature in {
            "suffix_not_effective",
            "effect_function_not_reached",
            "effect_reached_but_no_state_delta",
            "effect_probe_missing_or_uninformative",
            "missing_prestate_prefix",
        }

    def _semantic_delta_type_for_repair(
        self,
        *,
        parent_candidate: Dict[str, Any],
        changed_fields: Sequence[str],
        required_prefix: Sequence[str],
        suffix_actions: Sequence[str],
        effect_probe_fix: Sequence[Any],
        sender_binding_fix: str,
        beneficiary_binding_fix: str,
        receiver_binding_fix: str,
        amount_binding_fix: str,
    ) -> str:
        canonical_fields = _normalize_changed_fields(changed_fields)
        parent_prefix = _normalize_text_list(parent_candidate.get("required_prefix"))
        parent_suffix = _normalize_text_list(parent_candidate.get("suffix_actions"))
        parent_probe = _compact_probe_plan(parent_candidate.get("effect_probe_plan"))
        parent_sender = _text(parent_candidate.get("sender_policy") or parent_candidate.get("sender_constraints")).strip()
        parent_beneficiary = _text(parent_candidate.get("beneficiary_binding")).strip()
        parent_receiver = _text(parent_candidate.get("receiver_binding")).strip()
        parent_amount = _text(parent_candidate.get("value_policy") or parent_candidate.get("value_constraints")).strip()
        normalized_prefix = _normalize_text_list(required_prefix)
        normalized_suffix = _normalize_text_list(suffix_actions)
        normalized_probe = _compact_probe_plan(effect_probe_fix)
        if not canonical_fields:
            return "no_semantic_change"
        if set(canonical_fields) <= {"effect_probe"}:
            return "probe_only"
        binding_fields = {"receiver_binding", "beneficiary_binding", "effect_probe"}
        if set(canonical_fields).issubset(binding_fields):
            sender_same = _text(sender_binding_fix).strip() in {"", parent_sender}
            beneficiary_same = _text(beneficiary_binding_fix).strip() in {"", parent_beneficiary}
            receiver_same = _text(receiver_binding_fix).strip() in {"", parent_receiver}
            amount_same = _text(amount_binding_fix).strip() in {"", parent_amount}
            prefix_same = normalized_prefix == parent_prefix
            suffix_same = normalized_suffix == parent_suffix
            probe_changed = normalized_probe != parent_probe
            if sender_same and beneficiary_same and receiver_same and amount_same and prefix_same and suffix_same and probe_changed:
                return "probe_only"
            return "binding_only"
        if normalized_prefix != parent_prefix:
            return "prefix_changed"
        if normalized_suffix != parent_suffix:
            return "suffix_changed"
        if "sender_binding" in canonical_fields:
            return "sender_changed"
        if "amount_binding" in canonical_fields:
            return "amount_changed"
        if "arg_binding" in canonical_fields:
            return "arg_changed"
        return "concrete_tx_changed"

    def _semantic_delta_strength_for_repair(
        self,
        semantic_delta_type: str,
        changed_fields: Sequence[str],
    ) -> str:
        canonical_fields = set(_normalize_changed_fields(changed_fields))
        if semantic_delta_type in {"", "no_semantic_change"}:
            return "none"
        if semantic_delta_type in {"probe_only", "binding_only"}:
            return "weak"
        if semantic_delta_type in {"suffix_changed", "prefix_changed", "concrete_tx_changed"} and canonical_fields.intersection({"suffix_actions", "amount_binding", "sender_binding", "arg_binding", "prefix"}):
            return "strong"
        return "medium"

    def _observation_only_repair(self, checkpoint: str, semantic_delta_type: str) -> bool:
        if not self._missing_checkpoint_requires_semantic_repair(checkpoint):
            return False
        return semantic_delta_type in {"probe_only", "binding_only", "no_semantic_change"}

    def _missing_checkpoint_prompt_requirements(self, checkpoint: str) -> List[str]:
        lowered = _text(checkpoint).strip().lower()
        if lowered == "first_claim_success":
            return [
                "You must change at least one of sender_binding, beneficiary/receiver binding, amount_binding, arg_binding, suffix_actions, or prestate prefix.",
                "A probe-only repair is not sufficient; the sequence must plausibly improve tx success plus balance/state effect for the first claim path.",
            ]
        if lowered == "abnormal_balance_or_state_no_reset_observed":
            return [
                "You must include a repeated or accounting-changing action such as repeat claim/harvest/withdraw or another state-changing suffix.",
                "Changing only pendingReward/userInfo/poolInfo/owner probes is not sufficient semantic repair.",
            ]
        if lowered == "privileged_or_asset_effect_observed":
            return [
                "You must target a protected state-changing function or asset sink and preserve an unexpected sender path if available.",
                "Changing only owner/balanceOf/allowance/native_balance probes is not sufficient semantic repair.",
            ]
        if lowered in {"privileged_effect_after_takeover", "abnormal_effect_observed"}:
            return [
                "You must include a concrete state-changing prefix or dependent sink action, not only an observation probe.",
                "Prefer boundary arguments plus the affected privileged/business function when ABI-compatible.",
            ]
        if lowered == "invariant_check_possible":
            return [
                "You must preserve an accounting-changing path and attach invariant-relevant balance, pending, supply, or pool observations when ABI-compatible.",
                "Changing only generic owner, name, or decimals probes is not sufficient; prefer a repeated or downstream accounting action plus comparator views.",
            ]
        if lowered == "funds_flow_to_changed_or_unexpected_receiver":
            return [
                "You must preserve or create a receiver/fee/treasury mutation before the dependent asset-flow sink.",
                "Changing only balanceOf/allowance/native_balance probes is not sufficient; the repaired sequence must plausibly move value or touch a sink.",
            ]
        if lowered in {"restriction_active", "balance_storage_or_asset_change_under_restriction"}:
            return [
                "You must include a pause/blacklist/lock-style setup when ABI-compatible, followed by a restricted transfer/sell/withdraw path for the final checkpoint.",
                "Changing only paused/balanceOf/native_balance probes is not sufficient semantic repair.",
            ]
        if lowered == "state_or_asset_effect_after_failed_call":
            return [
                "You must keep the failed external-call path and add a state or asset-effect suffix when ABI-compatible.",
                "A probe-only repair is not sufficient unless the obligation is explicitly observation-only.",
            ]
        return []

    def _extract_nested_repair_payload(self, raw_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        candidates = [
            ("top_level", raw_payload),
            ("repair_candidate", _as_dict(raw_payload.get("repair_candidate"))),
            ("repair", _as_dict(raw_payload.get("repair"))),
            ("plan", _as_dict(raw_payload.get("plan"))),
            ("candidate", _as_dict(raw_payload.get("candidate"))),
            ("action_plan", _as_dict(raw_payload.get("action_plan"))),
            ("repair_plan", _as_dict(raw_payload.get("repair_plan"))),
        ]
        for label, payload in candidates:
            if payload:
                return payload, label
        return raw_payload, "top_level"

    def _normalize_scope_actions(self, value: Any) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        if not isinstance(value, list):
            return output
        for item in value:
            if not isinstance(item, dict):
                continue
            action = _text(item.get("action")).strip()
            if not action:
                continue
            output.append(
                {
                    "action": action,
                    "action_role": _text(item.get("action_role")).strip(),
                    "grounding_source": _text(item.get("grounding_source") or item.get("grounding_sources")).strip(),
                }
            )
        return output

    def _sanitize_effect_probe_fix(self, value: Any) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for item in _normalize_probe_plan(value):
            function_name = _canonical_action_name(item.get("function"))
            if not function_name:
                continue
            if function_name != "native_balance" and not self.abi_index.has(function_name):
                continue
            sanitized.append(
                {
                    "probe_type": _text(item.get("probe_type")).strip() or "agent_view_hint",
                    "function": function_name,
                    "targets": _normalize_text_list(item.get("targets")),
                }
            )
        return sanitized

    def _normalize_repair_payload(
        self,
        *,
        raw_payload: Dict[str, Any],
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
        observed: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = _as_dict(raw_payload)
        nested, raw_shape = self._extract_nested_repair_payload(payload)
        payload_binding_updates = _as_dict(payload.get("binding_updates"))
        nested_binding_updates = _as_dict(nested.get("binding_updates"))
        parent_id = _text(parent_candidate.get("candidate_id")).strip()
        failure_signature = _text(signature.get("failure_signature")).strip()
        reason_text = _text(payload.get("reason") or nested.get("reason") or payload.get("repair_reason") or nested.get("repair_reason")).strip()
        parent_snapshot = self._parent_binding_snapshot(parent_candidate)
        normalized: Dict[str, Any] = {
            "repair_action": self._infer_repair_action_from_signature(failure_signature, payload if payload else nested),
            "soft_hint_type": _text(payload.get("soft_hint_type") or nested.get("soft_hint_type")).strip(),
            "parent_candidate_id": _text(payload.get("parent_candidate_id") or nested.get("parent_candidate_id")).strip() or parent_id,
            "reuse_parent_prefix": bool(payload.get("reuse_parent_prefix", nested.get("reuse_parent_prefix", True))),
            "changed_fields_model": _dedup_text(
                _normalize_text_list(payload.get("changed_fields")) +
                _normalize_text_list(nested.get("changed_fields"))
            ),
            "changed_fields_inferred": [],
            "changed_fields_final": [],
            "additional_prefix": _normalize_text_list(
                payload.get("additional_prefix")
                or payload.get("prefix_updates")
                or nested.get("additional_prefix")
                or nested.get("prefix_updates")
                or nested.get("required_prefix")
            ),
            "suffix_actions": _normalize_text_list(
                payload.get("suffix_actions")
                or payload.get("suffix_updates")
                or nested.get("suffix_actions")
                or nested.get("suffix_updates")
                or nested.get("function_sequence")
                or nested.get("sequence_skeleton")
            ),
            "sender_binding_fix": _text(
                payload.get("sender_binding_fix")
                or payload_binding_updates.get("sender_binding")
                or nested.get("sender_binding_fix")
                or nested_binding_updates.get("sender_binding")
                or nested.get("sender_binding")
            ).strip(),
            "beneficiary_binding_fix": _text(
                payload.get("beneficiary_binding_fix")
                or payload_binding_updates.get("beneficiary_binding")
                or nested.get("beneficiary_binding_fix")
                or nested_binding_updates.get("beneficiary_binding")
                or nested.get("beneficiary_binding")
            ).strip(),
            "receiver_binding_fix": _text(
                payload.get("receiver_binding_fix")
                or payload_binding_updates.get("receiver_binding")
                or nested.get("receiver_binding_fix")
                or nested_binding_updates.get("receiver_binding")
                or nested.get("receiver_binding")
            ).strip(),
            "amount_binding_fix": _text(
                payload.get("amount_binding_fix")
                or payload_binding_updates.get("amount_binding")
                or nested.get("amount_binding_fix")
                or nested_binding_updates.get("amount_binding")
                or nested.get("amount_binding")
            ).strip(),
            "arg_binding_fix": _text(
                payload.get("arg_binding_fix")
                or payload_binding_updates.get("arg_binding")
                or nested.get("arg_binding_fix")
                or nested_binding_updates.get("arg_binding")
                or nested.get("arg_binding")
            ).strip(),
            "effect_probe_fix": self._sanitize_effect_probe_fix(
                payload.get("effect_probe_fix")
                or payload.get("effect_probe_plan")
                or nested.get("effect_probe_fix")
                or nested.get("effect_probe_plan")
            ),
            "parameter_plan": _clean_any_list(payload.get("parameter_plan") or nested.get("parameter_plan") or nested.get("param_sources")),
            "expected_checkpoint_delta": _text(payload.get("expected_checkpoint_delta") or nested.get("expected_checkpoint_delta") or nested.get("target_missing_checkpoint") or parent_candidate.get("target_missing_checkpoint")).strip(),
            "expected_evidence": _normalize_text_list(payload.get("expected_evidence") or nested.get("expected_evidence") or parent_candidate.get("expected_evidence")),
            "reason": reason_text or "normalized_failure_repair_payload",
            "confidence": _text(payload.get("confidence") or nested.get("confidence")).strip() or "medium",
            "no_repair": bool(payload.get("no_repair") or nested.get("no_repair")),
            "obligation_id": _text(payload.get("obligation_id") or nested.get("obligation_id")).strip(),
            "repair_scope_id": _text(payload.get("repair_scope_id") or nested.get("repair_scope_id")).strip(),
            "selected_scope_actions": self._normalize_scope_actions(payload.get("selected_scope_actions") or nested.get("selected_scope_actions")),
            "addressed_obligation_kinds": _normalize_text_list(payload.get("addressed_obligation_kinds") or nested.get("addressed_obligation_kinds")),
            "preserved_prefix_facts": _normalize_text_list(payload.get("preserved_prefix_facts") or nested.get("preserved_prefix_facts")),
            "expected_runtime_evidence": _normalize_text_list(payload.get("expected_runtime_evidence") or nested.get("expected_runtime_evidence")),
            "semantic_repair_claim": _text(payload.get("semantic_repair_claim") or nested.get("semantic_repair_claim")).strip(),
            "normalization_warnings": [],
            "raw_shape": raw_shape,
        }
        if normalized["soft_hint_type"] == "diagnostic_probe_only" and normalized["repair_action"] != "change_effect_probe":
            normalized["repair_action"] = "change_effect_probe"
            normalized["normalization_warnings"].append("repair_action_aligned_to_diagnostic_probe_only")
        if not normalized["effect_probe_fix"] and normalized["repair_action"] == "change_effect_probe":
            normalized["effect_probe_fix"] = list(parent_snapshot.get("effect_probe_plan") or [])
        reason_inference = self._repair_intent_from_reason(normalized["reason"])
        for key, value in _as_dict(reason_inference.get("values")).items():
            if not _text(normalized.get(key)).strip():
                normalized[key] = value
        normalized["normalization_warnings"].extend(_normalize_text_list(reason_inference.get("warnings")))
        if normalized["repair_action"] == "change_effect_probe" and not normalized["effect_probe_fix"]:
            normalized["effect_probe_fix"] = list(parent_snapshot.get("effect_probe_plan") or [])
        if normalized["repair_action"] == "add_missing_prestate_prefix" and not normalized["additional_prefix"]:
            normalized["additional_prefix"] = list(parent_snapshot.get("required_prefix") or [])
        if not normalized["suffix_actions"]:
            normalized["suffix_actions"] = list(parent_snapshot.get("suffix_actions") or [])
        if not normalized["parameter_plan"] and isinstance(nested.get("param_sources"), dict):
            plan_rows: List[Dict[str, Any]] = []
            for function_name, source_hint in nested.get("param_sources").items():
                text = _text(function_name).strip()
                if not text:
                    continue
                plan_rows.append(
                    {
                        "function": text,
                        "arg": "auto",
                        "source_type": "observed_trace",
                        "source_hint": _text(source_hint).strip() or "observed_trace",
                    }
                )
            normalized["parameter_plan"] = plan_rows
        normalized["reason_text_inference"] = reason_inference
        return normalized

    def _infer_changed_fields_from_parameter_plan(self, parameter_plan: Sequence[Any]) -> List[str]:
        inferred: List[str] = []
        for item in parameter_plan:
            if not isinstance(item, dict):
                continue
            arg_name = _text(item.get("arg")).strip().lower()
            source_hint = _text(item.get("source_hint")).strip().lower()
            if any(token in arg_name for token in ("sender", "caller", "from")) or "sender" in source_hint:
                inferred.append("sender_binding")
            if any(token in arg_name for token in ("beneficiary", "recipient")) or "beneficiary" in source_hint:
                inferred.append("beneficiary_binding")
            if any(token in arg_name for token in ("receiver", "to")) or "receiver" in source_hint or "attacker" in source_hint:
                inferred.append("receiver_binding")
            if any(token in arg_name for token in ("amount", "value", "qty", "shares")) or any(token in source_hint for token in ("allowance", "balance", "amount", "boundary")):
                inferred.append("amount_binding")
        return _dedup_text(inferred)

    def _scope_action_catalog(self, scope: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        catalog: Dict[str, Dict[str, Any]] = {}
        if not isinstance(scope, dict):
            return catalog
        for key in (
            "candidate_prefix_actions",
            "candidate_state_writer_actions",
            "candidate_sink_actions",
            "candidate_repeat_actions",
            "candidate_observation_actions",
        ):
            for item in list(scope.get(key) or []):
                if not isinstance(item, dict):
                    continue
                action = _text(item.get("action")).strip()
                if not action:
                    continue
                row = dict(item)
                raw_key = action
                canonical_key = _action_lookup_key(action)
                if raw_key not in catalog:
                    catalog[raw_key] = row
                if canonical_key and canonical_key not in catalog:
                    catalog[canonical_key] = row
        return catalog

    def _scope_aware_repair_validation(
        self,
        *,
        normalized_payload: Dict[str, Any],
        state_obligation: Dict[str, Any],
        repair_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        issues: List[str] = []
        warnings: List[str] = []
        valid = True
        obligation_id = _text(normalized_payload.get("obligation_id")).strip()
        scope_id = _text(normalized_payload.get("repair_scope_id")).strip()
        expected_obligation_id = _text(state_obligation.get("obligation_id")).strip()
        expected_scope_id = _text(repair_scope.get("repair_scope_id")).strip()
        obligation_kinds = set(_clean_list(state_obligation.get("obligation_kinds")))
        scope_catalog = self._scope_action_catalog(repair_scope)
        selected_actions = list(normalized_payload.get("selected_scope_actions") or [])
        selected_names = [_text(item.get("action")).strip() for item in selected_actions if isinstance(item, dict)]
        selected_keys = [_action_lookup_key(name) for name in selected_names]
        required_prefix_facts = {
            fact
            for fact in _clean_list(state_obligation.get("preserved_prefix_facts"))
            if not fact.startswith("functions_reached:")
        }
        preserved_prefix_facts = set(_clean_list(normalized_payload.get("preserved_prefix_facts")))
        expected_runtime_evidence = _clean_list(normalized_payload.get("expected_runtime_evidence"))
        semantic_claim = _text(normalized_payload.get("semantic_repair_claim")).strip()
        semantic_fields = set(_normalize_changed_fields(normalized_payload.get("changed_fields_final")))
        mutable_binding_fields = set(_normalize_changed_fields(repair_scope.get("mutable_binding_fields")))

        if expected_obligation_id and obligation_id != expected_obligation_id:
            valid = False
            issues.append("no_state_obligation")
        if expected_scope_id and scope_id != expected_scope_id:
            valid = False
            issues.append("no_dependency_grounded_scope")
        if bool(repair_scope.get("no_safe_semantic_scope")):
            if not bool(normalized_payload.get("no_repair")):
                valid = False
                issues.append("no_safe_semantic_repair")
            else:
                warnings.append("no_safe_semantic_scope")
        for action, action_key in zip(selected_names, selected_keys):
            if action and action not in scope_catalog and action_key not in scope_catalog:
                valid = False
                issues.append("selected_action_out_of_scope")
                break
        if required_prefix_facts and not required_prefix_facts.issubset(preserved_prefix_facts):
            valid = False
            issues.append("required_prefix_fact_not_preserved")
        if not expected_runtime_evidence and not bool(normalized_payload.get("no_repair")):
            valid = False
            issues.append("no_expected_runtime_evidence")
        observation_only = semantic_claim == "observation_only" or semantic_fields.issubset({"effect_probe"}) if semantic_fields else semantic_claim == "observation_only"
        if observation_only and "observe_already_produced_effect" not in obligation_kinds and not bool(normalized_payload.get("no_repair")):
            valid = False
            issues.append("observation_only_misaligned_with_obligation")
        binding_field_to_fix = {
            "sender_binding": _text(normalized_payload.get("sender_binding_fix")).strip(),
            "beneficiary_binding": _text(normalized_payload.get("beneficiary_binding_fix")).strip(),
            "receiver_binding": _text(normalized_payload.get("receiver_binding_fix")).strip(),
            "amount_binding": _text(normalized_payload.get("amount_binding_fix")).strip(),
            "arg_binding": _text(normalized_payload.get("arg_binding_fix")).strip(),
        }
        unsupported_binding_updates = [
            field
            for field, value in binding_field_to_fix.items()
            if value and field in semantic_fields and field not in mutable_binding_fields
        ]
        if unsupported_binding_updates and not bool(normalized_payload.get("no_repair")):
            valid = False
            issues.append("binding_update_not_supported_by_evidence")
        if selected_keys:
            semantic_scope_keys = {
                _action_lookup_key(item.get("action"))
                for item in list(repair_scope.get("candidate_state_writer_actions") or [])
                + list(repair_scope.get("candidate_sink_actions") or [])
                + list(repair_scope.get("candidate_repeat_actions") or [])
                + list(repair_scope.get("candidate_prefix_actions") or [])
                if isinstance(item, dict) and _action_lookup_key(item.get("action"))
            }
            observation_scope_keys = {
                _action_lookup_key(item.get("action"))
                for item in list(repair_scope.get("candidate_observation_actions") or [])
                if isinstance(item, dict) and _action_lookup_key(item.get("action"))
            }
            selected_semantic_keys = [key for key in selected_keys if key in semantic_scope_keys]
            if not observation_only and not bool(normalized_payload.get("no_repair")):
                if selected_actions and not selected_semantic_keys and semantic_fields.intersection({
                    "prefix",
                    "suffix_actions",
                    "sender_binding",
                    "beneficiary_binding",
                    "receiver_binding",
                    "amount_binding",
                    "arg_binding",
                }):
                    valid = False
                    issues.append("ungrounded_semantic_change")
                elif any(key not in semantic_scope_keys and key not in observation_scope_keys for key in selected_keys):
                    valid = False
                    issues.append("ungrounded_semantic_change")
        if not valid and "observation_only_misaligned_with_obligation" in issues:
            self._bump_pipeline_counter("observation_only_misaligned_repair_count", 1)
        if not valid and "ungrounded_semantic_change" in issues:
            self._bump_pipeline_counter("ungrounded_semantic_change_rejected_count", 1)
        if valid and selected_keys and not observation_only:
            self._bump_pipeline_counter("grounded_semantic_repair_proposed", 1)
        return {
            "valid": valid,
            "issues": _dedup_text(issues),
            "warnings": _dedup_text(warnings),
            "selected_scope_action_names": _dedup_text(selected_names),
        }

    def _hydrate_scope_bound_payload(
        self,
        *,
        normalized_payload: Dict[str, Any],
        state_obligation: Dict[str, Any],
        repair_scope: Dict[str, Any],
        parent_candidate: Dict[str, Any],
    ) -> None:
        if not _text(normalized_payload.get("obligation_id")).strip():
            normalized_payload["obligation_id"] = _text(state_obligation.get("obligation_id")).strip()
            normalized_payload.setdefault("normalization_warnings", []).append("obligation_id_defaulted_from_runtime_context")
        if not _text(normalized_payload.get("repair_scope_id")).strip():
            normalized_payload["repair_scope_id"] = _text(repair_scope.get("repair_scope_id")).strip()
            normalized_payload.setdefault("normalization_warnings", []).append("repair_scope_id_defaulted_from_runtime_context")
        if not _clean_list(normalized_payload.get("preserved_prefix_facts")) and bool(normalized_payload.get("reuse_parent_prefix")):
            normalized_payload["preserved_prefix_facts"] = list(state_obligation.get("preserved_prefix_facts") or [])
        if not _clean_list(normalized_payload.get("expected_runtime_evidence")):
            normalized_payload["expected_runtime_evidence"] = list(state_obligation.get("required_effect_evidence") or [])
        if not normalized_payload.get("effect_probe_fix") and "effect_probe" in set(_normalize_changed_fields(normalized_payload.get("changed_fields_final"))):
            observation_actions: List[Dict[str, Any]] = []
            for item in list(normalized_payload.get("selected_scope_actions") or []):
                if isinstance(item, dict) and _text(item.get("action_role")).strip() == "observation":
                    observation_actions.append(item)
            if not observation_actions:
                for item in list(repair_scope.get("candidate_observation_actions") or []):
                    if isinstance(item, dict):
                        observation_actions.append(item)
            probe_rows: List[Dict[str, Any]] = []
            for item in observation_actions:
                function_name = _canonical_action_name(item.get("action"))
                if not function_name:
                    continue
                probe_rows.append(
                    {
                        "probe_type": "agent_view_hint",
                        "function": function_name,
                        "targets": [],
                    }
                )
            if probe_rows:
                normalized_payload["effect_probe_fix"] = probe_rows[:6]
                normalized_payload.setdefault("normalization_warnings", []).append("effect_probe_fix_inferred_from_scope_observation_actions")
        if not _text(normalized_payload.get("semantic_repair_claim")).strip():
            changed = set(_normalize_changed_fields(normalized_payload.get("changed_fields_final")))
            if changed.issubset({"effect_probe"}) and changed:
                normalized_payload["semantic_repair_claim"] = "observation_only"
            elif "prefix" in changed:
                normalized_payload["semantic_repair_claim"] = "prefix"
            elif "suffix_actions" in changed:
                normalized_payload["semantic_repair_claim"] = "suffix"
            elif changed.intersection({"sender_binding", "beneficiary_binding", "receiver_binding", "amount_binding", "arg_binding"}):
                normalized_payload["semantic_repair_claim"] = "binding"
            elif changed:
                normalized_payload["semantic_repair_claim"] = "mixed"
        if (
            _text(normalized_payload.get("soft_hint_type")).strip() != "diagnostic_probe_only"
            and _text(normalized_payload.get("repair_action")).strip() == "change_effect_probe"
            and set(_normalize_changed_fields(normalized_payload.get("changed_fields_final"))) != {"effect_probe"}
        ):
            normalized_payload.setdefault("normalization_warnings", []).append("probe_fallback_not_allowed_for_non_probe_soft_hint")
        if list(normalized_payload.get("selected_scope_actions") or []):
            return
        scope_catalog = self._scope_action_catalog(repair_scope)
        inferred: List[Dict[str, Any]] = []
        candidate_actions = _dedup_text(
            _normalize_text_list(normalized_payload.get("additional_prefix"))
            + _normalize_text_list(normalized_payload.get("suffix_actions"))
        )
        if "effect_probe" in set(_normalize_changed_fields(normalized_payload.get("changed_fields_final"))):
            for probe in _normalize_probe_plan(normalized_payload.get("effect_probe_fix")):
                function = _text(probe.get("function")).strip()
                if function:
                    candidate_actions.append(function)
        for action in _dedup_text(candidate_actions):
            meta = scope_catalog.get(action)
            if not isinstance(meta, dict):
                continue
            inferred.append(
                {
                    "action": action,
                    "action_role": _text(meta.get("action_role")).strip(),
                    "grounding_source": ",".join(_clean_list(meta.get("grounding_sources"))[:2]),
                }
            )
        if inferred:
            normalized_payload["selected_scope_actions"] = inferred
            normalized_payload.setdefault("normalization_warnings", []).append("selected_scope_actions_inferred_from_scope_aligned_updates")

    def _infer_changed_fields(
        self,
        *,
        normalized_payload: Dict[str, Any],
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
    ) -> Dict[str, Any]:
        parent_snapshot = self._parent_binding_snapshot(parent_candidate)
        inferred: List[str] = []
        inference_reason_by_field: Dict[str, str] = {}
        inference_confidence_by_field: Dict[str, str] = {}

        def remember(field: str, reason: str, confidence: str = "high") -> None:
            if field not in inferred:
                inferred.append(field)
            inference_reason_by_field[field] = reason
            inference_confidence_by_field[field] = confidence

        sender_fix = _text(normalized_payload.get("sender_binding_fix")).strip()
        if sender_fix and sender_fix != _text(parent_snapshot.get("sender_binding")).strip():
            remember("sender_binding", "sender_binding_fix_differs_from_parent")
        beneficiary_fix = _text(normalized_payload.get("beneficiary_binding_fix")).strip()
        if beneficiary_fix and beneficiary_fix != _text(parent_snapshot.get("beneficiary_binding")).strip():
            remember("beneficiary_binding", "beneficiary_binding_fix_differs_from_parent")
        receiver_fix = _text(normalized_payload.get("receiver_binding_fix")).strip()
        if receiver_fix and receiver_fix != _text(parent_snapshot.get("receiver_binding")).strip():
            remember("receiver_binding", "receiver_binding_fix_differs_from_parent")
        amount_fix = _text(normalized_payload.get("amount_binding_fix")).strip()
        if amount_fix and amount_fix != _text(parent_snapshot.get("amount_binding")).strip():
            remember("amount_binding", "amount_binding_fix_differs_from_parent")
        arg_fix = _text(normalized_payload.get("arg_binding_fix")).strip()
        if arg_fix and arg_fix != _text(parent_candidate.get("argument_policy") or parent_candidate.get("arg_constraints")).strip():
            remember("arg_binding", "arg_binding_fix_differs_from_parent")
        if _normalize_text_list(normalized_payload.get("additional_prefix")) and _normalize_text_list(normalized_payload.get("additional_prefix")) != _normalize_text_list(parent_snapshot.get("required_prefix")):
            remember("prefix", "additional_prefix_differs_from_parent")
        if _normalize_text_list(normalized_payload.get("suffix_actions")) and _normalize_text_list(normalized_payload.get("suffix_actions")) != _normalize_text_list(parent_snapshot.get("suffix_actions")):
            remember("suffix_actions", "suffix_actions_differ_from_parent")
        if _compact_probe_plan(normalized_payload.get("effect_probe_fix")) and _compact_probe_plan(normalized_payload.get("effect_probe_fix")) != _compact_probe_plan(parent_snapshot.get("effect_probe_plan")):
            remember("effect_probe", "effect_probe_fix_differs_from_parent")
        for field in self._infer_changed_fields_from_parameter_plan(normalized_payload.get("parameter_plan") or []):
            remember(field, "parameter_plan_mentions_binding_field", "medium")

        failure_signature = _text(signature.get("failure_signature")).strip()
        fallback_map = {
            "wrong_sender_binding": "sender_binding",
            "wrong_beneficiary_binding": "beneficiary_binding",
            "wrong_receiver_binding": "receiver_binding",
            "wrong_amount_binding": "amount_binding",
            "suffix_not_effective": "suffix_actions",
            "effect_function_not_reached": "suffix_actions",
            "effect_reached_but_no_state_delta": "suffix_actions",
            "effect_probe_missing_or_uninformative": "effect_probe",
            "missing_prestate_prefix": "prefix",
        }
        fallback_field = fallback_map.get(failure_signature)
        if fallback_field and fallback_field not in inferred:
            remember(fallback_field, "failure_signature_default_field", "medium")
        reason_fields = _normalize_text_list(_as_dict(normalized_payload.get("reason_text_inference")).get("fields"))
        for field in reason_fields:
            canonical = {
                "sender_binding": "sender_binding",
                "beneficiary_binding": "beneficiary_binding",
                "receiver_binding": "receiver_binding",
                "amount_binding": "amount_binding",
                "effect_probe": "effect_probe",
                "prefix": "prefix",
                "suffix_actions": "suffix_actions",
            }.get(field, field)
            if canonical:
                remember(canonical, "reason_text_keyword_inference", "low")

        changed_fields_model = _normalize_text_list(normalized_payload.get("changed_fields_model"))
        changed_fields_final = _dedup_text(changed_fields_model + inferred)
        log_payload = {
            "parent_candidate_id": parent_candidate.get("candidate_id"),
            "repair_candidate_temp_id": _text(normalized_payload.get("parent_candidate_id")).strip() or _text(parent_candidate.get("candidate_id")).strip(),
            "changed_fields_model": changed_fields_model,
            "changed_fields_inferred": inferred,
            "changed_fields_final": changed_fields_final,
            "inference_reason_by_field": inference_reason_by_field,
            "inference_confidence_by_field": inference_confidence_by_field,
        }
        _append_jsonl(self.changed_fields_inference_log_path, log_payload)
        self.changed_fields_inference_summary["attempted"] += 1
        if inferred:
            self.changed_fields_inference_summary["changed_fields_inferred_count"] += 1
        for field in changed_fields_final:
            self.changed_fields_inference_summary["field:{}".format(field)] += 1
        normalized_payload["changed_fields_inferred"] = inferred
        normalized_payload["changed_fields_final"] = changed_fields_final
        return log_payload

    def _binding_signature(self, candidate: Dict[str, Any]) -> str:
        payload = {
            "sender_policy": _text(candidate.get("sender_policy") or candidate.get("sender_constraints")).strip(),
            "argument_policy": _text(candidate.get("argument_policy") or candidate.get("arg_constraints")).strip(),
            "value_policy": _text(candidate.get("value_policy") or candidate.get("value_constraints")).strip(),
            "receiver_binding": _text(candidate.get("receiver_binding")).strip(),
            "beneficiary_binding": _text(candidate.get("beneficiary_binding")).strip(),
            "effect_probe_plan": _normalize_probe_plan(candidate.get("effect_probe_plan")),
            "parameter_plan": _clean_any_list(candidate.get("parameter_plan")),
        }
        return _json_compact(payload)

    def _repair_signature(self, candidate: Dict[str, Any]) -> str:
        payload = {
            "sequence_signature": _json_compact(_normalize_text_list(candidate.get("guided_sequence") or candidate.get("function_sequence"))),
            "binding_signature": self._binding_signature(candidate),
            "target_missing_checkpoint": _text(candidate.get("target_missing_checkpoint")).strip(),
        }
        return _json_compact(payload)

    def _repair_attempt_key(
        self,
        *,
        bench_id: str,
        parent_candidate_id: str,
        failure_signature_id: str,
        repair_type: str,
        next_missing_checkpoint: str,
        normalized_changed_fields: Sequence[str],
        binding_signature: str,
    ) -> str:
        payload = {
            "bench_id": _text(bench_id).strip(),
            "parent_candidate_id": _text(parent_candidate_id).strip(),
            "failure_signature_id": _text(failure_signature_id).strip(),
            "repair_type": _text(repair_type).strip(),
            "next_missing_checkpoint": _text(next_missing_checkpoint).strip(),
            "normalized_changed_fields": _normalize_text_list(normalized_changed_fields),
            "binding_signature": _text(binding_signature).strip(),
        }
        return _json_compact(payload)

    def _record_rejected_repair_attempt(
        self,
        *,
        repair_attempt_key: str,
        reason: str,
        candidate: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if repair_attempt_key:
            self.rejected_repair_attempt_keys.add(repair_attempt_key)
        reason_text = _text(reason).strip() or "unknown"
        self.rejected_repair_attempt_reasons[reason_text] += 1
        self.repair_attempt_cache_summary["rejected_cache_size"] = len(self.rejected_repair_attempt_keys)
        payload = {
            "repair_attempt_key": repair_attempt_key,
            "reason": reason_text,
        }
        if isinstance(candidate, dict):
            payload.update(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "parent_candidate_id": candidate.get("parent_candidate_id"),
                    "failure_signature": candidate.get("failure_signature"),
                    "repair_action": candidate.get("repair_action"),
                    "repair_signature": candidate.get("repair_signature"),
                    "binding_signature": candidate.get("binding_signature"),
                    "sequence_signature": candidate.get("sequence_signature"),
                    "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
                }
            )
        if isinstance(extra, dict):
            payload.update(extra)
        _append_jsonl(self.rejected_repair_keys_path, payload)

    def _build_rule_repair_fallback_payload(
        self,
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
        rejected_reason: str,
        repair_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        fallback = self._deterministic_failure_repair_payload(parent_candidate, signature, repair_scope=repair_scope)
        fallback["fallback_from_llm_repair"] = True
        fallback["llm_repair_rejected_reason"] = rejected_reason
        fallback["fallback_origin"] = "llm_repair_rejected"
        if rejected_reason in {
            "selected_action_out_of_scope",
            "ungrounded_semantic_change",
            "materialization_failed_unsupported_hint",
            "no_safe_semantic_repair",
        }:
            fallback["no_repair"] = True
            fallback["repair_action"] = "no_repair"
            fallback["reason"] = "fallback_blocked_due_to_scope_or_hint_support"
        return fallback

    def _materialize_failure_repair_candidate(
        self,
        *,
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
        payload: Dict[str, Any],
        observed: Dict[str, Any],
        prompt_record: Optional[Dict[str, Any]] = None,
        state_obligation: Optional[Dict[str, Any]] = None,
        repair_scope: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        parent_candidate_id = _text(parent_candidate.get("candidate_id")).strip()
        failure_signature_id = _text(signature.get("failure_signature_id") or signature.get("signature_id") or signature.get("failure_signature")).strip()
        if int(self.failure_repair_pipeline_counters.get("repair_materialization_attempted", 0) or 0) >= self.max_repair_materialization_attempts_per_run:
            self._bump_pipeline_counter("repair_materialization_budget_exceeded", 1)
            self._bump_pipeline_counter("repair_storm_guard_triggered", 1)
            self.repair_attempt_cache_summary["repair_materialization_budget_exceeded"] += 1
            self._set_zero_reason("repair_materialized", "repair_materialization_budget_exceeded")
            return None
        self._bump_pipeline_counter("repair_materialization_attempted", 1)
        normalized_payload = self._normalize_repair_payload(
            raw_payload=_as_dict(payload),
            parent_candidate=parent_candidate,
            signature=signature,
            observed=observed,
        )
        state_obligation = state_obligation if isinstance(state_obligation, dict) and state_obligation else self._build_state_obligation(parent_candidate, signature, observed)
        repair_scope = repair_scope if isinstance(repair_scope, dict) and repair_scope else self._build_dependency_grounded_scope(parent_candidate, signature, observed, state_obligation)
        self._infer_changed_fields(
            normalized_payload=normalized_payload,
            parent_candidate=parent_candidate,
            signature=signature,
        )
        self._hydrate_scope_bound_payload(
            normalized_payload=normalized_payload,
            state_obligation=state_obligation,
            repair_scope=repair_scope,
            parent_candidate=parent_candidate,
        )
        scope_validation = self._scope_aware_repair_validation(
            normalized_payload=normalized_payload,
            state_obligation=state_obligation,
            repair_scope=repair_scope,
        )
        normalization_log = {
            "candidate_id": "",
            "parent_candidate_id": parent_candidate_id,
            "raw_shape": normalized_payload.get("raw_shape"),
            "normalized_success": not bool(normalized_payload.get("no_repair")),
            "changed_fields_model": list(normalized_payload.get("changed_fields_model") or []),
            "changed_fields_inferred": list(normalized_payload.get("changed_fields_inferred") or []),
            "changed_fields_final": list(normalized_payload.get("changed_fields_final") or []),
            "normalization_warnings": list(normalized_payload.get("normalization_warnings") or []),
            "normalization_reject_reason": "",
            "repair_action_final": normalized_payload.get("repair_action"),
            "no_repair": bool(normalized_payload.get("no_repair")),
            "obligation_id": normalized_payload.get("obligation_id"),
            "repair_scope_id": normalized_payload.get("repair_scope_id"),
            "scope_validation": scope_validation,
        }
        _append_jsonl(self.repair_payload_normalization_log_path, normalization_log)
        self.repair_payload_normalization_summary["attempted"] += 1
        if normalized_payload.get("changed_fields_inferred"):
            self.repair_payload_normalization_summary["changed_fields_inferred_count"] += 1
        if normalized_payload.get("changed_fields_final"):
            self.repair_payload_normalization_summary["normalized_repair_payload_count"] += 1
        else:
            self.repair_payload_normalization_summary["normalized_without_changed_fields"] += 1
        payload = normalized_payload
        parent_sequence = list(parent_candidate.get("guided_sequence") or [])
        if _text(payload.get("soft_hint_type")).strip() not in {"", "diagnostic_probe_only"}:
            semantic_changed_fields = set(_normalize_changed_fields(payload.get("changed_fields_final")))
            if semantic_changed_fields - {"effect_probe"}:
                selected_roles = {
                    _text(item.get("action_role")).strip()
                    for item in list(payload.get("selected_scope_actions") or [])
                    if isinstance(item, dict)
                }
                if "prefix" in semantic_changed_fields:
                    payload["repair_action"] = "add_missing_prestate_prefix"
                elif selected_roles.intersection({"repeat_action", "sink_action"}) or "suffix_actions" in semantic_changed_fields:
                    payload["repair_action"] = "strengthen_suffix"
                elif semantic_changed_fields.intersection({"sender_binding", "beneficiary_binding", "receiver_binding"}):
                    payload["repair_action"] = "fix_sender_binding"
                elif semantic_changed_fields.intersection({"amount_binding", "arg_binding"}):
                    payload["repair_action"] = "fix_amount_binding"
                else:
                    payload["repair_action"] = "combined_repair"
        repair_action = _text(payload.get("repair_action")).strip()
        target_missing = _text(parent_candidate.get("target_missing_checkpoint")).strip()
        changed_fields = _normalize_changed_fields(payload.get("changed_fields_final"))
        pre_candidate_binding_signature = _json_compact({
            "sender_binding_fix": _text(payload.get("sender_binding_fix")).strip(),
            "beneficiary_binding_fix": _text(payload.get("beneficiary_binding_fix")).strip(),
            "receiver_binding_fix": _text(payload.get("receiver_binding_fix")).strip(),
            "amount_binding_fix": _text(payload.get("amount_binding_fix")).strip(),
            "arg_binding_fix": _text(payload.get("arg_binding_fix")).strip(),
            "effect_probe_fix": _normalize_probe_plan(payload.get("effect_probe_fix")),
            "parameter_plan": _clean_any_list(payload.get("parameter_plan")),
        })
        repair_attempt_key = self._repair_attempt_key(
            bench_id=self._runtime_bench_label(),
            parent_candidate_id=parent_candidate_id,
            failure_signature_id=failure_signature_id,
            repair_type=repair_action or "combined_repair",
            next_missing_checkpoint=target_missing,
            normalized_changed_fields=changed_fields,
            binding_signature=pre_candidate_binding_signature,
        )
        self._bump_pipeline_counter("repair_attempt_key_created", 1)
        if payload.get("no_repair") or repair_action == "no_repair":
            self.failure_repair_summary["no_repair_count"] += 1
            if prompt_record is not None:
                prompt_record["normalization_result"] = normalization_log
            return None
        if (
            _text(payload.get("soft_hint_type")).strip() not in {"", "diagnostic_probe_only"}
            and repair_action == "change_effect_probe"
            and set(changed_fields) != {"effect_probe"}
        ):
            explicit_reason = "unsupported_non_probe_soft_hint_materialization"
            self._bump_pipeline_counter("repair_rejected", 1)
            self.repair_materialization_summary["rejected"] += 1
            self.repair_materialization_summary["materialization_failed_unsupported_hint"] += 1
            self.failure_repair_summary["rejected_repair_count"] += 1
            normalization_log["normalization_reject_reason"] = explicit_reason
            if prompt_record is not None:
                prompt_record["normalization_result"] = normalization_log
                prompt_record["materialization_result"] = {
                    "materialization_status_final": "rejected_unsupported_non_probe_soft_hint_materialization",
                    "materialization_reason": explicit_reason,
                }
            return None
        if not bool(scope_validation.get("valid")):
            reason = ",".join(list(scope_validation.get("issues") or [])) or "scope_validation_failed"
            self._bump_pipeline_counter("repair_rejected", 1)
            self.repair_materialization_summary["rejected"] += 1
            self.repair_materialization_summary["rejected_scope_validation"] += 1
            self.failure_repair_summary["rejected_repair_count"] += 1
            normalization_log["normalization_reject_reason"] = reason
            self._record_rejected_repair_attempt(
                repair_attempt_key="",
                reason=reason,
                extra={
                    "parent_candidate_id": parent_candidate_id,
                    "failure_signature": signature.get("failure_signature"),
                    "target_missing_checkpoint": target_missing,
                },
            )
            if prompt_record is not None:
                prompt_record["normalization_result"] = normalization_log
                prompt_record["materialization_result"] = {
                    "materialization_status_final": "rejected_scope_validation",
                    "materialization_reason": reason,
                }
            return None
        if not changed_fields:
            self._bump_pipeline_counter("repair_rejected_no_changed_fields", 1)
            self._bump_pipeline_counter("repair_rejected", 1)
            self._bump_pipeline_counter("repair_storm_guard_triggered", 1)
            self.repair_materialization_summary["rejected"] += 1
            self.repair_materialization_summary["rejected_no_changed_fields"] += 1
            self.failure_repair_summary["rejected_repair_count"] += 1
            normalization_log["normalization_reject_reason"] = "no_changed_fields"
            self._record_rejected_repair_attempt(
                repair_attempt_key=repair_attempt_key,
                reason="no_changed_fields",
                extra={
                    "parent_candidate_id": parent_candidate_id,
                    "failure_signature": signature.get("failure_signature"),
                    "target_missing_checkpoint": target_missing,
                },
            )
            if prompt_record is not None:
                prompt_record["normalization_result"] = normalization_log
                prompt_record["materialization_result"] = {
                    "materialization_status_final": "rejected_no_changed_fields",
                    "materialization_reason": "no_changed_fields",
                }
            return None
        required_prefix = _normalize_text_list(payload.get("additional_prefix"))
        if bool(payload.get("reuse_parent_prefix")):
            parent_prefix = _normalize_text_list(parent_candidate.get("required_prefix"))
            required_prefix = _dedup_text(parent_prefix + required_prefix) if (parent_prefix or required_prefix) else []
        suffix_actions = _normalize_text_list(payload.get("suffix_actions"))
        if not suffix_actions:
            suffix_actions = _normalize_text_list(parent_candidate.get("suffix_actions")) or parent_sequence[-2:]
        required_prefix, suffix_actions, sequence_sanitization_notes = self._sanitize_failure_repair_sequence(
            required_prefix=required_prefix,
            suffix_actions=suffix_actions,
            parent_candidate=parent_candidate,
            target_missing=target_missing,
        )
        if sequence_sanitization_notes:
            normalization_log["sequence_sanitization_notes"] = list(sequence_sanitization_notes)
        sequence = self._merge_prefix_suffix(required_prefix, suffix_actions)
        mapping = ACTION_MAPPING.get(self.runtime_pattern_family, {}).get(target_missing, {"goal": "failure aware repair"})
        candidate = self._new_candidate(
            source="agent" if self.failure_repair_use_agent else "tfi",
            next_missing=target_missing,
            mapping=mapping,
            sequence=sequence,
            observed=observed,
            build_reason="failure_aware_agent_repair",
            raw_agent_payload=payload,
        )
        candidate["candidate_type"] = "checkpoint_guided_candidate_v3_1_failure_repair"
        candidate["bench_id"] = self._runtime_bench_label()
        candidate["contract_name"] = self._runtime_contract_label()
        candidate["parent_candidate_id"] = parent_candidate_id
        candidate["failure_repair_candidate"] = True
        candidate["failure_signature"] = signature.get("failure_signature")
        candidate["failure_signature_id"] = failure_signature_id
        candidate["source_trigger_id"] = failure_signature_id or _text(signature.get("failure_signature")).strip()
        candidate["source_parent_id"] = parent_candidate_id
        candidate["source_scope_id"] = _text(parent_candidate.get("scope_id") or parent_candidate.get("source_task_id")).strip()
        candidate["source_candidate_kind"] = _text(candidate.get("candidate_type")).strip() or "checkpoint_guided_failure_repair"
        candidate["repair_action"] = repair_action or "combined_repair"
        candidate["repair_attempt_index"] = int(self.repair_attempts_by_parent[parent_candidate_id]) + 1
        candidate["required_prefix"] = list(required_prefix)
        candidate["suffix_actions"] = list(suffix_actions)
        candidate["effect_probe_plan"] = _normalize_probe_plan(payload.get("effect_probe_fix") or parent_candidate.get("effect_probe_plan") or [])
        candidate["probe_appended"] = bool(candidate.get("effect_probe_plan"))
        candidate["uses_llm_hypothesis"] = bool(self.failure_repair_use_agent)
        candidate["repair_no_changed_fields"] = not bool(changed_fields)
        candidate["repair_target_mismatch"] = target_missing != _text(parent_candidate.get("target_missing_checkpoint")).strip()
        candidate["repair_unresolved_binding"] = False
        candidate["repair_changed_fields_model"] = list(payload.get("changed_fields_model") or [])
        candidate["repair_changed_fields_inferred"] = list(payload.get("changed_fields_inferred") or [])
        candidate["repair_changed_fields_final"] = list(changed_fields)
        candidate["repair_sequence_sanitization_notes"] = list(sequence_sanitization_notes)
        candidate["sender_policy"] = (
            _text(payload.get("sender_binding_fix")).strip()
            or _text(parent_candidate.get("sender_policy")).strip()
            or _text(parent_candidate.get("sender_constraints")).strip()
            or "any_existing_sender"
        )
        candidate["value_policy"] = (
            _text(payload.get("amount_binding_fix")).strip()
            or _text(parent_candidate.get("value_policy")).strip()
            or _text(parent_candidate.get("value_constraints")).strip()
            or "keep_original"
        )
        candidate["argument_policy"] = (
            _text(payload.get("arg_binding_fix")).strip()
            or _text(parent_candidate.get("argument_policy")).strip()
            or _text(parent_candidate.get("arg_constraints")).strip()
            or "abi_default_or_boundary_mix"
        )
        candidate["sender_constraints"] = candidate["sender_policy"]
        candidate["value_constraints"] = candidate["value_policy"]
        candidate["arg_constraints"] = candidate["argument_policy"]
        candidate["parameter_plan"] = payload.get("parameter_plan") if isinstance(payload.get("parameter_plan"), list) else []
        candidate["target_missing_checkpoint"] = target_missing
        candidate["target_function"] = sequence[-1] if sequence else parent_candidate.get("target_function")
        candidate["prefix_replay"] = bool(payload.get("reuse_parent_prefix"))
        candidate["suffix_mutation_type"] = _text(payload.get("repair_action")).strip() or "failure_repair_suffix"
        candidate["priority"] = "checkpoint"
        candidate["confidence"] = _safe_float(payload.get("confidence"), parent_candidate.get("confidence", 0.65))
        candidate["receiver_binding"] = _text(payload.get("receiver_binding_fix")).strip() or _text(parent_candidate.get("receiver_binding")).strip()
        candidate["beneficiary_binding"] = _text(payload.get("beneficiary_binding_fix")).strip() or _text(parent_candidate.get("beneficiary_binding")).strip()
        candidate["sequence_signature"] = _json_compact(sequence)
        candidate["parent_sequence_signature"] = _json_compact(parent_sequence)
        candidate["binding_signature"] = self._binding_signature(candidate)
        parent_binding_signature = self._binding_signature(parent_candidate)
        candidate["repair_signature"] = self._repair_signature(candidate)
        candidate["repair_sender_binding_changed"] = "sender_binding" in changed_fields and candidate["sender_policy"] != _text(parent_candidate.get("sender_policy") or parent_candidate.get("sender_constraints")).strip()
        candidate["repair_beneficiary_binding_changed"] = "beneficiary_binding" in changed_fields and candidate["beneficiary_binding"] != _text(parent_candidate.get("beneficiary_binding")).strip()
        candidate["repair_receiver_binding_changed"] = "receiver_binding" in changed_fields and candidate["receiver_binding"] != _text(parent_candidate.get("receiver_binding")).strip()
        candidate["repair_amount_binding_changed"] = "amount_binding" in changed_fields and candidate["value_policy"] != _text(parent_candidate.get("value_policy") or parent_candidate.get("value_constraints")).strip()
        candidate["repair_probe_binding_changed"] = "effect_probe" in changed_fields and _compact_probe_plan(candidate.get("effect_probe_plan")) != _compact_probe_plan(parent_candidate.get("effect_probe_plan"))
        candidate["repair_binding_only"] = bool(changed_fields) and list(sequence) == list(parent_sequence) and candidate["binding_signature"] != parent_binding_signature
        candidate["repair_repeats_parent"] = list(sequence) == list(parent_sequence) and candidate["binding_signature"] == parent_binding_signature
        semantic_delta_type = self._semantic_delta_type_for_repair(
            parent_candidate=parent_candidate,
            changed_fields=changed_fields,
            required_prefix=required_prefix,
            suffix_actions=suffix_actions,
            effect_probe_fix=payload.get("effect_probe_fix") or [],
            sender_binding_fix=_text(payload.get("sender_binding_fix")).strip(),
            beneficiary_binding_fix=_text(payload.get("beneficiary_binding_fix")).strip(),
            receiver_binding_fix=_text(payload.get("receiver_binding_fix")).strip(),
            amount_binding_fix=_text(payload.get("amount_binding_fix")).strip(),
        )
        semantic_delta_strength = self._semantic_delta_strength_for_repair(
            semantic_delta_type,
            changed_fields,
        )
        observation_only = self._observation_only_repair(target_missing, semantic_delta_type)
        candidate["semantic_delta_type"] = semantic_delta_type
        candidate["semantic_delta_strength"] = semantic_delta_strength
        candidate["observation_only_repair"] = bool(observation_only)
        candidate["state_obligation"] = state_obligation
        candidate["state_obligation_id"] = state_obligation.get("obligation_id")
        candidate["dependency_grounded_scope"] = repair_scope
        candidate["dependency_grounded_scope_id"] = repair_scope.get("repair_scope_id")
        candidate["repair_scope_validation"] = scope_validation
        candidate["selected_scope_actions"] = list(normalized_payload.get("selected_scope_actions") or [])
        candidate["expected_runtime_evidence"] = list(normalized_payload.get("expected_runtime_evidence") or [])
        candidate["semantic_repair_claim"] = _text(normalized_payload.get("semantic_repair_claim")).strip()
        candidate["repair_attempt_key"] = repair_attempt_key
        if semantic_delta_type == "probe_only":
            self._bump_pipeline_counter("repair_probe_only_count", 1)
        if observation_only:
            self._bump_pipeline_counter("repair_observation_only_count", 1)
        if semantic_delta_strength == "weak":
            self._bump_pipeline_counter("repair_semantic_delta_weak", 1)
        elif semantic_delta_strength == "medium":
            self._bump_pipeline_counter("repair_semantic_delta_medium", 1)
        elif semantic_delta_strength == "strong":
            self._bump_pipeline_counter("repair_semantic_delta_strong", 1)
        if repair_attempt_key in self.rejected_repair_attempt_keys:
            candidate["materialization_status_final"] = "rejected_cached_attempt"
            candidate["materialization_reason"] = "repair_attempt_key_cached_rejected"
            self._bump_pipeline_counter("repair_attempt_key_skipped_cached", 1)
            self._bump_pipeline_counter("repair_rejected", 1)
            self._bump_pipeline_counter("repair_storm_guard_triggered", 1)
            self.repair_materialization_summary["rejected"] += 1
            self.repair_materialization_summary["rejected_cached_attempt"] += 1
            self.failure_repair_summary["rejected_repair_count"] += 1
            self._record_rejected_repair_attempt(
                repair_attempt_key=repair_attempt_key,
                reason="repair_attempt_key_cached_rejected",
                candidate=candidate,
            )
            return None
        parent_signature_key = "{}::{}".format(parent_candidate_id, failure_signature_id or repair_action or "unknown")
        if self.repair_attempts_by_parent_signature[parent_signature_key] >= self.max_repair_attempts_per_parent_signature:
            candidate["materialization_status_final"] = "rejected_repair_attempt_budget_exceeded"
            candidate["materialization_reason"] = "repair_attempt_budget_exceeded"
            self._bump_pipeline_counter("repair_attempt_budget_exceeded", 1)
            self._bump_pipeline_counter("repair_rejected", 1)
            self._bump_pipeline_counter("repair_storm_guard_triggered", 1)
            self.repair_materialization_summary["rejected"] += 1
            self.repair_materialization_summary["rejected_repair_attempt_budget_exceeded"] += 1
            self.failure_repair_summary["rejected_repair_count"] += 1
            self._record_rejected_repair_attempt(
                repair_attempt_key=repair_attempt_key,
                reason="repair_attempt_budget_exceeded",
                candidate=candidate,
                extra={"parent_signature_key": parent_signature_key},
            )
            return None
        self.repair_attempts_by_parent_signature[parent_signature_key] += 1
        self._mark_modules(
            candidate,
            ["failure_aware_agent_repair", "parameter_materializer"],
            "failure_aware_agent_repair",
        )
        if candidate.get("prefix_replay"):
            self._mark_modules(candidate, ["prefix_replay"], "failure_repair_prefix_replay")
        if candidate.get("required_prefix"):
            self._mark_modules(candidate, ["prestate_builder"], "failure_repair_prestate")
        if candidate.get("suffix_actions"):
            self._mark_modules(candidate, ["suffix_mutation"], "failure_repair_suffix")
        if candidate.get("probe_appended"):
            self._mark_modules(candidate, ["effect_probe"], "failure_repair_probe")
        candidate = self._finalize_candidate_materialization(candidate, observed)
        materialized_repair_signature = self._repair_signature(candidate)
        candidate["repair_signature"] = materialized_repair_signature
        if materialized_repair_signature in self.repair_recent_signatures:
            candidate["repair_repeats_parent"] = True
            candidate["materialization_status_final"] = "rejected_repair_repeats_parent"
            candidate["materialization_reason"] = ",".join(
                [bit for bit in [candidate.get("materialization_reason"), "duplicate_repair_signature"] if bit]
            )
            self._bump_pipeline_counter("repair_rejected_repair_repeats_parent", 1)
            self._bump_pipeline_counter("repair_rejected_duplicate_signature", 1)
            self._bump_pipeline_counter("repair_storm_guard_triggered", 1)
            self._record_rejected_repair_attempt(
                repair_attempt_key=repair_attempt_key,
                reason="repair_repeats_parent",
                candidate=candidate,
                extra={"parent_signature_key": parent_signature_key},
            )
        elif not _text(candidate.get("materialization_status_final")).strip().startswith("rejected_"):
            self.repair_recent_signatures.append(materialized_repair_signature)
        log_payload = {
            "candidate_id": candidate.get("candidate_id"),
            "parent_candidate_id": parent_candidate_id,
            "failure_signature": signature.get("failure_signature"),
            "repair_action": candidate.get("repair_action"),
            "changed_fields": changed_fields,
            "changed_fields_model": list(payload.get("changed_fields_model") or []),
            "changed_fields_inferred": list(payload.get("changed_fields_inferred") or []),
            "required_prefix": list(required_prefix),
            "suffix_actions": list(suffix_actions),
            "materialization_status_final": candidate.get("materialization_status_final"),
            "materialization_reason": candidate.get("materialization_reason"),
            "repair_attempt_index": candidate.get("repair_attempt_index"),
            "modules_used": list(candidate.get("modules_used") or []),
            "repair_binding_only": bool(candidate.get("repair_binding_only")),
            "repair_signature": candidate.get("repair_signature"),
            "binding_signature": candidate.get("binding_signature"),
            "sequence_signature": candidate.get("sequence_signature"),
            "repair_attempt_key": repair_attempt_key,
            "semantic_delta_type": semantic_delta_type,
            "semantic_delta_strength": semantic_delta_strength,
            "observation_only_repair": bool(observation_only),
            "state_obligation_id": candidate.get("state_obligation_id"),
            "repair_scope_id": candidate.get("dependency_grounded_scope_id"),
            "scope_validation_valid": bool(scope_validation.get("valid")),
            "scope_validation_issues": list(scope_validation.get("issues") or []),
            "selected_scope_actions": list(candidate.get("selected_scope_actions") or []),
            "expected_runtime_evidence": list(candidate.get("expected_runtime_evidence") or []),
            "semantic_repair_claim": candidate.get("semantic_repair_claim"),
            "sequence_sanitization_notes": list(sequence_sanitization_notes),
        }
        _append_jsonl(self.repair_materialization_log_path, log_payload)
        final_status = _text(candidate.get("materialization_status_final")).strip()
        is_rejected = final_status.startswith("rejected_")
        self.repair_materialization_summary["materialized" if not is_rejected else "rejected"] += 1
        self.repair_materialization_summary[final_status or "unknown"] += 1
        if prompt_record is not None:
            prompt_record["repair_candidate_id"] = candidate.get("candidate_id")
            prompt_record["normalization_result"] = normalization_log
            prompt_record["materialization_result"] = {
                "materialization_status_final": candidate.get("materialization_status_final"),
                "materialization_reason": candidate.get("materialization_reason"),
            }
        if is_rejected:
            self._bump_pipeline_counter("repair_rejected", 1)
            self.failure_repair_summary["rejected_repair_count"] += 1
            if final_status == "rejected_repair_repeats_parent":
                self._record_rejected_repair_attempt(
                    repair_attempt_key=repair_attempt_key,
                    reason="repair_repeats_parent",
                    candidate=candidate,
                    extra={"parent_signature_key": parent_signature_key},
                )
            if self.failure_repair_use_agent and not bool(payload.get("fallback_from_llm_repair")):
                fallback_payload = self._build_rule_repair_fallback_payload(
                    parent_candidate,
                    signature,
                    _text(candidate.get("materialization_status_final")).strip() or "rejected",
                    repair_scope=repair_scope,
                )
                fallback_log = {
                    "parent_candidate_id": parent_candidate_id,
                    "failure_signature": signature.get("failure_signature"),
                    "llm_repair_rejected_reason": _text(candidate.get("materialization_status_final")).strip() or "rejected",
                    "fallback_repair_action": fallback_payload.get("repair_action"),
                    "fallback_materialized": False,
                    "fallback_executed": False,
                }
                retry_prompt = dict(prompt_record or {})
                retry_prompt["fallback_from_llm_repair"] = True
                retry_candidate = self._materialize_failure_repair_candidate(
                    parent_candidate=parent_candidate,
                    signature=signature,
                    payload=fallback_payload,
                    observed=observed,
                    prompt_record=retry_prompt,
                    state_obligation=state_obligation,
                    repair_scope=repair_scope,
                )
                if retry_candidate is not None:
                    fallback_log["fallback_materialized"] = True
                    self.repair_fallback_summary["fallback_materialized"] += 1
                else:
                    self.repair_fallback_summary["fallback_rejected"] += 1
                _append_jsonl(self.repair_fallback_log_path, fallback_log)
                self.repair_fallback_summary["fallback_attempted"] += 1
                if retry_candidate is not None:
                    return retry_candidate
            return None
        self.repair_attempts_by_parent[parent_candidate_id] += 1
        self.failure_repair_summary["repair_candidate_materialized"] += 1
        self._bump_pipeline_counter("repair_materialized", 1)
        if not observation_only:
            self._bump_pipeline_counter("semantic_repair_materialized", 1)
        if not observation_only and not bool(repair_scope.get("no_safe_semantic_scope")):
            self._bump_pipeline_counter("grounded_semantic_repair_materialized", 1)
        self.repair_lifecycle.mark_materialized(
            candidate,
            extra={
                "source_trigger_id": candidate.get("source_trigger_id"),
                "source_parent_id": candidate.get("source_parent_id"),
                "source_scope_id": candidate.get("source_scope_id"),
                "source_candidate_kind": candidate.get("source_candidate_kind"),
            },
        )
        return candidate

    def _build_failure_repair_candidates(
        self,
        next_missing: str,
        observed: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self.enable_failure_repair_pipeline:
            self._set_zero_reason("repair_trigger_true", "failure_repair_pipeline_disabled")
            self._set_zero_reason("repair_materialized", "failure_repair_pipeline_disabled")
            self._set_zero_reason("repair_executed", "failure_repair_pipeline_disabled")
            return []
        built: List[Dict[str, Any]] = []
        signature_ids = list(self.failure_signatures_by_candidate.keys())
        pending_ids = [
            candidate_id
            for candidate_id in signature_ids
            if candidate_id in self.pending_failure_repair_signature_ids
        ]
        other_ids = [
            candidate_id
            for candidate_id in signature_ids
            if candidate_id not in self.pending_failure_repair_signature_ids
        ]
        # Pending signatures are the repair queue. Keep them at the end because
        # the bounded selection below takes the most recent tail.
        candidate_ids = other_ids + pending_ids
        if not candidate_ids:
            self._set_zero_reason("repair_trigger_true", "no_failure_signatures_available")
            self._set_zero_reason("llm_repair_call_attempted", "no_failure_signatures_available")
            self._set_zero_reason("repair_materialized", "no_failure_signatures_available")
            return []
        for candidate_id in candidate_ids[-max(1, self.max_per_round * 3):]:
            parent_candidate = self.candidates_by_id.get(candidate_id)
            signature = self.failure_signatures_by_candidate.get(candidate_id)
            if not isinstance(parent_candidate, dict) or not isinstance(signature, dict):
                continue
            if not self._repair_checkpoint_compatible(
                _text(parent_candidate.get("target_missing_checkpoint")).strip(),
                _text(next_missing).strip(),
                signature,
            ):
                continue
            trigger = self._should_trigger_failure_repair(signature, parent_candidate)
            if not bool(trigger.get("trigger_agent_repair")):
                self.pending_failure_repair_signature_ids.discard(candidate_id)
                continue
            prompt_record: Dict[str, Any] = {
                "stage": "failure_aware_agent_repair",
                "parent_candidate_id": parent_candidate.get("candidate_id"),
                "target_missing_checkpoint": next_missing,
                "runtime_pattern_family": self.runtime_pattern_family,
                "system_prompt": self._failure_repair_system_prompt(),
            }
            payload: Dict[str, Any]
            fallback_after_llm_error = False
            fallback_reason = ""
            state_obligation = self._build_state_obligation(parent_candidate, signature, observed)
            repair_scope = self._build_dependency_grounded_scope(parent_candidate, signature, observed, state_obligation)
            prompt_record["state_obligation_id"] = state_obligation.get("obligation_id")
            prompt_record["repair_scope_id"] = repair_scope.get("repair_scope_id")
            user_prompt = self._build_failure_repair_prompt(
                parent_candidate,
                signature,
                observed,
                state_obligation=state_obligation,
                repair_scope=repair_scope,
            )
            prompt_record["user_prompt"] = user_prompt
            if self.failure_repair_use_agent and self.llm_provider and getattr(self.llm_provider, "is_available", lambda: False)() and getattr(self.llm_provider, "is_configured", lambda: False)():
                started = time.time()
                self.llm_call_count += 1
                self._bump_pipeline_counter("llm_repair_call_attempted", 1)
                try:
                    payload = self.llm_provider.generate_json(
                        user_prompt,
                        system_prompt=self._failure_repair_system_prompt(),
                    )
                    ended = time.time()
                    duration = max(0.0, ended - started)
                    self.llm_success_count += 1
                    self.llm_interaction_time_total_sec += duration
                    self._bump_pipeline_counter("llm_repair_call_success", 1)
                    prompt_record["started_at"] = started
                    prompt_record["ended_at"] = ended
                    prompt_record["duration_sec"] = round(duration, 6)
                    prompt_record["model_output"] = payload
                    prompt_record["metadata"] = self.llm_provider.get_last_response_metadata()
                    prompt_record["raw_model_output"] = getattr(self.llm_provider, "get_last_response_text", lambda: "")()
                    if self.phase_cost_env is not None:
                        try:
                            record_phase_event(self.phase_cost_env, prompt_record)
                        except Exception:
                            pass
                except Exception as exc:
                    ended = time.time()
                    duration = max(0.0, ended - started)
                    self.llm_error_count += 1
                    self.llm_interaction_time_total_sec += duration
                    self.failure_repair_summary["llm_error"] += 1
                    self._bump_pipeline_counter("llm_repair_call_error", 1)
                    prompt_record["started_at"] = started
                    prompt_record["ended_at"] = ended
                    prompt_record["duration_sec"] = round(duration, 6)
                    prompt_record["error"] = str(exc)
                    if self.phase_cost_env is not None:
                        try:
                            record_phase_event(self.phase_cost_env, prompt_record)
                        except Exception:
                            pass
                    fallback_after_llm_error = True
                    fallback_reason = "llm_request_error"
                    payload = self._build_rule_repair_fallback_payload(
                        parent_candidate,
                        signature,
                        fallback_reason,
                        repair_scope=repair_scope,
                    )
                    prompt_record["fallback_from_llm_repair"] = True
                    prompt_record["fallback_reason"] = fallback_reason
                    prompt_record["fallback_output"] = payload
                    self.repair_fallback_summary["fallback_attempted"] += 1
                    self.repair_fallback_summary["fallback_after_llm_error"] += 1
            else:
                payload = self._deterministic_failure_repair_payload(parent_candidate, signature, repair_scope=repair_scope)
                prompt_record["model_output"] = payload
                prompt_record["raw_model_output"] = json.dumps(payload, ensure_ascii=True)
                prompt_record["metadata"] = {"mode": "rule_repair"}
                if self.failure_repair_use_agent:
                    self._set_zero_reason("llm_repair_call_attempted", "llm_provider_unavailable_or_not_configured")
            if isinstance(payload, dict):
                self._bump_pipeline_counter("repair_plan_parsed", 1)
                prompt_record["parsed_output"] = payload
            if isinstance(payload, dict) and (payload.get("no_repair") or _text(payload.get("repair_action")).strip() == "no_repair"):
                self._bump_pipeline_counter("llm_repair_no_repair", 1)
            try:
                candidate = self._materialize_failure_repair_candidate(
                    parent_candidate=parent_candidate,
                    signature=signature,
                    payload=payload,
                    observed=observed,
                    prompt_record=prompt_record,
                    state_obligation=state_obligation,
                    repair_scope=repair_scope,
                )
            except Exception as exc:
                self._bump_pipeline_counter("repair_subsystem_error", 1)
                self.failure_repair_summary["repair_subsystem_error"] += 1
                prompt_record["repair_subsystem_error"] = str(exc)
                _append_jsonl(
                    self.repair_subsystem_error_log_path,
                    {
                        "parent_candidate_id": parent_candidate.get("candidate_id"),
                        "failure_signature": signature.get("failure_signature"),
                        "target_missing_checkpoint": next_missing,
                        "error": str(exc),
                    },
                )
                self._remember_failure_repair_interaction(prompt_record)
                _append_jsonl(self.failure_repair_interactions_path, prompt_record)
                if fallback_after_llm_error:
                    self.repair_fallback_summary["fallback_rejected"] += 1
                    _append_jsonl(
                        self.repair_fallback_log_path,
                        {
                            "parent_candidate_id": parent_candidate.get("candidate_id"),
                            "failure_signature": signature.get("failure_signature"),
                            "llm_repair_rejected_reason": fallback_reason or "llm_request_error",
                            "fallback_repair_action": payload.get("repair_action") if isinstance(payload, dict) else "",
                            "fallback_materialized": False,
                            "fallback_executed": False,
                            "fallback_repair_candidate_id": "",
                            "fallback_materialization_status": "repair_subsystem_error",
                            "fallback_materialization_reason": str(exc),
                        },
                    )
                continue
            self._remember_failure_repair_interaction(prompt_record)
            _append_jsonl(self.failure_repair_interactions_path, prompt_record)
            if fallback_after_llm_error:
                fallback_log = {
                    "parent_candidate_id": parent_candidate.get("candidate_id"),
                    "failure_signature": signature.get("failure_signature"),
                    "llm_repair_rejected_reason": fallback_reason or "llm_request_error",
                    "fallback_repair_action": payload.get("repair_action") if isinstance(payload, dict) else "",
                    "fallback_materialized": bool(candidate is not None),
                    "fallback_executed": False,
                    "fallback_repair_candidate_id": candidate.get("candidate_id") if isinstance(candidate, dict) else "",
                    "fallback_materialization_status": candidate.get("materialization_status_final") if isinstance(candidate, dict) else "rejected",
                    "fallback_materialization_reason": candidate.get("materialization_reason") if isinstance(candidate, dict) else "candidate_not_materialized",
                }
                if candidate is None:
                    self.repair_fallback_summary["fallback_rejected"] += 1
                else:
                    self.repair_fallback_summary["fallback_materialized"] += 1
                _append_jsonl(self.repair_fallback_log_path, fallback_log)
            if candidate is None:
                continue
            self.pending_failure_repair_signature_ids.discard(candidate_id)
            built.append(candidate)
            self.failure_repair_summary["repair_candidate_built"] += 1
            self._bump_pipeline_counter("repair_queued", 1)
            self.repair_lifecycle.mark_stage(
                candidate.get("candidate_id"),
                stage="queued",
                seed={
                    "bench_id": candidate.get("bench_id") or self._runtime_bench_label(),
                    "contract": candidate.get("contract_name") or self._runtime_contract_label(),
                    "source_trigger_id": candidate.get("source_trigger_id"),
                    "source_parent_id": candidate.get("source_parent_id") or parent_candidate.get("candidate_id"),
                    "source_scope_id": candidate.get("source_scope_id"),
                    "source_candidate_kind": candidate.get("source_candidate_kind") or candidate.get("candidate_type"),
                },
            )
            _append_jsonl(
                self.repair_queue_events_path,
                {
                    "repair_candidate_id": candidate.get("candidate_id"),
                    "parent_candidate_id": parent_candidate.get("candidate_id"),
                    "repair_attempt_index": candidate.get("repair_attempt_index"),
                    "queued_at_global_tx": None,
                    "executed_at_global_tx": None,
                    "queue_delay": None,
                    "skipped_reason": "",
                    "cooldown_reason": "",
                    "group_name": _text(candidate.get("group_name")).strip(),
                    "failure_signature": candidate.get("failure_signature"),
                    "repair_action": candidate.get("repair_action"),
                    "sequence_signature": candidate.get("sequence_signature"),
                    "binding_signature": candidate.get("binding_signature"),
                    "repair_signature": candidate.get("repair_signature"),
                    "is_binding_only_repair": bool(candidate.get("repair_binding_only")),
                    "duplicate_reason": "",
                    "accepted_by_queue": True,
                },
            )
        return built

    def _deterministic_failure_repair_payload(
        self,
        parent_candidate: Dict[str, Any],
        signature: Dict[str, Any],
        repair_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        failure_signature = _text(signature.get("failure_signature")).strip()
        target_missing = _text(parent_candidate.get("target_missing_checkpoint")).strip().lower()
        parent_prefix = _clean_list(parent_candidate.get("required_prefix"))
        guided_sequence = _clean_list(parent_candidate.get("guided_sequence"))
        base_suffix = _clean_list(parent_candidate.get("suffix_actions")) or _clean_list(parent_candidate.get("guided_sequence"))[-2:]
        repair_scope = repair_scope if isinstance(repair_scope, dict) else {}
        scope_catalog = self._scope_action_catalog(repair_scope)
        scope_rows: List[Dict[str, Any]] = []
        for key in (
            "candidate_prefix_actions",
            "candidate_state_writer_actions",
            "candidate_sink_actions",
            "candidate_repeat_actions",
            "candidate_observation_actions",
        ):
            for item in list(repair_scope.get(key) or []):
                if isinstance(item, dict) and _text(item.get("action")).strip():
                    scope_rows.append(item)

        def pick_actions(keywords: Sequence[str], preferred: Sequence[str] = ()) -> List[str]:
            return self.abi_index.select_by_keywords(keywords, preferred=preferred)

        def merge_actions(*parts: Sequence[str], limit: int = 4) -> List[str]:
            merged: List[str] = []
            for part in parts:
                merged.extend(_normalize_text_list(part))
            return _dedup_text(merged, limit=max(1, int(limit)))

        def pick_scope_actions(
            *,
            roles: Sequence[str] = (),
            keywords: Sequence[str] = (),
            related_tokens: Sequence[str] = (),
            preferred: Sequence[str] = (),
            limit: int = 6,
        ) -> List[str]:
            lowered_roles = {_text(item).strip().lower() for item in roles if _text(item).strip()}
            lowered_keywords = [_text(item).strip().lower() for item in keywords if _text(item).strip()]
            lowered_related = [_text(item).strip().lower() for item in related_tokens if _text(item).strip()]
            output: List[str] = []

            def add(action_name: Any) -> None:
                name = _canonical_action_name(action_name)
                if not name or not self.abi_index.has(name):
                    return
                canonical = self.abi_index.by_lower.get(name.lower(), name)
                if canonical not in output:
                    output.append(canonical)

            def matches(row: Dict[str, Any]) -> bool:
                if lowered_roles:
                    role = _text(row.get("action_role")).strip().lower()
                    if role not in lowered_roles:
                        return False
                action = _text(row.get("action")).strip().lower()
                related = " ".join(
                    _clean_list(row.get("related_state_or_sink"))
                    + _clean_list(row.get("grounding_sources"))
                    + _clean_list(row.get("related_binding_requirements"))
                ).lower()
                if lowered_keywords and any(key in action or key in related for key in lowered_keywords):
                    return True
                if lowered_related and any(token in action or token in related for token in lowered_related):
                    return True
                return not lowered_keywords and not lowered_related

            for item in preferred:
                clean = _text(item).strip()
                if not clean:
                    continue
                row = scope_catalog.get(clean) or scope_catalog.get(_action_lookup_key(clean))
                if isinstance(row, dict) and matches(row):
                    add(clean)
            for row in scope_rows:
                if matches(row):
                    add(row.get("action"))
                if len(output) >= max(1, int(limit)):
                    break
            return output[: max(1, int(limit))]

        def build_probe_plan(preferred_names: Sequence[str], fallback: Sequence[Any] = ()) -> List[Dict[str, Any]]:
            existing = _normalize_probe_plan(fallback or parent_candidate.get("effect_probe_plan") or [])
            existing_by_key: Dict[str, Dict[str, Any]] = {}
            for row in existing:
                function_name = _canonical_action_name(row.get("function"))
                key = _action_lookup_key(function_name)
                if function_name and key and key not in existing_by_key:
                    existing_by_key[key] = dict(row)
            output: List[Dict[str, Any]] = []
            seen = set()
            for name in _dedup_text(list(preferred_names)):
                function_name = _canonical_action_name(name)
                key = _action_lookup_key(function_name)
                if not function_name or not key or key in seen:
                    continue
                seen.add(key)
                if key in existing_by_key:
                    output.append(dict(existing_by_key[key]))
                else:
                    output.append(
                        {
                            "probe_type": "native_balance_snapshot" if function_name == "native_balance" else "state_view",
                            "function": function_name,
                            "targets": [],
                        }
                    )
                if len(output) >= 6:
                    break
            if not output:
                output.extend(existing[:6])
            return output[:6]

        payload = {
            "repair_action": "combined_repair",
            "parent_candidate_id": parent_candidate.get("candidate_id"),
            "reuse_parent_prefix": True,
            "changed_fields": [],
            "additional_prefix": list(parent_candidate.get("required_prefix") or []),
            "suffix_actions": list(base_suffix),
            "sender_binding_fix": "",
            "beneficiary_binding_fix": "",
            "receiver_binding_fix": "",
            "amount_binding_fix": "",
            "effect_probe_fix": list(parent_candidate.get("effect_probe_plan") or []),
            "parameter_plan": list(parent_candidate.get("parameter_plan") or []),
            "expected_checkpoint_delta": parent_candidate.get("target_missing_checkpoint"),
            "expected_evidence": list(parent_candidate.get("expected_evidence") or []),
            "reason": "rule_based_failure_repair",
            "confidence": "medium",
            "no_repair": False,
        }
        if failure_signature == "wrong_sender_binding":
            payload["changed_fields"] = ["sender_binding"]
            if self._access_control_profile_requested():
                payload["sender_binding_fix"] = "rotate_existing_senders"
            else:
                payload["sender_binding_fix"] = "keep_same_sender_across_sequence"
            payload["repair_action"] = "fix_sender_binding"
        elif failure_signature in {"wrong_beneficiary_binding", "wrong_receiver_binding"}:
            payload["changed_fields"] = ["amount_binding", "suffix_actions"]
            payload["receiver_binding_fix"] = "mutate_transfer_arguments"
            payload["repair_action"] = "fix_receiver_binding"
        elif failure_signature == "wrong_amount_binding":
            payload["changed_fields"] = ["amount_binding"]
            payload["amount_binding_fix"] = "try_zero_one_max_boundary_values"
            payload["repair_action"] = "fix_amount_binding"
        elif failure_signature == "missing_prestate_prefix":
            payload["changed_fields"] = ["additional_prefix"]
            payload["additional_prefix"] = list(parent_candidate.get("required_prefix") or list(parent_candidate.get("guided_sequence") or [])[:1])
            payload["repair_action"] = "add_missing_prestate_prefix"
        elif failure_signature in {"suffix_not_effective", "effect_function_not_reached", "effect_reached_but_no_state_delta"}:
            payload["changed_fields"] = ["suffix_actions", "effect_probe_fix"]
            payload["suffix_actions"] = list(base_suffix) + (list(base_suffix[-1:]) or [])
            payload["effect_probe_fix"] = list(parent_candidate.get("effect_probe_plan") or [])
            payload["repair_action"] = "strengthen_suffix"
        elif failure_signature == "effect_probe_missing_or_uninformative":
            payload["changed_fields"] = ["effect_probe_fix"]
            payload["effect_probe_fix"] = list(parent_candidate.get("effect_probe_plan") or [])
            payload["repair_action"] = "change_effect_probe"
        else:
            payload["no_repair"] = True
            payload["repair_action"] = "no_repair"
            payload["reason"] = "rule_repair_not_applicable"
        if target_missing == "funds_flow_to_changed_or_unexpected_receiver":
            setters = pick_actions(
                list(ASSET_FLOW_SETTER_HINTS),
                preferred=parent_prefix,
            )
            sinks = pick_actions(
                list(ASSET_FLOW_BUSINESS_HINTS),
                preferred=base_suffix,
            )
            payload["no_repair"] = False
            payload["repair_action"] = "strengthen_suffix"
            payload["changed_fields"] = merge_actions(
                payload.get("changed_fields"),
                ["prefix", "suffix_actions"],
                limit=8,
            )
            payload["additional_prefix"] = merge_actions(parent_prefix, setters[:1], limit=3)
            payload["suffix_actions"] = merge_actions(base_suffix, sinks[:2], limit=3)
            if failure_signature == "wrong_receiver_binding":
                payload["receiver_binding_fix"] = "attacker_or_changed_receiver"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["receiver_binding"], limit=8)
            if failure_signature == "wrong_amount_binding":
                payload["amount_binding_fix"] = "try_zero_and_small_nonzero"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["amount_binding"], limit=8)
            payload["soft_hint_type"] = "prefer_dependency_local_reordering"
            payload["expected_runtime_evidence"] = merge_actions(
                payload.get("expected_runtime_evidence"),
                ["tx_end_success", "receiver_or_fee_state_written", "balance_change", "token_transfer", "unexpected_receiver"],
                limit=8,
            )
            payload["semantic_repair_claim"] = "mixed"
        elif target_missing in {"restriction_active", "balance_storage_or_asset_change_under_restriction"}:
            restriction_setters = pick_scope_actions(
                roles=("prefix_builder", "state_writer"),
                keywords=("pause", "blacklist", "lock", "ban", "block", "enable", "disable", "set", "freeze", "restrict", "deny", "forbid", "transfernewun"),
                related_tokens=("restriction", "paused", "blacklist", "lock", "freeze", "blocked", "disabled", "sellingdisabled", "tradingopen", "newun"),
                preferred=parent_prefix + guided_sequence,
                limit=4,
            )
            if not restriction_setters:
                restriction_setters = pick_actions(
                    ["pause", "blacklist", "lock", "ban", "block", "enable", "disable", "set", "freeze", "restrict", "deny", "forbid", "newun"],
                    preferred=parent_prefix,
                )
            restricted_ops = pick_scope_actions(
                roles=("sink_action", "state_writer", "repeat_action"),
                keywords=("transferfrom", "transfer", "sell", "withdraw", "burn", "claim", "redeem"),
                related_tokens=("token", "balance", "asset_or_external_sink", "allowance", "newun"),
                preferred=base_suffix + guided_sequence[-2:] + [self.affected_function],
                limit=5,
            )
            if not restricted_ops:
                restricted_ops = pick_actions(
                    ["transfer", "transferFrom", "sell", "withdraw", "burn", "claim", "redeem"],
                    preferred=base_suffix,
                )
            affected_name = _text(self.affected_function).strip()
            if affected_name and self.abi_index.has(affected_name):
                restricted_ops = merge_actions([affected_name], restricted_ops, limit=5)
            if self.abi_index.has("transfer"):
                restricted_ops = merge_actions(restricted_ops, ["transfer"], limit=5)
            payload["no_repair"] = False
            if target_missing == "restriction_active":
                prefix_plan = restriction_setters[:1]
                suffix_plan = merge_actions(restriction_setters[:1], restricted_ops[:1], base_suffix[:1], limit=2)
                payload["repair_action"] = "add_missing_prestate_prefix" if prefix_plan else "strengthen_suffix"
                payload["changed_fields"] = merge_actions(
                    payload.get("changed_fields"),
                    ["prefix", "suffix_actions"],
                    limit=8,
                )
                payload["additional_prefix"] = merge_actions(parent_prefix, prefix_plan, limit=3)
                payload["suffix_actions"] = suffix_plan
                payload["expected_runtime_evidence"] = merge_actions(
                    payload.get("expected_runtime_evidence"),
                    ["tx_end_success", "sstore", "restriction_active"],
                    limit=8,
                )
            else:
                prefix_plan = restriction_setters[:1]
                suffix_seed: List[str] = []
                if affected_name and self.abi_index.has(affected_name):
                    suffix_seed.append(affected_name)
                if self.abi_index.has("transfer") and affected_name.lower() != "transfer":
                    suffix_seed.append("transfer")
                suffix_plan = merge_actions(suffix_seed, restricted_ops[:3], base_suffix, limit=3)
                payload["repair_action"] = "add_missing_prestate_prefix" if prefix_plan else "strengthen_suffix"
                payload["changed_fields"] = merge_actions(
                    payload.get("changed_fields"),
                    ["prefix", "suffix_actions"],
                    limit=8,
                )
                prefix_extra: List[str] = []
                if any(_text(item).strip().lower() == "transferfrom" for item in suffix_plan) and self.abi_index.has("approve"):
                    prefix_extra.append("approve")
                payload["additional_prefix"] = merge_actions(parent_prefix, prefix_plan, prefix_extra, limit=3)
                payload["suffix_actions"] = suffix_plan
                payload["expected_runtime_evidence"] = merge_actions(
                    payload.get("expected_runtime_evidence"),
                    ["tx_end_success", "sstore", "token_transfer", "restricted_operation_success", "guarded_revert_or_restriction_effect"],
                    limit=8,
                )
                if failure_signature == "wrong_receiver_binding":
                    payload["receiver_binding_fix"] = "attacker_or_changed_receiver"
                    payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["receiver_binding"], limit=8)
            if failure_signature == "wrong_sender_binding":
                payload["sender_binding_fix"] = "keep_same_sender_across_sequence"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["sender_binding"], limit=8)
            if failure_signature == "wrong_amount_binding":
                payload["amount_binding_fix"] = "try_zero_and_small_nonzero"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["amount_binding"], limit=8)
            payload["soft_hint_type"] = "prefer_state_writer_before_frontier"
            payload["semantic_repair_claim"] = "mixed"
        elif target_missing in {"privileged_or_asset_effect_observed", "privileged_effect_after_takeover"}:
            protected = pick_actions(
                ["set", "admin", "owner", "grant", "role", "mint", "withdraw", "upgrade", "pause"],
                preferred=parent_prefix,
            )
            sinks = pick_actions(
                ["mint", "withdraw", "transfer", "upgrade", "execute"],
                preferred=base_suffix,
            )
            payload["no_repair"] = False
            payload["repair_action"] = "strengthen_suffix"
            payload["changed_fields"] = merge_actions(
                payload.get("changed_fields"),
                ["prefix", "suffix_actions"],
                limit=8,
            )
            payload["additional_prefix"] = merge_actions(parent_prefix, protected[:1], limit=3)
            payload["suffix_actions"] = merge_actions(base_suffix, sinks[:2], limit=3)
            if failure_signature == "wrong_sender_binding":
                payload["sender_binding_fix"] = "rotate_existing_senders"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["sender_binding"], limit=8)
            payload["soft_hint_type"] = "prefer_sender_context_variation"
            payload["expected_runtime_evidence"] = merge_actions(
                payload.get("expected_runtime_evidence"),
                ["tx_end_success", "critical_state_written", "state_write_or_storage_delta", "privileged_suffix_effect"],
                limit=8,
            )
            payload["semantic_repair_claim"] = "mixed"
        elif target_missing == "invariant_check_possible":
            invariant_prefix = pick_scope_actions(
                roles=("prefix_builder", "state_writer"),
                keywords=("approve", "deposit", "mint", "stake", "add", "update", "pool"),
                related_tokens=("balance", "allowance", "pool", "supply", "reward", "state_write"),
                preferred=parent_prefix + guided_sequence,
                limit=4,
            )
            invariant_writers = pick_scope_actions(
                roles=("state_writer", "sink_action"),
                keywords=("withdraw", "claim", "harvest", "deposit", "mint", "burn", "update", "massupdate", "redeem"),
                related_tokens=("pool", "supply", "balance", "reward", "debt", "asset_or_external_sink"),
                preferred=base_suffix + guided_sequence[-2:] + [self.affected_function],
                limit=4,
            )
            invariant_repeats = pick_scope_actions(
                roles=("repeat_action",),
                keywords=("withdraw", "claim", "harvest", "deposit", "mint", "burn", "redeem"),
                related_tokens=("repeat_relation", "same_sender", "same_amount", "pool", "reward"),
                preferred=base_suffix + guided_sequence[-2:],
                limit=3,
            )
            invariant_observers = pick_scope_actions(
                roles=("observation",),
                keywords=("balance", "allowance", "pending", "supply", "price", "ratio", "preview", "debt", "pool", "userinfo"),
                related_tokens=("state_view", "native_balance_snapshot", "erc20_balanceof", "erc20_allowance"),
                preferred=[item.get("function") for item in _normalize_probe_plan(parent_candidate.get("effect_probe_plan")) if isinstance(item, dict)],
                limit=6,
            )
            if not invariant_observers:
                invariant_observers = [
                    _text(item.get("function")).strip()
                    for item in _normalize_probe_plan(parent_candidate.get("effect_probe_plan"))
                    if isinstance(item, dict)
                    and any(
                        token in _text(item.get("function")).strip().lower()
                        for token in ("balance", "allowance", "pending", "supply", "pool")
                    )
                ][:6]

            prefix_extra = [name for name in invariant_prefix if name not in parent_prefix][:1]
            suffix_plan: List[str] = []
            for action in base_suffix[:1]:
                function_name = _canonical_action_name(action)
                if not function_name or not self.abi_index.has(function_name):
                    continue
                suffix_plan.append(self.abi_index.by_lower.get(function_name.lower(), function_name))
            for action in invariant_writers[:2]:
                function_name = _canonical_action_name(action)
                if not function_name or not self.abi_index.has(function_name):
                    continue
                canonical = self.abi_index.by_lower.get(function_name.lower(), function_name)
                if not suffix_plan or suffix_plan[-1] != canonical:
                    suffix_plan.append(canonical)
                if len(suffix_plan) >= 3:
                    break
            if invariant_repeats:
                function_name = _canonical_action_name(invariant_repeats[0])
                if function_name and self.abi_index.has(function_name) and len(suffix_plan) < 3:
                    suffix_plan.append(self.abi_index.by_lower.get(function_name.lower(), function_name))
            elif suffix_plan and len(suffix_plan) < 2:
                suffix_plan.append(suffix_plan[-1])
            if not suffix_plan:
                suffix_plan = list(base_suffix) + (list(base_suffix[-1:]) or [])

            payload["no_repair"] = False
            payload["repair_action"] = "combined_repair"
            payload["changed_fields"] = merge_actions(
                payload.get("changed_fields"),
                ["suffix_actions", "effect_probe_fix"],
                limit=8,
            )
            if prefix_extra:
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["prefix"], limit=8)
            payload["additional_prefix"] = merge_actions(parent_prefix, prefix_extra, limit=3)
            payload["suffix_actions"] = suffix_plan[:3]
            payload["effect_probe_fix"] = build_probe_plan(
                invariant_observers,
                fallback=parent_candidate.get("effect_probe_plan") or [],
            )
            if invariant_repeats or any(
                "same_sender" in item.lower() for item in _clean_list(repair_scope.get("must_preserve_relations"))
            ):
                payload["sender_binding_fix"] = "keep_same_sender_across_sequence"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["sender_binding"], limit=8)
            if failure_signature in {"effect_reached_but_no_state_delta", "wrong_amount_binding"}:
                payload["amount_binding_fix"] = "try_zero_one_max_boundary_values"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["amount_binding"], limit=8)
            payload["soft_hint_type"] = "prefer_repeat_suffix_preserving_prefix"
            payload["expected_runtime_evidence"] = merge_actions(
                payload.get("expected_runtime_evidence"),
                ["tx_end_success", "state_write_or_storage_delta", "asset_or_sink_effect"],
                limit=8,
            )
            payload["semantic_repair_claim"] = "mixed"
        elif target_missing == "abnormal_effect_observed":
            setters = pick_actions(
                ["set", "update", "config", "fee", "rate", "limit", "price", "slippage"],
                preferred=parent_prefix,
            )
            sinks = pick_actions(
                ["swap", "claim", "withdraw", "transfer", "buy", "sell", "redeem"],
                preferred=base_suffix,
            )
            payload["no_repair"] = False
            payload["repair_action"] = "strengthen_suffix"
            payload["changed_fields"] = merge_actions(
                payload.get("changed_fields"),
                ["prefix", "suffix_actions"],
                limit=8,
            )
            payload["additional_prefix"] = merge_actions(parent_prefix, setters[:1], limit=3)
            payload["suffix_actions"] = merge_actions(base_suffix, sinks[:2], limit=3)
            if failure_signature == "wrong_amount_binding":
                payload["amount_binding_fix"] = "try_zero_one_max_boundary_values"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["amount_binding"], limit=8)
            payload["soft_hint_type"] = "prefer_balance_or_allowance_setup"
            payload["expected_runtime_evidence"] = merge_actions(
                payload.get("expected_runtime_evidence"),
                ["tx_end_success", "sstore", "balance_change", "external_call"],
                limit=8,
            )
            payload["semantic_repair_claim"] = "mixed"
        elif target_missing == "state_or_asset_effect_after_failed_call":
            sinks = pick_actions(
                ["call", "transfer", "withdraw", "execute", "send"],
                preferred=base_suffix,
            )
            payload["no_repair"] = False
            payload["repair_action"] = "strengthen_suffix"
            payload["changed_fields"] = merge_actions(
                payload.get("changed_fields"),
                ["suffix_actions"],
                limit=8,
            )
            payload["suffix_actions"] = merge_actions(base_suffix, sinks[:2], base_suffix[-1:], limit=3)
            if failure_signature == "wrong_amount_binding":
                payload["amount_binding_fix"] = "try_zero_and_small_nonzero"
                payload["changed_fields"] = merge_actions(payload.get("changed_fields"), ["amount_binding"], limit=8)
            payload["soft_hint_type"] = "prefer_repeat_suffix_preserving_prefix"
            payload["expected_runtime_evidence"] = merge_actions(
                payload.get("expected_runtime_evidence"),
                ["external_call_failed", "tx_end_success", "sstore", "balance_change"],
                limit=8,
            )
            payload["semantic_repair_claim"] = "mixed"
        if self._access_control_profile_requested() and not payload.get("no_repair"):
            role_prefix = [
                name
                for name in self._access_control_role_writer_candidates({})
                if name not in _clean_list(parent_candidate.get("required_prefix"))
            ][:1]
            parent_prefix = _clean_list(parent_candidate.get("required_prefix"))
            preserved_prefix = self._access_control_replay_prefix(
                parent_prefix or _clean_list(parent_candidate.get("guided_sequence"))[:2],
                role_prefix,
            )
            sensitive_suffixes = self._access_control_sensitive_suffix_candidates({})
            suffix_actions = _clean_list(payload.get("suffix_actions")) or _clean_list(parent_candidate.get("suffix_actions"))
            if sensitive_suffixes:
                suffix_actions = _dedup_text(suffix_actions + sensitive_suffixes[:1], limit=2)
            if preserved_prefix:
                payload["additional_prefix"] = preserved_prefix
                payload["reuse_parent_prefix"] = True
                payload["changed_fields"] = _dedup_text(_normalize_changed_fields(payload.get("changed_fields")) + ["prefix"])
            if suffix_actions:
                payload["suffix_actions"] = suffix_actions
                payload["changed_fields"] = _dedup_text(_normalize_changed_fields(payload.get("changed_fields")) + ["suffix_actions"])
            if not payload.get("sender_binding_fix"):
                payload["sender_binding_fix"] = "rotate_existing_senders" if len(self.accounts) >= 2 else "try_owner_or_deployer_or_previous_success_sender"
                payload["changed_fields"] = _dedup_text(_normalize_changed_fields(payload.get("changed_fields")) + ["sender_binding"])
            payload["repair_action"] = "add_missing_prestate_prefix" if role_prefix or preserved_prefix else payload.get("repair_action")
            payload["soft_hint_type"] = "prefer_state_writer_before_frontier"
            payload["expected_runtime_evidence"] = _dedup_text(
                _normalize_text_list(payload.get("expected_runtime_evidence"))
                + _normalize_text_list(payload.get("expected_evidence"))
                + ["tx_end_success", "state_write_or_storage_delta", "privileged_suffix_effect"]
            )
            payload["semantic_repair_claim"] = "mixed"
            payload["targeted_patch_id"] = "TARGETED_ACCESS_CONTROL_NONE34_RESCUE_PATCH"
            if self._access_control_profile_requested():
                self.profile_effect_summary["unauthorized_revert_repair_candidates"] = int(
                    self.profile_effect_summary.get("unauthorized_revert_repair_candidates", 0) + 1
                )
                guided_sequence = _clean_list(parent_candidate.get("guided_sequence"))
                suffix_target = _clean_list(payload.get("suffix_actions")) or guided_sequence[-1:]
                transfer_names = self.abi_index.select_by_keywords(["transferownership"])
                accept_names = self.abi_index.select_by_keywords(["acceptownership"])
                role_setup_names = self.abi_index.select_by_keywords(["grantrole", "addrole", "addminter", "setoperator", "setadmin"])
                allowlist_names = self.abi_index.select_by_keywords(["whitelist", "allowlist", "allow", "include"])
                pause_names = self.abi_index.select_by_keywords(["pause"])
                unpause_names = self.abi_index.select_by_keywords(["unpause"])
                profile_prefix: List[str] = []
                if transfer_names:
                    profile_prefix.append(transfer_names[0])
                    if accept_names:
                        profile_prefix.append(accept_names[0])
                elif role_setup_names:
                    profile_prefix.append(role_setup_names[0])
                elif allowlist_names:
                    profile_prefix.append(allowlist_names[0])
                elif pause_names:
                    profile_prefix.append(pause_names[0])
                    if unpause_names:
                        profile_prefix.append(unpause_names[0])
                if profile_prefix:
                    payload["additional_prefix"] = self._access_control_replay_prefix(
                        _clean_list(parent_candidate.get("required_prefix")) or profile_prefix,
                        profile_prefix,
                    )
                    payload["changed_fields"] = _dedup_text(_normalize_changed_fields(payload.get("changed_fields")) + ["prefix"])
                if suffix_target:
                    payload["suffix_actions"] = _dedup_text(list(suffix_target), limit=2)
                payload["sender_binding_fix"] = "try_owner_or_deployer_or_previous_success_sender"
                payload["expected_runtime_evidence"] = _dedup_text(
                    _normalize_text_list(payload.get("expected_runtime_evidence"))
                    + [
                        "caller_changed_before_sensitive_call",
                        "privileged_state_storage_changed",
                        "setup_then_sensitive_success",
                        "new_storage_frontier",
                    ]
                )
                payload["guidance_profile"] = "access_control_state_push"
        if not payload["changed_fields"] and not payload["no_repair"]:
            payload["no_repair"] = True
            payload["repair_action"] = "no_repair"
            payload["reason"] = "rule_repair_no_changed_fields"
        return payload

    def _build_agent_effect_repair_candidates(
        self,
        next_missing: str,
        observed: Dict[str, Any],
        tfi_candidates: Sequence[Dict[str, Any]],
        recent_failed: Sequence[Dict[str, Any]],
        prestate_plans: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self.llm_provider or not getattr(self.llm_provider, "is_available", lambda: False)():
            self.agent_repair_summary["provider_unavailable"] += 1
            return []
        if not getattr(self.llm_provider, "is_configured", lambda: False)():
            self.agent_repair_summary["provider_not_configured"] += 1
            return []
        prompt = self._agent_effect_repair_prompt(next_missing, observed, tfi_candidates, recent_failed, prestate_plans)
        started = time.time()
        self.llm_call_count += 1
        try:
            payload = self.llm_provider.generate_json(prompt)
        except Exception as exc:
            ended = time.time()
            duration = max(0.0, ended - started)
            self.llm_error_count += 1
            self.llm_interaction_time_total_sec += duration
            self.agent_repair_summary["llm_error"] += 1
            self.llm_interaction_events.append(
                {
                    "stage": "effect_forcing_agent_repair",
                    "started_at": started,
                    "ended_at": ended,
                    "error": str(exc),
                    "target_missing_checkpoint": next_missing,
                    "runtime_pattern_family": self.runtime_pattern_family,
                    "duration_sec": round(duration, 6),
                }
            )
            if self.phase_cost_env is not None:
                try:
                    record_phase_event(self.phase_cost_env, self.llm_interaction_events[-1])
                except Exception:
                    pass
            return []
        ended = time.time()
        duration = max(0.0, ended - started)
        self.llm_success_count += 1
        self.llm_interaction_time_total_sec += duration
        metadata = {}
        try:
            metadata = self.llm_provider.get_last_response_metadata()
        except Exception:
            metadata = {}
        event_payload = {
            "stage": "effect_forcing_agent_repair",
            "started_at": started,
            "ended_at": ended,
            "target_missing_checkpoint": next_missing,
            "runtime_pattern_family": self.runtime_pattern_family,
            "duration_sec": round(duration, 6),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }
        if isinstance(metadata, dict):
            event_payload.update(
                {
                    "model": metadata.get("model"),
                    "usage": metadata.get("usage"),
                    "cost_usd": metadata.get("cost_usd"),
                    "cost_estimation_mode": metadata.get("cost_estimation_mode"),
                }
            )
        self.llm_interaction_events.append(event_payload)
        self.llm_interaction_events = self.llm_interaction_events[-256:]
        if self.phase_cost_env is not None:
            try:
                record_phase_event(self.phase_cost_env, event_payload)
            except Exception:
                pass
        if not isinstance(payload, dict):
            self.agent_repair_summary["output_not_json_object"] += 1
            return []
        if payload.get("no_candidate"):
            self.agent_repair_summary["no_candidate"] += 1
            return []
        required_prefix = _clean_list(payload.get("required_prefix"))
        suffix_actions = _clean_list(payload.get("suffix_actions")) or _clean_list(payload.get("sequence_skeleton"))
        sequence = self._merge_prefix_suffix(required_prefix, suffix_actions)
        mapping = ACTION_MAPPING.get(self.runtime_pattern_family, {}).get(next_missing, {"goal": "agent effect forcing repair"})
        candidate = self._new_candidate(
            source="agent",
            next_missing=next_missing,
            mapping=mapping,
            sequence=sequence,
            observed=observed,
            build_reason="effect_forcing_agent_repair",
            raw_agent_payload=payload,
        )
        self._mark_modules(candidate, ["agent_repair"], "agent_effect_repair")
        if required_prefix:
            candidate["required_prefix"] = required_prefix
            self._mark_modules(candidate, ["prestate_builder"], "agent_effect_repair_prefix")
        if suffix_actions:
            candidate["suffix_actions"] = suffix_actions
            self._mark_modules(candidate, ["suffix_mutation"], "agent_effect_repair_suffix")
        if bool(payload.get("reuse_prefix")):
            candidate["prefix_replay"] = True
            self._mark_modules(candidate, ["prefix_replay"], "agent_effect_repair_prefix_replay")
        candidate["agent_repair_payload"] = payload
        candidate["confidence"] = _safe_float(payload.get("confidence"), candidate.get("confidence", 0.65))
        candidate = self._maybe_append_effect_probe(candidate, observed, probe_plan=payload.get("effect_probe_plan"))
        candidate = self._finalize_candidate_materialization(candidate, observed)
        self.agent_repair_summary["candidates_built"] += 1
        return [candidate]

    def _novelty_signature(self, sequence: Sequence[str], checkpoint: str, mapping: Dict[str, Any]) -> str:
        payload = {
            "checkpoint": checkpoint,
            "sequence": list(sequence),
            "sender": mapping.get("sender_policy") or "",
            "value": mapping.get("value_policy") or "",
            "args": mapping.get("argument_policy") or "",
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)

    def _filter_candidates(self, candidates: Sequence[Dict[str, Any]], next_missing: str) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        if next_missing in self.progress.covered:
            self.filter_counts["target_checkpoint_already_covered"] += len(candidates)
            return []
        working = [item for item in candidates if isinstance(item, dict)]
        if self.enable_strong_useful_scheduling and working:
            working = self._prioritize_candidates(working)
        working = self._apply_access_control_profile_budget(working)
        for candidate in working:
            if _text(candidate.get("materialization_status")).strip() == "rejected_as_unresolved":
                self.filter_counts["materialization_rejected"] += 1
                continue
            if _safe_float(candidate.get("confidence"), 0.0) < self.confidence_threshold:
                self.filter_counts["low_confidence"] += 1
                continue
            signature = _text(candidate.get("novelty_signature")).strip()
            if bool(candidate.get("failure_repair_candidate")):
                signature = _text(candidate.get("repair_signature")).strip() or signature
            if not signature:
                self.filter_counts["missing_novelty_signature"] += 1
                continue
            if signature in self.recent_signatures:
                self.filter_counts["duplicate_recent_signature"] += 1
                if bool(candidate.get("failure_repair_candidate")):
                    _append_jsonl(
                        self.repair_queue_events_path,
                        {
                            "repair_candidate_id": candidate.get("candidate_id"),
                            "parent_candidate_id": candidate.get("parent_candidate_id"),
                            "repair_attempt_index": candidate.get("repair_attempt_index"),
                            "queued_at_global_tx": None,
                            "executed_at_global_tx": None,
                            "queue_delay": None,
                            "skipped_reason": "duplicate_recent_signature",
                            "cooldown_reason": "",
                            "group_name": _text(candidate.get("group_name")).strip(),
                            "failure_signature": candidate.get("failure_signature"),
                            "repair_action": candidate.get("repair_action"),
                            "sequence_signature": candidate.get("sequence_signature"),
                            "binding_signature": candidate.get("binding_signature"),
                            "repair_signature": candidate.get("repair_signature"),
                            "is_binding_only_repair": bool(candidate.get("repair_binding_only")),
                            "duplicate_reason": "duplicate_recent_signature",
                            "accepted_by_queue": False,
                        },
                    )
                continue
            self.recent_signatures.append(signature)
            output.append(candidate)
            self.source_counts[_text(candidate.get("source")).strip() or "unknown"] += 1
            if len(output) >= self.max_per_round:
                break
        self.filter_counts["accepted"] += len(output)
        return output

    def _prioritize_candidates(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prioritized: List[Dict[str, Any]] = []
        for candidate in candidates:
            candidate = dict(candidate)
            if self.enable_effect_forcing_v3:
                self._mark_modules(candidate, ["strong_useful_scheduling"], "strong_useful_scheduling")
            hint = 0
            if self.rq1_rescue_mode:
                hint += _safe_int(candidate.get("rq1_rescue_priority"), 0)
                if _text(candidate.get("target_missing_checkpoint")).strip() == _text(self.progress.next_missing()).strip():
                    hint += 10
            if bool(candidate.get("sink_touched")):
                hint += 2
            if bool(candidate.get("prefix_replay")):
                hint += 2
            if bool(candidate.get("access_control_rescue_candidate")):
                hint += 3
            if bool(candidate.get("access_control_prefix_preserving")):
                hint += 2
            if _text(candidate.get("semantic_delta_type")).strip() in {"prefix_changed", "suffix_changed", "concrete_tx_changed"}:
                hint += 2
            if bool(candidate.get("observation_only_repair")) or _text(candidate.get("semantic_delta_type")).strip() == "probe_only":
                hint -= 3
            if bool(candidate.get("probe_appended")):
                hint += 1
            if candidate.get("source") == "agent":
                hint += 1
            if self._access_control_profile_requested() and self._is_access_control_profile_candidate(candidate):
                sequence = " ".join(_clean_list(candidate.get("guided_sequence"))).lower()
                profile_bonus = 0
                if self._access_control_profile_boost_allowed(candidate):
                    if "transferownership" in sequence or "acceptownership" in sequence:
                        profile_bonus += 2
                    if any(token in sequence for token in ("grantrole", "addrole", "addminter", "setoperator", "setadmin")):
                        profile_bonus += 2
                    if any(token in sequence for token in ("whitelist", "allowlist", "allow")):
                        profile_bonus += 1
                    if any(token in sequence for token in ("pause", "unpause")):
                        profile_bonus += 1
                    if _text(candidate.get("sender_policy")).strip().lower() == "try_owner_or_deployer_or_previous_success_sender":
                        profile_bonus += 1
                    if bool(candidate.get("access_control_profile_active")):
                        profile_bonus += 1
                    if bool(candidate.get("failure_repair_candidate")) and self._access_control_runtime_evidence_seen():
                        profile_bonus += 1
                    if bool(candidate.get("prefix_replay")) and self._access_control_productive_signal_seen():
                        profile_bonus += 1
                elif bool(candidate.get("access_control_rescue_candidate")):
                    profile_bonus -= 2
                elif bool(candidate.get("access_control_profile_active")):
                    profile_bonus -= 1
                hint += profile_bonus
                if profile_bonus > 0:
                    self.profile_effect_summary["profile_boosted_candidate_count"] = int(
                        self.profile_effect_summary.get("profile_boosted_candidate_count", 0) + 1
                    )
            candidate["strong_useful_priority_hint"] = hint
            prioritized.append(candidate)
            self.strong_scheduling_summary["candidates_considered"] += 1
        prioritized.sort(
            key=lambda item: (
                -_safe_int(item.get("strong_useful_priority_hint"), 0),
                -_safe_float(item.get("confidence"), 0.0),
                _text(item.get("candidate_id")).strip(),
            )
        )
        self.strong_scheduling_summary["candidates_reordered"] += len(prioritized)
        return prioritized

    def _write_filter_summary(self) -> None:
        _write_json(
            self.filter_summary_path,
            {
                "filter_counts": dict(self.filter_counts),
                "dedup_window": int(self.dedup_window),
                "recent_signature_count": len(self.recent_signatures),
                "confidence_threshold": float(self.confidence_threshold),
            },
        )

    def record_injection(
        self,
        candidates: Sequence[Dict[str, Any]],
        injected_count: int,
        round_id: int,
        injected_candidate_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.injected_count += max(0, int(injected_count or 0))
        precise_ids = [_text(item).strip() for item in list(injected_candidate_ids or []) if _text(item).strip()]
        precise_id_set = set(precise_ids)
        precise_rank_map: Dict[str, int] = {}
        for rank, candidate_id in enumerate(precise_ids, start=1):
            if candidate_id not in precise_rank_map:
                precise_rank_map[candidate_id] = rank
        if precise_id_set:
            self.candidate_budget_used += sum(
                1
                for candidate in candidates
                if _text(candidate.get("candidate_id")).strip() in precise_id_set
            )
        else:
            self.candidate_budget_used += min(len(candidates), max(0, int(injected_count or 0)))
        before_level = self.progress.best_level
        for index, candidate in enumerate(candidates, start=1):
            candidate_id = _text(candidate.get("candidate_id")).strip()
            injected = candidate_id in precise_id_set if precise_id_set else index <= int(injected_count or 0)
            injected_rank = precise_rank_map.get(candidate_id) if precise_id_set else (index if injected else None)
            if bool(candidate.get("failure_repair_candidate")):
                if injected:
                    self._bump_pipeline_counter("repair_admitted", 1)
                    self._bump_pipeline_counter("repair_selected", 1)
                    self.repair_lifecycle.mark_stage(
                        candidate.get("candidate_id"),
                        stage="admitted",
                        extra={"round_id": int(round_id), "injected_rank": injected_rank},
                        seed={
                            "bench_id": candidate.get("bench_id") or self._runtime_bench_label(),
                            "contract": candidate.get("contract_name") or self._runtime_contract_label(),
                            "source_trigger_id": candidate.get("source_trigger_id"),
                            "source_parent_id": candidate.get("source_parent_id") or candidate.get("parent_candidate_id"),
                            "source_scope_id": candidate.get("source_scope_id"),
                            "source_candidate_kind": candidate.get("source_candidate_kind") or candidate.get("candidate_type"),
                        },
                    )
                    self.repair_lifecycle.mark_stage(
                        candidate.get("candidate_id"),
                        stage="selected",
                        extra={"round_id": int(round_id), "selected_rank": index},
                    )
                    self.register_selected_repair_candidate(
                        candidate,
                        round_id=int(round_id),
                        selected_rank=index,
                        injected_rank=int(injected_rank or index),
                    )
                else:
                    self._bump_pipeline_counter("repair_dropped", 1)
                    self.repair_lifecycle.mark_dropped(
                        candidate.get("candidate_id"),
                        reason="queued_but_not_admitted_this_round",
                        extra={"round_id": int(round_id), "injected_rank": injected_rank},
                    )
            if injected and before_level not in {"L2", "L3"}:
                self.candidate_executed_before_l2 += 1
            if injected and before_level != "L3":
                self.candidate_executed_before_l3 += 1
            if injected and before_level == "L3":
                self.candidate_delayed_after_l3 += 1
            if injected:
                self.executed_count += 1
                self.inflight_candidates[_text(candidate.get("candidate_id")).strip()] = {
                    "candidate": dict(candidate),
                    "before_best_level": before_level,
                    "before_strict_vspc": self.progress.strict_vspc(),
                    "injected_round": int(round_id),
                }
            _append_jsonl(
                self.delta_path,
                self._delta_record(
                    candidate,
                    injected=injected,
                    executed=False,
                    round_id=round_id,
                    before_level=before_level,
                    stage="injected" if injected else "injection_skipped",
                ),
            )
        self._write_summary()

    def _remember_failure_signature_for_candidate(
        self,
        candidate_id: str,
        signature: Dict[str, Any],
        *,
        replace_existing: bool = False,
    ) -> int:
        clean_id = _text(candidate_id).strip()
        if not clean_id or not isinstance(signature, dict):
            return 0
        if clean_id in self.failure_signatures_by_candidate and not replace_existing:
            return 0
        self.failure_signatures_by_candidate[clean_id] = dict(signature)
        _append_jsonl(self.failure_signature_log_path, signature)
        self.failure_signature_summary[_text(signature.get("failure_signature")).strip() or "unknown"] += 1
        self._bump_pipeline_counter("failure_signature_created", 1)
        if self._allowed_repair_actions_for_signature(_text(signature.get("failure_signature")).strip()):
            self.pending_failure_repair_signature_ids.add(clean_id)
            return 1
        return 0

    def _synthesize_unlinked_inflight_failure_signatures(
        self,
        *,
        round_id: int,
        linked_candidate_ids: Sequence[str],
    ) -> int:
        if not self.enable_failure_repair_pipeline or not self.inflight_candidates:
            return 0
        linked = set(_normalize_text_list(linked_candidate_ids))
        new_failure_signature_count = 0
        effect_modules = {
            "prestate_builder",
            "prefix_replay",
            "suffix_mutation",
            "effect_probe",
            "strong_useful_scheduling",
        }
        for candidate_id, payload in list(self.inflight_candidates.items()):
            clean_id = _text(candidate_id).strip()
            if not clean_id:
                continue
            if clean_id in linked or clean_id in self.observed_execution_ids:
                continue
            if clean_id in self.failure_signatures_by_candidate:
                continue
            injected_round = _safe_int(payload.get("injected_round"), 0)
            if int(round_id) <= injected_round:
                continue
            candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
            if not isinstance(candidate, dict) or bool(candidate.get("failure_repair_candidate")):
                continue
            target_checkpoint = _text(candidate.get("target_missing_checkpoint")).strip()
            if not target_checkpoint or target_checkpoint in self.progress.covered:
                continue
            modules = set(_clean_list(candidate.get("modules_used")))
            source = _text(candidate.get("source")).strip()
            if source != "tfi" and not modules.intersection(effect_modules):
                continue
            self._bump_pipeline_counter("no_delta_candidate_detected", 1)
            self._bump_pipeline_counter("failure_signature_attempted", 1)
            signature = self._failure_signature_for_candidate(
                candidate,
                [],
                before_level=_text(payload.get("before_best_level")).strip() or "none",
                after_level=self.progress.best_level,
                before_strict_vspc=_safe_float(payload.get("before_strict_vspc"), 0.0),
                after_strict_vspc=self.progress.strict_vspc(),
            )
            evidence = _dedup_text(
                _normalize_text_list(signature.get("evidence"))
                + [
                    "candidate_injected_but_no_candidate_id_linked_oracle_event",
                    "synthetic_no_delta_after_injection_grace",
                ]
            )
            signature["failure_signature_id"] = "{}:synthetic_no_link:{}".format(clean_id, int(round_id))
            signature["execution_status"] = "injected_without_linked_oracle_event"
            signature["tx_success"] = False
            signature["synthetic_failure_signature"] = True
            signature["synthetic_reason"] = "no_candidate_id_linked_oracle_event_after_injection_grace"
            signature["evidence"] = evidence
            signature["evidence_summary"] = ",".join(evidence[:4])
            new_failure_signature_count += self._remember_failure_signature_for_candidate(clean_id, signature)
            self.recent_failed_checkpoint_candidates.append(
                {
                    "candidate_id": clean_id,
                    "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
                    "failure_signature": signature.get("failure_signature"),
                    "evidence": list(signature.get("evidence") or []),
                    "sequence": list(candidate.get("guided_sequence") or []),
                    "synthetic_failure_signature": True,
                }
            )
        return int(new_failure_signature_count)

    def _observe_inflight_execution(self, new_events: Sequence[Dict[str, Any]], round_id: int) -> int:
        if not self.inflight_candidates:
            return 0
        new_failure_signature_count = 0
        by_candidate: Dict[str, List[Dict[str, Any]]] = {}
        for event in new_events:
            if not isinstance(event, dict):
                continue
            candidate_id = _text(event.get("candidate_id")).strip()
            if not candidate_id or candidate_id not in self.inflight_candidates:
                continue
            by_candidate.setdefault(candidate_id, []).append(event)
        for candidate_id, events in by_candidate.items():
            if candidate_id in self.observed_execution_ids:
                continue
            payload = self.inflight_candidates.get(candidate_id) or {}
            candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
            if self._access_control_profile_requested():
                observed_events_context = self._observed_context(events)
                if (
                    bool(candidate.get("access_control_profile_active"))
                    and not any(bool(event.get("reverted")) or _text(event.get("event_type")).strip() == "revert" for event in events if isinstance(event, dict))
                    and any(name in set(_clean_list(candidate.get("guided_sequence"))) for name in self._access_control_role_writer_candidates({})[:8])
                ):
                    self.profile_effect_summary["setup_then_sensitive_success_count"] = int(
                        self.profile_effect_summary.get("setup_then_sensitive_success_count", 0) + 1
                    )
                self.profile_effect_summary["ownership_or_role_event_seen_count"] = int(
                    self.profile_effect_summary.get("ownership_or_role_event_seen_count", 0)
                    + int(observed_events_context.get("ownership_or_role_event_seen_count", 0))
                )
                self.profile_effect_summary["access_control_storage_write_seen_count"] = int(
                    self.profile_effect_summary.get("access_control_storage_write_seen_count", 0)
                    + int(observed_events_context.get("access_control_storage_write_seen_count", 0))
                )
            self._bump_pipeline_counter("effect_forcing_candidate_executed", 1)
            before_level = _text(payload.get("before_best_level")).strip() or "none"
            record = self._delta_record(
                candidate,
                injected=True,
                executed=True,
                round_id=round_id,
                before_level=before_level,
                stage="observed_execution",
                related_events=events,
                before_strict_vspc=payload.get("before_strict_vspc"),
            )
            _append_jsonl(self.delta_path, record)
            if bool(candidate.get("failure_repair_candidate")):
                self._bump_pipeline_counter("repair_oracle_events_linked", 1)
                self._bump_pipeline_counter("repair_attributed", 1)
                if not bool(candidate.get("observation_only_repair")):
                    self._bump_pipeline_counter("semantic_repair_executed", 1)
                    if not bool(candidate.get("dependency_grounded_scope", {}).get("no_safe_semantic_scope")):
                        self._bump_pipeline_counter("grounded_semantic_repair_executed", 1)
                self.repair_lifecycle.mark_stage(
                    candidate.get("candidate_id"),
                    stage="executed",
                    extra={
                        "round_id": int(round_id),
                        "related_event_count": len(events),
                    },
                )
                self.repair_lifecycle.mark_stage(
                    candidate.get("candidate_id"),
                    stage="attributed",
                    extra={
                        "round_id": int(round_id),
                        "checkpoint_delta": record.get("checkpoint_delta"),
                        "strict_vspc_delta": record.get("strict_vspc_delta"),
                        "oracle_level_delta": record.get("oracle_level_delta"),
                    },
                )
                self.repair_lifecycle.mark_oracle_event(
                    candidate.get("candidate_id"),
                    extra={
                        "round_id": int(round_id),
                        "related_event_count": len(events),
                    },
                )
                if not bool(candidate.get("observation_only_repair")) and not bool(candidate.get("dependency_grounded_scope", {}).get("no_safe_semantic_scope")):
                    self._bump_pipeline_counter("grounded_semantic_repair_attributed", 1)
                self._update_failure_repair_interaction(
                    candidate.get("candidate_id"),
                    execution_result={
                        "executed": True,
                        "stage": "observed_execution",
                        "before_best_level": record.get("before_best_level"),
                        "after_best_level": record.get("after_best_level"),
                        "before_strict_vspc": record.get("before_strict_vspc"),
                        "after_strict_vspc": record.get("after_strict_vspc"),
                        "related_event_count": record.get("related_event_count"),
                        "related_event_type_counts": record.get("related_event_type_counts"),
                    },
                    delta_attribution={
                        "checkpoint_delta": record.get("checkpoint_delta"),
                        "strict_vspc_delta": record.get("strict_vspc_delta"),
                        "oracle_level_delta": record.get("oracle_level_delta"),
                        "useful_strength": record.get("useful_strength"),
                        "l3_confirmed_by_repair": record.get("l3_confirmed_by_candidate"),
                    },
                )
                _append_jsonl(self.failure_repair_delta_path, dict(record, parent_candidate_id=candidate.get("parent_candidate_id"), failure_signature=candidate.get("failure_signature"), repair_action=candidate.get("repair_action"), repair_candidate_id=candidate.get("candidate_id"), direct_or_delayed="direct", materialization_status=candidate.get("materialization_status_final"), executed=True, l3_confirmed_by_repair=bool(record.get("l3_confirmed_by_candidate"))))
                if _text(record.get("useful_strength")).strip() == "medium":
                    self.failure_repair_summary["repair_to_medium_delta"] += 1
                    self._bump_pipeline_counter("repair_direct_delta", 1)
                if _text(record.get("useful_strength")).strip() == "strong":
                    self.failure_repair_summary["repair_to_strong_delta"] += 1
                    self._bump_pipeline_counter("repair_direct_delta", 1)
                if not bool(candidate.get("observation_only_repair")) and _safe_int(record.get("checkpoint_delta"), 0) > 0:
                    self._bump_pipeline_counter("semantic_repair_to_checkpoint_delta", 1)
                    if not bool(candidate.get("dependency_grounded_scope", {}).get("no_safe_semantic_scope")):
                        self._bump_pipeline_counter("grounded_semantic_repair_to_checkpoint_delta", 1)
                if not bool(candidate.get("observation_only_repair")) and _safe_float(record.get("strict_vspc_delta"), 0.0) > 0.0:
                    self._bump_pipeline_counter("grounded_semantic_repair_to_vspc_delta", 1)
                if _text(record.get("after_best_level")).strip() == "L2" and _text(record.get("before_best_level")).strip() != "L2":
                    self.failure_repair_summary["repair_to_l2"] += 1
                    self._bump_pipeline_counter("repair_to_l2", 1)
                    if not bool(candidate.get("observation_only_repair")):
                        self._bump_pipeline_counter("semantic_repair_to_l2", 1)
                        self._bump_pipeline_counter("semantic_repair_to_L2", 1)
                        if not bool(candidate.get("dependency_grounded_scope", {}).get("no_safe_semantic_scope")):
                            self._bump_pipeline_counter("grounded_semantic_repair_to_L2", 1)
                if _text(record.get("after_best_level")).strip() == "L1" and _text(record.get("before_best_level")).strip() not in {"L1", "L2", "L3"}:
                    if not bool(candidate.get("observation_only_repair")):
                        self._bump_pipeline_counter("semantic_repair_to_l1", 1)
                        self._bump_pipeline_counter("semantic_repair_to_L1", 1)
                        if not bool(candidate.get("dependency_grounded_scope", {}).get("no_safe_semantic_scope")):
                            self._bump_pipeline_counter("grounded_semantic_repair_to_L1", 1)
                if bool(record.get("l3_confirmed_by_candidate")):
                    self.failure_repair_summary["repair_to_l3"] += 1
                    self._bump_pipeline_counter("repair_to_l3", 1)
                    if not bool(candidate.get("observation_only_repair")):
                        self._bump_pipeline_counter("semantic_repair_to_l3", 1)
                        self._bump_pipeline_counter("semantic_repair_to_L3", 1)
                        if not bool(candidate.get("dependency_grounded_scope", {}).get("no_safe_semantic_scope")):
                            self._bump_pipeline_counter("grounded_semantic_repair_to_L3", 1)
            else:
                if _text(record.get("useful_strength")).strip() in {"none", "weak"} or (
                    _text(record.get("before_best_level")).strip() == _text(record.get("after_best_level")).strip()
                    and _safe_float(record.get("strict_vspc_delta"), 0.0) == 0.0
                    and _safe_int(record.get("checkpoint_delta"), 0) == 0
                    and _safe_int(record.get("oracle_level_delta"), 0) == 0
                ):
                    self._bump_pipeline_counter("no_delta_candidate_detected", 1)
                self._bump_pipeline_counter("failure_signature_attempted", 1)
                signature = self._failure_signature_for_candidate(
                    candidate,
                    events,
                    before_level=before_level,
                    after_level=_text(record.get("after_best_level")).strip() or before_level,
                    before_strict_vspc=_safe_float(payload.get("before_strict_vspc"), 0.0),
                    after_strict_vspc=_safe_float(record.get("after_strict_vspc"), 0.0),
                )
                new_failure_signature_count += self._remember_failure_signature_for_candidate(
                    candidate_id,
                    signature,
                    replace_existing=True,
                )
                if _text(record.get("useful_strength")).strip() == "weak":
                    self.recent_failed_checkpoint_candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
                            "failure_signature": signature.get("failure_signature"),
                            "evidence": list(signature.get("evidence") or []),
                            "sequence": list(candidate.get("guided_sequence") or []),
                        }
                    )
            self.observed_execution_ids.add(candidate_id)
        new_failure_signature_count += self._synthesize_unlinked_inflight_failure_signatures(
            round_id=int(round_id),
            linked_candidate_ids=list(by_candidate.keys()),
        )
        self._write_summary()
        return int(new_failure_signature_count)

    def _delta_record(
        self,
        candidate: Dict[str, Any],
        *,
        injected: bool,
        executed: bool,
        round_id: int,
        before_level: str,
        stage: str,
        related_events: Optional[Sequence[Dict[str, Any]]] = None,
        before_strict_vspc: Any = None,
    ) -> Dict[str, Any]:
        after_level = self.progress.best_level
        before_strict = _safe_float(before_strict_vspc, self.progress.strict_vspc())
        after_strict = self.progress.strict_vspc()
        checkpoint_delta = 0
        oracle_delta = 0 if before_level == after_level else 1
        related_events = list(related_events or [])
        event_types = Counter(_text(event.get("event_type")).strip() for event in related_events if isinstance(event, dict))
        if executed and related_events and candidate.get("target_missing_checkpoint") in self.progress.covered:
            checkpoint_delta = 1
        if executed and checkpoint_delta > 0:
            self.checkpoint_delta_count += 1
        if executed and oracle_delta > 0:
            self.oracle_level_delta_count += 1
        useful_strength = "weak"
        if executed and (checkpoint_delta > 0 or after_strict > before_strict):
            useful_strength = "medium"
            self.medium_useful_count += 1
        if executed and (oracle_delta > 0 or after_level == "L3"):
            useful_strength = "strong"
            self.strong_useful_count += 1
        record = {
            "round": int(round_id),
            "candidate_id": candidate.get("candidate_id"),
            "target_missing_checkpoint": candidate.get("target_missing_checkpoint"),
            "candidate_goal": candidate.get("candidate_goal"),
            "materialization_status": candidate.get("materialization_status"),
            "injected": bool(injected),
            "executed": bool(executed),
            "stage": stage,
            "before_best_level": before_level,
            "after_best_level": after_level,
            "before_strict_vspc": round(before_strict, 6),
            "after_strict_vspc": round(after_strict, 6),
            "checkpoint_delta": checkpoint_delta,
            "strict_vspc_delta": round(after_strict - before_strict, 6),
            "evidence_vspc_delta": None,
            "oracle_level_delta": oracle_delta,
            "tx_to_delta_global": None,
            "fuzz_active_time_to_delta_s": None,
            "wall_time_to_delta_s": None,
            "sink_touched_delta": int(event_types.get("sink_touch", 0) > 0),
            "balance_change_delta": int(event_types.get("balance_change", 0) > 0),
            "transfer_delta": int(event_types.get("token_transfer", 0) > 0),
            "useful_strength": useful_strength,
            "related_event_count": len(related_events),
            "related_event_type_counts": dict(event_types),
            "modules_used": list(candidate.get("modules_used") or []),
            "module_source": candidate.get("module_source"),
            "before_level": before_level,
            "after_level": after_level,
            "before_vspc": round(before_strict, 6),
            "after_vspc": round(after_strict, 6),
            "l3_confirmed_by_candidate": bool(after_level == "L3" and oracle_delta > 0),
        }
        self._record_module_contribution(record)
        if self.enable_effect_forcing_v3 and _clean_list(candidate.get("modules_used")):
            _append_jsonl(self.effect_forcing_delta_path, record)
        return record

    def _record_module_contribution(self, record: Dict[str, Any]) -> None:
        modules = _clean_list(record.get("modules_used"))
        if not modules:
            return
        if not bool(record.get("executed")):
            return
        strict_delta = _safe_float(record.get("strict_vspc_delta"), 0.0)
        useful_strength = _text(record.get("useful_strength")).strip()
        for module in modules:
            counter = self.module_contribution.setdefault(module, Counter())
            counter["candidates_executed_by_module"] += 1
            if useful_strength == "medium":
                counter["medium_useful_by_module"] += 1
            if useful_strength == "strong":
                counter["strong_useful_by_module"] += 1
            if _safe_int(record.get("checkpoint_delta"), 0) > 0:
                counter["checkpoint_delta_by_module"] += 1
            if _safe_int(record.get("oracle_level_delta"), 0) > 0:
                counter["oracle_level_delta_by_module"] += 1
            if bool(record.get("l3_confirmed_by_candidate")):
                counter["l3_confirmed_by_module"] += 1
            self.module_vspc_delta_sum[module] += strict_delta

    def build_summary(self) -> Dict[str, Any]:
        materialization_summary = self.materializer.build_summary()
        module_contribution_summary = {}
        for module in EFFECT_FORCING_MODULES:
            counter = self.module_contribution.get(module, Counter())
            executed = int(counter.get("candidates_executed_by_module", 0))
            module_contribution_summary[module] = {
                "candidates_built_by_module": int(counter.get("candidates_built_by_module", 0)),
                "candidates_executed_by_module": executed,
                "medium_useful_by_module": int(counter.get("medium_useful_by_module", 0)),
                "strong_useful_by_module": int(counter.get("strong_useful_by_module", 0)),
                "checkpoint_delta_by_module": int(counter.get("checkpoint_delta_by_module", 0)),
                "oracle_level_delta_by_module": int(counter.get("oracle_level_delta_by_module", 0)),
                "l3_confirmed_by_module": int(counter.get("l3_confirmed_by_module", 0)),
                "avg_vspc_delta_by_module": round(
                    float(self.module_vspc_delta_sum.get(module, 0.0)) / float(executed),
                    6,
                ) if executed else 0.0,
            }
        prestate_summary = dict(self.prestate_summary)
        prefix_summary = dict(self.prefix_summary)
        effect_probe_summary = dict(self.effect_probe_summary)
        agent_repair_summary = dict(self.agent_repair_summary)
        strong_scheduling_summary = dict(self.strong_scheduling_summary)
        failure_signature_summary = dict(self.failure_signature_summary)
        failure_repair_trigger_summary = dict(self.failure_repair_trigger_summary)
        repair_materialization_summary = dict(self.repair_materialization_summary)
        repair_payload_normalization_summary = dict(self.repair_payload_normalization_summary)
        changed_fields_inference_summary = dict(self.changed_fields_inference_summary)
        repair_scheduling_summary = dict(self.repair_scheduling_summary)
        repair_fallback_summary = dict(self.repair_fallback_summary)
        failure_repair_summary = dict(self.failure_repair_summary)
        lifecycle_summary = self.repair_lifecycle.finalize()
        repair_attempt_cache_summary = dict(self.repair_attempt_cache_summary)
        repair_attempt_cache_summary["rejected_repair_attempt_keys"] = len(self.rejected_repair_attempt_keys)
        repair_attempt_cache_summary["rejected_repair_attempt_reasons"] = dict(self.rejected_repair_attempt_reasons)
        repair_attempt_cache_summary["max_repair_attempts_per_parent_signature"] = int(self.max_repair_attempts_per_parent_signature)
        repair_attempt_cache_summary["max_repair_materialization_attempts_per_run"] = int(self.max_repair_materialization_attempts_per_run)
        repair_attempt_cache_summary["repair_subsystem_error_log_path"] = self.repair_subsystem_error_log_path or None
        _write_json(self.prestate_summary_path, prestate_summary)
        _write_json(self.prefix_summary_path, prefix_summary)
        _write_json(self.effect_probe_summary_path, effect_probe_summary)
        _write_json(self.agent_effect_repair_summary_path, agent_repair_summary)
        _write_json(self.strong_scheduling_summary_path, strong_scheduling_summary)
        _write_json(self.module_contribution_summary_path, module_contribution_summary)
        _write_json(self.failure_signature_summary_path, failure_signature_summary)
        _write_json(self.failure_repair_trigger_summary_path, failure_repair_trigger_summary)
        _write_json(self.repair_materialization_summary_path, repair_materialization_summary)
        _write_json(self.repair_payload_normalization_summary_path, repair_payload_normalization_summary)
        _write_json(self.changed_fields_inference_summary_path, changed_fields_inference_summary)
        _write_json(self.repair_scheduling_summary_path, repair_scheduling_summary)
        _write_json(self.repair_fallback_summary_path, repair_fallback_summary)
        _write_json(self.candidate_lifecycle_summary_path, lifecycle_summary)
        _write_json(self.repair_attempt_cache_summary_path, repair_attempt_cache_summary)
        state_obligation_summary = self.state_obligation_builder.write_summary()
        dependency_grounded_scope_summary = self.dependency_grounded_scope_builder.write_summary()
        _write_json(self.state_obligation_summary_path, state_obligation_summary)
        _write_json(self.dependency_grounded_scope_summary_path, dependency_grounded_scope_summary)
        failure_repair_summary["repair_candidate_materialized"] = int(
            lifecycle_summary.get("repair_candidate_materialized", failure_repair_summary.get("repair_candidate_materialized", 0) or 0)
        )
        failure_repair_summary["repair_candidate_queued"] = int(
            lifecycle_summary.get("repair_candidate_queued", failure_repair_summary.get("repair_candidate_queued", 0) or 0)
        )
        failure_repair_summary["repair_candidate_executed"] = int(
            lifecycle_summary.get("repair_candidate_executed", failure_repair_summary.get("repair_candidate_executed", 0) or 0)
        )
        failure_repair_summary["repair_candidate_admitted"] = int(lifecycle_summary.get("repair_candidate_admitted", 0))
        failure_repair_summary["repair_candidate_selected"] = int(lifecycle_summary.get("repair_candidate_selected", 0))
        failure_repair_summary["repair_candidate_execution_submitted"] = int(lifecycle_summary.get("repair_candidate_execution_submitted", 0))
        failure_repair_summary["repair_candidate_execution_started"] = int(lifecycle_summary.get("repair_candidate_execution_started", 0))
        failure_repair_summary["repair_candidate_execution_exception"] = int(lifecycle_summary.get("repair_candidate_execution_exception", 0))
        failure_repair_summary["repair_candidate_attributed"] = int(lifecycle_summary.get("repair_candidate_attributed", 0))
        failure_repair_summary["repair_candidate_dropped"] = int(lifecycle_summary.get("repair_candidate_dropped", 0))
        submission_summary = self._repair_submission_summary_payload()
        failure_repair_summary.update(submission_summary)
        if failure_repair_summary:
            built = float(failure_repair_summary.get("repair_candidate_built", 0) or 0)
            executed = float(failure_repair_summary.get("repair_candidate_executed", 0) or 0)
            failure_repair_summary["repair_success_rate"] = round(executed / built, 6) if built > 0 else 0.0
        _write_json(self.failure_repair_summary_path, failure_repair_summary)
        _write_json(self.repair_selected_not_submitted_reasons_path, dict(self.repair_submit_false_by_reason))
        _write_json(self.repair_execution_submission_summary_path, submission_summary)
        self._write_failure_repair_prompt_catalog()
        if self._access_control_profile_requested():
            profile_row = {
                "bench_contract_id": self._runtime_bench_label(),
                "contract_name": self._runtime_contract_label(),
                "profile_enabled": bool(self.profile_effect_summary.get("profile_enabled")),
                "profile_name": self.profile_effect_summary.get("profile_name"),
                "access_control_heuristic_matched": bool(self.profile_effect_summary.get("access_control_heuristic_matched")),
                "matched_access_control_functions_count": int(self.profile_effect_summary.get("matched_access_control_functions_count", 0)),
                "matched_sensitive_functions_count": int(self.profile_effect_summary.get("matched_sensitive_functions_count", 0)),
                "caller_pool_size_before": int(self.profile_effect_summary.get("caller_pool_size_before", 0)),
                "caller_pool_size_after": int(self.profile_effect_summary.get("caller_pool_size_after", 0)),
                "observed_owner_admin_address_count": int(self.profile_effect_summary.get("observed_owner_admin_address_count", 0)),
                "generated_permission_setup_candidates": int(self.profile_effect_summary.get("generated_permission_setup_candidates", 0)),
                "generated_caller_permutation_candidates": int(self.profile_effect_summary.get("generated_caller_permutation_candidates", 0)),
                "unauthorized_revert_seen_count": int(self.profile_effect_summary.get("unauthorized_revert_seen_count", 0)),
                "unauthorized_revert_repair_candidates": int(self.profile_effect_summary.get("unauthorized_revert_repair_candidates", 0)),
                "setup_then_sensitive_success_count": int(self.profile_effect_summary.get("setup_then_sensitive_success_count", 0)),
                "ownership_or_role_event_seen_count": int(self.profile_effect_summary.get("ownership_or_role_event_seen_count", 0)),
                "access_control_storage_write_seen_count": int(self.profile_effect_summary.get("access_control_storage_write_seen_count", 0)),
                "profile_boosted_candidate_count": int(self.profile_effect_summary.get("profile_boosted_candidate_count", 0)),
                "runtime_pattern_family": self.runtime_pattern_family,
                "runtime_pattern_family_provenance": self.runtime_pattern_family_provenance,
                "runtime_pattern_family_source": self.profile_effect_summary.get("runtime_pattern_family_source"),
            }
            _write_json(self.profile_effect_summary_path, profile_row)
            _write_single_row_csv(
                self.profile_effect_csv_path,
                profile_row,
                ["bench_contract_id", "contract_name", *PROFILE_EFFECT_FIELDS, "runtime_pattern_family", "runtime_pattern_family_provenance", "runtime_pattern_family_source"],
            )
        _write_json(
            self.effect_forcing_summary_path,
            {
                "enabled": bool(self.enable_effect_forcing_v3),
                "modules": {
                    "prestate_builder": bool(self.enable_prestate_builder),
                    "prefix_replay_suffix_mutation": bool(self.enable_prefix_replay_suffix_mutation),
                    "effect_probes": bool(self.enable_effect_probes),
                    "agent_effect_repair": bool(self.enable_agent_effect_repair),
                    "failure_aware_agent_repair": bool(self.enable_failure_aware_agent_repair),
                    "strong_useful_scheduling": bool(self.enable_strong_useful_scheduling),
                },
                "prestate_builder_summary": prestate_summary,
                "prefix_replay_suffix_summary": prefix_summary,
                "effect_probe_summary": effect_probe_summary,
                "agent_effect_repair_summary": agent_repair_summary,
                "failure_signature_summary": failure_signature_summary,
                "failure_repair_trigger_summary": failure_repair_trigger_summary,
                "repair_materialization_summary": repair_materialization_summary,
                "repair_payload_normalization_summary": repair_payload_normalization_summary,
                "changed_fields_inference_summary": changed_fields_inference_summary,
                "repair_scheduling_summary": repair_scheduling_summary,
                "repair_execution_submission_summary": submission_summary,
                "repair_fallback_summary": repair_fallback_summary,
                "failure_repair_summary": failure_repair_summary,
                "profile_effect_summary": self.profile_effect_summary if self._access_control_profile_requested() else {},
                "state_obligation_summary": state_obligation_summary,
                "dependency_grounded_scope_summary": dependency_grounded_scope_summary,
                "candidate_lifecycle_summary": lifecycle_summary,
                "strong_useful_scheduling_summary": strong_scheduling_summary,
                "module_contribution_summary": module_contribution_summary,
            },
        )
        return {
            "enabled": True,
            "checkpoint_guided_injection_enabled": True,
            "rq1_rescue_mode": bool(self.rq1_rescue_mode),
            "effect_forcing_v3_enabled": bool(self.enable_effect_forcing_v3),
            "effect_forcing_v3_1_enabled": bool(self.enable_effect_forcing_v3_1),
            "prestate_builder_enabled": bool(self.enable_prestate_builder),
            "prefix_replay_suffix_mutation_enabled": bool(self.enable_prefix_replay_suffix_mutation),
            "effect_probes_enabled": bool(self.enable_effect_probes),
            "agent_effect_repair_enabled": bool(self.enable_agent_effect_repair),
            "failure_repair_pipeline_enabled": bool(self.enable_failure_repair_pipeline),
            "failure_aware_agent_repair_enabled": bool(self.enable_failure_aware_agent_repair),
            "strong_useful_scheduling_enabled": bool(self.enable_strong_useful_scheduling),
            "target_runtime_pattern_family": self.runtime_pattern_family,
            "current_best_level": self.progress.best_level,
            "current_level_vspc": round(self.progress.level_vspc(), 6),
            "current_strict_vspc": round(self.progress.strict_vspc(), 6),
            "covered_checkpoints": sorted(self.progress.covered),
            "next_missing_checkpoint": self.progress.next_missing(),
            "last_vspc_improvement_round": int(self.progress.last_vspc_improvement_round),
            "vspc_stagnation_rounds": int(self.progress.vspc_stagnation_rounds),
            "candidate_built_count": int(self.built_count),
            "candidate_injected_count": int(self.injected_count),
            "candidate_executed_count": int(self.executed_count),
            "candidate_executed_before_l2": int(self.candidate_executed_before_l2),
            "candidate_executed_before_l3": int(self.candidate_executed_before_l3),
            "candidate_delayed_after_l3": int(self.candidate_delayed_after_l3),
            "candidate_budget_used": int(self.candidate_budget_used),
            "source_counts": dict(self.source_counts),
            "no_candidate_reasons": dict(self.no_candidate_reasons),
            "filter_counts": dict(self.filter_counts),
            "medium_useful_candidate_count": int(self.medium_useful_count),
            "strong_useful_candidate_count": int(self.strong_useful_count),
            "checkpoint_delta_count": int(self.checkpoint_delta_count),
            "oracle_level_delta_count": int(self.oracle_level_delta_count),
            "observed_execution_count": len(self.observed_execution_ids),
            "llm_called": bool(self.llm_call_count > 0),
            "llm_call_count": int(self.llm_call_count),
            "llm_success_count": int(self.llm_success_count),
            "llm_error_count": int(self.llm_error_count),
            "llm_interaction_count": int(len(self.llm_interaction_events)),
            "llm_interaction_time_total_sec": round(float(self.llm_interaction_time_total_sec), 6),
            "llm_interaction_events": list(self.llm_interaction_events),
            "materialization": materialization_summary,
            "prestate_builder_summary": prestate_summary,
            "prefix_replay_suffix_summary": prefix_summary,
            "effect_probe_summary": effect_probe_summary,
            "agent_effect_repair_summary": agent_repair_summary,
            "failure_signature_summary": failure_signature_summary,
            "failure_repair_trigger_summary": failure_repair_trigger_summary,
            "repair_materialization_summary": repair_materialization_summary,
            "repair_payload_normalization_summary": repair_payload_normalization_summary,
            "changed_fields_inference_summary": changed_fields_inference_summary,
            "repair_scheduling_summary": repair_scheduling_summary,
            "repair_execution_submission_summary": submission_summary,
            "repair_fallback_summary": repair_fallback_summary,
            "repair_attempt_cache_summary": repair_attempt_cache_summary,
            "failure_repair_summary": failure_repair_summary,
            "profile_effect_summary": self.profile_effect_summary if self._access_control_profile_requested() else {},
            "state_obligation_summary": state_obligation_summary,
            "dependency_grounded_scope_summary": dependency_grounded_scope_summary,
            "candidate_lifecycle_summary": lifecycle_summary,
            "failure_repair_pipeline_counters": self._pipeline_counters_payload(),
            "strong_useful_scheduling_summary": strong_scheduling_summary,
            "module_contribution_summary": module_contribution_summary,
            "paths": {
                "checkpoint_progress": self.progress_path or None,
                "checkpoint_guided_candidates": self.candidates_path or None,
                "checkpoint_guided_candidate_delta": self.delta_path or None,
                "checkpoint_guided_filter_summary": self.filter_summary_path or None,
                "parameter_materialization_log": materialization_summary.get("log_path"),
                "parameter_materialization_summary": self.materializer.summary_path or None,
                "checkpoint_action_mapping_summary": self.mapping_summary_path or None,
                "checkpoint_guided_summary": self.summary_path or None,
                "prestate_builder_summary": self.prestate_summary_path or None,
                "prefix_replay_suffix_summary": self.prefix_summary_path or None,
                "effect_probe_summary": self.effect_probe_summary_path or None,
                "agent_effect_repair_summary": self.agent_effect_repair_summary_path or None,
                "failure_signature_summary": self.failure_signature_summary_path or None,
                "failure_repair_trigger_summary": self.failure_repair_trigger_summary_path or None,
                "repair_materialization_summary": self.repair_materialization_summary_path or None,
                "repair_payload_normalization_summary": self.repair_payload_normalization_summary_path or None,
                "changed_fields_inference_summary": self.changed_fields_inference_summary_path or None,
                "repair_scheduling_summary": self.repair_scheduling_summary_path or None,
                "repair_scheduler_decisions": self.repair_scheduler_decisions_path or None,
                "repair_selected_not_submitted_reasons": self.repair_selected_not_submitted_reasons_path or None,
                "repair_execution_submission_summary": self.repair_execution_submission_summary_path or None,
                "repair_execution_debug": self.repair_execution_debug_path or None,
                "repair_fallback_summary": self.repair_fallback_summary_path or None,
                "candidate_lifecycle": self.candidate_lifecycle_path or None,
                "candidate_lifecycle_summary": self.candidate_lifecycle_summary_path or None,
                "state_obligation_summary": self.state_obligation_summary_path or None,
                "dependency_grounded_scope_summary": self.dependency_grounded_scope_summary_path or None,
                "repair_attempt_cache_summary": self.repair_attempt_cache_summary_path or None,
                "rejected_repair_keys": self.rejected_repair_keys_path or None,
                "repair_subsystem_error_log": self.repair_subsystem_error_log_path or None,
                "failure_repair_summary": self.failure_repair_summary_path or None,
                "failure_repair_pipeline_counters": self.failure_repair_pipeline_counters_path or None,
                "strong_useful_scheduling_summary": self.strong_scheduling_summary_path or None,
                "module_contribution_summary": self.module_contribution_summary_path or None,
                "effect_forcing_summary": self.effect_forcing_summary_path or None,
                "profile_effect_summary_json": self.profile_effect_summary_path or None,
                "profile_effect_summary_csv": self.profile_effect_csv_path or None,
                "effect_forcing_candidate_delta": self.effect_forcing_delta_path or None,
                "failure_repair_candidate_delta": self.failure_repair_delta_path or None,
                "llm_failure_repair_interactions": self.failure_repair_interactions_path or None,
                "llm_failure_repair_prompt_catalog_json": self.failure_repair_prompt_catalog_json_path or None,
                "llm_failure_repair_prompt_catalog_md": self.failure_repair_prompt_catalog_md_path or None,
            },
            "no_main_search_replacement": True,
            "no_generator_mutation_crossover_selection_change": True,
        }

    def _write_summary(self) -> Dict[str, Any]:
        summary = self.build_summary()
        _write_json(self.summary_path, summary)
        _write_json(self.failure_repair_pipeline_counters_path, self._pipeline_counters_payload())
        self.materializer.write_summary()
        return summary

    def _remember_failure_repair_interaction(self, prompt_record: Dict[str, Any]) -> None:
        if not isinstance(prompt_record, dict):
            return
        stored = dict(prompt_record)
        repair_candidate_id = _text(stored.get("repair_candidate_id")).strip()
        if repair_candidate_id:
            self.repair_interaction_by_candidate_id[repair_candidate_id] = stored
        self.repair_interaction_records.append(stored)

    def _update_failure_repair_interaction(self, repair_candidate_id: Any, **updates: Any) -> None:
        candidate_id = _text(repair_candidate_id).strip()
        if not candidate_id:
            return
        record = self.repair_interaction_by_candidate_id.get(candidate_id)
        if not isinstance(record, dict):
            return
        for key, value in updates.items():
            if value is not None:
                record[key] = value
        self.repair_interaction_by_candidate_id[candidate_id] = record

    def _write_failure_repair_prompt_catalog(self) -> None:
        if not self.dump_failure_repair_prompts:
            return
        rows: List[Dict[str, Any]] = []
        for item in self.repair_interaction_records:
            if isinstance(item, dict):
                rows.append(dict(item))
        if not rows and self.failure_repair_interactions_path and os.path.exists(self.failure_repair_interactions_path):
            try:
                with open(self.failure_repair_interactions_path, "r", encoding="utf-8", errors="ignore") as handle:
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
                rows = []
        _write_json(self.failure_repair_prompt_catalog_json_path, rows)
        lines = [
            "# Failure Repair Prompt Catalog",
            "",
            "This file contains exact failure-aware repair interactions captured at runtime.",
            "",
        ]
        for row in rows[:128]:
            lines.append("## {}".format(_text(row.get("parent_candidate_id")).strip() or "UNKNOWN_PARENT"))
            lines.append("")
            lines.append("- target_missing_checkpoint: {}".format(_text(row.get("target_missing_checkpoint")).strip() or ""))
            lines.append("- repair_candidate_id: {}".format(_text(row.get("repair_candidate_id")).strip() or ""))
            lines.append("- error: {}".format(_text(row.get("error")).strip() or ""))
            lines.append("- metadata: `{}`".format(_json_compact(row.get("metadata"))))
            lines.append("- materialization_result: `{}`".format(_json_compact(row.get("materialization_result"))))
            lines.append("- execution_result: `{}`".format(_json_compact(row.get("execution_result"))))
            lines.append("- delta_attribution: `{}`".format(_json_compact(row.get("delta_attribution"))))
            lines.append("")
        if not rows:
            lines.append("- No failure-aware repair interactions captured.")
        try:
            with open(self.failure_repair_prompt_catalog_md_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        except Exception:
            pass


def _resolve_oracle_events_path(events_path: str) -> str:
    if not events_path:
        return ""
    if os.path.exists(events_path):
        return events_path
    directory = os.path.dirname(events_path)
    if not directory:
        return events_path
    for name in ("oracle_events.compact.jsonl.gz", "oracle_events.jsonl.gz", "oracle_events.jsonl"):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return events_path


def read_oracle_events(events_path: str, limit: int = 0) -> List[Dict[str, Any]]:
    events_path = _resolve_oracle_events_path(events_path)
    if not events_path or not os.path.exists(events_path):
        return []
    max_records = int(limit or 0)
    records: Any
    if max_records > 0:
        records = deque(maxlen=max(1, max_records))
    else:
        records = []
    try:
        opener = gzip.open if str(events_path).lower().endswith(".gz") else open
        with opener(events_path, "rt", encoding="utf-8", errors="ignore") as handle:
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
    return list(records)

