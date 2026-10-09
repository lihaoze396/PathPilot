# -*- coding: utf-8 -*-

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from fuzzer.utils import settings
except ImportError:
    from utils import settings


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _normalize_path(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = _text(item).strip()
        if text:
            output.append(text)
    return output


def _dedup_text(items: Sequence[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in items:
        text = _text(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _looks_like_address(value: Any) -> bool:
    text = _text(value).strip().lower()
    if len(text) != 42 or not text.startswith("0x"):
        return False
    return all(ch in "0123456789abcdef" for ch in text[2:])


def _is_zero_address(value: Any) -> bool:
    return _text(value).strip().lower() == "0x" + ("0" * 40)


class RepairSequenceConcretizer:
    PERMISSION_KEYWORDS = (
        "owner",
        "admin",
        "auth",
        "authorize",
        "whitelist",
        "blacklist",
        "exclude",
        "include",
        "role",
        "pause",
        "unpause",
    )
    CONFIG_KEYWORDS = (
        "set",
        "update",
        "config",
        "router",
        "pair",
        "swap",
        "fee",
        "limit",
        "launch",
        "trading",
    )
    RESTRICTION_KEYWORDS = (
        "blacklist",
        "ban",
        "block",
        "deny",
        "disable",
        "forbid",
        "freeze",
        "lock",
        "pause",
        "restrict",
    )

    def __init__(
        self,
        generator: Any,
        contract_name: Optional[str] = None,
        strategy_version: str = "v2",
    ):
        self.generator = generator
        self.contract_name = contract_name or getattr(generator, "contract_name", None)
        self.strategy_version = _text(strategy_version).strip() or "v2"

    def concretize_candidate(self, selected_candidate: Dict[str, Any]) -> Dict[str, Any]:
        chromosome, metadata = self.build_chromosome(selected_candidate)
        payload = dict(metadata)
        payload["guided_sequence"] = _normalize_path(selected_candidate.get("guided_sequence"))
        payload["candidate_id"] = _text(selected_candidate.get("candidate_id")).strip() or None
        payload["hypothesis_id"] = _text(selected_candidate.get("hypothesis_id")).strip() or None
        payload["target_function"] = _text(selected_candidate.get("target_function")).strip() or None
        payload["sequence_strategy"] = _text(selected_candidate.get("sequence_strategy")).strip() or None
        payload["chromosome"] = chromosome
        return payload

    def build_chromosome(self, selected_candidate: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        guided_sequence = _normalize_path(selected_candidate.get("guided_sequence"))
        if not guided_sequence:
            return None, self._base_metadata("failed", "empty_guided_sequence")

        generators = [self.generator] + list(getattr(self.generator, "other_generators", []) or [])
        account_pool = self._build_account_pool(selected_candidate)
        if not account_pool:
            return None, self._base_metadata("failed", "account_pool_empty")

        plan = self._plan_bindings(selected_candidate, guided_sequence, account_pool)
        genes: List[Dict[str, Any]] = []
        gene_debug: List[Dict[str, Any]] = []
        notes: List[str] = list(plan.get("notes") or [])
        partial = bool(plan.get("partial"))

        for gene_index, function_name in enumerate(guided_sequence):
            resolved = self._resolve_function(generators, function_name)
            if resolved is None:
                return None, self._base_metadata("failed", "function_not_resolved", notes=notes)

            target_generator, function_selector, argument_types, state_mutability = resolved
            temp_generator = self._clone_generator(target_generator)
            generated = temp_generator.generate_individual(function_selector, argument_types)
            if not generated:
                return None, self._base_metadata("failed", "generator_failed", notes=notes)

            gene = copy.deepcopy(generated[0])
            sender = self._pick_sender_for_gene(
                selected_candidate=selected_candidate,
                guided_sequence=guided_sequence,
                gene_index=gene_index,
                function_name=function_name,
                account_pool=account_pool,
                plan=plan,
            )
            if not sender:
                return None, self._base_metadata("failed", "sender_unavailable", notes=notes)

            value = self._pick_value_for_gene(
                selected_candidate=selected_candidate,
                function_name=function_name,
                gene=gene,
                gene_index=gene_index,
                state_mutability=state_mutability,
            )
            arguments, arg_partial, arg_notes = self._build_arguments(
                function_name=function_name,
                function_selector=function_selector,
                argument_types=argument_types,
                generated_gene=gene,
                account_pool=account_pool,
                sender=sender,
                gene_index=gene_index,
                guided_sequence=guided_sequence,
                selected_candidate=selected_candidate,
                plan=plan,
            )
            if arguments is None:
                return None, self._base_metadata("failed", "argument_construction_failed", notes=notes + arg_notes)

            partial = partial or arg_partial
            notes.extend(arg_notes)
            gene["account"] = sender
            gene["contract"] = getattr(target_generator, "contract", gene.get("contract"))
            gene["amount"] = value
            gene["arguments"] = arguments
            gene.setdefault("blocknumber", None)
            gene.setdefault("timestamp", None)
            gene.setdefault("gaslimit", settings.GAS_LIMIT)
            gene.setdefault("call_return", {})
            gene.setdefault("extcodesize", {})
            gene.setdefault("returndatasize", {})
            genes.append(gene)
            gene_debug.append(
                {
                    "gene_index": int(gene_index),
                    "function_name": _text(function_name).strip() or None,
                    "selector": _text(function_selector).strip() or None,
                    "sender": _text(sender).strip() or None,
                    "value": value,
                    "argument_types": list(argument_types or []),
                    "arguments": copy.deepcopy(arguments),
                }
            )

        metadata = self._base_metadata(
            "partial" if partial else "mappable",
            None,
            notes=notes,
        )
        metadata["sender_binding_reason"] = plan.get("sender_binding_reason")
        metadata["value_binding_reason"] = plan.get("value_binding_reason")
        metadata["argument_binding_reason"] = plan.get("argument_binding_reason")
        metadata["prestate_binding_reason"] = plan.get("prestate_binding_reason")
        metadata["preserve_prefix"] = bool(plan.get("preserve_prefix"))
        metadata["binding_plan"] = self._debug_plan_snapshot(plan)
        metadata["gene_debug"] = gene_debug
        return genes, metadata

    def _build_account_pool(self, selected_candidate: Dict[str, Any]) -> List[str]:
        base_pool = [item for item in list(getattr(self.generator, "accounts", []) or []) if _text(item).strip()]
        raw = selected_candidate.get("raw") if isinstance(selected_candidate.get("raw"), dict) else {}
        extra_pool = []
        for source in (selected_candidate, raw):
            extra_pool.extend(_normalize_path(source.get("access_control_address_pool")))
            hints = source.get("access_control_binding_hints") if isinstance(source.get("access_control_binding_hints"), dict) else {}
            extra_pool.extend(_normalize_path(hints.get("owner_candidates")))
            extra_pool.extend(_normalize_path(hints.get("admin_candidates")))
            extra_pool.extend(_normalize_path(hints.get("pending_owner_candidates")))
            extra_pool.extend(_normalize_path(hints.get("attacker_candidates")))
            extra_pool.extend(_normalize_path(hints.get("address_pool")))
            extra_pool.extend(_normalize_path([hints.get("deployer"), hints.get("contract_self")]))
        return _dedup_text(base_pool + extra_pool)

    def _base_metadata(self, status: str, skip_reason: Optional[str], notes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        return {
            "concretization_status": _text(status).strip() or "failed",
            "skip_reason": _text(skip_reason).strip() or None,
            "sender_binding_reason": None,
            "value_binding_reason": None,
            "argument_binding_reason": None,
            "prestate_binding_reason": None,
            "preserve_prefix": False,
            "concretization_notes": list(dict.fromkeys([_text(item).strip() for item in (notes or []) if _text(item).strip()])),
            "strategy_version": self.strategy_version,
        }

    @staticmethod
    def _debug_plan_snapshot(plan: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "owner",
            "admin",
            "pending_owner",
            "same_sender",
            "privileged_sender",
            "asset_flow_current_state",
            "asset_flow_target",
            "asset_flow_target_alt",
            "contract_self",
            "owner_candidates",
            "admin_candidates",
            "pending_owner_candidates",
            "preferred_sender_candidates",
            "state_change_candidates",
        )
        snapshot: Dict[str, Any] = {}
        for key in keys:
            if key in plan:
                snapshot[key] = copy.deepcopy(plan.get(key))
        return snapshot

    def _normalize_address_candidates(
        self,
        values: Any,
        *,
        allow_zero: bool = False,
        exclude: Optional[Sequence[Any]] = None,
    ) -> List[str]:
        raw_values = values if isinstance(values, (list, tuple, set)) else [values]
        excluded = {
            _text(item).strip().lower()
            for item in list(exclude or [])
            if _text(item).strip()
        }
        output: List[str] = []
        seen = set()
        for item in raw_values:
            text = _text(item).strip()
            lowered = text.lower()
            if not _looks_like_address(text):
                continue
            if not allow_zero and _is_zero_address(text):
                continue
            if lowered in excluded or lowered in seen:
                continue
            seen.add(lowered)
            output.append(text)
        return output

    def _pick_first_address(
        self,
        values: Any,
        *,
        allow_zero: bool = False,
        exclude: Optional[Sequence[Any]] = None,
    ) -> Optional[str]:
        normalized = self._normalize_address_candidates(
            values,
            allow_zero=allow_zero,
            exclude=exclude,
        )
        return normalized[0] if normalized else None

    def _resolve_function(self, generators: Sequence[Any], function_name: str):
        normalized_name = _text(function_name).strip()
        if not normalized_name:
            return None
        for generator in generators:
            if normalized_name in ("fallback", "receive") and "fallback" in getattr(generator, "interface", {}):
                state_mutability = self._resolve_state_mutability(
                    generator,
                    selector="fallback",
                    function_name=normalized_name,
                )
                return generator, "fallback", list(getattr(generator, "interface", {}).get("fallback") or []), state_mutability
            interface_mapper = getattr(generator, "interface_mapper", {}) or {}
            matches: List[Tuple[str, str]] = []
            for signature, selector in interface_mapper.items():
                if _text(signature).split("(", 1)[0] == normalized_name:
                    matches.append((_text(signature), _text(selector)))
            if not matches:
                continue
            matches.sort(key=lambda item: item[0])
            signature, selector = matches[0]
            interface = getattr(generator, "interface", {}) or {}
            argument_types = list(interface.get(selector) or [])
            state_mutability = self._resolve_state_mutability(
                generator,
                selector=selector,
                signature=signature,
                function_name=normalized_name,
            )
            return generator, selector, argument_types, state_mutability
        return None

    def _clone_generator(self, generator: Any):
        cloned = generator.__class__(
            interface=copy.deepcopy(getattr(generator, "interface", {})),
            bytecode=getattr(generator, "bytecode", None),
            accounts=list(getattr(generator, "accounts", []) or []),
            contract=getattr(generator, "contract", None),
            other_generators=list(getattr(generator, "other_generators", []) or []),
            interface_mapper=copy.deepcopy(getattr(generator, "interface_mapper", {})),
            contract_name=getattr(generator, "contract_name", None),
            sol_path=getattr(generator, "sol_path", None),
            function_state_mutability=copy.deepcopy(getattr(generator, "function_state_mutability", {})),
        )
        try:
            cloned.update_other_generators(
                list(getattr(generator, "other_generators", []) or []),
                copy.deepcopy(getattr(generator, "total_interface_mapper", {})),
            )
        except Exception:
            pass
        return cloned

    @staticmethod
    def _resolve_state_mutability(
        generator: Any,
        selector: Optional[str] = None,
        signature: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> str:
        resolver = getattr(generator, "resolve_state_mutability", None)
        if callable(resolver):
            try:
                return _text(
                    resolver(selector=selector, signature=signature, function_name=function_name)
                ).strip().lower() or "nonpayable"
            except Exception:
                pass
        mapping = getattr(generator, "function_state_mutability", {}) or {}
        for key in (signature, function_name, selector):
            text = _text(key).strip()
            if not text:
                continue
            state_mutability = _text(mapping.get(text)).strip().lower()
            if state_mutability:
                return state_mutability
        return "nonpayable"

    def _plan_bindings(
        self,
        selected_candidate: Dict[str, Any],
        guided_sequence: Sequence[str],
        account_pool: Sequence[str],
    ) -> Dict[str, Any]:
        sender_policy = _text(selected_candidate.get("sender_policy")).strip().lower()
        sequence_strategy = _text(selected_candidate.get("sequence_strategy")).strip().lower()
        value_policy = _text(selected_candidate.get("value_policy")).strip().lower()
        runtime_family = _text(
            selected_candidate.get("runtime_pattern_family")
            or selected_candidate.get("target_runtime_pattern_family")
        ).strip().lower()
        notes: List[str] = []
        partial = False

        raw = selected_candidate.get("raw") if isinstance(selected_candidate.get("raw"), dict) else {}
        hints = selected_candidate.get("access_control_binding_hints") if isinstance(selected_candidate.get("access_control_binding_hints"), dict) else {}
        if not hints and isinstance(raw.get("access_control_binding_hints"), dict):
            hints = dict(raw.get("access_control_binding_hints") or {})
        contract_self = _text(hints.get("contract_self")).strip() or None

        fallback_candidates = self._normalize_address_candidates(account_pool)
        owner_candidates = self._normalize_address_candidates(hints.get("owner_candidates"))
        admin_candidates = self._normalize_address_candidates(hints.get("admin_candidates"))
        pending_owner_candidates = self._normalize_address_candidates(hints.get("pending_owner_candidates"))
        preferred_sender_candidates = self._normalize_address_candidates(
            hints.get("preferred_sender_candidates")
        )
        current_state_candidates = self._normalize_address_candidates(
            hints.get("current_state_candidates")
        )
        state_change_candidates = self._normalize_address_candidates(
            hints.get("state_change_candidates")
        )
        attacker_candidates = self._normalize_address_candidates(
            hints.get("attacker_candidates")
        )

        owner = self._pick_first_address(
            owner_candidates + preferred_sender_candidates + fallback_candidates
        ) or (_text(account_pool[0]).strip() if account_pool else None)
        admin = self._pick_first_address(
            admin_candidates + owner_candidates + preferred_sender_candidates + fallback_candidates
        ) or owner
        pending_owner = self._pick_first_address(
            pending_owner_candidates + admin_candidates + fallback_candidates,
            exclude=[owner],
        ) or admin
        current_state = self._pick_first_address(current_state_candidates)
        change_target_candidates = (
            state_change_candidates
            + attacker_candidates
            + fallback_candidates
            + preferred_sender_candidates
        )
        target_missing_checkpoint = _text(selected_candidate.get("target_missing_checkpoint")).strip().lower()
        change_target_exclude = [contract_self, current_state]
        alt_1 = self._pick_first_address(change_target_candidates, exclude=change_target_exclude)
        if not alt_1 and owner and owner not in {contract_self, current_state}:
            alt_1 = owner
        alt_2 = self._pick_first_address(
            change_target_candidates + [admin, pending_owner, owner],
            exclude=change_target_exclude + [alt_1],
        ) or alt_1
        if target_missing_checkpoint == "receiver_or_fee_state_written":
            nonprivileged_target = self._pick_first_address(
                state_change_candidates + fallback_candidates,
                exclude=change_target_exclude + owner_candidates + admin_candidates + pending_owner_candidates + preferred_sender_candidates,
            )
            if nonprivileged_target:
                alt_2 = alt_1 or alt_2
                alt_1 = nonprivileged_target
                notes.append("asset_flow_receiver_state_prefers_nonprivileged_target")

        same_sender = owner
        privileged_sender = self._pick_first_address(
            preferred_sender_candidates + owner_candidates + [owner, admin] + fallback_candidates
        ) or owner
        if "rotate_existing_senders" in sender_policy:
            sender_reason = "rotate_existing_senders applied over generator account pool"
        elif "owner" in sender_policy or "deployer" in sender_policy:
            sender_reason = "privileged sender preference bound to first available account"
            partial = True
            notes.append("owner_or_deployer_not_confirmed_using_first_account")
        else:
            sender_reason = "same/default sender bound from generator account pool"

        if "permission_setup_then_sink" in sequence_strategy or "owner" in sender_policy or "deployer" in sender_policy:
            sender_reason = "permission setup uses privileged sender and keeps sender consistency where possible"

        prestate_reason = None
        if "approve_before_transferfrom" in sequence_strategy and "approve" in guided_sequence and "transferFrom" in guided_sequence:
            prestate_reason = "approve/transferFrom sender and amount bindings enabled"
        elif "config_before_sink" in sequence_strategy:
            prestate_reason = "config-like functions bound to privileged sender with conservative arguments"
            partial = True
        elif "permission_setup_then_sink" in sequence_strategy:
            prestate_reason = "permission-like functions bound before sink with sender consistency"
            partial = True

        value_reason = "default zero/original value policy"
        if "nonzero" in value_policy or any(item in ("fallback", "receive") for item in guided_sequence):
            value_reason = "small nonzero value used for nonzero/fallback-oriented sequence"

        argument_reason = "ABI-aware conservative mutation with transfer and permission binding rules"
        if runtime_family == "asset_flow_redirection":
            argument_reason = "ABI-aware conservative mutation with asset-flow state-change target preference"
            if current_state:
                notes.append("asset_flow_current_state_observed")
            if alt_1 and alt_1 != current_state:
                notes.append("asset_flow_distinct_target_selected")
        preserve_prefix = "preserve_prefix" in sequence_strategy
        if preserve_prefix:
            notes.append("preserve_prefix_enabled")

        return {
            "partial": partial,
            "notes": notes,
            "owner": owner,
            "admin": admin,
            "pending_owner": pending_owner,
            "alt_1": alt_1,
            "alt_2": alt_2,
            "contract_self": contract_self,
            "same_sender": same_sender,
            "privileged_sender": privileged_sender,
            "asset_flow_current_state": current_state,
            "asset_flow_target": alt_1,
            "asset_flow_target_alt": alt_2,
            "owner_candidates": list(owner_candidates),
            "admin_candidates": list(admin_candidates),
            "pending_owner_candidates": list(pending_owner_candidates),
            "preferred_sender_candidates": list(preferred_sender_candidates),
            "state_change_candidates": list(state_change_candidates),
            "sender_binding_reason": sender_reason,
            "value_binding_reason": value_reason,
            "argument_binding_reason": argument_reason,
            "prestate_binding_reason": prestate_reason,
            "preserve_prefix": preserve_prefix,
        }

    def _pick_sender_for_gene(
        self,
        selected_candidate: Dict[str, Any],
        guided_sequence: Sequence[str],
        gene_index: int,
        function_name: str,
        account_pool: Sequence[str],
        plan: Dict[str, Any],
    ) -> Optional[str]:
        sender_policy = _text(selected_candidate.get("sender_policy")).strip().lower()
        normalized_name = _text(function_name).strip().lower()
        runtime_family = _text(
            selected_candidate.get("runtime_pattern_family")
            or selected_candidate.get("target_runtime_pattern_family")
        ).strip().lower()
        target_missing_checkpoint = _text(selected_candidate.get("target_missing_checkpoint")).strip().lower()
        setter_seen_before = any(
            self._is_config_like(_text(name).strip().lower())
            for name in guided_sequence[:gene_index]
        )
        if not account_pool:
            return None
        if "rotate_existing_senders" in sender_policy:
            return _text(account_pool[gene_index % len(account_pool)]).strip() or None
        if "approve" == function_name and "transferFrom" in guided_sequence:
            return plan.get("owner") or plan.get("same_sender")
        if function_name == "transferFrom" and "approve" in guided_sequence:
            return plan.get("alt_1") or plan.get("same_sender")
        if normalized_name == "acceptownership":
            return plan.get("pending_owner") or plan.get("alt_1") or plan.get("same_sender")
        if runtime_family == "asset_flow_redirection" and self._is_config_like(normalized_name):
            return (
                plan.get("privileged_sender")
                or plan.get("asset_flow_current_state")
                or plan.get("same_sender")
            )
        if (
            runtime_family == "asset_flow_redirection"
            and target_missing_checkpoint == "receiver_or_fee_state_written"
            and setter_seen_before
            and any(token in normalized_name for token in ("supply", "mint", "burn", "claim", "collect", "withdraw"))
        ):
            return (
                plan.get("asset_flow_target")
                or plan.get("asset_flow_target_alt")
                or plan.get("same_sender")
            )
        if self._is_permission_like(normalized_name) or self._is_config_like(normalized_name):
            return plan.get("privileged_sender") or plan.get("same_sender")
        if "keep_same_sender_across_sequence" in sender_policy:
            return plan.get("same_sender")
        if "owner" in sender_policy or "deployer" in sender_policy:
            return plan.get("privileged_sender") or plan.get("same_sender")
        return plan.get("same_sender") or _text(account_pool[0]).strip() or None

    def _pick_value_for_gene(
        self,
        selected_candidate: Dict[str, Any],
        function_name: str,
        gene: Dict[str, Any],
        gene_index: int,
        state_mutability: Optional[str] = None,
    ) -> int:
        value_policy = _text(selected_candidate.get("value_policy")).strip().lower()
        target_function = _text(selected_candidate.get("target_function")).strip().lower()
        normalized_name = _text(function_name).strip().lower()
        normalized_mutability = _text(state_mutability).strip().lower() or "nonpayable"
        accepts_value = normalized_name in ("fallback", "receive") or normalized_mutability == "payable"
        if value_policy == "force_zero_value":
            return 0
        if normalized_name in ("fallback", "receive"):
            return 1
        if not accepts_value:
            amount = gene.get("amount")
            if isinstance(amount, int) and amount == 0:
                return 0
            return 0
        if target_function in ("fallback", "receive") and value_policy == "try_small_nonzero_value":
            return 1
        if "nonzero" in value_policy:
            return 1
        if value_policy == "try_zero_and_small_nonzero":
            return 1 if gene_index == 0 else 0
        amount = gene.get("amount")
        if isinstance(amount, int):
            return amount
        return 0

    def _build_arguments(
        self,
        function_name: str,
        function_selector: str,
        argument_types: Sequence[str],
        generated_gene: Dict[str, Any],
        account_pool: Sequence[str],
        sender: str,
        gene_index: int,
        guided_sequence: Sequence[str],
        selected_candidate: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Tuple[Optional[List[Any]], bool, List[str]]:
        generated_arguments = list(generated_gene.get("arguments") or [])
        generated_values = list(generated_arguments[1:]) if generated_arguments else []
        while len(generated_values) < len(argument_types):
            generated_values.append(None)

        notes: List[str] = []
        partial = False
        amount = 1
        next_function = _text(guided_sequence[gene_index + 1]).strip() if gene_index + 1 < len(guided_sequence) else ""
        previous_function = _text(guided_sequence[gene_index - 1]).strip() if gene_index > 0 else ""
        runtime_family = _text(
            selected_candidate.get("runtime_pattern_family")
            or selected_candidate.get("target_runtime_pattern_family")
        ).strip().lower()

        if function_name == "approve" and len(argument_types) == 2:
            spender = plan.get("alt_1") or sender
            if next_function == "transferFrom":
                spender = plan.get("alt_1") or spender
            return [function_selector, spender, amount], partial, notes

        if function_name == "transfer" and len(argument_types) == 2:
            to_value = plan.get("alt_1") if plan.get("alt_1") != sender else plan.get("alt_2")
            to_value = to_value or sender
            return [function_selector, to_value, amount], partial, notes

        if function_name == "transferFrom" and len(argument_types) == 3:
            from_value = plan.get("owner") or sender
            if runtime_family == "lock_pause_blacklist_bypass":
                to_value = plan.get("alt_1") if plan.get("alt_1") not in (None, "", from_value) else plan.get("alt_2")
            else:
                to_value = plan.get("alt_2") if plan.get("alt_2") not in (None, "", from_value) else plan.get("alt_1")
            to_value = to_value or from_value
            return [function_selector, from_value, to_value, amount], partial, notes

        values: List[Any] = []
        argument_policy = _text(selected_candidate.get("argument_policy")).strip().lower()
        asset_flow_current_state = _text(plan.get("asset_flow_current_state")).strip() or None
        asset_flow_target = _text(plan.get("asset_flow_target")).strip() or None
        asset_flow_target_alt = _text(plan.get("asset_flow_target_alt")).strip() or None
        target_missing_checkpoint = _text(selected_candidate.get("target_missing_checkpoint")).strip().lower()
        asset_flow_arg_exclude = [
            plan.get("contract_self"),
            asset_flow_current_state,
        ]
        for arg_index, arg_type in enumerate(argument_types):
            normalized_type = _text(arg_type).replace(" storage", "").replace(" memory", "").strip().lower()
            base_value = generated_values[arg_index]
            if "[" in normalized_type and "]" in normalized_type:
                array_value, array_partial, array_notes = self._build_array_argument(
                    arg_type=normalized_type,
                    account_pool=account_pool,
                    sender=sender,
                    ordinal=gene_index + arg_index,
                )
                values.append(array_value)
                partial = partial or array_partial
                notes.extend(array_notes)
                continue
            if normalized_type.startswith("address"):
                lowered_name = _text(function_name).strip().lower()
                if lowered_name == "transferownership" and arg_index == 0:
                    values.append(plan.get("pending_owner") or plan.get("alt_1") or sender)
                elif lowered_name in {"grantrole", "addrole", "addminter", "setoperator", "setadmin", "whitelist", "allowlist"} and arg_index == 0:
                    values.append(plan.get("alt_1") or plan.get("pending_owner") or sender)
                elif (
                    runtime_family == "lock_pause_blacklist_bypass"
                    and arg_index == 0
                    and self._is_restriction_setter_like(lowered_name)
                    and (
                        any(token in _text(previous_function).strip().lower() for token in ("transfer", "sell", "withdraw"))
                        or any(token in _text(next_function).strip().lower() for token in ("transfer", "sell", "withdraw"))
                    )
                ):
                    values.append(plan.get("alt_1") or plan.get("alt_2") or sender)
                    notes.append("restriction_setter_address_aligned_with_transfer_target")
                elif self._is_permission_like(lowered_name) and arg_index == 0:
                    values.append(plan.get("alt_1") or plan.get("owner") or sender)
                elif (
                    runtime_family == "asset_flow_redirection"
                    and arg_index == 0
                    and self._is_config_like(lowered_name)
                ):
                    preferred_target = None
                    if target_missing_checkpoint == "receiver_or_fee_state_written":
                        preferred_target = self._pick_first_address(
                            list(plan.get("state_change_candidates") or [])
                            + [asset_flow_target, asset_flow_target_alt, plan.get("alt_1"), plan.get("alt_2")]
                            + list(account_pool),
                            exclude=asset_flow_arg_exclude
                            + [sender, plan.get("owner"), plan.get("admin"), plan.get("pending_owner")]
                            + list(plan.get("preferred_sender_candidates") or []),
                        )
                        if preferred_target:
                            notes.append("asset_flow_receiver_state_prefers_nonprivileged_target")
                    preferred_target = self._pick_first_address(
                        [preferred_target, asset_flow_target, asset_flow_target_alt, plan.get("alt_1"), plan.get("alt_2")],
                        exclude=asset_flow_arg_exclude,
                    ) or preferred_target
                    if not preferred_target:
                        preferred_target = self._pick_first_address(
                            list(account_pool) + [plan.get("owner"), plan.get("admin"), sender],
                            exclude=asset_flow_arg_exclude,
                        )
                    values.append(preferred_target or sender)
                    if preferred_target and preferred_target != asset_flow_current_state:
                        notes.append("asset_flow_config_address_prefers_distinct_state_target")
                elif self._is_config_like(_text(function_name).strip().lower()) and arg_index == 0:
                    values.append(plan.get("alt_1") or sender)
                else:
                    addr = self._pick_address(account_pool, arg_index + gene_index + 1, fallback=plan.get("contract_self") or sender)
                    values.append(addr)
                continue
            if normalized_type.startswith("bool"):
                values.append(bool((gene_index + arg_index) % 2 == 0))
                partial = partial or "mutate_permission_and_sink_args" in argument_policy
                continue
            if normalized_type.startswith("uint") or normalized_type.startswith("int"):
                if self._is_config_like(_text(function_name).strip().lower()):
                    values.append(1 if (gene_index + arg_index) % 2 == 0 else 0)
                    partial = True
                elif "boundary" in argument_policy:
                    values.append(self._boundary_value(normalized_type, gene_index + arg_index))
                else:
                    values.append(1)
                continue
            if normalized_type.startswith("bytes") or normalized_type == "string":
                if base_value is None:
                    values.append("" if normalized_type == "string" else b"")
                    partial = True
                    notes.append("abi_type_without_specific_binding_placeholder_used")
                else:
                    values.append(base_value)
                continue
            if base_value is not None:
                values.append(base_value)
            else:
                values.append(0)
                partial = True
                notes.append("unknown_abi_type_placeholder_used")
        return [function_selector] + values, partial, notes

    def _build_array_argument(
        self,
        arg_type: str,
        account_pool: Sequence[str],
        sender: str,
        ordinal: int,
    ) -> Tuple[Any, bool, List[str]]:
        normalized_type = _text(arg_type).replace(" storage", "").replace(" memory", "").strip().lower()
        element_type = normalized_type.split("[", 1)[0]
        partial = True
        notes = ["array_argument_conservative_placeholder_used"]
        if element_type.startswith("address"):
            first = self._pick_address(account_pool, ordinal, fallback=sender)
            second = self._pick_address(account_pool, ordinal + 1, fallback=sender)
            if second == first and len(account_pool) > 1:
                second = self._pick_address(account_pool, ordinal + 2, fallback=sender)
            return [first] if first == second else [first, second], partial, notes
        if element_type.startswith("bool"):
            return [True, False], partial, notes
        if element_type.startswith("uint") or element_type.startswith("int"):
            return [0, 1], partial, notes
        if element_type.startswith("bytes") or element_type == "string":
            return ["" if element_type == "string" else b""], partial, notes
        return [0], partial, notes

    @staticmethod
    def _pick_address(account_pool: Sequence[str], ordinal: int, fallback: Optional[str] = None) -> Any:
        if account_pool:
            return account_pool[ordinal % len(account_pool)]
        return fallback

    @staticmethod
    def _boundary_value(arg_type: str, ordinal: int) -> int:
        normalized_type = _text(arg_type).strip().lower()
        if normalized_type.startswith("uint"):
            bit_text = normalized_type.replace("uint", "") or "256"
            try:
                bits = max(8, int(bit_text))
            except Exception:
                bits = 256
            values = [0, 1, (2 ** bits) - 1]
            return values[ordinal % len(values)]
        if normalized_type.startswith("int"):
            bit_text = normalized_type.replace("int", "") or "256"
            try:
                bits = max(8, int(bit_text))
            except Exception:
                bits = 256
            max_value = (2 ** (bits - 1)) - 1
            values = [0, 1, max_value]
            return values[ordinal % len(values)]
        return 1

    def _is_permission_like(self, function_name: str) -> bool:
        lowered = _text(function_name).strip().lower()
        return any(keyword in lowered for keyword in self.PERMISSION_KEYWORDS)

    def _is_config_like(self, function_name: str) -> bool:
        lowered = _text(function_name).strip().lower()
        return any(keyword in lowered for keyword in self.CONFIG_KEYWORDS)

    def _is_restriction_setter_like(self, function_name: str) -> bool:
        lowered = _text(function_name).strip().lower()
        if any(keyword in lowered for keyword in self.RESTRICTION_KEYWORDS):
            return True
        if lowered.startswith("set") or lowered.startswith("update"):
            return True
        if lowered.startswith("transfer") and lowered not in {"transfer", "transferfrom", "transferownership"}:
            return True
        return False

