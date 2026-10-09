# -*- coding: utf-8 -*-

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


CONFIG_KEYWORDS = (
    "set",
    "update",
    "config",
    "configure",
    "router",
    "pair",
    "whitelist",
    "blacklist",
    "pause",
    "unpause",
    "enable",
    "disable",
)
TRADING_KEYWORDS = ("trading",)
PERMISSION_KEYWORDS = (
    "owner",
    "admin",
    "authorized",
    "authorize",
    "permission",
    "role",
    "whitelist",
    "blacklist",
)


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


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _normalize_path(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            output.append(text)
    return output


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = _text(item).strip()
        if text:
            output.append(text)
    return output


def _function_matches(function_name: str, keywords: Sequence[str]) -> bool:
    lowered = _text(function_name).lower()
    return bool(lowered and any(keyword in lowered for keyword in keywords))


def _split_params(params_text: str) -> List[str]:
    text = params_text.strip()
    if not text:
        return []
    parts: List[str] = []
    current: List[str] = []
    depth_round = 0
    depth_square = 0
    for char in text:
        if char == "," and depth_round == 0 and depth_square == 0:
            item = "".join(current).strip()
            if item:
                parts.append(item)
            current = []
            continue
        current.append(char)
        if char == "(":
            depth_round += 1
        elif char == ")" and depth_round > 0:
            depth_round -= 1
        elif char == "[":
            depth_square += 1
        elif char == "]" and depth_square > 0:
            depth_square -= 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_comments(source_text: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", "", source_text, flags=re.DOTALL)
    without_line = re.sub(r"//.*?$", "", without_block, flags=re.MULTILINE)
    return without_line


class RepairSequenceDryRunValidator:
    def __init__(
        self,
        output_dir: str,
        repair_candidates_path: Optional[str] = None,
        source_path: Optional[str] = None,
        contract_name: Optional[str] = None,
        path_plan_path: Optional[str] = None,
        ksi_evidence_path: Optional[str] = None,
        blocker_evidence_path: Optional[str] = None,
    ):
        self.output_dir = os.path.abspath(output_dir) if output_dir else ""
        self.contract_name = contract_name

        self._requested_repair_candidates_path = (
            os.path.abspath(repair_candidates_path) if repair_candidates_path else None
        )
        self._requested_source_path = os.path.abspath(source_path) if source_path else None
        self._requested_path_plan_path = os.path.abspath(path_plan_path) if path_plan_path else None
        self._requested_ksi_evidence_path = os.path.abspath(ksi_evidence_path) if ksi_evidence_path else None
        self._requested_blocker_evidence_path = (
            os.path.abspath(blocker_evidence_path) if blocker_evidence_path else None
        )

        self.default_repair_candidates_path = (
            os.path.join(self.output_dir, "guided_sequence_candidates.json") if self.output_dir else ""
        )
        self.default_agent_path_plan_path = (
            os.path.join(self.output_dir, "agent_path_plan.json") if self.output_dir else ""
        )
        self.default_proposed_path_plan_path = (
            os.path.join(self.output_dir, "proposed_path_plan.json") if self.output_dir else ""
        )
        self.default_ksi_evidence_path = (
            os.path.join(self.output_dir, "ksi_evidence.jsonl") if self.output_dir else ""
        )
        self.default_blocker_evidence_path = (
            os.path.join(self.output_dir, "blocker_evidence.json") if self.output_dir else ""
        )

        self.loaded_inputs: Dict[str, bool] = {
            "repair_candidates": False,
            "path_plan": False,
            "ksi_evidence": False,
            "blocker_evidence": False,
            "source": False,
        }
        self.input_paths: Dict[str, Optional[str]] = {
            "repair_candidates": self._requested_repair_candidates_path or self.default_repair_candidates_path or None,
            "path_plan": self._requested_path_plan_path or self.default_agent_path_plan_path or None,
            "ksi_evidence": self._requested_ksi_evidence_path or self.default_ksi_evidence_path or None,
            "blocker_evidence": self._requested_blocker_evidence_path or self.default_blocker_evidence_path or None,
            "source": self._requested_source_path,
        }
        self.allowed_functions: Set[str] = set()
        self.function_arity: Dict[str, int] = {}
        self.last_results_payload: Dict[str, Any] = {}
        self.last_summary_payload: Dict[str, Any] = {}

    def validate(self) -> List[Dict[str, Any]]:
        repair_payload = self._load_json_with_fallback(
            self._requested_repair_candidates_path,
            self.default_repair_candidates_path,
            "repair_candidates",
        )
        path_plan_payload = self._load_path_plan_with_fallback()
        ksi_records = self._load_jsonl_with_fallback(
            self._requested_ksi_evidence_path,
            self.default_ksi_evidence_path,
            "ksi_evidence",
        )
        blocker_payload = self._load_json_with_fallback(
            self._requested_blocker_evidence_path,
            self.default_blocker_evidence_path,
            "blocker_evidence",
        )
        source_info = self._load_source_info(self._requested_source_path)

        candidates = repair_payload.get("candidates")
        if not isinstance(candidates, list):
            candidates = []

        sender_info = self._collect_sender_info(ksi_records)
        self.allowed_functions, self.function_arity = self._build_allowed_functions(
            repair_payload=repair_payload,
            path_plan_payload=path_plan_payload,
            ksi_records=ksi_records,
            blocker_payload=blocker_payload,
            source_info=source_info,
        )

        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            result = self._validate_candidate(candidate, sender_info)
            results.append(result)

        results.sort(key=lambda item: (_text(item.get("candidate_id")), _text(item.get("hypothesis_id"))))
        return results

    def write_results(
        self,
        results: List[Dict[str, Any]],
        filename: str = "guided_sequence_dryrun.json",
    ) -> str:
        if not self.output_dir:
            return ""
        output_path = os.path.join(self.output_dir, filename)
        payload = {
            "loaded_inputs": dict(self.loaded_inputs),
            "allowed_function_count": len(self.allowed_functions),
            "allowed_functions_preview": sorted(self.allowed_functions)[:50],
            "results": results,
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        self.last_results_payload = payload
        return output_path

    def write_summary(
        self,
        results: List[Dict[str, Any]],
        filename: str = "guided_sequence_dryrun_summary.json",
    ) -> str:
        if not self.output_dir:
            return ""
        status_counts = Counter()
        diagnosis_type_status_counts: Dict[str, Counter] = defaultdict(Counter)
        sequence_strategy_status_counts: Dict[str, Counter] = defaultdict(Counter)
        skip_reasons = Counter()
        bad_function_count = 0
        for result in results:
            status = _text(result.get("dryrun_status")) or "skipped"
            status_counts[status] += 1
            diagnosis_type = _text(result.get("diagnosis_type")) or "unknown"
            sequence_strategy = _text(result.get("sequence_strategy")) or "unknown"
            diagnosis_type_status_counts[diagnosis_type][status] += 1
            sequence_strategy_status_counts[sequence_strategy][status] += 1
            if result.get("all_functions_exist") is False:
                bad_function_count += 1
            skip_reason = _text(result.get("skip_reason"))
            if skip_reason:
                skip_reasons[skip_reason] += 1

        payload = {
            "enabled": True,
            "candidate_count": len(results),
            "mappable_count": int(status_counts.get("mappable", 0)),
            "partial_count": int(status_counts.get("partial", 0)),
            "unmappable_count": int(status_counts.get("unmappable", 0)),
            "skipped_count": int(status_counts.get("skipped", 0)),
            "bad_function_count": int(bad_function_count),
            "status_counts": dict(status_counts),
            "diagnosis_type_status_counts": {
                key: dict(counter) for key, counter in diagnosis_type_status_counts.items()
            },
            "sequence_strategy_status_counts": {
                key: dict(counter) for key, counter in sequence_strategy_status_counts.items()
            },
            "top_skip_reasons": dict(skip_reasons.most_common(10)),
            "loaded_inputs": dict(self.loaded_inputs),
            "output_path": os.path.join(self.output_dir, "guided_sequence_dryrun.json"),
        }
        output_path = os.path.join(self.output_dir, filename)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception:
            return ""
        self.last_summary_payload = payload
        return output_path

    def _load_json_with_fallback(
        self,
        requested_path: Optional[str],
        default_path: Optional[str],
        key: str,
    ) -> Dict[str, Any]:
        candidate_paths: List[str] = []
        if requested_path:
            candidate_paths.append(requested_path)
        if default_path and default_path not in candidate_paths:
            candidate_paths.append(default_path)
        self.loaded_inputs[key] = False
        self.input_paths[key] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_json(path)
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
        candidate_paths: List[str] = []
        if requested_path:
            candidate_paths.append(requested_path)
        if default_path and default_path not in candidate_paths:
            candidate_paths.append(default_path)
        self.loaded_inputs[key] = False
        self.input_paths[key] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_jsonl(path)
            if payload:
                self.loaded_inputs[key] = True
                self.input_paths[key] = path
                return payload
        return []

    def _load_path_plan_with_fallback(self) -> Dict[str, Any]:
        candidate_paths: List[str] = []
        if self._requested_path_plan_path:
            candidate_paths.append(self._requested_path_plan_path)
        if self.default_agent_path_plan_path and self.default_agent_path_plan_path not in candidate_paths:
            candidate_paths.append(self.default_agent_path_plan_path)
        if self.default_proposed_path_plan_path and self.default_proposed_path_plan_path not in candidate_paths:
            candidate_paths.append(self.default_proposed_path_plan_path)

        self.loaded_inputs["path_plan"] = False
        self.input_paths["path_plan"] = candidate_paths[0] if candidate_paths else None
        for path in candidate_paths:
            payload = _read_json(path)
            hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
            if isinstance(hypotheses, list):
                self.loaded_inputs["path_plan"] = True
                self.input_paths["path_plan"] = path
                return payload
        return {}

    def _load_source_info(self, requested_path: Optional[str]) -> Dict[str, Any]:
        self.loaded_inputs["source"] = False
        self.input_paths["source"] = requested_path
        if not requested_path or not os.path.exists(requested_path):
            return {"functions": set(), "arity": {}}
        try:
            with open(requested_path, "r", encoding="utf-8") as handle:
                source_text = handle.read()
        except Exception:
            return {"functions": set(), "arity": {}}

        self.loaded_inputs["source"] = True
        sanitized = _strip_comments(source_text)
        functions: Set[str] = set()
        arity: Dict[str, int] = {}

        for match in re.finditer(
            r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)",
            sanitized,
            flags=re.DOTALL,
        ):
            name = match.group(1).strip()
            params = match.group(2)
            functions.add(name)
            arity[name] = len(_split_params(params))

        for special_name in ("fallback", "receive"):
            if re.search(r"\b{}\s*\(".format(special_name), sanitized):
                functions.add(special_name)
                arity[special_name] = 0

        return {"functions": functions, "arity": arity}

    def _build_allowed_functions(
        self,
        repair_payload: Dict[str, Any],
        path_plan_payload: Dict[str, Any],
        ksi_records: Sequence[Dict[str, Any]],
        blocker_payload: Dict[str, Any],
        source_info: Dict[str, Any],
    ) -> Tuple[Set[str], Dict[str, int]]:
        allowed: Set[str] = set()
        arity: Dict[str, int] = {}

        functions_from_source = source_info.get("functions")
        if isinstance(functions_from_source, set):
            allowed.update(functions_from_source)
        source_arity = source_info.get("arity")
        if isinstance(source_arity, dict):
            arity.update({str(key): int(value) for key, value in source_arity.items()})

        candidates = repair_payload.get("candidates") if isinstance(repair_payload, dict) else None
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                allowed.update(_normalize_path(item.get("base_path")))
                allowed.update(_normalize_path(item.get("guided_sequence")))

        hypotheses = path_plan_payload.get("hypotheses") if isinstance(path_plan_payload, dict) else None
        if isinstance(hypotheses, list):
            for item in hypotheses:
                if not isinstance(item, dict):
                    continue
                allowed.update(_normalize_path(item.get("path")))
                sink = _text(item.get("sink"))
                if sink:
                    allowed.add(sink)

        for record in ksi_records:
            observations = record.get("observations") if isinstance(record.get("observations"), dict) else {}
            allowed.update(_normalize_path(observations.get("functions")))

        blocker_hypotheses = blocker_payload.get("hypotheses") if isinstance(blocker_payload, dict) else None
        if isinstance(blocker_hypotheses, list):
            for item in blocker_hypotheses:
                if not isinstance(item, dict):
                    continue
                allowed.update(_normalize_path(item.get("path")))
                sink = _text(item.get("sink"))
                if sink:
                    allowed.add(sink)

        allowed.update({"fallback", "receive"})
        allowed.discard("")
        return allowed, arity

    @staticmethod
    def _collect_sender_info(ksi_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        all_senders: Set[str] = set()
        success_senders: Set[str] = set()
        for record in ksi_records:
            observations = record.get("observations") if isinstance(record.get("observations"), dict) else {}
            senders = _normalize_string_list(observations.get("sender_sequence"))
            for sender in senders:
                all_senders.add(sender)
            revert = observations.get("revert") is True
            sink_ok = observations.get("sink_executed_successfully") is True or record.get("sink_executed_successfully") is True
            if not revert or sink_ok:
                for sender in senders:
                    success_senders.add(sender)
        return {
            "all_senders": all_senders,
            "success_senders": success_senders,
        }

    def _validate_candidate(self, candidate: Any, sender_info: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(candidate, dict):
            return {
                "candidate_id": None,
                "hypothesis_id": None,
                "diagnosis_type": None,
                "sequence_strategy": None,
                "guided_sequence": [],
                "dryrun_status": "skipped",
                "all_functions_exist": False,
                "missing_functions": [],
                "checks": {},
                "skip_reason": "candidate_not_dict",
                "source": "guided_sequence_dryrun_v1",
            }

        candidate_id = _text(candidate.get("candidate_id"))
        guided_sequence = _normalize_path(candidate.get("guided_sequence"))
        if not candidate_id:
            return {
                "candidate_id": None,
                "hypothesis_id": _text(candidate.get("hypothesis_id")) or None,
                "diagnosis_type": _text(candidate.get("diagnosis_type")) or None,
                "sequence_strategy": _text(candidate.get("sequence_strategy")) or None,
                "guided_sequence": guided_sequence,
                "dryrun_status": "skipped",
                "all_functions_exist": False,
                "missing_functions": [],
                "checks": {},
                "skip_reason": "missing_candidate_id",
                "source": "guided_sequence_dryrun_v1",
            }
        if not guided_sequence:
            return {
                "candidate_id": candidate_id,
                "hypothesis_id": _text(candidate.get("hypothesis_id")) or None,
                "diagnosis_type": _text(candidate.get("diagnosis_type")) or None,
                "sequence_strategy": _text(candidate.get("sequence_strategy")) or None,
                "guided_sequence": [],
                "dryrun_status": "skipped",
                "all_functions_exist": False,
                "missing_functions": [],
                "checks": {},
                "skip_reason": "empty_guided_sequence",
                "source": "guided_sequence_dryrun_v1",
            }

        sender_policy = _text(candidate.get("sender_policy"))
        value_policy = _text(candidate.get("value_policy"))
        argument_policy = _text(candidate.get("argument_policy"))
        prestate_policy = _text(candidate.get("prestate_policy"))
        target_function = _text(candidate.get("target_function")) or (guided_sequence[-1] if guided_sequence else "")

        missing_functions = [item for item in guided_sequence if item not in self.allowed_functions]
        all_functions_exist = len(missing_functions) == 0

        function_check = self._check_functions(guided_sequence, missing_functions)
        sender_policy_check = self._check_sender_policy(sender_policy, sender_info)
        value_policy_check = self._check_value_policy(value_policy)
        argument_policy_check = self._check_argument_policy(argument_policy, target_function)
        prestate_policy_check = self._check_prestate_policy(
            prestate_policy=prestate_policy,
            guided_sequence=guided_sequence,
            target_function=target_function,
        )

        checks = {
            "function_check": function_check,
            "sender_policy_check": sender_policy_check,
            "value_policy_check": value_policy_check,
            "argument_policy_check": argument_policy_check,
            "prestate_policy_check": prestate_policy_check,
        }

        statuses = [check.get("status") for check in checks.values()]
        if "unmappable" in statuses:
            dryrun_status = "unmappable"
        elif "partial" in statuses:
            dryrun_status = "partial"
        else:
            dryrun_status = "mappable"

        return {
            "candidate_id": candidate_id,
            "hypothesis_id": _text(candidate.get("hypothesis_id")) or None,
            "diagnosis_type": _text(candidate.get("diagnosis_type")) or None,
            "sequence_strategy": _text(candidate.get("sequence_strategy")) or None,
            "guided_sequence": guided_sequence,
            "dryrun_status": dryrun_status,
            "all_functions_exist": all_functions_exist,
            "missing_functions": missing_functions,
            "checks": checks,
            "skip_reason": None,
            "source": "guided_sequence_dryrun_v1",
        }

    @staticmethod
    def _check_functions(guided_sequence: Sequence[str], missing_functions: Sequence[str]) -> Dict[str, Any]:
        if not guided_sequence:
            return {"status": "skipped", "reason": "repair sequence is empty"}
        if missing_functions:
            return {
                "status": "unmappable",
                "reason": "missing functions in allowed function set: {}".format(", ".join(missing_functions)),
            }
        return {
            "status": "mappable",
            "reason": "all functions appear in allowed function set",
        }

    @staticmethod
    def _check_sender_policy(sender_policy: str, sender_info: Dict[str, Any]) -> Dict[str, Any]:
        all_senders = sender_info.get("all_senders", set()) if isinstance(sender_info, dict) else set()
        success_senders = sender_info.get("success_senders", set()) if isinstance(sender_info, dict) else set()

        if sender_policy == "any_existing_sender":
            if all_senders:
                return {"status": "mappable", "reason": "existing sender pool is available"}
            return {"status": "partial", "reason": "existing sender pool was not confirmed from evidence"}

        if sender_policy == "try_owner_or_deployer":
            return {
                "status": "partial",
                "reason": "owner or deployer identity is not confirmed from current evidence",
            }

        if sender_policy == "try_owner_or_deployer_or_previous_success_sender":
            if success_senders:
                return {
                    "status": "partial",
                    "reason": "previous successful senders exist, but owner or deployer identity is not confirmed",
                }
            return {
                "status": "partial",
                "reason": "previous successful senders and owner/deployer identity are not fully confirmed",
            }

        if sender_policy == "keep_same_sender_across_sequence":
            if all_senders:
                return {
                    "status": "mappable",
                    "reason": "a stable sender can be reused across the sequence",
                }
            return {
                "status": "partial",
                "reason": "same-sender policy is conceptually supported, but no sender evidence was loaded",
            }

        if sender_policy == "rotate_existing_senders":
            if all_senders:
                return {
                    "status": "mappable",
                    "reason": "existing senders can be rotated from the observed sender pool",
                }
            return {
                "status": "partial",
                "reason": "sender rotation is conceptually supported, but no sender pool was confirmed",
            }

        if sender_policy == "keep_original_sender":
            return {
                "status": "mappable",
                "reason": "original sender policy can be preserved conceptually",
            }

        if sender_policy == "keep_prefix_sender_stable":
            return {
                "status": "mappable",
                "reason": "prefix sender stability can be preserved conceptually",
            }

        return {
            "status": "partial",
            "reason": "sender policy is not explicitly modeled by the dry-run mapper",
        }

    @staticmethod
    def _check_value_policy(value_policy: str) -> Dict[str, Any]:
        if value_policy in (
            "keep_original",
            "try_small_nonzero_value",
            "force_zero_value",
            "try_zero_and_small_nonzero",
        ):
            return {
                "status": "mappable",
                "reason": "{} can be represented with the existing value model".format(value_policy),
            }
        return {
            "status": "partial",
            "reason": "value policy is not explicitly modeled by the dry-run mapper",
        }

    def _check_argument_policy(self, argument_policy: str, target_function: str) -> Dict[str, Any]:
        if not argument_policy:
            return {
                "status": "partial",
                "reason": "argument policy is missing",
            }
        supported = {
            "mutate_sink_arguments",
            "mutate_frontier_arguments",
            "mutate_transfer_arguments",
            "try_boolean_true_false",
            "try_zero_one_max_boundary_values",
        }
        if argument_policy not in supported:
            return {
                "status": "partial",
                "reason": "argument policy is not explicitly modeled by the dry-run mapper",
            }
        if target_function in self.function_arity:
            return {
                "status": "mappable",
                "reason": "argument mutation is conceptually supported and source-derived arity is available",
            }
        return {
            "status": "partial",
            "reason": "argument mutation policy is supported conceptually, but source-derived arity was not confirmed",
        }

    def _check_prestate_policy(
        self,
        prestate_policy: str,
        guided_sequence: Sequence[str],
        target_function: str,
    ) -> Dict[str, Any]:
        lowered_sequence = [item.lower() for item in guided_sequence]
        allowed_lower = {item.lower() for item in self.allowed_functions}

        if prestate_policy == "none":
            return {
                "status": "mappable",
                "reason": "no additional pre-state requirements are declared",
            }

        if prestate_policy == "permission_setup":
            if guided_sequence and _function_matches(guided_sequence[0], PERMISSION_KEYWORDS):
                return {
                    "status": "mappable",
                    "reason": "repair sequence begins with an observed permission-like function",
                }
            return {
                "status": "partial",
                "reason": "permission setup is conceptually supported, but a strong permission-like setup function was not confirmed",
            }

        if prestate_policy == "insert_approve_before_transferFrom":
            has_approve = "approve" in lowered_sequence or "approve" in allowed_lower
            has_transfer = "transferfrom" in lowered_sequence or "transferfrom" in allowed_lower
            if has_approve and has_transfer:
                return {
                    "status": "mappable",
                    "reason": "approve and transferFrom are both present in the sequence or allowed function set",
                }
            return {
                "status": "unmappable",
                "reason": "approve-before-transferFrom requires approve and transferFrom, but one of them is missing",
            }

        if prestate_policy == "insert_trading_enable":
            trading_functions = [item for item in self.allowed_functions if _function_matches(item, TRADING_KEYWORDS)]
            config_functions = [item for item in self.allowed_functions if _function_matches(item, CONFIG_KEYWORDS)]
            if trading_functions and config_functions:
                return {
                    "status": "mappable",
                    "reason": "observed function set contains trading-related setup functions",
                }
            if trading_functions:
                return {
                    "status": "partial",
                    "reason": "trading-related functions exist, but a concrete setup shape was not fully confirmed",
                }
            return {
                "status": "partial",
                "reason": "no explicit trading-enable function was confirmed in the allowed function set",
            }

        if prestate_policy == "insert_config_prestate":
            config_present = any(_function_matches(item, CONFIG_KEYWORDS) for item in guided_sequence) or any(
                _function_matches(item, CONFIG_KEYWORDS) for item in self.allowed_functions
            )
            if config_present:
                return {
                    "status": "mappable",
                    "reason": "observed function set contains configuration-like functions",
                }
            return {
                "status": "partial",
                "reason": "config pre-state is conceptually supported, but no strong config-like function was confirmed",
            }

        return {
            "status": "partial",
            "reason": "pre-state policy is not explicitly modeled by the dry-run mapper",
        }

