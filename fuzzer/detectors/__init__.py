#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

from utils.utils import initialize_logger, print_individual_solution_as_transaction

from .arbitrary_memory_access import ArbitraryMemoryAccessDetector
from .assertion_failure import AssertionFailureDetector
from .block_dependency import BlockDependencyDetector
from .integer_overflow import IntegerOverflowDetector
from .leaking_ether import LeakingEtherDetector
from .locking_ether import LockingEtherDetector
from .reentrancy import ReentrancyDetector
from .transaction_order_dependency import TransactionOrderDependencyDetector
from .unchecked_return_value import UncheckedReturnValueDetector
from .unprotected_selfdestruct import UnprotectedSelfdestructDetector
from .unsafe_delegatecall import UnsafeDelegatecallDetector


class DetectorExecutor:
    def __init__(self, source_map=None, function_signature_mapping=None):
        self.source_map = source_map
        self.function_signature_mapping = function_signature_mapping or {}
        self.logger = initialize_logger("Detector")

        self.integer_overflow_detector = IntegerOverflowDetector()
        self.assertion_failure_detector = AssertionFailureDetector()
        self.arbitrary_memory_access_detector = ArbitraryMemoryAccessDetector()
        self.reentrancy_detector = ReentrancyDetector()
        self.transaction_order_dependency_detector = TransactionOrderDependencyDetector()
        self.block_dependency_detector = BlockDependencyDetector()
        self.unchecked_return_value_detector = UncheckedReturnValueDetector()
        self.unsafe_delegatecall_detector = UnsafeDelegatecallDetector()
        self.leaking_ether_detector = LeakingEtherDetector()
        self.locking_ether_detector = LockingEtherDetector()
        self.unprotected_selfdestruct_detector = UnprotectedSelfdestructDetector()

    def initialize_detectors(self):
        self.integer_overflow_detector.init()
        self.assertion_failure_detector.init()
        self.arbitrary_memory_access_detector.init()
        self.reentrancy_detector.init()
        self.transaction_order_dependency_detector.init()
        self.block_dependency_detector.init()
        self.unchecked_return_value_detector.init()
        self.unsafe_delegatecall_detector.init()
        self.leaking_ether_detector.init()
        self.locking_ether_detector.init()
        self.unprotected_selfdestruct_detector.init()

    @staticmethod
    def error_exists(errors, error_type):
        for error in errors:
            if error["type"] == error_type:
                return True
        return False

    @staticmethod
    def normalize_function_selector(data):
        if not data:
            return "fallback"
        normalized = data[2:] if data.startswith("0x") else data
        normalized = normalized[:8]
        return normalized if normalized else "fallback"

    @staticmethod
    def resolve_triggering_function(function_signature_mapping, individual, transaction_index):
        if transaction_index is None or transaction_index >= len(individual.solution):
            return {}
        transaction = individual.solution[transaction_index]["transaction"]
        selector = DetectorExecutor.normalize_function_selector(transaction.get("data", ""))
        signature = function_signature_mapping.get(selector) or function_signature_mapping.get("0x" + selector)
        if selector == "fallback" and not signature:
            signature = "fallback"
        return {
            "triggering_transaction_index": transaction_index,
            "triggering_function_selector": selector,
            "triggering_function_signature": signature,
            "triggering_function_name": signature.split("(")[0] if signature and "(" in signature else signature,
        }

    @staticmethod
    def add_error(errors, pc, error_type, individual, mfe, detector, source_map, function_signature_mapping=None, transaction_index=None):
        elapsed = time.time() - mfe.execution_begin
        error = {
            "swc_id": detector.swc_id,
            "severity": detector.severity,
            "type": error_type,
            "individual": individual.solution,
            "time": elapsed,
            "pc": hex(pc),
        }
        if function_signature_mapping is None:
            function_signature_mapping = {}
        error.update(
            DetectorExecutor.resolve_triggering_function(
                function_signature_mapping,
                individual,
                transaction_index,
            )
        )
        if source_map and source_map.get_buggy_line(pc):
            error["line"] = source_map.get_location(pc)["begin"]["line"] + 1
            error["column"] = source_map.get_location(pc)["begin"]["column"] + 1
            error["source_code"] = source_map.get_buggy_line(pc)
        if mfe.results.get("first_bug_time") is None or elapsed < mfe.results["first_bug_time"]:
            mfe.results["first_bug_time"] = elapsed
        if pc not in errors:
            errors[pc] = [error]
            return True
        if not DetectorExecutor.error_exists(errors[pc], error_type):
            errors[pc].append(error)
            return True
        return False

    @staticmethod
    def get_color_for_severity(severity):
        if severity == "High":
            return "\u001b[31m"
        if severity == "Medium":
            return "\u001b[33m"
        if severity == "Low":
            return "\u001b[32m"
        return ""

    def _log_detection(self, detector, title, pc, individual, index):
        color = DetectorExecutor.get_color_for_severity(detector.severity)
        self.logger.title(color + "-----------------------------------------------------")
        self.logger.title(color + title)
        self.logger.title(color + "-----------------------------------------------------")
        self.logger.title(color + "SWC-ID:   " + str(detector.swc_id))
        self.logger.title(color + "Severity: " + detector.severity)
        self.logger.title(color + "-----------------------------------------------------")
        if self.source_map and self.source_map.get_buggy_line(pc):
            self.logger.title(color + "Source code line:")
            self.logger.title(color + "-----------------------------------------------------")
            line = self.source_map.get_location(pc)["begin"]["line"] + 1
            column = self.source_map.get_location(pc)["begin"]["column"] + 1
            self.logger.title(color + self.source_map.source.filename + ":" + str(line) + ":" + str(column))
            self.logger.title(color + self.source_map.get_buggy_line(pc))
            self.logger.title(color + "-----------------------------------------------------")
        self.logger.title(color + "Transaction sequence:")
        self.logger.title(color + "-----------------------------------------------------")
        print_individual_solution_as_transaction(
            self.logger,
            individual.solution,
            color,
            self.function_signature_mapping,
            index,
        )

    def run_detectors(self, previous_instruction, current_instruction, errors, tainted_record, individual, mfe, previous_branch, transaction_index):
        pc, index = self.arbitrary_memory_access_detector.detect_arbitrary_memory_access(
            tainted_record,
            individual,
            current_instruction,
            transaction_index,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Arbitrary Memory Access",
            individual,
            mfe,
            self.arbitrary_memory_access_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.arbitrary_memory_access_detector,
                "      !!! Arbitrary memory access detected !!!       ",
                pc,
                individual,
                index,
            )

        pc, index = self.assertion_failure_detector.detect_assertion_failure(current_instruction, transaction_index)
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Assertion Failure",
            individual,
            mfe,
            self.assertion_failure_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.assertion_failure_detector,
                "          !!! Assertion failure detected !!!         ",
                pc,
                individual,
                index,
            )

        pc, index, overflow_type = self.integer_overflow_detector.detect_integer_overflow(
            mfe,
            tainted_record,
            previous_instruction,
            current_instruction,
            individual,
            transaction_index,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Integer Overflow",
            individual,
            mfe,
            self.integer_overflow_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            if overflow_type == "overflow":
                title = "          !!! Integer overflow detected !!!          "
            else:
                title = "         !!! Integer underflow detected !!!          "
            self._log_detection(self.integer_overflow_detector, title, pc, individual, index)

        pc, index = self.reentrancy_detector.detect_reentrancy(tainted_record, current_instruction, transaction_index)
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Reentrancy",
            individual,
            mfe,
            self.reentrancy_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.reentrancy_detector,
                "            !!! Reentrancy detected !!!              ",
                pc,
                individual,
                index,
            )

        pc, index = self.transaction_order_dependency_detector.detect_transaction_order_dependency(
            current_instruction,
            tainted_record,
            individual,
            transaction_index,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Transaction Order Dependency",
            individual,
            mfe,
            self.transaction_order_dependency_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.transaction_order_dependency_detector,
                "    !!! Transaction order dependency detected !!!    ",
                pc,
                individual,
                index,
            )

        pc, index = self.block_dependency_detector.detect_block_dependency(
            tainted_record,
            current_instruction,
            previous_branch,
            transaction_index,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Block Dependency",
            individual,
            mfe,
            self.block_dependency_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.block_dependency_detector,
                "          !!! Block dependency detected !!!          ",
                pc,
                individual,
                index,
            )

        pc, index = self.unchecked_return_value_detector.detect_unchecked_return_value(
            previous_instruction,
            current_instruction,
            tainted_record,
            transaction_index,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Unchecked Return Value",
            individual,
            mfe,
            self.unchecked_return_value_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.unchecked_return_value_detector,
                "        !!! Unchecked return value detected !!!      ",
                pc,
                individual,
                index,
            )

        pc, index = self.unsafe_delegatecall_detector.detect_unsafe_delegatecall(
            current_instruction,
            tainted_record,
            individual,
            previous_instruction,
            transaction_index,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Unsafe Delegatecall",
            individual,
            mfe,
            self.unsafe_delegatecall_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.unsafe_delegatecall_detector,
                "        !!! Unsafe delegatecall detected !!!         ",
                pc,
                individual,
                index,
            )

        pc, index = self.leaking_ether_detector.detect_leaking_ether(
            current_instruction,
            tainted_record,
            individual,
            transaction_index,
            previous_branch,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Leaking Ether",
            individual,
            mfe,
            self.leaking_ether_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.leaking_ether_detector,
                "            !!! Leaking ether detected !!!           ",
                pc,
                individual,
                index,
            )

        pc, index = self.locking_ether_detector.detect_locking_ether(
            mfe.cfg,
            current_instruction,
            individual,
            transaction_index,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Locking Ether",
            individual,
            mfe,
            self.locking_ether_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.locking_ether_detector,
                "            !!! Locking ether detected !!!           ",
                pc,
                individual,
                index,
            )

        pc, index = self.unprotected_selfdestruct_detector.detect_unprotected_selfdestruct(
            current_instruction,
            tainted_record,
            individual,
            transaction_index,
        )
        if pc and DetectorExecutor.add_error(
            errors,
            pc,
            "Unprotected Selfdestruct",
            individual,
            mfe,
            self.unprotected_selfdestruct_detector,
            self.source_map,
            self.function_signature_mapping,
            index,
        ):
            self._log_detection(
                self.unprotected_selfdestruct_detector,
                "      !!! Unprotected selfdestruct detected !!!      ",
                pc,
                individual,
                index,
            )

    def analyze_transaction(self, transaction, result, errors, individual, mfe, transaction_index):
        return None

    def finalize_results(self, results):
        return results


__all__ = ["DetectorExecutor"]
