#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import shlex
import solcx
import logging
import subprocess
import importlib.util
import hashlib
from pathlib import Path

from solcx.install import _convert_and_validate_version
try:
    import eth_utils
except ModuleNotFoundError:
    eth_utils = None
try:
    from web3 import Web3
except ModuleNotFoundError:
    class Web3:
        @staticmethod
        def keccak(text=None, **_kwargs):
            return hashlib.sha3_256((text or "").encode("utf-8")).digest()

        sha3 = keccak
from .settings import LOGGING_LEVEL


def initialize_logger(name):
    logger = logging.getLogger(name)
    logger.title = lambda *a: logger.info(*[bold(x) for x in a])
    logger_error = logger.error
    logger.error = lambda *a: logger_error(*[red(bold(x)) for x in a])
    logger_warning = logger.warning
    logger.warning = lambda *a: logger_warning(*[red(bold(x)) for x in a])
    logger.setLevel(level=LOGGING_LEVEL)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logger


def bold(x):
    return "".join(['\033[1m', x, '\033[0m']) if isinstance(x, str) else x


def red(x):
    return "".join(['\033[91m', x, '\033[0m']) if isinstance(x, str) else x


def code_bool(value: bool):
    return str(int(value)).zfill(64)


def code_uint(value):
    return hex(value).replace("0x", "").zfill(64)


def code_int(value):
    return hex(value).replace("0x", "").zfill(64)


def code_address(value):
    return value.zfill(64)


def code_bytes(value):
    return value.ljust(64, "0")


def code_type(value, type):
    if type == "bool":
        return code_bool(value)
    elif type.startswith("uint"):
        return code_uint(value)
    elif type.startswith("int"):
        return code_int(value)
    elif type == "address":
        return code_address(value)
    elif type.startswith("bytes"):
        return code_bytes(value)
    else:
        raise Exception()


def run_command(cmd):
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=FNULL)
    return p.communicate()[0]


def _keccak_text(text):
    if hasattr(Web3, "keccak"):
        return Web3.keccak(text=text)
    return Web3.sha3(text=text)


def _compile_with_forge_support(solc_version, evm_version, source_code_file, outputs, optimizer_runs=200, source_code=None):
    try:
        current = Path(__file__).resolve()
        for parent in current.parents:
            support_path = parent / "forge_compile_support.py"
            if support_path.exists():
                spec = importlib.util.spec_from_file_location("forge_compile_support", support_path)
                support = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(support)
                out, _, _ = support.compile_standard_resolved(
                    source_code_file,
                    solc_version,
                    evm_version=evm_version,
                    outputs=outputs,
                    optimizer_runs=optimizer_runs,
                    source_override=source_code,
                )
                return out
    except Exception:
        return None
    return None


def compile(solc_version, evm_version, source_code_file):
    out = None
    source_code = ""
    with open(source_code_file, 'r', encoding='utf-8', errors='ignore') as file:
        source_code = file.read()
    outputs = [
        "abi",
        "evm.deployedBytecode",
        "evm.bytecode.object",
        "evm.legacyAssembly",
    ]
    out = _compile_with_forge_support(solc_version, evm_version, source_code_file, outputs, source_code=source_code)
    if out is not None:
        return out

    def build_compile_input(extra_settings=None):
        settings_payload = {
            "optimizer": {
                "enabled": True,
                "runs": 200,
            },
            "evmVersion": evm_version,
            "outputSelection": {
                source_code_file: {
                    "*":
                        outputs,
                }
            }
        }
        if extra_settings:
            settings_payload.update(extra_settings)
        return {
            'language': 'Solidity',
            'sources': {source_code_file: {'content': source_code}},
            'settings': settings_payload,
        }

    try:
        solc_version = str(solc_version or "").strip()
        if not solc_version or solc_version.lower() in {"n/a", "na", "none", "null", "unknown"}:
            raise ValueError("missing or placeholder solc version")
        if not solc_version.startswith("v"):
            solc_version = "v" + solc_version
        solc_version = _convert_and_validate_version(solc_version)
        if not solc_version in solcx.get_installed_solc_versions():
            solcx.install_solc(solc_version)
        solcx.set_solc_version(solc_version, True)
        out = solcx.compile_standard(
            build_compile_input(),
            allow_paths='.',
        )
    except Exception as e:
        message = getattr(e, "message", str(e))
        if "Stack too deep" in message:
            try:
                out = solcx.compile_standard(
                    build_compile_input(
                        {
                            "optimizer": {
                                "enabled": True,
                                "runs": 200
                            },
                            "viaIR": True,
                        }
                    ),
                    allow_paths='.',
                )
            except Exception as fallback_error:
                print("Error: Solidity compilation failed!")
                print(getattr(fallback_error, "message", str(fallback_error)))
        else:
            print("Error: Solidity compilation failed!")
            print(message)
    return out


def get_interface_from_abi(abi):
    interface = {}
    interface_mapper = {}  # 记录函数名和函数签名的映射
    for field in abi:
        if field['type'] == 'function':
            function_name = field['name']
            function_inputs = []
            signature = function_name + '('
            for i in range(len(field['inputs'])):
                input_type = field['inputs'][i]['type']
                function_inputs.append(input_type)
                signature += input_type
                if i < len(field['inputs']) - 1:
                    signature += ','
            signature += ')'
            hash = _keccak_text(signature)[0:4].hex()
            interface[hash] = function_inputs
            interface_mapper[signature] = hash
        elif field['type'] == 'constructor':
            function_inputs = []
            for i in range(len(field['inputs'])):
                input_type = field['inputs'][i]['type']
                function_inputs.append(input_type)
            interface['constructor'] = function_inputs
    if not "fallback" in interface:
        interface["fallback"] = []
    return interface, interface_mapper


def get_function_state_mutability_from_abi(abi):
    mapping = {}
    if not isinstance(abi, list):
        return mapping

    name_mutability = {}
    for field in abi:
        if not isinstance(field, dict):
            continue
        field_type = str(field.get("type") or "").strip()
        state_mutability = str(field.get("stateMutability") or "").strip().lower()
        if not state_mutability:
            state_mutability = "payable" if field_type == "receive" else "nonpayable"

        if field_type == "function":
            function_name = str(field.get("name") or "").strip()
            if not function_name:
                continue
            inputs = field.get("inputs") or []
            signature = function_name + '('
            for i in range(len(inputs)):
                input_item = inputs[i] if isinstance(inputs[i], dict) else {}
                input_type = str(input_item.get("type") or "").strip()
                signature += input_type
                if i < len(inputs) - 1:
                    signature += ','
            signature += ')'
            selector = _keccak_text(signature)[0:4].hex()
            mapping[signature] = state_mutability
            mapping[selector] = state_mutability
            existing = name_mutability.get(function_name)
            if existing is None:
                name_mutability[function_name] = state_mutability
            elif existing != state_mutability:
                name_mutability[function_name] = ""
        elif field_type in {"fallback", "receive"}:
            mapping[field_type] = state_mutability

    for function_name, state_mutability in name_mutability.items():
        if state_mutability:
            mapping[function_name] = state_mutability
    return mapping


def get_function_signature_mapping(abi):
    mapping = {}
    for field in abi:
        if field['type'] == 'function':
            function_name = field['name']
            function_inputs = []
            signature = function_name + '('
            for i in range(len(field['inputs'])):
                input_type = field['inputs'][i]['type']
                signature += input_type
                if i < len(field['inputs']) - 1:
                    signature += ','
            signature += ')'
            hash = _keccak_text(signature)[0:4].hex()
            mapping[hash] = signature
    if not "fallback" in mapping:
        mapping["fallback"] = "fallback"
    return mapping


def remove_swarm_hash(bytecode):
    if isinstance(bytecode, str):
        if bytecode.endswith("0029"):
            bytecode = re.sub(r"a165627a7a72305820\S{64}0029$", "", bytecode)
        if bytecode.endswith("0033"):
            bytecode = re.sub(r"5056fe.*?0033$", "5056", bytecode)
    return bytecode


def get_pcs_and_jumpis(bytecode):
    bytecode = bytes.fromhex(remove_swarm_hash(bytecode).replace("0x", ""))
    i = 0
    pcs = []
    jumpis = []
    while i < len(bytecode):
        opcode = bytecode[i]
        pcs.append(i)
        if opcode == 87:  # JUMPI
            jumpis.append(hex(i))
        if opcode >= 96 and opcode <= 127:  # PUSH
            size = opcode - 96 + 1
            i += size
        i += 1
    if len(pcs) == 0:
        pcs = [0]
    return (pcs, jumpis)


def convert_stack_value_to_int(stack_value):
    if isinstance(stack_value, int):
        return stack_value
    if isinstance(stack_value, (bytes, bytearray)):
        return int.from_bytes(stack_value, "big")
    if isinstance(stack_value, (list, tuple)) and len(stack_value) >= 2:
        if stack_value[0] == int:
            return stack_value[1]
        if stack_value[0] == bytes:
            return int.from_bytes(stack_value[1], "big")
    raise Exception("Error: Cannot convert stack value to int. Unknown type: " + str(type(stack_value)))


def convert_stack_value_to_hex(stack_value):
    if isinstance(stack_value, int):
        return hex(stack_value).replace("0x", "").zfill(64)
    if isinstance(stack_value, (bytes, bytearray)):
        return bytes(stack_value).hex().zfill(64)
    if isinstance(stack_value, (list, tuple)) and len(stack_value) >= 2:
        if stack_value[0] == int:
            return hex(stack_value[1]).replace("0x", "").zfill(64)
        if stack_value[0] == bytes:
            return stack_value[1].hex().zfill(64)
    raise Exception("Error: Cannot convert stack value to hex. Unknown type: " + str(type(stack_value)))


def is_fixed(value):
    return isinstance(value, int)


def split_len(seq, length):
    return [seq[i:i + length] for i in range(0, len(seq), length)]


def print_individual_solution_as_transaction(logger, individual_solution, color="", function_signature_mapping={}, transaction_index=None):
    for index, input in enumerate(individual_solution):
        transaction = input["transaction"]
        if not transaction["to"] == None:
            if transaction["data"].startswith("0x"):
                hash = transaction["data"][0:10]
            else:
                hash = transaction["data"][0:8]
            if len(individual_solution) == 1 or (transaction_index != None and transaction_index == 0):
                if hash in function_signature_mapping:
                    logger.title(color + "Transaction - " + function_signature_mapping[hash] + ":")
                else:
                    logger.title(color + "Transaction:")
            else:
                if hash in function_signature_mapping:
                    logger.title(color + "Transaction " + str(index + 1) + " - " + function_signature_mapping[hash] + ":")
                else:
                    logger.title(color + "Transaction " + str(index + 1) + ":")
            logger.title(color + "-----------------------------------------------------")
            logger.title(color + "From:      " + transaction["from"])
            logger.title(color + "To:        " + str(transaction["to"]))
            logger.title(color + "Value:     " + str(transaction["value"]) + " Wei")
            logger.title(color + "Gas Limit: " + str(transaction["gaslimit"]))
            i = 0
            for data in split_len("0x" + transaction["data"].replace("0x", ""), 42):
                if i == 0:
                    logger.title(color + "Input:     " + str(data))
                else:
                    logger.title(color + "           " + str(data))
                i += 1
            logger.title(color + "-----------------------------------------------------")
            if transaction_index != None and index + 1 > transaction_index:
                break


def normalize_32_byte_hex_address(value):
    as_bytes = eth_utils.to_bytes(hexstr=value)
    return eth_utils.to_normalized_address(as_bytes[-20:])
