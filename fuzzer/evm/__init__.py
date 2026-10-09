#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import string
import sys
import pickle
import logging
import time
from typing import Any, Dict, List

from eth import Chain, constants
from eth.chains.mainnet import (
    MAINNET_GENESIS_HEADER,
    HOMESTEAD_MAINNET_BLOCK,
    TANGERINE_WHISTLE_MAINNET_BLOCK,
    SPURIOUS_DRAGON_MAINNET_BLOCK,
    BYZANTIUM_MAINNET_BLOCK,
    PETERSBURG_MAINNET_BLOCK,
    ISTANBUL_MAINNET_BLOCK,
    BERLIN_MAINNET_BLOCK,
    LONDON_MAINNET_BLOCK
)
from eth.constants import ZERO_ADDRESS, CREATE_CONTRACT_ADDRESS
from eth.db.atomic import AtomicDB
from eth.db.backends.memory import MemoryDB
from eth.rlp.accounts import Account
from eth.rlp.headers import BlockHeader
try:
    from eth.tools.logging import DEBUG2_LEVEL_NUM
except ImportError:
    DEBUG2_LEVEL_NUM = logging.DEBUG
from eth.validation import validate_uint256
from eth.vm.spoof import SpoofTransaction
from eth_utils import to_canonical_address, decode_hex, encode_hex
from web3 import HTTPProvider
from web3 import Web3

from .storage_emulation import (
    FrontierVMForFuzzTesting,
    HomesteadVMForFuzzTesting,
    TangerineWhistleVMForFuzzTesting,
    SpuriousDragonVMForFuzzTesting,
    ByzantiumVMForFuzzTesting,
    PetersburgVMForFuzzTesting,
    IstanbulVMForFuzzTesting,
    BerlinVMForFuzzTesting,
    LondonVMForFuzzTesting
)

# 获取根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../'
# 将根目录添加到path中
sys.path.append(BASE_DIR)
from fuzzer.utils import settings
from fuzzer.utils.utils import initialize_logger
try:
    from eth_abi import encode_abi
except ImportError:
    from eth_abi import encode as encode_abi


def _diagnostic_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [_diagnostic_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _diagnostic_jsonable(item) for key, item in value.items()}
    return str(value)


def _diagnostic_error_payload(error: Any) -> Dict[str, Any]:
    if error is None:
        return {"type": "", "repr": "", "text": "", "bytes_hex": ""}
    payload = {
        "type": type(error).__name__,
        "repr": repr(error),
        "text": str(error),
        "bytes_hex": "",
    }
    if isinstance(error, (bytes, bytearray)):
        payload["bytes_hex"] = "0x" + bytes(error).hex()
    return payload


_UNKNOWN_CONSTRUCTOR_ADDRESS_BASE = int("1000000000000000000000000000000000000000", 16)

_STANDARD_DEX_FIXTURE_ADDRESSES = {
    "DexRouterMock": [
        "0x10ED43C718714eb63d5aA57B78B54704E256024E",
        "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        "0xD99D1c33F9fC3444f8101754aBC46c52416550D1",
        "0xCc7aDc94F3D80127849D2b41b6439b7CF1eB4Ae0",
    ],
    "DexFactoryMock": [
        "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
        "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
        "0x6725F303b657a9451d8BA641348b6761A6CC7a17",
    ],
    "DexPairMock": [
        "0x1111111111111111111111111111111111111111",
        "0x2222222222222222222222222222222222222222",
    ],
    "WrappedNativeMock": [
        "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "0xae13d989daC2f0dEbFf460aC112a837C89BAa7cd",
    ],
}

_STANDARD_FIXTURE_BYTECODE_CACHE = None


def _address_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return "0x%040x" % value
    if isinstance(value, (bytes, bytearray)):
        return encode_hex(value)
    text = str(value).strip()
    if text and not text.startswith("0x") and len(text) == 40:
        text = "0x" + text
    return text


def _is_address_like_type(param_type: str) -> bool:
    text = str(param_type or "").strip()
    return text in {"address", "address payable", "contract"} or text.startswith("contract ")


def _array_base_type(param_type: str) -> str:
    text = str(param_type or "").strip()
    while text.endswith("]") and "[" in text:
        text = text[:text.rfind("[")].strip()
    return text


def _default_array_lengths(param_type: str) -> List[int]:
    text = str(param_type or "").strip()
    lengths: List[int] = []
    while text.endswith("]") and "[" in text:
        open_index = text.rfind("[")
        raw = text[open_index + 1:-1].strip()
        if raw.isdigit():
            lengths.insert(0, max(0, int(raw)))
        else:
            lengths.insert(0, 1)
        text = text[:open_index].strip()
    return lengths


def _build_default_array_value(base_type: str, lengths: List[int], address_factory, ordinal_seed: int = 0):
    if not lengths:
        normalized = str(base_type or "").strip()
        if _is_address_like_type(normalized):
            return address_factory(ordinal_seed)
        if normalized.startswith("uint") or normalized.startswith("int"):
            return 0
        if normalized == "bool":
            return False
        if normalized == "string":
            return ""
        if normalized == "bytes":
            return bytearray()
        if normalized.startswith("bytes"):
            try:
                size = int(normalized[5:])
            except Exception:
                size = 0
            return bytearray(0 for _ in range(max(0, size)))
        return None
    count = max(0, int(lengths[0]))
    return [
        _build_default_array_value(base_type, lengths[1:], address_factory, ordinal_seed + idx)
        for idx in range(count)
    ]


def _compile_standard_fixture_bytecodes() -> Dict[str, Any]:
    global _STANDARD_FIXTURE_BYTECODE_CACHE
    if _STANDARD_FIXTURE_BYTECODE_CACHE is not None:
        return _STANDARD_FIXTURE_BYTECODE_CACHE
    payload: Dict[str, Any] = {
        "ok": False,
        "fixture_source": "",
        "solc_version": "",
        "bytecodes": {},
        "error": "",
    }
    fixture_source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dependencies",
        "mocks",
        "DexMocks.sol",
    )
    payload["fixture_source"] = fixture_source
    try:
        from solcx import compile_standard, get_installed_solc_versions, set_solc_version

        installed = [str(version) for version in get_installed_solc_versions()]
        preferred = "0.8.10" if "0.8.10" in installed else next((version for version in installed if version.startswith("0.8.")), "")
        if not preferred:
            raise RuntimeError("no local solc 0.8.x available for standard fixture compilation")
        set_solc_version(preferred)
        with open(fixture_source, "r", encoding="utf-8") as handle:
            source_text = handle.read()
        compiled = compile_standard(
            {
                "language": "Solidity",
                "sources": {"DexMocks.sol": {"content": source_text}},
                "settings": {
                    "evmVersion": "istanbul",
                    "optimizer": {"enabled": False},
                    "outputSelection": {
                        "*": {
                            "*": ["evm.deployedBytecode.object"],
                        }
                    },
                },
            }
        )
        contracts = compiled.get("contracts", {}).get("DexMocks.sol", {})
        bytecodes = {}
        for contract_name in _STANDARD_DEX_FIXTURE_ADDRESSES:
            bytecode = (
                contracts.get(contract_name, {})
                .get("evm", {})
                .get("deployedBytecode", {})
                .get("object", "")
            )
            if bytecode:
                bytecodes[contract_name] = decode_hex("0x" + bytecode)
        payload.update({"ok": bool(bytecodes), "solc_version": preferred, "bytecodes": bytecodes})
    except Exception as exc:
        payload["error"] = repr(exc)
    _STANDARD_FIXTURE_BYTECODE_CACHE = payload
    return payload


class InstrumentedEVM:
    def __init__(self, eth_node_ip=None, eth_node_port=None) -> None:
        chain_class = Chain.configure(
            __name__='Blockchain',
            vm_configuration=(
                (constants.GENESIS_BLOCK_NUMBER, FrontierVMForFuzzTesting),
                (HOMESTEAD_MAINNET_BLOCK, HomesteadVMForFuzzTesting),
                (TANGERINE_WHISTLE_MAINNET_BLOCK, TangerineWhistleVMForFuzzTesting),
                (SPURIOUS_DRAGON_MAINNET_BLOCK, SpuriousDragonVMForFuzzTesting),
                (BYZANTIUM_MAINNET_BLOCK, ByzantiumVMForFuzzTesting),
                (PETERSBURG_MAINNET_BLOCK, PetersburgVMForFuzzTesting),
                (ISTANBUL_MAINNET_BLOCK, IstanbulVMForFuzzTesting),
                (BERLIN_MAINNET_BLOCK, BerlinVMForFuzzTesting),
                (LONDON_MAINNET_BLOCK, LondonVMForFuzzTesting),
            ),
        )

        class MyMemoryDB(MemoryDB):
            def __init__(self) -> None:
                self.kv_store = {'storage': dict(), 'account': dict(), 'code': dict()}

            def rst(self) -> None:
                self.kv_store = {'storage': dict(), 'account': dict(), 'code': dict()}

        if eth_node_ip and eth_node_port and settings.REMOTE_FUZZING:
            self.w3 = Web3(HTTPProvider('http://%s:%s' % (eth_node_ip, eth_node_port)))
        else:
            self.w3 = None
        self.chain = chain_class.from_genesis_header(AtomicDB(MyMemoryDB()), MAINNET_GENESIS_HEADER)
        self.logger = initialize_logger("EVM")
        self.accounts = list()
        self.snapshot = None
        self.vm = None
        self.last_deploy_diagnostic = None
        self.standard_deployment_fixtures = []

    def get_block_by_blockid(self, block_identifier):
        validate_uint256(block_identifier)
        return self.w3.eth.getBlock(block_identifier)

    def get_cached_block_by_id(self, block_number):
        block = None
        block_path = os.path.dirname(os.path.abspath(__file__)) + "/" + ".".join([str(block_number), "block"])
        if not os.path.exists(block_path) and block_number in {ISTANBUL_MAINNET_BLOCK, BERLIN_MAINNET_BLOCK, LONDON_MAINNET_BLOCK}:
            block_path = os.path.dirname(os.path.abspath(__file__)) + "/" + ".".join([str(PETERSBURG_MAINNET_BLOCK), "block"])
        with open(block_path, "rb") as f:
            block = pickle.load(f)
        if getattr(block, "number", None) != block_number:
            payload = dict(block)
            payload["number"] = block_number
            block = type(block)(payload)
        return block

    @property
    def storage_emulator(self):
        return self.vm.state._account_db

    def set_vm(self, block_identifier='latest'):
        _block = None
        if self.w3:
            if block_identifier == 'latest':
                block_identifier = self.w3.eth.blockNumber
            validate_uint256(block_identifier)
            _block = self.w3.eth.getBlock(block_identifier)
        if not _block:
            if block_identifier in [
                HOMESTEAD_MAINNET_BLOCK,
                BYZANTIUM_MAINNET_BLOCK,
                PETERSBURG_MAINNET_BLOCK,
                ISTANBUL_MAINNET_BLOCK,
                BERLIN_MAINNET_BLOCK,
                LONDON_MAINNET_BLOCK,
            ]:
                _block = self.get_cached_block_by_id(block_identifier)
            else:
                _block = self.get_cached_block_by_id(PETERSBURG_MAINNET_BLOCK)
                _block = _block.copy(block_number=block_identifier)
        block_header = BlockHeader(difficulty=_block.difficulty,
                                   block_number=_block.number,
                                   gas_limit=max(int(_block.gasLimit), int(settings.GAS_LIMIT)),
                                   timestamp=_block.timestamp,
                                   coinbase=ZERO_ADDRESS,  # default value
                                   parent_hash=_block.parentHash,
                                   uncles_hash=_block.uncles,
                                   state_root=_block.stateRoot,
                                   transaction_root=_block.transactionsRoot,
                                   receipt_root=_block.receiptsRoot,
                                   bloom=0,  # default value
                                   gas_used=_block.gasUsed,
                                   extra_data=_block.extraData,
                                   mix_hash=_block.mixHash,
                                   nonce=_block.nonce)
        self.vm = self.chain.get_vm(block_header)

    def execute(self, tx, debug=True):  # debug默认是False
        if debug:
            logging.getLogger('eth.vm.computation.Computation')
            logging.basicConfig(level=DEBUG2_LEVEL_NUM)
        return self.vm.state.apply_transaction(tx)

    def reset(self):
        self.storage_emulator._raw_store_db.wrapped_db.rst()

    def create_fake_account(self, address, nonce=0, balance=settings.ACCOUNT_BALANCE, code='', storage=None):
        if storage is None:
            storage = {}
        address = to_canonical_address(address)
        account = Account(nonce=nonce, balance=balance)
        self.vm.state._account_db._set_account(address, account)
        if code and code != '':
            self.vm.state._account_db.set_code(address, code)
        if storage:
            for k, v in storage.items():
                self.vm.state._account_db.set_storage(address, int.from_bytes(decode_hex(k), byteorder="big"),
                                                      int.from_bytes(decode_hex(v), byteorder="big"))
        self.logger.debug("Created account %s with balance %s", encode_hex(address), account.balance)
        return encode_hex(address)

    def _account_code_size(self, address) -> int:
        try:
            address_text = _address_text(address)
            if not address_text:
                return -1
            return len(self.storage_emulator.get_code(to_canonical_address(address_text)) or b"")
        except Exception:
            return -1

    def _materialize_unknown_constructor_address(self, index: int) -> str:
        address = "0x%040x" % (_UNKNOWN_CONSTRUCTOR_ADDRESS_BASE + int(index or 0) + 1)
        if not self.has_account(address):
            self.create_fake_account(address)
        return address

    def install_standard_deployment_fixtures(self):
        compiled = _compile_standard_fixture_bytecodes()
        installed = []
        if not compiled.get("ok"):
            self.standard_deployment_fixtures = [
                {
                    "fixture_family": "standard_dex_fixture",
                    "installed": False,
                    "reason": compiled.get("error") or "fixture_compile_failed",
                    "fixture_source": compiled.get("fixture_source", ""),
                }
            ]
            self.logger.warning("Standard deployment fixtures unavailable: %s", compiled.get("error", "unknown"))
            return installed
        for contract_name, addresses in _STANDARD_DEX_FIXTURE_ADDRESSES.items():
            bytecode = compiled.get("bytecodes", {}).get(contract_name)
            if not bytecode:
                continue
            for address in addresses:
                self.create_fake_account(address, code=bytecode)
                installed.append(
                    {
                        "fixture_family": "standard_dex_fixture",
                        "contract_name": contract_name,
                        "address": address,
                        "code_size": len(bytecode),
                        "solc_version": compiled.get("solc_version", ""),
                        "fixture_source": compiled.get("fixture_source", ""),
                        "installed": True,
                    }
                )
        self.standard_deployment_fixtures = installed
        if installed:
            self.logger.info("Installed %s standard deployment fixture accounts", len(installed))
        return installed

    def has_account(self, address):
        address = to_canonical_address(address)
        return self.vm.state._account_db._has_account(address)

    def deploy_contract(self, creator, bin_code, amount=0, gas=None, gas_price=settings.GAS_PRICE,
                        debug=False, deploy_args: List[str] = None, deploy_mode=1):
        """
        部署合约
        """
        if gas is None:
            gas = settings.GAS_LIMIT
        raw_bin = str(bin_code or "")
        raw_no_prefix = raw_bin[2:] if raw_bin.startswith("0x") else raw_bin
        deploy_diagnostic: Dict[str, Any] = {
            "diagnostic_version": "deployment_diagnostics_v1",
            "timestamp_unix": time.time(),
            "creator": creator,
            "deployment_tx_value": amount,
            "deployment_tx_gas": gas,
            "deployment_tx_gas_price": gas_price,
            "deploy_mode": deploy_mode,
            "deployment_bytecode_length": len(raw_no_prefix) // 2 if raw_no_prefix else 0,
            "deployment_bytecode_prefix": raw_no_prefix[:24],
            "deployment_bytecode_suffix": raw_no_prefix[-24:] if raw_no_prefix else "",
            "linked_bytecode_status": "contains_placeholder" if "__" in raw_no_prefix else "no_placeholder_observed",
            "standard_deployment_fixtures": list(getattr(self, "standard_deployment_fixtures", []) or []),
            "constructor_args_raw": list(deploy_args) if deploy_args is not None else [],
            "constructor_arguments_materialized": False,
            "constructor_argument_sources": [],
            "dependency_addresses_materialized": [],
            "deployment_tx_created": False,
            "deployment_reverted": None,
            "deployment_receipt_status": "",
            "exception_signature": "",
            "first_failing_opcode_if_available": "",
            "evm_fork_or_runtime_context": getattr(settings, "EVM_VERSION", ""),
        }
        self.last_deploy_diagnostic = deploy_diagnostic
        if deploy_args is not None:
            assert len(deploy_args) % 3 == 0, "deploy_args必须是3的倍数, [name, type, name对应的contract或者YA_DO_NOT_KNOW]"
            encode_types = []
            encode_values = []
            unknown_address_index = 0
            for i in range(0, len(deploy_args), 3):
                param_name, param_type, param_value = deploy_args[i:i + 3]
                address_like = _is_address_like_type(param_type)
                array_type = "[" in str(param_type or "") and "]" in str(param_type or "")
                materialized_value = None
                source = "unhandled"
                if array_type and param_value == "YA_DO_NOT_KNOW":
                    base_type = _array_base_type(param_type)
                    lengths = _default_array_lengths(param_type)
                    materialized = _build_default_array_value(
                        base_type,
                        lengths,
                        lambda ordinal: self._materialize_unknown_constructor_address(
                            unknown_address_index + ordinal
                        ),
                    )
                    if _is_address_like_type(base_type):
                        unknown_address_index += max(1, sum(lengths) if lengths else 1)
                    encode_types.append(param_type)
                    encode_values.append(materialized)
                    materialized_value = materialized
                    source = "primitive_default"
                elif address_like and param_value != "YA_DO_NOT_KNOW":
                    encode_types.append("address")
                    if deploy_mode == 1:
                        materialized_value = settings.TRANS_INFO[param_value]
                        source = "settings.TRANS_INFO"
                    elif deploy_mode == 2:
                        materialized_value = random.choice(self.accounts)
                        source = "random_existing_account"
                    else:
                        materialized_value = 0x0000000000000000000000000000000000000000
                        source = "zero_address"
                    encode_values.append(materialized_value)
                elif address_like and param_value == "YA_DO_NOT_KNOW":
                    encode_types.append("address")
                    if deploy_mode == 1:
                        materialized_value = self._materialize_unknown_constructor_address(unknown_address_index)
                        unknown_address_index += 1
                        source = "deterministic_distinct_unknown_address"
                    elif deploy_mode == 2:
                        materialized_value = random.choice(self.accounts)
                        source = "random_existing_account"
                    else:
                        materialized_value = 0x0000000000000000000000000000000000000000
                        source = "zero_address"
                    encode_values.append(materialized_value)
                elif param_type.startswith("uint"):
                    encode_types.append(param_type)
                    materialized_value = 0 if param_value == "YA_DO_NOT_KNOW" else int(str(param_value), 0)
                    encode_values.append(materialized_value)
                    source = "primitive_default" if param_value == "YA_DO_NOT_KNOW" else "literal_uint"
                elif param_type == "bool":
                    encode_types.append(param_type)
                    if param_value == "YA_DO_NOT_KNOW":
                        materialized_value = False
                        source = "primitive_default"
                    else:
                        lowered = str(param_value).strip().lower()
                        if lowered in {"true", "1", "yes", "y"}:
                            materialized_value = True
                        elif lowered in {"false", "0", "no", "n"}:
                            materialized_value = False
                        else:
                            raise ValueError(f"unsupported bool constructor literal: {param_value}")
                        source = "literal_bool"
                    encode_values.append(materialized_value)
                elif param_type == "string":
                    encode_types.append(param_type)
                    materialized_value = "" if param_value == "YA_DO_NOT_KNOW" else str(param_value)
                    encode_values.append(materialized_value)
                    source = "primitive_default" if param_value == "YA_DO_NOT_KNOW" else "literal_string"
                elif param_type.startswith("bytes"):
                    if param_type == "bytes":
                        bytes_size = random.randint(1, 32) if param_value == "YA_DO_NOT_KNOW" else None
                    else:
                        bytes_size = int(param_type[5:])
                    encode_types.append(param_type)
                    if param_value == "YA_DO_NOT_KNOW":
                        materialized_value = bytearray(0 for _ in range(bytes_size))
                        source = "primitive_default"
                    else:
                        raw_value = str(param_value).strip()
                        if raw_value.startswith("0x"):
                            raw_bytes = bytes.fromhex(raw_value[2:])
                        else:
                            raw_bytes = raw_value.encode("utf-8")
                        if bytes_size is not None and len(raw_bytes) != bytes_size:
                            raise ValueError(
                                f"fixed bytes constructor literal has wrong size for {param_type}: "
                                f"expected {bytes_size}, got {len(raw_bytes)}"
                            )
                        materialized_value = raw_bytes if param_type == "bytes" else bytearray(raw_bytes)
                        source = "literal_bytes"
                    encode_values.append(materialized_value)
                elif param_type.startswith("int"):
                    encode_types.append(param_type)
                    materialized_value = 0 if param_value == "YA_DO_NOT_KNOW" else int(str(param_value), 0)
                    encode_values.append(materialized_value)
                    source = "primitive_default" if param_value == "YA_DO_NOT_KNOW" else "literal_int"
                deploy_diagnostic["constructor_argument_sources"].append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "raw_value": param_value,
                        "source": source,
                        "materialized_value": _diagnostic_jsonable(materialized_value),
                    }
                )
                if address_like:
                    deploy_diagnostic["dependency_addresses_materialized"].append(
                        {
                            "name": param_name,
                            "type": param_type,
                            "raw_value": param_value,
                            "source": source,
                            "address": _diagnostic_jsonable(materialized_value),
                            "code_size": self._account_code_size(materialized_value),
                        }
                    )
            deploy_diagnostic["constructor_encode_types"] = list(encode_types)
            deploy_diagnostic["constructor_encode_values"] = _diagnostic_jsonable(encode_values)
            deploy_diagnostic["constructor_arguments_materialized"] = True
            self.logger.info(f"encode_values: {encode_values}")
            bin_code += encode_abi(encode_types, encode_values).hex()
            final_raw = str(bin_code or "")
            final_raw = final_raw[2:] if final_raw.startswith("0x") else final_raw
            deploy_diagnostic["deployment_bytecode_length_after_constructor_args"] = len(final_raw) // 2 if final_raw else 0
        nonce = self.vm.state.get_nonce(decode_hex(creator))
        deploy_diagnostic["deployment_nonce"] = nonce
        tx = self.vm.create_unsigned_transaction(
            nonce=nonce,
            gas_price=gas_price,
            gas=gas,
            to=CREATE_CONTRACT_ADDRESS,
            value=amount,
            data=decode_hex(bin_code),
        )
        deploy_diagnostic["deployment_tx_created"] = True
        tx = SpoofTransaction(tx, from_=decode_hex(creator))
        result = self.execute(tx, debug=debug)
        deploy_diagnostic["deployment_reverted"] = bool(getattr(result, "is_error", False))
        deploy_diagnostic["deployment_receipt_status"] = "reverted" if deploy_diagnostic["deployment_reverted"] else "success"
        deploy_diagnostic["result_error"] = _diagnostic_error_payload(getattr(result, "_error", None))
        deploy_diagnostic["exception_signature"] = deploy_diagnostic["result_error"].get("repr", "")
        deploy_diagnostic["output_if_available"] = _diagnostic_jsonable(getattr(result, "output", b""))
        try:
            deploy_diagnostic["gas_remaining_if_available"] = result.get_gas_remaining() if hasattr(result, "get_gas_remaining") else ""
        except Exception as gas_exc:
            deploy_diagnostic["gas_remaining_error"] = repr(gas_exc)
        deploy_diagnostic["deployed_address"] = encode_hex(result.msg.storage_address)
        address = to_canonical_address(encode_hex(result.msg.storage_address))
        self.storage_emulator.set_balance(address, 1)
        try:
            deployed_code = self.storage_emulator.get_code(address)
            deploy_diagnostic["deployed_code_size"] = len(deployed_code or b"")
        except Exception as code_exc:
            deploy_diagnostic["deployed_code_size_error"] = repr(code_exc)
        self.last_deploy_diagnostic = deploy_diagnostic
        return result

    def deploy_transaction(self, input, gas_price=settings.GAS_PRICE, debug=False):
        transaction = input["transaction"]
        from_account = decode_hex(transaction["from"])
        nonce = self.vm.state.get_nonce(from_account)
        try:
            to = decode_hex(transaction["to"])
        except:
            to = transaction["to"]
        tx = self.vm.create_unsigned_transaction(
            nonce=nonce,
            gas_price=gas_price,
            gas=transaction["gaslimit"],
            to=to,
            value=transaction["value"],
            data=decode_hex(transaction["data"]),
        )
        tx = SpoofTransaction(tx, from_=from_account)

        block = input["block"]
        if "timestamp" in block and block["timestamp"] is not None:
            self.vm.state.fuzzed_timestamp = block["timestamp"]
        else:
            self.vm.state.fuzzed_timestamp = None
        if "blocknumber" in block and block["blocknumber"] is not None:
            self.vm.state.fuzzed_blocknumber = block["blocknumber"]
        else:
            self.vm.state.fuzzed_blocknumber = None

        global_state = input["global_state"]
        if "balance" in global_state and global_state["balance"] is not None:
            self.vm.state.fuzzed_balance = global_state["balance"]
        else:
            self.vm.state.fuzzed_balance = None

        if "call_return" in global_state and global_state["call_return"] is not None \
                and len(global_state["call_return"]) > 0:
            self.vm.state.fuzzed_call_return = global_state["call_return"]
        if "extcodesize" in global_state and global_state["extcodesize"] is not None \
                and len(global_state["extcodesize"]) > 0:
            self.vm.state.fuzzed_extcodesize = global_state["extcodesize"]

        environment = input["environment"]
        if "returndatasize" in environment and environment["returndatasize"] is not None:
            self.vm.state.fuzzed_returndatasize = environment["returndatasize"]

        self.storage_emulator.set_balance(from_account, settings.ACCOUNT_BALANCE)
        return self.execute(tx, debug=debug)

    def get_balance(self, address):
        return self.storage_emulator.get_balance(address)

    def get_code(self, address):
        return self.storage_emulator.get_code(address)

    def set_code(self, address, code):
        return self.storage_emulator.set_code(address, code)

    def create_snapshot(self):
        self.snapshot = self.storage_emulator.record()
        self.storage_emulator.set_snapshot(self.snapshot)

    def restore_from_snapshot(self):
        self.storage_emulator.discard(self.snapshot)

    def get_accounts(self):
        return [encode_hex(x) for x in self.storage_emulator._raw_store_db.wrapped_db["account"].keys()]

    def set_vm_by_name(self, EVM_VERSION):
        if EVM_VERSION == "homestead":
            self.set_vm(HOMESTEAD_MAINNET_BLOCK)
        elif EVM_VERSION == "byzantium":
            self.set_vm(BYZANTIUM_MAINNET_BLOCK)
        elif EVM_VERSION == "petersburg":
            self.set_vm(PETERSBURG_MAINNET_BLOCK)
        elif EVM_VERSION == "istanbul":
            self.set_vm(ISTANBUL_MAINNET_BLOCK)
        elif EVM_VERSION == "berlin":
            self.set_vm(BERLIN_MAINNET_BLOCK)
        elif EVM_VERSION == "london":
            self.set_vm(LONDON_MAINNET_BLOCK)
        else:
            raise Exception(
                "Unknown EVM version, please choose either "
                "'homestead', 'byzantium', 'petersburg' or 'istanbul'."
            )

    def create_fake_accounts(self):
        self.accounts.append(self.create_fake_account("0xcafebabecafebabecafebabecafebabecafebabe"))
        for address in settings.ATTACKER_ACCOUNTS:
            self.accounts.append(self.create_fake_account(address))
        self.install_standard_deployment_fixtures()
