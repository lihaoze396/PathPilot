import csv
import hashlib
import json
import os
import re
from pathlib import Path

import solcx
from packaging.version import Version


PRAGMA_RE = re.compile(r"pragma\s+solidity\s+([^;]+);", re.I)
IMPORT_RE = re.compile(
    r"^\s*import\s+(?:(?:[^;\"']+?)\s+from\s+)?[\"']([^\"']+)[\"']\s*;",
    re.M,
)
COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)


DEFAULT_OUTPUTS = [
    "abi",
    "evm.deployedBytecode",
    "evm.bytecode.object",
    "evm.bytecode.linkReferences",
    "evm.deployedBytecode.linkReferences",
    "evm.legacyAssembly",
    "metadata",
]


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")


def strip_comments(text):
    return COMMENT_RE.sub("", text or "")


def normalize_path(path):
    return str(Path(path).resolve()).replace("\\", "/")


def normalize_version(value):
    return Version(str(value).strip().lstrip("v"))


def installed_versions():
    return sorted(normalize_version(v) for v in solcx.get_installed_solc_versions())


def _expand_version(version_text):
    parts = str(version_text).strip().lstrip("v").split(".")
    while len(parts) < 3:
        parts.append("0")
    return Version(".".join(parts[:3]))


def parse_constraints(expr):
    alternatives = []
    for part in re.split(r"\s*\|\|\s*", str(expr or "").strip()):
        constraints = []
        tokens = re.findall(
            r"(\^|~|>=|<=|>|<|=|==)?\s*v?(\d+\.\d+\.\d+|\d+\.\d+)",
            part,
        )
        for op, version_text in tokens:
            version = _expand_version(version_text)
            op = op or "="
            if op == "^":
                constraints.append((">=", version))
                if version.major > 0:
                    constraints.append(("<", Version(f"{version.major + 1}.0.0")))
                elif version.minor > 0:
                    constraints.append(("<", Version(f"0.{version.minor + 1}.0")))
                else:
                    constraints.append(("<", Version(f"0.0.{version.micro + 1}")))
            elif op == "~":
                constraints.append((">=", version))
                constraints.append(("<", Version(f"{version.major}.{version.minor + 1}.0")))
            else:
                constraints.append((op, version))
        if constraints:
            alternatives.append(constraints)
    return alternatives


def satisfies_constraints(version, constraints):
    for op, target in constraints:
        if op == ">=" and not version >= target:
            return False
        if op == "<=" and not version <= target:
            return False
        if op == ">" and not version > target:
            return False
        if op == "<" and not version < target:
            return False
        if op in ("=", "==") and not version == target:
            return False
    return True


def satisfies_pragma(version, expr):
    alternatives = parse_constraints(expr)
    if not alternatives:
        return True
    return any(satisfies_constraints(version, constraints) for constraints in alternatives)


def candidate_solc_versions(source_text, versions=None):
    versions = versions or installed_versions()
    pragmas = PRAGMA_RE.findall(strip_comments(source_text))
    if not pragmas:
        preferred = Version("0.8.20")
        return [preferred] if preferred in versions else list(reversed(versions))
    matches = [
        version
        for version in reversed(versions)
        if all(satisfies_pragma(version, pragma) for pragma in pragmas)
    ]
    return matches


def supports_via_ir(version):
    try:
        return normalize_version(version) >= Version("0.8.13")
    except Exception:
        return False


def classify_error(message):
    text = str(message or "")
    if "Source " in text and "not found" in text:
        return "missing_import"
    if "requires different compiler version" in text:
        return "wrong_solc"
    if "Stack too deep" in text:
        return "stack_too_deep"
    if "Identifier not found" in text or "Undeclared identifier" in text:
        return "missing_symbol"
    if "TypeError" in text:
        return "type_error"
    if "ParserError" in text:
        return "parser_error"
    if "DeclarationError" in text:
        return "declaration_error"
    if "Invalid EVM version requested" in text:
        return "invalid_evm_version"
    return "other"


def dataset_manifest_candidates(source_code_file):
    source = Path(source_code_file).resolve()
    candidates = []
    env_manifest = os.environ.get("FORGE_DATASET_MANIFEST", "").strip()
    if env_manifest:
        candidates.append(Path(env_manifest))
    for parent in [source.parent, *source.parents]:
        candidates.append(parent / "manifest.csv")
        candidates.append(parent / "compile_ready_manifest.csv")
    seen = set()
    output = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        output.append(resolved)
    return output


def resolve_dataset_source(source_code_file):
    requested = Path(source_code_file).resolve()
    requested_norm = normalize_path(requested)
    requested_name = requested.name
    for manifest in dataset_manifest_candidates(requested):
        try:
            rows = csv.DictReader(manifest.open("r", encoding="utf-8-sig"))
            for row in rows:
                selected_values = [
                    row.get("selected_file", ""),
                    row.get("compile_source_file", ""),
                    row.get("selected_filename", ""),
                ]
                selected_norms = set()
                for value in selected_values:
                    if not value:
                        continue
                    try:
                        selected_norms.add(normalize_path(value))
                    except Exception:
                        selected_norms.add(str(value).replace("\\", "/"))
                if requested_norm not in selected_norms and requested_name not in selected_values:
                    continue
                source_value = row.get("source_file") or row.get("compile_source_file")
                if source_value and Path(source_value).exists():
                    return Path(source_value).resolve(), manifest, row
        except Exception:
            continue
    return requested, None, None


def extract_imports(source_text):
    return IMPORT_RE.findall(strip_comments(source_text))


def _common_remapping_targets(root):
    root = Path(root)
    return [
        ("@openzeppelin/contracts-upgradeable/", root / "lib" / "openzeppelin-contracts-upgradeable" / "contracts-upgradeable"),
        ("@openzeppelin/contracts/", root / "lib" / "openzeppelin-contracts" / "contracts"),
        ("@openzeppelin/contracts/", root / "node_modules" / "@openzeppelin" / "contracts"),
        ("@openzeppelin/", root / "@openzeppelin"),
        ("@openzeppelin/", root / "node_modules" / "@openzeppelin"),
        ("hardhat/", root / "hardhat"),
        ("hardhat/", root / "node_modules" / "hardhat"),
        ("forge-std/", root / "lib" / "forge-std" / "src"),
        ("solmate/", root / "lib" / "solmate" / "src"),
        ("solady/", root / "lib" / "solady" / "src"),
        ("ds-test/", root / "lib" / "ds-test" / "src"),
    ]


def _global_vendor_roots():
    roots = []
    local_vendor = Path(__file__).resolve().parent / "fuzzer" / "dependencies" / "vendor"
    if local_vendor.exists():
        roots.append(local_vendor)
    for value in (
        os.environ.get("FUZZ_VENDOR_ROOT", ""),
    ):
        if not value:
            continue
        root = Path(value) / "fuzzer" / "dependencies" / "vendor"
        if root.exists():
            roots.append(root.resolve())
    output = []
    seen = set()
    for root in roots:
        key = normalize_path(root)
        if key not in seen:
            seen.add(key)
            output.append(root)
    return output


def compiler_remappings(root, use_relative_targets=True):
    root = Path(root).resolve()
    remappings = []
    seen = set()
    seen_prefixes = set()

    def add(prefix, target):
        target = Path(target)
        if not target.exists():
            return
        if prefix in seen_prefixes:
            return
        target_resolved = target.resolve()
        try:
            target_text = target_resolved.relative_to(root).as_posix()
        except ValueError:
            target_text = normalize_path(target_resolved)
        if not use_relative_targets:
            target_text = normalize_path(target_resolved)
        key = (prefix, target_text)
        if key in seen:
            return
        seen.add(key)
        seen_prefixes.add(prefix)
        remappings.append(f"{prefix}={target_text.rstrip('/')}/")

    node_modules = root / "node_modules"
    if node_modules.exists():
        for child in node_modules.iterdir():
            if child.is_dir():
                add(child.name + "/", child)
                if child.name.startswith("@"):
                    for scoped_child in child.iterdir():
                        if scoped_child.is_dir():
                            add(child.name + "/" + scoped_child.name + "/", scoped_child)
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and child.name.startswith("@"):
            for scoped_child in child.iterdir():
                if scoped_child.is_dir():
                    add(child.name + "/" + scoped_child.name + "/", scoped_child)
            add(child.name + "/", child)
    for prefix, target in _common_remapping_targets(root):
        add(prefix, target)
    for vendor in _global_vendor_roots():
        add("@openzeppelin/contracts-upgradeable/", vendor / "@openzeppelin" / "contracts-upgradeable")
        add("@openzeppelin/contracts/math/", vendor / "@openzeppelin" / "contracts" / "utils" / "math")
        add("@openzeppelin/contracts/utils/ReentrancyGuard.sol", vendor / "@openzeppelin" / "contracts" / "security" / "ReentrancyGuard.sol")
        add("@openzeppelin/contracts/", vendor / "@openzeppelin" / "contracts")
        add("@openzeppelin/", vendor / "@openzeppelin")
        add("OpenZeppelin/openzeppelin-contracts@3.2.0/contracts/", vendor / "@openzeppelin" / "contracts")
        add("OpenZeppelin/openzeppelin-contracts@3.1.0/contracts/", vendor / "@openzeppelin" / "contracts")
        add("@solarity/solidity-lib/", vendor / "@solarity" / "solidity-lib")
        add("@solarity/", vendor / "@solarity")
    return remappings


def _resolve_import_candidate(root, source_rel_parent, import_path):
    root = Path(root)
    import_path = import_path.replace("\\", "/")
    if import_path.startswith("."):
        candidate = (root / source_rel_parent / import_path).resolve()
        return candidate if candidate.exists() else None

    direct = (root / import_path).resolve()
    if direct.exists():
        return direct
    node = (root / "node_modules" / import_path).resolve()
    if node.exists():
        return node
    for prefix, target in _common_remapping_targets(root):
        if import_path.startswith(prefix) and Path(target).exists():
            candidate = (Path(target) / import_path[len(prefix):]).resolve()
            if candidate.exists():
                return candidate
    for vendor in _global_vendor_roots():
        candidates = [
            ("@openzeppelin/contracts-upgradeable/", vendor / "@openzeppelin" / "contracts-upgradeable"),
            ("@openzeppelin/contracts/", vendor / "@openzeppelin" / "contracts"),
            ("@openzeppelin/contracts/math/", vendor / "@openzeppelin" / "contracts" / "utils" / "math"),
            ("@openzeppelin/contracts/utils/ReentrancyGuard.sol", vendor / "@openzeppelin" / "contracts" / "security" / "ReentrancyGuard.sol"),
            ("@openzeppelin/", vendor / "@openzeppelin"),
            ("OpenZeppelin/openzeppelin-contracts@3.2.0/contracts/", vendor / "@openzeppelin" / "contracts"),
            ("OpenZeppelin/openzeppelin-contracts@3.1.0/contracts/", vendor / "@openzeppelin" / "contracts"),
            ("@solarity/solidity-lib/", vendor / "@solarity" / "solidity-lib"),
            ("@solarity/", vendor / "@solarity"),
        ]
        for prefix, target in candidates:
            if import_path.startswith(prefix) and Path(target).exists():
                candidate = (Path(target) / import_path[len(prefix):]).resolve()
                if candidate.exists():
                    return candidate
    return None


def infer_project_root(source_path, source_text=None):
    source_path = Path(source_path).resolve()
    source_text = source_text if source_text is not None else read_text(source_path)
    imports = extract_imports(source_text)
    if not imports:
        return source_path.parent
    ancestors = [source_path.parent, *source_path.parents]
    best = None
    for root in ancestors:
        try:
            source_rel = source_path.relative_to(root).parent
        except ValueError:
            continue
        if all(_resolve_import_candidate(root, source_rel, imp) for imp in imports):
            best = root
            break
    return best or source_path.parent


def resolve_compile_source(source_code_file):
    requested = Path(source_code_file).resolve()
    actual, manifest, row = resolve_dataset_source(requested)
    source_text = read_text(actual)
    root = infer_project_root(actual, source_text)
    try:
        source_key = actual.relative_to(root).as_posix()
    except ValueError:
        source_key = normalize_path(actual)
    return {
        "requested_path": requested,
        "requested_key": normalize_path(requested),
        "actual_path": actual,
        "source_text": source_text,
        "project_root": Path(root).resolve(),
        "source_key": source_key,
        "manifest": manifest,
        "manifest_row": row,
    }


def _dedupe_path_values(values):
    output = []
    seen = set()
    for value in values:
        try:
            key = normalize_path(value)
        except Exception:
            key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(str(value))
    return output


def _slither_supports_base_path(solc_path):
    if not solc_path:
        return True
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(solc_path))
    if not match:
        return True
    return tuple(int(match.group(i)) for i in range(1, 4)) >= (0, 6, 0)


def _slither_remappings_for_context(resolved, context_root):
    project_root = Path(resolved["project_root"]).resolve()
    context_root = Path(context_root).resolve()
    if context_root == project_root:
        return compiler_remappings(project_root, use_relative_targets=True)

    def _relative_to_context(path_text):
        path_text = str(path_text).rstrip("/")
        try:
            target = Path(path_text).resolve()
            return target.relative_to(context_root).as_posix()
        except Exception:
            return path_text.replace("\\", "/")

    remappings = []
    for remapping in compiler_remappings(project_root, use_relative_targets=False):
        if "=" not in remapping:
            remappings.append(remapping)
            continue
        prefix, target = remapping.split("=", 1)
        remappings.append("{}={}/".format(prefix, _relative_to_context(target)))
    return remappings


def _build_slither_context(
    resolved,
    context_root,
    source_target,
    *,
    solc_path=None,
    evm_version=None,
    vendor_roots=None,
    context_label="",
    include_base_path_args=True,
):
    context_root = Path(context_root).resolve()
    vendor_roots = vendor_roots or []
    remappings = _slither_remappings_for_context(resolved, context_root)
    allow_paths = _dedupe_path_values(
        [
            str(context_root),
            str(resolved["project_root"]),
            str(resolved["actual_path"].parent),
            os.getcwd(),
            *[str(parent) for parent in list(resolved["actual_path"].parents)[:8]],
            *[str(path) for path in vendor_roots],
        ]
    )

    kwargs = {}
    if solc_path:
        kwargs["solc"] = str(solc_path)
    if remappings:
        kwargs["solc_remaps"] = remappings

    solc_args = []
    supports_base_path = _slither_supports_base_path(solc_path)
    if include_base_path_args and supports_base_path:
        solc_args.extend(
            [
                "--base-path {}".format(str(context_root)),
                "--allow-paths {}".format(",".join(allow_paths)),
            ]
        )
    if evm_version and (supports_base_path or not include_base_path_args):
        solc_args.append("--evm-version {}".format(str(evm_version)))
    if solc_args:
        kwargs["solc_args"] = " ".join(solc_args)

    return {
        "target": source_target,
        "cwd": str(context_root),
        "kwargs": kwargs,
        "resolved": {
            "requested_path": str(resolved["requested_path"]),
            "actual_path": str(resolved["actual_path"]),
            "project_root": str(resolved["project_root"]),
            "slither_context_root": str(context_root),
            "source_key": resolved["source_key"],
            "slither_target": source_target,
            "context_label": context_label,
            "remapping_count": len(remappings),
            "allow_path_count": len(allow_paths),
        },
    }


def compile_standard_resolved(
    source_code_file,
    solc_version,
    evm_version="byzantium",
    outputs=None,
    optimizer_runs=200,
    source_override=None,
):
    resolved = resolve_compile_source(source_code_file)
    source_text = source_override if source_override is not None else resolved["source_text"]
    outputs = outputs or DEFAULT_OUTPUTS
    version_text = str(solc_version).strip().lstrip("v")
    solcx.set_solc_version(version_text, silent=True)
    allow_path_values = [
        str(resolved["project_root"]),
        str(resolved["actual_path"].parent),
        os.getcwd(),
        *[str(parent) for parent in list(resolved["actual_path"].parents)[:8]],
        *[str(path) for path in _global_vendor_roots()],
    ]
    allow_paths = []
    seen_allow_paths = set()
    for value in allow_path_values:
        try:
            key = normalize_path(value)
        except Exception:
            key = str(value)
        if key in seen_allow_paths:
            continue
        seen_allow_paths.add(key)
        allow_paths.append(value)

    def build(source_key, via_ir=False, include_evm=True, use_relative_remappings=True):
        settings = {
            "optimizer": {"enabled": True, "runs": optimizer_runs},
            "remappings": compiler_remappings(
                resolved["project_root"],
                use_relative_targets=use_relative_remappings,
            ),
            "outputSelection": {
                "*": {"*": outputs},
                source_key: {"*": outputs},
            },
        }
        if include_evm and evm_version:
            settings["evmVersion"] = evm_version
        if via_ir:
            settings["viaIR"] = True
        return {
            "language": "Solidity",
            "sources": {source_key: {"content": source_text}},
            "settings": settings,
        }

    last_error = None
    source_variants = [
        {
            "source_key": resolved["source_key"],
            "base_path": str(resolved["project_root"]),
            "relative_remappings": True,
        },
        {
            "source_key": normalize_path(resolved["actual_path"]),
            "base_path": None,
            "relative_remappings": False,
        },
        {
            "source_key": resolved["source_key"],
            "base_path": None,
            "relative_remappings": False,
        },
    ]
    variant_seen = set()
    unique_variants = []
    for variant in source_variants:
        key = (variant["source_key"], variant["base_path"], variant["relative_remappings"])
        if key in variant_seen:
            continue
        variant_seen.add(key)
        unique_variants.append(variant)

    via_ir_options = (False, True) if supports_via_ir(version_text) else (False,)
    for variant in unique_variants:
        for via_ir in via_ir_options:
            for include_evm in (True, False):
                try:
                    output = solcx.compile_standard(
                        build(
                            variant["source_key"],
                            via_ir=via_ir,
                            include_evm=include_evm,
                            use_relative_remappings=variant["relative_remappings"],
                        ),
                        base_path=variant["base_path"],
                        allow_paths=allow_paths,
                    )
                    _alias_requested_source(output, resolved, compiled_source_key=variant["source_key"])
                    return output, resolved, {
                        "via_ir": via_ir,
                        "evm_version_used": evm_version if include_evm else "",
                        "base_path_used": variant["base_path"] or "",
                        "source_key_used": variant["source_key"],
                    }
                except Exception as exc:
                    last_error = exc
                    continue
    raise last_error


def slither_compilation_contexts(source_code_file, solc_path=None, evm_version=None):
    """Return ordered Slither target/cwd/kwargs candidates aligned with the forge compile resolver."""
    resolved = resolve_compile_source(source_code_file)
    vendor_roots = _global_vendor_roots()
    project_root = Path(resolved["project_root"]).resolve()
    source_target = resolved["source_key"]
    import_paths = extract_imports(resolved["source_text"])
    contexts = [
        _build_slither_context(
            resolved,
            project_root,
            source_target,
            solc_path=solc_path,
            evm_version=evm_version,
            vendor_roots=vendor_roots,
            context_label="project_local_no_base_path",
            include_base_path_args=False,
        ),
        _build_slither_context(
            resolved,
            project_root,
            source_target,
            solc_path=solc_path,
            evm_version=evm_version,
            vendor_roots=vendor_roots,
            context_label="project_local_with_base_path",
        )
    ]

    if any(str(import_path).replace("\\", "/").startswith("../") for import_path in import_paths):
        grandparent_root = resolved["actual_path"].parent.parent
        if grandparent_root.exists() and grandparent_root != project_root:
            try:
                grandparent_target = resolved["actual_path"].relative_to(grandparent_root).as_posix()
            except ValueError:
                grandparent_target = source_target
            contexts.append(
                _build_slither_context(
                    resolved,
                    grandparent_root,
                    grandparent_target,
                    solc_path=solc_path,
                    evm_version=evm_version,
                    vendor_roots=vendor_roots,
                    context_label="grandparent_import_fallback",
                )
            )

    context_roots = [project_root, resolved["actual_path"].parent, *vendor_roots]
    try:
        common_root = Path(os.path.commonpath([str(Path(item).resolve()) for item in context_roots])).resolve()
    except Exception:
        common_root = project_root
    try:
        broad_target = resolved["actual_path"].relative_to(common_root).as_posix()
    except ValueError:
        broad_target = resolved["source_key"]

    if common_root != project_root or broad_target != source_target:
        contexts.append(
            _build_slither_context(
                resolved,
                common_root,
                broad_target,
                solc_path=solc_path,
                evm_version=evm_version,
                vendor_roots=vendor_roots,
                context_label="shared_common_root_fallback",
            )
        )

    unique_contexts = []
    seen = set()
    for context in contexts:
        key = (
            context["cwd"],
            context["target"],
            tuple(context["kwargs"].get("solc_remaps", []) or []),
            context["kwargs"].get("solc_args", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_contexts.append(context)
    return unique_contexts


def slither_compilation_context(source_code_file, solc_path=None, evm_version=None):
    """Return the preferred Slither target/cwd/kwargs aligned with the forge compile resolver."""
    contexts = slither_compilation_contexts(
        source_code_file,
        solc_path=solc_path,
        evm_version=evm_version,
    )
    return contexts[0]


def _alias_requested_source(output, resolved, compiled_source_key=None):
    contracts = output.get("contracts")
    if not isinstance(contracts, dict):
        return
    source_key = compiled_source_key or resolved["source_key"]
    requested_variants = {
        str(resolved["requested_path"]),
        normalize_path(resolved["requested_path"]),
        str(resolved["actual_path"]),
        normalize_path(resolved["actual_path"]),
        resolved["source_key"],
    }
    if source_key not in contracts:
        return
    for requested_key in requested_variants:
        if requested_key not in contracts:
            contracts[requested_key] = contracts[source_key]
    sources = output.get("sources")
    if isinstance(sources, dict) and source_key in sources:
        for requested_key in requested_variants:
            sources.setdefault(requested_key, sources[source_key])


def compile_with_candidates(source_code_file, evm_version="byzantium", max_versions=None):
    resolved = resolve_compile_source(source_code_file)
    versions = candidate_solc_versions(resolved["source_text"])
    if max_versions:
        versions = versions[:max_versions]
    last_error = None
    attempts = []
    for version in versions:
        try:
            output, resolved, note = compile_standard_resolved(
                source_code_file,
                "v" + str(version),
                evm_version=evm_version,
            )
            return {
                "ok": True,
                "output": output,
                "resolved": resolved,
                "solc_version": "v" + str(version),
                "note": note,
                "attempts": attempts,
                "error": "",
            }
        except Exception as exc:
            last_error = exc
            message = getattr(exc, "message", str(exc))
            attempts.append({"solc_version": "v" + str(version), "error_class": classify_error(message), "error": message[:1000]})
    message = getattr(last_error, "message", str(last_error)) if last_error else "no matching installed solc"
    return {
        "ok": False,
        "output": None,
        "resolved": resolved,
        "solc_version": "",
        "note": {},
        "attempts": attempts,
        "error_class": classify_error(message),
        "error": message,
    }


def source_hash(path):
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()
