#!/usr/bin/env python3
"""Fail-closed native input contract for the G3-R partial-claim build HOLD."""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import subprocess

SOURCE_HEAD = "5c36a3f4431ce93d2676fb82349a972b513a3ebf"
SOURCE_TREE = "8b7eb1157feda6fa099371abc7df123c18faaa19"

# Object names and argv macros are source contracts, not a reviewed link recipe.
OBJECTS = (
    ("data_io", "native/data_io.c", ()),
    ("checkpoint_io", "native/checkpoint_io.c", ()),
    ("kernel_abi", "native/kernel_abi.c", ()),
    ("m3_i64_integration", "src/eshkol_transformer/m3_i64_integration.c",
     ("ET_M3_TESTING", "ET_I64_TENSOR_TESTING", "ET_I64_TENSOR_STORAGE_QUERY_PRIVATE")),
    ("t1_i64_shell", "native/t1_i64_shell.c", ()),
    ("m3_call_f32_integration", "src/eshkol_transformer/m3_call_f32_integration.c",
     ("ET_F32_TENSOR_TESTING", "ET_F32_TENSOR_STORAGE_QUERY_PRIVATE",
      "ET_G3R_SAME_OWNER_SAVE_PRIVATE")),
    ("g3t_transport", "src/eshkol_transformer/g3t_transport.c",
     ("ET_G3T_TESTING", "ET_F32_TENSOR_TESTING", "ET_A2_KV_CACHE_TESTING",
      "ET_I64_TENSOR_TESTING", "ET_G3T_PREFILL_SAMPLE_PRIVATE",
      "ET_G3T_OUTPUT_TEXT_PRIVATE", "ET_G3T_FINAL_PUBLICATION_PRIVATE",
      "ET_G3T_ZERO_BUDGET_PRIVATE", "ET_G3T_P2_ZERO_BUDGET_PRIVATE",
      "ET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE", "ET_G3T_MANUAL_LOGITS_PRIVATE",
      "ET_G3T_MANUAL_P1_PREFILL_PRIVATE", "ET_G3T_MANUAL_P2_PREFILL_PRIVATE",
      "ET_G3T_MANUAL_DECODE_PRIVATE", "ET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE",
      "ET_G3T_INPUT_FROM_T1_PRIVATE", "ET_G3T_OWNED_TOKEN_INPUT_PRIVATE",
      "ET_G3T_OUTPUT_RNG_CLONE_PRIVATE", "ET_G3T_GENERATOR_RNG_PRIVATE",
      "ET_G3R_COMMITTED_HISTORY_PRIVATE", "ET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE",
      "ET_G3R_SAME_OWNER_SAVE_PRIVATE", "ET_I64_TENSOR_STORAGE_QUERY_PRIVATE",
      "ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE", "ET_F32_TENSOR_STORAGE_QUERY_PRIVATE")),
    ("a2_kv_cache", "native/a2_kv_cache.c",
     ("ET_A2_KV_CACHE_TESTING", "ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE")),
    ("n2_primitives_provider", "native/n2_primitives_provider.c", ()),
    ("n3k_primitives_provider", "native/n3k_primitives_provider.c", ()),
    ("a2_attention_provider", "native/a2_attention_provider.c", ()),
    ("g3n_primitives_provider", "native/g3n_primitives_provider.c", ()),
    ("g3s_sampling_provider", "native/g3s_sampling_provider.c", ()),
    ("i2_helpers", "native/i2_wave2_package_bridge.c",
     ("ET_I2_NATIVE_HELPERS_ONLY", "ET_M3T_PACKAGE_BUILD")),
    ("p1_identity", "native/p1_identity.c", ("ET_P1_TRUSTED_BUILD=1",)),
    ("f32_tensor", "native/f32_tensor.c",
     ("ET_F32_TENSOR_TESTING", "ET_F32_TENSOR_STORAGE_QUERY_PRIVATE",
      "ET_G3R_SAVE_BIT_WITNESS_TESTING", "ET_G3R_SAME_OWNER_SAVE_PRIVATE")),
    ("o2_optimizer", "native/o2_optimizer.c",
     ("ET_O2_TESTING", "ET_G3R_SAME_OWNER_SAVE_PRIVATE")),
    ("g3r_same_owner_save", "native/g3r_same_owner_save.c",
     ("ET_G3R_SAME_OWNER_SAVE_PRIVATE", "ET_G3R_COMMITTED_HISTORY_PRIVATE")),
    ("g3r_o2_creation_bridge", "native/g3r_o2_creation_bridge.c",
     ("ET_G3R_SAME_OWNER_SAVE_PRIVATE", "ET_G3R_COMMITTED_HISTORY_PRIVATE")),
    ("capture_partial_claim_cut_bridge", "tests/g3r/capture_partial_claim_cut_bridge.c",
     ("ET_F32_TENSOR_TESTING", "ET_G3R_SAVE_BIT_WITNESS_TESTING",
      "ET_O2_TESTING", "ET_G3R_SAME_OWNER_SAVE_PRIVATE",
      "ET_G3R_COMMITTED_HISTORY_PRIVATE")),
)
NESTED = {
    "m3_i64_integration": ("native/i64_tensor.c",),
    "m3_call_f32_integration": ("src/eshkol_transformer/m3t_f32_integration.c",),
    "g3t_transport": ("src/eshkol_transformer/m3_model.c",
                      "src/eshkol_transformer/m3t_transport.c"),
}
DIRECT_HEADERS = (
    "native/f32_parameter_internal.h", "native/o2_optimizer_internal.h",
    "native/g3r_same_owner_save.h", "native/g3r_o2_creation_bridge.h",
    "src/eshkol_transformer/g3t_transport.h",
    "src/eshkol_transformer/m3t_transport.h",
)


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(root), *args))


def canonical_root(root: Path) -> Path:
    if not root.is_absolute() or root != root.resolve(strict=True):
        raise ValueError("native root must be an absolute canonical directory")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("native root is not a real directory")
    return root


def verify_file(root: Path, name: str) -> str:
    if name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"noncanonical native path: {name}")
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(f"missing or aliased native source: {name}")
    raw = path.read_bytes()
    try:
        committed = git(root, "show", f"{SOURCE_HEAD}:{name}")
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"uncommitted native dependency: {name}") from exc
    if raw != committed:
        raise ValueError(f"native source differs from pinned Git bytes: {name}")
    return sha256(raw).hexdigest()


def input_paths() -> tuple[str, ...]:
    return tuple(sorted({source for _, source, _ in OBJECTS}
                        | {source for group in NESTED.values() for source in group}
                        | set(DIRECT_HEADERS)))


def verify_root(root: Path, *, require_readonly: bool = True) -> dict[str, str]:
    root = canonical_root(root)
    if Path(git(root, "rev-parse", "--show-toplevel").decode().strip()) != root:
        raise ValueError("native root is not the Git checkout root")
    if git(root, "rev-parse", "HEAD").decode().strip() != SOURCE_HEAD or \
            git(root, "rev-parse", "HEAD^{tree}").decode().strip() != SOURCE_TREE:
        raise ValueError("native root has the wrong exact commit/tree")
    if git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("native root is dirty")
    if require_readonly and not (os.statvfs(root).f_flag & os.ST_RDONLY):
        raise ValueError("native root filesystem is not read-only")
    return {name: verify_file(root, name) for name in input_paths()}


def normalize_depfile(root: Path, object_name: str, depfile: Path,
                      expected_target: str) -> dict[str, str]:
    """Accept a single Clang -MMD -MF rule with canonical, pinned repo dependencies."""
    specs = {name: source for name, source, _ in OBJECTS}
    if object_name not in specs:
        raise ValueError("unknown native object")
    root = canonical_root(root)
    verify_root(root)
    if not depfile.is_absolute() or depfile.is_symlink() or not depfile.is_file() or \
            depfile.resolve() != depfile:
        raise ValueError("missing or aliased depfile")
    raw = depfile.read_text(encoding="utf-8")
    if raw.count(":") != 1 or "#" in raw:
        raise ValueError("unsupported depfile grammar")
    rule = raw.replace("\\\n", " ")
    if "\\" in rule:
        raise ValueError("escaped depfile paths are unsupported")
    target, dependencies = rule.split(":", 1)
    if target.strip() != expected_target or not dependencies.split():
        raise ValueError("depfile target or dependency list differs")
    paths: dict[str, str] = {}
    for token in dependencies.split():
        segments = token.lstrip("/").split("/")
        if token.startswith("//") or any(part in ("", ".", "..") for part in segments):
            raise ValueError(f"aliased native dependency: {token}")
        candidate = Path(token) if token.startswith("/") else root / token
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"native dependency escaped root: {token}") from exc
        if candidate != root / relative or relative in paths:
            raise ValueError(f"aliased or duplicate native dependency: {token}")
        paths[relative] = verify_file(root, relative)
    required = {specs[object_name], *NESTED.get(object_name, ())}
    if not required.issubset(paths):
        raise ValueError("depfile omitted required native source")
    return dict(sorted(paths.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-preflight", action="store_true",
                        help="check Git bytes without asserting a read-only builder mount")
    args = parser.parse_args()
    try:
        rows = verify_root(args.root, require_readonly=not args.source_preflight)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(74, f"native source HOLD: {exc}\n")
    for name, digest in rows.items():
        print(f"{digest}  {name}")


if __name__ == "__main__":
    main()
