"""Freeze the real Eshkol closure and one test-only post-adoption cut."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

from tests.g3r.test_same_owner_save_caller_source import definition, validate_cleanup_root


ROOT = Path(__file__).resolve().parents[2]
BASE = "3b8ea59a4527ab564d0f894ca6412ec62474cc34"
BASE_TREE = "86832ef9f1edb20383f8e7b1107ba85c97582fff"
CALLER = "native/g3r_same_owner_save_caller_extension.esk"
CALLER_SHA256 = "c4c6ddf6ff107719ae99d2ff00ba9131d0748c313887d9090a19a7e53eb10acc"
CLOSURE = "native/g3r_same_owner_o2_adoption_source_closure.txt"
CLOSURE_SHA256 = "096f6e8470fb56017333ba25c03e1bd53515a5f3eec5d22afa6598592608ecf8"
ADOPTION_SHA256 = "03bfb2868b238a02c8310067458646a12a0b67aa12af546a8871050b11ee837c"
MANUAL_PREFILL = ("native/g3t_manual_p1_extension.esk",
                  "native/g3m_prefill_p1_extension.esk")
FIXTURES = ("tests/g3r/save_post_adoption_cut_root.esk",
            "tests/g3r/save_post_adoption_cut_runtime.esk")
FE9 = "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9"
FE9_RUNNER_SHA256 = "7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa"
FE9_RUNTIME_SHA256 = "32cd446a3aeaa2e78bbe49b0c04cda961b8e1eeb7bb0ea53ec5e55c11f0c183e"
F31 = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"
LOCK_SHA256 = "8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe"
COMPILER_MANIFEST = "tests/g3r/post_adoption_fe9_compiler.tsv"
COMPILER_MANIFEST_SHA256 = "6aa744a3b6a7ebda7566c183e8fe1134d95cd1ee02ca0a4ad1ac6785a4e59f23"

ADOPTION_RETURN = """      (vector-set! shell-cell 0
        (g3r-save-o2-adopt-held-internal
          generator model resolved tokenizer dataset optimizer capsule))
"""
ABANDON = "(g3r-save-o2-abandon-internal! owned-shell)"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_caller(source: str) -> str:
    caller = definition(source,
                        ["g3r-save-c2-capture-internal", "generator", "model",
                         "resolved", "tokenizer", "dataset", "optimizer",
                         "capsule"])
    try:
        validate_cleanup_root(caller)
    except AssertionError as error:
        raise ValueError("accepted caller cleanup root changed") from error
    if source.count(ADOPTION_RETURN) != 1 or source.count(ABANDON) != 1:
        raise ValueError("accepted caller publication or cleanup point changed")
    source = source.replace(
        ADOPTION_RETURN,
        ADOPTION_RETURN +
        "      (g3r-save-cut-after-publication! (vector-ref shell-cell 0))\n",
    )
    return source.replace(ABANDON, "(g3r-save-cut-abandon! owned-shell)")


def validate_closure(files: list[str]) -> None:
    if len(files) != len(set(files)) or len(files) != 96 or files[0] != \
            "native/g3r_same_owner_o2_adoption_root.esk":
        raise ValueError("invalid recursive Eshkol source closure")
    for name in (*files, *MANUAL_PREFILL, *FIXTURES):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not name.endswith(".esk"):
            raise ValueError("unsafe Eshkol closure or fixture path")
    required = {CALLER, "native/g3r_same_owner_o2_adoption_extension.esk",
                "native/c2_training_state_extension.esk",
                "internal/p1/lib/transformer/module.esk",
                "native/o2_wave2_extension.esk",
                "native/g3r_training_controls_extension.esk"}
    if not required.issubset(files):
        raise ValueError("real P1/O2/C2/adoption leaves are missing")


def parse_unique_tsv(data: str) -> dict[str, str]:
    values = {}
    for row in data.splitlines():
        if row.startswith("#"):
            continue
        parts = row.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] in values:
            raise ValueError("malformed or duplicate compiler manifest row")
        values[parts[0]] = parts[1]
    if not values:
        raise ValueError("empty compiler manifest")
    return values


def validate_toolchain_manifest() -> dict[str, str]:
    if sha256((ROOT / "toolchain/eshkol.lock").read_bytes()) != LOCK_SHA256:
        raise ValueError("supported LLVM21 toolchain lock changed")
    manifest = (ROOT / COMPILER_MANIFEST).read_bytes()
    if sha256(manifest) != COMPILER_MANIFEST_SHA256:
        raise ValueError("dedicated fe9 compiler manifest changed")
    values = parse_unique_tsv(manifest.decode("utf-8"))
    expected = {
        "schema": "g3r-post-adoption-fe9-compiler-v1",
        "compiler_source_commit": FE9,
        "compiler_source_tree": "66c21f7ec19b1b4a42199fa30ed8e0e9727021bf",
        "runner_sha256": FE9_RUNNER_SHA256,
        "runtime_archive_sha256": FE9_RUNTIME_SHA256,
        "clang_sha256": "82481792aef943c1750ae5fd71e5a5737212741337debd0fe5d28bd82dd018e9",
        "container_digest": F31,
        "supported_os": "ubuntu-22.04",
        "supported_arch": "x86_64",
        "llvm_version": "21.1.8",
        "clang_version": "21.1.8",
        "toolchain_lock_sha256": LOCK_SHA256,
        "c7b_release_sha256": "736d282d210cf92b3141bd8b9405232f627356573231a10b2626f4262384d91e",
        "status": "source-only; capture build and runtime HOLD",
    }
    if values != expected:
        raise ValueError("dedicated fe9 compiler identity changed")
    lock = parse_unique_tsv((ROOT / "toolchain/eshkol.lock").read_text())
    for key, value in (("eshkol_commit", FE9), ("llvm_version", "21.1.8"),
                       ("clang_version", "21.1.8"), ("supported_os", "ubuntu-22.04"),
                       ("supported_arch", "x86_64")):
        if lock.get(key) != value:
            raise ValueError("compiler manifest conflicts with toolchain lock")
    return values


def validate_source() -> list[str]:
    validate_toolchain_manifest()
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"],
                   cwd=ROOT, check=True)
    base_tree = subprocess.check_output(["git", "rev-parse", f"{BASE}^{{tree}}"],
                                        cwd=ROOT, text=True).strip()
    if base_tree != BASE_TREE:
        raise ValueError("accepted source tree changed")
    for name, digest in ((CALLER, CALLER_SHA256), (CLOSURE, CLOSURE_SHA256)):
        if sha256((ROOT / name).read_bytes()) != digest:
            raise ValueError(f"pinned {name} changed")

    files = (ROOT / CLOSURE).read_text().splitlines()
    validate_closure(files)
    if sha256((ROOT / "native/g3r_same_owner_o2_adoption_extension.esk").read_bytes()) != \
            ADOPTION_SHA256:
        raise ValueError("pinned adoption leaf changed")
    changed = subprocess.run(["git", "diff", "--name-only", BASE, "--",
                              *files, *MANUAL_PREFILL],
                             cwd=ROOT, check=True, capture_output=True, text=True)
    if changed.stdout:
        raise ValueError("production Eshkol closure differs from accepted base")
    for name in FIXTURES:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"],
                                            cwd=ROOT)
        if (ROOT / name).read_bytes() != committed:
            raise ValueError(f"uncommitted fixture bytes: {name}")
    if (ROOT / COMPILER_MANIFEST).read_bytes() != subprocess.check_output(
            ["git", "show", f"HEAD:{COMPILER_MANIFEST}"], cwd=ROOT):
        raise ValueError("uncommitted compiler manifest")
    frozen_caller((ROOT / CALLER).read_text())
    return files


def prepare(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a fresh path")
    if not output.is_absolute():
        raise ValueError("output must be absolute")
    if ROOT.resolve() == output.resolve() or ROOT.resolve() in output.resolve().parents:
        raise ValueError("output must be outside the source checkout")
    files = validate_source()

    output.mkdir(parents=True)
    source_root = output / "source"
    copied = []
    for name in [*files, *MANUAL_PREFILL]:
        relative = Path(name)
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        copied.append(target)
    for name in FIXTURES:
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        copied.append(target)
    generated = source_root / "tests/g3r/g3r_same_owner_save_cut_caller.esk"
    generated.write_text(frozen_caller((ROOT / CALLER).read_text()))
    copied.append(generated)
    compiler_copy = output / "COMPILER.tsv"
    shutil.copyfile(ROOT / COMPILER_MANIFEST, compiler_copy)
    copied.append(compiler_copy)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"],
                                   cwd=ROOT, text=True).strip()
    manifest = output / "PREPARED.tsv"
    manifest.write_text("".join(f"{key}\t{value}\n" for key, value in (
        ("source_commit", commit), ("source_tree", tree),
        ("accepted_source_commit", BASE), ("accepted_source_tree", BASE_TREE),
        ("accepted_caller_sha256", CALLER_SHA256),
        ("accepted_adoption_sha256", ADOPTION_SHA256),
        ("closure_sha256", CLOSURE_SHA256),
        ("cut", "post-adoption longjmp and fail-stop abandon only"),
        ("fail_stop_mode", "fail-stop"),
        ("fail_stop_expected_exit", "134"),
        ("fail_stop_injection_marker",
         "G3-R post-adoption O2 release fault injected"),
        ("compiler_commit", FE9), ("compiler_runner_sha256", FE9_RUNNER_SHA256),
        ("compiler_source_tree", "66c21f7ec19b1b4a42199fa30ed8e0e9727021bf"),
        ("runtime_archive_sha256", FE9_RUNTIME_SHA256),
        ("clang_sha256", "82481792aef943c1750ae5fd71e5a5737212741337debd0fe5d28bd82dd018e9"),
        ("container_digest", F31),
        ("supported_host", "ubuntu-22.04/x86_64"),
        ("supported_llvm_clang", "21.1.8"),
        ("toolchain_lock_sha256", LOCK_SHA256),
        ("compiler_manifest_sha256", COMPILER_MANIFEST_SHA256),
        ("closure_count", str(len(files))),
        ("status", "source-prepared; compiled gate not run"),
    )))
    copied.append(manifest)
    sums = output / "SOURCE-SHA256SUMS"
    sums.write_text("".join(
        f"{sha256(path.read_bytes())}  {path.relative_to(output)}\n"
        for path in sorted(copied)))
    for path in copied + [sums]:
        path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
