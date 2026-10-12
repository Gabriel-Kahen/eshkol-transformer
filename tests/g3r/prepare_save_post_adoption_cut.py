"""Freeze the real Eshkol closure and one test-only post-adoption cut."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

from tests.g3r.test_same_owner_save_caller_source import definition, validate_cleanup_root


ROOT = Path(__file__).resolve().parents[2]
BASE = "ac0ac36e5da6ec4e7bc19c4f4dc91c78cbfc9f5e"
CALLER = "native/g3r_same_owner_save_caller_extension.esk"
CALLER_SHA256 = "01a14412f39fbe83413938a5536eec89900a9cf8e1bd541d488103dfca91bdef"
CLOSURE = "native/g3r_same_owner_o2_adoption_source_closure.txt"
CLOSURE_SHA256 = "096f6e8470fb56017333ba25c03e1bd53515a5f3eec5d22afa6598592608ecf8"
MANUAL_PREFILL = ("native/g3t_manual_p1_extension.esk",
                  "native/g3m_prefill_p1_extension.esk")
FIXTURES = ("tests/g3r/save_post_adoption_cut_root.esk",
            "tests/g3r/save_post_adoption_cut_runtime.esk")
FE9 = "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9"
FE9_RUNNER_SHA256 = "7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa"
FE9_RUNTIME_SHA256 = "32cd446a3aeaa2e78bbe49b0c04cda961b8e1eeb7bb0ea53ec5e55c11f0c183e"
F31 = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"

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


def prepare(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a fresh path")
    if not output.is_absolute():
        raise ValueError("output must be absolute")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"],
                   cwd=ROOT, check=True)
    for name, digest in ((CALLER, CALLER_SHA256), (CLOSURE, CLOSURE_SHA256)):
        if sha256((ROOT / name).read_bytes()) != digest:
            raise ValueError(f"pinned {name} changed")

    files = (ROOT / CLOSURE).read_text().splitlines()
    changed = subprocess.run(["git", "diff", "--name-only", BASE, "--",
                              *files, *MANUAL_PREFILL],
                             cwd=ROOT, check=True, capture_output=True, text=True)
    if changed.stdout:
        raise ValueError("production Eshkol closure differs from accepted base")
    if len(files) != len(set(files)) or not files or files[0] != \
            "native/g3r_same_owner_o2_adoption_root.esk":
        raise ValueError("invalid recursive Eshkol source closure")
    required = {CALLER, "native/g3r_same_owner_o2_adoption_extension.esk",
                "native/c2_training_state_extension.esk",
                "internal/p1/lib/transformer/module.esk",
                "native/o2_wave2_extension.esk",
                "native/g3r_training_controls_extension.esk"}
    if not required.issubset(files):
        raise ValueError("real P1/O2/C2/adoption leaves are missing")
    for name in FIXTURES:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"],
                                            cwd=ROOT)
        if (ROOT / name).read_bytes() != committed:
            raise ValueError(f"uncommitted fixture bytes: {name}")

    output.mkdir(parents=True)
    source_root = output / "source"
    copied = []
    for name in [*files, *MANUAL_PREFILL]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not name.endswith(".esk"):
            raise ValueError("unsafe Eshkol closure path")
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

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"],
                                   cwd=ROOT, text=True).strip()
    manifest = output / "PREPARED.tsv"
    manifest.write_text("".join(f"{key}\t{value}\n" for key, value in (
        ("source_commit", commit), ("source_tree", tree),
        ("accepted_caller_commit", BASE), ("accepted_caller_sha256", CALLER_SHA256),
        ("compiler_commit", FE9), ("compiler_runner_sha256", FE9_RUNNER_SHA256),
        ("runtime_archive_sha256", FE9_RUNTIME_SHA256),
        ("container_digest", F31),
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
