#!/usr/bin/env python3
"""Stage the exact approved production closure for the G3-R partial-claim cut."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess

from tests.d2.prepare_public_resources import FINGERPRINT, VOCAB
from tests.d2.test_resources import write_d1_resource

ROOT = Path(__file__).resolve().parents[2]
BASE = "3b8ea59a4527ab564d0f894ca6412ec62474cc34"
TREE = "86832ef9f1edb20383f8e7b1107ba85c97582fff"
FE9 = "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9"
F31 = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"
CLOSURE = "native/g3r_same_owner_o2_adoption_source_closure.txt"
MANIFEST = "native/g3r_same_owner_capture_main_source.sha256"
PINS = {
    CLOSURE: "096f6e8470fb56017333ba25c03e1bd53515a5f3eec5d22afa6598592608ecf8",
    MANIFEST: "950d967f30691934d6ff86b69fe78df2a7b877c0e24f60d62fd7a58fbbd5609f",
    "native/c2_training_state_extension.esk": "130a4e4d6de5abe3ae7bcb0fb14eb631b80b7834037b938fed00c25343454dab",
    "native/g3r_same_owner_save_caller_extension.esk": "c4c6ddf6ff107719ae99d2ff00ba9131d0748c313887d9090a19a7e53eb10acc",
    "native/o2_optimizer.c": "d81ff51c9b615087cefa804942031465a996ce26a924b77330fe9174e5b9d9f5",
    "native/g3t_manual_p1_extension.esk": "a034856ea2068b52d8717859309bb0dd030c375c6155e89de2eae3413504dabb",
    "native/g3m_prefill_p1_extension.esk": "11bcf13718a2ea15409c150db6ed7d45a613f7e88905c0cb5e23e04355923fa9",
}
EXTRA = (
    "native/g3t_manual_p1_extension.esk",
    "native/g3m_prefill_p1_extension.esk",
    "tests/g3r/capture_partial_claim_cut_root.esk",
    "tests/g3r/capture_partial_claim_cut_runtime.esk",
    "tests/g3r/capture_partial_claim_cut_bridge.c",
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def verify() -> list[str]:
    lock = (ROOT / "toolchain/eshkol.lock").read_text()
    candidate = (ROOT / "tests/tr3_lease/runtime_candidate.tsv").read_text()
    if f"eshkol_commit\t{FE9}\n" not in lock or \
            f"container_digest\t{F31}\n" not in candidate:
        raise ValueError("supported compiler or container source pin changed")
    if git("rev-parse", f"{BASE}^{{tree}}") != TREE:
        raise ValueError("approved commit/tree pair changed")
    if subprocess.run(("git", "merge-base", "--is-ancestor", BASE, "HEAD"),
                      cwd=ROOT, check=False).returncode:
        raise ValueError("old or divergent source history")
    for name, expected in PINS.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"pinned source changed: {name}")
    paths = (ROOT / CLOSURE).read_text().splitlines()
    if len(paths) != 96 or len(set(paths)) != 96 or paths[0] != \
            "native/g3r_same_owner_o2_adoption_root.esk":
        raise ValueError("expected exact 96-path production closure")
    rows = [line.split("  ", 1) for line in (ROOT / MANIFEST).read_text().splitlines()]
    if len(rows) != 96 or [name for _, name in rows] != paths:
        raise ValueError("production byte manifest and closure differ")
    for expected, name in rows:
        if Path(name).is_absolute() or ".." in Path(name).parts or \
                not name.endswith(".esk") or digest(ROOT / name) != expected:
            raise ValueError(f"production closure byte mismatch: {name}")
        if git("show", f"{BASE}:{name}") != (ROOT / name).read_text().rstrip("\n"):
            raise ValueError(f"closure differs from approved commit: {name}")
    for name in EXTRA[:2]:
        if git("show", f"{BASE}:{name}") != (ROOT / name).read_text().rstrip("\n"):
            raise ValueError(f"manual prefill differs from approved commit: {name}")
    return paths


def prepare(output: Path) -> None:
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("output must be a fresh absolute path")
    paths = verify()
    output.mkdir(parents=True)
    staged = output / "source"
    copied: list[Path] = []
    for name in (*paths, *EXTRA):
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        copied.append(target)
    # The installed root asks for this basename, although the pinned closure
    # records its source in internal/e3/lib.
    e3 = staged / "native/e3_d2_restore.esk"
    shutil.copyfile(staged / "internal/e3/lib/e3_d2_restore.esk", e3)
    copied.append(e3)
    corpus = output / "corpus"
    corpus.mkdir()
    write_d1_resource(corpus, ((1, 2, 3, 4),),
                      fingerprint=FINGERPRINT, vocab=VOCAB)
    copied.extend(sorted(corpus.iterdir()))
    expected = output / "expected-eshkol-closure.txt"
    expected.write_text("\n".join(sorted(
        (set(paths) - {paths[0], "internal/e3/lib/e3_d2_restore.esk"})
        | {"native/e3_d2_restore.esk", *EXTRA[:4]}
    )) + "\n")
    copied.append(expected)
    manifest = output / "PREPARED.tsv"
    manifest.write_text(
        f"approved_commit\t{BASE}\n"
        f"approved_tree\t{TREE}\n"
        f"fixture_commit\t{git('rev-parse', 'HEAD')}\n"
        f"fixture_tree\t{git('rev-parse', 'HEAD^{tree}')}\n"
        f"compiler_commit\t{FE9}\n"
        f"container_digest\t{F31}\n"
        "status\tsource-prepared; compiled gate HOLD\n"
    )
    copied.append(manifest)
    sums = output / "SOURCE-SHA256SUMS"
    sums.write_text("".join(
        f"{digest(path)}  {path.relative_to(output)}\n" for path in sorted(copied)
    ))
    for path in (*copied, sums):
        path.chmod(0o444)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
