#!/usr/bin/env python3
"""Stage the exact approved production closure for the G3-R partial-claim cut."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess

from tests.d2.test_resources import write_d1_resource

ROOT = Path(__file__).resolve().parents[2]
BASE = "3b8ea59a4527ab564d0f894ca6412ec62474cc34"
TREE = "86832ef9f1edb20383f8e7b1107ba85c97582fff"
FE9 = "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9"
F31 = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"
BYTE_FINGERPRINT = (
    "sha256:eshkol-byte-tokenizer-v1:"
    "aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704"
)
CLOSURE = "native/g3r_same_owner_o2_adoption_source_closure.txt"
MANIFEST = "native/g3r_same_owner_capture_main_source.sha256"
PINS = {
    "toolchain/eshkol.lock": "8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe",
    "tests/tr3_lease/runtime_candidate.tsv": "46097980f65a6fee1643590e7dd983cfebf91983a0d8487832a7cacafebd8b99",
    "tests/d2/test_resources.py": "0de1f3326b663c16e45d7fed5efc43e49a30b4fa540b13df5adc2c67ab2ee26d",
    "tests/c2/test_handler_order.py": "f100005dc13ecf7a01868ef107b02a276b87bda105abe080cddb82883097057d",
    "scripts/test-g3r-capture-partial-claim.sh": "2a1e7b0c7fc8e69b25b13c82a761705d927118688b865140774b9436ad70eade",
    "tests/g3r/capture_partial_claim_cut_root.esk": "a974d60b46fa43ed4e4fd9f8f3951359c08173def624b244d5083445ec01ecfa",
    "tests/g3r/capture_partial_claim_cut_runtime.esk": "c9bee87c65270cb336433ec2ecce0fe3c98de6a4f9e352b2a99ff74a6240dfdb",
    "tests/g3r/capture_partial_claim_cut_bridge.c": "83de21c7a6007df3170869bf0ec63fd2c6133d3de498cca71ade331b5c70923c",
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
TOOLS = (
    "tests/g3r/prepare_capture_partial_claim_cut.py",
    "tests/g3r/test_capture_partial_claim_cut_source.py",
    "scripts/test-g3r-capture-partial-claim.sh",
    "tests/d2/test_resources.py",
    "tests/c2/test_handler_order.py",
    "toolchain/eshkol.lock",
    "tests/tr3_lease/runtime_candidate.tsv",
)
ALLOWED_DELTA = set(EXTRA[2:]) | set(TOOLS[:3]) | {
    "docs/g3/G3_R_CAPTURE_PARTIAL_CLAIM_GATE.md",
}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def committed_bytes(revision: str, name: str) -> bytes:
    return subprocess.check_output(("git", "show", f"{revision}:{name}"), cwd=ROOT)


def verify_file(name: str, revision: str, path: Path | None = None) -> None:
    path = path or ROOT / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked source: {name}")
    raw = path.read_bytes()
    if name in PINS and sha256(raw).hexdigest() != PINS[name]:
        raise ValueError(f"pinned source changed: {name}")
    if raw != committed_bytes(revision, name):
        raise ValueError(f"working source differs from {revision}: {name}")


def verify(expected_head: str) -> list[str]:
    head = git("rev-parse", "HEAD")
    if head != expected_head:
        raise ValueError("fixture HEAD differs from explicit source commit")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("fixture worktree is not clean")
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
    delta = set(git("diff", "--name-only", BASE, head).splitlines())
    if not delta.issubset(ALLOWED_DELTA):
        raise ValueError("fixture HEAD changes unapproved source paths")
    for name, expected in PINS.items():
        verify_file(name, BASE if name not in ALLOWED_DELTA else head)
    for name in TOOLS:
        verify_file(name, head)
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
        verify_file(name, BASE)
    for name in EXTRA[:2]:
        verify_file(name, BASE)
    for name in EXTRA[2:]:
        verify_file(name, head)
    return paths


def prepare(output: Path, expected_head: str) -> None:
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("output must be a fresh absolute path")
    paths = verify(expected_head)
    output.mkdir(parents=True)
    staged = output / "source"
    copied: list[Path] = []
    for name in (*paths, *EXTRA, *TOOLS):
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(committed_bytes(expected_head, name))
        copied.append(target)
    # The installed root asks for this basename, although the pinned closure
    # records its source in internal/e3/lib.
    e3 = staged / "native/e3_d2_restore.esk"
    shutil.copyfile(staged / "internal/e3/lib/e3_d2_restore.esk", e3)
    copied.append(e3)
    corpus = output / "corpus"
    corpus.mkdir()
    write_d1_resource(corpus, ((1, 2, 3, 4),),
                      fingerprint=BYTE_FINGERPRINT, vocab=256)
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
        f"fixture_commit\t{expected_head}\n"
        f"fixture_tree\t{git('rev-parse', expected_head + '^{tree}')}\n"
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
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    prepare(args.output, args.head)
