#!/usr/bin/env python3
"""Close the supported candidate fixture's source and retained evidence sets."""

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import sys


ALLOWLIST_SHA256 = "02106f31375643c1744f7f4a8feb1aa96d992ef245eef9e9c02d367a2877ecd7"


def require_regular(path: Path) -> None:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"nonregular or symlink input: {path}")
    if path != path.resolve(strict=True):
        raise ValueError(f"noncanonical input path: {path}")


def load_allowlist(path: Path) -> dict:
    require_regular(path)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != ALLOWLIST_SHA256:
        raise ValueError("candidate depfile allowlist hash differs")
    value = json.loads(raw)
    if value.get("version") != 1:
        raise ValueError("candidate depfile allowlist version differs")
    return value


def canonical_dependency(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"relative depfile prerequisite: {path}")
    current = Path("/")
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"symlink depfile prerequisite: {path}")
    resolved = path.resolve(strict=True)
    require_regular(resolved)
    return resolved


def depfile_paths(path: Path, expected_target: Path) -> list[Path]:
    require_regular(path)
    raw = path.read_text()
    raw = raw.replace("\\\n", " ")
    if raw.count(":") != 1:
        raise ValueError(f"unexpected depfile format: {path}")
    target, prerequisites = raw.split(":", 1)
    if target.strip() != str(expected_target) or not prerequisites.strip():
        raise ValueError(f"wrong depfile target or empty prerequisites: {path}")
    # All admitted worktree/evidence/toolchain roots have no shell whitespace.
    # A Make escape here would need an independently reviewed path parser.
    if "\\" in prerequisites:
        raise ValueError(f"escaped depfile prerequisite: {path}")
    result = [canonical_dependency(Path(item)) for item in prerequisites.split()]
    return result


def compare_depfile(path: Path, target: Path, expected: set[Path]) -> list[Path]:
    actual = depfile_paths(path, target)
    if set(actual) != expected:
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        raise ValueError(f"{path} closure differs: missing={missing}, extra={extra}")
    return actual


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: checker PROJECT_ROOT EVIDENCE")
    project, evidence = map(Path, sys.argv[1:])
    allowlist = load_allowlist(project / "tests/g3r/candidate_depfile_allowlist.json")
    admitted = set()
    native = allowlist["native"]
    common = {project / item for item in allowlist["eshkol_common"]}
    private = evidence / "overlay/transformer/module.esk"
    ordinary = project / "internal/p1/lib/transformer/module.esk"
    baseline = evidence / "baseline/transformer/module.esk"
    staged = evidence / "off/stage/transformer/module.esk"
    expected = {}
    for mode in ("normal", "sanitize"):
        for stem in native:
            if stem in ("ordinary_bridge", "p1_identity"):
                continue
            expected[evidence / mode / f"{stem}.d"] = (
                evidence / mode / f"{stem}.o",
                {project / item for item in native[stem]},
            )
        expected[evidence / mode / "candidate.d"] = (
            evidence / mode / "candidate.raw.o",
            common | {project / item for item in allowlist["eshkol_candidate_extra"]} | {private},
        )
    expected[evidence / "off/ordinary_bridge.d"] = (
        evidence / "off/ordinary_bridge.o",
        {project / item for item in native["ordinary_bridge"]},
    )
    for variant in ("baseline", "current"):
        expected[evidence / "off" / f"{variant}.d"] = (
            evidence / "off" / f"{variant}.o",
            common | {project / item for item in allowlist["eshkol_off_extra"]} | {staged},
        )
        native_inputs = {project / item for item in native["p1_identity"]}
        if variant == "baseline":
            native_inputs.remove(project / "native/p1_identity.c")
            native_inputs.add(evidence / "baseline/p1_identity.c")
        expected[evidence / "off" / f"{variant}.native.d"] = (
            evidence / "off" / f"{variant}.native.o", native_inputs,
        )
    depfiles = set(evidence.glob("normal/*.d"))
    depfiles |= set(evidence.glob("sanitize/*.d"))
    depfiles |= set(evidence.glob("off/*.d"))
    if depfiles != set(expected):
        missing = sorted(set(expected) - depfiles)
        extra = sorted(depfiles - set(expected))
        raise ValueError(f"depfile set mismatch: missing={missing}, extra={extra}")
    for depfile in sorted(depfiles):
        target, dependencies = expected[depfile]
        admitted.update(compare_depfile(depfile, target, dependencies))
    for variant, original in (("baseline", baseline), ("current", ordinary)):
        snapshot = evidence / "off" / f"{variant}.source.esk"
        require_regular(snapshot)
        if snapshot.read_bytes() != original.read_bytes() or snapshot.read_bytes() != staged.read_bytes():
            raise ValueError(f"{variant} staged source differs from its retained origin")
        digest = sha256(snapshot.read_bytes()).hexdigest()
        require_regular(evidence / "off" / f"{variant}.source.sha256")
        source_record = (evidence / "off" / f"{variant}.source.sha256").read_text()
        if source_record != f"{digest}  {snapshot}\n":
            raise ValueError(f"{variant} source digest record changed")
        for phase in ("pre", "post"):
            record = evidence / "off" / f"{variant}.stage.{phase}.sha256"
            require_regular(record)
            if record.read_text() != f"{digest}  {staged}\n":
                raise ValueError(f"{variant} staged {phase} digest differs")
        phase_provenance = evidence / "off" / f"{variant}.compiler-provenance.tsv"
        require_regular(phase_provenance)
        if phase_provenance.read_bytes() != (evidence / "compiler-provenance.tsv").read_bytes():
            raise ValueError(f"{variant} compiler provenance differs")
    explicit = (
        "scripts/test-g3r-candidate-construction.sh",
        "scripts/build-g3r-candidate-p1-overlay.py",
        "scripts/common.sh",
        "tests/q0/test_g3r_candidate_construction_static.py",
        "tests/g3r/check_candidate_construction_inputs.py",
        "tests/g3r/candidate_depfile_allowlist.json",
        "tests/g3r/candidate_construction_test.esk",
        "tests/g3r/candidate_construction_bridge.c",
        "tests/p1/i2_construction_test.esk",
        "tests/p1/i2_construction_test_bridge.c",
        "native/g3r_candidate_construction_root.esk",
        "native/g3r_candidate_construction_extension.esk",
        "native/g3r_candidate_p1_inside.esk",
        "toolchain/eshkol.lock",
    )
    for relative in explicit:
        admitted.add(project / relative)
    for path in (private, ordinary, baseline, staged,
                 evidence / "off/baseline.source.esk",
                 evidence / "off/current.source.esk",
                 evidence / "baseline/p1_identity.c"):
        admitted.add(path)
    manifest = evidence / "source-inputs.sha256"
    with manifest.open("x") as stream:
        for path in sorted(admitted):
            require_regular(path)
            stream.write(f"{sha256(path.read_bytes()).hexdigest()}  {path}\n")
    # The final seal must cover every artifact the gate could inspect. An
    # evidence symlink/FIFO/socket cannot evade the regular-file hash walk.
    for directory, names, files in os.walk(evidence, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError(f"non-directory evidence entry: {path}")
        for name in files:
            require_regular(Path(directory) / name)
    print(f"closed source inputs: {len(admitted)}, raw depfiles: {len(depfiles)}")


if __name__ == "__main__":
    main()
