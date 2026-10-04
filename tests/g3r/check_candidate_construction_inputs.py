#!/usr/bin/env python3
"""Close the supported candidate fixture's source and retained evidence sets."""

from hashlib import sha256
import os
from pathlib import Path
import stat
import sys


def require_regular(path: Path) -> None:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"nonregular or symlink input: {path}")
    if path != path.resolve(strict=True):
        raise ValueError(f"noncanonical input path: {path}")


def depfile_paths(path: Path) -> list[Path]:
    require_regular(path)
    raw = path.read_text()
    raw = raw.replace("\\\n", " ")
    if raw.count(":") != 1:
        raise ValueError(f"unexpected depfile format: {path}")
    target, prerequisites = raw.split(":", 1)
    if not target.strip() or not prerequisites.strip():
        raise ValueError(f"empty depfile target/prerequisites: {path}")
    # All admitted worktree/evidence/toolchain roots have no shell whitespace.
    # A Make escape here would need an independently reviewed path parser.
    if "\\" in prerequisites:
        raise ValueError(f"escaped depfile prerequisite: {path}")
    result = [Path(item) for item in prerequisites.split()]
    if any(not item.is_absolute() for item in result):
        raise ValueError(f"relative depfile prerequisite: {path}")
    return result


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: checker PROJECT_ROOT EVIDENCE ESHKOL_SOURCE")
    project, evidence, eshkol = map(Path, sys.argv[1:])
    roots = (project, evidence / "overlay", evidence / "baseline",
             evidence / "off/stage", eshkol)
    admitted = set()
    c_stems = (
        "i2_bridge", "data_io", "checkpoint_io", "kernel_abi",
        "i64_tensor", "t1_i64_shell", "n3k_primitives_provider",
        "n2_primitives_provider", "a2_attention_provider", "m3t_transport",
        "m3t_f32_integration", "candidate_bridge", "candidate",
    )
    expected = {evidence / mode / f"{stem}.d"
                for mode in ("normal", "sanitize") for stem in c_stems}
    expected |= {evidence / "off" / f"{stem}.d"
                 for stem in ("ordinary_bridge", "baseline", "current",
                              "baseline.native", "current.native")}
    depfiles = set(evidence.glob("normal/*.d"))
    depfiles |= set(evidence.glob("sanitize/*.d"))
    depfiles |= set(evidence.glob("off/*.d"))
    if depfiles != expected:
        raise ValueError(f"depfile set mismatch: missing={expected - depfiles}, extra={depfiles - expected}")
    for depfile in sorted(depfiles):
        dependencies = depfile_paths(depfile)
        for path in dependencies:
            require_regular(path)
            if not any(path.is_relative_to(root) for root in roots):
                raise ValueError(f"depfile escaped source roots: {path}")
            admitted.add(path)
    private = evidence / "overlay/transformer/module.esk"
    ordinary = project / "internal/p1/lib/transformer/module.esk"
    baseline = evidence / "baseline/transformer/module.esk"
    for mode in ("normal", "sanitize"):
        deps = set(depfile_paths(evidence / mode / "candidate.d"))
        if private not in deps or ordinary in deps:
            raise ValueError(f"{mode} private overlay closure is wrong")
    staged = evidence / "off/stage/transformer/module.esk"
    for variant, original in (("baseline", baseline), ("current", ordinary)):
        deps = set(depfile_paths(evidence / "off" / f"{variant}.d"))
        if staged not in deps or private in deps or baseline in deps or ordinary in deps:
            raise ValueError(f"{variant} fixed-path ordinary closure is wrong")
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
