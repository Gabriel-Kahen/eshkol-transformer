#!/usr/bin/env python3
"""Fail closed on missing or substituted terminal compiler evidence."""

import json
from pathlib import Path
import re
import subprocess
import sys

from tests.shared.prepare_eshkol_asan_ir import prepare, sha


FIXTURES = ("terminal", "feature-off", "parity-original", "parity-overlay",
            "unmarked", "decode-first", "decode-second", "precommit", "postzero")
NATIVE = ("composer_native", "kernel_fail_allocator", "i2_native_helpers",
          "call_entry_allocation_shim", "p1_identity", "data_io",
          "checkpoint_io", "n3k_primitives_provider", "g3n_primitives_provider",
          "n2_primitives_provider", "g3c4_primitives_provider",
          "g3s_sampling_provider", "a2_attention_provider", "a2_kv_cache",
          "t1_i64_shell", "m3_i64_integration", "m3_model")
REPORT = re.compile(r"\b__asan_report_(?:load|store)\d+\b")


def require_files(directory: Path, expected: set[str]) -> None:
    actual = {entry.name for entry in directory.iterdir()}
    if actual != expected:
        raise ValueError(f"{directory}: file set differs: {sorted(actual ^ expected)}")
    if any(not entry.is_file() or entry.is_symlink() or not entry.stat().st_size
           for entry in directory.iterdir()):
        raise ValueError(f"{directory}: nonregular, symlink, or empty artifact")


def dep_target(depfile: Path, target: Path, source: Path) -> None:
    text = depfile.read_text()
    if (not text.startswith(f"{target}:") or not re.search(
            rf"(?:^|\s){re.escape(str(source))}(?:\s|$)", text)):
        raise ValueError(f"{depfile}: wrong target or missing actual source")


def source_identity(evidence: Path, label: str) -> Path:
    for suffix in ("before", "after"):
        file = evidence / f"{label}-source.{suffix}"
        if not file.is_file() or file.is_symlink():
            raise ValueError(f"{label}: unsafe source identity record")
    before = (evidence / f"{label}-source.before").read_text()
    after = (evidence / f"{label}-source.after").read_text()
    if before != after:
        raise ValueError(f"{label}: source changed during compile/link")
    digest, source = before.rstrip("\n").split("  ", 1)
    path = Path(source)
    if not path.is_file() or path.is_symlink() or sha(path.read_bytes()) != digest:
        raise ValueError(f"{label}: source identity changed after compile/link")
    return path


def recorded(evidence: Path, label: str, expected: int) -> None:
    for suffix in ("command", "stdout", "stderr", "exit"):
        file = evidence / f"{label}.{suffix}"
        if not file.is_file() or file.is_symlink():
            raise ValueError(f"{label}: missing/unsafe raw stream {suffix}")
    if (evidence / f"{label}.exit").read_text() != f"{expected}\n":
        raise ValueError(f"{label}: unexpected numeric exit")
    if not (evidence / f"{label}.command").stat().st_size:
        raise ValueError(f"{label}: command missing")


def verify(mode_dir: Path, evidence: Path, mode: str) -> None:
    if mode not in ("normal", "repeat", "sanitize"):
        raise ValueError("unsupported mode")
    generated = mode_dir / "generated"
    expected = set()
    for fixture in FIXTURES:
        expected.update(f"{fixture}.{suffix}" for suffix in
                        ("raw.o", "raw.o.bc", "raw.o.ll", "d"))
        if mode == "sanitize":
            expected.update(f"{fixture}.{suffix}" for suffix in
                            ("asan.ll", "asan.json", "asan.o"))
    require_files(generated, expected)
    objects = mode_dir / "objects"
    require_files(objects, {f"{name}.{suffix}" for name in NATIVE
                            for suffix in ("o", "d")})
    for name in NATIVE:
        recorded(evidence, f"{mode}-native-{name}-compile", 0)
        depfile = objects / f"{name}.d"
        root = Path(__file__).resolve().parents[2]
        native_source = {
            "composer_native": root / "tests/g3c4/p2_g2_terminal_composer_native.c",
            "kernel_fail_allocator": root / "tests/m3t/kernel_fail_allocator.c",
            "i2_native_helpers": root / "native/i2_wave2_package_bridge.c",
            "call_entry_allocation_shim": root / "tests/g3c4/call_entry_allocation_shim.cpp",
            "p1_identity": root / "native/p1_identity.c",
            "m3_i64_integration": root / "src/eshkol_transformer/m3_i64_integration.c",
            "m3_model": root / "src/eshkol_transformer/m3_model.c",
        }.get(name, root / f"native/{name}.c")
        dep_target(depfile, objects / f"{name}.o", native_source)
    recorded(evidence, f"{mode}-native-archive", 0)
    for fixture in FIXTURES:
        label = f"{mode}-{fixture}" if fixture != "terminal" else mode
        raw = generated / f"{fixture}.raw.o"
        source = source_identity(evidence, label)
        root = Path(__file__).resolve().parents[2]
        expected_source = {
            "terminal": root / "tests/g3c4/p2_g2_terminal_composer_test.esk",
            "feature-off": root / "tests/g3c4/p2_g2_terminal_composer_feature_off_test.esk",
        }.get(fixture, mode_dir / "overlays" / f"{fixture}.esk")
        if source != expected_source:
            raise ValueError(f"{label}: unexpected fixture source")
        dep_target(generated / f"{fixture}.d", raw, source)
        recorded(evidence, f"{label}-compile", 0)
        recorded(evidence, f"{label}-link", 0)
        recorded(evidence, f"{label}-run", 134 if fixture == "postzero" else 0)
        if not (mode_dir / f"composer-{fixture}").is_file():
            raise ValueError(f"{label}: linked executable missing")
        if mode == "sanitize":
            raw_ir = (generated / f"{fixture}.raw.o.ll").read_bytes()
            derived, definitions = prepare(raw_ir)
            actual_derived = (generated / f"{fixture}.asan.ll").read_bytes()
            receipt = json.loads((generated / f"{fixture}.asan.json").read_text())
            if actual_derived != derived or receipt != {
                "raw_sha256": sha(raw_ir), "derived_sha256": sha(derived),
                "definitions": len(definitions),
                "definition_line_sha256": definitions,
            }:
                raise ValueError(f"{label}: derived IR is not exact")
            for step in ("asan-prepare", "asan-compile", "asan-symbols"):
                recorded(evidence, f"{label}-{step}", 0)
            symbols = subprocess.run(
                ["nm", "-u", str(generated / f"{fixture}.asan.o")],
                capture_output=True, text=True, check=True).stdout
            if not REPORT.search(symbols):
                raise ValueError(f"{label}: generated object lacks ASan checks")
    print(f"G3-C4 terminal {mode} raw compiler evidence PASS: "
          f"native={len(NATIVE)} fixtures={len(FIXTURES)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: check_p2_g2_terminal_compiler_evidence.py "
                         "MODE_DIRECTORY EVIDENCE MODE")
    verify(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
