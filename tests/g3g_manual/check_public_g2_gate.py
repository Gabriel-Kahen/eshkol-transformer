"""Development-only source guard for the installed G2 sanitizer route."""

from __future__ import annotations

from pathlib import Path
import re


def check(builder: str, runner: str, caller: str) -> None:
    required_builder = (
        'case "${G3G_MANUAL_SANITIZE:-0}" in',
        '"${package_policy}" == g3g-manual-public-aggregate',
        'g3g_manual_test_sanitize=1',
        'package_test_sanitize_flags=(-O1 -g -fsanitize=address,undefined',
        '"${e1b_cc}" "${package_test_sanitize_flags[@]}" -c -x ir',
        '"${package_test_sanitize_flags[@]}" \\\n  "${package_bridge_flags[@]}"',
        'package_native_cflags+=("${package_test_sanitize_flags[@]}")',
        '"${e1b_cxx}" -r "${package_test_reloc_flags[@]}"',
        'e1b_compare_exact_undefined_allowlist',
        '"${e1b_tmp}/noninstrumentation-undefined.txt"',
        '"${e1b_tmp}/sanitizer-undefined.txt"',
    )
    required_runner = (
        'G3G_MANUAL_SANITIZE=0 /usr/bin/bash',
        'G3G_MANUAL_SANITIZE=1 /usr/bin/bash',
        'G2 evidence directory already exists',
        'scripts/test-g3g-manual-public.sh',
        'tests/g3g_manual/public_g2_runtime.esk',
        'check_public_closure.py',
        '-fsanitize=address,undefined -fno-omit-frame-pointer',
        'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1',
        'UBSAN_OPTIONS=halt_on_error=1',
        'G3G-MANUAL-PUBLIC-G2-PASS',
        'cmp "${source_pin}" "${source_after}"',
        'sanitizer-undefined.txt',
        'noninstrumentation-undefined.txt',
    )
    for required in required_builder:
        if required not in builder:
            raise ValueError(f"installed SAN build contract missing: {required}")
    for required in required_runner:
        if required not in runner:
            raise ValueError(f"installed G2 run contract missing: {required}")
    if builder.count('case "${G3G_MANUAL_SANITIZE:-0}" in') != 1 or \
       builder.count('g3g_manual_test_sanitize=1') != 1:
        raise ValueError("installed SAN selector duplicated")
    selector = builder.split('case "${G3G_MANUAL_SANITIZE:-0}" in', 1)[1].split('esac', 1)[0]
    if selector.count('"${package_policy}" == g3g-manual-public-aggregate') != 1:
        raise ValueError("installed SAN selector is not exact manual tuple")
    if "G3G_MANUAL_SANITIZE" in caller or re.search(r"\(load\b|g3c4-.*-private", caller):
        raise ValueError("installed caller imports test or private authority")
    if caller.count("(require transformer.") != 6:
        raise ValueError("installed caller requires changed")
    if caller.count("G3G-MANUAL-PUBLIC-G2-PASS") != 1:
        raise ValueError("installed caller terminal marker changed")


def main(root: Path) -> None:
    check((root / "scripts/build-e1b-consumer.sh").read_text(),
          (root / "scripts/test-g3g-manual-public-g2.sh").read_text(),
          (root / "tests/g3g_manual/public_g2_runtime.esk").read_text())


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
    print("installed public G2 source gate: PASS")
