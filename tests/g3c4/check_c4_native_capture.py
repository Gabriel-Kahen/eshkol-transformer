#!/usr/bin/env python3
"""Source-only closure and high-risk control guard for C4 native capture."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/g3c4_p2g2_terminal_close_source_closure.txt"
CLOSURE = ROOT / "native/g3c4_c4_native_capture_source_closure.txt"
OBSERVER = ROOT / "tests/g3c4/c4_native_capture.c"
PARSER = ROOT / "tests/g3c4/c4_native_transcript.py"
RUNNER = ROOT / "scripts/test-g3c4-c4-native-capture.sh"
BASE_SHA = "43f25cca40be7f18b2d5ff15f2806aa4b2d1e59a30ad0a905c15c62b033dc576"
ADDITIONS = (
    "tests/g3c4/c4_math_oracle.py",
    "tests/g3c4/c4_math_source_pins.sha256",
    "tests/g3c4/c4_math_closure_members.sha256",
    "tests/g3c4/c4_math_roadmap_delta.hex",
    "tests/g3c4/c4_native_capture.c",
    "tests/g3c4/c4_native_transcript.py",
    "tests/g3c4/test_c4_native_transcript.py",
    "tests/g3c4/test_c4_native_capture_source.py",
    "tests/g3c4/check_c4_native_capture.py",
    "scripts/test-g3c4-c4-native-capture.sh",
    "native/g3c4_c4_native_capture_source_closure.txt",
    "tests/q0/requirements-oracle.lock",
)


def check(observer: str, parser: str, runner: str,
          base: list[str], closure: list[str]) -> None:
    assert len(base) == 198 and closure == base + list(ADDITIONS)
    assert len(set(closure)) == len(closure)
    assert all((ROOT / member).is_file() for member in closure)
    assert "#include \"../../src/eshkol_transformer/m3_call_f32_integration.c\"" in observer
    assert "#include \"../../src/eshkol_transformer/g3c4_model_owner.c\"" in observer
    assert "et_g3c4_private_model_owner_create_seeded_v1(seed)" in observer
    assert "et_g3c4_private_model_owner_word_v1(owner, index + 4)" in observer
    assert "offset == 1192u" in observer
    assert "state[2] == 292" in observer
    assert "et_f32_parameter_value_borrow_begin_v1" in observer
    assert "check_view(view, \"f32\", rank, parameter_shapes[index]" in observer
    assert "mask[position] == (uint8_t)(position < (size_t)prefix)" in observer
    assert "(head * 4u + position) * 2u + dimension" in observer
    assert "(head * (size_t)prefix + position) * 2u + dimension" in observer
    assert "check_view(keys, \"f32\", 4u, kv_shape, 16u * sizeof(float))" in observer
    assert "REQUIRE(*(const int64_t *)lengths->data == prefix)" in observer
    assert "REQUIRE(target_abort_count == 1u && target_abort_result == 0)" in observer
    assert "REQUIRE(outside_abort_count == before)" in observer
    assert "et_g3c4_private_prefill1_v1" in observer
    assert "et_g3c4_private_prefill2_v1" in observer
    assert "et_g3c4_private_prefill3_v1" in observer
    assert "et_g3c4_private_full_prefix_forward_v1" in observer
    assert "gamma =" not in observer and "fork(" not in observer
    assert "2f3adec7636cebc39b735d7b6c461e2856e390ec1e9f12fccdc0c3074184f041" in parser
    for control in ("noncanonical fixture", "case identity/order",
                    "intended T4 abort", "invalid/nonfinite f32 bit word",
                    "native child timeout", "native executable drift",
                    "returncode", "stderr_bytes", "open(\"xb\")"):
        assert control in parser, control
    for control in ("verify_toolchain", "resolve_provenance_compilers",
                    '[[ "${Q0_PYTHON:-}" == /*', "q0_identity",
                    "3.14.6", "2.13.0+cpu", "source.pre.sha256",
                    "source.post.sha256", "source-git.pre", "source-git.post",
                    "normal/repeat.stdout", "normal/repeat.json",
                    "build_mode sanitize", "compiled-artifacts.sha256",
                    "libc4_native_capture.a", "archive-defined.symbols",
                    "archive-undefined.symbols", "objects/*.o",
                    "timeout --signal=TERM --kill-after=5s 600s",
                    "timeout --signal=TERM --kill-after=5s 120s",
                    "--wrap=et_a2_kv_cache_transaction_abort_v1"):
        assert control in runner, control
    assert "LOCK_FILE=" not in runner and "docker " not in runner.lower()


def main() -> None:
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise AssertionError("accepted terminal closure drift")
    check(OBSERVER.read_text(), PARSER.read_text(), RUNNER.read_text(),
          BASE.read_text().splitlines(), CLOSURE.read_text().splitlines())
    print("C4 native capture source/closure PASS")


if __name__ == "__main__":
    main()
