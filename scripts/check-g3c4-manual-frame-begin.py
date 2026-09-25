#!/usr/bin/env python3
"""Structural gate for the source-private manual frame-begin leaf."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

def body(source: str, signature: str) -> str:
    at = source.index(signature)
    start = source.index("{", at)
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{": depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0: return source[at:i + 1]
    raise ValueError(f"unterminated function: {signature}")

def ordered(source: str, parts: list[str], label: str) -> None:
    at = -1
    for part in parts:
        at = source.find(part, at + 1)
        require(at >= 0, f"{label} missing or misordered: {part}")

def check() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/check-g3c4-logits-reservation.py")],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_manual_frame_begin.c").read_text()
    runner = (ROOT / "scripts/test-g3c4-manual-frame-begin.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_MANUAL_FRAME_BEGIN_STEP23B_CONTRACT.md").read_text()
    require(source.count("int64_t et_g3c4_private_frame_begin_v1(") == 1 and
            header.count("int64_t et_g3c4_private_frame_begin_v1(") == 1,
            "exact source-private entry missing")
    begin = body(source, "int64_t et_g3c4_private_frame_begin_v1(")
    ordered(begin, ["et_g3c4_admit_active_call", "frame_kind != 1",
                    "et_g3c4_admit_input", "et_g3c4_pending_logits_lookup",
                    "et_g3c4_cache_idle_preflight",
                    "et_g3c4_prompt_prefill_borrow", "memcpy(ids, view->data",
                    "et_i64_tensor_borrow_end_v1", "et_g3c4_manual_frame_allocate",
                    "context->manual_frame = frame"], "frame admission and owned copy")
    for name in ("et_g3c4_private_call_prepare_end_v1",
                 "et_g3c4_private_call_finish_v1"):
        require("context->manual_frame != NULL" in body(source, f"int64_t {name}"),
                f"{name} accepts incomplete frame")
    abort = body(source, "int64_t et_g3c4_private_call_abort_v1")
    ordered(abort, ["et_f32_tensor_destroy_v1(&pending_logits->tensor",
                    "free(context->manual_frame)", "et_g3c4_active_call_drain"],
            "abort cleanup")
    for phrase in ("manual_prefill(owner, 1)", "manual_prefill(owner, 2)",
                   "borrowed_input(owner)", "manual_decode(owner)",
                   "et_g3c4_context_allocation_limit", "check_frame(context, 2"):
        require(phrase in test, f"native witness omits: {phrase}")
    for phrase in ("test-g3c4-logits-reservation.sh", "compile_mode normal",
                   "compile_mode sanitize", "runtime-repeat.stdout", "detect_leaks=1"):
        require(phrase in runner, f"runner omits: {phrase}")
    require("21 role calls" in contract and "No Eshkol entry" in contract,
            "scope contract missing")
    dependency = (ROOT / "native/g3c4_logits_reservation_source_closure.txt").read_text().splitlines()
    additions = ["native/g3c4_logits_reservation_source_closure.txt",
                 "tests/g3c4/test_manual_frame_begin.c",
                 "scripts/check-g3c4-manual-frame-begin.py",
                 "scripts/test-g3c4-manual-frame-begin.sh",
                 "docs/g3/G3_C4_MANUAL_FRAME_BEGIN_STEP23B_CONTRACT.md"]
    expected = dependency + [p for p in additions if p not in dependency]
    actual = (ROOT / "native/g3c4_manual_frame_begin_source_closure.txt").read_text().splitlines()
    require(actual == expected and len(actual) == len(set(actual)), "closure mismatch")
    for path in actual: require((ROOT / path).is_file(), f"missing file: {path}")

if __name__ == "__main__":
    try: check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 manual frame begin static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private manual frame begin contract: PASS")
