#!/usr/bin/env python3
"""Structural gate for the private manual logits reservation leaf."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

def ordered(source: str, parts: list[str], label: str) -> None:
    at = -1
    for part in parts:
        at = source.find(part, at + 1)
        require(at >= 0, f"{label} missing or misordered: {part}")

def function(source: str, signature: str) -> str:
    at = source.index(signature)
    start = source.index("{", at)
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{": depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0: return source[at:i + 1]
    raise ValueError(f"unterminated function: {signature}")

def check() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/check-g3c4-output-envelope.py")],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_logits_reservation.c").read_text()
    runner = (ROOT / "scripts/test-g3c4-logits-reservation.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_LOGITS_RESERVATION_STEP23A_CONTRACT.md").read_text()
    require(source.count("et_g3c4_private_logits_reserve_v1") == 1 and
            header.count("et_g3c4_private_logits_reserve_v1") == 1,
            "exact source-private symbol missing")
    require("ET_G3C4_LOGITS_RESERVATION_PRIVATE requires an active generator call and typed tensor release" in source,
            "macro prerequisite missing")
    for phrase in ("ET_G3C4_LOGITS_KIND = 3", "ET_G3C4_LOGITS_MAGIC",
                   "et_f32_tensor *tensor", "sizeof(et_g3c4_logits_internal) == 48u",
                   "et_g3c4_pending_logits_lookup"):
        require(phrase in source, f"closed logits owner omits: {phrase}")
    reserve = function(source, "void *et_g3c4_private_logits_reserve_v1")
    ordered(reserve, ["et_g3c4_admit_active_call(candidate)",
                      "context->call_kind != 0", "context->call_kind != 1",
                      "et_g3c4_pending_logits_lookup", "et_g3c4_logits_allocate()",
                      "et_f32_tensor_create_v1(2u, shape", "free(logits)",
                      "logits->transport.state = 0u", "et_g3c4_enroll_logits(logits)"],
            "complete-before-enroll reservation")
    release = function(source, "int64_t et_g3c4_private_tensor_release_v1")
    ordered(release, ["et_g3c4_admit_logits(candidate, 1)",
                      "et_f32_tensor_destroy_v1(&logits->tensor",
                      "logits->parent_ctx = NULL",
                      "logits->transport.state = ET_G3C4_CONTEXT_DEAD"],
            "typed logits release")
    abort = function(source, "int64_t et_g3c4_private_call_abort_v1")
    ordered(abort, ["et_g3c4_pending_logits_lookup", "et_g3c4_cache_idle_preflight",
                    "et_f32_tensor_destroy_v1(&pending_logits->tensor",
                    "pending_logits->transport.state = ET_G3C4_CONTEXT_DEAD",
                    "et_g3c4_active_call_drain(context)"], "manual rollback")
    for name in ("et_g3c4_private_call_prepare_end_v1",
                 "et_g3c4_private_call_finish_v1"):
        body = function(source, f"int64_t {name}")
        require("et_g3c4_pending_logits_lookup" in body and
                "if (pending_logits != NULL)" in body,
                f"{name} accepts pending manual result")
    for phrase in ("manual_abort(owner, 0)", "manual_abort(owner, 1)",
                   "et_f32_tensor_test_fail_alloc_after_v1(cut)",
                   "ET_F32_TENSOR_CODE_ACTIVE_BORROW", "explicit_release(owner)",
                   "generate_rejects(owner)", "f32-cuts=4 owner-cuts=1"):
        require(phrase in test, f"native witness omits: {phrase}")
    for phrase in ("test-g3c4-output-envelope.sh", "compile_mode normal",
                   "compile_mode sanitize", "runtime-repeat.stdout",
                   "detect_leaks=1", "python-isolation.stdout"):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in ("starts from integration commit", "29-row boundary inventory",
                   "No Eshkol entry", "frame begin, role step, frame prepare"):
        require(phrase in contract, f"contract omits: {phrase}")
    dependency = (ROOT / "native/g3c4_output_envelope_source_closure.txt").read_text().splitlines()
    additions = ["native/g3c4_output_envelope_source_closure.txt",
                 "tests/g3c4/test_logits_reservation.c",
                 "scripts/check-g3c4-logits-reservation.py",
                 "scripts/test-g3c4-logits-reservation.sh",
                 "docs/g3/G3_C4_LOGITS_RESERVATION_STEP23A_CONTRACT.md"]
    expected = dependency + [p for p in additions if p not in dependency]
    actual = (ROOT / "native/g3c4_logits_reservation_source_closure.txt").read_text().splitlines()
    require(actual == expected and len(actual) == len(set(actual)),
            "logits reservation closure is not exact dependency union")
    for path in actual: require((ROOT / path).is_file(), f"missing closure file: {path}")

if __name__ == "__main__":
    try: check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 logits reservation static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private manual logits reservation contract: PASS")
