#!/usr/bin/env python3
"""Structural gate for the translation-unit-private manual role-0 precursor."""
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

def ordered(source: str, parts: list[str]) -> None:
    at = -1
    for part in parts:
        at = source.find(part, at + 1)
        require(at >= 0, f"missing or misordered: {part}")

def check() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/check-g3c4-manual-frame-begin.py")],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    test = (ROOT / "tests/g3c4/test_manual_role0.c").read_text()
    runner = (ROOT / "scripts/test-g3c4-manual-role0.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_MANUAL_ROLE0_STEP23C_CONTRACT.md").read_text()
    require("et_g3c4_private_role_step_v1" not in source,
            "partial accepted role_step boundary exposed")
    require("static inline __attribute__((unused)) int64_t et_g3c4_manual_role0_run" in source,
            "translation-unit-private seam missing")
    role = body(source, "static inline __attribute__((unused)) int64_t et_g3c4_manual_role0_run")
    ordered(role, ["et_g3c4_admit_active_call", "frame->next_ordinal != 0",
                   "et_g3c4_pending_logits_lookup", "et_g3c4_cache_idle_preflight",
                   "et_g3c4_token_runtime_discover", "et_g3c4_token_dispatch",
                   "memcpy(frame->et, et_candidate", "frame->next_ordinal = 1"])
    for phrase in ("et_g3n_kernel_provider_v1()", "et_n3k_kernel_provider_v1()",
                   "g3n.embedding.forward", "n3k.embedding.forward",
                   "frame->input_ids", "context->pins.views[10]"):
        require(phrase in role, f"role0 route missing: {phrase}")
    for phrase in ("prefill_role0(owner, 1)", "prefill_role0(owner, 2)",
                   "decode_role0(owner)", "dispatch_cut(owner)",
                   "malformed_role0(owner)",
                   "frame->et", "ET_KERNEL_CODE_PROVIDER_REJECTED"):
        require(phrase in test, f"witness missing: {phrase}")
    for phrase in ("test-g3c4-manual-frame-begin.sh", "compile_mode normal",
                   "compile_mode sanitize", "runtime-repeat.stdout",
                   "test ! -s \"$evidence/added-defined.txt\""):
        require(phrase in runner, f"runner missing: {phrase}")
    require("No Eshkol entry" in contract and "21-role" in contract,
            "scope contract missing")
    dependency = (ROOT / "native/g3c4_manual_frame_begin_source_closure.txt").read_text().splitlines()
    additions = ["native/g3c4_manual_frame_begin_source_closure.txt",
                 "tests/g3c4/test_manual_role0.c",
                 "scripts/check-g3c4-manual-role0.py",
                 "scripts/test-g3c4-manual-role0.sh",
                 "docs/g3/G3_C4_MANUAL_ROLE0_STEP23C_CONTRACT.md"]
    expected = dependency + [p for p in additions if p not in dependency]
    actual = (ROOT / "native/g3c4_manual_role0_source_closure.txt").read_text().splitlines()
    require(actual == expected and len(actual) == len(set(actual)), "closure mismatch")
    for path in actual: require((ROOT / path).is_file(), f"missing file: {path}")

if __name__ == "__main__":
    try: check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 manual role0 static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 internal manual role0 contract: PASS")
