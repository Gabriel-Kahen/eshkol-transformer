#!/usr/bin/env python3
"""Structural gate for the internal manual ordinal 1..9 precursor."""
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
    subprocess.run([sys.executable, str(ROOT / "scripts/check-g3c4-manual-role0.py")],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    test = (ROOT / "tests/g3c4/test_manual_pre_a2.c").read_text()
    runner = (ROOT / "scripts/test-g3c4-manual-pre-a2.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_MANUAL_PRE_A2_STEP23D_CONTRACT.md").read_text()
    require("et_g3c4_private_role_step_v1" not in source,
            "partial accepted role_step boundary exposed")
    signature = "static inline __attribute__((unused)) int64_t et_g3c4_manual_pre_a2_run"
    role = body(source, signature)
    ordered(role, ["ordinal < 1 || ordinal > 9", "et_g3c4_admit_active_call",
                   "frame->next_ordinal != ordinal", "et_g3c4_pending_logits_lookup",
                   "et_g3c4_cache_idle_preflight", "switch (ordinal)",
                   "et_g3c4_token_runtime_discover", "et_g3c4_token_dispatch",
                   "memcpy(destination, step_candidate", "frame->next_ordinal = ordinal + 1"])
    for phrase in ("case 1:", "case 2:", "case 3:", "case 4: case 5: case 6:",
                   "g3n.embedding.forward", "n3k.embedding.forward",
                   "g3n.residual.forward", "n3k.residual.forward",
                   "g3n.layer-norm.forward", "layer-norm.forward",
                   "g3n.linear.forward-no-bias", "n3k.linear.forward-no-bias",
                   "g3n.heads.split.forward", "n3k.heads.split.forward",
                   "2u * 4u * sizeof(float)", "UINT32_C(0x3727c5ac)"):
        require(phrase in role, f"fixed route missing: {phrase}")
    for phrase in ("pre_a2_case(owner, 1, 1, 0)",
                   "pre_a2_case(owner, 1, 2, 0)",
                   "pre_a2_case(owner, 2, 1, 0)",
                   "pre_a2_case(owner, 2, 1, ordinal)",
                   "reference_outputs[ordinal]",
                   "ET_KERNEL_CODE_PROVIDER_REJECTED", "cuts=27 ordinals=1-9"):
        require(phrase in test, f"native witness missing: {phrase}")
    for phrase in ("test-g3c4-manual-role0.sh", "compile_mode normal",
                   "compile_mode sanitize", "runtime-repeat.stdout",
                   "test ! -s \"$evidence/added-defined.txt\""):
        require(phrase in runner, f"runner missing: {phrase}")
    require("No Eshkol entry" in contract and "Attention ordinal 10" in contract
            and "bitwise" in contract,
            "scope contract missing")
    dependency = (ROOT / "native/g3c4_manual_role0_source_closure.txt").read_text().splitlines()
    additions = ["native/g3c4_manual_role0_source_closure.txt",
                 "tests/g3c4/test_manual_pre_a2.c",
                 "scripts/check-g3c4-manual-pre-a2.py",
                 "scripts/test-g3c4-manual-pre-a2.sh",
                 "docs/g3/G3_C4_MANUAL_PRE_A2_STEP23D_CONTRACT.md"]
    expected = dependency + [p for p in additions if p not in dependency]
    actual = (ROOT / "native/g3c4_manual_pre_a2_source_closure.txt").read_text().splitlines()
    require(actual == expected and len(actual) == len(set(actual)), "closure mismatch")
    for path in actual: require((ROOT / path).is_file(), f"missing file: {path}")

if __name__ == "__main__":
    try: check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 manual pre-A2 static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 internal manual pre-A2 contract: PASS")
