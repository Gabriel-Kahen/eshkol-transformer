#!/usr/bin/env python3
"""Fast structural checks for accepted G3-C4 Step 5 active calls."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered(text, fragments, label):
    position = -1
    for fragment in fragments:
        found = text.find(fragment, position + 1)
        require(found >= 0, f"{label} missing or misordered: {fragment}")
        position = found


def without_conditional_feature(text, macro):
    """Remove source regions compiled only when a later feature is enabled."""
    kept = []
    depth = 0
    for line in text.splitlines(keepends=True):
        directive = line.lstrip()
        opens = (directive.startswith("#if ") or
                 directive.startswith("#ifdef ") or
                 directive.startswith("#ifndef "))
        if depth:
            if opens:
                depth += 1
            elif directive.startswith("#endif"):
                depth -= 1
            continue
        if opens and macro in directive:
            depth = 1
            continue
        kept.append(line)
    require(depth == 0, f"unterminated conditional feature block: {macro}")
    return "".join(kept)


def check():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-g3c4-context-cache.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    step5_source = without_conditional_feature(
        without_conditional_feature(
            without_conditional_feature(source, "ET_G3C4_MANUAL_ROLE0_PRIVATE"),
            "ET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE"),
        "ET_G3C4_GENERATOR_PRIVATE")
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_active_call.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_ACTIVE_CALL_STEP5_CONTRACT.md").read_text()

    require("root accepted" in contract.lower(),
            "Step 5 contract does not record root acceptance")
    require(source.count("#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE") >= 2,
            "active-call feature blocks are missing")
    require("requires context and C4 pins" in source,
            "invalid active-call macro tuple is not rejected")

    for symbol in [
        "et_g3c4_private_call_acquire_v1",
        "et_g3c4_private_call_prepare_end_v1",
        "et_g3c4_private_call_finish_v1",
        "et_g3c4_private_call_abort_v1",
    ]:
        require(header.count(symbol) == 1, f"header boundary changed: {symbol}")
        require(source.count(symbol) == 1, f"source boundary changed: {symbol}")

    for field in [
        "et_g3c4_model_pins_internal pins", "int64_t call_kind",
        "int64_t budget", "uint32_t acquired_mask",
    ]:
        require(field in source, f"active context omits {field}")
    for forbidden in [
        "generator_seed", "generator_rng", "sampler", "role_step",
        "frame_begin", "frame_commit", "g3c4-model-registry", "G3-S",
    ]:
        require(forbidden not in step5_source,
                f"Step 5 reaches deferred surface: {forbidden}")

    ordered(source, [
        "context = et_g3c4_admit_idle_call(candidate)",
        "et_g3c4_valid_call_tuple(call_kind, budget)",
        "et_g3c4_cache_idle_preflight(context)",
        "et_g3c4_model_pins_begin_internal(",
        "context->acquired_mask = ET_G3C4_ACQUIRED_PINS",
        "context->call_kind = call_kind",
        "context->acquired_mask |= ET_G3C4_ACQUIRED_METADATA",
        "ET_G3C4_CONTEXT_BUSY(context) = ET_G3C4_CALL_ACTIVE",
        "context->acquired_mask |= ET_G3C4_ACQUIRED_BUSY",
        "context->owner->active = context",
        "context->acquired_mask |= ET_G3C4_ACQUIRED_OWNER",
    ], "active acquire")
    ordered(source, [
        "ET_G3C4_ACQUIRED_OWNER) != 0u",
        "context->owner->active = NULL",
        "ET_G3C4_ACQUIRED_BUSY) != 0u",
        "ET_G3C4_CONTEXT_BUSY(context) = ET_G3C4_CALL_IDLE",
        "ET_G3C4_ACQUIRED_METADATA) != 0u",
        "context->call_kind = 0",
        "ET_G3C4_ACQUIRED_PINS) != 0u",
        "et_g3c4_model_pins_end_internal(&context->pins)",
    ], "partial-publication rollback")
    ordered(source, [
        "static int64_t et_g3c4_active_call_drain",
        "et_g3c4_model_pins_end_internal(&context->pins)",
        "context->owner->active = NULL",
        "context->call_kind = 0",
        "context->budget = 0",
        "ET_G3C4_CONTEXT_BUSY(context) = ET_G3C4_CALL_IDLE",
        "context->acquired_mask = 0u",
    ], "active drain")
    require(step5_source.count("et_g3c4_cache_idle_preflight(context)") == 3,
            "cache idle proof must occur only in acquire, prepare-end, abort")
    require("et_a2_kv_cache_read_borrow_layer_v1" not in step5_source,
            "idle probe must not inspect cache content")

    for phrase in [
        "accepted_tuples", "rejected_tuples", "a2_allocation_failures",
        "pin_and_publication_failures", "nested_leases",
        "identity_lifecycle_and_corruption", "check_owner_unchanged",
        "registry_1024", "registry_8192", "record_bytes=%zu",
    ]:
        require(phrase in test, f"focused test omits {phrase}")

    expected = [
        "src/eshkol_transformer/g3c4_context_internal.h",
        "src/eshkol_transformer/g3c4_model_owner_internal.h",
        "src/eshkol_transformer/g3c4_model_owner.c",
        "src/eshkol_transformer/m3_call_pins.h",
        "src/eshkol_transformer/m3_call_f32_integration.c",
        "src/eshkol_transformer/m3t_f32_scoped.h",
        "src/eshkol_transformer/m3t_f32_integration.c",
        "native/f32_parameter_internal.h",
        "native/f32_tensor.c",
        "include/eshkol_transformer/f32_tensor.h",
        "include/eshkol_transformer/i64_tensor.h",
        "include/eshkol_transformer/kernel_abi.h",
        "include/eshkol_transformer/n3k_primitives_abi.h",
        "include/eshkol_transformer/a2_kv_cache.h",
        "native/a2_kv_cache.c",
        "native/kernel_abi.c",
        "native/n3k_primitives_provider.c",
        "tests/m3t/kernel_fail_allocator.c",
        "tests/g3c4/test_active_call.c",
        "tests/g3c4/test_context_cache.c",
        "tests/g3c4/test_pins.c",
        "scripts/check-g3c4-active-call.py",
        "scripts/check-g3c4-context-cache.py",
        "scripts/test-g3c4-active-call.sh",
        "scripts/test-g3c4-context-cache.sh",
        "scripts/test-g3c4-pins.sh",
        "docs/g3/G3_C4_ACTIVE_CALL_STEP5_CONTRACT.md",
        "docs/g3/G3_C4_ACTIVE_CALL_STEP5.md",
    ]
    closure = ROOT / "native/g3c4_active_call_source_closure.txt"
    actual = closure.read_text().splitlines()
    require(actual == expected, "Step 5 source closure changed")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 active-call static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 active-call static contract: PASS")
