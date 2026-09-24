#!/usr/bin/env python3
"""Fast structural checks for the accepted G3-C4 Step 4a context/cache leaf."""

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
        [sys.executable, str(ROOT / "scripts/check-g3c4-model-authority.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_context_cache.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_CONTEXT_CACHE_STEP4A_CONTRACT.md").read_text()

    require("root accepted" in contract.lower(),
            "Step 4a contract does not record root acceptance")
    require(source.count("#ifdef ET_G3C4_CONTEXT_PRIVATE") == 1,
            "context implementation feature block changed")
    block = source[source.index("#ifdef ET_G3C4_CONTEXT_PRIVATE"):]
    step4_block = without_conditional_feature(
        block, "ET_G3C4_ACTIVE_CALL_PRIVATE")
    step4_block = without_conditional_feature(
        step4_block, "ET_G3C4_GENERATOR_PRIVATE")
    require(source.index("#ifdef ET_G3C4_CONTEXT_PRIVATE") >
            source.index("et_g3c4_private_model_owner_abort_v1"),
            "context block no longer follows accepted owner implementation")
    for symbol in [
        "et_g3c4_private_context_create_v1",
        "et_g3c4_private_context_close_v1",
    ]:
        require(header.count(symbol) == 1, f"header boundary changed: {symbol}")
        require(block.count(symbol) == 1, f"source boundary changed: {symbol}")

    for field in [
        "registry_next", "magic", "kind", "state", "busy", "owner", "cache",
    ]:
        require(field in block, f"context record omits {field}")
    for forbidden in [
        "et_g3c4_model_pins", "generator_seed", "generator_rng",
        "policy", "sampler", "forward", "owner->active =",
    ]:
        require(forbidden not in step4_block,
                f"Step 4a reaches deferred surface: {forbidden}")

    ordered(block, [
        "owner = et_g3c4_admit_owner(candidate, 0)",
        "owner->state != ET_G3C4_OWNER_SEALED",
        "owner->active != NULL",
        "context = et_g3c4_context_allocate()",
        "et_a2_kv_cache_create_v1(",
        "1u, 1u, 2u, 4u, 2u, &cache, &error",
        "context->owner = owner",
        "context->cache = cache",
        "context->registry_next = et_g3c4_context_registry",
        "et_g3c4_context_registry = context",
    ], "context construction")
    ordered(block, [
        "context = et_g3c4_admit_context(candidate)",
        "ET_G3C4_CONTEXT_STATE(context) == ET_G3C4_CONTEXT_DEAD",
        "owner = et_g3c4_admit_owner(context->owner, 0)",
        "owner->state != ET_G3C4_OWNER_SEALED",
        "owner->active != NULL",
        "et_a2_kv_cache_destroy_v1(&context->cache, &error)",
        "context->owner = NULL",
        "ET_G3C4_CONTEXT_STATE(context) = ET_G3C4_CONTEXT_DEAD",
    ], "context close")
    require("free(context);\n    et_g3c4_error_restore_internal(first);" in block,
            "constructor failure does not preserve first error")

    for phrase in [
        "construction_failures", "cache_geometry_and_multiplicity",
        "transaction_busy", "view_busy",
        "borrow_busy", "wrong_lifecycle", "registry_1024",
        "registry_8192", "owner_unchanged",
    ]:
        require(phrase in test, f"focused test omits {phrase}")

    expected = [
        "src/eshkol_transformer/g3c4_context_internal.h",
        "src/eshkol_transformer/g3c4_model_owner_internal.h",
        "src/eshkol_transformer/g3c4_model_owner.c",
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
        "tests/g3c4/test_model_owner.c",
        "tests/g3c4/test_context_cache.c",
        "scripts/check-g3c4-context-cache.py",
        "scripts/test-g3c4-native-owner.sh",
        "scripts/test-g3c4-context-cache.sh",
        "docs/g3/G3_C4_CONTEXT_CACHE_STEP4A_CONTRACT.md",
        "docs/g3/G3_C4_CONTEXT_CACHE_STEP4A.md",
    ]
    actual = (ROOT / "native/g3c4_context_source_closure.txt").read_text().splitlines()
    require(actual == expected, "Step 4a source closure changed")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 context/cache static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 context/cache static contract: PASS")
