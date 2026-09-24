#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 6A native generator slice."""

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


def check():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-g3c4-active-call.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_generator_native.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_CALL_ENTRY_STEP6_CONTRACT.md").read_text()

    require("Step 6A" in contract and "one native transport registry" in contract,
            "accepted Step 6A contract is missing")
    require("ET_G3C4_GENERATOR_PRIVATE requires context, C4 pins, and active call"
            in source, "invalid generator macro tuple is not rejected")
    require(source.count(
        "static et_g3c4_transport_header_internal *et_g3c4_transport_registry")
        == 1, "native transport registry is not singular")
    require("static et_g3c4_rng_internal *et_g3c4_rng_registry" not in source,
            "separate RNG registry is forbidden")
    require("static et_g3c4_context_internal *et_g3c4_generator_registry"
            not in source, "separate generator registry is forbidden")

    for symbol in [
        "et_g3c4_private_generator_seed_v1",
        "et_g3c4_private_generator_rng_v1",
        "et_g3c4_private_generator_close_v1",
        "et_g3c4_private_rng_seed_v1",
        "et_g3c4_private_rng_clone_v1",
        "et_g3c4_private_rng_word_v1",
        "et_g3c4_private_rng_release_v1",
    ]:
        require(header.count(symbol) == 1, f"header boundary changed: {symbol}")
        require(source.count(symbol) == 1, f"source boundary changed: {symbol}")

    for fragment in [
        "et_g3c4_transport_header_internal transport",
        "uint32_t generator_kind",
        "uint32_t generator_ready",
        "int64_t generator_policy[6]",
        "int64_t generator_rng_words[4]",
        "int64_t words[4]",
        "sizeof(et_g3c4_transport_header_internal) == 32u",
        "sizeof(et_g3c4_context_internal) == 1520u",
        "sizeof(et_g3c4_rng_internal) == 64u",
    ]:
        require(fragment in source, f"native layout omits {fragment}")

    ordered(source, [
        "context = et_g3c4_context_allocate()",
        "et_a2_kv_cache_create_v1(",
        "context->generator_kind = 1u",
        "context->generator_ready = 1u",
        "memcpy(context->generator_rng_words",
        "et_g3c4_enroll_context(context)",
    ], "generator construction")
    ordered(source, [
        "context = et_g3c4_admit_context(candidate)",
        "context->generator_kind != 1u",
        "et_a2_kv_cache_destroy_v1(&context->cache, &error)",
        "memset(context->generator_policy, 0",
        "memset(context->generator_rng_words, 0",
        "context->generator_ready = 0u",
        "context->owner = NULL",
        "ET_G3C4_CONTEXT_STATE(context) = ET_G3C4_CONTEXT_DEAD",
    ], "generator close")
    ordered(source, [
        "memset(rng->words, 0, sizeof(rng->words))",
        "rng->transport.state = ET_G3C4_CONTEXT_DEAD",
    ], "RNG release")

    for forbidden in ["philox", "counter_advance", "sample_token",
                      "tokenizer", "frame_commit", "result_publish"]:
        require(forbidden not in source.lower(),
                f"native ownership slice reaches held G3-S surface: {forbidden}")

    for phrase in [
        "layout_and_macro_surface", "rng_lifetime",
        "policy_and_construction", "type_boundaries",
        "close_retry_and_scrub", "contexts_1024=1024",
        "contexts_8192=8192", "rng_1024=1024", "rng_8192=8192",
    ]:
        require(phrase in test, f"focused native test omits {phrase}")

    expected = (ROOT / "native/g3c4_active_call_source_closure.txt"
                ).read_text().splitlines() + [
        "native/g3c4_active_call_source_closure.txt",
        "native/g3c4_context_source_closure.txt",
        "native/g3c4_model_source_closure.txt",
        "native/g3c4_generator_owner_deps.txt",
        "tests/g3c4/test_generator_native.c",
        "tests/g3c4/test_model_owner.c",
        "tests/m3cg/test_pins.c",
        "scripts/check-g3c4-generator.py",
        "scripts/check-g3c4-model-authority.py",
        "scripts/check-g3c4-i2-prepared-route.py",
        "scripts/test-g3c4-generator.sh",
        "scripts/test-g3c4-native-owner.sh",
        "scripts/common.sh",
        "docs/g3/G3_C4_CALL_ENTRY_STEP6_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_generator_source_closure.txt"
              ).read_text().splitlines()
    require(actual == expected, "native generator source closure changed")
    require(len(actual) == len(set(actual)), "source closure has duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")
    depfile = (ROOT / "native/g3c4_generator_owner_deps.txt").read_text()
    for path in [
        "src/eshkol_transformer/g3c4_model_owner.c",
        "src/eshkol_transformer/g3c4_model_owner_internal.h",
        "native/f32_parameter_internal.h",
        "src/eshkol_transformer/g3c4_context_internal.h",
        "include/eshkol_transformer/a2_kv_cache.h",
        "src/eshkol_transformer/m3_call_pins.h",
    ]:
        require(path in depfile or path.replace(
                    "native/f32_parameter_internal.h",
                    "src/eshkol_transformer/../../native/f32_parameter_internal.h")
                in depfile,
                f"owner depfile omits {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 native generator static check failed: {error}",
              file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 native generator static contract: PASS")
