#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 9A sampler transport leaf."""

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
        position = text.find(fragment, position + 1)
        require(position >= 0, f"{label} missing or misordered: {fragment}")


def check():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-g3c4-full-prefix-forward.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_sampler_transport.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_SAMPLER_TRANSPORT_STEP9A_CONTRACT.md").read_text()
    runner = (ROOT / "scripts/test-g3c4-sampler-transport.sh").read_text()

    require("ET_G3C4_SAMPLER_TRANSPORT_PRIVATE requires full-prefix forward"
            in source, "sampler macro is not nested under full-prefix forward")
    symbol = "et_g3c4_private_sample_last_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private sampler boundary changed")
    start = source.index(f"int64_t {symbol}")
    end = source.index("\n#endif", start)
    body = source[start:end]

    for fragment in [
        "full_logits + 3u * 256u",
        "context->call_kind != 2 || context->budget != 1",
        "memcpy(numeric_rng, context->generator_rng_words",
        '"g3s.greedy.forward"', '"g3s.categorical.forward"',
        '"g3s.greedy"', '"g3s.categorical"',
        "input_count = 2u", "input_count = 5u",
        "et_g3c4_sampler_runtime", "token_candidate < 0",
    ]:
        require(fragment in body, f"sampler transport omits {fragment}")
    ordered(body, [
        "memcpy(last_logits",
        "memcpy(numeric_rng",
        "et_kernel_runtime_dispatch(",
        "token_candidate < 0",
        "memcpy(token_output",
        "memcpy(successor_output",
    ], "speculative publication")
    require(body.count("context->generator_rng_words") == 1,
            "sampler transport writes or rereads generator RNG unexpectedly")
    for forbidden in ["transaction_commit", "transaction_begin",
                      "full_prefix_forward_v1(", "generator_rng_words[2] =",
                      "generator_rng_words[3] ="]:
        require(forbidden not in body,
                f"sampler transport reaches held ownership surface: {forbidden}")

    for phrase in ["expected_categorical", "force_dispatch_failure",
                   "inputs[0].data != expected_full_logits",
                   "memcmp(context->generator_rng_words",
                   "check_cache_empty", "categorical_cases",
                   "call_kind_rejection"]:
        require(phrase in test, f"focused test omits {phrase}")
    require("--wrap=et_kernel_runtime_dispatch" in runner,
            "runner does not interpose sampler dispatch")
    require("cannot establish that prefill was committed" in contract,
            "contract does not state the frame-ordering limit")

    prior = (ROOT / "native/g3c4_full_prefix_forward_source_closure.txt"
             ).read_text().splitlines()
    expected = prior + [
        "native/g3c4_full_prefix_forward_source_closure.txt",
        "tests/g3c4/test_sampler_transport.c",
        "scripts/check-g3c4-sampler-transport.py",
        "scripts/test-g3c4-sampler-transport.sh",
        "docs/g3/G3_C4_SAMPLER_TRANSPORT_STEP9A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_sampler_transport_source_closure.txt"
              ).read_text().splitlines()
    require(actual == expected,
            "sampler closure is not the exact Step 8A successor")
    require(len(actual) == len(set(actual)), "closure has duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 sampler static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private sampler transport static contract: PASS")
