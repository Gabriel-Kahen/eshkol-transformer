#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 7A provider-route leaf."""

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
        [sys.executable, str(ROOT / "scripts/check-g3c4-generator.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    test = (ROOT / "tests/g3c4/test_provider_routes.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_PROVIDER_ROUTES_STEP7A_CONTRACT.md").read_text()

    require("ET_G3C4_PROVIDER_ROUTES_PRIVATE requires the private generator"
            in source, "route macro is not nested under the generator")
    require(source.count("static et_kernel_runtime *et_g3c4_forward_runtime") == 1,
            "forward runtime authority is not singular")
    require(source.count("static et_kernel_runtime *et_g3c4_sampler_runtime") == 1,
            "sampler runtime authority is not singular")
    require(source.count("et_g3c4_kernel_provider_v1()") == 1,
            "fixed G3-C4 provider route changed")
    require(source.count("et_g3s_kernel_provider_v1()") == 1,
            "fixed G3-S provider route changed")

    rows = [
        ('"g3c4.embedding-forward", "g3c4.embedding.forward", 4u',
         '{1u, 4u, 4u, 4u}'),
        ('"g3c4.embedding-forward", "g3c4.embedding.forward", 4u',
         '{1u, 4u, 256u, 4u}'),
        ('"g3c4.linear", "g3c4.linear.forward-no-bias", 4u',
         '{1u, 4u, 4u, 4u}'),
        ('"g3c4.linear", "g3c4.linear.forward-no-bias", 4u',
         '{1u, 4u, 4u, 8u}'),
        ('"g3c4.linear", "g3c4.linear.forward-no-bias", 4u',
         '{1u, 4u, 8u, 4u}'),
        ('"g3c4.linear", "g3c4.linear.forward-no-bias", 4u',
         '{1u, 4u, 4u, 256u}'),
        ('"g3c4.layer-norm", "g3c4.layer-norm.forward", 3u',
         '{1u, 4u, 4u}'),
        ('"g3c4.gelu", "g3c4.gelu.forward", 3u', '{1u, 4u, 8u}'),
        ('"g3c4.residual", "g3c4.residual.forward", 3u', '{1u, 4u, 4u}'),
        ('"g3c4.head-layout", "g3c4.heads.split.forward", 4u',
         '{1u, 4u, 2u, 2u}'),
        ('"g3c4.head-layout", "g3c4.heads.merge.forward", 4u',
         '{1u, 4u, 2u, 2u}'),
        ('"g3c4.causal-attention", "g3c4.causal-attention.forward", 6u',
         '{1u, 2u, 2u, 4u, 4u, 2u}'),
    ]
    cursor = source.index("et_g3c4_forward_routes[]")
    for prefix, shape in rows:
        cursor = source.find(prefix, cursor)
        require(cursor >= 0, f"missing ordered route {prefix} {shape}")
        end = source.find("},", cursor)
        require(shape in source[cursor:end + 2], f"wrong shape for {prefix}")
        cursor = end + 2
    for row in [
        '{"g3s.greedy", "g3s.greedy.forward", 2u, {1u, 256u}}',
        '{"g3s.categorical", "g3s.categorical.forward", 2u, {1u, 256u}}',
    ]:
        require(row in source, f"missing sampler route: {row}")

    ordered(source, [
        "et_g3c4_stage_provider_routes(",
        "context = et_g3c4_context_allocate()",
        "et_a2_kv_cache_create_v1(",
        "et_g3c4_forward_runtime = forward_runtime",
        "et_g3c4_sampler_runtime = sampler_runtime",
        "et_g3c4_enroll_context(context)",
    ], "atomic construction")
    require(source.count("et_g3c4_abort_staged_routes(") >= 3,
            "staged route cleanup is incomplete")
    for forbidden in ["runtime_invoke", "sample_token", "frame_commit",
                      "result_publish", "tokenizer"]:
        require(forbidden not in source.lower(),
                f"provider-route leaf reaches held surface: {forbidden}")
    for phrase in ["exact_route_table", "staged_rejection_is_atomic",
                   "construction_atomicity", "accepted_routes_are_reused"]:
        require(phrase in test, f"focused test omits {phrase}")
    require("does not invoke either runtime" in contract,
            "contract does not state the numerical limit")

    prior = (ROOT / "native/g3c4_generator_source_closure.txt"
             ).read_text().splitlines()
    expected = prior + [
        "native/g3c4_generator_source_closure.txt",
        "include/eshkol_transformer/g3c4_primitives_abi.h",
        "include/eshkol_transformer/g3s_sampling_abi.h",
        "native/g3c4_primitives_provider.c",
        "native/g3s_sampling_provider.c",
        "tests/g3c4/test_provider_routes.c",
        "tests/q0/test_python_isolation.py",
        "scripts/check-g3c4-provider-routes.py",
        "scripts/test-g3c4-provider-routes.sh",
        "docs/g3/G3_C4_PROVIDER_ROUTES_STEP7A_CONTRACT.md",
        "docs/g3/G3_T_PRIVATE_CONTRACT.md",
        "docs/g3/G3_C4_N_ABI_PROPOSAL.md",
        "docs/g3/G3_S_ABI_PROPOSAL.md",
        "docs/ROADMAP.md",
    ]
    closure = (ROOT / "native/g3c4_provider_routes_source_closure.txt"
               ).read_text().splitlines()
    require(closure == expected,
            "provider-route closure is not the exact generator successor")
    require(len(closure) == len(set(closure)), "closure has duplicates")
    for path in closure:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 provider-route static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private provider-route static contract: PASS")
