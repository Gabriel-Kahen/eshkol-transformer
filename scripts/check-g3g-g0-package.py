#!/usr/bin/env python3
"""Closed public G0/G1 facade and package boundary."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "native"
facade = (ROOT / "lib/transformer/generation.esk").read_text()
root = (manifest / "g3g_g0_package_root.esk").read_text()
wrapper = (manifest / "g3g_g0_public_extension.esk").read_text()
bridge = (manifest / "g3g_g0_package_bridge.c").read_text()
exports = (manifest / "g3g_g0_package_public_exports.txt").read_text().splitlines()
strings = (manifest / "g3g_g0_package_public_strings.txt").read_text().splitlines()
renames = (manifest / "g3g_g0_package_private_renames.txt").read_text().splitlines()
source = (manifest / "g3g_g0_package_source_closure.txt").read_text().splitlines()
facades = (manifest / "g3g_g0_package_facades.txt").read_text().splitlines()
assert len(exports) == 100 and exports == sorted(set(exports))
assert exports == (manifest / "g3g_package_public_exports.txt").read_text().splitlines()
assert len(strings) == 106 and strings == sorted(set(strings))
assert strings == (manifest / "g3g_package_public_strings.txt").read_text().splitlines()
stems = (
    "generator_create", "generator_generate", "generation_input_create",
    "generation_token_input_create", "generation_output_ids",
    "generation_output_lengths", "generation_output_text",
    "generation_output_rng", "generation_output_cache_lengths",
    "generation_tensor_release", "generation_output_release",
    "generation_rng_release", "generator_close",
)
added_public = {f"et_e1b_public_g3_{stem}_v1" for stem in stems}
assert set(exports) == set((manifest / "m3_package_public_exports.txt").read_text().splitlines()) | added_public
added_private = {f"et_e1b_private_g3_{stem}_cabi_v1" for stem in stems}
assert {line.split()[1] for line in renames if line.startswith("g3g-")} == added_private
assert set(strings) == set((manifest / "m3_package_public_strings.txt").read_text().splitlines()) | added_public
assert len(facades) == 8 and facades == sorted(set(facades))
assert facades == sorted((manifest / "m3_package_facades.txt").read_text().splitlines()
                         + ["transformer/generation.esk"])
assert source[0] == "native/g3g_g0_package_root.esk"
assert source[-3:] == [
    "native/g3t_zero_budget_extension.esk",
    "native/g3t_p2_zero_budget_extension.esk",
    "native/g3g_g0_public_extension.esk",
]
native_source = (manifest / "g3g_g0_package_native_source_closure.txt").read_text().splitlines()
assert "native/g3g_g0_package_bridge.c" in native_source
assert "native/g3g_package_bridge.c" not in native_source
assert "include/eshkol_transformer/n3k_primitives_abi.h" in native_source
assert len(source) == len(set(source))
expected = (
    "generator-create", "generator-generate!", "generation-input-create",
    "generation-token-input-create", "generation-output-ids",
    "generation-output-lengths", "generation-output-text",
    "generation-output-rng", "generation-output-cache-lengths",
    "generation-tensor-release!", "generation-output-release!",
    "generation-rng-release!", "generator-close!",
)
provide = facade.split("(provide ", 1)[1].split(")", 1)[0].split()
assert provide == list(expected)
assert "generator-prefill!" not in facade and "generator-decode-step!" not in facade
assert '(load "g3m_seeded_p1_g1_extension.esk")' in root
for included in ("g3t_zero_budget_extension", "g3t_p2_zero_budget_extension"):
    assert included in root
for excluded in (
                 "g3m_manual_decode_extension", "g3m_prefill_p1_extension",
                 "g3m_prefill_p2_extension"):
    assert excluded not in root
assert len([line for line in renames if "et_e1b_private_g3_" in line]) == 13
assert bridge.count("et_e1b_ensure_private_initialized_v1();") == 13
for stem in stems:
    assert f"et_e1b_public_g3_{stem}_v1(" in bridge
    assert f"et_e1b_private_g3_{stem}_cabi_v1(" in bridge
assert wrapper.count("(g3g-public-boundary '") == 13
assert "(g3t-seed-config config 'generator-create)" in wrapper
assert "(g3m-generate-p1-g1! generator input)" in wrapper
assert "(g3t-zero-generate! generator input)" in wrapper
assert "(g3t-p2-zero-generate! generator input)" in wrapper
assert "(g3m-native-full-request-preflight" in wrapper
assert wrapper.count("(vector-set! g3g-g0-input-provenance 0 next)") == 2
assert "ET_G3T_TESTING" not in wrapper
build = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
assert "-DET_G3T_ZERO_BUDGET_PRIVATE" in build
assert "-DET_G3T_P2_ZERO_BUDGET_PRIVATE" in build
assert "g3g-g0-public-aggregate" in build
assert "-DET_G3T_TESTING" not in (ROOT / "scripts/build-e1b-consumer.sh").read_text()
private_test = (ROOT / "tests/g3t/p2_zero_budget_test.esk").read_text()
v2_private = (ROOT / "tests/g3g_g0/private_fault_runtime.esk").read_text()
for witness in (
    "P2/G1 leaves cache and pins",
    "failed release preserves provenance and owner",
    "A2 rollback keeps empty cache and drains pins",
    "P2/G0 detached exact clone values",
):
    assert witness in v2_private
public_test = (ROOT / "tests/g3g_g0/public_runtime.esk").read_text()
assert "private G3-M rejection has public E1 operation and data-only details" in public_test
assert "T1 P2 public G0" in public_test
for witness in (
    "G3-G constructor cut maps invoked public operation",
    "G3-G ID clone cut maps invoked public operation",
    "G3-G RNG clone cut maps invoked public operation",
    "G3-G public adapter cut retry and owners release",
):
    assert witness in private_test
print("G3-G G0/G1 public source contract: PASS")
