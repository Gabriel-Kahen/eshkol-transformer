#!/usr/bin/env python3
"""Closed fixed-profile G3-G revision-3 package source contract."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
native = root / "native"
prefix = native / "g3g_manual_package"
read = lambda suffix: Path(f"{prefix}_{suffix}.txt").read_text().splitlines()
exports = read("public_exports")
strings = read("public_strings")
renames = read("private_renames")
sources = read("source_closure")
native_sources = read("native_source_closure")
facades = read("facades")
assert len(exports) == 105 and exports == sorted(set(exports))
assert len(strings) == 111 and strings == sorted(set(strings))
assert len(sources) == 45 and len(sources) == len(set(sources))
assert len(native_sources) == 57 and len(native_sources) == len(set(native_sources))
assert len(read("native_objects")) == 5
assert len(facades) == 8 and facades == sorted(set(facades))
assert read("archive_members") == ["g3g_manual_package.o"]
added = {
    "et_e1b_public_g3c4_model_create_seeded_v1",
    "et_e1b_public_g3c4_generation_input_create_v1",
    "et_e1b_public_g3_generator_prefill_v1",
    "et_e1b_public_g3_generator_decode_step_v1",
    "et_e1b_public_g3_diagnostic_generation_logits_bits_v1",
}
assert set(exports) == set((native / "g3g_g0_package_public_exports.txt").read_text().splitlines()) | added
assert set(strings) == set((native / "g3g_g0_package_public_strings.txt").read_text().splitlines()) | added
assert {line.split()[1] for line in renames if line.startswith(("g3g-", "g3c4-public-"))} == {
    name.replace("et_e1b_public_", "et_e1b_private_").replace("_v1", "_cabi_v1")
    for name in added | {name for name in exports if name.startswith("et_e1b_public_g3_")}
}
assert sources[0] == "native/g3g_manual_package_root.esk"
assert sources[-6:] == [
    f"native/{name}.esk" for name in (
        "g3c4_i2_construction_extension", "g3c4_p1_construction_extension",
        "g3c4_model_extension", "g3c4_call_entry_extension",
        "g3c4_public_model_input_extension", "g3g_manual_public_extension",
    )
]
assert "native/g3g_manual_package_bridge.c" in native_sources
assert "src/eshkol_transformer/g3c4_model_owner.c" in native_sources
assert "-DET_G3C4_NATIVE_PINS_PRIVATE" in (
    root / "scripts/build-e1b-consumer.sh"
).read_text()
assert (root / "scripts/build-e1b-consumer.sh").read_text().count(
    "-DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE") == 3
assert "native/i64_t1_pair_private.h" in native_sources
assert "native/g3g_g0_package_bridge.c" not in native_sources
package_root = (native / "g3g_manual_package_root.esk").read_text()
for name in (name for name in sources[-6:]
             if name != "native/g3c4_p1_construction_extension.esk"):
    assert package_root.count(f'(load "{Path(name).name}")') == 1
bridge = (native / "g3g_manual_package_bridge.c").read_text()
assert bridge.count("et_e1b_ensure_private_initialized_v1();") == 18
facade = (native / "g3g_manual_facades/transformer/generation.esk").read_text()
for name in ("generator-prefill!", "generator-decode-step!",
             "diagnostic-generation-logits-bits", "generation-c4-input-create"):
    assert facade.count(f"(define ({name} ") == 1
assert "(define (diagnostic-c4-model-create-seeded " in (
    native / "g3g_manual_facades/transformer/diagnostic_transport.esk"
).read_text()
assert "(define (diagnostic-c4-model-create-seeded " not in (
    root / "lib/transformer/diagnostic_transport.esk"
).read_text()
assert "et_g3t_private_logits_copy_bits_v1" not in bridge
assert "ET_G3T_TESTING" not in (root / "scripts/build-e1b-consumer.sh").read_text()
print("G3-G manual package source contract: PASS")
