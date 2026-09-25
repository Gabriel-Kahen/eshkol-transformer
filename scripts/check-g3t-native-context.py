#!/usr/bin/env python3
"""Structural gate for the authentic private G3-T native owner precursor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check() -> None:
    source = (ROOT / "src/eshkol_transformer/g3t_transport.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3t_transport.h").read_text()
    test = (ROOT / "tests/g3t/native_context_test.esk").read_text()
    runner = (ROOT / "scripts/test-g3t-native-context.sh").read_text()
    note = (ROOT / "docs/g3/G3_T_NATIVE_CONTEXT_LEAF.md").read_text()
    assert '#include "m3_model.c"' in source
    assert "g3c4" not in source.lower() and "g3c4" not in header.lower()
    for name in ("generator_seed", "generator_close", "call_acquire",
                 "call_abort", "last_error_domain", "last_error_category",
                 "last_error_code"):
        spelling = f"et_g3t_private_{name}_v1"
        assert spelling in source and spelling in header, spelling
    for phrase in ("while (r && r != candidate)",
                   "o->r.state != 1", "o->initialized != 14",
                   "et_f32_parameter_validate_identity_v1",
                   "et_a2_kv_cache_create_v1(1, 1, 2, 2, 2",
                   "et_g3t_model_pins_begin_internal",
                   "et_g3t_model_pins_check_internal",
                   "et_g3t_model_pins_end_internal",
                   "o->active = c", "o->active = NULL",
                   "et_a2_kv_cache_destroy_v1", "c->h.state = G3T_DEAD"):
        assert phrase in source, phrase
    for phrase in ("foreign model pointer rejected", "initializer pointer is wrong kind",
                   "context allocation failure", "A2 allocation failure",
                   "same-model native frame excludes G3", "all fourteen pinned",
                   "busy close rejected", "idempotent exact close",
                   "M3T model remains usable"):
        assert phrase in test, phrase
    for phrase in ("build_mode normal", "build_mode sanitize", "repeat.stdout",
                   "detect_leaks=1", "ET_G3T_TESTING", "production.o",
                   "test hook escaped"):
        assert phrase in runner, phrase
    assert "cannot prefill, decode, sample, commit output" in note
    dependency = (ROOT / "native/g3t_model_admission_source_closure.txt").read_text().splitlines()
    native = (ROOT / "native/m3_package_native_source_closure.txt").read_text().splitlines()
    additions = ["native/m3_package_native_source_closure.txt", *native,
                 "src/eshkol_transformer/m3_call_f32_integration.c",
                 "src/eshkol_transformer/g3t_transport.c",
                 "src/eshkol_transformer/g3t_transport.h",
                 "src/eshkol_transformer/m3_call_pins.h",
                 "native/a2_kv_cache.c", "include/eshkol_transformer/a2_kv_cache.h",
                 "tests/g3t/native_context_test.esk",
                 "scripts/check-g3t-native-context.py",
                 "scripts/test-g3t-native-context.sh",
                 "docs/g3/G3_T_NATIVE_CONTEXT_LEAF.md"]
    expected = dependency + [path for path in additions if path not in dependency]
    actual = (ROOT / "native/g3t_native_context_source_closure.txt").read_text().splitlines()
    assert actual == expected, "native context closure differs from exact dependency union"
    assert len(actual) == len(set(actual)), "duplicate source closure path"
    for path in actual:
        if path.startswith(".deps/eshkol-src/"):
            assert path in native, path
        else:
            assert (ROOT / path).is_file(), path


if __name__ == "__main__":
    check()
    print("G3-T authentic native context source contract: PASS")
