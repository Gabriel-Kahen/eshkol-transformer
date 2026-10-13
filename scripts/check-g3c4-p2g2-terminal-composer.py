#!/usr/bin/env python3
"""Static, source-private Eshkol terminal-composer boundary checks."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3c4_p2g2_t1_prefix_coordinator_source_closure.txt").read_text().splitlines()
additions = [
    "docs/g3/G3_C4_P2_G2_NATIVE_TERMINAL_CONTRACT.md",
    "docs/g3/G3_C4_P2_G2_LOGICAL_I1_CLONE_CONTRACT.md",
    "docs/g3/G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md",
    "native/g3c4_p2g2_terminal_composer_extension.esk",
    "tests/g3c4/p2_g2_terminal_composer_native.c",
    "tests/g3c4/p2_g2_terminal_composer_test.esk",
    "tests/g3c4/p2_g2_terminal_composer_feature_off_test.esk",
    "tests/g3c4/p2_g2_terminal_fault_fixture.esk.in",
    "tests/g3c4/generate_p2_g2_terminal_fault_overlays.py",
    "tests/g3c4/check_p2_g2_terminal_composer_oracle.py",
    "tests/g3c4/test_context_cache_checker.py",
    "tests/g3c4/test_active_call_checker.py",
    "scripts/check-g3c4-p2g2-terminal-composer.py",
    "scripts/test-g3c4-p2g2-terminal-composer.sh",
    "tests/g3c4/call_entry_allocation_shim.cpp",
    "native/g3c4_p2g2_terminal_composer_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)
source = (root / "native/g3c4_p2g2_terminal_composer_extension.esk").read_text()
fixture = (root / "tests/g3c4/p2_g2_terminal_composer_test.esk").read_text()
native_fixture = (root / "tests/g3c4/p2_g2_terminal_composer_native.c").read_text()
runner = (root / "scripts/test-g3c4-p2g2-terminal-composer.sh").read_text()
oracle = (root / "tests/g3c4/check_p2_g2_terminal_composer_oracle.py").read_text()
off = (root / "tests/g3c4/p2_g2_terminal_composer_feature_off_test.esk").read_text()
shim = (root / "tests/g3c4/call_entry_allocation_shim.cpp").read_text()
for symbol in (
    "et_g3c4_composer_emergency_raise_v1",
    "et_g3c4_composer_emergency_capture_v1",
    "et_g3c4_composer_emergency_exact_v1",
):
    assert shim.count(f" {symbol}(") == 1, symbol
    assert f":real {symbol})" in fixture, symbol
    assert symbol not in source and symbol not in off, symbol
assert "call_entry_allocation_shim.cpp" in runner
assert "eshkol_runtime_emergency_raise_v1((int32_t)condition)" in shim
assert "eshkol_get_raised_value(&raised)" in shim
assert "g_current_exception == allocation_condition5" in shim
assert "raised.data.ptr_val ==" in shim
assert "object or exception-handler allocation failed" in shim
assert "condition == 4 || condition == 5" in shim
assert "different runtime emergency is not condition 5" in fixture
assert "(composer-emergency-raise 4) #f" in fixture
sweep = fixture[fixture.index("(define (allocation-sweep kind)"):
                fixture.index("(allocation-sweep 1)")]
assert "(= (composer-emergency-exact) 1)" in sweep
assert "(and caught caught-condition5)" in sweep
assert "(transformer-error? caught)" not in sweep
assert all(f"(allocation-sweep {kind})" in fixture for kind in (1, 2, 3))
for symbol in (
    "et_g3c4_private_p2g2_eos_first_frame_carrier_v1",
    "et_g3c4_private_p2g2_second_frame_carrier_v1",
    "et_g3c4_private_p2g2_terminal_commit_v1",
    "et_g3c4_private_p2g2_terminal_snapshot_v1",
    "et_g3c4_private_p2g2_terminal_ids_clone_v1",
    "et_g3c4_private_p2g2_terminal_ids_clone_copy_v1",
    "et_g3c4_private_p2g2_terminal_ids_clone_release_v1",
):
    assert source.count(symbol) == 1, symbol
for operation in (
    "g3c4-p2g2-terminal-generate-internal",
    "g3c4-p2g2-terminal-snapshot-internal",
    "g3c4-p2g2-terminal-release-internal!",
    "g3c4-p2g2-terminal-ids-clone-internal",
    "g3c4-p2g2-terminal-ids-copy-internal",
    "g3c4-p2g2-terminal-ids-release-internal!",
):
    assert f"(define ({operation} " in source, operation
assert source.index("(g3c4-p2g2-native-eos-first") < source.index(
    "(t1-private-g3-decode-raw-into!") < source.index(
    "(g3c4-p2g2-native-prefix-commit") < source.index(
    "(g3c4-p2g2-native-second") < source.index(
    "(g3c4-p2g2-native-terminal")
assert source.count("(t1-private-g3-decode-raw-into!") == 2
assert "(define g3c4-p2g2-terminal-originals" in source
assert "(eq? (vector-ref original index)" in source
assert "(vector-ref anchor index)" in source
assert "(g3c4-p2g2-protected-abort! call)" in source
assert "(vector-set! barrier 0 #t)" in source
tail = source[source.index("(vector-set! barrier 0 #t)"):
              source.index("(define (g3c4-p2g2-terminal-entry-live")]
assert "(g3c4-call-success!" not in tail
assert "(g3c4-native-" not in tail
assert "(g3c4-fail" not in tail
assert "(cons " not in tail and "(make-" not in tail
assert tail.count("(vector-set!") >= 20
for phrase in (
    "held source I1 prevents terminal release",
    "foreign owner cannot end held source I1 lease",
    "held clone I1 prevents clone release",
    "equal-length terminal text substitution rejected",
    "equal-length clone readback substitution rejected",
    "dead terminal original anchor substitution rejected",
    "dead clone original readback substitution rejected",
    "P2/P3 cache and original/first RNG survive abort",
    "independent P3 prefix logits and K/V reference",
    "actual P3/T4 cache equals independent direct prefix",
    "detached native I1 survives source release",
    "emit actual second-logit and RNG oracle row",
    "one sampled second token differs from its first token",
    "snapshot error leaves terminal usable",
    "clone copy error leaves owner usable",
    "allocation cut retains exact P0/P2/P3 cache and RNG",
):
    assert phrase in fixture, phrase
for macro in (
    "ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE",
    "ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE",
    "ET_G3C4_P2_G2_TERMINAL_PRIVATE",
    "ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE",
    "ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE",
):
    assert macro in native_fixture, macro
assert "--wrap=et_g3c4_private_p2g2_terminal_commit_v1" in runner
assert "--wrap=et_g3c4_private_p2g2_terminal_snapshot_v1" in runner
assert "--wrap=et_g3c4_private_p2g2_terminal_ids_clone_copy_v1" in runner
assert "--wrap=arena_allocate_vector_with_header" in runner
assert "--wrap=arena_allocate_cons_with_header" in runner
assert "--wrap=arena_allocate_with_header" in runner
assert "ASAN_OPTIONS=detect_leaks=1" in runner
assert "check_p2_g2_t1_prefix_oracle.py" in runner
assert "prefix-oracle-input.stdout" in runner
assert "check_p2_g2_terminal_composer_oracle.py" in runner
assert "p2_g2_terminal_composer_feature_off_test.esk" in runner
assert "composer rejects feature-off entry" in off
assert '(load "g3c4_p2g2_ownership_extension.esk")' not in off
assert '(load "g3c4_p2g2_terminal_composer_extension.esk")' in off
assert '(config 1)' in off and '(config 2)' in off
assert 'feature-off constructor rejects budget two' in off
assert 'private P2/G2 gate remains off' in off
assert "COMPOSER_SECOND " in native_fixture
assert "et_g3c4_p2g2_coordinator_test_prefix_v1(" in native_fixture
assert "et_g3c4_p2g2_coordinator_test_observe_v1(" in native_fixture
assert "reference.philox" in oracle and "reference.literal" in oracle
assert "normal-run.stdout" in runner and "sanitize-run.stdout" in runner
overlay_generator = (root / additions[8]).read_text()
assert "T1_SHA =" in overlay_generator and "COMPOSER_SHA =" in overlay_generator
assert "result.replace(insertion, \"\", 1) != source" in overlay_generator
assert "ambiguous/wrong-marker negative control escaped" in overlay_generator
assert "LEASE_MARKER =" in overlay_generator
assert "LEASE_INSERT =" in overlay_generator
assert "composer-held-lease-overlay.esk" in overlay_generator
assert "lease-i1" in overlay_generator and "lease-a2" in overlay_generator
assert "result.replace(insertion, \"\", 1) != source" in overlay_generator
assert "et_g3c4_p2g2_composer_test_lease_rejected_v1" in native_fixture
assert "ET_I64_TENSOR_CODE_ACTIVE_BORROW" in native_fixture
assert "ET_KERNEL_CODE_PROVIDER_REJECTED" in native_fixture
assert "composer_lease_cache_snapshot" in native_fixture
assert "composer_lease_pending_zero(pending, kind)" in native_fixture
assert "composer_lease_i1, &view, &error" in native_fixture
assert "composer_lease_calls != 2" in native_fixture
assert "--wrap=et_g3c4_private_call_abort_v1" in runner
assert "(overlay-lease-end" in overlay_generator
assert "native #f kind) -1" in overlay_generator
for root_name in ("terminal-head", "terminal-original-head", "record-head",
                  "protected-head", "prefix-head"):
    assert overlay_generator.count(root_name) == 2, root_name
assert "decode-first decode-second" in runner
assert "test \"$exit_status\" -eq 134" in runner
assert "cmp \"$evidence/$mode-parity-original-run.stdout\"" in runner
assert "categorical-singleton" in fixture and "composer-exhausted" in fixture
assert "categorical-carry" in native_fixture and "categorical-final" in native_fixture
assert "composer_clone_owner = clone" in native_fixture
assert "candidate == composer_clone_owner" in native_fixture
assert "foreign live clone cannot end held clone I1 lease" in fixture
assert "held clone still rejects release after foreign end" in fixture
print("G3-C4 private Eshkol terminal source closure and tail PASS")
