#!/usr/bin/env python3
"""Closed source and private symbol contract for G3-T logits snapshots."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
closure = (root / "native/g3t_manual_logits_materialize_source_closure.txt").read_text().splitlines()
base = (root / "native/g3m_manual_decode_source_closure.txt").read_text().splitlines()
assert closure == base + [
    "native/g3t_manual_logits_materialize_extension.esk",
    "native/g3t_manual_logits_materialize_local_symbols.txt",
    "scripts/check-g3t-manual-logits-materialize.py",
    "docs/g3/G3_T_MANUAL_LOGITS_MATERIALIZATION_PROPOSAL.md",
    "tests/g3t/test_manual_logits_materialize.c",
    "scripts/test-g3t-manual-logits-materialize-native.sh",
]
assert len(closure) == len(set(closure))
assert all((root / item).is_file() for item in closure if not item.startswith(".deps/"))
assert (root / "native/g3t_manual_logits_materialize_local_symbols.txt").read_text().splitlines() == ["g3t-logits-bits"]
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
header = (root / "src/eshkol_transformer/g3t_transport.h").read_text()
wrapper = (root / "native/g3t_manual_logits_materialize_extension.esk").read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
stem = "et_g3t_private_logits_copy_bits_v1"
assert native.count(stem) == 1 and header.count(stem) == 1
assert "#ifdef ET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE" in native
assert "g3t_admit_record(\n      candidate, G3T_LOGITS, 0)" in native
assert "et_f32_tensor_borrow_begin_v1(logits->tensor" in native
assert "et_f32_tensor_copy_bits_to_v1(" in native
assert "et_f32_tensor_borrow_end_v1(&borrow" in native
assert "view->shape[0] != 1 || view->shape[1] != 256" in native
assert "stride0 != 1024 || stride1 != 4" in native
assert "byte_count != 1024" in native
assert "(define (g3t-logits-bits logits)" in wrapper
assert "(make-bytevector 1024 0)" in wrapper
assert "(provide" not in wrapper
assert '(load "g3t_manual_logits_materialize_extension.esk")' in test
for witness in (
    "manual logits snapshot exact 256 bits and shape",
    "manual logits snapshot P2 detached last-row bits",
    "manual logits snapshot decode detached last-row bits",
    "manual logits snapshot borrowed owner rejects atomically",
    "manual logits snapshot pending owner rejects atomically",
    "manual logits snapshot I2 allocation cut leaves destination",
    "manual logits snapshot dead owner rejected",
):
    assert witness in test, witness
assert "scripts/check-g3t-manual-logits-materialize.py" in runner
assert "native/g3t_manual_logits_materialize_source_closure.txt" in runner
assert "-DET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE" in runner
for public in (
    "native/g3g_package_root.esk", "native/g3g_g0_package_root.esk",
    "native/g3g_package_public_exports.txt", "native/g3g_g0_package_public_exports.txt",
    "lib/transformer/generation.esk",
):
    assert stem not in (root / public).read_text()
    assert "g3t-logits-bits" not in (root / public).read_text()
print("G3-T private manual logits materialization source contract: PASS")
