#!/usr/bin/env python3
"""Static source closure for the pending G1 ID and raw text leaf."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
predecessor = (root / "native/g3t_prefill_sample_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_t1_output_decode_extension.esk",
    "native/g3t_output_text_extension.esk",
    "tests/g3t/output_text_test.esk",
    "scripts/check-g3t-output-text.py",
    "scripts/test-g3t-output-text.sh",
    "docs/g3/G3_T_OUTPUT_TEXT_LEAF.md",
]
closure = (root / "native/g3t_output_text_source_closure.txt").read_text().splitlines()
assert closure == predecessor + [item for item in additions if item not in predecessor]
assert len(closure) == len(set(closure))
assert all((root / item).is_file() for item in additions)
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
wrapper = (root / "native/g3t_output_text_extension.esk").read_text()
decoder = (root / additions[0]).read_text()
test = (root / "tests/g3t/output_text_test.esk").read_text()
for symbol in ("output_copy_decode_ids", "output_accept_text"):
    assert f"et_g3t_private_{symbol}_v1" in native
assert "et_i64_tensor_private_storage_overlap_v1" in native
assert "et_a2_kv_cache_private_storage_overlap_v1" in native
assert "g3t_check_binding(c)" in native
assert "(t1-private-g3-decode-raw-into! tokenizer staging raw)" in wrapper
assert "(define (t1-private-g3-decode-raw-into!" in decoder
assert '(load "g3c4_t1_output_decode_extension.esk")' in test
assert "still no final frame prepare" in test
assert "byte0 staging" in test and "source wrapper byte255" in test
assert "(provide" not in wrapper
assert "g3c4" not in native.lower()
print("G3-T pending G1 ID/raw-text source contract: PASS")
