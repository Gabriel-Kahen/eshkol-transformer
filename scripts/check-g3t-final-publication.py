#!/usr/bin/env python3
"""Closed source inventory for the private G1 final publication leaf."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_output_text_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3t_final_publication_extension.esk",
    "tests/g3t/final_publication_test.esk",
    "scripts/check-g3t-final-publication.py",
    "scripts/test-g3t-final-publication.sh",
    "docs/g3/G3_T_FINAL_PUBLICATION_LEAF.md",
]
closure = (root / "native/g3t_final_publication_source_closure.txt").read_text().splitlines()
assert closure == base + [path for path in additions if path not in base]
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in additions)
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
source = (root / additions[0]).read_text()
test = (root / additions[1]).read_text()
for symbol in ("call_prepare_end", "call_finish", "frame_commit"):
    assert f"et_g3t_private_{symbol}_v1" in native
assert "g3t_final_preflight" in native
assert "et_a2_kv_cache_transaction_view_begin_v1" in native
assert "et_i64_tensor_borrow_begin_v1" in native
assert "c->final_committed = 1" in native
assert "(define (g3t-final-commit!" in source
assert "(define (g3t-output-release!" in source
assert "all 256 final logits equal independent M3T" in test
assert "commit revalidation cut before publication" in test
assert "g3c4" not in native.lower()
assert "(provide" not in source
print("G3-T private G1 final publication source contract: PASS")
