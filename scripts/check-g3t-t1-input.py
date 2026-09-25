#!/usr/bin/env python3
"""Closed source and authority-order check for private G3-T T1 input."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_p2_zero_budget_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3t_t1_input_extension.esk",
    "scripts/check-g3t-t1-input.py",
    "docs/g3/G3_T_INPUT_FROM_T1_LEAF.md",
]
closure = (root / "native/g3t_t1_input_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
assert "native/t1_i64_shell.c" in closure
assert "native/t1_i64_shell.h" in closure
assert "src/eshkol_transformer/g3t_transport.c" in closure
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
extension = (root / additions[0]).read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
constructor = native.split("void *et_g3t_private_input_from_t1_v1", 1)[1].split(
    "int64_t et_g3t_private_tensor_release_v1", 1
)[0]
assert constructor.index("et_t1_i64_shell_length_v1") < constructor.index("calloc")
assert constructor.index("et_t1_i64_shell_read_v1") < constructor.index("calloc")
assert constructor.index("et_i64_tensor_create_v1") < constructor.index("g3t_registry =")
assert constructor.index("et_i64_tensor_copy_from_v1") < constructor.index("g3t_registry =")
assert extension.index("t1-wave1-tensor-admitted?") < extension.index(
    "g3t-native-input-from-t1 encoded"
)
assert "(provide" not in extension
assert "g3c4" not in native.lower()
assert "T1-owned I1 borrow blocks release" in test
assert "T1 I1 cut atomic" in test
print("G3-T private T1 input source contract: PASS")
