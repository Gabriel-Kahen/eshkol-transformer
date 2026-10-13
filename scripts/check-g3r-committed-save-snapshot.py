#!/usr/bin/env python3
"""Bounded source check for the private same-owner save copy seam."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
c = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
h = (root / "src/eshkol_transformer/g3t_transport.h").read_text()
esk = (root / "native/g3r_committed_save_snapshot_extension.esk").read_text()
name = "et_g3t_private_committed_save_copy_v1"
assert name in c and name in h and name in esk
copy = c.split(f"int64_t {name}(", 1)[1].split("\n#endif", 1)[0]
for required in (
    "g3t_admit(generator, 0)", "g3t_model(expected_model)",
    "c->model != o", "g3r_save_span(destination_header, 136",
    "g3r_save_span(staging_header, 136", "(d0 < s1 && s0 < d1)",
    "et_a2_kv_cache_private_storage_overlap_v1",
    "et_f32_tensor_private_storage_overlap_v1",
    "et_g3t_model_pins_begin_internal", "et_g3t_model_pins_check_internal",
    "et_a2_kv_cache_read_borrow_begin_v1", "et_a2_kv_cache_read_borrow_layer_v1",
    "et_g3t_private_committed_history_snapshot_v1(c, &snapshot)",
    "((const int64_t *)lengths->data)[0] != snapshot.length",
    "et_a2_kv_cache_read_borrow_end_v1", "et_g3t_model_pins_end_internal",
    "if (!status) memcpy((unsigned char *)staging_header + 8, &snapshot, 136)",
):
    assert required in copy, required
assert copy.index("et_g3t_model_pins_begin_internal") < copy.index(
    "et_a2_kv_cache_read_borrow_begin_v1"
) < copy.index("et_g3t_private_committed_history_snapshot_v1(c, &snapshot)")
assert copy.index("et_a2_kv_cache_read_borrow_end_v1") < copy.index(
    "if (!status) memcpy((unsigned char *)staging_header + 8, &snapshot, 136)"
)
for required in (
    "(m3-call operation", "(g3t-generator-find generator operation)",
    "(g3t-model-entry-live", "(g3r-save-tokenizer?",
    "(g3r-native-committed-save-copy", "(g3r-save-data staging entry)",
    "(bytevector-copy! destination 0 staging 0 g3r-save-size)",
    "(eq? (vector-ref registered 6) #f)",
):
    assert required in esk, required
assert esk.index("(g3r-save-data staging entry)") < esk.index(
    "(bytevector-copy! destination 0 staging 0 g3r-save-size)"
)
for public in (
    "native/g3g_package_root.esk", "native/g3g_manual_package_root.esk",
    "native/cli3_generate_root.esk",
):
    assert "g3r_committed_save_snapshot_extension.esk" not in (root / public).read_text()

depth = 0
in_string = False
escaped = False
for line in esk.splitlines():
    for ch in line:
        if not in_string and ch == ";":
            break
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            assert depth >= 0
assert depth == 0 and not in_string
print("G3-R committed save snapshot source contract PASS")
