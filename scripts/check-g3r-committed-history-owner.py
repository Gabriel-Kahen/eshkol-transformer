#!/usr/bin/env python3
"""Check that every accepted G3-T commit tail updates one native owner."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
header = (root / "src/eshkol_transformer/g3t_transport.h").read_text()
commit = source.split("int64_t et_g3t_private_frame_commit_v1", 1)[1].split(
    "int64_t et_g3t_private_call_finish_v1", 1
)[0]
assert commit.count("g3t_history_commit_prompt(c);") == 3
assert commit.count("g3t_history_commit_append(c, 0);") == 1
assert commit.count("g3t_history_commit_append(c, 1);") == 1
assert commit.count("g3t_history_prompt_preflight(c)") == 1
assert "g3t_history_prompt_preflight(c)" in source.split(
    "static int64_t g3t_manual_p1_preflight", 1
)[1].split("static int64_t g3t_manual_decode_preflight", 1)[0]
assert "g3t_history_append_preflight(c)" in source.split(
    "static int64_t g3t_manual_decode_preflight", 1
)[1].split("static int64_t g3t_final_preflight", 1)[0]
assert "g3t_history_append_preflight(c)" in source.split(
    "static int64_t g3t_final_preflight", 1
)[1].split("et_g3t_private_call_prepare_end_v1", 1)[0]
assert "g3t_history_prompt_preflight(c)" in source.split(
    "static int64_t g3t_zero_preflight", 1
)[1].split("static int64_t g3t_manual_p1_preflight", 1)[0]
assert "memset(c->history_ids, 0" in source
assert "et_g3t_private_committed_history_snapshot_v1" in header
assert "et_g3t_private_committed_history_snapshot_v1" in source
prefill = (root / "native/g3t_prefill_sample_extension.esk").read_text()
final = (root / "native/g3t_final_publication_extension.esk").read_text()
manual_transport = (root / "native/g3c4_manual_transport_extension.esk").read_text()
for name in ("g3m_prefill_p1_extension.esk", "g3m_prefill_p2_extension.esk",
             "g3m_manual_decode_extension.esk"):
    wrapper = (root / "native" / name).read_text()
    assert "(g3t-generator-find generator operation)" in wrapper
    assert "(g3t-call-acquire generator-entry " in wrapper
    assert "(g3t-frame-commit call)" in wrapper
assert "(g3t-native-frame-commit (vector-ref entry 3))" in prefill
assert "(g3t-native-frame-commit native)" in final
assert "(g3t-native-frame-commit (vector-ref generator-entry 3))" in manual_transport
for name in ("g3t_zero_budget_extension.esk", "g3t_p2_zero_budget_extension.esk",
             "g3m_seeded_p1_g1_extension.esk"):
    assert "(g3t-final-commit! call output)" in (root / "native" / name).read_text()
print("G3-R committed history owner source contract: PASS")
