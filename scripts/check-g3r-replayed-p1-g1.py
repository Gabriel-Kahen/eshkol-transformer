#!/usr/bin/env python3
"""Causal source checks for the private replayed P1/G1 transition."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
source = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
roles = (root / "src/eshkol_transformer/g3t_prefill_roles.inc").read_text()
header = (root / "src/eshkol_transformer/g3t_transport.h").read_text()
wrapper = (root / "native/g3r_replayed_p1_g1_extension.esk").read_text()
fixture = (root / "tests/g3r/replayed_p1_g1_same_process_test.esk").read_text()
base = (root / "native/g3m_seeded_p1_g1_source_closure.txt").read_text().splitlines()
added = [
    "native/g3r_replayed_p1_g1_extension.esk",
    "native/g3r_replayed_p1_g1_local_symbols.txt",
    "tests/g3r/replayed_p1_g1_native_test.c",
    "tests/g3r/replayed_p1_g1_same_process_test.esk",
    "scripts/check-g3r-replayed-p1-g1.py",
    "scripts/test-g3r-replayed-p1-g1-host.sh",
    "docs/g3/G3_R_REPLAYED_P1_G1_PRIVATE_LEAF.md",
]
closure = (root / "native/g3r_replayed_p1_g1_source_closure.txt").read_text().splitlines()
assert closure == base + added and len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
assert (root / "native/g3r_replayed_p1_g1_local_symbols.txt").read_text().splitlines() == [
    "g3r-native-replay-preflight", "g3r-native-replay-acquire",
    "g3r-replay-tokenizer-live", "g3r-replay-call-acquire",
    "g3r-private-replayed-p1-g1!",
]
assert "et_g3t_private_replayed_p1_g1_preflight_v1" in header
assert "et_g3t_private_replayed_p1_g1_acquire_v1" in header
assert "#ifdef ET_G3R_REPLAYED_P1_G1_PRIVATE" in source
assert "c->prefill_committed))" in source.split(
    "int64_t et_g3t_private_call_acquire_v1", 1)[1].split(
    "static int g3t_replay_coherent", 1)[0]
assert "c->prefill_committed || input->h.busy" in source.split(
    "et_g3t_private_full_request_preflight_v1", 1)[1].split(
    "et_g3t_private_prompt_preflight_v1", 1)[0]


def section(name: str, next_name: str, text: str = source) -> str:
    return text.split(name, 1)[1].split(next_name, 1)[0]


def guards(native: str = source, frame_roles: str = roles) -> None:
    assert "&& !c->replayed_p1_g1_call" in native
    assert "!c->history_pending_g1 && c->replayed_p1_g1_call" in native
    admission = section("static int64_t g3t_replay_idle", "static int64_t g3t_replay_binding_cache", native)
    for required in ("!c->prefill_committed", "c->history_length != 1",
                     "c->history_prompt_length != 1", "c->history_generated_count",
                     "c->history_manual_count", "c->history_generated_eos_terminal",
                     "c->frame.next_ordinal", "c->sample_token", "c->rng[2] == -1 && c->rng[3] == -1", "c->save_hold"):
        assert required in admission, required
    binding = section("static int64_t g3t_replay_binding_cache",
                      "et_g3t_private_replayed_p1_g1_preflight_v1", native)
    for required in ("et_g3t_model_pins_check_internal", "binding_identities",
                     "binding_values", "et_a2_kv_cache_read_borrow_begin_v1",
                     "et_a2_kv_cache_read_borrow_layer_v1",
                     "((const int64_t *)len->data)[0] != 1",
                     "((const uint8_t *)mask->data)[1] != 0",
                     "et_a2_kv_cache_read_borrow_end_v1"):
        assert required in binding, required
    for name, next_name in (
        ("et_g3t_private_output_reserve_v1", "et_g3t_private_output_release_v1"),
        ("et_g3t_private_output_prepare_v1", "static int g3t_span_ok"),
        ("static g3t_output *g3t_ready_output", "static int64_t g3t_decode_carrier"),
        ("et_g3t_private_frame_begin_v1", "et_g3t_private_frame_prepare_v1"),
        ("et_g3t_private_frame_prepare_v1", "static int64_t g3t_zero_preflight"),
        ("static int64_t g3t_final_preflight", "et_g3t_private_call_prepare_end_v1"),
        ("et_g3t_private_call_prepare_end_v1", "et_g3t_private_frame_commit_v1"),
        ("et_g3t_private_frame_commit_v1", "et_g3t_private_call_finish_v1"),
        ("et_g3t_private_sample_v1", "et_g3t_private_call_abort_v1"),
    ):
        assert "g3t_replay_coherent(c)" in section(name, next_name, native), name
    assert "g3t_replay_coherent(c)" in section(
        "et_g3t_private_role_step_v1", "if (ordinal < 0", frame_roles)
    assert "c->replayed_p1_g1_call = 0;" in section(
        "et_g3t_private_call_finish_v1", "et_g3t_private_sample_v1", native)
    assert "c->replayed_p1_g1_call = 0;" in section(
        "et_g3t_private_call_abort_v1", "#ifdef ET_G3T_TESTING", native)
    assert "if (!g3t_replay_coherent(c)) abort();" in section(
        "et_g3t_private_call_finish_v1", "et_g3t_private_sample_v1", native)


guards()
# Each deletion is a source mutation of one causal guard, not a copied
# implementation assertion. The check must reject all mutated variants.
for start, end, original, changed in (
    ("static int64_t g3t_replay_idle", "static int64_t g3t_replay_binding_cache",
     "c->history_length != 1", "c->history_length > 2"),
    ("static int64_t g3t_replay_idle", "static int64_t g3t_replay_binding_cache",
     "c->rng[2] == -1 && c->rng[3] == -1", "0"),
    ("static int64_t g3t_replay_binding_cache",
     "et_g3t_private_replayed_p1_g1_preflight_v1",
     "((const int64_t *)len->data)[0] != 1", "0"),
    ("static int64_t g3t_history_append_preflight",
     "static void g3t_history_commit_prompt",
     "!c->history_pending_g1 && c->replayed_p1_g1_call", "0"),
    ("et_g3t_private_call_finish_v1", "et_g3t_private_sample_v1",
     "c->replayed_p1_g1_call = 0;", "/* marker retained */"),
):
    prefix, tail = source.split(start, 1)
    body, suffix = tail.split(end, 1)
    assert original in body
    mutant = prefix + start + body.replace(original, changed, 1) + end + suffix
    try:
        guards(mutant)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"source mutation survived: {original}")
assert "(provide" not in wrapper
for required in ("(t1-private-entry tokenizer)", "(eq? (car found) tokenizer)",
                 "(eq? (vector-ref core 1) 'raw)",
                 "(= (t1-tokenizer-core-vocab-size core) 256)",
                 "sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704",
                 "(g3t-model-entry-live", "(eq? (vector-ref generator-entry 9) #f)",
                 "(eq? (vector-ref model-entry 10) #f)"):
    assert required in wrapper
assert wrapper.index("(g3r-replay-tokenizer-live generator-entry operation)") < wrapper.index(
    "(g3r-native-replay-preflight") < wrapper.index("(set! call (g3r-replay-call-acquire")
assert wrapper.index("(guard (caught") < wrapper.index("(set! call (g3r-replay-call-acquire")
assert re.findall(r"\(g3t-role-step call (\d+)\)", wrapper) == [
    str(i) for i in range(21)]
for required in ("(g3t-output-reserve call 1)", "(g3t-sample call)",
                 "(g3t-frame-begin call #f 2)", "(g3t-output-prepare call output)",
                 "(g3t-t1-decode-output! call output)", "(g3t-final-commit! call output)"):
    assert required in wrapper
assert '(load "g3r_replayed_p1_g1_extension.esk")' in fixture
for required in ("all P1 logits", "first A2 K/V row", "successor RNG words",
                 "full context rejects", "wrong-kind receiver"):
    assert required in fixture
print("G3-R replayed P1/G1 source and mutation checks: PASS")
