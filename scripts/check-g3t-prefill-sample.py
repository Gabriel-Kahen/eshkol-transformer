#!/usr/bin/env python3
"""Static scope check for the private P1/G1 prefill/sample leaf."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
core = (ROOT / "src/eshkol_transformer/g3t_transport.c").read_text()
roles = (ROOT / "src/eshkol_transformer/g3t_prefill_roles.inc").read_text()
source = (ROOT / "native/g3t_prefill_sample_extension.esk").read_text()
test = (ROOT / "tests/g3t/prefill_sample_test.esk").read_text()
base = (ROOT / "native/g3t_generator_constructor_source_closure.txt").read_text().splitlines()
additions = [
    "include/eshkol_transformer/g3n_primitives_abi.h",
    "include/eshkol_transformer/g3s_sampling_abi.h",
    "native/g3n_primitives_provider.c",
    "native/g3s_sampling_provider.c",
    "templates/p1/module_roots.esk.tmpl",
    "scripts/test-p1-construction-schedule.sh",
    "tests/p1/construction_schedule_identity_test.esk",
    "src/eshkol_transformer/g3t_prefill_roles.inc",
    "native/g3t_prefill_sample_extension.esk",
    "native/g3t_prefill_sample_local_symbols.txt",
    "tests/g3t/prefill_sample_test.esk",
    "scripts/check-g3t-prefill-sample.py",
    "scripts/test-g3t-prefill-sample.sh",
    "docs/g3/G3_T_PREFILL_SAMPLE_LEAF.md",
    "docs/ROADMAP.md",
]
closure = (ROOT / "native/g3t_prefill_sample_source_closure.txt").read_text().splitlines()
assert closure == base + [path for path in additions if path not in base]
assert len(closure) == len(set(closure))
assert all((ROOT / path).is_file() for path in additions)
assert '#include "m3_model.c"' in core
assert '#include "g3t_prefill_roles.inc"' in core
assert "g3c4" not in (core + roles + source).lower()
assert "(provide" not in source
symbols = (ROOT / "native/g3t_prefill_sample_local_symbols.txt").read_text().splitlines()
assert symbols == ["g3t-prefill-dead!", "g3t-prefill-entry",
                   "g3t-prefill-call-linked", "g3t-prefill-call-entry"]
for symbol in symbols:
    assert f"(define ({symbol} " in source
for name in ("input_from_token", "tensor_release", "output_reserve",
             "frame_begin", "role_step", "frame_prepare", "frame_commit",
             "sample"):
    assert f"et_g3t_private_{name}_v1" in core + roles
for symbol in ("g3t-with-call", "g3t-frame-begin", "g3t-role-step",
               "g3t-frame-prepare", "g3t-frame-commit", "g3t-sample"):
    assert symbol in source and symbol in test
assert test.count("(g3t-native-role-step native ") >= 22
assert test.count("(g3t-native-role-step partial-native ") == 12
assert "all 256 prefill logits equal independent M3T" in test
assert "all 256 token-frame logits equal independent M3T" in test
assert "sampled attention failure preserves cache length" in test
assert "profile drift rejects sampled frame" in test
assert "if (c->frame.kind == 2)" in core
assert "return g3t_bad(G3T_STATE, G3T_LIFECYCLE);" in core
assert "c->frame.kind == 2 ? 1 : 0" in roles
assert (ROOT / "docs/g3/G3_T_TOKEN_FRAME_LEAF.md").is_file()
assert "G3T_ONLY_NORMAL" in (ROOT / "scripts/test-g3t-prefill-sample.sh").read_text()
assert "-DET_G3T_PREFILL_SAMPLE_PRIVATE" in (
    ROOT / "scripts/test-g3t-prefill-sample.sh").read_text()
assert "construction-schedule-equal?" in (
    ROOT / "internal/p1/lib/transformer/module.esk").read_text()
print("G3-T private P1/G1 prefill/sample source contract: PASS")
