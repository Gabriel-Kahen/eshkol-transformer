#!/usr/bin/env python3
"""Causal source gate for the feature-private held O2 count witness."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
native = (root / "native/o2_optimizer.c").read_text()
bridge = (root / "native/o2_wave2_package_bridge.c").read_text()
adopter = (root / "native/g3r_held_o2_adopter_extension.esk").read_text()
header = (root / "native/o2_optimizer_internal.h").read_text()


def body(source: str, name: str, next_name: str) -> str:
    return source.split(name, 1)[1].split(next_name, 1)[0]


def check(c: str, b: str, s: str, h: str) -> None:
    name = "int32_t et_g3r_private_o2_save_completed_updates_v1("
    witness = body(c, name, "int32_t et_g3r_private_o2_save_issued_match_v1(")
    assert name in h
    assert witness.index("et_o2_g3r_save_buffers_conflict(") < witness.index(
        "et_g3r_private_o2_save_owner_match_v1(")
    assert "if (result != 0) return result;" in witness
    assert witness.index("if (result != 0) return result;") < witness.index(
        "*completed_updates = hold->completed_updates;")
    assert "hold->completed_updates > (uint64_t)INT64_MAX" in witness
    assert "hold->optimizer->completed_updates != hold->completed_updates" in witness
    assert "et_o2_g3r_hold_admit(candidate)" in witness
    assert "return et_o2_success(error);" in witness

    held_bridge = body(
        b, "int64_t et_g3r_private_o2_adopter_completed_updates_v1(",
        "void *et_g3r_private_o2_adopter_issue_v1(")
    assert "et_g3r_private_o2_save_completed_updates_v1(" in held_bridge
    assert "&completed, &et_o2_bridge_error) != 0" in held_bridge
    assert "return -1;" in held_bridge
    assert "return (int64_t)completed;" in held_bridge

    adopt = body(s, "(define (g3r-held-o2-adopt!", ";;; Called only after C2")
    assert "et_g3r_private_o2_adopter_completed_updates_v1" in s
    assert "(g3r-held-o2-completed-updates" in adopt
    assert "o2-native-optimizer-completed-updates" not in adopt
    assert adopt.index("(g3r-held-o2-completed-updates") < adopt.index(
        '(o2-make-identity "transformer.optimizer-state:v1")')
    assert adopt.index("(if (< completed 0) (o2-native-fail operation))") < adopt.index(
        "(let ((issued (g3r-held-o2-issue hold)))")
    assert "(g3r-held-o2-owner-match hold native-optimizer)" in adopt


check(native, bridge, adopter, header)
mutants = (
    (native.replace("et_g3r_private_o2_save_owner_match_v1(\n      candidate, optimizer, error)",
                    "et_g3r_private_o2_save_check_v1(candidate, error)"),
     bridge, adopter, header),
    (native.replace("if (result != 0) return result;\n  et_o2_g3r_save_hold *hold",
                    "if (result == 0) return result;\n  et_o2_g3r_save_hold *hold"),
     bridge, adopter, header),
    (native.replace("*completed_updates = hold->completed_updates;",
                    "*completed_updates = 0u;"), bridge, adopter, header),
    (native, bridge,
     adopter.replace("(g3r-held-o2-completed-updates\n                       hold native-optimizer)",
                     "(o2-native-optimizer-completed-updates native-optimizer)"),
     header),
    (native, bridge,
     adopter.replace("(if (< completed 0) (o2-native-fail operation))", "#t"),
     header),
)
for mutant in mutants:
    try:
        check(*mutant)
    except (AssertionError, ValueError):
        continue
    raise AssertionError("held-count authority mutation escaped source gate")

print("held O2 count source gate PASS (5 causal mutations)")
