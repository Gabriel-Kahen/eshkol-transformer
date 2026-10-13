#!/usr/bin/env python3
"""Bounded structural gate for the private joint original-owner hold."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "native/g3r_joint_hold.c").read_text()
header = (root / "native/g3r_joint_hold_internal.h").read_text()


def check(s: str, h: str) -> None:
    begin = s[s.index("et_g3r_joint_hold *et_g3r_joint_hold_begin_v1"):s.index(
        "int64_t et_g3r_joint_hold_check_v1")]
    roster = begin.index("et_m3t_private_owner_save_roster_v1")
    g3 = begin.index("et_g3r_private_g3t_save_begin_v1")
    o2 = begin.index("et_g3r_private_o2_save_begin_v1")
    assert begin.index("calloc(1, sizeof(*hold))") < roster < g3 < o2
    assert "generator, parameters, identities" in begin
    assert "optimizer, (et_f32_parameter *const *)parameters, identities" in begin
    assert "et_g3r_private_g3t_save_end_v1(hold->g3_hold)" in begin[o2:]
    assert "joint_o2_error(&o2_error);" in begin[o2:]
    issue = s[s.index("et_o2_optimizer_state *et_g3r_joint_hold_issue_v1"):s.index(
        "int64_t et_g3r_joint_hold_issued_match_v1")]
    assert "hold->issued || hold->issued_once" in issue
    assert issue.index("hold->issued = issued;") < issue.index(
        "et_g3r_joint_hold_issued_match_v1(hold, issued)")
    assert issue.index("hold->issued = issued;") < issue.index(
        "hold->issued_once = 1u;") < issue.index(
            "et_g3r_joint_hold_issued_match_v1(hold, issued)")
    match = s[s.index("int64_t et_g3r_joint_hold_issued_match_v1"):s.index(
        "int64_t et_g3r_joint_hold_copy_words_v1")]
    assert "state != hold->issued" in match
    assert "et_g3r_private_o2_save_issued_match_v1" in match
    assert "et_g3r_private_o2_save_owner_match_v1" in s
    release = s[s.index("int64_t et_g3r_joint_hold_release_issued_v1"):s.index(
        "int64_t et_g3r_joint_hold_end_v1")]
    assert "et_o2_optimizer_state_lifecycle_v1" in release
    assert "ET_O2_OPTIMIZER_STATE_LIVE" in release
    assert "ET_O2_OPTIMIZER_STATE_DEAD" in release
    assert "abort(); /* RELEASING" in release
    end = s[s.index("int64_t et_g3r_joint_hold_end_v1"):s.index(
        "int64_t et_g3r_joint_hold_last_domain_v1")]
    assert end.index("et_g3r_private_o2_save_end_v1") < end.index(
        "et_g3r_private_g3t_save_end_v1")
    assert "if (!hold || hold->issued) return joint_invalid();" in end
    assert "if (hold->o2_hold != NULL) abort()" in end
    assert "ET_G3R_COMMITTED_HISTORY_PRIVATE" in h
    assert "ET_G3R_SAME_OWNER_SAVE_PRIVATE" in h


check(source, header)
for altered in (
    source.replace("state != hold->issued", "state == hold->issued"),
    source.replace("hold->issued = issued;", "/* lost issued root */"),
    source.replace("hold->issued_once = 1u;", "/* repeat allowed */"),
    source.replace("et_g3r_private_o2_save_end_v1(&hold->o2_hold, &error)",
                   "et_g3r_private_o2_save_check_v1(hold->o2_hold, &error)"),
    source.replace("optimizer, (et_f32_parameter *const *)parameters, identities",
                   "optimizer, foreign_parameters, identities"),
):
    try:
        check(altered, header)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("joint-hold mutation escaped source gate")

print("joint G3/O2 hold source gate: pass (5 mutation controls)")
