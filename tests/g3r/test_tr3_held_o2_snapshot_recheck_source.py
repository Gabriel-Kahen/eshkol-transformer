"""Source gate for one enrolled TR3 snapshot lease with its original O2 hold."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXT = (ROOT / "native/g3r_tr3_held_o2_snapshot_recheck_extension.esk").read_text()
ROOT_ESK = (ROOT / "native/g3r_tr3_held_o2_snapshot_recheck_root.esk").read_text()
TR3 = (ROOT / "native/tr3_lease_core_extension.esk").read_text()
O2 = (ROOT / "native/o2_optimizer.c").read_text()
BRIDGE = (ROOT / "native/o2_wave2_package_bridge.c").read_text()


def section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def check_held(source: str) -> None:
    held = section(
        source,
        "(define (tr3-lease-snapshot-recheck-held-o2-internal",
        "    enrolled))",
    )
    ordered = (
        "tr3-lease-authenticate-minimal record operation",
        "tr3-lease-authenticate\n             minimal",
        "(eq? (vector-ref enrolled 3) 'snapshotting)",
        "(eq? (vector-ref enrolled 4) parent)",
        "tr3-lease-snapshot-parent-header!",
        "(eq? (vector-ref parent 3) 'constructing)",
        "tr3-lease-m3-idle! (vector-ref enrolled 9) operation",
        "tr3-lease-d2-idle! (vector-ref enrolled 16) operation",
        "o2-require-optimizer operation (vector-ref enrolled 13)",
        "(eq? optimizer optimizer-record)",
        "(tr3-held-o2-completed-updates\n                hold (vector-ref optimizer-record 1))",
        "(if (< completed 0) (o2-native-fail operation))",
        "(= completed (vector-ref enrolled 20))",
    )
    positions = [held.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert "o2-native-optimizer-completed-updates" not in held
    assert "o2-native-optimizer-require-absent-gradients" not in held
    assert "tr3-lease-restore-enter-admission!" not in held


check_held(EXT)
for old, new in (
    ("tr3-lease-authenticate-minimal record operation", "record"),
    ("(eq? (vector-ref enrolled 4) parent)", "#t"),
    ("tr3-lease-m3-idle! (vector-ref enrolled 9) operation", "#t"),
    ("tr3-lease-d2-idle! (vector-ref enrolled 16) operation", "#t"),
    ("(eq? optimizer optimizer-record)", "#t"),
    ("hold (vector-ref optimizer-record 1)", "hold (vector-ref enrolled 9)"),
    ("(if (< completed 0) (o2-native-fail operation))", "#t"),
    ("(= completed (vector-ref enrolled 20))", "#t"),
):
    assert old in EXT
    try:
        check_held(EXT.replace(old, new, 1))
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError(f"held TR3 authority mutation escaped: {old}")

assert ':real et_g3r_private_o2_adopter_completed_updates_v1' in EXT
assert '(load "tr3_c_snapshot_root.esk")' in ROOT_ESK
assert '(load "g3r_tr3_held_o2_snapshot_recheck_extension.esk")' in ROOT_ESK
assert "(provide" not in EXT + ROOT_ESK

enter = section(TR3, "(define (tr3-lease-snapshot-enter-internal!", "(define (tr3-lease-snapshot-recheck-internal")
ordinary = section(TR3, "(define (tr3-lease-snapshot-recheck-internal", "(define (tr3-lease-snapshot-abort-internal!")
assert "tr3-lease-restore-enter-admission! record operation" in enter
assert "tr3-lease-restore-enter-admission! enrolled operation" in ordinary
assert "tr3-lease-o2-restore-idle! record operation" in TR3

witness = section(O2, "int32_t et_g3r_private_o2_save_completed_updates_v1(", "int32_t et_g3r_private_o2_save_issued_match_v1(")
owner = section(O2, "int32_t et_g3r_private_o2_save_owner_match_v1(", "int32_t et_g3r_private_o2_save_completed_updates_v1(")
held_check = section(O2, "int32_t et_g3r_private_o2_save_check_v1(", "int32_t et_g3r_private_o2_save_owner_match_v1(")
assert "et_g3r_private_o2_save_owner_match_v1(" in witness
assert "hold->completed_updates > (uint64_t)INT64_MAX" in witness
assert "hold->optimizer->completed_updates != hold->completed_updates" in witness
assert "et_o2_find_optimizer(optimizer) != hold->optimizer" in owner
assert "return et_g3r_private_o2_save_check_v1(hold, error);" in owner
assert "et_o2_validate_absent_gradients(optimizer, operation, error)" in held_check
assert "et_g3r_private_o2_save_completed_updates_v1(" in BRIDGE
assert "return -1;" in section(BRIDGE, "int64_t et_g3r_private_o2_adopter_completed_updates_v1(", "void *et_g3r_private_o2_adopter_issue_v1(")

print("TR3 held O2 snapshot recheck source PASS (8 authority mutations)")
