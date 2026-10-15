"""Source-order gate for the private TR3/G3/O2 original-owner hold ledger."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ESK = (ROOT / "native/g3r_original_owner_coordinator_extension.esk").read_text()
ROOT_ESK = (ROOT / "native/g3r_original_owner_coordinator_root.esk").read_text()
BRIDGE = (ROOT / "native/o2_wave2_package_bridge.c").read_text()
G3 = (ROOT / "native/g3r_trainer_g3_save_admission_extension.esk").read_text()
TR3 = (ROOT / "native/g3r_tr3_held_o2_snapshot_recheck_extension.esk").read_text()


def section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def check(source: str) -> None:
    begin = section(source, "(define (g3r-original-owner-begin!", "(define (g3r-original-owner-end!")
    ordered = (
        "(g3-cell (vector #f))",
        "(o2-cell (vector #f))",
        "(ledger (vector g3r-original-owner-tag",
        "(guard (caught",
        "(g3r-original-o2-native-end",
        "(g3r-trainer-g3-save-end! (vector-ref g3-cell 0))",
        "(g3r-trainer-g3-save-begin! trainer generator)",
        "(vector-set! g3-cell 0 g3-ledger)",
        "(g3r-original-o2-native-begin",
        "(vector-ref optimizer-record 1)",
        "(vector-ref model-entry 3)",
        "(vector-set! o2-cell 0 hold)",
        "(tr3-lease-snapshot-recheck-held-o2-internal",
        "(g3r-original-owner-g3-recheck! trainer generator g3-ledger)",
        "(g3r-trainer-g3-native-check",
        "(vector-set! ledger 5 'held)",
    )
    positions = [begin.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert begin.count("(tr3-lease-snapshot-recheck-held-o2-internal") == 2
    assert "(if (not (= status 0)) (tr3-lease-fail-stop 134))" in begin
    assert "(guard (cleanup-failure (#t (tr3-lease-fail-stop 134)))" in begin
    end = section(source, "(define (g3r-original-owner-end!", "  #t)")
    assert end.index("(g3r-original-o2-native-end") < end.index("(g3r-trainer-g3-save-end!")
    assert end.index("(g3r-trainer-g3-save-end!") < end.index("(vector-set! ledger 5 'dead)")
    assert "(tr3-lease-fail-stop 134)" in end
    assert "c2-training-state-compose-internal" not in source
    assert "c2-save-image-receipt" not in source


check(ESK)
for old, new in (
    ("(g3-cell (vector #f))", "(g3-cell #f)"),
    ("(g3r-original-o2-native-end\n                               (vector-ref o2-cell 0))", "0"),
    ("(g3r-trainer-g3-save-end! (vector-ref g3-cell 0))", "#t"),
    ("(vector-ref optimizer-record 1)\n                  (vector-ref model-entry 3)",
     "(vector-ref model-entry 3)\n                  (vector-ref optimizer-record 1)"),
    ("(vector-set! o2-cell 0 hold)", "#t"),
    ("(tr3-lease-snapshot-recheck-held-o2-internal\n          record parent (vector-ref o2-cell 0))", "#t"),
    ("(g3r-trainer-g3-native-check", "(g3r-original-o2-native-end"),
    ("(vector-set! ledger 5 'held)", "#t"),
):
    assert old in ESK
    try:
        check(ESK.replace(old, new, 1))
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError(f"original-owner mutation escaped: {old}")

g3_recheck = section(ESK, "(define (g3r-original-owner-g3-recheck!", "(define (g3r-original-owner-begin!")
for required in (
    "(vector-ref record 9) model-entry",
    "(module-mode-internal (vector-ref record 8)) 'train",
    "(g3t-generator-find generator operation) entry",
    "(vector-ref entry 5) model-entry",
    "(t1-private-entry tokenizer)",
    "g3r-trainer-g3-fingerprint",
):
    assert required in g3_recheck
assert "(load \"g3r_trainer_g3_save_root.esk\")" in ROOT_ESK
assert ROOT_ESK.index("g3r_trainer_g3_save_root.esk") < ROOT_ESK.index(
    "g3r_tr3_held_o2_snapshot_recheck_extension.esk") < ROOT_ESK.index(
    "g3r_original_owner_coordinator_extension.esk")
assert "(provide" not in ESK + ROOT_ESK

o2_begin = section(BRIDGE, "void *et_g3r_private_trainer_o2_save_begin_v1(", "int64_t et_g3r_private_trainer_o2_save_end_v1(")
assert o2_begin.index("et_m3t_private_owner_save_roster_v1(") < o2_begin.index(
    "et_g3r_private_o2_save_begin_v1(")
assert "parameters[i] = (et_f32_parameter *)raw_parameters[i]" in o2_begin
assert "if (hold == NULL) abort();" in o2_begin
assert "et_g3r_private_o2_save_end_v1(" in BRIDGE
assert "g3r-trainer-g3-native-end" in G3
assert "tr3-held-o2-completed-updates" in TR3
print("G3R original-owner coordinator source PASS (8 causal mutations)")
