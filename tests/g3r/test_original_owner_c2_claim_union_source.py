"""Source gate for the single-registry original-owner C2 claim boundary."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"
UNION = NATIVE / "g3r_original_owner_c2_claim_root.esk"
CLAIM = (NATIVE / "g3r_held_o2_c2_claim_extension.esk").read_text()
COMPOSE = (NATIVE / "c2_training_state_extension.esk").read_text()
O2 = (NATIVE / "o2_wave2_extension.esk").read_text()


def loads(path):
    yield path.name
    for name in re.findall(r'^\(load "([^"]+)"\)', path.read_text(), re.M):
        child = NATIVE / name
        if child.is_file():
            yield from loads(child)


chain = list(loads(UNION))
for name in (
    "g3r_original_owner_preclaim_root.esk",
    "g3r_original_owner_coordinator_root.esk",
    "g3r_trainer_g3_save_root.esk",
    "g3r_tr3_held_o2_snapshot_recheck_extension.esk",
    "g3r_held_o2_adopter_extension.esk",
    "g3r_held_o2_c2_claim_extension.esk",
    "c2_training_state_extension.esk",
    "o2_wave2_extension.esk",
):
    assert chain.count(name) == 1, (name, chain.count(name))
assert chain.index("c2_training_state_extension.esk") < chain.index(
    "g3r_held_o2_c2_claim_extension.esk"
)
assert chain.index("o2_wave2_extension.esk") < chain.index(
    "g3r_held_o2_c2_claim_extension.esk"
)
assert "(provide" not in UNION.read_text() + CLAIM


def check_claim(source):
    required = (
        "(eq? (vector-ref request 1) #f)",
        "(eq? (vector-ref request 2) #f)",
        "(c2-training-state-find shell)",
        "(eq? (vector-ref owner 1) p1)",
        "(eq? (vector-ref owner 2) o2-shell)",
        "(eq? (vector-ref owner 7) 'live)",
        "(eq? (vector-ref o2-state 6) (vector-ref owner 6))",
        "(eq? o2-state (o2-find-identity o2-shell o2-states))",
        "(g3r-held-o2-fail-stop)",
        "(vector-set! ledger 8 'claimed)",
    )
    for item in required:
        assert item in source, item
    assert source.index("(g3r-held-o2-fail-stop)") < source.index(
        "(vector-set! ledger 8 'claimed)"
    )


check_claim(CLAIM)
for item in (
    "(eq? (vector-ref request 2) #f)",
    "(c2-training-state-find shell)",
    "(eq? (vector-ref owner 2) o2-shell)",
    "(eq? (vector-ref o2-state 6) (vector-ref owner 6))",
    "(eq? o2-state (o2-find-identity o2-shell o2-states))",
    "(g3r-held-o2-fail-stop)",
):
    try:
        check_claim(CLAIM.replace(item, "#t"))
    except AssertionError:
        pass
    else:
        raise AssertionError(f"claim mutation escaped: {item}")

assert "(claim-progress (vector #f #f))" in COMPOSE
assert COMPOSE.index("(vector-set! claim-progress 1 #t)") < COMPOSE.index(
    "(c2-training-state-claim-o2! o2 owner-token)"
)
assert "(o2-state-claim-c2-rollback-or-live-internal!" in O2
print("G3R original-owner C2 claim union source PASS (single registry, 6 mutations)")
