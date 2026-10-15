"""Keep the preclaim root separate from the C2 transfer successor."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
root = (ROOT / "native/g3r_original_owner_preclaim_root.esk").read_text()
adopter = (ROOT / "native/g3r_held_o2_adopter_extension.esk").read_text()
c2 = (ROOT / "native/c2_training_state_extension.esk").read_text()
o2 = (ROOT / "native/o2_wave2_extension.esk").read_text()

assert root.count('(load "g3r_original_owner_coordinator_root.esk")') == 1
assert root.count('(load "g3r_held_o2_adopter_extension.esk")') == 1
assert root.index("g3r_original_owner_coordinator_root.esk") < root.index(
    "g3r_held_o2_adopter_extension.esk"
)
assert "(provide" not in root + adopter
assert "(define (g3r-held-o2-adopt!" in adopter
assert "(define (g3r-held-o2-adopter-release!" in adopter
assert "(define (g3r-held-o2-adopter-c2-claim!" not in adopter
assert "c2-training-state-compose-internal" not in root + adopter

# The C2 successor has the reviewed two-child rollback, but this root still
# stops at preclaim enrollment and never transfers a child into C2.
compose = c2.split("(define (c2-training-state-compose-internal request)", 1)[1]
compose = compose.split("(define (c2-training-state-borrow-activation-rollback!", 1)[0]
assert "(claim-progress (vector #f #f))" in compose
assert "(c2-training-state-claim-o2! o2 owner-token)" in compose
assert "c2-training-state-rollback-o2-or-live!" in compose
assert "o2-state-claim-c2-rollback-or-live-internal!" in o2

print("G3R original-owner preclaim boundary PASS (C2 transfer separate)")
