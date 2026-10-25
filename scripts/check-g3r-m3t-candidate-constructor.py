#!/usr/bin/env python3
"""Pin the narrow candidate constructor ownership and seal ordering."""

from hashlib import sha256
from pathlib import Path
import re


root = Path(__file__).resolve().parents[1]
source = (root / "native/g3r_m3t_candidate_constructor_extension.esk").read_text()
ordinary = root / "native/m3t_transport_extension.esk"
assert sha256(ordinary.read_bytes()).hexdigest() == (
    "71247e722d72764ce1b3400cb3f818ba12c3754dd11a6c742ffcec63942d5815"
), "ordinary M3T source changed"


def section(name):
    start = source.index(f"(define ({name}")
    end = source.find("\n(define ", start + 1)
    return source[start:] if end < 0 else source[start:end]


begin = section("g3r-candidate-ledger-begin-internal!")
create = section("g3r-candidate-model-create-internal!")
assert begin.index("(m3t-stage 'model)") < begin.index(
    "(vector-set! g3r-candidate-ledger-registry 0 next)"
)
assert begin.index("(vector #f #f (vector #f #f #f #f) #f)") < begin.index(
    "(vector-set! g3r-candidate-ledger-registry 0 next)"
)
assert begin.index("(vector-set! g3r-candidate-ledger-registry 0 next)") < begin.index(
    "(car (vector-ref g3r-candidate-ledger-registry 0)) 0)"
)
assert create.index("(guard (caught") < create.index("(m3t-native-owner-create")
assert create.index("(set! native created)") < create.index(
    "(vector-set! record 8 created)"
) < create.index("(vector-set! entry 3 created)") < create.index(
    "(i2-candidate-construction-begin-internal! cell)"
)
assert create.index("(i2-candidate-construction-abort-internal! cell)") < create.index(
    "(m3t-native-owner-abort native)"
)
assert "(vector-ref record 9)" in create[: create.index("(m3t-native-owner-create")]
assert "(vector-ref record 9)" not in create[create.index("(guard (caught"):]
assert create.index("(vector-set! (vector-ref construction 8) 2") < create.index(
    "(i2-module-register-parameter-internal!"
)
for fragment in (
    "(vector-ref m3t-paths index)",
    "(m3t-child-parent root path cache)",
    "(i2-module-register-parameter-internal!",
    "(m3t-native-owner-bind created index handle)",
    "(i2-module-register-tied-parameter-internal!",
    "(m3t-validate-schedule schedule (vector-ref entry 8)",
    "(m3t-native-owner-initialize created index)",
    "(m3t-native-owner-successor created)",
):
    assert fragment in create, f"fixed schedule missing {fragment}"
assert create.index("(vector-set! successor 3 next)") < create.index(
    "(vector-set! record 11 #t)"
) < create.index("(i2-candidate-construction-seal-internal! cell)")
assert "(if (vector-ref record 11) (exit 134))" in create
tail = create.split("(i2-candidate-construction-seal-internal! cell)", 1)[1]
assert re.findall(r"\(vector-set! ([\w-]+) (\d+) ([^)]+)\)", tail) == [
    ("successor", "2", "'live"),
    ("entry", "2", "'live"),
    ("record", "1", "'sealed"),
], "postseal tail changed"
assert "(exit 134)" in tail and "(m3t-native-owner-seal created)" in tail
assert "(m3t-native-initializer-seal" in tail
assert not re.search(r"\((?:cons|vector|make-vector|list)(?:\s|\))", tail)
fixture = (root / "tests/g3r/m3t_candidate_constructor_test.esk").read_text()
assert '"candidate exact head/embedding handle tie"' in fixture
assert '"ordinary exact tie parity"' in fixture
assert fixture.count("(parameter-tree-handle") >= 5
assert "(parameter-tree-tie-groups candidate-tree)" in fixture
print("G3-R M3T candidate source contract PASS")
