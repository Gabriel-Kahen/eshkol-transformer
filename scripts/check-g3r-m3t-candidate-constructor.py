#!/usr/bin/env python3
"""Pin the narrow candidate constructor ownership and seal ordering."""

from hashlib import sha256
from pathlib import Path
import re


root = Path(__file__).resolve().parents[1]


def check_fixture_delimiters(path):
    """Catch unclosed lists before the expensive compiled gate."""
    stack = []
    string = escaped = comment = False
    for line, row in enumerate(path.read_text().splitlines(), 1):
        comment = False
        for column, char in enumerate(row, 1):
            if comment:
                continue
            if string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    string = False
            elif char == ";":
                comment = True
            elif char == '"':
                string = True
            elif char == "(":
                stack.append((line, column))
            elif char == ")":
                assert stack, f"{path}:{line}:{column}: unexpected close"
                stack.pop()
    assert not string, f"{path}: unterminated string"
    assert not stack, f"{path}:{stack[-1][0]}:{stack[-1][1]}: unclosed list"


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
preflight = section("g3r-candidate-graph-identity-preflight-internal!")
assert source.count("et_m3t_private_candidate_binding_preflight_v1") == 1
for fragment in (
    "(g3r-candidate-ledger-record ledger operation)",
    "(m3t-entry model 'model operation #t)",
    "(i2-candidate-construction-preflight-internal! cell)",
    "(m3t-config-check",
    "(module-parameters model)",
    "(parameter-handle-path handle)",
    "(parameter-tree-handle",
    "(parameter-tree-tie-groups tree)",
    "(g3r-m3t-native-binding-preflight owner index handle)",
    "(m3t-native-owner-successor owner)",
):
    assert fragment in preflight, f"graph identity leaf missing {fragment}"
assert "(vector-set!" not in preflight, "graph identity leaf writes state"
assert not re.search(r"(?:revoke|retire|release|abort|commit)-internal!", preflight)
native = (root / "src/eshkol_transformer/m3t_transport.c").read_text()
header = (root / "src/eshkol_transformer/m3t_transport.h").read_text()
binding_start = native.index("int64_t et_m3t_private_candidate_binding_preflight_v1(")
binding_end = native.index("\n#endif", binding_start)
binding = native[binding_start:binding_end]
assert "#ifdef ET_G3R_CANDIDATE_RETIRE_PRIVATE" in native[:binding_start]
assert "#ifdef ET_G3R_CANDIDATE_RETIRE_PRIVATE" in header
assert binding.index("admit(candidate, OWNER, 0)") < binding.index(
    "owner_roster_preflight(o)"
) < binding.index("o->handles[index] != exact_handle")
assert "return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_TOPOLOGY)" in binding
assert not re.search(r"\b(?:memcpy|malloc|calloc|free)\s*\(", binding)
roster_start = native.index("static int64_t owner_roster_preflight(owner *o)")
roster_end = native.index("\n#endif", roster_start)
roster = native[roster_start:roster_end]
assert "et_f32_parameter_validate_identity_v1(" in roster
assert "o->p[i] == o->p[j] || o->handles[i] == o->handles[j]" in roster
assert native.count("owner_roster_preflight(o)") == 2
runner = (root / "scripts/test-g3r-m3t-candidate-constructor.sh").read_text()
assert "-DET_G3R_CANDIDATE_RETIRE_PRIVATE=1" in runner
assert "2a8521181277d65c95875e3478ad3e68031c496e87b3be6e2c7ac56431ea324c" in runner
assert "m3t_candidate_binding_preflight_native_test.c" in runner
fixture = (root / "tests/g3r/m3t_candidate_constructor_test.esk").read_text()
check_fixture_delimiters(root / "tests/g3r/m3t_candidate_constructor_test.esk")
assert '"candidate exact head/embedding handle tie"' in fixture
assert '"ordinary exact tie parity"' in fixture
assert '"ordinary and candidate fourteen paths agree"' in fixture
assert fixture.count("(parameter-tree-handle") >= 5
assert "(parameter-tree-tie-groups candidate-tree)" in fixture
assert '"foreign native owner rejects candidate P1 shell"' in fixture
native_fixture = (root / "tests/g3r/m3t_candidate_binding_preflight_native_test.c").read_text()
for fragment in (
    "exact->handles[0] = exact->handles[1]",
    "exact->p[0] = exact->p[1]",
    "et_m3t_private_owner_abort_v1(stale)",
    "exact, 0, &handles[14]",
):
    assert fragment in native_fixture, f"native binding negative missing {fragment}"
for negative in (
    '"foreign graph ledger"',
    '"ordinary model is not this graph"',
    '"changed model root"',
    '"changed native owner"',
    '"stale successor"',
    '"changed config fingerprint"',
    '"changed I2 count"',
    '"changed P1 roster"',
    '"changed canonical handle"',
):
    assert negative in fixture, f"graph identity negative missing {negative}"
feature_off = (root / "tests/g3r/m3t_candidate_feature_off_test.esk").read_text()
check_fixture_delimiters(root / "tests/g3r/m3t_candidate_feature_off_test.esk")
assert "(parameter-handle-path (vector-ref handles index))" in feature_off
assert "(vector-ref m3t-paths index)" in feature_off
print("G3-R M3T candidate source contract PASS")
