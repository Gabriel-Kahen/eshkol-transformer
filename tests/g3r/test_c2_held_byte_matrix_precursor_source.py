"""Source gate for the noninstalled authentic-owner C2 matrix precursor."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/g3r/c2_held_byte_matrix_precursor.esk"
SOURCE = FIXTURE.read_text()


def balanced(text: str) -> bool:
    depth = 0
    quoted = escaped = comment = False
    for char in text:
        if comment:
            if char == "\n":
                comment = False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == ";":
            comment = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted


def check(text: str) -> None:
    assert balanced(text)
    assert '(load "g3r_c2_held_transaction_root.esk")' in text
    assert "g3r-c2-held-transaction-internal" in text
    assert "(define (fixture)" in text
    for needle in (
        "(t2-wave2-tokenizer-byte resolved)",
        "(d2-token-dataset-open (dataset-config corpus) tokenizer)",
        "(m3t-public-model-create",
        "(o2-public-optimizer-create",
        "(g3t-generator-create model tokenizer (policy))",
        "(module-train! model)",
        "(tr3-lease-create-internal",
        "(module-state-dict-held-clone-internal (vector-ref f 2))",
        "(o2-public-optimizer-state (vector-ref f 3))",
        "(c2-training-state-compose-internal request)",
        "(c2-training-state-borrow-begin-internal",
        "(state-dict-c2-owned-tensor-borrow-begin-internal",
        "(i2-checkpoint-encode",
        "(o2-state-copy-moment-bits-c2-owned-internal",
        "(bytes=? (vector-ref (vector-ref a 2) 0)",
        "(bytes=? (vector-ref (vector-ref a 3) 0)",
        "(raw-for-shell 'g3r-c2-held-transaction-internal",
        "(eq? (vector-ref before 0) (vector-ref raw 7))",
        "(same-path-objects? (vector-ref before 1) (vector-ref f 4))",
        "(define twin (fixture))",
        "(eq? (vector-ref held-owner 5) 'held)",
        "(= (vector-length request) 6)",
        "(not (vector-ref request 5))",
        "(set! g3r-c2-held-compose-call",
        "(set! g3r-held-o2-adopter-release!",
        "(eq? ledger release-ledger)",
        "(= release-calls 2)",
        "(eq? (vector-ref held-owner 5) 'dead)",
        "(image=? reference-bytes (owner-image held-shell))",
        "(image=? twin-bytes (owner-image shell))",
    ):
        assert needle in text, needle
    assert text.index("(check \"six-slot request before claim\"") < text.index(
        "(m3t-fail 'internal 'g3r-c2-held-transaction-internal"
    )
    assert text.index("(set! g3r-c2-held-compose-call compose-original)") < text.index(
        "(define held-shell"
    )
    assert text.index("(set! g3r-held-o2-adopter-release! release-original)") < text.index(
        "(define held-shell"
    )
    assert text.index("(check \"original 42 parameter") < text.index(
        "(check \"held shell releases\""
    )
    for banned in ("checkpoint-save", "save-receipt", "file-write", "file-open"):
        assert banned not in text


check(SOURCE)
mutations = (
    ('(load "g3r_c2_held_transaction_root.esk")', ""),
    ("(module-state-dict-held-clone-internal (vector-ref f 2))",
     "(module-state-dict (vector-ref f 2))"),
    ("(o2-public-optimizer-state (vector-ref f 3))", "#f"),
    ("(bytes=? (vector-ref (vector-ref a 2) 0)", "(equal? (vector-ref (vector-ref a 2) 0)"),
    ("(eq? (vector-ref before 0) (vector-ref raw 7))", "#t"),
    ("(eq? (vector-ref held-owner 5) 'held)", "#t"),
    ("(= (vector-length request) 6)", "#t"),
    ("(eq? ledger release-ledger)", "#t"),
    ("(= release-calls 2)", "#t"),
    ("(image=? reference-bytes (owner-image held-shell))", "#t"),
    ("(image=? twin-bytes (owner-image shell))", "#t"),
)
for before, after in mutations:
    assert before in SOURCE, before
    try:
        check(SOURCE.replace(before, after))
    except AssertionError:
        continue
    raise AssertionError(f"matrix mutation escaped source gate: {before}")

print(f"C2 held byte matrix precursor source PASS ({len(mutations)} mutations)")
