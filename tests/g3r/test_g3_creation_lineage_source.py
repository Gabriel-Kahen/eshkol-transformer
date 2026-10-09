"""Bounded private G3 creation source controls; no generated package claim."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "native/g3r_g3_creation_lineage_extension.esk"
AGGREGATE = ROOT / "native/g3r_g3_creation_lineage_root.esk"
MANIFEST = ROOT / "native/g3r_g3_creation_lineage_source_closure.txt"


def check(source: str, aggregate: str, manifest: str) -> None:
    forms = parse(source)
    if parse(aggregate) != [
        ["load", '"g3r_m3_x1_lineage_root.esk"'],
        ["load", '"g3r_g3_creation_lineage_extension.esk"'],
    ]:
        raise ValueError("private aggregate load order changed")
    expected = [
        "native/g3r_g3_creation_lineage_root.esk",
        *(ROOT / "native/g3r_m3_x1_lineage_source_closure.txt").read_text().splitlines(),
        "native/g3r_g3_creation_lineage_extension.esk",
    ]
    if manifest.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("recursive source closure changed")
    externs = [form for form in forms if isinstance(form, list) and form[:1] == ["extern"]]
    actual = [form[1:3] + [form[-1]] for form in externs]
    if actual != [
        ["ptr", "g3r-creation-o2-begin", "et_g3r_private_o2_creation_begin_v1"],
        ["i64", "g3r-creation-o2-check", "et_g3r_private_o2_creation_check_v1"],
        ["i64", "g3r-creation-o2-end", "et_g3r_private_o2_creation_end_v1"],
        ["i64", "g3r-creation-o2-error-word", "et_g3r_private_o2_creation_error_word_v1"],
    ]:
        raise ValueError("scalar native bridge ABI changed")
    if "(define g3r-g3-creation-limit 64)" not in source:
        raise ValueError("finite retained-row bound changed")
    quota = source[source.index("(define (g3r-g3-creation-capacity"):
                   source.index("(define (g3r-g3-creation-copy-error!")]
    if quota.index("((= left 0)") > quota.index("((null? tail)"):
        raise ValueError("65th row is admitted")
    route = next((f for f in forms if isinstance(f, list) and f[:2] ==
                  ["define", "g3r-g3-creation-route"]), None)
    if (route is None or route[2][0] != "let*" or
            [x[0] for x in route[2][1]] != ["registry", "create", "required"] or
            route[2][1][0] != ["registry", ["vector", ["quote", []], "#f"]]):
        raise ValueError("issuer/completed list no longer lexical")
    fixed = [
        "(g3r-g3-creation-route 'create model original-x1 tokenizer",
        "(g3r-g3-creation-route 'require generator model original-x1 tokenizer",
        "(g3t-generator-create model tokenizer config)",
        "(g3r-x1-model-require model original-x1)",
        "(g3r-d2-lineage-require dataset tokenizer)",
        "(tr3-lease-d2-idle! dataset-state operation)",
        "(eq? (t1-private-entry tokenizer) tokenizer-entry)",
        "(eq? (vector-ref optimizer-entry 5)",
        "(eq? (car assignments)",
        "(vector-ref saved-handles index)",
        "(= (o2-optimizer-completed-updates-internal optimizer)",
        "(eq? (g3t-generator-find shell",
        "(eq? (vector-ref entry 3) (vector-ref roots 13))",
        "(eq? (vector-ref entry 7) policy)",
        "(equal? policy expected-policy)",
    ]
    if any(item not in source for item in fixed):
        raise ValueError("original graph or fixed constructor check changed")
    stages = [
        "(row (vector #f 'pending #f))",
        "(roots (vector model original-x1 tokenizer dataset optimizer",
        "(record (cons row roots))",
        "(next (cons record (vector-ref registry 0)))",
        "(pending (vector row next #f #f))",
        "(first (vector #f 0 0 0 0))",
        "(vector-set! registry 1 pending)",
        "(g3t-generator-create model tokenizer config)",
        "(vector-set! pending 2 shell)",
        "(g3r-creation-o2-begin",
        "(vector-set! pending 3 hold)",
        "(g3r-creation-o2-check hold)",
        "(g3r-creation-o2-end hold)",
        "(vector-set! pending 3 #f)",
        "(g3r-g3-creation-check-roots roots shell entry)",
        "(vector-set! row 0 shell)",
        "(vector-set! registry 0 next)",
        "(vector-set! registry 1 #f)",
    ]
    create = source[source.index("(create\n"):source.index("(required\n")]
    points = [create.find(stage) for stage in stages]
    # Cleanup has earlier copies of these writes; use the success-tail sites.
    points[13] = create.rfind(stages[13])
    points[14] = create.rfind(stages[14])
    points[17] = create.rfind(stages[17])
    if -1 in points or points != sorted(points):
        raise ValueError("preallocation, native interval, or publication moved")
    if create.rfind("(g3r-g3-creation-check-roots roots shell entry)") > \
            create.index("(vector-set! row 0 shell)"):
        raise ValueError("fallible recheck after publication")
    cleanup = [
        "(g3r-g3-creation-copy-error! first hold)",
        "(g3r-g3-creation-copy-error! first",
        "(g3r-creation-o2-end",
        "(guard (close-failure (#t (exit 134)))",
        "(g3t-generator-close! (vector-ref pending 2))",
        "(vector-set! registry 1 #f)",
        "(g3r-g3-creation-raise-error first)",
    ]
    if any(part not in create for part in cleanup):
        raise ValueError("first-error or unpublished cleanup changed")
    copy = source[source.index("(define (g3r-g3-creation-copy-error!"):
                  source.index("(define (g3r-g3-creation-raise-error")]
    for index in range(4):
        if f"(g3r-creation-o2-error-word hold {index})" not in copy:
            raise ValueError("native error tuple was truncated")
    # The public three-argument constructor and its registry remain byte-for-byte
    # upstream; this test only asserts the new wrapper never changes them.
    prior = (ROOT / "native/g3t_generator_constructor_extension.esk").read_text()
    if "(define (g3t-generator-create model tokenizer config)" not in prior:
        raise ValueError("upstream G3 constructor changed")


class G3CreationLineageSource(unittest.TestCase):
    def setUp(self):
        self.source = EXT.read_text()
        self.aggregate = AGGREGATE.read_text()
        self.manifest = MANIFEST.read_text()

    def verify(self, **replacement):
        values = {"source": self.source, "aggregate": self.aggregate,
                  "manifest": self.manifest}
        values.update(replacement)
        check(**values)

    def test_exact_source(self):
        self.verify()

    def test_issuer_and_manifest_mutants(self):
        changes = [
            ("(registry (vector '() #f))", "(registry #f)"),
            ("(g3t-generator-create model tokenizer config)",
             "(g3t-generator-create model tokenizer #f)"),
            ("(g3r-x1-model-require model original-x1)",
             "(g3r-x1-model-require model model)"),
            ("(g3r-d2-lineage-require dataset tokenizer)", "#t"),
            ("(eq? (t1-private-entry tokenizer) tokenizer-entry)", "#t"),
            ("(tr3-lease-d2-idle! dataset-state operation)", "#t"),
            ("(g3r-creation-o2-check hold)", "0"),
            ("(g3r-creation-o2-end hold)", "0"),
            ("(vector-set! registry 0 next)", "(vector-set! registry 0 '())"),
            ("(guard (close-failure (#t (exit 134)))", "(guard (close-failure (#t #f))"),
        ]
        for before, after in changes:
            self.assertIn(before, self.source)
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(before, after))
        with self.assertRaises(ValueError):
            self.verify(manifest=self.manifest.replace(
                "native/g3r_g3_creation_lineage_extension.esk", ""))

    def test_first_error_and_identity_mutants(self):
        for before, after in [
            ("(g3r-creation-o2-error-word hold 2)", "0"),
            ("(eq? (vector-ref entry 3) (vector-ref roots 13))", "#t"),
            ("(eq? (vector-ref entry 7) policy)", "#t"),
            ("(equal? policy expected-policy)", "#t"),
            ("(vector-ref saved-handles index)", "(vector-ref handles index)"),
            ("(= (o2-optimizer-completed-updates-internal optimizer)", "#t"),
        ]:
            self.assertIn(before, self.source)
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(before, after))


if __name__ == "__main__":
    unittest.main()
