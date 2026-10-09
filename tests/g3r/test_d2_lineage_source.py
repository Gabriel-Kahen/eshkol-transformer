"""Private D2/T2 same-call source controls; compiled behavior remains unproved."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "native/g3r_d2_lineage_extension.esk"
PRIVATE_ROOT = ROOT / "native/g3r_d2_lineage_root.esk"
MANIFEST = ROOT / "native/g3r_d2_lineage_source_closure.txt"


def check_lineage(source: str, root: str, manifest: str) -> None:
    forms = parse(source)
    names = [form[1][0] for form in forms if isinstance(form, list)
             and len(form) > 2 and form[0] == "define"
             and isinstance(form[1], list)]
    if names != ["g3r-d2-lineage-check-capacity",
                 "g3r-d2-lineage-byte-core", "g3r-d2-lineage-open",
                 "g3r-d2-lineage-require", "g3r-d2-lineage-retire!"]:
        raise ValueError("private lineage entries differ")
    if parse(root) != [["load", '"cli3_generate_root.esk"'],
                       ["load", '"g3r_d2_lineage_extension.esk"']]:
        raise ValueError("same-aggregate load order differs")
    expected = ["native/g3r_d2_lineage_root.esk",
                *(ROOT / "native/cli3_generate_source_closure.txt").read_text().splitlines(),
                "native/g3r_d2_lineage_extension.esk"]
    if manifest.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("source closure differs")
    old_root = (ROOT / "native/cli3_generate_root.esk").read_text()
    if old_root.index('(load "cli3_generate_t2_bridge.esk")') > old_root.index(
            '(load "g3t_generator_constructor_extension.esk")'):
        raise ValueError("T2 bridge must precede G3 constructor")
    bridge = (ROOT / "native/cli3_generate_t2_bridge.esk").read_text()
    if "(define t1-private-entry t2-private-entry)" not in bridge:
        raise ValueError("T1 must use the same T2 registry")
    stages = ["(tokenizer-entry (g3r-d2-lineage-byte-core tokenizer operation))",
              "(row (vector #f 'pending tokenizer tokenizer-entry",
              "(next (cons row (vector-ref g3r-d2-lineage-registry 0)))",
              "(vector-set! g3r-d2-lineage-registry 1 pending)",
              "(d2-token-dataset-open config tokenizer)",
              "(vector-set! pending 2 shell)",
              "(d2-open-dataset-state shell operation)",
              "(vector-set! row 0 shell)",
              "(vector-set! row 4 state)",
              "(vector-set! row 7 witness)",
              "(vector-set! row 1 'live)",
              "(vector-set! g3r-d2-lineage-registry 0 next)",
              "(vector-set! g3r-d2-lineage-registry 1 #f)"]
    positions = [source.rfind(s) if s == stages[-1] else source.find(s)
                 for s in stages]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("same-call pending/publication order differs")
    required = ["(d2-token-dataset-close! (vector-ref pending 2))",
                "(guard (cleanup-failure (#t (exit 134)))",
                "(raise caught)",
                "(eq? candidate row)",
                "(eq? (vector-ref candidate 2) tokenizer)",
                "(eq? (vector-ref candidate 3) tokenizer-entry)",
                "(eq? (vector-ref candidate 4) state)",
                "(eq? original tokenizer)",
                "(eq? (vector-ref row 3) tokenizer-entry)",
                "(eq? (vector-ref row 4) state)",
                "((vector-ref row 7) row)",
                "(d2-open-dataset-state dataset operation)",
                "(d2-core-state-identity=? core decoded)",
                "(<= (vector-ref decoded 15) (vector-ref core 14))"]
    if any(s not in source for s in required):
        raise ValueError("identity, cursor, or failure cleanup differs")
    if "(vector-set! g3r-d2-lineage-registry 0 next)" in source[:positions[4]]:
        raise ValueError("dataset is published before D2 open")
    if ("(define g3r-d2-lineage-maximum-rows 64)" not in source or
            source.index("(g3r-d2-lineage-check-capacity\n") > positions[4] or
            "(= capacity 0)" not in source):
        raise ValueError("ordinary-close retention is unbounded")
    retire = source[source.index("(define (g3r-d2-lineage-retire!") :]
    retire_order = ["(next (append (reverse prefix) (cdr rows)))",
                    "(d2-token-dataset-close! dataset)",
                    "(vector-set! g3r-d2-lineage-registry 0 next)",
                    "(vector-set! row 2 #f)",
                    "(vector-set! row 7 #f)"]
    points = [retire.find(item) for item in retire_order]
    if -1 in points or points != sorted(points):
        raise ValueError("retirement publication or cleanup order differs")
    d2 = (ROOT / "internal/d2/lib/d2_dataset.esk").read_text()
    native = (ROOT / "native/d2_native.c").read_text()
    open_body = d2[d2.index("(define (d2-token-dataset-open config tokenizer)"):
                   d2.index("(define (d2-token-dataset-next-batch dataset)")]
    if ("(d2-make-dataset-shell state)" not in open_body or
            "(d2-native-dataset-open shell shuffle-count)" not in open_body or
            "(define (d2-make-dataset-shell state)" not in d2 or
            "et_d2_dataset_open_v1" not in native):
        raise ValueError("reviewed fixed D2 constructor path differs")


class D2LineageSource(unittest.TestCase):
    def setUp(self) -> None:
        self.source = EXT.read_text()
        self.root = PRIVATE_ROOT.read_text()
        self.manifest = MANIFEST.read_text()

    def check(self, *, source=None, root=None, manifest=None) -> None:
        check_lineage(source if source is not None else self.source,
                      root if root is not None else self.root,
                      manifest if manifest is not None else self.manifest)

    def test_exact_route(self) -> None:
        self.check()

    def test_reordered_aggregate_denied(self) -> None:
        root = self.root.replace('cli3_generate_root.esk',
                                 'g3g_manual_package_root.esk')
        with self.assertRaises(ValueError):
            self.check(root=root)

    def test_missing_dependency_denied(self) -> None:
        with self.assertRaises(ValueError):
            self.check(manifest=self.manifest.replace(
                'native/cli3_generate_t2_bridge.esk\n', ''))

    def test_premature_publication_denied(self) -> None:
        start = '(vector-set! g3r-d2-lineage-registry 0 next)'
        early = '(d2-token-dataset-open config tokenizer)'
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(start, '(begin ' + start + ')', 1)
                       .replace(early, start + '\n          ' + early, 1))

    def test_equal_byte_alternate_shell_denied(self) -> None:
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace('(eq? original tokenizer)',
                                                  '(equal? original tokenizer)'))

    def test_forged_companion_denied(self) -> None:
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                '((vector-ref row 7) row)', '#t'))
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace('(eq? candidate row)', '#t'))

    def test_close_failure_cannot_be_swallowed(self) -> None:
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace('(exit 134)', '#f'))

    def test_closed_or_changed_cursor_denied(self) -> None:
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                '(d2-open-dataset-state dataset operation)',
                '(d2-dataset-state dataset operation)'))
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                '(d2-core-state-identity=? core decoded)', '#t'))

    def test_ordinary_close_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                '(define g3r-d2-lineage-maximum-rows 64)',
                '(define g3r-d2-lineage-maximum-rows 999999)'))

    def test_retire_only_after_real_close(self) -> None:
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                '(d2-token-dataset-close! dataset)', '#t'))
        with self.assertRaises(ValueError):
            self.check(source=self.source.replace(
                '(next (append (reverse prefix) (cdr rows)))',
                '(next (cdr rows))'))


if __name__ == "__main__":
    unittest.main()
