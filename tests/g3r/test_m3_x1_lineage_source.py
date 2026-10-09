"""Private M3/X1 creation source controls; generated code remains unrun."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "native/m3t_transport_extension.esk"
X1 = ROOT / "native/x1_config_private.esk"
EXT = ROOT / "native/g3r_m3_x1_lineage_extension.esk"
PRIVATE_ROOT = ROOT / "native/g3r_m3_x1_lineage_root.esk"
MANIFEST = ROOT / "native/g3r_m3_x1_lineage_source_closure.txt"


def ordered(source: str, parts: list[str]) -> None:
    points = [source.find(part) for part in parts]
    if -1 in points or points != sorted(points):
        raise ValueError("creation or publication order differs")


def check(core: str, ext: str, root: str, manifest: str, x1: str) -> None:
    parse(core)
    parse(ext)
    parse(x1)
    if parse(root) != [["load", '"g3r_d2_lineage_root.esk"'],
                       ["load", '"g3r_m3_x1_lineage_extension.esk"']]:
        raise ValueError("private aggregate differs")
    expected = ["native/g3r_m3_x1_lineage_root.esk",
                *((ROOT / "native/g3r_d2_lineage_source_closure.txt")
                  .read_text().splitlines()),
                "native/g3r_m3_x1_lineage_extension.esk"]
    if manifest.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("exact source closure differs")
    if ("x1-resolved? x1-private-resolved-entry)" not in x1 or
            "(define (x1-private-resolved-entry value)" not in x1 or
            "(entry (x1-config-entry value))" not in x1 or
            "(eq? (car (cdr entry)) 'resolved)" not in x1):
        raise ValueError("registered resolved X1 entry authority differs")
    if "(define (m3t-public-model-create config state)\n  (m3t-model-create-core config state #f))" not in core:
        raise ValueError("ordinary public constructor differs")
    construction = core[core.index("(define (m3t-model-create-core"):
                        core.index("(define (m3t-public-model-create")]
    ordered(construction, ["(config-data (m3t-config-check config initial operation))",
                   "(m3t-model-lineage-preflight config initial config-data lineage operation)",
                   "(entry (m3t-stage 'model))",
                   "(m3t-native-owner-create (vector-ref initial 3))",
                   "(if lineage (vector-set! (vector-ref lineage 1) 2 'sealing))",
                   "(i2-construction-seal-internal! construction)",
                   "(vector-set! successor 2 'live)",
                   "(vector-set! entry 2 'live)",
                   "(vector-set! identity 0 root)",
                   "(vector-set! row 1 'live)",
                   "(vector-set! registry 0 next)",
                   "(vector-set! registry 1 #f)"])
    required_core = ["(eq? (vector-ref (vector-ref lineage 0) 1)",
                     "(eq? (vector-ref (vector-ref lineage 2) 2) config)",
                     "(x1-private-resolved-entry config)",
                     "(eq? (vector-ref (vector-ref lineage 2) 8) initial)",
                     "(string=? (vector-ref (vector-ref lineage 2) 5)",
                     "(string=? (vector-ref (vector-ref lineage 2) 6)",
                     "(if (not (vector-ref cleanup 1))", "(raise caught)"]
    if any(part not in core for part in required_core):
        raise ValueError("private preflight or original abort differs")
    ordered(ext, ["(g3r-x1-model-lineage-capacity",
                  "(x1-private-config-validate config)",
                  "(x1-resolved? config)",
                  "(identity (vector #f #f))",
                  "(row (vector #f 'pending config x1-entry #f",
                  "(next (cons row (vector-ref g3r-x1-model-lineage-registry 0)))",
                  "(vector-set! g3r-x1-model-lineage-registry 1 pending)",
                  "(m3t-model-create-core config initializer lineage)"])
    if ("(define g3r-x1-model-lineage-limit 64)" not in ext or
            ext.index("((= capacity 0)") > ext.index("((null? remaining)")):
        raise ValueError("retained rows are not bounded at capacity")
    needed = ["(eq? candidate row)", "(eq? (vector-ref candidate 2) config)",
              "(eq? (vector-ref candidate 3) x1-entry)",
              "(eq? (vector-ref candidate 4) (vector-ref identity 1))",
              "(eq? (vector-ref candidate 8) initial)",
              "(eq? (x1-private-resolved-entry config) x1-entry)",
              "(x1-resolved? config)", "(eq? original config)",
              "(eq? (vector-ref row 4) entry)",
              "(eq? (x1-private-resolved-entry original) x1-entry)",
              "(eq? (vector-ref row 8) (vector-ref entry 5))",
              "((vector-ref row 7) row)",
              "(m3t-entry model 'model operation #t)",
              "(string=? (vector-ref data 0) canonical)",
              "(string=? (vector-ref data 1) fingerprint)",
              "(if (vector-ref pending 2) (exit 134))",
              "(vector-set! g3r-x1-model-lineage-registry 1 #f)",
              "(raise caught)"]
    if any(part not in ext for part in needed):
        raise ValueError("identity, failure cut, or typed failure differs")
    # No new callback is accepted as the authority for M3 construction.
    if "(m3t-model-create-core config initializer lineage)" not in ext:
        raise ValueError("fixed constructor route differs")


class M3X1LineageSource(unittest.TestCase):
    def setUp(self) -> None:
        self.core = CORE.read_text()
        self.ext = EXT.read_text()
        self.root = PRIVATE_ROOT.read_text()
        self.manifest = MANIFEST.read_text()
        self.x1 = X1.read_text()

    def check(self, *, core=None, ext=None, root=None, manifest=None, x1=None) -> None:
        check(self.core if core is None else core,
              self.ext if ext is None else ext,
              self.root if root is None else root,
              self.manifest if manifest is None else manifest,
              self.x1 if x1 is None else x1)

    def test_exact_route(self) -> None:
        self.check()

    def test_public_constructor_cannot_mint_lineage(self) -> None:
        with self.assertRaises(ValueError):
            self.check(core=self.core.replace("m3t-model-create-core config state #f",
                                               "m3t-model-create-core config state lineage"))

    def test_preflight_and_seal_order(self) -> None:
        with self.assertRaises(ValueError):
            self.check(core=self.core.replace(
                "(lineage-checked\n           (m3t-model-lineage-preflight config initial config-data lineage operation))",
                "(lineage-checked #t)", 1))
        with self.assertRaises(ValueError):
            self.check(core=self.core.replace(
                "(if lineage (vector-set! (vector-ref lineage 1) 2 'sealing))", "#t"))

    def test_no_retroactive_or_alternate_x1(self) -> None:
        for before, after in [("(eq? original config)", "(equal? original config)"),
                              ("(x1-resolved? config)", "#t"),
                              ("(m3t-entry model 'model operation #t)",
                               "(m3t-entry model 'model operation #f)")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(ext=self.ext.replace(before, after))

    def test_copied_row_and_changed_model_denied(self) -> None:
        for before in ["(eq? candidate row)", "((vector-ref row 7) row)",
                       "(eq? (vector-ref row 4) entry)",
                       "(string=? (vector-ref data 0) canonical)"]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(ext=self.ext.replace(before, "#t"))

    def test_original_entry_and_initializer_denied_when_changed(self) -> None:
        for before in ["(eq? (vector-ref candidate 3) x1-entry)",
                       "(eq? (vector-ref candidate 8) initial)",
                       "(eq? (x1-private-resolved-entry original) x1-entry)",
                       "(eq? (vector-ref row 8) (vector-ref entry 5))"]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(ext=self.ext.replace(before, "#t"))
        with self.assertRaises(ValueError):
            self.check(x1=self.x1.replace("x1-resolved? x1-private-resolved-entry)",
                                            "x1-resolved?)"))

    def test_failure_cut_and_quota(self) -> None:
        for before, after in [("(if (vector-ref pending 2) (exit 134))", "#t"),
                              ("(raise caught)", "#t"),
                              ("(define g3r-x1-model-lineage-limit 64)",
                               "(define g3r-x1-model-lineage-limit 1000000)"),
                              ("((= capacity 0)", "((null? remaining)")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(ext=self.ext.replace(before, after))

    def test_missing_or_reordered_source_denied(self) -> None:
        with self.assertRaises(ValueError):
            self.check(manifest=self.manifest.replace(
                "native/g3r_m3_x1_lineage_extension.esk\n", ""))
        with self.assertRaises(ValueError):
            self.check(root=self.root.replace("g3r_d2_lineage_root.esk",
                                              "g3g_manual_package_root.esk"))


if __name__ == "__main__":
    unittest.main()
