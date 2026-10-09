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
    if ("(define m3t-model-lineage-registry (vector '() #f))" not in core or
            "(define g3r-x1-model-lineage-limit 64)" not in core or
            "(define g3r-x1-model-lineage-registry m3t-model-lineage-registry)" not in ext):
        raise ValueError("canonical registry or finite retained-row quota differs")
    if ("(define m3t-model-lineage-routes\n  (let ((secret" not in core or
            "(current (vector #f))" not in core or
            "(lambda (lineage)" not in core or
            "(eq? (vector-ref current 0) lineage)" not in core or
            "(eq? (vector-ref lineage 5) secret)" not in core or
            "(lambda (config initializer)" not in core):
        raise ValueError("lexical issuer or exact-current verifier differs")
    routes = [form for form in parse(core) if isinstance(form, list) and
              len(form) == 3 and form[0] == "define" and
              form[1] == "m3t-model-lineage-routes"]
    if (len(routes) != 1 or routes[0][2][0] != "let" or
            routes[0][2][2][0] != "vector" or
            len(routes[0][2][2]) != 3 or
            routes[0][2][2][1][0] != "lambda" or
            routes[0][2][2][2][0] != "lambda"):
        raise ValueError("secret or issuer route escaped lexical closure")
    construction = core[core.index("(define (m3t-model-create-core"):
                        core.index("(define (m3t-public-model-create")]
    ordered(construction, ["(config-data (m3t-config-check config initial operation))",
                           "(m3t-model-lineage-preflight config initial config-data lineage operation)",
                           "(entry (m3t-stage 'model))",
                           "(m3t-native-owner-create (vector-ref initial 3))",
                           "(if lineage (vector-set! (vector-ref lineage 1) 2 'building))",
                           "(if lineage\n                             (vector-set! (vector-ref lineage 1) 2 'aborted))",
                           "(if lineage (vector-set! (vector-ref lineage 1) 2 'sealing))",
                           "(i2-construction-seal-internal! construction)",
                           "(vector-set! successor 2 'live)",
                           "(vector-set! entry 2 'live)",
                           "(vector-set! identity 0 root)",
                           "(vector-set! row 1 'live)",
                           "(vector-set! registry 0 next)",
                           "(vector-set! registry 1 #f)"])
    required_core = ["(= (vector-length lineage) 6)",
                     "(eq? (vector-ref lineage 0) m3t-model-lineage-registry)",
                     "(eq? (vector-ref m3t-model-lineage-registry 1) lineage)",
                     "((vector-ref m3t-model-lineage-routes 0) lineage)",
                     "(eq? (vector-ref (vector-ref lineage 2) 2) config)",
                     "(x1-private-resolved-entry config)",
                     "(eq? (vector-ref (vector-ref lineage 2) 8) initial)",
                     "(if (not (vector-ref cleanup 1))", "(raise caught)"]
    if any(part not in core for part in required_core):
        raise ValueError("direct core forgery or original abort is possible")
    issuer = core[core.index("(define m3t-model-lineage-routes"):]
    ordered(issuer, ["(g3r-x1-model-lineage-capacity",
                     "(x1-private-config-validate config)",
                     "(x1-resolved? config)",
                     "(identity (vector #f #f))",
                     "(row (vector #f 'pending config x1-entry #f",
                     "(next (cons row (vector-ref m3t-model-lineage-registry 0)))",
                     "(lineage (vector m3t-model-lineage-registry pending row",
                     "(vector-set! m3t-model-lineage-registry 1 lineage)",
                     "(vector-set! current 0 lineage)",
                     "(m3t-model-create-core config initializer lineage)",
                     "(vector-set! current 0 #f)\n            model"])
    quota = core[core.index("(define (g3r-x1-model-lineage-capacity"):
                 core.index("(define (m3t-model-lineage-preflight")]
    if quota.index("((= capacity 0)") > quota.index("((null? remaining)"):
        raise ValueError("retained rows are not bounded at capacity")
    needed_issuer = ["(eq? candidate row)",
                     "(eq? (vector-ref candidate 2) config)",
                     "(eq? (vector-ref candidate 3) x1-entry)",
                     "(eq? (vector-ref candidate 8) initial)",
                     "(eq? (x1-private-resolved-entry config) x1-entry)",
                     "(eq? (vector-ref pending 2) 'building)",
                     "(eq? (vector-ref pending 2) 'sealing)",
                     "(exit 134)",
                     "(vector-set! current 0 #f)",
                     "(vector-set! m3t-model-lineage-registry 1 #f)",
                     "(raise caught)"]
    if any(part not in issuer for part in needed_issuer):
        raise ValueError("issuer identity, abort, or replay denial differs")
    if "(define (g3r-x1-model-create config initializer)\n  ((vector-ref m3t-model-lineage-routes 1) config initializer))" not in ext:
        raise ValueError("private route can bypass lexical issuer")
    required_ext = ["(m3t-entry model 'model operation #t)",
                    "(eq? original config)",
                    "(eq? (x1-private-resolved-entry original) x1-entry)",
                    "(eq? (vector-ref row 8) (vector-ref entry 5))",
                    "(eq? (vector-ref row 4) entry)",
                    "((vector-ref row 7) row)",
                    "(string=? (vector-ref data 0) canonical)",
                    "(string=? (vector-ref data 1) fingerprint)"]
    if any(part not in ext for part in required_ext):
        raise ValueError("original X1/M3 entry authentication differs")


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

    def test_public_cannot_mint_lineage(self) -> None:
        with self.assertRaises(ValueError):
            self.check(core=self.core.replace("m3t-model-create-core config state #f",
                                               "m3t-model-create-core config state lineage"))

    def test_direct_core_forged_or_foreign_pending_denied(self) -> None:
        for before in ["(eq? (vector-ref lineage 0) m3t-model-lineage-registry)",
                       "(eq? (vector-ref m3t-model-lineage-registry 1) lineage)",
                       "((vector-ref m3t-model-lineage-routes 0) lineage)",
                       "(eq? (vector-ref current 0) lineage)",
                       "(eq? (vector-ref lineage 5) secret)"]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(core=self.core.replace(before, "#t"))

    def test_no_retroactive_or_alternate_x1(self) -> None:
        for before, after in [("(eq? original config)", "(equal? original config)"),
                              ("(m3t-entry model 'model operation #t)",
                               "(m3t-entry model 'model operation #f)")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(ext=self.ext.replace(before, after))
        with self.assertRaises(ValueError):
            self.check(core=self.core.replace("(x1-resolved? config)", "#t"))

    def test_copied_row_or_changed_entry_denied(self) -> None:
        for before in ["(eq? candidate row)",
                       "(eq? (vector-ref candidate 8) initial)"]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(core=self.core.replace(before, "#t"))
        for before in ["((vector-ref row 7) row)",
                       "(eq? (x1-private-resolved-entry original) x1-entry)",
                       "(eq? (vector-ref row 8) (vector-ref entry 5))"]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(ext=self.ext.replace(before, "#t"))

    def test_failure_cut_and_quota(self) -> None:
        for before, after in [("(eq? (vector-ref pending 2) 'building)", "#f"),
                              ("(eq? (vector-ref pending 2) 'sealing)", "#f"),
                              ("(raise caught)", "#t"),
                              ("(define g3r-x1-model-lineage-limit 64)",
                               "(define g3r-x1-model-lineage-limit 1000000)"),
                              ("(vector-set! current 0 #f)", "#t")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(core=self.core.replace(before, after))
        with self.assertRaises(ValueError):
            self.check(core=self.core.replace("(lineage-checked\n           (m3t-model-lineage-preflight config initial config-data lineage operation))",
                                               "(lineage-checked #t)"))

    def test_missing_source_or_route_denied(self) -> None:
        with self.assertRaises(ValueError):
            self.check(manifest=self.manifest.replace(
                "native/g3r_m3_x1_lineage_extension.esk\n", ""))
        with self.assertRaises(ValueError):
            self.check(root=self.root.replace("g3r_d2_lineage_root.esk",
                                              "g3g_manual_package_root.esk"))
        with self.assertRaises(ValueError):
            self.check(ext=self.ext.replace("(vector-ref m3t-model-lineage-routes 1)",
                                            "m3t-model-create-core"))


if __name__ == "__main__":
    unittest.main()
