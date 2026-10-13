"""Private M3/X1 authority and state controls; generated code remains unrun."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "native/m3t_transport_extension.esk"
X1 = ROOT / "native/x1_config_private.esk"
EXT = ROOT / "native/g3r_m3_x1_lineage_extension.esk"
PRIVATE_ROOT = ROOT / "native/g3r_m3_x1_lineage_root.esk"
MANIFEST = ROOT / "native/g3r_m3_x1_lineage_source_closure.txt"


def forms_named(source: str, name: str):
    return [f for f in parse(source) if isinstance(f, list) and len(f) > 2 and
            f[0] == "define" and (f[1] == name or
                                  isinstance(f[1], list) and f[1][0] == name)]


def lineage_predicate(source: str):
    form, = forms_named(source, "m3t-model-lineage-preflight")
    return form[2][2][1][1]


def descendant(expr, tag):
    if isinstance(expr, list):
        if expr and expr[0] == tag:
            yield expr
        for child in expr:
            yield from descendant(child, tag)


def eval_pure(expr, env):
    """Interpret only the reviewed identity predicate, without M3/native calls."""
    if isinstance(expr, str):
        if expr in env:
            return env[expr]
        if expr.isdecimal():
            return int(expr)
        raise ValueError(f"unexpected source atom: {expr}")
    op, *args = expr
    if op == "and":
        return all(eval_pure(a, env) for a in args)
    if op == "quote":
        return args[0]
    if op == "vector?":
        return isinstance(eval_pure(args[0], env), list)
    if op == "pair?":
        return isinstance(eval_pure(args[0], env), tuple)
    if op == "procedure?":
        return callable(eval_pure(args[0], env))
    if op == "vector-ref":
        return eval_pure(args[0], env)[eval_pure(args[1], env)]
    if op == "vector-length":
        return len(eval_pure(args[0], env))
    if op == "car":
        return eval_pure(args[0], env)[0]
    if op == "cdr":
        return eval_pure(args[0], env)[1]
    if op == "x1-private-resolved-entry":
        assert eval_pure(args[0], env) is env["config"]
        return env["x1-entry"]
    if op in {"x1-private-config-canonical", "x1-private-config-fingerprint"}:
        assert eval_pure(args[0], env) is env["config"]
        return env["canonical" if op.endswith("canonical") else "fingerprint"]
    if op in {"=", "string=?"}:
        return eval_pure(args[0], env) == eval_pure(args[1], env)
    if op == "eq?":
        a, b = (eval_pure(x, env) for x in args)
        return a is b or isinstance(a, str) and a == b
    raise ValueError(f"unexpected source predicate: {op}")


def issued_state():
    config, initial, x1_entry, secret = object(), object(), object(), object()
    canonical, fingerprint = "canonical", "fingerprint"
    snapshot = [None, config, x1_entry, None, initial, canonical, fingerprint]
    row = [None, "pending", config, x1_entry, None,
           canonical, fingerprint, initial]
    record = (row, snapshot)
    next_rows = (record, None)
    pending = [row, next_rows, "prestage"]
    registry = [None, None]
    lineage = [registry, pending, row, next_rows, snapshot, secret]
    registry[1] = lineage
    return {"config": config, "initial": initial, "x1-entry": x1_entry,
            "secret": secret, "current": [lineage], "registry": registry,
            "config-data": [canonical, fingerprint], "lineage": lineage}


def completed_state():
    state = issued_state()
    model, entry = object(), [None] * 8
    canonical, fingerprint = state["config-data"]
    entry[5], entry[7] = state["initial"], [canonical, fingerprint]
    row, snapshot = state["lineage"][2], state["lineage"][4]
    row[0], row[1], row[4] = model, "live", entry
    snapshot[0], snapshot[3] = model, entry
    state.update(model=model, entry=entry, original=state["config"],
                 row=row, snapshot=snapshot, data=entry[7],
                 canonical=canonical, fingerprint=fingerprint)
    return state


def completed_predicate(source):
    route, = forms_named(source, "m3t-model-create-route")
    binding = route[2][1][-1][1]
    matches = [a for a in descendant(binding, "and") if
               ["eq?", "original", "config"] in a]
    assert len(matches) == 1
    return matches[0]


def check(core: str, ext: str, root: str, manifest: str, x1: str):
    for source in (core, ext, x1):
        parse(source)
    if parse(root) != [["load", '"g3r_d2_lineage_root.esk"'],
                       ["load", '"g3r_m3_x1_lineage_extension.esk"']]:
        raise ValueError("private aggregate changed")
    expected = ["native/g3r_m3_x1_lineage_root.esk",
                *(ROOT / "native/g3r_d2_lineage_source_closure.txt").read_text().splitlines(),
                "native/g3r_m3_x1_lineage_extension.esk"]
    if manifest.splitlines() != expected or len(set(expected)) != len(expected):
        raise ValueError("source closure changed")
    if ("x1-resolved? x1-private-resolved-entry)" not in x1 or
            "(entry (x1-config-entry value))" not in x1 or
            "(eq? (car (cdr entry)) 'resolved)" not in x1):
        raise ValueError("X1 registered entry accessor changed")
    if forms_named(core, "registry") or forms_named(core, "m3t-model-lineage-registry") or \
            forms_named(core, "m3t-model-create-core") or \
            "g3r-x1-model-lineage-registry" in ext:
        raise ValueError("authority escaped lexical scope")
    route, = forms_named(core, "m3t-model-create-route")
    if route[2][0] != "let*" or [b[0] for b in route[2][1]] != \
            ["secret", "current", "registry", "core", "private", "required"] or \
            route[2][1][2] != ["registry", ["vector", ["quote", []], "#f"]] or \
            route[2][2][0] != "lambda":
        raise ValueError("core, registry, or issuer can be replaced")
    if ("(define (m3t-public-model-create config state)\n"
            "  (m3t-model-create-route 'public config state))" not in core or
            "((eq? mode 'public) (core config state #f))" not in core or
            "((eq? mode 'private) (private config state))" not in core or
            "((eq? mode 'require) (required config state))" not in core or
            "(m3t-model-create-route 'private config initializer)" not in ext or
            "(m3t-model-create-route 'require model config)" not in ext):
        raise ValueError("fixed public/private/require route changed")
    pre = lineage_predicate(core)
    if not eval_pure(pre, issued_state()):
        raise ValueError("genuine issued lineage cannot enter M3")
    pre_text = core[core.index("(define (m3t-model-lineage-preflight"):route_start(core)]
    for part in ("(eq? (vector-ref lineage 0) registry)",
                 "(eq? (vector-ref registry 1) lineage)",
                 "(eq? (vector-ref current 0) lineage)",
                 "(eq? (vector-ref lineage 5) secret)",
                 "(eq? (car (car (vector-ref lineage 3)))",
                 "(eq? (cdr (car (vector-ref lineage 3)))",
                 "(eq? (vector-ref (vector-ref lineage 4) 1) config)"):
        if part not in pre_text:
            raise ValueError("pending/secret/snapshot binding changed")
    ordered = ["(m3t-model-lineage-preflight config initial config-data lineage operation secret current registry)",
               "(entry (m3t-stage 'model))",
               "(if lineage (vector-set! (vector-ref lineage 1) 2 'building))",
               "(if lineage (vector-set! (vector-ref lineage 1) 2 'sealing))",
               "(i2-construction-seal-internal! construction)",
               "(vector-set! row 1 'live)", "(vector-set! registry 0 next)",
               "(vector-set! registry 1 #f)"]
    points = [core.find(p, route_start(core)) for p in ordered]
    if -1 in points or points != sorted(points):
        raise ValueError("preflight or genuine publication order changed")
    for part in ("(next (cons record (vector-ref registry 0)))",
                 "(record (cons row snapshot))", "(pending (vector row next 'prestage))",
                 "(lineage (vector registry pending row",
                 "(vector-set! registry 1 lineage)",
                 "(vector-set! current 0 lineage)",
                 "(eq? (vector-ref pending 2) 'building)",
                 "(eq? (vector-ref pending 2) 'sealing)",
                 "(exit 134)", "(raise caught)"):
        if part not in core:
            raise ValueError("issuance, abort, or failstop changed")
    quota = core[core.index("(define (g3r-x1-model-lineage-capacity"):
                 core.index("(define (m3t-model-lineage-preflight")]
    if ("(define g3r-x1-model-lineage-limit 64)" not in core or
            quota.index("((= capacity 0)") > quota.index("((null? remaining)")):
        raise ValueError("retained row quota changed")
    auth = completed_predicate(core)
    needed = [["eq?", ["vector-ref", "row", "0"], "model"],
              ["eq?", ["vector-ref", "row", "2"], "original"],
              ["eq?", ["vector-ref", "row", "4"], "entry"],
              ["eq?", ["vector-ref", "snapshot", "3"], "entry"],
              ["eq?", ["vector-ref", "entry", "5"],
               ["vector-ref", "snapshot", "4"]]]
    if any(item not in auth for item in needed):
        raise ValueError("issuer-held snapshot does not authenticate mutable row")
    if ("((eq? (vector-ref (cdr (car records)) 0) model)" not in core or
            "(let search ((records (vector-ref registry 0)))" not in core or
            "(m3t-fail 'invalid-argument operation" not in core or
            "\n                      #t))\n                   (else (search (cdr records))))))))" not in core):
        raise ValueError("forged inserted row or returned authority permitted")


def route_start(core):
    return core.index("(define m3t-model-create-route")


class M3X1Source(unittest.TestCase):
    def setUp(self):
        self.core, self.ext = CORE.read_text(), EXT.read_text()
        self.root, self.manifest = PRIVATE_ROOT.read_text(), MANIFEST.read_text()
        self.x1 = X1.read_text()

    def check(self, **changes):
        data = dict(core=self.core, ext=self.ext, root=self.root,
                    manifest=self.manifest, x1=self.x1)
        data.update(changes)
        check(**data)

    def test_exact_source(self):
        self.check()

    def test_positive_preflight_and_invalid_states(self):
        pred = lineage_predicate(self.core)
        self.assertTrue(eval_pure(pred, issued_state()))
        for mutation in (
            lambda e: e["registry"].__setitem__(1, e["lineage"][1]),
            lambda e: e["current"].__setitem__(0, None),
            lambda e: e["lineage"].__setitem__(5, object()),
            lambda e: e["lineage"].__setitem__(0, [None, e["lineage"]]),
            lambda e: e["lineage"].__setitem__(2, list(e["lineage"][2])),
            lambda e: e["lineage"].__setitem__(4, list(e["lineage"][4])),
        ):
            state = issued_state()
            mutation(state)
            self.assertFalse(eval_pure(pred, state))
        contradiction = self.core.replace(
            "(eq? (vector-ref registry 1) lineage)",
            "(eq? (vector-ref registry 1) (vector-ref lineage 1))")
        self.assertFalse(eval_pure(lineage_predicate(contradiction), issued_state()))

    def test_mutable_verifier_and_forged_registry_denied(self):
        for before, after in (
            ("(registry (vector '() #f))", "(registry #f)"),
            ("(core (lambda (config state lineage)",
             "(core (vector (lambda (config state lineage)"),
            ("(eq? (vector-ref registry 1) lineage)", "#t"),
            ("(eq? (vector-ref lineage 5) secret)", "#t"),
        ):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(core=self.core.replace(before, after))
        with self.assertRaises(ValueError):
            self.check(ext=self.ext.replace("'require", "'public"))

    def test_completed_identity_and_forged_ordinary_model(self):
        pred = completed_predicate(self.core)
        genuine = completed_state()
        self.assertTrue(eval_pure(pred, genuine))
        ordinary_model = object()
        # Search is keyed by the issuer-held snapshot, not the mutable row.
        self.assertIsNot(genuine["snapshot"][0], ordinary_model)
        for mutation in (
            lambda e: e["row"].__setitem__(0, ordinary_model),
            lambda e: e["row"].__setitem__(2, object()),
            lambda e: e["row"].__setitem__(4, [None] * 8),
            lambda e: e["snapshot"].__setitem__(3, [None] * 8),
            lambda e: e["row"].__setitem__(5, "alternate"),
            lambda e: e["entry"].__setitem__(5, object()),
        ):
            state = completed_state()
            mutation(state)
            self.assertFalse(eval_pure(pred, state))

    def test_issuer_snapshot_and_row_mutants(self):
        for before, after in (
            ("(record (cons row snapshot))", "(record (cons row row))"),
            ("(eq? (vector-ref row 0) model)", "#t"),
            ("(eq? (vector-ref row 2) original)", "#t"),
            ("(eq? (vector-ref snapshot 3) entry)", "#t"),
            ("(let search ((records (vector-ref registry 0)))",
             "(let search ((records '()))"),
        ):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(core=self.core.replace(before, after))
        with self.assertRaises(ValueError):
            self.check(core=self.core.replace(
                "\n                      #t))\n                   (else (search (cdr records))))))))",
                "\n                      row))\n                   (else (search (cdr records))))))))"))

    def test_public_and_failure_cuts(self):
        for before, after in (
            ("((eq? mode 'public) (core config state #f))",
             "((eq? mode 'public) (private config state))"),
            ("(eq? (vector-ref pending 2) 'building)", "#f"),
            ("(eq? (vector-ref pending 2) 'sealing)", "#f"),
            ("(define g3r-x1-model-lineage-limit 64)",
             "(define g3r-x1-model-lineage-limit 1000000)"),
        ):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.check(core=self.core.replace(before, after))

    def test_aggregate_and_x1_identity(self):
        with self.assertRaises(ValueError):
            self.check(root=self.root.replace("g3r_d2_lineage_root.esk",
                                              "g3g_manual_package_root.esk"))
        with self.assertRaises(ValueError):
            self.check(manifest=self.manifest.replace(
                "native/g3r_m3_x1_lineage_extension.esk\n", ""))
        with self.assertRaises(ValueError):
            self.check(x1=self.x1.replace("x1-resolved? x1-private-resolved-entry)",
                                            "x1-resolved?)"))


if __name__ == "__main__":
    unittest.main()
