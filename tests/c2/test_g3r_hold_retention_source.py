"""Source guards for the feature-private active-only hold pair lifecycle."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import form, parse


ROOT = Path(__file__).resolve().parents[2]
ESK = ROOT / "native/g3r_same_owner_hold_bridge_extension.esk"
C = ROOT / "native/g3r_same_owner_hold_bridge.c"
HEADER = ROOT / "native/g3r_same_owner_hold_bridge.h"


def definition(forms, name):
    return next(x for x in forms if x[0] == "define" and
                isinstance(x[1], list) and x[1][0] == name)


def contains(tree, target):
    return tree == target or (isinstance(tree, list) and
                              any(contains(x, target) for x in tree))


def validate(esk, c, header):
    forms = parse(esk)
    begin = definition(forms, "g3r-hold-pair-begin!")
    end = definition(forms, "g3r-hold-pair-end!")
    scrub = definition(forms, "g3r-hold-pair-scrub!")
    unlink = definition(forms, "g3r-hold-pair-unlink!")
    if not contains(begin, form("(g3r-hold-pair-scrub! ledger)")) or \
            not contains(end, form("(g3r-hold-pair-scrub! ledger)")):
        raise ValueError("failed begin or proved end retains Eshkol pair")
    if not contains(scrub, form("(vector-set! ledger index #f)")) or \
            not contains(scrub, form("(vector-set! (vector-ref ledger 1) 0 #f)")) or \
            not contains(scrub, form("(g3r-hold-pair-unlink! ledger)")):
        raise ValueError("ended pair retains owner references")
    for exact in ("(eq? ledger (car remaining))",
                  "(set-cdr! previous (cdr remaining))",
                  "(vector-set! g3r-hold-pair-registry 0 (cdr remaining))",
                  "(g3r-holds-fail-stop)"):
        if not contains(unlink, form(exact)):
            raise ValueError("pair registry exact unlink weakened")
    for exact in ("int64_t ticket;", "static int64_t next_pair_ticket = 1;",
                  "pair->ticket = next_pair_ticket++;",
                  "if (p->ticket == candidate && p->live && p->g3 && p->o2)",
                  "*slot = pair->next;", "--pair_count;", "free(pair);"):
        if exact not in c:
            raise ValueError("native ticket or active-only retirement weakened")
    if "int64_t et_g3r_private_same_owner_holds_begin_v1" not in header or \
            "int64_t pair, void *generator" not in header or \
            "(extern i64 g3r-holds-native-begin ptr ptr ptr" not in esk:
        raise ValueError("private pair ticket ABI changed")


class HoldRetentionSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.esk = ESK.read_text()
        cls.c = C.read_text()
        cls.header = HEADER.read_text()

    def test_contract(self):
        validate(self.esk, self.c, self.header)

    def test_retirement_and_stale_ticket_mutations_fail(self):
        for old in ("(begin (g3r-hold-pair-scrub! ledger)",
                    "\n        (g3r-hold-pair-scrub! ledger)\n",
                    "(vector-set! ledger index #f)",
                    "(set-cdr! previous (cdr remaining))"):
            with self.subTest(old=old), self.assertRaises(ValueError):
                validate(self.esk.replace(old, "#t"), self.c, self.header)
        for old in ("if (p->ticket == candidate && p->live && p->g3 && p->o2)",
                    "--pair_count;", "free(pair);"):
            with self.subTest(old=old), self.assertRaises(ValueError):
                validate(self.esk, self.c.replace(old, "/* removed */"), self.header)


if __name__ == "__main__":
    unittest.main()
