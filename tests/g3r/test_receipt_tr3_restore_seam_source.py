"""Source-only owner, phase, and trainer authority contract for G3-R restore."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import definition, parse

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3r_c2_receipt_tr3_restore_extension.esk"
ROOT_FILE = ROOT / "native/g3r_c2_receipt_tr3_restore_root.esk"
CLOSURE = ROOT / "native/g3r_c2_receipt_tr3_restore_source_closure.txt"
PREDECESSOR = ROOT / "native/g3r_trained_pair_reader_source_closure.txt"


def verify(source: str, root: str, closure: str, predecessor: str) -> None:
    forms = parse(source)
    if len(forms) != 1:
        raise ValueError("restore seam must add exactly one private entry")
    entry = definition({"source": forms},
                       "g3r-c2-receipt-tr3-restore-internal!")
    if entry[1] != ["g3r-c2-receipt-tr3-restore-internal!",
                    "receipt", "trainer"]:
        raise ValueError("restore seam signature changed")
    required = (
        "(g3r-c2-receipt-find receipt)",
        "(if (not receipt-record)",
        "(not (eq? (vector-ref receipt-record 1) 'live))",
        "(vector-ref receipt-record 4)",
        "(vector-ref receipt-record 2)",
        "(c2-load-exact-live-owner? owner)",
        "(tr3-lease-record-for-shell-internal trainer operation)",
        "(not (eq? (vector-ref trainer-record 3) 'idle))",
        "(vector-ref trainer-record 4)",
        "(tr3-lease-restore-enter-admission! trainer-record operation)",
        "(vector-set! receipt-record 1 'restoring)",
        "(guard (caught",
        "(vector-set! receipt-record 1 'live)",
        "(raise caught)",
        "(tr3-c-joint-restore-internal! trainer owner)",
        "(if (not (eq? restored #t)) (tr3-lease-fail-stop 134))",
        "(vector-set! receipt-record 1 'live)",
    )
    positions = [source.find(part) for part in required[:-1]]
    positions.append(source.rfind(required[-1]))
    if -1 in positions or positions != sorted(positions):
        raise ValueError("authentic owner, trainer, or phase order changed")
    if source.count("(vector-set! receipt-record 1 'live)") != 2:
        raise ValueError("failure and success must both restore receipt phase")
    if source.count("(tr3-c-joint-restore-internal! trainer owner)") != 1:
        raise ValueError("restore must use the retained owner exactly once")
    if source.count("(vector-ref receipt-record 2)") != 1:
        raise ValueError("owner escaped its single lexical binding")
    forbidden = ("c2-public-checkpoint-load", "g3r-c2-load-with-sha-internal",
                 "t1-native-read-exact", "c2-training-state-release-internal!",
                 "g3r-c2-receipt-release-internal!", "(load ", "extern ")
    if any(item in source for item in forbidden):
        raise ValueError("restore seam reopened or widened owner authority")
    if parse(root) != [
        ["load", '"g3r_trained_pair_reader_root.esk"'],
        ["load", '"g3r_c2_receipt_tr3_restore_extension.esk"'],
    ]:
        raise ValueError("restore root must load one accepted reader registry")
    expected = ["native/g3r_c2_receipt_tr3_restore_root.esk",
                *predecessor.splitlines(),
                "native/g3r_c2_receipt_tr3_restore_extension.esk"]
    if closure.splitlines() != expected or len(expected) != len(set(expected)):
        raise ValueError("restore source closure changed")
    for member in ("native/cli3_root.esk",
                   "native/tr3_c_joint_restore_extension.esk",
                   "native/g3r_c2_sha_receipt_extension.esk"):
        if expected.count(member) != 1:
            raise ValueError("restore root duplicated an owning registry")


class ReceiptTr3RestoreSeamSourceTest(unittest.TestCase):
    def test_exact_authority_contract(self):
        verify(SOURCE.read_text(), ROOT_FILE.read_text(),
               CLOSURE.read_text(), PREDECESSOR.read_text())

    def test_forged_dead_busy_and_wrong_trainer_mutations(self):
        source = SOURCE.read_text()
        root = ROOT_FILE.read_text()
        closure = CLOSURE.read_text()
        predecessor = PREDECESSOR.read_text()
        for old, new in (
            ("(g3r-c2-receipt-find receipt)", "(vector receipt)"),
            ("(if (not receipt-record)", "(if receipt-record"),
            ("(not (eq? (vector-ref receipt-record 1) 'live))",
             "(eq? (vector-ref receipt-record 1) 'dead)"),
            ("(vector-ref receipt-record 4)", "#f"),
            ("(c2-load-exact-live-owner? owner)", "owner"),
            ("(tr3-lease-record-for-shell-internal trainer operation)",
             "(vector trainer)"),
            ("(tr3-lease-restore-enter-admission! trainer-record operation)",
             "#t"),
            ("(vector-set! receipt-record 1 'restoring)", "#t"),
            ("(tr3-c-joint-restore-internal! trainer owner)",
             "(tr3-c-joint-restore-internal! trainer receipt)"),
            ("(raise caught)", "#f"),
        ):
            with self.subTest(old=old):
                mutant = source.replace(old, new, 1)
                self.assertNotEqual(mutant, source)
                with self.assertRaises(ValueError):
                    verify(mutant, root, closure, predecessor)
        with self.assertRaises(ValueError):
            verify(source + "\n(c2-public-checkpoint-load #f #f #f)\n",
                   root, closure, predecessor)


if __name__ == "__main__":
    unittest.main()
