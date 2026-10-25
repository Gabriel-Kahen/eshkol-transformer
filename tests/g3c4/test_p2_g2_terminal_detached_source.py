"""Source guards for private detached terminal adapters, not runtime proof."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/g3c4_p2g2_terminal_detached_extension.esk"
HEADER = ROOT / "src/eshkol_transformer/g3c4_context_internal.h"


def balanced(source: str) -> bool:
    depth = 0
    quoted = False
    escaped = False
    for line in source.splitlines():
        for char in line.split(";;;", 1)[0]:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0 and not quoted


def preflight_has_original_extent(segment: str, native_call: str,
                                  payload_length: str) -> bool:
    before = segment.split(native_call, 1)[0]
    return all((
        "(eq? destination original-destination)" in before,
        "(bytevector? original-destination)" in before,
        payload_length in before,
    ))


class TerminalDetachedSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.header = HEADER.read_text()

    def test_private_abi_and_no_public_dispatch(self):
        self.assertTrue(balanced(self.source))
        for symbol in (
            "et_g3c4_private_p2g2_terminal_snapshot_v1",
            "et_g3c4_private_p2g2_terminal_ids_clone_v1",
            "et_g3c4_private_p2g2_terminal_ids_clone_copy_v1",
            "et_g3c4_private_p2g2_terminal_ids_clone_release_v1",
        ):
            self.assertIn(symbol, self.source)
            self.assertIn(symbol, self.header)
        self.assertNotIn("g3c4-public-", self.source)
        self.assertNotIn("g3c4-native-result-tensor-create", self.source)

    def test_clone_probed_definition_is_top_level(self):
        marker = "(define (g3c4-p2g2-terminal-ids-clone-probed-internal"
        before = self.source.split(marker, 1)[0]
        self.assertTrue(balanced(before))
        mutant = self.source.replace(
            "(bytevector-u8-ref text 1)))))))",
            "(bytevector-u8-ref text 1))))))", 1,
        ).replace("answer)))))", "answer))))))", 1)
        self.assertTrue(balanced(mutant))
        self.assertFalse(balanced(mutant.split(marker, 1)[0]))

    def test_snapshot_destination_precedes_ptr_and_no_cached_failure(self):
        segment = self.source.split(
            "(define (g3c4-p2g2-terminal-snapshot-internal shell)", 1
        )[1].split("(define g3c4-p2g2-terminal-ids-owners", 1)[0]
        native = segment.index("(g3c4-p2g2-native-terminal-snapshot\n")
        self.assertIn("(eq? destination original-destination)", segment[:native])
        self.assertIn("(= (bytevector-length original-destination) 64)", segment[:native])
        self.assertIn("native original-destination 72", segment[native:])
        self.assertIn("(g3c4-p2g2-zero-bytes! original-destination 64)", segment)
        self.assertIn("(= logical count)", segment[native:])
        self.assertIn("(= cache (+ 2 count))", segment[native:])
        self.assertIn("(= id-one 0)", segment[native:])
        self.assertIn("(= rng-zero 1)", segment[native:])
        self.assertIn("(>= rng-one 0)", segment[native:])
        self.assertLess(segment.index("(g3c4-p2g2-zero-bytes! original-destination 64)", native),
                        segment.index("(vector id-zero id-one logical cache", native))

    def test_clone_originals_copy_and_cleanup(self):
        segment = self.source.split(
            "(define (g3c4-p2g2-terminal-ids-clone-probed-internal", 1
        )[1].split("(define (g3c4-p2g2-terminal-ids-copy-internal shell)", 1)[0]
        copy = segment.index("(g3c4-p2g2-native-terminal-ids-clone-copy\n")
        publish = segment.index("(vector-set! entry 2 'live)")
        self.assertLess(copy, publish)
        self.assertIn("(eq? destination original-destination)", segment[:copy])
        self.assertIn("(= (bytevector-length original-destination)\n                           (* 8 count))", segment[:copy])
        self.assertIn("native count original-destination (+ 8 (* 8 count))", segment[copy:])
        self.assertIn("'register shell original-check", segment[:copy])
        self.assertIn("(eq? (vector-ref owner 2)\n                           (if (null? native) #f native))", segment[:copy])
        self.assertIn("(g3c4-p2g2-terminal-snapshot-matches-source", segment[:copy])
        self.assertIn("(g3c4-p2g2-native-terminal-ids-clone-release", segment)
        self.assertIn("(repair-clone native)", segment)
        self.assertIn("(after-clone native)", segment)
        self.assertLess(segment.index("(g3c4-p2g2-native-terminal-ids-clone-release"),
                        segment.index("(g3c4-entry-dead! entry)"))
        self.assertIn("(eq? (vector-ref g3c4-registry 0)\n                                            entry-node)", segment)
        self.assertIn("(g3c4-p2g2-terminal-ids-original-authority\n                                        'remove shell token)", segment)
        self.assertIn("(set! published #t)", segment)

    def test_detached_copy_and_release_reject_wrong_owner(self):
        copy = self.source.split(
            "(define (g3c4-p2g2-terminal-ids-copy-internal shell)", 1
        )[1].split("(define (g3c4-p2g2-terminal-ids-release-internal! shell)", 1)[0]
        release = self.source.split(
            "(define (g3c4-p2g2-terminal-ids-release-internal! shell)", 1
        )[1]
        self.assertIn("(g3c4-p2g2-terminal-ids-entry shell operation #f)", copy)
        self.assertIn("(eq? destination original-destination)", copy)
        self.assertIn("native count original-destination (+ 8 (* 8 count))", copy)
        self.assertIn("(vector-ref snapshot 0)", copy)
        self.assertIn("(vector-ref snapshot 1)", copy)
        self.assertIn("(g3c4-p2g2-terminal-ids-entry shell operation #t)", release)
        self.assertIn("(if (eq? (vector-ref owner 5) 'dead)\n            #t", release)
        self.assertLess(release.index("(g3c4-p2g2-native-terminal-ids-clone-release native)"),
                        release.index("(g3c4-entry-dead! entry)"))

    def test_negative_static_mutants_lose_allocation_or_owner_gate(self):
        snapshot = self.source.split(
            "(define (g3c4-p2g2-terminal-snapshot-internal shell)", 1
        )[1].split("(define g3c4-p2g2-terminal-ids-owners", 1)[0]
        clone = self.source.split(
            "(define (g3c4-p2g2-terminal-ids-clone-probed-internal", 1
        )[1].split("(define (g3c4-p2g2-terminal-ids-clone-internal", 1)[0]
        snap_call = "(g3c4-p2g2-native-terminal-snapshot\n"
        clone_call = "(g3c4-p2g2-native-terminal-ids-clone-copy\n"
        snap_extent = "(= (bytevector-length original-destination) 64)"
        clone_extent = "(= (bytevector-length original-destination)\n                           (* 8 count))"
        self.assertTrue(preflight_has_original_extent(snapshot, snap_call,
                                                      snap_extent))
        self.assertTrue(preflight_has_original_extent(clone, clone_call,
                                                      clone_extent))
        self.assertFalse(preflight_has_original_extent(
            snapshot.replace(snap_extent, "#t", 1), snap_call, snap_extent))
        self.assertFalse(preflight_has_original_extent(
            clone.replace("(eq? destination original-destination)",
                          "#t"), clone_call, clone_extent))
        original_owner_gate = "'register shell original-check"
        self.assertIn(original_owner_gate, clone)
        self.assertNotIn(original_owner_gate,
                         clone.replace(original_owner_gate, "'register shell owner"))


if __name__ == "__main__":
    unittest.main()
