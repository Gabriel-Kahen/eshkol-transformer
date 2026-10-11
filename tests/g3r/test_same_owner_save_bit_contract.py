"""Deterministic contract oracle; it does not execute a SAVE implementation."""

from dataclasses import dataclass, replace
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/g3/G3_R_SAME_OWNER_SAVE_BIT_WITNESS_CONTRACT.md"
M3T = ROOT / "native/m3t_transport_extension.esk"
ZERO = 0x00000000


def roster(document):
    rows = re.findall(r"^\| (\d+) \| `([^`]+)` \| `\[([^]]+)\]` \|$",
                      document, re.M)
    return [(int(i), path, tuple(map(int, shape.split(","))))
            for i, path, shape in rows]


def m3t_roster(source):
    paths = source.split("(define m3t-paths", 1)[1].split("(define m3t-shapes", 1)[0]
    shapes = source.split("(define m3t-shapes", 1)[1].split("(define (m3t-append-name", 1)[0]
    names = ["/".join(re.findall(r'"([^"]+)"', row))
             for row in re.findall(r"\((?:\"[^\"]+\"\s*)+\)", paths)]
    dims = [tuple(map(int, row.split()))
            for row in re.findall(r"\((\d+(?:\s+\d+)*)\)", shapes)]
    return list(zip(range(len(names)), names, dims))


@dataclass(frozen=True)
class Observation:
    original: object
    source: object
    head: object
    tie: object
    before: tuple[int, ...]
    after: tuple[int, ...]
    detached: tuple[int, ...]
    gradient_before: tuple[int, ...]
    gradient_after: tuple[int, ...]
    gradient_state: str
    weight_before: int
    weight_after: int
    count_before: int
    count_after: int
    controls_before: tuple
    controls_after: tuple


def admits(obs):
    return (
        obs.source is obs.original and obs.tie is obs.head
        and obs.before == obs.after == obs.detached
        and obs.gradient_before == obs.gradient_after
        and obs.gradient_state == "absent"
        and all(bit == ZERO for bit in obs.gradient_before)
        and obs.weight_before == obs.weight_after
        and obs.count_before == obs.count_after
        and obs.controls_before == obs.controls_after
    )


@dataclass(frozen=True)
class Cut:
    recoverable: bool
    published: bool
    original_unchanged: bool
    detached_drained: bool
    first_error_retained: bool
    leases_drained: bool
    retry_allowed: bool
    uncertainty: bool = False
    authority_retained: bool = False
    fail_stop: bool = False


def atomic(cut):
    if cut.uncertainty:
        return (not cut.recoverable and not cut.published
                and cut.authority_retained and cut.fail_stop)
    return (cut.recoverable and not cut.published
            and cut.original_unchanged and cut.detached_drained
            and cut.first_error_retained and cut.leases_drained
            and cut.retry_allowed)


class SameOwnerSaveBitContract(unittest.TestCase):
    def setUp(self):
        original, head = object(), object()
        controls = ("rng4", "cursor", "epoch-cursor", "tokenizer",
                    "config", "updates", "tokens", "history")
        self.obs = Observation(
            original, original, head, head, (0x3F800000, 0x80000000),
            (0x3F800000, 0x80000000), (0x3F800000, 0x80000000),
            (ZERO, ZERO), (ZERO, ZERO), "absent", ZERO, ZERO, 0, 0,
            controls, controls)

    def test_exact_roster_and_tie_source(self):
        doc, source = CONTRACT.read_text(), M3T.read_text()
        self.assertEqual(roster(doc), m3t_roster(source))
        self.assertEqual(len(roster(doc)), 14)
        self.assertIn('(eq? (vector-ref (car rows) 1) (vector-ref handles 10))',
                      source)
        self.assertIn('unexpected diagnostic parameter alias', source)
        self.assertIn('"token_embedding" "weight"', source)
        self.assertNotEqual(roster(doc.replace('`[2,4]`', '`[4,4]`')),
                            m3t_roster(source))
        self.assertNotEqual(roster(doc.replace('`head/weight`', '`head/copy`')),
                            m3t_roster(source))

    def test_same_handle_bits_gradient_and_detached_baseline(self):
        self.assertTrue(admits(self.obs))
        negatives = (
            replace(self.obs, source=object()),
            replace(self.obs, tie=object()),
            replace(self.obs, after=(0x3F800001, 0x80000000)),
            replace(self.obs, after=(0x3F800000, ZERO)),
            replace(self.obs, detached=(0x3F800001, 0x80000000)),
            replace(self.obs, gradient_after=(ZERO, 0x80000000)),
            replace(self.obs, gradient_before=(ZERO, 0x80000000),
                    gradient_after=(ZERO, 0x80000000)),
            replace(self.obs, gradient_state="present"),
            replace(self.obs, weight_after=0x3F800000),
            replace(self.obs, count_after=1),
        )
        for candidate in negatives:
            with self.subTest(candidate=candidate):
                self.assertFalse(admits(candidate))

    def test_rng_cursor_token_and_journal_drift(self):
        self.assertTrue(admits(self.obs))
        for index in range(len(self.obs.controls_before)):
            changed = list(self.obs.controls_after)
            changed[index] = "changed"
            self.assertFalse(admits(replace(self.obs, controls_after=tuple(changed))))

    def test_recoverable_and_uncertain_failure_atomicity(self):
        good = Cut(True, False, True, True, True, True, True)
        self.assertTrue(atomic(good))
        for field in ("published", "original_unchanged", "detached_drained",
                      "first_error_retained", "leases_drained", "retry_allowed"):
            self.assertFalse(atomic(replace(good, **{field: field == "published"})))
        self.assertFalse(atomic(replace(good, uncertainty=True)))
        uncertain = Cut(False, False, True, False, True, False, False,
                        uncertainty=True, authority_retained=True, fail_stop=True)
        self.assertTrue(atomic(uncertain))
        self.assertFalse(atomic(replace(uncertain, authority_retained=False)))
        self.assertFalse(atomic(replace(uncertain, fail_stop=False)))


if __name__ == "__main__":
    unittest.main()
