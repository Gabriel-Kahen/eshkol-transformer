"""Host-only sticky K1 transcript and source-contract negative controls."""

import unittest
from pathlib import Path

from tests.e3_private.parse_n4_sticky_observation import parse
from tests.e3_private.test_n4_sparse_tuple import sample

ROOT = Path(__file__).resolve().parents[2]


def transcript(mode: int, before_sw: str, after_sw: str,
               before_mx: str, after_mx: str) -> bytes:
    original = sample().splitlines(keepends=True)
    record = (f"E3N4FENV\t{mode}\t037f\t037f\t{before_sw}\t{after_sw}"
              f"\t{before_mx}\t{after_mx}\n").encode("ascii")
    return b"".join(original[:1] + [record] + original[1:])


class StickyTupleTest(unittest.TestCase):
    def test_pair_and_mutants(self):
        clear = transcript(0, "0000", "0020", "00001f80", "00001fa0")
        seed = transcript(1, "0004", "0004", "00001fa1", "00001fa1")
        parse(clear, "absent", "clear")
        parse(seed, "absent", "seed")
        for changed, mode in (
            (transcript(2, "0000", "0000", "00001f80", "00001f80"), "clear-before"),
            (transcript(3, "0004", "0000", "00001fa1", "00001f80"), "clear-after"),
        ):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "seeded sticky flags"):
                parse(changed, "absent", mode)
        mutations = (
            (transcript(1, "0000", "0004", "00001f80", "00001fa1"), "seed"),
            (transcript(1, "0004", "0000", "00001fa1", "00001f80"), "seed"),
            (transcript(1, "0004", "0004", "00009fa1", "00009fa1"), "seed"),
            (transcript(1, "0004", "0004", "00001fa1", "00005fa1"), "seed"),
            (transcript(0, "0004", "0004", "00001f80", "00001f80"), "clear"),
            (seed.replace(b"00000001", b"00000000", 1), "seed"),
            (seed.replace(b"E3ROW\t0\t3,0\t4,0\t1,0",
                          b"E3ROW\t0\t3,0\t4,0\t0,1", 1), "seed"),
            (seed.replace(b"E3N4FENV", b"E3N4OTHER", 1), "seed"),
        )
        for changed, mode in mutations:
            with self.subTest(mode=mode, tail=changed[-48:]), self.assertRaises(ValueError):
                parse(changed, "absent", mode)

    def test_mode_is_private_and_after_actual_dispatch(self):
        source = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('ET_E3_N4_STICKY_MODE) && !defined(ET_E3_N4_SPARSE_TESTING)', source)
        self.assertIn('"fnclex"', source)
        self.assertIn('fdivrp %%st, %%st(1)', source)
        self.assertIn('if (ET_E3_N4_STICKY_MODE == 2)', source)
        self.assertIn('if (ET_E3_N4_STICKY_MODE == 3)', source)
        self.assertLess(source.index('const unsigned n4_mx_before = _mm_getcsr()'),
                        source.index('const int32_t dispatch_status ='))
        self.assertLess(source.index('const int32_t dispatch_status ='),
                        source.index('const unsigned n4_mx_after = _mm_getcsr()'))
        self.assertIn('"${e3_tuple_kind}" == retention-overflow', builder)
        self.assertIn('"-DET_E3_N4_STICKY_MODE=${E3_N4_STICKY_MODE}"', builder)
        self.assertNotIn("et_e3_n4_sticky", (ROOT / "include/eshkol_transformer/e3_evaluation_metrics_abi.h").read_text())


if __name__ == "__main__":
    unittest.main()
