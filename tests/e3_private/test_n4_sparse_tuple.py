"""Reject forged or incomplete N4 compiled transcripts before admission."""

import unittest

from tests.e3_private.parse_n4_sparse_observation import _expected_role, parse
from tests.e3_private.parse_numerical_observation import ROLE_NAMES
from tests.e3_private.parse_overflow_observation import SHAPES


def sample() -> bytes:
    rows = ["E3N4K1\t037f\t00001f80\t" + ",".join(_expected_role("Z", 512)),
            "E3ROW\t0\t3,0\t4,0\t1,0"]
    for name, shape in zip(ROLE_NAMES, SHAPES, strict=True):
        count = 1
        for dimension in shape:
            count *= dimension
        rows.append("\t".join(("E3ROLE", "0", name,
                                ",".join(map(str, shape)),
                                ",".join(_expected_role(name, count)))))
    rows.extend(("E3CE\t0\t40a00000,40b17218",
                 "E3BATCH\t0\t40a00000,3f800000,00000000,00000000\t1",
                 "E3STATE\t0\t40a00000,3f800000,00000000\t1,1",
                 "E3FINAL\t40a00000,3f800000,43700000,00000000\t1,1",
                 "E3-N4-SPARSE-PASS gradient=absent"))
    return ("\n".join(rows) + "\n").encode("ascii")


class SparseTupleTest(unittest.TestCase):
    def test_complete_and_rejections(self):
        good = sample()
        parse(good, "absent")
        mutations = (
            good.replace(b"E3ROW\t0\t3,0\t4,0\t1,0",
                         b"E3ROW\t0\t3,0\t4,0\t0,1", 1),
            good.replace(b"00000001", b"00000000", 1),
            good.replace(b"E3N4K1\t037f\t00001f80",
                         b"E3N4K1\t037f\t00009f80", 1),
            good.replace(b"E3BATCH\t0", b"E3BATCH\t1", 1),
            good.replace(b"E3-N4-SPARSE-PASS", b"E3-N4-SPARSE-FAIL", 1),
            good[:-1],
        )
        for changed in mutations:
            with self.subTest(changed=changed[-70:]), self.assertRaises(ValueError):
                parse(changed, "absent")


if __name__ == "__main__":
    unittest.main()
