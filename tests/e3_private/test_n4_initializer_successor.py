"""Pin the C1-loaded N4 driver's initializer assertion to the Q0 oracle."""

import unittest
from pathlib import Path

from tests.n3k.test_initializer_reference import initialize


ROOT = Path(__file__).resolve().parents[2]


class N4InitializerSuccessorTest(unittest.TestCase):
    def test_tied_unique_matrix_draws_and_driver_assertion(self):
        state = (1, 1729, 0, 0)
        for shape in ((256, 4), (2, 4), (4, 4), (4, 4), (4, 4),
                      (4, 4), (8, 4), (4, 8)):
            _, state = initialize(shape, state)
        self.assertEqual(state, (1, 1729, 290, 0))
        source = (ROOT / "tests/e3_private/overflow_driver.esk").read_text()
        self.assertIn("(equal? initializer '(1 1729 290 0))", source)
        self.assertNotIn("(equal? initializer '(1 1729 580 0))", source)


if __name__ == "__main__":
    unittest.main()
