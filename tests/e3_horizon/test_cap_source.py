from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_preinvoke_cap(frame: str, restore: str) -> None:
    if not (
        frame.index("(e3-stage-set! entry 2)")
        < frame.index("(e3-d2-prepare! frame dataset)")
        < frame.index("(e3-native-acquire (vector-ref entry 4))")
    ):
        raise ValueError("D2 cap is no longer before native acquire")
    cap = restore.index("(> remaining 8192)")
    if not (
        cap < restore.index('"D2 dataset exceeds E3 traversal bounds"')
        < restore.index("(vector-set! record 4 dataset)")
    ):
        raise ValueError("D2 one-over cap is no longer before prepare publication")


class CapCutSourceTests(unittest.TestCase):
    def test_existing_preinvoke_cut_and_negative_mutations(self) -> None:
        frame = (ROOT / "native/e3_private_extension.esk").read_text()
        restore = (ROOT / "internal/e3/lib/e3_d2_restore.esk").read_text()
        check_preinvoke_cap(frame, restore)
        with self.assertRaises(ValueError):
            check_preinvoke_cap(
                frame.replace(
                    "(e3-d2-prepare! frame dataset)",
                    "(e3-native-acquire (vector-ref entry 4))\n"
                    "(e3-d2-prepare! frame dataset)", 1,
                ),
                restore,
            )
        with self.assertRaises(ValueError):
            check_preinvoke_cap(frame, restore.replace("(> remaining 8192)", "(> remaining 8193)"))


if __name__ == "__main__":
    unittest.main()
