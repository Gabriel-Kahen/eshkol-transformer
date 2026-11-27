"""Keep the private C2 admission roster aligned with the live V256 model."""

import json
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse
from tests.eval5.mod5_reference_oracle import PATHS, SHAPES_256

SOURCE = Path(__file__).resolve().parents[2] / "native" / "c2_mod5_v256_stage_policy.esk"


class Mod5V256StagePolicy(unittest.TestCase):
    def test_checkpoint_paths_shapes_and_alias_match_live_geometry(self):
        definitions = {
            form[1]: form[2] for form in parse(SOURCE.read_text())
            if form[0] == "define" and isinstance(form[1], str)
        }
        quote, rows = definitions["c2-mod5-v256-roster"]
        self.assertEqual(quote, "quote")
        self.assertEqual(
            tuple(".".join(map(json.loads, path)) for path, _ in rows), PATHS
        )
        self.assertEqual(
            tuple(tuple(map(int, shape)) for _, shape in rows), SHAPES_256
        )
        quote, aliases = definitions["c2-mod5-v256-tie"]
        self.assertEqual(quote, "quote")
        self.assertEqual(
            tuple(PATHS[int(ordinal)] for ordinal in aliases),
            ("head.weight", "token_embedding.weight"),
        )
