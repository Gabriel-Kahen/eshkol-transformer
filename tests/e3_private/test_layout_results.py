from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.d2.reference import ReferenceConfig
from tests.e3_reference.corpus import CASES, materialize
from tests.e3_reference.join import validate_five_case_join
from tests.e3_reference.test_join import _bundle, _record_sha
from tests.e3_private.verify_layout_results import parse_stdout, verify_results


def _stdout(results: dict[str, dict[str, object]]) -> bytes:
    lines = []
    for name, *_ in CASES:
        result = results[name]
        before = bytes.fromhex(result["cursor_before"])
        after = bytes.fromhex(result["cursor_after"])
        lines.append("\t".join((
            "E3RESULT", name,
            ",".join(map(str, result["outputs"])),
            ",".join(map(str, before)),
            ",".join(map(str, after)),
        )))
    return ("\n".join((*lines, "E3-PRIVATE-RUNTIME-PASS")) + "\n").encode("ascii")


class LayoutResultTests(unittest.TestCase):
    def test_driver_cursor_config_matches_frozen_reference(self) -> None:
        driver = (Path(__file__).resolve().parents[2] / "tests/e3_private/driver.esk").read_text()
        config = ReferenceConfig(1, 2)
        for key, value in (
            ("maximum-manifest-bytes", config.maximum_manifest_bytes),
            ("maximum-shard-bytes", config.maximum_shard_bytes),
            ("maximum-total-tokens", config.maximum_total_tokens),
            ("maximum-batch-bytes", config.maximum_batch_bytes),
            ("shuffle-window-rows", config.shuffle_window_rows),
        ):
            self.assertIn(f"':{key} {value}", driver)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="e3-result-join-")
        self.root = Path(self.temporary.name) / "corpus"
        self.corpus = materialize(self.root)
        self.bundle = _bundle(self.corpus)
        with patch("tests.e3_reference.join.FROZEN_BUNDLE_SHA256", _record_sha(self.bundle)):
            self.joined = validate_five_case_join(self.root, self.corpus, self.bundle)
        self.results = {}
        for case in self.joined:
            final = case["synthetic"]["final"]
            self.results[case["name"]] = {
                "outputs": [int(word, 16) for word in final["metrics_f32"]]
                           + final["counters"],
                "cursor_before": case["start_cursor"],
                "cursor_after": case["start_cursor"],
            }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_five_results_bind_six_outputs_and_restored_start_cursor(self) -> None:
        parsed = parse_stdout(_stdout(self.results))
        verify_results(self.joined, parsed)
        self.assertNotEqual(self.joined[0]["start_cursor"], self.joined[0]["end_cursor"])

    def test_missing_reordered_and_extra_result_fail(self) -> None:
        lines = _stdout(self.results).splitlines()
        for raw in (
            b"\n".join(lines[:-2] + lines[-1:]) + b"\n",
            b"\n".join(lines[1:2] + lines[:1] + lines[2:]) + b"\n",
            b"\n".join(lines[:1] + lines) + b"\n",
        ):
            with self.assertRaisesRegex(ValueError, "result count|ordering"):
                parse_stdout(raw)
        with self.assertRaisesRegex(ValueError, "canonical LF"):
            parse_stdout(_stdout(self.results).replace(b"\n", b"\r\n"))

    def test_cursor_and_publication_mutations_fail(self) -> None:
        for name, field, value, message in (
            ("packed-suffix", "cursor_after", self.joined[2]["end_cursor"], "cursor"),
            ("packed-single", "cursor_before", self.joined[1]["start_cursor"], "cursor"),
            ("packed-single", "outputs", [0] * 6, "weight"),
        ):
            changed = copy.deepcopy(self.results)
            changed[name][field] = value
            with self.assertRaisesRegex(ValueError, message):
                verify_results(self.joined, changed)
        changed = copy.deepcopy(self.results)
        changed["packed-sharded"]["outputs"][0] += 1
        with self.assertRaisesRegex(ValueError, "packed repartition|perplexity"):
            verify_results(self.joined, changed)

    def test_noncanonical_cursor_and_output_words_fail(self) -> None:
        raw = _stdout(self.results)
        with self.assertRaisesRegex(ValueError, "malformed private result integers"):
            parse_stdout(raw.replace(b"E3RESULT\tpacked-single\t", b"E3RESULT\tpacked-single\t01,", 1))
        fields = raw.splitlines()[0].split(b"\t")
        fields[3] = b"256," + fields[3].split(b",", 1)[1]
        changed = b"\t".join(fields) + b"\n" + b"\n".join(raw.splitlines()[1:]) + b"\n"
        with self.assertRaisesRegex(ValueError, "malformed private result integers"):
            parse_stdout(changed)


if __name__ == "__main__":
    unittest.main()
