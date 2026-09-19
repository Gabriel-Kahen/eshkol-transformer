"""Result-decoder tests; synthetic observations are not compiled parity evidence."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from tests.m3.check_public_parity import WEIGHT_BITS, check


class PublicParityDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="m3-parity-decoder-")
        cls.root = Path(cls.temporary.name)
        cls.reference = cls.root / "reference.json"
        subprocess.run([sys.executable, "-m", "tests.m3.reference", "--output", str(cls.reference)], check=True, capture_output=True)
        cls.document = json.loads(cls.reference.read_text())
        cls.observed = cls.root / "observed.txt"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def rows(self, instrumented):
        def row(prefix, values):
            raw = struct.pack("<" + "f" * len(values), *values)
            return prefix + " " + " ".join(map(str, raw))

        result = []
        for case, item in enumerate(self.document["cases"][:3]):
            result.append(row(f"M3-LOGITS {case}", item["logits"]["values"]))
            if instrumented:
                for index, path in enumerate(sorted(item["parameters"])):
                    result.append(row(f"M3-PARAMETER {case} {index}", item["parameters"][path]["values"]))
                    result.append(row(f"M3-GRADIENT {case} {index}", item["unique_vjp"][path]["values"]))
                    result.append(f"M3-METADATA {case} {index} 1 1 {WEIGHT_BITS[case]}")
        return [*result, "M3-PUBLIC-MODEL-PASS"]

    def invoke(self, rows, instrumented):
        self.observed.write_text("\n".join(rows) + "\n")
        with redirect_stdout(io.StringIO()):
            check(self.reference, self.observed, instrumented)

    def test_valid_public_and_instrumented_result_shapes(self):
        self.invoke(self.rows(False), False)
        self.invoke(self.rows(True), True)

    def test_missing_duplicate_metadata_and_numerical_mutations_reject(self):
        baseline = self.rows(True)
        mutations = [baseline[:-1], [*baseline[:-1], baseline[1], baseline[-1]], baseline[1:]]
        for prefix, replacement in (("M3-GRADIENT 0 0 ", " ".join(["0"] * 64)),
                                    ("M3-METADATA 0 0 ", "1 2 1065353216"),
                                    ("M3-METADATA 0 0 ", "1 1 1073741824")):
            mutations.append([prefix + replacement if line.startswith(prefix) else line for line in baseline])
        # Lost embedding contribution in the tied unique parameter.
        head = self.document["cases"][0]["tied_edge_vjp"]["head"]["values"]
        raw = struct.pack("<" + "f" * len(head), *head)
        prefix = "M3-GRADIENT 0 10 "
        mutations.append([prefix + " ".join(map(str, raw)) if line.startswith(prefix) else line for line in baseline])
        for target in ("M3-LOGITS 0 ", "M3-PARAMETER 0 0 "):
            changed = []
            for line in baseline:
                if line.startswith(target):
                    values = line[len(target):].split()
                    values[0] = str(int(values[0]) ^ 1)
                    if target.startswith("M3-LOGITS"):
                        values[:4] = ["0", "0", "128", "127"]
                    line = target + " ".join(values)
                changed.append(line)
            mutations.append(changed)
        for index, rows in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises((ValueError, struct.error)):
                self.invoke(rows, True)
        with self.assertRaises(ValueError):
            self.invoke(baseline, False)
