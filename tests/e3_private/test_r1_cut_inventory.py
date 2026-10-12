"""Source-only guard for the bounded E3 §9 R1 inventory."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs/e3/E3_SECTION9_R1_CUT_INVENTORY.md"
EXPECTED = {
    "admission", "p1-prepare", "d2-prepare", "frame-allocation",
    "tensor-allocation", "provider-discovery", "epsilon-copy",
    "acquisition-pin", "p1-enter", "batch-allocation", "late-shard-read",
    "batch-copy", "provider-preflight", "post-provider", "finalize",
    "final-publication", "public-seek", "k1-invoke",
}
DISPOSITIONS = {"existing", "partial", "gap", "excluded"}


def inventory_errors(document: str) -> list[str]:
    rows = [line for line in document.splitlines() if line.startswith("| ")]
    errors: list[str] = []
    seen: set[str] = set()
    for line in rows:
        fields = [field.strip() for field in line.strip("|").split("|")]
        if fields[0] == "Cut":
            continue
        if len(fields) != 7:
            errors.append("row must have seven fields")
            continue
        cut, stage, reach, owner, seam, disposition, assertion = fields
        if cut in seen or cut not in EXPECTED:
            errors.append(f"duplicate or unknown cut: {cut}")
        seen.add(cut)
        for label, value in (("stage", stage), ("reachability", reach),
                             ("owner", owner), ("assertion", assertion)):
            if not value or value in {"-", "?", "TBD"}:
                errors.append(f"{cut}: missing {label}")
        if disposition not in DISPOSITIONS:
            errors.append(f"{cut}: missing disposition")
        if disposition == "gap" and "No " not in assertion:
            errors.append(f"{cut}: gap must name the absent cut")
        match = re.fullmatch(r"`([\w/.-]+\.(?:c|esk))::([\w!?-]+)`", seam)
        if not match:
            errors.append(f"{cut}: missing source seam")
            continue
        path, symbol = match.groups()
        source_path = ROOT / path
        if ".." in Path(path).parts or not source_path.is_file():
            errors.append(f"{cut}: missing source file {path}")
            continue
        source = source_path.read_text()
        if path.endswith(".esk"):
            definition = rf"(?m)^\s*\(define\s+\({re.escape(symbol)}(?:\s|\))"
        else:
            definition = rf"(?m)^\s*(?:static\s+)?(?:[\w*]+\s+)+{re.escape(symbol)}\s*\("
        if not re.search(definition, source):
            errors.append(f"{cut}: stale or missing definition {path}::{symbol}")
    if seen != EXPECTED:
        errors.append(f"missing cuts: {sorted(EXPECTED - seen)}")
    return errors


class R1CutInventoryTest(unittest.TestCase):
    def test_source_inventory_is_complete_and_current(self) -> None:
        self.assertEqual(inventory_errors(INVENTORY.read_text()), [])

    def test_rejects_missing_row_fields_and_stale_symbol(self) -> None:
        original = INVENTORY.read_text()
        for old, new in (
            ("| I2 | `native/f32_tensor.c::et_f32_tensor_copy_bits_from_v1`",
             "|  | `native/f32_tensor.c::et_f32_tensor_copy_bits_from_v1`"),
            ("Frame creation copies epsilon bits after discovery.", ""),
            ("| gap | No copy cut at the later epsilon copy",
             "|  | No copy cut at the later epsilon copy"),
            ("et_f32_tensor_copy_bits_from_v1`", "missing_copy_symbol`"),
        ):
            self.assertIn(old, original)
            self.assertTrue(inventory_errors(original.replace(old, new, 1)))
        without_row = "\n".join(
            line for line in original.splitlines()
            if not line.startswith("| epsilon-copy |"))
        self.assertTrue(inventory_errors(without_row))


if __name__ == "__main__":
    unittest.main()
