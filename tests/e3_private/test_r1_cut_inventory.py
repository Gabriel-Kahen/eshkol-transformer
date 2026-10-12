"""Source-only guard for the bounded E3 §9 R1 inventory."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs/e3/E3_SECTION9_R1_CUT_INVENTORY.md"
EXPECTED = {
    "admission", "invocation-envelope", "p1-prepare", "d2-prepare", "frame-allocation",
    "tensor-allocation", "provider-discovery", "epsilon-copy",
    "acquisition-pin", "p1-enter", "batch-envelope", "d2-source-allocation",
    "batch-allocation", "late-shard-read", "batch-copy",
    "provider-preflight-common", "correct-preflight", "accumulate-preflight",
    "post-provider", "finalize",
    "final-publication", "public-seek", "k1-invoke",
}
DISPOSITIONS = {"existing", "partial", "gap", "excluded"}
PINNED_SEAMS = {
    "invocation-envelope": "native/e3_private_extension.esk::e3-evaluate-into!",
    "batch-envelope": "native/e3_private_extension.esk::e3-run-one-batch!",
    "d2-source-allocation": "internal/d2/lib/d2_dataset.esk::d2-token-dataset-next-batch",
    "provider-preflight-common": "src/eshkol_transformer/e3_frame.c::e3_dispatch",
    "correct-preflight": "src/eshkol_transformer/e3_frame.c::et_e3_private_correct_v1",
    "accumulate-preflight": "src/eshkol_transformer/e3_frame.c::e3_accumulate_dispatch",
}


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
        if cut in PINNED_SEAMS and f"{path}::{symbol}" != PINNED_SEAMS[cut]:
            errors.append(f"{cut}: wrong call-site seam")
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

    def test_invocation_allocations_and_all_three_k1_dispatch_sites(self) -> None:
        source = (ROOT / "native/e3_private_extension.esk").read_text()
        entry = source.split("(define (e3-evaluate-into!", 1)[1].split(
            "(define (e3-publish!", 1)[0]
        batch = source.split("(define (e3-run-one-batch!", 1)[1].split(
            "(define (e3-traverse!", 1)[0]
        traverse = source.split("(define (e3-traverse!", 1)[1].split(
            "(define (e3-evaluate-closed!", 1)[0]
        d2 = (ROOT / "internal/d2/lib/d2_dataset.esk").read_text()
        next_batch = d2.split("(define (d2-token-dataset-next-batch", 1)[1].split(
            "(define (d2-token-dataset-cursor", 1)[0]
        self.assertIn("(cons #f '())", entry)
        self.assertIn("(vector #f 0 #f (cons #f '()))", batch)
        self.assertIn("(d2-token-dataset-next-batch dataset)", batch)
        self.assertIn("(with-region ('e3-evaluation-batch 65536)", traverse)
        self.assertIn("(batch-cap (vector", next_batch)
        self.assertIn("(boundary-input (make-bytevector", next_batch)

        frame = (ROOT / "src/eshkol_transformer/e3_frame.c").read_text()
        sites = (
            ("static int64_t e3_dispatch(", "int64_t et_e3_private_forward_role_v1("),
            ("int64_t et_e3_private_correct_v1(", "static int64_t e3_accumulate_dispatch("),
            ("static int64_t e3_accumulate_dispatch(", "int64_t et_e3_private_accumulate_v1("),
        )
        for begin, end in sites:
            body = frame.split(begin, 1)[1].split(end, 1)[0]
            self.assertEqual(body.count("et_kernel_runtime_dispatch("), 1, begin)


if __name__ == "__main__":
    unittest.main()
