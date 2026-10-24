"""Source checks for the prepared authentic SAVE producer gate."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import parse

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts/test-g3r-c2-authentic-save.sh"
PRODUCER = ROOT / "tests/c2/public_runtime.esk"
CONTROL = ROOT / "tests/g3r/public_without_save.esk"


def verify(gate: str, producer: str, control: str) -> None:
    parse(producer)
    parse(control)
    for source, required in ((producer, "checkpoint-save!"),
                             (control, "checkpoint-load")):
        if required not in source:
            raise ValueError(f"missing public lifecycle operation: {required}")
    if not (producer.index("checkpoint-load input") <
            producer.index("checkpoint-save!") <
            producer.index("trainer-state-release! owner")):
        raise ValueError("accepted LOAD/SAVE/release order changed")
    if "checkpoint-save!" in control or "trainer-state-release! owner" not in control:
        raise ValueError("missing-SAVE control must load and release only")

    sequence = (
        "scripts/build-c2.sh", "compile_public no-save", "compile_public producer",
        'test ! -e "$saved" && test ! -L "$saved"',
        '"$out/no-save/program"',
        'die "missing-SAVE control unexpectedly produced the image"',
        '"$out/producer/program"',
        "C2 PUBLIC RUNTIME PASS: 10 checks",
        '(( size <= 1048576 ))',
        'cmp "$out/seed/valid.c2" "$saved"',
        'prepare_pair_file_inputs.py',
        '"$directory/pair-file-reader" "$saved" "$out/companions"',
        'cmp "$out/pair-before-reader.sha256" "$out/pair-after-reader.sha256"',
    )
    positions = [gate.find(marker) for marker in sequence]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("SAVE provenance and private-reader gate order changed")
    for marker in ("et_e1b_public_c2_checkpoint_save_v1",
                   "et_c2_private_checkpoint_load_stage_with_sha_v1",
                   '"$out/producer/program.ll" >"$out/producer/save-call-ir.txt"',
                   '"$out/no-save/program.ll" >"$out/no-save/linked-facade-save-ir.txt"',
                   'build_reader normal', 'build_reader sanitize',
                   'cmp "$out/private-normal/run.stdout" "$out/private-normal/repeat.stdout"',
                   'cmp "$out/private-normal/run.stdout" "$out/private-sanitize/run.stdout"'):
        if marker not in gate:
            raise ValueError(f"missing SAVE/reader evidence: {marker}")


class AuthenticSaveGateSourceTest(unittest.TestCase):
    def test_contract(self):
        verify(GATE.read_text(), PRODUCER.read_text(), CONTROL.read_text())

    def test_missing_save_and_freshness_mutations(self):
        gate, producer, control = (path.read_text() for path in
                                   (GATE, PRODUCER, CONTROL))
        for marker in ('test ! -e "$saved" && test ! -L "$saved"',
                       'die "missing-SAVE control unexpectedly produced the image"',
                       'cmp "$out/seed/valid.c2" "$saved"'):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                verify(gate.replace(marker, "", 1), producer, control)
        with self.assertRaises(ValueError):
            verify(gate, producer, control.replace("checkpoint-load", "load", 1))
        with self.assertRaises(ValueError):
            verify(gate, producer.replace("checkpoint-save!", "save", 1), control)


if __name__ == "__main__":
    unittest.main()
