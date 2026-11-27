import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from tests.eval5 import mod5_reference_oracle as oracle
from tests.mod5 import mod5_v256_decoder_adamw as witness
from tests.mod5.mod5_v256_c2_fixture import DOCUMENTS, verify as verify_training
from tests.mod5.mod5_v256_gradient import loss, section, source_for


class DecoderAdamWTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        witness.generate(self.root)

    def test_generated_decoder_and_tied_backward_are_reused(self):
        generated = (self.root / "witness.esk").read_text()
        tied = (self.root / "tied.etcp").read_bytes()
        gradient = source_for(self.root / "evaluation", witness.evaluation.sha(tied))
        for start, end in (
                ("(define (decoder ", "(define (tensor-dimensions "),
                ("(define (load-checked-weights ", "(define (close-workspace)"),
                ("(define (canonicalize-tied! ", "(define (gradient-witness ")):
            expected = section(gradient, start, end)
            self.assertEqual(generated.count(expected), 1)
        self.assertEqual(generated.count('(load "d2_wave2_root.esk")'), 1)
        self.assertEqual(generated.count('(load "o2_wave2_extension.esk")'), 1)
        self.assertNotIn('(load "i2_wave2_root.esk")', generated)
        self.assertNotIn("@@", generated)
        self.assertNotIn("k2-public", generated)
        self.assertNotIn("c2-public", generated)

    def synthetic_trace(self, variant):
        tied = variant == "tied"
        count = 11 if tied else 12
        model = "tied.etcp" if tied else "evaluation/model.etcp"
        initial = oracle.weights_from_c1((self.root / model).read_bytes())
        pre = tuple(value for row in initial[:count] for value in row)
        average = square = (0.0,) * len(pre)
        bits = bytearray(witness.pack(pre))
        lines = [f"PROFILE {variant} {count}"]
        cursors = verify_training(self.root / "training")["oracle_cursor_sha256_by_next_ordinal"]
        for ordinal, doc in enumerate(DOCUMENTS):
            lines.append(f"CURSOR {ordinal} {cursors[ordinal]}")
            grad = tuple((i % 17 - 8) / 4096 for i in range(len(pre)))
            post, average, square = zip(*(witness.adamw(*row, ordinal + 1) for row in
                                         zip(pre, grad, average, square, strict=True)), strict=True)
            before_loss, after_loss = (loss(witness.roster(values, tied), doc) for values in (pre, post))
            lines.append(f"ROW {ordinal} {doc[:-1].hex()} {doc[1:].hex()} 8 {before_loss} {after_loss}")
            for plane in (grad, post, average, square):
                bits.extend(witness.pack(plane))
            pre = post
        lines.extend((f"CURSOR 6 {cursors[6]}", witness.SUCCESS))
        transcript, trace = self.root / f"{variant}.stdout", self.root / f"{variant}.bits"
        transcript.write_text("\n".join(lines) + "\n")
        trace.write_bytes(bits)
        return transcript, trace

    def test_evidence_oracle_accepts_both_rosters_and_rejects_bad_moment_bits(self):
        for variant in ("untied", "tied"):
            with self.subTest(variant=variant), contextlib.redirect_stdout(io.StringIO()):
                transcript, trace = self.synthetic_trace(variant)
                witness.verify(self.root, variant, transcript, trace)
                bits = bytearray(trace.read_bytes())
                bits[-1] ^= 1
                trace.write_bytes(bits)
                with self.assertRaisesRegex(ValueError, "update 6 exp_avg_sq"):
                    witness.verify(self.root, variant, transcript, trace)

    def test_evidence_oracle_rejects_incomplete_mask_and_cursor_drift(self):
        transcript, trace = self.synthetic_trace("tied")
        original = transcript.read_text().splitlines(keepends=True)
        for index, previous, replacement, message in (
                (1, "CURSOR 0 ", "CURSOR 1 ", "incorrect live D2 cursor"),
                (2, " 8 ", " 7 ", "incorrect live D2 tokens or loss mask")):
            modified = original.copy()
            modified[index] = modified[index].replace(previous, replacement)
            transcript.write_text("".join(modified))
            with self.assertRaisesRegex(ValueError, message):
                witness.verify(self.root, "tied", transcript, trace)


if __name__ == "__main__":
    unittest.main()
