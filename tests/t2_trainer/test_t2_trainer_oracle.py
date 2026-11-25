"""Independent Python reference BPE trainer oracle for T2 native C11 parity."""
from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tests.t2 import reference as accepted_t2

MAGIC = b"ETBPE100"
VERSION_MAJOR = 1
VERSION_MINOR = 0
TOKEN_NONE = 0xFFFFFFFF
MAX_TOKEN_BYTES = 256


def ref_train_bpe(
    docs: list[bytes],
    target_vocab_size: int,
    min_frequency: int,
    special_mask: int,
) -> tuple[dict[str, int], list[tuple[int, int, int]]]:
    """Pure-Python independent reference BPE trainer."""
    special_bits = [1, 2, 4, 8]
    special_count = sum(1 for b in special_bits if special_mask & b)
    max_merges = target_vocab_size - 256 - special_count
    assert max_merges >= 0

    seqs = [[b for b in d] for d in docs]
    tok_lens: dict[int, int] = {b: 1 for b in range(256)}
    merges: list[tuple[int, int, int]] = []

    for rank in range(max_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for seq in seqs:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1

        candidates = [
            (cnt, left, right)
            for (left, right), cnt in counts.items()
            if cnt >= min_frequency
            and tok_lens[left] + tok_lens[right] <= MAX_TOKEN_BYTES
        ]
        if not candidates:
            break
        # Maximize count, tie-break by lowest lexicographic (left, right).
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, best_left, best_right = candidates[0]
        new_id = 256 + rank
        merges.append((best_left, best_right, new_id))
        tok_lens[new_id] = tok_lens[best_left] + tok_lens[best_right]

        new_seqs = []
        for seq in seqs:
            out: list[int] = []
            i = 0
            n = len(seq)
            while i < n:
                if i + 1 < n and seq[i] == best_left and seq[i + 1] == best_right:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_seqs.append(out)
        seqs = new_seqs

    merge_count = len(merges)
    vocab_size = 256 + merge_count + special_count
    next_sp = 256 + merge_count
    ids = []
    for b in special_bits:
        if special_mask & b:
            ids.append(next_sp)
            next_sp += 1
        else:
            ids.append(TOKEN_NONE)

    info = {
        "vocab_size": vocab_size,
        "merge_count": merge_count,
        "special_mask": special_mask,
        "bos_id": ids[0],
        "eos_id": ids[1],
        "pad_id": ids[2],
        "unk_id": ids[3],
    }
    return info, merges


def ref_encode_bpe(data: bytes, merges: list[tuple[int, int, int]]) -> list[int]:
    """Direct rank-order sweep BPE encode (independent algorithm from C hash loop)."""
    seq = list(data)
    for left, right, new_id in merges:
        if len(seq) < 2:
            break
        out: list[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == left and seq[i + 1] == right:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return seq


def ref_serialize_artifact(
    info: dict[str, int], merges: list[tuple[int, int, int]]
) -> tuple[bytes, str]:
    header = MAGIC + struct.pack(
        "<HHIIIIIIII",
        VERSION_MAJOR,
        VERSION_MINOR,
        0,
        info["vocab_size"],
        info["merge_count"],
        info["special_mask"],
        info["bos_id"],
        info["eos_id"],
        info["pad_id"],
        info["unk_id"],
    )
    body = b"".join(struct.pack("<III", l, r, nid) for l, r, nid in merges)
    payload = header + body
    digest = hashlib.sha256(payload).digest()
    return payload + digest, digest.hex()


class TestT2BpeTrainerOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bin_path = os.environ.get("T2_TRAINER_NATIVE_BIN")
        if not bin_path or not Path(bin_path).is_file():
            raise RuntimeError(
                "T2_TRAINER_NATIVE_BIN must point to compiled test_t2_trainer_native"
            )
        cls.native_bin = bin_path

    def _run_case(
        self,
        docs: list[bytes],
        probe: bytes,
        target_vocab: int,
        min_freq: int,
        special_mask: int,
    ) -> bytes:
        ref_info, ref_merges = ref_train_bpe(
            docs, target_vocab, min_freq, special_mask
        )
        ref_tokens = ref_encode_bpe(probe, ref_merges)
        ref_artifact, ref_sha = ref_serialize_artifact(ref_info, ref_merges)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            blob_path = tmp_dir / "docs.bin"
            probe_path = tmp_dir / "probe.bin"
            art_path = tmp_dir / "out.etbpe"

            blob = struct.pack("<I", len(docs)) + b"".join(
                struct.pack("<I", len(d)) + d for d in docs
            )
            blob_path.write_bytes(blob)
            probe_path.write_bytes(probe)

            proc = subprocess.run(
                [
                    self.native_bin,
                    "--oracle-cli",
                    str(blob_path),
                    str(probe_path),
                    str(target_vocab),
                    str(min_freq),
                    str(special_mask),
                    str(art_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            c_info: dict[str, int] = {}
            c_merges: list[tuple[int, int, int]] = []
            c_tokens: list[int] = []
            c_decoded = b""
            c_sha = ""
            for line in proc.stdout.splitlines():
                parts = line.split("\t")
                tag = parts[0]
                if tag == "INFO":
                    keys = [
                        "vocab_size",
                        "merge_count",
                        "special_mask",
                        "bos_id",
                        "eos_id",
                        "pad_id",
                        "unk_id",
                    ]
                    c_info = {k: int(v) for k, v in zip(keys, parts[1:])}
                elif tag == "MERGE":
                    rank, l, r, nid = (int(x) for x in parts[1:])
                    self.assertEqual(rank, len(c_merges))
                    c_merges.append((l, r, nid))
                elif tag == "TOKENS":
                    c_tokens = [int(x) for x in parts[1:] if x != ""]
                elif tag == "DECODED":
                    c_decoded = bytes.fromhex(parts[1]) if len(parts) > 1 else b""
                elif tag == "SHA256":
                    c_sha = parts[1]

            self.assertEqual(c_info, ref_info, "model info mismatch")
            self.assertEqual(c_merges, ref_merges, "merge table mismatch")
            self.assertEqual(c_tokens, ref_tokens, "encoded token sequence mismatch")
            self.assertEqual(c_decoded, probe, "lossless decode mismatch")
            self.assertEqual(c_sha, ref_sha, "artifact SHA-256 mismatch")
            native_artifact = art_path.read_bytes()
            self.assertEqual(
                native_artifact,
                ref_artifact,
                "ETBPE100 binary artifact bytes mismatch",
            )

            # The admitted raw-byte subset must produce the same merge IDs and
            # ordinary encode result as the accepted Wave-2 T2 reference.
            maximum_merges = target_vocab - 256 - special_mask.bit_count()
            if maximum_merges <= accepted_t2.MAX_MERGES:
                t2_core = accepted_t2.train(
                    [[doc] for doc in docs], maximum_merges, min_freq, "raw"
                )
                self.assertEqual(
                    c_merges,
                    [(m.left_id, m.right_id, m.result_id) for m in t2_core.merges],
                    "accepted T2 merge table mismatch",
                )
                self.assertEqual(
                    c_tokens, list(accepted_t2.encode_bytes(t2_core, probe)),
                    "accepted T2 raw encode mismatch",
                )
            with self.assertRaises(accepted_t2.FormatError):
                accepted_t2.decode_artifact(native_artifact)
            return native_artifact

    def test_structured_english_and_compression(self) -> None:
        docs = [
            b"the quick brown fox jumps over the lazy dog " * 6,
            b"deterministic byte pair encoding in eshkol transformer " * 5,
            b"eshkol transformer attention rmsnorm swiglu checkpoint " * 4,
        ]
        probe = b"the deterministic eshkol transformer byte pair encoding checkpoint"
        self._run_case(docs, probe, target_vocab=340, min_freq=2, special_mask=0xF)

    def test_canonical_tie_breaking_stress(self) -> None:
        # Deliberately symmetric equal-frequency pairs in reverse order.
        docs = [
            b"zy", b"zy", b"yx", b"yx", b"ab", b"ab", b"ac", b"ac",
            b"aa", b"aa", b"ba", b"ba", b"aaa", b"aaa",
        ]
        probe = b"zyyxabacaabaaaa"
        self._run_case(docs, probe, target_vocab=280, min_freq=2, special_mask=0x3)

    def test_nul_utf8_and_all_256_bytes(self) -> None:
        all_bytes = bytes(range(256))
        docs = [
            b"\x00\x01\x02\x00\x01\x02\xff\xfe\xff\xfe" * 4,
            "café 日本語 🚀🧪 eshkol 日本語 🚀🧪 ".encode("utf-8") * 5,
            all_bytes,
            b"",
        ]
        probe = b"\x00" + all_bytes + "日本語🚀".encode("utf-8") + b"\x00\xff\xfe"
        self._run_case(docs, probe, target_vocab=320, min_freq=2, special_mask=0x0)

    def test_early_stop_and_zero_merges(self) -> None:
        docs = [b"abcdefg", b"hijklmn"]
        self._run_case(docs, b"abcdefg", target_vocab=512, min_freq=2, special_mask=0x5)
        self._run_case(docs, b"abcdefg", target_vocab=256, min_freq=1, special_mask=0x0)

    def test_document_order_does_not_change_artifact(self) -> None:
        docs = [b"banana bandana" * 3, b"ananas" * 4, b"band" * 5, b""]
        forward = self._run_case(docs, b"banana bandana", 300, 2, 0)
        reverse = self._run_case(list(reversed(docs)), b"banana bandana", 300, 2, 0)
        self.assertEqual(forward, reverse)


if __name__ == "__main__":
    unittest.main()
