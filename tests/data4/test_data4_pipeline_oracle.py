"""Independent stdlib Python oracle for the DATA4 corpus pipeline."""
from __future__ import annotations

import hashlib
import struct
import subprocess
import sys
import unittest
from pathlib import Path

DRIVER, DATA_DIR = (Path(arg).resolve() for arg in sys.argv[1:3])

MASK64 = (1 << 64) - 1


def splitmix64_mix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def hash_bytes64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & MASK64
    return splitmix64_mix(h)


def is_printable_cp(cp: int) -> bool:
    if cp in (0x09, 0x0A, 0x0D, 0x20):
        return True
    if 0x21 <= cp <= 0x7E:
        return True
    if cp < 0xA0:
        return False
    if 0xFDD0 <= cp <= 0xFDEF:
        return False
    if (cp & 0xFFFE) == 0xFFFE:
        return False
    return True


def is_alnum_cp(cp: int) -> bool:
    if (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A) or (0x30 <= cp <= 0x39):
        return True
    if (0x00C0 <= cp <= 0x00D6) or (0x00D8 <= cp <= 0x00F6) or (0x00F8 <= cp <= 0x024F):
        return True
    if (0x0370 <= cp <= 0x03FF) or (0x0400 <= cp <= 0x04FF):
        return True
    if (0x3040 <= cp <= 0x30FF) or (0x4E00 <= cp <= 0x9FFF) or (0xAC00 <= cp <= 0xD7A3):
        return True
    return False


def validate_utf8_strict(raw: bytes) -> tuple[bool, int, int, int]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False, 0, 0, 0
    cps = [ord(ch) for ch in text]
    return (
        True,
        len(cps),
        sum(1 for cp in cps if is_printable_cp(cp)),
        sum(1 for cp in cps if is_alnum_cp(cp)),
    )


def dup_line_ratio(raw: bytes) -> float:
    if not raw:
        return 0.0
    lines = raw[:-1].split(b"\n") if raw.endswith(b"\n") else raw.split(b"\n")
    return (len(lines) - len(set(lines))) / len(lines)


def byte_ngrams(raw: bytes, n: int) -> list[bytes]:
    if len(raw) < n:
        return []
    return [raw[i : i + n] for i in range(len(raw) - n + 1)]


def repeated_ngram_ratio(raw: bytes, n: int) -> float:
    grams = byte_ngrams(raw, n)
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def compute_minhash(shingles: set[bytes], num_perms: int, seed: int) -> list[int]:
    if not shingles:
        return [MASK64] * num_perms
    h0s = [hash_bytes64(s) for s in shingles]
    sig = [MASK64] * num_perms
    for k in range(num_perms):
        a = splitmix64_mix(seed ^ (k * 2 + 1)) | 1
        b = splitmix64_mix(seed ^ (k * 2 + 2))
        sig[k] = min(splitmix64_mix((a * h + b) & MASK64) for h in h0s)
    return sig


def f64_hex(v: float) -> str:
    return struct.pack(">d", v).hex()


def seal_report(payload: bytes) -> bytes:
    digest = hashlib.sha256(payload).hexdigest().encode("ascii")
    return payload + b"checksum\t" + digest + b"\n"


ORACLE_CONFIG = {
    "min_bytes": 24,
    "max_bytes": 512,
    "min_codepoints": 20,
    "max_codepoints": 512,
    "min_printable_ratio": 0.90,
    "min_alnum_ratio": 0.50,
    "max_dup_line_ratio": 0.25,
    "max_repeated_ngram_ratio": 0.35,
    "rep_ngram_size": 5,
    "near_dup_enabled": 1,
    "dedup_ngram_size": 5,
    "minhash_num_perms": 64,
    "minhash_bands": 16,
    "minhash_seed": 0x4441544134534545,
    "near_dup_jaccard_threshold": 0.75,
}

D0 = (
    b"Eshkol transformer training requires deterministic corpus filtering and "
    b"exact provenance verification across shards."
)
ORACLE_CORPUS: list[tuple[str, bytes]] = [
    ("src.clean.0", D0),
    ("src.bad_utf8.1", b"bad utf8 \xed\xa0\x80 surrogate text"),
    ("src.short.2", b"too short doc"),
    ("src.symbols.3", b"!!!@@@###$$$%%%^^^&&&***((()))___+++===---///|||~~~"),
    (
        "src.dup_lines.4",
        b"Repeated spam line for testing.\n"
        b"Repeated spam line for testing.\n"
        b"Repeated spam line for testing.\n"
        b"One distinct closing sentence.\n",
    ),
    (
        "src.rep_ngram.5",
        b"abcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcde1234",
    ),
    ("src.exact_dup.6", D0),
    (
        "src.near_dup.7",
        b"Eshkol transformer training requires deterministic corpus filtering and "
        b"exact provenance verification across shards!",
    ),
    (
        "src.utf8_multi.8",
        "Multilingual UTF-8 corpus sample: café naïve 日本語 привет "
        "with high alphanumeric density.".encode("utf-8"),
    ),
    (
        "src.distinct.9",
        b"Completely distinct technical note on memory-safe C11 ring buffers, "
        b"checksummed manifests, and reproducible LLVM builds.",
    ),
]


def run_oracle_pipeline(corpus: list[tuple[str, bytes]], cfg: dict) -> tuple[bytes, list[dict]]:
    kept_exact: dict[bytes, str] = {}
    kept_records: list[dict] = []
    bands = cfg["minhash_bands"]
    rows_per_band = cfg["minhash_num_perms"] // bands
    band_buckets: list[dict[tuple[int, ...], list[int]]] = [{} for _ in range(bands)]

    counts = {
        "kept": 0,
        "rejected_utf8": 0,
        "rejected_length": 0,
        "rejected_quality": 0,
        "rejected_repetition": 0,
        "duplicate_exact": 0,
        "duplicate_near": 0,
    }
    input_bytes = sum(len(raw) for _, raw in corpus)
    kept_bytes = 0
    rows: list[dict] = []

    for ordinal, (src_id, raw) in enumerate(corpus):
        digest_hex = hashlib.sha256(raw).hexdigest()
        valid, cp_count, pr_count, al_count = validate_utf8_strict(raw)
        row = {
            "ordinal": ordinal,
            "source_id": src_id,
            "byte_count": len(raw),
            "codepoint_count": cp_count if valid else 0,
            "sha256": digest_hex,
            "disposition": "",
            "matched": "",
            "jnum": 0,
            "jden": 0,
        }
        if not valid:
            row["disposition"] = "rejected_utf8"
            counts["rejected_utf8"] += 1
            rows.append(row)
            continue

        if not (cfg["min_bytes"] <= len(raw) <= cfg["max_bytes"]) or not (
            cfg["min_codepoints"] <= cp_count <= cfg["max_codepoints"]
        ):
            row["disposition"] = "rejected_length"
            counts["rejected_length"] += 1
            rows.append(row)
            continue

        pr_ratio = 0.0 if cp_count == 0 else pr_count / cp_count
        al_ratio = 0.0 if cp_count == 0 else al_count / cp_count
        if pr_ratio < cfg["min_printable_ratio"] or al_ratio < cfg["min_alnum_ratio"]:
            row["disposition"] = "rejected_quality"
            counts["rejected_quality"] += 1
            rows.append(row)
            continue

        if (
            dup_line_ratio(raw) > cfg["max_dup_line_ratio"]
            or repeated_ngram_ratio(raw, cfg["rep_ngram_size"])
            > cfg["max_repeated_ngram_ratio"]
        ):
            row["disposition"] = "rejected_repetition"
            counts["rejected_repetition"] += 1
            rows.append(row)
            continue

        if raw in kept_exact:
            row["disposition"] = "duplicate_exact"
            row["matched"] = kept_exact[raw]
            counts["duplicate_exact"] += 1
            rows.append(row)
            continue

        shingles = set(byte_ngrams(raw, cfg["dedup_ngram_size"]))
        sig = compute_minhash(shingles, cfg["minhash_num_perms"], cfg["minhash_seed"])
        if cfg["near_dup_enabled"] and shingles:
            cands: set[int] = set()
            for b in range(bands):
                key = tuple(sig[b * rows_per_band : (b + 1) * rows_per_band])
                for kidx in band_buckets[b].get(key, ()):
                    cands.add(kidx)
            matched_cand = None
            for kidx in sorted(cands):
                other = kept_records[kidx]["shingles"]
                inter = len(shingles & other)
                uni = len(shingles | other)
                if uni > 0 and (inter / uni) >= cfg["near_dup_jaccard_threshold"]:
                    matched_cand = (kidx, inter, uni)
                    break
            if matched_cand is not None:
                kidx, inter, uni = matched_cand
                row["disposition"] = "duplicate_near"
                row["matched"] = kept_records[kidx]["source_id"]
                row["jnum"] = inter
                row["jden"] = uni
                counts["duplicate_near"] += 1
                rows.append(row)
                continue

        row["disposition"] = "kept"
        counts["kept"] += 1
        kept_bytes += len(raw)
        kept_exact[raw] = src_id
        kidx = len(kept_records)
        kept_records.append({"source_id": src_id, "shingles": shingles, "sig": sig})
        if cfg["near_dup_enabled"] and shingles:
            for b in range(bands):
                key = tuple(sig[b * rows_per_band : (b + 1) * rows_per_band])
                band_buckets[b].setdefault(key, []).append(kidx)
        rows.append(row)

    lines = [
        "format\teshkol-data4-corpus-pipeline-report",
        "version\t1\t0",
        f"min-bytes\t{cfg['min_bytes']}",
        f"max-bytes\t{cfg['max_bytes']}",
        f"min-codepoints\t{cfg['min_codepoints']}",
        f"max-codepoints\t{cfg['max_codepoints']}",
        f"min-printable-ratio-hex\t{f64_hex(cfg['min_printable_ratio'])}",
        f"min-alnum-ratio-hex\t{f64_hex(cfg['min_alnum_ratio'])}",
        f"max-dup-line-ratio-hex\t{f64_hex(cfg['max_dup_line_ratio'])}",
        f"max-repeated-ngram-ratio-hex\t{f64_hex(cfg['max_repeated_ngram_ratio'])}",
        f"rep-ngram-size\t{cfg['rep_ngram_size']}",
        f"near-dup-enabled\t{cfg['near_dup_enabled']}",
        f"dedup-ngram-size\t{cfg['dedup_ngram_size']}",
        f"minhash-num-perms\t{cfg['minhash_num_perms']}",
        f"minhash-bands\t{cfg['minhash_bands']}",
        f"minhash-seed\t{cfg['minhash_seed']}",
        f"near-dup-jaccard-threshold-hex\t{f64_hex(cfg['near_dup_jaccard_threshold'])}",
        f"input-documents\t{len(corpus)}",
        f"kept-documents\t{counts['kept']}",
        f"rejected-utf8-documents\t{counts['rejected_utf8']}",
        f"rejected-length-documents\t{counts['rejected_length']}",
        f"rejected-quality-documents\t{counts['rejected_quality']}",
        f"rejected-repetition-documents\t{counts['rejected_repetition']}",
        f"duplicate-exact-documents\t{counts['duplicate_exact']}",
        f"duplicate-near-documents\t{counts['duplicate_near']}",
        f"input-bytes\t{input_bytes}",
        f"kept-bytes\t{kept_bytes}",
    ]
    for r in rows:
        lines.append(
            f"document\t{r['ordinal']}\t{r['source_id']}\t{r['byte_count']}\t"
            f"{r['codepoint_count']}\t{r['sha256']}\t{r['disposition']}\t"
            f"{r['matched']}\t{r['jnum']}\t{r['jden']}"
        )
    lines.append("checksum-algorithm\tsha256")
    payload = ("\n".join(lines) + "\n").encode("ascii")
    return seal_report(payload), rows


class Data4PipelineOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.actual_report = (DATA_DIR / "report.tsv").read_bytes()
        cls.minhash_lines = (DATA_DIR / "minhash.txt").read_text("ascii").splitlines()

    def verify_status(self, report_bytes: bytes) -> int:
        path = DATA_DIR / "mutant-report.tsv"
        try:
            path.write_bytes(report_bytes)
            out = subprocess.check_output(
                [str(DRIVER), "--verify-report", str(path)], text=True
            )
            return int(out.strip())
        finally:
            path.unlink(missing_ok=True)

    def test_minhash_bit_exact_parity(self) -> None:
        shingles = set(byte_ngrams(D0, ORACLE_CONFIG["dedup_ngram_size"]))
        expected_sig = compute_minhash(
            shingles, ORACLE_CONFIG["minhash_num_perms"], ORACLE_CONFIG["minhash_seed"]
        )
        self.assertEqual(self.minhash_lines[0], f"uniq {len(shingles)}")
        actual_sig = [int(h, 16) for h in self.minhash_lines[1:]]
        self.assertEqual(actual_sig, expected_sig)

    def test_canonical_report_byte_exact_parity(self) -> None:
        expected_report, rows = run_oracle_pipeline(ORACLE_CORPUS, ORACLE_CONFIG)
        self.assertEqual(self.actual_report, expected_report)
        self.assertEqual(rows[0]["disposition"], "kept")
        self.assertEqual(rows[1]["disposition"], "rejected_utf8")
        self.assertEqual(rows[2]["disposition"], "rejected_length")
        self.assertEqual(rows[3]["disposition"], "rejected_quality")
        self.assertEqual(rows[4]["disposition"], "rejected_repetition")
        self.assertEqual(rows[5]["disposition"], "rejected_repetition")
        self.assertEqual(rows[6]["disposition"], "duplicate_exact")
        self.assertEqual(rows[7]["disposition"], "duplicate_near")
        self.assertEqual(rows[7]["matched"], "src.clean.0")
        self.assertGreater(rows[7]["jden"], 0)
        self.assertGreaterEqual(
            rows[7]["jnum"] / rows[7]["jden"],
            ORACLE_CONFIG["near_dup_jaccard_threshold"],
        )
        self.assertEqual(self.verify_status(self.actual_report), 0)

    def test_tampered_and_resealed_invalid_reports(self) -> None:
        payload, sep, _ = self.actual_report.rpartition(b"checksum\t")
        self.assertEqual(sep, b"checksum\t")

        # Raw body tamper without resealing -> ET_DATA4_ERR_CHECKSUM (-8)
        self.assertEqual(
            self.verify_status(self.actual_report.replace(b"src.clean.0", b"src.clean.X", 1)),
            -8,
        )

        # Resealed invalid reports -> explicit format/version error codes.
        cases = [
            (b"version\t1\t0\n", b"version\t2\t0\n", -9),
            (b"format\teshkol-data4-corpus-pipeline-report\n", b"format\twrong\n", -7),
            (b"kept-documents\t3\n", b"kept-documents\t4\n", -7),
            (b"kept-bytes\t", b"kept-bytes\t0", -7),  # leading zero
            (b"src.distinct.9", b"src.clean.0", -7),  # duplicate source_id
            (b"src.clean.0\t0\t0\n", b"src.bad_utf8.1\t0\t0\n", -7),  # non-kept canonical
            (b"duplicate_near\tsrc.clean.0", b"duplicate_near\tsrc.distinct.9", -7),  # forward ref
            (b"min-bytes\t24\n", b"min-bytes\t24\r\n", -7),  # CRLF
            (b"min-printable-ratio-hex\t3feccccccccccccd\n",
             b"min-printable-ratio-hex\t7ff8000000000000\n", -7),  # NaN
        ]
        for before, after, expected_code in cases:
            with self.subTest(before=before, after=after):
                self.assertGreaterEqual(payload.count(before), 1)
                mutated = seal_report(payload.replace(before, after, 1))
                self.assertEqual(self.verify_status(mutated), expected_code)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
