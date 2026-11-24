"""Independent Python binary64 reference oracle for Unigram Viterbi & EM Trainer."""
from __future__ import annotations

import hashlib
import math
import os
import struct
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

MAGIC = b"ETUG1000"
VERSION_MAJOR = 1
VERSION_MINOR = 0
TOKEN_NONE = 0xFFFFFFFF
MSTEP_MLE = 0
MSTEP_DIGAMMA_VB = 1


def hex_to_f64(h: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def f64_to_hex(v: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', v))[0]:016x}"


def log_add_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def log_sum_exp_array(arr: list[float]) -> float:
    max_v = -math.inf
    for x in arr:
        if x > max_v:
            max_v = x
    if not math.isfinite(max_v):
        return max_v
    sum_exp = 0.0
    for x in arr:
        sum_exp += math.exp(x - max_v)
    return max_v + math.log(sum_exp)


def digamma_f64(x: float) -> float:
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += (
        math.log(x)
        - 0.5 * inv
        - inv2
        * (
            (1.0 / 12.0)
            - inv2
            * (
                (1.0 / 120.0)
                - inv2
                * (
                    (1.0 / 252.0)
                    - inv2
                    * (
                        (1.0 / 240.0)
                        - inv2 * ((1.0 / 132.0) - inv2 * (691.0 / 32760.0))
                    )
                )
            )
        )
    )
    return result


def build_lookup(pieces: list[bytes]) -> tuple[dict[bytes, int], int]:
    lut = {p: i for i, p in enumerate(pieces)}
    max_len = max((len(p) for p in pieces), default=1)
    return lut, max_len


def ref_viterbi(
    pieces: list[bytes],
    theta: list[float],
    data: bytes,
    exclude_id: int = -1,
) -> tuple[list[int], float]:
    n = len(data)
    if n == 0:
        return [], 0.0
    lut, max_len = build_lookup(pieces)
    best_score = [-math.inf] * (n + 1)
    best_tok = [-1] * (n + 1)
    best_len = [0] * (n + 1)
    best_score[0] = 0.0

    for pos in range(n):
        base = best_score[pos]
        if base == -math.inf:
            continue
        limit = min(max_len, n - pos)
        for plen in range(1, limit + 1):
            tok = lut.get(data[pos : pos + plen], -1)
            if tok < 0 or tok == exclude_id:
                continue
            nxt = pos + plen
            cand = base + theta[tok]
            better = False
            if best_tok[nxt] < 0:
                better = True
            elif cand > best_score[nxt]:
                better = True
            elif cand == best_score[nxt]:
                if plen > best_len[nxt] or (
                    plen == best_len[nxt] and tok < best_tok[nxt]
                ):
                    better = True
            if better:
                best_score[nxt] = cand
                best_tok[nxt] = tok
                best_len[nxt] = plen

    toks: list[int] = []
    cur = n
    while cur > 0:
        toks.append(best_tok[cur])
        cur -= best_len[cur]
    toks.reverse()
    return toks, best_score[n]


def ref_fb_one(
    lut: dict[bytes, int],
    max_len: int,
    theta: list[float],
    data: bytes,
    acc_exp: list[float] | None,
) -> float:
    n = len(data)
    if n == 0:
        return 0.0
    alpha = [-math.inf] * (n + 1)
    alpha[0] = 0.0
    for pos in range(n):
        a_pos = alpha[pos]
        if a_pos == -math.inf:
            continue
        limit = min(max_len, n - pos)
        for plen in range(1, limit + 1):
            tok = lut.get(data[pos : pos + plen], -1)
            if tok < 0:
                continue
            nxt = pos + plen
            alpha[nxt] = log_add_exp(alpha[nxt], a_pos + theta[tok])
    log_z = alpha[n]

    if acc_exp is not None:
        beta = [-math.inf] * (n + 1)
        beta[n] = 0.0
        for p in range(n - 1, -1, -1):
            limit = min(max_len, n - p)
            for plen in range(1, limit + 1):
                tok = lut.get(data[p : p + plen], -1)
                if tok < 0:
                    continue
                nxt = p + plen
                if beta[nxt] == -math.inf:
                    continue
                beta[p] = log_add_exp(beta[p], theta[tok] + beta[nxt])
        for pos in range(n):
            a_pos = alpha[pos]
            if a_pos == -math.inf:
                continue
            limit = min(max_len, n - pos)
            for plen in range(1, limit + 1):
                tok = lut.get(data[pos : pos + plen], -1)
                if tok < 0:
                    continue
                nxt = pos + plen
                if beta[nxt] == -math.inf:
                    continue
                acc_exp[tok] += math.exp(a_pos + theta[tok] + beta[nxt] - log_z)

    return log_z


def ref_corpus_estep(
    pieces: list[bytes],
    theta: list[float],
    docs: list[bytes],
    need_expected: bool,
) -> tuple[float, list[float] | None]:
    lut, max_len = build_lookup(pieces)
    exp_counts = [0.0] * len(pieces) if need_expected else None
    ll = 0.0
    for doc in docs:
        if len(doc) == 0:
            continue
        ll += ref_fb_one(lut, max_len, theta, doc, exp_counts)
    return ll, exp_counts


def ref_mstep(
    expected: list[float], mstep_mode: int, smoothing_alpha: float
) -> list[float]:
    sum_c = 0.0
    for c in expected:
        sum_c += c + smoothing_alpha
    if mstep_mode == MSTEP_MLE:
        log_sum = math.log(sum_c)
        theta = [math.log(c + smoothing_alpha) - log_sum for c in expected]
    else:
        psi_sum = digamma_f64(sum_c)
        theta = [digamma_f64(c + smoothing_alpha) - psi_sum for c in expected]
    lse = log_sum_exp_array(theta)
    return [t - lse for t in theta]


def ref_train_unigram(
    docs: list[bytes],
    target_vocab_size: int,
    seed_vocab_size: int,
    max_piece_bytes: int,
    min_frequency: int,
    num_em_sub_iters: int,
    final_em_iters: int,
    mstep_mode: int,
    special_mask: int,
    shrinking_factor: float,
    smoothing_alpha: float,
) -> tuple[dict[str, int], dict[str, float | int], list[bytes], list[float]]:
    special_bits = [1, 2, 4, 8]
    special_count = sum(1 for b in special_bits if special_mask & b)
    target_active = target_vocab_size - special_count

    byte_counts = [0] * 256
    sub_counts: Counter[bytes] = Counter()
    for d in docs:
        n = len(d)
        for i in range(n):
            byte_counts[d[i]] += 1
            if target_active > 256:
                max_l = min(max_piece_bytes, n - i)
                for l in range(2, max_l + 1):
                    sub_counts[d[i : i + l]] += 1

    cands = [
        (cnt * len(sub), cnt, len(sub), sub)
        for sub, cnt in sub_counts.items()
        if cnt >= min_frequency
    ]
    cands.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    cands = cands[: seed_vocab_size - 256]

    pieces: list[bytes] = [bytes([b]) for b in range(256)] + [c[3] for c in cands]
    raw_counts = [byte_counts[b] for b in range(256)] + [c[1] for c in cands]
    weights = [float(rc) + smoothing_alpha for rc in raw_counts]
    total_w = 0.0
    for w in weights:
        total_w += w
    log_tw = math.log(total_w)
    theta = [math.log(w) - log_tw for w in weights]
    lse = log_sum_exp_array(theta)
    theta = [t - lse for t in theta]

    seed_active_count = len(pieces)
    pruning_rounds = 0
    total_em_iters = 0
    initial_ll, _ = ref_corpus_estep(pieces, theta, docs, False)

    while len(pieces) > target_active:
        pruning_rounds += 1
        for _ in range(num_em_sub_iters):
            _, expected = ref_corpus_estep(pieces, theta, docs, True)
            assert expected is not None
            theta = ref_mstep(expected, mstep_mode, smoothing_alpha)
            total_em_iters += 1

        _, expected = ref_corpus_estep(pieces, theta, docs, True)
        assert expected is not None

        cur_extra = len(pieces) - 256
        target_extra = target_active - 256
        desired_extra = int(math.floor(float(cur_extra) * shrinking_factor))
        if desired_extra >= cur_extra:
            desired_extra = cur_extra - 1
        if desired_extra < target_extra:
            desired_extra = target_extra

        prune_entries = []
        for k in range(cur_extra):
            tid = 256 + k
            _, alt_score = ref_viterbi(pieces, theta, pieces[tid], exclude_id=tid)
            delta_l = expected[tid] * (theta[tid] - alt_score)
            prune_entries.append((delta_l, expected[tid], len(pieces[tid]), pieces[tid], theta[tid]))

        prune_entries.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        kept = prune_entries[:desired_extra]
        pieces = [bytes([b]) for b in range(256)] + [e[3] for e in kept]
        theta = theta[:256] + [e[4] for e in kept]
        lse = log_sum_exp_array(theta)
        theta = [t - lse for t in theta]

    final_em_start_ll = 0.0
    for it in range(final_em_iters):
        iter_ll, expected = ref_corpus_estep(pieces, theta, docs, True)
        assert expected is not None
        if it == 0:
            final_em_start_ll = iter_ll
        theta = ref_mstep(expected, mstep_mode, smoothing_alpha)
        total_em_iters += 1

    learned = [
        (theta[256 + k], len(pieces[256 + k]), pieces[256 + k])
        for k in range(len(pieces) - 256)
    ]
    learned.sort(key=lambda item: (-item[0], item[1], item[2]))
    pieces = [bytes([b]) for b in range(256)] + [e[2] for e in learned]
    theta = theta[:256] + [e[0] for e in learned]
    lse = log_sum_exp_array(theta)
    theta = [t - lse for t in theta]

    final_em_end_ll, _ = ref_corpus_estep(pieces, theta, docs, False)

    active_count = len(pieces)
    learned_count = active_count - 256
    vocab_size = active_count + special_count
    next_sp = active_count
    sp_ids = []
    for b in special_bits:
        if special_mask & b:
            sp_ids.append(next_sp)
            next_sp += 1
        else:
            sp_ids.append(TOKEN_NONE)

    info = {
        "vocab_size": vocab_size,
        "active_count": active_count,
        "learned_count": learned_count,
        "special_mask": special_mask,
        "bos_id": sp_ids[0],
        "eos_id": sp_ids[1],
        "pad_id": sp_ids[2],
        "unk_id": sp_ids[3],
    }
    stats: dict[str, float | int] = {
        "seed_active_count": seed_active_count,
        "final_active_count": active_count,
        "pruning_rounds": pruning_rounds,
        "total_em_iters": total_em_iters,
        "initial_ll": initial_ll,
        "final_em_start_ll": final_em_start_ll,
        "final_em_end_ll": final_em_end_ll,
    }
    return info, stats, pieces, theta


def ref_serialize_etug(
    info: dict[str, int], pieces: list[bytes], theta: list[float]
) -> tuple[bytes, str]:
    body = b"".join(
        struct.pack("<dH", lp, len(p)) + p for p, lp in zip(pieces, theta)
    )
    header = MAGIC + struct.pack(
        "<HHIIIIIIIII",
        VERSION_MAJOR,
        VERSION_MINOR,
        0,
        info["vocab_size"],
        info["active_count"],
        info["special_mask"],
        info["bos_id"],
        info["eos_id"],
        info["pad_id"],
        info["unk_id"],
        len(body),
    )
    payload = header + body
    digest = hashlib.sha256(payload).digest()
    return payload + digest, digest.hex()


class TestUnigramViterbiOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bin_path = os.environ.get("UNIGRAM_TOK_NATIVE_BIN")
        if not bin_path or not Path(bin_path).is_file():
            raise RuntimeError(
                "UNIGRAM_TOK_NATIVE_BIN must point to compiled test_unigram_tok_native"
            )
        cls.native_bin = bin_path

    def _run_case(
        self,
        docs: list[bytes],
        probe: bytes,
        target_vocab: int,
        seed_vocab: int,
        max_piece: int,
        min_freq: int,
        sub_iters: int,
        final_iters: int,
        mstep_mode: int,
        special_mask: int,
        shrink: float,
        alpha: float,
    ) -> None:
        ref_info, ref_stats, ref_pieces, ref_theta = ref_train_unigram(
            docs,
            target_vocab,
            seed_vocab,
            max_piece,
            min_freq,
            sub_iters,
            final_iters,
            mstep_mode,
            special_mask,
            shrink,
            alpha,
        )
        ref_tokens, ref_vscore = ref_viterbi(ref_pieces, ref_theta, probe)
        lut, max_len = build_lookup(ref_pieces)
        ref_exp = [0.0] * len(ref_pieces)
        ref_log_z = ref_fb_one(lut, max_len, ref_theta, probe, ref_exp)
        ref_artifact, ref_sha = ref_serialize_etug(ref_info, ref_pieces, ref_theta)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            docs_path = tmp_dir / "docs.bin"
            probe_path = tmp_dir / "probe.bin"
            art_path = tmp_dir / "model.etug"

            docs_blob = struct.pack("<I", len(docs)) + b"".join(
                struct.pack("<I", len(d)) + d for d in docs
            )
            docs_path.write_bytes(docs_blob)
            probe_path.write_bytes(probe)

            proc = subprocess.run(
                [
                    self.native_bin,
                    "--oracle-cli",
                    str(docs_path),
                    str(probe_path),
                    str(target_vocab),
                    str(seed_vocab),
                    str(max_piece),
                    str(min_freq),
                    str(sub_iters),
                    str(final_iters),
                    str(mstep_mode),
                    str(special_mask),
                    repr(shrink),
                    repr(alpha),
                    str(art_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            c_info: dict[str, int] = {}
            c_stats: dict[str, int | float] = {}
            c_pieces: list[bytes] = []
            c_theta_hex: list[str] = []
            c_tokens: list[int] = []
            c_vscore_hex = ""
            c_decoded = b""
            c_log_z_hex = ""
            c_exp: dict[int, float] = {}
            c_sha = ""

            for line in proc.stdout.splitlines():
                parts = line.split("\t")
                tag = parts[0]
                if tag == "INFO":
                    keys = [
                        "vocab_size",
                        "active_count",
                        "learned_count",
                        "special_mask",
                        "bos_id",
                        "eos_id",
                        "pad_id",
                        "unk_id",
                    ]
                    c_info = {k: int(v) for k, v in zip(keys, parts[1:])}
                elif tag == "STATS":
                    c_stats = {
                        "seed_active_count": int(parts[1]),
                        "final_active_count": int(parts[2]),
                        "pruning_rounds": int(parts[3]),
                        "total_em_iters": int(parts[4]),
                        "initial_ll": hex_to_f64(parts[5]),
                        "final_em_start_ll": hex_to_f64(parts[6]),
                        "final_em_end_ll": hex_to_f64(parts[7]),
                    }
                elif tag == "PIECE":
                    tid = int(parts[1])
                    self.assertEqual(tid, len(c_pieces))
                    c_theta_hex.append(parts[2])
                    c_pieces.append(bytes.fromhex(parts[3]) if len(parts) > 3 else b"")
                elif tag == "TOKENS":
                    c_tokens = [int(x) for x in parts[1:] if x != ""]
                elif tag == "VSCORE":
                    c_vscore_hex = parts[1]
                elif tag == "DECODED":
                    c_decoded = bytes.fromhex(parts[1]) if len(parts) > 1 else b""
                elif tag == "FBLOGZ":
                    c_log_z_hex = parts[1]
                elif tag == "FBEXP":
                    c_exp[int(parts[1])] = hex_to_f64(parts[2])
                elif tag == "SHA256":
                    c_sha = parts[1]

            self.assertEqual(c_info, ref_info, "info mismatch")
            self.assertEqual(c_stats["seed_active_count"], ref_stats["seed_active_count"])
            self.assertEqual(c_stats["final_active_count"], ref_stats["final_active_count"])
            self.assertEqual(c_stats["pruning_rounds"], ref_stats["pruning_rounds"])
            self.assertEqual(c_stats["total_em_iters"], ref_stats["total_em_iters"])
            self.assertEqual(c_pieces, ref_pieces, "trained piece bytes mismatch")
            self.assertEqual(
                c_theta_hex,
                [f64_to_hex(t) for t in ref_theta],
                "piece binary64 log_probs bitwise mismatch",
            )
            self.assertEqual(c_tokens, ref_tokens, "Viterbi token IDs mismatch")
            self.assertEqual(c_vscore_hex, f64_to_hex(ref_vscore), "Viterbi score mismatch")
            self.assertEqual(c_decoded, probe, "lossless decode mismatch")
            self.assertEqual(c_log_z_hex, f64_to_hex(ref_log_z), "Forward-Backward log Z mismatch")
            for tid, val in enumerate(ref_exp):
                self.assertEqual(
                    f64_to_hex(c_exp.get(tid, 0.0)),
                    f64_to_hex(val),
                    f"expected count mismatch for piece {tid}",
                )
            self.assertEqual(c_sha, ref_sha, "ETUG1000 SHA-256 mismatch")
            self.assertEqual(
                art_path.read_bytes(),
                ref_artifact,
                "ETUG1000 binary artifact bytes mismatch",
            )

    def test_mle_english_compression(self) -> None:
        docs = [
            b"eshkol transformer viterbi lattice unigram tokenizer checkpoint " * 4,
            b"forward backward expectation maximization subword pruning " * 4,
            b"deterministic binary64 log probability normalization sha256 " * 3,
        ]
        probe = b"eshkol transformer viterbi unigram tokenizer expectation maximization"
        self._run_case(
            docs,
            probe,
            target_vocab=320,
            seed_vocab=1024,
            max_piece=14,
            min_freq=2,
            sub_iters=2,
            final_iters=4,
            mstep_mode=MSTEP_MLE,
            special_mask=0xF,
            shrink=0.75,
            alpha=1e-3,
        )

    def test_digamma_vb_mode(self) -> None:
        docs = [
            b"the quick brown fox jumps over the lazy dog " * 5,
            b"unigram language model sentencepiece digamma variational bayes " * 4,
        ]
        probe = b"the quick brown fox sentencepiece digamma unigram model"
        self._run_case(
            docs,
            probe,
            target_vocab=300,
            seed_vocab=768,
            max_piece=12,
            min_freq=2,
            sub_iters=2,
            final_iters=4,
            mstep_mode=MSTEP_DIGAMMA_VB,
            special_mask=0x3,
            shrink=0.75,
            alpha=0.5,
        )

    def test_nul_utf8_and_all_256_bytes(self) -> None:
        all_bytes = bytes(range(256))
        docs = [
            b"\x00\x01\x02\x00\x01\x02\xff\xfe\xff\xfe" * 4,
            "café 日本語 🚀 eshkol 日本語 🚀 ".encode("utf-8") * 4,
            all_bytes,
            b"",
        ]
        probe = b"\x00" + all_bytes + "日本語🚀".encode("utf-8") + b"\x00\xff\xfe"
        self._run_case(
            docs,
            probe,
            target_vocab=296,
            seed_vocab=640,
            max_piece=12,
            min_freq=2,
            sub_iters=2,
            final_iters=3,
            mstep_mode=MSTEP_MLE,
            special_mask=0x0,
            shrink=0.75,
            alpha=1e-3,
        )

    def test_byte_only_vocab(self) -> None:
        docs = [b"abcdefg", b"hijklmn"]
        self._run_case(
            docs,
            b"abcdefg",
            target_vocab=256,
            seed_vocab=256,
            max_piece=8,
            min_freq=1,
            sub_iters=1,
            final_iters=2,
            mstep_mode=MSTEP_MLE,
            special_mask=0x0,
            shrink=0.75,
            alpha=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
