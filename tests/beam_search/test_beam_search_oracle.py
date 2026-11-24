#!/usr/bin/env python3
"""Independent Python binary64 oracle for Deterministic Beam Search & Logit Processors.

Verifies exact bit-level/ULP-tight parity between native C11 and an independent
Python IEEE-754 binary64 reference implementation across:
  1. Composable logit processors (Keskar/CTRL repetition penalty, exact
     no-repeat N-gram blocking, min-length EOS suppression) and log-softmax.
  2. Trap-logits tree proving B=1 exact greedy equivalence and B=4 strictly
     recovering the global optimum (+3.68 nats over greedy).
  3. Wu et al. (2016) length penalty ranking flip (alpha=0.0 vs alpha=1.5).
  4. Six multi-parameter stress cases (B in 1..8, ngram in {0,2,3}, min_new,
     repetition_penalty, length_penalty, early_stopping in {0,1}) verifying
     exact step count, early-stopping flag, token sequences, rankings, scores,
     and zero repeated N-grams.
"""
from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass


def wu_length_penalty(gen_len: int, alpha: float) -> float:
    if alpha == 0.0:
        return 1.0
    if alpha == 1.0:
        return (5.0 + float(gen_len)) / 6.0
    return math.pow((5.0 + float(gen_len)) / 6.0, alpha)


def wu_score(sum_log_prob: float, gen_len: int, alpha: float) -> float:
    if alpha == 0.0:
        return sum_log_prob
    return sum_log_prob / wu_length_penalty(gen_len, alpha)


def apply_repetition_penalty(
    history: list[int], logits: list[float], penalty: float
) -> list[float]:
    if penalty == 1.0:
        return list(logits)
    seen = set(history)
    out = list(logits)
    for t in seen:
        x = out[t]
        if math.isfinite(x):
            out[t] = (x / penalty) if x > 0.0 else (x * penalty)
    return out


def apply_no_repeat_ngram(
    seq: list[int], logits: list[float], ngram_size: int
) -> list[float]:
    out = list(logits)
    if ngram_size < 2 or len(seq) < ngram_size:
        return out
    prefix_len = ngram_size - 1
    target = seq[len(seq) - prefix_len :]
    for start in range(len(seq) - ngram_size + 1):
        if seq[start : start + prefix_len] == target:
            out[seq[start + prefix_len]] = -math.inf
    return out


def apply_min_length(
    gen_len: int, min_new: int, eos_id: int, logits: list[float]
) -> list[float]:
    out = list(logits)
    if gen_len < min_new:
        out[eos_id] = -math.inf
    return out


def log_softmax(logits: list[float]) -> list[float]:
    finite = [x for x in logits if math.isfinite(x)]
    assert finite, "all logits masked"
    m = max(finite)
    s = sum(math.exp(x - m) for x in finite)
    lse = math.log(s)
    return [((x - m) - lse) if math.isfinite(x) else -math.inf for x in logits]


@dataclass
class Hyp:
    tokens: tuple[int, ...]
    gen_len: int
    finished: int
    sum_log_prob: float
    score: float

    def sort_key(self) -> tuple[float, float, int, tuple[int, ...]]:
        return (-self.score, -self.sum_log_prob, len(self.tokens), self.tokens)


def run_beam_search(
    logits_fn,
    prompt: list[int],
    num_beams: int,
    vocab_size: int,
    eos_id: int,
    max_new: int,
    min_new: int,
    ngram_size: int,
    early_stopping: int,
    length_penalty: float,
    rep_penalty: float,
) -> tuple[int, int, list[Hyp]]:
    active: list[tuple[list[int], float]] = [(list(prompt), 0.0)]
    finished_pool: list[Hyp] = []
    steps_done = 0
    stopped_early = 0

    while steps_done < max_new and active:
        # Collect all finite candidates: (log_prob, parent_idx, token_id)
        cands: list[tuple[float, int, int]] = []
        for b_idx, (seq, parent_lp) in enumerate(active):
            raw = logits_fn(seq)
            assert len(raw) == vocab_size
            p = apply_repetition_penalty(seq, raw, rep_penalty)
            p = apply_no_repeat_ngram(seq, p, ngram_size)
            p = apply_min_length(steps_done, min_new, eos_id, p)
            lps = log_softmax(p)
            for v, lp in enumerate(lps):
                if math.isfinite(lp):
                    cands.append((parent_lp + lp, b_idx, v))

        # Exact strict total order: log_prob DESC, parent_idx ASC, token_id ASC
        cands.sort(key=lambda c: (-c[0], c[1], c[2]))
        top_2b = cands[: 2 * num_beams]

        new_gen_len = steps_done + 1
        next_active: list[tuple[list[int], float]] = []
        for rank, (cand_lp, b_idx, tok) in enumerate(top_2b):
            parent_seq = active[b_idx][0]
            new_seq = parent_seq + [tok]
            if tok == eos_id:
                if rank < num_beams:
                    sc = wu_score(cand_lp, new_gen_len, length_penalty)
                    finished_pool.append(
                        Hyp(tuple(new_seq), new_gen_len, 1, cand_lp, sc)
                    )
                    finished_pool.sort(key=lambda h: h.sort_key())
                    finished_pool = finished_pool[: 2 * num_beams]
            else:
                if len(next_active) < num_beams:
                    next_active.append((new_seq, cand_lp))
            if len(next_active) == num_beams and rank + 1 >= num_beams:
                break

        active = next_active
        steps_done = new_gen_len

        if steps_done >= max_new or not active:
            break
        if early_stopping == 1 and len(finished_pool) >= num_beams:
            best_possible = wu_score(active[0][1], max_new, length_penalty)
            kth_score = finished_pool[num_beams - 1].score
            if best_possible <= kth_score:
                stopped_early = 1
                break

    merged = list(finished_pool)
    if len(finished_pool) < num_beams:
        for seq, lp in active:
            sc = wu_score(lp, steps_done, length_penalty)
            merged.append(Hyp(tuple(seq), steps_done, 0, lp, sc))
    merged.sort(key=lambda h: h.sort_key())
    return steps_done, stopped_early, merged[:num_beams]


def trap_tree_logits(seq: list[int]) -> list[float]:
    last = seq[-1]
    if len(seq) == 1 and last == 1:
        out = [-6.0] * 8
        out[2] = 1.2
        out[3] = 0.9
        return out
    if last == 2:
        out = [0.0] * 8
        out[4] = 0.2
        return out
    if last == 4:
        return [0.0] * 8
    if last == 3:
        out = [-4.0] * 8
        out[5] = 8.0
        return out
    if last == 5:
        out = [-4.0] * 8
        out[6] = 8.0
        return out
    if last == 6:
        out = [-4.0] * 8
        out[0] = 8.0
        return out
    out = [0.0] * 8
    out[0] = 0.5
    return out


def wu_shift_logits(seq: list[int]) -> list[float]:
    out = [-10.0] * 6
    last = seq[-1]
    if len(seq) == 1 and last == 1:
        out[2] = 0.0
        out[3] = -0.2
    elif last == 2:
        out[0] = 10.0
    elif last == 3:
        out[4] = 10.0
    elif last == 4:
        out[5] = 10.0
    elif last == 5:
        out[2] = 10.0
    else:
        out[0] = 10.0
    return out


def hash_model_logits(seed: int, vocab: int):
    mask64 = (1 << 64) - 1

    def fn(seq: list[int]) -> list[float]:
        h = (seed ^ 0x9E3779B97F4A7C15) & mask64
        for tok in seq:
            h ^= (tok + 0x9E3779B97F4A7C15 + ((h << 6) & mask64) + (h >> 2)) & mask64
            h &= mask64
        out = []
        for v in range(vocab):
            x = (h ^ ((v * 0xBF58476D1CE4E5B9) & mask64)) & mask64
            x ^= x >> 30
            x = (x * 0xBF58476D1CE4E5B9) & mask64
            x ^= x >> 27
            x = (x * 0x94D049BB133111EB) & mask64
            x ^= x >> 31
            bucket = (x % 41) - 20
            out.append(float(bucket) * 0.125)
        return out

    return fn


def assert_close_vec(name: str, got: list[float], ref: list[float]) -> None:
    assert len(got) == len(ref), f"{name}: length mismatch"
    for i, (g, r) in enumerate(zip(got, ref)):
        if math.isinf(r) and r < 0.0:
            assert math.isinf(g) and g < 0.0, f"{name}[{i}]: expected -inf, got {g}"
        else:
            err = abs(g - r)
            assert err <= 1e-14, f"{name}[{i}]: got={g!r} ref={r!r} err={err}"


def assert_no_repeated_ngram(tokens: tuple[int, ...], n: int) -> None:
    if n < 2:
        return
    seen: set[tuple[int, ...]] = set()
    for i in range(len(tokens) - n + 1):
        gram = tokens[i : i + n]
        assert gram not in seen, f"repeated {n}-gram {gram} in {tokens}"
        seen.add(gram)


def parse_dump(stdout: str):
    vecs: dict[str, list[float]] = {}
    metas: dict[str, tuple[int, int, int]] = {}
    hyps: dict[str, list[Hyp]] = {}
    negs: dict[str, str] = {}
    for ln in stdout.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        tag = parts[0]
        if tag.startswith("PROC_"):
            vecs[tag] = [float(x) for x in parts[1:]]
        elif tag == "NEG":
            for p in parts[1:]:
                k, v = p.split("=", 1)
                negs[k] = v
        elif tag.endswith("_META"):
            base = tag[: -len("_META")]
            kv = dict(p.split("=", 1) for p in parts[1:])
            metas[base] = (int(kv["steps"]), int(kv["early"]), int(kv["count"]))
        else:
            kv = dict(p.split("=", 1) for p in parts[1:])
            toks = tuple(int(t) for t in kv["tokens"].split(","))
            h = Hyp(
                tokens=toks,
                gen_len=int(kv["gen_len"]),
                finished=int(kv["finished"]),
                sum_log_prob=float(kv["sum_lp"]),
                score=float(kv["score"]),
            )
            hyps.setdefault(tag, []).append(h)
    return vecs, metas, hyps, negs


def compare_case(
    tag: str,
    metas: dict[str, tuple[int, int, int]],
    hyps: dict[str, list[Hyp]],
    ref_steps: int,
    ref_early: int,
    ref_hyps: list[Hyp],
    ngram_size: int,
) -> float:
    got_steps, got_early, got_count = metas[tag]
    assert got_steps == ref_steps, f"{tag}: steps {got_steps} != {ref_steps}"
    assert got_early == ref_early, f"{tag}: early {got_early} != {ref_early}"
    assert got_count == len(ref_hyps), f"{tag}: count {got_count} != {len(ref_hyps)}"
    got_list = hyps[tag]
    assert len(got_list) == len(ref_hyps)
    max_err = 0.0
    for r, (g, ref) in enumerate(zip(got_list, ref_hyps)):
        assert g.tokens == ref.tokens, f"{tag}[{r}]: tokens {g.tokens} != {ref.tokens}"
        assert g.finished == ref.finished, f"{tag}[{r}]: finished mismatch"
        assert g.gen_len == ref.gen_len, f"{tag}[{r}]: gen_len mismatch"
        lp_err = abs(g.sum_log_prob - ref.sum_log_prob)
        sc_err = abs(g.score - ref.score)
        max_err = max(max_err, lp_err, sc_err)
        assert lp_err <= 1e-13, f"{tag}[{r}]: lp_err={lp_err}"
        assert sc_err <= 1e-13, f"{tag}[{r}]: sc_err={sc_err}"
        assert_no_repeated_ngram(g.tokens, ngram_size)
    return max_err


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <test_beam_search_native>")
    proc = subprocess.run(
        [sys.argv[1], "--oracle"], check=True, capture_output=True, text=True
    )
    vecs, metas, hyps, negs = parse_dump(proc.stdout)

    # 1. Verify logit processor chain + log_softmax bit/ULP parity.
    hist = [2, 5, 2, 0, 7]
    raw = [1.5, -2.25, 3.75, 0.0, -0.0, -4.5, 2.0, 0.25]
    ng_seq = [1, 2, 3, 4, 1, 2, 3]
    p1 = apply_repetition_penalty(hist, raw, 1.35)
    p2 = apply_no_repeat_ngram(ng_seq, p1, 3)
    p3 = apply_min_length(1, 3, 0, p2)
    lsm = log_softmax(p3)
    assert_close_vec("PROC_REP", vecs["PROC_REP"], p1)
    assert_close_vec("PROC_NGRAM", vecs["PROC_NGRAM"], p2)
    assert_close_vec("PROC_MINLEN", vecs["PROC_MINLEN"], p3)
    assert_close_vec("PROC_LSM", vecs["PROC_LSM"], lsm)

    max_err = 0.0

    # 2. Trap tree B=1 vs B=4.
    for tag, B in (("TRAP_B1", 1), ("TRAP_B4", 4)):
        st, es, rh = run_beam_search(
            trap_tree_logits, [1], B, 8, 0, 4, 0, 0, 1, 0.0, 1.0
        )
        max_err = max(max_err, compare_case(tag, metas, hyps, st, es, rh, 0))
    assert hyps["TRAP_B1"][0].tokens == (1, 2, 4, 0)
    assert hyps["TRAP_B4"][0].tokens == (1, 3, 5, 6, 0)
    assert hyps["TRAP_B4"][0].sum_log_prob > hyps["TRAP_B1"][0].sum_log_prob + 3.5

    # 3. Wu length penalty shift (alpha=0.0 vs alpha=1.5).
    for tag, alpha in (("WU_A00", 0.0), ("WU_A15", 1.5)):
        st, es, rh = run_beam_search(
            wu_shift_logits, [1], 4, 6, 0, 5, 0, 0, 0, alpha, 1.0
        )
        max_err = max(max_err, compare_case(tag, metas, hyps, st, es, rh, 0))
    assert hyps["WU_A00"][0].tokens == (1, 2, 0)
    assert hyps["WU_A15"][0].tokens == (1, 3, 4, 5, 2, 0)

    # 4. Six multi-parameter stress cases.
    stress_specs = [
        ("STRESS_0", 101, 1, 16, 8, 0, 0, 1, 0.0, 1.0),
        ("STRESS_1", 202, 4, 16, 10, 2, 2, 1, 0.6, 1.25),
        ("STRESS_2", 303, 6, 16, 12, 3, 3, 0, 1.0, 1.5),
        ("STRESS_3", 404, 8, 16, 10, 1, 2, 1, 1.4, 1.15),
        ("STRESS_4", 505, 4, 16, 9, 0, 0, 1, 0.0, 1.0),
        ("STRESS_5", 606, 3, 16, 8, 4, 3, 1, 2.0, 1.8),
    ]
    for tag, seed, B, V, max_n, min_n, ng, es_flag, lp, rp in stress_specs:
        st, es, rh = run_beam_search(
            hash_model_logits(seed, V),
            [2, 5, 7],
            B,
            V,
            0,
            max_n,
            min_n,
            ng,
            es_flag,
            lp,
            rp,
        )
        max_err = max(max_err, compare_case(tag, metas, hyps, st, es, rh, ng))

    # 5. Verify negative fail-closed status codes and sentinel preservation.
    expected_negs = {
        "all_mask": -3,
        "nan": -2,
        "inf": -2,
        "alias": -4,
        "b0": -1,
        "b9": -7,
        "v1": -1,
        "ng1": -1,
        "rp": -1,
        "mm": -1,
    }
    for k, exp_code in expected_negs.items():
        assert int(negs[k]) == exp_code, f"NEG[{k}]: got {negs[k]} != {exp_code}"
    assert float(negs["sentinel"]) == 42.5, f"sentinel corrupted: {negs['sentinel']}"

    print(
        f"BEAM_SEARCH Python binary64 oracle: PASS "
        f"(10 scenarios, 4 processors, 10 negative codes, max_abs_err={max_err:.3e})"
    )


if __name__ == "__main__":
    main()
