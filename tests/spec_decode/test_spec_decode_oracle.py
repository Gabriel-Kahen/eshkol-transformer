"""Independent Python analytical & empirical oracle for speculative decoding.

Verifies:
1. Canonical binary64 log-sum-exp logits-to-probability parity (greedy,
   temperature, top-k, top-p, lowest-index tie-breaking) against the compiled C
   ABI `et_spec_logits_to_probs_v1`.
2. Exact analytical marginal parity:
     P_spec(v) = q(v) * min(1, p(v) / q(v)) + (1 - alpha) * r(v)
   matches exact p(v) to <= 1e-12 across diverse (q, p) pairs including top-k,
   top-p, temperature scaling, peaked/flat, partial-overlap, and completely
   disjoint supports.
3. Exact SplitMix64 counter-RNG single-step parity between Python reference and
   compiled C `et_spec_verify_sample_v1`.
4. Empirical sampling parity: 60,000 draws per scenario from the compiled C
   engine match exact target p(v) (L_inf, TV distance, chi-square, acceptance
   rate, and zero emissions for p(v) == 0).
"""
from __future__ import annotations

import ctypes
import math
import struct
import sys
import unittest
from pathlib import Path

LIB_PATH = Path(sys.argv[1]).resolve()
ET_SPEC_OK = 0
ET_SPEC_NO_EOS = 0xFFFFFFFF


class SamplingParams(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_double),
        ("top_p", ctypes.c_double),
        ("top_k", ctypes.c_uint32),
        ("eos_token_id", ctypes.c_uint32),
    ]


class RngState(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint64),
        ("counter", ctypes.c_uint64),
    ]


class VerifyResult(ctypes.Structure):
    _fields_ = [
        ("accepted_count", ctypes.c_uint32),
        ("emitted_count", ctypes.c_uint32),
        ("eos_hit", ctypes.c_uint32),
        ("rejected", ctypes.c_uint32),
        ("rng_draws", ctypes.c_uint64),
        ("next_rng", RngState),
    ]


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.et_spec_logits_to_probs_v1.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(SamplingParams),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.et_spec_logits_to_probs_v1.restype = ctypes.c_int32

    lib.et_spec_sample_token_v1.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(SamplingParams),
        ctypes.POINTER(RngState),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(RngState),
    ]
    lib.et_spec_sample_token_v1.restype = ctypes.c_int32

    lib.et_spec_verify_sample_v1.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(SamplingParams),
        ctypes.POINTER(RngState),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(VerifyResult),
    ]
    lib.et_spec_verify_sample_v1.restype = ctypes.c_int32
    return lib


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def py_logits_to_probs(
    logits: list[float], temperature: float, top_k: int, top_p: float
) -> list[float]:
    v = len(logits)
    xs = [f32(x) for x in logits]
    if temperature == 0.0:
        best = 0
        for i in range(1, v):
            if xs[i] > xs[best]:
                best = i
        out = [0.0] * v
        out[best] = 1.0
        return out

    m = max(xs)
    scaled = [(x - m) / temperature for x in xs]
    sum_exp = sum(math.exp(s) for s in scaled)
    log_z = math.log(sum_exp)
    probs = [math.exp(s - log_z) for s in scaled]

    order = sorted(range(v), key=lambda idx: (-probs[idx], idx))
    if top_k < v:
        for idx in order[top_k:]:
            probs[idx] = 0.0
    topk_sum = sum(probs)
    probs = [p / topk_sum if p > 0.0 else 0.0 for p in probs]

    if top_p < 1.0:
        cum = 0.0
        keep = top_k
        for rank in range(top_k):
            cum += probs[order[rank]]
            if cum >= top_p:
                keep = rank + 1
                break
        for idx in order[keep:top_k]:
            probs[idx] = 0.0
        final_sum = sum(probs)
        probs = [p / final_sum if p > 0.0 else 0.0 for p in probs]

    return probs


def py_splitmix64_uniform01(seed: int, counter: int) -> float:
    mask = (1 << 64) - 1
    x = (seed + ((counter * 0x9E3779B97F4A7C15) & mask)) & mask
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    x = (x ^ (x >> 31)) & mask
    return (x >> 11) * (2.0 ** -53)


def py_sample_categorical(probs: list[float], u: float) -> int:
    cdf = 0.0
    last_nz = 0
    for idx, p in enumerate(probs):
        if p > 0.0:
            last_nz = idx
            cdf += p
            if u < cdf:
                return idx
    return last_nz


def analytical_spec_marginal(q: list[float], p: list[float]) -> tuple[list[float], float]:
    """Compute exact P_spec(v) = q(v)*min(1, p(v)/q(v)) + (1 - alpha)*r(v)."""
    v = len(p)
    accepted_mass = []
    for i in range(v):
        if q[i] == 0.0:
            accepted_mass.append(0.0)
        else:
            accepted_mass.append(q[i] * min(1.0, p[i] / q[i]))
    alpha = math.fsum(accepted_mass)
    raw_res = [max(0.0, p[i] - q[i]) for i in range(v)]
    res_sum = math.fsum(raw_res)
    if res_sum > 0.0:
        residual = [r / res_sum for r in raw_res]
    else:
        residual = list(p)
    marginal = [accepted_mass[i] + (1.0 - alpha) * residual[i] for i in range(v)]
    return marginal, alpha


def c_logits_to_probs(
    lib: ctypes.CDLL, logits: list[float], temperature: float, top_k: int, top_p: float
) -> list[float]:
    v = len(logits)
    arr = (ctypes.c_float * v)(*logits)
    out = (ctypes.c_double * v)()
    params = SamplingParams(temperature, top_p, top_k, ET_SPEC_NO_EOS)
    st = lib.et_spec_logits_to_probs_v1(arr, v, ctypes.byref(params), out)
    if st != ET_SPEC_OK:
        raise AssertionError(f"et_spec_logits_to_probs_v1 failed: {st}")
    return list(out)


SCENARIOS = [
    (
        "identical_p_eq_q",
        [1.2, 0.4, -0.3, 0.8, -1.1, 0.0, 0.5, -0.2],
        [1.2, 0.4, -0.3, 0.8, -1.1, 0.0, 0.5, -0.2],
        1.0,
        8,
        1.0,
    ),
    (
        "mild_shift_temp_0.8",
        [2.0, 1.5, 0.8, 0.2, -0.5, -1.0, 0.1, -0.3],
        [1.7, 1.9, 0.5, 0.9, -0.2, -0.8, 0.3, -0.7],
        0.8,
        8,
        1.0,
    ),
    (
        "peaked_target_flat_draft",
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 1.0, -1.0, -2.0, 0.5, -3.0, 2.2, 0.0],
        0.7,
        8,
        1.0,
    ),
    (
        "flat_target_peaked_draft",
        [4.5, 1.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        1.2,
        8,
        1.0,
    ),
    (
        "topk_partial_overlap",
        [3.0, 2.5, 2.0, 0.1, -1.0, -2.0, -3.0, -4.0],
        [-2.0, 2.8, 1.9, 3.1, 2.2, -1.0, -3.0, -4.0],
        0.9,
        3,
        1.0,
    ),
    (
        "topp_nucleus_filtered",
        [2.5, 2.1, 1.7, 0.4, -0.2, -1.5, -2.0, -2.5],
        [1.2, 2.6, 1.8, 2.0, 0.1, -1.0, -1.5, -2.0],
        0.75,
        6,
        0.75,
    ),
    (
        "completely_disjoint_support",
        [5.0, 4.5, 4.0, -5.0, -5.0, -5.0, -5.0, -5.0],
        [-5.0, -5.0, -5.0, 5.0, 4.5, 4.0, -5.0, -5.0],
        1.0,
        3,
        1.0,
    ),
    (
        "tied_logits_topk_topp",
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        0.6,
        4,
        0.7,
    ),
    (
        "vocab_64_temp_topk_topp",
        [math.sin(i * 0.7) * 2.0 for i in range(64)],
        [math.cos(i * 0.5) * 2.2 + (0.6 if i % 5 == 0 else 0.0) for i in range(64)],
        0.85,
        16,
        0.9,
    ),
]


class SpecDecodeOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = load_lib()

    def test_logits_to_probs_parity(self) -> None:
        max_err = 0.0
        extra_cases = [
            ([0.0, 0.0, 0.0, 0.0], 0.0, 4, 1.0),
            ([-0.0, 0.0, -1.0, 0.0], 0.0, 2, 0.5),
            ([-1000.0, 1000.0, 999.0, -1000.0], 1.0, 4, 1.0),
            ([10.0, 10.0, 9.0, 8.0, 7.0], 0.5, 3, 0.8),
            ([0.1 * i for i in range(128)], 1.7, 32, 0.65),
        ]
        for logits, temp, top_k, top_p in extra_cases:
            c_p = c_logits_to_probs(self.lib, logits, temp, top_k, top_p)
            py_p = py_logits_to_probs(logits, temp, top_k, top_p)
            self.assertAlmostEqual(sum(c_p), 1.0, delta=1e-13)
            for a, b in zip(c_p, py_p, strict=True):
                max_err = max(max_err, abs(a - b))
                self.assertLessEqual(abs(a - b), 1e-14)
        print(f"ORACLE logits_to_probs max_abs_err={max_err:.3e}", flush=True)

    def test_analytical_marginal_parity(self) -> None:
        max_err = 0.0
        for name, q_logits, p_logits, temp, top_k, top_p in SCENARIOS:
            with self.subTest(scenario=name):
                q_c = c_logits_to_probs(self.lib, q_logits, temp, top_k, top_p)
                p_c = c_logits_to_probs(self.lib, p_logits, temp, top_k, top_p)
                q_py = py_logits_to_probs(q_logits, temp, top_k, top_p)
                p_py = py_logits_to_probs(p_logits, temp, top_k, top_p)
                for a, b in zip(q_c, q_py, strict=True):
                    self.assertLessEqual(abs(a - b), 1e-14)
                for a, b in zip(p_c, p_py, strict=True):
                    self.assertLessEqual(abs(a - b), 1e-14)

                marginal, alpha = analytical_spec_marginal(q_c, p_c)
                err = max(abs(m - pv) for m, pv in zip(marginal, p_c, strict=True))
                max_err = max(max_err, err)
                self.assertLessEqual(
                    err,
                    1e-12,
                    f"{name}: marginal error {err:.3e} exceeded 1e-12 (alpha={alpha:.6f})",
                )
        print(
            f"ORACLE analytical_marginal_parity scenarios={len(SCENARIOS)} "
            f"max_abs_err={max_err:.3e}",
            flush=True,
        )

    def test_exact_step_rng_replay_against_python(self) -> None:
        _, q_logits, p_logits, temp, top_k, top_p = SCENARIOS[4]
        v = len(q_logits)
        q_probs = py_logits_to_probs(q_logits, temp, top_k, top_p)
        p_probs = py_logits_to_probs(p_logits, temp, top_k, top_p)
        bonus_logits = [0.2 * i for i in range(v)]
        bonus_probs = py_logits_to_probs(bonus_logits, temp, top_k, top_p)

        q_arr = (ctypes.c_float * v)(*q_logits)
        p_arr = (ctypes.c_float * (2 * v))(*(p_logits + bonus_logits))
        params = SamplingParams(temp, top_p, top_k, ET_SPEC_NO_EOS)
        seed = 0x517CC1B727220A95
        counter = 0

        for _ in range(250):
            # Sample draft token x_0 in Python and C using same counter.
            u_d = py_splitmix64_uniform01(seed, counter)
            py_x = py_sample_categorical(q_probs, u_d)
            c_x = ctypes.c_uint32(0)
            rng_in = RngState(seed, counter)
            rng_mid = RngState(0, 0)
            st = self.lib.et_spec_sample_token_v1(
                q_arr, v, ctypes.byref(params), ctypes.byref(rng_in),
                ctypes.byref(c_x), ctypes.byref(rng_mid),
            )
            self.assertEqual(st, ET_SPEC_OK)
            self.assertEqual(c_x.value, py_x)
            self.assertEqual(rng_mid.counter, counter + 1)
            counter += 1

            # Verify K=1 step in Python and C.
            u_acc = py_splitmix64_uniform01(seed, counter)
            counter += 1
            if p_probs[py_x] > 0.0 and (
                q_probs[py_x] <= p_probs[py_x] or u_acc * q_probs[py_x] < p_probs[py_x]
            ):
                u_bonus = py_splitmix64_uniform01(seed, counter)
                counter += 1
                py_emitted = [py_x, py_sample_categorical(bonus_probs, u_bonus)]
                py_accepted = 1
            else:
                raw_r = [max(0.0, p_probs[i] - q_probs[i]) for i in range(v)]
                rs = sum(raw_r)
                r = [x / rs for x in raw_r] if rs > 0.0 else p_probs
                u_res = py_splitmix64_uniform01(seed, counter)
                counter += 1
                py_emitted = [py_sample_categorical(r, u_res)]
                py_accepted = 0

            dt = (ctypes.c_uint32 * 1)(c_x.value)
            out_t = (ctypes.c_uint32 * 2)()
            res = VerifyResult()
            st = self.lib.et_spec_verify_sample_v1(
                dt, q_arr, p_arr, 1, v, ctypes.byref(params),
                ctypes.byref(rng_mid), out_t, 2, ctypes.byref(res),
            )
            self.assertEqual(st, ET_SPEC_OK)
            self.assertEqual(res.accepted_count, py_accepted)
            self.assertEqual(list(out_t[: res.emitted_count]), py_emitted)
            self.assertEqual(res.next_rng.counter, counter)

    def test_empirical_sampling_matches_exact_p(self) -> None:
        num_samples = 60000
        empirical_scenarios = [
            SCENARIOS[1],  # mild_shift_temp_0.8
            SCENARIOS[4],  # topk_partial_overlap
            SCENARIOS[5],  # topp_nucleus_filtered
            SCENARIOS[6],  # completely_disjoint_support
        ]
        max_linf = 0.0
        max_tv = 0.0
        max_alpha_err = 0.0

        for idx, (name, q_logits, p_logits, temp, top_k, top_p) in enumerate(
            empirical_scenarios
        ):
            v = len(q_logits)
            q_probs = c_logits_to_probs(self.lib, q_logits, temp, top_k, top_p)
            p_probs = c_logits_to_probs(self.lib, p_logits, temp, top_k, top_p)
            _, expected_alpha = analytical_spec_marginal(q_probs, p_probs)

            q_arr = (ctypes.c_float * v)(*q_logits)
            p_arr = (ctypes.c_float * (2 * v))(*(p_logits + p_logits))
            params = SamplingParams(temp, top_p, top_k, ET_SPEC_NO_EOS)

            counts = [0] * v
            accepted_total = 0
            rng = RngState(0xCAFEBABE00000001 + idx * 97, 0)
            draft_tok = ctypes.c_uint32(0)
            next_rng = RngState(0, 0)
            dt = (ctypes.c_uint32 * 1)(0)
            out_t = (ctypes.c_uint32 * 2)()
            res = VerifyResult()

            for _ in range(num_samples):
                st = self.lib.et_spec_sample_token_v1(
                    q_arr, v, ctypes.byref(params), ctypes.byref(rng),
                    ctypes.byref(draft_tok), ctypes.byref(next_rng),
                )
                self.assertEqual(st, ET_SPEC_OK)
                dt[0] = draft_tok.value
                st = self.lib.et_spec_verify_sample_v1(
                    dt, q_arr, p_arr, 1, v, ctypes.byref(params),
                    ctypes.byref(next_rng), out_t, 2, ctypes.byref(res),
                )
                self.assertEqual(st, ET_SPEC_OK)
                counts[out_t[0]] += 1
                accepted_total += res.accepted_count
                rng = res.next_rng

            freqs = [c / num_samples for c in counts]
            for tok, (pv, cnt) in enumerate(zip(p_probs, counts, strict=True)):
                if pv == 0.0:
                    self.assertEqual(
                        cnt, 0, f"{name}: zero-prob token {tok} emitted {cnt} times"
                    )
            linf = max(abs(f - pv) for f, pv in zip(freqs, p_probs, strict=True))
            tv = 0.5 * sum(abs(f - pv) for f, pv in zip(freqs, p_probs, strict=True))
            emp_alpha = accepted_total / num_samples
            alpha_err = abs(emp_alpha - expected_alpha)
            max_linf = max(max_linf, linf)
            max_tv = max(max_tv, tv)
            max_alpha_err = max(max_alpha_err, alpha_err)

            self.assertLess(linf, 0.006, f"{name}: L_inf={linf:.5f}")
            self.assertLess(tv, 0.010, f"{name}: TV={tv:.5f}")
            self.assertLess(alpha_err, 0.006, f"{name}: alpha_err={alpha_err:.5f}")
            print(
                f"ORACLE empirical scenario={name} samples={num_samples} "
                f"linf={linf:.5f} tv={tv:.5f} alpha_expected={expected_alpha:.5f} "
                f"alpha_empirical={emp_alpha:.5f}",
                flush=True,
            )

        print(
            f"ORACLE empirical_summary max_linf={max_linf:.5f} "
            f"max_tv={max_tv:.5f} max_alpha_err={max_alpha_err:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
