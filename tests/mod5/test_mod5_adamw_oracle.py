"""Independent Python oracle for MOD5 AdamW, clipping, LR schedules, and snapshots."""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest

LIB_ENV = "MOD5_ADAMW_LIB"

LR_CONSTANT = 0
LR_LINEAR_WARMUP_LINEAR_DECAY = 1
LR_LINEAR_WARMUP_COSINE_DECAY = 2

SNAPSHOT_MAGIC = b"ETM5TRN1"
SNAPSHOT_HEADER_BYTES = 224
SNAPSHOT_CHECKSUM_BYTES = 32


class LrScheduleConfig(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("base_lr", ctypes.c_double),
        ("min_lr", ctypes.c_double),
        ("warmup_steps", ctypes.c_uint64),
        ("total_steps", ctypes.c_uint64),
    ]


class DatasetCursor(ctypes.Structure):
    _fields_ = [
        ("shard_index", ctypes.c_uint64),
        ("token_offset", ctypes.c_uint64),
        ("epoch", ctypes.c_uint64),
    ]


class AdamWConfig(ctypes.Structure):
    _fields_ = [
        ("param_count", ctypes.c_uint64),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("weight_decay", ctypes.c_float),
        ("max_grad_norm", ctypes.c_float),
        ("reserved", ctypes.c_uint32),
        ("schedule", LrScheduleConfig),
        ("model_config_hash", ctypes.c_uint8 * 32),
        ("tokenizer_id", ctypes.c_char_p),
    ]


class StepInfo(ctypes.Structure):
    _fields_ = [
        ("step", ctypes.c_uint64),
        ("lr", ctypes.c_double),
        ("grad_norm", ctypes.c_double),
        ("clip_scale", ctypes.c_double),
        ("clipped", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ipow_f64(base: float, exp: int) -> float:
    result = 1.0
    factor = float(base)
    while exp > 0:
        if exp & 1:
            result *= factor
        exp >>= 1
        if exp > 0:
            factor *= factor
    return result


def oracle_lr(
    kind: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    step: int,
) -> float:
    if kind == LR_CONSTANT:
        return base_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (float(step) / float(warmup_steps))
    if step >= total_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    if kind == LR_LINEAR_WARMUP_LINEAR_DECAY:
        return min_lr + (base_lr - min_lr) * (1.0 - progress)
    if kind == LR_LINEAR_WARMUP_COSINE_DECAY:
        return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule kind {kind}")


def oracle_adamw_step_f64(
    params: list[float],
    m: list[float],
    v: list[float],
    grads: list[float],
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    max_grad_norm: float,
) -> tuple[list[float], list[float], list[float], float, float, bool]:
    """High-precision binary64 mathematical reference for AdamW + L2 clipping."""
    grad_norm = math.sqrt(sum(float(g) * float(g) for g in grads))
    clipped = max_grad_norm > 0.0 and grad_norm > max_grad_norm
    clip_scale = (max_grad_norm / grad_norm) if clipped else 1.0
    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    out_p, out_m, out_v = [], [], []
    for p, mv, vv, g in zip(params, m, v, grads):
        ge = g * clip_scale if clipped else g
        m_new = beta1 * mv + (1.0 - beta1) * ge
        v_new = beta2 * vv + (1.0 - beta2) * (ge * ge)
        m_hat = m_new / bc1
        v_hat = v_new / bc2
        p_new = p * (1.0 - lr * weight_decay) - lr * m_hat / (math.sqrt(v_hat) + eps)
        out_p.append(p_new)
        out_m.append(m_new)
        out_v.append(v_new)
    return out_p, out_m, out_v, grad_norm, clip_scale, clipped


def oracle_adamw_step_f32_exact(
    params: list[float],
    m: list[float],
    v: list[float],
    grads: list[float],
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    max_grad_norm: float,
) -> tuple[list[float], list[float], list[float], float, float, bool]:
    """Exact IEEE-754 binary32 evaluation order matching the native C contract."""
    b1 = f32(beta1)
    b2 = f32(beta2)
    ep = f32(eps)
    wd = f32(weight_decay)
    mgn = f32(max_grad_norm)
    lr32 = f32(lr)
    one_m_b1 = f32(1.0 - b1)
    one_m_b2 = f32(1.0 - b2)
    bc1 = f32(1.0 - ipow_f64(float(b1), step))
    bc2 = f32(1.0 - ipow_f64(float(b2), step))
    decay = f32(1.0 - f32(lr32 * wd))

    sum_sq = 0.0
    for g in grads:
        gd = float(f32(g))
        sum_sq += gd * gd
    grad_norm = math.sqrt(sum_sq)
    clipped = mgn > 0.0 and grad_norm > float(mgn)
    clip_scale = (float(mgn) / grad_norm) if clipped else 1.0

    out_p, out_m, out_v = [], [], []
    for p, mv, vv, g in zip(params, m, v, grads):
        g32 = f32(float(f32(g)) * clip_scale) if clipped else f32(g)
        g_sq = f32(g32 * g32)
        m_new = f32(f32(b1 * mv) + f32(one_m_b1 * g32))
        v_new = f32(f32(b2 * vv) + f32(one_m_b2 * g_sq))
        m_hat = f32(m_new / bc1)
        v_hat = f32(v_new / bc2)
        denom = f32(f32(math.sqrt(v_hat)) + ep)
        p_decay = f32(p * decay)
        p_new = f32(p_decay - f32(lr32 * f32(m_hat / denom)))
        out_p.append(p_new)
        out_m.append(m_new)
        out_v.append(v_new)
    return out_p, out_m, out_v, grad_norm, clip_scale, clipped


class Mod5AdamWOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lib_path = os.environ.get(LIB_ENV)
        if not lib_path:
            raise unittest.SkipTest(f"Set {LIB_ENV} via scripts/test-mod5-adamw.sh")
        lib = ctypes.CDLL(lib_path)
        lib.et_mod5_adamw_compute_lr.argtypes = [
            ctypes.POINTER(LrScheduleConfig),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.et_mod5_adamw_compute_lr.restype = ctypes.c_int32

        lib.et_mod5_adamw_create.argtypes = [
            ctypes.POINTER(AdamWConfig),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.et_mod5_adamw_create.restype = ctypes.c_int32

        lib.et_mod5_adamw_destroy.argtypes = [ctypes.c_void_p]
        lib.et_mod5_adamw_destroy.restype = None

        lib.et_mod5_adamw_set_params.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.et_mod5_adamw_set_params.restype = ctypes.c_int32

        lib.et_mod5_adamw_get_params.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.et_mod5_adamw_get_params.restype = ctypes.c_int32

        lib.et_mod5_adamw_get_moments.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.et_mod5_adamw_get_moments.restype = ctypes.c_int32

        lib.et_mod5_adamw_set_cursor.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(DatasetCursor),
        ]
        lib.et_mod5_adamw_set_cursor.restype = ctypes.c_int32

        lib.et_mod5_adamw_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(StepInfo),
        ]
        lib.et_mod5_adamw_step.restype = ctypes.c_int32

        lib.et_mod5_adamw_save_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.et_mod5_adamw_save_snapshot.restype = ctypes.c_int32

        lib.et_mod5_adamw_load_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.et_mod5_adamw_load_snapshot.restype = ctypes.c_int32

        cls.lib = lib

    def test_lr_schedules_match_oracle(self) -> None:
        max_abs_err = 0.0
        for kind, name in (
            (LR_CONSTANT, "CONSTANT"),
            (LR_LINEAR_WARMUP_LINEAR_DECAY, "LINEAR_WARMUP_LINEAR_DECAY"),
            (LR_LINEAR_WARMUP_COSINE_DECAY, "LINEAR_WARMUP_COSINE_DECAY"),
        ):
            sched = LrScheduleConfig(
                kind=kind,
                reserved=0,
                base_lr=0.05,
                min_lr=0.005,
                warmup_steps=4,
                total_steps=16,
            )
            for step in range(1, 21):
                got = ctypes.c_double(-1.0)
                rc = self.lib.et_mod5_adamw_compute_lr(
                    ctypes.byref(sched), step, ctypes.byref(got)
                )
                self.assertEqual(rc, 0, f"{name} step {step} failed")
                expected = oracle_lr(kind, 0.05, 0.005, 4, 16, step)
                err = abs(got.value - expected)
                max_abs_err = max(max_abs_err, err)
                self.assertLessEqual(err, 1e-15, f"{name} step {step}: {got.value} vs {expected}")
        print(f"ORACLE LR schedules max_abs_err={max_abs_err:.3e} PASS", flush=True)

    def test_adamw_clipping_and_moments_match_oracle(self) -> None:
        n = 16
        beta1, beta2, eps, wd, max_norm = 0.9, 0.999, 1e-8, 0.01, 0.5
        base_lr, min_lr, warmup, total = 0.05, 0.005, 4, 16
        tok_id = b"mod5-oracle-token-v1"
        model_hash = hashlib.sha256(b"mod5-oracle-model-v1").digest()

        cfg = AdamWConfig()
        cfg.param_count = n
        cfg.beta1 = beta1
        cfg.beta2 = beta2
        cfg.eps = eps
        cfg.weight_decay = wd
        cfg.max_grad_norm = max_norm
        cfg.schedule = LrScheduleConfig(
            kind=LR_LINEAR_WARMUP_COSINE_DECAY,
            reserved=0,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup,
            total_steps=total,
        )
        for i, b in enumerate(model_hash):
            cfg.model_config_hash[i] = b
        cfg.tokenizer_id = tok_id

        state = ctypes.c_void_p()
        self.assertEqual(self.lib.et_mod5_adamw_create(ctypes.byref(cfg), ctypes.byref(state)), 0)
        try:
            init_p = [f32(0.125 * ((i % 7) - 3)) for i in range(n)]
            p_arr = (ctypes.c_float * n)(*init_p)
            self.assertEqual(self.lib.et_mod5_adamw_set_params(state, p_arr, n), 0)

            p_f64 = [float(x) for x in init_p]
            m_f64 = [0.0] * n
            v_f64 = [0.0] * n
            p_f32 = list(init_p)
            m_f32 = [0.0] * n
            v_f32 = [0.0] * n

            max_p_err_f64 = 0.0
            max_m_err_f64 = 0.0
            max_v_err_f64 = 0.0
            max_norm_err = 0.0
            clipped_steps = 0

            for step in range(1, total + 1):
                # Early steps have large gradients (clipped); later steps decay (unclipped).
                scale = 0.4 / (1.0 + 0.35 * (step - 1))
                grads = [
                    f32(scale * math.sin(0.7 * (i + 1) + 0.5 * step))
                    for i in range(n)
                ]
                g_arr = (ctypes.c_float * n)(*grads)
                info = StepInfo()
                self.assertEqual(
                    self.lib.et_mod5_adamw_step(state, g_arr, n, 8, ctypes.byref(info)),
                    0,
                )

                lr = oracle_lr(
                    LR_LINEAR_WARMUP_COSINE_DECAY, base_lr, min_lr, warmup, total, step
                )
                p_f64, m_f64, v_f64, gn64, cs64, cl64 = oracle_adamw_step_f64(
                    p_f64, m_f64, v_f64, [float(g) for g in grads],
                    step, lr, beta1, beta2, eps, wd, max_norm,
                )
                p_f32, m_f32, v_f32, gn32, cs32, cl32 = oracle_adamw_step_f32_exact(
                    p_f32, m_f32, v_f32, grads,
                    step, lr, beta1, beta2, eps, wd, max_norm,
                )

                self.assertEqual(info.step, step)
                self.assertEqual(bool(info.clipped), cl32)
                if cl32:
                    clipped_steps += 1
                max_norm_err = max(max_norm_err, abs(info.grad_norm - gn64))
                self.assertLessEqual(abs(info.grad_norm - gn64), 1e-12)
                self.assertLessEqual(abs(info.clip_scale - cs64), 1e-12)
                self.assertLessEqual(abs(info.lr - lr), 1e-15)

                got_p = (ctypes.c_float * n)()
                got_m = (ctypes.c_float * n)()
                got_v = (ctypes.c_float * n)()
                self.assertEqual(self.lib.et_mod5_adamw_get_params(state, got_p, n), 0)
                self.assertEqual(self.lib.et_mod5_adamw_get_moments(state, got_m, got_v, n), 0)

                for i in range(n):
                    # Exact 0-ULP bitwise match against f32 reference contract
                    self.assertEqual(f32_bits(got_p[i]), f32_bits(p_f32[i]), f"step {step} p[{i}]")
                    self.assertEqual(f32_bits(got_m[i]), f32_bits(m_f32[i]), f"step {step} m[{i}]")
                    self.assertEqual(f32_bits(got_v[i]), f32_bits(v_f32[i]), f"step {step} v[{i}]")
                    # Tight tolerance against independent f64 mathematical oracle
                    max_p_err_f64 = max(max_p_err_f64, abs(got_p[i] - p_f64[i]))
                    max_m_err_f64 = max(max_m_err_f64, abs(got_m[i] - m_f64[i]))
                    max_v_err_f64 = max(max_v_err_f64, abs(got_v[i] - v_f64[i]))

            self.assertGreater(clipped_steps, 0)
            self.assertLess(clipped_steps, total)
            self.assertLessEqual(max_p_err_f64, 5e-6)
            self.assertLessEqual(max_m_err_f64, 5e-7)
            self.assertLessEqual(max_v_err_f64, 5e-7)

            # Independently verify binary snapshot wire format + SHA-256
            cur = DatasetCursor(shard_index=1, token_offset=8, epoch=4)
            self.assertEqual(self.lib.et_mod5_adamw_set_cursor(state, ctypes.byref(cur)), 0)
            with tempfile.TemporaryDirectory() as td:
                snap = Path(td) / "oracle.etsnap"
                self.assertEqual(self.lib.et_mod5_adamw_save_snapshot(state, str(snap).encode()), 0)
                raw = snap.read_bytes()
                expected_size = SNAPSHOT_HEADER_BYTES + len(tok_id) + 12 * n + SNAPSHOT_CHECKSUM_BYTES
                self.assertEqual(len(raw), expected_size)
                self.assertEqual(raw[:8], SNAPSHOT_MAGIC)
                self.assertEqual(struct.unpack_from("<HHI", raw, 8), (1, 0, 0))
                self.assertEqual(
                    struct.unpack_from("<QQQQQQ", raw, 16),
                    (n, total, 8 * total, 1, 8, 4),
                )
                self.assertEqual(
                    hashlib.sha256(raw[:-SNAPSHOT_CHECKSUM_BYTES]).digest(),
                    raw[-SNAPSHOT_CHECKSUM_BYTES:],
                )
                tok_len, reserved = struct.unpack_from("<II", raw, 216)
                self.assertEqual((tok_len, reserved), (len(tok_id), 0))
                self.assertEqual(raw[224 : 224 + tok_len], tok_id)
                off = 224 + tok_len
                snap_p = struct.unpack_from(f"<{n}f", raw, off)
                snap_m = struct.unpack_from(f"<{n}f", raw, off + 4 * n)
                snap_v = struct.unpack_from(f"<{n}f", raw, off + 8 * n)
                for i in range(n):
                    self.assertEqual(f32_bits(snap_p[i]), f32_bits(p_f32[i]))
                    self.assertEqual(f32_bits(snap_m[i]), f32_bits(m_f32[i]))
                    self.assertEqual(f32_bits(snap_v[i]), f32_bits(v_f32[i]))

            print(
                f"ORACLE AdamW 16 steps (clipped={clipped_steps}/{total}): "
                f"f32_exact=0-ULP f64_max_err(p={max_p_err_f64:.3e}, "
                f"m={max_m_err_f64:.3e}, v={max_v_err_f64:.3e}, "
                f"grad_norm={max_norm_err:.3e}) PASS",
                flush=True,
            )
        finally:
            self.lib.et_mod5_adamw_destroy(state)


if __name__ == "__main__":
    unittest.main()
