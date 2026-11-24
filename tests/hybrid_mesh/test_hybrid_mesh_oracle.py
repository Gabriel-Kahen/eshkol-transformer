#!/usr/bin/env python3
"""Independent binary64 dense reference oracle for the 2D Hybrid Mesh (DP x TP) engine.

Verifies:
  1. Forward loss, analytic parameter gradients (454 params), and dX (64 inputs)
     against native C dumps.
  2. 10-step decoupled AdamW + global L2 clipping trajectory against all 6 native
     mesh topologies ((1x1), (2x1), (1x2), (2x2), (4x1), (1x4)) within <= 1e-6.
  3. (--check-aot) Compiled Eshkol AOT dumped weights across (1x1), (2x1), (1x2),
     and (2x2) against each other and the binary64 reference within <= 1e-6.
"""
from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

D_MODEL = 8
D_FF = 16
NUM_CLASSES = 6
GLOBAL_BATCH = 8
RMS_EPS = 1e-5
MODEL_SEED = 20261124
LR = 0.05
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5
STEPS = 10
MASK64 = (1 << 64) - 1

REPLICATED_PARAMS = D_MODEL + D_MODEL + NUM_CLASSES * D_MODEL + NUM_CLASSES  # 70
FULL_SHARDED_PARAMS = 3 * D_FF * D_MODEL  # 384
FULL_PARAMS = REPLICATED_PARAMS + FULL_SHARDED_PARAMS  # 454


def f32(x: float) -> float:
    val = struct.unpack("<f", struct.pack("<f", x))[0]
    return 0.0 if val == 0.0 else val


def splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = (z ^ (z >> 31)) & MASK64
    return state, z


def splitmix64_unit_signed(state: int) -> tuple[int, float]:
    state, z = splitmix64_next(state)
    u = (z >> 40) & 0xFFFFFF
    return state, (float(u) / 8388608.0) - 1.0


def init_full_params(seed: int) -> list[float]:
    rng = (seed ^ 0x45543244494E4954) & MASK64
    params: list[float] = []
    for _ in range(D_MODEL):
        rng, u = splitmix64_unit_signed(rng)
        params.append(f32(1.0 + 0.1 * u))
    for _ in range(D_MODEL):
        rng, u = splitmix64_unit_signed(rng)
        params.append(f32(0.05 * u))
    for _ in range(NUM_CLASSES * D_MODEL):
        rng, u = splitmix64_unit_signed(rng)
        params.append(f32(0.25 * u))
    for _ in range(NUM_CLASSES):
        rng, u = splitmix64_unit_signed(rng)
        params.append(f32(0.05 * u))
    for _ in range(3 * D_FF * D_MODEL):
        rng, u = splitmix64_unit_signed(rng)
        params.append(f32(0.25 * u))
    assert len(params) == FULL_PARAMS
    return params


def make_full_batch(step_idx: int) -> tuple[list[list[float]], list[int]]:
    xs: list[list[float]] = []
    ys: list[int] = []
    for sample_idx in range(GLOBAL_BATCH):
        target = (sample_idx + step_idx) % NUM_CLASSES
        ys.append(target)
        rng = (
            0x5450445032303236
            ^ ((step_idx * 0x9E3779B97F4A7C15) & MASK64)
            ^ ((sample_idx * 0xBF58476D1CE4E5B9) & MASK64)
        ) & MASK64
        row: list[float] = []
        for j in range(D_MODEL):
            rng, u = splitmix64_unit_signed(rng)
            noise = 0.6 * u
            signal = 0.75 if (j % NUM_CLASSES) == target else -0.15
            row.append(f32(noise + signal))
        xs.append(row)
    return xs, ys


def unpack_params(p: list[float]):
    off = 0
    gamma = p[off : off + D_MODEL]
    off += D_MODEL
    b_down = p[off : off + D_MODEL]
    off += D_MODEL
    w_head = [p[off + c * D_MODEL : off + (c + 1) * D_MODEL] for c in range(NUM_CLASSES)]
    off += NUM_CLASSES * D_MODEL
    b_head = p[off : off + NUM_CLASSES]
    off += NUM_CLASSES
    w_gate = [p[off + k * D_MODEL : off + (k + 1) * D_MODEL] for k in range(D_FF)]
    off += D_FF * D_MODEL
    w_up = [p[off + k * D_MODEL : off + (k + 1) * D_MODEL] for k in range(D_FF)]
    off += D_FF * D_MODEL
    w_down = [p[off + j * D_FF : off + (j + 1) * D_FF] for j in range(D_MODEL)]
    off += D_MODEL * D_FF
    assert off == FULL_PARAMS
    return gamma, b_down, w_head, b_head, w_gate, w_up, w_down


def forward_backward_dense_f64(
    params: list[float], xs: list[list[float]], ys: list[int]
) -> tuple[float, list[float], list[float]]:
    gamma, b_down, w_head, b_head, w_gate, w_up, w_down = unpack_params(params)
    b = len(xs)
    inv_b = 1.0 / float(b)

    dgamma = [0.0] * D_MODEL
    db_down = [0.0] * D_MODEL
    dw_head = [[0.0] * D_MODEL for _ in range(NUM_CLASSES)]
    db_head = [0.0] * NUM_CLASSES
    dw_gate = [[0.0] * D_MODEL for _ in range(D_FF)]
    dw_up = [[0.0] * D_MODEL for _ in range(D_FF)]
    dw_down = [[0.0] * D_FF for _ in range(D_MODEL)]
    dx_flat: list[float] = []

    total_loss = 0.0
    for n in range(b):
        xn = xs[n]
        y = ys[n]
        ms = sum(v * v for v in xn) / float(D_MODEL)
        rrms = 1.0 / math.sqrt(ms + RMS_EPS)
        u = [xn[j] * rrms * gamma[j] for j in range(D_MODEL)]

        gate = [sum(u[j] * w_gate[k][j] for j in range(D_MODEL)) for k in range(D_FF)]
        up = [sum(u[j] * w_up[k][j] for j in range(D_MODEL)) for k in range(D_FF)]
        sig = [1.0 / (1.0 + math.exp(-g)) for g in gate]
        silu = [gate[k] * sig[k] for k in range(D_FF)]
        act = [silu[k] * up[k] for k in range(D_FF)]

        z = [sum(act[k] * w_down[j][k] for k in range(D_FF)) for j in range(D_MODEL)]
        h = [xn[j] + z[j] + b_down[j] for j in range(D_MODEL)]

        logits = [
            b_head[c] + sum(h[j] * w_head[c][j] for j in range(D_MODEL))
            for c in range(NUM_CLASSES)
        ]
        m_lg = max(logits)
        exps = [math.exp(lg - m_lg) for lg in logits]
        s_exp = sum(exps)
        probs = [e / s_exp for e in exps]
        total_loss += (m_lg + math.log(s_exp)) - logits[y]

        dh = [0.0] * D_MODEL
        for c in range(NUM_CLASSES):
            dl = (probs[c] - (1.0 if c == y else 0.0)) * inv_b
            db_head[c] += dl
            for j in range(D_MODEL):
                dw_head[c][j] += dl * h[j]
                dh[j] += dl * w_head[c][j]

        for j in range(D_MODEL):
            db_down[j] += dh[j]

        du = [0.0] * D_MODEL
        for k in range(D_FF):
            da = 0.0
            for j in range(D_MODEL):
                dw_down[j][k] += dh[j] * act[k]
                da += dh[j] * w_down[j][k]
            dsilu = sig[k] * (1.0 + gate[k] * (1.0 - sig[k]))
            dp = da * silu[k]
            dg = da * up[k] * dsilu
            for j in range(D_MODEL):
                dw_gate[k][j] += dg * u[j]
                dw_up[k][j] += dp * u[j]
                du[j] += dg * w_gate[k][j] + dp * w_up[k][j]

        dot = 0.0
        for j in range(D_MODEL):
            dgamma[j] += du[j] * xn[j] * rrms
            dot += du[j] * gamma[j] * xn[j]
        coeff = dot / float(D_MODEL)
        rrms2 = rrms * rrms
        for j in range(D_MODEL):
            dx_norm = rrms * (du[j] * gamma[j] - xn[j] * rrms2 * coeff)
            dx_flat.append(dh[j] + dx_norm)

    grads: list[float] = []
    grads.extend(dgamma)
    grads.extend(db_down)
    for row in dw_head:
        grads.extend(row)
    grads.extend(db_head)
    for row in dw_gate:
        grads.extend(row)
    for row in dw_up:
        grads.extend(row)
    for row in dw_down:
        grads.extend(row)
    assert len(grads) == FULL_PARAMS
    return total_loss * inv_b, grads, dx_flat


def run_reference_train10() -> tuple[list[float], list[float], list[float], list[float]]:
    params = init_full_params(MODEL_SEED)
    m = [0.0] * FULL_PARAMS
    v = [0.0] * FULL_PARAMS
    losses: list[float] = []
    step0_grads: list[float] = []
    step0_dx: list[float] = []

    for s in range(STEPS):
        xs, ys = make_full_batch(0)
        loss, grads, dx = forward_backward_dense_f64(params, xs, ys)
        if s == 0:
            step0_grads = grads
            step0_dx = dx
        losses.append(loss)

        gnorm = math.sqrt(sum(g * g for g in grads))
        scale = (MAX_GRAD_NORM / gnorm) if (MAX_GRAD_NORM > 0.0 and gnorm > MAX_GRAD_NORM) else 1.0

        t = s + 1
        bc1 = 1.0 - (BETA1**t)
        bc2 = 1.0 - (BETA2**t)
        decay = 1.0 - LR * WEIGHT_DECAY
        for i in range(FULL_PARAMS):
            g = grads[i] * scale
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * g
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * g * g
            m_hat = m[i] / bc1
            v_hat = v[i] / bc2
            params[i] = params[i] * decay - LR * m_hat / (math.sqrt(v_hat) + EPS)
    return params, losses, step0_grads, step0_dx


def read_f32_file(path: Path, count: int) -> list[float]:
    raw = path.read_bytes()
    assert len(raw) == count * 4, f"{path}: expected {count * 4} bytes, got {len(raw)}"
    return list(struct.unpack(f"<{count}f", raw))


def read_f64_file(path: Path, count: int) -> list[float]:
    raw = path.read_bytes()
    assert len(raw) == count * 8, f"{path}: expected {count * 8} bytes, got {len(raw)}"
    return list(struct.unpack(f"<{count}d", raw))


def max_abs_diff(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b))


def verify_native_binary(native_bin: str) -> None:
    ref_params, ref_losses, ref_g0, ref_dx0 = run_reference_train10()
    with tempfile.TemporaryDirectory(prefix="hybrid_mesh_oracle_") as tmp:
        subprocess.run([native_bin, "--dump-dir", tmp], check=True)
        d = Path(tmp)
        l0 = read_f64_file(d / "step0_loss.f64", 1)[0]
        g0 = read_f32_file(d / "step0_grads.f32", FULL_PARAMS)
        dx0 = read_f32_file(d / "step0_dx.f32", GLOBAL_BATCH * D_MODEL)

        l0_err = abs(l0 - ref_losses[0])
        g0_err = max_abs_diff(g0, ref_g0)
        dx0_err = max_abs_diff(dx0, ref_dx0)
        assert l0_err <= 1e-6, f"step0 loss err {l0_err}"
        assert g0_err <= 1e-6, f"step0 grad err {g0_err}"
        assert dx0_err <= 1e-6, f"step0 dx err {dx0_err}"

        topos = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4)]
        worst_l = 0.0
        worst_w = 0.0
        for dp, tp in topos:
            w = read_f32_file(d / f"weights_dp{dp}_tp{tp}.f32", FULL_PARAMS)
            ls = read_f64_file(d / f"losses_dp{dp}_tp{tp}.f64", STEPS)
            le = max_abs_diff(ls, ref_losses)
            we = max_abs_diff(w, ref_params)
            assert le <= 1e-6, f"({dp}x{tp}) loss err {le} > 1e-6"
            assert we <= 1e-6, f"({dp}x{tp}) weight err {we} > 1e-6"
            worst_l = max(worst_l, le)
            worst_w = max(worst_w, we)

        print(
            f"PYTHON ORACLE NATIVE: step0(loss={l0_err:.2e}, grad={g0_err:.2e}, dx={dx0_err:.2e}) "
            f"all_6_topos(max_loss_err={worst_l:.2e}, max_weight_err={worst_w:.2e}) PASS"
        )


def verify_aot_dumps(paths: list[str]) -> None:
    ref_params, _, _, _ = run_reference_train10()
    loaded = [(Path(p).name, read_f32_file(Path(p), FULL_PARAMS)) for p in paths]
    base_name, base_w = loaded[0]
    worst_vs_ref = 0.0
    worst_vs_base = 0.0
    for name, w in loaded:
        err_ref = max_abs_diff(w, ref_params)
        err_base = max_abs_diff(w, base_w)
        assert err_ref <= 1e-6, f"AOT {name} vs oracle err {err_ref} > 1e-6"
        assert err_base <= 1e-6, f"AOT {name} vs {base_name} err {err_base} > 1e-6"
        worst_vs_ref = max(worst_vs_ref, err_ref)
        worst_vs_base = max(worst_vs_base, err_base)
    print(
        f"PYTHON ORACLE AOT ({len(loaded)} meshes): "
        f"max_vs_1x1={worst_vs_base:.2e} max_vs_f64_ref={worst_vs_ref:.2e} PASS"
    )


def main() -> int:
    if len(sys.argv) == 2:
        verify_native_binary(sys.argv[1])
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == "--check-aot":
        verify_aot_dumps(sys.argv[2:])
        return 0
    print(f"Usage: {sys.argv[0]} <native_bin> | --check-aot <w1.bin> ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
