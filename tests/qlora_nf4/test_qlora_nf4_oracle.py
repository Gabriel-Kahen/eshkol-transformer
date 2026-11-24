"""Independent binary64 Python reference oracle for the native QLoRA NF4 engine."""
from __future__ import annotations

import math
import struct
import subprocess
import sys

NF4_CODEBOOK = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
NF4_MIDPOINTS = tuple(
    0.5 * (NF4_CODEBOOK[i] + NF4_CODEBOOK[i + 1]) for i in range(15)
)


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def read_vec(lines) -> list[float]:
    return [float.fromhex(tok) for tok in next(lines).split()]


def nf4_quantize_normalized(v: float) -> int:
    q = 0
    for i, mid in enumerate(NF4_MIDPOINTS):
        if v >= mid:
            q = i + 1
        else:
            break
    return q


def nf4_quantize_ref(
    rows_out: int,
    cols_in: int,
    bs: int,
    dqs: int,
    mode: int,
    weights: list[float],
) -> tuple[bytes, list[float]]:
    num_params = rows_out * cols_in
    num_blocks = num_params // bs
    nibbles = bytearray(num_params // 2)
    absmaxes: list[float] = []

    half = bs // 2
    for b in range(num_blocks):
        blk = weights[b * bs : (b + 1) * bs]
        absmax = max(abs(v) for v in blk)
        absmaxes.append(absmax)
        if absmax == 0.0:
            for p in range(half):
                nibbles[b * half + p] = 0x77
        else:
            for p in range(half):
                q0 = nf4_quantize_normalized(blk[2 * p] / absmax)
                q1 = nf4_quantize_normalized(blk[2 * p + 1] / absmax)
                nibbles[b * half + p] = (q0 & 0x0F) | ((q1 & 0x0F) << 4)

    out = bytearray(nibbles)
    if mode == 1:
        for s in absmaxes:
            out.extend(struct.pack("<f", s))
        rec_scales = list(absmaxes)
    elif mode == 2:
        num_dq = (num_blocks + dqs - 1) // dqs
        qs = bytearray(num_blocks)
        dq_meta = bytearray()
        rec_scales = [0.0] * num_blocks
        for g in range(num_dq):
            start = g * dqs
            end = min(start + dqs, num_blocks)
            grp = absmaxes[start:end]
            offset_f32 = f32(min(grp))
            s_max = max(grp)
            rng = s_max - offset_f32
            scale_f32 = f32(rng / 255.0) if rng > 0.0 else 0.0
            dq_meta.extend(struct.pack("<ff", offset_f32, scale_f32))
            for j, s in enumerate(grp):
                if scale_f32 == 0.0:
                    q_s = 0
                else:
                    scaled = (s - offset_f32) / scale_f32
                    if scaled <= 0.0:
                        q_s = 0
                    elif scaled >= 255.0:
                        q_s = 255
                    else:
                        q_s = max(0, min(255, int(math.floor(scaled + 0.5))))
                qs[start + j] = q_s
                rec_scales[start + j] = f32(offset_f32 + scale_f32 * q_s)
        out.extend(qs)
        out.extend(dq_meta)
    else:
        raise AssertionError(f"unknown quant mode {mode}")

    # Dequantize using reconstructed block scales.
    deq = [0.0] * num_params
    for b in range(num_blocks):
        s = rec_scales[b]
        for p in range(half):
            byte = nibbles[b * half + p]
            deq[b * bs + 2 * p] = f32(s * NF4_CODEBOOK[byte & 0x0F])
            deq[b * bs + 2 * p + 1] = f32(s * NF4_CODEBOOK[(byte >> 4) & 0x0F])
    return bytes(out), deq


def linear_forward_ref(
    m: int,
    n: int,
    k: int,
    r: int,
    alpha: float,
    x: list[float],
    w_deq: list[float],
    bias: list[float],
    a: list[float],
    b: list[float],
) -> list[float]:
    scale = alpha / float(r)
    h = [0.0] * (m * r)
    for mi in range(m):
        for ri in range(r):
            h[mi * r + ri] = math.fsum(
                x[mi * k + ki] * a[ri * k + ki] for ki in range(k)
            )
    y = [0.0] * (m * n)
    for mi in range(m):
        for ni in range(n):
            base = math.fsum(
                w_deq[ni * k + ki] * x[mi * k + ki] for ki in range(k)
            )
            lora = math.fsum(
                h[mi * r + ri] * b[ni * r + ri] for ri in range(r)
            )
            y[mi * n + ni] = f32(base + bias[ni] + scale * lora)
    return y


def linear_backward_ref(
    m: int,
    n: int,
    k: int,
    r: int,
    alpha: float,
    x: list[float],
    w_deq: list[float],
    a: list[float],
    b: list[float],
    dy: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    scale = alpha / float(r)
    h = [0.0] * (m * r)
    dh = [0.0] * (m * r)
    for mi in range(m):
        for ri in range(r):
            h[mi * r + ri] = math.fsum(
                x[mi * k + ki] * a[ri * k + ki] for ki in range(k)
            )
            dh[mi * r + ri] = scale * math.fsum(
                dy[mi * n + ni] * b[ni * r + ri] for ni in range(n)
            )
    db = [0.0] * (n * r)
    for ni in range(n):
        for ri in range(r):
            db[ni * r + ri] = f32(
                scale * math.fsum(dy[mi * n + ni] * h[mi * r + ri] for mi in range(m))
            )
    da = [0.0] * (r * k)
    for ri in range(r):
        for ki in range(k):
            da[ri * k + ki] = f32(
                math.fsum(dh[mi * r + ri] * x[mi * k + ki] for mi in range(m))
            )
    dx = [0.0] * (m * k)
    for mi in range(m):
        for ki in range(k):
            base_g = math.fsum(
                dy[mi * n + ni] * w_deq[ni * k + ki] for ni in range(n)
            )
            lora_g = math.fsum(
                dh[mi * r + ri] * a[ri * k + ki] for ri in range(r)
            )
            dx[mi * k + ki] = f32(base_g + lora_g)
    dbias = [
        f32(math.fsum(dy[mi * n + ni] for mi in range(m))) for ni in range(n)
    ]
    return dx, da, db, dbias


def merge_ref(
    n: int, k: int, r: int, alpha: float, w_deq: list[float], a: list[float], b: list[float]
) -> list[float]:
    scale = alpha / float(r)
    out = [0.0] * (n * k)
    for ni in range(n):
        for ki in range(k):
            delta = scale * math.fsum(
                b[ni * r + ri] * a[ri * k + ki] for ri in range(r)
            )
            out[ni * k + ki] = f32(w_deq[ni * k + ki] + delta)
    return out


def silu(z: float) -> float:
    return z / (1.0 + math.exp(-z)) if z >= 0.0 else (z * math.exp(z)) / (1.0 + math.exp(z))


def silu_grad(z: float) -> float:
    s = 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))
    return s * (1.0 + z * (1.0 - s))


def adamw_update(
    params: list[float],
    grads: list[float],
    m_st: list[float],
    v_st: list[float],
    t: int,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    wd: float,
) -> None:
    bc1 = 1.0 - (b1 ** t)
    bc2 = 1.0 - (b2 ** t)
    decay = 1.0 - lr * wd
    for i in range(len(params)):
        g = grads[i]
        mn = b1 * m_st[i] + (1.0 - b1) * g
        vn = b2 * v_st[i] + (1.0 - b2) * g * g
        mh = mn / bc1
        vh = vn / bc2
        pn = params[i] * decay - lr * (mh / (math.sqrt(vh) + eps))
        m_st[i] = f32(mn)
        v_st[i] = f32(vn)
        params[i] = f32(pn)


def assert_close_vec(name: str, act: list[float], exp: list[float], tol: float) -> float:
    if len(act) != len(exp):
        raise AssertionError(f"{name}: length mismatch {len(act)} != {len(exp)}")
    max_err = 0.0
    for idx, (va, ve) in enumerate(zip(act, exp)):
        err = abs(va - ve)
        if not math.isclose(va, ve, rel_tol=tol, abs_tol=tol):
            raise AssertionError(
                f"{name}[{idx}]: native={va:.10g} oracle={ve:.10g} err={err:.3e}"
            )
        if err > max_err:
            max_err = err
    return max_err


def verify(binary: str) -> None:
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    linear_cases = 0
    mlp_cases = 0
    max_linear_err = 0.0
    max_mlp_err = 0.0

    for header in lines:
        parts = header.split()
        if parts[0] == "LINEAR":
            m, n, k, r, bs, dqs, mode = (int(parts[i]) for i in range(1, 8))
            alpha = float.fromhex(parts[8])
            w_f32 = read_vec(lines)
            x = read_vec(lines)
            bias = read_vec(lines)
            a = read_vec(lines)
            b = read_vec(lines)
            dy = read_vec(lines)

            act_packed = bytes.fromhex(next(lines).strip())
            act_deq = read_vec(lines)
            act_y = read_vec(lines)
            act_dx = read_vec(lines)
            act_da = read_vec(lines)
            act_db = read_vec(lines)
            act_dbias = read_vec(lines)
            act_merged = read_vec(lines)

            exp_packed, exp_deq = nf4_quantize_ref(n, k, bs, dqs, mode, w_f32)
            if act_packed != exp_packed:
                raise AssertionError(
                    f"case {linear_cases}: packed NF4 bytes differ from Python oracle!"
                )
            exp_y = linear_forward_ref(m, n, k, r, alpha, x, exp_deq, bias, a, b)
            exp_dx, exp_da, exp_db, exp_dbias = linear_backward_ref(
                m, n, k, r, alpha, x, exp_deq, a, b, dy
            )
            exp_merged = merge_ref(n, k, r, alpha, exp_deq, a, b)

            for tag, act_v, exp_v in (
                ("W_deq", act_deq, exp_deq),
                ("Y", act_y, exp_y),
                ("dX", act_dx, exp_dx),
                ("dA", act_da, exp_da),
                ("dB", act_db, exp_db),
                ("dbias", act_dbias, exp_dbias),
                ("W_merged", act_merged, exp_merged),
            ):
                err = assert_close_vec(f"case{linear_cases}.{tag}", act_v, exp_v, 5e-6)
                max_linear_err = max(max_linear_err, err)
            linear_cases += 1

        elif parts[0] == "MLP":
            m, d_in, d_h, d_out, r, bs, dqs, mode = (int(parts[i]) for i in range(1, 9))
            alpha = float.fromhex(parts[9])
            lr = float.fromhex(parts[10])
            b1 = float.fromhex(parts[11])
            b2 = float.fromhex(parts[12])
            eps = float.fromhex(parts[13])
            wd = float.fromhex(parts[14])
            steps = int(parts[15])

            w1_f = read_vec(lines)
            w2_f = read_vec(lines)
            bias1 = read_vec(lines)
            bias2 = read_vec(lines)
            x = read_vec(lines)
            target = read_vec(lines)
            a1 = read_vec(lines)
            b1_l = read_vec(lines)
            a2 = read_vec(lines)
            b2_l = read_vec(lines)

            act_losses = read_vec(lines)
            act_a1 = read_vec(lines)
            act_b1 = read_vec(lines)
            act_a2 = read_vec(lines)
            act_b2 = read_vec(lines)
            act_y = read_vec(lines)

            _, w1_deq = nf4_quantize_ref(d_h, d_in, bs, dqs, mode, w1_f)
            _, w2_deq = nf4_quantize_ref(d_out, d_h, bs, dqs, mode, w2_f)

            ma1 = [0.0] * len(a1)
            va1 = [0.0] * len(a1)
            mb1 = [0.0] * len(b1_l)
            vb1 = [0.0] * len(b1_l)
            ma2 = [0.0] * len(a2)
            va2 = [0.0] * len(a2)
            mb2 = [0.0] * len(b2_l)
            vb2 = [0.0] * len(b2_l)

            exp_losses: list[float] = []
            y_n = m * d_out
            for s in range(steps):
                z1 = linear_forward_ref(m, d_h, d_in, r, alpha, x, w1_deq, bias1, a1, b1_l)
                h1 = [f32(silu(v)) for v in z1]
                y = linear_forward_ref(m, d_out, d_h, r, alpha, h1, w2_deq, bias2, a2, b2_l)
                loss = 0.5 * math.fsum((yv - tv) ** 2 for yv, tv in zip(y, target)) / float(y_n)
                exp_losses.append(loss)
                dy = [f32((yv - tv) / float(y_n)) for yv, tv in zip(y, target)]
                dh1, da2, db2, _ = linear_backward_ref(
                    m, d_out, d_h, r, alpha, h1, w2_deq, a2, b2_l, dy
                )
                dz1 = [f32(g * silu_grad(zv)) for g, zv in zip(dh1, z1)]
                _, da1, db1, _ = linear_backward_ref(
                    m, d_h, d_in, r, alpha, x, w1_deq, a1, b1_l, dz1
                )
                t = s + 1
                adamw_update(a1, da1, ma1, va1, t, lr, b1, b2, eps, wd)
                adamw_update(b1_l, db1, mb1, vb1, t, lr, b1, b2, eps, wd)
                adamw_update(a2, da2, ma2, va2, t, lr, b1, b2, eps, wd)
                adamw_update(b2_l, db2, mb2, vb2, t, lr, b1, b2, eps, wd)

            z1 = linear_forward_ref(m, d_h, d_in, r, alpha, x, w1_deq, bias1, a1, b1_l)
            h1 = [f32(silu(v)) for v in z1]
            exp_y = linear_forward_ref(m, d_out, d_h, r, alpha, h1, w2_deq, bias2, a2, b2_l)

            for tag, act_v, exp_v in (
                ("losses", act_losses, exp_losses),
                ("A1", act_a1, a1),
                ("B1", act_b1, b1_l),
                ("A2", act_a2, a2),
                ("B2", act_b2, b2_l),
                ("Y", act_y, exp_y),
            ):
                err = assert_close_vec(f"mlp.{tag}", act_v, exp_v, 1e-5)
                max_mlp_err = max(max_mlp_err, err)
            mlp_cases += 1
        else:
            raise AssertionError(f"unexpected oracle header: {header!r}")

    if linear_cases != 4 or mlp_cases != 1:
        raise AssertionError(
            f"expected 4 linear + 1 MLP oracle cases, saw {linear_cases} + {mlp_cases}"
        )
    print(
        f"QLORA Python oracle: PASS ({linear_cases} linear exact-packed cases "
        f"max_err={max_linear_err:.3e}, {mlp_cases} 12-step AdamW MLP max_err={max_mlp_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
