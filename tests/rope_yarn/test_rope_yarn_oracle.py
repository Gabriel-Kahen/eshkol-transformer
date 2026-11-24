"""Independent binary64 reference oracle for RoPE Long-Context Extension Engine
(STANDARD, LINEAR_PI, NTK_AWARE, YARN + mscale, NEOX_SPLIT_HALF & INTERLEAVED,
partial rotary, central FD gradients, relative-position QK dot invariance, and
negative rejections).
"""
from __future__ import annotations

import math
import subprocess
import sys

MODE_STANDARD = 0
MODE_LINEAR_PI = 1
MODE_NTK_AWARE = 2
MODE_YARN = 3

LAYOUT_NEOX = 0
LAYOUT_INTERLEAVED = 1


def compute_freqs_ref(
    mode: int,
    d_rot: int,
    orig_max_pos: int,
    base: float,
    scale_factor: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[list[float], float]:
    half = d_rot // 2
    mscale = 1.0
    if mode == MODE_STANDARD:
        inv_freq = [math.pow(base, (-2.0 * i) / d_rot) for i in range(half)]
    elif mode == MODE_LINEAR_PI:
        inv_freq = [math.pow(base, (-2.0 * i) / d_rot) / scale_factor for i in range(half)]
    elif mode == MODE_NTK_AWARE:
        eff_base = base * math.pow(scale_factor, d_rot / (d_rot - 2))
        inv_freq = [math.pow(eff_base, (-2.0 * i) / d_rot) for i in range(half)]
    elif mode == MODE_YARN:
        inv_freq = []
        for i in range(half):
            orig = math.pow(base, (-2.0 * i) / d_rot)
            wavelength = (2.0 * math.pi) / orig
            rotations = orig_max_pos / wavelength
            gamma = (rotations - beta_slow) / (beta_fast - beta_slow)
            gamma = max(0.0, min(1.0, gamma))
            inv_freq.append((1.0 - gamma) * (orig / scale_factor) + gamma * orig)
        mscale = 1.0 if scale_factor <= 1.0 else (0.1 * math.log(scale_factor) + 1.0)
    else:
        raise AssertionError(f"unknown mode {mode}")
    return inv_freq, mscale


def rope_apply_ref(
    layout: int,
    head_dim: int,
    rotary_dim: int,
    inv_freq: list[float],
    mscale: float,
    batch: int,
    tokens: int,
    heads: int,
    positions: list[int],
    values: list[float],
    backward: bool,
) -> list[float]:
    half = rotary_dim // 2
    out = [0.0] * len(values)
    for n in range(batch):
        for t in range(tokens):
            pos = positions[n * tokens + t]
            cos_m = [mscale * math.cos(pos * inv_freq[k]) for k in range(half)]
            sin_m = [mscale * math.sin(pos * inv_freq[k]) for k in range(half)]
            for h in range(heads):
                base = ((n * tokens + t) * heads + h) * head_dim
                for k in range(half):
                    if layout == LAYOUT_INTERLEAVED:
                        i0 = base + 2 * k
                        i1 = base + 2 * k + 1
                    else:
                        i0 = base + k
                        i1 = base + half + k
                    v0 = values[i0]
                    v1 = values[i1]
                    c = cos_m[k]
                    s = sin_m[k]
                    if backward:
                        out[i0] = v0 * c + v1 * s
                        out[i1] = -v0 * s + v1 * c
                    else:
                        out[i0] = v0 * c - v1 * s
                        out[i1] = v0 * s + v1 * c
                for d in range(rotary_dim, head_dim):
                    out[base + d] = values[base + d]
    return out


def relative_dot_ref(
    layout: int,
    head_dim: int,
    rotary_dim: int,
    inv_freq: list[float],
    mscale: float,
    q: list[float],
    k: list[float],
    pos_m: int,
    pos_n: int,
) -> float:
    half = rotary_dim // 2
    rel = pos_m - pos_n
    rot_dot = 0.0
    for i in range(half):
        if layout == LAYOUT_INTERLEAVED:
            i0, i1 = 2 * i, 2 * i + 1
        else:
            i0, i1 = i, half + i
        phi = rel * inv_freq[i]
        c = math.cos(phi)
        s = math.sin(phi)
        rq0 = q[i0] * c - q[i1] * s
        rq1 = q[i0] * s + q[i1] * c
        rot_dot += rq0 * k[i0] + rq1 * k[i1]
    tail_dot = sum(q[d] * k[d] for d in range(rotary_dim, head_dim))
    return (mscale * mscale) * rot_dot + tail_dot


def verify(binary: str) -> None:
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())

    cases = 0
    comparisons = 0
    fd_checks = 0
    max_freq_err = 0.0
    max_mscale_err = 0.0
    max_val_err = 0.0
    max_fd_err = 0.0
    max_inv_err = 0.0
    neg_seen: dict[str, int] = {}

    for line in lines:
        parts = line.split()
        if parts[0] == "NEG":
            if len(parts) != 3:
                raise AssertionError(f"malformed NEG line: {line!r}")
            neg_seen[parts[1]] = int(parts[2])
            continue
        if len(parts) != 13 or parts[0] != "CASE":
            raise AssertionError(f"malformed oracle header: {line!r}")

        mode = int(parts[1])
        layout = int(parts[2])
        head_dim = int(parts[3])
        rotary_dim = int(parts[4])
        orig_max_pos = int(parts[5])
        base = float.fromhex(parts[6])
        scale_factor = float.fromhex(parts[7])
        beta_fast = float.fromhex(parts[8])
        beta_slow = float.fromhex(parts[9])
        batch = int(parts[10])
        tokens = int(parts[11])
        heads = int(parts[12])

        freq_tokens = next(lines).split()
        actual_inv_freq = [float.fromhex(v) for v in freq_tokens[:-1]]
        actual_mscale = float.fromhex(freq_tokens[-1])
        positions = [int(v) for v in next(lines).split()]
        x = [float.fromhex(v) for v in next(lines).split()]
        dy = [float.fromhex(v) for v in next(lines).split()]
        actual_y = [float.fromhex(v) for v in next(lines).split()]
        actual_dx = [float.fromhex(v) for v in next(lines).split()]
        inv_parts = next(lines).split()
        if len(inv_parts) != 10 or inv_parts[0] != "INV":
            raise AssertionError(f"malformed INV line: {inv_parts!r}")

        exp_inv_freq, exp_mscale = compute_freqs_ref(
            mode, rotary_dim, orig_max_pos, base, scale_factor, beta_fast, beta_slow
        )
        if len(actual_inv_freq) != len(exp_inv_freq):
            raise AssertionError("inv_freq length mismatch")
        for got, want in zip(actual_inv_freq, exp_inv_freq):
            err = abs(got - want)
            if err > 1e-13 * max(1.0, abs(want)):
                raise AssertionError(f"inv_freq mismatch: {got} vs {want} (err={err})")
            max_freq_err = max(max_freq_err, err)

        m_err = abs(actual_mscale - exp_mscale)
        if m_err > 1e-14:
            raise AssertionError(f"mscale mismatch: {actual_mscale} vs {exp_mscale}")
        max_mscale_err = max(max_mscale_err, m_err)

        exp_y = rope_apply_ref(
            layout, head_dim, rotary_dim, exp_inv_freq, exp_mscale,
            batch, tokens, heads, positions, x, False
        )
        exp_dx = rope_apply_ref(
            layout, head_dim, rotary_dim, exp_inv_freq, exp_mscale,
            batch, tokens, heads, positions, dy, True
        )

        for kind, actual, expected in (("forward", actual_y, exp_y), ("backward", actual_dx, exp_dx)):
            if len(actual) != len(expected):
                raise AssertionError(f"{kind} length mismatch")
            for idx, (got, want) in enumerate(zip(actual, expected)):
                diff = abs(got - want)
                if diff > 1e-6:
                    raise AssertionError(
                        f"RoPE YaRN {kind} case={cases} idx={idx}: got={got:.10g}, want={want:.10g}, diff={diff:.3e}"
                    )
                max_val_err = max(max_val_err, diff)
                comparisons += 1

        # Independent Python central FD verification against native backward dx.
        eps = 1e-5
        sample_indices = sorted({0, rotary_dim // 2 - 1, rotary_dim - 1, head_dim - 1, len(x) - 1})
        x_work = list(x)
        for idx in sample_indices:
            orig = x_work[idx]
            x_work[idx] = orig + eps
            yp = rope_apply_ref(
                layout, head_dim, rotary_dim, exp_inv_freq, exp_mscale,
                batch, tokens, heads, positions, x_work, False
            )
            x_work[idx] = orig - eps
            ym = rope_apply_ref(
                layout, head_dim, rotary_dim, exp_inv_freq, exp_mscale,
                batch, tokens, heads, positions, x_work, False
            )
            x_work[idx] = orig
            lp = sum(a * b for a, b in zip(yp, dy))
            lm = sum(a * b for a, b in zip(ym, dy))
            num = (lp - lm) / (2.0 * eps)
            err = abs(actual_dx[idx] - num)
            if err > 2e-6:
                raise AssertionError(
                    f"Python FD mismatch case={cases} idx={idx}: native_dx={actual_dx[idx]}, fd={num}, err={err}"
                )
            max_fd_err = max(max_fd_err, err)
            fd_checks += 1

        # Independent relative-position QK dot-product verification.
        pos_m = int(inv_parts[1])
        pos_n = int(inv_parts[2])
        dot_mn = float.fromhex(inv_parts[4])
        dot_sh = float.fromhex(inv_parts[5])
        dot_rel = float.fromhex(inv_parts[6])
        shift_err = float.fromhex(inv_parts[7])
        rel_err = float.fromhex(inv_parts[8])
        inv_mscale = float.fromhex(inv_parts[9])

        exp_rel_dot = relative_dot_ref(
            layout, head_dim, rotary_dim, exp_inv_freq, exp_mscale,
            x[:head_dim], dy[:head_dim], pos_m, pos_n
        )
        if abs(inv_mscale - exp_mscale) > 1e-14:
            raise AssertionError(f"INV mscale mismatch on case={cases}")
        if abs(dot_rel - exp_rel_dot) > 1e-12:
            raise AssertionError(f"INV relative_ref mismatch: {dot_rel} vs {exp_rel_dot}")
        if abs(dot_mn - exp_rel_dot) > 2e-5 or abs(dot_sh - exp_rel_dot) > 2e-5:
            raise AssertionError(f"INV dot_mn/dot_sh mismatch on case={cases}")
        if shift_err > 2e-5 or rel_err > 2e-5:
            raise AssertionError(f"INV err too high on case={cases}: shift={shift_err}, rel={rel_err}")
        max_inv_err = max(max_inv_err, shift_err, rel_err)

        cases += 1

    expected_neg = {
        "odd_d_rot": -2,
        "scale_lt_1": -1,
        "beta_order": -1,
        "ntk_d_rot_2": -2,
        "alias": -4,
        "nonfinite": -5,
        "pos_oob": -8,
    }
    if neg_seen != expected_neg:
        raise AssertionError(f"negative witness mismatch: got={neg_seen}, want={expected_neg}")
    if cases < 11:
        raise AssertionError(f"expected at least 11 oracle cases, got {cases}")

    print(
        f"RoPE YaRN Python binary64 oracle: PASS "
        f"(cases={cases}, values={comparisons}, fd_checks={fd_checks}, "
        f"max_freq_err={max_freq_err:.3e}, max_mscale_err={max_mscale_err:.3e}, "
        f"max_val_err={max_val_err:.3e}, max_fd_err={max_fd_err:.3e}, "
        f"max_inv_err={max_inv_err:.3e}, neg_witnesses={len(neg_seen)})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
