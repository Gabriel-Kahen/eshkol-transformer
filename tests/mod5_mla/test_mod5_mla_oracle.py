"""Independent binary64 reference oracle for MOD5 Multi-Head Latent Attention."""
from __future__ import annotations

import math
import subprocess
import sys


def matmul_nt(a: list[float], rows_a: int, k_dim: int,
              b: list[float], rows_b: int) -> list[float]:
    """Compute C = A @ B^T where A is [rows_a, k_dim] and B is [rows_b, k_dim]."""
    out = [0.0] * (rows_a * rows_b)
    for i in range(rows_a):
        a_off = i * k_dim
        c_off = i * rows_b
        for j in range(rows_b):
            b_off = j * k_dim
            acc = 0.0
            for k in range(k_dim):
                acc += a[a_off + k] * b[b_off + k]
            out[c_off + j] = acc
    return out


def lse_softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    sum_exp = sum(math.exp(s - m) for s in scores)
    lse = m + math.log(sum_exp)
    return [math.exp(s - lse) for s in scores]


def mla_reference(n: int, t: int, d: int, h: int, dh: int, dc: int,
                  x: list[float], w_q: list[float], w_dkv: list[float],
                  w_uk: list[float], w_uv: list[float], w_o: list[float],
                  dy: list[float]):
    hd = h * dh
    scale = 1.0 / math.sqrt(float(dh))

    out = [0.0] * (n * t * d)
    dx = [0.0] * (n * t * d)
    dw_q = [0.0] * (hd * d)
    dw_dkv = [0.0] * (dc * d)
    dw_uk = [0.0] * (hd * dc)
    dw_uv = [0.0] * (hd * dc)
    dw_o = [0.0] * (d * hd)

    for b in range(n):
        xb = x[b * t * d:(b + 1) * t * d]
        dyb = dy[b * t * d:(b + 1) * t * d]

        c_kv = matmul_nt(xb, t, d, w_dkv, dc)   # [T, dc]
        q = matmul_nt(xb, t, d, w_q, hd)        # [T, HD]
        k = matmul_nt(c_kv, t, dc, w_uk, hd)    # [T, HD]
        v = matmul_nt(c_kv, t, dc, w_uv, hd)    # [T, HD]

        attn = [0.0] * (t * hd)
        probs_all = [[None] * t for _ in range(h)]
        for head in range(h):
            base_h = head * dh
            for tq in range(t):
                scores = []
                for tk in range(tq + 1):
                    dot = 0.0
                    for kk in range(dh):
                        dot += q[tq * hd + base_h + kk] * k[tk * hd + base_h + kk]
                    scores.append(dot * scale)
                p = lse_softmax(scores)
                probs_all[head][tq] = p
                for kk in range(dh):
                    acc = 0.0
                    for tk in range(tq + 1):
                        acc += p[tk] * v[tk * hd + base_h + kk]
                    attn[tq * hd + base_h + kk] = acc

        ob = matmul_nt(attn, t, hd, w_o, d)
        out[b * t * d:(b + 1) * t * d] = ob

        # Backward B1: O = Attn @ W_o^T
        d_attn = [0.0] * (t * hd)
        for tok in range(t):
            for dd in range(d):
                dy_val = dyb[tok * d + dd]
                for ii in range(hd):
                    dw_o[dd * hd + ii] += dy_val * attn[tok * hd + ii]
            for ii in range(hd):
                acc = 0.0
                for dd in range(d):
                    acc += dyb[tok * d + dd] * w_o[dd * hd + ii]
                d_attn[tok * hd + ii] = acc

        # Backward B2: Causal SDPA
        d_q = [0.0] * (t * hd)
        d_k = [0.0] * (t * hd)
        d_v = [0.0] * (t * hd)
        for head in range(h):
            base_h = head * dh
            for tq in range(t):
                p = probs_all[head][tq]
                dp = []
                for tk in range(tq + 1):
                    val = 0.0
                    for kk in range(dh):
                        val += d_attn[tq * hd + base_h + kk] * v[tk * hd + base_h + kk]
                    dp.append(val)
                mean_dp = sum(p[tk] * dp[tk] for tk in range(tq + 1))
                for tk in range(tq + 1):
                    ds = p[tk] * (dp[tk] - mean_dp)
                    ddot = ds * scale
                    for kk in range(dh):
                        d_q[tq * hd + base_h + kk] += ddot * k[tk * hd + base_h + kk]
                        d_k[tk * hd + base_h + kk] += ddot * q[tq * hd + base_h + kk]
                        d_v[tk * hd + base_h + kk] += p[tk] * d_attn[tq * hd + base_h + kk]

        # Backward B3: K = C_kv @ W_uk^T, V = C_kv @ W_uv^T
        d_ckv = [0.0] * (t * dc)
        for tok in range(t):
            for ii in range(hd):
                dki = d_k[tok * hd + ii]
                dvi = d_v[tok * hd + ii]
                for r in range(dc):
                    dw_uk[ii * dc + r] += dki * c_kv[tok * dc + r]
                    dw_uv[ii * dc + r] += dvi * c_kv[tok * dc + r]
            for r in range(dc):
                acc = 0.0
                for ii in range(hd):
                    acc += (d_k[tok * hd + ii] * w_uk[ii * dc + r] +
                            d_v[tok * hd + ii] * w_uv[ii * dc + r])
                d_ckv[tok * dc + r] = acc

        # Backward B4: Q = X @ W_q^T, C_kv = X @ W_dkv^T
        for tok in range(t):
            for ii in range(hd):
                dqi = d_q[tok * hd + ii]
                for dd in range(d):
                    dw_q[ii * d + dd] += dqi * xb[tok * d + dd]
            for r in range(dc):
                dcr = d_ckv[tok * dc + r]
                for dd in range(d):
                    dw_dkv[r * d + dd] += dcr * xb[tok * d + dd]
            for dd in range(d):
                acc_q = sum(d_q[tok * hd + ii] * w_q[ii * d + dd] for ii in range(hd))
                acc_c = sum(d_ckv[tok * dc + r] * w_dkv[r * d + dd] for r in range(dc))
                dx[(b * t + tok) * d + dd] = acc_q + acc_c

    return out, dx, dw_q, dw_dkv, dw_uk, dw_uv, dw_o


def verify(binary: str) -> None:
    proc = subprocess.run([binary, "--oracle"], check=True,
                          capture_output=True, text=True)
    lines = iter(proc.stdout.splitlines())
    cases = 0
    comparisons = 0
    max_abs_diff = 0.0
    max_cached_diff = 0.0

    for header in lines:
        parts = header.split()
        if len(parts) != 8 or parts[0] != "CASE":
            raise AssertionError(f"malformed oracle header: {header!r}")
        n, t, d, h, dh, dc, _prefill = (int(parts[i]) for i in range(1, 8))
        if not (dc < 2 * h * dh):
            raise AssertionError(f"invalid compression ratio on {(h, dh, dc)}")

        x = [float.fromhex(v) for v in next(lines).split()]
        w_q = [float.fromhex(v) for v in next(lines).split()]
        w_dkv = [float.fromhex(v) for v in next(lines).split()]
        w_uk = [float.fromhex(v) for v in next(lines).split()]
        w_uv = [float.fromhex(v) for v in next(lines).split()]
        w_o = [float.fromhex(v) for v in next(lines).split()]
        dy = [float.fromhex(v) for v in next(lines).split()]

        got_o = [float.fromhex(v) for v in next(lines).split()]
        got_dx = [float.fromhex(v) for v in next(lines).split()]
        got_dw_q = [float.fromhex(v) for v in next(lines).split()]
        got_dw_dkv = [float.fromhex(v) for v in next(lines).split()]
        got_dw_uk = [float.fromhex(v) for v in next(lines).split()]
        got_dw_uv = [float.fromhex(v) for v in next(lines).split()]
        got_dw_o = [float.fromhex(v) for v in next(lines).split()]
        got_o_cached = [float.fromhex(v) for v in next(lines).split()]

        want_o, want_dx, want_dw_q, want_dw_dkv, want_dw_uk, want_dw_uv, want_dw_o = (
            mla_reference(n, t, d, h, dh, dc, x, w_q, w_dkv, w_uk, w_uv, w_o, dy)
        )

        pairs = (
            ("O", got_o, want_o, 1e-5),
            ("dX", got_dx, want_dx, 1e-5),
            ("dW_q", got_dw_q, want_dw_q, 1e-5),
            ("dW_dkv", got_dw_dkv, want_dw_dkv, 1e-5),
            ("dW_uk", got_dw_uk, want_dw_uk, 1e-5),
            ("dW_uv", got_dw_uv, want_dw_uv, 1e-5),
            ("dW_o", got_dw_o, want_dw_o, 1e-5),
            ("O_cached", got_o_cached, want_o, 1e-5),
        )
        for name, actual, expected, tol in pairs:
            if len(actual) != len(expected):
                raise AssertionError(f"{name} length mismatch on {(n, t, d, h, dh, dc)}")
            for idx, (got, want) in enumerate(zip(actual, expected)):
                diff = abs(got - want)
                if diff > tol:
                    raise AssertionError(
                        f"MLA {name} shape={(n, t, d, h, dh, dc)} idx={idx}: "
                        f"got {got:.10g}, want {want:.10g}, diff={diff:.3e}"
                    )
                if diff > max_abs_diff:
                    max_abs_diff = diff
                if name == "O_cached" and diff > max_cached_diff:
                    max_cached_diff = diff
                comparisons += 1
        cases += 1

    if cases < 8:
        raise AssertionError(f"expected at least 8 oracle cases, got {cases}")
    print(f"MOD5 MLA Python oracle: PASS (cases={cases}, values={comparisons}, "
          f"max_abs_diff={max_abs_diff:.3e}, max_cached_diff={max_cached_diff:.3e})")


if __name__ == "__main__":
    verify(sys.argv[1])
