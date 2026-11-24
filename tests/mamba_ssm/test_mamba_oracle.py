"""Independent binary64 reference oracle for Mamba / S6 Selective State-Space Model."""
from __future__ import annotations

import math
import subprocess
import sys


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def silu(x: float) -> float:
    return x * sigmoid(x)


def silu_grad(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def softplus(x: float) -> float:
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def conv1d_silu_forward(n: int, t: int, d: int, k: int,
                        u: list[float], conv_w: list[float],
                        conv_b: list[float]) -> list[float]:
    x = [0.0] * (n * t * d)
    for ni in range(n):
        for ti in range(t):
            for di in range(d):
                acc = conv_b[di]
                for ki in range(k):
                    lag = (k - 1) - ki
                    if ti >= lag:
                        tau = ti - lag
                        acc += conv_w[di * k + ki] * u[(ni * t + tau) * d + di]
                x[(ni * t + ti) * d + di] = silu(acc)
    return x


def conv1d_silu_backward(n: int, t: int, d: int, k: int,
                         u: list[float], conv_w: list[float],
                         conv_b: list[float], dx: list[float]):
    du = [0.0] * (n * t * d)
    dw = [0.0] * (d * k)
    db = [0.0] * d
    for ni in range(n):
        for ti in range(t):
            for di in range(d):
                c_val = conv_b[di]
                for ki in range(k):
                    lag = (k - 1) - ki
                    if ti >= lag:
                        tau = ti - lag
                        c_val += conv_w[di * k + ki] * u[(ni * t + tau) * d + di]
                dc = dx[(ni * t + ti) * d + di] * silu_grad(c_val)
                db[di] += dc
                for ki in range(k):
                    lag = (k - 1) - ki
                    if ti >= lag:
                        tau = ti - lag
                        u_idx = (ni * t + tau) * d + di
                        dw[di * k + ki] += dc * u[u_idx]
                        du[u_idx] += dc * conv_w[di * k + ki]
    return du, dw, db


def selective_scan_forward(n: int, t: int, d: int, s: int, gated: bool,
                           x: list[float], dt_raw: list[float],
                           a_log: list[float], b_in: list[float],
                           c_in: list[float], d_skip: list[float],
                           z_in: list[float]) -> list[float]:
    a_neg = [-math.exp(v) for v in a_log]
    out = [0.0] * (n * t * d)
    for ni in range(n):
        for di in range(d):
            h = [0.0] * s
            for ti in range(t):
                ntd_idx = (ni * t + ti) * d + di
                nts_base = (ni * t + ti) * s
                xv = x[ntd_idx]
                dtv = softplus(dt_raw[ntd_idx])
                yv = d_skip[di] * xv
                for si in range(s):
                    da = math.exp(dtv * a_neg[di * s + si])
                    h[si] = da * h[si] + dtv * b_in[nts_base + si] * xv
                    yv += h[si] * c_in[nts_base + si]
                out[ntd_idx] = yv * silu(z_in[ntd_idx]) if gated else yv
    return out


def selective_scan_backward(n: int, t: int, d: int, s: int, gated: bool,
                            x: list[float], dt_raw: list[float],
                            a_log: list[float], b_in: list[float],
                            c_in: list[float], d_skip: list[float],
                            z_in: list[float], dout: list[float]):
    a_neg = [-math.exp(v) for v in a_log]
    dx = [0.0] * (n * t * d)
    ddt_raw = [0.0] * (n * t * d)
    da_log = [0.0] * (d * s)
    db = [0.0] * (n * t * s)
    dc = [0.0] * (n * t * s)
    dd_skip = [0.0] * d
    dz = [0.0] * (n * t * d) if gated else None

    for ni in range(n):
        for di in range(d):
            h_hist = [[0.0] * s for _ in range(t)]
            da_hist = [[0.0] * s for _ in range(t)]
            dt_hist = [0.0] * t
            y_hist = [0.0] * t

            for ti in range(t):
                ntd_idx = (ni * t + ti) * d + di
                nts_base = (ni * t + ti) * s
                xv = x[ntd_idx]
                dtv = softplus(dt_raw[ntd_idx])
                dt_hist[ti] = dtv
                yv = d_skip[di] * xv
                for si in range(s):
                    h_prev = h_hist[ti - 1][si] if ti > 0 else 0.0
                    da = math.exp(dtv * a_neg[di * s + si])
                    h_cur = da * h_prev + dtv * b_in[nts_base + si] * xv
                    da_hist[ti][si] = da
                    h_hist[ti][si] = h_cur
                    yv += h_cur * c_in[nts_base + si]
                y_hist[ti] = yv

            dh_next = [0.0] * s
            for ti in range(t - 1, -1, -1):
                ntd_idx = (ni * t + ti) * d + di
                nts_base = (ni * t + ti) * s
                xv = x[ntd_idx]
                dtv = dt_hist[ti]
                dov = dout[ntd_idx]
                if gated:
                    zv = z_in[ntd_idx]
                    dy = dov * silu(zv)
                    dz[ntd_idx] = dov * y_hist[ti] * silu_grad(zv)
                else:
                    dy = dov

                dd_skip[di] += dy * xv
                dx_acc = dy * d_skip[di]
                ddt_acc = 0.0
                for si in range(s):
                    h_cur = h_hist[ti][si]
                    h_prev = h_hist[ti - 1][si] if ti > 0 else 0.0
                    da = da_hist[ti][si]
                    av = a_neg[di * s + si]
                    bv = b_in[nts_base + si]
                    cv = c_in[nts_base + si]

                    dh = dh_next[si] + dy * cv
                    dc[nts_base + si] += dy * h_cur

                    d_dt_a = dh * h_prev * da
                    ddt_acc += d_dt_a * av + dh * bv * xv
                    da_log[di * s + si] += d_dt_a * dtv * av
                    db[nts_base + si] += dh * dtv * xv
                    dx_acc += dh * dtv * bv
                    dh_next[si] = dh * da

                dx[ntd_idx] = dx_acc
                ddt_raw[ntd_idx] = ddt_acc * sigmoid(dt_raw[ntd_idx])

    return dx, ddt_raw, da_log, db, dc, dd_skip, dz


def self_verify_oracle_fd() -> float:
    """Verify the binary64 Python oracle's own backward against binary64 central FD."""
    n, t, d, s, k = 1, 5, 3, 2, 3
    ntd, nts, dk, ds = n * t * d, n * t * s, d * k, d * s
    u = [((i * 7 + 3) % 13 - 6) * 0.1 for i in range(ntd)]
    conv_w = [((i * 5 + 1) % 9 - 4) * 0.12 for i in range(dk)]
    conv_b = [0.05 * (i - 1) for i in range(d)]
    dt_raw = [((i * 11 + 2) % 11 - 5) * 0.15 for i in range(ntd)]
    a_log = [math.log(float(si + 1)) - 0.1 * di for di in range(d) for si in range(s)]
    b_in = [((i * 3 + 5) % 11 - 5) * 0.1 for i in range(nts)]
    c_in = [((i * 9 + 1) % 11 - 5) * 0.1 for i in range(nts)]
    d_skip = [0.2 + 0.1 * i for i in range(d)]
    z_in = [((i * 13 + 4) % 13 - 6) * 0.1 for i in range(ntd)]
    dout = [((i * 17 + 7) % 11 - 5) * 0.1 for i in range(ntd)]

    def loss_fn() -> float:
        xc = conv1d_silu_forward(n, t, d, k, u, conv_w, conv_b)
        out = selective_scan_forward(n, t, d, s, True, xc, dt_raw, a_log, b_in, c_in, d_skip, z_in)
        return sum(o * g for o, g in zip(out, dout))

    xc = conv1d_silu_forward(n, t, d, k, u, conv_w, conv_b)
    dx, ddt, dal, db, dc, ddsk, dz = selective_scan_backward(
        n, t, d, s, True, xc, dt_raw, a_log, b_in, c_in, d_skip, z_in, dout
    )
    du, dw, dcb = conv1d_silu_backward(n, t, d, k, u, conv_w, conv_b, dx)

    eps = 1e-6
    max_fd_err = 0.0
    for arr, grad in (
        (u, du), (conv_w, dw), (conv_b, dcb), (dt_raw, ddt),
        (a_log, dal), (b_in, db), (c_in, dc), (d_skip, ddsk), (z_in, dz),
    ):
        for idx in range(len(arr)):
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = loss_fn()
            arr[idx] = orig - eps
            lm = loss_fn()
            arr[idx] = orig
            num = (lp - lm) / (2.0 * eps)
            err = abs(grad[idx] - num)
            if err > 1e-8:
                raise AssertionError(f"Oracle binary64 FD mismatch: {err:.3e}")
            if err > max_fd_err:
                max_fd_err = err
    return max_fd_err


def verify(binary: str) -> None:
    oracle_fd_err = self_verify_oracle_fd()

    proc = subprocess.run([binary, "--oracle"], check=True,
                          capture_output=True, text=True)
    lines = iter(proc.stdout.splitlines())
    cases = 0
    comparisons = 0
    max_abs_diff = 0.0
    max_cache_diff = 0.0

    for header in lines:
        parts = header.split()
        if len(parts) != 7 or parts[0] != "CASE":
            raise AssertionError(f"malformed oracle header: {header!r}")
        n, t, d, s, k, gated_int = (int(parts[i]) for i in range(1, 7))
        gated = bool(gated_int)

        u = [float.fromhex(v) for v in next(lines).split()]
        conv_w = [float.fromhex(v) for v in next(lines).split()]
        conv_b = [float.fromhex(v) for v in next(lines).split()]
        dt_raw = [float.fromhex(v) for v in next(lines).split()]
        a_log = [float.fromhex(v) for v in next(lines).split()]
        b_in = [float.fromhex(v) for v in next(lines).split()]
        c_in = [float.fromhex(v) for v in next(lines).split()]
        d_skip = [float.fromhex(v) for v in next(lines).split()]
        z_in = [float.fromhex(v) for v in next(lines).split()]
        dout = [float.fromhex(v) for v in next(lines).split()]

        got_x = [float.fromhex(v) for v in next(lines).split()]
        got_out_full = [float.fromhex(v) for v in next(lines).split()]
        got_out_cached = [float.fromhex(v) for v in next(lines).split()]
        got_dx = [float.fromhex(v) for v in next(lines).split()]
        got_ddt = [float.fromhex(v) for v in next(lines).split()]
        got_dalog = [float.fromhex(v) for v in next(lines).split()]
        got_db = [float.fromhex(v) for v in next(lines).split()]
        got_dc = [float.fromhex(v) for v in next(lines).split()]
        got_ddskip = [float.fromhex(v) for v in next(lines).split()]
        got_dz = [float.fromhex(v) for v in next(lines).split()] if gated else None
        got_du = [float.fromhex(v) for v in next(lines).split()]
        got_dw = [float.fromhex(v) for v in next(lines).split()]
        got_db_conv = [float.fromhex(v) for v in next(lines).split()]

        want_x = conv1d_silu_forward(n, t, d, k, u, conv_w, conv_b)
        # Feed exact binary32-rounded Stage A output into Stage B reference.
        want_out = selective_scan_forward(
            n, t, d, s, gated, got_x, dt_raw, a_log, b_in, c_in, d_skip, z_in
        )
        want_dx, want_ddt, want_dalog, want_db, want_dc, want_ddskip, want_dz = (
            selective_scan_backward(
                n, t, d, s, gated, got_x, dt_raw, a_log, b_in, c_in, d_skip, z_in, dout
            )
        )
        want_du, want_dw, want_db_conv = conv1d_silu_backward(
            n, t, d, k, u, conv_w, conv_b, got_dx
        )

        pairs = [
            ("x_conv", got_x, want_x, 1e-6),
            ("out_full", got_out_full, want_out, 1e-6),
            ("out_cached", got_out_cached, want_out, 1e-6),
            ("dx_scan", got_dx, want_dx, 1e-6),
            ("ddt_raw", got_ddt, want_ddt, 1e-6),
            ("da_log", got_dalog, want_dalog, 1e-6),
            ("db", got_db, want_db, 1e-6),
            ("dc", got_dc, want_dc, 1e-6),
            ("dd_skip", got_ddskip, want_ddskip, 1e-6),
            ("du", got_du, want_du, 1e-6),
            ("dconv_w", got_dw, want_dw, 1e-6),
            ("dconv_b", got_db_conv, want_db_conv, 1e-6),
        ]
        if gated:
            pairs.append(("dz", got_dz, want_dz, 1e-6))

        for name, actual, expected, tol in pairs:
            if len(actual) != len(expected):
                raise AssertionError(f"{name} length mismatch on {(n, t, d, s, k, gated)}")
            for idx, (got, want) in enumerate(zip(actual, expected)):
                diff = abs(got - want)
                if diff > tol:
                    raise AssertionError(
                        f"MAMBA {name} shape={(n, t, d, s, k, gated)} idx={idx}: "
                        f"got {got:.10g}, want {want:.10g}, diff={diff:.3e}"
                    )
                if diff > max_abs_diff:
                    max_abs_diff = diff
                comparisons += 1

        # Exact cache vs full-sequence parity check
        for idx, (gc, gf) in enumerate(zip(got_out_cached, got_out_full)):
            cdiff = abs(gc - gf)
            if cdiff > 1e-6:
                raise AssertionError(
                    f"MAMBA cache parity mismatch shape={(n, t, d, s, k, gated)} idx={idx}: "
                    f"cached={gc:.10g}, full={gf:.10g}, diff={cdiff:.3e}"
                )
            if cdiff > max_cache_diff:
                max_cache_diff = cdiff
        cases += 1

    if cases < 8:
        raise AssertionError(f"expected at least 8 oracle cases, got {cases}")
    print(f"MAMBA_SSM Python binary64 oracle: PASS (cases={cases}, values={comparisons}, "
          f"oracle_fd_err={oracle_fd_err:.3e}, max_abs_diff={max_abs_diff:.3e}, "
          f"max_cache_diff={max_cache_diff:.3e})")


if __name__ == "__main__":
    verify(sys.argv[1])
