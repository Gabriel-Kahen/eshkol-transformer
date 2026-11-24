"""Independent Python binary64 dense reference & FD oracle for FlashAttention-2."""
from __future__ import annotations

import math
import subprocess
import sys


def pair_allowed(q_pos: int, k_pos: int, causal: int, window: int) -> bool:
    if causal:
        if k_pos > q_pos:
            return False
        if window > 0 and (q_pos - k_pos) >= window:
            return False
        return True
    if window > 0 and abs(q_pos - k_pos) >= window:
        return False
    return True


def expected_tile_counts(
    b: int, hq: int, tq: int, tk: int, br: int, bc: int, causal: int, window: int
) -> tuple[int, int]:
    evaluated = 0
    skipped = 0
    for qs in range(0, tq, br):
        qe = min(qs + br, tq)
        for ks in range(0, tk, bc):
            ke = min(ks + bc, tk)
            any_ok = any(
                pair_allowed(i, j, causal, window)
                for i in range(qs, qe)
                for j in range(ks, ke)
            )
            if any_ok:
                evaluated += 1
            else:
                skipped += 1
    heads = b * hq
    return evaluated * heads, skipped * heads


def dense_attention_reference(
    b: int,
    hq: int,
    hkv: int,
    tq: int,
    tk: int,
    dh: int,
    causal: int,
    window: int,
    scale_cfg: float,
    softcap: float,
    q: list[float],
    k: list[float],
    v: list[float],
    do: list[float],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    scale = (1.0 / math.sqrt(float(dh))) if scale_cfg == 0.0 else scale_cfg
    group = hq // hkv

    o = [0.0] * (b * hq * tq * dh)
    lse = [0.0] * (b * hq * tq)
    dq = [0.0] * (b * hq * tq * dh)
    dk = [0.0] * (b * hkv * tk * dh)
    dv = [0.0] * (b * hkv * tk * dh)

    for bi in range(b):
        for h in range(hq):
            kv_h = h // group
            for i in range(tq):
                q_row = (bi * hq + h) * tq + i
                q_off = q_row * dh

                valid_j = []
                s_raws = []
                s_effs = []
                for j in range(tk):
                    if not pair_allowed(i, j, causal, window):
                        continue
                    k_off = ((bi * hkv + kv_h) * tk + j) * dh
                    dot = 0.0
                    for d in range(dh):
                        dot += q[q_off + d] * k[k_off + d]
                    sr = dot * scale
                    se = (softcap * math.tanh(sr / softcap)) if softcap > 0.0 else sr
                    valid_j.append(j)
                    s_raws.append(sr)
                    s_effs.append(se)

                if not valid_j:
                    raise AssertionError(f"empty mask at row {i}")

                m = max(s_effs)
                sum_exp = sum(math.exp(s - m) for s in s_effs)
                lse_val = m + math.log(sum_exp)
                lse[q_row] = lse_val

                probs = [math.exp(s - lse_val) for s in s_effs]
                for idx_j, j in enumerate(valid_j):
                    v_off = ((bi * hkv + kv_h) * tk + j) * dh
                    p = probs[idx_j]
                    for d in range(dh):
                        o[q_off + d] += p * v[v_off + d]

                dp_list = []
                for j in valid_j:
                    v_off = ((bi * hkv + kv_h) * tk + j) * dh
                    dp = 0.0
                    for d in range(dh):
                        dp += do[q_off + d] * v[v_off + d]
                    dp_list.append(dp)

                d_i = sum(probs[idx_j] * dp_list[idx_j] for idx_j in range(len(valid_j)))
                for idx_j, j in enumerate(valid_j):
                    kv_off = ((bi * hkv + kv_h) * tk + j) * dh
                    p = probs[idx_j]
                    ds_cap = p * (dp_list[idx_j] - d_i)
                    if softcap > 0.0:
                        t = math.tanh(s_raws[idx_j] / softcap)
                        sc_deriv = 1.0 - t * t
                    else:
                        sc_deriv = 1.0
                    ddot = ds_cap * sc_deriv * scale
                    for d in range(dh):
                        dq[q_off + d] += ddot * k[kv_off + d]
                        dk[kv_off + d] += ddot * q[q_off + d]
                        dv[kv_off + d] += p * do[q_off + d]

    return o, lse, dq, dk, dv


def check_python_fd(
    b: int,
    hq: int,
    hkv: int,
    tq: int,
    tk: int,
    dh: int,
    causal: int,
    window: int,
    scale_cfg: float,
    softcap: float,
    q: list[float],
    k: list[float],
    v: list[float],
    do: list[float],
    dq: list[float],
    dk: list[float],
    dv: list[float],
) -> float:
    eps = 1e-6
    worst = 0.0
    q_mut = list(q)
    k_mut = list(k)
    v_mut = list(v)

    def loss_fn() -> float:
        out, _, _, _, _ = dense_attention_reference(
            b, hq, hkv, tq, tk, dh, causal, window, scale_cfg, softcap,
            q_mut, k_mut, v_mut, do,
        )
        return sum(a * b_val for a, b_val in zip(out, do))

    for buf, grad in ((q_mut, dq), (k_mut, dk), (v_mut, dv)):
        step = max(1, len(buf) // 6)
        for idx in range(0, len(buf), step):
            orig = buf[idx]
            buf[idx] = orig + eps
            lp = loss_fn()
            buf[idx] = orig - eps
            lm = loss_fn()
            buf[idx] = orig
            fd = (lp - lm) / (2.0 * eps)
            diff = abs(grad[idx] - fd)
            if diff > 1e-6 + 1e-5 * abs(fd):
                raise AssertionError(
                    f"Python oracle FD mismatch idx={idx}: an={grad[idx]:.10g} fd={fd:.10g}"
                )
            if diff > worst:
                worst = diff
    return worst


def verify(binary: str) -> None:
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    cases = 0
    values = 0
    max_err_o = 0.0
    max_err_lse = 0.0
    max_err_dq = 0.0
    max_err_dk = 0.0
    max_err_dv = 0.0
    max_fd_err = 0.0

    for header in lines:
        parts = header.split()
        if len(parts) != 17 or parts[0] != "CASE":
            raise AssertionError(f"malformed oracle header: {header!r}")
        b, hq, hkv, tq, tk, dh, br, bc, causal, window = (
            int(parts[i]) for i in range(1, 11)
        )
        scale_cfg = float.fromhex(parts[11])
        softcap = float.fromhex(parts[12])
        feval, fskip, beval, bskip = (int(parts[i]) for i in range(13, 17))

        exp_eval, exp_skip = expected_tile_counts(
            b, hq, tq, tk, br, bc, causal, window
        )
        if (feval, fskip) != (exp_eval, exp_skip) or (beval, bskip) != (
            exp_eval,
            exp_skip,
        ):
            raise AssertionError(
                f"tile stats mismatch case={cases}: got fwd=({feval},{fskip}) "
                f"bwd=({beval},{bskip}), want ({exp_eval},{exp_skip})"
            )

        q = [float.fromhex(x) for x in next(lines).split()]
        k = [float.fromhex(x) for x in next(lines).split()]
        v = [float.fromhex(x) for x in next(lines).split()]
        do = [float.fromhex(x) for x in next(lines).split()]
        got_o = [float.fromhex(x) for x in next(lines).split()]
        got_lse = [float.fromhex(x) for x in next(lines).split()]
        got_dq = [float.fromhex(x) for x in next(lines).split()]
        got_dk = [float.fromhex(x) for x in next(lines).split()]
        got_dv = [float.fromhex(x) for x in next(lines).split()]

        want_o, want_lse, want_dq, want_dk, want_dv = dense_attention_reference(
            b, hq, hkv, tq, tk, dh, causal, window, scale_cfg, softcap, q, k, v, do
        )

        # Independently verify Python oracle gradients with binary64 central FD
        if cases in (0, 2, 4, 6, 7):
            fd_err = check_python_fd(
                b, hq, hkv, tq, tk, dh, causal, window, scale_cfg, softcap,
                q, k, v, do, want_dq, want_dk, want_dv,
            )
            if fd_err > max_fd_err:
                max_fd_err = fd_err

        for name, got, want in (
            ("O", got_o, want_o),
            ("LSE", got_lse, want_lse),
            ("dQ", got_dq, want_dq),
            ("dK", got_dk, want_dk),
            ("dV", got_dv, want_dv),
        ):
            if len(got) != len(want):
                raise AssertionError(f"{name} length mismatch on case {cases}")
            tol = 1e-12 if name == "LSE" else 1e-6
            for idx, (g, wv) in enumerate(zip(got, want)):
                diff = abs(g - wv)
                if diff > tol:
                    raise AssertionError(
                        f"FlashAttention-2 {name} case={cases} "
                        f"(B={b},Hq={hq},Hkv={hkv},Tq={tq},Tk={tk},Dh={dh},"
                        f"Br={br},Bc={bc},causal={causal},win={window},cap={softcap}) "
                        f"idx={idx}: got={g:.10g}, want={wv:.10g}, diff={diff:.3e}"
                    )
                if name == "O" and diff > max_err_o:
                    max_err_o = diff
                elif name == "LSE" and diff > max_err_lse:
                    max_err_lse = diff
                elif name == "dQ" and diff > max_err_dq:
                    max_err_dq = diff
                elif name == "dK" and diff > max_err_dk:
                    max_err_dk = diff
                elif name == "dV" and diff > max_err_dv:
                    max_err_dv = diff
                values += 1
        cases += 1

    if cases < 9:
        raise AssertionError(f"expected at least 9 oracle cases, got {cases}")
    print(
        f"FLASH_ATTN2 Python binary64 oracle: PASS (cases={cases}, values={values}, "
        f"max_err O={max_err_o:.3e} LSE={max_err_lse:.3e} "
        f"dQ={max_err_dq:.3e} dK={max_err_dk:.3e} dV={max_err_dv:.3e}, "
        f"py_fd_err={max_fd_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
