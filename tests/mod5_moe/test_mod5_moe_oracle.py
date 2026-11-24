"""Independent binary64 reference oracle for MOD5 Sparse Top-K MoE forward & backward."""
from __future__ import annotations

import math
import subprocess
import sys


def stable_sigmoid(g: float) -> float:
    if g >= 0.0:
        return 1.0 / (1.0 + math.exp(-g))
    eg = math.exp(g)
    return eg / (1.0 + eg)


def moe_reference(
    n: int,
    t: int,
    d: int,
    f: int,
    e: int,
    k: int,
    alpha_aux: float,
    x: list[float],
    w_gate: list[float],
    w_ge: list[float],
    w_ue: list[float],
    w_de: list[float],
    dy: list[float],
) -> tuple[list[float], float, list[int], int, list[float], list[float], list[float], list[float], list[float]]:
    total_tokens = n * t
    exp_stride = f * d

    probs: list[list[float]] = []
    top_idx: list[list[int]] = []
    top_w: list[list[float]] = []
    top_sum_p: list[float] = []
    counts = [0] * e
    p_sum = [0.0] * e

    for s in range(total_tokens):
        x_row = x[s * d : (s + 1) * d]
        logits = [
            sum(x_row[j] * w_gate[ex * d + j] for j in range(d))
            for ex in range(e)
        ]
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        denom = sum(exps)
        p = [v / denom for v in exps]
        probs.append(p)
        for ex in range(e):
            p_sum[ex] += p[ex]

        # Deterministic top-K with lowest-expert-index tie-breaking.
        ranked = sorted(range(e), key=lambda ex: (-p[ex], ex))
        chosen = ranked[:k]
        sum_p = sum(p[ex] for ex in chosen)
        weights = [p[ex] / sum_p for ex in chosen]
        for ex in chosen:
            counts[ex] += 1
        top_idx.append(chosen)
        top_w.append(weights)
        top_sum_p.append(sum_p)

    aux_loss = alpha_aux * e * sum(
        (counts[ex] / (total_tokens * k)) * (p_sum[ex] / total_tokens)
        for ex in range(e)
    )

    y = [0.0] * (total_tokens * d)
    dx = [0.0] * (total_tokens * d)
    dw_gate = [0.0] * (e * d)
    dw_ge = [0.0] * (e * exp_stride)
    dw_ue = [0.0] * (e * exp_stride)
    dw_de = [0.0] * (e * exp_stride)
    active_evals = 0

    for s in range(total_tokens):
        x_row = x[s * d : (s + 1) * d]
        dy_row = dy[s * d : (s + 1) * d]
        dw_tok = [0.0] * k

        for slot in range(k):
            ex = top_idx[s][slot]
            w = top_w[s][slot]
            active_evals += 1
            base = ex * exp_stride

            g = [0.0] * f
            u = [0.0] * f
            sig = [0.0] * f
            silu = [0.0] * f
            h = [0.0] * f
            for fi in range(f):
                gv = sum(w_ge[base + fi * d + di] * x_row[di] for di in range(d))
                uv = sum(w_ue[base + fi * d + di] * x_row[di] for di in range(d))
                sv = stable_sigmoid(gv)
                sl = gv * sv
                g[fi], u[fi], sig[fi], silu[fi], h[fi] = gv, uv, sv, sl, sl * uv

            exp_out = [
                sum(w_de[base + di * f + fi] * h[fi] for fi in range(f))
                for di in range(d)
            ]
            for di in range(d):
                y[s * d + di] += w * exp_out[di]

            dw_tok[slot] = sum(dy_row[di] * exp_out[di] for di in range(d))
            do = [w * dy_row[di] for di in range(d)]

            for di in range(d):
                for fi in range(f):
                    dw_de[base + di * f + fi] += do[di] * h[fi]

            for fi in range(f):
                dh = sum(w_de[base + di * f + fi] * do[di] for di in range(d))
                du = dh * silu[fi]
                dg = (dh * u[fi]) * sig[fi] * (1.0 + g[fi] * (1.0 - sig[fi]))
                for di in range(d):
                    dw_ge[base + fi * d + di] += dg * x_row[di]
                    dw_ue[base + fi * d + di] += du * x_row[di]
                    dx[s * d + di] += (
                        w_ge[base + fi * d + di] * dg
                        + w_ue[base + fi * d + di] * du
                    )

        # Router backward (top-K renormalization + Switch aux loss).
        dp = [
            (alpha_aux * e * (counts[ex] / (total_tokens * k))) / total_tokens
            for ex in range(e)
        ]
        w_dot_dw = sum(top_w[s][j] * dw_tok[j] for j in range(k))
        for j in range(k):
            ex = top_idx[s][j]
            dp[ex] += (dw_tok[j] - w_dot_dw) / top_sum_p[s]

        p_dot_dp = sum(probs[s][ex] * dp[ex] for ex in range(e))
        for ex in range(e):
            dr = probs[s][ex] * (dp[ex] - p_dot_dp)
            for di in range(d):
                dw_gate[ex * d + di] += dr * x_row[di]
                dx[s * d + di] += dr * w_gate[ex * d + di]

    return y, aux_loss, counts, active_evals, dx, dw_gate, dw_ge, dw_ue, dw_de


def verify(binary: str) -> None:
    proc = subprocess.run([binary, "--oracle"], check=True,
                          capture_output=True, text=True)
    lines = iter(proc.stdout.splitlines())
    cases = 0
    comparisons = 0
    max_abs_diff = 0.0

    for header in lines:
        parts = header.split()
        if len(parts) != 10 or parts[0] != "CASE":
            raise AssertionError(f"malformed oracle header: {header!r}")
        n, t, d, f, e, k = (int(parts[i]) for i in range(1, 7))
        alpha_aux = float.fromhex(parts[7])
        actual_aux = float.fromhex(parts[8])
        actual_evals = int(parts[9])

        actual_counts = [int(v) for v in next(lines).split()]
        x = [float.fromhex(v) for v in next(lines).split()]
        w_gate = [float.fromhex(v) for v in next(lines).split()]
        w_ge = [float.fromhex(v) for v in next(lines).split()]
        w_ue = [float.fromhex(v) for v in next(lines).split()]
        w_de = [float.fromhex(v) for v in next(lines).split()]
        dy = [float.fromhex(v) for v in next(lines).split()]

        actual_y = [float.fromhex(v) for v in next(lines).split()]
        actual_dx = [float.fromhex(v) for v in next(lines).split()]
        actual_dwg = [float.fromhex(v) for v in next(lines).split()]
        actual_dwge = [float.fromhex(v) for v in next(lines).split()]
        actual_dwue = [float.fromhex(v) for v in next(lines).split()]
        actual_dwde = [float.fromhex(v) for v in next(lines).split()]

        (
            exp_y,
            exp_aux,
            exp_counts,
            exp_evals,
            exp_dx,
            exp_dwg,
            exp_dwge,
            exp_dwue,
            exp_dwde,
        ) = moe_reference(n, t, d, f, e, k, alpha_aux, x, w_gate, w_ge, w_ue, w_de, dy)

        if actual_evals != exp_evals or actual_counts != exp_counts:
            raise AssertionError(
                f"telemetry mismatch shape={(n,t,d,f,e,k)}: "
                f"evals={actual_evals} vs {exp_evals}, "
                f"counts={actual_counts} vs {exp_counts}"
            )

        aux_diff = abs(actual_aux - exp_aux)
        if aux_diff > 1e-5:
            raise AssertionError(
                f"aux_loss mismatch shape={(n,t,d,f,e,k)}: "
                f"got {actual_aux:.10g} want {exp_aux:.10g} diff={aux_diff:.3e}"
            )
        max_abs_diff = max(max_abs_diff, aux_diff)
        comparisons += 1

        for name, got_list, want_list in (
            ("y", actual_y, exp_y),
            ("dx", actual_dx, exp_dx),
            ("dw_gate", actual_dwg, exp_dwg),
            ("dw_gate_exp", actual_dwge, exp_dwge),
            ("dw_up_exp", actual_dwue, exp_dwue),
            ("dw_down_exp", actual_dwde, exp_dwde),
        ):
            if len(got_list) != len(want_list):
                raise AssertionError(f"{name} length mismatch on {(n,t,d,f,e,k)}")
            for idx, (got, want) in enumerate(zip(got_list, want_list)):
                diff = abs(got - want)
                if diff > 1e-5:
                    raise AssertionError(
                        f"MoE {name} shape={(n,t,d,f,e,k)} idx={idx}: "
                        f"got {got:.10g}, want {want:.10g}, diff={diff:.3e}"
                    )
                if diff > max_abs_diff:
                    max_abs_diff = diff
                comparisons += 1
        cases += 1

    if cases < 6:
        raise AssertionError(f"expected at least 6 oracle cases, got {cases}")
    print(
        f"MOD5 MoE Python oracle: PASS (cases={cases}, values={comparisons}, "
        f"max_abs_diff={max_abs_diff:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
