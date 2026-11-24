"""Independent binary64 Python oracle for O2 Muon + AdamW + WSD optimizer."""
from __future__ import annotations

import math
import subprocess
import sys

NS_A = 3.4445
NS_B = -4.7750
NS_C = 2.0315


def read_vec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def matmul(a, b, m, k, n):
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            out[i * n + j] = math.fsum(
                a[i * k + p] * b[p * n + j] for p in range(k)
            )
    return out


def matmul_at(a, m, k):
    """Return A @ A^T of shape [m, m] for A [m, k]."""
    out = [0.0] * (m * m)
    for i in range(m):
        for j in range(m):
            out[i * m + j] = math.fsum(
                a[i * k + p] * a[j * k + p] for p in range(k)
            )
    return out


def transpose(a, rows, cols):
    out = [0.0] * (rows * cols)
    for i in range(rows):
        for j in range(cols):
            out[j * rows + i] = a[i * cols + j]
    return out


def newton_schulz5_ref(rows, cols, g, ns_steps, eps):
    norm = math.sqrt(math.fsum(v * v for v in g))
    denom = norm + eps
    transposed = rows > cols
    if transposed:
        r, c = cols, rows
        x = [v / denom for v in transpose(g, rows, cols)]
    else:
        r, c = rows, cols
        x = [v / denom for v in g]

    for _ in range(ns_steps):
        a = matmul_at(x, r, c)
        a2 = matmul(a, a, r, r, r)
        b = [NS_B * a[i] + NS_C * a2[i] for i in range(r * r)]
        bx = matmul(b, x, r, r, c)
        x = [NS_A * x[i] + bx[i] for i in range(r * c)]

    return transpose(x, r, c) if transposed else x


def ortho_error_ref(rows, cols, u):
    if rows <= cols:
        gram = matmul_at(u, rows, cols)
        r = rows
    else:
        ut = transpose(u, rows, cols)
        gram = matmul_at(ut, cols, rows)
        r = cols
    sq = []
    for i in range(r):
        for j in range(r):
            diff = gram[i * r + j] - (1.0 if i == j else 0.0)
            sq.append(diff * diff)
    return math.sqrt(math.fsum(sq))


def wsd_ref(warmup, stable, total, max_lr, min_lr, decay_kind, step):
    stable_end = warmup + stable
    decay_steps = total - stable_end
    if step < warmup:
        return max_lr * (float(step + 1) / float(warmup)), 0
    if step < stable_end:
        return max_lr, 1
    if decay_steps == 0 or step >= total:
        return min_lr, 2
    if step == stable_end:
        return max_lr, 2
    t = float(step - stable_end) / float(decay_steps)
    if decay_kind == 0:
        return max_lr - (max_lr - min_lr) * t, 2
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t)), 2


def muon_step_ref(rows, cols, w, g, b, lr, beta, eps, wd, ns_steps, nesterov,
                  apply_wd):
    n = rows * cols
    b_next = [beta * b[i] + g[i] for i in range(n)]
    m_in = [g[i] + beta * b_next[i] for i in range(n)] if nesterov else b_next
    u = newton_schulz5_ref(rows, cols, m_in, ns_steps, eps)
    scale = math.sqrt(max(1.0, float(rows) / float(cols)))
    if apply_wd:
        w_next = [w[i] - lr * (wd * w[i] + scale * u[i]) for i in range(n)]
    else:
        w_next = [w[i] - lr * (scale * u[i]) for i in range(n)]
    return w_next, b_next


def adamw_step_ref(w, g, m, v, lr, lr_scale, step_index, beta1, beta2, eps, wd,
                   apply_wd):
    eff_lr = lr * lr_scale
    bc1 = 1.0 - (beta1 ** step_index)
    bc2 = 1.0 - (beta2 ** step_index)
    decay = (1.0 - eff_lr * wd) if apply_wd else 1.0
    m_next, v_next, w_next = [], [], []
    for wi, gi, mi, vi in zip(w, g, m, v):
        mn = beta1 * mi + (1.0 - beta1) * gi
        vn = beta2 * vi + (1.0 - beta2) * gi * gi
        m_hat = mn / bc1
        v_hat = vn / bc2
        base = wi * decay if apply_wd else wi
        wn = base - eff_lr * (m_hat / (math.sqrt(v_hat) + eps))
        m_next.append(mn)
        v_next.append(vn)
        w_next.append(wn)
    return w_next, m_next, v_next


def verify(binary):
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    ns_cases = 0
    wsd_cases = 0
    hybrid_cases = 0
    comparisons = 0
    max_ns_err = 0.0
    max_hybrid_err = 0.0
    max_wsd_err = 0.0

    for header in lines:
        parts = header.split()
        tag = parts[0]

        if tag == "NS5":
            rows, cols, ns_steps = int(parts[1]), int(parts[2]), int(parts[3])
            eps = float.fromhex(parts[4])
            actual_ortho = float.fromhex(parts[5])
            g = read_vec(lines)
            u_act = read_vec(lines)
            u_exp = newton_schulz5_ref(rows, cols, g, ns_steps, eps)
            exp_ortho = ortho_error_ref(rows, cols, u_exp)
            for idx, (va, ve) in enumerate(zip(u_act, u_exp)):
                err = abs(va - ve)
                if err > 1e-5:
                    raise AssertionError(
                        f"NS5 ({rows},{cols}) idx={idx}: native={va} oracle={ve} err={err:.3e}"
                    )
                max_ns_err = max(max_ns_err, err)
                comparisons += 1
            o_err = abs(actual_ortho - exp_ortho)
            if o_err > 1e-5:
                raise AssertionError(
                    f"NS5 ({rows},{cols}) ortho: native={actual_ortho} oracle={exp_ortho} err={o_err:.3e}"
                )
            max_ns_err = max(max_ns_err, o_err)
            comparisons += 1
            ns_cases += 1

        elif tag == "WSD":
            warmup, stable, total = int(parts[1]), int(parts[2]), int(parts[3])
            max_lr, min_lr = float.fromhex(parts[4]), float.fromhex(parts[5])
            decay_kind, end_step = int(parts[6]), int(parts[7])
            for expected_step in range(end_step + 1):
                pt = next(lines).split()
                assert pt[0] == "WSD_PT"
                step = int(pt[1])
                assert step == expected_step
                lr_act = float.fromhex(pt[2])
                phase_act = int(pt[3])
                lr_exp, phase_exp = wsd_ref(
                    warmup, stable, total, max_lr, min_lr, decay_kind, step
                )
                if phase_act != phase_exp:
                    raise AssertionError(
                        f"WSD phase mismatch at step {step}: native={phase_act} oracle={phase_exp}"
                    )
                err = abs(lr_act - lr_exp)
                if err > 1e-12:
                    raise AssertionError(
                        f"WSD lr mismatch at step {step}: native={lr_act} oracle={lr_exp} err={err:.3e}"
                    )
                max_wsd_err = max(max_wsd_err, err)
                comparisons += 1
            wsd_cases += 1

        elif tag == "HYBRID":
            nesterov = int(parts[1])
            m_beta = float.fromhex(parts[2])
            m_eps = float.fromhex(parts[3])
            m_wd = float.fromhex(parts[4])
            m_ns = int(parts[5])
            a_b1 = float.fromhex(parts[6])
            a_b2 = float.fromhex(parts[7])
            a_eps = float.fromhex(parts[8])
            a_wd = float.fromhex(parts[9])
            a_scale = float.fromhex(parts[10])
            w_warm, w_stab, w_tot = int(parts[11]), int(parts[12]), int(parts[13])
            w_max, w_min = float.fromhex(parts[14]), float.fromhex(parts[15])
            w_kind = int(parts[16])
            num_steps = int(parts[17])

            w_wide = read_vec(lines)
            w_tall = read_vec(lines)
            w_norm = read_vec(lines)
            w_vec = read_vec(lines)
            b_wide = [0.0] * 24
            b_tall = [0.0] * 24
            m_norm = [0.0] * 6
            v_norm = [0.0] * 6
            m_vec = [0.0] * 6
            v_vec = [0.0] * 6

            for s in range(num_steps):
                g_wide = read_vec(lines)
                g_tall = read_vec(lines)
                g_norm = read_vec(lines)
                g_vec = read_vec(lines)

                st_line = next(lines).split()
                assert st_line[0] == "STEP_OUT" and int(st_line[1]) == s
                lr_act = float.fromhex(st_line[2])
                phase_act = int(st_line[3])
                lr_exp, phase_exp = wsd_ref(
                    w_warm, w_stab, w_tot, w_max, w_min, w_kind, s
                )
                assert phase_act == phase_exp and abs(lr_act - lr_exp) <= 1e-12

                w_wide, b_wide = muon_step_ref(
                    4, 6, w_wide, g_wide, b_wide, lr_exp, m_beta, m_eps, m_wd,
                    m_ns, nesterov, True
                )
                w_tall, b_tall = muon_step_ref(
                    6, 4, w_tall, g_tall, b_tall, lr_exp, m_beta, m_eps, m_wd,
                    m_ns, nesterov, False
                )
                w_norm, m_norm, v_norm = adamw_step_ref(
                    w_norm, g_norm, m_norm, v_norm, lr_exp, a_scale, s + 1,
                    a_b1, a_b2, a_eps, a_wd, False
                )
                w_vec, m_vec, v_vec = adamw_step_ref(
                    w_vec, g_vec, m_vec, v_vec, lr_exp, a_scale, s + 1,
                    a_b1, a_b2, a_eps, a_wd, True
                )

                expected_vecs = [
                    ("w_wide", w_wide), ("b_wide", b_wide),
                    ("w_tall", w_tall), ("b_tall", b_tall),
                    ("w_norm", w_norm), ("m_norm", m_norm), ("v_norm", v_norm),
                    ("w_vec", w_vec), ("m_vec", m_vec), ("v_vec", v_vec),
                ]
                for name, exp in expected_vecs:
                    act = read_vec(lines)
                    assert len(act) == len(exp)
                    for idx, (va, ve) in enumerate(zip(act, exp)):
                        err = abs(va - ve)
                        if err > 1e-5:
                            raise AssertionError(
                                f"HYBRID nesterov={nesterov} step={s} {name}[{idx}]: "
                                f"native={va:.10g} oracle={ve:.10g} err={err:.3e}"
                            )
                        max_hybrid_err = max(max_hybrid_err, err)
                        comparisons += 1
            hybrid_cases += 1
        else:
            raise AssertionError(f"unexpected oracle tag: {tag!r}")

    assert ns_cases == 8 and wsd_cases == 5 and hybrid_cases == 2
    print(
        f"O2 MUON Python oracle: PASS (ns={ns_cases}, wsd={wsd_cases}, "
        f"hybrid={hybrid_cases}, comparisons={comparisons}, "
        f"max_ns_err={max_ns_err:.3e}, max_hybrid_err={max_hybrid_err:.3e}, "
        f"max_wsd_err={max_wsd_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
