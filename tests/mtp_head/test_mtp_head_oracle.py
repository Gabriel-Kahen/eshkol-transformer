#!/usr/bin/env python3
"""Independent Python IEEE-754 binary64 reference oracle for DeepSeek-V3 MTP.

Invokes the compiled native C binary in `--oracle-dump` mode across D in 1..4
(untied) and D=3 (tied), and verifies:
  1. Forward losses (L_total, L_main, L_mtp_avg, L_1..L_D) <= 1e-12
  2. Exact analytic gradients (d_h0 and ALL parameters including shared E and
     shared W_lm) <= 1e-6
  3. 12-step decoupled AdamW trajectory (per-step losses, grad norms, and final
     params / m / v) <= 1e-5
  4. Self-speculative greedy draft_verify tokens & acceptance exact match
"""
from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import tempfile


def f32(x: float) -> float:
    v = struct.unpack("<f", struct.pack("<f", x))[0]
    return 0.0 if v == 0.0 else v


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def rmsnorm_fwd(x: list[float], gamma: list[float], eps: float):
    H = len(x)
    ms = sum(v * v for v in x) / H
    inv_rms = 1.0 / math.sqrt(ms + eps)
    y = [x[j] * inv_rms * gamma[j] for j in range(H)]
    return y, inv_rms


def rmsnorm_bwd_add(
    x: list[float],
    gamma: list[float],
    inv_rms: float,
    dy: list[float],
    dx_acc: list[float],
    dgamma_acc: list[float],
) -> None:
    H = len(x)
    g_dot = 0.0
    for j in range(H):
        g_dot += dy[j] * gamma[j] * x[j]
        dgamma_acc[j] += dy[j] * x[j] * inv_rms
    coeff = (inv_rms * inv_rms / H) * g_dot
    for j in range(H):
        dx_acc[j] += inv_rms * (dy[j] * gamma[j] - x[j] * coeff)


def compute_layout(V: int, H: int, F: int, D: int, tie: int) -> dict:
    off = 0
    embed_off = off
    embed_len = V * H
    off += embed_len
    if tie == 1:
        lm_off = embed_off
        lm_len = 0
    else:
        lm_off = off
        lm_len = V * H
        off += lm_len
    gm_off = off
    off += H
    rel_gh = 0
    rel_ge = rel_gh + H
    rel_wp = rel_ge + H
    rel_gf = rel_wp + H * 2 * H
    rel_wg = rel_gf + H
    rel_wu = rel_wg + F * H
    rel_wd = rel_wu + F * H
    rel_go = rel_wd + H * F
    stride = rel_go + H
    head_offs = []
    for _ in range(D):
        head_offs.append(off)
        off += stride
    return {
        "total": off,
        "embed_off": embed_off,
        "lm_off": lm_off,
        "gm_off": gm_off,
        "head_offs": head_offs,
        "rel_gh": rel_gh,
        "rel_ge": rel_ge,
        "rel_wp": rel_wp,
        "rel_gf": rel_gf,
        "rel_wg": rel_wg,
        "rel_wu": rel_wu,
        "rel_wd": rel_wd,
        "rel_go": rel_go,
    }


def mtp_block_fwd(
    params: list[float], lay: dict, V: int, H: int, F: int, eps: float,
    k: int, h_prev: list[float], tok_embed: int
) -> dict:
    base = lay["head_offs"][k]
    eo = lay["embed_off"] + tok_embed * H
    e_in = [params[eo + j] for j in range(H)]
    gh = params[base + lay["rel_gh"] : base + lay["rel_gh"] + H]
    ge = params[base + lay["rel_ge"] : base + lay["rel_ge"] + H]
    wp = params[base + lay["rel_wp"] : base + lay["rel_wp"] + H * 2 * H]
    gf = params[base + lay["rel_gf"] : base + lay["rel_gf"] + H]
    wg = params[base + lay["rel_wg"] : base + lay["rel_wg"] + F * H]
    wu = params[base + lay["rel_wu"] : base + lay["rel_wu"] + F * H]
    wd = params[base + lay["rel_wd"] : base + lay["rel_wd"] + H * F]
    go = params[base + lay["rel_go"] : base + lay["rel_go"] + H]

    u_h, inv_h = rmsnorm_fwd(h_prev, gh, eps)
    u_e, inv_e = rmsnorm_fwd(e_in, ge, eps)
    cat = u_h + u_e
    two_H = 2 * H
    z = [sum(wp[i * two_H + j] * cat[j] for j in range(two_H)) for i in range(H)]
    r, inv_f = rmsnorm_fwd(z, gf, eps)
    g = [sum(wg[m * H + j] * r[j] for j in range(H)) for m in range(F)]
    u = [sum(wu[m * H + j] * r[j] for j in range(H)) for m in range(F)]
    a = [(g[m] * sigmoid(g[m])) * u[m] for m in range(F)]
    h_out = [
        z[i] + sum(wd[i * F + m] * a[m] for m in range(F)) for i in range(H)
    ]
    o, inv_o = rmsnorm_fwd(h_out, go, eps)
    return {
        "h_in": list(h_prev),
        "e_in": e_in,
        "u_h": u_h,
        "u_e": u_e,
        "inv_h": inv_h,
        "inv_e": inv_e,
        "z": z,
        "r": r,
        "inv_f": inv_f,
        "g": g,
        "u": u,
        "a": a,
        "h_out": h_out,
        "o": o,
        "inv_o": inv_o,
    }


def lm_logits(params: list[float], lm_off: int, o: list[float], V: int, H: int):
    return [
        sum(params[lm_off + v * H + j] * o[j] for j in range(H))
        for v in range(V)
    ]


def log_softmax_ce(logits: list[float], target: int):
    mx = max(logits)
    lse = mx + math.log(sum(math.exp(v - mx) for v in logits))
    return lse - logits[target], lse


def mtp_forward_backward_py(
    params: list[float], lay: dict, V: int, H: int, F: int, D: int,
    rms_eps: float, lambda_mtp: float, B: int, T: int,
    h0: list[float], tokens: list[int]
):
    n = lay["total"]
    d_params = [0.0] * n
    d_h0 = [0.0] * (B * T * H)

    lm_off = lay["lm_off"]
    gm_off = lay["gm_off"]
    gm = params[gm_off : gm_off + H]
    scale_main = 1.0 / (B * (T - 1))
    scale_depth = [(lambda_mtp / D) / (B * (T - 2 - k)) for k in range(D)]

    sum_main = 0.0
    sum_depth = [0.0] * D

    for b in range(B):
        tok_row = tokens[b * T : (b + 1) * T]
        for t in range(T - 1):
            h0_bt = h0[(b * T + t) * H : (b * T + t + 1) * H]
            k_limit = min(D, T - 2 - t)
            caches = []
            prev = h0_bt
            for k in range(k_limit):
                c = mtp_block_fwd(
                    params, lay, V, H, F, rms_eps, k, prev, tok_row[t + k + 1]
                )
                caches.append(c)
                prev = c["h_out"]

            dh_carry = [0.0] * H
            for k in range(k_limit - 1, -1, -1):
                c = caches[k]
                base = lay["head_offs"][k]
                embed_tok = tok_row[t + k + 1]
                target_tok = tok_row[t + k + 2]

                logits = lm_logits(params, lm_off, c["o"], V, H)
                loss_s, lse = log_softmax_ce(logits, target_tok)
                sum_depth[k] += loss_s

                sc = scale_depth[k]
                do_vec = [0.0] * H
                for v in range(V):
                    dl = sc * (math.exp(logits[v] - lse) - (1.0 if v == target_tok else 0.0))
                    for j in range(H):
                        d_params[lm_off + v * H + j] += dl * c["o"][j]
                        do_vec[j] += dl * params[lm_off + v * H + j]

                dh_total = list(dh_carry)
                go_off = base + lay["rel_go"]
                dgo = [0.0] * H
                rmsnorm_bwd_add(
                    c["h_out"], params[go_off : go_off + H], c["inv_o"],
                    do_vec, dh_total, dgo
                )
                for j in range(H):
                    d_params[go_off + j] += dgo[j]

                dz = list(dh_total)
                wd_off = base + lay["rel_wd"]
                wg_off = base + lay["rel_wg"]
                wu_off = base + lay["rel_wu"]
                dg = [0.0] * F
                du = [0.0] * F
                for m in range(F):
                    da = 0.0
                    for i in range(H):
                        da += dh_total[i] * params[wd_off + i * F + m]
                        d_params[wd_off + i * F + m] += dh_total[i] * c["a"][m]
                    gm_v = c["g"][m]
                    sig = sigmoid(gm_v)
                    silu = gm_v * sig
                    silu_p = sig * (1.0 + gm_v * (1.0 - sig))
                    du[m] = da * silu
                    dg[m] = da * c["u"][m] * silu_p

                dr = [0.0] * H
                for m in range(F):
                    for j in range(H):
                        dr[j] += (
                            dg[m] * params[wg_off + m * H + j]
                            + du[m] * params[wu_off + m * H + j]
                        )
                        d_params[wg_off + m * H + j] += dg[m] * c["r"][j]
                        d_params[wu_off + m * H + j] += du[m] * c["r"][j]

                gf_off = base + lay["rel_gf"]
                dgf = [0.0] * H
                rmsnorm_bwd_add(
                    c["z"], params[gf_off : gf_off + H], c["inv_f"],
                    dr, dz, dgf
                )
                for j in range(H):
                    d_params[gf_off + j] += dgf[j]

                wp_off = base + lay["rel_wp"]
                two_H = 2 * H
                du_h = [0.0] * H
                du_e = [0.0] * H
                for i in range(H):
                    dzi = dz[i]
                    for j in range(H):
                        d_params[wp_off + i * two_H + j] += dzi * c["u_h"][j]
                        du_h[j] += dzi * params[wp_off + i * two_H + j]
                    for j in range(H):
                        d_params[wp_off + i * two_H + H + j] += dzi * c["u_e"][j]
                        du_e[j] += dzi * params[wp_off + i * two_H + H + j]

                ge_off = base + lay["rel_ge"]
                de_tok = [0.0] * H
                dge = [0.0] * H
                rmsnorm_bwd_add(
                    c["e_in"], params[ge_off : ge_off + H], c["inv_e"],
                    du_e, de_tok, dge
                )
                eo = lay["embed_off"] + embed_tok * H
                for j in range(H):
                    d_params[eo + j] += de_tok[j]
                    d_params[ge_off + j] += dge[j]

                gh_off = base + lay["rel_gh"]
                dh_prev = [0.0] * H
                dgh = [0.0] * H
                rmsnorm_bwd_add(
                    c["h_in"], params[gh_off : gh_off + H], c["inv_h"],
                    du_h, dh_prev, dgh
                )
                for j in range(H):
                    d_params[gh_off + j] += dgh[j]
                dh_carry = dh_prev

            # Main head (k = 0)
            o_main, inv_m = rmsnorm_fwd(h0_bt, gm, rms_eps)
            logits = lm_logits(params, lm_off, o_main, V, H)
            loss_s, lse = log_softmax_ce(logits, tok_row[t + 1])
            sum_main += loss_s
            do_vec = [0.0] * H
            for v in range(V):
                dl = scale_main * (math.exp(logits[v] - lse) - (1.0 if v == tok_row[t + 1] else 0.0))
                for j in range(H):
                    d_params[lm_off + v * H + j] += dl * o_main[j]
                    do_vec[j] += dl * params[lm_off + v * H + j]
            dgm = [0.0] * H
            rmsnorm_bwd_add(h0_bt, gm, inv_m, do_vec, dh_carry, dgm)
            for j in range(H):
                d_params[gm_off + j] += dgm[j]
                d_h0[(b * T + t) * H + j] = f32(dh_carry[j])

    loss_main = sum_main / (B * (T - 1))
    loss_depth = [sum_depth[k] / (B * (T - 2 - k)) for k in range(D)]
    loss_mtp_avg = sum(loss_depth) / D
    loss_total = loss_main + lambda_mtp * loss_mtp_avg
    d_params_f32 = [f32(v) for v in d_params]
    return (loss_total, loss_main, loss_mtp_avg, loss_depth), d_h0, d_params_f32


def adamw_step_py(
    params: list[float], m: list[float], v: list[float], grads: list[float],
    step: int, lr: float, b1: float, b2: float, eps: float, wd: float,
    max_gn: float
):
    gn = math.sqrt(sum(g * g for g in grads))
    clip_scale = (max_gn / gn) if (max_gn > 0.0 and gn > max_gn) else 1.0
    t = step + 1
    bc1 = 1.0 - (b1 ** t)
    bc2 = 1.0 - (b2 ** t)
    for i in range(len(params)):
        g = grads[i] * clip_scale
        mn = b1 * m[i] + (1.0 - b1) * g
        vn = b2 * v[i] + (1.0 - b2) * g * g
        mh = mn / bc1
        vh = vn / bc2
        pn = params[i] - lr * (mh / (math.sqrt(vh) + eps) + wd * params[i])
        params[i] = f32(pn)
        m[i] = f32(mn)
        v[i] = f32(vn)
    return gn


def verify_case(native_bin: str, D: int, tie: int) -> None:
    with tempfile.TemporaryDirectory(prefix="mtp_oracle_") as tmp:
        dump_path = os.path.join(tmp, f"dump_d{D}_t{tie}.bin")
        subprocess.run(
            [native_bin, "--oracle-dump", str(D), str(tie), dump_path],
            check=True,
        )
        with open(dump_path, "rb") as fp:
            blob = fp.read()

    off = 0
    def take(fmt: str):
        nonlocal off
        sz = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, blob, off)
        off += sz
        return vals

    V, H, F, D_c, tie_c, B, T, n = take("<8I")
    assert D_c == D and tie_c == tie
    rms_eps, lambda_mtp, lr, b1, b2, eps, wd, max_gn = take("<8d")

    lay = compute_layout(V, H, F, D, tie)
    assert lay["total"] == n

    init_params = list(take(f"<{n}f"))
    h0 = list(take(f"<{B * T * H}f"))
    tokens = list(take(f"<{B * T}I"))
    c_losses0 = take("<7d")
    c_d_h0 = list(take(f"<{B * T * H}f"))
    c_d_params = list(take(f"<{n}f"))

    (lt, lm, lavg, ldepths), py_d_h0, py_d_params = mtp_forward_backward_py(
        init_params, lay, V, H, F, D, rms_eps, lambda_mtp, B, T, h0, tokens
    )

    loss_err = max(
        abs(lt - c_losses0[0]),
        abs(lm - c_losses0[1]),
        abs(lavg - c_losses0[2]),
        *(abs(ldepths[k] - c_losses0[3 + k]) for k in range(D)),
    )
    assert loss_err <= 1e-12, f"D={D} tie={tie} loss_err={loss_err}"

    dh0_err = max(abs(a - b) for a, b in zip(py_d_h0, c_d_h0))
    dp_err = max(abs(a - b) for a, b in zip(py_d_params, c_d_params))
    assert dh0_err <= 1e-6, f"D={D} tie={tie} dh0_err={dh0_err}"
    assert dp_err <= 1e-6, f"D={D} tie={tie} dp_err={dp_err}"

    # Verify full 12-step AdamW trajectory
    params = list(init_params)
    m_buf = [0.0] * n
    v_buf = [0.0] * n
    max_traj_err = 0.0
    for step in range(12):
        c_row = take("<8d")
        (lt_s, lm_s, lavg_s, ldep_s), _, g_s = mtp_forward_backward_py(
            params, lay, V, H, F, D, rms_eps, lambda_mtp, B, T, h0, tokens
        )
        gn_s = adamw_step_py(params, m_buf, v_buf, g_s, step, lr, b1, b2, eps, wd, max_gn)
        step_err = max(
            abs(lt_s - c_row[0]),
            abs(lm_s - c_row[1]),
            abs(lavg_s - c_row[2]),
            *(abs(ldep_s[k] - c_row[3 + k]) for k in range(D)),
            abs(gn_s - c_row[7]),
        )
        max_traj_err = max(max_traj_err, step_err)

    c_final_p = list(take(f"<{n}f"))
    c_final_m = list(take(f"<{n}f"))
    c_final_v = list(take(f"<{n}f"))
    p_err = max(abs(a - b) for a, b in zip(params, c_final_p))
    m_err = max(abs(a - b) for a, b in zip(m_buf, c_final_m))
    v_err = max(abs(a - b) for a, b in zip(v_buf, c_final_v))
    assert max_traj_err <= 1e-6, f"D={D} tie={tie} traj_err={max_traj_err}"
    assert p_err <= 1e-6 and m_err <= 1e-6 and v_err <= 1e-6, (
        f"D={D} tie={tie} final state errs p={p_err} m={m_err} v={v_err}"
    )

    # Verify self-speculative greedy draft_verify at t=0
    c_anchor, c_acc, c_emit, c_rej = take("<4I")
    c_drafts = list(take(f"<{D}I"))
    c_emitted = list(take(f"<{c_emit}I"))
    assert off == len(blob)

    # Compute Python greedy anchor + sequential MTP drafts + target verification
    gm = params[lay["gm_off"] : lay["gm_off"] + H]
    o0, _ = rmsnorm_fwd(h0[:H], gm, rms_eps)
    l0 = lm_logits(params, lay["lm_off"], o0, V, H)
    py_anchor = max(range(V), key=lambda idx: l0[idx])
    assert py_anchor == c_anchor, f"anchor mismatch {py_anchor} vs {c_anchor}"

    h_cur = list(h0[:H])
    prev_tok = py_anchor
    py_drafts = []
    for k in range(D):
        c = mtp_block_fwd(params, lay, V, H, F, rms_eps, k, h_cur, prev_tok)
        lk = lm_logits(params, lay["lm_off"], c["o"], V, H)
        dk = max(range(V), key=lambda idx: lk[idx])
        py_drafts.append(dk)
        prev_tok = dk
        h_cur = c["h_out"]
    assert py_drafts == c_drafts, f"drafts mismatch {py_drafts} vs {c_drafts}"

    py_targets = []
    for j in range(D + 1):
        hj = h0[(1 + j) * H : (2 + j) * H]
        oj, _ = rmsnorm_fwd(hj, gm, rms_eps)
        lj = [f32(v) for v in lm_logits(params, lay["lm_off"], oj, V, H)]
        py_targets.append(max(range(V), key=lambda idx: lj[idx]))
    py_emitted = []
    py_acc = 0
    py_rej = 0
    for i in range(D):
        if py_drafts[i] == py_targets[i]:
            py_emitted.append(py_drafts[i])
            py_acc += 1
        else:
            py_emitted.append(py_targets[i])
            py_rej = 1
            break
    if py_rej == 0:
        py_emitted.append(py_targets[D])

    assert py_acc == c_acc and py_rej == c_rej and py_emitted == c_emitted, (
        f"spec verify mismatch py=({py_acc},{py_rej},{py_emitted}) "
        f"c=({c_acc},{c_rej},{c_emitted})"
    )

    print(
        f"MTP PYTHON ORACLE D={D} tie={tie} loss_err={loss_err:.3e} "
        f"dh0_err={dh0_err:.3e} dp_err={dp_err:.3e} "
        f"adamw12_err={max(max_traj_err, p_err, m_err, v_err):.3e} "
        f"spec_acc={py_acc}/{D} PASS"
    )


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(f"usage: {sys.argv[0]} <test_mtp_head_native>\n")
        return 2
    native_bin = sys.argv[1]
    for D in (1, 2, 3, 4):
        verify_case(native_bin, D, 0)
    verify_case(native_bin, 3, 1)
    print("MTP PYTHON ORACLE ALL CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
