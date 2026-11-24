"""Independent binary64 Python oracle for the native DPO alignment engine."""
import math
import subprocess
import sys


def read_fvec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def read_ivec(lines):
    return [int(tok) for tok in next(lines).split()]


def stable_softplus(x: float) -> float:
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def stable_sigmoid(x: float) -> float:
    if x >= 0.0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    ez = math.exp(x)
    return ez / (1.0 + ez)


def seq_logps(B, T, V, logits, targets, mask):
    out = [0.0] * B
    for b in range(B):
        acc = 0.0
        for t in range(T):
            bt = b * T + t
            if mask[bt] == 0:
                continue
            row = logits[bt * V : (bt + 1) * V]
            m = max(row)
            s = math.fsum(math.exp(z - m) for z in row)
            acc += (row[targets[bt]] - m) - math.log(s)
        out[b] = acc
    return out


def seq_grads(B, T, V, logits, targets, mask, scales):
    out = [0.0] * (B * T * V)
    for b in range(B):
        scale = scales[b]
        for t in range(T):
            bt = b * T + t
            if mask[bt] == 0:
                continue
            row = logits[bt * V : (bt + 1) * V]
            m = max(row)
            exps = [math.exp(z - m) for z in row]
            s = math.fsum(exps)
            tgt = targets[bt]
            base = bt * V
            for v in range(V):
                p = exps[v] / s
                ind = 1.0 if v == tgt else 0.0
                out[base + v] = scale * (ind - p)
    return out


def dpo_reference(B, T, V, ref_free, beta, eps, pi_w, pi_l, ref_w, ref_l,
                  tgt_w, tgt_l, mask_w, mask_l):
    lp_pi_w = seq_logps(B, T, V, pi_w, tgt_w, mask_w)
    lp_pi_l = seq_logps(B, T, V, pi_l, tgt_l, mask_l)
    if ref_free == 0:
        lp_ref_w = seq_logps(B, T, V, ref_w, tgt_w, mask_w)
        lp_ref_l = seq_logps(B, T, V, ref_l, tgt_l, mask_l)
    else:
        lp_ref_w = [0.0] * B
        lp_ref_l = [0.0] * B

    inv_b = 1.0 / float(B)
    losses = []
    rw_list = []
    rl_list = []
    margins = []
    accs = []
    gw = []
    gl = []

    for b in range(B):
        dw = lp_pi_w[b] - lp_ref_w[b]
        dl = lp_pi_l[b] - lp_ref_l[b]
        rw = beta * dw
        rl = beta * dl
        margin = beta * (dw - dl)
        if eps == 0.0:
            loss_b = stable_softplus(-margin)
        else:
            loss_b = (
                (1.0 - eps) * stable_softplus(-margin)
                + eps * stable_softplus(margin)
            )
        if margin >= 0.0:
            dl_dm = eps - stable_sigmoid(-margin)
        else:
            dl_dm = stable_sigmoid(margin) - (1.0 - eps)
        gw_b = (beta * inv_b) * dl_dm
        losses.append(loss_b)
        rw_list.append(rw)
        rl_list.append(rl)
        margins.append(margin)
        accs.append(1.0 if margin > 0.0 else 0.0)
        gw.append(gw_b)
        gl.append(-gw_b)

    telem = [
        math.fsum(losses) * inv_b,
        math.fsum(rw_list) * inv_b,
        math.fsum(rl_list) * inv_b,
        math.fsum(margins) * inv_b,
        math.fsum(accs) * inv_b,
    ]
    d_w = seq_grads(B, T, V, pi_w, tgt_w, mask_w, gw)
    d_l = seq_grads(B, T, V, pi_l, tgt_l, mask_l, gl)
    return telem, d_w, d_l


def verify(binary: str) -> None:
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    cases = 0
    comparisons = 0
    max_abs = 0.0
    telem_names = (
        "mean_loss",
        "chosen_reward_mean",
        "rejected_reward_mean",
        "reward_margin_mean",
        "preference_accuracy",
    )

    for header in lines:
        parts = header.split()
        if len(parts) != 7 or parts[0] != "CASE":
            raise AssertionError(f"unexpected oracle header: {header!r}")
        B, T, V, ref_free = (int(parts[i]) for i in range(1, 5))
        beta = float.fromhex(parts[5])
        eps = float.fromhex(parts[6])

        pi_w = read_fvec(lines)
        pi_l = read_fvec(lines)
        ref_w = read_fvec(lines)
        ref_l = read_fvec(lines)
        tgt_w = read_ivec(lines)
        tgt_l = read_ivec(lines)
        mask_w = read_ivec(lines)
        mask_l = read_ivec(lines)
        act_telem = read_fvec(lines)
        act_dw = read_fvec(lines)
        act_dl = read_fvec(lines)

        exp_telem, exp_dw, exp_dl = dpo_reference(
            B, T, V, ref_free, beta, eps, pi_w, pi_l, ref_w, ref_l,
            tgt_w, tgt_l, mask_w, mask_l
        )

        for name, va, ve in zip(telem_names, act_telem, exp_telem):
            err = abs(va - ve)
            if err > 1e-6:
                raise AssertionError(
                    f"case {cases} {name}: native={va:.16g} oracle={ve:.16g} "
                    f"err={err:.3e}"
                )
            if err > max_abs:
                max_abs = err
            comparisons += 1

        for gname, act_g, exp_g, mvec in (
            ("d_logits_chosen", act_dw, exp_dw, mask_w),
            ("d_logits_rejected", act_dl, exp_dl, mask_l),
        ):
            if len(act_g) != len(exp_g):
                raise AssertionError(f"{gname} length mismatch in case {cases}")
            for idx, (va, ve) in enumerate(zip(act_g, exp_g)):
                bt = idx // V
                if mvec[bt] == 0 and va != 0.0:
                    raise AssertionError(
                        f"case {cases} {gname}[{idx}] masked pos non-zero: {va}"
                    )
                err = abs(va - ve)
                if err > 1e-6:
                    raise AssertionError(
                        f"case {cases} ({B},{T},{V}) {gname}[{idx}]: "
                        f"native={va:.10g} oracle={ve:.10g} err={err:.3e}"
                    )
                if err > max_abs:
                    max_abs = err
                comparisons += 1

        cases += 1

    if cases != 8:
        raise AssertionError(f"expected 8 oracle cases, saw {cases}")
    print(
        f"DPO Python binary64 oracle: PASS ({cases} cases, {comparisons} "
        f"values, max_abs_diff={max_abs:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
