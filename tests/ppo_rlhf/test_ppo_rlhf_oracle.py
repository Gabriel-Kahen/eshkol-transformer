"""Independent binary64 Python reference oracle & f64 FD checker for PPO-RLHF."""
import math
import struct
import subprocess
import sys


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def read_f32_vec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def read_int_vec(lines):
    return [int(tok) for tok in next(lines).split()]


def log_softmax_row(row):
    m = max(row)
    lse = m + math.log(math.fsum(math.exp(x - m) for x in row))
    lp = [x - lse for x in row]
    p = [math.exp(y) for y in lp]
    ent = -math.fsum(pv * lpv for pv, lpv in zip(p, lp) if pv > 0.0)
    return lse, lp, p, ent


def kl_shaped_ref(B, T, beta_kl, r_env, old_lp, ref_lp, mask):
    out = []
    for i in range(B * T):
        if mask[i] == 0:
            out.append(0.0)
        else:
            out.append(r_env[i] - beta_kl * (old_lp[i] - ref_lp[i]))
    return out


def gae_ref(B, T, gamma, lam, r_shaped, values, mask):
    adv = [0.0] * (B * T)
    ret = [0.0] * (B * T)
    for b in range(B):
        next_v = 0.0
        next_adv = 0.0
        for t in range(T - 1, -1, -1):
            idx = b * T + t
            if mask[idx] == 0:
                adv[idx] = 0.0
                ret[idx] = 0.0
                next_v = 0.0
                next_adv = 0.0
            else:
                v_t = values[idx]
                delta = r_shaped[idx] + gamma * next_v - v_t
                a_t = delta + gamma * lam * next_adv
                adv[idx] = a_t
                ret[idx] = a_t + v_t
                next_v = v_t
                next_adv = a_t
    return adv, ret


def whiten_ref(B, T, eps, adv, mask):
    active = [adv[i] for i in range(B * T) if mask[i] == 1]
    cnt = float(len(active))
    mean = math.fsum(active) / cnt
    var = math.fsum((x - mean) ** 2 for x in active) / cnt
    denom = math.sqrt(var + eps)
    out = []
    for i in range(B * T):
        if mask[i] == 0:
            out.append(0.0)
        else:
            diff = adv[i] - mean
            out.append(0.0 if diff == 0.0 else diff / denom)
    return out


def ppo_loss_and_grad_ref(
    B, T, V, use_vc, clip_lo, clip_hi, v_eps, vf_coef, ent_coef,
    logits, values, old_values, actions, mask, old_lp, adv, ret
):
    bt = B * T
    active = sum(mask)
    m_d = float(active)
    c_lo = 1.0 - clip_lo
    c_hi = 1.0 + clip_hi

    sum_pg = 0.0
    sum_vl = 0.0
    sum_ent = 0.0
    sum_kl = 0.0
    clipped_tokens = 0
    v_clipped_tokens = 0
    d_logits = [0.0] * (bt * V)
    d_values = [0.0] * bt

    for i in range(bt):
        if mask[i] == 0:
            continue
        row = logits[i * V : (i + 1) * V]
        lse, lp, probs, ent = log_softmax_row(row)
        act = actions[i]
        logpi = lp[act]
        delta_lp = logpi - old_lp[i]
        ratio = math.exp(delta_lp)
        a_d = adv[i]
        r_clip = min(max(ratio, c_lo), c_hi)
        clipped = (a_d > 0.0 and ratio > c_hi) or (a_d < 0.0 and ratio < c_lo)
        pg_obj = r_clip * a_d if clipped else ratio * a_d
        d_pg = 0.0 if clipped else ratio * a_d
        kl = 0.5 * delta_lp * delta_lp

        v_d = values[i]
        r_d = ret[i]
        diff_u = v_d - r_d
        sq_u = diff_u * diff_u
        v_loss = 0.5 * sq_u
        d_v = diff_u
        v_clipped = False
        if use_vc == 1:
            dv = v_d - old_values[i]
            if dv < -v_eps or dv > v_eps:
                v_c = (old_values[i] - v_eps) if dv < -v_eps else (old_values[i] + v_eps)
                diff_c = v_c - r_d
                sq_c = diff_c * diff_c
                if sq_c > sq_u:
                    v_loss = 0.5 * sq_c
                    d_v = 0.0
                    v_clipped = True

        sum_pg += pg_obj
        sum_vl += v_loss
        sum_ent += ent
        sum_kl += kl
        if clipped:
            clipped_tokens += 1
        if v_clipped:
            v_clipped_tokens += 1

        d_values[i] = (vf_coef * d_v) / m_d
        g_lp = -d_pg / m_d
        g_ent = ent_coef / m_d
        for v in range(V):
            d_logits[i * V + v] = (
                g_lp * ((1.0 if v == act else 0.0) - probs[v])
                + g_ent * probs[v] * (lp[v] + ent)
            )

    policy_loss = -sum_pg / m_d
    value_loss = sum_vl / m_d
    mean_ent = sum_ent / m_d
    approx_kl = sum_kl / m_d
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * mean_ent

    act_r = [ret[i] for i in range(bt) if mask[i] == 1]
    act_e = [ret[i] - values[i] for i in range(bt) if mask[i] == 1]
    mean_r = math.fsum(act_r) / m_d
    mean_e = math.fsum(act_e) / m_d
    var_r = math.fsum((x - mean_r) ** 2 for x in act_r) / m_d
    var_e = math.fsum((x - mean_e) ** 2 for x in act_e) / m_d
    ev = 0.0 if var_r <= 0.0 else (1.0 - var_e / var_r)

    metrics = (
        total_loss, policy_loss, value_loss, mean_ent, approx_kl,
        float(clipped_tokens) / m_d, float(v_clipped_tokens) / m_d, ev,
        active, clipped_tokens, v_clipped_tokens
    )
    return metrics, d_logits, d_values


def actor_critic_forward_f32(B, T, D, V, feats, params):
    bt = B * T
    b_pi_off = V * D
    w_v_off = b_pi_off + V
    b_v_off = w_v_off + D
    logits = [0.0] * (bt * V)
    values = [0.0] * bt
    for i in range(bt):
        frow = feats[i * D : (i + 1) * D]
        for v in range(V):
            acc = params[b_pi_off + v]
            for d in range(D):
                acc += params[v * D + d] * frow[d]
            logits[i * V + v] = f32(acc)
        vacc = params[b_v_off]
        for d in range(D):
            vacc += params[w_v_off + d] * frow[d]
        values[i] = f32(vacc)
    return logits, values


def verify_trajectory(lines, header):
    parts = header.split()
    B, T, D, V, n_iters = (int(parts[i]) for i in range(1, 6))
    bt = B * T
    feats = read_f32_vec(lines)
    ref_p = read_f32_vec(lines)
    params = read_f32_vec(lines)
    acts = read_int_vec(lines)
    mask = read_int_vec(lines)
    r_env = read_f32_vec(lines)

    pcount = len(params)
    adam_m = [0.0] * pcount
    adam_v = [0.0] * pcount
    step = 0
    max_param_err = 0.0

    # Frozen reference logprobs
    ref_lp = [0.0] * bt
    b_pi_off = V * D
    for i in range(bt):
        if mask[i] == 0:
            continue
        frow = feats[i * D : (i + 1) * D]
        row = []
        for v in range(V):
            acc = ref_p[b_pi_off + v]
            for d in range(D):
                acc += ref_p[v * D + d] * frow[d]
            row.append(f32(acc))
        _, lp, _, _ = log_softmax_row(row)
        ref_lp[i] = f32(lp[acts[i]])

    w_v_off = b_pi_off + V
    b_v_off = w_v_off + D

    beta_kl = f32(0.05)
    gamma = f32(0.95)
    gae_lam = f32(0.90)
    adv_eps = f32(1e-6)
    clip_lo = f32(0.2)
    clip_hi = f32(0.2)
    v_eps = f32(0.2)
    vf_c = f32(0.5)
    ent_c = f32(0.02)
    lr = f32(0.05)
    b1 = f32(0.9)
    b2 = f32(0.999)
    a_eps = f32(1e-8)
    wd = f32(0.01)
    max_gn = f32(1.0)
    target_kl = f32(0.5)

    for _ in range(n_iters):
        old_logits, old_values = actor_critic_forward_f32(
            B, T, D, V, feats, params
        )
        old_lp = [0.0] * bt
        for i in range(bt):
            if mask[i] == 1:
                _, lp, _, _ = log_softmax_row(old_logits[i * V : (i + 1) * V])
                old_lp[i] = f32(lp[acts[i]])
        r_shaped = [f32(x) for x in kl_shaped_ref(B, T, beta_kl, r_env, old_lp, ref_lp, mask)]
        adv_raw, ret = gae_ref(B, T, gamma, gae_lam, r_shaped, old_values, mask)
        adv_raw = [f32(x) for x in adv_raw]
        ret = [f32(x) for x in ret]
        adv_w = [f32(x) for x in whiten_ref(B, T, adv_eps, adv_raw, mask)]

        last_m = None
        epochs_done = 0
        early = 0
        for _ep in range(3):
            cur_logits, cur_values = actor_critic_forward_f32(
                B, T, D, V, feats, params
            )
            m_ep, d_logits, d_values = ppo_loss_and_grad_ref(
                B, T, V, 1, clip_lo, clip_hi, v_eps, vf_c, ent_c,
                cur_logits, cur_values, old_values, acts, mask, old_lp, adv_w, ret
            )
            last_m = m_ep
            if f32(m_ep[4]) > target_kl:
                early = 1
                break
            g = [0.0] * pcount
            for i in range(bt):
                if mask[i] == 0:
                    continue
                frow = feats[i * D : (i + 1) * D]
                dv = f32(d_values[i])
                g[b_v_off] += dv
                for d in range(D):
                    g[w_v_off + d] += dv * frow[d]
                for v in range(V):
                    dz = f32(d_logits[i * V + v])
                    g[b_pi_off + v] += dz
                    for d in range(D):
                        g[v * D + d] += dz * frow[d]
            gf = [f32(x) for x in g]
            gnorm = math.sqrt(math.fsum(x * x for x in gf))
            scale = max_gn / gnorm if gnorm > max_gn else 1.0
            step += 1
            bc1 = 1.0 - (b1 ** step)
            bc2 = 1.0 - (b2 ** step)
            for p_i in range(pcount):
                gi = gf[p_i] * scale
                mi = b1 * adam_m[p_i] + (1.0 - b1) * gi
                vi = b2 * adam_v[p_i] + (1.0 - b2) * gi * gi
                pi = params[p_i] * (1.0 - lr * wd) - lr * (mi / bc1) / (
                    math.sqrt(vi / bc2) + a_eps
                )
                adam_m[p_i] = f32(mi)
                adam_v[p_i] = f32(vi)
                params[p_i] = f32(pi)
            epochs_done += 1

        step_line = next(lines).split()
        assert step_line[0] == "STEP"
        act_step = int(step_line[1])
        act_ep = int(step_line[2])
        act_es = int(step_line[3])
        act_loss = float.fromhex(step_line[4])
        act_kl = float.fromhex(step_line[5])
        act_params = read_f32_vec(lines)
        assert act_step == step and act_ep == epochs_done and act_es == early, (
            f"step/ep/es mismatch: {(act_step, act_ep, act_es)} vs {(step, epochs_done, early)}"
        )
        assert math.isclose(act_loss, last_m[0], rel_tol=2e-5, abs_tol=2e-6), (
            f"traj loss mismatch: native={act_loss:.9g} oracle={last_m[0]:.9g}"
        )
        assert math.isclose(act_kl, last_m[4], rel_tol=2e-5, abs_tol=2e-6), (
            f"traj kl mismatch: native={act_kl:.9g} oracle={last_m[4]:.9g}"
        )
        for a, e in zip(act_params, params):
            err = abs(a - e)
            max_param_err = max(max_param_err, err)
            assert err < 1e-5, f"traj param mismatch: {a} vs {e}"
    return max_param_err


def verify(binary: str):
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    cases = 0
    comparisons = 0
    max_abs = 0.0
    max_fd_l = 0.0
    max_fd_v = 0.0
    traj_err = None

    for header in lines:
        if header.startswith("TRAJ "):
            traj_err = verify_trajectory(lines, header)
            continue
        parts = header.split()
        if len(parts) != 14 or parts[0] != "CASE":
            raise AssertionError(f"unexpected header: {header!r}")
        B, T, V, use_vc = (int(parts[i]) for i in range(1, 5))
        gamma, lam, beta_kl, adv_eps, clip_lo, clip_hi, v_eps, vf_c, ent_c = (
            float.fromhex(parts[i]) for i in range(5, 14)
        )

        logits = read_f32_vec(lines)
        values = read_f32_vec(lines)
        old_values = read_f32_vec(lines)
        actions = read_int_vec(lines)
        mask = read_int_vec(lines)
        r_env = read_f32_vec(lines)
        old_lp = read_f32_vec(lines)
        ref_lp = read_f32_vec(lines)

        act_r_shaped = read_f32_vec(lines)
        act_adv_raw = read_f32_vec(lines)
        act_ret = read_f32_vec(lines)
        act_adv_w = read_f32_vec(lines)
        m_parts = next(lines).split()
        act_metrics = tuple(float.fromhex(m_parts[i]) for i in range(8)) + tuple(
            int(m_parts[i]) for i in range(8, 11)
        )
        act_d_logits = read_f32_vec(lines)
        act_d_values = read_f32_vec(lines)

        exp_r_shaped = kl_shaped_ref(B, T, beta_kl, r_env, old_lp, ref_lp, mask)
        exp_adv_raw, exp_ret = gae_ref(B, T, gamma, lam, exp_r_shaped, old_values, mask)
        exp_adv_w = whiten_ref(B, T, adv_eps, exp_adv_raw, mask)
        exp_metrics, exp_d_logits, exp_d_values = ppo_loss_and_grad_ref(
            B, T, V, use_vc, clip_lo, clip_hi, v_eps, vf_c, ent_c,
            logits, values, old_values, actions, mask, old_lp,
            act_adv_w, act_ret
        )

        for idx in range(8):
            va, ve = act_metrics[idx], exp_metrics[idx]
            err = abs(va - ve)
            if not math.isclose(va, ve, rel_tol=3e-6, abs_tol=2e-6):
                raise AssertionError(f"case {cases} metric[{idx}]: {va} vs {ve}")
            max_abs = max(max_abs, err)
            comparisons += 1
        if act_metrics[8:] != exp_metrics[8:]:
            raise AssertionError(
                f"case {cases} counts: {act_metrics[8:]} vs {exp_metrics[8:]}"
            )

        for name, act, exp in (
            ("r_shaped", act_r_shaped, exp_r_shaped),
            ("adv_raw", act_adv_raw, exp_adv_raw),
            ("ret", act_ret, exp_ret),
            ("adv_w", act_adv_w, exp_adv_w),
            ("d_logits", act_d_logits, exp_d_logits),
            ("d_values", act_d_values, exp_d_values),
        ):
            assert len(act) == len(exp)
            for idx, (va, ve) in enumerate(zip(act, exp)):
                err = abs(va - ve)
                if not math.isclose(va, ve, rel_tol=5e-6, abs_tol=2e-6):
                    raise AssertionError(
                        f"case {cases} {name}[{idx}]: {va} vs {ve} err={err:.3e}"
                    )
                max_abs = max(max_abs, err)
                comparisons += 1

        # Independent f64 central finite differences on d_logits and d_values
        h = 1e-6
        lw = list(logits)
        for idx in range(len(lw)):
            orig = lw[idx]
            lw[idx] = orig + h
            mp, _, _ = ppo_loss_and_grad_ref(
                B, T, V, use_vc, clip_lo, clip_hi, v_eps, vf_c, ent_c,
                lw, values, old_values, actions, mask, old_lp, act_adv_w, act_ret
            )
            lw[idx] = orig - h
            mm, _, _ = ppo_loss_and_grad_ref(
                B, T, V, use_vc, clip_lo, clip_hi, v_eps, vf_c, ent_c,
                lw, values, old_values, actions, mask, old_lp, act_adv_w, act_ret
            )
            lw[idx] = orig
            fd = (mp[0] - mm[0]) / (2.0 * h)
            err = abs(act_d_logits[idx] - fd)
            max_fd_l = max(max_fd_l, err)
            if err > 1e-6 + 1e-5 * max(abs(act_d_logits[idx]), abs(fd)):
                raise AssertionError(
                    f"case {cases} f64 FD d_logits[{idx}]: {act_d_logits[idx]} vs {fd}"
                )

        vw = list(values)
        for idx in range(len(vw)):
            orig = vw[idx]
            vw[idx] = orig + h
            mp, _, _ = ppo_loss_and_grad_ref(
                B, T, V, use_vc, clip_lo, clip_hi, v_eps, vf_c, ent_c,
                logits, vw, old_values, actions, mask, old_lp, act_adv_w, act_ret
            )
            vw[idx] = orig - h
            mm, _, _ = ppo_loss_and_grad_ref(
                B, T, V, use_vc, clip_lo, clip_hi, v_eps, vf_c, ent_c,
                logits, vw, old_values, actions, mask, old_lp, act_adv_w, act_ret
            )
            vw[idx] = orig
            fd = (mp[0] - mm[0]) / (2.0 * h)
            err = abs(act_d_values[idx] - fd)
            max_fd_v = max(max_fd_v, err)
            if err > 1e-6 + 1e-5 * max(abs(act_d_values[idx]), abs(fd)):
                raise AssertionError(
                    f"case {cases} f64 FD d_values[{idx}]: {act_d_values[idx]} vs {fd}"
                )
        cases += 1

    assert cases == 6, f"expected 6 oracle cases, got {cases}"
    assert traj_err is not None, "missing trajectory oracle check"
    print(
        f"PPO Python oracle + f64 FD + trajectory: PASS ({cases} cases, "
        f"{comparisons} values, max_abs_err={max_abs:.3e}, "
        f"max_f64_fd_logits={max_fd_l:.3e}, max_f64_fd_values={max_fd_v:.3e}, "
        f"traj_max_param_err={traj_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
