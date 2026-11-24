"""Independent double-precision Python oracle + f64 FD check for native GRPO."""
import math
import subprocess
import sys


def read_f32_vec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def read_int_vec(lines):
    return [int(tok) for tok in next(lines).split()]


def group_advantages_ref(B, G, rewards, adv_mode, eps):
    out = []
    for b in range(B):
        grp = rewards[b * G : (b + 1) * G]
        mean = math.fsum(grp) / float(G)
        if adv_mode == 1:  # ET_GRPO_ADV_DR_GRPO
            for r in grp:
                out.append(r - mean)
        else:  # ET_GRPO_ADV_STANDARD (population variance, sqrt(var) + eps)
            var = math.fsum((r - mean) ** 2 for r in grp) / float(G)
            denom = math.sqrt(var) + eps
            for r in grp:
                out.append((r - mean) / denom)
    return out


def log_softmax_row(row):
    m = max(row)
    lse = m + math.log(math.fsum(math.exp(x - m) for x in row))
    return lse, [x - lse for x in row]


def token_logprobs_ref(N, T, V, logits, targets, mask):
    tok_lp = [0.0] * (N * T)
    seq_lp = [0.0] * N
    for n in range(N):
        acc = 0.0
        for t in range(T):
            idx = n * T + t
            if mask[idx] == 0:
                tok_lp[idx] = 0.0
            else:
                row = logits[idx * V : (idx + 1) * V]
                _, ls = log_softmax_row(row)
                lp = ls[targets[idx]]
                tok_lp[idx] = lp
                acc += lp
        seq_lp[n] = acc
    return tok_lp, seq_lp


def grpo_loss_and_grad_ref(
    N, T, V, norm_mode, clip_low, clip_high, kl_beta,
    logits, targets, mask, log_old, log_ref, adv, rewards
):
    c_low = 1.0 - clip_low
    c_high = 1.0 + clip_high
    seq_counts = [sum(mask[n * T : (n + 1) * T]) for n in range(N)]
    active_tokens = sum(seq_counts)
    active_seqs = sum(1 for c in seq_counts if c > 0)
    mean_reward = math.fsum(rewards) / float(N)

    pg_loss = 0.0
    mean_kl = 0.0
    clipped_tokens = 0
    grad_log_pi = [0.0] * (N * T)
    grad_logits = [0.0] * (N * T * V)

    for n in range(N):
        if seq_counts[n] == 0:
            continue
        w = (
            1.0 / float(active_tokens)
            if norm_mode == 0
            else 1.0 / float(active_seqs * seq_counts[n])
        )
        a = adv[n]
        for t in range(T):
            idx = n * T + t
            if mask[idx] == 0:
                continue
            row = logits[idx * V : (idx + 1) * V]
            lse, ls = log_softmax_row(row)
            tgt = targets[idx]
            log_pi = ls[tgt]

            rho = math.exp(log_pi - log_old[idx])
            rho_clip = min(max(rho, c_low), c_high)
            clipped = (a > 0.0 and rho > c_high) or (a < 0.0 and rho < c_low)
            pg_obj = rho_clip * a if clipped else rho * a
            dpg = 0.0 if clipped else rho * a

            delta_ref = log_ref[idx] - log_pi
            em1 = math.expm1(delta_ref)
            kl = max(0.0, em1 - delta_ref)
            dkl = -em1

            pg_loss += w * (-pg_obj)
            mean_kl += w * kl
            if clipped:
                clipped_tokens += 1

            g_lp = w * (-dpg + kl_beta * dkl)
            grad_log_pi[idx] = g_lp
            for v in range(V):
                p = math.exp(row[v] - lse)
                grad_logits[idx * V + v] = g_lp * ((1.0 if v == tgt else 0.0) - p)

    loss = pg_loss + kl_beta * mean_kl
    clip_frac = float(clipped_tokens) / float(active_tokens)
    metrics = (
        loss, pg_loss, mean_kl, clip_frac, mean_reward,
        active_tokens, clipped_tokens, active_seqs
    )
    return metrics, grad_log_pi, grad_logits


def verify(binary):
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    cases = 0
    comparisons = 0
    max_abs = 0.0
    max_fd_rel = 0.0
    max_fd_abs = 0.0

    for header in lines:
        parts = header.split()
        if len(parts) != 11 or parts[0] != "CASE":
            raise AssertionError(f"unexpected oracle header: {header!r}")
        B, G, T, V, adv_mode, norm_mode = (int(parts[i]) for i in range(1, 7))
        adv_eps = float.fromhex(parts[7])
        clip_low = float.fromhex(parts[8])
        clip_high = float.fromhex(parts[9])
        kl_beta = float.fromhex(parts[10])
        N = B * G

        rewards = read_f32_vec(lines)
        logits = read_f32_vec(lines)
        targets = read_int_vec(lines)
        mask = read_int_vec(lines)
        log_old = read_f32_vec(lines)
        log_ref = read_f32_vec(lines)

        act_adv = read_f32_vec(lines)
        act_tok_lp = read_f32_vec(lines)
        act_seq_lp = read_f32_vec(lines)
        m_parts = next(lines).split()
        act_metrics = (
            float.fromhex(m_parts[0]),
            float.fromhex(m_parts[1]),
            float.fromhex(m_parts[2]),
            float.fromhex(m_parts[3]),
            float.fromhex(m_parts[4]),
            int(m_parts[5]),
            int(m_parts[6]),
            int(m_parts[7]),
        )
        act_grad_lp = read_f32_vec(lines)
        act_grad_logits = read_f32_vec(lines)

        exp_adv = group_advantages_ref(B, G, rewards, adv_mode, adv_eps)
        exp_tok_lp, exp_seq_lp = token_logprobs_ref(
            N, T, V, logits, targets, mask
        )
        exp_metrics, exp_grad_lp, exp_grad_logits = grpo_loss_and_grad_ref(
            N, T, V, norm_mode, clip_low, clip_high, kl_beta,
            logits, targets, mask, log_old, log_ref, exp_adv, rewards
        )

        # Check scalar metrics (5 floats + 3 exact uint32 counts).
        for idx in range(5):
            va, ve = act_metrics[idx], exp_metrics[idx]
            err = abs(va - ve)
            if not math.isclose(va, ve, rel_tol=2e-6, abs_tol=1e-6):
                raise AssertionError(
                    f"case {cases} metric[{idx}]: native={va} oracle={ve}"
                )
            max_abs = max(max_abs, err)
            comparisons += 1
        if act_metrics[5:] != exp_metrics[5:]:
            raise AssertionError(
                f"case {cases} counts mismatch: {act_metrics[5:]} vs {exp_metrics[5:]}"
            )

        # Check all vector outputs against double-precision oracle.
        for name, act, exp in (
            ("adv", act_adv, exp_adv),
            ("tok_lp", act_tok_lp, exp_tok_lp),
            ("seq_lp", act_seq_lp, exp_seq_lp),
            ("grad_log_pi", act_grad_lp, exp_grad_lp),
            ("grad_logits", act_grad_logits, exp_grad_logits),
        ):
            if len(act) != len(exp):
                raise AssertionError(f"case {cases} {name} length mismatch")
            for idx, (va, ve) in enumerate(zip(act, exp)):
                err = abs(va - ve)
                if not math.isclose(va, ve, rel_tol=5e-6, abs_tol=1e-6):
                    raise AssertionError(
                        f"case {cases} {name}[{idx}]: native={va} oracle={ve} err={err:.3e}"
                    )
                max_abs = max(max_abs, err)
                comparisons += 1

        # Independent f64 central finite-difference verification of native grad_logits
        # using the exact f32 advantages supplied to et_grpo_loss_backward_v1.
        h = 1e-6
        logits_work = list(logits)
        for idx in range(len(logits_work)):
            orig = logits_work[idx]
            logits_work[idx] = orig + h
            mp, _, _ = grpo_loss_and_grad_ref(
                N, T, V, norm_mode, clip_low, clip_high, kl_beta,
                logits_work, targets, mask, log_old, log_ref, act_adv, rewards
            )
            logits_work[idx] = orig - h
            mm, _, _ = grpo_loss_and_grad_ref(
                N, T, V, norm_mode, clip_low, clip_high, kl_beta,
                logits_work, targets, mask, log_old, log_ref, act_adv, rewards
            )
            logits_work[idx] = orig
            fd = (mp[0] - mm[0]) / (2.0 * h)
            an = act_grad_logits[idx]
            err = abs(an - fd)
            scale = max(abs(an), abs(fd))
            max_fd_abs = max(max_fd_abs, err)
            if scale >= 1e-5:
                rel = err / scale
                max_fd_rel = max(max_fd_rel, rel)
                if rel > 1e-5:
                    raise AssertionError(
                        f"case {cases} f64 FD rel grad_logits[{idx}]: "
                        f"native={an:.10g} fd={fd:.10g} rel={rel:.3e}"
                    )
            if err > 1e-8 + 1e-5 * scale:
                raise AssertionError(
                    f"case {cases} f64 FD abs grad_logits[{idx}]: "
                    f"native={an:.10g} fd={fd:.10g} err={err:.3e}"
                )

        cases += 1

    if cases != 8:
        raise AssertionError(f"expected 8 oracle cases, saw {cases}")
    print(
        f"GRPO Python oracle + f64 FD: PASS ({cases} cases, {comparisons} values, "
        f"max_abs_err={max_abs:.3e}, max_f64_fd_abs_err={max_fd_abs:.3e}, "
        f"max_f64_fd_rel_err={max_fd_rel:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
