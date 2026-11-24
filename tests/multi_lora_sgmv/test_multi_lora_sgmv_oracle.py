"""Independent binary64 Python oracle for Multi-LoRA SGMV forward, backward, and AdamW."""
import math
import struct
import subprocess
import sys


def f32(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", val))[0]


def read_vec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def sgmv_forward_ref(M, in_dim, out_dim, w_base, bias, x, row_ids, adapters):
    y = [0.0] * (M * out_dim)
    for i in range(M):
        aid = row_ids[i]
        ad = None if aid == -1 else adapters[aid]
        h = None
        if ad is not None:
            r = ad["rank"]
            a_w = ad["a"]
            h = [0.0] * r
            for ri in range(r):
                acc = 0.0
                for k in range(in_dim):
                    acc += a_w[ri * in_dim + k] * x[i * in_dim + k]
                h[ri] = acc
        for o in range(out_dim):
            base_acc = 0.0
            for k in range(in_dim):
                base_acc += w_base[o * in_dim + k] * x[i * in_dim + k]
            if bias is not None:
                base_acc += bias[o]
            total = base_acc
            if ad is not None:
                r = ad["rank"]
                b_w = ad["b"]
                scale = ad["alpha"] / float(r)
                lora_acc = 0.0
                for ri in range(r):
                    lora_acc += b_w[o * r + ri] * h[ri]
                total = base_acc + scale * lora_acc
            y[i * out_dim + o] = f32(total)
    return y


def sgmv_backward_ref(M, in_dim, out_dim, w_base, x, row_ids, dy, adapters):
    dx = [0.0] * (M * in_dim)
    da_map = {aid: [0.0] * (ad["rank"] * in_dim) for aid, ad in adapters.items()}
    db_map = {aid: [0.0] * (out_dim * ad["rank"]) for aid, ad in adapters.items()}

    # Base backward per row.
    dx_acc = [0.0] * (M * in_dim)
    for i in range(M):
        for k in range(in_dim):
            acc = 0.0
            for o in range(out_dim):
                acc += dy[i * out_dim + o] * w_base[o * in_dim + k]
            dx_acc[i * in_dim + k] = acc
            if row_ids[i] == -1:
                dx[i * in_dim + k] = f32(acc)

    # Segmented per-adapter backward (preserving ascending row index order).
    for aid, ad in adapters.items():
        rows = [i for i in range(M) if row_ids[i] == aid]
        if not rows:
            continue
        r = ad["rank"]
        scale = ad["alpha"] / float(r)
        a_w = ad["a"]
        b_w = ad["b"]
        h_rows = []
        dh_rows = []
        for i in rows:
            hr = [0.0] * r
            dhr = [0.0] * r
            for ri in range(r):
                h_acc = 0.0
                for k in range(in_dim):
                    h_acc += a_w[ri * in_dim + k] * x[i * in_dim + k]
                dh_acc = 0.0
                for o in range(out_dim):
                    dh_acc += dy[i * out_dim + o] * b_w[o * r + ri]
                hr[ri] = h_acc
                dhr[ri] = scale * dh_acc
            h_rows.append(hr)
            dh_rows.append(dhr)

        for o in range(out_dim):
            for ri in range(r):
                acc = 0.0
                for p, i in enumerate(rows):
                    acc += dy[i * out_dim + o] * h_rows[p][ri]
                db_map[aid][o * r + ri] = f32(scale * acc)

        for ri in range(r):
            for k in range(in_dim):
                acc = 0.0
                for p, i in enumerate(rows):
                    acc += dh_rows[p][ri] * x[i * in_dim + k]
                da_map[aid][ri * in_dim + k] = f32(acc)

        for p, i in enumerate(rows):
            for k in range(in_dim):
                lg = 0.0
                for ri in range(r):
                    lg += dh_rows[p][ri] * a_w[ri * in_dim + k]
                dx[i * in_dim + k] = f32(dx_acc[i * in_dim + k] + lg)

    return dx, da_map, db_map


def pow_u32(base: float, exp: int) -> float:
    acc = 1.0
    for _ in range(exp):
        acc *= base
    return acc


def adamw_step_ref(ad, da, db, lr, beta1, beta2, eps, wd):
    ad["step"] += 1
    t = ad["step"]
    bc1 = 1.0 - pow_u32(beta1, t)
    bc2 = 1.0 - pow_u32(beta2, t)
    for key_p, key_m, key_v, g in (
        ("a", "m_a", "v_a", da),
        ("b", "m_b", "v_b", db),
    ):
        p_arr = ad[key_p]
        m_arr = ad[key_m]
        v_arr = ad[key_v]
        for idx in range(len(p_arr)):
            p = p_arr[idx]
            gi = g[idx]
            m = beta1 * m_arr[idx] + (1.0 - beta1) * gi
            v = beta2 * v_arr[idx] + (1.0 - beta2) * gi * gi
            m_hat = m / bc1
            v_hat = v / bc2
            p_new = p - lr * wd * p - lr * (m_hat / (math.sqrt(v_hat) + eps))
            p_arr[idx] = f32(p_new)
            m_arr[idx] = f32(m)
            v_arr[idx] = f32(v)


def verify(binary: str) -> None:
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    sgmv_cases = 0
    traj_seen = 0
    comparisons = 0
    max_sgmv_err = 0.0
    max_traj_err = 0.0

    for header in lines:
        parts = header.split()
        if parts[0] == "SGMV_CASE":
            M, in_dim, out_dim, has_bias, num_adapters = (
                int(parts[i]) for i in range(1, 6)
            )
            w_base = read_vec(lines)
            bias = read_vec(lines) if has_bias else None
            x = read_vec(lines)
            dy = read_vec(lines)
            row_ids = [int(tok) for tok in next(lines).split()]
            adapters = {}
            adapter_order = []
            for _ in range(num_adapters):
                ah = next(lines).split()
                assert ah[0] == "ADAPTER"
                aid = int(ah[1])
                rank = int(ah[2])
                alpha = float.fromhex(ah[3])
                a_w = read_vec(lines)
                b_w = read_vec(lines)
                adapters[aid] = {"rank": rank, "alpha": alpha, "a": a_w, "b": b_w}
                adapter_order.append(aid)

            act_y = read_vec(lines)
            act_dx = read_vec(lines)
            exp_y = sgmv_forward_ref(
                M, in_dim, out_dim, w_base, bias, x, row_ids, adapters
            )
            exp_dx, exp_da, exp_db = sgmv_backward_ref(
                M, in_dim, out_dim, w_base, x, row_ids, dy, adapters
            )

            for name, act, exp in (("Y", act_y, exp_y), ("dX", act_dx, exp_dx)):
                assert len(act) == len(exp), f"{name} len mismatch"
                for idx, (va, ve) in enumerate(zip(act, exp)):
                    err = abs(va - ve)
                    if err > 1e-7:
                        raise AssertionError(
                            f"SGMV case {sgmv_cases} {name}[{idx}]: "
                            f"native={va!r} oracle={ve!r} err={err:.3e}"
                        )
                    max_sgmv_err = max(max_sgmv_err, err)
                    comparisons += 1

            for aid in adapter_order:
                act_da = read_vec(lines)
                act_db = read_vec(lines)
                for name, act, exp in (
                    (f"dA[{aid}]", act_da, exp_da[aid]),
                    (f"dB[{aid}]", act_db, exp_db[aid]),
                ):
                    assert len(act) == len(exp)
                    for idx, (va, ve) in enumerate(zip(act, exp)):
                        err = abs(va - ve)
                        if err > 1e-7:
                            raise AssertionError(
                                f"SGMV case {sgmv_cases} {name}[{idx}]: "
                                f"native={va!r} oracle={ve!r} err={err:.3e}"
                            )
                        max_sgmv_err = max(max_sgmv_err, err)
                        comparisons += 1
            sgmv_cases += 1

        elif parts[0] == "ADAMW_TRAJ":
            M, in_dim, out_dim, steps = (int(parts[i]) for i in range(1, 5))
            lr, beta1, beta2, eps, wd = (
                float.fromhex(parts[i]) for i in range(5, 10)
            )
            w_base = read_vec(lines)
            bias = read_vec(lines)
            x = read_vec(lines)
            target = read_vec(lines)
            row_ids = [int(tok) for tok in next(lines).split()]
            adapters = {}
            ids = []
            for _ in range(3):
                ah = next(lines).split()
                assert ah[0] == "TRAJ_ADAPTER"
                aid = int(ah[1])
                rank = int(ah[2])
                alpha = float.fromhex(ah[3])
                a_w = read_vec(lines)
                b_w = read_vec(lines)
                adapters[aid] = {
                    "rank": rank,
                    "alpha": alpha,
                    "a": a_w,
                    "b": b_w,
                    "m_a": [0.0] * len(a_w),
                    "v_a": [0.0] * len(a_w),
                    "m_b": [0.0] * len(b_w),
                    "v_b": [0.0] * len(b_w),
                    "step": 0,
                }
                ids.append(aid)

            for step in range(steps + 1):
                st_line = next(lines).split()
                assert st_line[0] == "STEP" and int(st_line[1]) == step
                act_losses = [float.fromhex(st_line[2 + t]) for t in range(3)]
                y = sgmv_forward_ref(
                    M, in_dim, out_dim, w_base, bias, x, row_ids, adapters
                )
                exp_losses = [0.0, 0.0, 0.0]
                dy = [0.0] * (M * out_dim)
                for i in range(M):
                    if row_ids[i] not in ids:
                        continue
                    task = ids.index(row_ids[i])
                    inv = 1.0 / float(3 * out_dim)
                    for o in range(out_dim):
                        idx = i * out_dim + o
                        diff = y[idx] - target[idx]
                        exp_losses[task] += 0.5 * inv * diff * diff
                        dy[idx] = f32(inv * diff)
                for t in range(3):
                    err = abs(act_losses[t] - exp_losses[t])
                    if err > 1e-7:
                        raise AssertionError(
                            f"AdamW step {step} task {t}: "
                            f"native={act_losses[t]:.12g} oracle={exp_losses[t]:.12g} "
                            f"err={err:.3e}"
                        )
                    max_traj_err = max(max_traj_err, err)
                    comparisons += 1
                if step < steps:
                    _, da_map, db_map = sgmv_backward_ref(
                        M, in_dim, out_dim, w_base, x, row_ids, dy, adapters
                    )
                    for aid in ids:
                        adamw_step_ref(
                            adapters[aid],
                            da_map[aid],
                            db_map[aid],
                            lr,
                            beta1,
                            beta2,
                            eps,
                            wd,
                        )
            act_final_y = read_vec(lines)
            for idx, (va, ve) in enumerate(zip(act_final_y, y)):
                err = abs(va - ve)
                if err > 1e-7:
                    raise AssertionError(
                        f"AdamW final Y[{idx}]: native={va!r} oracle={ve!r} err={err:.3e}"
                    )
                max_traj_err = max(max_traj_err, err)
                comparisons += 1
            traj_seen += 1
        else:
            raise AssertionError(f"unexpected header: {header!r}")

    assert sgmv_cases == 3 and traj_seen == 1
    print(
        f"MULTI-LORA SGMV Python oracle: PASS ({sgmv_cases} SGMV cases + "
        f"12-step AdamW trajectory, {comparisons} values, "
        f"sgmv_max_err={max_sgmv_err:.3e}, adamw_max_err={max_traj_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
