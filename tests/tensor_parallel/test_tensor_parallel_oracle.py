"""Independent binary64 Python dense reference + f64 FD oracle for 1D Tensor Parallelism."""
import math
import struct
import subprocess
import sys


def read_vec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def max_abs(a, b):
    assert len(a) == len(b), (len(a), len(b))
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def dot_loss(y, dy):
    return math.fsum(a * b for a, b in zip(y, dy))


def linear_fwd(M, K, N, x, w, b):
    y = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = b[j] if b is not None else 0.0
            for k in range(K):
                s += x[i * K + k] * w[j * K + k]
            y[i * N + j] = s
    return y


def linear_bwd(M, K, N, x, w, dy):
    dx = [0.0] * (M * K)
    dw = [0.0] * (N * K)
    db = [0.0] * N
    for i in range(M):
        for k in range(K):
            dx[i * K + k] = math.fsum(dy[i * N + j] * w[j * K + k] for j in range(N))
    for j in range(N):
        db[j] = math.fsum(dy[i * N + j] for i in range(M))
        for k in range(K):
            dw[j * K + k] = math.fsum(dy[i * N + j] * x[i * K + k] for i in range(M))
    return dx, dw, db


def sigmoid(g):
    if g >= 0.0:
        return 1.0 / (1.0 + math.exp(-g))
    eg = math.exp(g)
    return eg / (1.0 + eg)


def swiglu_fwd(T, D, F, x, wg, wu, wd):
    gate = linear_fwd(T, D, F, x, wg, None)
    up = linear_fwd(T, D, F, x, wu, None)
    act = [g * sigmoid(g) * u for g, u in zip(gate, up)]
    y = linear_fwd(T, F, D, act, wd, None)
    return y, gate, up, act


def swiglu_bwd(T, D, F, x, wg, wu, wd, gate, up, act, dy):
    dact, dwd, _ = linear_bwd(T, F, D, act, wd, dy)
    dgate = []
    dup = []
    for g, u, da in zip(gate, up, dact):
        sig = sigmoid(g)
        silu = g * sig
        dsilu = sig * (1.0 + g * (1.0 - sig))
        dgate.append(da * u * dsilu)
        dup.append(da * silu)
    dx_g, dwg, _ = linear_bwd(T, D, F, x, wg, dgate)
    dx_u, dwu, _ = linear_bwd(T, D, F, x, wu, dup)
    dx = [a + b for a, b in zip(dx_g, dx_u)]
    return dx, dwg, dwu, dwd


def gqa_fwd(B, T, D, Hq, Hkv, Dh, x, wq, wk, wv, wo):
    M = B * T
    kv_dim = Hkv * Dh
    ratio = Hq // Hkv
    scale = 1.0 / math.sqrt(float(Dh))
    q = linear_fwd(M, D, D, x, wq, None)
    k = linear_fwd(M, D, kv_dim, x, wk, None)
    v = linear_fwd(M, D, kv_dim, x, wv, None)
    attn = [0.0] * (M * D)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // ratio
            for i in range(T):
                q_off = (b * T + i) * D + hq * Dh
                scores = []
                for j in range(i + 1):
                    k_off = (b * T + j) * kv_dim + hkv * Dh
                    scores.append(
                        scale * math.fsum(q[q_off + d] * k[k_off + d] for d in range(Dh))
                    )
                ms = max(scores)
                exps = [math.exp(s - ms) for s in scores]
                denom = math.fsum(exps)
                probs = [e / denom for e in exps]
                for d in range(Dh):
                    attn[q_off + d] = math.fsum(
                        probs[j] * v[(b * T + j) * kv_dim + hkv * Dh + d]
                        for j in range(i + 1)
                    )
    y = linear_fwd(M, D, D, attn, wo, None)
    return y, q, k, v, attn


def gqa_bwd(B, T, D, Hq, Hkv, Dh, x, wq, wk, wv, wo, q, k, v, attn, dy):
    M = B * T
    kv_dim = Hkv * Dh
    ratio = Hq // Hkv
    scale = 1.0 / math.sqrt(float(Dh))
    dattn, dwo, _ = linear_bwd(M, D, D, attn, wo, dy)
    dq = [0.0] * (M * D)
    dk = [0.0] * (M * kv_dim)
    dv = [0.0] * (M * kv_dim)
    for b in range(B):
        for hq in range(Hq):
            hkv = hq // ratio
            for i in range(T):
                q_off = (b * T + i) * D + hq * Dh
                scores = []
                for j in range(i + 1):
                    k_off = (b * T + j) * kv_dim + hkv * Dh
                    scores.append(
                        scale * math.fsum(q[q_off + d] * k[k_off + d] for d in range(Dh))
                    )
                ms = max(scores)
                exps = [math.exp(s - ms) for s in scores]
                denom = math.fsum(exps)
                probs = [e / denom for e in exps]
                dp = []
                for j in range(i + 1):
                    v_off = (b * T + j) * kv_dim + hkv * Dh
                    dp.append(
                        math.fsum(dattn[q_off + d] * v[v_off + d] for d in range(Dh))
                    )
                    for d in range(Dh):
                        dv[v_off + d] += probs[j] * dattn[q_off + d]
                p_dot = math.fsum(probs[j] * dp[j] for j in range(i + 1))
                for j in range(i + 1):
                    ds = probs[j] * (dp[j] - p_dot) * scale
                    k_off = (b * T + j) * kv_dim + hkv * Dh
                    for d in range(Dh):
                        dq[q_off + d] += ds * k[k_off + d]
                        dk[k_off + d] += ds * q[q_off + d]
    dx_q, dwq, _ = linear_bwd(M, D, D, x, wq, dq)
    dx_k, dwk, _ = linear_bwd(M, D, kv_dim, x, wk, dk)
    dx_v, dwv, _ = linear_bwd(M, D, kv_dim, x, wv, dv)
    dx = [a + b + c for a, b, c in zip(dx_q, dx_k, dx_v)]
    return dx, dwq, dwk, dwv, dwo


def rmsnorm_fwd(M, D, eps, x, gamma):
    y = [0.0] * (M * D)
    for i in range(M):
        row = x[i * D : (i + 1) * D]
        inv_rms = 1.0 / math.sqrt(math.fsum(v * v for v in row) / float(D) + eps)
        for d in range(D):
            y[i * D + d] = row[d] * inv_rms * gamma[d]
    return y


def rmsnorm_bwd(M, D, eps, x, gamma, dy):
    dx = [0.0] * (M * D)
    dg = [0.0] * D
    for i in range(M):
        row = x[i * D : (i + 1) * D]
        dyr = dy[i * D : (i + 1) * D]
        mean_sq = math.fsum(v * v for v in row) / float(D) + eps
        inv_rms = 1.0 / math.sqrt(mean_sq)
        dot = math.fsum(dyr[d] * gamma[d] * row[d] for d in range(D))
        scale = dot / (float(D) * mean_sq)
        for d in range(D):
            dg[d] += dyr[d] * row[d] * inv_rms
            dx[i * D + d] = inv_rms * (dyr[d] * gamma[d] - row[d] * scale)
    return dx, dg


def block_fwd(B, T, D, F, Hq, Hkv, Dh, eps, x, w):
    M = B * T
    n1 = rmsnorm_fwd(M, D, eps, x, w[0])
    ao, q, k, v, attn = gqa_fwd(B, T, D, Hq, Hkv, Dh, n1, w[1], w[2], w[3], w[4])
    h1 = [a + b for a, b in zip(x, ao)]
    n2 = rmsnorm_fwd(M, D, eps, h1, w[5])
    mo, gate, up, act = swiglu_fwd(M, D, F, n2, w[6], w[7], w[8])
    y = [a + b for a, b in zip(h1, mo)]
    return y, (n1, q, k, v, attn, h1, n2, gate, up, act)


def block_bwd(B, T, D, F, Hq, Hkv, Dh, eps, x, w, cache, dy):
    M = B * T
    n1, q, k, v, attn, h1, n2, gate, up, act = cache
    dn2, dwg, dwu, dwd = swiglu_bwd(M, D, F, n2, w[6], w[7], w[8], gate, up, act, dy)
    dh1_m, dg2 = rmsnorm_bwd(M, D, eps, h1, w[5], dn2)
    dh1 = [a + b for a, b in zip(dy, dh1_m)]
    dn1, dwq, dwk, dwv, dwo = gqa_bwd(
        B, T, D, Hq, Hkv, Dh, n1, w[1], w[2], w[3], w[4], q, k, v, attn, dh1
    )
    dx_a, dg1 = rmsnorm_bwd(M, D, eps, x, w[0], dn1)
    dx = [a + b for a, b in zip(dh1, dx_a)]
    return dx, [dg1, dwq, dwk, dwv, dwo, dg2, dwg, dwu, dwd]


def f32_round(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def det_target(count):
    out = []
    mask = (1 << 64) - 1
    for idx in range(count):
        x = ((202 * 6364136223846793005) + ((idx + 1) * 1442695040888963407)) & mask
        x ^= x >> 33
        x = (x * 0xFF51AFD7ED558CCD) & mask
        x ^= x >> 33
        bucket = (x % 2001) - 1000
        out.append(f32_round((float(bucket) / 1000.0) * 0.55))
    return out


def fd_check_f64(eval_loss, arr, analytic, step_stride=1, h=1e-6):
    max_err = 0.0
    for idx in range(0, len(arr), step_stride):
        orig = arr[idx]
        arr[idx] = orig + h
        lp = eval_loss()
        arr[idx] = orig - h
        lm = eval_loss()
        arr[idx] = orig
        fd = (lp - lm) / (2.0 * h)
        err = abs(fd - analytic[idx])
        if err > max_err:
            max_err = err
    return max_err


def verify(binary):
    proc = subprocess.run([binary, "--oracle"], check=True, capture_output=True, text=True)
    lines = iter(proc.stdout.splitlines())
    max_ref_err = 0.0
    max_tp_parity_err = 0.0
    max_fd_err = 0.0

    # 1. ColumnParallelLinear & RowParallelLinear
    hdr = next(lines).split()
    assert hdr[0] == "LINEAR"
    M, K, N = map(int, hdr[1:4])
    lx, lw, lb, ldy = read_vec(lines), read_vec(lines), read_vec(lines), read_vec(lines)
    ref_y = linear_fwd(M, K, N, lx, lw, lb)
    ref_dx, ref_dw, ref_db = linear_bwd(M, K, N, lx, lw, ldy)
    max_fd_err = max(
        max_fd_err,
        fd_check_f64(lambda: dot_loss(linear_fwd(M, K, N, lx, lw, lb), ldy), lx, ref_dx, 3),
        fd_check_f64(lambda: dot_loss(linear_fwd(M, K, N, lx, lw, lb), ldy), lw, ref_dw, 7),
        fd_check_f64(lambda: dot_loss(linear_fwd(M, K, N, lx, lw, lb), ldy), lb, ref_db, 1),
    )
    col_w1 = row_w1 = None
    for _ in range(3):
        ch = next(lines).split()
        assert ch[0] == "COL_W"
        w_sz = int(ch[1])
        cy, cdx, cdw, cdb = read_vec(lines), read_vec(lines), read_vec(lines), read_vec(lines)
        for act, ref in ((cy, ref_y), (cdx, ref_dx), (cdw, ref_dw), (cdb, ref_db)):
            e = max_abs(act, ref)
            max_ref_err = max(max_ref_err, e)
            assert e <= 1e-6, f"COL W={w_sz} ref err {e}"
        if w_sz == 1:
            col_w1 = (cy, cdx, cdw, cdb)
        else:
            for a, b in zip((cy, cdx, cdw, cdb), col_w1):
                e = max_abs(a, b)
                max_tp_parity_err = max(max_tp_parity_err, e)
                assert e <= 1e-6, f"COL W={w_sz} vs W=1 err {e}"

        rh = next(lines).split()
        assert rh[0] == "ROW_W"
        ry, rdx, rdw, rdb = read_vec(lines), read_vec(lines), read_vec(lines), read_vec(lines)
        for act, ref in ((ry, ref_y), (rdx, ref_dx), (rdw, ref_dw), (rdb, ref_db)):
            e = max_abs(act, ref)
            max_ref_err = max(max_ref_err, e)
            assert e <= 1e-6, f"ROW W={w_sz} ref err {e}"
        if w_sz == 1:
            row_w1 = (ry, rdx, rdw, rdb)
        else:
            for a, b in zip((ry, rdx, rdw, rdb), row_w1):
                e = max_abs(a, b)
                max_tp_parity_err = max(max_tp_parity_err, e)
                assert e <= 1e-6, f"ROW W={w_sz} vs W=1 err {e}"

    # 2. TP-SwiGLU
    hdr = next(lines).split()
    assert hdr[0] == "SWIGLU"
    T, D, F = map(int, hdr[1:4])
    sx, swg, swu, swd, sdy = (read_vec(lines) for _ in range(5))
    sy_ref, sg, su, sa = swiglu_fwd(T, D, F, sx, swg, swu, swd)
    sdx_ref, sdwg_ref, sdwu_ref, sdwd_ref = swiglu_bwd(
        T, D, F, sx, swg, swu, swd, sg, su, sa, sdy
    )
    max_fd_err = max(
        max_fd_err,
        fd_check_f64(lambda: dot_loss(swiglu_fwd(T, D, F, sx, swg, swu, swd)[0], sdy), sx, sdx_ref, 5),
        fd_check_f64(lambda: dot_loss(swiglu_fwd(T, D, F, sx, swg, swu, swd)[0], sdy), swg, sdwg_ref, 17),
        fd_check_f64(lambda: dot_loss(swiglu_fwd(T, D, F, sx, swg, swu, swd)[0], sdy), swu, sdwu_ref, 17),
        fd_check_f64(lambda: dot_loss(swiglu_fwd(T, D, F, sx, swg, swu, swd)[0], sdy), swd, sdwd_ref, 17),
    )
    sw_w1 = None
    for _ in range(3):
        sh = next(lines).split()
        assert sh[0] == "SWIGLU_W"
        w_sz = int(sh[1])
        vals = tuple(read_vec(lines) for _ in range(5))
        refs = (sy_ref, sdx_ref, sdwg_ref, sdwu_ref, sdwd_ref)
        for act, ref in zip(vals, refs):
            e = max_abs(act, ref)
            max_ref_err = max(max_ref_err, e)
            assert e <= 1e-6, f"SWIGLU W={w_sz} ref err {e}"
        if w_sz == 1:
            sw_w1 = vals
        else:
            for a, b in zip(vals, sw_w1):
                e = max_abs(a, b)
                max_tp_parity_err = max(max_tp_parity_err, e)
                assert e <= 1e-6, f"SWIGLU W={w_sz} vs W=1 err {e}"

    # 3. TP-GQA
    hdr = next(lines).split()
    assert hdr[0] == "GQA"
    B, T, D, Hq, Hkv, Dh = map(int, hdr[1:7])
    gx, gwq, gwk, gwv, gwo, gdy = (read_vec(lines) for _ in range(6))
    gy_ref, gq, gk, gv, ga = gqa_fwd(B, T, D, Hq, Hkv, Dh, gx, gwq, gwk, gwv, gwo)
    gdx_ref, gdwq_ref, gdwk_ref, gdwv_ref, gdwo_ref = gqa_bwd(
        B, T, D, Hq, Hkv, Dh, gx, gwq, gwk, gwv, gwo, gq, gk, gv, ga, gdy
    )
    max_fd_err = max(
        max_fd_err,
        fd_check_f64(lambda: dot_loss(gqa_fwd(B, T, D, Hq, Hkv, Dh, gx, gwq, gwk, gwv, gwo)[0], gdy), gx, gdx_ref, 11),
        fd_check_f64(lambda: dot_loss(gqa_fwd(B, T, D, Hq, Hkv, Dh, gx, gwq, gwk, gwv, gwo)[0], gdy), gwq, gdwq_ref, 29),
        fd_check_f64(lambda: dot_loss(gqa_fwd(B, T, D, Hq, Hkv, Dh, gx, gwq, gwk, gwv, gwo)[0], gdy), gwk, gdwk_ref, 19),
        fd_check_f64(lambda: dot_loss(gqa_fwd(B, T, D, Hq, Hkv, Dh, gx, gwq, gwk, gwv, gwo)[0], gdy), gwv, gdwv_ref, 19),
        fd_check_f64(lambda: dot_loss(gqa_fwd(B, T, D, Hq, Hkv, Dh, gx, gwq, gwk, gwv, gwo)[0], gdy), gwo, gdwo_ref, 29),
    )
    gqa_w1 = None
    for _ in range(3):
        gh = next(lines).split()
        assert gh[0] == "GQA_W"
        w_sz = int(gh[1])
        vals = tuple(read_vec(lines) for _ in range(6))
        refs = (gy_ref, gdx_ref, gdwq_ref, gdwk_ref, gdwv_ref, gdwo_ref)
        for act, ref in zip(vals, refs):
            e = max_abs(act, ref)
            max_ref_err = max(max_ref_err, e)
            assert e <= 1e-6, f"GQA W={w_sz} ref err {e}"
        if w_sz == 1:
            gqa_w1 = vals
        else:
            for a, b in zip(vals, gqa_w1):
                e = max_abs(a, b)
                max_tp_parity_err = max(max_tp_parity_err, e)
                assert e <= 1e-6, f"GQA W={w_sz} vs W=1 err {e}"

    # 4. Full TP Transformer Block
    hdr = next(lines).split()
    assert hdr[0] == "BLOCK"
    B, T, D, F, Hq, Hkv, Dh = map(int, hdr[1:8])
    eps = float.fromhex(hdr[8])
    bx = read_vec(lines)
    bw = [read_vec(lines) for _ in range(9)]
    target = det_target(B * T * D)
    by_ref, bcache = block_fwd(B, T, D, F, Hq, Hkv, Dh, eps, bx, bw)
    inv_n = 1.0 / float(B * T * D)
    bloss_ref = 0.5 * inv_n * math.fsum((a - b) ** 2 for a, b in zip(by_ref, target))
    bdy = [(a - b) * inv_n for a, b in zip(by_ref, target)]
    bdx_ref, bg_ref = block_bwd(B, T, D, F, Hq, Hkv, Dh, eps, bx, bw, bcache, bdy)

    def block_mse():
        y, _ = block_fwd(B, T, D, F, Hq, Hkv, Dh, eps, bx, bw)
        return 0.5 * inv_n * math.fsum((a - b) ** 2 for a, b in zip(y, target))

    max_fd_err = max(max_fd_err, fd_check_f64(block_mse, bx, bdx_ref, 13))
    for p in range(9):
        stride = max(1, len(bw[p]) // 8)
        max_fd_err = max(max_fd_err, fd_check_f64(block_mse, bw[p], bg_ref[p], stride))

    blk_w1 = None
    for _ in range(3):
        bh = next(lines).split()
        assert bh[0] == "BLOCK_W"
        w_sz = int(bh[1])
        act_loss = float.fromhex(bh[2])
        assert abs(act_loss - bloss_ref) <= 1e-6, f"BLOCK W={w_sz} loss err"
        by = read_vec(lines)
        bdx = read_vec(lines)
        bgs = [read_vec(lines) for _ in range(9)]
        vals = (by, bdx, *bgs)
        refs = (by_ref, bdx_ref, *bg_ref)
        for act, ref in zip(vals, refs):
            e = max_abs(act, ref)
            max_ref_err = max(max_ref_err, e)
            assert e <= 1e-6, f"BLOCK W={w_sz} ref err {e}"
        if w_sz == 1:
            blk_w1 = (act_loss, vals)
        else:
            max_tp_parity_err = max(max_tp_parity_err, abs(act_loss - blk_w1[0]))
            for a, b in zip(vals, blk_w1[1]):
                e = max_abs(a, b)
                max_tp_parity_err = max(max_tp_parity_err, e)
                assert e <= 1e-6, f"BLOCK W={w_sz} vs W=1 err {e}"

    assert max_fd_err <= 1e-7, f"f64 FD error too large: {max_fd_err}"
    print(
        f"TP Python binary64 oracle + f64 FD check: PASS "
        f"(max_ref_err={max_ref_err:.3e} max_w24_vs_w1_err={max_tp_parity_err:.3e} "
        f"max_f64_fd_err={max_fd_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
