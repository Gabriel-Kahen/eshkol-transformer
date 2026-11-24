"""Independent binary64 dense attention & FD gradient oracle for Ring Attention."""
from __future__ import annotations

import math
import subprocess
import sys


def dense_attention_reference(
    n: int,
    t: int,
    h: int,
    hkv: int,
    dh: int,
    causal: int,
    q: list[float],
    k: list[float],
    v: list[float],
    do: list[float],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    scale = 1.0 / math.sqrt(float(dh))
    group = h // hkv

    o = [0.0] * (n * t * h * dh)
    lse = [0.0] * (n * t * h)
    dq = [0.0] * (n * t * h * dh)
    dk = [0.0] * (n * t * hkv * dh)
    dv = [0.0] * (n * t * hkv * dh)

    for b in range(n):
        for tq in range(t):
            j_end = (tq + 1) if causal else t
            for head in range(h):
                kv_head = head // group
                q_idx = (b * t + tq) * h + head
                q_off = q_idx * dh

                scores = []
                for tk in range(j_end):
                    k_off = ((b * t + tk) * hkv + kv_head) * dh
                    dot = 0.0
                    for d in range(dh):
                        dot += q[q_off + d] * k[k_off + d]
                    scores.append(dot * scale)

                m = max(scores)
                sum_exp = sum(math.exp(s - m) for s in scores)
                lse_val = m + math.log(sum_exp)
                lse[q_idx] = lse_val

                probs = [math.exp(s - lse_val) for s in scores]
                for tk in range(j_end):
                    v_off = ((b * t + tk) * hkv + kv_head) * dh
                    p = probs[tk]
                    for d in range(dh):
                        o[q_off + d] += p * v[v_off + d]

                # Exact backward directly from probs (independent of O_f32 shortcut)
                dp_list = []
                for tk in range(j_end):
                    v_off = ((b * t + tk) * hkv + kv_head) * dh
                    dp = 0.0
                    for d in range(dh):
                        dp += do[q_off + d] * v[v_off + d]
                    dp_list.append(dp)

                d_i = sum(probs[tk] * dp_list[tk] for tk in range(j_end))
                for tk in range(j_end):
                    kv_off = ((b * t + tk) * hkv + kv_head) * dh
                    p = probs[tk]
                    ds = p * (dp_list[tk] - d_i)
                    ddot = ds * scale
                    for d in range(dh):
                        dq[q_off + d] += ddot * k[kv_off + d]
                        dk[kv_off + d] += ddot * q[q_off + d]
                        dv[kv_off + d] += p * do[q_off + d]

    return o, lse, dq, dk, dv


def check_python_fd(
    n: int,
    t: int,
    h: int,
    hkv: int,
    dh: int,
    causal: int,
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
            n, t, h, hkv, dh, causal, q_mut, k_mut, v_mut, do
        )
        return sum(a * b for a, b in zip(out, do))

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
        if len(parts) != 10 or parts[0] != "CASE":
            raise AssertionError(f"malformed oracle header: {header!r}")
        n, t, h, hkv, dh, w, causal, fskip, bskip = (int(parts[i]) for i in range(1, 10))
        expected_skip = (w * (w - 1) // 2) if causal else 0
        if fskip != expected_skip or bskip != expected_skip:
            raise AssertionError(
                f"causal skip mismatch W={w} causal={causal}: got ({fskip}, {bskip}), want {expected_skip}"
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
            n, t, h, hkv, dh, causal, q, k, v, do
        )

        # Cross-verify our Python reference itself via binary64 central FD on first 4 cases
        if cases < 4:
            fd_err = check_python_fd(
                n, t, h, hkv, dh, causal, q, k, v, do, want_dq, want_dk, want_dv
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
            for idx, (g, wv) in enumerate(zip(got, want)):
                diff = abs(g - wv)
                if diff > 1e-6:
                    raise AssertionError(
                        f"Ring Attention {name} case={cases} (N={n},T={t},H={h},Hkv={hkv},"
                        f"Dh={dh},W={w},causal={causal}) idx={idx}: "
                        f"got={g:.10g}, want={wv:.10g}, diff={diff:.3e}"
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
        f"RING_ATTN Python binary64 oracle: PASS (cases={cases}, values={values}, "
        f"max_err O={max_err_o:.3e} LSE={max_err_lse:.3e} "
        f"dQ={max_err_dq:.3e} dK={max_err_dk:.3e} dV={max_err_dv:.3e}, "
        f"py_fd_err={max_fd_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
