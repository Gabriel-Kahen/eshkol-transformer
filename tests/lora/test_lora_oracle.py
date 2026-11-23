"""Independent double-precision Python oracle for the native LoRA engine."""
import math
import subprocess
import sys


def read_vec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def lora_reference(M, K, N, R, alpha, x, w, a, b, dy):
    scale = alpha / float(R)

    # H = X @ A^T  [M, R]
    h = [0.0] * (M * R)
    for m in range(M):
        for i in range(R):
            h[m * R + i] = math.fsum(
                x[m * K + k] * a[i * K + k] for k in range(K)
            )

    # Y = X @ W^T + scale * (H @ B^T)  [M, N]
    y = [0.0] * (M * N)
    for m in range(M):
        for n in range(N):
            base = math.fsum(x[m * K + k] * w[n * K + k] for k in range(K))
            lora = math.fsum(h[m * R + i] * b[n * R + i] for i in range(R))
            y[m * N + n] = base + scale * lora

    # dB = scale * dY^T @ H  [N, R]
    db = [0.0] * (N * R)
    for n in range(N):
        for i in range(R):
            db[n * R + i] = scale * math.fsum(
                dy[m * N + n] * h[m * R + i] for m in range(M)
            )

    # dH = scale * dY @ B  [M, R]
    dh = [0.0] * (M * R)
    for m in range(M):
        for i in range(R):
            dh[m * R + i] = scale * math.fsum(
                dy[m * N + n] * b[n * R + i] for n in range(N)
            )

    # dA = dH^T @ X  [R, K]
    da = [0.0] * (R * K)
    for i in range(R):
        for k in range(K):
            da[i * K + k] = math.fsum(
                dh[m * R + i] * x[m * K + k] for m in range(M)
            )

    # dX = dY @ W + dH @ A  [M, K]
    dx = [0.0] * (M * K)
    for m in range(M):
        for k in range(K):
            base_g = math.fsum(dy[m * N + n] * w[n * K + k] for n in range(N))
            lora_g = math.fsum(dh[m * R + i] * a[i * K + k] for i in range(R))
            dx[m * K + k] = base_g + lora_g

    # W_merged = W + scale * (B @ A), W_restored = W_merged - scale * (B @ A)
    w_merged = [0.0] * (N * K)
    w_restored = [0.0] * (N * K)
    for n in range(N):
        for k in range(K):
            delta = scale * math.fsum(
                b[n * R + i] * a[i * K + k] for i in range(R)
            )
            w_merged[n * K + k] = w[n * K + k] + delta
            w_restored[n * K + k] = w_merged[n * K + k] - delta

    return y, dx, da, db, w_merged, w_restored


def verify(binary):
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    names = ("Y", "dX", "dA", "dB", "W_merged", "W_restored")
    cases = 0
    comparisons = 0
    max_abs = 0.0

    for header in lines:
        parts = header.split()
        if len(parts) != 6 or parts[0] != "CASE":
            raise AssertionError(f"unexpected oracle header: {header!r}")
        M, K, N, R = (int(parts[i]) for i in range(1, 5))
        alpha = float.fromhex(parts[5])
        x = read_vec(lines)
        w = read_vec(lines)
        a = read_vec(lines)
        b = read_vec(lines)
        dy = read_vec(lines)
        actuals = [read_vec(lines) for _ in range(6)]
        expected = lora_reference(M, K, N, R, alpha, x, w, a, b, dy)

        for name, act, exp in zip(names, actuals, expected):
            if len(act) != len(exp):
                raise AssertionError(f"{name} length mismatch in case {cases}")
            for idx, (va, ve) in enumerate(zip(act, exp)):
                err = abs(va - ve)
                if not math.isclose(va, ve, rel_tol=5e-6, abs_tol=1e-6):
                    raise AssertionError(
                        f"case {cases} ({M},{K},{N},{R}) {name}[{idx}]: "
                        f"native={va:.10g} oracle={ve:.10g} err={err:.3e}"
                    )
                if err > max_abs:
                    max_abs = err
                comparisons += 1
        cases += 1

    if cases != 8:
        raise AssertionError(f"expected 8 oracle cases, saw {cases}")
    print(
        f"LORA Python oracle: PASS ({cases} cases, {comparisons} values, "
        f"max_abs_err={max_abs:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
