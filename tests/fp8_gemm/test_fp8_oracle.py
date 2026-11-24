"""Independent exact-rational + binary64 Python oracle for OCP FP8 & Block GEMM."""
from fractions import Fraction
import math
import struct
import subprocess
import sys


def f32_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def float_to_f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def f32_bits_to_fraction(bits: int):
    sign = -1 if (bits >> 31) else 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0xFF:
        return None  # Inf or NaN
    if exp == 0:
        if mant == 0:
            return Fraction(0)
        return sign * Fraction(mant, 1 << (126 + 23))
    val = Fraction((1 << 23) | mant, 1 << 23)
    shift = exp - 127
    if shift >= 0:
        return sign * val * (1 << shift)
    return sign * val / (1 << (-shift))


def ref_e4m3_decode(code: int):
    sign = (code >> 7) & 1
    exp = (code >> 3) & 0xF
    mant = code & 0x7
    if exp == 15 and mant == 7:
        return None  # NaN
    if exp == 0:
        if mant == 0:
            return (0x80000000 if sign else 0x00000000), Fraction(0)
        val = Fraction(mant, 1 << 9)
    else:
        e = exp - 7
        base = Fraction(8 + mant, 8)
        val = base * (1 << e) if e >= 0 else base / (1 << (-e))
    if sign:
        val = -val
    return float_to_f32_bits(float(val)), val


def ref_e5m2_decode(code: int):
    sign = (code >> 7) & 1
    exp = (code >> 2) & 0x1F
    mant = code & 0x3
    if exp == 31:
        return None  # Inf or NaN
    if exp == 0:
        if mant == 0:
            return (0x80000000 if sign else 0x00000000), Fraction(0)
        val = Fraction(mant, 1 << 16)
    else:
        e = exp - 15
        base = Fraction(4 + mant, 4)
        val = base * (1 << e) if e >= 0 else base / (1 << (-e))
    if sign:
        val = -val
    return float_to_f32_bits(float(val)), val


def build_positive_grid(fmt: str):
    """Build sorted positive representable grid + one extra overflow slot."""
    grid = []
    if fmt == "e4m3":
        for c in range(0x00, 0x7F):
            _, frac = ref_e4m3_decode(c)
            grid.append((c, frac))
        # Next unbounded grid point after 448.0 (0x7E) is 480.0 (0x7F, odd).
        grid.append((0x7F, Fraction(480)))
        max_code = 0x7E
    else:
        for c in range(0x00, 0x7C):
            _, frac = ref_e5m2_decode(c)
            grid.append((c, frac))
        # Next unbounded grid point after 57344.0 (0x7B) is 65536.0 (0x7C, even).
        grid.append((0x7C, Fraction(65536)))
        max_code = 0x7B
    return grid, max_code


def ref_fp8_encode(bits: int, fmt: str, reject_overflow: bool):
    """Pure exact-Fraction IEEE-754 RNE reference encoder."""
    frac = f32_bits_to_fraction(bits)
    if frac is None:
        return -2, 0  # ET_FP8_ERR_NONFINITE
    sign_bit = 0x80 if (bits >> 31) else 0x00
    a = abs(frac)
    if a == 0:
        return 0, sign_bit

    grid, max_code = build_positive_grid(fmt)
    overflow_slot = grid[-1][0]
    overflow_val = grid[-1][1]

    if a >= overflow_val:
        return (-3, 0) if reject_overflow else (0, sign_bit | max_code)

    # Locate bracketing interval [lo, hi] in grid.
    lo_idx = 0
    hi_idx = len(grid) - 1
    while lo_idx + 1 < hi_idx:
        mid = (lo_idx + hi_idx) // 2
        if grid[mid][1] <= a:
            lo_idx = mid
        else:
            hi_idx = mid

    c_lo, v_lo = grid[lo_idx]
    c_hi, v_hi = grid[hi_idx]
    d_lo = a - v_lo
    d_hi = v_hi - a
    if d_lo < d_hi:
        chosen = c_lo
    elif d_hi < d_lo:
        chosen = c_hi
    else:
        # Exact tie: round to even LSB of code.
        chosen = c_lo if (c_lo & 1) == 0 else c_hi

    if chosen == overflow_slot:
        return (-3, 0) if reject_overflow else (0, sign_bit | max_code)
    return 0, sign_bit | chosen


def read_f32_line(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def read_u8_line(lines):
    return [int(tok, 16) for tok in next(lines).split()]


def verify(binary: str) -> None:
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())

    # 1. Exhaustive 256-codepoint E4M3FN
    assert next(lines) == "SECTION E4M3_ALL256"
    for c in range(256):
        parts = next(lines).split()
        code, dst, fbits, rst, re = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2], 16),
            int(parts[3]),
            int(parts[4]),
        )
        assert code == c
        ref = ref_e4m3_decode(c)
        if ref is None:
            assert dst == -2, f"E4M3 code 0x{c:02x} expected -2, got {dst}"
        else:
            exp_bits, _ = ref
            assert dst == 0 and fbits == exp_bits, (
                f"E4M3 decode 0x{c:02x}: got {fbits:#010x}, exp {exp_bits:#010x}"
            )
            assert rst == 0 and re == c, f"E4M3 roundtrip 0x{c:02x}: got {re:#04x}"

    # 2. Exhaustive 256-codepoint E5M2
    assert next(lines) == "SECTION E5M2_ALL256"
    for c in range(256):
        parts = next(lines).split()
        code, dst, fbits, rst, re = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2], 16),
            int(parts[3]),
            int(parts[4]),
        )
        assert code == c
        ref = ref_e5m2_decode(c)
        if ref is None:
            assert dst == -2, f"E5M2 code 0x{c:02x} expected -2, got {dst}"
        else:
            exp_bits, _ = ref
            assert dst == 0 and fbits == exp_bits, (
                f"E5M2 decode 0x{c:02x}: got {fbits:#010x}, exp {exp_bits:#010x}"
            )
            assert rst == 0 and re == c, f"E5M2 roundtrip 0x{c:02x}: got {re:#04x}"

    # 3 & 4. RNE boundary checks for E4M3FN and E5M2
    rne_checked = 0
    for fmt, tag in (("e4m3", "SECTION E4M3_RNE"), ("e5m2", "SECTION E5M2_RNE")):
        hdr = next(lines).split()
        assert f"{hdr[0]} {hdr[1]}" == tag
        n_cases = int(hdr[2])
        for _ in range(n_cases):
            parts = next(lines).split()
            bits = int(parts[0], 16)
            s_sat, q_sat, s_rej, q_rej = (int(parts[i]) for i in range(1, 5))
            es_sat, eq_sat = ref_fp8_encode(bits, fmt, reject_overflow=False)
            es_rej, eq_rej = ref_fp8_encode(bits, fmt, reject_overflow=True)
            assert s_sat == es_sat and (s_sat != 0 or q_sat == eq_sat), (
                f"{fmt} SAT {bits:#010x}: got ({s_sat}, 0x{q_sat:02x}), "
                f"exp ({es_sat}, 0x{eq_sat:02x})"
            )
            assert s_rej == es_rej and (s_rej != 0 or q_rej == eq_rej), (
                f"{fmt} REJ {bits:#010x}: got ({s_rej}, 0x{q_rej:02x}), "
                f"exp ({es_rej}, 0x{eq_rej:02x})"
            )
            rne_checked += 1

    # 5. Delayed-scaling manager trajectories
    ds_steps = 0
    for _ in range(2):
        hdr = next(lines).split()
        assert hdr[0] == "SECTION" and hdr[1] == "DELAYED_SCALE"
        fmt = int(hdr[2])
        hlen = int(hdr[3])
        margin = int(hdr[4])
        eps = float.fromhex(hdr[5])
        min_s = float.fromhex(hdr[6])
        max_s = float.fromhex(hdr[7])
        n_steps = int(hdr[8])
        fp8_max = 448.0 if fmt == 1 else 57344.0
        ring = []
        for _ in range(n_steps):
            obs, sc, ha = (float.fromhex(t) for t in next(lines).split())
            ring.append(obs)
            if len(ring) > hlen:
                ring.pop(0)
            exp_ha = max(ring) if ring else 0.0
            denom = max(exp_ha, eps)
            exp_sc = (fp8_max / denom) * (2.0 ** (-margin))
            exp_sc = min(max(exp_sc, min_s), max_s)
            exp_sc_f32 = f32_bits_to_float(float_to_f32_bits(exp_sc))
            assert math.isclose(ha, exp_ha, rel_tol=0.0, abs_tol=0.0)
            assert math.isclose(sc, exp_sc_f32, rel_tol=1e-6, abs_tol=1e-6), (
                f"delayed scale mismatch: got {sc}, exp {exp_sc_f32}"
            )
            ds_steps += 1

    # 6. Block-scaled W8A8 GEMM + STE backward parity
    hdr = next(lines).split()
    assert hdr[0] == "SECTION" and hdr[1] == "BLOCK_GEMM"
    n_gemm = int(hdr[2])
    gemm_max_err = 0.0
    bwd_max_err = 0.0
    total_vals = 0
    for _ in range(n_gemm):
        ch = next(lines).split()
        assert ch[0] == "CASE"
        M, N, K, BK, BN, has_bias = (int(ch[i]) for i in range(1, 7))
        KB = K // BK
        Xq = read_u8_line(lines)
        Sx = read_f32_line(lines)
        Wq = read_u8_line(lines)
        Sw = read_f32_line(lines)
        bias = read_f32_line(lines) if has_bias else None
        Y = read_f32_line(lines)
        dY = read_f32_line(lines)
        dX = read_f32_line(lines)
        dW = read_f32_line(lines)
        dB = read_f32_line(lines)

        X_deq = [
            float(ref_e4m3_decode(Xq[m * K + k])[1]) * Sx[m * KB + (k // BK)]
            for m in range(M)
            for k in range(K)
        ]
        W_deq = [
            float(ref_e4m3_decode(Wq[n * K + k])[1])
            * Sw[(n // BN) * KB + (k // BK)]
            for n in range(N)
            for k in range(K)
        ]

        for m in range(M):
            for n in range(N):
                ref = math.fsum(
                    X_deq[m * K + k] * W_deq[n * K + k] for k in range(K)
                )
                if bias is not None:
                    ref += bias[n]
                ref_f32 = f32_bits_to_float(float_to_f32_bits(ref))
                err = abs(Y[m * N + n] - ref_f32)
                assert err <= 1e-6 and math.isclose(
                    Y[m * N + n], ref, rel_tol=1e-6, abs_tol=1e-6
                ), (
                    f"GEMM ({M},{N},{K},bk={BK},bn={BN}) Y[{m},{n}]: "
                    f"got={Y[m*N+n]} ref_f32={ref_f32} ref={ref} err={err:.3e}"
                )
                gemm_max_err = max(gemm_max_err, err)
                total_vals += 1

        for m in range(M):
            for k in range(K):
                ref = math.fsum(
                    dY[m * N + n] * W_deq[n * K + k] for n in range(N)
                )
                ref_f32 = f32_bits_to_float(float_to_f32_bits(ref))
                err = abs(dX[m * K + k] - ref_f32)
                assert err <= 1e-6 and math.isclose(
                    dX[m * K + k], ref, rel_tol=1e-6, abs_tol=1e-6
                ), f"dX[{m},{k}] err={err:.3e}"
                bwd_max_err = max(bwd_max_err, err)
                total_vals += 1

        for n in range(N):
            for k in range(K):
                ref = math.fsum(
                    dY[m * N + n] * X_deq[m * K + k] for m in range(M)
                )
                ref_f32 = f32_bits_to_float(float_to_f32_bits(ref))
                err = abs(dW[n * K + k] - ref_f32)
                assert err <= 1e-6 and math.isclose(
                    dW[n * K + k], ref, rel_tol=1e-6, abs_tol=1e-6
                ), f"dW[{n},{k}] err={err:.3e}"
                bwd_max_err = max(bwd_max_err, err)
                total_vals += 1

        for n in range(N):
            ref = math.fsum(dY[m * N + n] for m in range(M))
            ref_f32 = f32_bits_to_float(float_to_f32_bits(ref))
            err = abs(dB[n] - ref_f32)
            assert err <= 1e-6 and math.isclose(
                dB[n], ref, rel_tol=1e-6, abs_tol=1e-6
            ), f"dB[{n}] err={err:.3e}"
            bwd_max_err = max(bwd_max_err, err)
            total_vals += 1

    print(
        f"FP8 Python oracle: PASS (512 FP8 codepoints, {rne_checked} RNE boundaries, "
        f"{ds_steps} delayed-scale steps, {n_gemm} block-GEMM cases, {total_vals} tensor values, "
        f"gemm_max_err={gemm_max_err:.3e}, ste_bwd_max_err={bwd_max_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
