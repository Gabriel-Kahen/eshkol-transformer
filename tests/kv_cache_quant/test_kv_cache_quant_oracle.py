"""Independent exact-Fraction + binary64 Python oracle for Quantized KV-Cache."""
from fractions import Fraction
import math
import struct
import subprocess
import sys


def f32_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def float_to_f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def to_f32(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", val))[0]


def f32_bits_to_fraction(bits: int):
    sign = -1 if (bits >> 31) else 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0xFF:
        return None
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
        return None
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


def build_e4m3_grid():
    grid = []
    for c in range(0x00, 0x7F):
        _, frac = ref_e4m3_decode(c)
        grid.append((c, frac))
    grid.append((0x7F, Fraction(480)))
    return grid


E4M3_GRID = build_e4m3_grid()


def ref_e4m3_encode_saturating(bits: int):
    frac = f32_bits_to_fraction(bits)
    if frac is None:
        return -4, 0  # ET_KVQ_ERR_NONFINITE
    sign_bit = 0x80 if (bits >> 31) else 0x00
    a = abs(frac)
    if a == 0:
        return 0, sign_bit
    if a >= E4M3_GRID[-1][1]:
        return 0, sign_bit | 0x7E
    lo, hi = 0, len(E4M3_GRID) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if E4M3_GRID[mid][1] <= a:
            lo = mid
        else:
            hi = mid
    c_lo, v_lo = E4M3_GRID[lo]
    c_hi, v_hi = E4M3_GRID[hi]
    d_lo = a - v_lo
    d_hi = v_hi - a
    if d_lo < d_hi:
        chosen = c_lo
    elif d_hi < d_lo:
        chosen = c_hi
    else:
        chosen = c_lo if (c_lo & 1) == 0 else c_hi
    if chosen == 0x7F:
        return 0, sign_bit | 0x7E
    return 0, sign_bit | chosen


def ref_int8_quant_dequant(vec_bits):
    vals = [f32_bits_to_float(b) for b in vec_bits]
    amax = max(abs(x) for x in vals)
    if amax == 0.0:
        scale_bits = 0x00000000
        q_u8 = [0] * len(vals)
        deq_bits = [0x00000000] * len(vals)
        return scale_bits, q_u8, deq_bits
    scale = to_f32(amax / 127.0)
    if scale == 0.0:
        scale = f32_bits_to_float(1)
    scale_bits = float_to_f32_bits(scale)
    q_u8 = []
    deq_bits = []
    for x in vals:
        v = x / scale
        if v >= 127.0:
            qi = 127
        elif v <= -127.0:
            qi = -127
        else:
            qi = int(round(v))
        q_u8.append(qi & 0xFF)
        deq_bits.append(float_to_f32_bits(to_f32(float(qi) * scale)))
    return scale_bits, q_u8, deq_bits


def ref_fp8_quant_dequant(vec_bits):
    vals = [f32_bits_to_float(b) for b in vec_bits]
    amax = max(abs(x) for x in vals)
    if amax == 0.0:
        scale_bits = 0x00000000
        q_u8 = [0] * len(vals)
        deq_bits = [0x00000000] * len(vals)
        return scale_bits, q_u8, deq_bits
    scale = to_f32(amax / 448.0)
    if scale == 0.0:
        scale = f32_bits_to_float(1)
    scale_bits = float_to_f32_bits(scale)
    q_u8 = []
    deq_bits = []
    for x in vals:
        scaled = to_f32(x / scale)
        st, code = ref_e4m3_encode_saturating(float_to_f32_bits(scaled))
        assert st == 0
        q_u8.append(code)
        dec_bits, _ = ref_e4m3_decode(code)
        dec_f = f32_bits_to_float(dec_bits)
        deq_bits.append(float_to_f32_bits(to_f32(dec_f * scale)))
    return scale_bits, q_u8, deq_bits


def ref_gqa_decode_twopass(seq_len, hq, hkv, hd, window, scale, q, k_deq, v_deq):
    group = hq // hkv
    t_start = 0 if (window == 0 or window >= seq_len) else (seq_len - window)
    out = [0.0] * (hq * hd)
    for qh in range(hq):
        kv_h = qh // group
        q_off = qh * hd
        scores = []
        for t in range(t_start, seq_len):
            k_vec = k_deq[t][kv_h * hd : (kv_h + 1) * hd]
            dot = math.fsum(q[q_off + d] * k_vec[d] for d in range(hd))
            scores.append(dot * scale)
        m = max(scores)
        ws = [math.exp(s - m) for s in scores]
        denom = math.fsum(ws)
        for d in range(hd):
            acc = math.fsum(
                (ws[i] / denom) * v_deq[t_start + i][kv_h * hd + d]
                for i in range(len(scores))
            )
            out[q_off + d] = to_f32(acc)
    return out


def read_hex_u32(lines):
    return [int(tok, 16) for tok in next(lines).split()]


def read_hex_u8(lines):
    return [int(tok, 16) for tok in next(lines).split()]


def verify(binary: str):
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())

    # 1. Exhaustive 256 FP8 E4M3FN decode check (0 ULP).
    assert next(lines) == "E4M3_DECODE_256"
    for c in range(256):
        c_hex, st_s, b_hex = next(lines).split()
        assert int(c_hex, 16) == c
        st = int(st_s)
        bits = int(b_hex, 16)
        ref = ref_e4m3_decode(c)
        if ref is None:
            assert st == -4, f"code {c:#x} expected -4 got {st}"
        else:
            ref_bits, _ = ref
            assert st == 0 and bits == ref_bits, (
                f"decode {c:#x}: got {bits:#010x} expected {ref_bits:#010x}"
            )

    # 2. Boundary/tie FP8 E4M3FN encode check (0 ULP).
    hdr = next(lines).split()
    assert hdr[0] == "E4M3_ENCODE"
    n_enc = int(hdr[1])
    for _ in range(n_enc):
        b_hex, st_s, c_hex = next(lines).split()
        bits = int(b_hex, 16)
        st = int(st_s)
        code = int(c_hex, 16)
        ref_st, ref_code = ref_e4m3_encode_saturating(bits)
        assert st == ref_st and code == ref_code, (
            f"encode {bits:#010x}: got ({st},{code:#x}) expected ({ref_st},{ref_code:#x})"
        )

    # 3. Vector INT8 & FP8 quantize/dequantize bit-exact check (0.0 ULP).
    hdr = next(lines).split()
    assert hdr[0] == "VEC_CASES"
    n_vec = int(hdr[1])
    for ci in range(n_vec):
        vhdr = next(lines).split()
        assert vhdr[0] == "VEC" and int(vhdr[1]) == ci
        src_bits = read_hex_u32(lines)
        sc_i8 = int(next(lines).strip(), 16)
        q_i8 = read_hex_u8(lines)
        deq_i8 = read_hex_u32(lines)
        sc_f8 = int(next(lines).strip(), 16)
        q_f8 = read_hex_u8(lines)
        deq_f8 = read_hex_u32(lines)

        r_sc_i8, r_q_i8, r_deq_i8 = ref_int8_quant_dequant(src_bits)
        assert sc_i8 == r_sc_i8 and q_i8 == r_q_i8 and deq_i8 == r_deq_i8, (
            f"INT8 vec {ci} mismatch"
        )
        r_sc_f8, r_q_f8, r_deq_f8 = ref_fp8_quant_dequant(src_bits)
        assert sc_f8 == r_sc_f8 and q_f8 == r_q_f8 and deq_f8 == r_deq_f8, (
            f"FP8 vec {ci} mismatch"
        )

    # 4. End-to-end GQA decode cases across FP32, INT8, FP8 + sliding window.
    hdr = next(lines).split()
    assert hdr[0] == "GQA_CASES"
    n_gqa = int(hdr[1])
    max_gqa_err = 0.0
    total_vals = 0

    for gi in range(n_gqa):
        ghdr = next(lines).split()
        assert ghdr[0] == "GQA"
        fmt, seq_len, hq, hkv, hd, win = (int(ghdr[i]) for i in range(1, 7))
        scale = f32_bits_to_float(int(ghdr[7], 16))
        q_bits = read_hex_u32(lines)
        q_f = [f32_bits_to_float(b) for b in q_bits]

        k_deq_all = []
        v_deq_all = []
        for pos in range(seq_len):
            k_orig_bits = read_hex_u32(lines)
            v_orig_bits = read_hex_u32(lines)
            dk_bits = read_hex_u32(lines)
            dv_bits = read_hex_u32(lines)

            # Verify exact quantization bytes, scales, and dequantized bits (0 ULP)!
            if fmt == 1:  # FP32
                assert dk_bits == k_orig_bits
                assert dv_bits == v_orig_bits
            else:
                qfn = ref_int8_quant_dequant if fmt == 2 else ref_fp8_quant_dequant
                for side, (orig_b, deq_b) in enumerate(
                    ((k_orig_bits, dk_bits), (v_orig_bits, dv_bits))
                ):
                    for h in range(hkv):
                        toks = next(lines).split()
                        got_sc = int(toks[0], 16)
                        got_raw = [int(x, 16) for x in toks[1:]]
                        sl_orig = orig_b[h * hd : (h + 1) * hd]
                        sl_deq = deq_b[h * hd : (h + 1) * hd]
                        r_sc, r_raw, r_deq = qfn(sl_orig)
                        assert got_sc == r_sc, f"scale mismatch gqa={gi} pos={pos} side={side} h={h}"
                        assert got_raw == r_raw, f"raw bytes mismatch gqa={gi} pos={pos} side={side} h={h}"
                        assert sl_deq == r_deq, f"dequant bits mismatch gqa={gi} pos={pos} side={side} h={h}"

            k_deq_all.append([f32_bits_to_float(b) for b in dk_bits])
            v_deq_all.append([f32_bits_to_float(b) for b in dv_bits])

        actual_bits = read_hex_u32(lines)
        actual = [f32_bits_to_float(b) for b in actual_bits]
        expected = ref_gqa_decode_twopass(
            seq_len, hq, hkv, hd, win, scale, q_f, k_deq_all, v_deq_all
        )
        for idx, (va, ve) in enumerate(zip(actual, expected)):
            err = abs(va - ve)
            if err > 1e-6:
                raise AssertionError(
                    f"GQA case {gi} fmt={fmt} win={win} idx={idx}: "
                    f"got={va:.10g} ref={ve:.10g} err={err:.3e}"
                )
            if err > max_gqa_err:
                max_gqa_err = err
            total_vals += 1

    print(
        f"KVQ Python oracle: PASS (256 FP8 codes + {n_enc} ties + "
        f"{n_vec} vec pairs 0.0 ULP + {n_gqa} GQA cases / {total_vals} vals "
        f"max_abs_err={max_gqa_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
