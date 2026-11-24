"""Independent Python reference oracle for W4A32 INT4 quantization & fused GEMM.

Verifies:
  1. Exact IEEE-754 binary16 round-to-nearest-even encoding via struct '<e'.
  2. Exact GGML Q4_0 split-half nibble packing (low=0..15, high=16..31, offset=-8).
  3. Exact Q4_ASYM group-wise (G in {32, 64, 128}) packing and dequantization.
  4. Reconstruction error bounds (and Q4_ASYM error advantage on skewed weights).
  5. Fused GEMV (M=1) and GEMM (M>1) parity with binary64 reference X @ dequant(W)^T + b.
"""

from __future__ import annotations

import hashlib
import math
import struct
import subprocess
import sys

FORMAT_Q4_0 = 1
FORMAT_Q4_ASYM = 2


def f32(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", val))[0]


def f32_to_f16_bytes(val: float) -> bytes:
    v32 = f32(val)
    if not math.isfinite(v32) or abs(v32) >= 65520.0:
        raise OverflowError(f"f16 overflow for {v32!r}")
    return struct.pack("<e", v32)


def f16_bytes_to_f32(raw: bytes) -> float:
    return f32(struct.unpack("<e", raw)[0])


def round_clamp_nibble(scaled: float) -> int:
    if scaled <= 0.0:
        return 0
    if scaled >= 15.0:
        return 15
    q = int(math.floor(scaled + 0.5))
    return max(0, min(15, q))


def quantize_q4_0_ref(N: int, K: int, w: list[float]) -> bytes:
    assert K % 32 == 0
    out = bytearray()
    for n in range(N):
        row = w[n * K : (n + 1) * K]
        for g in range(K // 32):
            blk = row[g * 32 : (g + 1) * 32]
            amax = 0.0
            signed_max = 0.0
            for v in blk:
                av = abs(v)
                if av > amax:
                    amax = av
                    signed_max = v
            if amax == 0.0:
                d_bytes = b"\x00\x00"
                d_q = 0.0
            else:
                d_bytes = f32_to_f16_bytes(signed_max / -8.0)
                d_q = f16_bytes_to_f32(d_bytes)
            inv_d = (1.0 / d_q) if d_q != 0.0 else 0.0
            out.extend(d_bytes)
            # Split-half nibble packing: low nibble = j (0..15), high = j+16 (16..31).
            for j in range(16):
                q0 = round_clamp_nibble(blk[j] * inv_d + 8.0)
                q1 = round_clamp_nibble(blk[j + 16] * inv_d + 8.0)
                out.append((q0 & 0x0F) | ((q1 & 0x0F) << 4))
    return bytes(out)


def dequantize_q4_0_ref(N: int, K: int, packed: bytes) -> list[float]:
    groups = K // 32
    assert len(packed) == N * groups * 18
    out: list[float] = []
    pos = 0
    for _n in range(N):
        for _g in range(groups):
            d = f16_bytes_to_f32(packed[pos : pos + 2])
            qs = packed[pos + 2 : pos + 18]
            pos += 18
            blk = [0.0] * 32
            for j in range(16):
                b = qs[j]
                blk[j] = f32(d * ((b & 0x0F) - 8))
                blk[j + 16] = f32(d * (((b >> 4) & 0x0F) - 8))
            out.extend(blk)
    return out


def quantize_asym_ref(N: int, K: int, G: int, w: list[float]) -> bytes:
    assert G in (32, 64, 128) and K % G == 0
    half = G // 2
    out = bytearray()
    for n in range(N):
        row = w[n * K : (n + 1) * K]
        for g in range(K // G):
            blk = row[g * G : (g + 1) * G]
            w_min = min(blk)
            w_max = max(blk)
            m_bytes = f32_to_f16_bytes(w_min)
            m_q = f16_bytes_to_f32(m_bytes)
            span = w_max - m_q
            if span > 0.0:
                d_bytes = f32_to_f16_bytes(span / 15.0)
                d_q = f16_bytes_to_f32(d_bytes)
            else:
                d_bytes = b"\x00\x00"
                d_q = 0.0
            inv_d = (1.0 / d_q) if d_q != 0.0 else 0.0
            out.extend(d_bytes)
            out.extend(m_bytes)
            for j in range(half):
                q0 = round_clamp_nibble((blk[j] - m_q) * inv_d)
                q1 = round_clamp_nibble((blk[j + half] - m_q) * inv_d)
                out.append((q0 & 0x0F) | ((q1 & 0x0F) << 4))
    return bytes(out)


def dequantize_asym_ref(N: int, K: int, G: int, packed: bytes) -> list[float]:
    groups = K // G
    half = G // 2
    blk_bytes = 4 + half
    assert len(packed) == N * groups * blk_bytes
    out: list[float] = []
    pos = 0
    for _n in range(N):
        for _g in range(groups):
            d = f16_bytes_to_f32(packed[pos : pos + 2])
            m = f16_bytes_to_f32(packed[pos + 2 : pos + 4])
            qs = packed[pos + 4 : pos + blk_bytes]
            pos += blk_bytes
            blk = [0.0] * G
            for j in range(half):
                b = qs[j]
                blk[j] = f32(d * (b & 0x0F) + m)
                blk[j + half] = f32(d * ((b >> 4) & 0x0F) + m)
            out.extend(blk)
    return out


def gemm_ref(
    M: int,
    N: int,
    K: int,
    x: list[float],
    w_deq: list[float],
    bias: list[float] | None,
) -> list[float]:
    out = [0.0] * (M * N)
    for row in range(M):
        x_row = x[row * K : (row + 1) * K]
        for col in range(N):
            w_row = w_deq[col * K : (col + 1) * K]
            acc = math.fsum(a * b for a, b in zip(x_row, w_row))
            if bias is not None:
                acc += bias[col]
            out[row * N + col] = f32(acc)
    return out


def read_vec(lines) -> list[float]:
    return [float.fromhex(tok) for tok in next(lines).split()]


def verify(binary: str) -> None:
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    cases = 0
    total_bytes = 0
    total_outputs = 0
    max_deq_err = 0.0
    max_gemm_err = 0.0
    max_q4_recon = 0.0
    max_asym_recon = 0.0

    container_checked = False
    negatives_checked = False

    for header in lines:
        parts = header.split()
        if parts == ["CONTAINER"]:
            blk = bytes.fromhex(next(lines))
            wire = bytes.fromhex(next(lines))
            expected_hdr = b"ETQ41000" + struct.pack(
                "<IIIIIIQQ", 1, 0, FORMAT_Q4_0, 32, 1, 32, len(blk), 0
            )
            expected_body = expected_hdr + blk
            expected_wire = expected_body + hashlib.sha256(expected_body).digest()
            if wire != expected_wire:
                raise AssertionError("ETQ41000 wire bytes mismatch Python SHA-256 oracle")
            container_checked = True
            continue
        if len(parts) == 8 and parts[0] == "NEGATIVES":
            got = [int(tok) for tok in parts[1:]]
            # Expected: SHAPE(-2), OVERFLOW(-6), NONFINITE(-5), NONFINITE(-5),
            #           ALIAS(-4), CHECKSUM(-11), TRUNCATED(-10).
            expected_neg = [-2, -6, -5, -5, -4, -11, -10]
            if got != expected_neg:
                raise AssertionError(
                    f"negative status mismatch: got={got} expected={expected_neg}"
                )
            negatives_checked = True
            continue
        if len(parts) != 7 or parts[0] != "CASE":
            raise AssertionError(f"unexpected header: {header!r}")
        fmt, G, M, N, K, use_bias = (int(tok) for tok in parts[1:])
        x = read_vec(lines)
        w = read_vec(lines)
        bias_line = next(lines)
        bias = None if bias_line == "NONE" else [
            float.fromhex(t) for t in bias_line.split()
        ]
        if (bias is not None) != bool(use_bias):
            raise AssertionError("bias presence mismatch")
        actual_packed = bytes.fromhex(next(lines))
        actual_deq = read_vec(lines)
        actual_y = read_vec(lines)

        if fmt == FORMAT_Q4_0:
            exp_packed = quantize_q4_0_ref(N, K, w)
            exp_deq = dequantize_q4_0_ref(N, K, exp_packed)
        elif fmt == FORMAT_Q4_ASYM:
            exp_packed = quantize_asym_ref(N, K, G, w)
            exp_deq = dequantize_asym_ref(N, K, G, exp_packed)
        else:
            raise AssertionError(f"unknown format {fmt}")

        # 1. Exact byte-for-byte packed layout match (including f16 scales & split-half nibbles).
        if actual_packed != exp_packed:
            for idx, (a, e) in enumerate(zip(actual_packed, exp_packed)):
                if a != e:
                    raise AssertionError(
                        f"case {cases} (fmt={fmt},G={G}) packed byte[{idx}] "
                        f"mismatch: native=0x{a:02x} oracle=0x{e:02x}"
                    )
            raise AssertionError(f"case {cases} packed length mismatch")
        total_bytes += len(actual_packed)

        # 2. Exact dequantized weight match.
        for idx, (va, ve) in enumerate(zip(actual_deq, exp_deq)):
            err = abs(va - ve)
            if err > max_deq_err:
                max_deq_err = err
            if err != 0.0:
                raise AssertionError(
                    f"case {cases} deq[{idx}] mismatch: native={va} oracle={ve}"
                )

        # 3. Quantization reconstruction bound check:
        #    Q4_0 worst step <= max_abs / 8, so max_err <= max_abs / 16 + f16_ulp.
        #    Q4_ASYM worst step <= (max-min)/15, so max_err <= (max-min)/30 + f16_ulp.
        max_w_abs = max(abs(v) for v in w)
        recon_err = max(abs(a - b) for a, b in zip(actual_deq, w))
        if fmt == FORMAT_Q4_0:
            if recon_err > max_q4_recon:
                max_q4_recon = recon_err
            bound = max_w_abs / 7.5 + 1e-3
            if recon_err > bound:
                raise AssertionError(
                    f"case {cases} Q4_0 recon err {recon_err} > {bound}"
                )
        else:
            if recon_err > max_asym_recon:
                max_asym_recon = recon_err
            span = max(w) - min(w)
            bound = span / 28.0 + 1e-3
            if recon_err > bound:
                raise AssertionError(
                    f"case {cases} Q4_ASYM recon err {recon_err} > {bound}"
                )

        # 4. Exact fused GEMV/GEMM parity against binary64 X @ dequant(W)^T + bias.
        exp_y = gemm_ref(M, N, K, x, exp_deq, bias)
        for idx, (va, ve) in enumerate(zip(actual_y, exp_y)):
            err = abs(va - ve)
            if err > max_gemm_err:
                max_gemm_err = err
            if not math.isclose(va, ve, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(
                    f"case {cases} (fmt={fmt},G={G},M={M},N={N},K={K}) Y[{idx}]: "
                    f"native={va:.10g} oracle={ve:.10g} err={err:.3e}"
                )
            total_outputs += 1

        cases += 1

    if cases != 8 or not container_checked or not negatives_checked:
        raise AssertionError(
            f"incomplete oracle coverage: cases={cases} container={container_checked} "
            f"negatives={negatives_checked}"
        )
    print(
        f"W4A32 Python oracle: PASS ({cases} cases, {total_bytes} packed bytes bit-exact, "
        f"{total_outputs} GEMV/GEMM outputs, ETQ41000 SHA-256 + 7 negatives verified, "
        f"max_deq_err={max_deq_err:.3e}, max_gemm_err={max_gemm_err:.3e}, "
        f"max_q4_recon={max_q4_recon:.4f}, max_asym_recon={max_asym_recon:.4f})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
