"""Independent Paged GQA attention and canonical prefix key oracles."""
import hashlib
import math
import struct
import subprocess
import sys


def read_vec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


def dense_gqa_decode_reference(seq_len, hq, hkv, hd, scale, q, k, v):
    assert hq % hkv == 0
    group_size = hq // hkv
    out = [0.0] * (hq * hd)
    for qh in range(hq):
        kv_h = qh // group_size
        q_off = qh * hd
        scores = []
        for t in range(seq_len):
            k_off = (t * hkv + kv_h) * hd
            dot = math.fsum(q[q_off + d] * k[k_off + d] for d in range(hd))
            scores.append(dot * scale)
        max_s = max(scores)
        weights = [math.exp(s - max_s) for s in scores]
        denom = math.fsum(weights)
        for d in range(hd):
            acc = math.fsum(
                (weights[t] / denom) * v[(t * hkv + kv_h) * hd + d]
                for t in range(seq_len)
            )
            out[q_off + d] = acc
    return out


def verify(binary):
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    cases = 0
    comparisons = 0
    max_abs = 0.0

    for header in lines:
        parts = header.split()
        if len(parts) != 6 or parts[0] != "CASE":
            raise AssertionError(f"unexpected oracle header: {header!r}")
        seq_len, hq, hkv, hd = (int(parts[i]) for i in range(1, 5))
        scale = float.fromhex(parts[5])
        q = read_vec(lines)
        k = read_vec(lines)
        v = read_vec(lines)
        actual = read_vec(lines)

        if len(q) != hq * hd or len(actual) != hq * hd:
            raise AssertionError(f"Q/out length mismatch in case {cases}")
        if len(k) != seq_len * hkv * hd or len(v) != seq_len * hkv * hd:
            raise AssertionError(f"K/V length mismatch in case {cases}")

        expected = dense_gqa_decode_reference(
            seq_len, hq, hkv, hd, scale, q, k, v
        )
        for idx, (va, ve) in enumerate(zip(actual, expected)):
            err = abs(va - ve)
            if not math.isclose(va, ve, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(
                    f"case {cases} (len={seq_len}, hq={hq}, hkv={hkv}, hd={hd}) "
                    f"out[{idx}]: paged={va:.10g} oracle={ve:.10g} err={err:.3e}"
                )
            if err > max_abs:
                max_abs = err
            comparisons += 1
        cases += 1

    if cases != 8:
        raise AssertionError(f"expected 8 oracle cases, saw {cases}")
    print(
        f"PAGED_KV Python f64 oracle: PASS ({cases} cases, {comparisons} values, "
        f"max_abs_err={max_abs:.3e})"
    )


def verify_hashes(binary):
    actual = subprocess.run(
        [binary, "--hash-oracle"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    expected = []
    for block_size in range(1, 65):
        tokens = [0xFFFFFFFF - 0x01234567 * i for i in range(block_size)]
        payload = struct.pack(f"<{block_size + 1}I", block_size, *tokens)
        legacy = b"ETPKV1\0\0"
        bound = b"ETPKV2\0\0" + bytes(range(96))
        zeros = bytes(32)
        legacy_root = hashlib.sha256(legacy + zeros + payload).digest()
        bound_root = hashlib.sha256(bound + b"\0" + zeros + payload).digest()
        expected.extend([
            legacy_root.hex(),
            hashlib.sha256(legacy + legacy_root + payload).hexdigest(),
            bound_root.hex(),
            hashlib.sha256(bound + b"\1" + bound_root + payload).hexdigest(),
            hashlib.sha256(bound + b"\1" + zeros + payload).hexdigest(),
        ])
    if len(actual) != len(expected):
        raise AssertionError(f"expected {len(expected)} hashes, saw {len(actual)}")
    for index, (produced, reference) in enumerate(zip(actual, expected)):
        if produced != reference:
            raise AssertionError(f"hash {index}: native={produced} oracle={reference}")
    print(f"PAGED_KV Python SHA-256 oracle: PASS ({len(expected)} exact digests)")


if __name__ == "__main__":
    verify(sys.argv[1])
    verify_hashes(sys.argv[1])
