"""Independent double-precision oracle for the native MOD5 subprocess."""
from collections import Counter
import math
import subprocess
import sys


def rmsnorm(shape, x, gamma, eps, upstream):
    width = shape[2]
    y, dx, dgamma = [], [], [0.0] * width
    for start in range(0, len(x), width):
        row = x[start : start + width]
        grad = upstream[start : start + width]
        rstd = 1.0 / math.sqrt(eps + math.fsum(value**2 for value in row) / width)
        inner = math.fsum(g * w * value for g, w, value in zip(grad, gamma, row))
        for d, (value, w, g) in enumerate(zip(row, gamma, grad)):
            y.append(value * rstd * w)
            dx.append(g * w * rstd - value * inner * rstd**3 / width)
            dgamma[d] += g * value * rstd
    return y, dx, dgamma


def swiglu(_shape, gates, up, upstream):
    y, dgate, dup = [], [], []
    for gate, value, grad in zip(gates, up, upstream):
        sigmoid = 1.0 / (1.0 + math.exp(-gate))
        silu = gate * sigmoid
        slope = sigmoid + gate * sigmoid * (1.0 - sigmoid)
        y.append(silu * value)
        dgate.append(grad * value * slope)
        dup.append(grad * silu)
    return y, dgate, dup


def attention(shape, query, key, value, qpos, kpos, mask, slopes, window, upstream):
    batches, heads, kv_heads, queries, keys, width = shape
    out, dq = [0.0] * len(query), [0.0] * len(query)
    dk, dv = [0.0] * len(key), [0.0] * len(value)
    for batch in range(batches):
        for head in range(heads):
            kv_head = head * kv_heads // heads
            for token in range(queries):
                start = ((batch * heads + head) * queries + token) * width
                q = query[start : start + width]
                g = upstream[start : start + width]
                admitted = []
                for index in range(keys):
                    distance = qpos[batch * queries + token] - kpos[batch * keys + index]
                    if mask[(batch * queries + token) * keys + index] and 0 <= distance < window:
                        offset = ((batch * kv_heads + kv_head) * keys + index) * width
                        k = key[offset : offset + width]
                        score = math.fsum(x * y for x, y in zip(q, k)) / math.sqrt(width)
                        admitted.append((offset, score - slopes[head] * distance))
                if not admitted:
                    continue
                peak = max(score for _, score in admitted)
                weights = [math.exp(score - peak) for _, score in admitted]
                normalizer = math.fsum(weights)
                active = [(offset, w / normalizer) for (offset, _), w in zip(admitted, weights)]
                row = [
                    math.fsum(p * value[offset + d] for offset, p in active)
                    for d in range(width)
                ]
                out[start : start + width] = row
                for offset, p in active:
                    dscore = p * math.fsum(
                        grad * (value[offset + d] - row[d]) for d, grad in enumerate(g)
                    )
                    ddot = dscore / math.sqrt(width)
                    for d in range(width):
                        dq[start + d] += ddot * key[offset + d]
                        dk[offset + d] += ddot * q[d]
                        dv[offset + d] += p * g[d]
    return out, dq, dk, dv


def read_vector(lines, parse=float.fromhex):
    return [parse(value) for value in next(lines).split()]


def verify(binary):
    native = subprocess.run([binary, "--oracle"], check=True, capture_output=True, text=True)
    lines = iter(native.stdout.splitlines())
    seen = Counter()
    comparisons = 0
    largest_error = 0.0
    for header in lines:
        operation, *extents = header.split()
        shape = tuple(map(int, extents))
        seen[operation] += 1
        if operation == "rmsnorm.forward":
            x, gamma = read_vector(lines), read_vector(lines)
            (epsilon,) = read_vector(lines)
            expected = rmsnorm(shape, x, gamma, epsilon, read_vector(lines))
        elif operation == "swiglu.forward":
            expected = swiglu(shape, *(read_vector(lines) for _ in range(3)))
        elif operation == "variant-attention.forward":
            q, k, v = (read_vector(lines) for _ in range(3))
            qpos, kpos, mask = (read_vector(lines, int) for _ in range(3))
            slopes = read_vector(lines)
            (window,) = read_vector(lines, int)
            expected = attention(shape, q, k, v, qpos, kpos, mask, slopes,
                                 window, read_vector(lines))
        else:
            raise AssertionError(f"unexpected operation {operation}")

        for output, wanted in enumerate(expected):
            actual = read_vector(lines)
            if len(actual) != len(wanted):
                raise AssertionError(f"{operation} {shape} output {output} length mismatch")
            for i, (value, target) in enumerate(zip(actual, wanted)):
                if not math.isclose(value, target, rel_tol=3e-6, abs_tol=3e-7):
                    raise AssertionError(
                        f"{operation} {shape} output {output} index {i}: "
                        f"native {value:.10g}, reference {target:.10g}"
                    )
                largest_error = max(largest_error, abs(value - target))
                comparisons += 1
    if seen != {"rmsnorm.forward": 5, "swiglu.forward": 5, "variant-attention.forward": 30}:
        raise AssertionError(f"unexpected case roster {seen}")
    print(f"MOD5 Python reference: PASS ({sum(seen.values())} cases, "
          f"{comparisons} values, maximum absolute error {largest_error:.3g})")


if __name__ == "__main__":
    verify(sys.argv[1])
