"""Independent binary64 reference oracle for MOD5 RoPE forward & backward."""
from __future__ import annotations

import math
import subprocess
import sys


def rope_reference(n: int, h: int, t: int, dh: int,
                   positions: list[int], values: list[float],
                   theta: float, scale: float, backward: bool) -> list[float]:
    half = dh // 2
    inv_freq = [math.pow(theta, (-2.0 * k) / dh) for k in range(half)]
    out = [0.0] * len(values)
    for batch in range(n):
        for token in range(t):
            pos = positions[batch * t + token]
            scaled_pos = pos / scale
            cos_v = [math.cos(scaled_pos * inv_freq[k]) for k in range(half)]
            sin_v = [math.sin(scaled_pos * inv_freq[k]) for k in range(half)]
            for head in range(h):
                base = ((batch * h + head) * t + token) * dh
                for k in range(half):
                    v0 = values[base + 2 * k]
                    v1 = values[base + 2 * k + 1]
                    c = cos_v[k]
                    s = sin_v[k]
                    if backward:
                        out[base + 2 * k] = v0 * c + v1 * s
                        out[base + 2 * k + 1] = -v0 * s + v1 * c
                    else:
                        out[base + 2 * k] = v0 * c - v1 * s
                        out[base + 2 * k + 1] = v0 * s + v1 * c
    return out


def verify(binary: str) -> None:
    proc = subprocess.run([binary, "--oracle"], check=True,
                          capture_output=True, text=True)
    lines = iter(proc.stdout.splitlines())
    cases = 0
    comparisons = 0
    max_abs_diff = 0.0
    for header in lines:
        parts = header.split()
        if len(parts) != 7 or parts[0] != "CASE":
            raise AssertionError(f"malformed oracle header: {header!r}")
        n, h, t, dh = (int(parts[i]) for i in range(1, 5))
        theta = float.fromhex(parts[5])
        scale = float.fromhex(parts[6])
        positions = [int(v) for v in next(lines).split()]
        x = [float.fromhex(v) for v in next(lines).split()]
        dy = [float.fromhex(v) for v in next(lines).split()]
        actual_y = [float.fromhex(v) for v in next(lines).split()]
        actual_dx = [float.fromhex(v) for v in next(lines).split()]

        expected_y = rope_reference(n, h, t, dh, positions, x, theta, scale, False)
        expected_dx = rope_reference(n, h, t, dh, positions, dy, theta, scale, True)

        for kind, actual, expected in (
            ("forward", actual_y, expected_y),
            ("backward", actual_dx, expected_dx),
        ):
            if len(actual) != len(expected):
                raise AssertionError(f"{kind} length mismatch on {(n, h, t, dh)}")
            for idx, (got, want) in enumerate(zip(actual, expected)):
                diff = abs(got - want)
                if diff > 1e-6:
                    raise AssertionError(
                        f"RoPE {kind} shape={(n, h, t, dh)} idx={idx}: "
                        f"got {got:.10g}, want {want:.10g}, diff={diff:.3e}"
                    )
                if diff > max_abs_diff:
                    max_abs_diff = diff
                comparisons += 1
        cases += 1

    if cases < 9:
        raise AssertionError(f"expected at least 9 oracle cases, got {cases}")
    print(f"MOD5 RoPE Python oracle: PASS (cases={cases}, values={comparisons}, "
          f"max_abs_diff={max_abs_diff:.3e})")


if __name__ == "__main__":
    verify(sys.argv[1])
