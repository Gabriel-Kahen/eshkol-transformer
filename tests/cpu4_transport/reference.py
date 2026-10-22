#!/usr/bin/env python3
"""Independent CPU4 equations and binary64 central differences; stdlib only."""
import argparse
import hashlib
import math
from pathlib import Path
import struct
import time


PARAMETERS = (
    ("blocks/0/attention/key/weight", (4, 4)),
    ("blocks/0/attention/output/weight", (4, 4)),
    ("blocks/0/attention/query/weight", (4, 4)),
    ("blocks/0/attention/value/weight", (4, 4)),
    ("blocks/0/ffn/down/weight", (4, 8)),
    ("blocks/0/ffn/up/weight", (8, 4)),
    ("blocks/0/norm1/bias", (4,)),
    ("blocks/0/norm1/weight", (4,)),
    ("blocks/0/norm2/bias", (4,)),
    ("blocks/0/norm2/weight", (4,)),
    ("head/weight", (256, 4)),
    ("norm_final/bias", (4,)),
    ("norm_final/weight", (4,)),
    ("position_embedding/weight", (4, 4)),
)
IDS = (3, 197, 3, 41)
EPSILON_BITS = 0x3727C5AC
STEP = 1e-5
# Suggested f32 comparison envelopes, fixed before native execution.
FORWARD_ATOL, FORWARD_RTOL = 4e-5, 4e-5
GRADIENT_ATOL, GRADIENT_RTOL = 3e-4, 5e-4


def from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def binary32(value):
    return from_bits(bits(value))


EPSILON = from_bits(EPSILON_BITS)


def parameters():
    """Conditioned strict-load fixture, deliberately not a production initializer."""
    result = []
    for ordinal, (path, shape) in enumerate(PARAMETERS):
        values = [((i * 37 + ordinal * 19) % 127 - 63) / 89
                  for i in range(math.prod(shape))]
        if len(shape) == 1 and path.endswith("weight"):
            values = [value + 1 for value in values]
        result.append([binary32(value) for value in values])
    return result


def upstream():
    return [binary32(((i * 29 + 17) % 101 - 50) / 64) for i in range(1024)]


def linear(rows, weight, output_width):
    width = len(rows[0])
    return [[sum(x[d] * weight[o * width + d] for d in range(width))
             for o in range(output_width)] for x in rows]


def norm(rows, gamma, beta):
    result = []
    for row in rows:
        mean = sum(row) / 4
        centered = [x - mean for x in row]
        inverse = 1 / math.sqrt(sum(x * x for x in centered) / 4 + EPSILON)
        result.append([centered[d] * inverse * gamma[d] + beta[d] for d in range(4)])
    return result


def add(left, right):
    return [[a + b for a, b in zip(x, y)] for x, y in zip(left, right)]


def forward(p, ids=IDS, *, causal=True, wrong_layout=False,
            embedding_weight=None, head_weight=None, embedding_rows=None):
    """Direct token/head equations; no role descriptors or native code imports.

    Optional edge overrides and mutations are reference self-checks only.
    The returned 1024 values are token-major [1,4,256].
    """
    if len(p) != 14 or any(len(value) != math.prod(shape)
                           for value, (_, shape) in zip(p, PARAMETERS)):
        raise ValueError("expected the fourteen complete CPU4 parameter arrays")
    if len(ids) != 4 or any(type(token) is not int or not 0 <= token < 256 for token in ids):
        raise ValueError("expected four exact byte-token IDs")
    table = p[10] if embedding_weight is None else embedding_weight
    head = p[10] if head_weight is None else head_weight
    embedding = ([[table[token * 4 + d] for d in range(4)] for token in ids]
                 if embedding_rows is None else embedding_rows)
    x = [[embedding[t][d] + p[13][t * 4 + d] for d in range(4)] for t in range(4)]
    n = norm(x, p[7], p[6])
    q, k, v = (linear(n, p[index], 4) for index in (2, 0, 3))

    def component(projected, h, t, d):
        if wrong_layout:
            # Incorrectly reinterpret contiguous token storage as [H,T,Dh].
            address = h * 8 + t * 2 + d
            return projected[address // 4][address % 4]
        return projected[t][h * 2 + d]

    attended = [[[0.0, 0.0] for _ in range(4)] for _ in range(2)]
    for h in range(2):
        for t in range(4):
            keys = range(t + 1) if causal else range(4)
            scores = [sum(component(q, h, t, d) * component(k, h, j, d)
                          for d in range(2)) / math.sqrt(2) for j in keys]
            maximum = max(scores)
            probabilities = [math.exp(score - maximum) for score in scores]
            denominator = sum(probabilities)
            for d in range(2):
                attended[h][t][d] = sum(probability * component(v, h, j, d)
                                        for probability, j in zip(probabilities, keys)) / denominator
    merged = [[attended[h][t][d] for h in range(2) for d in range(2)] for t in range(4)]
    if wrong_layout:
        flat = [value for head_rows in attended for row in head_rows for value in row]
        merged = [flat[t * 4:t * 4 + 4] for t in range(4)]
    residual = add(x, linear(merged, p[1], 4))
    wide = linear(norm(residual, p[9], p[8]), p[5], 8)
    activated = [[.5 * z * (1 + math.erf(z / math.sqrt(2))) for z in row] for row in wide]
    y = add(residual, linear(activated, p[4], 4))
    logits = linear(norm(y, p[12], p[11]), head, 256)
    return [value for row in logits for value in row]


def objective(p, seed, **options):
    return sum(value * weight for value, weight in zip(forward(p, **options), seed))


def derivative(p, seed, parameter, index, step=STEP, **options):
    old = parameter[index]
    delta = step * max(1, abs(old))
    try:
        parameter[index] = old + delta
        plus = objective(p, seed, **options)
        parameter[index] = old - delta
        minus = objective(p, seed, **options)
    finally:
        parameter[index] = old
    return (plus - minus) / (2 * delta)


def vjp(p, seed):
    return [[derivative(p, seed, parameter, i) for i in range(len(parameter))]
            for parameter in p]


def near(left, right, atol=2e-6, rtol=2e-5):
    if not (math.isfinite(left) and math.isfinite(right)
            and abs(left - right) <= atol + rtol * abs(right)):
        raise AssertionError(f"reference self-check mismatch: {left} versus {right}")


def self_check(p, seed, logits, gradient):
    assert sum(map(len, p)) == sum(map(len, gradient)) == 1192
    assert len(logits) == len(seed) == 1024
    assert all(math.isfinite(x) for values in (*p, *gradient, logits, seed) for x in values)
    assert all(max(map(abs, values)) > 1e-6 for values in gradient)
    assert len(set(IDS)) == 3 and IDS[0] == IDS[2]
    assert all(x != 0 for index in (7, 9, 12) for x in p[index])

    # An unseen future input cannot change any earlier token's causal output.
    changed = forward(p, (3, 197, 3, 99))
    assert changed[:768] == logits[:768]
    assert max(abs(a - b) for a, b in zip(changed[768:], logits[768:])) > 1e-3
    causal_delta = max(abs(a - b) for a, b in zip(forward(p, causal=False)[:256], logits[:256]))
    layout_delta = max(abs(a - b) for a, b in zip(forward(p, wrong_layout=True), logits))
    assert causal_delta > 1e-3 and layout_delta > 1e-3

    # Independent directional probes test flattened gradient assembly at a
    # different step, including mixed cross-parameter and position-tail changes.
    for salt in (1, 7, 23):
        direction = [[((i * 13 + ordinal * 7 + salt) % 17 - 8) / 64
                      for i in range(len(parameter))] for ordinal, parameter in enumerate(p)]
        step = 2e-5
        plus = [[x + step * d for x, d in zip(values, row)] for values, row in zip(p, direction)]
        minus = [[x - step * d for x, d in zip(values, row)] for values, row in zip(p, direction)]
        measured = (objective(plus, seed) - objective(minus, seed)) / (2 * step)
        predicted = sum(g * d for row, vector in zip(gradient, direction) for g, d in zip(row, vector))
        near(predicted, measured)
    for ordinal, parameter in enumerate(p):
        index = len(parameter) - 1
        near(gradient[ordinal][index], derivative(p, seed, parameter, index, step=5e-6))

    # The tied row must receive both projection and embedding edges; repeated
    # token3 additionally has two separately nonzero embedding contributions.
    fixed_embedding, fixed_head = p[10][:], p[10][:]
    embedding = [[fixed_embedding[token * 4 + d] for d in range(4)] for token in IDS]
    for d in range(4):
        index = IDS[0] * 4 + d
        head_edge = derivative(p, seed, p[10], index, embedding_weight=fixed_embedding)
        embedding_edge = derivative(p, seed, p[10], index, head_weight=fixed_head)
        near(gradient[10][index], head_edge + embedding_edge)
        scatter = [derivative(p, seed, embedding[t], d,
                              embedding_rows=embedding, head_weight=fixed_head) for t in (0, 2)]
        near(embedding_edge, sum(scatter))
        assert all(abs(value) > 1e-5 for value in scatter)
    print(f"CPU4 self-check PASS: 1192 gradients; causal-mutant delta={causal_delta:.6g}; "
          f"head-layout-mutant delta={layout_delta:.6g}; tied/repeated-token scatter; 3 directions")


def c_array(ctype, name, values, encoder):
    encoded = [encoder(value) for value in values]
    rows = ["  " + ", ".join(encoded[i:i + 6]) for i in range(0, len(encoded), 6)]
    return f"static const {ctype} {name}[{len(values)}] = {{\n" + ",\n".join(rows) + "\n};\n"


def header(p, seed, logits, gradient):
    offsets = [0]
    for parameter in p:
        offsets.append(offsets[-1] + len(parameter))
    text = ("/* Generated by independent stdlib CPU4 mathematical reference. */\n"
            "#ifndef CPU4_TRANSPORT_REFERENCE_H\n#define CPU4_TRANSPORT_REFERENCE_H\n"
            "#include <stddef.h>\n#include <stdint.h>\n"
            "#define CPU4_REF_PARAMETER_COUNT 14\n#define CPU4_REF_VALUE_COUNT 1192\n"
            "#define CPU4_REF_LOGIT_COUNT 1024\n"
            f"#define CPU4_REF_EPSILON_BITS UINT32_C(0x{EPSILON_BITS:08x})\n"
            f"#define CPU4_REF_FORWARD_ATOL {FORWARD_ATOL.hex()}\n"
            f"#define CPU4_REF_FORWARD_RTOL {FORWARD_RTOL.hex()}\n"
            f"#define CPU4_REF_GRADIENT_ATOL {GRADIENT_ATOL.hex()}\n"
            f"#define CPU4_REF_GRADIENT_RTOL {GRADIENT_RTOL.hex()}\n")
    text += c_array("size_t", "cpu4_ref_parameter_offsets", offsets, str)
    text += c_array("int64_t", "cpu4_ref_ids", IDS, str)
    text += c_array("uint32_t", "cpu4_ref_parameter_bits", [x for row in p for x in row],
                    lambda x: f"UINT32_C(0x{bits(x):08x})")
    text += c_array("uint32_t", "cpu4_ref_upstream_bits", seed,
                    lambda x: f"UINT32_C(0x{bits(x):08x})")
    text += c_array("double", "cpu4_ref_logits", logits, float.hex)
    text += c_array("double", "cpu4_ref_gradients", [x for row in gradient for x in row], float.hex)
    text += "static const char *const cpu4_ref_parameter_paths[14] = {\n"
    text += ",\n".join(f'  "{path}"' for path, _ in PARAMETERS) + "\n};\n"
    text += c_array("size_t", "cpu4_ref_parameter_ranks", [len(shape) for _, shape in PARAMETERS], str)
    text += "static const uint64_t cpu4_ref_parameter_shapes[14][2] = {\n"
    text += ",\n".join("  {" + ", ".join(map(str, shape if len(shape) == 2 else (*shape, 0))) + "}"
                       for _, shape in PARAMETERS) + "\n};\n#endif\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, help="write generated C header to this path")
    parser.add_argument("--self-check", action="store_true", help="run mathematical sensitivity and derivative checks")
    args = parser.parse_args()
    started = time.monotonic()
    p, seed = parameters(), upstream()
    original = [[bits(x) for x in row] for row in p]
    logits = forward(p)
    gradient = vjp(p, seed)
    assert all(math.isfinite(x) for row in (*p, *gradient, logits, seed) for x in row)
    if args.self_check:
        self_check(p, seed, logits, gradient)
    assert original == [[bits(x) for x in row] for row in p]
    result = header(p, seed, logits, gradient)
    if args.header:
        args.header.write_text(result, encoding="ascii")
    print(f"CPU4 reference: 14 parameters, 1192 values, 1024 logits/upstream; "
          f"header-sha256={hashlib.sha256(result.encode('ascii')).hexdigest()}; "
          f"elapsed={time.monotonic() - started:.3f}s")


if __name__ == "__main__":
    main()
