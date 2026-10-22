#!/usr/bin/env python3
"""Independent binary64 T4 oracles; generate temporary C fixtures, never Git data."""
import math
from pathlib import Path
import struct
import sys


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def values(count, salt):
    return [f32(((i * 17 + salt * 11) % 71 - 35) / 31) for i in range(count)]


def tensor(shape, data, dtype="f32"):
    return {"shape": shape, "data": data, "dtype": dtype}


def gradients(function, parameters, upstream):
    """Central differences of independent forward, never provider forward."""
    answer = []
    for parameter in parameters:
        gradient = []
        for index, original in enumerate(parameter):
            step = 1e-5 * max(1, abs(original))
            parameter[index] = original + step
            positive = sum(a * b for a, b in zip(function(), upstream))
            parameter[index] = original - step
            negative = sum(a * b for a, b in zip(function(), upstream))
            parameter[index] = original
            gradient.append((positive - negative) / (2 * step))
        answer.append(gradient)
    return answer


CASES = []


def pair(provider, capability, operation, request, inputs, output_shape,
         function, parameters, backward_inputs, gradient_shapes):
    expected = function()
    upstream = values(len(expected), 19)
    derivative = gradients(function, parameters, upstream)
    CASES.append((provider, capability, operation + ".forward", request,
                  inputs, [(output_shape, expected)]))
    CASES.append((provider, capability, operation + ".backward", request,
                  backward_inputs(tensor(output_shape, upstream)),
                  list(zip(gradient_shapes, derivative))))


def make_cases():
    for vocabulary, ids in ((256, [3, 197, 3, 41]), (4, [0, 1, 2, 3])):
        weight = values(vocabulary * 4, 1)
        index = tensor([1, 4], ids, "i64")
        table = tensor([vocabulary, 4], weight)
        # Forward/backward have distinct existing capability names.
        pair(0, "n3k.embedding-forward", "n3k.embedding", [1, 4, vocabulary, 4],
             [index, table], [1, 4, 4],
             lambda: [weight[token * 4 + d] for token in ids for d in range(4)],
             [weight], lambda up: [index, up], [[vocabulary, 4]])
        row = CASES[-1]
        CASES[-1] = (row[0], "n3k.embedding-backward", *row[2:])

    for width, output in ((4, 4), (4, 8), (8, 4), (4, 256)):
        x, weight = values(4 * width, 2), values(output * width, 3)
        inputs = [tensor([1, 4, width], x), tensor([output, width], weight)]
        def linear():
            return [sum(x[t * width + d] * weight[o * width + d]
                        for d in range(width)) for t in range(4) for o in range(output)]
        pair(0, "n3k.linear", "n3k.linear", [1, 4, width, output],
             inputs, [1, 4, output], linear, [x, weight],
             lambda up: inputs + [up], [[1, 4, width], [output, width]])
        # Existing operation spelling retains the no-bias suffix.
        for i in (-2, -1):
            row = CASES[i]
            CASES[i] = (*row[:2], row[2] + "-no-bias", *row[3:])

    x = values(32, 4)
    pair(0, "n3k.gelu", "n3k.gelu", [1, 4, 8], [tensor([1, 4, 8], x)],
         [1, 4, 8], lambda: [v * .5 * (1 + math.erf(v / math.sqrt(2))) for v in x],
         [x], lambda up: [tensor([1, 4, 8], x), up], [[1, 4, 8]])

    for capability, operation, edges in (("n3k.residual", "n3k.residual", 2),
                                        ("n3k.activation-sum", "n3k.sum2", 2),
                                        ("n3k.activation-sum", "n3k.sum3", 3)):
        parameters = [values(16, i + 5) for i in range(edges)]
        pair(0, capability, operation, [1, 4, 4],
             [tensor([1, 4, 4], v) for v in parameters], [1, 4, 4],
             lambda: [sum(v[i] for v in parameters) for i in range(16)],
             parameters, lambda up: [up], [[1, 4, 4]] * edges)

    # Nonsymmetric T4/H2: token-major and head-major addresses cannot coincide.
    for split in (True, False):
        x = values(16, 8)
        input_shape = [1, 4, 4] if split else [1, 2, 4, 2]
        output_shape = [1, 2, 4, 2] if split else [1, 4, 4]
        def layout():
            if split:
                return [x[t * 4 + h * 2 + d]
                        for h in range(2) for t in range(4) for d in range(2)]
            return [x[h * 8 + t * 2 + d]
                    for t in range(4) for h in range(2) for d in range(2)]
        pair(0, "n3k.head-layout", "n3k.heads." + ("split" if split else "merge"),
             [1, 4, 2, 2], [tensor(input_shape, x)], output_shape, layout,
             [x], lambda up: [up], [input_shape])

    x, gamma, beta = values(16, 9), values(4, 10), values(4, 11)
    epsilon = tensor([], [f32(1e-5)])
    inputs = [tensor([1, 4, 4], x), tensor([4], gamma), tensor([4], beta), epsilon]
    def norm():
        y = []
        for t in range(4):
            row = x[t * 4:t * 4 + 4]
            mean = sum(row) / 4
            inv = 1 / math.sqrt(sum((v - mean) ** 2 for v in row) / 4 + epsilon["data"][0])
            y.extend((v - mean) * inv * gamma[d] + beta[d] for d, v in enumerate(row))
        return y
    pair(1, "kernel.norm", "layer-norm", [1, 4, 4], inputs, [1, 4, 4],
         norm, [x, gamma, beta], lambda up: [inputs[0], inputs[1], epsilon, up],
         [[1, 4, 4], [4], [4]])

    q, k, v = values(16, 12), values(16, 13), values(16, 14)
    positions = [0, 1, 2, 3]
    # A hole exercises the actual mask together with the positional causal cut.
    mask = [1] * 16
    mask[3 * 4 + 1] = 0
    inputs = [tensor([1, 2, 4, 2], a) for a in (q, k, v)]
    inputs += [tensor([1, 4], positions[:], "i64"),
               tensor([1, 4], positions[:], "i64"), tensor([1, 4, 4], mask, "bool")]
    def attention():
        result = []
        for h in range(2):
            for t in range(4):
                keys = [j for j in range(4) if positions[j] <= positions[t] and mask[t * 4 + j]]
                scores = [sum(q[h * 8 + t * 2 + d] * k[h * 8 + j * 2 + d]
                              for d in range(2)) / math.sqrt(2) for j in keys]
                peak = max(scores)
                weights = [math.exp(a - peak) for a in scores]
                total = sum(weights)
                result.extend(sum(w * v[h * 8 + j * 2 + d] for w, j in zip(weights, keys)) / total
                              for d in range(2))
        return result
    pair(2, "kernel.causal-attention", "causal-attention", [1, 2, 2, 4, 4, 2],
         inputs, [1, 2, 4, 2], attention, [q, k, v], lambda up: inputs + [up],
         [[1, 2, 4, 2]] * 3)


def header():
    lines = ["/* Generated independent CPU4 forward/central-difference fixtures. */"]
    records = []
    for case_index, (provider, capability, operation, shape, inputs, outputs) in enumerate(CASES):
        prefix = f"c{case_index}"
        for index, value in enumerate(inputs):
            dtype = value["dtype"]
            name = f"{prefix}_i{index}"
            ctype = {"f32": "float", "i64": "int64_t", "bool": "unsigned char"}[dtype]
            data = ",".join(v.hex() + "f" if dtype == "f32" else str(v) for v in value["data"])
            lines.append(f"static {ctype} {name}[] = {{{data}}};")
        for index, (_, expected) in enumerate(outputs):
            lines += [f"static float {prefix}_o{index}[{len(expected)}];",
                      f"static const double {prefix}_e{index}[] = {{{','.join(v.hex() for v in expected)}}};"]
        def descriptor(name, dtype, tensor_shape, count, expected="NULL"):
            dims = ",".join(map(str, tensor_shape or [0]))
            return '{' + f'{name},sizeof({name}),"{dtype}",{len(tensor_shape)},{{{dims}}},{expected},{count}' + '}'
        ins = [descriptor(f"{prefix}_i{i}", item["dtype"], item["shape"], len(item["data"]))
               for i, item in enumerate(inputs)]
        outs = [descriptor(f"{prefix}_o{i}", "f32", dims, len(expected), f"{prefix}_e{i}")
                for i, (dims, expected) in enumerate(outputs)]
        records.append('{' + f'{provider},"{capability}","{operation}",{len(shape)},'
                       + '{' + ','.join(map(str, shape)) + '},'
                       + f'{len(inputs)},{len(outputs)},' + '{' + ','.join(ins) + '},'
                       + '{' + ','.join(outs) + '}' + '}')
    lines.append("static test_case cases[] = {\n" + ",\n".join(records) + "\n};")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    make_cases()
    Path(sys.argv[1]).write_text(header())
    print(f"CPU4 independent reference: {len(CASES)} forward/VJP cases; central differences step=1e-5")
