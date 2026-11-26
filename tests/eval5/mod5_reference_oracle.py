"""Development oracle for the fixed V=16 MOD5/D1 EVAL5 fixture; never publishes a manifest."""

import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/io5"), str(ROOT / "tests/d1")]
import model_mapping
import reference_format
import safetensors_interop as safe

FINGERPRINT = "eval5-mod5-token-v1"
SHAPES = ((8, 16), (16, 16), (16, 16), (8, 16), (16, 32),
          (32, 16), (32, 16), (16,), (16,), (16, 16), (16,), (16, 16))
PATHS = (
    "blocks.0.attention.key.weight", "blocks.0.attention.output.weight",
    "blocks.0.attention.query.weight", "blocks.0.attention.value.weight",
    "blocks.0.ffn.down.weight", "blocks.0.ffn.gate.weight",
    "blocks.0.ffn.up.weight", "blocks.0.norm1.weight",
    "blocks.0.norm2.weight", "head.weight", "norm_final.weight",
    "token_embedding.weight",
)


def weights_from_c1(raw):
    state = safe.decode_c1(raw)
    model_mapping.mapped_names(state, model_mapping.MOD5, "huggingface")
    if state.provider != "i2-dense-cpu-f32-v1" or state.aliases not in ((), ((9, 11),)):
        raise ValueError("unexpected MOD5 provider or alias topology")
    if tuple(".".join(t.path) for t in state.tensors) != PATHS:
        raise ValueError("unexpected MOD5 parameter roster")
    if tuple(t.shape for t in state.tensors) != SHAPES:
        raise ValueError("unexpected MOD5 parameter shapes")
    result = [struct.unpack("<" + "f" * (len(t.data) // 4), t.data)
              for t in state.tensors]
    if not all(math.isfinite(x) for tensor in result for x in tensor):
        raise ValueError("nonfinite MOD5 weight")
    return result


def project(rows, weight, output, input_width):
    return [[math.fsum(row[d] * weight[o * input_width + d]
                       for d in range(input_width)) for o in range(output)]
            for row in rows]


def rms(rows, gamma):
    return [[x * gamma[d] / math.sqrt(0.125 + math.fsum(v * v for v in row) / 16)
             for d, x in enumerate(row)] for row in rows]


def attention(q, k, v):
    rows = []
    for t in range(8):
        merged = []
        for head in range(4):
            kv = head // 2
            scores = [math.fsum(q[t][head * 4 + d] * k[j][kv * 4 + d]
                                for d in range(4)) / 2 for j in range(t + 1)]
            peak = max(scores)
            exp = [math.exp(s - peak) for s in scores]
            denom = math.fsum(exp)
            merged.extend(math.fsum(exp[j] * v[j][kv * 4 + d]
                                    for j in range(t + 1)) / denom for d in range(4))
        rows.append(merged)
    return rows


def logits(weights, context):
    if len(context) != 8 or any(type(t) is not int or not 0 <= t < 16 for t in context):
        raise ValueError("expected eight V=16 token IDs")
    x = [list(weights[11][token * 16:(token + 1) * 16]) for token in context]
    norm = rms(x, weights[7])
    q = project(norm, weights[2], 16, 16)
    k = project(norm, weights[0], 8, 16)
    v = project(norm, weights[3], 8, 16)
    attended = project(attention(q, k, v), weights[1], 16, 16)
    residual = [[a + b for a, b in zip(left, right)]
                for left, right in zip(x, attended)]
    norm = rms(residual, weights[8])
    gate = project(norm, weights[5], 32, 16)
    up = project(norm, weights[6], 32, 16)
    hidden = [[(g / (1 + math.exp(-g))) * u for g, u in zip(gr, ur)]
              for gr, ur in zip(gate, up)]
    down = project(hidden, weights[4], 16, 32)
    block = [[a + b for a, b in zip(left, right)]
             for left, right in zip(residual, down)]
    return project(rms(block, weights[10]), weights[9], 16, 16)


def nll(rows, targets):
    def row_loss(row, target):
        peak = max(row)
        return peak - row[target] + math.log(math.fsum(math.exp(x - peak) for x in row))
    return math.fsum(row_loss(row, target) for row, target in zip(rows, targets, strict=True))


def documents(corpus):
    if (corpus["fingerprint"] != FINGERPRINT or corpus["vocab_size"] != 16
            or corpus["shard_count"] != 1 or len(corpus["tokens"]) != 18):
        raise ValueError("unexpected D1 fixture identity or shape")
    tokens = corpus["tokens"]
    if any(type(t) is not int or not 0 <= t < 16 for t in tokens):
        raise ValueError("D1 token outside model vocabulary")
    return [tuple(tokens[i:i + 9]) for i in (0, 9)]


def audit(train, evaluation):
    train_bytes = {bytes("".join(f"{t:02x}" for t in doc), "ascii") for doc in train}
    grams = {doc[i:i + 3] for doc in train for i in range(7)}
    exact = overlap = matched = contaminated = 0
    for doc in evaluation:
        raw = bytes("".join(f"{t:02x}" for t in doc), "ascii")
        e = raw in train_bytes
        count = sum(doc[i:i + 3] in grams for i in range(7))
        n = count > 0 and count / 7 >= 0.25
        exact += e
        overlap += n
        matched += count
        contaminated += e or n
    return {"eval_documents": 2, "contaminated_documents": contaminated,
            "exact_doc_matches": exact, "ngram_contaminated_docs": overlap,
            "matched_eval_ngrams": matched, "total_eval_ngrams": 14}


def evaluate(weights, train, evaluation):
    counts = audit(train, evaluation)
    if counts["contaminated_documents"]:
        return {"audit": counts, "status": "contaminated"}
    outputs = [logits(weights, doc[:8]) for doc in evaluation]
    total = math.fsum(nll(rows, doc[1:]) for rows, doc in zip(outputs, evaluation))
    choices = (
        ((outputs[0], evaluation[0][7:]), (outputs[0], (1, 4)), 0),
        ((outputs[1], (4, 1)), (outputs[1], evaluation[1][7:]), 1),
    )
    a = logits(weights, (10, 11, 12, 13, 14, 15, 0, 3))
    b = logits(weights, (3, 0, 15, 14, 13, 12, 11, 10))
    choices += (((a, (10, 11)), (b, (12, 13)), 0),
                ((b, (14, 15)), (a, (3, 0)), 1))
    correct = sum(min(range(2), key=lambda i: nll(item[i][0][6:8], item[i][1]) / 2)
                  == item[2] for item in choices)
    return {"audit": counts, "status": "clean", "lm": {
        "token_count": 16, "byte_count": 32, "total_nll": total,
        "mean_nll": total / 16, "perplexity": math.exp(total / 16),
        "bits_per_byte": total / (math.log(2) * 32)},
        "mc": {"correct_count": correct, "total_count": 4, "accuracy": correct / 4}}


def main(argv):
    if len(argv) != 4:
        raise SystemExit("usage: mod5_reference_oracle.py CHECKPOINT TRAIN_D1 EVAL_D1")
    weights = weights_from_c1(Path(argv[1]).read_bytes())
    train = documents(reference_format.parse_corpus(Path(argv[2])))
    evaluation = documents(reference_format.parse_corpus(Path(argv[3])))
    print(json.dumps(evaluate(weights, train, evaluation), sort_keys=True))


if __name__ == "__main__":
    main(sys.argv)
