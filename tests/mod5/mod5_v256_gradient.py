"""Prepare and verify the bounded MOD5-C V256 gradient witness."""

from dataclasses import replace
import json
import math
from pathlib import Path
import sys

from tests.eval5 import mod5_reference_oracle as oracle, mod5_v256_fixture as fixture
from tests.eval5.build_mod5_v256_witness import OWNER, digest_literal, generate, one


HEAD, EMBEDDING = 9, 11
STEP = 1 / 512
PROBES = {
    "head": ((3, 0), (201, 12)),
    "embedding": ((15, 15), (200, 5)),
    "tied": ((3, 0), (15, 15), (200, 0)),
}
TAIL = Path(__file__).with_name("mod5_v256_gradient_tail.esk")


def section(text, start, end):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"MOD5 source boundary drift: {start!r} / {end!r}")
    return text[text.index(start):text.index(end)]


def fresh_write(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def tied_model(raw):
    state = oracle.safe.decode_c1(raw)
    if state.aliases or tuple(tensor.shape for tensor in state.tensors) != oracle.SHAPES_256:
        raise ValueError("expected the canonical untied V256 roster")
    tensors = list(state.tensors)
    tensors[HEAD] = replace(tensors[HEAD], data=tensors[EMBEDDING].data)
    return oracle.safe.encode_c1(replace(
        state, tensors=tuple(tensors), aliases=((HEAD, EMBEDDING),)))


def source_for(fixture_root, tied_sha):
    generated = generate(fixture_root)
    marker = ";;; Test-only V=256 EVAL5 score witness."
    if generated.count(marker) != 1:
        raise ValueError("EVAL5 specialization boundary drift")
    prefix, eval_tail = generated.split(marker, 1)
    checked_inputs = section(eval_tail, "(define (check-tokenizer ", "(define (audit-splits ")
    one_hot = section(eval_tail, "(define (one-hot-doc ", "(define (model-logits ")
    loader = section(eval_tail, "(define (load-checked-weights ", "(define (copy-digest ")
    loader = one(loader, "(define (load-checked-weights path)",
                 "(define (load-checked-weights path expected-sha)")
    loader = one(loader, "expected-model-sha", "expected-sha")
    pins = section(eval_tail, "(define expected-model-sha ", "(define (check-tokenizer ")
    cleanup = section(OWNER.read_text(), "(define (close-workspace)", "(define (mod5-e2e-main ")
    tail = one(TAIL.read_text(), "@@TIED_SHA@@", digest_literal(tied_sha))
    for kind, probes in PROBES.items():
        positions = "(" + " ".join(f"({row} {col})" for row, col in probes) + ")"
        tail = one(tail, f"@@{kind.upper()}_PROBES@@", positions)
    return prefix + pins + checked_inputs + one_hot + loader + cleanup + tail


def loss(weights, document):
    rows = oracle.logits(weights, tuple(document[:-1]))
    losses = []
    for row, target in zip(rows, document[1:], strict=True):
        peak = max(row)
        losses.append(peak - row[target] + math.log(math.fsum(
            math.exp(value - peak) for value in row)))
    return math.fsum(losses) / 8


def derivative(weights, document, row, col, *slots):
    index = row * 16 + col

    def perturb(sign):
        changed = weights.copy()
        for slot in slots:
            values = weights[slot]
            changed[slot] = values[:index] + (values[index] + sign * STEP,) + values[index + 1:]
        return loss(changed, document)

    return (perturb(1) - perturb(-1)) / (2 * STEP)


def reference(directory):
    root = directory / "fixture"
    record = fixture.verify(root)
    raw = (root / "model.etcp").read_bytes()
    tied = (directory / "tied.etcp").read_bytes()
    if tied != tied_model(raw):
        raise ValueError("wrong tied V256 fixture identity")
    weights = oracle.weights_from_c1(raw)
    shared = oracle.weights_from_c1(tied)
    document = fixture.SOURCE["eval-clean"][0]
    probes = {}
    for kind, positions in PROBES.items():
        current = shared if kind == "tied" else weights
        slots = {"head": (HEAD,), "embedding": (EMBEDDING,), "tied": (HEAD, EMBEDDING)}[kind]
        for row, col in positions:
            value = derivative(current, document, row, col, *slots)
            head = derivative(current, document, row, col, HEAD)
            embedding = derivative(current, document, row, col, EMBEDDING)
            if abs(value) < 0.001 or (kind == "tied" and min(abs(head), abs(embedding)) < 0.0005):
                raise ValueError(f"degenerate {kind} probe {row},{col}")
            probes[f"{kind} {row} {col}"] = [value, value, head, embedding]
    return {
        "mask": "11111111",
        "untied_loss": loss(weights, document), "tied_loss": loss(shared, document),
        "untied_sha256": record["checkpoint_sha256"], "tied_sha256": fixture.sha(tied),
        "probes": probes,
    }


def build(directory):
    fixture.build(directory / "fixture")
    raw = (directory / "fixture/model.etcp").read_bytes()
    tied = tied_model(raw)
    fresh_write(directory / "tied.etcp", tied)
    expected = reference(directory)
    fresh_write(directory / "reference.json", fixture.canonical(expected))
    fresh_write(directory / "witness.esk", source_for(
        directory / "fixture", expected["tied_sha256"]).encode())


def close(observed, expected, absolute, relative):
    return math.isfinite(observed) and abs(observed - expected) <= absolute + relative * abs(expected)


def verify(directory, output):
    expected = reference(directory)
    if (directory / "reference.json").read_bytes() != fixture.canonical(expected):
        raise ValueError("reference record changed")
    lines = output.read_text().splitlines()
    if (len(lines) != 5 + len(expected["probes"]) or lines[0] != "MASK 11111111"
            or lines[-2:] != ["TIED_CANONICAL_HEAD 4096", "MOD5_V256_GRADIENT_PASS"]):
        raise ValueError("unexpected gradient transcript envelope")
    for line, mode in zip(lines[1:3], ("untied", "tied"), strict=True):
        prefix, value = line.rsplit(" ", 1)
        if prefix != f"LOSS {mode}" or not close(float(value), expected[f"{mode}_loss"], 0.000002, 0):
            raise ValueError(f"bad {mode} mean loss")
    for line, (probe, reference_values) in zip(lines[3:-2], expected["probes"].items(), strict=True):
        words = line.split(" ")
        if len(words) != 8 or " ".join(words[:4]) != f"GRAD {probe}":
            raise ValueError(f"unexpected gradient probe {line!r}")
        observed = tuple(map(float, words[4:]))
        if not close(observed[0], observed[1], 0.000015, 0.005):
            raise ValueError(f"live analytic/finite-difference mismatch for {probe}")
        if not all(close(value, target, 0.00005, 0.01)
                   for value, target in zip(observed, reference_values, strict=True)):
            raise ValueError(f"independent gradient mismatch for {probe}")
    print(f"MOD5 V256 full-mask gradient oracle: {len(expected['probes'])} probes passed")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "build":
        build(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "verify":
        verify(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("usage: python3 -m tests.mod5.mod5_v256_gradient build DIR | verify DIR OUTPUT")
