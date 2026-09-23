"""Pinned mathematical M3/CE reference and labeled synthetic f32 transcript.

The synthetic observation exercises the standalone comparator. It is not output
from E3, a native test hook, or evidence that the evaluator exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

from tests.e3_metrics.reference_probe import accumulate, divide, finalize, reduce_bool
from tests.e3_reference.corpus import validate_payload as validate_corpus_payload
from tests.e3_reference.transcript import (
    MAX_BYTES, canonical, decode, encode, f32_bits, f64_bits,
    validate_mathematical_case, validate_observed_case,
)
from tests.m3 import reference as m3


torch = m3.torch
functional = torch.nn.functional
ROOT = Path(__file__).resolve().parents[2]
ROLE_NAMES = (
    "ET", "EP", "X", "N1", "QT", "KT", "VT", "QH", "KH", "VH", "AH",
    "AT", "AO", "R", "N2", "FU", "FG", "FD", "Y", "NF", "Z",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role_trace(parameters: dict[str, torch.Tensor], tokens: tuple[int, int]):
    """Direct accepted equations with named development observations."""

    p = parameters
    block = "blocks/0/"

    def norm(value, prefix):
        centered = value - value.mean(dim=-1, keepdim=True)
        scaled = centered / torch.sqrt(
            centered.square().mean(dim=-1, keepdim=True) + m3.EPSILON
        )
        return scaled * p[prefix + "/weight"] + p[prefix + "/bias"]

    et = p[m3.TIED][list(tokens)].unsqueeze(0)
    ep = p["position_embedding/weight"].unsqueeze(0)
    x = et + ep
    n1 = norm(x, block + "norm1")
    qt = n1 @ p[block + "attention/query/weight"].T
    kt = n1 @ p[block + "attention/key/weight"].T
    vt = n1 @ p[block + "attention/value/weight"].T
    qh = qt.reshape(1, 2, 2, 2).transpose(1, 2)
    kh = kt.reshape(1, 2, 2, 2).transpose(1, 2)
    vh = vt.reshape(1, 2, 2, 2).transpose(1, 2)
    scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(2)
    scores = scores.masked_fill(
        torch.tensor([[False, True], [False, False]]), -math.inf
    )
    ah = torch.softmax(scores, dim=-1) @ vh
    at = ah.transpose(1, 2).reshape(1, 2, 4)
    ao = at @ p[block + "attention/output/weight"].T
    residual = x + ao
    n2 = norm(residual, block + "norm2")
    fu = n2 @ p[block + "ffn/up/weight"].T
    fg = 0.5 * fu * (1 + torch.erf(fu / math.sqrt(2)))
    fd = fg @ p[block + "ffn/down/weight"].T
    y = residual + fd
    nf = norm(y, "norm_final")
    z = nf @ p[m3.TIED].T
    values = (
        et, ep, x, n1, qt, kt, vt, qh, kh, vh, ah, at, ao, residual,
        n2, fu, fg, fd, y, nf, z,
    )
    if not torch.equal(z, m3.forward(p, tokens)):
        raise AssertionError("development role trace differs from accepted M3 forward")
    return tuple(zip(ROLE_NAMES, values))


def _roles(trace) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "values_f64": [f64_bits(item) for item in value.detach().reshape(-1).tolist()],
        }
        for name, value in trace
    ]


def _small_f32(value: int) -> int:
    return int(f32_bits(float(value)), 16)


def _synthetic_case(
    batches: list[dict[str, object]], parameters: dict[str, torch.Tensor]
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    state = (0, 0, 0, 0, 0)
    observed = []
    mathematical = []
    active_mathematical_ce: list[float] = []
    for ordinal, batch in enumerate(batches):
        inputs = tuple(batch["inputs"])
        targets = tuple(batch["targets"])
        mask = tuple(batch["mask"])
        trace = role_trace(parameters, inputs)
        logits64 = trace[-1][1]
        logits32 = logits64.to(torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.int64)
        ce64 = functional.cross_entropy(
            logits64.reshape(2, 256), target_tensor, reduction="none"
        )
        ce32 = functional.cross_entropy(
            logits32.reshape(2, 256), target_tensor, reduction="none"
        )
        ce_words = tuple(int(f32_bits(item), 16) for item in ce32.tolist())
        numerator, weight = reduce_bool(ce_words, mask)
        winners = logits32.reshape(2, 256).argmax(-1).tolist()
        correct_count = sum(
            int(mask[position] and winners[position] == targets[position])
            for position in range(2)
        )
        correct = _small_f32(correct_count)
        state = accumulate(state, (numerator, weight, correct, sum(mask)))
        observed.append(
            {
                "ce_f32": [f"{word:08x}" for word in ce_words],
                "correct_f32": f"{correct:08x}",
                "counters": list(state[3:]),
                "inputs": list(inputs),
                "logits_f32": [f32_bits(item) for item in logits32.reshape(-1).tolist()],
                "mask": list(mask),
                "ordinal": ordinal,
                "reduction_f32": [
                    f"{numerator:08x}", f"{weight:08x}",
                    f"{divide(numerator, weight):08x}",
                ],
                "state_f32": [f"{word:08x}" for word in state[:3]],
                "targets": list(targets),
            }
        )
        mathematical.append(
            {
                "ce_f64": [f64_bits(item) for item in ce64.tolist()],
                "inputs": list(inputs),
                "mask": list(mask),
                "ordinal": ordinal,
                "roles": _roles(trace),
                "targets": list(targets),
            }
        )
        active_mathematical_ce.extend(
            ce64[position].item() for position in range(2) if mask[position]
        )
    final_words = finalize(state)
    return (
        {
            "batches": observed,
            "final": {
                "counters": list(state[3:]),
                "metrics_f32": [f"{word:08x}" for word in final_words],
            },
            "kind": "synthetic-observation",
            "profile": "e3-private-n1-t2-v256-d4-v1",
        },
        mathematical,
        f64_bits(math.fsum(active_mathematical_ce) / len(active_mathematical_ce)),
    )


def build(corpus_payload: dict[str, object]) -> dict[str, object]:
    validate_corpus_payload(corpus_payload)
    m3.setup()
    parameters, successor = m3.parameters(1729)
    cases = []
    for corpus_case in corpus_payload["cases"]:
        synthetic, mathematical, mathematical_loss = _synthetic_case(
            corpus_case["batches"], parameters
        )
        cases.append(
            {
                "corpus": corpus_case,
                "mathematical": mathematical,
                "mathematical_loss_f64": mathematical_loss,
                "name": corpus_case["name"],
                "synthetic": synthetic,
            }
        )
    parameter_words = b"".join(
        bytes.fromhex(f32_bits(item))
        for path, _ in m3.PARAMETERS
        for item in parameters[path].to(torch.float32).detach().reshape(-1).tolist()
    )
    result = {
        "cases": cases,
        "kind": "reference-bundle",
        "model": {
            "initializer_seed": 1729,
            "initializer_successor": list(successor),
            "parameter_f32_sha256": hashlib.sha256(parameter_words).hexdigest(),
            "reference_manifest_sha256": _digest(ROOT / "tests/m3/reference_manifest.json"),
        },
        "profile": m3.PROFILE,
        "provenance": {
            "corpus_source_sha256": _digest(ROOT / "tests/e3_reference/corpus.py"),
            "generator_source_sha256": _digest(Path(__file__)),
            "m3_reference_source_sha256": _digest(ROOT / "tests/m3/reference.py"),
            "initializer_source_sha256": _digest(
                ROOT / "tests/n3k/test_initializer_reference.py"
            ),
            "python": platform.python_version(),
            "q0_lock_sha256": _digest(ROOT / "tests/q0/requirements-oracle.lock"),
            "rational_reference_source_sha256": _digest(
                ROOT / "tests/e3_metrics/reference_probe.py"
            ),
            "torch": torch.__version__,
        },
    }
    validate_bundle(result)
    return result


def validate_bundle(payload: dict[str, object]) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "cases", "kind", "model", "profile", "provenance"
    }:
        raise ValueError("reference bundle fields differ from version 1")
    if payload["kind"] != "reference-bundle" or payload["profile"] != m3.PROFILE:
        raise ValueError("reference bundle kind/profile mismatch")
    cases = payload["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 16:
        raise ValueError("reference bundle case count mismatch")
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "corpus", "mathematical", "mathematical_loss_f64", "name", "synthetic"
        }:
            raise ValueError("reference bundle case fields differ from version 1")
        if case["name"] != case["corpus"].get("name"):
            raise ValueError("reference bundle case identity mismatch")
        expected = tuple(case["corpus"]["batches"])
        validate_observed_case(case["synthetic"], expected)
        validate_mathematical_case(
            case["synthetic"], case["mathematical"], expected,
            case["mathematical_loss_f64"],
        )
    if not isinstance(payload["model"], dict) or set(payload["model"]) != {
        "initializer_seed", "initializer_successor", "parameter_f32_sha256",
        "reference_manifest_sha256",
    }:
        raise ValueError("reference bundle model fields differ from version 1")
    if not isinstance(payload["provenance"], dict) or set(payload["provenance"]) != {
        "corpus_source_sha256", "generator_source_sha256",
        "initializer_source_sha256", "m3_reference_source_sha256", "python",
        "q0_lock_sha256", "rational_reference_source_sha256", "torch",
    }:
        raise ValueError("reference bundle provenance fields differ from version 1")


def _read_limited(path: Path) -> bytes:
    if path.is_symlink() or path.parent.resolve() != path.parent.absolute():
        raise ValueError("corpus reference path must not be a symlink")
    size = path.stat().st_size
    if not 1 <= size <= MAX_BYTES:
        raise ValueError("corpus reference exceeds the transcript byte limit")
    return path.read_bytes()


def _write_exclusive(path: Path, raw: bytes) -> None:
    ancestor = path.parent
    while not ancestor.exists():
        if ancestor.is_symlink():
            raise ValueError("reference output ancestors must not be symlinks")
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve() != ancestor.absolute():
        raise ValueError("reference output ancestors must not be symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    corpus_payload = decode(_read_limited(arguments.corpus_reference))
    _write_exclusive(arguments.output, encode(build(corpus_payload)))


if __name__ == "__main__":
    main()
