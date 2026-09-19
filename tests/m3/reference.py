"""Development-only complete M3 mathematical reference with reviewed manifest.

No production role tables, native kernels, or model schedules are imported.
PyTorch expresses the graph directly; the independently tested integer Philox
oracle supplies exact initializer bits. This is not a production runtime or a
Q0 record. Generate records under build/ or a temporary directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

PINNED_PYTHON = (3, 14, 6)
PINNED_TORCH = "2.13.0+cpu"
if sys.version_info[:3] != PINNED_PYTHON:
    raise RuntimeError("M3 reference requires Python 3.14.6; see tests/m3/README.md")
for name, expected in (("ATEN_CPU_CAPABILITY", "default"), ("MKL_CBWR", "COMPATIBLE")):
    if os.environ.get(name) != expected:
        raise RuntimeError(f"M3 reference requires explicit {name}={expected} before torch import")
import torch
if torch.__version__ != PINNED_TORCH:
    raise RuntimeError("M3 reference requires PyTorch 2.13.0+cpu; see tests/q0/requirements-oracle.lock")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.n3k.test_initializer_reference import initialize, signed


# Explicit model contract, deliberately independent of native role descriptors.
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
    ("position_embedding/weight", (2, 4)),
)
TIED = "head/weight"
EPSILON = struct.unpack("<f", struct.pack("<I", 0x3727C5AC))[0]
PROFILE = {"identity": "eshkol-diagnostic-byte-decoder-v1", "N": 1, "T": 2,
           "V": 256, "D": 4, "Hq": 2, "Hkv": 2, "Dh": 2, "L": 1, "F": 8,
           "device": "cpu", "parameter_dtype": "f32", "compute_dtype": "f64",
           "positions": "learned", "normalization": "pre-LN/final-LN",
           "epsilon_bits": "3727c5ac", "activation": "exact-erf-gelu",
           "projection_bias": False, "tied_head": True, "dropout": False, "cache": False}


def setup() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def parameters(seed: int = 1729, counter: int = 0, *, conditioned: bool = False,
               dtype: torch.dtype = torch.float64):
    if type(seed) is not int or not 0 <= seed < 1 << 63:
        raise ValueError("seed must be a nonnegative exact signed-i64")
    if type(counter) is not int or not 0 <= counter < (1 << 128) - 290:
        raise ValueError("counter lacks complete model capacity")
    state = (1, seed, signed(counter & ((1 << 64) - 1)), signed(counter >> 64))
    result = {}
    for ordinal, (path, shape) in enumerate(PARAMETERS):
        if len(shape) == 2:
            bits, state = initialize(shape, state)
            values = [struct.unpack("<f", struct.pack("<I", bit))[0] for bit in bits]
        else:
            values = [1.0 if path.endswith("weight") else 0.0] * shape[0]
        if conditioned:
            # A test-only strict-state-load fixture, not another initializer.
            values = [((i * 37 + ordinal * 19) % 127 - 63) / 89
                      for i in range(math.prod(shape))]
            if len(shape) == 1 and path.endswith("weight"):
                values = [value + 1.0 for value in values]
        result[path] = torch.tensor(values, dtype=torch.float32).to(dtype).reshape(shape).requires_grad_()
    return result, state


def upstream(salt: int = 1, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor([((i * 29 + salt * 17) % 101 - 50) / 64
                         for i in range(512)], dtype=dtype).reshape(1, 2, 256)


def _tensor(value, shape, dtype, label):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{label}: expected tensor shape {shape}")
    if value.dtype != dtype or value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError(f"{label}: expected common CPU {dtype} dense tensor")
    if not value.is_contiguous() or value.storage_offset() != 0:
        raise ValueError(f"{label}: expected contiguous zero-offset tensor")
    if not torch.isfinite(value.detach()).all().item():
        raise ValueError(f"{label}: expected finite values")


def _parameters(p):
    if not isinstance(p, dict) or set(p) != {path for path, _ in PARAMETERS}:
        raise ValueError("expected exactly the fourteen canonical parameter paths")
    dtype = getattr(p[TIED], "dtype", None)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("expected CPU f32 or f64 reference parameters")
    for path, shape in PARAMETERS:
        _tensor(p[path], shape, dtype, path)
    return dtype


def _tokens(tokens):
    if len(tokens) != 2 or any(type(t) is not int or not 0 <= t < 256 for t in tokens):
        raise ValueError("expected two exact byte-token IDs")


def forward(p: dict[str, torch.Tensor], tokens=(3, 197), *,
            embedding_weight=None, head_weight=None, causal=True,
            split_heads=True, swap_qk=False) -> torch.Tensor:
    """Validated mathematical model with test-only wiring mutation switches."""
    dtype = _parameters(p)
    _tokens(tokens)
    for label, value in (("embedding edge", embedding_weight), ("head edge", head_weight)):
        if value is not None:
            _tensor(value, (256, 4), dtype, label)
    return _equations(p, tokens, embedding_weight=embedding_weight, head_weight=head_weight,
                      causal=causal, split_heads=split_heads, swap_qk=swap_qk)


def _equations(p, tokens, *, embedding_weight=None, head_weight=None,
               causal=True, split_heads=True, swap_qk=False):
    """Direct tensor equations; callers validate exact operands first."""
    embedding_weight = p[TIED] if embedding_weight is None else embedding_weight
    head_weight = p[TIED] if head_weight is None else head_weight
    block = "blocks/0/"

    def norm(x, prefix):
        centered = x - x.mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + EPSILON)
        return normalized * p[prefix + "/weight"] + p[prefix + "/bias"]

    x = embedding_weight[list(tokens)].unsqueeze(0) + p["position_embedding/weight"].unsqueeze(0)
    n = norm(x, block + "norm1")
    projected = [n @ p[block + "attention/" + role + "/weight"].T
                 for role in (("key", "query", "value") if swap_qk else ("query", "key", "value"))]
    heads = [value.reshape(1, 2, 2, 2).transpose(1, 2) if split_heads
             else value.reshape(1, 2, 2, 2) for value in projected]
    q, k, v = heads
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(2)
    if causal:
        scores = scores.masked_fill(torch.tensor([[False, True], [False, False]]), -math.inf)
    attended = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(1, 2, 4)
    r = x + attended @ p[block + "attention/output/weight"].T
    u = norm(r, block + "norm2") @ p[block + "ffn/up/weight"].T
    activated = 0.5 * u * (1 + torch.erf(u / math.sqrt(2)))
    y = r + activated @ p[block + "ffn/down/weight"].T
    return norm(y, "norm_final") @ head_weight.T


def vjp(p, tokens=(3, 197), seed=None):
    dtype = _parameters(p)
    _tokens(tokens)
    seed = upstream(dtype=dtype) if seed is None else seed
    _tensor(seed, (1, 2, 256), dtype, "upstream")
    if not all(value.requires_grad for value in p.values()):
        raise ValueError("VJP parameters must require gradients")
    logits = _equations(p, tokens)
    values = torch.autograd.grad((logits * seed).sum(), tuple(p.values()))
    embedding = p[TIED].detach().clone().requires_grad_()
    head = p[TIED].detach().clone().requires_grad_()
    untied = _equations(p, tokens, embedding_weight=embedding, head_weight=head)
    edges = torch.autograd.grad((untied * seed).sum(), (head, embedding))
    return logits.detach(), dict(zip(p, values)), {"head": edges[0], "embedding": edges[1]}


def finite_difference(p, path, tokens, seed, relative_step=1e-5):
    dtype = _parameters(p)
    _tokens(tokens)
    _tensor(seed, (1, 2, 256), dtype, "upstream")
    if not math.isfinite(relative_step) or relative_step <= 0:
        raise ValueError("finite-difference step must be positive and finite")
    flat = p[path].view(-1)
    result = torch.empty_like(flat)
    with torch.no_grad():
        for index in range(flat.numel()):
            old = flat[index].item()
            step = relative_step * max(1, abs(old))
            try:
                flat[index] = old + step
                plus = (_equations(p, tokens) * seed).sum().item()
                flat[index] = old - step
                minus = (_equations(p, tokens) * seed).sum().item()
            finally:
                flat[index] = old
            result[index] = (plus - minus) / (2 * step)
    return result.reshape(p[path].shape)


def record(seed, tokens, salt, conditioned=False):
    p, successor = parameters(seed, conditioned=conditioned)
    seed_tensor = upstream(salt)
    logits, gradients, edges = vjp(p, tokens, seed_tensor)

    def tensor(value):
        return {"shape": list(value.shape), "values": value.detach().flatten().tolist()}

    return {"initializer_seed": seed, "initializer_successor": successor,
            "conditioned_state_fixture": conditioned, "tokens": tokens,
            "upstream": tensor(seed_tensor), "parameters": {k: tensor(v) for k, v in p.items()},
            "logits": tensor(logits), "unique_vjp": {k: tensor(v) for k, v in gradients.items()},
            "tied_edge_vjp": {k: tensor(v) for k, v in edges.items()}}


def document():
    setup()
    return {"format": "eshkol-m3-mathematical-reference", "version": 1, "profile": PROFILE,
               "torch": torch.__version__, "python": ".".join(map(str, PINNED_PYTHON)),
               "reference_environment": {name: os.environ[name] for name in ("ATEN_CPU_CAPABILITY", "MKL_CBWR")},
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__).resolve(), ROOT / "tests/n3k/test_initializer_reference.py",
                                            ROOT / "tests/q0/requirements-oracle.lock")},
               "logical_alias": {"token_embedding/weight": TIED},
               "cases": [record(1729, (3, 197), 1), record(0, (0, 255), 2),
                         record((1 << 63) - 1, (7, 7), 3), record(1729, (3, 197), 2, True)]}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def manifest(content, encoded):
    return {"format": "eshkol-m3-reference-manifest", "version": 1,
            "reference_format": content["format"], "reference_version": content["version"],
            "reference_sha256": hashlib.sha256(encoded).hexdigest(),
            "profile": content["profile"], "python": content["python"], "torch": content["torch"],
            "reference_environment": content["reference_environment"],
            "source_sha256": content["source_sha256"],
            "unique_parameter_count": 14, "logical_parameter_count": 15, "unique_value_count": 1184,
            "parameter_shapes": {path: list(shape) for path, shape in PARAMETERS},
            "logical_alias": content["logical_alias"],
            "cases": [{key: item[key] for key in ("initializer_seed", "initializer_successor", "tokens", "conditioned_state_fixture")}
                      | {"upstream_shape": item["upstream"]["shape"], "logits_shape": item["logits"]["shape"]}
                      for item in content["cases"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-manifest", type=Path, help="deliberately propose a manifest for independent review")
    args = parser.parse_args()
    content = document()
    encoded = canonical(content)
    expected = json.loads(canonical(manifest(content, encoded)))
    if args.candidate_manifest is not None:
        args.candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_manifest.write_bytes(canonical(expected))
    elif json.loads(Path(__file__).with_name("reference_manifest.json").read_text()) != expected:
        raise RuntimeError("M3 reference differs from its reviewed manifest; regenerate only for explicit independent review")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(expected["reference_sha256"])
