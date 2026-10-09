#!/usr/bin/env python3
"""Development-only independent G3-C4 prefix mathematics and transcript check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import torch
from tests.g3s import reference as sampler_reference

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SEAL = "ace96aee0bba7caa1ab38434aefbf730ad710ff6e24f009d32fdc49b7389b460"
SOURCE_PINS_SHA = "722ea76906c272dad59a3412bb33312787289baf56e1938667e605824d74288d"
MEMBER_PINS_SHA = "5d06779a7de7e06a94f1bfc64474f7c3f4eb1c6c020d58bcf3d63f20b8b1d164"
SAMPLER_REFERENCE_SHA = "dfef749ccf1aff32f268e2db7b981a993e2bd3023e27f0044aeb412fdbec53dd"
BASE_COMMIT = "8b4b7d2f892f1307c8df3646841733196f5aa0fc"
BASE_TREE = "69a2c63303632449eb44352acb1bdc71207f5c73"
BASE_ROADMAP_SHA = "7f8b7615bc3d06aede6231dee83e903997e759eab5540a027ec83db5e91f9032"
NEXT_ROADMAP_SHA = "5df226a94bc765b5f0fd6d8a2f76f11b09eb49fa5a19efb5f8b04daba2fa2f10"
ROADMAP_DELTA_SHA = "53651c815d8fd7a94377e052729f0c7f94f8538d9a4b0db045cc8bc703ccc9e2"
PATHS = (
    "blocks/0/attention/key/weight", "blocks/0/attention/output/weight",
    "blocks/0/attention/query/weight", "blocks/0/attention/value/weight",
    "blocks/0/ffn/down/weight", "blocks/0/ffn/up/weight",
    "blocks/0/norm1/bias", "blocks/0/norm1/weight",
    "blocks/0/norm2/bias", "blocks/0/norm2/weight",
    "head/weight", "norm_final/bias", "norm_final/weight",
    "position_embedding/weight",
)
SHAPES = ((4, 4), (4, 4), (4, 4), (4, 4), (4, 8), (8, 4),
          (4,), (4,), (4,), (4,), (256, 4), (4,), (4,), (4, 4))
PROMPTS = ((0,), (255,), (0, 0), (255, 255), (0, 255), (7, 11),
           (0, 255, 7), (7, 11, 0), (0, 255, 7, 11), (7, 11, 0, 255))
SEEDS = (0, 1729, (1 << 63) - 1)
U32 = (1 << 32) - 1
U64 = (1 << 64) - 1
ABS_TOL = 2e-6
REL_TOL = 2e-5


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pins(raw: bytes) -> list[tuple[str, str]]:
    rows = [line.split("  ", 1) for line in raw.decode("utf-8").splitlines()]
    if any(len(row) != 2 or len(row[0]) != 64 or
           any(c not in "0123456789abcdef" for c in row[0]) or
           not row[1] or row[1].startswith("/") or ".." in Path(row[1]).parts
           for row in rows):
        raise ValueError("invalid source pin row")
    if len(set(path for _, path in rows)) != len(rows):
        raise ValueError("duplicate source pin")
    return [(digest, path) for digest, path in rows]


def check_source_pins(overrides: dict[str, bytes] | None = None,
                      source_pins: bytes | None = None,
                      member_pins: bytes | None = None,
                      roadmap_basis: bytes | None = None,
                      roadmap_delta: bytes | None = None) -> None:
    """Fail before any fixture generation on changed production/source inputs."""
    source_pins = source_pins if source_pins is not None else (
        ROOT / "tests/g3c4/c4_math_source_pins.sha256").read_bytes()
    member_pins = member_pins if member_pins is not None else (
        ROOT / "tests/g3c4/c4_math_closure_members.sha256").read_bytes()
    if _sha(source_pins) != SOURCE_PINS_SHA or _sha(member_pins) != MEMBER_PINS_SHA:
        raise ValueError("pin manifest drift")
    sources, members = _pins(source_pins), _pins(member_pins)
    if len(sources) != 13 or len(members) != 61:
        raise ValueError("pin count drift")
    sampler_path = "tests/g3s/reference.py"
    sampler_bytes = (overrides or {}).get(sampler_path,
                                          (ROOT / sampler_path).read_bytes())
    if _sha(sampler_bytes) != SAMPLER_REFERENCE_SHA:
        raise ValueError("sampler reference drift")
    closure_path = "native/g3c4_full_prefix_forward_source_closure.txt"
    closure = (overrides or {}).get(closure_path,
                                    (ROOT / closure_path).read_bytes())
    if closure.decode("utf-8").splitlines() != [path for _, path in members]:
        raise ValueError("closure order or member drift")
    roadmap_path = "docs/ROADMAP.md"
    pinned = dict((path, digest) for digest, path in members)
    if pinned[roadmap_path] != BASE_ROADMAP_SHA:
        raise ValueError("original ROADMAP member pin drift")
    actual_tree = subprocess.run(
        ["git", "--no-optional-locks", "rev-parse", f"{BASE_COMMIT}^{{tree}}"],
        cwd=ROOT, check=True, capture_output=True, timeout=30).stdout.decode().strip()
    if actual_tree != BASE_TREE:
        raise ValueError("proof-B Git tree drift")
    if roadmap_basis is None:
        roadmap_basis = subprocess.run(
            ["git", "--no-optional-locks", "show",
             f"{BASE_COMMIT}:{roadmap_path}"], cwd=ROOT, check=True,
            capture_output=True, timeout=30).stdout
    if _sha(roadmap_basis) != BASE_ROADMAP_SHA:
        raise ValueError("original ROADMAP Git blob drift")
    actual_roadmap = (overrides or {}).get(
        roadmap_path, (ROOT / roadmap_path).read_bytes())
    if _sha(actual_roadmap) != NEXT_ROADMAP_SHA:
        raise ValueError("successor ROADMAP drift")
    encoded_delta = (ROOT / "tests/g3c4/c4_math_roadmap_delta.hex").read_text()
    try:
        expected_delta = bytes.fromhex(encoded_delta)
    except ValueError as error:
        raise ValueError("approved ROADMAP delta encoding drift") from error
    if encoded_delta != expected_delta.hex() + "\n":
        raise ValueError("approved ROADMAP delta encoding drift")
    if _sha(expected_delta) != ROADMAP_DELTA_SHA:
        raise ValueError("approved ROADMAP delta pin drift")
    if roadmap_delta is None:
        roadmap_delta = subprocess.run(
            ["git", "--no-optional-locks", "diff", "--no-ext-diff",
             BASE_COMMIT, "--", roadmap_path], cwd=ROOT, check=True,
            capture_output=True, timeout=30).stdout
    if roadmap_delta != expected_delta:
        raise ValueError("successor ROADMAP diff drift")
    for digest, path in sources + members:
        if path == roadmap_path:
            continue
        data = (overrides or {}).get(path)
        if data is None:
            data = (ROOT / path).read_bytes()
        if _sha(data) != digest:
            raise ValueError(f"source drift: {path}")


def require_q0() -> None:
    if sys.version_info[:3] != (3, 14, 6) or torch.__version__ != "2.13.0+cpu":
        raise RuntimeError("Q0 Python 3.14.6 / Torch 2.13.0+cpu required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def check_schedule(paths: tuple[str, ...], shapes: tuple[tuple[int, ...], ...],
                   token_alias: str) -> None:
    if (paths != PATHS or shapes != SHAPES or token_alias != PATHS[10] or
        len(paths) != 14 or len(set(paths)) != 14 or
        sum(math.prod(shape) for shape in shapes) != 1192):
        raise ValueError("canonical parameter schedule drift")


def philox(counter: int, key: int) -> tuple[int, int, int, int]:
    """Integer Philox4x32-10; no production or other oracle import."""
    if not 0 <= counter < 1 << 128 or not 0 <= key <= U64:
        raise ValueError("Philox counter/key range")
    x = [(counter >> shift) & U32 for shift in (0, 32, 64, 96)]
    k0, k1 = key & U32, key >> 32
    for turn in range(10):
        left = x[0] * 0xD2511F53
        right = x[2] * 0xCD9E8D57
        x = [((right >> 32) ^ x[1] ^ k0) & U32, right & U32,
             ((left >> 32) ^ x[3] ^ k1) & U32, left & U32]
        if turn != 9:
            k0 = (k0 + 0x9E3779B9) & U32
            k1 = (k1 + 0xBB67AE85) & U32
    return tuple(x)  # type: ignore[return-value]


def weight_bits(lane: int) -> int:
    """Exact binary32 bits of ((lane>>8)*2^-24 - 1/2)*2^-4."""
    if not 0 <= lane <= U32:
        raise ValueError("initializer lane range")
    numerator = (lane >> 8) - (1 << 23)
    if numerator == 0:
        return 0
    magnitude = abs(numerator)
    lead = magnitude.bit_length() - 1
    return ((0x80000000 if numerator < 0 else 0) |
            ((lead + 99) << 23) |
            ((magnitude << (23 - lead)) & 0x7FFFFF))


def initialize(seed: int) -> tuple[list[list[int]], tuple[int, int, int, int]]:
    check_schedule(PATHS, SHAPES, "head/weight")
    if seed not in SEEDS:
        raise ValueError("unapproved model seed")
    counter = 0
    all_bits: list[list[int]] = []
    for index, shape in enumerate(SHAPES):
        if len(shape) == 1:
            all_bits.append([0x3F800000 if index in (7, 9, 12) else 0]
                            * shape[0])
            continue
        count = shape[0] * shape[1]
        blocks = (count + 3) // 4
        lanes = [lane for offset in range(blocks)
                 for lane in philox(counter + offset, seed)]
        all_bits.append([weight_bits(lane) for lane in lanes[:count]])
        counter += blocks
    if counter != 292 or sum(map(len, all_bits)) != 1192:
        raise AssertionError("C4 initializer schedule drift")
    return all_bits, (1, seed, counter, 0)


def _f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _round32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def tensors(bits: list[list[int]]) -> tuple[torch.Tensor, ...]:
    if len(bits) != 14 or any(len(row) != math.prod(shape)
                              for row, shape in zip(bits, SHAPES)):
        raise ValueError("parameter shape/count")
    result = tuple(torch.tensor([_f32(word) for word in row],
                                dtype=torch.float64, device="cpu").reshape(shape)
                   for row, shape in zip(bits, SHAPES))
    if any(not bool(torch.isfinite(tensor).all()) for tensor in result):
        raise ValueError("nonfinite parameter")
    return result


def _norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
          variant: str) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    if variant == "wrong_norm_variance":
        variance = ((x - mean) ** 2).sum(dim=-1, keepdim=True)
    if variant == "wrong_norm_swap":
        gamma, beta = beta, gamma
    return (x - mean) / torch.sqrt(variance + _f32(0x3727C5AC)) * gamma + beta


def forward(weights: tuple[torch.Tensor, ...], tokens: tuple[int, ...],
            variant: str = "accepted") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent float64 mathematical prefix, no native kernel calls."""
    if variant not in {
        "accepted", "no_position", "wrong_norm_variance", "wrong_norm_swap",
        "wrong_matrix_orientation", "wrong_qk", "wrong_head_split",
        "wrong_scale", "wrong_causal", "wrong_gelu", "wrong_head_tie",
    }:
        raise ValueError("unknown equation variant")
    if len(weights) != 14 or any(tuple(weight.shape) != shape or
                                 weight.dtype != torch.float64 or
                                 weight.device.type != "cpu" or
                                 not bool(torch.isfinite(weight).all())
                                 for weight, shape in zip(weights, SHAPES)):
        raise ValueError("parameter dtype/device/shape/finite")
    if not 1 <= len(tokens) <= 4 or any(type(i) is not int or not 0 <= i < 256
                                        for i in tokens):
        raise ValueError("prefix token range/length")
    ids = torch.tensor(tokens, dtype=torch.int64, device="cpu")
    x = weights[10][ids] + weights[13][:len(tokens)]
    if variant == "no_position":
        x = weights[10][ids]
    n1 = _norm(x, weights[7], weights[6], variant)
    q = n1 @ weights[2].T
    k = n1 @ weights[0].T
    v = n1 @ weights[3].T
    if variant == "wrong_matrix_orientation":
        q = n1 @ weights[2]
    if variant == "wrong_qk":
        q, k = k, q
    length = len(tokens)
    qh = q.reshape(length, 2, 2).permute(1, 0, 2)
    kh = k.reshape(length, 2, 2).permute(1, 0, 2)
    vh = v.reshape(length, 2, 2).permute(1, 0, 2)
    if variant == "wrong_head_split":
        qh = q.reshape(2, length, 2)
    scores = (qh @ kh.transpose(-1, -2)) / (1 if variant == "wrong_scale" else math.sqrt(2))
    mask = torch.ones((length, length), dtype=torch.bool, device="cpu").tril()
    if variant == "wrong_causal":
        mask = torch.ones_like(mask)
    probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    attention = (probabilities @ vh).permute(1, 0, 2).reshape(length, 4)
    r = x + attention @ weights[1].T
    n2 = _norm(r, weights[9], weights[8], variant)
    up = n2 @ weights[5].T
    gelu = 0.5 * up * (1 + torch.erf(up / math.sqrt(2)))
    if variant == "wrong_gelu":
        gelu = 0.5 * up * (1 + torch.tanh(math.sqrt(2 / math.pi) *
                                         (up + 0.044715 * up ** 3)))
    y = r + gelu @ weights[4].T
    nf = _norm(y, weights[12], weights[11], variant)
    head = weights[10].flip(0) if variant == "wrong_head_tie" else weights[10]
    logits = nf @ head.T
    result = (logits, kh.unsqueeze(0), vh.unsqueeze(0))
    if any(not bool(torch.isfinite(part).all()) for part in result):
        raise ValueError("nonfinite forward")
    return result


def _f64_rows(tensor: torch.Tensor) -> dict[str, object]:
    return {"dtype": "float64", "device": "cpu", "shape": list(tensor.shape),
            "bits": [struct.pack(">d", float(value)).hex()
                     for value in tensor.reshape(-1).tolist()]}


def _f32_words(words: list[int], shape: list[int]) -> dict[str, object]:
    return {"dtype": "float32", "device": "cpu", "shape": shape,
            "bits": [f"{word:08x}" for word in words]}


def _independent_selection(logits: torch.Tensor, mode: str,
                           state: tuple[int, int, int, int]
                           ) -> tuple[int, tuple[int, int, int, int]]:
    """Frozen diagnostic IDs; actual-logit G3-S proof remains separate."""
    values = [_round32(float(value)) for value in logits.tolist()]
    if len(values) != 256:
        raise ValueError("selection needs V256")
    if mode == "greedy":
        return max(range(256), key=lambda i: (values[i], -i)), state
    if mode != "categorical":
        raise ValueError("unknown sampling mode")
    lanes = sampler_reference.philox(state[1], state[2], state[3])
    uniform = sampler_reference.f32((lanes[0] >> 8) * 2**-24)
    selected = sampler_reference.literal(values, 1.0, 17, 0.5,
                                         uniform)["token"]
    return selected, (1, state[1], state[2] + 1, state[3])


def build_fixture() -> dict[str, object]:
    require_q0()
    check_source_pins()
    models = []
    cases = []
    terminal_diagnostics = []
    case_keys = set()
    for seed in SEEDS:
        bits, successor = initialize(seed)
        weights = tensors(bits)
        models.append({"seed": seed, "parameters": _f32_words(
            [word for row in bits for word in row], [1192]),
            "successor_words": list(successor)})
        for tokens in PROMPTS:
            logits, keys, values = forward(weights, tokens)
            cases.append({"seed": seed, "tokens": list(tokens),
                          "logits": _f64_rows(logits),
                          "keys": _f64_rows(keys),
                          "values": _f64_rows(values)})
            case_keys.add((seed, tokens))
        if seed == 1729:
            for prompt in ((0, 255), (7, 11)):
                first_logits = forward(weights, prompt)[0][-1]
                for mode, sampler_seed in (("greedy", 7),
                                           ("categorical", 1729)):
                    initial = (1, sampler_seed, 0, 0)
                    first, after_first = _independent_selection(
                        first_logits, mode, initial)
                    prefix3 = (*prompt, first)
                    second_logits = forward(weights, prefix3)[0][-1]
                    second, after_second = _independent_selection(
                        second_logits, mode, after_first)
                    prefix4 = (*prefix3, second)
                    for tokens in (prefix3, prefix4):
                        if (seed, tokens) not in case_keys:
                            logits, keys, values = forward(weights, tokens)
                            cases.append({"seed": seed, "tokens": list(tokens),
                                          "logits": _f64_rows(logits),
                                          "keys": _f64_rows(keys),
                                          "values": _f64_rows(values)})
                            case_keys.add((seed, tokens))
                    terminal_diagnostics.append({
                        "prompt": list(prompt), "mode": mode,
                        "sampling_seed": sampler_seed, "first_id": first,
                        "second_id": second, "initial_rng": list(initial),
                        "after_first_rng": list(after_first),
                        "after_second_rng": list(after_second),
                        "ids_are_diagnostic_until_margin_proof": True})
    return {"version": 1, "name": "g3c4.independent-prefix-math",
            "contract_sha256": CONTRACT_SEAL,
            "source_pins_sha256": SOURCE_PINS_SHA,
            "member_pins_sha256": MEMBER_PINS_SHA,
            "python": "3.14.6", "torch": "2.13.0+cpu",
            "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
            "models": models, "cases": cases,
            "terminal_diagnostics": terminal_diagnostics}


def encode(record: dict[str, object]) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def decode(raw: bytes) -> dict:
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("oversized JSON record")
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def forbidden(value: str) -> None:
        raise ValueError(f"nonfinite JSON literal {value}")

    record = json.loads(raw, object_pairs_hook=unique,
                        parse_constant=forbidden)
    if type(record) is not dict or encode(record) != raw:
        raise ValueError("noncanonical JSON record")
    return record


def _exact_keys(row: dict, names: set[str]) -> None:
    if type(row) is not dict or set(row) != names:
        raise ValueError("missing/extra record field")


def _words(row: dict, shape: list[int], dtype: str) -> list[float]:
    _exact_keys(row, {"dtype", "device", "shape", "bits"})
    if (row["dtype"] != dtype or row["device"] != "cpu" or
        type(row["shape"]) is not list or
        any(type(x) is not int for x in row["shape"]) or
        row["shape"] != shape):
        raise ValueError("dtype/device/shape drift")
    width = 8 if dtype == "float32" else 16
    if type(row["bits"]) is not list or len(row["bits"]) != math.prod(shape):
        raise ValueError("truncated tensor")
    values = []
    for word in row["bits"]:
        if type(word) is not str or len(word) != width or any(
                c not in "0123456789abcdef" for c in word):
            raise ValueError("invalid float word")
        value = struct.unpack(">f" if width == 8 else ">d",
                              bytes.fromhex(word))[0]
        if not math.isfinite(value):
            raise ValueError("nonfinite tensor")
        values.append(value)
    return values


def compare_observation(fixture: dict, observed: dict,
                        frozen_sha256: str) -> dict[str, float]:
    """Compare a future independently captured native transcript, fail closed."""
    require_q0()
    check_source_pins()
    if (type(frozen_sha256) is not str or len(frozen_sha256) != 64 or
        _sha(encode(fixture)) != frozen_sha256):
        raise ValueError("frozen fixture SHA mismatch")
    _exact_keys(fixture, {"version", "name", "contract_sha256",
                          "source_pins_sha256", "member_pins_sha256",
                          "python", "torch", "absolute_tolerance",
                          "relative_tolerance", "models", "cases",
                          "terminal_diagnostics"})
    if (type(fixture["version"]) is not int or fixture["version"] != 1 or
        fixture["name"] != "g3c4.independent-prefix-math" or
        fixture["contract_sha256"] != CONTRACT_SEAL or
        fixture["source_pins_sha256"] != SOURCE_PINS_SHA or
        fixture["member_pins_sha256"] != MEMBER_PINS_SHA or
        fixture["python"] != "3.14.6" or fixture["torch"] != "2.13.0+cpu" or
        fixture["absolute_tolerance"] != ABS_TOL or
        fixture["relative_tolerance"] != REL_TOL):
        raise ValueError("fixture authority drift")
    _exact_keys(observed, {"version", "models", "cases"})
    if type(observed["version"]) is not int or observed["version"] != 1:
        raise ValueError("observed version")
    if type(fixture["models"]) is not list or type(observed["models"]) is not list or \
       type(fixture["cases"]) is not list or type(observed["cases"]) is not list or \
       type(fixture["terminal_diagnostics"]) is not list or \
       len(fixture["terminal_diagnostics"]) != 4 or \
       len(fixture["models"]) != len(observed["models"]) or \
       len(fixture["cases"]) != len(observed["cases"]):
        raise ValueError("record count")
    maximum = 0.0
    for expected, actual in zip(fixture["models"], observed["models"]):
        _exact_keys(expected, {"seed", "parameters", "successor_words"})
        _exact_keys(actual, {"seed", "parameters", "successor_words"})
        if expected["seed"] != actual["seed"] or \
           expected["successor_words"] != actual["successor_words"] or \
           expected["parameters"] != actual["parameters"]:
            raise ValueError("initializer image/state drift")
        _words(expected["parameters"], [1192], "float32")
    for expected, actual in zip(fixture["cases"], observed["cases"]):
        _exact_keys(expected, {"seed", "tokens", "logits", "keys", "values"})
        _exact_keys(actual, {"seed", "tokens", "logits", "keys", "values"})
        if expected["seed"] != actual["seed"] or \
           expected["tokens"] != actual["tokens"]:
            raise ValueError("case identity drift")
        tokens = expected["tokens"]
        if type(tokens) is not list or not 1 <= len(tokens) <= 4 or any(
                type(i) is not int or not 0 <= i < 256 for i in tokens):
            raise ValueError("invalid expected prefix")
        for name, shape in (("logits", [len(tokens), 256]),
                            ("keys", [1, 2, len(tokens), 2]),
                            ("values", [1, 2, len(tokens), 2])):
            lhs = _words(expected[name], shape, "float64")
            rhs = _words(actual[name], shape, "float32")
            for math_value, native_value in zip(lhs, rhs):
                error = abs(math_value - native_value)
                if error > ABS_TOL + REL_TOL * abs(math_value):
                    raise ValueError(f"{name} numerical drift")
                maximum = max(maximum, error)
    return {"maximum_absolute_error": maximum}


def main() -> None:
    parser = argparse.ArgumentParser()
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument("--output", type=Path)
    route.add_argument("--fixture", type=Path)
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--observation", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        if args.frozen_sha256 is not None or args.observation is not None:
            parser.error("generation accepts only --output")
        if args.output.exists():
            raise ValueError("refusing to overwrite frozen fixture")
        payload = encode(build_fixture())
        try:
            with args.output.open("xb") as stream:
                stream.write(payload)
        except FileExistsError as error:
            raise ValueError("refusing to overwrite frozen fixture") from error
    else:
        if args.frozen_sha256 is None or args.observation is None:
            parser.error("comparison needs frozen SHA and observation")
        require_q0()
        check_source_pins()
        expected = decode(args.fixture.read_bytes())
        if _sha(encode(expected)) != args.frozen_sha256:
            raise ValueError("frozen fixture SHA mismatch")
        actual = decode(args.observation.read_bytes())
        print(encode(compare_observation(expected, actual,
                                         args.frozen_sha256)).decode(), end="")


if __name__ == "__main__":
    main()
