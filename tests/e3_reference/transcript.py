"""Strict standalone records for development-only E3 observations.

This format is deliberately separate from Q0 and from any future E3 runtime API.
It contains data supplied by a test harness; it never opens a native library.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, NoReturn

from tests.e3_metrics.reference_probe import (
    INF_BITS, accumulate, divide, finalize, reduce_bool,
)


FORMAT = "eshkol-e3-development-transcript"
VERSION = 1
MAX_BYTES = 8 * 1024 * 1024
MAX_BATCHES = 8_192
F32_TOLERANCE = (2.0e-6, 2.0e-5)


class TranscriptError(ValueError):
    """A malformed record or failed independent comparison."""


def _fail(message: str) -> NoReturn:
    raise TranscriptError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def encode(payload: dict[str, object]) -> bytes:
    payload_bytes = canonical(payload)
    result = canonical(
        {
            "format": FORMAT,
            "payload": payload,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "version": VERSION,
        }
    )
    if len(result) > MAX_BYTES:
        _fail("record exceeds the development transcript byte limit")
    return result


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode(raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_BYTES:
        _fail("record exceeds the development transcript byte limit")
    if not raw.endswith(b"\n") or raw.startswith(b"\xef\xbb\xbf"):
        _fail("record is not canonical LF-terminated UTF-8 JSON")
    try:
        document = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TranscriptError("record is not valid UTF-8 JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "format", "payload", "payload_sha256", "version"
    }:
        _fail("record envelope fields differ from version 1")
    if document["format"] != FORMAT or document["version"] != VERSION:
        _fail("record format or version mismatch")
    payload = document["payload"]
    digest = document["payload_sha256"]
    if not isinstance(payload, dict) or not isinstance(digest, str):
        _fail("record payload or checksum has the wrong type")
    if payload.get("kind") not in {
        "corpus-reference", "reference-bundle", "observed-f32", "synthetic-observation"
    }:
        _fail("record payload kind is unknown")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        _fail("record checksum is not canonical lowercase SHA-256")
    if hashlib.sha256(canonical(payload)).hexdigest() != digest:
        _fail("record payload checksum mismatch")
    if encode(payload) != raw:
        _fail("record JSON is not canonical")
    return payload


def f32_bits(value: float) -> str:
    return struct.pack(">f", value).hex()


def f64_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def bits_f32(word: object, where: str) -> tuple[int, float]:
    if not isinstance(word, str) or len(word) != 8 or any(
        ch not in "0123456789abcdef" for ch in word
    ):
        _fail(f"{where}: expected lowercase binary32 word")
    bits = int(word, 16)
    value = struct.unpack(">f", bytes.fromhex(word))[0]
    if not math.isfinite(value):
        _fail(f"{where}: expected finite binary32")
    return bits, value


def bits_f64(word: object, where: str) -> float:
    if not isinstance(word, str) or len(word) != 16 or any(
        ch not in "0123456789abcdef" for ch in word
    ):
        _fail(f"{where}: expected lowercase binary64 word")
    value = struct.unpack(">d", bytes.fromhex(word))[0]
    if not math.isfinite(value):
        _fail(f"{where}: expected finite binary64")
    return value


def _i64(value: object, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < 1 << 63:
        _fail(f"{where}: expected exact signed-i64")
    return value


def _token(value: object, where: str) -> int:
    result = _i64(value, where)
    if result >= 256:
        _fail(f"{where}: expected raw V256 token ID")
    return result


def _pair(values: object, where: str, element) -> tuple[Any, Any]:
    if not isinstance(values, list) or len(values) != 2:
        _fail(f"{where}: expected length 2")
    return element(values[0], f"{where}[0]"), element(values[1], f"{where}[1]")


def _bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{where}: expected Boolean")
    return value


def _close(actual: float, expected: float, where: str) -> None:
    absolute, relative = F32_TOLERANCE
    if abs(actual - expected) > absolute + relative * abs(expected):
        _fail(f"{where}: outside mathematical tolerance")


def _ce(logits: tuple[float, ...], target: int) -> float:
    maximum = max(logits)
    return (
        math.log(math.fsum(math.exp(value - maximum) for value in logits))
        + maximum
        - logits[target]
    )


def validate_mathematical_case(
    observed: dict[str, object], mathematical: list[dict[str, object]],
    expected_batches: tuple[dict[str, object], ...], expected_loss_word: str,
) -> dict[str, float]:
    """Check model/CE tolerance separately from exact observed-f32 arithmetic."""

    validate_observed_case(observed, expected_batches)
    batches = observed.get("batches")
    if not isinstance(batches, list) or len(batches) != len(mathematical):
        _fail("mathematical/observed batch count mismatch")
    active_model_ce: list[float] = []
    active_observed_ce: list[float] = []
    local_budgets: list[float] = []
    maximum_logit_error = 0.0
    for index, (actual, reference, expected) in enumerate(
        zip(batches, mathematical, expected_batches)
    ):
        where = f"mathematical[{index}]"
        if reference.get("inputs") != expected["inputs"] or reference.get("targets") != expected["targets"] or reference.get("mask") != expected["mask"]:
            _fail(f"{where}: corpus wiring differs from the D2 reference")
        roles = reference.get("roles")
        if not isinstance(roles, list) or not roles or roles[-1].get("name") != "Z":
            _fail(f"{where}: missing final mathematical Z role")
        words = roles[-1].get("values_f64")
        if not isinstance(words, list) or len(words) != 512:
            _fail(f"{where}.Z: expected shape [1,2,256]")
        model_logits = tuple(bits_f64(word, where + ".Z") for word in words)
        observed_words = actual.get("logits_f32")
        if not isinstance(observed_words, list) or len(observed_words) != 512:
            _fail(f"batches[{index}].logits_f32: expected shape [1,2,256]")
        observed_logits = tuple(
            bits_f32(word, f"batches[{index}].logits_f32")[1]
            for word in observed_words
        )
        delta = max(abs(left - right) for left, right in zip(observed_logits, model_logits))
        maximum_logit_error = max(maximum_logit_error, delta)
        scale = max(abs(value) for value in model_logits)
        if delta > 1.0e-10 + 3.0e-4 * scale:
            _fail(f"{where}.Z: exceeds the accepted M3 mathematical tolerance")
        ce_words = reference.get("ce_f64")
        if not isinstance(ce_words, list) or len(ce_words) != 2:
            _fail(f"{where}.ce_f64: expected length 2")
        targets = tuple(expected["targets"])
        mask = tuple(expected["mask"])
        actual_ce_words = actual.get("ce_f32")
        if not isinstance(actual_ce_words, list) or len(actual_ce_words) != 2:
            _fail(f"batches[{index}].ce_f32: expected length 2")
        for position in range(2):
            model_ce = bits_f64(ce_words[position], f"{where}.ce_f64[{position}]")
            recomputed = _ce(
                model_logits[position * 256 : (position + 1) * 256], targets[position]
            )
            if abs(model_ce - recomputed) > 1.0e-12:
                _fail(f"{where}.ce_f64[{position}]: equation mismatch")
            observed_ce = bits_f32(
                actual_ce_words[position], f"batches[{index}].ce_f32[{position}]"
            )[1]
            independent_observed_ce = _ce(
                observed_logits[position * 256 : (position + 1) * 256], targets[position]
            )
            local = F32_TOLERANCE[0] + F32_TOLERANCE[1] * abs(independent_observed_ce)
            if mask[position]:
                active_model_ce.append(model_ce)
                active_observed_ce.append(observed_ce)
                local_budgets.append(local + 2.0 * delta)
    if not active_model_ce:
        _fail("mathematical case has no active token")
    mathematical_loss = math.fsum(active_model_ce) / len(active_model_ce)
    declared_loss = bits_f64(expected_loss_word, "mathematical_loss_f64")
    if abs(declared_loss - mathematical_loss) > 1.0e-12:
        _fail("declared mathematical loss differs from per-position CE")
    final = observed.get("final")
    if not isinstance(final, dict) or not isinstance(final.get("metrics_f32"), list):
        _fail("observed final metrics are missing")
    final_loss = bits_f32(final["metrics_f32"][0], "final.metrics_f32[0]")[1]
    observed_mean = math.fsum(active_observed_ce) / len(active_observed_ce)
    rounding = abs(final_loss - observed_mean)
    budget = rounding + math.fsum(local_budgets) / len(local_budgets) + 1.0e-12
    if abs(final_loss - mathematical_loss) > budget:
        _fail("final loss exceeds propagated model/CE/rounding tolerance")
    return {
        "budget": budget,
        "error": abs(final_loss - mathematical_loss),
        "maximum_logit_error": maximum_logit_error,
        "rounding": rounding,
    }


def validate_observed_case(
    observed: dict[str, object], expected_batches: tuple[dict[str, object], ...]
) -> None:
    """Validate a future harness transcript without depending on a harness API.

    M3 logits and CE are tolerance comparisons. Reduction, accumulation, counters,
    accuracy and final metrics are exact functions of the observed binary32 words.
    """

    if set(observed) != {"batches", "final", "kind", "profile"}:
        _fail("observed case fields differ from version 1")
    if observed["kind"] not in {"observed-f32", "synthetic-observation"}:
        _fail("observed case kind is not explicit")
    if observed["profile"] != "e3-private-n1-t2-v256-d4-v1":
        _fail("observed case profile mismatch")
    batches = observed["batches"]
    if not isinstance(batches, list) or not 1 <= len(batches) <= MAX_BATCHES:
        _fail("observed case has invalid batch count")
    if len(batches) != len(expected_batches):
        _fail("observed case batch count differs from D2 reference")
    state = (0, 0, 0, 0, 0)
    for index, (batch, expected) in enumerate(zip(batches, expected_batches)):
        where = f"batches[{index}]"
        if not isinstance(batch, dict) or set(batch) != {
            "ce_f32", "correct_f32", "counters", "inputs", "logits_f32",
            "mask", "ordinal", "reduction_f32", "state_f32", "targets"
        }:
            _fail(f"{where}: fields differ from version 1")
        if _i64(batch["ordinal"], where + ".ordinal") != index:
            _fail(f"{where}: ordinal mismatch")
        inputs = _pair(batch["inputs"], where + ".inputs", _token)
        targets = _pair(batch["targets"], where + ".targets", _i64)
        mask = _pair(batch["mask"], where + ".mask", _bool)
        if list(inputs) != expected["inputs"]:
            _fail(f"{where}.inputs: differs from the independent D2 emission")
        if list(targets) != expected["targets"]:
            _fail(f"{where}.targets: differs from the independent D2 emission")
        if list(mask) != expected["mask"]:
            _fail(f"{where}.mask: differs from the independent D2 emission")
        if mask not in ((True, True), (True, False)):
            _fail(f"{where}.mask: valid D2 rows require an active prefix")
        raw_logits = batch["logits_f32"]
        if not isinstance(raw_logits, list) or len(raw_logits) != 512:
            _fail(f"{where}.logits_f32: expected shape [1,2,256]")
        logits = tuple(bits_f32(word, where + ".logits_f32")[1] for word in raw_logits)
        ce_words = _pair(batch["ce_f32"], where + ".ce_f32", bits_f32)
        for position in range(2):
            _close(
                ce_words[position][1],
                _ce(logits[position * 256 : (position + 1) * 256], targets[position]),
                f"{where}.ce_f32[{position}]",
            )
        ce_bits = (ce_words[0][0], ce_words[1][0])
        numerator, weight = reduce_bool(ce_bits, mask)
        reduction = (numerator, weight, divide(numerator, weight))
        actual_reduction = tuple(
            bits_f32(word, where + ".reduction_f32")[0]
            for word in _triple(batch["reduction_f32"], where + ".reduction_f32")
        )
        if actual_reduction != reduction:
            _fail(f"{where}.reduction_f32: exact rational reduction mismatch")
        correct_count = 0
        for position in range(2):
            row = logits[position * 256 : (position + 1) * 256]
            winner = max(range(256), key=lambda token: (row[token], -token))
            correct_count += int(mask[position] and winner == targets[position])
        correct = bits_f32(batch["correct_f32"], where + ".correct_f32")[0]
        if correct != _exact_small_f32(correct_count):
            _fail(f"{where}.correct_f32: target/argmax wiring mismatch")
        state = accumulate(state, (numerator, weight, correct, sum(mask)))
        actual_state = tuple(
            bits_f32(word, where + ".state_f32")[0]
            for word in _triple(batch["state_f32"], where + ".state_f32")
        )
        if actual_state != state[:3]:
            _fail(f"{where}.state_f32: exact accumulation mismatch")
        counters = _pair(batch["counters"], where + ".counters", _i64)
        if counters != state[3:]:
            _fail(f"{where}.counters: token/batch counter mismatch")
    final = observed["final"]
    if not isinstance(final, dict) or set(final) != {"counters", "metrics_f32"}:
        _fail("final fields differ from version 1")
    words = final["metrics_f32"]
    if not isinstance(words, list) or len(words) != 4:
        _fail("final.metrics_f32: expected four words")
    actual = tuple(bits_f32(word, "final.metrics_f32")[0] for word in words)
    expected = finalize(state)
    if actual[0:2] != expected[0:2] or actual[3] != expected[3]:
        _fail("final.metrics_f32: loss/weight/accuracy mismatch")
    if expected[2] == INF_BITS:
        _fail("final.metrics_f32: successful finite transcript expected overflow")
    if abs(actual[2] - expected[2]) > 2:
        _fail("final.metrics_f32: exponential exceeds two binary32 ULP")
    if _pair(final["counters"], "final.counters", _i64) != state[3:]:
        _fail("final.counters mismatch")


def _triple(value: object, where: str) -> tuple[object, object, object]:
    if not isinstance(value, list) or len(value) != 3:
        _fail(f"{where}: expected length 3")
    return value[0], value[1], value[2]


def _exact_small_f32(value: int) -> int:
    return int.from_bytes(struct.pack(">f", float(value)), byteorder="big")
