"""Strict stdlib validator for generated TR3 reference artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource
from tests.tr3_reference.constants import (
    FORMAT,
    HELDOUT_SHARDS,
    INITIALIZER_SEED,
    OPTIMIZER,
    PARAMETER_SHAPES,
    PROFILE,
    PROPOSAL,
    RESUME_A1_SHARDS,
    RESUME_A2_SHARDS,
    SOURCE_PATHS,
    TOKENIZER_FINGERPRINT,
    TOLERANCES,
    TRAIN_SHARDS,
    UPDATE_COUNT,
    VERSION,
    VOCAB_SIZE,
)


PATHS = tuple(PARAMETER_SHAPES)
SHAPES = PARAMETER_SHAPES


def _fail(message: str) -> None:
    raise ValueError(message)


def _exact_int(value: object, where: str, low: int = 0, high: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        _fail(f"{where}: expected exact integer in range")
    return value


def _finite(value: object, where: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _fail(f"{where}: expected finite float")
    return value


def _strict_equal(value: object, expected: object, where: str) -> None:
    if type(value) is not type(expected):
        _fail(f"{where}: value type differs from frozen contract")
    if isinstance(expected, dict):
        if set(value) != set(expected):
            _fail(f"{where}: mapping fields differ from frozen contract")
        for key in expected:
            _strict_equal(value[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            _fail(f"{where}: list length differs from frozen contract")
        for index, (item, expected_item) in enumerate(zip(value, expected)):
            _strict_equal(item, expected_item, f"{where}[{index}]")
    elif value != expected:
        _fail(f"{where}: value differs from frozen contract")


def _sha(value: object, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        _fail(f"{where}: expected lowercase SHA-256")
    return value


def _load(path: Path, maximum: int) -> object:
    raw = path.read_bytes()
    if not 1 <= len(raw) <= maximum or not raw.endswith(b"\n"):
        _fail(f"{path.name}: noncanonical size or termination")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name}: invalid JSON") from error
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    if raw != canonical:
        _fail(f"{path.name}: noncanonical JSON")
    return value


def _canonical_sha(value: object) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _f32_word(word: object, where: str) -> float:
    if not isinstance(word, str) or len(word) != 8 or any(ch not in "0123456789abcdef" for ch in word):
        _fail(f"{where}: malformed f32 word")
    value = struct.unpack(">f", bytes.fromhex(word))[0]
    if not math.isfinite(value):
        _fail(f"{where}: nonfinite f32 word")
    return value


def _f64_word(word: object, where: str) -> float:
    if not isinstance(word, str) or len(word) != 16 or any(ch not in "0123456789abcdef" for ch in word):
        _fail(f"{where}: malformed f64 word")
    value = struct.unpack(">d", bytes.fromhex(word))[0]
    if not math.isfinite(value):
        _fail(f"{where}: nonfinite f64 word")
    return value


def _as_f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _gradient_record(record: object, where: str) -> dict[str, list[float]]:
    if not isinstance(record, dict) or list(record) != list(PATHS):
        _fail(f"{where}: gradient parameter paths/order mismatch")
    decoded = {}
    for path in PATHS:
        tensor = record[path]
        _tensor(tensor, SHAPES[path], f"{where}.{path}")
        decoded[path] = [
            _f32_word(word, f"{where}.{path}.data[{index}]")
            for index, word in enumerate(tensor["data"])
        ]
    return decoded


def _f32_words(values: object, count: int, where: str) -> None:
    if not isinstance(values, list) or len(values) != count:
        _fail(f"{where}: incorrect value count")
    for index, word in enumerate(values):
        _f32_word(word, f"{where}[{index}]")


def _tensor(record: object, shape: list[int], where: str) -> None:
    if not isinstance(record, dict) or set(record) != {"data", "encoding", "shape"}:
        _fail(f"{where}: tensor fields differ from version 1")
    if record["encoding"] != "ieee754-f32-hex-be":
        _fail(f"{where}: tensor encoding mismatch")
    _strict_equal(record["shape"], shape, where + ".shape")
    count = math.prod(shape)
    _f32_words(record["data"], count, where + ".data")


def _batch(record: object, where: str) -> int:
    expected = {
        "argmax", "inputs", "mask", "numerator_bits", "per_token_loss_bits",
        "targets", "unique_argmax", "weight",
    }
    if not isinstance(record, dict) or set(record) != expected:
        _fail(f"{where}: batch fields differ from version 1")
    for field in ("inputs", "targets", "argmax"):
        values = record[field]
        if not isinstance(values, list) or len(values) != 2:
            _fail(f"{where}.{field}: expected two token IDs")
        for index, value in enumerate(values):
            _exact_int(value, f"{where}.{field}[{index}]", 0, 255)
    mask = record["mask"]
    unique = record["unique_argmax"]
    if not isinstance(mask, list) or len(mask) != 2 or any(type(item) is not bool for item in mask):
        _fail(f"{where}.mask: expected two booleans")
    if not any(mask):
        _fail(f"{where}.mask: empty masks are outside this reference")
    if not isinstance(unique, list) or len(unique) != 2 or any(type(item) is not bool for item in unique):
        _fail(f"{where}.unique_argmax: expected two booleans")
    weight = _exact_int(record["weight"], where + ".weight", 1, 2)
    if weight != sum(mask):
        _fail(f"{where}: weight and mask differ")
    _f32_word(record["numerator_bits"], where + ".numerator_bits")
    loss_words = record["per_token_loss_bits"]
    if not isinstance(loss_words, list) or len(loss_words) != 2:
        _fail(f"{where}.per_token_loss_bits: incorrect value count")
    per_token = [
        _f32_word(word, f"{where}.per_token_loss_bits[{index}]")
        for index, word in enumerate(loss_words)
    ]
    reconstructed = 0.0
    for loss, active in zip(per_token, mask):
        reconstructed = _as_f32(reconstructed + (loss if active else 0.0))
    if record["numerator_bits"] != struct.pack(">f", reconstructed).hex():
        _fail(f"{where}: numerator word differs from masked per-token f32 sum")
    return weight


def _d2_batch(batch) -> dict[str, object]:
    return {
        "inputs": list(batch.inputs[0]),
        "targets": list(batch.targets[0]),
        "mask": list(batch.loss_mask[0]),
        "weight": sum(batch.loss_mask[0]),
    }


def _evaluation(
    record: object,
    where: str,
    expected_batches: list[dict[str, object]] | None = None,
) -> None:
    expected = {
        "batch_count", "batches", "mean", "mean_bits_f64", "numerator",
        "numerator_bits_after_f64_sum", "token_count", "weight",
    }
    if not isinstance(record, dict) or set(record) != expected:
        _fail(f"{where}: evaluation fields differ from version 1")
    batches = record["batches"]
    if not isinstance(batches, list) or not batches:
        _fail(f"{where}.batches: expected nonempty list")
    weights = [_batch(batch, f"{where}.batches[{index}]") for index, batch in enumerate(batches)]
    if expected_batches is not None:
        observed_batches = [
            {field: batch[field] for field in ("inputs", "targets", "mask", "weight")}
            for batch in batches
        ]
        if observed_batches != expected_batches:
            _fail(f"{where}: evaluation batches differ from authenticated corpus")
    batch_numerators = [
        _f32_word(batch["numerator_bits"], f"{where}.batches[{index}].numerator_bits")
        for index, batch in enumerate(batches)
    ]
    if _exact_int(record["batch_count"], where + ".batch_count", 1) != len(batches):
        _fail(f"{where}: batch count mismatch")
    total = sum(weights)
    if (
        _exact_int(record["weight"], where + ".weight", 1) != total
        or _exact_int(record["token_count"], where + ".token_count", 1) != total
    ):
        _fail(f"{where}: token/weight count mismatch")
    numerator = _finite(record["numerator"], where + ".numerator")
    mean = _finite(record["mean"], where + ".mean")
    expected_numerator = sum(batch_numerators, 0.0)
    expected_mean = expected_numerator / total
    if numerator != expected_numerator or mean != expected_mean or numerator <= 0 or mean <= 0:
        _fail(f"{where}: invalid global numerator/weight mean")
    encoded_numerator = _f64_word(
        record["numerator_bits_after_f64_sum"], where + ".numerator_bits_after_f64_sum"
    )
    encoded_mean = _f64_word(record["mean_bits_f64"], where + ".mean_bits_f64")
    if encoded_numerator != numerator or encoded_mean != mean:
        _fail(f"{where}: f64 words differ from numeric accumulation")


def validate_trajectory(payload: object, training_batches: list[dict[str, object]]) -> None:
    expected = {
        "entries", "format", "kind", "logical_alias", "parameter_paths",
        "parameter_shapes", "profile", "version",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        _fail("trajectory fields differ from version 1")
    if (
        payload["format"] != FORMAT
        or _exact_int(payload["version"], "trajectory.version") != VERSION
        or payload["kind"] != "complete-training-trajectory"
        or payload["profile"] != PROFILE
    ):
        _fail("trajectory identity mismatch")
    if payload["parameter_paths"] != list(PATHS):
        _fail("trajectory parameter contract mismatch")
    _strict_equal(payload["parameter_shapes"], SHAPES, "trajectory.parameter_shapes")
    if payload["logical_alias"] != {"token_embedding/weight": "head/weight"}:
        _fail("trajectory logical alias mismatch")
    entries = payload["entries"]
    if not isinstance(entries, list) or len(entries) != UPDATE_COUNT + 1:
        _fail("trajectory must contain step zero and every committed update")
    for update, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "completed_updates", "optimizer_state", "parameters", "training_evaluation"
        }:
            _fail(f"trajectory.entries[{update}]: fields differ")
        if _exact_int(entry["completed_updates"], f"trajectory.entries[{update}].completed_updates") != update:
            _fail(f"trajectory.entries[{update}]: nonconsecutive update")
        if list(entry["parameters"]) != list(PATHS) or list(entry["optimizer_state"]) != list(PATHS):
            _fail(f"trajectory.entries[{update}]: parameter order/path mismatch")
        _evaluation(
            entry["training_evaluation"],
            f"trajectory.entries[{update}].training_evaluation",
            training_batches,
        )
        for path in PATHS:
            _tensor(entry["parameters"][path], SHAPES[path], f"trajectory.entries[{update}].parameters.{path}")
            state = entry["optimizer_state"][path]
            if not isinstance(state, dict) or set(state) != {"exp_avg", "exp_avg_sq", "step"}:
                _fail(f"trajectory.entries[{update}].optimizer_state.{path}: invalid state")
            if _exact_int(state["step"], f"trajectory.entries[{update}].optimizer_state.{path}.step") != update:
                _fail(f"trajectory.entries[{update}].optimizer_state.{path}: invalid state")
            _tensor(state["exp_avg"], SHAPES[path], f"trajectory.entries[{update}].optimizer_state.{path}.exp_avg")
            _tensor(state["exp_avg_sq"], SHAPES[path], f"trajectory.entries[{update}].optimizer_state.{path}.exp_avg_sq")


def validate_corpora(payload: object, root: Path) -> dict[str, list[dict[str, object]]]:
    if not isinstance(payload, dict) or set(payload) != {
        "datasets", "format", "kind", "profile", "resume_cases", "tokenizer", "version"
    }:
        _fail("corpora fields differ from version 1")
    if (
        payload["format"] != FORMAT
        or _exact_int(payload["version"], "corpora.version") != VERSION
        or payload["kind"] != "corpora-and-cursors"
        or payload["profile"] != PROFILE
    ):
        _fail("corpora identity mismatch")
    tokenizer = payload["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != {"fingerprint", "identity", "path", "sha256", "vocab_size"}:
        _fail("tokenizer record fields differ")
    if (
        tokenizer["path"] != "tokenizer.etok"
        or _exact_int(tokenizer["vocab_size"], "tokenizer.vocab_size", 1) != VOCAB_SIZE
        or tokenizer["fingerprint"] != TOKENIZER_FINGERPRINT
        or tokenizer["identity"] != {"utf8_policy": "raw", "specials": [], "prefix": [], "suffix": []}
    ):
        _fail("tokenizer identity mismatch")
    _sha(tokenizer["sha256"], "tokenizer.sha256")
    if hashlib.sha256((root / tokenizer["path"]).read_bytes()).hexdigest() != tokenizer["sha256"]:
        _fail("tokenizer artifact digest mismatch")
    datasets = payload["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != {"train", "heldout", "resume-a1", "resume-a2"}:
        _fail("dataset set mismatch")
    dataset_contracts = {
        "train": (TRAIN_SHARDS, True),
        "heldout": (HELDOUT_SHARDS, False),
        "resume-a1": (RESUME_A1_SHARDS, True),
        "resume-a2": (RESUME_A2_SHARDS, False),
    }
    resources = {}
    authenticated_batches = {}
    for name, record in datasets.items():
        if not isinstance(record, dict) or set(record) != {
            "batches", "end_cursor", "files", "manifest_trailer_sha256", "packing", "shards", "start_cursor"
        }:
            _fail(f"datasets.{name}: fields differ")
        expected_shards, expected_packing = dataset_contracts[name]
        if record["packing"] is not expected_packing:
            _fail(f"datasets.{name}: differs from frozen profile shards/packing")
        _strict_equal(
            record["shards"],
            [list(shard) for shard in expected_shards],
            f"datasets.{name}.shards",
        )
        if type(record["packing"]) is not bool or not isinstance(record["batches"], list):
            _fail(f"datasets.{name}: invalid packing/batches")
        if not isinstance(record["shards"], list) or not record["shards"]:
            _fail(f"datasets.{name}.shards: expected nonempty shards")
        for shard_index, shard in enumerate(record["shards"]):
            if not isinstance(shard, list) or not shard:
                _fail(f"datasets.{name}.shards[{shard_index}]: expected nonempty token list")
            for token_index, token in enumerate(shard):
                _exact_int(token, f"datasets.{name}.shards[{shard_index}][{token_index}]", 0, 255)
        for index, batch in enumerate(record["batches"]):
            if not isinstance(batch, dict) or set(batch) != {"inputs", "mask", "targets", "weight"}:
                _fail(f"datasets.{name}.batches[{index}]: fields differ")
            shadow = batch | {
                "argmax": batch["targets"], "unique_argmax": [True, True],
                "numerator_bits": "00000000", "per_token_loss_bits": ["00000000", "00000000"],
            }
            _batch(shadow, f"datasets.{name}.batches[{index}]")
        trailer = _sha(record["manifest_trailer_sha256"], f"datasets.{name}.manifest_trailer_sha256")
        files = record["files"]
        if not isinstance(files, list) or not files:
            _fail(f"datasets.{name}.files: expected nonempty list")
        expected_names = ["manifest.etm"] + [
            f"shard-{index:016d}.ets" for index in range(len(record["shards"]))
        ]
        if [item.get("path") if isinstance(item, dict) else None for item in files] != expected_names:
            _fail(f"datasets.{name}.files: physical filenames mismatch")
        for item in files:
            if not isinstance(item, dict) or set(item) != {"bytes", "path", "sha256"}:
                _fail(f"datasets.{name}.files: malformed record")
            if not isinstance(item["path"], str) or "/" in item["path"] or "\\" in item["path"]:
                _fail(f"datasets.{name}.files: unsafe path")
            path = root / name / item["path"]
            size = _exact_int(item["bytes"], "file.bytes", 1, 1_048_576)
            if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != _sha(item["sha256"], "file.sha256"):
                _fail(f"datasets.{name}.{item['path']}: file metadata mismatch")
        loaded_shards, digest = load_d1_resource(
            root / name,
            expected_fingerprint=TOKENIZER_FINGERPRINT,
            expected_vocab=256,
            maximum_total_tokens=sum(map(len, record["shards"])),
        )
        normalized_shards = tuple(tuple(shard) for shard in record["shards"])
        if loaded_shards != normalized_shards or digest.hex() != trailer:
            _fail(f"datasets.{name}: physical D1 resource differs from schema")
        config = ReferenceConfig(1, 2, None, 1, record["packing"])
        dataset = ReferenceDataset(
            loaded_shards,
            config,
            manifest_digest=digest,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=256,
        )
        if dataset.snapshot().hex() != record["start_cursor"]:
            _fail(f"datasets.{name}: start cursor differs from authentic D2 state")
        replayed = []
        while (batch := dataset.next_batch()) is not None:
            replayed.append(_d2_batch(batch))
        if replayed != record["batches"] or dataset.snapshot().hex() != record["end_cursor"]:
            _fail(f"datasets.{name}: batches or end cursor differ from authentic D2 replay")
        resources[name] = (loaded_shards, digest, config)
        authenticated_batches[name] = replayed
    cases = payload["resume_cases"]
    if not isinstance(cases, dict) or set(cases) != {"accumulation-1", "accumulation-2-unequal-mask"}:
        _fail("resume case set mismatch")
    for name, dataset_name, expected_a, expected_weights in (
        ("accumulation-1", "resume-a1", 1, [2]),
        ("accumulation-2-unequal-mask", "resume-a2", 2, [1, 2]),
    ):
        case = cases[name]
        if not isinstance(case, dict) or set(case) != {
            "K", "R", "accumulation_steps", "contributions", "end_cursor", "epoch_crossings",
            "epoch_start_cursor", "saved_cursor", "saved_next_ordinal", "total_weight",
        }:
            _fail(f"resume_cases.{name}: fields differ")
        if (
            _exact_int(case["K"], f"resume_cases.{name}.K", 1) != 1
            or _exact_int(case["R"], f"resume_cases.{name}.R", 3) != 3
            or _exact_int(case["accumulation_steps"], f"resume_cases.{name}.accumulation_steps", 1) != expected_a
        ):
            _fail(f"resume_cases.{name}: boundary contract mismatch")
        contributions = case["contributions"]
        if not isinstance(contributions, list) or len(contributions) != expected_a:
            _fail(f"resume_cases.{name}: contribution count mismatch")
        loaded_shards, digest, config = resources[dataset_name]
        dataset = ReferenceDataset(
            loaded_shards,
            config,
            manifest_digest=digest,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            vocab_size=256,
        )
        epoch_start = dataset.snapshot()
        if case["epoch_start_cursor"] != epoch_start.hex():
            _fail(f"resume_cases.{name}: epoch-start cursor mismatch")
        for _ in range(case["K"]):
            if dataset.next_batch() is None:
                _fail(f"resume_cases.{name}: saved position exceeds dataset")
        saved = dataset.snapshot()
        saved_fields = decode_reference_cursor(saved)
        if (
            case["saved_cursor"] != saved.hex()
            or _exact_int(case["saved_next_ordinal"], f"resume_cases.{name}.saved_next_ordinal")
            != saved_fields.next_ordinal
        ):
            _fail(f"resume_cases.{name}: saved cursor/ordinal mismatch")
        weights = []
        crossings = 0
        epoch = 0
        for index, contribution in enumerate(contributions):
            if not isinstance(contribution, dict) or set(contribution) != {"batch", "contribution_ordinal", "epoch"}:
                _fail(f"resume_cases.{name}.contributions[{index}]: fields differ")
            if _exact_int(
                contribution["contribution_ordinal"],
                f"resume_cases.{name}.contributions[{index}].contribution_ordinal",
            ) != index:
                _fail(f"resume_cases.{name}.contributions[{index}]: ordinal mismatch")
            emitted_batch = dataset.next_batch()
            if emitted_batch is None:
                crossings += 1
                epoch += 1
                dataset.seek(epoch_start)
                emitted_batch = dataset.next_batch()
            if emitted_batch is None:
                _fail(f"resume_cases.{name}: replay remained exhausted after epoch reset")
            if _exact_int(
                contribution["epoch"],
                f"resume_cases.{name}.contributions[{index}].epoch",
            ) != epoch:
                _fail(f"resume_cases.{name}.contributions[{index}]: epoch mismatch")
            recorded_batch = contribution["batch"]
            if not isinstance(recorded_batch, dict) or set(recorded_batch) != {"inputs", "mask", "targets", "weight"}:
                _fail(f"resume_cases.{name}.contributions[{index}].batch: fields differ")
            shadow = recorded_batch | {
                "argmax": recorded_batch["targets"], "unique_argmax": [True, True],
                "numerator_bits": "00000000", "per_token_loss_bits": ["00000000", "00000000"],
            }
            weight = _batch(shadow, f"resume_cases.{name}.contributions[{index}].batch")
            actual = {
                "batch": _d2_batch(emitted_batch),
                "contribution_ordinal": index,
                "epoch": epoch,
            }
            if contribution != actual:
                _fail(f"resume_cases.{name}.contributions[{index}]: authentic replay mismatch")
            weights.append(weight)
        if (
            weights != expected_weights
            or _exact_int(case["total_weight"], f"resume_cases.{name}.total_weight", 1) != sum(weights)
            or _exact_int(case["epoch_crossings"], f"resume_cases.{name}.epoch_crossings") != crossings
            or case["end_cursor"] != dataset.snapshot().hex()
        ):
            _fail(f"resume_cases.{name}: contribution sequence mismatch")
    if (
        len(authenticated_batches["train"]) != 1
        or [batch["weight"] for batch in authenticated_batches["train"]] != [2]
        or [batch["weight"] for batch in authenticated_batches["heldout"]] != [2, 1]
        or [
            target
            for batch in authenticated_batches["train"]
            for target, active in zip(batch["targets"], batch["mask"])
            if active
        ]
        != [197, 41]
    ):
        _fail("authenticated corpora differ from frozen profile batch invariants")
    return authenticated_batches


def _active_transitions(batches: list[dict[str, object]]) -> set[tuple[int, int]]:
    return {
        (left, right)
        for batch in batches
        for left, right, active in zip(batch["inputs"], batch["targets"], batch["mask"])
        if active
    }


def _active_evaluation_values(evaluation: dict[str, object], field: str) -> list[object]:
    return [
        value
        for batch in evaluation["batches"]
        for value, active in zip(batch[field], batch["mask"])
        if active
    ]


def _summary_measurements(payload: dict[str, object]) -> dict[str, object]:
    training = payload["training"]
    heldout = payload["heldout"]
    final_training = training["after"]
    final_heldout = heldout["after"]
    return {
        "initial_training_mean_ce": training["before"]["mean"],
        "final_training_mean_ce": final_training["mean"],
        "final_active_targets": _active_evaluation_values(final_training, "targets"),
        "final_active_argmax": _active_evaluation_values(final_training, "argmax"),
        "final_active_unique_argmax": _active_evaluation_values(final_training, "unique_argmax"),
        "initial_heldout_global_mean_ce": heldout["before"]["mean"],
        "final_heldout_global_mean_ce": final_heldout["mean"],
        "heldout_weight": final_heldout["weight"],
        "heldout_batches": final_heldout["batch_count"],
        "heldout_mask_weights": [batch["weight"] for batch in final_heldout["batches"]],
    }


def validate_summary(
    payload: object,
    corpus_batches: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "accumulation_reference", "format", "heldout", "initializer", "kind", "optimizer",
        "profile", "provenance", "scope", "thresholds", "training", "updates", "version",
    }:
        _fail("summary fields differ from version 1")
    if (
        payload["format"] != FORMAT
        or _exact_int(payload["version"], "summary.version") != VERSION
        or payload["kind"] != "acceptance-summary"
        or payload["profile"] != PROFILE
        or _exact_int(payload["updates"], "summary.updates") != UPDATE_COUNT
    ):
        _fail("summary identity/configuration mismatch")
    _strict_equal(payload["optimizer"], OPTIMIZER, "summary.optimizer")
    if not isinstance(payload["initializer"], dict) or set(payload["initializer"]) != {"seed", "successor"}:
        _fail("summary initializer mismatch")
    if _exact_int(payload["initializer"]["seed"], "summary.initializer.seed") != INITIALIZER_SEED:
        _fail("summary initializer mismatch")
    _strict_equal(
        payload["initializer"]["successor"],
        [1, INITIALIZER_SEED, 290, 0],
        "summary.initializer.successor",
    )
    if not isinstance(payload["training"], dict) or set(payload["training"]) != {"before", "after"}:
        _fail("summary training fields differ")
    if not isinstance(payload["heldout"], dict) or set(payload["heldout"]) != {"before", "after", "train_transitions", "heldout_transitions"}:
        _fail("summary heldout fields differ")
    _evaluation(payload["training"]["before"], "summary.training.before", corpus_batches["train"])
    _evaluation(payload["training"]["after"], "summary.training.after", corpus_batches["train"])
    _evaluation(payload["heldout"]["before"], "summary.heldout.before", corpus_batches["heldout"])
    _evaluation(payload["heldout"]["after"], "summary.heldout.after", corpus_batches["heldout"])
    train_transitions = _active_transitions(corpus_batches["train"])
    heldout_transitions = _active_transitions(corpus_batches["heldout"])
    if train_transitions & heldout_transitions:
        _fail("authenticated held-out transitions overlap training transitions")
    for label, expected_transitions in (
        ("train_transitions", train_transitions),
        ("heldout_transitions", heldout_transitions),
    ):
        transitions = payload["heldout"][label]
        if not isinstance(transitions, list) or not transitions:
            _fail(f"summary.heldout.{label}: expected nonempty transitions")
        normalized = []
        for index, pair in enumerate(transitions):
            if not isinstance(pair, list) or len(pair) != 2:
                _fail(f"summary.heldout.{label}[{index}]: expected token pair")
            normalized.append(
                tuple(
                    _exact_int(token, f"summary.heldout.{label}[{index}]", 0, 255)
                    for token in pair
                )
            )
        if normalized != sorted(expected_transitions):
            _fail(f"summary.heldout.{label}: differs from authenticated corpus")
    measurements = _summary_measurements(payload)
    if (
        measurements["final_active_argmax"] != measurements["final_active_targets"]
        or not all(measurements["final_active_unique_argmax"])
    ):
        _fail("final active targets are not unique logits argmax")
    if payload["training"]["after"]["mean"] >= 0.05:
        _fail("final training mean CE threshold failed")
    if payload["heldout"]["after"]["mean"] >= payload["heldout"]["before"]["mean"]:
        _fail("held-out global loss did not decrease")
    proof = payload["accumulation_reference"]
    if not isinstance(proof, dict) or set(proof) != {
        "bit_identical", "combined_numerator_gradient_sha256", "combined_numerator_gradients",
        "contribution_ordinals", "max_abs_difference", "parameter_count",
        "summed_numerator_gradient_sha256", "summed_numerator_gradients",
        "total_weight", "weights", "within_parameter_tolerance", "wrong_per_batch_mean_gradients",
        "wrong_per_batch_mean_max_abs_difference",
    }:
        _fail("accumulation proof fields differ")
    weights = proof["weights"]
    ordinals = proof["contribution_ordinals"]
    if not isinstance(weights, list) or len(weights) != 2 or not isinstance(ordinals, list) or len(ordinals) != 2:
        _fail("accumulation proof contract mismatch")
    if (
        [_exact_int(value, f"accumulation.weights[{index}]", 1) for index, value in enumerate(weights)] != [2, 1]
        or [_exact_int(value, f"accumulation.ordinals[{index}]") for index, value in enumerate(ordinals)] != [0, 1]
        or _exact_int(proof["total_weight"], "accumulation.total_weight", 1) != 3
        or _exact_int(proof["parameter_count"], "accumulation.parameter_count", 1) != len(PATHS)
    ):
        _fail("accumulation proof contract mismatch")
    summed = _gradient_record(proof["summed_numerator_gradients"], "accumulation.summed")
    combined = _gradient_record(proof["combined_numerator_gradients"], "accumulation.combined")
    wrong = _gradient_record(proof["wrong_per_batch_mean_gradients"], "accumulation.wrong_mean")
    summed_digest = _canonical_sha(proof["summed_numerator_gradients"])
    combined_digest = _canonical_sha(proof["combined_numerator_gradients"])
    reported_summed_digest = _sha(
        proof["summed_numerator_gradient_sha256"], "accumulation.summed.sha256"
    )
    reported_combined_digest = _sha(
        proof["combined_numerator_gradient_sha256"], "accumulation.combined.sha256"
    )
    if reported_summed_digest != summed_digest or reported_combined_digest != combined_digest:
        _fail("accumulation gradient digest mismatch")
    bit_identical = proof["summed_numerator_gradients"] == proof["combined_numerator_gradients"]
    if type(proof["bit_identical"]) is not bool or proof["bit_identical"] != bit_identical:
        _fail("accumulation bit-identity flag mismatch")
    differences = []
    wrong_differences = []
    for path in PATHS:
        for left, right, wrong_value in zip(summed[path], combined[path], wrong[path]):
            differences.append(abs(_as_f32(left - right)))
            normalized_left = _as_f32(left / 3)
            normalized_wrong = _as_f32(wrong_value / 2)
            wrong_differences.append(abs(_as_f32(normalized_left - normalized_wrong)))
    measured_delta = _finite(proof["max_abs_difference"], "accumulation max delta")
    if measured_delta < 0 or measured_delta != max(differences):
        _fail("accumulation max delta differs from gradient records")
    within = measured_delta <= TOLERANCES["parameter"]["absolute"]
    if type(proof["within_parameter_tolerance"]) is not bool or proof["within_parameter_tolerance"] != within or not within:
        _fail("combined numerator gradient parity failed")
    wrong_delta = _finite(proof["wrong_per_batch_mean_max_abs_difference"], "wrong mean delta")
    if wrong_delta <= 0 or wrong_delta != max(wrong_differences):
        _fail("wrong mean-seed control did not differ")
    _strict_equal(
        payload["thresholds"],
        {"final_training_mean_ce_strict_upper": 0.05, "tolerances": TOLERANCES},
        "summary.thresholds",
    )
    scope = payload["scope"]
    if not isinstance(scope, dict) or scope.get("development_reference_only") is not True or any(
        scope.get(field) is not False
        for field in ("native_acceptance", "native_resume_proof", "checkpoint_format_or_api", "runtime_or_production_dependency")
    ):
        _fail("scope must remain development-only")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"environment", "proposal", "python", "source_sha256", "torch"}:
        _fail("provenance fields differ")
    if provenance["python"] != "3.14.6" or provenance["torch"] != "2.13.0+cpu" or provenance["environment"] != {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE"}:
        _fail("pinned provenance mismatch")
    _strict_equal(provenance["proposal"], PROPOSAL, "summary.provenance.proposal")
    source_hashes = provenance["source_sha256"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_PATHS):
        _fail("provenance source inventory mismatch")
    for path in SOURCE_PATHS:
        digest = source_hashes[path]
        _sha(digest, "provenance.source_sha256")
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts or hashlib.sha256((Path(__file__).resolve().parents[2] / candidate).read_bytes()).hexdigest() != digest:
            _fail(f"provenance source mismatch: {path}")
    return measurements


def validate_output(root: Path, manifest: object) -> None:
    if not isinstance(manifest, dict) or set(manifest) != {
        "artifacts", "configuration", "format", "measurements", "profile", "provenance", "scope", "version"
    }:
        _fail("manifest fields differ from version 1")
    if (
        manifest["format"] != "eshkol-tr3-training-reference-manifest"
        or _exact_int(manifest["version"], "manifest.version") != VERSION
        or manifest["profile"] != PROFILE
    ):
        _fail("manifest identity mismatch")
    _strict_equal(
        manifest["configuration"],
        {"initializer_seed": INITIALIZER_SEED, "optimizer": OPTIMIZER, "updates": UPDATE_COUNT},
        "manifest.configuration",
    )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"corpora.json", "summary.json", "trajectory.json"}:
        _fail("manifest artifact set mismatch")
    payloads = {}
    limits = {"corpora.json": 100_000, "summary.json": 100_000, "trajectory.json": 8_000_000}
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            _fail(f"manifest artifact record malformed: {name}")
        path = root / name
        if path.stat().st_size != _exact_int(record["bytes"], name + ".bytes", 1, limits[name]):
            _fail(f"manifest artifact size mismatch: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != _sha(record["sha256"], name + ".sha256"):
            _fail(f"manifest artifact digest mismatch: {name}")
        payloads[name] = _load(path, limits[name])
    corpus_batches = validate_corpora(payloads["corpora.json"], root)
    validate_trajectory(payloads["trajectory.json"], corpus_batches["train"])
    measurements = validate_summary(payloads["summary.json"], corpus_batches)
    summary = payloads["summary.json"]
    trajectory = payloads["trajectory.json"]
    if (
        summary["training"]["before"] != trajectory["entries"][0]["training_evaluation"]
        or summary["training"]["after"] != trajectory["entries"][-1]["training_evaluation"]
    ):
        _fail("summary training endpoints differ from complete trajectory")
    _strict_equal(manifest["provenance"], summary["provenance"], "manifest.provenance")
    _strict_equal(manifest["scope"], summary["scope"], "manifest.scope")
    _strict_equal(manifest["measurements"], measurements, "manifest.measurements")
