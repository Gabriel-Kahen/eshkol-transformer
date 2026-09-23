"""Strict stdlib validator for generated TR3 reference artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

from tests.d2.reference import decode_reference_cursor
from tests.tr3_reference.constants import (
    FORMAT,
    OPTIMIZER,
    PARAMETER_SHAPES,
    PROFILE,
    PROPOSAL,
    TOKENIZER_FINGERPRINT,
    TOLERANCES,
    UPDATE_COUNT,
    VERSION,
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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{where}: expected finite number")
    return float(value)


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


def _f32_words(values: object, count: int, where: str) -> None:
    if not isinstance(values, list) or len(values) != count:
        _fail(f"{where}: incorrect value count")
    for index, word in enumerate(values):
        if not isinstance(word, str) or len(word) != 8 or any(ch not in "0123456789abcdef" for ch in word):
            _fail(f"{where}[{index}]: malformed f32 word")
        if not math.isfinite(struct.unpack(">f", bytes.fromhex(word))[0]):
            _fail(f"{where}[{index}]: nonfinite f32 word")


def _tensor(record: object, shape: list[int], where: str) -> None:
    if not isinstance(record, dict) or set(record) != {"data", "encoding", "shape"}:
        _fail(f"{where}: tensor fields differ from version 1")
    if record["encoding"] != "ieee754-f32-hex-be" or record["shape"] != shape:
        _fail(f"{where}: tensor encoding or shape mismatch")
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
    _f32_words([record["numerator_bits"]], 1, where + ".numerator_bits")
    _f32_words(record["per_token_loss_bits"], 2, where + ".per_token_loss_bits")
    return weight


def _evaluation(record: object, where: str) -> None:
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
    if _exact_int(record["batch_count"], where + ".batch_count", 1) != len(batches):
        _fail(f"{where}: batch count mismatch")
    total = sum(weights)
    if record["weight"] != total or record["token_count"] != total:
        _fail(f"{where}: token/weight count mismatch")
    numerator = _finite(record["numerator"], where + ".numerator")
    mean = _finite(record["mean"], where + ".mean")
    if numerator <= 0 or mean <= 0 or not math.isclose(mean, numerator / total, rel_tol=0, abs_tol=1e-15):
        _fail(f"{where}: invalid global numerator/weight mean")
    for field in ("mean_bits_f64", "numerator_bits_after_f64_sum"):
        word = record[field]
        if not isinstance(word, str) or len(word) != 16 or any(ch not in "0123456789abcdef" for ch in word):
            _fail(f"{where}.{field}: malformed f64 word")


def validate_trajectory(payload: object) -> None:
    expected = {
        "entries", "format", "kind", "logical_alias", "parameter_paths",
        "parameter_shapes", "profile", "version",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        _fail("trajectory fields differ from version 1")
    if (payload["format"], payload["version"], payload["kind"], payload["profile"]) != (
        FORMAT, VERSION, "complete-training-trajectory", PROFILE
    ):
        _fail("trajectory identity mismatch")
    if payload["parameter_paths"] != list(PATHS) or payload["parameter_shapes"] != SHAPES:
        _fail("trajectory parameter contract mismatch")
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
        if entry["completed_updates"] != update:
            _fail(f"trajectory.entries[{update}]: nonconsecutive update")
        if list(entry["parameters"]) != list(PATHS) or list(entry["optimizer_state"]) != list(PATHS):
            _fail(f"trajectory.entries[{update}]: parameter order/path mismatch")
        _evaluation(entry["training_evaluation"], f"trajectory.entries[{update}].training_evaluation")
        for path in PATHS:
            _tensor(entry["parameters"][path], SHAPES[path], f"trajectory.entries[{update}].parameters.{path}")
            state = entry["optimizer_state"][path]
            if not isinstance(state, dict) or set(state) != {"exp_avg", "exp_avg_sq", "step"} or state["step"] != update:
                _fail(f"trajectory.entries[{update}].optimizer_state.{path}: invalid state")
            _tensor(state["exp_avg"], SHAPES[path], f"trajectory.entries[{update}].optimizer_state.{path}.exp_avg")
            _tensor(state["exp_avg_sq"], SHAPES[path], f"trajectory.entries[{update}].optimizer_state.{path}.exp_avg_sq")


def validate_corpora(payload: object, root: Path) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "datasets", "format", "kind", "profile", "resume_cases", "tokenizer", "version"
    }:
        _fail("corpora fields differ from version 1")
    if (payload["format"], payload["version"], payload["kind"], payload["profile"]) != (
        FORMAT, VERSION, "corpora-and-cursors", PROFILE
    ):
        _fail("corpora identity mismatch")
    tokenizer = payload["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != {"fingerprint", "identity", "path", "sha256", "vocab_size"}:
        _fail("tokenizer record fields differ")
    if (
        tokenizer["path"] != "tokenizer.etok"
        or tokenizer["vocab_size"] != 256
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
    for name, record in datasets.items():
        if not isinstance(record, dict) or set(record) != {
            "batches", "end_cursor", "files", "manifest_trailer_sha256", "packing", "shards", "start_cursor"
        }:
            _fail(f"datasets.{name}: fields differ")
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
                "numerator_bits": "3f800000", "per_token_loss_bits": ["3f800000", "3f800000"],
            }
            _batch(shadow, f"datasets.{name}.batches[{index}]")
        cursor_fields = {}
        for cursor_name in ("start_cursor", "end_cursor"):
            word = record[cursor_name]
            if not isinstance(word, str) or len(word) % 2 or any(ch not in "0123456789abcdef" for ch in word):
                _fail(f"datasets.{name}.{cursor_name}: malformed cursor hex")
            fields = decode_reference_cursor(bytes.fromhex(word))
            cursor_fields[cursor_name] = fields
            if fields.vocab_size != 256 or fields.tokenizer_fingerprint != tokenizer["fingerprint"]:
                _fail(f"datasets.{name}.{cursor_name}: cursor identity mismatch")
        if cursor_fields["start_cursor"].next_ordinal != 0 or cursor_fields["end_cursor"].next_ordinal != cursor_fields["end_cursor"].total_rows:
            _fail(f"datasets.{name}: cursor endpoints mismatch")
        _sha(record["manifest_trailer_sha256"], f"datasets.{name}.manifest_trailer_sha256")
        files = record["files"]
        if not isinstance(files, list) or not files:
            _fail(f"datasets.{name}.files: expected nonempty list")
        for item in files:
            if not isinstance(item, dict) or set(item) != {"bytes", "path", "sha256"}:
                _fail(f"datasets.{name}.files: malformed record")
            if not isinstance(item["path"], str) or "/" in item["path"] or "\\" in item["path"]:
                _fail(f"datasets.{name}.files: unsafe path")
            path = root / name / item["path"]
            size = _exact_int(item["bytes"], "file.bytes", 1, 1_048_576)
            if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != _sha(item["sha256"], "file.sha256"):
                _fail(f"datasets.{name}.{item['path']}: file metadata mismatch")
    cases = payload["resume_cases"]
    if not isinstance(cases, dict) or set(cases) != {"accumulation-1", "accumulation-2-unequal-mask"}:
        _fail("resume case set mismatch")
    for name, expected_a, expected_weights in (
        ("accumulation-1", 1, [2]),
        ("accumulation-2-unequal-mask", 2, [1, 2]),
    ):
        case = cases[name]
        if not isinstance(case, dict) or set(case) != {
            "K", "R", "accumulation_steps", "contributions", "end_cursor", "epoch_crossings",
            "epoch_start_cursor", "saved_cursor", "saved_next_ordinal", "total_weight",
        }:
            _fail(f"resume_cases.{name}: fields differ")
        if case["K"] != 1 or case["R"] != 3 or case["accumulation_steps"] != expected_a or case["epoch_crossings"] != 1:
            _fail(f"resume_cases.{name}: boundary contract mismatch")
        contributions = case["contributions"]
        if not isinstance(contributions, list) or len(contributions) != expected_a:
            _fail(f"resume_cases.{name}: contribution count mismatch")
        weights = []
        ordinals = []
        for index, contribution in enumerate(contributions):
            if not isinstance(contribution, dict) or set(contribution) != {"batch", "contribution_ordinal", "epoch"}:
                _fail(f"resume_cases.{name}.contributions[{index}]: fields differ")
            if contribution["epoch"] not in (0, 1):
                _fail(f"resume_cases.{name}.contributions[{index}]: invalid epoch")
            batch = contribution["batch"]
            if not isinstance(batch, dict) or set(batch) != {"inputs", "mask", "targets", "weight"}:
                _fail(f"resume_cases.{name}.contributions[{index}].batch: fields differ")
            shadow = batch | {
                "argmax": batch["targets"], "unique_argmax": [True, True],
                "numerator_bits": "3f800000", "per_token_loss_bits": ["3f800000", "3f800000"],
            }
            weights.append(_batch(shadow, f"resume_cases.{name}.contributions[{index}].batch"))
            ordinals.append(contribution["contribution_ordinal"])
        if weights != expected_weights or ordinals != list(range(expected_a)) or case["total_weight"] != sum(weights):
            _fail(f"resume_cases.{name}: contribution sequence mismatch")
        for cursor_name in ("epoch_start_cursor", "saved_cursor", "end_cursor"):
            decode_reference_cursor(bytes.fromhex(case[cursor_name]))


def validate_summary(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "accumulation_reference", "format", "heldout", "initializer", "kind", "optimizer",
        "profile", "provenance", "scope", "thresholds", "training", "updates", "version",
    }:
        _fail("summary fields differ from version 1")
    if (payload["format"], payload["version"], payload["kind"], payload["profile"]) != (
        FORMAT, VERSION, "acceptance-summary", PROFILE
    ) or payload["optimizer"] != OPTIMIZER or payload["updates"] != UPDATE_COUNT:
        _fail("summary identity/configuration mismatch")
    if not isinstance(payload["initializer"], dict) or set(payload["initializer"]) != {"seed", "successor"} or payload["initializer"]["seed"] != 1729:
        _fail("summary initializer mismatch")
    if not isinstance(payload["training"], dict) or set(payload["training"]) != {"before", "after"}:
        _fail("summary training fields differ")
    if not isinstance(payload["heldout"], dict) or set(payload["heldout"]) != {"before", "after", "train_transitions", "heldout_transitions"}:
        _fail("summary heldout fields differ")
    _evaluation(payload["training"]["before"], "summary.training.before")
    _evaluation(payload["training"]["after"], "summary.training.after")
    _evaluation(payload["heldout"]["before"], "summary.heldout.before")
    _evaluation(payload["heldout"]["after"], "summary.heldout.after")
    transition_sets = []
    for label in ("train_transitions", "heldout_transitions"):
        transitions = payload["heldout"][label]
        if not isinstance(transitions, list) or not transitions:
            _fail(f"summary.heldout.{label}: expected nonempty transitions")
        normalized = set()
        for index, pair in enumerate(transitions):
            if not isinstance(pair, list) or len(pair) != 2:
                _fail(f"summary.heldout.{label}[{index}]: expected token pair")
            normalized.add(tuple(_exact_int(token, f"summary.heldout.{label}[{index}]", 0, 255) for token in pair))
        transition_sets.append(normalized)
    if transition_sets[0] & transition_sets[1]:
        _fail("held-out transitions overlap training transitions")
    if payload["training"]["after"]["mean"] >= 0.05:
        _fail("final training mean CE threshold failed")
    if payload["heldout"]["after"]["mean"] >= payload["heldout"]["before"]["mean"]:
        _fail("held-out global loss did not decrease")
    proof = payload["accumulation_reference"]
    if not isinstance(proof, dict) or set(proof) != {
        "bit_identical", "combined_numerator_gradient_sha256", "contribution_ordinals",
        "max_abs_difference", "parameter_count", "summed_numerator_gradient_sha256",
        "total_weight", "weights", "within_parameter_tolerance",
        "wrong_per_batch_mean_max_abs_difference",
    }:
        _fail("accumulation proof fields differ")
    if proof["weights"] != [2, 1] or proof["total_weight"] != 3 or proof["parameter_count"] != 14:
        _fail("accumulation proof contract mismatch")
    if proof["within_parameter_tolerance"] is not True or proof["max_abs_difference"] > 2.0e-6:
        _fail("combined numerator gradient parity failed")
    if _finite(proof["wrong_per_batch_mean_max_abs_difference"], "wrong mean delta") <= 0:
        _fail("wrong mean-seed control did not differ")
    for field in ("summed_numerator_gradient_sha256", "combined_numerator_gradient_sha256"):
        _sha(proof[field], "accumulation_reference." + field)
    thresholds = payload["thresholds"]
    if thresholds != {"final_training_mean_ce_strict_upper": 0.05, "tolerances": TOLERANCES}:
        _fail("summary thresholds differ from frozen contract")
    scope = payload["scope"]
    if not isinstance(scope, dict) or scope.get("development_reference_only") is not True or any(
        scope.get(field) is not False
        for field in ("native_acceptance", "native_resume_proof", "checkpoint_format_or_api", "runtime_or_production_dependency")
    ):
        _fail("scope must remain development-only")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"environment", "proposal", "python", "source_sha256", "torch"}:
        _fail("provenance fields differ")
    if provenance["python"] != "3.14.6" or provenance["torch"] != "2.13.0+cpu" or provenance["environment"] != {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE"} or provenance["proposal"] != PROPOSAL:
        _fail("pinned provenance mismatch")
    for path, digest in provenance["source_sha256"].items():
        _sha(digest, "provenance.source_sha256")
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts or hashlib.sha256((Path(__file__).resolve().parents[2] / candidate).read_bytes()).hexdigest() != digest:
            _fail(f"provenance source mismatch: {path}")


def validate_output(root: Path, manifest: object) -> None:
    if not isinstance(manifest, dict) or set(manifest) != {
        "artifacts", "configuration", "format", "measurements", "profile", "provenance", "scope", "version"
    }:
        _fail("manifest fields differ from version 1")
    if manifest["format"] != "eshkol-tr3-training-reference-manifest" or manifest["version"] != VERSION or manifest["profile"] != PROFILE:
        _fail("manifest identity mismatch")
    if manifest["configuration"] != {"initializer_seed": 1729, "optimizer": OPTIMIZER, "updates": UPDATE_COUNT}:
        _fail("manifest configuration mismatch")
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
    validate_corpora(payloads["corpora.json"], root)
    validate_trajectory(payloads["trajectory.json"])
    validate_summary(payloads["summary.json"])
    if payloads["summary.json"]["provenance"] != manifest["provenance"] or payloads["summary.json"]["scope"] != manifest["scope"]:
        _fail("manifest summary provenance/scope mismatch")
    expected_measurements = {
        "initial_training_mean_ce": payloads["summary.json"]["training"]["before"]["mean"],
        "final_training_mean_ce": payloads["summary.json"]["training"]["after"]["mean"],
        "final_active_targets": [197, 41],
        "final_active_argmax": [197, 41],
        "final_active_unique_argmax": [True, True],
        "initial_heldout_global_mean_ce": payloads["summary.json"]["heldout"]["before"]["mean"],
        "final_heldout_global_mean_ce": payloads["summary.json"]["heldout"]["after"]["mean"],
        "heldout_weight": 3,
        "heldout_batches": 2,
        "heldout_mask_weights": [2, 1],
    }
    if manifest["measurements"] != expected_measurements:
        _fail("manifest measurements differ from generated summary")
