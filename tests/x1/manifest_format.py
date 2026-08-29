"""Independent reader for the X1 canonical resolved-run manifest.

This module is development-only.  Production configuration parsing and hashing
remain in Eshkol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FORMAT = "eshkol-resolved-run"
FORMAT_VERSION = [1, 0]
SCHEMA_VERSION = [1, 0]
CANONICALIZATION = "eshkol-config-json-v1"
MAX_CANONICAL_BYTES = 16_384

ROOT_KEYS = {
    "canonicalization",
    "checksum-algorithm",
    "checksum-coverage",
    "config-schema-version",
    "format",
    "format-version",
    "limits",
    "provenance",
    "required-features",
    "resolved",
}
CONFIG_KEYS = {
    "config-schema-major",
    "config-schema-minor",
    "model.context-length",
    "model.device",
    "model.dtype",
    "model.head-size",
    "model.hidden-size",
    "model.kv-head-count",
    "model.layer-count",
    "model.query-head-count",
    "model.vocabulary-size",
    "run.deterministic",
    "run.seed",
    "training.accumulation-steps",
}
LIMITS = {
    "integer-digits": 19,
    "max-input-bytes": 16_384,
    "max-input-keys": 14,
    "max-input-nesting-depth": 1,
}
PROVENANCE_VALUES = {"default", "derived", "input", "override"}
PROVENANCE_BY_KEY = {
    "config-schema-major": {"input"},
    "config-schema-minor": {"input"},
    "model.context-length": {"input", "override"},
    "model.device": {"default", "input", "override"},
    "model.dtype": {"default", "input", "override"},
    "model.head-size": {"derived", "input"},
    "model.hidden-size": {"input", "override"},
    "model.kv-head-count": {"default", "input", "override"},
    "model.layer-count": {"input", "override"},
    "model.query-head-count": {"input", "override"},
    "model.vocabulary-size": {"input", "override"},
    "run.deterministic": {"default", "input", "override"},
    "run.seed": {"input", "override"},
    "training.accumulation-steps": {"default", "input", "override"},
}


class ManifestError(ValueError):
    """Raised when resolved-run bytes violate the public X1 contract."""


def _reject_float(_: str) -> None:
    raise ManifestError("JSON floating-point numbers are forbidden")


def _parse_int(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    if len(digits) > 19:
        raise ManifestError("integer lexical width exceeds 19 digits")
    value = int(token)
    if not -(2**63) <= value <= 2**63 - 1:
        raise ManifestError("integer is outside signed 64-bit range")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate key: {key}")
        result[key] = value
    return result


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ManifestError(f"{where} keys differ: missing={missing}, unknown={unknown}")
    return value


def _version(value: Any, expected: list[int], where: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value != expected
    ):
        raise ManifestError(f"unsupported {where}: {value!r}")


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def load_manifest_bytes(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_CANONICAL_BYTES:
        raise ManifestError("canonical manifest exceeds 16384 bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ManifestError("UTF-8 BOM is forbidden")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ManifestError("manifest must end in exactly one LF")
    if b"\r" in raw or any(byte >= 0x80 for byte in raw):
        raise ManifestError("version 1 canonical bytes must be ASCII with LF newlines")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_int=_parse_int,
            parse_float=_reject_float,
            parse_constant=_reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("invalid UTF-8 JSON") from exc
    root = _exact_keys(value, ROOT_KEYS, "root")
    if raw != canonical_bytes(root):
        raise ManifestError("bytes are not canonical")
    if root["canonicalization"] != CANONICALIZATION:
        raise ManifestError("unsupported canonicalization")
    if root["checksum-algorithm"] != "sha256":
        raise ManifestError("unsupported checksum algorithm")
    if root["checksum-coverage"] != "whole-document-including-final-lf":
        raise ManifestError("unsupported checksum coverage")
    if root["format"] != FORMAT:
        raise ManifestError("unsupported format")
    _version(root["format-version"], FORMAT_VERSION, "format version")
    _version(root["config-schema-version"], SCHEMA_VERSION, "config schema version")
    if root["required-features"] != []:
        raise ManifestError("version 1.0 required-features must be empty")
    limits = _exact_keys(root["limits"], set(LIMITS), "limits")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in limits.values()):
        raise ManifestError("limits must be exact integers")
    if limits != LIMITS:
        raise ManifestError("limits do not match version 1.0")

    config = _exact_keys(root["resolved"], CONFIG_KEYS, "resolved")
    provenance = _exact_keys(root["provenance"], CONFIG_KEYS, "provenance")
    if any(
        not isinstance(source, str) or source not in PROVENANCE_VALUES
        for source in provenance.values()
    ):
        raise ManifestError("unknown provenance source")
    for key, source in provenance.items():
        if source not in PROVENANCE_BY_KEY[key]:
            raise ManifestError(f"impossible provenance for {key}: {source}")
    for key, value in config.items():
        if key == "config-schema-major":
            if isinstance(value, bool) or not isinstance(value, int) or value != 1:
                raise ManifestError("config-schema-major must be 1")
        elif key == "config-schema-minor":
            if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                raise ManifestError("config-schema-minor must be 0")
        elif key == "run.deterministic":
            if value is not True:
                raise ManifestError("run.deterministic must be true")
        elif key == "model.dtype":
            if value != "f32":
                raise ManifestError("only f32 is admitted by schema 1.0")
        elif key == "model.device":
            if value != "cpu":
                raise ManifestError("only cpu is admitted by schema 1.0")
        elif isinstance(value, bool) or not isinstance(value, int):
            raise ManifestError(f"{key} must be an integer")
    if config["run.seed"] < 0:
        raise ManifestError("run.seed must be nonnegative")
    for key in CONFIG_KEYS - {
        "config-schema-major",
        "config-schema-minor",
        "run.seed",
        "run.deterministic",
        "model.dtype",
        "model.device",
    }:
        if config[key] < 1:
            raise ManifestError(f"{key} must be positive")
    if config["model.hidden-size"] != (
        config["model.query-head-count"] * config["model.head-size"]
    ):
        raise ManifestError("hidden/head-size combination is incompatible")
    if config["model.query-head-count"] % config["model.kv-head-count"]:
        raise ManifestError("query heads must be divisible by kv heads")
    declared_defaults = {
        "model.device": "cpu",
        "model.dtype": "f32",
        "model.kv-head-count": config["model.query-head-count"],
        "run.deterministic": True,
        "training.accumulation-steps": 1,
    }
    for key, default in declared_defaults.items():
        if provenance[key] == "default" and config[key] != default:
            raise ManifestError(f"default provenance disagrees with {key}")
    return root


def load_manifest(path: Path) -> dict[str, Any]:
    return load_manifest_bytes(path.read_bytes())


def fingerprint(raw: bytes) -> str:
    return f"sha256:{CANONICALIZATION}:{hashlib.sha256(raw).hexdigest()}"
