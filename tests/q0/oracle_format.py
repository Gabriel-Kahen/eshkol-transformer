"""Frozen, non-executable Q0 oracle fixture format (version 1)."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import struct
import unicodedata
from pathlib import Path
from typing import Any, Mapping, NoReturn

FORMAT_NAME = "eshkol-oracle"
FORMAT_VERSION = 1
MAX_FIXTURE_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 32

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HEX = re.compile(r"[0-9a-f]+\Z")
_KINDS = frozenset(
    {
        "known_value",
        "parity",
        "gradient",
        "repeated_input",
        "special_value",
        "boundary",
        "malformed",
    }
)
_ROLES = frozenset({"input", "expected", "analytic_gradient"})
_DTYPES = {
    "bool": ("bool01", 1),
    "int64": ("twos-complement-hex-be", 16),
    "float32": ("ieee754-hex-be", 8),
    "float64": ("ieee754-hex-be", 16),
}


class FixtureError(ValueError):
    """Malformed, corrupt, noncanonical, or unsupported fixture."""


def _fail(message: str) -> NoReturn:
    raise FixtureError(message)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{where}: expected object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{where}: expected array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str):
        _fail(f"{where}: expected string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise FixtureError(f"{where}: string is not valid Unicode scalar text") from error
    if unicodedata.normalize("NFC", value) != value:
        _fail(f"{where}: string is not Unicode NFC")
    return value


def _name(value: Any, where: str) -> str:
    result = _string(value, where)
    if _NAME.fullmatch(result) is None:
        _fail(f"{where}: invalid identifier")
    return result


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{where}: expected integer")
    if value < minimum or value > 2**63 - 1:
        _fail(f"{where}: integer outside [{minimum}, 2^63-1]")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{where}: keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate object key: {key}")
        result[key] = value
    return result


def _forbid_float(token: str) -> NoReturn:
    _fail(f"JSON floating-point number is forbidden: {token}")


def _forbid_constant(token: str) -> NoReturn:
    _fail(f"non-standard JSON number is forbidden: {token}")


def _validate_json(value: Any, where: str = "$", depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        _fail(f"{where}: JSON nesting exceeds {MAX_JSON_DEPTH}")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            _fail(f"{where}: integer outside signed 64-bit range")
        return
    if isinstance(value, float):
        _fail(f"{where}: JSON floating-point values are forbidden")
    if isinstance(value, str):
        _string(value, where)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{where}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _string(key, f"{where}.<key>")
            _validate_json(item, f"{where}.{key}", depth + 1)
        return
    _fail(f"{where}: unsupported JSON type {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON without a trailing LF."""
    _validate_json(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeError) as error:
        raise FixtureError("fixture contains text that cannot be encoded as UTF-8") from error


def _count(shape: list[Any], where: str) -> int:
    result = 1
    for index, raw_dimension in enumerate(shape):
        dimension = _integer(raw_dimension, f"{where}[{index}]")
        if dimension > 2**31 - 1:
            _fail(f"{where}[{index}]: dimension exceeds 2^31-1")
        result *= dimension
        if result > 2**63 - 1:
            _fail(f"{where}: element count overflow")
    return result


def _validate_tensor(raw: Any, where: str) -> tuple[str, str, str]:
    tensor = _object(raw, where)
    _keys(
        tensor,
        {"data", "device", "dtype", "encoding", "layout", "name", "role", "shape"},
        where,
    )
    name = _name(tensor["name"], f"{where}.name")
    role = _string(tensor["role"], f"{where}.role")
    if role not in _ROLES:
        _fail(f"{where}.role: unsupported role {role!r}")
    if tensor["device"] != "cpu":
        _fail(f"{where}.device: version 1 supports only explicit cpu")
    if tensor["layout"] != "row_major":
        _fail(f"{where}.layout: version 1 supports only row_major")
    dtype = _string(tensor["dtype"], f"{where}.dtype")
    if dtype not in _DTYPES:
        _fail(f"{where}.dtype: unsupported dtype {dtype!r}")
    encoding, width = _DTYPES[dtype]
    if tensor["encoding"] != encoding:
        _fail(f"{where}.encoding: dtype {dtype!r} requires {encoding!r}")
    shape = _array(tensor["shape"], f"{where}.shape")
    expected_count = _count(shape, f"{where}.shape")
    data = _array(tensor["data"], f"{where}.data")
    if len(data) != expected_count:
        _fail(
            f"{where}.data: count {len(data)} does not match shape count "
            f"{expected_count}"
        )
    for index, raw_element in enumerate(data):
        element = _string(raw_element, f"{where}.data[{index}]")
        if len(element) != width:
            _fail(f"{where}.data[{index}]: expected encoded width {width}")
        if dtype == "bool":
            if element not in {"0", "1"}:
                _fail(f"{where}.data[{index}]: bool must be 0 or 1")
        elif _HEX.fullmatch(element) is None:
            _fail(f"{where}.data[{index}]: expected lowercase hexadecimal")
    return name, role, dtype


def _validate_tolerance(raw: Any, where: str) -> tuple[float, float, bool]:
    tolerance = _object(raw, where)
    _keys(tolerance, {"absolute", "equal_nan", "relative"}, where)
    if not isinstance(tolerance["equal_nan"], bool):
        _fail(f"{where}.equal_nan: expected boolean")
    decoded: dict[str, float] = {}
    for field in ("absolute", "relative"):
        bits = _string(tolerance[field], f"{where}.{field}")
        if len(bits) != 16 or _HEX.fullmatch(bits) is None:
            _fail(f"{where}.{field}: expected 16 lowercase hexadecimal digits")
        number = struct.unpack(">d", bytes.fromhex(bits))[0]
        if not math.isfinite(number) or number < 0.0:
            _fail(f"{where}.{field}: expected finite non-negative float64")
        decoded[field] = number
    return decoded["absolute"], decoded["relative"], tolerance["equal_nan"]


def _validate_generator(raw: Any) -> None:
    where = "$.payload.generator"
    generator = _object(raw, where)
    _keys(
        generator,
        {
            "dependency_lock_sha256",
            "framework",
            "name",
            "seed",
            "source_sha256",
            "version",
        },
        where,
    )
    _name(generator["name"], f"{where}.name")
    _integer(generator["version"], f"{where}.version", 1)
    _integer(generator["seed"], f"{where}.seed")
    for field in ("dependency_lock_sha256", "source_sha256"):
        digest = _string(generator[field], f"{where}.{field}")
        if _SHA256.fullmatch(digest) is None:
            _fail(f"{where}.{field}: invalid SHA-256")
    framework = _object(generator["framework"], f"{where}.framework")
    _keys(framework, {"name", "version"}, f"{where}.framework")
    _name(framework["name"], f"{where}.framework.name")
    version = _string(framework["version"], f"{where}.framework.version")
    if not version or len(version) > 64:
        _fail(f"{where}.framework.version: invalid version")


def validate_payload(raw: Any) -> dict[str, Any]:
    """Validate a version-1 payload without executing generator metadata."""
    payload = _object(raw, "$.payload")
    _keys(payload, {"cases", "generator", "tensors"}, "$.payload")
    _validate_generator(payload["generator"])

    tensors = _array(payload["tensors"], "$.payload.tensors")
    tensor_records = [
        _validate_tensor(tensor, f"$.payload.tensors[{index}]")
        for index, tensor in enumerate(tensors)
    ]
    tensor_names = [name for name, _role, _dtype in tensor_records]
    if tensor_names != sorted(tensor_names):
        _fail("$.payload.tensors: records must be sorted by name")
    if len(tensor_names) != len(set(tensor_names)):
        _fail("$.payload.tensors: duplicate tensor name")
    available = set(tensor_names)
    tensor_roles = {name: role for name, role, _dtype in tensor_records}
    tensor_dtypes = {name: dtype for name, _role, dtype in tensor_records}

    cases = _array(payload["cases"], "$.payload.cases")
    case_names: list[str] = []
    for index, raw_case in enumerate(cases):
        where = f"$.payload.cases[{index}]"
        case = _object(raw_case, where)
        _keys(
            case,
            {"expectation", "inputs", "kind", "name", "operation", "tolerance"},
            where,
        )
        case_names.append(_name(case["name"], f"{where}.name"))
        kind = _string(case["kind"], f"{where}.kind")
        if kind not in _KINDS:
            _fail(f"{where}.kind: unsupported kind {kind!r}")
        _name(case["operation"], f"{where}.operation")
        raw_inputs = _array(case["inputs"], f"{where}.inputs")
        inputs = [
            _name(reference, f"{where}.inputs[{input_index}]")
            for input_index, reference in enumerate(raw_inputs)
        ]
        if len(inputs) != len(set(inputs)):
            _fail(f"{where}.inputs: duplicate tensor reference")
        for input_index, reference in enumerate(inputs):
            if reference not in available:
                _fail(f"{where}.inputs[{input_index}]: unknown tensor {reference!r}")
            if tensor_roles[reference] != "input":
                _fail(
                    f"{where}.inputs[{input_index}]: tensor {reference!r} has role "
                    f"{tensor_roles[reference]!r}, expected 'input'"
                )

        expectation = _object(case["expectation"], f"{where}.expectation")
        _keys(expectation, {"error", "outputs"}, f"{where}.expectation")
        raw_outputs = _array(
            expectation["outputs"], f"{where}.expectation.outputs"
        )
        outputs = [
            _name(reference, f"{where}.expectation.outputs[{output_index}]")
            for output_index, reference in enumerate(raw_outputs)
        ]
        if len(outputs) != len(set(outputs)):
            _fail(f"{where}.expectation.outputs: duplicate tensor reference")
        for output_index, reference in enumerate(outputs):
            if reference not in available:
                _fail(
                    f"{where}.expectation.outputs[{output_index}]: unknown tensor "
                    f"{reference!r}"
                )
            required_role = "analytic_gradient" if kind == "gradient" else "expected"
            if tensor_roles[reference] != required_role:
                _fail(
                    f"{where}.expectation.outputs[{output_index}]: tensor "
                    f"{reference!r} has role {tensor_roles[reference]!r}, expected "
                    f"{required_role!r} for kind {kind!r}"
                )
        error = expectation["error"]
        if error is None:
            if not outputs:
                _fail(f"{where}.expectation: success requires output tensors")
        else:
            error = _string(error, f"{where}.expectation.error")
            if not error or outputs:
                _fail(f"{where}.expectation: error requires text and no outputs")
        absolute, relative, equal_nan = _validate_tolerance(
            case["tolerance"], f"{where}.tolerance"
        )
        for reference in outputs:
            if tensor_dtypes[reference] in {"bool", "int64"} and (
                absolute != 0.0 or relative != 0.0 or equal_nan
            ):
                _fail(
                    f"{where}.tolerance: integer/bool output {reference!r} "
                    "requires zero tolerance and equal_nan=false"
                )

    if case_names != sorted(case_names):
        _fail("$.payload.cases: records must be sorted by name")
    if len(case_names) != len(set(case_names)):
        _fail("$.payload.cases: duplicate case name")
    return payload


def _integrity_object(format_name: Any, version: Any, payload: Any) -> dict[str, Any]:
    return {"format": format_name, "payload": payload, "version": version}


def encode_fixture(payload: Mapping[str, Any]) -> bytes:
    """Validate and encode one checksummed canonical fixture."""
    validated = validate_payload(dict(payload))
    integrity = _integrity_object(FORMAT_NAME, FORMAT_VERSION, validated)
    digest = hashlib.sha256(canonical_json(integrity)).hexdigest()
    envelope = {
        "checksum": {"algorithm": "sha256", "digest": digest},
        **integrity,
    }
    return canonical_json(envelope) + b"\n"


def load_fixture(source: bytes | bytearray | memoryview | str | Path) -> dict[str, Any]:
    """Load strict data-only fixture bytes and reject corruption."""
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if len(raw) > MAX_FIXTURE_BYTES:
        _fail(f"fixture exceeds {MAX_FIXTURE_BYTES} bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail("fixture must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise FixtureError("fixture is not valid UTF-8") from error
    try:
        envelope = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_forbid_float,
            parse_constant=_forbid_constant,
        )
    except json.JSONDecodeError as error:
        raise FixtureError(f"invalid JSON: {error.msg}") from error
    except RecursionError as error:
        raise FixtureError("fixture JSON nesting exceeds parser limit") from error
    envelope = _object(envelope, "$")
    if raw != canonical_json(envelope) + b"\n":
        _fail("fixture bytes are not canonical")
    _keys(envelope, {"checksum", "format", "payload", "version"}, "$")
    checksum = _object(envelope["checksum"], "$.checksum")
    _keys(checksum, {"algorithm", "digest"}, "$.checksum")
    if checksum["algorithm"] != "sha256":
        _fail(f"unsupported checksum algorithm {checksum['algorithm']!r}")
    expected = _string(checksum["digest"], "$.checksum.digest")
    if _SHA256.fullmatch(expected) is None:
        _fail("$.checksum.digest: invalid SHA-256")
    integrity = _integrity_object(
        envelope["format"], envelope["version"], envelope["payload"]
    )
    actual = hashlib.sha256(canonical_json(integrity)).hexdigest()
    if not hmac.compare_digest(expected, actual):
        _fail("fixture checksum mismatch")
    format_name = _string(envelope["format"], "$.format")
    version = _integer(envelope["version"], "$.version", 1)
    if format_name != FORMAT_NAME:
        _fail(f"unsupported fixture format {format_name!r}")
    if version != FORMAT_VERSION:
        _fail(f"unsupported fixture version {version!r}")
    return validate_payload(envelope["payload"])


def decode_tensor(tensor: Mapping[str, Any]) -> tuple[Any, ...]:
    """Decode one validated tensor record into immutable scalar values."""
    checked = dict(tensor)
    _validate_tensor(checked, "$.tensor")
    dtype = checked["dtype"]
    values: list[Any] = []
    for encoded in checked["data"]:
        if dtype == "bool":
            values.append(encoded == "1")
        elif dtype == "int64":
            values.append(int.from_bytes(bytes.fromhex(encoded), "big", signed=True))
        elif dtype == "float32":
            values.append(struct.unpack(">f", bytes.fromhex(encoded))[0])
        else:
            values.append(struct.unpack(">d", bytes.fromhex(encoded))[0])
    return tuple(values)


def tensor_by_name(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Find a tensor by its stable name."""
    for tensor in payload["tensors"]:
        if tensor["name"] == name:
            return tensor
    _fail(f"unknown tensor {name!r}")
