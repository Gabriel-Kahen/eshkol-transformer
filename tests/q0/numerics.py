"""Reusable comparison and central finite-difference helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


class NumericalContractError(ValueError):
    """Shape, dtype, device, policy, or objective contract violation."""


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class ComparisonPolicy:
    atol: float
    rtol: float
    equal_nan: bool = False


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    actual: TensorMetadata
    expected: TensorMetadata
    policy: ComparisonPolicy
    mismatch_count: int
    first_failure_flat_index: int | None
    first_failure_index: tuple[int, ...] | None
    max_absolute_error: float
    max_relative_error: float
    error: str | None


@dataclass(frozen=True)
class FiniteDifferenceResult:
    gradient: tuple[float, ...]
    steps: tuple[float, ...]
    metadata: TensorMetadata
    scheme: str
    error: str | None


def element_count(shape: Sequence[int]) -> int:
    result = 1
    for index, dimension in enumerate(shape):
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise NumericalContractError(f"shape[{index}] is not an integer")
        if dimension < 0:
            raise NumericalContractError(f"shape[{index}] is negative")
        result *= dimension
    return result


def _multi_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    result = [0] * len(shape)
    remaining = flat_index
    for axis in range(len(shape) - 1, -1, -1):
        dimension = shape[axis]
        result[axis] = remaining % dimension
        remaining //= dimension
    return tuple(result)


def _validate_metadata(metadata: TensorMetadata) -> None:
    if metadata.dtype not in {"bool", "int64", "float32", "float64"}:
        raise NumericalContractError(f"unsupported dtype {metadata.dtype!r}")
    if metadata.device not in {"cpu", "cuda", "hip"}:
        raise NumericalContractError(f"unsupported device {metadata.device!r}")
    element_count(metadata.shape)


def _validate_runtime_value(value: object, dtype: str, where: str) -> None:
    if dtype == "bool":
        if type(value) is not bool:
            raise NumericalContractError(f"{where} must be a bool for dtype bool")
        return
    if dtype == "int64":
        if type(value) is not int:
            raise NumericalContractError(f"{where} must be an int for dtype int64")
        if value < -(2**63) or value > 2**63 - 1:
            raise NumericalContractError(f"{where} is outside signed int64 range")
        return
    if type(value) is not float:
        raise NumericalContractError(f"{where} must be a float for dtype {dtype}")


def compare_values(
    actual_values: Sequence[float | int | bool],
    expected_values: Sequence[float | int | bool],
    *,
    actual: TensorMetadata,
    expected: TensorMetadata,
    policy: ComparisonPolicy,
) -> ComparisonResult:
    """Compare flat tensors without implicit cast, reshape, or device transfer."""
    _validate_metadata(actual)
    _validate_metadata(expected)
    if (actual.shape, actual.dtype, actual.device) != (
        expected.shape,
        expected.dtype,
        expected.device,
    ):
        raise NumericalContractError(
            "metadata mismatch: "
            f"actual={actual.shape}/{actual.dtype}/{actual.device}, "
            f"expected={expected.shape}/{expected.dtype}/{expected.device}"
        )
    for field, value in (("atol", policy.atol), ("rtol", policy.rtol)):
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise NumericalContractError(f"{field} must be finite and non-negative")
    if type(policy.equal_nan) is not bool:
        raise NumericalContractError("equal_nan must be a bool")
    required = element_count(expected.shape)
    if len(actual_values) != required or len(expected_values) != required:
        raise NumericalContractError(
            f"value count mismatch: shape requires {required}, "
            f"actual={len(actual_values)}, expected={len(expected_values)}"
        )
    if expected.dtype in {"bool", "int64"} and (
        policy.atol != 0.0 or policy.rtol != 0.0
    ):
        raise NumericalContractError("integer/bool comparison requires zero tolerance")
    if expected.dtype in {"bool", "int64"} and policy.equal_nan:
        raise NumericalContractError("integer/bool comparison forbids equal_nan")

    for index, value in enumerate(actual_values):
        _validate_runtime_value(value, actual.dtype, f"actual_values[{index}]")
    for index, value in enumerate(expected_values):
        _validate_runtime_value(value, expected.dtype, f"expected_values[{index}]")

    mismatch_count = 0
    first_flat: int | None = None
    max_absolute = 0.0
    max_relative = 0.0
    error: str | None = None
    for index, (observed_raw, reference_raw) in enumerate(
        zip(actual_values, expected_values)
    ):
        if expected.dtype in {"bool", "int64"}:
            passed = observed_raw == reference_raw
            absolute = 0.0 if passed else math.inf
            relative = absolute
        else:
            observed = float(observed_raw)
            reference = float(reference_raw)
            if math.isnan(observed) or math.isnan(reference):
                passed = (
                    policy.equal_nan
                    and math.isnan(observed)
                    and math.isnan(reference)
                )
                absolute = 0.0 if passed else math.inf
                relative = absolute
            elif math.isinf(observed) or math.isinf(reference):
                passed = observed == reference
                absolute = 0.0 if passed else math.inf
                relative = absolute
            else:
                absolute = abs(observed - reference)
                relative = (
                    absolute / abs(reference)
                    if reference != 0.0
                    else (0.0 if absolute == 0.0 else math.inf)
                )
                passed = absolute <= policy.atol + policy.rtol * abs(reference)
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)
        if not passed:
            mismatch_count += 1
            if first_flat is None:
                first_flat = index
                error = (
                    f"{expected.name}[{index}] mismatch on "
                    f"{expected.device}/{expected.dtype}: actual={observed_raw!r}, "
                    f"expected={reference_raw!r}, absolute_error={absolute!r}, "
                    f"relative_error={relative!r}"
                )
    return ComparisonResult(
        passed=mismatch_count == 0,
        actual=actual,
        expected=expected,
        policy=policy,
        mismatch_count=mismatch_count,
        first_failure_flat_index=first_flat,
        first_failure_index=(
            None if first_flat is None else _multi_index(first_flat, expected.shape)
        ),
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
        error=error,
    )


def central_finite_difference(
    function: Callable[[tuple[float, ...]], float],
    point: Sequence[float],
    *,
    epsilon: float = 1.0e-6,
    name: str = "gradient",
    dtype: str = "float64",
    device: str = "cpu",
) -> FiniteDifferenceResult:
    """Two-sided CPU finite difference for a scalar objective.

    Each coordinate uses h_i = epsilon * max(1, abs(x_i)). This development helper
    is not an autodiff fallback.
    """
    if dtype != "float64":
        raise NumericalContractError(
            "finite differences support only float64 Python arithmetic"
        )
    if device != "cpu":
        raise NumericalContractError(
            "finite differences are CPU-only; no device fallback is performed"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise NumericalContractError("epsilon must be finite and positive")
    base_values: list[float] = []
    for index, value in enumerate(point):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NumericalContractError(f"point[{index}] must be a real scalar")
        try:
            base_values.append(float(value))
        except (TypeError, ValueError, OverflowError) as error:
            raise NumericalContractError(f"point[{index}] must be finite") from error
    base = tuple(base_values)
    if not all(math.isfinite(value) for value in base):
        raise NumericalContractError("point values must be finite")

    def evaluate(values: tuple[float, ...], dimension: int | None) -> float:
        location = "empty point" if dimension is None else f"dimension {dimension}"
        try:
            raw = function(values)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("objective result is not a real scalar")
            result = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise NumericalContractError(
                f"objective is not a finite scalar at {location}"
            ) from error
        if not math.isfinite(result):
            raise NumericalContractError(
                f"objective returned non-finite value at {location}"
            )
        return result

    if not base:
        evaluate(base, None)

    gradient: list[float] = []
    steps: list[float] = []
    for index, value in enumerate(base):
        step = epsilon * max(1.0, abs(value))
        if not math.isfinite(step) or step <= 0.0:
            raise NumericalContractError(
                f"finite-difference step is non-finite at dimension {index}"
            )
        plus = list(base)
        minus = list(base)
        plus[index] = value + step
        minus[index] = value - step
        if not math.isfinite(plus[index]) or not math.isfinite(minus[index]):
            raise NumericalContractError(
                f"finite-difference perturbation is non-finite at dimension {index}"
            )
        if plus[index] == value or minus[index] == value or plus[index] == minus[index]:
            raise NumericalContractError(
                f"finite-difference perturbation is not representable at dimension {index}"
            )
        high = evaluate(tuple(plus), index)
        low = evaluate(tuple(minus), index)
        derivative = (high - low) / (plus[index] - minus[index])
        if not math.isfinite(derivative):
            raise NumericalContractError(
                f"finite-difference gradient is non-finite at dimension {index}"
            )
        gradient.append(derivative)
        steps.append(step)
    return FiniteDifferenceResult(
        gradient=tuple(gradient),
        steps=tuple(steps),
        metadata=TensorMetadata(name, (len(base),), dtype, device),
        scheme="central-2point-scaled",
        error=None,
    )
