from __future__ import annotations

import math
import unittest

from tests.q0.numerics import (
    ComparisonPolicy,
    NumericalContractError,
    TensorMetadata,
    central_finite_difference,
    compare_values,
)


class ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = TensorMetadata("value", (2,), "float64", "cpu")

    def test_known_value_and_tolerance(self) -> None:
        result = compare_values(
            [1.0, 2.0000001],
            [1.0, 2.0],
            actual=self.metadata,
            expected=self.metadata,
            policy=ComparisonPolicy(atol=1.0e-6, rtol=0.0),
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.mismatch_count, 0)
        self.assertIsNone(result.error)

    def test_failure_has_shape_dtype_device_and_index_metadata(self) -> None:
        actual = TensorMetadata("actual", (2, 2), "float32", "cpu")
        expected = TensorMetadata("expected", (2, 2), "float32", "cpu")
        result = compare_values(
            [1.0, 2.0, 7.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            actual=actual,
            expected=expected,
            policy=ComparisonPolicy(0.0, 0.0),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.mismatch_count, 1)
        self.assertEqual(result.first_failure_flat_index, 2)
        self.assertEqual(result.first_failure_index, (1, 0))
        self.assertIn("cpu/float32", result.error)

    def test_nan_inf_zero_and_empty_policies(self) -> None:
        special = TensorMetadata("special", (4,), "float64", "cpu")
        result = compare_values(
            [math.nan, math.inf, -math.inf, 0.0],
            [math.nan, math.inf, -math.inf, -0.0],
            actual=special,
            expected=special,
            policy=ComparisonPolicy(0.0, 0.0, equal_nan=True),
        )
        self.assertTrue(result.passed)
        without_nan = compare_values(
            [math.nan], [math.nan],
            actual=TensorMetadata("nan", (1,), "float64", "cpu"),
            expected=TensorMetadata("nan", (1,), "float64", "cpu"),
            policy=ComparisonPolicy(0.0, 0.0),
        )
        self.assertFalse(without_nan.passed)
        nonzero_at_zero = compare_values(
            [1.0], [0.0],
            actual=TensorMetadata("zero", (1,), "float64", "cpu"),
            expected=TensorMetadata("zero", (1,), "float64", "cpu"),
            policy=ComparisonPolicy(0.0, 0.0),
        )
        self.assertEqual(nonzero_at_zero.max_relative_error, math.inf)
        empty = TensorMetadata("empty", (0,), "float64", "cpu")
        self.assertTrue(
            compare_values(
                [], [],
                actual=empty,
                expected=empty,
                policy=ComparisonPolicy(0.0, 0.0),
            ).passed
        )

    def test_malformed_metadata_and_policy_are_rejected(self) -> None:
        with self.assertRaisesRegex(NumericalContractError, "metadata mismatch"):
            compare_values(
                [1.0], [1.0],
                actual=TensorMetadata("a", (1,), "float32", "cpu"),
                expected=TensorMetadata("e", (1,), "float64", "cpu"),
                policy=ComparisonPolicy(0.0, 0.0),
            )
        with self.assertRaisesRegex(NumericalContractError, "value count mismatch"):
            compare_values(
                [1.0], [],
                actual=TensorMetadata("a", (1,), "float64", "cpu"),
                expected=TensorMetadata("e", (1,), "float64", "cpu"),
                policy=ComparisonPolicy(0.0, 0.0),
            )
        with self.assertRaisesRegex(NumericalContractError, "zero tolerance"):
            integer = TensorMetadata("integer", (1,), "int64", "cpu")
            compare_values(
                [1], [1],
                actual=integer,
                expected=integer,
                policy=ComparisonPolicy(1.0, 0.0),
            )

    def test_runtime_scalar_types_and_integer_policies_are_strict(self) -> None:
        cases = (
            ("bool", [1], [True], "must be a bool"),
            ("int64", [True], [1], "must be an int"),
            ("float32", [1], [1.0], "must be a float"),
            ("float64", [1.0], [1], "must be a float"),
        )
        for dtype, actual_values, expected_values, message in cases:
            with self.subTest(dtype=dtype):
                metadata = TensorMetadata("strict", (1,), dtype, "cpu")
                with self.assertRaisesRegex(NumericalContractError, message):
                    compare_values(
                        actual_values,
                        expected_values,
                        actual=metadata,
                        expected=metadata,
                        policy=ComparisonPolicy(0.0, 0.0),
                    )

        integer = TensorMetadata("integer", (1,), "int64", "cpu")
        with self.assertRaisesRegex(NumericalContractError, "forbids equal_nan"):
            compare_values(
                [1],
                [1],
                actual=integer,
                expected=integer,
                policy=ComparisonPolicy(0.0, 0.0, equal_nan=True),
            )
        with self.assertRaisesRegex(NumericalContractError, "outside signed int64"):
            compare_values(
                [2**63],
                [0],
                actual=integer,
                expected=integer,
                policy=ComparisonPolicy(0.0, 0.0),
            )
        with self.assertRaisesRegex(NumericalContractError, "equal_nan must be a bool"):
            compare_values(
                [1.0],
                [1.0],
                actual=TensorMetadata("float", (1,), "float64", "cpu"),
                expected=TensorMetadata("float", (1,), "float64", "cpu"),
                policy=ComparisonPolicy(0.0, 0.0, equal_nan=1),  # type: ignore[arg-type]
            )


class FiniteDifferenceTests(unittest.TestCase):
    def test_quadratic_and_scaled_steps(self) -> None:
        result = central_finite_difference(
            lambda values: values[0] ** 2 + 3.0 * values[1] ** 2,
            (2.0, -4.0),
            epsilon=1.0e-6,
        )
        self.assertAlmostEqual(result.gradient[0], 4.0, places=7)
        self.assertAlmostEqual(result.gradient[1], -24.0, places=7)
        self.assertEqual(result.steps, (2.0e-6, 4.0e-6))
        self.assertEqual(result.metadata.shape, (2,))

    def test_linear_repeated_coordinate_and_empty(self) -> None:
        linear = central_finite_difference(
            lambda values: 2.0 * values[0] - 5.0 * values[1],
            (1.0, 1.0),
        )
        self.assertAlmostEqual(linear.gradient[0], 2.0, places=8)
        self.assertAlmostEqual(linear.gradient[1], -5.0, places=8)
        repeated = central_finite_difference(
            lambda values: (values[0] + values[1]) ** 2,
            (2.0, 2.0),
        )
        self.assertAlmostEqual(repeated.gradient[0], 8.0, places=7)
        self.assertAlmostEqual(repeated.gradient[1], 8.0, places=7)
        empty = central_finite_difference(lambda values: 0.0, ())
        self.assertEqual(empty.gradient, ())
        self.assertEqual(empty.steps, ())
        self.assertEqual(empty.metadata.shape, (0,))
        with self.assertRaisesRegex(NumericalContractError, "empty point"):
            central_finite_difference(lambda values: math.nan, ())

    def test_invalid_inputs_and_objective_are_rejected(self) -> None:
        for epsilon in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(NumericalContractError, "epsilon"):
                    central_finite_difference(lambda values: values[0], (1.0,), epsilon=epsilon)
        with self.assertRaisesRegex(NumericalContractError, "point values"):
            central_finite_difference(lambda values: values[0], (math.nan,))
        with self.assertRaisesRegex(NumericalContractError, "non-finite"):
            central_finite_difference(lambda values: math.inf, (1.0,))
        with self.assertRaisesRegex(NumericalContractError, "finite scalar"):
            central_finite_difference(lambda values: [values[0]], (1.0,))
        with self.assertRaisesRegex(NumericalContractError, "CPU-only"):
            central_finite_difference(lambda values: values[0], (1.0,), device="hip")
        with self.assertRaisesRegex(NumericalContractError, "only float64"):
            central_finite_difference(lambda values: values[0], (1.0,), dtype="int64")
        with self.assertRaisesRegex(NumericalContractError, "only float64"):
            central_finite_difference(lambda values: values[0], (1.0,), dtype="float32")
        with self.assertRaisesRegex(NumericalContractError, "real scalar"):
            central_finite_difference(lambda values: values[0], ("1.0",))
        with self.assertRaisesRegex(NumericalContractError, "perturbation"):
            central_finite_difference(lambda values: values[0], (1.0e308,), epsilon=1.0)
        with self.assertRaisesRegex(NumericalContractError, "not representable"):
            central_finite_difference(
                lambda values: values[0],
                (1.0e16,),
                epsilon=1.0e-16,
            )
        with self.assertRaisesRegex(NumericalContractError, "gradient"):
            central_finite_difference(
                lambda values: 1.0e308 if values[0] > 0.0 else -1.0e308,
                (0.0,),
            )


if __name__ == "__main__":
    unittest.main()
