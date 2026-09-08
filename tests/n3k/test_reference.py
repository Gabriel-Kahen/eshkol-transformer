"""Independent oracle self-checks; native parity is a separate mandatory gate."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.n3k import reference as ref


class MathematicalReferenceTests(unittest.TestCase):
    def assert_vjp(self, function, inputs, upstream, expected):
        for index, analytic in enumerate(expected):
            measured = ref.central_vjp(function, inputs, upstream, index)
            self.assertEqual(len(analytic), len(measured))
            for element, (a, b) in enumerate(zip(analytic, measured)):
                self.assertTrue(math.isclose(a, b, abs_tol=3e-7, rel_tol=3e-6),
                                f"operand {index}, element {element}: analytic={a}, finite difference={b}")

    def test_every_embedding_row_repeated_and_boundary_weight_gradient(self):
        for row in ref.EMBEDDING_ROWS:
            _, _, vocabulary, width = row
            for ids in ((0, vocabulary - 1), (min(7, vocabulary - 1),) * 2):
                with self.subTest(row=row, ids=ids):
                    weights = ref.values(vocabulary * width, 1)
                    upstream = ref.values(2 * width, 2)
                    self.assert_vjp(lambda candidate: ref.embedding(ids, candidate, width),
                                    (weights,), upstream,
                                    (ref.embedding_vjp(ids, upstream, vocabulary, width),))

    def test_every_linear_row_both_operand_gradients(self):
        for row in ref.LINEAR_ROWS:
            _, _, din, dout = row
            with self.subTest(row=row):
                x, weight, upstream = ref.values(2 * din, 3), ref.values(din * dout, 4), ref.values(2 * dout, 5)
                self.assert_vjp(lambda a, b: ref.linear(a, b, din, dout), (x, weight), upstream,
                                ref.linear_vjp(x, weight, upstream, din, dout))

    def test_gelu_row_gradient_negative_zero_and_tails(self):
        forward = next(case for case in ref.cases() if case.operation == "gelu.forward")
        x = forward.inputs[0].data
        upstream = ref.values(len(x), 6)
        self.assert_vjp(ref.gelu, (x,), upstream, (ref.gelu_vjp(x, upstream),))
        self.assertEqual(math.copysign(1, ref.gelu((-0.0,))[0]), -1)
        self.assertEqual(ref.gelu_vjp((0.0,), (1.0,)), (0.5,))
        # The exact-erf forward/VJP must remain distinguishable from tanh GELU.
        tanh = tuple(0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3))) for v in x)
        self.assertGreater(max(abs(a - b) for a, b in zip(ref.gelu(x), tanh)), 1e-4)

    def test_residual_each_edge_and_shared_primal_gradient(self):
        x, branch, upstream = (ref.values(8, salt) for salt in (7, 8, 9))
        self.assert_vjp(ref.residual, (x, branch), upstream, (upstream, upstream))
        self.assert_vjp(lambda value: ref.residual(value, value), (x,), upstream,
                        (tuple(2 * value for value in upstream),))

    def test_oracle_detects_overwrite_repeated_embedding_and_transposed_linear(self):
        width, vocabulary, ids = 4, 256, (7, 7)
        upstream = ref.values(8, 2)
        dw = ref.embedding_vjp(ids, upstream, vocabulary, width)
        self.assertNotEqual(dw[7 * width:8 * width], upstream[width:])
        self.assertEqual(dw[:7 * width], (0.0,) * (7 * width))
        x, weight = ref.values(8, 3), ref.values(16, 4)
        transposed = tuple(weight[column * 4 + row] for row in range(4) for column in range(4))
        self.assertNotEqual(ref.linear(x, weight, 4, 4), ref.linear(x, transposed, 4, 4))


class FixtureTests(unittest.TestCase):
    def test_exact_operation_row_coverage_and_dense_shapes(self):
        cases = ref.cases()
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case.name for case in cases}), len(cases))
        expected = ({(op, row) for row in ref.EMBEDDING_ROWS for op in ("embedding.forward", "embedding.backward")}
                    | {(op, row) for row in ref.LINEAR_ROWS for op in ("linear.forward-no-bias", "linear.backward-no-bias")}
                    | {(op, ref.GELU_ROW) for op in ("gelu.forward", "gelu.backward")}
                    | {(op, ref.RESIDUAL_ROW) for op in ("residual.forward", "residual.backward")})
        self.assertEqual({(case.operation, case.row) for case in cases}, expected)
        for case in cases:
            for tensor in case.inputs + case.outputs:
                self.assertEqual(math.prod(tensor.shape), len(tensor.data))
                self.assertTrue(all(math.isfinite(value) for value in tensor.data))

    def test_generated_header_fresh_process_determinism(self):
        script = Path(ref.__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"reference-{i}.h" for i in range(2)]
            for path in paths:
                subprocess.run([sys.executable, str(script), "--header", str(path)],
                               check=True, capture_output=True, cwd=directory)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertEqual(paths[0].read_bytes(), ref.render_header())


if __name__ == "__main__":
    unittest.main()
