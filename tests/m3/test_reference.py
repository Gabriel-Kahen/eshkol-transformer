"""Tests for the unfrozen independent graph; no production parity claim."""
import unittest

from tests.m3.reference import PARAMETERS, TIED, finite_difference, forward, parameters, setup, upstream, vjp
import torch


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup()

    def test_initializer_contract_and_continuation(self):
        self.assertEqual(len(PARAMETERS), 14)
        for seed in (0, 1729, (1 << 63) - 1):
            p, successor = parameters(seed)
            self.assertEqual(sum(value.numel() for value in p.values()), 1184)
            self.assertEqual(successor, (1, seed, 290, 0))
            continued, final = parameters(seed, 290)
            self.assertEqual(final, (1, seed, 580, 0))
            self.assertFalse(torch.equal(p[TIED], continued[TIED]))
            for path, shape in PARAMETERS:
                if len(shape) == 1:
                    self.assertTrue(torch.equal(p[path], torch.full(shape, 1.0 if path.endswith("weight") else 0.0, dtype=torch.float64)))
        with self.assertRaises(ValueError):
            parameters(-1)

    def test_all_unique_vjps_and_tied_edges(self):
        for init_seed, tokens, salt in ((1729, (3, 197), 1), (0, (0, 255), 2), ((1 << 63) - 1, (7, 7), 3)):
            p, _ = parameters(init_seed)
            logits, gradients, edges = vjp(p, tokens, upstream(salt))
            self.assertEqual(tuple(logits.shape), (1, 2, 256))
            self.assertEqual(tuple(gradients), tuple(path for path, _ in PARAMETERS))
            torch.testing.assert_close(gradients[TIED], edges["head"] + edges["embedding"], atol=1e-12, rtol=1e-12)
            for edge in edges.values():
                self.assertGreater(edge.abs().max().item(), 1e-8)
            for mutant in (edges["head"], edges["embedding"], edges["head"] * 2 + edges["embedding"], edges["head"] + edges["embedding"] * 2):
                self.assertFalse(torch.allclose(gradients[TIED], mutant, atol=1e-9, rtol=1e-9))

    def test_all_1184_values_central_differences_two_steps_two_fixtures(self):
        for conditioned, salt, steps, absolute in ((True, 2, (1e-5, 5e-6), 3e-8),
                                                  (False, 1, (2e-6, 1e-6), 1e-8)):
            p, _ = parameters(conditioned=conditioned)
            seed = upstream(salt)
            _, gradients, _ = vjp(p, (3, 197), seed)
            for path, _ in PARAMETERS:
                self.assertGreater(gradients[path].abs().max().item(), 1e-8, path)
                for step in steps:
                    numeric = finite_difference(p, path, (3, 197), seed, step)
                    torch.testing.assert_close(gradients[path], numeric, rtol=2e-5, atol=absolute, msg=path)

    def test_causality_checks_embedding_edge_not_tied_sum(self):
        p, _ = parameters(conditioned=True)
        torch.testing.assert_close(forward(p, (3, 197))[0, 0], forward(p, (3, 41))[0, 0], rtol=0, atol=0)
        seed = upstream()
        seed[:, 1] = 0
        _, gradients, edges = vjp(p, (3, 197), seed)
        self.assertEqual(torch.count_nonzero(edges["embedding"][197]).item(), 0)
        self.assertGreater(torch.count_nonzero(gradients[TIED][197]).item(), 0)

    def test_distinguishable_wiring_mutations(self):
        p, _ = parameters(conditioned=True)
        expected = forward(p)
        for mutation in ({"causal": False}, {"split_heads": False}, {"swap_qk": True}):
            self.assertFalse(torch.allclose(expected, forward(p, **mutation), rtol=1e-7, atol=1e-9), mutation)

    def test_zero_seed_has_all_fourteen_zero_contributions(self):
        p, _ = parameters()
        _, gradients, edges = vjp(p, seed=torch.zeros((1, 2, 256), dtype=torch.float64))
        self.assertEqual(len(gradients), 14)
        for value in (*gradients.values(), *edges.values()):
            self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_malformed_tensors_reject_without_broadcast_or_cast(self):
        p, _ = parameters()
        malformed = [dict(p, extra=p[TIED]), {key: value for key, value in p.items() if key != TIED}]
        for value in (torch.tensor(0.0, dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64),
                      torch.zeros(4, dtype=torch.float32), torch.full((4,), float("nan"), dtype=torch.float64)):
            malformed.append({**p, "norm_final/bias": value})
        for candidate in malformed:
            with self.assertRaises(ValueError):
                forward(candidate)
        for seed in (torch.tensor(1.0, dtype=torch.float64), torch.zeros(1, 1, 256, dtype=torch.float64),
                     torch.zeros(1, 2, 256, dtype=torch.float32), torch.full((1, 2, 256), float("inf"), dtype=torch.float64)):
            with self.assertRaises(ValueError):
                vjp(p, seed=seed)
        for edge in (torch.zeros(4, dtype=torch.float64), torch.zeros(256, 4, dtype=torch.float32)):
            for keyword in ("embedding_weight", "head_weight"):
                with self.assertRaises(ValueError):
                    forward(p, **{keyword: edge})


if __name__ == "__main__":
    unittest.main()
