"""Guard the public lease/mode ordering of the one-batch acceptance caller."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class OverfitSourceContract(unittest.TestCase):
    def test_public_observations_outside_lease_and_no_mode_mutation(self) -> None:
        source = (ROOT / "tests/tr3_public_installed/one_batch_overfit.esk").read_text()
        self.assertNotIn("(module-eval!", source)
        self.assertNotIn("(module-train!", source)
        self.assertIn("(model-forward model input ':deterministic? #t)", source)
        ordered = (
            "(define initial-state (module-state-dict model))",
            "(define initial-logits (forward-bits model input))",
            "(define trainer (trainer-create config tokenizer training model optimizer))",
            "(check 'enrolled-model-forward-rejected",
            "(define before (trainer-evaluate! trainer evaluation))",
            "(define summary (trainer-train! trainer (trainer-stop-policy #f 32 #f)))",
            "(define after (trainer-evaluate! trainer evaluation))",
            "(check 'release (trainer-release! trainer))",
            "(define trained-state (module-state-dict model))",
            "(define trained-logits (forward-bits model input))",
            "(check 'restore-initial-state (module-load-state-dict! model initial-state))",
            "(check 'restore-trained-state (module-load-state-dict! model trained-state))",
        )
        positions = [source.index(item) for item in ordered]
        self.assertEqual(positions, sorted(positions))
        rejection = source.split("(check 'enrolled-model-forward-rejected", 1)[1].split(
            "(define before", 1
        )[0]
        self.assertIn("(model-forward model input ':deterministic? #t)", rejection)
        self.assertIn("(transformer-error-category raised) 'invalid-state", rejection)
        self.assertIn("(transformer-error-operation raised) 'model-forward", rejection)

    def test_provider_contract_for_mode_and_lease(self) -> None:
        gate = (ROOT / "native/tr3_lease_public_gates_extension.esk").read_text()
        step = (ROOT / "native/tr3_step_transaction_extension.esk").read_text()
        model = (ROOT / "native/m3_model_extension.esk").read_text()
        state = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
        self.assertIn("(tr3-public-gate-model model 'model-forward)", gate)
        self.assertIn("(tr3-public-gate-module module 'module-load-state-dict!)", gate)
        self.assertIn("(if (not (eq? mode 'train))", step)
        self.assertIn("((eq? key ':deterministic?)", model)
        load = state.split("(define (module-load-state-dict-scoped!", 1)[1].split(
            "(define (module-load-state-dict!", 1
        )[0]
        self.assertNotIn("set-mode-modules!", load)

    def test_public_m3_probe_uses_same_aggregate_package_closure(self) -> None:
        native = ROOT / "native"
        installed_root = (native / "tr3_public_installed_root.esk").read_text()
        self.assertEqual(installed_root.count('(load "tr3_public_release_candidate_root.esk")'), 1)
        self.assertEqual(installed_root.count('(load "m3_call_adapters.esk")'), 1)

        accepted = (native / "m3_package_bridge.c").read_text().split("\n\n", 1)[1]
        bridge = (native / "tr3_public_installed_bridge.c").read_text()
        self.assertIn(accepted, bridge)
        self.assertIn("transformer/model.esk\n",
                      (native / "tr3_public_installed_facades.txt").read_text())

        renames = set((native / "tr3_public_installed_private_renames.txt")
                      .read_text().splitlines())
        exports = set((native / "tr3_public_installed_exports.txt")
                      .read_text().splitlines())
        accepted_renames = {
            row.replace("m3-public-", "tr3-public-", 1)
            for row in (native / "m3_package_private_renames.txt")
            .read_text().splitlines() if row.startswith("m3-public-")
        }
        accepted_exports = {
            row for row in (native / "m3_package_public_exports.txt")
            .read_text().splitlines() if row.startswith("et_e1b_public_m3_")
        }
        self.assertEqual(len(accepted_renames), 8)
        self.assertEqual(len(accepted_exports), 8)
        self.assertLessEqual(accepted_renames, renames)
        self.assertLessEqual(accepted_exports, exports)
        self.assertFalse(any(row.startswith("m3-public-") for row in renames))
        gates = (native / "tr3_lease_public_gates_extension.esk").read_text()
        for row in accepted_renames:
            gate = row.split()[0]
            self.assertIn(f"(define ({gate} ", gates)


if __name__ == "__main__":
    unittest.main()
