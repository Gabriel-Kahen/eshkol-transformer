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


if __name__ == "__main__":
    unittest.main()
