"""Static contract for the private checked-pair to typed G3-T RNG bridge."""
from pathlib import Path
import unittest

from tests.c2.test_handler_order import definition, parse, render


ROOT = Path(__file__).resolve().parents[2]


class PairRngBridgeSourceTest(unittest.TestCase):
    def test_one_aggregate_and_no_public_export(self):
        loads = parse((ROOT / "native/g3r_pair_rng_bridge_root.esk").read_text())
        self.assertEqual(loads, [
            ["load", '"g3r_file_pair_root.esk"'],
            ["load", '"m3t_transport_extension.esk"'],
            ["load", '"m3_schedule.esk"'],
            ["load", '"m3_model_extension.esk"'],
            ["load", '"m3_call_adapters.esk"'],
            ["load", '"g3t_model_admission_extension.esk"'],
            ["load", '"g3t_generator_constructor_extension.esk"'],
            ["load", '"g3t_prefill_sample_extension.esk"'],
            ["load", '"g3t_rng_owner_extension.esk"'],
            ["load", '"g3r_rng_import_extension.esk"'],
            ["load", '"g3r_pair_rng_bridge_extension.esk"'],
        ])
        for name in ("native/g3g_public_extension.esk",
                     "native/g3g_package_root.esk",
                     "native/cli3_generate_root.esk"):
            self.assertNotIn("g3r_pair_rng_bridge_extension.esk",
                             (ROOT / name).read_text())

    def test_checked_source_precedes_import_and_atomic_publication(self):
        source = (ROOT / "native/g3r_pair_rng_bridge_extension.esk").read_text()
        form = render(definition({"bridge": parse(source)},
                                 "g3r-pair-rng-admit-internal"))
        order = [
            "(make-vector 9 #f)",
            "(g3r-file-pair-admit-internal c2-path companion-path policy report pair)",
            "(vector-set! receipt-cell 0 (vector-ref pair 0))",
            "(g3r-rng-owner-from-checked-words",
            "(vector (vector-ref receipt-cell 0) (vector-ref rng-owner-cell 0)",
            "(vector-copy! destination 0 staged 0 10)",
            "(vector-set! rng-owner-cell 0 #f)",
            "(vector-set! receipt-cell 0 #f)",
        ]
        positions = [form.index(part) for part in order]
        self.assertEqual(positions, sorted(positions))
        release_rng = "(g3t-generation-rng-release! (vector-ref rng-owner-cell 0))"
        release_receipt = "(g3r-c2-receipt-release-internal! (if (vector-ref receipt-cell 0)"
        self.assertIn(release_rng, form)
        self.assertIn(release_receipt, form)
        self.assertLess(form.index(release_rng), form.index(release_receipt))
        self.assertIn("(g3r-c2-sha-fail-stop 134)", form)
        self.assertEqual(form.count("(vector-copy! destination"), 1)
        self.assertNotIn("(m3-call", form)

    def test_fault_hook_cannot_touch_checked_payload(self):
        pair = (ROOT / "native/g3r_file_pair_extension.esk").read_text()
        self.assertIn("(g3r-file-pair-before-publish-internal #f)", pair)
        self.assertNotIn("(g3r-file-pair-before-publish-internal staged)", pair)
        for name in ("tests/g3r/file_pair_test.esk",
                     "tests/g3r/pair_rng_bridge_test.esk"):
            test = (ROOT / name).read_text()
            parse(test)
            self.assertIn("(vector-set! payload 0", test)
            self.assertIn("(vector-set! payload 2", test)
            self.assertIn("(not hook-saw-vector)", test)
            self.assertIn("(not hook-mutated)", test)
        fixture = (ROOT / "tests/g3r/pair_rng_bridge_test.esk").read_text()
        for label in ("RNG allocation cut releases checked receipt",
                      "failed ten-slot promotion releases both owners",
                      "raw words vector is not typed RNG owner",
                      "native owner is independent of decoded vector",
                      "receipt release preserves independent RNG owner",
                      "bridge owner consumed by same-aggregate :rng constructor",
                      "typed RNG release leaves copied generator RNG live"):
            self.assertIn(label, fixture)


if __name__ == "__main__":
    unittest.main()
