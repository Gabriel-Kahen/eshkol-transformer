"""Keep the dedicated joint RSS witness equivalent to its multi-mode predecessor."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from tests.c2.c2_operational_source_manifest import closure

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "tests/c2/c2_checkpoint_operational_runtime.esk"
JOINT = ROOT / "tests/c2/c2_checkpoint_operational_joint_runtime.esk"
BODY = ROOT / "tests/c2/c2_checkpoint_operational_joint_body.esk"
ROOT_LOAD = '(load "c2_wave2_root.esk")'
BODY_LOAD = '(load "c2_checkpoint_operational_joint_body.esk")'
EXPECTED_CHECKS = (
    "metric baseline is available",
    "test ABI reports positive f32 tensor control size",
    "test ABI reports positive I2 decode control size",
    "joint public load requires every one of 64 descriptors exactly once",
    "joint exact reconstruction publishes one exact C2 owner",
    "joint E=62/P=1 owner has exact live storage",
    "joint exact owner count is live/nonbusy",
    "first joint operational save succeeds",
    "first joint save retains under two exact file images",
    "first joint save only publishes 124 lazy P1 identities",
    "second joint operational save succeeds",
    "warmed second joint save retains under 32 KiB",
    "second joint operational save is control-flat",
    "overwrite joint operational save succeeds",
    "warmed overwrite joint save retains under 32 KiB",
    "overwrite joint operational save is control-flat",
    "joint phase probes observe exactly three complete saves",
    "joint exact owner releases",
    "joint exact lifecycle restores all storage live counters",
    "joint exact lifecycle has exact retired controls",
    "joint exact stage is consumed and clears four large references",
    "joint exact C2 owner retires once",
    "joint lifecycle drains all image/reader/fd state",
    "measured retained RSS is below 512 MiB",
    "measured process peak RSS is below 512 MiB",
    "measured runtime is inside the explicit watchdog",
)


def check_labels(source: str) -> tuple[str, ...]:
    return tuple(re.findall(r'\(check\s+"([^"]+)"', source))


class OperationalJointSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.multi = MULTI.read_text(encoding="utf-8")
        cls.joint = JOINT.read_text(encoding="utf-8")
        cls.body = BODY.read_text(encoding="utf-8")

    def test_both_entries_load_the_same_production_and_joint_closures(self) -> None:
        for source in (self.multi, self.joint):
            self.assertEqual(source.count(ROOT_LOAD), 1)
            self.assertEqual(source.count(BODY_LOAD), 1)
        self.assertNotIn(ROOT_LOAD, self.body)
        self.assertNotIn(BODY_LOAD, self.body)

        roots = tuple(
            ROOT / path
            for path in (
                "internal/p1/lib",
                "internal/c1/lib",
                "internal/t2/lib",
                "internal/t1/lib",
                "internal/d2/lib",
                "tests/c2",
                "src",
                "lib",
                "native",
            )
        )
        multi_closure = closure(MULTI, roots) - {MULTI.resolve()}
        joint_closure = closure(JOINT, roots) - {JOINT.resolve()}
        self.assertEqual(multi_closure, joint_closure)
        self.assertIn(BODY.resolve(), joint_closure)
        self.assertIn((ROOT / "native/c2_wave2_root.esk").resolve(), joint_closure)

    def test_joint_entry_runs_only_the_shared_joint_body(self) -> None:
        self.assertEqual(self.joint.count("(run-joint-mode)"), 1)
        self.assertNotIn("(define (run-over-mode", self.joint)
        self.assertNotIn("(define (run-models-mode", self.joint)
        self.assertNotIn("(define (run-k2-fail-mode", self.joint)
        self.assertNotIn("(define (run-rank-mode", self.joint)
        self.assertNotIn("(define (run-alias-mode", self.joint)

    def test_exact_joint_assertion_manifest_is_preserved(self) -> None:
        entry_labels = check_labels(self.joint)
        body_labels = check_labels(self.body)
        self.assertEqual(entry_labels + body_labels, EXPECTED_CHECKS)
        self.assertEqual(len(EXPECTED_CHECKS), 26)

    def test_three_save_and_phase_contract_is_exact(self) -> None:
        self.assertEqual(self.body.count("c2-public-checkpoint-save!"), 3)
        self.assertIn("'(:overwrite? #t)", self.body)
        self.assertIn("(= joint-measure-calls 3)", self.body)
        self.assertIn("(= joint-encode-calls 3)", self.body)
        self.assertIn("(= joint-publish-calls 3)", self.body)
        self.assertIn("C2 OPERATIONAL MEASURE: mode=joint", self.body)
        self.assertIn("C2 OPERATIONAL PHASE: mode=joint", self.body)
        self.assertIn("(< peak-rss 524288)", self.body)

    def test_multi_mode_entry_uses_the_shared_joint_body(self) -> None:
        self.assertNotIn("(define (run-joint-mode)", self.multi)
        self.assertIn('((string=? mode "joint") (run-joint-mode))', self.multi)


if __name__ == "__main__":
    unittest.main()
