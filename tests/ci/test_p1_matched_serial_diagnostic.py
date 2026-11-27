"""Bounded source and synthetic receipt checks for the P1 serial control."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

from tests.ci import p1_serial_comparison as comparison


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/p1-matched-serial-diagnostic.yml"
CANDIDATE = "15f2522946b8ee13b9f166967a99f2665263d4d0"
TREE = "672e9da3d40061278e94ca0aa1954e98936afa7e"
BASE = "codex/p1-matched-serial-receipt-20261127"
P1_PASS = ("P1 PASS: E1B-integrated packaging, structural/native/registry checks, "
           "root-publication IR and poison proof, sanitizers, negatives, atomicity, and determinism")
PAIR_BASE = "f9fd2de24ff10cd819a70b51f1697eb6e8fc3bfe"
TIME_FAILURE = "Command exited with non-zero status 1\n"


def block(source, begin, end):
    return source.split(begin, 1)[1].split(end, 1)[0]


def step_script(workflow, begin, end):
    section = block(workflow, f"      - name: {begin}\n", f"      - name: {end}\n")
    return textwrap.dedent(section.split("        run: |\n", 1)[1])


class MatchedSerialDiagnosticTests(unittest.TestCase):
    def test_branch_only_exact_source_and_supported_lane(self):
        text = WORKFLOW.read_text()
        self.assertIn(f"  pull_request:\n    branches: [{BASE}]", text)
        self.assertIn(f'test "$GITHUB_BASE_REF" = {BASE}', text)
        self.assertNotIn("  push:\n", text)
        self.assertNotIn("  workflow_dispatch:\n", text)
        self.assertIn("startsWith(github.head_ref, 'codex/p1-matched-serial-trigger-')", text)
        self.assertIn("ref: " + CANDIDATE, text)
        self.assertIn("test \"$(git rev-parse HEAD)\" = " + CANDIDATE, text)
        self.assertIn("test \"$(git rev-parse 'HEAD^{tree}')\" = " + TREE, text)
        self.assertIn("P1_TWO_COMPILERS: '0'", text)
        self.assertIn("P1_PAIR_MEMORY_TRACE: '0'", text)
        self.assertIn("P1_KEEP_TMP: '1'", text)
        self.assertIn("test ! -e build", text)
        self.assertIn("test ! -e .deps", text)
        self.assertIn("compgen -G '.deps/eshkol-transformer-p1.*'", text)
        self.assertIn("scripts/ci-build-prerequisites.sh parameter-state", text)
        self.assertIn("make test-ci-parameters-after-build", text)
        self.assertNotIn("make test-acceptance", text)
        self.assertIn("if: always()", text)
        self.assertIn("sha256sum -c SHA256SUMS", text)
        self.assertNotIn(".deps/eshkol-transformer-p1.*\n          path:", text)


    def test_preserves_paired_job_contract_and_timing_validator(self):
        paired = subprocess.check_output(
            ["git", "show", f"{PAIR_BASE}:.github/workflows/p1-pair-memory-diagnostic.yml"],
            cwd=ROOT, text=True)
        serial = WORKFLOW.read_text()
        serial = serial.replace("p1-matched-serial-diagnostic", "p1-pair-memory-diagnostic")
        self.assertEqual(
            block(paired, "    name: Supported parameter-state only", "      P1_TWO_COMPILERS:"),
            block(serial, "    name: Supported parameter-state only", "      P1_TWO_COMPILERS:"))
        for begin, end in (
                ("      - uses: actions/checkout", "      - name: Verify exact candidate source"),
                ("      - name: Install host dependencies", "      - name: Verify pinned Eshkol toolchain"),
                ("      - name: Build parameter-state prerequisites", "      - name: Collect validated"),
                ("          def validate_timing", "          root, outcome =")):
            self.assertEqual(block(paired, begin, end), block(serial, begin, end))


class MatchedSerialCollectorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.source = self.work / ".deps/eshkol-transformer-p1.abc123"
        self.source.mkdir(parents=True)
        self.timing = self.source / "compiler-timing.log"
        self.workflow = WORKFLOW.read_text()
        self.collect_script = step_script(self.workflow, "Collect validated serial P1 evidence",
                                          "Seal diagnostic receipt")
        self.positive = self.time_record(0, "cache-compiler.abcd12", "-L trusted positive.esk")
        self.timing.write_text(self.positive)

    def time_record(self, status, cache, args, emit_object=False):
        mode = "--emit-object " if emit_object else ""
        return (f"elapsed_s=1.00 max_rss_kib=2000 exit={status} command="
                "timeout --foreground --signal=TERM --kill-after=5s 1200s "
                "env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR -u ESHKOL_AOT_MODULE_CACHE_DIR "
                f"ESHKOL_JIT_CACHE=0 XDG_CACHE_HOME={self.source}/{cache} "
                f"{self.work}/.deps/eshkol-build/eshkol-run --strict-types {mode}--no-stdlib {args}\n")

    def complete_timing(self):
        rows = [self.time_record(0, f"cache-compiler.{i:06d}", "-L trusted positive.esk")
                for i in range(8)]
        for name in ("transformer-error-make", "transformer-error-raise",
                     "transformer-error-wrap-foreign", "e1-internal-dispatch", "et-e1b-private-raise"):
            rows.append(TIME_FAILURE + self.time_record(1, f"cache-public-surface-{name}",
                        f"-e (begin (require transformer.module) {name})"))
        for run in (1, 2):
            for name in ("module", "example", "public-stubs", "public-reverse"):
                rows.append(self.time_record(0, f"cache-object-{name}-{run}.o", "positive.esk", True))
            for name in ("example", "public-stubs", "public-reverse"):
                rows.append(self.time_record(0, f"cache-aot-{name}-{run}", "positive.esk"))
            fixture = self.work / "tests/p1/negative_guessed_private_native.esk"
            rows.append(self.time_record(0, f"cache-guessed-object-{run}",
                        f"{fixture} -o {self.source}/guessed-private.o", True))
            rows.append(TIME_FAILURE + self.time_record(1, f"cache-guessed-aot-{run}",
                        f"{fixture} -o {self.source}/guessed-private"))
            for kind in ("provider", "module", "handle", "state_entry", "state_dict"):
                stem = f"negative_public_{kind}_internal"
                fixture = self.work / "tests/p1" / f"{stem}.esk"
                for phase, args in (("source", f"-r {fixture}"),
                                    ("object", f"{fixture} -o {self.source}/{stem}.o"),
                                    ("aot", f"{fixture} -o {self.source}/{stem}")):
                    rows.append(TIME_FAILURE + self.time_record(1, f"cache-{stem}-{phase}-{run}",
                                args, phase == "object"))
            for stem in ("negative_wrong_arity", "negative_fixture_not_public", "negative_provider_callback_arity"):
                fixture = self.work / "tests/p1" / f"{stem}.esk"
                rows.append(TIME_FAILURE + self.time_record(1, f"cache-{stem}-{run}",
                            f"{fixture} -o {self.source}/{stem}.o", True))
        return "".join(rows)

    def run_step(self, script, receipt, **status):
        env = {**os.environ, "RUNNER_TEMP": str(receipt.parent), **status}
        return subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=self.work,
                              env=env, capture_output=True, text=True)

    def collect(self, name, outcome="success", final_pass=P1_PASS):
        receipt = self.work / name / "p1-matched-serial-diagnostic"
        receipt.mkdir(parents=True)
        (receipt / "parameter-state.log").write_text(
            f"{final_pass}\nP1 preserved temporary evidence: {self.source}\n")
        return receipt, self.run_step(self.collect_script, receipt, SUITE_OUTCOME=outcome)

    def test_collects_expected_gnu_time_failures_and_successes(self):
        payload = self.complete_timing()
        self.timing.write_text(payload)
        (self.source / "build-scratch.bin").write_bytes(b"not evidence")
        receipt, result = self.collect("valid")
        self.assertEqual(result.returncode, 0, result.stderr)
        copied = receipt / "validated-p1"
        self.assertEqual({path.name for path in copied.iterdir()}, {self.timing.name})
        self.assertEqual((copied / self.timing.name).read_text(), payload)
        self.assertFalse((copied / "build-scratch.bin").exists())
        self.assertIn("timing_outcome\tvalidated\nreceipt_status\tcomplete\n",
                      (receipt / "collection.tsv").read_text())

    def test_timing_failures_keep_receipt_partial(self):
        negative = self.time_record(1, "cache-public-surface-transformer-error-make",
                                    "-e (begin (require transformer.module) transformer-error-make)")
        complete = self.complete_timing()
        fixture = self.work / "tests/p1/negative_wrong_arity.esk"
        object_record = self.time_record(1, "cache-negative_wrong_arity-1",
                                         f"{fixture} -o {self.source}/negative_wrong_arity.o", True)
        for name, payload in (("missing", None), ("empty", ""), ("truncated", self.positive[:-1]),
                              ("valid-row-truncation", self.positive),
                              ("missing-negative", complete.replace(TIME_FAILURE + negative, "", 1)),
                              ("bad-row", self.positive.replace("1.00", "NaN")),
                              ("bare-failure", self.positive + negative),
                              ("orphan-banner", self.positive + TIME_FAILURE),
                              ("reversed-banner", TIME_FAILURE + self.positive),
                              ("unexpected-status", TIME_FAILURE + negative.replace("exit=1", "exit=124")),
                              ("unexpected-negative", TIME_FAILURE + self.positive.replace("exit=0", "exit=1")),
                              ("unexpected-success", self.positive + negative.replace("exit=1", "exit=0")),
                              ("unexpected-cache", self.positive.replace("cache-compiler.abcd12", "cache-unknown")),
                              ("duplicate-cache", complete + self.time_record(0, "cache-compiler.000000", "positive.esk")),
                              ("missing-object-mode", complete.replace(object_record, object_record.replace("--emit-object ", ""))),
                              ("wrong-fixture", complete.replace(negative, negative.replace("module) transformer-error-make",
                                                                                          "module) transformer-error-raise")))):
            with self.subTest(name=name):
                if payload is None:
                    self.timing.unlink()
                else:
                    self.timing.write_text(payload)
                receipt, result = self.collect(name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid compiler timing log", result.stderr)
                self.assertEqual(list((receipt / "validated-p1").iterdir()), [])
                self.assertIn("timing_outcome\tinvalid\nreceipt_status\tpartial\n",
                              (receipt / "collection.tsv").read_text())
        seal_script = step_script(self.workflow, "Seal diagnostic receipt", "Upload diagnostic-only receipt")
        sealed = self.run_step(seal_script, receipt, JOB_STATUS="failure", TOOLCHAIN_OUTCOME="success",
                               PREREQUISITES_OUTCOME="success", SUITE_OUTCOME="success", COLLECT_OUTCOME="failure")
        self.assertEqual(sealed.returncode, 0, sealed.stderr)
        seal = (receipt / "SHA256SUMS").read_text()
        self.assertIn("./collection.tsv", seal)
        self.assertNotIn("compiler-timing.log", seal)

    def test_failed_suite_can_preserve_partial_timing_without_a_complete_receipt(self):
        receipt, result = self.collect("failed-suite", outcome="failure")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((receipt / "validated-p1" / self.timing.name).read_text(), self.positive)
        self.assertIn("timing_outcome\tvalidated\nreceipt_status\tpartial\n",
                      (receipt / "collection.tsv").read_text())

    def test_rejects_pair_traces_and_missing_or_duplicate_pass(self):
        self.timing.write_text(self.complete_timing())
        for index, suffix in enumerate(("tsv", "summary.json")):
            unexpected = self.source / f"pair-1-memory.{suffix}"
            unexpected.touch()
            receipt, result = self.collect(f"pair-trace-{index}")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unexpected paired trace", result.stderr)
            self.assertFalse((receipt / "validated-p1").exists())
            unexpected.unlink()
        for index, proof in enumerate(("", f"{P1_PASS}\n{P1_PASS}")):
            receipt, result = self.collect(f"no-pass-{index}", final_pass=proof)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly one final P1 PASS", result.stderr)
            self.assertFalse((receipt / "validated-p1").exists())


class MatchedSerialComparisonTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.serial_base = "a" * 40
        self.serial_trigger = "b" * 40
        self.pair, self.pair_job, self.pair_log = self.write_evidence("pair", True)
        self.serial, self.serial_job, self.serial_log = self.write_evidence("serial", False)

    def seal(self, root):
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (root / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(root).as_posix()}\n"
            for p in paths))

    def write_evidence(self, name, paired):
        root = self.work / name
        (root / "validated-p1").mkdir(parents=True)
        base = comparison.PAIR_BASE if paired else self.serial_base
        trigger = comparison.PAIR_TRIGGER if paired else self.serial_trigger
        run, flag = (102, 1) if paired else (111, 0)
        identity = {
            "mode": f"P1_TWO_COMPILERS={flag} P1_PAIR_MEMORY_TRACE={flag} P1_KEEP_TMP=1",
            "status": "started", "workflow_base": base, "trigger_head": trigger,
            "source_head": CANDIDATE, "source_tree": TREE, "final_job_status": "success",
            "run_id": run, "run_attempt": 1,
            **{f"{step}_outcome": "success" for step in ("toolchain", "prerequisites", "suite", "collect")},
        }
        collection = {
            "scope": "P1 compiler process groups only" if paired else "serial P1 compiler timing only",
            "suite_outcome": "success", "validated_pairs": 6 if paired else 0,
            "timing_outcome": "validated", "receipt_status": "complete",
        }
        for filename, fields in (("identity.tsv", identity), ("collection.tsv", collection)):
            (root / filename).write_text("".join(f"{key}\t{value}\n" for key, value in fields.items()))
        for filename in (*comparison.MATCHED_FILES, "prerequisites.log", "toolchain.log", "initial-memory.txt"):
            (root / filename).write_text(f"synthetic {filename}\n")
        if paired:
            for filename in comparison.TRACE_FILES:
                (root / filename).write_text("Synthetic data sealed by fixture setup.\n")
        workspace = Path("/home/runner/work/eshkol-transformer/eshkol-transformer")
        scratch = workspace / f".deps/eshkol-transformer-p1.abc{run}"
        (root / "parameter-state.log").write_text(
            f"{P1_PASS}\nP1 preserved temporary evidence: {scratch}\n")
        fixture = MatchedSerialCollectorTests()
        fixture.work = workspace
        fixture.source = scratch
        timing = fixture.complete_timing()
        if paired:
            timing = timing.replace("cache-compiler.000000", "cache-compiler.abcdef")
        (root / comparison.TIMING).write_text(timing)
        self.seal(root)

        job = self.work / f"{name}-job.json"
        job.write_text(json.dumps({
            "name": comparison.JOB, "status": "completed", "conclusion": "success",
            "labels": ["ubuntu-22.04"], "runner_group_name": "GitHub Actions",
            "id": run * 9, "run_id": run, "run_attempt": 1,
            "steps": [{"name": comparison.SUITE, "conclusion": "success",
                       "started_at": "2026-11-26T00:00:00Z", "completed_at": "2026-11-26T00:01:00Z"}],
        }))
        log = self.work / f"{name}-job.log"
        log.write_text(
            "\ufeff2026-11-26T00:00:00.1234567Z ##[group]Runner Image\n"
            "2026-11-26T00:00:00.1234568Z Image: ubuntu-22.04\n"
            "2026-11-26T00:00:00.1234569Z Version: 20261120.303.1\n"
            "2026-11-26T00:00:00.1234570Z Cache restored from key: eshkol-v1-Linux-X64-llvm-21.1.8-"
            + "f" * 64 + "\n")
        return root, job, log

    def compare(self):
        return comparison.compare(self.pair, self.serial, self.pair_job, self.serial_job,
                                  self.pair_log, self.serial_log, self.serial_base, self.serial_trigger)

    def test_comparison_matches_normalized_commands_and_uses_provider_wall_time(self):
        report = self.compare()
        self.assertEqual(report["serial"]["suite_wall_seconds"], 60)
        self.assertEqual(report["runner_image"], "20261120.303.1")
        self.assertEqual(sum(len(row["paired"]) for row in report["compiler_records"]), 67)
        self.assertEqual(sum(len(row["serial"]) for row in report["compiler_records"] if row["exit"]), 43)
        self.assertNotIn("eshkol-transformer-p1.abc", json.dumps(report))

    def test_comparison_requires_sealed_complete_paired_result(self):
        trace = self.pair / "validated-p1/pair-6-memory.tsv"
        trace.write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "SHA256SUMS mismatch"):
            self.compare()
        trace.unlink()
        self.seal(self.pair)
        with self.assertRaisesRegex(ValueError, "missing receipt files"):
            self.compare()

    def test_comparison_rejects_wrong_trigger_or_partial_collection(self):
        identity = self.pair / "identity.tsv"
        original = identity.read_text()
        identity.write_text(original.replace(comparison.PAIR_TRIGGER, self.serial_trigger))
        self.seal(self.pair)
        with self.assertRaisesRegex(ValueError, "identity or outcomes"):
            self.compare()
        identity.write_text(original)
        collection = self.pair / "collection.tsv"
        collection.write_text(collection.read_text().replace("receipt_status\tcomplete", "receipt_status\tpartial"))
        self.seal(self.pair)
        with self.assertRaisesRegex(ValueError, "collection is incomplete"):
            self.compare()

    def test_comparison_rejects_mismatch_in_host_cache_toolchain_and_compiler_commands(self):
        original = self.serial_log.read_text()
        self.serial_log.write_text(original.replace("20261120.303.1", "20261120.304.1"))
        with self.assertRaisesRegex(ValueError, "runner image differs"):
            self.compare()
        self.serial_log.write_text(original.replace("Cache restored from key", "Cache not found for input keys"))
        with self.assertRaisesRegex(ValueError, "cache state differs"):
            self.compare()
        self.serial_log.write_text(original)
        toolchain = self.serial / "toolchain-provenance.tsv"
        original = toolchain.read_text()
        toolchain.write_text("different compiler hash\n")
        self.seal(self.serial)
        with self.assertRaisesRegex(ValueError, "toolchain-provenance.tsv differs"):
            self.compare()
        toolchain.write_text(original)
        timing = self.serial / comparison.TIMING
        timing.write_text(timing.read_text().replace("trusted positive.esk", "different.esk", 1))
        self.seal(self.serial)
        with self.assertRaisesRegex(ValueError, "compiler commands differ"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
