"""Bounded source and synthetic receipt checks for the P1 diagnostic workflow."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/p1-pair-memory-diagnostic.yml"
CANDIDATE = "15f2522946b8ee13b9f166967a99f2665263d4d0"
TREE = "672e9da3d40061278e94ca0aa1954e98936afa7e"
BASE = "codex/p1-pair-memory-timing-receipt-20261126"
HEADER = ("utc\tmonotonic_ns\tpgid_first\tpgid_second\t"
          "rss_first_bytes\trss_second_bytes\trss_sum_bytes\t"
          "processes_first\tprocesses_second\n")
TIME_FAILURE = "Command exited with non-zero status 1\n"


def block(source, begin, end):
    return source.split(begin, 1)[1].split(end, 1)[0]


def step_script(workflow, begin, end):
    section = block(workflow, f"      - name: {begin}\n", f"      - name: {end}\n")
    return textwrap.dedent(section.split("        run: |\n", 1)[1])


class PairMemoryDiagnosticTests(unittest.TestCase):
    def test_branch_only_exact_source_and_supported_lane(self):
        text = WORKFLOW.read_text()
        self.assertIn(f"  pull_request:\n    branches: [{BASE}]", text)
        self.assertIn(f'test "$GITHUB_BASE_REF" = {BASE}', text)
        self.assertNotIn("  push:\n", text)
        self.assertNotIn("  workflow_dispatch:\n", text)
        self.assertIn("startsWith(github.head_ref, 'codex/p1-pair-memory-trigger-')", text)
        self.assertIn("ref: " + CANDIDATE, text)
        self.assertIn("test \"$(git rev-parse HEAD)\" = " + CANDIDATE, text)
        self.assertIn("test \"$(git rev-parse 'HEAD^{tree}')\" = " + TREE, text)
        for setting in ("P1_TWO_COMPILERS", "P1_PAIR_MEMORY_TRACE", "P1_KEEP_TMP"):
            self.assertIn(setting + ": '1'", text)
        self.assertIn("scripts/ci-build-prerequisites.sh parameter-state", text)
        self.assertIn("make test-ci-parameters-after-build", text)
        self.assertNotIn("make test-acceptance", text)
        self.assertIn("if: always()", text)
        self.assertIn("sha256sum -c SHA256SUMS", text)
        self.assertNotIn(".deps/eshkol-transformer-p1.*\n          path:", text)


class PairMemoryCollectorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.source = self.work / ".deps/eshkol-transformer-p1.abc123"
        self.source.mkdir(parents=True)
        self.timing = self.source / "compiler-timing.log"
        self.workflow = WORKFLOW.read_text()
        self.collect_script = step_script(self.workflow, "Collect validated pair-memory evidence",
                                          "Seal diagnostic receipt")
        self.write_traces()
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

    def write_traces(self, positive_pairs=6, count=1):
        for i in range(1, 7):
            trace = self.source / f"pair-{i}-memory.tsv"
            first, second = (100, 200) if i <= positive_pairs else (0, 0)
            row = (f"2026-11-26T10:00:00.000+00:00\t1\t10\t11\t"
                   f"{first}\t{second}\t{first + second}\t{count}\t{count}\n")
            trace.write_text(HEADER + row)
            trace.with_suffix(".summary.json").write_text(json.dumps({
                "scope": "sampled RSS of two P1 compiler process groups only",
                "samples": 1, "both_active_samples": int(count > 0),
                "max_sampled_pair_rss_bytes": first + second, "interval_seconds": 0.25}))

    def run_step(self, script, receipt, **status):
        env = {**os.environ, "RUNNER_TEMP": str(receipt.parent), **status}
        return subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=self.work,
                              env=env, capture_output=True, text=True)

    def collect(self, name, outcome="success"):
        receipt = self.work / name / "p1-pair-memory-diagnostic"
        receipt.mkdir(parents=True)
        (receipt / "parameter-state.log").write_text(
            f"P1 preserved temporary evidence: {self.source}\n")
        return receipt, self.run_step(self.collect_script, receipt, SUITE_OUTCOME=outcome)

    def test_collects_expected_gnu_time_failures_and_successes(self):
        payload = self.complete_timing()
        self.timing.write_text(payload)
        (self.source / "build-scratch.bin").write_bytes(b"not evidence")
        receipt, result = self.collect("valid")
        self.assertEqual(result.returncode, 0, result.stderr)
        copied = receipt / "validated-p1"
        self.assertEqual(len(list(copied.iterdir())), 13)
        self.assertEqual((copied / self.timing.name).read_text(), payload)
        self.assertFalse((copied / "build-scratch.bin").exists())
        self.assertIn("timing_outcome\tvalidated\nreceipt_status\tcomplete\n",
                      (receipt / "collection.tsv").read_text())

    def test_timing_failures_preserve_and_seal_valid_traces(self):
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
                self.assertEqual({p.name for p in (receipt / "validated-p1").iterdir()},
                                 {p.name for p in self.source.glob("pair-*-memory.*")})
                self.assertIn("timing_outcome\tinvalid\nreceipt_status\tpartial\n",
                              (receipt / "collection.tsv").read_text())
        seal_script = step_script(self.workflow, "Seal diagnostic receipt", "Upload diagnostic-only receipt")
        sealed = self.run_step(seal_script, receipt, JOB_STATUS="failure", TOOLCHAIN_OUTCOME="success",
                               PREREQUISITES_OUTCOME="success", SUITE_OUTCOME="success", COLLECT_OUTCOME="failure")
        self.assertEqual(sealed.returncode, 0, sealed.stderr)
        seal = (receipt / "SHA256SUMS").read_text()
        self.assertIn("./validated-p1/pair-6-memory.tsv", seal)
        self.assertIn("./validated-p1/pair-6-memory.summary.json", seal)
        self.assertNotIn("compiler-timing.log", seal)

    def test_failed_suite_can_preserve_partial_timing_without_a_complete_receipt(self):
        receipt, result = self.collect("failed-suite", outcome="failure")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((receipt / "validated-p1" / self.timing.name).read_text(), self.positive)
        self.assertIn("timing_outcome\tvalidated\nreceipt_status\tpartial\n",
                      (receipt / "collection.tsv").read_text())

    def test_rejects_corrupt_missing_or_nonoverlapping_trace_evidence(self):
        summary = self.source / "pair-1-memory.summary.json"
        summary.write_text("{}")
        receipt, result = self.collect("corrupt")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pair trace/summary mismatch", result.stderr)
        self.assertFalse((receipt / "validated-p1").exists())
        self.write_traces()
        (self.source / "pair-1-memory.tsv").unlink()
        receipt, result = self.collect("missing-trace")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly six ordered pair traces", result.stderr)
        self.assertFalse((receipt / "validated-p1").exists())
        for name, positive_pairs, count in (("no-processes", 0, 0), ("zero-rss", 0, 1),
                                            ("last-pair-zero", 5, 1)):
            with self.subTest(name=name):
                self.write_traces(positive_pairs, count)
                receipt, result = self.collect(name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("no simultaneous positive RSS sample", result.stderr)
                self.assertFalse((receipt / "validated-p1").exists())


if __name__ == "__main__":
    unittest.main()
