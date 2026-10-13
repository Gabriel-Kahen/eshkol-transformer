"""Offline negative controls for receipt graph and emitted companion inputs."""
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
import shutil
import tempfile
import unittest
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def load(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECEIPTS = load("check-g3c4-proof-receipts.py")
EMITTED = load("check-g3c4-proof-emitted.py")
SOURCE = load("check-g3c4-proof-source.py")


class SourceAuditTests(unittest.TestCase):
    def test_current_nine_scripts(self):
        self.assertEqual(SOURCE.audit(ROOT), [])

    def test_unwrapped_direct_pipeline_and_substitution(self):
        for line in ("if grep -q x file", "nm file | sort -u",
                     "proof_command nm file | sort -u",
                     'tmp="$(mktemp -d /tmp/x)"'):
            with self.subTest(line=line):
                self.assertTrue(SOURCE.scan_text("runner.sh", line))

    def test_operator_continuation_starts_new_command(self):
        for operator in ('&&', '||', ';', '|'):
            with self.subTest(operator=operator):
                line = 'proof_command nm f ' + operator + ' \\\nsort -u'
                self.assertTrue(SOURCE.scan_text('runner.sh', line))

    def test_unwrapped_common_probe_rejected(self):
        for command in ('awk -F x file', 'git -C repo status',
                        'readlink -f path', 'sha256sum binary'):
            with self.subTest(command=command):
                self.assertIsNotNone(SOURCE.COMMON_EXTERNAL.search(f'value="$({command})"'))
                self.assertIsNone(SOURCE.COMMON_EXTERNAL.search(f'value="$(common_probe {command})"'))


class ReceiptPostflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for i, script in enumerate(RECEIPTS.SCRIPTS):
            path = self.root / f"command.{i}"
            path.mkdir()
            argv = f"bash\0{script}\0".encode()
            (path / "argv.nul").write_bytes(argv)
            (path / "cwd").write_text("/tmp\n")
            (path / "environment.sha256").write_text("0" * 64 + "\n")
            (path / "stdout.raw").write_bytes(b"")
            (path / "stderr.raw").write_bytes(b"")
            meta = {"exit": 0, "start_monotonic_ns": 1, "end_monotonic_ns": 2,
                    "wall_ns": 1, "stdout_bytes": 0, "stderr_bytes": 0,
                    "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                    "argv_sha256": hashlib.sha256(argv).hexdigest(),
                    "script": script, "site": f"{script}:1"}
            (path / "receipt.json").write_text(json.dumps(meta))

    def test_complete_graph(self):
        self.assertEqual(RECEIPTS.verify(self.root, set())["receipts"], 9)

    def test_native_log_seam_requires_every_command_image(self):
        log = self.root / "native-events.log"
        events = ["g3c4-native-exec-v2"]
        digest = "0" * 64

        def exec_event(pid, parent, argv):
            return (f"exec\t1\t{pid}\t{parent}\t1\t1\t1\t{argv.hex()}\t"
                    f"{b'/bin/bash'.hex()}\t{b'/tmp'.hex()}\t{digest}")

        argv = [(self.root / f"command.{i}" / "argv.nul").read_bytes()
                for i in range(9)]
        events.append(exec_event(100, 0, argv[0]))
        for i in range(1, 9):
            events.extend((f"spawn\t1\t100\t{100+i}\tfork\t{100+i}\t1",
                           exec_event(100+i, 100, argv[i]),
                           f"exit\t1\t{100+i}\t0\t0"))
        events.extend(("exit\t2\t100\t0\t0", "complete\t2\t100\t0\t8\t9\t9\t0"))
        log.write_text("\n".join(events) + "\n")
        summary = RECEIPTS.verify(self.root, set(), log)
        self.assertEqual(summary["matched_receipts"], 9)
        self.assertEqual(summary["unmatched_native_execs"], 0)
        log.write_text("\n".join(events[:-1] +
                                 [events[-1].replace("\t8\t9\t9\t0", "\t8\t8\t9\t0")]) + "\n")
        with self.assertRaisesRegex(ValueError, "event count mismatch"):
            RECEIPTS.verify(self.root, set(), log)

    def test_thread_merge_needs_group_provenance_and_following_exec(self):
        log = self.root / "native-events.log"
        argv = b"/bin/true\0"
        digest = "0" * 64

        def exec_event(when, ticks):
            return (f"exec\t{when}\t100\t0\t{ticks}\t1\t1\t{argv.hex()}\t"
                    f"{b'/bin/true'.hex()}\t{b'/tmp'.hex()}\t{digest}")

        def check(kind, tgid, following, *, next_ticks=2):
            events = ["g3c4-native-exec-v2", exec_event(1, 1),
                      f"spawn\t2\t100\t101\t{kind}\t{tgid}\t2",
                      "merge\t3\t101\t100"]
            if following:
                events.append(exec_event(4, next_ticks))
            events.extend(("exit\t5\t100\t0\t0",
                           f"complete\t6\t100\t0\t1\t{1+int(following)}\t1\t1"))
            log.write_text("\n".join(events) + "\n")
            return RECEIPTS.g3c4_native_trace_receipt.verify(log, [(argv, 0, 5)])

        self.assertEqual(check("clone", 100, True)["unmatched_native_execs"], 1)
        with self.assertRaisesRegex(ValueError, "invalid thread merge provenance"):
            check("fork", 101, False)
        with self.assertRaisesRegex(ValueError, "invalid thread merge provenance"):
            check("clone", 101, True)
        with self.assertRaisesRegex(ValueError, "invalid spawn thread group"):
            check("fork", 100, True)
        with self.assertRaisesRegex(ValueError, "merge lacks immediate destination exec"):
            check("clone", 100, False)
        with self.assertRaisesRegex(ValueError, "invalid exec lineage or task identity"):
            check("clone", 100, True, next_ticks=99)
        check("clone", 100, True)
        log.write_text(log.read_text().replace("g3c4-native-exec-v2",
                                               "g3c4-native-exec-v1", 1))
        with self.assertRaisesRegex(ValueError, "bad header"):
            RECEIPTS.g3c4_native_trace_receipt.verify(log, [(argv, 0, 5)])

    def test_missing_nested_receipt(self):
        path = self.root / "command.8" / "receipt.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing receipt"):
            RECEIPTS.verify(self.root, set())

    def test_missing_field(self):
        path = self.root / "command.0" / "receipt.json"
        meta = json.loads(path.read_text())
        del meta["exit"]
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "missing field"):
            RECEIPTS.verify(self.root, set())

    def test_duplicate_field(self):
        path = self.root / "command.0" / "receipt.json"
        text = path.read_text()
        path.write_text(text.replace('"exit": 0', '"exit": 0, "exit": 0'))
        with self.assertRaisesRegex(ValueError, "duplicate receipt field"):
            RECEIPTS.verify(self.root, set())

    def test_raw_stream_corruption(self):
        (self.root / "command.0" / "stdout.raw").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "stream mismatch"):
            RECEIPTS.verify(self.root, set())

    def test_symlink_root(self):
        alias = self.root.parent / (self.root.name + "-alias")
        alias.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(alias.unlink)
        with self.assertRaisesRegex(ValueError, "symlink"):
            RECEIPTS.verify(alias, set())

    def test_unexecuted_site(self):
        with self.assertRaisesRegex(ValueError, "unexecuted command sites"):
            RECEIPTS.verify(self.root, {"test-g3c4-p2g2-terminal-close.sh:999"})

    def test_same_line_pipeline_requires_both_programs(self):
        site = "test-g3c4-p2g2-terminal-close.sh:77"
        for number, program in ((9, "awk"), (10, "sort")):
            path = self.root / f"command.{number}"
            shutil.copytree(self.root / "command.0", path)
            argv = f"{program}\0x\0".encode()
            (path / "argv.nul").write_bytes(argv)
            meta = json.loads((path / "receipt.json").read_text())
            meta["argv_sha256"] = hashlib.sha256(argv).hexdigest()
            meta["site"] = site
            (path / "receipt.json").write_text(json.dumps(meta))
        expected = {site: Counter({"awk": 1, "sort": 1})}
        RECEIPTS.verify(self.root, expected)
        path = self.root / "command.10" / "receipt.json"
        meta = json.loads(path.read_text())
        meta["site"] = "other:1"
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "missing command receipt.*sort"):
            RECEIPTS.verify(self.root, expected)

    def test_common_and_q0_are_required_source_sites(self):
        calls = RECEIPTS.source_calls()
        self.assertIn(RECEIPTS.Q0_SITE, calls)
        self.assertTrue(any(site.startswith("common.sh:") for site in calls))

    def test_common_probe_records_call_site(self):
        env = dict(os.environ, G3C4_PROOF_RECEIPTS=str(self.root),
                   G3C4_PROOF_SCRIPT=RECEIPTS.SCRIPTS[0])
        command = "source scripts/common.sh; source scripts/g3c4_proof_command.sh; lock_value supported_cc >/dev/null"
        subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, check=True)
        sites = {json.loads(path.read_text())["site"] for path in self.root.glob("command.*/receipt.json")}
        self.assertIn(next(site for site in RECEIPTS.source_calls()
                           if site.startswith("common.sh:") and RECEIPTS.source_calls()[site]["awk"]), sites)

    def test_q0_python_git_child_is_receipted(self):
        env = dict(os.environ, G3C4_PROOF_RECEIPTS=str(self.root),
                   G3C4_PROOF_SCRIPT=RECEIPTS.SCRIPTS[0])
        helper = ROOT / "scripts/g3c4_proof_command.py"
        run = subprocess.run(["python3", str(helper), str(self.root), "--",
                              "python3", "-m", "unittest", "-q",
                              "tests.q0.test_python_isolation"],
                             cwd=ROOT, env=env, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        RECEIPTS.verify(self.root, {RECEIPTS.Q0_SITE: Counter({"git": 1})})


class EmittedPostflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for mode in EMITTED.MODES:
            for fixture in EMITTED.FIXTURES:
                directory = self.root / mode
                emitted = directory / "emitted"
                emitted.mkdir(parents=True, exist_ok=True)
                prefix = emitted / f"{fixture}.raw.o"
                for path in (prefix, Path(str(prefix) + ".bc"),
                             Path(str(prefix) + ".ll"), emitted / f"{fixture}.d",
                             emitted / f"{fixture}.map",
                             emitted / f"{fixture}.relinked",
                             directory / f"composer-{fixture}"):
                    path.write_bytes(b"same")

    def test_complete(self):
        self.assertEqual(EMITTED.verify(self.root)["files"], 126)

    def test_missing_ir_depfile_and_map(self):
        for suffix in ("terminal.raw.o.ll", "terminal.d", "terminal.map"):
            with self.subTest(suffix=suffix):
                path = self.root / "normal" / "emitted" / suffix
                original = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, "missing emitted input"):
                    EMITTED.verify(self.root)
                path.write_bytes(original)

    def test_binary_mismatch(self):
        (self.root / "sanitize" / "emitted" / "terminal.relinked").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "binary equivalence mismatch"):
            EMITTED.verify(self.root)


if __name__ == "__main__":
    unittest.main()
