"""Owned source-only controls for full F image-loader closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.g3c4 import f_artifact_attest as attest
from tests.g3c4 import f_image_loader_collect as collect
from tests.g3c4 import check_f_image_loader_collect as source_check


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class ImageCollectorTests(unittest.TestCase):
    def test_source_closure_and_mutants(self) -> None:
        source = (source_check.ROOT / source_check.ADDED[0]).read_text()
        base = source_check.BASE.read_text().splitlines()
        closure = source_check.CLOSURE.read_text().splitlines()
        originals = {name: (source_check.ROOT / name).read_bytes()
                     for name in source_check.UNCHANGED}
        source_check.validate(source, base, closure, originals)
        for changed in (source.replace("os.WNOWAIT", "0"),
                        source.replace("unadmitted loader dependency", "ignored", 1),
                        source.replace('"--debug-dump=no-follow-links", ', "")):
            with self.assertRaises(AssertionError):
                source_check.validate(changed, base, closure, originals)
        with self.assertRaises(AssertionError):
            source_check.validate(source, base, closure[:-1], originals)

    def test_actual_saved_f31_pin_sources(self) -> None:
        base = Path("/home/gabe/.codex/evidence/eshkol-transformer/"
                    "g3c4-c4-split-outer-run-20261009-b/inputs")
        tools, runtime = collect.pin_tables(base / "tools.tsv",
                                            base / "runtimes.tsv")
        self.assertEqual(len(runtime), 33)
        self.assertEqual(tools["/usr/bin/readelf"][1],
                         "04db0000749aff89e4af21429340b00b536fc6f80e811c872c006507881a5560")
        self.assertIn("/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2", runtime)
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "runtime.tsv"
            changed.write_bytes((base / "runtimes.tsv").read_bytes() +
                                b"0" * 64 + b"\t/unreviewed\n")
            with self.assertRaisesRegex(ValueError, "pin SHA drift"):
                collect.pin_tables(base / "tools.tsv", changed)

    def test_owned_capture_success_overflow_timeout_and_actual_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = collect.capture([sys.executable, "-c",
                                   "import sys;sys.stdout.write('ok\\n')"],
                                  base / "success", 3)
            self.assertEqual(raw, b"ok\n")
            receipt = json.loads((base / "success/status.json").read_bytes())
            self.assertTrue(receipt["quiescent"])
            self.assertEqual(receipt["returncode"], 0)
            self.assertIsInstance(receipt["wait_status"], int)
            with self.assertRaisesRegex(ValueError, "stdout overflow"):
                collect.capture([sys.executable, "-c",
                                 "import os;os.write(1,b'x'*2100000)"],
                                base / "stdout-overflow", 3)
            self.assertEqual((base / "stdout-overflow/stdout.raw").stat().st_size,
                             collect.STDOUT_LIMIT)
            with self.assertRaisesRegex(ValueError, "stderr overflow"):
                collect.capture([sys.executable, "-c",
                                 "import os;os.write(2,b'x'*70000)"],
                                base / "stderr-overflow", 3)
            self.assertEqual((base / "stderr-overflow/stderr.raw").stat().st_size,
                             collect.STDERR_LIMIT)
            with self.assertRaisesRegex(ValueError, "timeout"):
                collect.capture([sys.executable, "-c", "import time;time.sleep(5)"],
                                base / "timeout", 1)
            self.assertTrue((base / "timeout/status.json").is_file())
            with self.assertRaisesRegex(ValueError, "exit=7"):
                collect.capture([sys.executable, "-c", "raise SystemExit(7)"],
                                base / "nonzero", 3)
            with self.assertRaisesRegex(ValueError, "unexpected collector descendants"):
                collect.capture([sys.executable, "-c",
                                 "import os,time; pid=os.fork(); "
                                 "time.sleep(5) if pid==0 else os._exit(0)"],
                                base / "descendant-writer", 3)
            self.assertTrue((base / "descendant-writer/status.json").is_file())

    def test_loader_exact_runtime_subset_and_mutants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            loader = base / "ld-linux-x86-64.so.2"
            direct = base / "libdirect.so.1"
            transitive = base / "libtransitive.so.2"
            for path in (loader, direct, transitive):
                path.write_bytes(path.name.encode())
            runtime = {str(path): sha(path.read_bytes())
                       for path in (loader, direct, transitive)}
            raw = (f"\tlibdirect.so.1 => {direct} (0x0123)\n"
                   f"\tlibtransitive.so.2 => {transitive} (0x0456)\n"
                   f"\t{loader} (0x0789)\n").encode()
            pin_raw = collect.loader_rows(raw, runtime)
            self.assertEqual(len(pin_raw.splitlines()), 3)
            self.assertIn(b"libtransitive.so.2", pin_raw)
            with self.assertRaisesRegex(ValueError, "unadmitted loader dependency"):
                collect.loader_rows(raw, {str(loader): runtime[str(loader)],
                                          str(direct): runtime[str(direct)]})
            with self.assertRaisesRegex(ValueError, "unknown loader row"):
                collect.loader_rows(raw + b"foreign line\n", runtime)
            transitive.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "resolved image member drift"):
                collect.loader_rows(raw, runtime)

    def test_synthetic_two_phase_collector_and_extra_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build = base / "build"
            build.mkdir()
            pins = {}
            for name in attest.ARTIFACTS:
                path = build / Path(name).name
                path.write_bytes(name.encode())
                pins[name] = sha(path.read_bytes())
            binary = build / "eshkol-run"
            direct = base / "libdirect.so.1"
            direct.write_bytes(b"direct")
            loader = base / "ld-linux-x86-64.so.2"
            readelf = base / "readelf"
            elf = " 0x1 (NEEDED) Shared library: [libdirect.so.1]\\n 0x1 (RUNPATH) Library runpath: [/pinned]\\n"
            readelf.write_text(
                "#!/bin/sh\n"
                "test \"$1\" = --debug-dump=no-follow-links && "
                "test \"$2\" = -d && test \"$3\" = "
                f"'{binary}' || {{ printf 'wrong readelf argv\\n' >&2; exit 9; }}\n"
                f"printf '{elf}'\n")
            loader.write_text(f"#!/bin/sh\nprintf '\\tlibdirect.so.1 => {direct} (0x0123)\\n\\t{loader} (0x0456)\\n'\n")
            readelf.chmod(0o700)
            loader.chmod(0o700)
            tools = base / "tools.tsv"
            runtimes = base / "runtime.tsv"
            tools.write_text(f"{readelf}\t{readelf}\t{sha(readelf.read_bytes())}\n")
            runtimes.write_text(
                f"{sha(loader.read_bytes())}\t{loader}\n"
                f"{sha(direct.read_bytes())}\t{direct}\n")
            real_resolve = attest.resolved_image_libs
            def resolve(raw_elf, _synthetic_argv, exit_raw, stdout, stderr, pins):
                canonical_argv = (b"/lib64/ld-linux-x86-64.so.2\0--list\0"
                                  b"/out/build-off/eshkol-run\0")
                return real_resolve(raw_elf, canonical_argv, exit_raw,
                                    stdout, stderr, pins)
            with patch.object(attest, "F_PINS", pins), \
                 patch.object(attest, "ELF_NEEDED", frozenset({"libdirect.so.1"})), \
                 patch.object(attest, "ELF_RUNPATH", "/pinned"), \
                 patch.object(attest, "resolved_image_libs", side_effect=resolve):
                receipt = collect.collect(
                    binary, tools, runtimes, base / "evidence",
                    expected_tools=sha(tools.read_bytes()),
                    expected_runtimes=sha(runtimes.read_bytes()),
                    build=build, readelf=readelf, loader=loader)
                readelf_status = json.loads(
                    (base / "evidence/readelf/status.json").read_bytes())
                self.assertEqual(readelf_status["argv"],
                                 [str(readelf), "--debug-dump=no-follow-links",
                                  "-d", str(binary)])
                self.assertEqual([row["name"] for row in receipt["loader_closure"]],
                                 ["ld-linux-x86-64.so.2", "libdirect.so.1"])
                self.assertEqual(len(receipt["direct_needed"]), 1)
                self.assertEqual(len((base / "evidence/image-libraries.tsv").read_text().splitlines()), 2)
                readelf.write_text("#!/bin/sh\nprintf 'pinned warning\\n' >&2\n"
                                   f"printf '{elf}'\n")
                tools.write_text(f"{readelf}\t{readelf}\t{sha(readelf.read_bytes())}\n")
                with self.assertRaisesRegex(ValueError, "stderr=15"):
                    collect.collect(binary, tools, runtimes, base / "warning",
                                    expected_tools=sha(tools.read_bytes()),
                                    expected_runtimes=sha(runtimes.read_bytes()),
                                    build=build, readelf=readelf, loader=loader)
                warning_status = json.loads(
                    (base / "warning/readelf/status.json").read_bytes())
                self.assertEqual((base / "warning/readelf/stderr.raw").read_bytes(),
                                 b"pinned warning\n")
                self.assertEqual(warning_status["returncode"], 0)
                self.assertTrue(warning_status["quiescent"])
                self.assertFalse((base / "warning/collector.json").exists())
                self.assertFalse((base / "warning/loader").exists())
                readelf.write_text(
                    "#!/bin/sh\n"
                    "test \"$1\" = --debug-dump=no-follow-links && "
                    "test \"$2\" = -d && test \"$3\" = "
                    f"'{binary}' || {{ printf 'wrong readelf argv\\n' >&2; exit 9; }}\n"
                    f"printf '{elf}'\n")
                tools.write_text(f"{readelf}\t{readelf}\t{sha(readelf.read_bytes())}\n")
                extra = base / "unadmitted.so"
                extra.write_bytes(b"unadmitted")
                loader.write_text(f"#!/bin/sh\nprintf '\\tlibdirect.so.1 => {direct} (0x0123)\\n\\tlibunadmitted.so => {extra} (0x0234)\\n\\t{loader} (0x0456)\\n'\n")
                runtimes.write_text(
                    f"{sha(loader.read_bytes())}\t{loader}\n"
                    f"{sha(direct.read_bytes())}\t{direct}\n")
                with self.assertRaisesRegex(ValueError, "unadmitted loader dependency"):
                    collect.collect(binary, tools, runtimes, base / "denied",
                                    expected_tools=sha(tools.read_bytes()),
                                    expected_runtimes=sha(runtimes.read_bytes()),
                                    build=build, readelf=readelf, loader=loader)
                self.assertTrue((base / "denied/loader/status.json").is_file())


if __name__ == "__main__":
    unittest.main()
