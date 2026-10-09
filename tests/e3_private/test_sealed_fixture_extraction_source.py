"""Pure controls for the inactive, nonpublishing E3 inventory extraction."""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build-e1b-consumer.sh"
WRAPPER = ROOT / "scripts/extract-e3-sealed-fixture-inventories.sh"
OWNED = ROOT / "scripts/e3-sealed-fixture-extract-owned-tmp.sh"
RECEIPTS = ROOT / "scripts/e3-sealed-fixture-check-receipts.py"


def source_boundary(body: str) -> None:
    guard = body.index('if [[ "${E3_SEALED_FIXTURE_EXTRACT:-0}" != 0 ]]')
    admission = body.index('[[ "${e3_tuple_kind}" == sealed-fixture', guard)
    observed = body.index('if [[ "${e3_sealed_fixture_extract}" == 1 ]]; then\n'
                          '  # Observation stops here')
    raw_undefined = body.index('nm -u --format=posix "${e1b_tmp}/combined.o"', observed - 200)
    ordinary = body.index('e3_retention_sanitizer_inventory_verified=0', observed)
    publication = body.index('mv "${temporary_output}" "${output_object}"', ordinary)
    stop = body.index('  exit 0\nfi', observed)
    if not (guard < admission < raw_undefined < observed < stop < ordinary < publication):
        raise ValueError("extraction crosses the ordinary acceptance boundary")
    required = ('private.ll', 'private.d', 'source-closure.txt',
                'native-source-closure.txt', 'combined.raw.o', 'combined.o',
                'raw-global-defined.posix', 'raw-undefined.posix',
                'localized-readelf-symbols.txt', 'probe-archive-members.txt',
                'private.asan.ll', 'private.asan-imports.txt')
    if any(name not in body[observed:stop] for name in required):
        raise ValueError("extraction omits a raw emitted artifact")


class ExtractionSource(unittest.TestCase):
    def test_selected_boundary_and_mutations(self) -> None:
        body = BUILDER.read_text()
        source_boundary(body)
        begin = body.index('  # Observation stops here')
        end = body.index('e3_retention_sanitizer_inventory_verified=0', begin)
        for missing in ('  exit 0\nfi', 'private.asan-imports.txt',
                        'probe-archive-members.txt'):
            selected = body[begin:end]
            with self.assertRaises(ValueError):
                source_boundary(body[:begin] + selected.replace(missing, '', 1) + body[end:])
        self.assertIn('if [[ "${e3_sealed_fixture_extract}" != 1 ]]; then\n'
                      '  [[ -f "${undefined_symbols}" ]]', body)
        self.assertIn('sealed-fixture) [[ "${e3_sealed_fixture_extract}" == 1 ]]', body)

    def test_foreign_tuple_denied_before_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            prefix = ROOT / "native/e3_private_package"
            argv = ["/usr/bin/bash", str(BUILDER),
                    str(ROOT / "native/e3_private_driver_root.esk"),
                    str(ROOT / "native/e3_private_bridge.c"),
                    str(prefix) + "_private_renames.txt",
                    str(prefix) + "_public_exports.txt",
                    str(Path(temp) / "foreign.o"),
                    *(str(ROOT / name) for name in
                      ("internal/p1/lib", "internal/c1/lib", "internal/t1/lib",
                       "src", "internal/d2/lib", "internal/e3/lib"))]
            env = dict(os.environ, E3_SEALED_FIXTURE_EXTRACT="1")
            result = subprocess.run(argv, cwd=ROOT, env=env, text=True,
                                    capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exact private sealed-fixture tuple", result.stderr)
            self.assertFalse((Path(temp) / "foreign.o").exists())

    def test_wrapper_preflight_and_normal_denial(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "extract"
            unknown = subprocess.run(["/usr/bin/bash", str(WRAPPER),
                                      str(destination), "foreign"], cwd=ROOT,
                                     capture_output=True, text=True)
            self.assertNotEqual(unknown.returncode, 0)
            destination.mkdir()
            reused = subprocess.run(["/usr/bin/bash", str(WRAPPER),
                                     str(destination), "normal"], cwd=ROOT,
                                    capture_output=True, text=True)
            self.assertNotEqual(reused.returncode, 0)
            self.assertIn("must be fresh", reused.stderr)
            self.assertFalse((destination / "e3_sealed_fixture_package.o").exists())
            ordinary = subprocess.run(
                ["/usr/bin/bash", str(ROOT / "scripts/build-e3-sealed-fixture-test.sh"),
                 str(Path(temp) / "ordinary"), "normal"], cwd=ROOT,
                capture_output=True, text=True)
            self.assertNotEqual(ordinary.returncode, 0)
            self.assertIn("emitted manifest is not admitted", ordinary.stderr)

    def test_inputs_and_status_not_reused_as_baseline(self) -> None:
        wrapper = WRAPPER.read_text()
        self.assertIn('sha256sum -- "${inputs[@]}" >"${stage}/inputs.before.sha256"', wrapper)
        self.assertIn('sha256sum -- "${inputs[@]}" >"${stage}/inputs.after.sha256"', wrapper)
        self.assertIn('cmp -s "${stage}/inputs.before.sha256"', wrapper)
        self.assertIn('! -e "${stage}/e3_sealed_fixture_package.o"', wrapper)
        self.assertIn('! -e "${stage}/e3_sealed_fixture_package.o.evidence"', wrapper)
        self.assertIn('compiler.exit.raw', OWNED.read_text())
        self.assertIn(str(RECEIPTS.relative_to(ROOT)), wrapper)
        self.assertIn('E3_SEALED_FIXTURE_EXTRACT=1', wrapper)
        self.assertIn('e3_extract_owned_command "${stage}" /usr/bin/env', wrapper)

    def test_mocked_failure_retains_builder_tmp_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / ".e3-sealed-fixture.ABC123"
            stage.mkdir()
            script = r'''
set -euo pipefail
source "$1"
shift
e3_extract_owned_command "$@"
'''
            child = 'mkdir "$TMPDIR/emitted"; printf raw-private-ir >"$TMPDIR/emitted/private.ll"; exit 23'
            result = subprocess.run(["/usr/bin/bash", "-c", script, "test",
                                     str(OWNED), str(stage), "/usr/bin/bash", "-c", child],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 23)
            self.assertEqual((stage / "compiler.exit.raw").read_text(), "23\n")
            self.assertEqual((stage / "tmp/emitted/private.ll").read_bytes(), b"raw-private-ir")
            self.assertFalse((stage / "e3_sealed_fixture_package.o").exists())

    def test_child_success_with_failed_status_write_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "stage"
            stage.mkdir(mode=0o700)
            command = 'ln -s /dev/full "$1/compiler.exit.raw"; printf retained-stdout; exit 0'
            script = 'set -euo pipefail; source "$1"; shift; e3_extract_owned_command "$@"'
            result = subprocess.run(
                ["/usr/bin/bash", "-c", script, "test", str(OWNED), str(stage),
                 "/usr/bin/bash", "-c", command, "child", str(stage)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((stage / "compiler.stdout.raw").read_bytes(), b"retained-stdout")
            self.assertTrue((stage / "compiler.exit.raw").is_symlink())
            self.assertIn("compiler.exit.raw", result.stderr)

    def test_receipt_checker_denies_foreign_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / name for name in
                     ("argv", "stdout", "stderr", "exit")]
            for path in paths[:-1]:
                path.write_bytes(b"raw")
            paths[-1].write_bytes(b"23\n")
            cmd = ["/usr/bin/python3", "-B", str(RECEIPTS), *(str(p) for p in paths)]
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            for bad in (b"23\nextra", b"256\n", b"+0\n"):
                paths[-1].write_bytes(bad)
                self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            paths[-1].unlink()
            paths[-1].symlink_to("/dev/full")
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            paths[-1].unlink()
            paths[-1].write_bytes(b"0\n")
            paths[1].unlink()
            paths[1].symlink_to(paths[0])
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            paths[1].unlink()
            os.link(paths[0], paths[1])
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(cmd[:-1] + [str(paths[0])],
                                              capture_output=True).returncode, 0)

    def test_checked_receipt_write_rejects_full_device(self) -> None:
        spec = importlib.util.spec_from_file_location("e3_receipts", RECEIPTS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fd = os.open("/dev/full", os.O_WRONLY)
        try:
            with self.assertRaises(OSError):
                module._write_checked(fd, b"0\n")
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
