"""Executable admission negatives for the versioned I2 wrapper imports."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "i2_guard_v1_normal_undefined_symbols.txt",
    "i2_guard_v1_sanitized_noninstrumentation_undefined_symbols.txt",
    "i2_guard_v1_sanitized_instrumentation_undefined_symbols.txt",
)
CHECK = r'''
set -euo pipefail
PROJECT_ROOT=$1
raw_private_root="$PROJECT_ROOT/tests/e3_private/zero_control_driver_root.esk"
raw_package_bridge="$PROJECT_ROOT/tests/e3_private/zero_control_bridge.c"
raw_package_renames="$PROJECT_ROOT/tests/e3_private/zero_control_package_private_renames.txt"
raw_public_exports="$PROJECT_ROOT/tests/e3_private/zero_control_package_public_exports.txt"
raw_include_dirs=("$PROJECT_ROOT/internal/p1/lib" "$PROJECT_ROOT/internal/c1/lib"
  "$PROJECT_ROOT/internal/t1/lib" "$PROJECT_ROOT/src"
  "$PROJECT_ROOT/internal/d2/lib" "$PROJECT_ROOT/internal/e3/lib")
die() { printf '%s\n' "$*" >&2; exit 77; }
source "$PROJECT_ROOT/scripts/e3-private-package-policy.sh"
source "$PROJECT_ROOT/scripts/e3-i2-guard-import-policy.sh"
PROJECT_ROOT=$2
package_policy=$3
e3_tuple_kind=$4
e3_i2_guard_length=$5
e3_i2_mutant=$6
e3_zero_sanitize=$7
e3_i2_guard_import_manifest "$8"
'''


def check(fake: Path, role: str, *, policy: str = "e3-private-aggregate",
          tuple_kind: str = "zero", length: str = "1", mutant: str = "",
          sanitizer: str = "0") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", CHECK, "check", str(ROOT), str(fake), policy,
         tuple_kind, length, mutant, sanitizer, role],
        capture_output=True, text=True, check=False,
    )


class I2GuardImportPolicyTests(unittest.TestCase):
    def prepare(self, temporary: str) -> Path:
        fake = Path(temporary) / "fake"
        manifests = fake / "tests/e3_private"
        manifests.mkdir(parents=True)
        for name in NAMES:
            shutil.copyfile(ROOT / "tests/e3_private" / name, manifests / name)
        return fake

    def test_exact_tuple_and_mode_admission(self) -> None:
        with TemporaryDirectory(prefix="e3-i2-import-policy-") as temporary:
            fake = self.prepare(temporary)
            for length in ("1", "3"):
                for mutant in ("", "omit-abort"):
                    for role in ("normal", "sanitized_noninstrumentation",
                                 "sanitized_instrumentation"):
                        sanitizer = "0" if role == "normal" else "1"
                        with self.subTest(length=length, mutant=mutant, role=role):
                            result = check(fake, role, length=length,
                                           mutant=mutant, sanitizer=sanitizer)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertEqual(
                                result.stdout,
                                f"{fake}/tests/e3_private/i2_guard_v1_{role}_undefined_symbols.txt\n",
                            )
            for changes in (
                {"policy": "m3-model-aggregate"}, {"tuple_kind": "base"},
                {"length": "2"}, {"mutant": "different"},
                {"sanitizer": "0"},
            ):
                with self.subTest(changes=changes):
                    result = check(fake, "sanitized_instrumentation", **changes)
                    self.assertEqual(result.returncode, 77, result.stderr)
            self.assertEqual(check(fake, "unknown", sanitizer="1").returncode, 77)

    def test_missing_extra_and_symlinked_manifest_rejected(self) -> None:
        with TemporaryDirectory(prefix="e3-i2-import-policy-") as temporary:
            fake = self.prepare(temporary)
            manifest = fake / "tests/e3_private" / NAMES[0]
            original = manifest.read_bytes()
            manifest.write_bytes(original + b"unexpected_import\n")
            self.assertEqual(check(fake, "normal").returncode, 77)
            manifest.write_bytes(original[:-1])
            self.assertEqual(check(fake, "normal").returncode, 77)
            manifest.unlink()
            self.assertEqual(check(fake, "normal").returncode, 77)
            genuine = Path(temporary) / "genuine.txt"
            genuine.write_bytes(original)
            manifest.symlink_to(genuine)
            self.assertEqual(check(fake, "normal").returncode, 77)
            manifest.unlink()
            manifest.write_bytes(original)
            directory = manifest.parent
            alias_target = Path(temporary) / "authentic"
            directory.rename(alias_target)
            directory.symlink_to(alias_target, target_is_directory=True)
            self.assertEqual(check(fake, "normal").returncode, 77)

    def test_builder_dispatch_leaves_unwrapped_control_on_own_policy(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('undefined_symbols="$(e3_i2_guard_import_manifest normal)"', builder)
        self.assertIn('zero_sanitized="$(e3_i2_guard_import_manifest sanitized_noninstrumentation)"',
                      builder)
        self.assertIn('zero_instrumentation="$(e3_i2_guard_import_manifest sanitized_instrumentation)"',
                      builder)
        self.assertEqual(builder.count('e3_i2_guard_import_manifest'), 3)
        self.assertIn('zero_sanitized="${PROJECT_ROOT}/tests/e3_private/zero_control_package_sanitized_undefined_symbols.txt"',
                      builder)
        self.assertIn('zero_instrumentation="${PROJECT_ROOT}/tests/e3_private/zero_control_package_sanitizer_undefined_symbols.txt"',
                      builder)


if __name__ == "__main__":
    unittest.main()
