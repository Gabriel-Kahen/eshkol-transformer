#!/usr/bin/env python3
"""Small real-process controls for terminal command receipts."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/g3c4_terminal_command_receipt.sh"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_terminal_command_receipts import check_link_mapping, verify


class CommandReceiptTests(unittest.TestCase):
    def run_case(self, body: str):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            (evidence / "commands").mkdir()
            script = f'''set -euo pipefail
evidence={str(evidence)!r}
terminal_commands_root="$evidence/commands"
terminal_command_index=0
source {str(HELPER)!r}
{body}
'''
            process = subprocess.run(["bash", "-c", script], cwd=ROOT,
                                     capture_output=True, text=True)
            receipts = sorted((evidence / "commands").iterdir())
            result = [(path.name, (path / "exit").read_text(),
                       (path / "argv.nul").read_bytes(),
                       (path / "cwd").read_text(),
                       (path / "stdout.raw").read_bytes(),
                       (path / "stderr.raw").read_bytes())
                      for path in receipts]
            return process, result, evidence

    def test_success_records_exact_argv_cwd_streams_and_aliases(self):
        process, receipts, _ = self.run_case('''
terminal_record_command good good.stdout good.stderr -- \
  bash -c 'printf "out:%s" "$1"; printf "err" >&2' bash 'a b'
cmp "$evidence/good.stdout" "$evidence/commands/0001-good/stdout.raw"
cmp "$evidence/good.stderr" "$evidence/commands/0001-good/stderr.raw"
''')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(receipts[0][1], "0\n")
        self.assertEqual(receipts[0][2],
                         b'bash\0-c\0printf "out:%s" "$1"; printf "err" >&2\0bash\0a b\0')
        self.assertEqual(receipts[0][3], f"{ROOT}\n")
        self.assertEqual(receipts[0][4:], (b"out:a b", b"err"))

    def test_existing_evidence_alias_is_replaced_without_losing_old_raw(self):
        process, receipts, _ = self.run_case('''
terminal_record_command first latest.stdout latest.stderr -- bash -c 'printf first'
terminal_record_command second latest.stdout latest.stderr -- bash -c 'printf second'
test "$(cat "$evidence/latest.stdout")" = second
test "$(cat "$evidence/commands/0001-first/stdout.raw")" = first
''')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual([row[4] for row in receipts], [b"first", b"second"])

    def test_failure_preserves_raw_streams_and_numeric_exit(self):
        process, receipts, _ = self.run_case('''
status=0
terminal_record_command bad bad.stdout bad.stderr -- \
  bash -c 'printf fail; printf detail >&2; exit 17' || status=$?
test "$status" -eq 17
test -f "$evidence/bad.stdout"
test -f "$evidence/bad.stderr"
''')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(receipts[0][1], "17\n")
        self.assertEqual(receipts[0][4:], (b"fail", b"detail"))

    def test_timeout_is_not_masked(self):
        process, receipts, _ = self.run_case('''
status=0
terminal_record_command timed - - -- timeout 0.1s bash -c 'sleep 2' || status=$?
test "$status" -eq 124
''')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(receipts[0][1], "124\n")

    def test_toolchain_export_restores_common_shell_contract(self):
        process, receipts, _ = self.run_case('''
mkdir "$evidence/build"
printf 'cxx_path\\t/usr/bin/clang++-21\\n' >"$evidence/build/eshkol-transformer-provenance.tsv"
eshkol_build_dir() { printf '%s\\n' "$evidence/build"; }
tsv_value() { awk -F '\\t' -v key="$2" '$1 == key { print $2 }' "$1"; }
terminal_restore_toolchain_export
terminal_record_command export-check - - -- \
  bash -c 'test "$ESHKOL_CXX_COMPILER" = /usr/bin/clang++-21'
''')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(receipts[0][1], "0\n")

    def test_nested_link_receipts_are_collision_safe_and_replay_streams(self):
        process, receipts, _ = self.run_case('''
terminal_record_nested_link fixture bash -c 'printf one; printf err1 >&2' &
first=$!
terminal_record_nested_link fixture bash -c 'printf two; printf err2 >&2' &
second=$!
wait "$first"
wait "$second"
''')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(receipts), 2)
        self.assertNotEqual(receipts[0][0], receipts[1][0])
        self.assertEqual({row[4] for row in receipts}, {b"one", b"two"})
        self.assertEqual({row[5] for row in receipts}, {b"err1", b"err2"})
        self.assertEqual({row[1] for row in receipts}, {"0\n"})
        self.assertIn("one", process.stdout)
        self.assertIn("two", process.stdout)

    def test_nested_link_failure_exit_and_raw_streams(self):
        process, receipts, _ = self.run_case('''
status=0
terminal_record_nested_link failing bash -c 'printf out; printf err >&2; exit 23' || status=$?
test "$status" -eq 23
''')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(receipts[0][1], "23\n")
        self.assertEqual(receipts[0][4:], (b"out", b"err"))
        self.assertEqual(process.stdout, "out")
        self.assertEqual(process.stderr, "err")

    def test_receipt_audit_rejects_missing_exit_and_raw_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            receipt = evidence / "commands/0001-fixture"
            receipt.mkdir(parents=True)
            for name, data in {
                "label": b"fixture\n", "cwd": b"/tmp\n",
                "argv.nul": b"true\0", "exit": b"0\n",
                "stdout.raw": b"", "stderr.raw": b"",
            }.items():
                (receipt / name).write_bytes(data)
            self.assertEqual(verify(evidence), 1)
            (receipt / "exit").unlink()
            with self.assertRaises(ValueError):
                verify(evidence)
            (receipt / "exit").write_bytes(b"0\n")
            (receipt / "stderr.raw").unlink()
            with self.assertRaises(ValueError):
                verify(evidence)

    def test_missing_duplicate_and_orphan_nested_link_rejected(self):
        with self.assertRaises(ValueError):
            check_link_mapping({"normal-link"}, [], False)
        with self.assertRaises(ValueError):
            check_link_mapping({"normal-link"}, ["normal-link", "normal-link"], False)
        with self.assertRaises(ValueError):
            check_link_mapping(set(), ["normal-link"], False)
        check_link_mapping({"normal-link"}, ["normal-link"], False)


if __name__ == "__main__":
    unittest.main()
