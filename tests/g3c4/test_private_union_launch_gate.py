"""Synthetic decision denials for the still-HOLD one-root AOT route."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import tempfile
from unittest import TestCase, mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run-g3c4-public-t1-private-union.py"
SPEC = importlib.util.spec_from_file_location("private_union_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def decision() -> dict:
    rows = []
    for index, role in enumerate(sorted(gate.REQUIRED_ROLES)):
        name = f"/synthetic/{role}-{index}"
        if role.endswith("_archive"):
            lane, mode = role.split("_")[:2]
            name = f"/synthetic/libg3c4_private_{lane}_{mode}.a"
        rows.append({"role": role, "path": name, "sha256": "a" * 64,
                     "size": 1, "mode": 0o644, "uid": 1, "gid": 1})
    return {"format": gate.FORMAT, "source_basis_head": gate.SOURCE_BASIS_HEAD,
            "source_basis_tree": gate.SOURCE_BASIS_TREE, "base_head": gate.BASE_HEAD,
            "runner_projection_sha256": gate.runner_projection(),
            "preflight_sha256": gate.sha(gate.ordinary(gate.PREFLIGHT)),
            "root_review_sha256": gate.ROOT_REVIEW, "image": gate.IMAGE,
            "pins": rows, "limits": {key: 1 for key in gate.LIMITS},
            "slot_receipt_sha256": "a" * 64}


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class LaunchGateTest(TestCase):
    def admit(self, value: dict, **overrides):
        raw = encoded(value)
        good = b'(load "g3g_manual_package_root.esk")\n'
        def git(_, command, *args):
            if command == "rev-parse":
                return (gate.SOURCE_BASIS_TREE + "\n").encode()
            if command == "merge-base":
                return b""
            self.fail(f"unexpected git request: {command} {args}")
        with mock.patch.object(gate.source, "git", side_effect=git):
            return gate.admit(raw, hashlib.sha256(raw).hexdigest(),
                              check_source=overrides.get("check_source", lambda root: None),
                              check_pin=overrides.get("check_pin", lambda row: None),
                              read_source=overrides.get("read_source",
                                                        lambda root, name: good))

    def test_exact_synthetic_shape_and_private_argv(self) -> None:
        _, rows = self.admit(decision())
        with tempfile.TemporaryDirectory() as temporary:
            owned = Path(temporary) / "owned"
            owned.mkdir(mode=0o700)
            source_root = owned / "source"
            source_root.mkdir()
            rows["eshkol_run"]["path"] = str(owned / "pins/eshkol_run/compiler")
            rows["history_normal_archive"]["path"] = str(
                owned / "pins/history_normal_archive/libg3c4_private_history_normal.a")
            argv = gate.aot_argv(rows, "history", "normal", owned / "history",
                                 source_root)
            self.assertEqual(argv.count(str(source_root / gate.source.HISTORY)), 1)
            self.assertEqual(argv.count(str(source_root / gate.source.T1)), 0)
            rows["eshkol_run"]["path"] = "/live/compiler"
            with self.assertRaisesRegex(ValueError, "owned staged"):
                gate.aot_argv(rows, "history", "normal", owned / "history",
                              source_root)
        self.assertEqual(argv[1:5], ("--strict-types", "--optimize", "0", "--no-stdlib"))
        self.assertEqual(argv[argv.index("--lib") + 1], "g3c4_private_history_normal")

    def test_no_embedded_launch_authority(self) -> None:
        self.assertEqual(gate.ACCEPTED_DECISION_SHA256, "")
        with self.assertRaisesRegex(ValueError, "no ROOT-approved"):
            gate.admit(encoded(decision()), gate.ACCEPTED_DECISION_SHA256)

    def test_source_basis_is_frozen_before_successor(self) -> None:
        self.admit(decision())
        value = decision()
        value["source_basis_tree"] = "2" * 40
        with self.assertRaisesRegex(ValueError, "source basis mismatch"):
            self.admit(value)
        value = decision()
        value["source_head"] = "3" * 40
        with self.assertRaisesRegex(ValueError, "schema"):
            self.admit(value)
        value = decision()
        value["runner_projection_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source projection"):
            self.admit(value)

    def test_runner_projection_erases_only_approval_value(self) -> None:
        raw = gate.ordinary(SCRIPT)
        approved = re.sub(rb'(?m)^ACCEPTED_DECISION_SHA256 = ""$',
                          b'ACCEPTED_DECISION_SHA256 = "' + b'a' * 64 + b'"', raw)
        self.assertNotEqual(raw, approved)
        with mock.patch.object(gate, "ordinary", return_value=approved):
            self.assertEqual(gate.runner_projection(), gate.sha(raw))
        changed = approved.replace(b'--no-stdlib', b'--with-stdlib', 1)
        with mock.patch.object(gate, "ordinary", return_value=changed):
            self.assertNotEqual(gate.runner_projection(), gate.sha(raw))

    def test_missing_pin_or_limit_denies(self) -> None:
        value = decision()
        value["pins"].pop()
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.admit(value)
        value = decision()
        value["limits"].pop("memory_bytes")
        with self.assertRaisesRegex(ValueError, "resource bounds"):
            self.admit(value)

    def test_changed_pin_and_dirty_source_deny(self) -> None:
        with self.assertRaisesRegex(ValueError, "drift"):
            self.admit(decision(), check_pin=lambda row: (_ for _ in ()).throw(
                ValueError("selected pin drift")))
        with self.assertRaisesRegex(ValueError, "dirty"):
            self.admit(decision(), check_source=lambda root: (_ for _ in ()).throw(
                ValueError("source checkout is dirty")))

    def test_duplicate_root_denies(self) -> None:
        duplicate = b'(load "g3g_manual_package_root.esk")\n' * 2
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.admit(decision(), read_source=lambda root, name: duplicate)

    def test_duplicate_pin_path_denies(self) -> None:
        value = decision()
        value["pins"][1]["path"] = value["pins"][0]["path"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.admit(value)

    def test_unbound_heavy_slot_denies(self) -> None:
        value = decision()
        value["slot_receipt_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "heavy slot"):
            self.admit(value)

    def test_actual_selected_byte_and_symlink_drift_denies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected"
            path.write_bytes(b"reviewed")
            info = path.stat()
            row = {"role": "tool_selected", "path": str(path),
                   "sha256": hashlib.sha256(b"reviewed").hexdigest(),
                   "size": 8, "mode": stat.S_IMODE(info.st_mode),
                   "uid": info.st_uid, "gid": info.st_gid}
            gate.verify_pin(row)
            path.write_bytes(b"changed!")
            with self.assertRaisesRegex(ValueError, "drift"):
                gate.verify_pin(row)
            path.unlink()
            path.symlink_to(Path(temporary) / "target")
            with self.assertRaisesRegex(ValueError, "ordinary"):
                gate.verify_pin(row)

    def test_owned_snapshot_survives_live_path_replacement(self) -> None:
        fixture = b'(load "g3g_manual_package_root.esk")\n'
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            live = base / "libg3c4_private_history_normal.a"
            live.write_bytes(b"reviewed")
            info = live.stat()
            row = {"role": "history_normal_archive", "path": str(live),
                   "sha256": hashlib.sha256(b"reviewed").hexdigest(),
                   "size": 8, "mode": stat.S_IMODE(info.st_mode),
                   "uid": info.st_uid, "gid": info.st_gid}
            owned = base / "owned"
            owned.mkdir(mode=0o700)
            with mock.patch.object(gate.source, "source_names",
                                   return_value={gate.source.HISTORY}), \
                 mock.patch.object(gate.source, "ordinary", return_value=fixture), \
                 mock.patch.object(gate.source, "git", side_effect=lambda _, command,
                                   *args: fixture if command == "show" else b"100644 blob"):
                staged, source_root = gate.stage_inputs({row["role"]: row}, owned)
            live.write_bytes(b"replaced")
            gate.verify_pin(staged[row["role"]])
            self.assertEqual(Path(staged[row["role"]]["path"]).read_bytes(), b"reviewed")
            self.assertEqual((source_root / gate.source.HISTORY).read_bytes(), fixture)
            self.assertNotIn(str(live), gate.aot_argv(
                {"eshkol_run": {"path": str(owned / "pins/eshkol_run/compiler")},
                 **staged}, "history", "normal", owned / "output", source_root))

    def test_stage_rejects_drift_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            live = base / "selected"
            live.write_bytes(b"changed")
            info = live.stat()
            row = {"role": "selected", "path": str(live),
                   "sha256": hashlib.sha256(b"reviewed").hexdigest(),
                   "size": 8, "mode": stat.S_IMODE(info.st_mode),
                   "uid": info.st_uid, "gid": info.st_gid}
            owned = base / "owned"
            owned.mkdir(mode=0o700)
            with self.assertRaisesRegex(ValueError, "drift while staging"):
                gate.stage_inputs({row["role"]: row}, owned)
