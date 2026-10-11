"""Tiny generated inputs for the inert OCI export acquisition source."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/e3-n1-oci-export-acquisition.py"
spec = importlib.util.spec_from_file_location("e3_n1_oci_export_acquisition", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(path, with_sha=False):
    st = path.lstat()
    result = {"path": str(path), "resolved_path": str(path.resolve()),
              "mode": st.st_mode, "size": st.st_size, "device": st.st_dev,
              "inode": st.st_ino, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}
    if with_sha:
        result["sha256"] = sha(path.read_bytes())
    return result


def fixture(root):
    layer_data = [b"tiny layer zero", b"tiny layer one"]
    layers = [{"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
               "digest": "sha256:" + sha(data), "size": len(data)} for data in layer_data]
    config = json.dumps({"architecture": "amd64", "os": "linux", "rootfs": {
        "diff_ids": ["sha256:" + sha(b"expanded0"), "sha256:" + sha(b"expanded1")]}}).encode()
    config_desc = {"mediaType": "application/vnd.oci.image.config.v1+json",
                   "digest": "sha256:" + sha(config), "size": len(config)}
    manifest = json.dumps({"config": config_desc, "layers": layers}).encode()
    manifest_desc = {"mediaType": "application/vnd.oci.image.manifest.v1+json",
                     "digest": "sha256:" + sha(manifest), "size": len(manifest),
                     "platform": {"architecture": "amd64", "os": "linux"}}
    index = json.dumps({"manifests": [manifest_desc]}).encode()
    small = {"index": index, "manifest": manifest, "config": config}
    expected = {name: (sha(data), len(data)) for name, data in small.items()}
    expected["layers"] = [(sha(data), len(data)) for data in layer_data]
    expected["diff_ids"] = ["sha256:" + sha(b"expanded0"), "sha256:" + sha(b"expanded1")]
    paths = {}
    for name, data in small.items():
        paths[name] = root / (name + ".json")
        paths[name].write_bytes(data)
    sources = []
    for i, data in enumerate(layer_data):
        path = root / f"layer-{i}.blob"
        path.write_bytes(data)
        sources.append(path)
    st = root.stat()
    pin = {"schema": 1, "small": {name: identity(path, True) for name, path in paths.items()},
           "layers": layers, "sources": [identity(path) for path in sources],
           "destination": {"path": str(root / "export"), "parent": str(root),
                           "parent_mode": st.st_mode, "parent_device": st.st_dev,
                           "parent_inode": st.st_ino,
                           "free_bytes": os.statvfs(root).f_bavail * os.statvfs(root).f_frsize},
           "limits": {"seconds": 10, "max_rss_bytes": 2 * 1024 * 1024 * 1024,
                      "max_disk_bytes": 1024 * 1024,
                      "max_blob_bytes": 1024, "max_combined_bytes": 1024}}
    return pin, expected, sources


def acquire(root, pin, expected):
    path = root / "pin.json"
    raw = json.dumps(pin).encode()
    path.write_bytes(raw)
    with mock.patch.object(mod, "EXPECTED", expected), \
         mock.patch.object(mod, "RELEASED_PIN_SHA256", sha(raw)):
        return mod.run(str(path))


class AcquisitionTests(unittest.TestCase):
    def test_production_release_absent_before_pin_or_layer_access(self):
        self.assertIsNone(mod.RELEASED_PIN_SHA256)
        with self.assertRaisesRegex(mod.Hold, "release-pin-absent"):
            mod.run("/definitely/missing")
        result = subprocess.run([sys.executable, str(SCRIPT), "--pin", "/definitely/missing"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("release-pin-absent", result.stdout)

    def test_exact_export_and_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, sources = fixture(root)
            output = Path(acquire(root, pin, expected))
            self.assertEqual(output, root / "export")
            self.assertEqual((output / "index.json").read_bytes(), (root / "index.json").read_bytes())
            self.assertEqual(json.loads((output / "oci-layout").read_text()), {"imageLayoutVersion": "1.0.0"})
            for source in sources:
                blob = output / "blobs" / "sha256" / sha(source.read_bytes())
                self.assertEqual(blob.read_bytes(), source.read_bytes())
                self.assertTrue(blob.is_file() and not blob.is_symlink())
            self.assertEqual(json.loads((output / "ACQUISITION.json").read_text())["status"], "VERIFIED")
            self.assertFalse(list(root.glob(".e3-n1-HOLD-*")))
            for line in (output / "SHA256SUMS-ACQUISITION").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(sha((output / name).read_bytes()), digest)

    def test_malformed_pin_and_caps_hold_before_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, _ = fixture(root)
            pin["limits"]["max_combined_bytes"] = 1
            with self.assertRaisesRegex(mod.Hold, "descriptor-byte-cap"):
                acquire(root, pin, expected)
            pin["limits"]["max_combined_bytes"] = 1024
            pin["limits"]["max_blob_bytes"] = 1
            with self.assertRaisesRegex(mod.Hold, "descriptor-byte-cap"):
                acquire(root, pin, expected)
            pin["limits"]["max_blob_bytes"] = 1024
            pin["limits"]["max_disk_bytes"] = 0
            with self.assertRaisesRegex(mod.Hold, "disk-limit"):
                acquire(root, pin, expected)
            pin["limits"]["max_disk_bytes"] = 65536
            with self.assertRaisesRegex(mod.Hold, "descriptor-disk-cap"):
                acquire(root, pin, expected)
            pin["limits"]["max_disk_bytes"] = 1024 * 1024
            pin["limits"]["max_rss_bytes"] = 1
            with self.assertRaisesRegex(mod.Hold, "rss-cap"):
                acquire(root, pin, expected)
            pin["limits"]["max_rss_bytes"] = 2 * 1024 * 1024 * 1024
            pin["small"]["index"]["size"] = "wrong"
            with self.assertRaises(mod.Hold):
                acquire(root, pin, expected)
            pin["small"]["index"]["size"] = expected["index"][1]
            pin["destination"]["path"] = str(root / "index.json")
            with self.assertRaisesRegex(mod.Hold, "destination-drift"):
                acquire(root, pin, expected)
            self.assertFalse(list(root.glob(".e3-n1-HOLD-*")))

    def test_symlink_and_identity_swap_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, sources = fixture(root)
            sources[0].rename(root / "original")
            sources[0].symlink_to(root / "original")
            with self.assertRaises(mod.Hold):
                acquire(root, pin, expected)
            self.assertFalse(list(root.glob(".e3-n1-HOLD-*")))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, sources = fixture(root)
            real_copy = mod.copy_layer

            def swap_before_copy(source, descriptor, path, limits, deadline, hashes, used):
                if source["path"] == str(sources[1]):
                    sources[1].rename(root / "old-second")
                    sources[1].write_bytes(b"tiny layer one")
                return real_copy(source, descriptor, path, limits, deadline, hashes, used)

            with mock.patch.object(mod, "copy_layer", side_effect=swap_before_copy):
                with self.assertRaisesRegex(mod.Hold, "source-identity-drift"):
                    acquire(root, pin, expected)
            self.assert_hold_partial(root)

    def test_second_blob_hash_failure_seals_partial_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, sources = fixture(root)
            sources[1].write_bytes(b"tiny layer onE")
            pin["sources"][1] = identity(sources[1])
            with self.assertRaisesRegex(mod.Hold, "layer-byte-drift"):
                acquire(root, pin, expected)
            self.assert_hold_partial(root)

    def test_source_mutation_during_read_and_late_time_cap_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, sources = fixture(root)
            original_read = mod.os.read
            changed = False

            def change_after_read(fd, count):
                nonlocal changed
                data = original_read(fd, count)
                if not changed and data == b"tiny layer one":
                    sources[1].write_bytes(b"tiny layer onE")
                    changed = True
                return data

            with mock.patch.object(mod.os, "read", side_effect=change_after_read):
                with self.assertRaisesRegex(mod.Hold, "source-identity-drift"):
                    acquire(root, pin, expected)
            self.assertTrue(changed)
            self.assert_hold_partial(root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, _ = fixture(root)
            original_sample = mod.sample

            def expire_after_stage(deadline, rss_cap):
                if list(root.glob(".e3-n1-HOLD-*")):
                    raise mod.Hold("time-cap")
                return original_sample(deadline, rss_cap)

            with mock.patch.object(mod, "sample", side_effect=expire_after_stage):
                with self.assertRaisesRegex(mod.Hold, "time-cap"):
                    acquire(root, pin, expected)
            self.assert_hold_partial(root)

    def test_parent_swap_at_actual_rename_retracts_verified_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, _ = fixture(root)
            moved = root.with_name(root.name + "-moved")
            actual_rename = mod.rename_noreplace
            swapped = False

            def swap_at_rename(parent_fd, source_name, target_name):
                nonlocal swapped
                if target_name == "export" and not swapped:
                    root.rename(moved)
                    root.symlink_to(moved, target_is_directory=True)
                    swapped = True
                return actual_rename(parent_fd, source_name, target_name)

            try:
                with mock.patch.object(mod, "rename_noreplace", side_effect=swap_at_rename):
                    with self.assertRaisesRegex(mod.Hold, "destination-parent-drift"):
                        acquire(root, pin, expected)
                self.assertTrue(swapped)
                self.assertFalse((moved / "export").exists())
                self.assert_hold_partial(moved)
            finally:
                if root.is_symlink():
                    root.unlink()
                    moved.rename(root)

    def test_parent_drift_with_retraction_error_marks_anchored_target_hold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin, expected, _ = fixture(root)
            moved = root.with_name(root.name + "-moved")
            actual_rename = mod.rename_noreplace
            calls = 0

            def fail_retraction(parent_fd, source_name, target_name):
                nonlocal calls
                calls += 1
                if calls == 1:
                    root.rename(moved)
                    root.symlink_to(moved, target_is_directory=True)
                    return actual_rename(parent_fd, source_name, target_name)
                raise OSError("synthetic rollback failure")

            try:
                with mock.patch.object(mod, "rename_noreplace", side_effect=fail_retraction):
                    with self.assertRaisesRegex(mod.Hold, "destination-parent-drift"):
                        acquire(root, pin, expected)
                self.assertEqual(calls, 2)
                self.assertEqual(json.loads((moved / "export" / "HOLD.json").read_text())["status"], "HOLD")
                self.assertFalse((moved / "export" / "ACQUISITION.json").exists())
                for line in (moved / "export" / "SHA256SUMS-ACQUISITION").read_text().splitlines():
                    digest, name = line.split("  ", 1)
                    self.assertEqual(sha((moved / "export" / name).read_bytes()), digest)
            finally:
                if root.is_symlink():
                    root.unlink()
                    moved.rename(root)
    def assert_hold_partial(self, root):
        self.assertFalse((root / "export").exists())
        stages = list(root.glob(".e3-n1-HOLD-*"))
        self.assertEqual(len(stages), 1)
        stage = stages[0]
        self.assertEqual(json.loads((stage / "HOLD.json").read_text())["status"], "HOLD")
        self.assertFalse((stage / "ACQUISITION.json").exists())
        for line in (stage / "SHA256SUMS-ACQUISITION").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(sha((stage / name).read_bytes()), digest)


if __name__ == "__main__":
    unittest.main()
