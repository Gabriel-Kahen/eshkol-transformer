"""Synthetic-only negatives for the inert E3 N1 capture preflight."""

import importlib.util
import gzip
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/e3-n1-raw-capture.py"
spec = importlib.util.spec_from_file_location("e3_n1_raw_capture", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PreflightTests(unittest.TestCase):
    def test_production_has_no_reviewed_pin(self):
        self.assertIsNone(mod.REVIEWED_PLAN_SHA256)
        with self.assertRaisesRegex(mod.Hold, "reviewed-plan-pin-absent"):
            mod.preflight("/missing/plan.json")

    def test_cli_emits_bounded_partial_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "fresh"
            run = subprocess.run([sys.executable, str(SCRIPT), "--plan", "/missing",
                                  "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            receipt = json.loads((out / "preflight.json").read_text())
            self.assertEqual(receipt["status"], "HOLD")
            self.assertEqual(receipt["action"], "none")
            self.assertEqual(receipt["reason"], "reviewed-plan-pin-absent")
            self.assertLess((out / "preflight.json").stat().st_size, 512)

    def test_capture_entry_is_disabled_before_any_input_access(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "fresh"
            run = subprocess.run([sys.executable, str(SCRIPT), "--capture", "--plan", "/missing",
                                  "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            receipt = json.loads((out / "preflight.json").read_text())
            self.assertEqual(receipt["reason"], "capture-disabled-by-source")
            self.assertEqual(receipt["action"], "none")

    def test_route_drift_and_missing_route(self):
        with self.assertRaisesRegex(mod.Hold, "mount-route-count"):
            mod.validate_routes([])
        routes = [dict(host=h, guest=g, mode=m) for h, g, m in [
            ("/home/gabe/.codex/venvs/eshkol-oracle-py3146", "/home/gabe/.codex/venvs/eshkol-oracle-py3146", "ro"),
            ("/home/gabe/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu", "/home/gabe/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu", "ro"),
            ("/home/gabe/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu", "/home/gabe/.local/share/uv/python/cpython-3.14.6-linux-x86_64-gnu", "ro"),
            ("/home/gabe/.codex/worktrees/e3-numerical-stable-source/eshkol-transformer", "/workspace", "ro"),
            ("/home/gabe/.codex/worktrees/wave3-runtime-repin/eshkol-transformer/.deps", "/workspace/.deps", "ro"),
            ("/home/gabe/Documents/ChatGPT/eshkol-transformer/.git", "/home/gabe/Documents/ChatGPT/eshkol-transformer/.git", "ro"),
            ("/tmp/e3-fresh", "/out", "rw")]]
        mod.validate_routes(routes)
        routes[3]["guest"] = "/wrong"
        with self.assertRaisesRegex(mod.Hold, "mount-route-drift"):
            mod.validate_routes(routes)

    def test_capture_recipe_requires_pinned_tools_and_all_categories(self):
        plan = {"selected_guest_files": [{"role": "tool", "path": "/usr/bin/readelf"}],
                "limits": {"seconds": 60, "max_output_bytes": 1024}}
        plan["capture_commands"] = [
            {"category": category, "tool": "/usr/bin/readelf",
             "argv": ["/usr/bin/readelf", "-a", "/out/input"], "env": {},
             "seconds": 10, "max_output_bytes": 512}
            for category in sorted(mod.CAPTURE_CATEGORIES)]
        mod.validate_capture_recipe(plan)
        plan["capture_commands"][0]["tool"] = "/usr/bin/unpinned"
        with self.assertRaisesRegex(mod.Hold, "capture-argv-unpinned"):
            mod.validate_capture_recipe(plan)
        plan["capture_commands"][0]["tool"] = "/usr/bin/readelf"
        plan["capture_commands"].pop()
        with self.assertRaisesRegex(mod.Hold, "capture-category-coverage"):
            mod.validate_capture_recipe(plan)

    def test_selected_layer_bytes_and_whiteout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layer = root / "layer.tar"
            with tarfile.open(layer, "w") as out:
                data = b"pinned guest tool"
                info = tarfile.TarInfo("usr/bin/tool")
                info.size = len(data)
                info.mode = 0o755
                out.addfile(info, io.BytesIO(data))
            entry = {"path": "/usr/bin/tool", "layer": 0, "kind": "file", "mode": 0o755,
                     "sha256": mod.hashlib.sha256(data).hexdigest(), "size": len(data), "role": "tool"}
            # The layer scanner compares only the independently pinned file fields.
            selected = [{k: v for k, v in entry.items() if k != "role"}]
            self.assertEqual(mod.scan_selected_layers([layer], selected, 1024, time.monotonic() + 5), 1)
            selected[0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(mod.Hold, "selected-guest-drift"):
                mod.scan_selected_layers([layer], selected, 1024, time.monotonic() + 5)
            selected[0]["sha256"] = entry["sha256"]
            with self.assertRaisesRegex(mod.Hold, "selected-byte-cap"):
                mod.scan_selected_layers([layer], selected, 4, time.monotonic() + 5)
            with self.assertRaisesRegex(mod.Hold, "rss-cap"):
                mod.scan_selected_layers([layer], selected, 1024, time.monotonic() + 5,
                                         rss_cap=1)
            whiteout = root / "whiteout.tar"
            with tarfile.open(whiteout, "w") as out:
                info = tarfile.TarInfo("usr/bin/.wh.tool")
                info.size = 0
                out.addfile(info, io.BytesIO())
            with self.assertRaisesRegex(mod.Hold, "selected-whiteout"):
                mod.scan_selected_layers([layer, whiteout], selected, 1024, time.monotonic() + 5)
            symlink = root / "symlink.tar"
            with tarfile.open(symlink, "w") as out:
                info = tarfile.TarInfo("usr/bin/tool")
                info.type = tarfile.SYMTYPE
                info.linkname = "unselected"
                info.mode = 0o777
                out.addfile(info)
            selected_link = [{"path": "/usr/bin/tool", "layer": 0,
                              "kind": "symlink", "target": "unselected", "mode": 0o777}]
            with self.assertRaisesRegex(mod.Hold, "symlink-target-unpinned"):
                mod.scan_selected_layers([symlink], selected_link, 1024, time.monotonic() + 5)

    def test_oci_image_identity_is_index_not_config(self):
        with tempfile.TemporaryDirectory() as temp:
            layout = Path(temp)
            (layout / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
            blobs = layout / "blobs" / "sha256"
            blobs.mkdir(parents=True)

            def store(data, media_type):
                digest = mod.hashlib.sha256(data).hexdigest()
                (blobs / digest).write_bytes(data)
                return {"mediaType": media_type, "digest": "sha256:" + digest,
                        "size": len(data)}

            data = b"selected tool"
            layer = layout / "layer.tar"
            with tarfile.open(layer, "w") as out:
                info = tarfile.TarInfo("usr/bin/tool")
                info.mode = 0o755
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
            layer_desc = store(layer.read_bytes(), "application/vnd.oci.image.layer.v1.tar")
            config_desc = store(json.dumps({"architecture": "amd64", "os": "linux",
                                           "rootfs": {"diff_ids": [layer_desc["digest"]]}}
                                          ).encode(), "application/vnd.oci.image.config.v1+json")
            manifest_desc = store(json.dumps({"config": config_desc, "layers": [layer_desc]}
                                             ).encode(), "application/vnd.oci.image.manifest.v1+json")
            manifest_desc["platform"] = {"architecture": "amd64", "os": "linux"}
            index = json.dumps({"manifests": [manifest_desc]}).encode()
            (layout / "index.json").write_bytes(index)
            index_hash = mod.hashlib.sha256(index).hexdigest()
            selected = {"path": "/usr/bin/tool", "layer": 0, "kind": "file",
                        "mode": 0o755, "sha256": mod.hashlib.sha256(data).hexdigest(),
                        "size": len(data), "role": "tool"}
            plan = {"oci_layout": str(layout), "index_sha256": index_hash,
                    "manifest": manifest_desc, "config": config_desc,
                    "layers": [layer_desc], "diff_ids": [layer_desc["digest"]],
                    "selected_guest_files": [selected],
                    "limits": {"max_blob_bytes": 1024 * 1024,
                               "max_selected_bytes": 1024, "max_rss_bytes": 2 * 1024**3,
                               "max_uncompressed_layer_bytes": 1024 * 1024,
                               "max_total_uncompressed_bytes": 1024 * 1024}}
            self.assertNotEqual("sha256:" + index_hash, config_desc["digest"])
            self.assertEqual(mod.validate_oci(plan, time.monotonic() + 5,
                                              expected_image="sha256:" + index_hash), 1)
            with self.assertRaisesRegex(mod.Hold, "oci-index-drift"):
                mod.validate_oci(plan, time.monotonic() + 5,
                                 expected_image=config_desc["digest"])
            wrong_platform = dict(plan)
            wrong_platform["manifest"] = dict(manifest_desc,
                                              platform={"architecture": "unknown", "os": "unknown"})
            with self.assertRaisesRegex(mod.Hold, "oci-amd64-manifest-drift"):
                mod.validate_oci(wrong_platform, time.monotonic() + 5,
                                 expected_image="sha256:" + index_hash)

            gzip_desc = store(gzip.compress(layer.read_bytes(), mtime=0),
                              "application/vnd.oci.image.layer.v1.tar+gzip")
            gzip_manifest = store(json.dumps({"config": config_desc, "layers": [gzip_desc]}
                                             ).encode(), "application/vnd.oci.image.manifest.v1+json")
            gzip_manifest["platform"] = {"architecture": "amd64", "os": "linux"}
            gzip_index = json.dumps({"manifests": [gzip_manifest]}).encode()
            (layout / "index.json").write_bytes(gzip_index)
            gzip_index_hash = mod.hashlib.sha256(gzip_index).hexdigest()
            plan.update(index_sha256=gzip_index_hash, manifest=gzip_manifest,
                        layers=[gzip_desc])
            self.assertEqual(mod.validate_oci(plan, time.monotonic() + 5,
                                              expected_image="sha256:" + gzip_index_hash), 1)

    def test_gzip_layer_integrity_and_resource_negatives(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "layer.tar.gz"
            tar = io.BytesIO()
            with tarfile.open(fileobj=tar, mode="w") as archive:
                payload = b"pinned guest tool"
                entry = tarfile.TarInfo("usr/bin/tool")
                entry.mode = 0o755
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
            raw = tar.getvalue()
            packed = gzip.compress(raw, mtime=0)
            selected = [{"path": "/usr/bin/tool", "layer": 0, "kind": "file",
                         "mode": 0o755, "sha256": mod.hashlib.sha256(payload).hexdigest(),
                         "size": len(payload)}]

            def check(data, *, expanded_cap=1024 * 1024, diff_id=None,
                      compressed_digest=None, declared_size=None, deadline=None,
                      chosen=None):
                path.write_bytes(data)
                descriptor = {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                              "digest": "sha256:" + (compressed_digest or mod.hashlib.sha256(data).hexdigest()),
                              "size": len(data) if declared_size is None else declared_size}
                layer = {"descriptor": descriptor,
                         "diff_id": diff_id or "sha256:" + mod.hashlib.sha256(raw).hexdigest()}
                return mod.scan_selected_layers([path], chosen or selected, 1024,
                                                deadline or time.monotonic() + 5,
                                                layers=[layer], uncompressed_cap=expanded_cap,
                                                total_uncompressed_cap=expanded_cap)

            self.assertEqual(check(packed), 1)
            with self.assertRaisesRegex(mod.Hold, "gzip-compressed-drift"):
                check(packed, compressed_digest="0" * 64)
            with self.assertRaisesRegex(mod.Hold, "gzip-compressed-size"):
                check(packed, declared_size=len(packed) - 1)
            with self.assertRaisesRegex(mod.Hold, "gzip-diffid-drift"):
                check(packed, diff_id="sha256:" + "0" * 64)
            with self.assertRaisesRegex(mod.Hold, "gzip-short"):
                check(packed[:-8])
            with self.assertRaisesRegex(mod.Hold, "gzip-trailing"):
                check(packed + b"trailing")
            with self.assertRaisesRegex(mod.Hold, "gzip-trailing"):
                check(packed + packed)
            with self.assertRaisesRegex(mod.Hold, "gzip-corrupt"):
                check(packed[:-1] + bytes([packed[-1] ^ 1]))
            with self.assertRaisesRegex(mod.Hold, "gzip-expanded-byte-cap"):
                check(packed, expanded_cap=16)
            layer = {"descriptor": {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "digest": "sha256:" + mod.hashlib.sha256(packed).hexdigest(),
                "size": len(packed)},
                "diff_id": "sha256:" + mod.hashlib.sha256(raw).hexdigest()}
            with self.assertRaisesRegex(mod.Hold, "gzip-expanded-byte-cap"):
                mod.scan_selected_layers([path, path], selected, 1024,
                                         time.monotonic() + 5, layers=[layer, layer],
                                         uncompressed_cap=1024 * 1024,
                                         total_uncompressed_cap=len(raw) + 16)
            with self.assertRaisesRegex(mod.Hold, "time-cap"):
                check(packed, deadline=time.monotonic() - 1)
            changed = [dict(selected[0], mode=0o700)]
            with self.assertRaisesRegex(mod.Hold, "selected-guest-drift"):
                check(packed, chosen=changed)

    def test_gzip_whiteout_and_symlink_negatives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            layers = []
            for index, name in enumerate(("usr/bin/tool", "usr/bin/.wh.tool")):
                tar = io.BytesIO()
                with tarfile.open(fileobj=tar, mode="w") as archive:
                    entry = tarfile.TarInfo(name)
                    entry.size = 0
                    archive.addfile(entry, io.BytesIO())
                raw = tar.getvalue()
                packed = gzip.compress(raw, mtime=0)
                path = root / f"layer-{index}.gz"
                path.write_bytes(packed)
                paths.append(path)
                layers.append({"descriptor": {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": "sha256:" + mod.hashlib.sha256(packed).hexdigest(),
                    "size": len(packed)},
                    "diff_id": "sha256:" + mod.hashlib.sha256(raw).hexdigest()})
            selected = [{"path": "/usr/bin/tool", "layer": 0, "kind": "file",
                         "mode": 0o644, "size": 0,
                         "sha256": mod.hashlib.sha256(b"").hexdigest()}]
            with self.assertRaisesRegex(mod.Hold, "selected-whiteout"):
                mod.scan_selected_layers(paths, selected, 1024, time.monotonic() + 5,
                                         layers=layers, uncompressed_cap=1024 * 1024,
                                         total_uncompressed_cap=2 * 1024 * 1024)

            tar = io.BytesIO()
            with tarfile.open(fileobj=tar, mode="w") as archive:
                entry = tarfile.TarInfo("usr/bin/tool")
                entry.type = tarfile.SYMTYPE
                entry.linkname = "missing"
                entry.mode = 0o777
                archive.addfile(entry)
            raw = tar.getvalue()
            packed = gzip.compress(raw, mtime=0)
            paths[0].write_bytes(packed)
            layers[0]["descriptor"].update(digest="sha256:" + mod.hashlib.sha256(packed).hexdigest(),
                                            size=len(packed))
            layers[0]["diff_id"] = "sha256:" + mod.hashlib.sha256(raw).hexdigest()
            selected = [{"path": "/usr/bin/tool", "layer": 0, "kind": "symlink",
                         "mode": 0o777, "target": "missing"}]
            with self.assertRaisesRegex(mod.Hold, "symlink-target-unpinned"):
                mod.scan_selected_layers(paths[:1], selected, 1024, time.monotonic() + 5,
                                         layers=layers[:1], uncompressed_cap=1024 * 1024,
                                         total_uncompressed_cap=1024 * 1024)

    def test_capture_receipt_requires_39_and_f_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.txt"
            raw.write_bytes(b"synthetic")
            record = {"path": str(raw), "sha256": mod.hashlib.sha256(b"synthetic").hexdigest(),
                      "size": 9}
            rows = []
            for object_name, count in mod.F_OBJECTS.items():
                for offset in range(count):
                    rows.append({"object": object_name, "section": ".text", "offset": offset,
                                 "producer": "synthetic", "abi": "synthetic", "declaration": "synthetic",
                                 "raw_records": [str(raw)]})
            plan = {"f_objects": [{"object": k, "sha256": str(i) * 64}
                                  for i, k in enumerate(mod.F_OBJECTS, 1)],
                    "limits": {"max_raw_record_bytes": 1024}}
            receipt = {"schema": 1, "status": "complete", "call_sites": rows,
                       "provider_graph": [str(raw)], "loader_observation": [str(raw)],
                       "system_headers": [str(raw)], "argv_env": [str(raw)],
                       "declarations": [str(raw)],
                       "f_object_parity": {x["object"]: x["sha256"] for x in plan["f_objects"]},
                       "raw_records": [record], "reviewed_semantics": False}
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            self.assertEqual(mod.validate_capture_receipt(receipt_path, plan, time.monotonic() + 5), 1)
            receipt["call_sites"] = rows[:-1]
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(mod.Hold, "call-count"):
                mod.validate_capture_receipt(receipt_path, plan, time.monotonic() + 5)
            receipt["call_sites"] = rows
            receipt["f_object_parity"]["private.o"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(mod.Hold, "f-object-parity"):
                mod.validate_capture_receipt(receipt_path, plan, time.monotonic() + 5)
            receipt["f_object_parity"]["private.o"] = plan["f_objects"][0]["sha256"]
            receipt["provider_graph"] = ["/nonexistent/provider.raw"]
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(mod.Hold, "call-raw-path-missing"):
                mod.validate_capture_receipt(receipt_path, plan, time.monotonic() + 5)


if __name__ == "__main__":
    unittest.main()
