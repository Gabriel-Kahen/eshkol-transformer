"""Synthetic-only negatives for the inert E3 N1 capture preflight."""

import importlib.util
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
