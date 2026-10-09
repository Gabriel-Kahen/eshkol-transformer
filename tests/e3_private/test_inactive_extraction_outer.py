"""Pure controls for the inactive E3 adapter; never invoke run_once."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from copy import deepcopy

from tests.e3_private import inactive_extraction_outer as outer
from tests.e3_private import inactive_extraction_guest as guest


H_REPO = Path("/home/gabe/.codex/worktrees/g3c4-public-g2-f-loader-launcher/eshkol-transformer")
EVIDENCE = Path("/home/gabe/.codex/evidence/eshkol-transformer")


class Contract(unittest.TestCase):
    def test_actual_h_source_and_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in outer.H_FILES:
                data = subprocess.run(
                    ["git", "--no-optional-locks", "-C", str(H_REPO), "show",
                     f"{outer.H_HEAD}:{relative}"], check=True,
                    capture_output=True).stdout
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            for name, source in (
                ("SHA256SUMS-SOURCE", EVIDENCE / "g3c4-public-g2-f-loader-monitor-source-20261009-h/SHA256SUMS-SOURCE"),
                ("SHA256SUMS-ROOT-REVIEW", EVIDENCE / "g3c4-public-g2-f-loader-monitor-root-source-review-20261009-h/SHA256SUMS-ROOT-REVIEW"),
                ("SHA256SUMS-INDEPENDENT-SOURCE-REVIEW", EVIDENCE / "g3c4-public-g2-f-loader-monitor-independent-source-review-20261009-h/SHA256SUMS-INDEPENDENT-SOURCE-REVIEW"),
            ):
                (root / name).write_bytes(source.read_bytes())
            outer.h_source(root, Path.read_bytes)
            path = root / "tests/g3c4/f_loader_outer.py"
            path.write_bytes(path.read_bytes() + b"# foreign\n")
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.h_source(root, Path.read_bytes)

    def test_material_exact_complete_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "e3-material").mkdir()
            pins = {}
            for name in outer.MATERIAL:
                body = name.encode()
                (root / "e3-material" / name).write_bytes(body)
                pins[name] = sha256(body).hexdigest()
            raw = json.dumps(pins, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            (root / "e3-material-pins.json").write_bytes(raw)
            bodies, actual = outer.material(root, sha256(raw).hexdigest(), Path.read_bytes)
            self.assertEqual(actual, pins)
            self.assertEqual(set(bodies), outer.MATERIAL)
            (root / "e3-material" / "loader_attestation").write_bytes(b"foreign")
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.material(root, sha256(raw).hexdigest(), Path.read_bytes)
            missing = dict(pins)
            missing.pop("loader_attestation")
            bad = json.dumps(missing, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            (root / "e3-material-pins.json").write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "complete canonical"):
                outer.material(root, sha256(bad).hexdigest(), Path.read_bytes)

    def test_h_profile_tail_only_and_stale_cmd_denial(self) -> None:
        class H:
            @staticmethod
            def require(ok, why):
                if not ok:
                    raise ValueError(why)

            @staticmethod
            def create_argv(decision):
                return ["/home/gabe/bin/docker", "create", "--read-only", "--network", "none",
                        "--mount", f"type=bind,src={decision['source']},dst=/source,readonly",
                        "--entrypoint", "/usr/bin/python3.10", outer.recipe.E3_COMMIT,
                        "-B", "-m", "tests.g3c4.f_loader_guest",
                        "--source-admission-sha256", decision["source_admission_sha256"]]

        decision = {"source": "/selected", "e3_material_pins_sha256": "a" * 64}
        result = outer.create_argv(H, decision)
        self.assertEqual(result[:9], H.create_argv(dict(decision, source_admission_sha256="0" * 64))[:9])
        self.assertEqual(result[-4:], outer.GUEST + ["a" * 64])
        with self.assertRaisesRegex(ValueError, "H guest tail changed"):
            outer.create_argv(types.SimpleNamespace(create_argv=lambda _: ["foreign"],
                                                   require=H.require), decision)

    def test_inactive_no_authority_and_two_modes(self) -> None:
        plan = outer.contract()
        self.assertEqual(plan["status"], "SOURCE_ONLY_INACTIVE")
        self.assertEqual(plan["h_source_sha256"], outer.H_SOURCE)
        self.assertEqual([row[-1] for row in plan["guest_sequence"]], ["normal", "sanitize"])
        self.assertEqual(len(plan["missing_runtime_authority"]), 2)
        self.assertNotIn("if __name__", Path(outer.__file__).read_text())
        self.assertEqual(outer.H_FILES, guest.H_FILES)
        self.assertEqual(outer.RECIPE_SHA, guest.RECIPE_SHA)
        self.assertEqual(sha256(Path(outer.recipe.__file__).read_bytes()).hexdigest(),
                         outer.RECIPE_SHA)

    def test_guest_input_nofollow_and_phase_receipt_denial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "build-off").mkdir()
            (root / "phases").mkdir()
            ordinary = root / "ordinary"
            ordinary.write_bytes(b"retained")
            self.assertEqual(guest.ordinary(ordinary), b"retained")
            link = root / "link"
            link.symlink_to(ordinary)
            with self.assertRaises(OSError):
                guest.ordinary(link)
            for mode in ("normal", "sanitize"):
                phase = root / "phases" / mode
                phase.mkdir(parents=True)
                extraction = root / f"emissions-{mode}" / "e3_sealed_fixture_package.o.extract"
                extraction.mkdir(parents=True)
                (extraction / "STATUS").write_bytes(
                    b"EXTRACTION_ONLY: emitted bytes require separate review; no package admitted\n")
                (phase / "stderr.raw").write_bytes(b"")
                (phase / "status.json").write_text(json.dumps({
                    "returncode": 0, "quiescent": True, "failure": "",
                    "argv": list(outer.recipe.guest_command(mode))}))
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "extraction-guest.json").write_text(json.dumps({
                "format": "e3-sealed-fixture-extraction-guest-v1",
                "modes": ["normal", "sanitize"],
                "material_pins_sha256": "a" * 64,
                "status": "EXTRACTION_ONLY"}))

            class H:
                ordinary = staticmethod(Path.read_bytes)
                directory = staticmethod(lambda path: self.assertTrue(path.is_dir()))
                attest = types.SimpleNamespace(copied_artifacts=lambda *_: None)

                @staticmethod
                def require(ok, why):
                    if not ok:
                        raise ValueError(why)

            decision = {"out": str(root), "e3_material_pins_sha256": "a" * 64}
            outer.guest_output(H, decision)
            (root / "phases/sanitize/status.json").write_text(json.dumps({
                "returncode": 1, "quiescent": True, "failure": "",
                "argv": list(outer.recipe.guest_command("sanitize"))}))
            with self.assertRaisesRegex(ValueError, "sanitize receipt failed"):
                outer.guest_output(H, decision)
            (root / "phases/sanitize/status.json").write_text(json.dumps({
                "returncode": 0, "quiescent": True, "failure": "",
                "argv": list(outer.recipe.guest_command("sanitize"))}))
            (root / "emissions-normal/e3_sealed_fixture_package.o").symlink_to("absent")
            with self.assertRaisesRegex(ValueError, "not extraction-only"):
                outer.guest_output(H, decision)

    def test_exact_e3_inspect_fields_and_foreign_negatives(self) -> None:
        roles = (("compiler", "/workspace", True), ("source", "/source", True),
                 ("f_output", "/f-output", True), ("bundle", "/bundle", True),
                 ("candidate", "/stage", False), ("out", "/out", False))
        decision = {key: "/" + key for key, _, _ in roles}
        decision.update(name="e3-sealed-fixture-extraction-test",
                        e3_material_pins_sha256="a" * 64)
        mounts = [{"Type": "bind", "Source": decision[key], "Destination": target,
                   "RW": not readonly, "Mode": "", "Propagation": "rprivate"}
                  for key, target, readonly in roles]
        host_mounts = [{"Type": "bind", "Source": decision[key], "Target": target,
                        **({"ReadOnly": True} if readonly else {})}
                       for key, target, readonly in roles]

        class H:
            IMAGE = "sha256:" + "b" * 64
            FSIZE_LIMIT = 128 * 1024 * 1024

            @staticmethod
            def require(ok, why):
                if not ok:
                    raise ValueError(why)

            @staticmethod
            def env_map(rows):
                return dict(row.split("=", 1) for row in rows)

            @staticmethod
            def expected_env(_):
                return {"PATH": "/usr/bin:/bin"}

        row = {"Id": "c" * 64, "Name": "/" + decision["name"], "Image": H.IMAGE,
               "Config": {"Labels": {"eshkol.gate": decision["name"]}, "User": "0:0",
                          "WorkingDir": "/source", "Entrypoint": ["/usr/bin/python3.10"],
                          "Cmd": outer.GUEST + ["a" * 64], "Env": ["PATH=/usr/bin:/bin"]},
               "State": {"Running": False}, "Mounts": mounts,
               "HostConfig": {"ReadonlyRootfs": True, "NetworkMode": "none",
                              "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=268435456,mode=1777"},
                              "Memory": 4 * 1024**3, "MemorySwap": 4 * 1024**3,
                              "PidsLimit": 128, "NanoCpus": 2_000_000_000,
                              "Ulimits": [{"Name": "fsize", "Hard": H.FSIZE_LIMIT,
                                           "Soft": H.FSIZE_LIMIT}],
                              "Mounts": host_mounts, "Binds": None}}
        raw = json.dumps([row]).encode()
        self.assertEqual(outer.inspect_owned(H, raw, decision, "c" * 64, False, b"[]"), row)
        for change in (lambda r: r["Config"].__setitem__("Cmd", ["-c", "foreign"]),
                       lambda r: r["HostConfig"].__setitem__("ReadonlyRootfs", False),
                       lambda r: r["Mounts"].append({"Type": "bind", "Source": "/foreign",
                                                      "Destination": "/foreign", "RW": False,
                                                      "Mode": "", "Propagation": "rprivate"}),
                       lambda r: r["State"].__setitem__("Running", True)):
            foreign = deepcopy(row)
            change(foreign)
            with self.assertRaises(ValueError):
                outer.inspect_owned(H, json.dumps([foreign]).encode(), decision,
                                    "c" * 64, False, b"[]")


if __name__ == "__main__":
    unittest.main()
