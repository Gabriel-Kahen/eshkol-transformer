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
from unittest.mock import patch

from tests.e3_private import inactive_extraction_outer as outer
from tests.e3_private import inactive_extraction_guest as guest


J_REPO = Path("/home/gabe/.codex/worktrees/g3c4-f-loader-proc-disappearance-k/eshkol-transformer")
EVIDENCE = Path("/home/gabe/.codex/evidence/eshkol-transformer")
K_SOURCE = EVIDENCE / "g3c4-f-loader-proc-disappearance-source-20261009-k/SHA256SUMS-SOURCE"
K_ROOT = EVIDENCE / "g3c4-f-loader-proc-disappearance-root-source-review-20261009-k/SHA256SUMS-ROOT-SOURCE-REVIEW"
K_IND = EVIDENCE / "g3c4-f-loader-proc-disappearance-independent-review-20261009-k/SHA256SUMS-INDEPENDENT-REVIEW"


class Contract(unittest.TestCase):
    def test_reviewed_recipe_e_source_and_denied_predecessor(self) -> None:
        e = EVIDENCE / "e3-section9-sealed-fixture-inventory-recipe-source-20261009-e"
        self.assertEqual(sha256((e / "SHA256SUMS-RECIPE-SOURCE").read_bytes()).hexdigest(),
                         "aa854ebbd94a171f2081a6ca6b286ad9a3fb64671ce13d1c6fbfd79431bf039a")
        self.assertEqual(sha256((EVIDENCE / "e3-section9-sealed-fixture-inventory-recipe-root-source-review-20261009-e/"
                                  "SHA256SUMS-ROOT-RECIPE-REVIEW").read_bytes()).hexdigest(),
                         "9d9724967f68ba3870a9ec98050a42b4c8a569c4e3c34d58c03005bf2530ca3a")
        self.assertEqual(sha256((EVIDENCE / "e3-section9-sealed-fixture-inventory-recipe-independent-review-20261009-e/"
                                  "SHA256SUMS-INDEPENDENT-REVIEW").read_bytes()).hexdigest(),
                         "c8ca178e457a6f55107db03218010ba7cdfd3722e98d72d2918c525e79d915d9")
        self.assertEqual(sha256((e / "source/tests/e3_private/inactive_extraction_recipe.py").read_bytes()).hexdigest(),
                         outer.RECIPE_SHA)
        self.assertEqual(outer.recipe.F_ATTEST_SHA,
                         "2c57e29015a76e2ff733862bb3ef05fcf7a5c0dba284dfad8913b3c10d7e3433")
        self.assertNotEqual(outer.RECIPE_SHA,
                            "215346061c87c6cf2c3475fdd7f2cea4f88e3f30239bc1b8a987d079f683761d")

    def test_actual_k_source_and_reviews(self) -> None:
        self.assertEqual(sha256(K_SOURCE.read_bytes()).hexdigest(), outer.J_SOURCE)
        self.assertEqual(sha256(K_ROOT.read_bytes()).hexdigest(), outer.J_ROOT)
        self.assertEqual(sha256(K_IND.read_bytes()).hexdigest(), outer.J_IND)
        actual_head, actual_tree = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(J_REPO), "rev-parse",
             "HEAD", "HEAD^{tree}"], check=True, capture_output=True, text=True).stdout.splitlines()
        self.assertEqual((actual_head, actual_tree), (outer.J_HEAD, outer.J_TREE))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in outer.J_FILES:
                data = subprocess.run(
                    ["git", "--no-optional-locks", "-C", str(J_REPO), "show",
                     f"{outer.J_HEAD}:{relative}"], check=True,
                    capture_output=True).stdout
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            for name, source in (
                ("SHA256SUMS-SOURCE", K_SOURCE),
                ("SHA256SUMS-ROOT-REVIEW", K_ROOT),
                ("SHA256SUMS-INDEPENDENT-SOURCE-REVIEW", K_IND),
            ):
                (root / name).write_bytes(source.read_bytes())
            outer.h_source(root, Path.read_bytes)
            path = root / "tests/g3c4/f_loader_outer.py"
            original_loader = path.read_bytes()
            path.write_bytes(original_loader + b"# foreign\n")
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.h_source(root, Path.read_bytes)
            path.write_bytes(original_loader)
            collector = root / "tests/g3c4/f_image_loader_collect.py"
            original = collector.read_bytes()
            collector.write_bytes(original + b"# stale collector\n")
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.h_source(root, Path.read_bytes)
            collector.write_bytes(original)
            old_collector = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(J_REPO), "show",
                 "07a2ae53cbdf5620fd10319b8c95bf0261dfeb51:tests/g3c4/f_image_loader_collect.py"],
                check=True, capture_output=True).stdout
            collector.write_bytes(old_collector)
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.h_source(root, Path.read_bytes)
            collector.write_bytes(original)
            (root / "SHA256SUMS-ROOT-REVIEW").write_bytes(K_ROOT.read_bytes() + b"forged\n")
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.h_source(root, Path.read_bytes)
            (root / "SHA256SUMS-ROOT-REVIEW").write_bytes(K_ROOT.read_bytes())
            (root / "SHA256SUMS-INDEPENDENT-SOURCE-REVIEW").write_bytes(K_IND.read_bytes() + b"forged\n")
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.h_source(root, Path.read_bytes)
            (root / "SHA256SUMS-INDEPENDENT-SOURCE-REVIEW").write_bytes(K_IND.read_bytes())
            (root / "SHA256SUMS-SOURCE").write_bytes((
                EVIDENCE / "g3c4-public-g2-f-loader-readelf-source-20261009-j/"
                "SOURCE-EVIDENCE/SHA256SUMS-SOURCE").read_bytes())
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

    def test_actual_flat_attester_precreate_pin(self) -> None:
        source = (EVIDENCE / "g3c4-public-g2-f-attest-lock-rebase-source-20261009-b/"
                  "f_artifact_attest.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "f_artifact_attest.py"
            candidate.write_bytes(source.read_bytes())
            outer.flat_attester(root, Path.read_bytes)
            candidate.write_bytes(candidate.read_bytes() + b"# drift\n")
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.flat_attester(root, Path.read_bytes)
            candidate.write_bytes((J_REPO / "tests/g3c4/f_artifact_attest.py").read_bytes())
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.flat_attester(root, Path.read_bytes)
            candidate.write_bytes((EVIDENCE / "g3c4-public-g2-f-attester-flat-source-20261009-c/"
                                   "f_artifact_attest.py").read_bytes())
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.flat_attester(root, Path.read_bytes)

    def test_j_profile_tail_only_and_stale_cmd_denial(self) -> None:
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
        result = outer.create_argv(H, H.create_argv, decision)
        self.assertEqual(result[:9], H.create_argv(dict(decision, source_admission_sha256="0" * 64))[:9])
        self.assertEqual(result[-4:], outer.GUEST + ["a" * 64])
        with self.assertRaisesRegex(ValueError, "H guest tail changed"):
            outer.create_argv(types.SimpleNamespace(require=H.require),
                              lambda _: ["foreign"], decision)

    def test_actual_binding_calls_original_builder_and_restores_h(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "h-source"
            module_path = source / "tests/g3c4/f_loader_outer.py"
            module_path.parent.mkdir(parents=True)
            module_path.write_bytes(b"source binding fixture")
            decision_path = root / "decision.json"
            decision = {"bundle": str(root), "source": "/selected",
                        "e3_material_pins_sha256": "a" * 64}
            raw = json.dumps(decision, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            decision_path.write_bytes(raw)
            prior_tail = ["-B", "-m", "tests.g3c4.f_loader_guest",
                          "--source-admission-sha256", "0" * 64]
            calls = []

            def original_create(current):
                calls.append(current["source_admission_sha256"])
                return ["docker", "create", "--mount", current["source"]] + prior_tail

            def require(ok, why):
                if not ok:
                    raise ValueError(why)

            h = types.SimpleNamespace(
                __file__=str(module_path), ordinary=Path.read_bytes,
                require=require, create_argv=original_create,
                selected=lambda *_args, **_kwargs: None,
                inspect_owned=lambda *_args: None,
                guest_output=lambda *_args: None)

            def run_once(_path, _pin, _evidence, _monitor):
                actual = h.create_argv(decision)
                self.assertEqual(actual[:4], ["docker", "create", "--mount", "/selected"])
                self.assertEqual(actual[-4:], outer.GUEST + ["a" * 64])

            h.run_once = run_once
            with patch.object(outer, "h_source") as authenticated:
                outer.bind_and_run(h, source, decision_path, sha256(raw).hexdigest(),
                                   root / "never-written", None)
            authenticated.assert_called_once()
            self.assertEqual(calls, ["0" * 64])
            self.assertIs(h.create_argv, original_create)

    def test_inactive_no_authority_and_two_modes(self) -> None:
        plan = outer.contract()
        self.assertEqual(plan["status"], "SOURCE_ONLY_INACTIVE")
        self.assertEqual(plan["h_source_sha256"], outer.J_SOURCE)
        self.assertEqual([row[-1] for row in plan["guest_sequence"]], ["normal", "sanitize"])
        self.assertEqual(len(plan["missing_runtime_authority"]), 2)
        self.assertNotIn("if __name__", Path(outer.__file__).read_text())
        self.assertEqual(outer.J_FILES, guest.J_FILES)
        self.assertEqual(outer.RECIPE_SHA, guest.RECIPE_SHA)
        self.assertEqual(sha256(Path(outer.recipe.__file__).read_bytes()).hexdigest(),
                         outer.RECIPE_SHA)
        previous = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(Path(outer.__file__).parents[2]),
             "show", "652e0c64b67f74f812ca275725a6b7419cbcd0fb:"
             "tests/e3_private/inactive_extraction_recipe.py"],
            check=True, capture_output=True).stdout
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "e3_recipe.py"
            stale.write_bytes(previous)
            with self.assertRaisesRegex(ValueError, "selected bytes changed"):
                outer.exact_bytes(stale, outer.RECIPE_SHA, Path.read_bytes)

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
               "Config": {"Labels": {"org.opencontainers.image.version": "22.04",
                                     "eshkol.gate": decision["name"]}, "User": "0:0",
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
        image = json.dumps([{"Id": H.IMAGE,
                             "Config": {"Labels": {"org.opencontainers.image.version": "22.04"}}}]).encode()
        self.assertEqual(outer.inspect_owned(H, raw, decision, "c" * 64, False, image), row)
        for change in (lambda r: r["Config"].__setitem__("Cmd", ["-c", "foreign"]),
                       lambda r: r["HostConfig"].__setitem__("ReadonlyRootfs", False),
                       lambda r: r["Mounts"].append({"Type": "bind", "Source": "/foreign",
                                                      "Destination": "/foreign", "RW": False,
                                                      "Mode": "", "Propagation": "rprivate"}),
                       lambda r: r["State"].__setitem__("Running", True),
                       lambda r: r["Config"]["Labels"].update(foreign="x")):
            foreign = deepcopy(row)
            change(foreign)
            with self.assertRaises(ValueError):
                outer.inspect_owned(H, json.dumps([foreign]).encode(), decision,
                                    "c" * 64, False, image)
        changed_image = json.dumps([{"Id": H.IMAGE,
                                     "Config": {"Labels": {"org.opencontainers.image.version": "22.05"}}}]).encode()
        with self.assertRaisesRegex(ValueError, "immutable f31 image labels"):
            outer.inspect_owned(H, raw, decision, "c" * 64, False, changed_image)


if __name__ == "__main__":
    unittest.main()
