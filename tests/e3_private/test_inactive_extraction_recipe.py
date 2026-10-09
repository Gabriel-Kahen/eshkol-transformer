"""Pure source-backed checks; never compile or launch a container."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.e3_private import inactive_extraction_recipe as recipe


E3 = Path("/home/gabe/.codex/worktrees/e3-section9-sealed-fixture-inventory-extraction-c/eshkol-transformer")
F_ATTEST = Path("/home/gabe/.codex/evidence/eshkol-transformer/g3c4-public-g2-f-attest-lock-rebase-source-20261009-b/f_artifact_attest.py")
F_ORIGINAL_PREPARED = Path("/home/gabe/.codex/evidence/eshkol-transformer/g3c4-public-g2-f-loader-prepared-j-20261009-c/out/evidence/F-copy-prepared.json")
E3_MANIFEST = Path("/home/gabe/.codex/evidence/eshkol-transformer/e3-section9-sealed-fixture-inventory-extraction-source-20261009-c/SHA256SUMS-EXTRACTION-SOURCE")


class SourceRecipe(unittest.TestCase):
    def test_actual_accepted_source_and_copy_contract(self) -> None:
        self.assertEqual(sha256(E3_MANIFEST.read_bytes()).hexdigest(), recipe.E3_SOURCE_MANIFEST)
        self.assertEqual(sha256(F_ATTEST.read_bytes()).hexdigest(), recipe.F_ATTEST_SHA)
        self.assertEqual(sha256(F_ORIGINAL_PREPARED.read_bytes()).hexdigest(),
                         recipe.F_ORIGINAL_PREPARED_SHA)
        actual = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(E3), "rev-parse", "HEAD", "HEAD^{tree}"],
            check=True, capture_output=True, text=True).stdout.splitlines()
        self.assertEqual(actual, [recipe.E3_COMMIT, recipe.E3_TREE])
        original = (E3 / "scripts/common.sh").read_bytes()
        transformed = recipe.common_candidate(original)
        spec = importlib.util.spec_from_file_location("f_attest", F_ATTEST)
        assert spec is not None and spec.loader is not None
        admitted = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(admitted)
        self.assertEqual(transformed, admitted.staged_common(original))
        self.assertEqual(recipe.TRANSFORMS, admitted.COMMON_TRANSFORMS)
        self.assertEqual(recipe.EXTERNAL_LOCK_SHA, admitted.CANDIDATE_LOCK_SHA)
        tracked = (E3 / "toolchain/eshkol.lock").read_bytes()
        self.assertEqual(sha256(tracked.replace(admitted.OLD_ROW,
                                                 admitted.NEW_ROW)).hexdigest(),
                         recipe.EXTERNAL_LOCK_SHA)
        with self.assertRaises(ValueError):
            recipe.common_candidate(original + b"\n")

    def test_exact_mounts_and_sequential_nonpublishing_outputs(self) -> None:
        plan = recipe.contract()
        self.assertEqual(plan["status"], "SOURCE_ONLY_INACTIVE")
        mounts = plan["mounts"]
        self.assertEqual(len(mounts), 6)
        self.assertEqual(len({row[1] for row in mounts}), len(mounts))
        self.assertEqual([row[2] for row in mounts], ["ro", "ro", "ro", "rw", "ro", "rw"])
        self.assertEqual(tuple(cmd[-1] for cmd in plan["sequence"]),
                         ("normal", "sanitize"))
        for mode, cmd in zip(("normal", "sanitize"), plan["sequence"]):
            self.assertEqual(cmd, recipe.guest_command(mode))
            self.assertEqual(cmd[-2], f"/out/emissions-{mode}")
            self.assertEqual(cmd[6], "/stage/candidate/scripts/extract-e3-sealed-fixture-inventories.sh")
            self.assertIn("ESHKOL_BUILD_DIR=/out/build-off", cmd)
            self.assertNotIn("build-e3-sealed-fixture-test.sh", " ".join(cmd))
        with self.assertRaises(ValueError):
            recipe.guest_command("public")

    def test_future_admission_remains_explicitly_missing(self) -> None:
        plan = recipe.contract()
        self.assertEqual(len(plan["missing_runtime_authority"]), 3)
        self.assertIn("F compatibility TSV", plan["missing_runtime_authority"][0])
        self.assertNotIn("if __name__", Path(recipe.__file__).read_text())
        with self.assertRaisesRegex(ValueError, "authority is missing"):
            recipe.prepare({}, {})

    def test_prepare_loads_reviewed_flat_attester_and_corrected_readelf(self) -> None:
        base = Path("/home/gabe/.codex/evidence/eshkol-transformer")
        self.assertEqual(sha256((base / "g3c4-public-g2-f-attest-lock-rebase-source-20261009-b/"
                                     "SHA256SUMS-SOURCE").read_bytes()).hexdigest(),
                         "8fcc584357e9937bbaa673965f3d6e8e6268b1bfdeebdd177b45ebc0a0fa0fef")
        self.assertEqual(sha256((base / "g3c4-public-g2-f-attest-lock-rebase-root-source-review-20261009-b/"
                                     "SHA256SUMS-ROOT-SOURCE-REVIEW").read_bytes()).hexdigest(),
                         "2d90f2d6c178c2bcd67001249671b37207c983d9c2e87909602aa1b2df0389cf")
        self.assertEqual(sha256((base / "g3c4-public-g2-f-attest-lock-rebase-independent-review-20261009-b/"
                                     "SHA256SUMS-INDEPENDENT-REVIEW").read_bytes()).hexdigest(),
                         "43bb2146eaaae518d5d175e191b5242b69ce5d449fddaab0c5b1c1a7e136c116")
        material = {name: name.encode() for name in recipe.MATERIAL_NAMES}
        pins = {name: sha256(raw).hexdigest() for name, raw in material.items()}
        captured = {}

        def selected_path(value):
            return F_ATTEST if value == "/bundle/f_artifact_attest.py" else Path(value)

        def prepare_source(attester, admission, loader, compatibility, readelf):
            captured["attester"] = attester
            self.assertEqual((admission, loader), ({}, {"elf": "selected"}))
            self.assertEqual((compatibility, readelf),
                             (material["loader_compatibility"], material["loader_readelf"]))
            return {"status": "PREPARED_FOR_EXTRACTION_ONLY"}

        with patch.object(recipe, "Path", side_effect=selected_path), \
                patch.object(recipe, "e3_admission", return_value={}), \
                patch.object(recipe, "loader_proof", return_value={"elf": "selected"}), \
                patch.object(recipe, "prepare_source_and_compiler", side_effect=prepare_source):
            self.assertEqual(recipe.prepare(material, pins)["status"],
                             "PREPARED_FOR_EXTRACTION_ONLY")
        attester = captured["attester"]
        binary = Path("/out/build-off/eshkol-run")
        recorded = b"selected dynamic metadata\n"
        with patch.object(attester, "ordinary", return_value=b"selected binary"), \
                patch.object(attester, "version_output", return_value=
                             "Eshkol Compiler v1.3.4-evolve\n"), \
                patch.object(attester, "elf_members", return_value=({"libc.so.6"}, "$ORIGIN")), \
                patch.dict(attester.F_PINS, {"build-off/eshkol-run":
                           sha256(b"selected binary").hexdigest()}), \
                patch.object(attester.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0, stdout=recorded, stderr=b"")) as run:
            self.assertEqual(attester.compiler_binary_and_elf(binary, recorded)
                             ["readelf_sha256"], sha256(recorded).hexdigest())
            self.assertEqual(run.call_args.args[0],
                             ["/usr/bin/readelf", "--debug-dump=no-follow-links",
                              "-d", str(binary)])
        with tempfile.TemporaryDirectory() as temporary:
            stale = Path(temporary) / "f_artifact_attest.py"
            stale.write_bytes((base / "g3c4-public-g2-f-loader-readelf-source-20261009-j/"
                                    "checkout/tests/g3c4/f_artifact_attest.py").read_bytes())
            with patch.object(recipe, "Path", side_effect=lambda value:
                              stale if value == "/bundle/f_artifact_attest.py" else Path(value)):
                with self.assertRaisesRegex(ValueError, "reviewed F attester bytes"):
                    recipe.accepted_attester(stale)
            stale.write_bytes((base / "g3c4-public-g2-f-attester-flat-source-20261009-c/"
                                    "f_artifact_attest.py").read_bytes())
            with patch.object(recipe, "Path", side_effect=lambda value:
                              stale if value == "/bundle/f_artifact_attest.py" else Path(value)):
                with self.assertRaisesRegex(ValueError, "reviewed F attester bytes"):
                    recipe.accepted_attester(stale)

    def test_e3_schema_and_exact_review_denials(self) -> None:
        data = {
            "format": "e3-sealed-fixture-source-admission-v1",
            "head": recipe.E3_COMMIT, "tree": recipe.E3_TREE,
            "tracked_file_count": recipe.E3_TRACKED_COUNT,
            "tracked_manifest_sha256": recipe.E3_TRACKED_SHA,
            "source_manifest_sha256": recipe.E3_SOURCE_MANIFEST,
            "root_source_review_sha256": recipe.E3_ROOT_REVIEW,
            "independent_source_review_sha256": recipe.E3_IND_REVIEW,
        }
        raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        base = Path("/home/gabe/.codex/evidence/eshkol-transformer")
        root = (base / "e3-section9-sealed-fixture-inventory-extraction-root-source-review-20261009-c/SHA256SUMS-ROOT-REVIEW").read_bytes()
        independent = (base / "e3-section9-sealed-fixture-inventory-extraction-independent-source-review-20261009-c/SHA256SUMS-INDEPENDENT-SOURCE-REVIEW").read_bytes()
        self.assertEqual(recipe.e3_admission(raw, sha256(raw).hexdigest(), root,
                                             independent)["head"], recipe.E3_COMMIT)
        for field, value in (("head", "0" * 40), ("format", "g3c4-public-g2-source-admission-v1"),
                             ("tracked_file_count", 0)):
            bad = dict(data, **{field: value})
            altered = json.dumps(bad, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            with self.assertRaises(ValueError):
                recipe.e3_admission(altered, sha256(altered).hexdigest(), root, independent)
        with self.assertRaises(ValueError):
            recipe.e3_admission(raw, sha256(raw).hexdigest(), root + b"\n", independent)

    def test_loader_fields_and_missing_pin_denials(self) -> None:
        entries = {
            "format_version": "1", "eshkol_repository": "https://github.com/Gabriel-Kahen/eshkol.git",
            "eshkol_commit": "6803dbe24fb65eaad2901767bf010787517946ef",
            "eshkol_source_dir": "/workspace",
            "eshkol_binary_sha256": "379143cdb5385b3cc5f64473628adcca6f32e138649e6d0f84bb11d397f23aa9",
            "cc_path": "/usr/bin/clang-21", "cxx_path": "/usr/bin/clang++-21",
            **{key: "v" for key in ("llvm_version", "cc_version", "cxx_version",
                                        "cmake_version", "ninja_version", "make_version",
                                        "bash_version", "git_version")},
            "feature_profile": "run-f-off-RelWithDebInfo-testsON-test-only",
            "attestation_origin": "new-test-only-F-copy;not-bootstrap;original-F-failed",
        }
        compat = "".join(f"{key}\t{value}\n" for key, value in entries.items()).encode()
        receipt = {
            "format": "g3c4-F-copy-test-attestation-v2",
            "origin": "new-test-only;historical-F-remains-failed",
            "compiler": {"head": "6803dbe24fb65eaad2901767bf010787517946ef",
                         "tree": "539d22b1b6a89c290aa13c91263982332b01b87b"},
            "project": {}, "original": {}, "copied": {}, "lock": {},
            "offline_review_sha256": "a" * 64,
            "prepared_sha256": "b" * 64,
            "original_prepared_sha256": recipe.F_ORIGINAL_PREPARED_SHA,
            "image_libraries": [], "image_pins_sha256": "c" * 64,
            "loader_stdout_sha256": "d" * 64,
            "compatibility_tsv_sha256": sha256(compat).hexdigest(), "elf": {"raw": "test"},
            "tool_versions": {key: "v" for key in ("llvm_version", "cc_version",
                                                    "cxx_version", "cmake_version",
                                                    "ninja_version", "make_version",
                                                    "bash_version", "git_version")},
        }
        raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        readelf = b"observed only\n"
        params = (raw, compat, readelf, sha256(raw).hexdigest(),
                  sha256(compat).hexdigest(), sha256(readelf).hexdigest())
        self.assertEqual(recipe.loader_proof(*params)["elf"], receipt["elf"])
        with self.assertRaises(ValueError):
            recipe.loader_proof(*params[:-1], "0" * 64)
        altered = dict(receipt, origin="historical-F-passed")
        bad = json.dumps(altered, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with self.assertRaises(ValueError):
            recipe.loader_proof(bad, compat, readelf, sha256(bad).hexdigest(),
                                sha256(compat).hexdigest(), sha256(readelf).hexdigest())
        for changed in ({"format": "g3c4-F-copy-test-attestation-v1"},
                        {"original_prepared_sha256": "0" * 64},
                        {"prepared_sha256": "not-a-sha"},
                        {"prepared_sha256": recipe.F_ORIGINAL_PREPARED_SHA}):
            altered = dict(receipt, **changed)
            bad = json.dumps(altered, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            with self.assertRaises(ValueError):
                recipe.loader_proof(bad, compat, readelf, sha256(bad).hexdigest(),
                                    sha256(compat).hexdigest(), sha256(readelf).hexdigest())
        missing = dict(receipt)
        del missing["original_prepared_sha256"]
        bad = json.dumps(missing, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with self.assertRaises(ValueError):
            recipe.loader_proof(bad, compat, readelf, sha256(bad).hexdigest(),
                                sha256(compat).hexdigest(), sha256(readelf).hexdigest())

    def test_actual_accepted_v2_loader_material(self) -> None:
        base = Path("/home/gabe/.codex/evidence/eshkol-transformer")
        result = base / "g3c4-public-g2-f-attest-success-result-20261009-e/outputs"
        staged = base / "g3c4-public-g2-f-attest-staged-20261009-e/out"
        attestation = (result / "evidence/F-copy-attestation.json").read_bytes()
        compatibility = (result / "build-off/eshkol-transformer-provenance.tsv").read_bytes()
        readelf = (staged / "evidence/loader/readelf/stdout.raw").read_bytes()
        self.assertEqual(sha256(attestation).hexdigest(),
                         "05f26d035215ef2ce361a201a171c52ba360058badc579ded2543c87cdd59bca")
        self.assertEqual(sha256(compatibility).hexdigest(),
                         "1b381ce2e2400a3a841138ec39e1ec8e45ee55149ca7f6da3a21da0e93b18ae4")
        self.assertEqual(sha256(readelf).hexdigest(),
                         "08620deb9ee0cec55d50f0b9c2f067a9388c9eff46e7af5410aa2aa6f0dd19de")
        pins = tuple(sha256(raw).hexdigest() for raw in
                     (attestation, compatibility, readelf))
        self.assertEqual(recipe.loader_proof(attestation, compatibility, readelf, *pins)
                         ["compatibility_tsv_sha256"], pins[1])
        stale = compatibility.replace(b"e138649e6d0f", b"e1386496d0f")
        stale_receipt = json.loads(attestation)
        stale_receipt["compatibility_tsv_sha256"] = sha256(stale).hexdigest()
        stale_attestation = (json.dumps(stale_receipt, sort_keys=True,
                                       separators=(",", ":")).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "selected 17-row"):
            recipe.loader_proof(stale_attestation, stale, readelf,
                                sha256(stale_attestation).hexdigest(),
                                sha256(stale).hexdigest(), pins[2])

    def test_preparation_uses_exact_reviewed_utility_order(self) -> None:
        calls = []

        class ObservedUtility:
            def compiler_source(self, path):
                calls.append(("compiler_source", str(path)))

            def original_output(self, path):
                calls.append(("original_output", str(path)))

            def offline_review(self, path):
                calls.append(("offline_review", str(path)))

            def ordinary(self, path):
                calls.append(("ordinary", str(path)))
                return b"tracked-lock"

            def candidate_lock(self, path, tracked):
                calls.append(("candidate_lock", str(path), tracked))

            def git(self, path, *args):
                if args[0] == "rev-parse":
                    return recipe.E3_COMMIT + "\n" + recipe.E3_TREE
                return ""

            def stage_project_copy(self, source, destination, admission, expected):
                calls.append(("stage_project_copy", str(source), str(destination)))
                return {"staged": "exact"}

            def copy_f_artifacts(self, source, destination):
                calls.append(("copy_f_artifacts", str(source), str(destination)))
                return {"eshkol-run": "exact"}

            def exclusive_bytes(self, path, raw):
                calls.append(("exclusive_bytes", str(path), raw))

            def final_copied_artifacts(self, destination, sha, expected):
                calls.append(("final_copied_artifacts", str(destination), sha))
                return {"eshkol-run": "exact"}

            def compiler_binary_and_elf(self, path, raw):
                calls.append(("compiler_binary_and_elf", str(path)))
                return {"elf": "exact"}

            def project_copy(self, source, candidate, admission, expected):
                calls.append(("project_copy", str(source), str(candidate)))
                return {"staged": "exact"}

        with patch.object(Path, "exists", return_value=False):
            prepared = recipe.prepare_source_and_compiler(
                ObservedUtility(), {"head": recipe.E3_COMMIT},
                {"elf": {"elf": "exact"}}, b"compat", b"readelf")
        self.assertEqual(prepared["status"], "PREPARED_FOR_EXTRACTION_ONLY")
        self.assertLess([c[0] for c in calls].index("stage_project_copy"),
                        [c[0] for c in calls].index("copy_f_artifacts"))
        self.assertEqual(next(c for c in calls if c[0] == "copy_f_artifacts"),
                         ("copy_f_artifacts", "/f-output", "/out/build-off"))
        self.assertEqual(next(c for c in calls if c[0] == "stage_project_copy"),
                         ("stage_project_copy", "/source", "/stage/candidate"))
        self.assertEqual(next(c for c in calls if c[0] == "exclusive_bytes")[1],
                         "/out/build-off/eshkol-transformer-provenance.tsv")
        calls.clear()
        with patch.object(Path, "exists", return_value=True), self.assertRaises(ValueError):
            recipe.prepare_source_and_compiler(ObservedUtility(), {}, {}, b"", b"")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
