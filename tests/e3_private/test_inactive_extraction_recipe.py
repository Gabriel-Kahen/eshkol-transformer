"""Pure source-backed checks; never compile or launch a container."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from tests.e3_private import inactive_extraction_recipe as recipe


E3 = Path("/home/gabe/.codex/worktrees/e3-section9-sealed-fixture-inventory-extraction-c/eshkol-transformer")
F_ATTEST = Path("/home/gabe/.codex/worktrees/g3c4-public-g2-f-loader-launcher/eshkol-transformer/tests/g3c4/f_artifact_attest.py")
E3_MANIFEST = Path("/home/gabe/.codex/evidence/eshkol-transformer/e3-section9-sealed-fixture-inventory-extraction-source-20261009-c/SHA256SUMS-EXTRACTION-SOURCE")


class SourceRecipe(unittest.TestCase):
    def test_actual_accepted_source_and_copy_contract(self) -> None:
        self.assertEqual(sha256(E3_MANIFEST.read_bytes()).hexdigest(), recipe.E3_SOURCE_MANIFEST)
        self.assertEqual(sha256(F_ATTEST.read_bytes()).hexdigest(), recipe.F_ATTEST_SHA)
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
            "eshkol_binary_sha256": "379143cdb5385b3cc5f64473628adcca6f32e1386496d0f84bb11d397f23aa9",
            "cc_path": "/usr/bin/clang-21", "cxx_path": "/usr/bin/clang++-21",
            **{key: "v" for key in ("llvm_version", "cc_version", "cxx_version",
                                        "cmake_version", "ninja_version", "make_version",
                                        "bash_version", "git_version")},
            "feature_profile": "run-f-off-RelWithDebInfo-testsON-test-only",
            "attestation_origin": "new-test-only-F-copy;not-bootstrap;original-F-failed",
        }
        compat = "".join(f"{key}\t{value}\n" for key, value in entries.items()).encode()
        receipt = {
            "format": "g3c4-F-copy-test-attestation-v1",
            "origin": "new-test-only;historical-F-remains-failed",
            "compiler": {"head": "6803dbe24fb65eaad2901767bf010787517946ef",
                         "tree": "539d22b1b6a89c290aa13c91263982332b01b87b"},
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
