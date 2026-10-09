"""Owned source-only fixtures for the explicit C4 external-6803 lock route."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tests.g3c4 import c4_toolchain_profile as profile


def commit(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm",
                    "owned fixture"], check=True)


@contextmanager
def isolated_profile():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "c4"
        source = base / "compiler"
        build = base / "build"
        for directory in (root / "scripts", root / "toolchain", source, build):
            directory.mkdir(parents=True)
        common = root / "scripts/common.sh"
        common.write_bytes((profile.ROOT / "scripts/common.sh").read_bytes())
        tracked = root / "toolchain/eshkol.lock"
        tracked.write_bytes((profile.ROOT / "toolchain/eshkol.lock").read_bytes())
        (source / "README").write_text("owned compiler fixture\n")
        commit(root)
        commit(source)
        compiler_commit = profile.git(source, "rev-parse", "HEAD")
        compiler_tree = profile.git(source, "rev-parse", "HEAD^{tree}")
        provenance = build / "eshkol-transformer-provenance.tsv"
        provenance.write_text("eshkol_commit\t" + compiler_commit + "\n" +
                              "eshkol_source_dir\t" + str(source.resolve()) + "\n")
        (build / "CMakeCache.txt").write_text(
            "CMAKE_HOME_DIRECTORY:INTERNAL=" + str(source.resolve()) + "\n")
        binary = build / "eshkol-run"
        binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        binary.chmod(0o700)
        candidate = base / "candidate.lock"
        candidate.write_bytes(tracked.read_bytes().replace(profile.OLD_ROW,
                                                           profile.NEW_ROW))
        with mock.patch.object(profile, "ROOT", root), \
             mock.patch.object(profile, "COMPILER_COMMIT", compiler_commit), \
             mock.patch.object(profile, "COMPILER_TREE", compiler_tree), \
             mock.patch.dict(os.environ, {"ESHKOL_ALLOW_UNSUPPORTED_HOST": "0"}):
            yield root, source, build, candidate


class C4ToolchainProfileTests(unittest.TestCase):
    def test_exact_external_candidate_and_repeat_observation(self) -> None:
        with isolated_profile() as (root, source, build, candidate):
            first = profile.observe(candidate, source, build)
            self.assertEqual(first, profile.observe(candidate, source, build))
            self.assertEqual(first["candidate_lock"]["sha256"],
                             profile.CANDIDATE_LOCK_SHA)
            self.assertEqual(first["candidate_lock"]["file_identity"]["nlink"], 1)
            self.assertEqual(first["c4_commit"], profile.git(root, "rev-parse", "HEAD"))
            self.assertEqual(first["compiler_commit"], profile.COMPILER_COMMIT)

    def test_bad_candidate_and_inside_checkout_are_rejected(self) -> None:
        for damage in ("relative", "inside-checkout", "symlink", "parent-symlink",
                       "hardlink", "extra", "wrong-commit", "missing"):
            with self.subTest(damage=damage), isolated_profile() as (
                    root, source, build, candidate):
                if damage == "relative":
                    candidate = Path("candidate.lock")
                elif damage == "inside-checkout":
                    candidate = root / ".git/candidate.lock"
                    candidate.write_bytes((root / "toolchain/eshkol.lock").read_bytes().replace(
                        profile.OLD_ROW, profile.NEW_ROW))
                elif damage == "symlink":
                    link = candidate.with_name("candidate-link.lock")
                    link.symlink_to(candidate)
                    candidate = link
                elif damage == "parent-symlink":
                    link = candidate.parent / "linked-parent"
                    link.symlink_to(candidate.parent, target_is_directory=True)
                    candidate = link / candidate.name
                elif damage == "hardlink":
                    os.link(candidate, candidate.with_name("other-link.lock"))
                elif damage == "extra":
                    candidate.write_bytes(candidate.read_bytes() + b"extra\n")
                elif damage == "wrong-commit":
                    candidate.write_bytes(candidate.read_bytes().replace(
                        profile.NEW_ROW, profile.OLD_ROW))
                else:
                    candidate.unlink()
                with self.assertRaises((ValueError, FileNotFoundError, OSError)):
                    profile.observe(candidate, source, build)

    def test_dirty_source_and_wrong_selected_compiler_are_rejected(self) -> None:
        for damage in ("dirty-common", "indexed-common", "untracked", "tracked-lock",
                       "wrong-head", "wrong-tree", "dirty-compiler",
                       "wrong-provenance", "wrong-cache", "unsupported-host"):
            with self.subTest(damage=damage), isolated_profile() as (
                    root, source, build, candidate):
                if damage in ("dirty-common", "indexed-common"):
                    common = root / "scripts/common.sh"
                    common.write_bytes(common.read_bytes() + b"# drift\n")
                    if damage == "indexed-common":
                        subprocess.run(["git", "-C", str(root), "add",
                                        "scripts/common.sh"], check=True)
                elif damage == "untracked":
                    (root / "stray.txt").write_text("stray\n")
                elif damage == "tracked-lock":
                    (root / "toolchain/eshkol.lock").write_bytes(candidate.read_bytes())
                elif damage == "wrong-head":
                    with mock.patch.object(profile, "COMPILER_COMMIT", "0" * 40):
                        with self.assertRaises(ValueError):
                            profile.observe(candidate, source, build)
                    continue
                elif damage == "wrong-tree":
                    with mock.patch.object(profile, "COMPILER_TREE", "0" * 40):
                        with self.assertRaises(ValueError):
                            profile.observe(candidate, source, build)
                    continue
                elif damage == "dirty-compiler":
                    (source / "stray.txt").write_text("stray\n")
                elif damage == "wrong-provenance":
                    (build / "eshkol-transformer-provenance.tsv").write_text(
                        "eshkol_commit\twrong\n" +
                        "eshkol_source_dir\t" + str(source.resolve()) + "\n")
                elif damage == "wrong-cache":
                    (build / "CMakeCache.txt").write_text(
                        "CMAKE_HOME_DIRECTORY:INTERNAL=/wrong\n")
                else:
                    with mock.patch.dict(os.environ,
                                         {"ESHKOL_ALLOW_UNSUPPORTED_HOST": "1"}):
                        with self.assertRaises(ValueError):
                            profile.observe(candidate, source, build)
                    continue
                with self.assertRaises(ValueError):
                    profile.observe(candidate, source, build)

    def test_parent_file_lock_and_tool_post_drift(self) -> None:
        with isolated_profile() as (_, source, build, candidate):
            before = profile.observe(candidate, source, build)
            replacement = candidate.with_name("replacement.lock")
            replacement.write_bytes(candidate.read_bytes())
            replacement.replace(candidate)
            self.assertNotEqual(before, profile.observe(candidate, source, build))
            candidate.write_bytes(candidate.read_bytes() + b"drift\n")
            with self.assertRaises(ValueError):
                profile.observe(candidate, source, build)
        with isolated_profile() as (_, source, build, candidate):
            before = profile.observe(candidate, source, build)
            binary = build / "eshkol-run"
            binary.write_bytes(binary.read_bytes() + b"# tool drift\n")
            self.assertNotEqual(before, profile.observe(candidate, source, build))


if __name__ == "__main__":
    unittest.main()
