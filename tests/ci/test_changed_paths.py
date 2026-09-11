from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.ci import changed_paths as ci_changed_paths


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = PROJECT_ROOT / "tests" / "ci" / "changed_paths.py"


class ChangedPathSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name)
        self._git("init", "--quiet")
        self._git("config", "user.name", "CI Test")
        self._git("config", "user.email", "ci-test@example.invalid")

        self._write("README.md", "initial\n")
        self._write("CONTRIBUTING.md", "initial\n")
        self._write("docs/guide.md", "initial\n")
        self._write("src/program.esk", "initial\n")
        self.base = self._commit("initial")

    def _git(self, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=self.repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    def _write(self, relative_path: str, contents: str) -> None:
        path = self.repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    def _commit(self, message: str) -> str:
        self._git("add", "--all")
        self._git("commit", "--quiet", "--message", message)
        return self._git("rev-parse", "HEAD")

    def _requires_full_suite(self, head: str) -> bool:
        return ci_changed_paths.requires_full_suite(
            self.base, head, repository=self.repository
        )

    def _run_selector(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SELECTOR), *arguments],
            cwd=self.repository,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_root_and_nested_markdown_prose_can_skip_full_suite(self) -> None:
        self._write("README.md", "updated\n")
        self._write("CONTRIBUTING.md", "updated\n")
        self._write("docs/reference/nested.md", "new\n")
        head = self._commit("prose only")
        self.assertFalse(self._requires_full_suite(head))

    def test_code_change_requires_full_suite(self) -> None:
        self._write("src/program.esk", "updated\n")
        head = self._commit("code")
        self.assertTrue(self._requires_full_suite(head))

    def test_mixed_prose_and_code_requires_full_suite(self) -> None:
        self._write("docs/guide.md", "updated\n")
        self._write("src/program.esk", "updated\n")
        head = self._commit("mixed")
        self.assertTrue(self._requires_full_suite(head))

    def test_unknown_markdown_path_requires_full_suite(self) -> None:
        self._write("notes/design.md", "new\n")
        head = self._commit("unknown markdown")
        self.assertTrue(self._requires_full_suite(head))

    def test_docs_agents_file_requires_full_suite(self) -> None:
        self._write("docs/AGENTS.md", "instructions\n")
        head = self._commit("operational instructions")
        self.assertTrue(self._requires_full_suite(head))

    def test_root_agents_file_requires_full_suite(self) -> None:
        self._write("AGENTS.md", "instructions\n")
        head = self._commit("root operational instructions")
        self.assertTrue(self._requires_full_suite(head))

    def test_docs_deletion_can_skip_full_suite(self) -> None:
        (self.repository / "docs/guide.md").unlink()
        head = self._commit("delete prose")
        self.assertFalse(self._requires_full_suite(head))

    def test_rename_from_code_to_docs_is_seen_as_code_and_requires_full_suite(
        self,
    ) -> None:
        self._git("mv", "src/program.esk", "docs/program.md")
        head = self._commit("rename code as prose")
        self.assertTrue(self._requires_full_suite(head))

    def test_rename_between_prose_paths_can_skip_full_suite(self) -> None:
        self._git("mv", "docs/guide.md", "docs/renamed.md")
        head = self._commit("rename prose")
        self.assertFalse(self._requires_full_suite(head))

    def test_nul_delimited_diff_accepts_newline_in_docs_filename(self) -> None:
        self._write("docs/line\nbreak.md", "new\n")
        head = self._commit("newline path")
        self.assertFalse(self._requires_full_suite(head))

    def test_empty_diff_requires_full_suite(self) -> None:
        self.assertTrue(self._requires_full_suite(self.base))

    def test_shallow_checkout_without_base_commit_requires_full_suite(self) -> None:
        self._write("docs/guide.md", "updated\n")
        head = self._commit("shallow head")
        clone_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(clone_temporary.cleanup)
        shallow = Path(clone_temporary.name) / "shallow"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth=1",
                self.repository.as_uri(),
                str(shallow),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertTrue(
            ci_changed_paths.requires_full_suite(self.base, head, repository=shallow)
        )

    def test_invalid_or_missing_revisions_require_full_suite(self) -> None:
        cases = (
            ("", self.base),
            (self.base, ""),
            ("missing-base", self.base),
            (self.base, "missing-head"),
            ("--help", self.base),
            ("bad\0base", self.base),
        )
        for base, head in cases:
            with self.subTest(base=base, head=head):
                self.assertTrue(
                    ci_changed_paths.requires_full_suite(
                        base, head, repository=self.repository
                    )
                )

    def test_cli_prints_only_the_boolean_decision(self) -> None:
        self._write("docs/guide.md", "updated\n")
        head = self._commit("cli prose")

        prose = self._run_selector("--base", self.base, "--head", head)
        self.assertEqual(
            (prose.returncode, prose.stdout, prose.stderr), (0, "false\n", "")
        )

        invalid = self._run_selector("--base", "missing", "--head", head)
        self.assertEqual(
            (invalid.returncode, invalid.stdout, invalid.stderr), (0, "true\n", "")
        )

        missing = self._run_selector()
        self.assertEqual(
            (missing.returncode, missing.stdout, missing.stderr), (0, "true\n", "")
        )


if __name__ == "__main__":
    unittest.main()
