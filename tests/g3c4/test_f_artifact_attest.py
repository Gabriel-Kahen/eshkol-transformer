"""Owned source-only negatives for the conditional F artifact admission."""

from __future__ import annotations

import hashlib
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.g3c4 import f_artifact_attest as attest
from tests.g3c4 import check_public_g2_f_artifact as source_check


ROOT = Path(__file__).resolve().parents[2]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FArtifactAttest(unittest.TestCase):
    def test_exact_offline_review_is_native_controls_only(self) -> None:
        path = Path("/home/gabe/.codex/evidence/eshkol-transformer/"
                    "e3-dual8-native-union-run-f-offline-audit-root-result-review-"
                    "20261009-c/SHA256SUMS-ROOT-REVIEW")
        self.assertEqual(attest.offline_review(path),
                         attest.OFFLINE_ROOT_REVIEW_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "SHA256SUMS-ROOT-REVIEW"
            wrong.write_bytes(b"forged review\n")
            with self.assertRaisesRegex(ValueError, "offline review missing"):
                attest.offline_review(wrong)

    def test_installed_caller_source_mutations(self) -> None:
        fixture = (ROOT / "tests/g3g_manual/public_g2_runtime.esk").read_text()
        attester = (ROOT / "tests/g3c4/f_artifact_attest.py").read_text()
        base = source_check.BASE.read_text().splitlines()
        closure = source_check.CLOSURE.read_text().splitlines()
        source = {name: (ROOT / name).read_bytes()
                  for name in source_check.UNCHANGED}
        source_check.validate(fixture, attester, base, closure, source)
        for changed in (fixture.replace("(require transformer.generation)",
                                        '(load "g3g_manual_package_root.esk")', 1),
                        fixture.replace("(generation-rng-release! two-rng)",
                                        "#t", 1),
                        fixture.replace("':profile 'diagnostic-c4-p2g2",
                                        "':profile 'diagnostic-c4-p2g1", 1)):
            with self.assertRaises(AssertionError):
                source_check.validate(changed, attester, base, closure, source)
        with self.assertRaises(AssertionError):
            source_check.validate(fixture, attester, base, closure[:-1], source)

    def test_exact_common_three_changes_only(self) -> None:
        original = (ROOT / "scripts/common.sh").read_bytes()
        staged = attest.staged_common(original)
        self.assertEqual(staged.count(b'LOCK_FILE="/bundle/public-g2-6803.lock"'), 1)
        self.assertEqual(staged.count(b'CMAKE_C_COMPILER:STRING'), 1)
        self.assertEqual(staged.count(b'CMAKE_CXX_COMPILER:STRING'), 1)
        self.assertEqual(staged.count(b'eshkol-transformer-provenance.tsv'),
                         original.count(b'eshkol-transformer-provenance.tsv'))
        with self.assertRaisesRegex(ValueError, "production common drift"):
            attest.staged_common(original.replace(b"verify_toolchain", b"skip_toolchain", 1))

    def test_exact_external_lock_and_wrong_path(self) -> None:
        tracked = (ROOT / "toolchain/eshkol.lock").read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.lock"
            path.write_bytes(tracked.replace(attest.OLD_ROW, attest.NEW_ROW))
            self.assertEqual(attest.candidate_lock(path, tracked, path)["sha256"],
                             attest.CANDIDATE_LOCK_SHA)
            with self.assertRaisesRegex(ValueError, "wrong candidate lock path"):
                attest.candidate_lock(path, tracked, path.parent / "other")
            path.write_bytes(tracked)
            with self.assertRaisesRegex(ValueError, "external lock differs"):
                attest.candidate_lock(path, tracked, path)
            path.unlink()
            (Path(tmp) / "target").write_bytes(tracked)
            path.symlink_to(Path(tmp) / "target")
            with self.assertRaises((ValueError, OSError)):
                attest.candidate_lock(path, tracked, path)

    def test_cache_and_receipt_malformations(self) -> None:
        original = Path("/home/gabe/.codex/evidence/eshkol-transformer/"
                        "e3-dual8-native-union-run-f-durable-evidence-20261009-a/output")
        cache = (original / "build-off/CMakeCache.txt").read_bytes()
        self.assertEqual(attest.cache_fields(cache)["ESHKOL_BUILD_TESTS:BOOL"], "ON")
        for wrong in (cache.replace(b"CMAKE_C_COMPILER:STRING", b"CMAKE_C_COMPILER:FILEPATH", 1),
                      cache.replace(b"ESHKOL_BUILD_TESTS:BOOL=ON", b"ESHKOL_BUILD_TESTS:BOOL=OFF", 1),
                      cache.replace(b"CMAKE_HOME_DIRECTORY:INTERNAL=/workspace",
                                    b"CMAKE_HOME_DIRECTORY:INTERNAL=/foreign", 1)):
            with self.assertRaisesRegex(ValueError, "F cache"):
                attest.cache_fields(wrong)
        self.assertIn("eshkol-run", attest.receipts(original)["build_argv"])
        with self.assertRaisesRegex(ValueError, "incomplete NUL argv"):
            attest.nul_argv(b"/usr/bin/cmake\0--build")
        with self.assertRaisesRegex(ValueError, "empty argv word"):
            attest.nul_argv(b"/usr/bin/cmake\0\0--build\0")

    def test_original_inventory_rejects_added_symlink_and_drift(self) -> None:
        observed = Path("/home/gabe/.codex/evidence/eshkol-transformer/"
                        "e3-dual8-native-union-run-f-durable-evidence-20261009-a/output")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            files = {}
            for relative in attest.F_PINS:
                if relative == "SHA256SUMS-RUN":
                    continue
                raw = ((observed / relative).read_bytes()
                       if relative.startswith("logs/") or
                       relative == "build-off/CMakeCache.txt" else
                       relative.encode())
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                files[relative] = sha(raw)
            link = output / "build-off/generated/llvm-selected-2cc6322eadfd/llvm"
            link.parent.mkdir(parents=True)
            link.symlink_to("/usr/lib/llvm-21/include/llvm")
            directory_rows = "path_hex\n" + "".join(
                f"{p.relative_to(output).as_posix().encode().hex()}\n"
                for p in sorted(output.rglob("*")) if p.is_dir() and
                not p.is_symlink())
            (output / "directory-inventory.tsv").write_text(directory_rows)
            files["directory-inventory.tsv"] = sha(directory_rows.encode())
            manifest = "".join(f"{digest}  ./{name}\n"
                               for name, digest in sorted(files.items())).encode()
            (output / "SHA256SUMS-RUN").write_bytes(manifest)
            files["SHA256SUMS-RUN"] = sha(manifest)
            with patch.object(attest, "F_PINS", files), \
                 patch.object(attest, "GENERATED", {
                     "build-off/generated/llvm-selected-2cc6322eadfd/llvm":
                         "/usr/lib/llvm-21/include/llvm"}):
                self.assertEqual(len(attest.original_output(output)["links"]), 1)
                extra = output / "build-off/foreign"
                extra.symlink_to("/tmp/foreign")
                with self.assertRaisesRegex(ValueError, "extra F symlink"):
                    attest.original_output(output)
                extra.unlink()
                fifo = output / "build-off/unlisted-fifo"
                os.mkfifo(fifo)
                with self.assertRaisesRegex(ValueError, "unclassified filesystem node"):
                    attest.original_output(output)
                fifo.unlink()
                (output / "build-off/libeshkol-static.a").write_bytes(b"drift")
                with self.assertRaisesRegex(ValueError, "F sealed member drift"):
                    attest.original_output(output)

    def test_exact_four_artifact_copy_and_partial_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            original.mkdir()
            pins = {}
            for index, name in enumerate(attest.ARTIFACTS):
                path = original / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = f"artifact-{index}".encode()
                path.write_bytes(raw)
                pins[name] = sha(raw)
            with patch.object(attest, "F_PINS", pins):
                copied = root / "copy"
                self.assertEqual(len(attest.copy_f_artifacts(original, copied)), 4)
                self.assertEqual({p.name for p in copied.iterdir()},
                                 {Path(p).name for p in attest.ARTIFACTS})
                with self.assertRaisesRegex(ValueError, "fresh"):
                    attest.copy_f_artifacts(original, copied)
                (copied / "libeshkol-runtime.a").write_bytes(b"drift")
                with self.assertRaisesRegex(ValueError, "copied F artifact drift"):
                    attest.copied_artifacts(copied, copied)
            incomplete = root / "incomplete"
            with patch.object(attest, "F_PINS", pins), \
                 patch("tests.g3c4.f_artifact_attest.os.fsync",
                       side_effect=OSError("owned fsync fault")):
                with self.assertRaisesRegex(OSError, "owned fsync fault"):
                    attest.copy_f_artifacts(original, incomplete)
            self.assertTrue(incomplete.is_dir())
            self.assertTrue(any(incomplete.iterdir()))

    def test_final_copy_requires_exact_compatibility_and_membership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp)
            pins = {}
            for name in attest.ARTIFACTS:
                path = copied / Path(name).name
                path.write_bytes(name.encode())
                pins[name] = sha(path.read_bytes())
            compatibility = copied / "eshkol-transformer-provenance.tsv"
            compatibility.write_bytes(b"test-only compatibility\n")
            with patch.object(attest, "F_PINS", pins), \
                 patch.object(attest, "ARTIFACTS", tuple(pins)):
                self.assertEqual(attest.final_copied_artifacts(
                    copied, sha(compatibility.read_bytes()), copied),
                    {Path(name).name: pin for name, pin in pins.items()})
                compatibility.write_bytes(b"post-publication drift\n")
                with self.assertRaisesRegex(ValueError, "compatibility view drift"):
                    attest.final_copied_artifacts(copied, sha(b"test-only compatibility\n"),
                                                  copied)
                compatibility.write_bytes(b"test-only compatibility\n")
                (copied / "extra").write_bytes(b"unreviewed")
                with self.assertRaisesRegex(ValueError, "membership drift"):
                    attest.final_copied_artifacts(
                        copied, sha(compatibility.read_bytes()), copied)

    def test_attestation_publication_keeps_partial_on_fsync_fault(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new-attestation.json"
            with patch("tests.g3c4.f_artifact_attest.os.fsync",
                       side_effect=OSError("owned fsync fault")):
                with self.assertRaisesRegex(OSError, "owned fsync fault"):
                    attest.exclusive_bytes(path, b"partial evidence\n")
            self.assertEqual(path.read_bytes(), b"partial evidence\n")
            with self.assertRaisesRegex(ValueError, "already exists"):
                attest.exclusive_bytes(path, b"replacement\n")

    def test_short_write_completes_and_eio_keeps_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original_write = os.write
            calls = 0

            def short_first(fd: int, raw: bytes) -> int:
                nonlocal calls
                calls += 1
                return original_write(fd, raw[:1] if calls == 1 else raw)

            good = Path(tmp) / "short.json"
            with patch("tests.g3c4.f_artifact_attest.os.write",
                       side_effect=short_first):
                attest.exclusive_bytes(good, b"complete\n")
            self.assertGreater(calls, 1)
            self.assertEqual(good.read_bytes(), b"complete\n")

            bad = Path(tmp) / "eio.json"
            calls = 0

            def eio_second(fd: int, raw: bytes) -> int:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("owned EIO")
                return original_write(fd, raw[:1])

            with patch("tests.g3c4.f_artifact_attest.os.write",
                       side_effect=eio_second):
                with self.assertRaisesRegex(OSError, "owned EIO"):
                    attest.exclusive_bytes(bad, b"no false success\n")
            self.assertEqual(bad.read_bytes(), b"n")

    def test_python310_cleanup_keeps_primary_and_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real_close = os.close
            for close_index in (1, 2):
                path = Path(tmp) / f"close-{close_index}.json"
                closes = 0
                syncs = 0
                def failing_close(fd: int) -> None:
                    nonlocal closes
                    closes += 1
                    real_close(fd)
                    if closes == close_index:
                        raise OSError("secondary close fault")
                real_sync = os.fsync
                def sync(fd: int) -> None:
                    nonlocal syncs
                    syncs += 1
                    if syncs == close_index:
                        raise OSError("primary fsync fault")
                    real_sync(fd)
                with patch("tests.g3c4.f_artifact_attest.os.fsync", side_effect=sync), \
                     patch("tests.g3c4.f_artifact_attest.os.close",
                           side_effect=failing_close):
                    with self.assertRaisesRegex(OSError, "primary fsync fault") as caught:
                        attest.exclusive_bytes(path, b"retained partial\n")
                self.assertEqual(path.read_bytes(), b"retained partial\n")
                self.assertIn("secondary close fault", " ".join(
                    caught.exception._attestation_cleanup))

    def test_source_copy_rejects_extra_and_wrong_common(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "scripts").mkdir()
            (source / "scripts/common.sh").write_bytes(
                (ROOT / "scripts/common.sh").read_bytes())
            (source / "data.txt").write_text("reviewed\n")
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Codex",
                            "-c", "user.email=codex@openai.com", "commit", "-qm",
                            "fixture"], check=True)
            head, tree = attest.git(source, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
            listing = subprocess.run(["git", "--no-optional-locks", "-C",
                                      str(source), "ls-files", "-s", "-z"],
                                     check=True, capture_output=True).stdout
            manifest = ""
            for entry in listing.rstrip(b"\0").split(b"\0"):
                metadata, raw_name = entry.split(b"\t", 1)
                mode = metadata.decode().split()[0]
                name = raw_name.decode()
                manifest += f"{name}\t{mode}\t{sha((source / name).read_bytes())}\n"
            decision = {"head": head, "tree": tree,
                        "tracked_file_count": 2,
                        "tracked_manifest_sha256": sha(manifest.encode()),
                        "admission_sha256": "synthetic-owned-decision"}
            prepared = base / "prepared"
            staged = attest.stage_project_copy(source, prepared, decision,
                                               prepared)
            self.assertEqual(staged["tracked_file_count"], "2")
            self.assertFalse((prepared / ".git").exists())
            with self.assertRaisesRegex(ValueError, "fresh"):
                attest.stage_project_copy(source, prepared, decision, prepared)
            candidate = base / "candidate"
            (candidate / "scripts").mkdir(parents=True)
            (candidate / "scripts/common.sh").write_bytes(attest.staged_common(
                (source / "scripts/common.sh").read_bytes()))
            (candidate / "data.txt").write_text("reviewed\n")
            self.assertEqual(attest.project_copy(source, candidate, decision,
                                                 candidate)["tracked_file_count"], "2")
            (candidate / "extra").write_text("unreviewed")
            with self.assertRaisesRegex(ValueError, "membership drift"):
                attest.project_copy(source, candidate, decision, candidate)
            (candidate / "extra").unlink()
            (candidate / "scripts/common.sh").write_bytes(
                (source / "scripts/common.sh").read_bytes())
            with self.assertRaisesRegex(ValueError, "candidate tracked source drift"):
                attest.project_copy(source, candidate, decision, candidate)
            (candidate / "scripts/common.sh").write_bytes(attest.staged_common(
                (source / "scripts/common.sh").read_bytes()))
            changed = dict(decision)
            changed["tracked_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "reviewed source content closure drift"):
                attest.project_copy(source, candidate, changed, candidate)
            changed = dict(decision)
            changed["tree"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "source checkout identity"):
                attest.project_copy(source, candidate, changed, candidate)
            extra_directory = candidate / "empty-extra"
            extra_directory.mkdir()
            with self.assertRaisesRegex(ValueError, "directory membership drift"):
                attest.project_copy(source, candidate, decision, candidate)
            extra_directory.rmdir()
            extra_fifo = candidate / "unlisted-fifo"
            os.mkfifo(extra_fifo)
            with self.assertRaisesRegex(ValueError, "unclassified filesystem node"):
                attest.project_copy(source, candidate, decision, candidate)
            extra_fifo.unlink()

    def test_external_source_admission_authenticates_reviews_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviews = []
            for label in ("INDEPENDENT", "ROOT"):
                path = root / f"SHA256SUMS-{label}"
                path.write_text(f"owned {label} review\n")
                reviews.append({"path": str(path), "sha256": sha(path.read_bytes())})
            admission = root / "decision.json"
            decision = {"format": "g3c4-public-g2-source-admission-v1",
                        "head": "a" * 40, "tree": "b" * 40,
                        "tracked_manifest_sha256": "c" * 64,
                        "tracked_file_count": 2,
                        "independent_review": reviews[0], "root_review": reviews[1]}
            admission.write_bytes(attest.canonical_json(decision))
            expected_sha = sha(admission.read_bytes())
            self.assertEqual(attest.source_admission(
                admission, expected_sha, (root / "candidate",), admission)["head"],
                "a" * 40)
            with self.assertRaisesRegex(ValueError, "digest drift"):
                attest.source_admission(admission, "0" * 64, (), admission)
            with self.assertRaisesRegex(ValueError, "outside the reviewed boundary"):
                attest.source_admission(admission, expected_sha, (root,), admission)
            with self.assertRaises(FileNotFoundError):
                attest.source_admission(root / "missing", expected_sha, (),
                                        root / "missing")
            symlink = root / "symlink.json"
            symlink.symlink_to(admission)
            with self.assertRaises((ValueError, OSError)):
                attest.source_admission(symlink, expected_sha, (), symlink)
            changed = dict(decision)
            changed["head"] = "z" * 40
            admission.write_bytes(attest.canonical_json(changed))
            with self.assertRaisesRegex(ValueError, "schema/identity malformed"):
                attest.source_admission(admission, sha(admission.read_bytes()), (),
                                        admission)
            admission.write_bytes(attest.canonical_json(decision))
            reviews[0]["sha256"] = "0" * 64
            admission.write_bytes(attest.canonical_json(decision))
            with self.assertRaisesRegex(ValueError, "review reference drift"):
                attest.source_admission(admission, sha(admission.read_bytes()), (),
                                        admission)

    def test_loader_and_compatibility_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / "libsample.so.1"
            transitive = Path(tmp) / "libtransitive.so.2"
            library.write_bytes(b"actual-image-library")
            transitive.write_bytes(b"actual-transitive-image-library")
            raw_elf = b" 0x1 (NEEDED) Shared library: [libsample.so.1]\n" \
                      b" 0x1 (RUNPATH) Library runpath: [/pinned]\n"
            loader = (f"\tlibsample.so.1 => {library} (0x0123)\n"
                      f"\tlibtransitive.so.2 => {transitive} (0x0456)\n").encode()
            pins = (f"libsample.so.1\t{library}\t{sha(library.read_bytes())}\n"
                    f"libtransitive.so.2\t{transitive}\t{sha(transitive.read_bytes())}\n").encode()
            argv = b"/lib64/ld-linux-x86-64.so.2\0--list\0/out/build-off/eshkol-run\0"
            with patch.object(attest, "ELF_NEEDED", frozenset({"libsample.so.1"})), \
                 patch.object(attest, "ELF_RUNPATH", "/pinned"):
                self.assertEqual(len(attest.resolved_image_libs(raw_elf, argv, b"0\n",
                                                                loader, b"", pins)), 2)
                omitted = loader.splitlines(keepends=True)[1]
                with self.assertRaisesRegex(ValueError, "direct ELF dependency missing"):
                    attest.resolved_image_libs(raw_elf, argv, b"0\n", omitted, b"", pins)
                with self.assertRaisesRegex(ValueError, "complete loader closure"):
                    attest.resolved_image_libs(raw_elf, argv, b"0\n", loader, b"",
                                               pins.splitlines(keepends=True)[0])
                extra = loader + f"\tlibunreviewed.so.3 => {transitive} (0x0789)\n".encode()
                with self.assertRaisesRegex(ValueError, "complete loader closure"):
                    attest.resolved_image_libs(raw_elf, argv, b"0\n", extra, b"", pins)
                with self.assertRaisesRegex(ValueError, "image library identity"):
                    attest.resolved_image_libs(raw_elf, argv, b"0\n", loader, b"",
                                               pins.replace(sha(library.read_bytes()).encode(),
                                                            b"0" * 64))
                transitive.write_bytes(b"post-pin mutation")
                with self.assertRaisesRegex(ValueError, "image library identity"):
                    attest.resolved_image_libs(raw_elf, argv, b"0\n", loader, b"", pins)
                with self.assertRaisesRegex(ValueError, "loader receipt invalid"):
                    attest.resolved_image_libs(raw_elf, argv, b"1\n", loader, b"", pins)
            versions = {key: "21.1.8" for key in
                        ("llvm_version", "cc_version", "cxx_version",
                         "cmake_version", "ninja_version", "make_version",
                         "bash_version", "git_version")}
            compat = attest.compatibility_tsv(Path("/workspace"), "a" * 64, versions)
            self.assertIn(b"attestation_origin\tnew-test-only-F-copy;not-bootstrap", compat)
            with self.assertRaisesRegex(ValueError, "tool version inventory"):
                attest.compatibility_tsv(Path("/workspace"), "a" * 64, {})

    def test_prepared_receipt_must_be_exact_and_current(self) -> None:
        expected = attest.prepared_payload({"head": "selected"},
                                           {"tree": "selected"},
                                           {"inventory": "original"},
                                           {"binary": "copy"},
                                           {"lock": "ordinary"},
                                           "offline-seal")
        raw = attest.canonical_json(expected)
        attest.verify_prepared(raw, expected)
        for forged in (raw.replace(b"selected", b"foreign", 1),
                       raw[:-1], raw + b"extra\n"):
            with self.assertRaisesRegex(ValueError, "prepared attestation"):
                attest.verify_prepared(forged, expected)
        changed = dict(expected)
        changed["copied"] = {"binary": "other"}
        with self.assertRaisesRegex(ValueError, "prepared attestation"):
            attest.verify_prepared(raw, changed)

    def test_compiler_elf_receipt_needs_same_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "eshkol-run"
            binary.write_bytes(b"selected compiler")
            elf = b" 0x1 (NEEDED) Shared library: [libsample.so.1]\n" \
                  b" 0x1 (RUNPATH) Library runpath: [/pinned]\n"
            completed = subprocess.CompletedProcess(
                ["readelf", "-d", str(binary)], 0, elf, b"")
            with patch.dict(attest.F_PINS, {"build-off/eshkol-run": sha(binary.read_bytes())}), \
                 patch.object(attest, "ELF_NEEDED", frozenset({"libsample.so.1"})), \
                 patch.object(attest, "ELF_RUNPATH", "/pinned"), \
                 patch.object(attest, "version_output",
                              return_value="Eshkol Compiler v1.3.4-evolve\n"), \
                 patch("tests.g3c4.f_artifact_attest.subprocess.run",
                       return_value=completed):
                self.assertEqual(attest.compiler_binary_and_elf(binary, elf)[
                    "needed_count"], "1")
                with self.assertRaisesRegex(ValueError, "readelf receipt differs"):
                    attest.compiler_binary_and_elf(binary, elf + b"forged")
                binary.write_bytes(b"other compiler")
                with self.assertRaisesRegex(ValueError, "copied compiler changed"):
                    attest.compiler_binary_and_elf(binary, elf)


if __name__ == "__main__":
    unittest.main()
