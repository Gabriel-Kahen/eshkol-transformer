"""Development-only split-capture controls; no compiler or native gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.g3c4 import c4_split_capture as split
from tests.g3c4 import c4_native_transcript as transcript
from tests.g3c4 import check_c4_split_capture as checker
from tests.g3c4.test_c4_native_transcript import fixture as synthetic_fixture


def synthetic_tree(root: Path) -> dict:
    """Fabricate a labelled schema fixture; never accepted as native evidence."""
    (root / "tmp").mkdir()
    for member in split.expected_members():
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    origin = "/guest-output"
    state = {"source": {"commit": split.BASE_COMMIT, "tree": split.BASE_TREE,
                        "closure": []},
             "fixture": split.FIXTURE_SHA,
             "candidate_lock": {"sha256": split.LOCK_SHA},
             "guest_output": origin}
    (root / "INPUTS.pre.json").write_bytes(split.canonical(state))
    (root / "INPUTS.post.json").write_bytes(split.canonical(state))
    for mode in ("normal", "sanitize"):
        binary = root / mode / "c4-native-capture"
        binary.write_bytes(b"synthetic-binary")
        inspections = ("archive-defined.symbols", "archive-undefined.symbols",
                       "defined.symbols", "undefined.symbols", "elf.symbols")
        stage_names = (["compile-" + Path(source).stem
                        for source in split.SOURCES] + ["archive", "link"] +
                       ["inspect-" + name.replace(".", "-")
                        for name in inspections])
        for stage in stage_names:
            prefix = root / mode / stage
            args = ["/usr/bin/true"]
            wrapper = b"#!/usr/bin/bash\nexec /usr/bin/true\n"
            prefix.with_suffix(".sh").write_bytes(wrapper)
            prefix.with_suffix(".argv.json").write_bytes(split.canonical(args))
            receipt = receipt_for(str(Path(origin) / mode / (stage + ".sh")),
                                  hashlib.sha256(wrapper).hexdigest(), b"")
            prefix.with_suffix(".status.json").write_bytes(split.canonical(receipt))
    for channel in split.CHANNELS:
        prefix = root / channel
        raw = b"C4CAP1\nENDCAPTURE\n"  # parser will reject; transport test only
        prefix.with_suffix(".stdout").write_bytes(raw)
        sha = split.file_digest(root / channel.split("/")[0] /
                                "c4-native-capture")
        receipt = receipt_for(str(Path(origin) / channel.split("/")[0] /
                                  "c4-native-capture"), sha, raw)
        prefix.with_suffix(".status.json").write_bytes(split.canonical(receipt))
    (root / "GUEST-INVENTORY.json").write_bytes(split.canonical(split.inventory(root)))
    return state


def receipt_for(binary: str, sha: str, stdout: bytes) -> dict:
    return {"binary": binary, "sha256_pre": sha, "sha256_post": sha,
            "returncode": 0, "leader_identity": [123, 123, 123, 456, "Z"],
            "uid": 1000, "elapsed_seconds": 0.1, "timeout_seconds": 120,
            "wait_status": 0, "quiescent": True, "timed_out": False,
            "overflow": "", "stdout_seen": len(stdout), "stderr_seen": 0,
            "teardown": [], "failure": "", "stdout_bytes": len(stdout),
            "stderr_bytes": 0}


def synthetic_raw(frozen: dict) -> bytes:
    lines = ["C4CAP1"]
    for model in frozen["models"]:
        seed = model["seed"]
        lines.extend((f"MODEL {seed} 1 {seed} 292 0",
                      "PARAMETERS 1192 " + " ".join(["00000000"] * 1192)))
        for row in (case for case in frozen["cases"] if case["seed"] == seed):
            tokens = row["tokens"]
            prefix = len(tokens)
            lines.append(f"CASE {seed} {prefix} " + " ".join(map(str, tokens)))
            if prefix == 4:
                lines.append("T4ABORT 1 0")
            for tag, count in (("LOGITS", prefix * 256),
                               ("KEYS", prefix * 4),
                               ("VALUES", prefix * 4)):
                lines.append(f"{tag} {count} " +
                             " ".join(["00000000"] * count))
            lines.append("ENDCASE")
    lines.append("ENDCAPTURE")
    return ("\n".join(lines) + "\n").encode()


def populate_synthetic_observations(root: Path, fixture: Path) -> None:
    frozen = transcript.load_fixture(fixture)
    raw = synthetic_raw(frozen)
    observation = transcript.encode_observation(transcript.parse(raw, frozen))
    for channel in split.CHANNELS:
        prefix = root / channel
        prefix.with_suffix(".stdout").write_bytes(raw)
        prefix.with_suffix(".json").write_bytes(observation)
        status = json.loads(prefix.with_suffix(".status.json").read_bytes())
        status["stdout_bytes"] = status["stdout_seen"] = len(raw)
        prefix.with_suffix(".status.json").write_bytes(split.canonical(status))
    (root / "GUEST-INVENTORY.json").write_bytes(split.canonical(split.inventory(root)))


class SplitCaptureTests(unittest.TestCase):
    def test_source_closure_and_310_shape(self) -> None:
        checker.main()
        text = checker.SOURCE.read_text()
        base = checker.BASE.read_text().splitlines()
        closure = checker.CLOSURE.read_text().splitlines()
        for marker in ("transcript.run_native", "oracle.compare_observation",
                       "verify_closed_tree", "q0_before == q0_state()",
                       "before == after"):
            with self.subTest(marker=marker):
                with self.assertRaises(AssertionError):
                    checker.check(text.replace(marker, "removed"),
                                  base, closure)
        with self.assertRaises(AssertionError):
            checker.check(text, base, closure[:-1])
        with self.assertRaises(AssertionError):
            checker.check(text + "\nimport docker\n", base, closure)
        with self.assertRaises(AssertionError):
            checker.check(text.replace('"native/a2_kv_cache.c"',
                                       '"native/unreviewed.c"', 1), base,
                          closure)

    def test_exact_member_census(self) -> None:
        self.assertEqual(len(split.SOURCES), 9)
        self.assertEqual(len(split.expected_members()), 207)
        self.assertIn("normal/repeat.stdout", split.expected_members())
        self.assertIn("sanitize/first.status.json", split.expected_members())
        self.assertNotIn("sanitize/repeat.stdout", split.expected_members())

    def test_c11_build_argv_is_nine_sources_and_actual_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = []

            def stage_stub(where, name, argv, timeout=120):
                observed.append((name, argv))
                if name.startswith("compile-") or name == "link":
                    artifact = Path(argv[argv.index("-o") + 1])
                    artifact.write_bytes(b"object")
                    if name == "link":
                        artifact.chmod(0o700)
                elif name == "archive":
                    Path(argv[2]).write_bytes(b"archive")
                return b"symbols" if name.startswith("inspect-") else b""

            with mock.patch.object(split, "stage", side_effect=stage_stub):
                binary = split.build_mode(root, "normal")
                self.assertTrue(binary.is_file())
                self.assertEqual(sum(name.startswith("compile-")
                                     for name, _ in observed), 9)
                self.assertEqual(len({argv[argv.index("-c") + 1]
                                      for name, argv in observed
                                      if name.startswith("compile-")}), 9)
                link = next(argv for name, argv in observed if name == "link")
                self.assertEqual(link[0], "/usr/bin/clang-21")
                self.assertIn("-Wl,--wrap=et_a2_kv_cache_transaction_abort_v1",
                              link)
                self.assertIn("-lm", link)
                self.assertFalse(any("eshkol-run" in arg for _, argv in observed
                                     for arg in argv))
                observed.clear()
                split.build_mode(root, "sanitize")
                sanitizer_link = next(argv for name, argv in observed
                                      if name == "link")
                self.assertIn("-fsanitize=address,undefined", sanitizer_link)

    def test_pin_manifest_wrong_digest_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pin = Path(directory) / "pins.tsv"
            raw = b"/usr/bin/python3\t/usr/bin/python3.10\t" + b"a" * 64 + b"\n"
            pin.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "pin manifest drift"):
                split._pin_rows(pin, "0" * 64, "tool")
            with self.assertRaisesRegex(ValueError, "pin count"):
                split._pin_rows(pin, hashlib.sha256(raw).hexdigest(), "tool")
            pin.write_bytes(b"garbage\n")
            with self.assertRaisesRegex(ValueError, "pin fields"):
                split._pin_rows(pin, split.file_digest(pin), "tool")

    def test_exact_source_admission_pin_and_dirty_rejection(self) -> None:
        head, tree = "a" * 40, "b" * 40

        def fake_git(*args):
            if args == ("rev-parse", f"{split.BASE_COMMIT}^{{tree}}"):
                return split.BASE_TREE
            if args == ("merge-base", "--is-ancestor", split.BASE_COMMIT,
                        head):
                return ""
            if args == ("rev-parse", "HEAD"):
                return "c" * 40
            if args == ("rev-parse", "HEAD^{tree}"):
                return tree
            if args == ("status", "--porcelain=v1", "--untracked-files=all"):
                return ""
            raise AssertionError(args)

        with mock.patch.object(split, "git", side_effect=fake_git):
            with self.assertRaisesRegex(ValueError, "exact clean source"):
                split.source_state(head, tree)

        def dirty_git(*args):
            if args == ("status", "--porcelain=v1", "--untracked-files=all"):
                return "?? unknown"
            if args == ("rev-parse", "HEAD"):
                return head
            return fake_git(*args)

        with mock.patch.object(split, "git", side_effect=dirty_git):
            with self.assertRaisesRegex(ValueError, "exact clean source"):
                split.source_state(head, tree)
        with self.assertRaisesRegex(ValueError, "pin format"):
            split.source_state("not-a-hash", tree)

    def test_owned_stage_receipt_and_failure_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(split.stage(root, "ok", ["/usr/bin/true"]), b"")
            split.require_receipt(root / "ok", str(root / "ok.sh"), 0)
            with self.assertRaisesRegex(ValueError, "native child failed"):
                split.stage(root, "fail", ["/usr/bin/false"])
            self.assertTrue((root / "fail.status.json").is_file())
            status = json.loads((root / "fail.status.json").read_bytes())
            self.assertEqual(status["returncode"], 1)
            self.assertIsNotNone(status["wait_status"])
            self.assertTrue(status["quiescent"])

    def test_guest_requires_three_matching_owned_raw_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            lock = root / "lock"
            tools = root / "tools"
            runtimes = root / "runtimes"
            for path in (fixture, lock, tools, runtimes):
                path.write_bytes(b"synthetic-input")
            raw = b"synthetic-native-raw"
            seen = []
            replies = [raw, raw, raw]

            def capture(binary, request, evidence):
                seen.append((str(binary), request,
                             bool(__import__("os").environ.get("ASAN_OPTIONS"))))
                result = replies.pop(0)
                evidence.parent.mkdir(parents=True, exist_ok=True)
                evidence.with_suffix(".stdout").write_bytes(result)
                return result

            with mock.patch.object(split, "guest_state", return_value={"pin": True}), \
                 mock.patch.object(split.transcript, "load_fixture", return_value={}), \
                 mock.patch.object(split.transcript, "make_input", return_value=b"request"), \
                 mock.patch.object(split.transcript, "run_native", side_effect=capture), \
                 mock.patch.object(split.transcript, "parse", return_value={}), \
                 mock.patch.object(split, "build_mode",
                                   side_effect=lambda output, mode: root / (mode + "-bin")), \
                 mock.patch.object(split, "inventory", return_value={"fake": "sha"}), \
                 mock.patch.object(split, "expected_members", return_value={"fake"}):
                split.guest(fixture, lock, tools, runtimes, root / "pass",
                            split.BASE_COMMIT, split.BASE_TREE)
                self.assertEqual(len(seen), 3)
                self.assertEqual([entry[2] for entry in seen],
                                 [False, False, True])
                self.assertEqual([entry[1] for entry in seen], [b"request"] * 3)
                self.assertTrue((root / "pass/GUEST-INVENTORY.json").exists())

                seen.clear()
                replies[:] = [raw, raw, b"different"]
                with self.assertRaisesRegex(ValueError, "raw drift"):
                    split.guest(fixture, lock, tools, runtimes, root / "fail",
                                split.BASE_COMMIT, split.BASE_TREE)
                self.assertTrue((root / "fail/GUEST-FAILURE.json").exists())
                self.assertFalse((root / "fail/GUEST-INVENTORY.json").exists())

    def test_durable_inventory_receipt_and_extra_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = synthetic_tree(root)
            self.assertEqual(split.verify_closed_tree(root, split.BASE_COMMIT,
                                                      split.BASE_TREE), state)
            bad = root / "normal/first.status.json"
            original = bad.read_bytes()
            record = json.loads(original)
            for field, value in (("wait_status", 9), ("quiescent", False),
                                 ("stderr_bytes", 1), ("overflow", "stdout"),
                                 ("returncode", 1), ("leader_identity", [1])):
                with self.subTest(field=field):
                    changed = dict(record)
                    changed[field] = value
                    bad.write_bytes(split.canonical(changed))
                    (root / "GUEST-INVENTORY.json").write_bytes(
                        split.canonical(split.inventory(root)))
                    with self.assertRaises(ValueError):
                        split.verify_closed_tree(root, split.BASE_COMMIT,
                                                 split.BASE_TREE)
            bad.write_bytes(original)
            (root / "GUEST-INVENTORY.json").write_bytes(
                split.canonical(split.inventory(root)))
            extra = root / "foreign"
            extra.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "inventory"):
                split.verify_closed_tree(root, split.BASE_COMMIT,
                                         split.BASE_TREE)
            extra.unlink()
            extra.mkdir()
            with self.assertRaisesRegex(ValueError, "directory census"):
                split.verify_closed_tree(root, split.BASE_COMMIT,
                                         split.BASE_TREE)
            extra.rmdir()
            (root / "normal/first.stdout").write_bytes(b"tamper")
            with self.assertRaisesRegex(ValueError, "inventory"):
                split.verify_closed_tree(root, split.BASE_COMMIT,
                                         split.BASE_TREE)

    def test_host_rejects_unclosed_or_wrong_raw_before_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guest = root / "guest"
            guest.mkdir()
            state = synthetic_tree(guest)
            fixture = root / "fixture.json"
            fixture.write_bytes(b"fixture")
            with mock.patch.object(split, "FIXTURE_SHA", split.file_digest(fixture)), \
                 mock.patch.object(split, "source_state", return_value=state["source"]):
                result = root / "result"
                with self.assertRaises(ValueError):
                    split.host(fixture, guest, result,
                               split.BASE_COMMIT, split.BASE_TREE)
                self.assertTrue((result / "HOST-FAILURE.json").is_file())
                self.assertFalse((result / "HOST-COMPARISON.json").exists())

    def test_closed_synthetic_raw_to_host_comparison_route(self) -> None:
        # Only the transport is synthetic; the mathematical comparator is mocked.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture.json"
            fixture.write_bytes(transcript.encode_observation(
                synthetic_fixture()))
            fixture_sha = split.file_digest(fixture)
            guest = root / "guest"
            guest.mkdir()
            state = synthetic_tree(guest)
            state["fixture"] = fixture_sha
            for name in ("INPUTS.pre.json", "INPUTS.post.json"):
                (guest / name).write_bytes(split.canonical(state))
            from tests.g3c4 import c4_math_oracle as oracle
            with mock.patch.object(split, "FIXTURE_SHA", fixture_sha), \
                 mock.patch.object(transcript, "FIXTURE_SHA256", fixture_sha), \
                 mock.patch.object(split, "source_state", return_value=state["source"]), \
                 mock.patch.object(split, "q0_state", return_value={"synthetic": True}), \
                 mock.patch.object(oracle, "require_q0"), \
                 mock.patch.object(oracle, "check_source_pins"), \
                 mock.patch.object(oracle, "decode", return_value={"synthetic": True}), \
                 mock.patch.object(oracle, "compare_observation",
                                   return_value={"maximum_absolute_error": 0.0}) as compare:
                populate_synthetic_observations(guest, fixture)
                self.assertEqual(split.verify_closed_tree(guest, split.BASE_COMMIT,
                                                          split.BASE_TREE), state)
                split.host(fixture, guest, root / "host",
                           split.BASE_COMMIT, split.BASE_TREE)
                self.assertEqual(compare.call_count, 3)
                self.assertTrue((root / "host/HOST-COMPARISON.json").is_file())
                bad = guest / "sanitize/first.stdout"
                original = bad.read_bytes()
                bad.write_bytes(bad.read_bytes().replace(b"T4ABORT 1 0",
                                                       b"T4ABORT 0 0", 1))
                (guest / "GUEST-INVENTORY.json").write_bytes(
                    split.canonical(split.inventory(guest)))
                with self.assertRaisesRegex(ValueError, "raw repeat"):
                    split.host(fixture, guest, root / "host-wrong",
                               split.BASE_COMMIT, split.BASE_TREE)
                self.assertEqual(compare.call_count, 3)
                bad.write_bytes(original)
                canonical = guest / "sanitize/first.json"
                canonical.write_bytes(b"{}\n")
                (guest / "GUEST-INVENTORY.json").write_bytes(
                    split.canonical(split.inventory(guest)))
                with self.assertRaisesRegex(ValueError, "canonical drift"):
                    split.host(fixture, guest, root / "host-wrong-canonical",
                               split.BASE_COMMIT, split.BASE_TREE)
                self.assertEqual(compare.call_count, 5)


if __name__ == "__main__":
    unittest.main()
