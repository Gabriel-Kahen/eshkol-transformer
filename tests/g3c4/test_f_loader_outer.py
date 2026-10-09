"""Owned source-only controls for the inactive F loader outer launcher."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.g3c4 import f_loader_outer as outer
from tests.g3c4 import check_f_loader_outer as source_check


class OuterTests(unittest.TestCase):
    def test_first_actual_image_and_created_inspect_label_union(self) -> None:
        fixture = Path(__file__).parent / "fixtures/f_loader_first_created"
        image = (fixture / "image.stdout.raw").read_bytes()
        created = (fixture / "created.stdout.raw").read_bytes()
        self.assertEqual(hashlib.sha256(image).hexdigest(),
                         "5110b38bb787512feb076ce27d0de79688edbf4ff3c5e1c5ac6b735d7c4754c1")
        self.assertEqual(hashlib.sha256(created).hexdigest(),
                         "0918c6df1efc5679f5a38537fff718669bb46b787bc5650db5b9f355f7c532d1")
        row = json.loads(created)[0]
        mounts = {entry["Target"]: entry["Source"]
                  for entry in row["HostConfig"]["Mounts"]}
        decision = {"name": row["Name"].removeprefix("/"),
                    "source_admission_sha256": row["Config"]["Cmd"][-1],
                    **{key: mounts[target] for key, target in (
                        ("compiler", "/workspace"), ("source", "/source"),
                        ("f_output", "/f-output"), ("bundle", "/bundle"),
                        ("candidate", "/stage"), ("out", "/out"))}}
        self.assertEqual(outer.expected_labels(image, decision["name"]),
                         row["Config"]["Labels"])
        self.assertEqual(outer.inspect_owned(created, decision, row["Id"],
                                             False, image), row)
        missing = copy.deepcopy(row)
        del missing["Config"]["Labels"]["org.opencontainers.image.version"]
        with self.assertRaisesRegex(ValueError, "owned container inspect mismatch"):
            outer.inspect_owned(json.dumps([missing]).encode(), decision,
                                row["Id"], False, image)

    def test_source_guard_denies_root_pin_and_profile_mutants(self) -> None:
        source = (source_check.ROOT / source_check.ADDED[1]).read_text()
        guest = (source_check.ROOT / source_check.ADDED[0]).read_text()
        original = source_check.OLD
        roadmap = source_check.NEW
        base = source_check.BASE.read_text().splitlines()
        closure = source_check.CLOSURE.read_text().splitlines()
        with self.assertRaises(AssertionError):
            source_check.validate(base, closure, source.replace(
                'external immutable ROOT wrapper required',
                'caller-provided pin is accepted'),
                guest, roadmap, original)
        with self.assertRaises(AssertionError):
            source_check.validate(base, closure,
                                  source.replace('"--network", "none"',
                                                 '"--network", "host"'),
                                  guest, roadmap, original)
        with self.assertRaises(AssertionError):
            source_check.validate(base, closure,
                                  source.replace(
                                      'row["Config"]["Labels"] == expected_labels(image_raw, decision["name"])',
                                      'row["Config"]["Labels"] == {"eshkol.gate": decision["name"]}'),
                                  guest, roadmap, original)
        with self.assertRaises(AssertionError):
            source_check.validate(base, closure, source,
                                  guest.replace('"--outer-stage-child"',
                                                '"--unsupported-stage"'),
                                  roadmap, original)
        with self.assertRaises(AssertionError):
            source_check.validate(base, closure[:-1], source, guest,
                                  roadmap, original)

    def test_inactive_exact_root_pin(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact external ROOT decision pin"):
            outer.selected(Path("/missing"), "caller-selected", fresh=True)

    def test_evidence_cannot_write_selected_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = {key: str(root / key) for key in
                     ("compiler", "source", "f_output", "bundle", "candidate", "out")}
            for path in roots.values():
                Path(path).mkdir()
            decision = {**roots, "name": "g3c4-f-loader-owned",
                        "source_admission_sha256": "a" * 64}
            with patch.object(outer, "selected", return_value=decision):
                with self.assertRaisesRegex(ValueError, "overlaps selected input"):
                    outer.run_once(root / "decision.json", "a" * 64,
                                   Path(roots["f_output"]) / "evidence")
            self.assertFalse((Path(roots["f_output"]) / "evidence").exists())

    def test_create_profile_and_guest_route(self) -> None:
        d = {"name": "g3c4-f-loader-owned", "compiler": "/one",
             "source": "/two", "f_output": "/three", "bundle": "/four",
             "candidate": "/five", "out": "/six",
             "source_admission_sha256": "a" * 64}
        argv = outer.create_argv(d)
        self.assertEqual(argv.count("--mount"), 6)
        self.assertEqual(argv.count("--read-only"), 1)
        self.assertEqual(argv[argv.index("--network") + 1], "none")
        self.assertEqual(argv[argv.index("--memory") + 1], "4g")
        self.assertEqual(argv[argv.index("--pids-limit") + 1], "128")
        self.assertEqual(argv[argv.index("--ulimit") + 1],
                         "fsize=134217728:134217728")
        self.assertGreater(outer.FSIZE_LIMIT, 85_874_424)
        self.assertEqual(argv[-2:], ["--source-admission-sha256", "a" * 64])
        self.assertIn("tests.g3c4.f_loader_guest", argv)
        self.assertEqual(argv[-4], "-m")
        self.assertIn("type=bind,src=/five,dst=/stage", argv)

    def test_real_wait_and_raw_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "normal"
            r = outer.capture([sys.executable, "-B", "-c",
                               "import sys;sys.stdout.write('ok')"], path, 5)
            self.assertTrue(r["quiescent"])
            self.assertEqual(r["returncode"], 0)
            self.assertEqual(r["wait_status"], 0)
            self.assertEqual((path / "stdout.raw").read_bytes(), b"ok")
            self.assertEqual((path / "stderr.raw").read_bytes(), b"")
            launch = json.loads((path / "argv.json").read_bytes())
            self.assertEqual(launch["cwd"], "/")
            self.assertEqual(launch["env"]["DOCKER_HOST"],
                             f"unix:///run/user/{outer.os.getuid()}/docker.sock")

    def test_owned_overflow_and_timeout_are_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(outer, "STREAM_LIMIT", 32):
            path = Path(tmp)
            overflow = outer.capture([sys.executable, "-B", "-c",
                                      "import sys;sys.stderr.write('x'*1000)"],
                                     path / "overflow", 5)
            self.assertIn("overflow", overflow["failure"])
            self.assertLessEqual(len((path / "overflow/stderr.raw").read_bytes()), 32)
            timed = outer.capture([sys.executable, "-B", "-c",
                                   "import time;time.sleep(10)"], path / "timeout", 1)
            self.assertIn("timeout", timed["failure"])
            self.assertIsNotNone(timed["wait_status"])
            self.assertTrue((path / "timeout/status.json").is_file())

    def test_owned_monitor_denial_keeps_actual_wait_and_raw_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "monitored"
            calls = [0]
            def deny() -> None:
                calls[0] += 1
                raise ValueError("aggregate bound")
            result = outer.capture([sys.executable, "-B", "-c",
                                    "import time;time.sleep(10)"],
                                   path, 5, deny)
            self.assertGreater(calls[0], 0)
            self.assertIn("monitor denied: ValueError: aggregate bound",
                          result["failure"])
            self.assertIsNotNone(result["wait_status"])
            self.assertTrue(result["quiescent"])
            self.assertTrue((path / "status.json").is_file())

    def test_inspect_requires_owned_exact_mounts(self) -> None:
        d = {"name": "g3c4-f-loader-owned", "compiler": "/one",
             "source": "/two", "f_output": "/three", "bundle": "/four",
             "candidate": "/five", "out": "/six",
             "source_admission_sha256": "a" * 64}
        mounts = [{"Type": "bind", "Source": d[key], "Destination": dest,
                   "RW": not ro, "Mode": "", "Propagation": "rprivate"}
                  for key, dest, ro in
                  (("compiler", "/workspace", True), ("source", "/source", True),
                   ("f_output", "/f-output", True), ("bundle", "/bundle", True),
                   ("candidate", "/stage", False), ("out", "/out", False))]
        host_mounts = [{"Type": "bind", "Source": d[key], "Target": dest,
                        **({"ReadOnly": True} if ro else {})}
                       for key, dest, ro in
                       (("compiler", "/workspace", True), ("source", "/source", True),
                        ("f_output", "/f-output", True), ("bundle", "/bundle", True),
                        ("candidate", "/stage", False), ("out", "/out", False))]
        image = json.dumps([{"Id": outer.IMAGE,
                             "Config": {"Env": ["PATH=/usr/bin", "DEBIAN_FRONTEND=noninteractive"],
                                        "Labels": {"org.opencontainers.image.version": "22.04"}}}]).encode()
        env = [f"{k}={v}" for k, v in outer.expected_env(image).items()]
        row = {"Id": "b" * 64, "Name": "/" + d["name"],
               "Image": outer.IMAGE, "Config": {
                   "Labels": {"org.opencontainers.image.version": "22.04",
                              "eshkol.gate": d["name"]}, "User": "0:0",
                   "WorkingDir": "/source", "Entrypoint": ["/usr/bin/python3.10"],
                   "Cmd": ["-B", "-m", "tests.g3c4.f_loader_guest",
                           "--source-admission-sha256", "a" * 64], "Env": env},
               "State": {"Running": False}, "Mounts": mounts,
               "HostConfig": {"ReadonlyRootfs": True, "NetworkMode": "none",
                              "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=268435456,mode=1777"},
                              "Memory": 4 * 1024**3, "MemorySwap": 4 * 1024**3,
                              "PidsLimit": 128, "NanoCpus": 2_000_000_000,
                              "Ulimits": [{"Name": "fsize", "Hard": outer.FSIZE_LIMIT,
                                           "Soft": outer.FSIZE_LIMIT}],
                              "Mounts": host_mounts, "Binds": None}}
        self.assertEqual(outer.inspect_owned(json.dumps([row]).encode(), d,
                                             "b" * 64, False, image), row)
        self.assertEqual(outer.expected_labels(image, d["name"]),
                         row["Config"]["Labels"])
        inherited = copy.deepcopy(row)
        del inherited["Config"]["Labels"]["org.opencontainers.image.version"]
        with self.assertRaisesRegex(ValueError, "owned container inspect mismatch"):
            outer.inspect_owned(json.dumps([inherited]).encode(), d,
                                "b" * 64, False, image)
        extra = copy.deepcopy(row)
        extra["Config"]["Labels"]["unreviewed"] = "value"
        with self.assertRaisesRegex(ValueError, "owned container inspect mismatch"):
            outer.inspect_owned(json.dumps([extra]).encode(), d,
                                "b" * 64, False, image)
        conflict = json.dumps([{"Id": outer.IMAGE, "Config": {
            "Env": [], "Labels": {"eshkol.gate": "foreign"}}}]).encode()
        with self.assertRaisesRegex(ValueError, "image reserves gate label"):
            outer.expected_labels(conflict, d["name"])
        for inherited in (["not-a-map"], {"wrong": 7}, {"": "empty-key"}):
            with self.subTest(inherited=inherited):
                malformed = json.dumps([{"Id": outer.IMAGE, "Config": {
                    "Env": [], "Labels": inherited}}]).encode()
                with self.assertRaisesRegex(ValueError, "image labels malformed"):
                    outer.expected_labels(malformed, d["name"])
        label_free = json.dumps([{"Id": outer.IMAGE, "Config": {
            "Env": [], "Labels": None}}]).encode()
        self.assertEqual(outer.expected_labels(label_free, d["name"]),
                         {"eshkol.gate": d["name"]})
        row["Mounts"].append({"Type": "bind", "Source": "/extra",
                              "Destination": "/extra", "RW": False,
                              "Mode": "", "Propagation": "rprivate"})
        with self.assertRaisesRegex(ValueError, "mounts differ"):
            outer.inspect_owned(json.dumps([row]).encode(), d, "b" * 64, False, image)
        row["Mounts"].pop()
        mutants = [
            ("user", lambda r: r["Config"].__setitem__("User", "1:1")),
            ("workdir", lambda r: r["Config"].__setitem__("WorkingDir", "/tmp")),
            ("entrypoint", lambda r: r["Config"].__setitem__("Entrypoint", ["/bin/sh"])),
            ("cmd", lambda r: r["Config"]["Cmd"].__setitem__(0, "-c")),
            ("env", lambda r: r["Config"]["Env"].append("LD_PRELOAD=/evil")),
            ("duplicate_env", lambda r: r["Config"]["Env"].append("PATH=/bin")),
            ("readonly", lambda r: r["HostConfig"].__setitem__("ReadonlyRootfs", False)),
            ("cpus", lambda r: r["HostConfig"].__setitem__("NanoCpus", 1_000_000_000)),
            ("fsize", lambda r: r["HostConfig"]["Ulimits"][0].__setitem__("Soft", 1)),
            ("old_fsize", lambda r: r["HostConfig"]["Ulimits"].__setitem__(
                0, {"Name": "fsize", "Hard": 16_777_216, "Soft": 16_777_216})),
            ("host_mount", lambda r: r["HostConfig"]["Mounts"][0].__setitem__("ReadOnly", False)),
            ("propagation", lambda r: r["Mounts"][0].__setitem__("Propagation", "shared")),
        ]
        for name, mutate in mutants:
            with self.subTest(name=name):
                bad = copy.deepcopy(row)
                mutate(bad)
                with self.assertRaises(ValueError):
                    outer.inspect_owned(json.dumps([bad]).encode(), d,
                                        "b" * 64, False, image)

    def test_guest_receipt_must_be_durable_and_bound_to_library_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            evidence = out / "evidence"
            (evidence / "loader").mkdir(parents=True)
            with patch.object(outer.attest, "copied_artifacts"):
                with self.assertRaisesRegex(ValueError, "durable guest receipts"):
                    outer.guest_output({"out": str(out)})
                (evidence / "F-copy-prepared.json").write_text("{}")
                (evidence / "loader/image-libraries.tsv").write_text("lib\n")
                (evidence / "loader/collector.json").write_text(json.dumps({
                    "format": "g3c4-F-image-loader-collector-v1",
                    "image_pins_sha256": "0" * 64}))
                with self.assertRaisesRegex(ValueError, "output identity"):
                    outer.guest_output({"out": str(out)})
                (evidence / "loader/collector.json").write_text(json.dumps({
                    "format": "g3c4-F-image-loader-collector-v1",
                    "image_pins_sha256": outer.sha(b"lib\n")}))
                outer.guest_output({"out": str(out)})

    def test_real_fresh_stage_child_under_mounted_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            (source / "scripts").mkdir()
            (source / "scripts/common.sh").write_bytes(b"test common\n")
            (source / "x.txt").write_bytes(b"tracked\n")
            for argv in (["git", "-C", str(source), "init", "-q"],
                         ["git", "-C", str(source), "add", "."],
                         ["git", "-C", str(source), "-c", "user.name=Test",
                          "-c", "user.email=test@example.invalid", "commit", "-qm", "owned"]):
                subprocess.run(argv, check=True, capture_output=True)
            head, tree = subprocess.check_output(
                ["git", "--no-optional-locks", "-C", str(source), "rev-parse",
                 "HEAD", "HEAD^{tree}"], text=True).splitlines()
            rows = "".join(f"{name}\t100644\t{hashlib.sha256((source/name).read_bytes()).hexdigest()}\n"
                           for name in ("scripts/common.sh", "x.txt"))
            decision = {"head": head, "tree": tree,
                        "tracked_file_count": 2,
                        "tracked_manifest_sha256": hashlib.sha256(rows.encode()).hexdigest(),
                        "admission_sha256": "a" * 64}
            stage = base / "stage"
            stage.mkdir()
            child = stage / "candidate"
            with patch.object(outer.attest, "staged_common", side_effect=lambda raw: raw):
                with self.assertRaisesRegex(ValueError, "destination must be fresh"):
                    outer.attest.stage_project_copy(source, child, decision)
                self.assertFalse(child.exists())
                copied = outer.attest.stage_project_copy(source, child, decision,
                                                          expected_candidate=child)
                self.assertEqual(copied["head"], head)
                self.assertEqual((child / "x.txt").read_bytes(), b"tracked\n")
                with self.assertRaisesRegex(ValueError, "destination must be fresh"):
                    outer.attest.stage_project_copy(source, child, decision,
                                                     expected_candidate=child)

    def test_same_canonical_admission_on_host_and_guest_review_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            roots = [base / "host-bundle", base / "guest-bundle"]
            review_bytes = {"independent_review": b"independent seal\n",
                            "root_review": b"root seal\n"}
            refs = {
                "independent_review": "/bundle/reviews/independent/SHA256SUMS-INDEPENDENT-SOURCE-REVIEW",
                "root_review": "/bundle/reviews/root/SHA256SUMS-ROOT-REVIEW",
            }
            data = {"format": "g3c4-public-g2-source-admission-v1",
                    "head": "a" * 40, "tree": "b" * 40,
                    "tracked_manifest_sha256": "c" * 64,
                    "tracked_file_count": 2}
            for key, logical in refs.items():
                data[key] = {"path": logical,
                             "sha256": outer.sha(review_bytes[key])}
            raw = outer.attest.canonical_json(data)
            for root in roots:
                for key, logical in refs.items():
                    path = root / Path(logical).relative_to("/bundle")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(review_bytes[key])
                (root / "public-g2-source-admission.json").write_bytes(raw)
            observations = []
            for root in roots:
                path = root / "public-g2-source-admission.json"
                observations.append(outer.attest.source_admission(
                    path, outer.sha(raw), (), expected_path=path,
                    outer_bundle_root=root))
            self.assertEqual(observations[0], observations[1])
            with self.assertRaisesRegex(ValueError, "digest drift"):
                outer.attest.source_admission(
                    roots[0] / "public-g2-source-admission.json", "0" * 64, (),
                    expected_path=roots[0] / "public-g2-source-admission.json",
                    outer_bundle_root=roots[0])
            wrong = copy.deepcopy(data)
            wrong["root_review"]["path"] = "/bundle/reviews/other/SHA256SUMS-ROOT-REVIEW"
            (roots[0] / "public-g2-source-admission.json").write_bytes(
                outer.attest.canonical_json(wrong))
            with self.assertRaisesRegex(ValueError, "logical review path"):
                outer.attest.source_admission(
                    roots[0] / "public-g2-source-admission.json",
                    outer.sha(outer.attest.canonical_json(wrong)), (),
                    expected_path=roots[0] / "public-g2-source-admission.json",
                    outer_bundle_root=roots[0])
            review = roots[1] / Path(refs["independent_review"]).relative_to("/bundle")
            review.write_bytes(b"mutated seal\n")
            with self.assertRaisesRegex(ValueError, "review reference drift"):
                outer.attest.source_admission(
                    roots[1] / "public-g2-source-admission.json", outer.sha(raw), (),
                    expected_path=roots[1] / "public-g2-source-admission.json",
                    outer_bundle_root=roots[1])

    def test_stopped_state_and_exact_docker_wait(self) -> None:
        state = {"Status": "exited", "Running": False, "Pid": 0,
                 "OOMKilled": False, "Dead": False, "Error": "", "ExitCode": 17}
        self.assertEqual(outer.stopped_exit({"State": state}), 17)
        outer.wait_exit(b"17\n", 17)
        for key, bad in (("Status", "created"), ("Running", True),
                         ("Pid", 42), ("OOMKilled", True), ("Dead", True),
                         ("Error", "failure"), ("ExitCode", None),
                         ("ExitCode", True)):
            with self.subTest(key=key, bad=bad):
                mutated = dict(state)
                mutated[key] = bad
                with self.assertRaisesRegex(ValueError, "authentically stopped"):
                    outer.stopped_exit({"State": mutated})
        for raw in (b"", b"17", b"17\nextra", b"017\n", b"18\n", b"-1\n"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "contradicts"):
                    outer.wait_exit(raw, 17)
        created = {**state, "Status": "created", "ExitCode": 0}
        outer.never_started({"State": created})
        for key, bad in (("Pid", True), ("OOMKilled", True),
                         ("Dead", True), ("Error", "bad"),
                         ("ExitCode", 1)):
            with self.subTest(created_key=key):
                altered = dict(created)
                altered[key] = bad
                with self.assertRaisesRegex(ValueError, "identity uncertain"):
                    outer.never_started({"State": altered})

    def test_mocked_one_shot_wait_precedes_remove_and_mismatch_denied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            roots = {key: str(base / key) for key in
                     ("compiler", "source", "f_output", "bundle", "candidate", "out")}
            for path in roots.values():
                Path(path).mkdir()
            decision = {**roots, "name": "g3c4-f-loader-owned",
                        "source_admission_sha256": "a" * 64}
            image = json.dumps([{"Id": outer.IMAGE, "Config": {"Env": []}}]).encode()
            state = {"Status": "exited", "Running": False, "Pid": 0,
                     "OOMKilled": False, "Dead": False, "Error": "", "ExitCode": 0}
            for answer, succeeds in ((b"0\n", True), (b"1\n", False)):
                with self.subTest(answer=answer):
                    calls = []
                    evidence = base / ("success" if succeeds else "mismatch")

                    def command(argv: list[str], phase: Path, timeout: int = 120) -> bytes:
                        action = argv[1]
                        calls.append(action)
                        if action == "image":
                            return image
                        if action == "info":
                            return b'["name=rootless"]\n'
                        if action == "ps":
                            return b""
                        if action == "create":
                            return ("b" * 64 + "\n").encode()
                        if action == "wait":
                            return answer
                        return b"[]"

                    def guest(argv: list[str], phase: Path,
                              timeout: int = outer.PHASE_LIMIT) -> dict:
                        phase.mkdir()
                        (phase / "stderr.raw").write_bytes(b"")
                        return {"quiescent": True, "returncode": 0, "failure": ""}

                    inspect_count = [0]

                    def inspect(*_args: object) -> dict:
                        inspect_count[0] += 1
                        if inspect_count[0] == 1:
                            return {"State": {**state, "Status": "created"}}
                        return {"State": state}

                    with patch.object(outer, "selected", return_value=decision), \
                         patch.object(outer, "checked", side_effect=command), \
                         patch.object(outer, "capture", side_effect=guest), \
                         patch.object(outer, "inspect_owned", side_effect=inspect), \
                         patch.object(outer, "guest_output"):
                        if succeeds:
                            outer.run_once(base / "decision.json", "a" * 64, evidence)
                        else:
                            with self.assertRaisesRegex(ValueError, "closure uncertain"):
                                outer.run_once(base / "decision.json", "a" * 64, evidence)
                    self.assertLess(calls.index("wait"), calls.index("rm"))
                    result = json.loads((evidence / "result.json").read_bytes())
                    self.assertIs(result["success"], succeeds)
                    if not succeeds:
                        self.assertIn("Docker wait uncertain", result["closure"][0])

    def test_monitor_failure_does_not_block_owned_docker_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            decision = {key: str(base / key) for key in
                        ("compiler", "source", "f_output", "bundle", "candidate", "out")}
            for path in decision.values():
                Path(path).mkdir()
            decision.update({"name": "g3c4-f-loader-owned",
                             "source_admission_sha256": "a" * 64})
            image = json.dumps([{"Id": outer.IMAGE, "Config": {"Env": []}}]).encode()
            state = {"Status": "exited", "Running": False, "Pid": 0,
                     "OOMKilled": False, "Dead": False, "Error": "", "ExitCode": 0}
            calls = []
            def command(argv, path, timeout=120, monitor=None):
                action = argv[1]
                calls.append((action, monitor is not None))
                if monitor is not None:
                    monitor()
                if action == "image": return image
                if action == "info": return b'["name=rootless"]\n'
                if action == "ps": return b""
                if action == "create":
                    self.assertIsNone(monitor)
                    return ("b" * 64 + "\n").encode()
                if action == "wait": return b"0\n"
                return b"[]"
            def guest(argv, path, timeout=outer.PHASE_LIMIT, monitor=None):
                self.fail("guest attach must not start after created inspect denial")
            def inspect(*args):
                return {"State": {**state, "Status": "created"}} if len(
                    [action for action, _ in calls if action == "inspect"]) == 1 else {"State": state}
            monitor_calls = [0]
            def deny():
                monitor_calls[0] += 1
                if monitor_calls[0] == 5:
                    raise ValueError("aggregate bound")
            with patch.object(outer, "selected", return_value=decision), \
                 patch.object(outer, "checked", side_effect=command), \
                 patch.object(outer, "capture", side_effect=guest), \
                 patch.object(outer, "inspect_owned", side_effect=inspect):
                with self.assertRaisesRegex(ValueError, "outer observation failed"):
                    outer.run_once(base / "decision.json", "a" * 64,
                                   base / "evidence", deny)
            self.assertLess([a for a, _ in calls].index("wait"),
                            [a for a, _ in calls].index("rm"))
            self.assertTrue(all(not watched for action, watched in calls
                                if action in ("wait", "rm")))
            result = json.loads((base / "evidence/result.json").read_bytes())
            self.assertEqual(monitor_calls[0], 5)
            self.assertIn("aggregate bound", result["primary"])
            self.assertEqual(result["closure"], [])

    def test_precreate_monitor_denial_never_calls_docker_create(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            decision = {key: str(base / key) for key in
                        ("compiler", "source", "f_output", "bundle", "candidate", "out")}
            for path in decision.values():
                Path(path).mkdir()
            decision.update({"name": "g3c4-f-loader-owned",
                             "source_admission_sha256": "a" * 64})
            image = json.dumps([{"Id": outer.IMAGE, "Config": {"Env": []}}]).encode()
            actions = []
            monitored = [0]
            def command(argv, path, timeout=120, monitor=None):
                action = argv[1]
                actions.append(action)
                if monitor is not None:
                    monitor()
                if action == "image": return image
                if action == "info": return b'["name=rootless"]\n'
                if action == "ps": return b""
                self.fail(f"unexpected Docker action: {action}")
            def deny():
                monitored[0] += 1
                if monitored[0] == 4:
                    raise ValueError("precreate aggregate bound")
            with patch.object(outer, "selected", return_value=decision), \
                 patch.object(outer, "checked", side_effect=command):
                with self.assertRaisesRegex(ValueError, "outer observation failed"):
                    outer.run_once(base / "decision.json", "a" * 64,
                                   base / "evidence", deny)
            self.assertNotIn("create", actions)
            self.assertEqual(monitored[0], 4)
            self.assertIn("precreate aggregate bound",
                          json.loads((base / "evidence/result.json").read_bytes())["primary"])

    def test_failed_create_exact_cid_recovery_enters_owned_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            decision = {key: str(base / key) for key in
                        ("compiler", "source", "f_output", "bundle", "candidate", "out")}
            for path in decision.values(): Path(path).mkdir()
            decision.update({"name": "g3c4-f-loader-owned",
                             "source_admission_sha256": "a" * 64})
            image = json.dumps([{"Id": outer.IMAGE, "Config": {"Env": []}}]).encode()
            cid = "b" * 64
            state = {"Status": "exited", "Running": False, "Pid": 0,
                     "OOMKilled": False, "Dead": False, "Error": "", "ExitCode": 0}
            actions = []
            def command(argv, path, timeout=120):
                action = argv[1]
                actions.append((action, path.name))
                if action == "image": return image
                if action == "info": return b'["name=rootless"]\n'
                if action == "ps" and path.name == "create-recovery-census":
                    return (cid + " " + decision["name"] + "\n").encode()
                if action == "ps": return b""
                if action == "create": raise ValueError("client receipt lost")
                if action == "wait": return b"0\n"
                return b"[]"
            def authenticated(raw, selected, candidate, *_args):
                self.assertEqual(selected, decision)
                self.assertEqual(candidate, cid)
                return {"State": state}
            with patch.object(outer, "selected", return_value=decision), \
                 patch.object(outer, "checked", side_effect=command), \
                 patch.object(outer, "inspect_owned", side_effect=authenticated):
                with self.assertRaisesRegex(ValueError, "outer observation failed"):
                    outer.run_once(base / "decision.json", "a" * 64,
                                   base / "evidence")
            labels = [name for _, name in actions]
            self.assertLess(labels.index("create-recovery-census"),
                            labels.index("create-recovery-inspect"))
            self.assertLess(labels.index("docker-wait"), labels.index("remove"))
            result = json.loads((base / "evidence/result.json").read_bytes())
            self.assertEqual(result["cid"], cid)
            self.assertIn("client receipt lost", result["primary"])
            self.assertEqual(result["closure"], [])

    def test_failed_create_absent_or_foreign_never_signals_unknown_cid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            decision = {key: str(base / key) for key in
                        ("compiler", "source", "f_output", "bundle", "candidate", "out")}
            for path in decision.values(): Path(path).mkdir()
            decision.update({"name": "g3c4-f-loader-owned",
                             "source_admission_sha256": "a" * 64})
            image = json.dumps([{"Id": outer.IMAGE, "Config": {"Env": []}}]).encode()
            for label, recovery in (("absent", b""),
                                    ("blank", b"\n"),
                                    ("foreign-name", ("b" * 64 + " other\n").encode()),
                                    ("duplicate", ("b" * 64 + " g3c4-f-loader-owned\n").encode() * 2)):
                actions = []
                evidence = base / label
                def command(argv, path, timeout=120):
                    action = argv[1]
                    actions.append(action)
                    if action == "image": return image
                    if action == "info": return b'["name=rootless"]\n'
                    if action == "ps" and path.name == "create-recovery-census":
                        return recovery
                    if action == "ps": return b""
                    if action == "create": raise ValueError("client receipt lost")
                    self.fail(f"unknown container must not be touched: {action}")
                with self.subTest(label=label), \
                     patch.object(outer, "selected", return_value=decision), \
                     patch.object(outer, "checked", side_effect=command):
                    with self.assertRaisesRegex(ValueError, "closure uncertain"):
                        outer.run_once(base / "decision.json", "a" * 64, evidence)
                self.assertNotIn("kill", actions)
                self.assertNotIn("rm", actions)
                result = json.loads((evidence / "result.json").read_bytes())
                self.assertIsNone(result["cid"])
                self.assertTrue(result["closure"])

    def test_failed_create_matching_name_with_foreign_profile_denied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp)
            decision = {"name": "g3c4-f-loader-owned"}
            candidate = "b" * 64
            calls = []
            def command(argv, path, timeout=120):
                calls.append(argv[1])
                if argv[1] == "ps":
                    return (candidate + " " + decision["name"] + "\n").encode()
                if argv[1] == "inspect": return b"[]"
                self.fail("unknown candidate action")
            with patch.object(outer, "checked", side_effect=command), \
                 patch.object(outer, "inspect_owned",
                              side_effect=ValueError("foreign effective profile")):
                with self.assertRaisesRegex(ValueError, "foreign effective profile"):
                    outer.recover_failed_create(decision, b"[]", evidence)
            self.assertEqual(calls, ["ps", "inspect"])
            self.assertFalse((evidence / "create-recovery.json").exists())


if __name__ == "__main__":
    unittest.main()
