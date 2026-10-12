#!/usr/bin/env python3
"""Offline controls for the exact-e3e terminal one-use launcher."""

import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import socket
import stat
import tempfile
import time
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2] / "scripts/g3c4_terminal_prelaunch.py"
BOOTSTRAP = SOURCE.with_name("g3c4_terminal_cgroup_bootstrap.py")
spec = importlib.util.spec_from_file_location("terminal_prelaunch", SOURCE)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def prepared(root: Path, plan: dict) -> Path:
    root.mkdir()
    for name in launch.FILES - {"SHA256SUMS-PREPARED"}:
        (root / name).write_text(json.dumps(plan) if name == "plan.json" else "")
    (root / "SHA256SUMS-PREPARED").write_text("".join(
        f"{launch.digest(root / name)}  {name}\n"
        for name in sorted(launch.FILES - {"SHA256SUMS-PREPARED"})))
    return root


def owned(cid="a" * 64, extra_mount=False):
    mounts = [
        {"Source": "/source", "Destination": "/workspace", "RW": False},
        {"Source": "/deps", "Destination": "/workspace/.deps", "RW": False},
        {"Source": "/out", "Destination": "/out", "RW": True},
    ]
    if extra_mount:
        mounts.append({"Source": "/secret", "Destination": "/secret", "RW": True})
    return {"Id": cid, "Name": "/name", "Image": "sha256:image",
            "Config": {"Labels": {launch.CLAIM_LABEL: "nonce"}, "Cmd": ["bash"],
                       "Image": "tag", "Entrypoint": ["/usr/bin/env"],
                       "WorkingDir": "/workspace", "User": "1000:1000",
                       "Tty": False, "OpenStdin": False,
                       "Env": ["PATH=/usr/bin", "ESHKOL_SOURCE_DIR=/workspace/.deps/eshkol-src",
                               "ESHKOL_BUILD_DIR=/workspace/.deps/eshkol-build",
                               "G3C4_TERMINAL_RETAIN_EMITTED=1"]},
            "Mounts": mounts, "HostConfig": {"AutoRemove": False,
                "Privileged": False, "Devices": [], "Binds": None, "CapAdd": None,
                "ReadonlyRootfs": True, "NetworkMode": "none",
                "Memory": 4 * 1024**3, "NanoCpus": 2_000_000_000,
                "PidsLimit": 512, "Tmpfs": {"/tmp": "rw,size=8589934592"}}}


def check_owned(data):
    launch.require_owned(data, "a" * 64, "name", "nonce", "sha256:image",
                         ["bash"], Path("/source"), Path("/deps"), Path("/out"),
                         "1000:1000", "tag", 4 * 1024**3, 2, ["PATH=/usr/bin"], {})


def socket_plan(endpoint: Path) -> dict:
    info = endpoint.stat()
    return {"docker_cli_path": "/bin/true", "docker_socket": str(endpoint),
            "docker_socket_uid": info.st_uid, "docker_socket_gid": info.st_gid,
            "docker_socket_mode": stat.S_IMODE(info.st_mode)}


class FakeCgroup:
    def __init__(self, root):
        self.path = root
        root.mkdir()
        (root / "cgroup.procs").write_text("")

    def member(self, pid):
        return (self.path / "cgroup.procs").read_text().strip() == str(pid)

    def empty(self):
        raw = (self.path / "cgroup.procs").read_text().strip()
        if not raw:
            return True
        signed = launch.process_identity(int(raw))
        return signed is None or not launch.signed_process_live(*signed)

    def kill(self):
        raw = (self.path / "cgroup.procs").read_text().strip()
        if raw:
            try:
                os.killpg(int(raw), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def wait_empty(self):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.empty():
                return
            time.sleep(.02)
        raise launch.Unsealed("fake cgroup not empty")


class PrelaunchSourceTests(unittest.TestCase):
    def test_exact_detached_source_tree_excludes_worktree_git_file(self):
        target = Path("/home/gabe/.codex/worktrees/g3-p2g2-terminal-e3e-gate-target/eshkol-transformer")
        if target.exists():
            self.assertEqual(launch.git(target, "rev-parse", "HEAD"), launch.EXPECTED_HEAD)
            launch.full_tracked_tree(target)
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp)
            (fake / ".git").symlink_to("/tmp/foreign-git")
            with self.assertRaises(ValueError):
                launch.full_tracked_tree(fake)

    def test_lock_failure_precedes_claim_and_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slot = root / "slot"
            slot.touch()
            bundle = prepared(root / "bundle", {"sole_slot_lock": str(slot),
                                                 "wall_seconds": 60,
                                                 "launch_enabled": True})
            with mock.patch.object(launch, "SOLE_SLOT", slot), \
                 mock.patch.object(launch, "lock_slot", side_effect=BlockingIOError):
                with self.assertRaises(BlockingIOError):
                    launch.run_live(bundle)
            self.assertEqual(set(p.name for p in bundle.iterdir()), launch.FILES)

    def test_claim_is_single_use_and_preflight_failure_seals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slot = root / "slot"
            slot.touch()
            bundle = prepared(root / "bundle", {"sole_slot_lock": str(slot),
                                                 "wall_seconds": 60,
                                                 "launch_enabled": True})
            with mock.patch.object(launch, "SOLE_SLOT", slot), \
                 mock.patch.object(launch, "verify_inputs", side_effect=ValueError("pin mismatch")), \
                 mock.patch.object(launch, "named_writers", return_value=[]):
                self.assertEqual(launch.run_live(bundle), 1)
            self.assertEqual(json.loads((bundle / "RESULT/outcome.json").read_text())["status"],
                             "PRELAUNCH_FAILURE")
            self.assertEqual(json.loads((bundle / "RESULT/lifecycle-state.json").read_text())["status"],
                             "UNSEALED_QUARANTINE")
            launch.verify_run_seal(bundle, bundle / "RESULT")
            with self.assertRaises(ValueError):
                launch.claim(bundle, "other")

    def test_bad_wall_is_rejected_before_one_use_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slot = root / "slot"
            slot.touch()
            bundle = prepared(root / "bundle", {"sole_slot_lock": str(slot),
                                                 "wall_seconds": "bogus",
                                                 "launch_enabled": True})
            with mock.patch.object(launch, "SOLE_SLOT", slot):
                with self.assertRaises(ValueError):
                    launch.run_live(bundle)
            self.assertFalse((bundle / "CLAIMED").exists())

    def test_source_only_bundle_rejects_launch_before_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slot = root / "slot"
            slot.touch()
            bundle = prepared(root / "bundle", {"sole_slot_lock": str(slot),
                                                 "wall_seconds": 60,
                                                 "launch_enabled": False})
            with mock.patch.object(launch, "SOLE_SLOT", slot):
                with self.assertRaises(launch.PrelaunchFailure):
                    launch.run_live(bundle)
            self.assertFalse((bundle / "CLAIMED").exists())

    def test_concurrent_claim_has_one_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = prepared(Path(temp) / "bundle", {})
            ready_r, ready_w = os.pipe()
            go_r, go_w = os.pipe()
            children = []
            try:
                for index in range(2):
                    pid = os.fork()
                    if pid == 0:
                        os.close(ready_r)
                        os.close(go_w)
                        os.write(ready_w, b"R")
                        os.read(go_r, 1)
                        try:
                            launch.claim(bundle, f"child-{index}")
                        except (FileExistsError, ValueError):
                            os._exit(1)
                        os._exit(0)
                    children.append(pid)
                os.close(ready_w)
                os.close(go_r)
                ready = b""
                while len(ready) < 2:
                    ready += os.read(ready_r, 2 - len(ready))
                self.assertEqual(ready, b"RR")
                os.write(go_w, b"GG")
                outcomes = [os.waitpid(pid, 0)[1] for pid in children]
                children.clear()
                self.assertEqual(sorted(os.waitstatus_to_exitcode(value)
                                        for value in outcomes), [0, 1])
                self.assertTrue((bundle / "CLAIMED").is_file())
            finally:
                for pid in children:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    os.waitpid(pid, 0)
                for fd in (ready_r, ready_w, go_r, go_w):
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def test_ownership_requires_exact_config_and_mounts(self):
        check_owned(owned())
        changed_cases = [owned(extra_mount=True)]
        for key, value in (("Entrypoint", ["/bin/bash"]),
                           ("WorkingDir", "/tmp"), ("User", "0:0")):
            changed = owned()
            changed["Config"][key] = value
            changed_cases.append(changed)
        changed = owned()
        changed["HostConfig"]["AutoRemove"] = True
        changed_cases.append(changed)
        for key, value in (("PidMode", "host"), ("IpcMode", "host"),
                           ("SecurityOpt", ["seccomp=unconfined"]),
                           ("UsernsMode", "host"), ("CgroupnsMode", "host"),
                           ("GroupAdd", ["0"])):
            changed = owned()
            changed["HostConfig"][key] = value
            changed_cases.append(changed)
        for index in (1, 2, 3):
            changed = owned()
            changed["Config"]["Env"][index] += "-changed"
            changed_cases.append(changed)
        changed = owned()
        changed["Config"]["Env"].append("EXTRA=1")
        changed_cases.append(changed)
        changed = owned()
        changed["Config"]["Labels"]["EXTRA"] = "1"
        changed_cases.append(changed)
        changed = owned()
        changed["Config"]["Tty"] = True
        changed_cases.append(changed)
        for changed in changed_cases:
            with self.assertRaises(launch.Unsealed):
                check_owned(changed)

    def test_guards_capture_wait_and_two_raw_eofs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            group = FakeCgroup(root / "group")
            out = root / "out"
            out.mkdir()
            result = launch.guarded_command(["/bin/bash", "-c", "printf a; printf b >&2"],
                                            out, "one", group, time.monotonic() + 5,
                                            BOOTSTRAP, {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
            self.assertEqual(result["exit"], 0)
            self.assertEqual(result["stdout"], b"a")
            self.assertEqual(result["stderr"], b"b")
            wait = json.loads((out / "one/wait.json").read_text())
            self.assertTrue(wait["stdout_eof"] and wait["stderr_eof"])
            with self.assertRaises(launch.Unsealed):
                launch.guarded_command(["/bin/bash", "-c", "sleep 10"], out,
                                       "timeout", group, time.monotonic() + .2,
                                       BOOTSTRAP, {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
            wait = json.loads((out / "timeout/wait.json").read_text())
            self.assertTrue(wait["timed_out"])
            self.assertTrue(group.empty())

    def test_ambient_docker_and_git_overrides_are_not_forwarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            endpoint = root / "daemon.sock"
            sock = socket.socket(socket.AF_UNIX)
            sock.bind(str(endpoint))
            result = root / "RESULT"
            result.mkdir()
            plan = socket_plan(endpoint)
            try:
                client = launch.DockerClient(plan, result, None, time.monotonic() + 5,
                                             BOOTSTRAP)
                with mock.patch.dict(os.environ, {"DOCKER_HOST": "tcp://evil:2375",
                                                   "DOCKER_CONTEXT": "evil",
                                                   "GIT_DIR": "/tmp/evil",
                                                   "PYTHONPATH": "/tmp/evil"}), \
                     mock.patch.object(launch, "guarded_command", return_value={"exit": 0}) as call:
                    client.call("check", "version")
                    sent = call.call_args.args
                    self.assertEqual(sent[0], ["/bin/true", "version"])
                    env = sent[-1]
                    self.assertEqual(env["DOCKER_HOST"], "unix://" + str(endpoint))
                    self.assertNotIn("DOCKER_CONTEXT", env)
                    self.assertNotIn("PYTHONPATH", env)
                    self.assertNotIn("GIT_DIR", launch.clean_git_env())
            finally:
                sock.close()

    def test_daemon_label_owns_cid_not_create_stdout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            endpoint = root / "daemon.sock"
            sock = socket.socket(socket.AF_UNIX)
            sock.bind(str(endpoint))
            result = root / "RESULT"
            result.mkdir()
            plan = socket_plan(endpoint) | {"image_tag": "tag", "gate_command": ["bash"],
                                            "container_memory_bytes": 4 * 1024**3,
                                            "container_cpus": 2}
            cid = "a" * 64
            try:
                client = launch.DockerClient(plan, result, None,
                                             time.monotonic() + 5, BOOTSTRAP)
                client.image = mock.Mock()
                client.candidates = mock.Mock(side_effect=[[], [cid], [cid]])
                client.call = mock.Mock(return_value={"exit": 0, "stdout": b"b" * 64})
                client.remove_owned = mock.Mock()
                client.verify_candidate = mock.Mock()
                with self.assertRaises(launch.Unsealed):
                    client.execute("nonce", Path("/source"), Path("/deps"),
                                   result / "out")
                client.remove_owned.assert_called_once()
                self.assertEqual(client.remove_owned.call_args.args[0], cid)
                self.assertFalse(any(call.args[0] == "container-start-attach"
                                     for call in client.call.call_args_list))
            finally:
                sock.close()

    def test_guardian_retains_slot_on_parent_pipe_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slot = root / "slot"
            slot.touch()
            fd = launch.lock_slot(slot)
            result = root / "RESULT"
            result.mkdir()
            group_path = root / "absent"
            group_path.mkdir()
            (group_path / "memory.max").write_text("100")
            (group_path / "pids.max").write_text("10")
            fake = launch.PrivateCgroup(group_path)
            guardian = launch.Guardian.start(fd, result, fake)
            try:
                (group_path / "memory.max").unlink()
                (group_path / "pids.max").unlink()
                group_path.rmdir()
                os.close(guardian.write_fd)
                guardian.write_fd = -1
                os.close(fd)
                for _ in range(100):
                    if (result / "guardian-state.json").exists():
                        break
                    time.sleep(.02)
                self.assertEqual(json.loads((result / "guardian-state.json").read_text())["status"],
                                 "PARENT_LOST_CGROUP_REMOVED")
                with self.assertRaises(BlockingIOError):
                    launch.lock_slot(slot)
                launch.release_orphan_guardian(guardian.identity, slot)
            finally:
                if launch.signed_process_live(*guardian.identity):
                    os.kill(guardian.pid, signal.SIGUSR1)
                os.waitpid(guardian.pid, 0)

    def test_guardian_retains_slot_after_real_parent_sigkill(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slot = root / "slot"
            slot.touch()
            result = root / "RESULT"
            result.mkdir()
            read_fd, write_fd = os.pipe()
            child = os.fork()
            if child == 0:
                os.close(read_fd)
                try:
                    fd = launch.lock_slot(slot)
                    group_path = root / "group"
                    group_path.mkdir()
                    (group_path / "memory.max").write_text("100")
                    (group_path / "pids.max").write_text("10")
                    guardian = launch.Guardian.start(fd, result,
                                                     launch.PrivateCgroup(group_path))
                    (group_path / "memory.max").unlink()
                    (group_path / "pids.max").unlink()
                    group_path.rmdir()
                    os.write(write_fd, f"{guardian.pid}:{guardian.identity[1]}\n".encode())
                    while True:
                        signal.pause()
                except BaseException:
                    os._exit(2)
            os.close(write_fd)
            guardian_identity = None
            try:
                self.assertTrue(select.select([read_fd], [], [], 2)[0])
                raw = os.read(read_fd, 128).decode().strip()
                guardian_identity = tuple(map(int, raw.split(":")))
                self.assertEqual(len(guardian_identity), 2)
                os.kill(child, signal.SIGKILL)
                os.waitpid(child, 0)
                child = -1
                for _ in range(100):
                    if (result / "guardian-state.json").exists():
                        break
                    time.sleep(.02)
                self.assertEqual(json.loads((result / "guardian-state.json").read_text())["status"],
                                 "PARENT_LOST_CGROUP_REMOVED")
                with self.assertRaises(BlockingIOError):
                    launch.lock_slot(slot)
                launch.release_orphan_guardian(guardian_identity, slot)
            finally:
                os.close(read_fd)
                if child != -1:
                    os.kill(child, signal.SIGKILL)
                    os.waitpid(child, 0)
                if guardian_identity and launch.signed_process_live(*guardian_identity):
                    os.kill(guardian_identity[0], signal.SIGUSR1)

    def test_recovery_seal_rejects_traversal_or_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = root / "RESULT"
            result.mkdir()
            (root / "CLAIMED").write_text("claim")
            launch.seal_result(root, result, {"status": "test"})
            launch.verify_run_seal(root, result)
            (result / "outcome.json").write_text("altered")
            with self.assertRaises(launch.Unsealed):
                launch.verify_run_seal(root, result)
            (root / "SHA256SUMS-RUN").write_text(
                f"{launch.digest(root / 'CLAIMED')}  RESULT/../CLAIMED\n")
            with self.assertRaises(launch.Unsealed):
                launch.verify_run_seal(root, result)

    def test_recovery_refuses_live_signed_parent_before_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = prepared(root / "bundle", {})
            (bundle / "CLAIMED").write_text(json.dumps({"nonce": "a" * 32}))
            result = bundle / "RESULT"
            result.mkdir()
            pid, started = launch.process_identity(os.getpid())
            (result / "guardian.json").write_text(json.dumps({
                "pid": pid, "starttime": started,
                "parent_pid": pid, "parent_starttime": started,
                "cgroup": "/sys/fs/cgroup/g3c4-p2g2-" + "a" * 32}))
            with mock.patch.object(launch, "load_plan", return_value={"launch_enabled": True}):
                with self.assertRaisesRegex(launch.Unsealed, "parent is still live"):
                    launch.recover_failure(bundle)
            self.assertFalse((bundle / "RECOVERY-CLAIMED").exists())

    def test_recovery_seal_preserves_partial_original_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            result = bundle / "RESULT"
            result.mkdir()
            (bundle / "CLAIMED").write_text("claim")
            (bundle / "RECOVERY-CLAIMED").write_text("recovery")
            (result / "outcome.json").write_text("original partial outcome")
            (result / "directories.json").write_text("original partial directories")
            launch.seal_result(bundle, result, {"status": "RECOVERED_FAILURE_UNADMITTED"})
            self.assertEqual((result / "outcome.json").read_text(),
                             "original partial outcome")
            self.assertEqual((result / "directories.json").read_text(),
                             "original partial directories")
            launch.verify_run_seal(bundle, result)


if __name__ == "__main__":
    unittest.main()
