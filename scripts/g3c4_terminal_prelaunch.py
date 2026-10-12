#!/usr/bin/env python3
"""One-use private terminal gate launcher. Importing this module never invokes Docker."""

import sys
sys.dont_write_bytecode = True

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import select
import shutil
import signal
import stat
import subprocess
import threading
import time


FILES = frozenset({
    "README.md", "plan.json", "source-closure.sha256", "input-files.sha256",
    "stage-matrix.tsv", "launcher.py", "bootstrap.py", "RECOVERY-LOCK",
    "SHA256SUMS-PREPARED",
})
EXPECTED_HEAD = "e3ebca2d0e7122a8335029a0d94c8c04c8a20d4b"
EXPECTED_TREE = "38963dbcca449c59fcace054292309fd8aa8da44"
CLAIM_LABEL = "g3c4.p2g2.claim"
SOLE_SLOT = Path("/home/gabe/.codex/eshkol-heavy-docker.lock")
GIT = Path("/usr/bin/git")


def clean_git_env() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": "/nonexistent",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}


class Unsealed(RuntimeError):
    """Raw evidence or ownership closure remains uncertain; retain the slot."""


class GateFailure(RuntimeError):
    """A complete, owned command failed with closure still provable."""


class PrelaunchFailure(RuntimeError):
    """A complete predispatch guard failed after one-use claim."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(path: Path) -> None:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"unsafe regular file: {path}")


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, data: dict) -> None:
    payload = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("atomic JSON write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    fsync_dir(path.parent)


def pristine(bundle: Path) -> None:
    if not stat.S_ISDIR(bundle.lstat().st_mode):
        raise ValueError("bundle is not a real directory")
    actual = {entry.name for entry in bundle.iterdir()}
    if actual != FILES:
        raise ValueError(f"bundle has claim/results/unknown members: {sorted(actual ^ FILES)}")
    for name in FILES:
        regular(bundle / name)
    expected = FILES - {"SHA256SUMS-PREPARED"}
    lines = (bundle / "SHA256SUMS-PREPARED").read_text().splitlines()
    seen = set()
    for line in lines:
        value, separator, name = line.partition("  ")
        if separator != "  " or name not in expected or name in seen or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid prepared manifest")
        if digest(bundle / name) != value:
            raise ValueError(f"prepared member changed: {name}")
        seen.add(name)
    if seen != expected:
        raise ValueError("prepared manifest incomplete")


def claim(bundle: Path, nonce: str) -> None:
    pristine(bundle)
    payload = (json.dumps({"nonce": nonce, "one_use": True,
                           "status": "UNSEALED_QUARANTINE"}, sort_keys=True) + "\n").encode()
    fd = os.open(bundle / "CLAIMED", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("one-use claim write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(bundle)


def lock_slot(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("sole-slot is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def git(directory: Path, *args: str) -> str:
    return subprocess.check_output([str(GIT), "-C", str(directory), *args],
                                   text=True, timeout=30, env=clean_git_env()).strip()


def tsv(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        key, sep, value = line.partition("\t")
        if sep != "\t" or not key or not value or key in values:
            raise ValueError(f"malformed TSV: {path}")
        values[key] = value
    return values


def manifest(source: Path, path: Path, count: int) -> list[str]:
    lines = path.read_text().splitlines()
    if len(lines) != count:
        raise ValueError(f"wrong manifest count: {path}")
    result = []
    for line in lines:
        expected, sep, relative = line.partition("  ")
        item = Path(relative)
        if sep != "  " or not re.fullmatch(r"[0-9a-f]{64}", expected) or \
                item.is_absolute() or ".." in item.parts or relative in result:
            raise ValueError(f"unsafe manifest entry: {line}")
        candidate = source / item
        regular(candidate)
        if digest(candidate) != expected:
            raise ValueError(f"source input changed: {relative}")
        result.append(relative)
    return result


def full_tracked_tree(source: Path) -> None:
    regular(source / ".git")
    tracked = set(filter(None, subprocess.check_output(
        [str(GIT), "-C", str(source), "ls-files", "-z"],
        timeout=30, env=clean_git_env()).decode().split("\0")))
    observed = set()
    for root, dirs, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        if any((root_path / name).is_symlink() for name in dirs):
            raise ValueError("source contains directory symlink")
        if root_path == source:
            dirs[:] = [name for name in dirs if name != ".git"]
        elif ".git" in dirs:
            raise ValueError("nested .git directory in exact source")
        for name in files:
            item = root_path / name
            relative = item.relative_to(source).as_posix()
            if relative == ".git":
                continue
            if relative.startswith(".deps/") or item.is_symlink():
                raise ValueError(f"unexpected source file or symlink: {relative}")
            regular(item)
            observed.add(relative)
    if observed != tracked:
        raise ValueError(f"full source file set differs: missing={len(tracked-observed)} extra={len(observed-tracked)}")


def verify_inputs(bundle: Path, plan: dict, *, mountpoint: bool = False) -> None:
    if plan["kind"] != "g3c4-p2g2-terminal-e3e-one-use" or \
            plan["source_head"] != EXPECTED_HEAD or plan["source_tree"] != EXPECTED_TREE or \
            plan["guest_source_mount"] != "/workspace:ro" or \
            plan["guest_deps_mount"] != "/workspace/.deps:ro" or \
            plan["gate_command"] != ["bash", "scripts/test-g3c4-p2g2-terminal-composer.sh", "/out/gate"] or \
            plan["sole_slot_lock"] != str(SOLE_SLOT):
        raise ValueError("recipe authority or exact source target changed")
    source = Path(plan["source_worktree"])
    deps = Path(plan["deps_root"])
    if source.is_symlink() or deps.is_symlink() or not source.is_dir() or not deps.is_dir():
        raise ValueError("source/dependency root differs")
    regular(source / ".git")
    if (source / ".git").read_text() != f"gitdir: {plan['source_gitdir']}\n" or \
            plan["source_git_common"] != "/home/gabe/Documents/ChatGPT/eshkol-transformer/.git":
        raise ValueError("detached worktree Git authority differs")
    if git(source, "rev-parse", "HEAD") != EXPECTED_HEAD or \
            git(source, "rev-parse", "HEAD^{tree}") != EXPECTED_TREE or \
            git(source, "rev-parse", "--git-common-dir") != plan["source_git_common"] or \
            git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("exact source Git state differs")
    if git(source, "branch", "--show-current"):
        raise ValueError("exact source target must remain detached")
    mount = source / ".deps"
    if mount.exists() or mount.is_symlink():
        if not stat.S_ISDIR(mount.lstat().st_mode) or any(mount.iterdir()):
            raise ValueError("source .deps mountpoint unsafe or nonempty")
    elif mountpoint:
        raise ValueError("source .deps mountpoint absent")
    full_tracked_tree(source)
    closure_file = source / "native/g3c4_p2g2_terminal_composer_source_closure.txt"
    if digest(closure_file) != plan["closure_file_sha256"]:
        raise ValueError("source closure changed")
    closure = manifest(source, bundle / "source-closure.sha256", plan["closure_count"])
    if closure != closure_file.read_text().splitlines():
        raise ValueError("source closure order differs")
    inputs = manifest(source, bundle / "input-files.sha256", plan["input_count"])
    if inputs != plan["input_paths"]:
        raise ValueError("extra input manifest differs")
    compiler = deps / "eshkol-src"
    if git(compiler, "rev-parse", "HEAD") != plan["compiler_head"] or \
            git(compiler, "rev-parse", "HEAD^{tree}") != plan["compiler_tree"] or \
            git(compiler, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("compiler source differs")
    build = deps / "eshkol-build"
    for filename, key in plan["compiler_artifacts"].items():
        path = build / filename
        regular(path)
        if digest(path) != key:
            raise ValueError(f"compiler artifact differs: {filename}")
    lock = tsv(source / "toolchain/eshkol.lock")
    provenance = tsv(build / "eshkol-transformer-provenance.tsv")
    if lock["eshkol_commit"] != plan["compiler_head"] or \
            lock["llvm_version"] != plan["llvm_version"] or \
            lock["clang_version"] != plan["llvm_version"] or \
            lock["supported_cc"] != plan["supported_cc"] or \
            lock["supported_cxx"] != plan["supported_cxx"] or \
            provenance["eshkol_commit"] != lock["eshkol_commit"] or \
            provenance["eshkol_source_dir"] != "/workspace/.deps/eshkol-src" or \
            provenance["eshkol_binary_sha256"] != plan["compiler_artifacts"]["eshkol-run"] or \
            provenance["cc_path"] != plan["cc_path"] or provenance["cxx_path"] != plan["cxx_path"] or \
            provenance["cc_version"] != lock["clang_version"] or \
            provenance["cxx_version"] != lock["clang_version"]:
        raise ValueError("toolchain lock/provenance differs")
    cache = (build / "CMakeCache.txt").read_text().splitlines()
    for entry in ("CMAKE_HOME_DIRECTORY:INTERNAL=/workspace/.deps/eshkol-src",
                  "CMAKE_C_COMPILER:FILEPATH=" + plan["cc_path"],
                  "CMAKE_CXX_COMPILER:FILEPATH=" + plan["cxx_path"]):
        if cache.count(entry) != 1:
            raise ValueError("CMake provenance cache differs")
    if int(plan["min_free_disk_bytes"]) < 68719476736 or \
            int(plan["min_mem_available_kib"]) < 4194304 or \
            not 60 <= int(plan["wall_seconds"]) <= 14400:
        raise ValueError("resource/wall floor changed")
    if plan["image_tag"] != "eshkol-checked-promotion-llvm21:20260922" or \
            plan["image_id"] != "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6":
        raise ValueError("image ID changed")
    if plan["closure_count"] != 216 or \
            plan["closure_file_sha256"] != "dd775982fb8e3071a6bb3ea38237c7e9999ccb1986f9bec88af052d1958f0264" or \
            plan["input_paths"] != ["toolchain/eshkol.lock", "eshkol.toml"] or \
            plan["input_count"] != 2 or \
            set(plan["compiler_artifacts"]) != {
                "eshkol-run", "libeshkol-runtime.a", "libeshkol-static.a",
                "eshkol-transformer-provenance.tsv", "CMakeCache.txt"}:
        raise ValueError("exact source/compiler closure plan differs")
    if int(plan["cli_memory_bytes"]) < 536870912 or \
            int(plan["container_memory_bytes"]) < 4294967296 or \
            int(plan["container_cpus"]) < 2:
        raise ValueError("process/container resource caps weakened")
    if plan["git_path"] != str(GIT) or digest(GIT) != plan["git_sha256"]:
        raise ValueError("pinned Git executable changed")
    python = Path(os.path.realpath(sys.executable))
    if plan["python_path"] != str(python) or digest(python) != plan["python_sha256"]:
        raise ValueError("pinned bootstrap Python changed")
    socket = Path(f"/run/user/{os.getuid()}/docker.sock")
    socket_info = socket.lstat()
    if plan["docker_socket"] != str(socket) or socket.is_symlink() or \
            not stat.S_ISSOCK(socket_info.st_mode) or \
            (socket_info.st_uid, socket_info.st_gid,
             stat.S_IMODE(socket_info.st_mode)) != \
            (plan["docker_socket_uid"], plan["docker_socket_gid"],
             plan["docker_socket_mode"]):
        raise ValueError("local Docker daemon socket differs")
    docker = Path(plan["docker_cli_path"])
    regular(docker)
    if digest(docker) != plan["docker_cli_sha256"]:
        raise ValueError("Docker CLI binary differs")


def resource_guard(bundle: Path, plan: dict) -> None:
    if shutil.disk_usage(bundle.parent).free < int(plan["min_free_disk_bytes"]):
        raise ValueError("insufficient available disk")
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        fields[key] = rest.strip().split()
    if int(fields["MemAvailable"][0]) < int(plan["min_mem_available_kib"]):
        raise ValueError("insufficient available memory")


def process_identity(pid: int) -> tuple[int, int] | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = data[data.rindex(")") + 2:].split()
    return pid, int(fields[19])


def signed_process_live(pid: int, starttime: int) -> bool:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    fields = data[data.rindex(")") + 2:].split()
    return int(fields[19]) == starttime and fields[0] not in ("Z", "X")


def named_writers(root: Path) -> list[str]:
    prefix = str(root.resolve()) + "/"
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            fds = list((proc / "fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise Unsealed(f"writer census denied: {proc}") from exc
        for fd in fds:
            try:
                target = os.readlink(fd)
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                raise Unsealed(f"writer FD census denied: {fd}") from exc
            if target.startswith(prefix):
                found.append(f"{proc.name}/{fd.name}:{target}")
    return found


class PrivateCgroup:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def create(cls, nonce: str, memory_max: int, pids_max: int) -> "PrivateCgroup":
        entries = [line for line in Path("/proc/self/cgroup").read_text().splitlines()
                   if line.startswith("0::/")]
        if len(entries) != 1:
            raise Unsealed("unified cgroup v2 membership unavailable")
        parent = Path("/sys/fs/cgroup") / entries[0][4:]
        if parent.is_symlink() or not parent.is_dir():
            raise Unsealed("private cgroup parent unsafe")
        path = parent / f"g3c4-p2g2-{nonce}"
        path.mkdir(mode=0o700)
        result = cls(path)
        try:
            for name, value in (("memory.max", str(memory_max)),
                                ("pids.max", str(pids_max))):
                (path / name).write_text(value)
            if not (path / "cgroup.kill").is_file() or \
                    not (path / "cgroup.events").is_file() or \
                    not (path / "cgroup.procs").is_file():
                raise Unsealed("private cgroup control files absent")
            if not result.empty():
                raise Unsealed("new private cgroup is populated")
        except BaseException:
            if result.empty():
                path.rmdir()
            raise
        return result

    def member(self, pid: int) -> bool:
        try:
            entries = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        except FileNotFoundError:
            return False
        expected = "/" + str(self.path.relative_to("/sys/fs/cgroup"))
        return entries == ["0::" + expected]

    def identity(self) -> dict:
        if self.path.is_symlink():
            raise Unsealed("private cgroup path is a symlink")
        info = self.path.stat()
        if not stat.S_ISDIR(info.st_mode):
            raise Unsealed("private cgroup path is not a directory")
        return {"device": info.st_dev, "inode": info.st_ino,
                "memory_max": (self.path / "memory.max").read_text().strip(),
                "pids_max": (self.path / "pids.max").read_text().strip()}

    def empty(self) -> bool:
        events = (self.path / "cgroup.events").read_text().splitlines()
        population = [line for line in events if line.startswith("populated ")]
        if population not in (["populated 0"], ["populated 1"]):
            raise Unsealed("private cgroup populated state unreadable")
        return population == ["populated 0"]

    def kill(self) -> None:
        (self.path / "cgroup.kill").write_text("1")

    def wait_empty(self, seconds: float = 5.0) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.empty():
                return
            time.sleep(0.02)
        raise Unsealed("private cgroup remains populated after kill")

    def kill_and_wait(self, seconds: float = 5.0) -> None:
        self.kill()
        self.wait_empty(seconds)

    def remove(self) -> None:
        if not self.empty():
            raise Unsealed("cannot remove populated private cgroup")
        self.path.rmdir()


def _pipe_to_file(pipe, path: Path, result: dict, key: str) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            while True:
                block = os.read(pipe.fileno(), 65536)
                if not block:
                    break
                view = memoryview(block)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("raw pipe write made no progress")
                    view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        result[key] = True
    except BaseException as exc:
        result[key] = repr(exc)
    finally:
        pipe.close()


def _ready(fd: int, deadline: float) -> str:
    remaining = max(0.0, deadline - time.monotonic())
    if not select.select([fd], [], [], remaining)[0]:
        raise Unsealed("cgroup bootstrap readiness timed out")
    raw = os.read(fd, 128)
    if not re.fullmatch(rb"[1-9][0-9]*:[1-9][0-9]*\n", raw):
        raise Unsealed("cgroup bootstrap readiness malformed")
    return raw.decode().strip()


def guarded_command(argv: list[str], output: Path, label: str,
                    cgroup: PrivateCgroup, deadline: float,
                    bootstrap: Path, env: dict[str, str]) -> dict:
    """Capture one authenticated direct CLI wait and both raw pipe EOFs."""
    if not argv or not Path(argv[0]).is_absolute() or not re.fullmatch(r"[a-z0-9-]+", label):
        raise ValueError("unguarded CLI argv/label")
    path = output / label
    path.mkdir(mode=0o700)
    atomic_json(path / "command.json", {"argv": argv, "cwd": os.getcwd(),
                                       "env": env})
    ready_r, ready_w = os.pipe2(os.O_CLOEXEC)
    go_r, go_w = os.pipe2(os.O_CLOEXEC)
    proc = None
    readers = []
    pipes = {}
    signed = None
    status = None
    timed_out = False
    failure = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-B", str(bootstrap), str(cgroup.path),
             str(ready_w), str(go_r), "--", *argv],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True,
            pass_fds=(ready_w, go_r), start_new_session=True, env=env)
        os.close(ready_w)
        os.close(go_r)
        ready_w = go_r = -1
        for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            thread = threading.Thread(target=_pipe_to_file,
                                      args=(stream, path / f"{name}.raw", pipes, name),
                                      daemon=True)
            thread.start()
            readers.append(thread)
        signed = process_identity(proc.pid)
        ready = _ready(ready_r, deadline)
        if signed is None or ready != f"{signed[0]}:{signed[1]}" or \
                not cgroup.member(proc.pid):
            raise Unsealed("direct child identity/cgroup bootstrap differs")
        os.write(go_w, b"G")
        os.close(go_w)
        go_w = -1
        try:
            status = proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
    except BaseException as exc:
        failure = repr(exc)
    finally:
        for fd in (ready_r, ready_w, go_r, go_w):
            if fd >= 0:
                os.close(fd)
        if proc is not None:
            try:
                if timed_out or failure or not cgroup.empty():
                    cgroup.kill()
                status = proc.wait(timeout=5)
                cgroup.wait_empty()
            except BaseException as exc:
                failure = f"teardown failed: {exc!r}"
            for thread in readers:
                thread.join(timeout=5)
            if proc.stdout and not proc.stdout.closed and not any(t.is_alive() for t in readers):
                proc.stdout.close()
            if proc.stderr and not proc.stderr.closed and not any(t.is_alive() for t in readers):
                proc.stderr.close()
        atomic_json(path / "wait.json", {
            "pid": proc.pid if proc else None, "starttime": signed[1] if signed else None,
            "exit": status, "timed_out": timed_out, "failure": failure,
            "stdout_eof": pipes.get("stdout") is True,
            "stderr_eof": pipes.get("stderr") is True,
        })
    if proc is None or failure or timed_out or \
            pipes.get("stdout") is not True or pipes.get("stderr") is not True or \
            not cgroup.empty():
        raise Unsealed(f"guarded CLI incomplete: {label}: {failure or 'timeout/pipe/cgroup'}")
    return {"exit": status, "stdout": (path / "stdout.raw").read_bytes(),
            "stderr": (path / "stderr.raw").read_bytes(),
            "pid": signed[0], "starttime": signed[1]}


class Guardian:
    """A separate lock holder survives parent hard kill and quarantines the slot."""

    def __init__(self, pid: int, write_fd: int, identity: tuple[int, int]):
        self.pid, self.write_fd, self.identity = pid, write_fd, identity

    @classmethod
    def start(cls, lock_fd: int, result: Path, cgroup: PrivateCgroup) -> "Guardian":
        parent_signed = process_identity(os.getpid())
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        ack_read, ack_write = os.pipe2(os.O_CLOEXEC)
        pid = os.fork()
        if pid == 0:
            os.close(write_fd)
            os.close(ack_read)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGUSR1, lambda _signal, _frame: os._exit(0))
            try:
                signed = process_identity(os.getpid())
                atomic_json(result / "guardian.json", {
                    "pid": os.getpid(), "starttime": signed[1],
                    "parent_pid": parent_signed[0],
                    "parent_starttime": parent_signed[1],
                    "cgroup": str(cgroup.path),
                    "cgroup_identity": cgroup.identity(), "slot_retained": True,
                })
                os.write(ack_write, f"{signed[0]}:{signed[1]}\n".encode())
                os.close(ack_write)
                message = os.read(read_fd, 1)
                if message == b"S":
                    os._exit(0)
                if (result.parent / "SHA256SUMS-RUN").exists():
                    while True:
                        signal.pause()
                if message != b"Q":
                    try:
                        if cgroup.path.exists():
                            cgroup.kill_and_wait()
                            state = "PARENT_LOST_CGROUP_KILLED"
                        else:
                            state = "PARENT_LOST_CGROUP_REMOVED"
                    except BaseException as exc:
                        state = f"PARENT_LOST_KILL_UNCERTAIN:{exc!r}"
                else:
                    state = "QUARANTINED_BY_PARENT"
                atomic_json(result / "guardian-state.json", {"status": state})
                while True:
                    signal.pause()
            except BaseException as exc:
                try:
                    atomic_json(result / "guardian-state.json",
                                {"status": "GUARDIAN_FAILURE", "error": repr(exc)})
                finally:
                    while True:
                        signal.pause()
        os.close(read_fd)
        os.close(ack_write)
        try:
            if not select.select([ack_read], [], [], 5)[0]:
                raise Unsealed("guardian readiness timed out")
            raw = os.read(ack_read, 128)
            signed = process_identity(pid)
            if signed is None or raw != f"{signed[0]}:{signed[1]}\n".encode():
                raise Unsealed("guardian signed identity differs")
            return cls(pid, write_fd, signed)
        except BaseException:
            os.close(write_fd)
            raise
        finally:
            os.close(ack_read)

    def release(self, seconds: float = 5.0) -> None:
        if process_identity(self.pid) != self.identity:
            raise Unsealed("guardian PID identity changed")
        os.write(self.write_fd, b"S")
        os.close(self.write_fd)
        self.write_fd = -1
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            waited, status = os.waitpid(self.pid, os.WNOHANG)
            if waited == self.pid:
                if status != 0:
                    raise Unsealed("guardian did not release slot cleanly")
                return
            time.sleep(0.02)
        raise Unsealed("guardian release wait timed out")

    def quarantine(self) -> None:
        if process_identity(self.pid) != self.identity:
            raise Unsealed("guardian PID identity changed before quarantine")
        if self.write_fd >= 0:
            os.write(self.write_fd, b"Q")
            os.close(self.write_fd)
            self.write_fd = -1


def require_owned(inspected: dict, cid: str, name: str, nonce: str,
                  image_id: str, command: list[str], source: Path,
                  deps: Path, output: Path, user: str, image_tag: str,
                  memory_bytes: int, cpus: int,
                  image_env: list[str], image_labels: dict) -> None:
    config = inspected.get("Config", {})
    host = inspected.get("HostConfig", {})
    mounts = inspected.get("Mounts", [])
    required_mounts = {
        (str(source), "/workspace", False),
        (str(deps), "/workspace/.deps", False),
        (str(output), "/out", True),
    }
    actual_mounts = {(entry.get("Source"), entry.get("Destination"),
                      entry.get("RW")) for entry in mounts}
    def env_map(values: list[str]) -> dict[str, str]:
        if not isinstance(values, list):
            raise Unsealed("daemon environment absent")
        parsed = {}
        for entry in values:
            if not isinstance(entry, str) or "=" not in entry:
                raise Unsealed("daemon environment malformed")
            key, value = entry.split("=", 1)
            if not key or key in parsed:
                raise Unsealed("daemon environment duplicate key")
            parsed[key] = value
        return parsed
    expected_env = env_map(image_env)
    expected_env.update({
        "ESHKOL_SOURCE_DIR": "/workspace/.deps/eshkol-src",
        "ESHKOL_BUILD_DIR": "/workspace/.deps/eshkol-build",
        "G3C4_TERMINAL_RETAIN_EMITTED": "1",
    })
    actual_env = env_map(config.get("Env"))
    if not isinstance(image_labels, dict):
        raise Unsealed("pinned image labels unavailable")
    expected_labels = dict(image_labels)
    expected_labels[CLAIM_LABEL] = nonce
    if inspected.get("Id") != cid or inspected.get("Name") != "/" + name or \
            inspected.get("Image") != image_id or \
            config.get("Labels") != expected_labels or \
            config.get("Cmd") != command or \
            config.get("Image") != image_tag or \
            config.get("Entrypoint") != ["/usr/bin/env"] or \
            config.get("WorkingDir") != "/workspace" or \
            config.get("User") != user or \
            config.get("Tty") is not False or \
            config.get("OpenStdin") is not False or \
            actual_env != expected_env or \
            host.get("AutoRemove") is not False or \
            host.get("Privileged") is not False or \
            host.get("Devices") not in (None, []) or \
            host.get("Binds") not in (None, []) or \
            host.get("CapAdd") not in (None, []) or \
            host.get("CapDrop") not in (None, []) or \
            host.get("GroupAdd") not in (None, []) or \
            host.get("SecurityOpt") not in (None, []) or \
            host.get("PidMode") not in (None, "") or \
            host.get("IpcMode") not in (None, "", "private") or \
            host.get("UsernsMode") not in (None, "") or \
            host.get("CgroupnsMode") not in (None, "", "private") or \
            host.get("VolumesFrom") not in (None, []) or \
            required_mounts != actual_mounts or \
            host.get("ReadonlyRootfs") is not True or \
            host.get("NetworkMode") != "none" or \
            host.get("Memory") != memory_bytes or \
            host.get("NanoCpus") != cpus * 1_000_000_000 or \
            host.get("PidsLimit") != 512 or \
            host.get("Tmpfs") != {"/tmp": "rw,size=8589934592"}:
        raise Unsealed("daemon-issued container identity/config/mounts differ")


def parsed_json(raw: bytes, context: str):
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Unsealed(f"{context} JSON malformed") from exc


class DockerClient:
    def __init__(self, plan: dict, result: Path, cgroup: PrivateCgroup,
                 deadline: float, bootstrap: Path):
        self.plan, self.result, self.cgroup = plan, result, cgroup
        self.deadline, self.bootstrap = deadline, bootstrap
        existing = [int(path.name.split("-", 1)[0]) for path in result.iterdir()
                    if re.fullmatch(r"[0-9]{3}-[a-z0-9-]+", path.name)]
        self.index = max(existing, default=0)
        self.binary = plan["docker_cli_path"]
        self.image_env = None
        self.image_labels = None
        socket = Path(plan["docker_socket"])
        info = socket.stat()
        if socket.is_symlink() or not stat.S_ISSOCK(info.st_mode) or \
                (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != \
                (plan["docker_socket_uid"], plan["docker_socket_gid"],
                 plan["docker_socket_mode"]):
            raise Unsealed("Docker endpoint socket unsafe")
        self.socket_identity = info.st_dev, info.st_ino
        config = result / "docker-config"
        if config.exists() or config.is_symlink():
            if not config.is_dir() or config.is_symlink() or any(config.iterdir()):
                raise Unsealed("private Docker config directory is not empty")
        else:
            config.mkdir(mode=0o700)
        self.env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C",
                    "HOME": "/nonexistent", "DOCKER_CONFIG": str(config),
                    "DOCKER_HOST": "unix://" + plan["docker_socket"],
                    "DOCKER_CLI_HINTS": "false"}

    def call(self, label: str, *args: str) -> dict:
        info = Path(self.plan["docker_socket"]).stat()
        if (info.st_dev, info.st_ino) != self.socket_identity or \
                not stat.S_ISSOCK(info.st_mode):
            raise Unsealed("Docker endpoint socket changed during gate")
        self.index += 1
        return guarded_command([self.binary, *args], self.result,
                               f"{self.index:03d}-{label}", self.cgroup,
                               self.deadline, self.bootstrap, self.env)

    def image(self) -> None:
        result = self.call("image-inspect", "image", "inspect", self.plan["image_tag"])
        if result["exit"] != 0:
            raise PrelaunchFailure("live image inspect failed")
        observed = parsed_json(result["stdout"], "image inspect")
        if result["exit"] != 0 or not isinstance(observed, list) or len(observed) != 1 or \
                observed[0].get("Id") != self.plan["image_id"]:
            raise PrelaunchFailure("live image ID differs")
        self.image_env = observed[0].get("Config", {}).get("Env")
        self.image_labels = observed[0].get("Config", {}).get("Labels") or {}
        if not isinstance(self.image_env, list):
            raise Unsealed("pinned image environment unavailable")
        if not isinstance(self.image_labels, dict):
            raise Unsealed("pinned image labels unavailable")

    def candidates(self, nonce: str) -> list[str]:
        result = self.call("claim-list", "ps", "-a", "--no-trunc", "--filter",
                           f"label={CLAIM_LABEL}={nonce}", "--format", "{{.ID}}")
        if result["exit"] != 0:
            raise Unsealed("daemon claim listing failed")
        values = result["stdout"].decode().splitlines()
        if len(values) != len(set(values)) or any(not re.fullmatch(r"[0-9a-f]{64}", value)
                                              for value in values):
            raise Unsealed("daemon claim candidate list malformed")
        return values

    def inspect(self, cid: str) -> dict:
        result = self.call("container-inspect", "inspect", cid)
        observed = parsed_json(result["stdout"], "container inspect")
        if result["exit"] != 0 or not isinstance(observed, list) or len(observed) != 1:
            raise Unsealed("daemon container inspect failed")
        return observed[0]

    def create_args(self, nonce: str, source: Path, deps: Path,
                    output: Path, name: str, user: str) -> list[str]:
        return [
            "create", "--name", name, "--label", f"{CLAIM_LABEL}={nonce}",
            "--read-only", "--network", "none", "--pids-limit", "512",
            "--memory", str(self.plan["container_memory_bytes"]),
            "--cpus", str(self.plan["container_cpus"]),
            "--user", user, "--workdir", "/workspace",
            "--entrypoint", "/usr/bin/env", "--tmpfs", "/tmp:rw,size=8589934592",
            "--mount", f"type=bind,src={source},dst=/workspace,readonly",
            "--mount", f"type=bind,src={deps},dst=/workspace/.deps,readonly",
            "--mount", f"type=bind,src={output},dst=/out",
            "--env", "ESHKOL_SOURCE_DIR=/workspace/.deps/eshkol-src",
            "--env", "ESHKOL_BUILD_DIR=/workspace/.deps/eshkol-build",
            "--env", "G3C4_TERMINAL_RETAIN_EMITTED=1",
            self.plan["image_tag"], *self.plan["gate_command"],
        ]

    def verify_candidate(self, cid: str, nonce: str, name: str,
                         source: Path, deps: Path, output: Path,
                         user: str) -> dict:
        observed = self.inspect(cid)
        require_owned(observed, cid, name, nonce, self.plan["image_id"],
                      self.plan["gate_command"], source, deps, output, user,
                      self.plan["image_tag"], int(self.plan["container_memory_bytes"]),
                      int(self.plan["container_cpus"]), self.image_env,
                      self.image_labels)
        return observed

    def remove_owned(self, cid: str, nonce: str, name: str,
                     source: Path, deps: Path, output: Path, user: str) -> None:
        observed = self.verify_candidate(cid, nonce, name, source, deps, output, user)
        if observed.get("State", {}).get("Running") is True:
            killed = self.call("container-kill", "kill", cid)
            if killed["exit"] != 0:
                raise Unsealed("owned container kill failed")
            waited = self.call("container-wait-cleanup", "wait", cid)
            if waited["exit"] != 0:
                raise Unsealed("owned container wait failed")
        removed = self.call("container-rm", "rm", cid)
        if removed["exit"] != 0 or removed["stdout"].decode().strip() != cid:
            raise Unsealed("owned container remove failed")
        if self.candidates(nonce):
            raise Unsealed("owned claim candidate remains after remove")

    def execute(self, nonce: str, source: Path, deps: Path, output: Path) -> int:
        name = f"g3c4-p2g2-{nonce}"
        user = f"{os.getuid()}:{os.getgid()}"
        self.image()
        if self.candidates(nonce):
            raise Unsealed("claim label already has daemon candidates")
        args = self.create_args(nonce, source, deps, output, name, user)
        cid = None
        create_complete = False
        try:
            created = self.call("container-create", *args)
            create_complete = True
            if created["exit"] != 0:
                raise Unsealed("Docker create nonzero; partial daemon side effect uncertain")
            listed = self.candidates(nonce)
            if len(listed) != 1:
                raise Unsealed("claim label lacks a sole daemon-issued CID")
            cid = listed[0]
            if created["stdout"].decode().strip() != cid:
                raise Unsealed("Docker create text differs from daemon-issued CID")
            self.verify_candidate(cid, nonce, name, source, deps, output, user)
            atomic_json(self.result / "create-accepted.json", {"cid": cid, "nonce": nonce,
                                                              "name": name})
            attached = self.call("container-start-attach", "start", "-a", cid)
            observed = self.verify_candidate(cid, nonce, name, source, deps, output, user)
            state = observed.get("State", {})
            guest_exit = state.get("ExitCode")
            if state.get("Running") is not False or \
                    not isinstance(guest_exit, int) or \
                    attached["exit"] not in (0, guest_exit) or \
                    attached["exit"] in (124, 125, 126, 127):
                raise Unsealed("owned container attached wait/state differs")
            return guest_exit
        finally:
            # Even an uncertain create is resolved by the unique daemon label.
            # The caller quarantines if create/attach lifecycle was incomplete.
            candidates = self.candidates(nonce)
            if len(candidates) > 1 or (cid is not None and candidates not in ([], [cid])):
                raise Unsealed("ambiguous claim-label daemon candidates")
            if candidates:
                self.remove_owned(candidates[0], nonce, name, source, deps, output, user)
            if not create_complete:
                raise Unsealed("Docker create completion uncertain; candidate cleanup recorded")


def seal_result(bundle: Path, result: Path, outcome: dict) -> None:
    recovering = (bundle / "RECOVERY-CLAIMED").exists()
    outcome_name = "recovery-outcome.json" if recovering else "outcome.json"
    directories_name = "recovery-directories.json" if recovering else "directories.json"
    atomic_json(result / outcome_name, outcome)
    files = [bundle / "CLAIMED"]
    if (bundle / "RECOVERY-CLAIMED").exists():
        regular(bundle / "RECOVERY-CLAIMED")
        files.append(bundle / "RECOVERY-CLAIMED")
    directory_names = []
    for root, dirs, names in os.walk(result, followlinks=False):
        current = Path(root)
        directory_names.append(str(current.relative_to(bundle)))
        if any((current / name).is_symlink() for name in dirs):
            raise Unsealed("result directory symlink")
        for name in names:
            path = current / name
            regular(path)
            if path != result / directories_name:
                files.append(path)
    atomic_json(result / directories_name, {"directories": sorted(directory_names)})
    files.append(result / directories_name)
    lines = "".join(f"{digest(path)}  {path.relative_to(bundle)}\n"
                    for path in sorted(files))
    seal = bundle / "SHA256SUMS-RUN"
    fd = os.open(seal, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        payload = lines.encode()
        while payload:
            written = os.write(fd, payload)
            if written <= 0:
                raise OSError("run seal write made no progress")
            payload = payload[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(bundle)


def load_plan(bundle: Path) -> dict:
    regular(bundle / "plan.json")
    value = json.loads((bundle / "plan.json").read_text())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("malformed plan")
    return value


def check_source(bundle: Path) -> None:
    pristine(bundle)
    plan = load_plan(bundle)
    verify_inputs(bundle, plan)
    print(f"source-only prelaunch pins PASS: head={EXPECTED_HEAD}; "
          "live image inspection, claim, and compiled gate unrun")


def _result(bundle: Path) -> Path:
    result = bundle / "RESULT"
    result.mkdir(mode=0o700)
    fsync_dir(bundle)
    return result


def run_live(bundle: Path) -> int:
    # No result mutation precedes sole-slot acquisition and atomic claim.
    slot_fd = lock_slot(SOLE_SLOT)
    guardian = None
    cgroup = None
    result = None
    claimed = False
    cgroup_removed = False
    old_alarm = None
    try:
        pristine(bundle)
        plan = load_plan(bundle)
        if plan["sole_slot_lock"] != str(SOLE_SLOT):
            raise ValueError("prepared plan selected a different sole slot")
        if plan.get("launch_enabled") is not True:
            raise PrelaunchFailure("this prepared source-only bundle has no dispatch authority")
        wall = int(plan["wall_seconds"])
        if not 60 <= wall <= 14400:
            raise ValueError("prepared plan wall bound invalid")
        nonce = secrets.token_hex(16)
        claim(bundle, nonce)
        claimed = True
        deadline = time.monotonic() + wall
        def expired(_signum, _frame):
            raise Unsealed("whole-gate wall clock expired")
        old_alarm = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, wall)
        result = _result(bundle)
        atomic_json(result / "lifecycle-state.json", {
            "status": "UNSEALED_QUARANTINE", "nonce": nonce,
            "reason": "one-use claim consumed; closure not yet proven"})
        atomic_json(result / "launch.json", {"nonce": nonce,
                                             "source_head": EXPECTED_HEAD,
                                             "source_tree": EXPECTED_TREE,
                                             "wall_seconds": plan["wall_seconds"]})
        try:
            verify_inputs(bundle, plan)
            source = Path(plan["source_worktree"])
            deps = Path(plan["deps_root"])
            mount = source / ".deps"
            if not mount.exists():
                mount.mkdir(mode=0o700)
            verify_inputs(bundle, plan, mountpoint=True)
            resource_guard(bundle, plan)
            output = result / "out"
            output.mkdir(mode=0o700)
            cgroup = PrivateCgroup.create(nonce,
                                          int(plan["cli_memory_bytes"]), 128)
            guardian = Guardian.start(slot_fd, result, cgroup)
            docker = DockerClient(plan, result, cgroup, deadline,
                                  bundle / "bootstrap.py")
            guest_exit = docker.execute(nonce, source, deps, output)
            if guest_exit != 0:
                raise GateFailure(f"guest gate exited {guest_exit}")
            outcome = {"status": "COMPILED_GATE_EXECUTED_UNADMITTED",
                       "guest_exit": guest_exit,
                       "limitations": ["successful-zero emitted proof HOLD",
                                       "predecessor/deeper command receipts HOLD"]}
            code = 0
        except PrelaunchFailure as exc:
            outcome = {"status": "PRELAUNCH_FAILURE", "error": repr(exc),
                       "one_use_consumed": True}
            code = 1
        except GateFailure as exc:
            outcome = {"status": "GATE_FAILURE", "error": repr(exc),
                       "one_use_consumed": True}
            code = 1
        except Unsealed:
            raise
        except BaseException as exc:
            if guardian is not None:
                raise Unsealed(f"unexpected postdispatch failure: {exc!r}") from exc
            outcome = {"status": "PRELAUNCH_FAILURE", "error": repr(exc),
                       "one_use_consumed": True}
            code = 1
        if cgroup is not None:
            if not cgroup.empty():
                raise Unsealed("private cgroup populated at final seal")
        if named_writers(result):
            raise Unsealed("named result writer remains")
        try:
            seal_result(bundle, result, outcome)
        except BaseException as exc:
            raise Unsealed(f"run seal incomplete: {exc!r}") from exc
        if cgroup is not None:
            try:
                cgroup.remove()
                cgroup_removed = True
            except BaseException as exc:
                raise Unsealed(f"postseal private cgroup removal incomplete: {exc!r}") from exc
        if guardian is not None:
            guardian.release()
            guardian = None
        return code
    except Unsealed as exc:
        if result is not None and not (bundle / "SHA256SUMS-RUN").exists():
            atomic_json(result / "quarantine.json", {
                "status": "UNSEALED_QUARANTINE", "error": repr(exc),
                "one_use_consumed": claimed})
        if guardian is not None and guardian.write_fd >= 0 and \
                not (bundle / "SHA256SUMS-RUN").exists():
            guardian.quarantine()
        elif cgroup is not None and not cgroup_removed and not cgroup.empty():
            # Guardian startup failed before any Docker dispatch. A populated
            # cgroup is still uncertain and must not be silently released.
            cgroup.kill_and_wait()
        return 75
    finally:
        if old_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        os.close(slot_fd)


def verify_prepared_after_claim(bundle: Path) -> None:
    """Authenticate immutable prepared files without requiring pristine runtime state."""
    actual = {entry.name for entry in bundle.iterdir()}
    allowed = FILES | {"CLAIMED", "RESULT", "RECOVERY-CLAIMED", "SHA256SUMS-RUN"}
    if not FILES <= actual or not actual <= allowed:
        raise Unsealed("unexpected recovered bundle member")
    for name in FILES:
        regular(bundle / name)
    lines = (bundle / "SHA256SUMS-PREPARED").read_text().splitlines()
    expected = FILES - {"SHA256SUMS-PREPARED"}
    seen = set()
    for line in lines:
        value, sep, name = line.partition("  ")
        if sep != "  " or name not in expected or name in seen or digest(bundle / name) != value:
            raise Unsealed("prepared seal changed during recovery")
        seen.add(name)
    if seen != expected:
        raise Unsealed("prepared seal incomplete during recovery")


def recover_failure(bundle: Path) -> int:
    """Seal a known-CID hard-kill as failure; ambiguous create stays quarantined."""
    verify_prepared_after_claim(bundle)
    recovery_fd = lock_slot(bundle / "RECOVERY-LOCK")
    try:
        return _recover_failure_locked(bundle)
    finally:
        os.close(recovery_fd)


def _recover_failure_locked(bundle: Path) -> int:
    plan = load_plan(bundle)
    if plan.get("launch_enabled") is not True:
        raise Unsealed("source-only bundle has no recovery authority")
    result = bundle / "RESULT"
    regular(bundle / "CLAIMED")
    claim_data = json.loads((bundle / "CLAIMED").read_text())
    nonce = claim_data["nonce"]
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise Unsealed("claim nonce malformed")
    guardian_data = json.loads((result / "guardian.json").read_text())
    identity = guardian_data["pid"], guardian_data["starttime"]
    if not signed_process_live(*identity):
        raise Unsealed("quarantine guardian identity absent")
    if signed_process_live(guardian_data["parent_pid"],
                           guardian_data["parent_starttime"]):
        raise Unsealed("original launcher parent is still live")
    cgroup = PrivateCgroup(Path(guardian_data["cgroup"]))
    if cgroup.path.name != f"g3c4-p2g2-{nonce}" or \
            not cgroup.path.is_relative_to("/sys/fs/cgroup"):
        raise Unsealed("quarantine cgroup identity differs")
    if cgroup.path.exists() and cgroup.identity() != guardian_data.get("cgroup_identity"):
        raise Unsealed("quarantine cgroup was replaced or its limits changed")
    sealed = bundle / "SHA256SUMS-RUN"
    if sealed.exists():
        verify_run_seal(bundle, result)
        if cgroup.path.exists():
            if not cgroup.empty():
                raise Unsealed("sealed run has newly populated private cgroup")
            cgroup.remove()
        release_orphan_guardian(identity, SOLE_SLOT)
        return 1
    state_file = result / "guardian-state.json"
    regular(state_file)
    state = json.loads(state_file.read_text()).get("status", "")
    if state not in ("QUARANTINED_BY_PARENT", "PARENT_LOST_CGROUP_KILLED",
                     "PARENT_LOST_CGROUP_REMOVED"):
        raise Unsealed("guardian did not authenticate quarantine/parent loss")
    slot = Path(plan["sole_slot_lock"])
    if slot != SOLE_SLOT:
        raise Unsealed("recovery sole slot changed")
    try:
        contender = lock_slot(slot)
    except BlockingIOError:
        pass
    else:
        os.close(contender)
        raise Unsealed("quarantine slot is not held")
    recovery_claim = bundle / "RECOVERY-CLAIMED"
    fd = os.open(recovery_claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600)
    try:
        payload = f"{nonce}\n".encode()
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise Unsealed("recovery claim write incomplete")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_dir(bundle)
    verify_inputs(bundle, plan, mountpoint=True)
    incomplete = []
    for command in sorted(result.glob("[0-9][0-9][0-9]-*")):
        wait = command / "wait.json"
        if not wait.exists():
            incomplete.append(command.name)
            continue
        regular(wait)
        status = json.loads(wait.read_text())
        if status.get("exit") is None or status.get("failure") or \
                status.get("timed_out") or status.get("stdout_eof") is not True or \
                status.get("stderr_eof") is not True:
            incomplete.append(command.name)
    if cgroup.path.exists() and not cgroup.empty():
        cgroup.kill_and_wait()
    accepted = result / "create-accepted.json"
    create_began = bool(list(result.glob("[0-9][0-9][0-9]-container-create")))
    deadline = time.monotonic() + 60
    docker = DockerClient(plan, result, cgroup, deadline, bundle / "bootstrap.py")
    docker.image()
    source = Path(plan["source_worktree"])
    deps = Path(plan["deps_root"])
    output = result / "out"
    name = f"g3c4-p2g2-{nonce}"
    user = f"{os.getuid()}:{os.getgid()}"
    candidates = docker.candidates(nonce)
    if len(candidates) > 1:
        raise Unsealed("multiple daemon candidates in recovery")
    if candidates:
        if accepted.exists():
            accepted_data = json.loads(accepted.read_text())
            if candidates != [accepted_data.get("cid")]:
                raise Unsealed("recovery candidate differs from accepted CID")
        # When create died before its receipt, the daemon's sole label candidate
        # is still inspected against the entire pinned config before removal.
        docker.remove_owned(candidates[0], nonce, name, source, deps, output, user)
    time.sleep(1)
    if docker.candidates(nonce) or (cgroup.path.exists() and not cgroup.empty()) or \
            named_writers(result):
        raise Unsealed("recovery absence/writer proof incomplete")
    if incomplete or (create_began and not accepted.exists()):
        raise Unsealed(f"raw CLI/acceptance evidence incomplete: {incomplete}; retain quarantine")
    if cgroup.path.exists():
        cgroup.remove()
    seal_result(bundle, result, {"status": "RECOVERED_FAILURE_UNADMITTED",
                                 "original_raw_may_be_incomplete": True,
                                 "one_use_consumed": True})
    release_orphan_guardian(identity, SOLE_SLOT)
    return 1


def verify_run_seal(bundle: Path, result: Path) -> None:
    lines = (bundle / "SHA256SUMS-RUN").read_text().splitlines()
    seen = set()
    for line in lines:
        expected, sep, relative = line.partition("  ")
        path = bundle / relative
        components = Path(relative).parts
        if sep != "  " or not re.fullmatch(r"[0-9a-f]{64}", expected) or \
                relative in seen or not components or ".." in components or \
                Path(relative).is_absolute() or \
                relative not in ("CLAIMED", "RECOVERY-CLAIMED") and \
                not relative.startswith("RESULT/"):
            raise Unsealed("run seal malformed")
        ancestor = path.parent
        while ancestor != bundle:
            if ancestor.is_symlink():
                raise Unsealed("run seal symlinked ancestor")
            ancestor = ancestor.parent
        regular(path)
        if digest(path) != expected:
            raise Unsealed("run seal member differs")
        seen.add(relative)
    observed = {"CLAIMED"}
    if (bundle / "RECOVERY-CLAIMED").exists():
        observed.add("RECOVERY-CLAIMED")
    directories = []
    for root, dirs, files in os.walk(result, followlinks=False):
        directories.append(str(Path(root).relative_to(bundle)))
        if any((Path(root) / name).is_symlink() for name in dirs):
            raise Unsealed("run seal directory symlink")
        observed.update(str((Path(root) / name).relative_to(bundle)) for name in files)
    if seen != observed:
        raise Unsealed("run seal member set differs")
    directories_name = ("recovery-directories.json"
                        if (bundle / "RECOVERY-CLAIMED").exists()
                        else "directories.json")
    if json.loads((result / directories_name).read_text()) != \
            {"directories": sorted(directories)}:
        raise Unsealed("run seal directory set differs")


def release_orphan_guardian(identity: tuple[int, int], slot: Path) -> None:
    if not signed_process_live(*identity):
        raise Unsealed("orphan guardian identity changed")
    os.kill(identity[0], signal.SIGUSR1)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not signed_process_live(*identity):
            try:
                fd = lock_slot(slot)
            except BlockingIOError:
                pass
            else:
                os.close(fd)
                return
        time.sleep(0.02)
    raise Unsealed("guardian/sole-slot release unproved")


if __name__ == "__main__":
    bundle = Path(__file__).resolve().parent
    if sys.argv[1:] == ["--check-source"]:
        check_source(bundle)
    elif sys.argv[1:] == ["--launch"]:
        raise SystemExit(run_live(bundle))
    elif sys.argv[1:] == ["--recover"]:
        raise SystemExit(recover_failure(bundle))
    else:
        raise SystemExit("usage: launcher.py --check-source|--launch|--recover")
