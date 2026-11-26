"""Development-only custody for one fresh generated G3-R replay or C2 SAVE gate.

Run: python3 -B -m tests.g3r.one_use_gate PACKAGE EXPECTED_SEAL_SHA256
PACKAGE/preparation contains config.json, inside.sh and a complete seal.json.
The caller pins seal.json independently. Outputs/CLAIM never belong to that seal.
No old preparation, marker, client exit or native-only sanitizer result grants
runtime admission; the retained result still needs independent review.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
from typing import Callable

SCHEMA = "g3r-one-use-v1"
LOCKS = (Path("/tmp/eshkol-transformer-heavy-build.lock"),
         Path("/home/gabe/.codex/eshkol-heavy-docker.lock"))
MEM_KIB, SWAP_KIB, DISK_BYTES = 16 * 1024**2, 8 * 1024**2, 32 * 1024**3
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
SAVE_APP_CALL = re.compile(
    rb'(?m)^[ \t]*%[A-Za-z0-9._-]+[ \t]*=[ \t]*call[ \t]+'
    rb'%eshkol_tagged_value[ \t]+@"checkpoint-save!"\(')
SAVE_NATIVE_CALL = re.compile(
    rb'(?m)^[ \t]*call void @et_e1b_public_c2_checkpoint_save_v1\(')


class Hold(RuntimeError):
    """Inputs, capacity, ownership or evidence cannot establish this gate."""


class ClientInterrupted(RuntimeError):
    def __init__(self, reason: str, returncode: int = 130):
        super().__init__(reason)
        self.returncode = returncode


class ResourceHold(Hold):
    pass


def require(value: object, reason: str) -> None:
    if not value:
        raise Hold(reason)


def confirmed_absence(code: int, data: bytes, error: bytes, name: str) -> bool:
    expected = ("Error response from daemon: No such container: " + name + "\n").encode()
    if code != 1 or error != expected:
        return False
    if not data:
        return True
    try:
        return load_json(data) == []
    except (ValueError, Hold):
        return False


def relative(value: str) -> str:
    require(isinstance(value, str) and value and "\x00" not in value,
            "invalid relative path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and all(x not in ("", ".", "..")
                                      for x in value.split("/")), "path escape")
    return value


def absolute(value: str | Path) -> Path:
    p = Path(value)
    require(p.is_absolute() and str(p) == str(value)
            and not any(x in str(p) for x in (",", "\n", "\r", "\x00"))
            and ".." not in p.parts, "noncanonical mount/path")
    return p


def open_path(path: Path, flags: int = os.O_RDONLY) -> int:
    """Open every ancestor using dirfd and O_NOFOLLOW, including the leaf."""
    path = absolute(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return os.open(path.name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)


def read_file(path: Path) -> bytes:
    fd = open_path(path)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "nonregular file")
        return stream.read()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    fd = open_path(path)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "nonregular file")
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def load_json(data: bytes) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def manifest(root: Path, excluded: tuple[str, ...] = (), *, special: bool = False) -> dict[str, str]:
    """Complete regular-file inventory; never traverse links or special nodes."""
    result = {}
    def visit(fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(fd)):
            rel = prefix + name
            if rel in excluded:
                continue
            relative(rel)
            mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=fd)
                try:
                    visit(child, rel + "/")
                finally:
                    os.close(child)
            else:
                if stat.S_ISREG(mode):
                    result[rel] = sha(root / rel)
                elif special:
                    result[rel] = ("symlink:" + os.readlink(name, dir_fd=fd)
                                   if stat.S_ISLNK(mode) else "mode:" + str(stat.S_IFMT(mode)))
                else:
                    raise Hold("symlink/special inventory child: " + rel)
    fd = open_path(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        visit(fd, "")
    finally:
        os.close(fd)
    return result


def write_new(path: Path, data: bytes) -> None:
    # The enclosing directory was opened/checked without following any links.
    parent = open_path(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(parent)


def mkdir_new(path: Path) -> None:
    parent = open_path(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
    finally:
        os.close(parent)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def git(root: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   text=True, env=env).strip()


def source_pin(root: Path) -> dict:
    """Preparation helper; call on the committed source, not a dirty candidate."""
    fd = open_path(root, os.O_RDONLY | os.O_DIRECTORY)
    os.close(fd)
    metadata = root / ".git"
    if metadata.is_file():
        pointer = read_file(metadata).decode().strip()
        require(pointer.startswith("gitdir: "), "worktree Git pointer")
        target = Path(pointer[8:])
        if not target.is_absolute():
            target = metadata.parent / target
        fd = open_path(target, os.O_RDONLY | os.O_DIRECTORY)
    else:
        fd = open_path(metadata, os.O_RDONLY | os.O_DIRECTORY)
    os.close(fd)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], env=env)
    files = {p.decode(): sha(root / p.decode()) for p in names.split(b"\0") if p}
    common = Path(git(root, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = root / common
    common = common.resolve()
    return {"root": str(root), "head": git(root, "rev-parse", "HEAD"),
            "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "git_common": str(common), "files": files}


def seal_preparation(package: Path) -> str:
    """Seal new preparation inputs only; never reseal or reuse an attempted gate."""
    require(set(os.listdir(package)) == {"preparation"}, "preparation already used")
    prep = package / "preparation"
    files = manifest(prep)
    require("config.json" in files and "inside.sh" in files, "preparation inputs")
    write_new(prep / "seal.json", json_bytes({"schema": SCHEMA, "files": files}))
    return sha(prep / "seal.json")


class Inputs:
    def __init__(self, package: Path, expected: str):
        self.package, self.prep, self.out = absolute(package), package / "preparation", package / "out"
        self.guest = self.out / "guest"
        require(HEX.fullmatch(expected), "independent preparation digest required")
        self.expected = expected
        seal_data = read_file(self.prep / "seal.json")
        require(digest(seal_data) == expected, "preparation seal identity")
        seal = load_json(seal_data)
        require(isinstance(seal, dict) and seal.get("schema") == SCHEMA
                and isinstance(seal.get("files"), dict), "preparation seal schema")
        self.files = seal["files"]
        self.config = load_json(read_file(self.prep / "config.json"))
        c = self.config
        require(isinstance(c, dict) and c.get("schema") == SCHEMA, "config schema")
        require(c.get("gate") in ("replay", "save"), "unsupported gate")
        require(re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,100}", c.get("container_name", "")), "container name")
        self.source = c["source"]
        require(COMMIT.fullmatch(self.source["head"]) and COMMIT.fullmatch(self.source["tree"]), "source identity")
        self.source_root = absolute(self.source["root"])
        self.common = absolute(self.source["git_common"])
        self.deps = c["dependencies"]
        require(isinstance(self.deps, list) and len(self.deps) == 2, "dependency mounts")
        require([d["guest"] for d in self.deps] ==
                ["/workspace/.deps/eshkol-src", "/workspace/.deps/eshkol-build"], "dependency mount paths")
        for d in self.deps:
            absolute(d["root"])
        self.docker = c["docker"]
        for key in ("client", "socket", "root"):
            absolute(self.docker[key])
        require(re.fullmatch(r"sha256:[0-9a-f]{64}", self.docker["image"]), "image digest")
        require(HEX.fullmatch(self.docker["client_sha256"]), "client digest")
        policy = c["evidence"]
        steps = policy["steps"]
        require(isinstance(steps, list) and steps and len(set(steps)) == len(steps)
                and all(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", n) for n in steps), "step inventory")
        needed = {"pins-before", "pins-after", "toolchain"}
        if c["gate"] == "replay":
            needed |= {"normal-run", "repeat-run", "sanitize-run", "normal-oracle", "repeat-oracle", "sanitize-oracle"}
        else:
            needed |= {"source-tests", "compiled-gate"}
        require(needed <= set(steps), "profile steps missing")
        require(needed <= set(policy["empty_stderr"]) <= set(steps), "strict step stderr")
        for name, value in policy.get("artifact_pins", {}).items():
            relative(name)
            require(HEX.fullmatch(value), "artifact digest")

    def verify(self) -> None:
        require(sha(self.prep / "seal.json") == self.expected, "changed preparation seal")
        require(manifest(self.prep, ("seal.json",)) == self.files, "preparation inventory")
        require(source_pin(self.source_root) == self.source, "source manifest/head/tree")
        require(not git(self.source_root, "status", "--porcelain=v1", "--untracked-files=all"), "dirty source")
        require(manifest(self.source_root, (".git",)) == self.source["files"],
                "source inventory includes ignored or untracked inputs")
        fd = open_path(self.common, os.O_RDONLY | os.O_DIRECTORY)
        os.close(fd)
        for dep in self.deps:
            require(manifest(Path(dep["root"])) == dep["files"], "dependency inventory")
        fd = open_path(Path(self.docker["root"]), os.O_RDONLY | os.O_DIRECTORY)
        os.close(fd)
        require(sha(Path(self.docker["client"])) == self.docker["client_sha256"], "Docker client pin")
        # A socket is the sole admitted special input; never dereference a link.
        sock = Path(self.docker["socket"])
        fd = open_path(sock.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            st = os.stat(sock.name, dir_fd=fd, follow_symlinks=False)
            require(stat.S_ISSOCK(st.st_mode) and st.st_uid == os.getuid(), "Docker socket ownership/type")
        finally:
            os.close(fd)

    def command(self) -> list[str]:
        c, d = self.config, self.docker
        args = [d["client"], "--host", "unix://" + d["socket"], "--config", str(self.out / "docker-config")]
        args += ["run", "--sig-proxy=false", "--pull", "never", "--name", c["container_name"],
                 "--label", "codex.source-head=" + self.source["head"],
                 "--label", "codex.evidence-kind=g3r-" + c["gate"],
                 "--label", "codex.preparation-sha256=" + self.expected,
                 "--cidfile", str(self.out / "container.cid"), "--network", "none", "--read-only",
                 "--user", "0:0", "--memory", "12g", "--memory-swap", "12g", "--cpus", "4",
                 "--pids-limit", "256", "--ulimit", "core=0", "--cap-drop", "ALL",
                 "--security-opt", "no-new-privileges"]
        mounts = [(self.prep, "/package"), (self.source_root, str(self.source_root)),
                  (self.common, str(self.common))]
        mounts += [(Path(dep["root"]), dep["guest"]) for dep in self.deps]
        for src, dst in mounts:
            args += ["--mount", f"type=bind,src={src},dst={dst},readonly"]
        args += ["--mount", f"type=bind,src={self.guest},dst=/out", "--workdir", str(self.source_root),
                 "--env", "TMPDIR=/out/tmp", "--env", "XDG_CACHE_HOME=/out/cache",
                 "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=1",
                 d["image"], "/usr/bin/bash", "/package/inside.sh"]
        return args


def resources(inputs: Inputs) -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemAvailable", "SwapFree"):
            mem[key] = int(value.split()[0])
    return {"mem_available_kib": mem.get("MemAvailable", 0),
            "swap_free_kib": mem.get("SwapFree", 0),
            "scratch_free_bytes": shutil.disk_usage(inputs.package).free,
            "docker_free_bytes": shutil.disk_usage(inputs.docker["root"]).free}


def require_resources(sample: dict) -> None:
    for key, floor in (("mem_available_kib", MEM_KIB), ("swap_free_kib", SWAP_KIB),
                       ("scratch_free_bytes", DISK_BYTES), ("docker_free_bytes", DISK_BYTES)):
        if type(sample.get(key)) is not int or sample[key] < floor:
            raise ResourceHold("resource floor: " + key)


@contextmanager
def heavy_locks(paths: tuple[Path, ...]):
    fds = []
    try:
        for path in paths:
            fd = open_path(path, os.O_RDWR | os.O_CREAT)
            fds.append(fd)
            require(stat.S_ISREG(os.fstat(fd).st_mode), "shared lock type")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Hold("shared heavy slot occupied") from exc
        yield
    finally:
        for fd in reversed(fds):
            os.close(fd)


class Process:
    """Only the Docker client process group may be terminated on interruption."""
    def execute(self, argv: list[str], stdout, stderr, timeout: int | None) -> int:
        env = {k: v for k, v in os.environ.items() if not k.startswith("DOCKER_")}
        env.update(PYTHONDONTWRITEBYTECODE="1", LC_ALL="C")
        proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr,
                                env=env, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except BaseException as exc:
            previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in SIGNALS}
            try:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            raise ClientInterrupted("Docker client interrupted: " + repr(exc), proc.returncode) from exc


class Runner:
    def __init__(self, inputs: Inputs, *, process=None, sample: Callable = resources,
                 sleep: Callable = time.sleep, clock: Callable = time.time,
                 locks: tuple[Path, ...] = LOCKS):
        self.i, self.process, self.sample = inputs, process or Process(), sample
        self.sleep, self.clock, self.locks = sleep, clock, locks
        self.counter, self.cid = 0, None
        self.records: dict[str, str] = {}

    def record(self, name: str, args: list[str], timeout: int | None = 30) -> tuple[int, bytes, bytes]:
        out = self.i.out / "host"
        write_new(out / (name + ".argv.json"), json_bytes(args))
        code = 127
        stdout_path, stderr_path = out / (name + ".stdout"), out / (name + ".stderr")
        # Pre-create both streams exclusively; all parent components are checked.
        write_new(stdout_path, b"")
        write_new(stderr_path, b"")
        failure = None
        try:
            with os.fdopen(open_path(stdout_path, os.O_WRONLY), "wb") as stdout, \
                    os.fdopen(open_path(stderr_path, os.O_WRONLY), "wb") as stderr:
                code = self.process.execute(args, stdout, stderr, timeout)
        except BaseException as exc:
            failure = exc
            code = getattr(exc, "returncode", 127)
            write_new(out / (name + ".error"), (repr(exc) + "\n").encode())
        finally:
            write_new(out / (name + ".exit"), (str(code) + "\n").encode())
        for suffix in ("argv.json", "stdout", "stderr", "exit"):
            rel = "host/" + name + "." + suffix
            self.records[rel] = sha(self.i.out / rel)
        if failure is not None:
            raise failure
        return code, read_file(stdout_path), read_file(stderr_path)

    def docker(self, name: str, args: list[str], timeout: int | None = 30) -> tuple[int, bytes, bytes]:
        d = self.i.docker
        return self.record(name, [d["client"], "--host", "unix://" + d["socket"],
                                 "--config", str(self.i.out / "docker-config"), *args], timeout)

    def daemon(self) -> None:
        code, data, _ = self.docker("security", ["info", "--format", "{{json .SecurityOptions}}"])
        require(code == 0 and "name=rootless" in load_json(data), "rootless daemon proof")
        code, data, _ = self.docker("docker-root", ["info", "--format", "{{.DockerRootDir}}"])
        require(code == 0 and data.decode().strip() == self.i.docker["root"], "Docker root identity")
        code, data, _ = self.docker("image", ["image", "inspect", self.i.docker["image"]])
        rows = load_json(data) if code == 0 else []
        require(len(rows) == 1 and rows[0]["Id"] == self.i.docker["image"]
                and rows[0].get("Os") == "linux" and rows[0].get("Architecture") == "amd64", "image identity/platform")

    def inspect(self) -> dict | None:
        self.counter += 1
        code, data, error = self.docker(f"inspect-{self.counter:04d}",
            ["container", "inspect", self.i.config["container_name"]])
        if confirmed_absence(code, data, error, self.i.config["container_name"]):
            return None
        require(code == 0, "daemon inspect unavailable")
        rows = load_json(data)
        require(isinstance(rows, list) and len(rows) == 1, "inspect row count")
        row = rows[0]
        labels = row["Config"]["Labels"]
        require(HEX.fullmatch(row["Id"]) and row["Name"] == "/" + self.i.config["container_name"]
                and row["Image"] == self.i.docker["image"]
                and labels.get("codex.source-head") == self.i.source["head"]
                and labels.get("codex.evidence-kind") == "g3r-" + self.i.config["gate"]
                and labels.get("codex.preparation-sha256") == self.i.expected, "container ownership")
        cid_path = self.i.out / "container.cid"
        if os.path.lexists(cid_path):
            cid = read_file(cid_path).decode().strip()
            require(HEX.fullmatch(cid) and cid == row["Id"], "CID identity")
        elif self.cid is None:
            # A launch may fail before cidfile publication. Such a container can
            # establish custody, but cannot establish success without the CID.
            cid = row["Id"]
        else:
            cid = self.cid
        require(self.cid is None or self.cid == row["Id"], "container ID changed")
        self.cid = cid
        return row

    @staticmethod
    def terminal(row: dict | None) -> bool:
        return row is not None and row["State"]["Running"] is False \
            and row["State"]["Status"] == "exited" \
            and type(row["State"]["ExitCode"]) is int

    def custody(self) -> dict | None:
        """Never kill containers or release locks on unknown daemon/ownership state."""
        absent = 0
        previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in SIGNALS}
        def note(name: str, value: dict) -> None:
            try:
                write_new(self.i.out / name, json_bytes(value))
            except OSError:
                # Failed logging never licenses releasing an uncertain container.
                pass
        try:
            while True:
                try:
                    row = self.inspect()
                    if row is None:
                        absent += 1
                        if absent == 2:
                            note("custody.json", {"status": "confirmed-absent", "observations": 2})
                            return None
                    else:
                        absent = 0
                        if self.terminal(row):
                            note("custody.json", {"status": "owned-terminal", "container": row})
                            return row
                except BaseException as exc:
                    absent = 0
                    note(f"host/custody-error-{self.counter:04d}.json",
                         {"error": repr(exc), "time": self.clock()})
                self.sleep(5 if absent else 30)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def run(self) -> dict:
        require(not os.environ.get("ESHKOL_ALLOW_UNSUPPORTED_HOST"), "unsupported-host override")
        self.i.verify()
        require(set(os.listdir(self.i.package)) == {"preparation"}, "stale/reused package")
        with heavy_locks(self.locks):
            # Revalidate identities and capacity after acquiring BOTH slots.
            self.i.verify()
            require(set(os.listdir(self.i.package)) == {"preparation"}, "one-use package already claimed")
            sample = self.sample(self.i)
            require_resources(sample)
            write_new(self.i.package / "CLAIM.json", json_bytes({"schema": SCHEMA,
                "source_head": self.i.source["head"], "source_tree": self.i.source["tree"],
                "preparation_sha256": self.i.expected, "time": self.clock()}))
            mkdir_new(self.i.out)
            for name in ("host", "docker-config", "guest"):
                mkdir_new(self.i.out / name)
            for name in ("tmp", "cache"):
                mkdir_new(self.i.guest / name)
            write_new(self.i.out / "resources.json", json_bytes(sample))
            write_new(self.i.out / "run.argv.json", json_bytes(self.i.command()))
            attempted, row, failure, status = False, None, None, "FAIL"
            old_handlers = {sig: signal.signal(sig, self.interrupted) for sig in SIGNALS}
            try:
                self.daemon()
                code, data, _ = self.docker("prelaunch-name", ["container", "inspect", self.i.config["container_name"]])
                err = read_file(self.i.out / "host/prelaunch-name.stderr")
                require(confirmed_absence(code, data, err, self.i.config["container_name"]),
                        "container name occupied or unavailable")
                # Once any client launch is attempted, cleanup owns both locks.
                attempted = True
                code, _, _ = self.record("run", self.i.command(), timeout=None)
                require(code == 0, "Docker client failed; no restart")
            except BaseException as exc:
                failure = repr(exc)
            finally:
                if attempted:
                    row = self.custody()
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
            try:
                require(failure is None and attempted, "launch failure: " + str(failure))
                require(self.terminal(row) and row["State"]["ExitCode"] == 0, "container exit identity/result")
                require(read_file(self.i.out / "container.cid").decode().strip() == row["Id"], "missing launch CID")
                self.i.verify()
                require(all(sha(self.i.out / rel) == value for rel, value in self.records.items()),
                        "host log identity changed")
                require(load_json(read_file(self.i.out / "run.argv.json")) == self.i.command(),
                        "recorded launch vector changed")
                artifacts = validate_evidence(self.i)
                status = "PASS"
                result = {"schema": SCHEMA, "status": status,
                          "source_head": self.i.source["head"], "source_tree": self.i.source["tree"],
                          "preparation_sha256": self.i.expected, "image": self.i.docker["image"],
                          "container_id": row["Id"], "artifacts": artifacts,
                          "scope": "generated CPU gate; sanitizer native providers only; independent review required"}
            except BaseException as exc:
                failure = repr(exc)
                result = {"schema": SCHEMA, "status": "FAIL", "failure": failure}
            result.update(source_head=self.i.source["head"], source_tree=self.i.source["tree"],
                          preparation_sha256=self.i.expected, image=self.i.docker["image"],
                          claim_sha256=sha(self.i.package / "CLAIM.json"))
            write_new(self.i.out / "result.json", json_bytes(result))
            # Complete terminal logs/results must exist before the output seal.
            sealed = manifest(self.i.out, special=True)
            write_new(self.i.out / "seal.json", json_bytes({"schema": SCHEMA, "files": sealed}))
            require(status == "PASS", failure or "gate failed")
            return result

    @staticmethod
    def interrupted(signum, _frame):
        raise ClientInterrupted("signal " + str(signum))


def elf(path: Path) -> None:
    data = read_file(path)
    require(len(data) >= 64 and data[:6] == b"\x7fELF\x02\x01"
            and struct.unpack_from("<H", data, 18)[0] == 62
            and struct.unpack_from("<H", data, 16)[0] in (2, 3), "generated x86_64 ELF: " + str(path))


def validate_evidence(inputs: Inputs) -> dict[str, str]:
    out, policy = inputs.guest, inputs.config["evidence"]
    files = manifest(out, special=True)  # Inspect fixture nodes; never dereference.
    allowed_special = ({"authentic-save/companions/link.g3r": "symlink:valid.g3r",
                        "authentic-save/companions/fifo.g3r": "mode:" + str(stat.S_IFIFO)}
                       if inputs.config["gate"] == "save" else {})
    require(all(HEX.fullmatch(value) or allowed_special.get(name) == value
                for name, value in files.items()), "unexpected output link/special node")
    require(read_file(out / "inside.exit") == b"0\n", "guest exit missing/nonzero")
    steps = set(policy["steps"])
    actual = {p for p in files if p.startswith("steps/")}
    expected = {f"steps/{name}.{suffix}" for name in steps
                for suffix in ("argv", "stdout", "stderr", "exit")}
    require(actual == expected, "incomplete/unexpected guest step inventory")
    for name in steps:
        for suffix in ("argv", "stdout", "stderr", "exit"):
            require(f"steps/{name}.{suffix}" in files, "missing step stream/exit")
        require(read_file(out / f"steps/{name}.exit") == b"0\n", "failed guest step")
        require(read_file(out / f"steps/{name}.argv").strip(), "missing exact guest argv")
        if name in policy["empty_stderr"]:
            require(not read_file(out / f"steps/{name}.stderr"), "unexpected guest stderr")
    if inputs.config["gate"] == "replay":
        traces = [read_file(out / f"steps/{mode}-run.stdout") for mode in ("normal", "repeat", "sanitize")]
        require(traces[0] == traces[1] == traces[2], "replay normal/repeat/sanitizer drift")
        for mode in ("normal", "sanitize"):
            elf(out / f"build/{mode}/caller")
            ir = read_file(out / f"build/{mode}/caller.ll")
            require(b"et_g3t_private_replayed_p1_g1_acquire_v1" in ir, "replay emitted IR identity")
        sanitized = read_file(out / "build/sanitize/caller")
        require(b"__asan_init" in sanitized and b"__ubsan_handle_" in sanitized,
                "native sanitizer runtime identity")
        require(re.search(rb"^G3-R replayed P1/G1 private PASS: checks=[1-9][0-9]*$", traces[0], re.M), "replay checks")
        # Recompute the accepted independent oracle against actual trace bytes;
        # never admit the static synthetic oracle controls as runtime evidence.
        from tests.g3r.replay_gate_oracle import check_trace
        check_trace(traces[0])
        for mode in ("normal", "repeat", "sanitize"):
            require(read_file(out / f"steps/{mode}-oracle.stdout").strip()
                    == b"G3-R replayed P1/G1 independent G3-S oracle PASS", "runtime oracle result")
    else:
        base = out / "authentic-save"
        for rel in ("no-save/program", "producer/program", "private-normal/pair-file-reader", "private-sanitize/pair-file-reader"):
            elf(base / rel)
        producer_ir = read_file(base / "producer/program.ll")
        control_ir = read_file(base / "no-save/program.ll")
        require(SAVE_NATIVE_CALL.search(producer_ir), "SAVE native bridge call")
        require(SAVE_APP_CALL.search(producer_ir) and not SAVE_APP_CALL.search(control_ir),
                "SAVE application call/no-SAVE IR identity")
        saved, companion = read_file(base / "roundtrip.c2"), read_file(base / "companions/valid.g3r")
        require(0 < len(saved) <= 1048576 and saved == read_file(base / "seed/valid.c2"), "fresh C2 SAVE bytes")
        require(len(saved) >= 288 and saved[:16] == bytes.fromhex("894553484b54524e53544154450d0a00")
                and struct.unpack_from("<Q", saved, 40)[0] == len(saved)
                and saved[-32:] == hashlib.sha256(b"eshkol-training-state-container-v1\0" + saved[:-32]).digest(), "C2 image format/checksum")
        require(len(companion) == 288 and companion[:8] == b"G3RCV1\0\0"
                and companion[224:256] == hashlib.sha256(saved).digest()
                and companion[256:] == hashlib.sha256(companion[:256]).digest(), "companion C2 digest")
        sanitized = read_file(base / "private-sanitize/pair-file-reader")
        require(b"__asan_init" in sanitized and b"__ubsan_handle_" in sanitized,
                "native sanitizer reader identity")
        before, after = read_file(base / "pair-before-reader.sha256"), read_file(base / "pair-after-reader.sha256")
        require(before == after, "reader pair changed")
        expected = {"/out/authentic-save/roundtrip.c2": digest(saved),
                    "/out/authentic-save/companions/valid.g3r": digest(companion)}
        rows = {}
        for line in before.decode().splitlines():
            value, name = line.split("  ", 1)
            require(name not in rows, "duplicate pair digest")
            rows[name] = value
        require(rows == expected, "pair receipt identity")
        require(read_file(base / "no-save/output-absence.txt").strip()
                == b"absent after compiled LOAD/release without SAVE", "no-SAVE absence evidence")
        require(read_file(base / "no-save/run.stdout").strip() == b"G3-R NO-SAVE CONTROL PASS", "compiled no-SAVE result")
        require(read_file(base / "producer/run.stdout").strip() == b"C2 PUBLIC RUNTIME PASS: 10 checks", "public SAVE result")
        provenance = dict(line.split("\t", 1) for line in read_file(base / "provenance.tsv").decode().splitlines())
        require(provenance.get("source_head") == inputs.source["head"]
                and provenance.get("source_tree") == inputs.source["tree"], "SAVE source identity")
        normal, repeat, sanitized = (read_file(base / name) for name in
            ("private-normal/run.stdout", "private-normal/repeat.stdout", "private-sanitize/run.stdout"))
        require(normal == repeat == sanitized and re.fullmatch(
            rb"G3-R companion file read PASS: [1-9][0-9]* checks\n", normal), "reader modes/result")
        for rel in ("no-save/run.stderr", "producer/run.stderr", "private-normal/run.stderr", "private-normal/repeat.stderr", "private-sanitize/run.stderr"):
            require(not read_file(base / rel), "SAVE runtime stderr")
    for name, value in policy.get("artifact_pins", {}).items():
        require(files.get(name) == value, "artifact identity: " + name)
    return files


def main() -> None:
    require(len(sys.argv) == 3, "usage: python3 -B -m tests.g3r.one_use_gate PACKAGE SEAL_SHA256")
    Runner(Inputs(Path(sys.argv[1]), sys.argv[2])).run()


if __name__ == "__main__":
    try:
        main()
    except ResourceHold as exc:
        print("G3-R one-use DEFERRED: " + str(exc), file=sys.stderr)
        sys.exit(75)
    except Hold as exc:
        print("G3-R one-use HOLD: " + str(exc), file=sys.stderr)
        sys.exit(1)
