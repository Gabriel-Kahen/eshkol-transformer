"""Inactive, test-only outer preparation for a copied-F loader observation.

ROOT must seal an external one-shot wrapper that pins an exact reviewed
decision SHA and invokes run_once. This module has no runnable CLI or decision
pin of its own and never admits the failed F gate or a public G2 artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time
from typing import Callable

from tests.g3c4 import f_artifact_attest as attest
from tests.g3c4 import f_image_loader_collect as collector


IMAGE = "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6"
DOCKER = Path("/home/gabe/bin/docker")
PHASE_LIMIT = 600
STREAM_LIMIT = 16 * 1024 * 1024
FSIZE_LIMIT = 128 * 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ordinary(path: Path) -> bytes:
    return attest.ordinary(path)


def directory(path: Path) -> None:
    require(path.is_absolute(), "relative selected directory")
    mode = path.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode) and
            path.resolve(strict=True) == path,
            f"selected directory is not ordinary: {path}")


def source_state(source: Path, admission: dict) -> None:
    head, tree = attest.git(source, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    require((head, tree) == (admission["head"], admission["tree"]) and
            attest.git(source, "status", "--porcelain=v1", "--untracked-files=all") == "",
            "source commit/tree/cleanliness drift")
    listing = subprocess.run(["git", "--no-optional-locks", "-C", str(source),
                              "ls-files", "-s", "-z"], check=True,
                             capture_output=True, timeout=30).stdout
    rows = []
    for entry in listing.rstrip(b"\0").split(b"\0"):
        metadata, name = entry.split(b"\t", 1)
        mode, _, stage = metadata.decode().split()
        require(mode in ("100644", "100755") and stage == "0" and
                b".." not in name.split(b"/"), "tracked source mode/path")
        rows.append(f"{name.decode()}\t{mode}\t"
                    f"{sha(ordinary(source / name.decode()))}\n")
    require(len(rows) == admission["tracked_file_count"] and
            sha("".join(rows).encode()) == admission["tracked_manifest_sha256"],
            "tracked source byte closure drift")


def selected(decision_path: Path, expected_sha: str, *, fresh: bool) -> dict:
    """Caller must be a separately admitted ROOT wrapper with an exact pin."""
    require(type(expected_sha) is str and HEX64.fullmatch(expected_sha),
            "exact external ROOT decision pin missing")
    require(decision_path.is_absolute() and
            decision_path.resolve(strict=True) == decision_path,
            "ROOT decision path is not canonical")
    data = ordinary(decision_path)
    require(sha(data) == expected_sha, "ROOT decision changed")
    decision = json.loads(data)
    keys = {"format", "name", "image", "docker_sha256", "socket_dev",
            "socket_ino", "compiler", "source",
            "f_output", "bundle", "candidate", "out", "source_admission_sha256",
            "tool_pins_sha256", "runtime_pins_sha256", "offline_review_sha256"}
    require(type(decision) is dict and set(decision) == keys and
            decision["format"] == "g3c4-F-loader-outer-decision-v1" and
            decision["image"] == IMAGE and
            re.fullmatch(r"g3c4-f-loader-[a-z0-9-]+", decision["name"]) is not None,
            "ROOT decision schema")
    for key in ("docker_sha256", "source_admission_sha256", "tool_pins_sha256",
                "runtime_pins_sha256", "offline_review_sha256"):
        require(type(decision[key]) is str and HEX64.fullmatch(decision[key]),
                f"decision SHA malformed: {key}")
    require(all(type(decision[key]) is int and decision[key] > 0
                for key in ("socket_dev", "socket_ino")),
            "rootless socket decision malformed")
    require(decision["tool_pins_sha256"] == collector.TOOL_PINS_SHA and
            decision["runtime_pins_sha256"] == collector.RUNTIME33_SHA and
            decision["offline_review_sha256"] == attest.OFFLINE_ROOT_REVIEW_SHA,
            "accepted pin mismatch")
    for key in ("compiler", "source", "f_output", "bundle", "candidate", "out"):
        require(type(decision[key]) is str, f"decision path malformed: {key}")
        directory(Path(decision[key]))
    roots = [Path(decision[key]).resolve() for key in
             ("compiler", "source", "f_output", "bundle", "candidate", "out")]
    require(len(set(roots)) == len(roots) and
            all(not (a in b.parents or b in a.parents) for i, a in enumerate(roots)
                for b in roots[i + 1:]), "selected mounts overlap")
    require(sha(ordinary(DOCKER)) == decision["docker_sha256"],
            "selected Docker client changed")
    socket_path = Path(f"/run/user/{os.getuid()}/docker.sock")
    socket_state = socket_path.lstat()
    require(stat.S_ISSOCK(socket_state.st_mode) and
            socket_state.st_uid == os.getuid() and
            (socket_state.st_dev, socket_state.st_ino) ==
            (decision["socket_dev"], decision["socket_ino"]),
            "rootless Docker socket identity")
    compiler = Path(decision["compiler"])
    head, tree = attest.git(compiler, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    require((head, tree) == (attest.COMPILER_HEAD, attest.COMPILER_TREE) and
            attest.git(compiler, "status", "--porcelain=v1", "--untracked-files=all") == "",
            "compiler source identity/cleanliness")
    source = Path(decision["source"])
    # The external source admission authenticates the exact head/tree/contents.
    bundle = Path(decision["bundle"])
    admission = attest.source_admission(
        bundle / "public-g2-source-admission.json",
        decision["source_admission_sha256"],
        tuple(root for root in roots if root != bundle.resolve()),
        expected_path=bundle / "public-g2-source-admission.json",
        outer_bundle_root=bundle)
    source_state(source, admission)
    require(
            sha(ordinary(bundle / "accepted-tools.tsv")) == collector.TOOL_PINS_SHA and
            sha(ordinary(bundle / "accepted-runtime33.tsv")) == collector.RUNTIME33_SHA and
            sha(ordinary(bundle / "offline/SHA256SUMS-ROOT-REVIEW")) ==
            attest.OFFLINE_ROOT_REVIEW_SHA,
            "external bundle input drift")
    attest.candidate_lock(bundle / "public-g2-6803.lock",
                          ordinary(source / "toolchain/eshkol.lock"),
                          bundle / "public-g2-6803.lock")
    attest.original_output(Path(decision["f_output"]))
    if fresh:
        for key in ("candidate", "out"):
            require(not any(Path(decision[key]).iterdir()),
                    f"fresh {key} is not empty")
    return decision


def create_argv(decision: dict) -> list[str]:
    name = decision["name"]
    argv = [str(DOCKER), "create", "--name", name,
            "--label", f"eshkol.gate={name}", "--read-only", "--network", "none",
            "--memory", "4g", "--memory-swap", "4g", "--pids-limit", "128",
            "--cpus", "2", "--ulimit", "fsize=134217728:134217728",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=268435456,mode=1777",
            "--user", "0:0", "--workdir", "/source"]
    for key, target, readonly in (
            ("compiler", "/workspace", True), ("source", "/source", True),
            ("f_output", "/f-output", True), ("bundle", "/bundle", True),
            ("candidate", "/stage", False), ("out", "/out", False)):
        spec = f"type=bind,src={decision[key]},dst={target}"
        argv += ["--mount", spec + (",readonly" if readonly else "")]
    argv += ["--env", "PATH=/usr/bin:/bin", "--env", "LC_ALL=C",
             "--env", "TZ=UTC", "--env", "PYTHONDONTWRITEBYTECODE=1",
             "--env", "ESHKOL_ALLOW_UNSUPPORTED_HOST=0",
             "--env", "GIT_CONFIG_COUNT=2", "--env", "GIT_CONFIG_KEY_0=safe.directory",
             "--env", "GIT_CONFIG_VALUE_0=/workspace",
             "--env", "GIT_CONFIG_KEY_1=safe.directory",
             "--env", "GIT_CONFIG_VALUE_1=/source",
             "--entrypoint", "/usr/bin/python3.10", IMAGE, "-B", "-m",
             "tests.g3c4.f_loader_guest", "--source-admission-sha256",
             decision["source_admission_sha256"]]
    return argv


def capture(argv: list[str], output: Path, timeout: int = PHASE_LIMIT,
            monitor: Callable[[], None] | None = None) -> dict:
    """Owned bounded subprocess; never use Popen.poll before session closure."""
    require(1 <= timeout <= PHASE_LIMIT and not output.exists(), "phase output exists")
    output.mkdir(mode=0o700)
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC",
                   "DOCKER_HOST": f"unix:///run/user/{os.getuid()}/docker.sock"}
    attest.exclusive_bytes(output / "argv.json", canonical({
        "argv": argv, "cwd": "/", "env": environment}))
    streams = [bytearray(), bytearray()]
    seen = [0, 0]
    selector = selectors.DefaultSelector()
    child = leader = None
    wait_status = returncode = None
    quiescent = False
    failure = ""
    cleanup = []
    started = time.monotonic()
    uid = os.getuid()

    def drain() -> None:
        nonlocal failure
        for key, _ in selector.select(0.05):
            channel = key.data
            raw = os.read(key.fd, 65536)
            if not raw:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            seen[channel] += len(raw)
            room = max(0, STREAM_LIMIT - len(streams[channel]))
            streams[channel].extend(raw[:room])
            if len(raw) > room:
                failure = failure or f"stream {channel} overflow"

    try:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 start_new_session=True, close_fds=True,
                                 cwd="/", env=environment)
        leader = collector.identity(child.pid)
        require(leader[1:3] == (child.pid, child.pid) and
                (Path("/proc") / str(child.pid)).stat().st_uid == uid,
                "child ownership uncertain")
        for index, pipe in enumerate((child.stdout, child.stderr)):
            assert pipe is not None
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, index)
        term_at = kill_at = None
        while True:
            exited = os.waitid(os.P_PID, child.pid,
                               os.WEXITED | os.WNOHANG | os.WNOWAIT)
            live = collector.members(child.pid, uid)
            if monitor is not None and not failure:
                try:
                    monitor()
                except BaseException as error:
                    failure = f"monitor denied: {type(error).__name__}: {error}"
            if (exited and not selector.get_map() and
                    all(row[0] == child.pid for row in live)):
                quiescent = True
                break
            now = time.monotonic()
            if exited and any(row[0] != child.pid for row in live):
                failure = failure or "unexpected descendant"
            if now - started >= timeout:
                failure = failure or "phase timeout"
            if failure and term_at is None:
                collector.signal_owned(leader, uid, signal.SIGTERM)
                cleanup.append("owned TERM")
                term_at = now
            elif term_at is not None and now - term_at >= 1 and kill_at is None:
                collector.signal_owned(leader, uid, signal.SIGKILL)
                cleanup.append("owned KILL")
                kill_at = now
            elif kill_at is not None and now - kill_at >= 5:
                raise RuntimeError("writers did not close")
            drain()
    except BaseException as error:
        failure = failure or f"{type(error).__name__}: {error}"
    finally:
        if child is not None and not quiescent:
            try:
                require(leader is not None, "leader unavailable")
                collector.signal_owned(leader, uid, signal.SIGKILL)
                cleanup.append("cleanup owned KILL")
                until = time.monotonic() + 5
                while time.monotonic() < until:
                    drain()
                    exited = os.waitid(os.P_PID, child.pid,
                                       os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    live = collector.members(child.pid, uid)
                    if (exited and not selector.get_map() and
                            all(row[0] == child.pid for row in live)):
                        quiescent = True
                        break
                if not quiescent:
                    cleanup.append("writer closure uncertain")
            except BaseException as error:
                cleanup.append(f"cleanup uncertain: {type(error).__name__}: {error}")
        if child is not None:
            until = time.monotonic() + 5
            while time.monotonic() < until and wait_status is None:
                try:
                    pid, status, _ = os.wait4(child.pid, os.WNOHANG)
                except InterruptedError:
                    continue
                except BaseException as error:
                    cleanup.append(f"wait uncertain: {type(error).__name__}: {error}")
                    break
                if pid:
                    wait_status = status
                    returncode = os.waitstatus_to_exitcode(status)
                    child.returncode = returncode
                else:
                    time.sleep(0.05)
            if wait_status is None:
                cleanup.append("wait uncertain")
        for key in list(selector.get_map().values()):
            try:
                selector.unregister(key.fileobj)
                key.fileobj.close()
            except BaseException as error:
                cleanup.append(f"stream uncertain: {type(error).__name__}: {error}")
        try:
            selector.close()
        except BaseException as error:
            cleanup.append(f"selector uncertain: {type(error).__name__}: {error}")
    if not quiescent or wait_status is None or any("uncertain" in item for item in cleanup):
        failure = failure or "physical closure uncertain"
    receipt = {"argv": argv, "cwd": "/", "env": environment,
               "leader": leader, "wait_status": wait_status,
               "returncode": returncode, "quiescent": quiescent,
               "seen": seen, "failure": failure, "cleanup": cleanup,
               "elapsed_seconds": time.monotonic() - started}
    attest.exclusive_bytes(output / "stdout.raw", bytes(streams[0]))
    attest.exclusive_bytes(output / "stderr.raw", bytes(streams[1]))
    attest.exclusive_bytes(output / "status.json", canonical(receipt))
    return receipt


def checked(argv: list[str], path: Path, timeout: int = 120,
            monitor: Callable[[], None] | None = None) -> bytes:
    receipt = capture(argv, path, timeout, monitor)
    require(receipt["returncode"] == 0 and receipt["quiescent"] and
            not receipt["failure"] and
            ordinary(path / "stderr.raw") == b"", "outer command failed")
    return ordinary(path / "stdout.raw")


def env_map(rows: list[str]) -> dict[str, str]:
    require(type(rows) is list and all(type(row) is str and "=" in row
                                       for row in rows), "effective env malformed")
    result = {}
    for row in rows:
        key, value = row.split("=", 1)
        require(key and key not in result, "effective env duplicate")
        result[key] = value
    return result


def expected_env(image_raw: bytes) -> dict[str, str]:
    image = json.loads(image_raw)
    require(type(image) is list and len(image) == 1 and
            image[0]["Id"] == IMAGE, "image env source mismatch")
    result = env_map(image[0]["Config"]["Env"])
    result.update({"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC",
                   "PYTHONDONTWRITEBYTECODE": "1",
                   "ESHKOL_ALLOW_UNSUPPORTED_HOST": "0",
                   "GIT_CONFIG_COUNT": "2",
                   "GIT_CONFIG_KEY_0": "safe.directory",
                   "GIT_CONFIG_VALUE_0": "/workspace",
                   "GIT_CONFIG_KEY_1": "safe.directory",
                   "GIT_CONFIG_VALUE_1": "/source"})
    return result


def inspect_owned(raw: bytes, decision: dict, cid: str,
                  running: bool | None, image_raw: bytes) -> dict:
    data = json.loads(raw)
    require(type(data) is list and len(data) == 1, "inspect cardinality")
    row = data[0]
    require(row["Id"] == cid and row["Name"] == "/" + decision["name"] and
            row["Config"]["Labels"] == {"eshkol.gate": decision["name"]} and
            row["Image"] == IMAGE and
            (running is None or row["State"]["Running"] is running) and
            row["Config"]["User"] == "0:0" and
            row["Config"]["WorkingDir"] == "/source" and
            row["Config"]["Entrypoint"] == ["/usr/bin/python3.10"] and
            row["Config"]["Cmd"] ==
            ["-B", "-m", "tests.g3c4.f_loader_guest",
             "--source-admission-sha256", decision["source_admission_sha256"]] and
            env_map(row["Config"]["Env"]) == expected_env(image_raw) and
            row["HostConfig"]["ReadonlyRootfs"] is True and
            row["HostConfig"]["NetworkMode"] == "none" and
            row["HostConfig"]["Tmpfs"] ==
            {"/tmp": "rw,nosuid,nodev,size=268435456,mode=1777"} and
            row["HostConfig"]["Memory"] == 4 * 1024**3 and
            row["HostConfig"]["MemorySwap"] == 4 * 1024**3 and
            row["HostConfig"]["PidsLimit"] == 128 and
            row["HostConfig"]["NanoCpus"] == 2_000_000_000 and
            row["HostConfig"]["Ulimits"] ==
            [{"Name": "fsize", "Hard": FSIZE_LIMIT, "Soft": FSIZE_LIMIT}],
            "owned container inspect mismatch")
    expected = {decision[key]: (target, read_only) for key, target, read_only in
                (("compiler", "/workspace", True), ("source", "/source", True),
                 ("f_output", "/f-output", True), ("bundle", "/bundle", True),
                 ("candidate", "/stage", False), ("out", "/out", False))}
    mounts = {mount["Source"]: (mount["Destination"], not mount["RW"])
              for mount in row["Mounts"] if mount["Type"] == "bind"}
    require(mounts == expected and len(row["Mounts"]) == len(expected) and
            all(mount["Mode"] == "" and mount["Propagation"] == "rprivate"
                for mount in row["Mounts"]),
            "owned mounts differ")
    expected_host_mounts = [
        {"Type": "bind", "Source": decision[key], "Target": target,
         **({"ReadOnly": True} if readonly else {})}
        for key, target, readonly in
        (("compiler", "/workspace", True), ("source", "/source", True),
         ("f_output", "/f-output", True), ("bundle", "/bundle", True),
         ("candidate", "/stage", False), ("out", "/out", False))]
    require(row["HostConfig"]["Mounts"] == expected_host_mounts and
            row["HostConfig"]["Binds"] is None,
            "owned host mount options differ")
    return row


def guest_output(decision: dict) -> None:
    out = Path(decision["out"])
    build = out / "build-off"
    evidence = out / "evidence"
    attest.copied_artifacts(build, build)
    require((evidence / "F-copy-prepared.json").is_file() and
            (evidence / "loader/collector.json").is_file() and
            (evidence / "loader/image-libraries.tsv").is_file(),
            "durable guest receipts missing")
    receipt = json.loads(ordinary(evidence / "loader/collector.json"))
    require(receipt["format"] == "g3c4-F-image-loader-collector-v1" and
            receipt["image_pins_sha256"] ==
            sha(ordinary(evidence / "loader/image-libraries.tsv")),
            "collector output identity mismatch")


def stopped_exit(row: dict) -> int:
    state = row["State"]
    code = state["ExitCode"]
    require(state["Status"] == "exited" and state["Running"] is False and
            type(state["Pid"]) is int and state["Pid"] == 0 and
            state["OOMKilled"] is False and state["Dead"] is False and
            state["Error"] == "" and type(code) is int and 0 <= code <= 255,
            "owned container is not authentically stopped")
    return code


def wait_exit(raw: bytes, expected: int) -> None:
    require(re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n", raw) is not None and
            int(raw[:-1]) == expected,
            "Docker wait contradicts stopped inspect")


def never_started(row: dict) -> None:
    state = row["State"]
    require(state["Status"] == "created" and state["Running"] is False and
            type(state["Pid"]) is int and state["Pid"] == 0 and
            state["OOMKilled"] is False and state["Dead"] is False and
            state["Error"] == "" and type(state["ExitCode"]) is int and
            state["ExitCode"] == 0,
            "unstarted container identity uncertain")


def run_once(decision_path: Path, expected_sha: str, evidence: Path,
             monitor: Callable[[], None] | None = None) -> None:
    """Only a future immutable ROOT wrapper may supply the decision SHA."""
    decision = selected(decision_path, expected_sha, fresh=True)
    roots = [Path(decision[key]) for key in
             ("compiler", "source", "f_output", "bundle", "candidate", "out")]
    require(evidence.is_absolute() and not evidence.exists() and
            evidence.parent.resolve(strict=True) == evidence.parent and
            all(evidence != root and root not in evidence.parents and
                evidence not in root.parents for root in roots) and
            all(decision_path != root and root not in decision_path.parents and
                decision_path not in root.parents for root in roots),
            "fresh outer evidence/decision overlaps selected input")
    evidence.mkdir(mode=0o700)
    attest.exclusive_bytes(evidence / "once.json", canonical({
        "decision_sha256": expected_sha, "state": "consumed-before-create"}))
    cid = None
    primary = ""
    image = security = None
    def active_checked(argv: list[str], path: Path, timeout: int = 120) -> bytes:
        if monitor is None:
            return checked(argv, path, timeout)
        return checked(argv, path, timeout, monitor)

    def active_capture(argv: list[str], path: Path, timeout: int) -> dict:
        if monitor is None:
            return capture(argv, path, timeout)
        return capture(argv, path, timeout, monitor)

    try:
        image = active_checked([str(DOCKER), "image", "inspect", IMAGE],
                        evidence / "image")
        require(json.loads(image)[0]["Id"] == IMAGE, "image identity")
        expected_env(image)
        security = active_checked([str(DOCKER), "info", "--format",
                            "{{json .SecurityOptions}}"], evidence / "daemon")
        require("rootless" in security.decode(), "daemon is not rootless")
        absence = active_checked([str(DOCKER), "ps", "-a", "--no-trunc", "--format",
                           "{{.ID}} {{.Names}}"], evidence / "initial-absence")
        require(decision["name"].encode() not in absence, "name already exists")
        selected(decision_path, expected_sha, fresh=False)
        created = active_checked(create_argv(decision), evidence / "create")
        require(HEX64.fullmatch(created.decode().strip()) is not None,
                "created CID malformed")
        cid = created.decode().strip()
        created_row = inspect_owned(active_checked([str(DOCKER), "inspect", cid],
                                    evidence / "created"), decision, cid, False, image)
        never_started(created_row)
        started = active_capture([str(DOCKER), "start", "--attach", cid],
                          evidence / "guest", PHASE_LIMIT)
        require(started["quiescent"] and started["returncode"] == 0 and
                not started["failure"] and
                ordinary(evidence / "guest/stderr.raw") == b"", "guest failed")
        guest_output(decision)
        selected(decision_path, expected_sha, fresh=False)
    except BaseException as error:
        primary = f"{type(error).__name__}: {error}"
    finally:
        closure = []
        if cid is not None:
            try:
                raw = checked([str(DOCKER), "inspect", cid],
                              evidence / "closure-inspect")
                row = inspect_owned(raw, decision, cid, None, image)
                if row["State"]["Running"]:
                    checked([str(DOCKER), "kill", cid], evidence / "owned-kill")
                    raw = checked([str(DOCKER), "inspect", cid],
                                  evidence / "stopped")
                    row = inspect_owned(raw, decision, cid, False, image)
                if row["State"]["Status"] == "created":
                    never_started(row)
                    closure.append("guest never started; Docker wait unavailable")
                else:
                    code = stopped_exit(row)
                    try:
                        wait_exit(checked([str(DOCKER), "wait", cid],
                                          evidence / "docker-wait"), code)
                    except BaseException as error:
                        closure.append(f"Docker wait uncertain: {type(error).__name__}: {error}")
                    if code != 0:
                        primary = primary or f"guest exit={code}"
                checked([str(DOCKER), "rm", cid], evidence / "remove")
                after = checked([str(DOCKER), "ps", "-a", "--no-trunc",
                                 "--format", "{{.ID}} {{.Names}}"],
                                evidence / "absence")
                require(cid.encode() not in after and
                        decision["name"].encode() not in after,
                        "owned CID still present")
            except BaseException as error:
                closure.append(f"closure uncertain: {type(error).__name__}: {error}")
        try:
            selected(decision_path, expected_sha, fresh=False)
        except BaseException as error:
            closure.append(f"input drift: {type(error).__name__}: {error}")
        if image is not None and security is not None:
            try:
                after_image = checked([str(DOCKER), "image", "inspect", IMAGE],
                                      evidence / "image-after")
                after_security = checked([str(DOCKER), "info", "--format",
                                          "{{json .SecurityOptions}}"],
                                         evidence / "daemon-after")
                require(after_image == image and after_security == security,
                        "image or rootless daemon changed")
            except BaseException as error:
                closure.append(f"image/daemon drift: {type(error).__name__}: {error}")
        attest.exclusive_bytes(evidence / "result.json", canonical({
            "primary": primary, "closure": closure, "cid": cid,
            "success": not primary and not closure and cid is not None}))
    require(not primary and not closure and cid is not None,
            "outer observation failed or physical closure uncertain")


if __name__ == "__main__":
    raise SystemExit("inactive source: external immutable ROOT wrapper required")
