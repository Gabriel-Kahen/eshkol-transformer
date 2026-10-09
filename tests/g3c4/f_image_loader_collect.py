"""Test-only bounded readelf/loader observation for a copied run-F compiler.

The outer ROOT capsule, not this module, admits the f31 image and source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

from tests.g3c4 import f_artifact_attest as attest


TOOL_PINS_SHA = "18764bbd7e634c0c01c3ca8f0337fa34f73d92b0631bffb23e25c3a1ef850cc7"
RUNTIME33_SHA = "94a7c39142c213681da20ba640996d14504d4c697dcc88c925af35577dedb53c"
BUILD = Path("/out/build-off")
TOOL_PINS = Path("/bundle/accepted-tools.tsv")
RUNTIME_PINS = Path("/bundle/accepted-runtime33.tsv")
READ_ELF = Path("/usr/bin/readelf")
LOADER = Path("/lib64/ld-linux-x86-64.so.2")
TIMEOUT_SECONDS = 120
STDOUT_LIMIT = 2 * 1024 * 1024
STDERR_LIMIT = 64 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def unique_rows(raw: bytes, columns: int) -> list[list[str]]:
    rows = []
    seen = set()
    for line in raw.decode("utf-8").splitlines():
        cells = line.split("\t")
        require(len(cells) == columns and cells[0] not in seen,
                "pin inventory malformed/duplicated")
        seen.add(cells[0])
        rows.append(cells)
    return rows


def pin_tables(tools: Path, runtimes: Path,
               expected_tools: str = TOOL_PINS_SHA,
               expected_runtimes: str = RUNTIME33_SHA) -> tuple[dict, dict]:
    require(tools.is_absolute() and runtimes.is_absolute(),
            "pin paths must be absolute")
    tools_raw, runtime_raw = attest.ordinary(tools), attest.ordinary(runtimes)
    require(sha(tools_raw) == expected_tools and
            sha(runtime_raw) == expected_runtimes,
            "saved f31 tool/runtime pin SHA drift")
    tool_rows = unique_rows(tools_raw, 3)
    runtime_rows = unique_rows(runtime_raw, 2)
    require(len(runtime_rows) == 33 or expected_runtimes != RUNTIME33_SHA,
            "accepted runtime33 member count drift")
    tool_map = {}
    for name, resolved, digest in tool_rows:
        require(Path(name).is_absolute() and Path(resolved).is_absolute() and
                re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "tool pin format")
        tool_map[name] = (resolved, digest)
    runtime_map = {}
    for digest, resolved in runtime_rows:
        require(Path(resolved).is_absolute() and
                re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                resolved not in runtime_map, "runtime pin format")
        runtime_map[resolved] = digest
    return tool_map, runtime_map


def pinned_file(path: Path, expected_path: str, expected_sha: str) -> dict:
    resolved = path.resolve(strict=True)
    require(str(resolved) == expected_path and
            sha(attest.ordinary(resolved)) == expected_sha,
            f"resolved image member drift: {path}")
    st = resolved.lstat()
    return {"path": str(path), "resolved": str(resolved),
            "sha256": expected_sha, "dev": st.st_dev, "ino": st.st_ino,
            "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns}


def image_state(tools: dict, runtimes: dict,
                readelf: Path = READ_ELF, loader: Path = LOADER) -> dict:
    require(str(readelf) in tools, "readelf absent from accepted tools")
    readelf_state = pinned_file(readelf, *tools[str(readelf)])
    loader_resolved = str(loader.resolve(strict=True))
    require(loader_resolved in runtimes, "loader absent from accepted runtime33")
    loader_state = pinned_file(loader, loader_resolved, runtimes[loader_resolved])
    runtime_state = {path: pinned_file(Path(path), path, digest)
                     for path, digest in runtimes.items()}
    return {"readelf": readelf_state, "loader": loader_state,
            "runtime33": runtime_state}


def identity(pid: int) -> tuple[int, int, int, int, str]:
    raw = (Path("/proc") / str(pid) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return pid, int(fields[2]), int(fields[3]), int(fields[19]), fields[0]


def members(sid: int, uid: int) -> list[tuple[int, int, int, int, str]]:
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            row = identity(int(path.name))
            owner = path.stat().st_uid
        except FileNotFoundError:
            continue
        if row[2] == sid and row[4] != "Z":
            require(owner == uid, "foreign member in collector session")
            found.append(row)
    return found


def signal_owned(leader: tuple[int, int, int, int, str], uid: int,
                 sig: signal.Signals) -> None:
    pid, group, session, start, _ = leader
    current = identity(pid)
    require(current[1:4] == (group, session, start) and
            (Path("/proc") / str(pid)).stat().st_uid == uid and
            all(row[1] == group for row in members(session, uid)),
            "collector session identity changed")
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def capture(argv: list[str], evidence: Path,
            timeout: int = TIMEOUT_SECONDS) -> bytes:
    """Bound two streams while keeping the session leader unreaped until closure."""
    require(argv and all(type(word) is str and word for word in argv) and
            1 <= timeout <= TIMEOUT_SECONDS and not evidence.exists(),
            "collector phase admission")
    evidence.mkdir(mode=0o700)
    attest.exclusive_bytes(evidence / "argv.json", canonical({"argv": argv}))
    selector = selectors.DefaultSelector()
    streams = [bytearray(), bytearray()]
    seen = [0, 0]
    limits = [STDOUT_LIMIT, STDERR_LIMIT]
    child = None
    leader = None
    wait_status = None
    returncode = None
    quiescent = False
    failure = ""
    overflow = ""
    teardown = []
    uid = os.getuid()
    started = time.monotonic()

    def drain() -> None:
        nonlocal failure, overflow
        for key, _ in selector.select(0.05):
            channel = key.data
            data = os.read(key.fd, 65536)
            if not data:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            seen[channel] += len(data)
            space = max(0, limits[channel] - len(streams[channel]))
            streams[channel].extend(data[:space])
            if len(data) > space and not overflow:
                overflow = "stdout" if channel == 0 else "stderr"
                failure = f"collector {overflow} overflow"

    try:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 start_new_session=True, close_fds=True)
        leader = identity(child.pid)
        require(leader[1:3] == (child.pid, child.pid) and
                (Path("/proc") / str(child.pid)).stat().st_uid == uid,
                "collector leader identity differs")
        for channel, pipe in enumerate((child.stdout, child.stderr)):
            assert pipe is not None
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, channel)
        term_at = kill_at = None
        while True:
            exited = os.waitid(os.P_PID, child.pid,
                               os.WEXITED | os.WNOHANG | os.WNOWAIT)
            live = members(child.pid, uid)
            descendants = [row for row in live if row[0] != child.pid]
            if exited and not selector.get_map() and not descendants:
                quiescent = True
                break
            now = time.monotonic()
            if exited and descendants and term_at is None:
                failure = failure or "unexpected collector descendants"
            if now - started >= timeout and term_at is None:
                failure = failure or "collector timeout"
            if failure and term_at is None:
                signal_owned(leader, uid, signal.SIGTERM)
                teardown.append("owned session SIGTERM")
                term_at = now
            elif term_at is not None and now - term_at >= 1 and kill_at is None:
                signal_owned(leader, uid, signal.SIGKILL)
                teardown.append("owned session SIGKILL")
                kill_at = now
            elif kill_at is not None and now - kill_at >= 5:
                raise RuntimeError("collector writers did not close")
            drain()
    except BaseException as error:
        failure = failure or f"collector error: {type(error).__name__}: {error}"
    finally:
        if child is not None and not quiescent:
            try:
                require(leader is not None, "collector leader identity unavailable")
                signal_owned(leader, uid, signal.SIGKILL)
                teardown.append("cleanup owned session SIGKILL")
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    drain()
                    exited = os.waitid(os.P_PID, child.pid,
                                       os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    live = members(child.pid, uid)
                    if (exited and not selector.get_map() and
                        all(row[0] == child.pid for row in live)):
                        quiescent = True
                        break
                if not quiescent:
                    teardown.append("collector writer closure uncertain")
            except BaseException as error:
                teardown.append(f"cleanup uncertainty: {type(error).__name__}: {error}")
        if child is not None:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and wait_status is None:
                try:
                    pid, status, _ = os.wait4(child.pid, os.WNOHANG)
                except InterruptedError:
                    teardown.append("wait4 interrupted; retrying")
                    continue
                except BaseException as error:
                    teardown.append(f"wait4 uncertainty: {type(error).__name__}: {error}")
                    break
                if pid:
                    wait_status = status
                    returncode = os.waitstatus_to_exitcode(status)
                    child.returncode = returncode
                    break
                time.sleep(0.05)
            if wait_status is None:
                teardown.append("collector child wait uncertain")
        try:
            for key in list(selector.get_map().values()):
                selector.unregister(key.fileobj)
                key.fileobj.close()
            selector.close()
        except BaseException as error:
            teardown.append(f"stream close uncertainty: {type(error).__name__}: {error}")
        if child is not None:
            for pipe in (child.stdout, child.stderr):
                if pipe is not None and not pipe.closed:
                    try:
                        pipe.close()
                    except BaseException as error:
                        teardown.append(f"pipe close uncertainty: {type(error).__name__}: {error}")

    stdout, stderr = bytes(streams[0]), bytes(streams[1])
    if (not quiescent or wait_status is None or
        any("uncertain" in note for note in teardown)):
        failure = failure or "collector closure uncertain"
    if returncode != 0 or stderr:
        failure = failure or f"collector child exit={returncode}, stderr={len(stderr)}"
    receipt = {"argv": argv, "leader": leader, "uid": uid,
               "wait_status": wait_status, "returncode": returncode,
               "quiescent": quiescent, "failure": failure,
               "overflow": overflow, "seen": seen, "teardown": teardown,
               "elapsed_seconds": time.monotonic() - started}
    attest.exclusive_bytes(evidence / "stdout.raw", stdout)
    attest.exclusive_bytes(evidence / "stderr.raw", stderr)
    attest.exclusive_bytes(evidence / "status.json", canonical(receipt))
    if failure:
        raise ValueError(failure)
    return stdout


def loader_rows(raw: bytes, runtime_map: dict[str, str]) -> bytes:
    observed = {}
    for line in raw.decode("utf-8").splitlines():
        mapped = re.fullmatch(r"\s*(\S+) => (/[\S]+) \(0x[0-9a-fA-F]+\)", line)
        if mapped:
            name, path = mapped.groups()
        else:
            direct = re.fullmatch(r"\s*(/\S+) \(0x[0-9a-fA-F]+\)", line)
            if direct:
                path = direct.group(1)
                name = Path(path).name
            else:
                require(re.fullmatch(r"\s*linux-vdso\.so\.1 \(0x[0-9a-fA-F]+\)",
                                     line) is not None, "unknown loader row")
                continue
        require(name not in observed, "duplicate loader name")
        resolved = str(Path(path).resolve(strict=True))
        require(resolved in runtime_map, "unadmitted loader dependency")
        pinned_file(Path(path), resolved, runtime_map[resolved])
        observed[name] = (path, runtime_map[resolved])
    require(observed and "ld-linux-x86-64.so.2" in observed,
            "loader closure incomplete")
    return "".join(f"{name}\t{path}\t{digest}\n"
                   for name, (path, digest) in sorted(observed.items())).encode()


def collect(binary: Path, tools_path: Path, runtimes_path: Path,
            evidence: Path, *, expected_tools: str = TOOL_PINS_SHA,
            expected_runtimes: str = RUNTIME33_SHA,
            build: Path = BUILD, readelf: Path = READ_ELF,
            loader: Path = LOADER) -> dict:
    require(binary == build / "eshkol-run" and evidence.is_absolute() and
            not evidence.exists(), "collector selected path/evidence admission")
    tools, runtimes = pin_tables(tools_path, runtimes_path,
                                expected_tools, expected_runtimes)
    original = attest.copied_artifacts(build, build)
    before = image_state(tools, runtimes, readelf, loader)
    evidence.mkdir(mode=0o700)
    readelf_raw = capture([str(readelf), "-d", str(binary)], evidence / "readelf")
    require(len(readelf_raw) <= 65536, "readelf semantic receipt oversized")
    needed, _ = attest.elf_members(readelf_raw)
    loader_raw = capture([str(loader), "--list", str(binary)], evidence / "loader")
    require(len(loader_raw) <= 65536, "loader semantic receipt oversized")
    pins = loader_rows(loader_raw, runtimes)
    attest.exclusive_bytes(evidence / "image-libraries.tsv", pins)
    argv = (str(loader) + "\0--list\0" + str(binary) + "\0").encode()
    closure = attest.resolved_image_libs(readelf_raw, argv, b"0\n",
                                         loader_raw, b"", pins)
    require(needed <= {row["name"] for row in closure},
            "direct ELF dependency absent from pinned loader closure")
    require(attest.copied_artifacts(build, build) == original and
            image_state(tools, runtimes, readelf, loader) == before and
            pin_tables(tools_path, runtimes_path,
                       expected_tools, expected_runtimes) == (tools, runtimes) and
            loader_rows(loader_raw, runtimes) == pins,
            "collector selected input changed before publication")
    receipt = {"format": "g3c4-F-image-loader-collector-v1",
               "origin": "test-only;historical-F-failed",
               "copied_artifacts": original,
               "tool_pins_sha256": expected_tools,
               "runtime33_sha256": expected_runtimes,
               "image_state": before,
               "readelf_sha256": sha(readelf_raw),
               "loader_sha256": sha(loader_raw),
               "image_pins_sha256": sha(pins),
               "direct_needed": sorted(needed),
               "loader_closure": closure}
    attest.exclusive_bytes(evidence / "collector.json", canonical(receipt))
    require(attest.ordinary(evidence / "collector.json") == canonical(receipt) and
            attest.copied_artifacts(build, build) == original and
            image_state(tools, runtimes, readelf, loader) == before and
            pin_tables(tools_path, runtimes_path,
                       expected_tools, expected_runtimes) == (tools, runtimes),
            "collector post-publication identity drift")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tool-pins", type=Path, required=True)
    parser.add_argument("--runtime-pins", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    require(args.tool_pins == TOOL_PINS and args.runtime_pins == RUNTIME_PINS,
            "collector requires fixed external f31 pin mounts")
    receipt = collect(args.binary, args.tool_pins, args.runtime_pins,
                      args.evidence)
    print(canonical(receipt).decode(), end="")


if __name__ == "__main__":
    main()
