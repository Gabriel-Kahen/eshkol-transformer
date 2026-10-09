"""Development-only strict raw-bit transport for the C4 native observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import tempfile
import time

FIXTURE_SHA256 = "2f3adec7636cebc39b735d7b6c461e2856e390ec1e9f12fccdc0c3074184f041"
SEEDS = (0, 1729, (1 << 63) - 1)
MAX_RAW_BYTES = 2 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024


def _canonical(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def load_fixture(path: Path) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError("frozen fixture SHA drift")
    fixture = json.loads(raw, object_pairs_hook=_unique)
    if type(fixture) is not dict or _canonical(fixture) != raw:
        raise ValueError("noncanonical fixture")
    if (type(fixture.get("models")) is not list or
        [model.get("seed") for model in fixture["models"]] != list(SEEDS) or
        type(fixture.get("cases")) is not list or
        len(fixture["cases"]) != 38):
        raise ValueError("frozen fixture identity")
    return fixture


def make_input(fixture: dict) -> bytes:
    lines = ["C4CAP1 3"]
    for seed in SEEDS:
        cases = [case for case in fixture["cases"] if case["seed"] == seed]
        if not cases or len(cases) > 32:
            raise ValueError("case count")
        lines.append(f"MODEL {seed} {len(cases)}")
        for case in cases:
            tokens = case["tokens"]
            if (type(tokens) is not list or not 1 <= len(tokens) <= 4 or
                any(type(token) is not int or not 0 <= token <= 255
                    for token in tokens)):
                raise ValueError("prefix identity")
            lines.append("CASE " + str(len(tokens)) + " " +
                         " ".join(map(str, tokens)))
    lines.append("END")
    return ("\n".join(lines) + "\n").encode("ascii")


def _integer(word: str, minimum: int, maximum: int) -> int:
    if (not word or word != str(int(word)) or
        not minimum <= int(word) <= maximum):
        raise ValueError("noncanonical integer")
    return int(word)


def _float_words(words: list[str], count: int, shape: list[int]) -> dict:
    if len(words) != count or math.prod(shape) != count:
        raise ValueError("float count/shape")
    for word in words:
        if (len(word) != 8 or any(char not in "0123456789abcdef"
                                  for char in word) or
            not math.isfinite(struct.unpack(">f", bytes.fromhex(word))[0])):
            raise ValueError("invalid/nonfinite f32 bit word")
    return {"dtype": "float32", "device": "cpu", "shape": shape,
            "bits": words}


def _word_line(line: str, tag: str, count: int, shape: list[int]) -> dict:
    fields = line.split(" ")
    if (fields[:2] != [tag, str(count)] or any(not word for word in fields)):
        raise ValueError(f"{tag} marker/count")
    return _float_words(fields[2:], count, shape)


def parse(raw: bytes, fixture: dict) -> dict:
    if (len(raw) > MAX_RAW_BYTES or not raw.endswith(b"\n") or
        b"\r" in raw or b"\x00" in raw):
        raise ValueError("raw transcript framing")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("raw transcript encoding") from error
    cursor = 0

    def next_line() -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError("truncated native transcript")
        line = lines[cursor]
        cursor += 1
        return line

    if next_line() != "C4CAP1":
        raise ValueError("native transcript version")
    models: list[dict] = []
    cases: list[dict] = []
    for expected_model in fixture["models"]:
        seed = expected_model["seed"]
        fields = next_line().split(" ")
        if (len(fields) != 6 or fields[0] != "MODEL" or
            _integer(fields[1], 0, (1 << 63) - 1) != seed):
            raise ValueError("model identity")
        state = [_integer(word, 0, (1 << 63) - 1) for word in fields[2:]]
        if state != [1, seed, 292, 0]:
            raise ValueError("initializer state")
        models.append({"seed": seed, "successor_words": state,
                       "parameters": _word_line(next_line(), "PARAMETERS",
                                                1192, [1192])})
        for expected_case in (row for row in fixture["cases"]
                              if row["seed"] == seed):
            tokens = expected_case["tokens"]
            prefix = len(tokens)
            fields = next_line().split(" ")
            if (fields[:3] != ["CASE", str(seed), str(prefix)] or
                len(fields) != prefix + 3 or
                [_integer(word, 0, 255) for word in fields[3:]] != tokens):
                raise ValueError("case identity/order")
            if prefix == 4 and next_line() != "T4ABORT 1 0":
                raise ValueError("intended T4 abort")
            cases.append({"seed": seed, "tokens": tokens,
                          "logits": _word_line(next_line(), "LOGITS",
                                               prefix * 256, [prefix, 256]),
                          "keys": _word_line(next_line(), "KEYS",
                                             prefix * 4, [1, 2, prefix, 2]),
                          "values": _word_line(next_line(), "VALUES",
                                               prefix * 4, [1, 2, prefix, 2])})
            if next_line() != "ENDCASE":
                raise ValueError("case terminator")
    if next_line() != "ENDCAPTURE" or cursor != len(lines):
        raise ValueError("capture terminator/extra record")
    return {"version": 1, "models": models, "cases": cases}


def encode_observation(observation: dict) -> bytes:
    return _canonical(observation)


def _identity(pid: int) -> tuple[int, int, int, int, str]:
    raw = (Path("/proc") / str(pid) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return pid, int(fields[2]), int(fields[3]), int(fields[19]), fields[0]


def _members(sid: int, uid: int) -> list[tuple[int, int, int, int, str]]:
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            member = _identity(int(path.name))
            owner = path.stat().st_uid
        except FileNotFoundError:
            continue
        if member[2] == sid and member[4] != "Z":
            if owner != uid:
                raise RuntimeError("foreign member in native capture session")
            members.append(member)
    return members


def _signal_session(leader: tuple[int, int, int, int, str], uid: int,
                    sig: signal.Signals) -> None:
    pid, group, session, start, _ = leader
    current = _identity(pid)
    if (current[1:4] != (group, session, start) or
        (Path("/proc") / str(pid)).stat().st_uid != uid or
        any(member[1] != group for member in _members(session, uid))):
        raise RuntimeError("native capture session identity changed")
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def run_native(binary: Path, request: bytes, evidence: Path,
               timeout_seconds: int = 120) -> bytes:
    """Cap both streams and retain the owned child through pipe closure and wait4."""
    if (not binary.is_absolute() or not binary.is_file() or
        type(request) is not bytes or len(request) > 65536 or
        timeout_seconds < 1 or timeout_seconds > 120):
        raise ValueError("native executable/timeout admission")
    before = hashlib.sha256(binary.read_bytes()).hexdigest()
    streams = [bytearray(), bytearray()]
    seen = [0, 0]
    limits = [MAX_RAW_BYTES, MAX_STDERR_BYTES]
    returncode: int | None = None
    wait_status: int | None = None
    child = None
    leader = None
    quiescent = False
    timed_out = False
    overflow = ""
    failure = ""
    teardown: list[str] = []
    uid = os.getuid()
    started = time.monotonic()
    selector = selectors.DefaultSelector()

    def drain(events: list[tuple[selectors.SelectorKey, int]]) -> None:
        nonlocal failure, overflow
        for key, _ in events:
            channel = key.data
            data = os.read(key.fd, 65536)
            if not data:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            seen[channel] += len(data)
            remaining = max(0, limits[channel] - len(streams[channel]))
            streams[channel].extend(data[:remaining])
            if len(data) > remaining and not overflow:
                overflow = ("stdout" if channel == 0 else "stderr")
                failure = f"native {overflow} overflow"

    try:
        with tempfile.TemporaryFile() as input_file:
            input_file.write(request)
            input_file.seek(0)
            child = subprocess.Popen([str(binary)], stdin=input_file,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     start_new_session=True, close_fds=True)
        leader = _identity(child.pid)
        if (leader[1:3] != (child.pid, child.pid) or
            (Path("/proc") / str(child.pid)).stat().st_uid != uid):
            raise RuntimeError("native capture leader identity differs")
        for channel, pipe in enumerate((child.stdout, child.stderr)):
            assert pipe is not None
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, channel)
        termination_at = killed_at = None
        while True:
            exited = os.waitid(os.P_PID, child.pid,
                               os.WEXITED | os.WNOHANG | os.WNOWAIT)
            members = _members(child.pid, uid)
            descendants = [member for member in members if member[0] != child.pid]
            if exited and not selector.get_map() and not descendants:
                quiescent = True
                break
            now = time.monotonic()
            if exited and descendants and termination_at is None:
                failure = failure or "unexpected native capture descendants"
            if now - started >= timeout_seconds and termination_at is None:
                timed_out = True
                failure = failure or "native child timeout"
            if failure and termination_at is None:
                _signal_session(leader, uid, signal.SIGTERM)
                teardown.append("owned session: SIGTERM")
                termination_at = now
            elif termination_at is not None and now - termination_at >= 1 and \
                    killed_at is None:
                _signal_session(leader, uid, signal.SIGKILL)
                teardown.append("owned session: SIGKILL")
                killed_at = now
            elif killed_at is not None and now - killed_at >= 5:
                raise RuntimeError("native capture writers did not close")
            drain(selector.select(0.05))
    except BaseException as error:
        failure = failure or f"native capture error: {type(error).__name__}: {error}"
    finally:
        if child is not None and not quiescent:
            try:
                if leader is None:
                    raise RuntimeError("native capture leader identity unavailable")
                _signal_session(leader, uid, signal.SIGKILL)
                teardown.append("cleanup: owned session SIGKILL")
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    drain(selector.select(0.05))
                    exited = os.waitid(os.P_PID, child.pid,
                                       os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    members = _members(child.pid, uid)
                    if (exited and not selector.get_map() and
                        all(member[0] == child.pid for member in members)):
                        quiescent = True
                        break
                if not quiescent:
                    teardown.append("native capture writer closure uncertain")
            except BaseException as error:
                teardown.append(f"cleanup uncertainty: {type(error).__name__}: {error}")
        if child is not None:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and wait_status is None:
                try:
                    pid, raw_wait, _ = os.wait4(child.pid, os.WNOHANG)
                except InterruptedError:
                    teardown.append("wait4 interrupted; retrying")
                    continue
                except BaseException as error:
                    teardown.append(f"wait4 uncertainty: {type(error).__name__}: {error}")
                    break
                if pid:
                    wait_status = raw_wait
                    returncode = os.waitstatus_to_exitcode(raw_wait)
                    child.returncode = returncode
                    break
                time.sleep(0.05)
            if wait_status is None:
                teardown.append("native capture child wait uncertain")
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

    stdout, stderr = map(bytes, streams)
    try:
        after = hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError:
        after = "missing"
    if before != after:
        failure = failure or "native executable drift"
    if not quiescent or wait_status is None or teardown and any(
            "uncertain" in note for note in teardown):
        failure = failure or "native capture closure uncertain"
    if returncode != 0 or stderr:
        failure = failure or (f"native child failed: returncode={returncode}, "
                              f"stderr_bytes={len(stderr)}")
    receipt = {"binary": str(binary), "sha256_pre": before,
               "sha256_post": after, "returncode": returncode,
               "leader_identity": leader, "uid": uid,
               "elapsed_seconds": time.monotonic() - started,
               "timeout_seconds": timeout_seconds,
               "wait_status": wait_status, "quiescent": quiescent,
               "timed_out": timed_out, "overflow": overflow,
               "stdout_seen": seen[0], "stderr_seen": seen[1],
               "teardown": teardown,
               "failure": failure, "stdout_bytes": len(stdout),
               "stderr_bytes": len(stderr)}
    with evidence.with_suffix(".stdout").open("xb") as stream:
        stream.write(stdout)
    with evidence.with_suffix(".stderr").open("xb") as stream:
        stream.write(stderr)
    with evidence.with_suffix(".status.json").open("xb") as stream:
        stream.write(_canonical(receipt))
    if failure:
        raise ValueError(failure)
    return stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--input", action="store_true")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fixture = load_fixture(args.fixture)
    if args.binary is not None:
        if (args.request is None or args.evidence is None or
            args.input or args.raw is not None or
            args.output is not None):
            parser.error("binary route requires --request and --evidence only")
        if args.request.read_bytes() != make_input(fixture):
            raise ValueError("native request drift")
        run_native(args.binary, make_input(fixture), args.evidence)
        return
    if args.input == (args.raw is not None) or args.output is None:
        parser.error("exactly one of --input or --raw needs --output")
    if args.request is not None or args.evidence is not None:
        parser.error("unexpected capture route argument")
    payload = (make_input(fixture) if args.input else
               encode_observation(parse(args.raw.read_bytes(), fixture)))
    try:
        with args.output.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("refusing to replace native capture artifact") from error


if __name__ == "__main__":
    main()
