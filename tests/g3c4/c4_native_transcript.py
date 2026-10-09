"""Development-only strict raw-bit transport for the C4 native observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

FIXTURE_SHA256 = "2f3adec7636cebc39b735d7b6c461e2856e390ec1e9f12fccdc0c3074184f041"
SEEDS = (0, 1729, (1 << 63) - 1)
MAX_RAW_BYTES = 2 * 1024 * 1024


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


def run_native(binary: Path, request: bytes, evidence: Path,
               timeout_seconds: int = 120) -> bytes:
    """Run one exact fixed-source observer, denying failed or drifting children."""
    if (not binary.is_absolute() or not binary.is_file() or
        timeout_seconds < 1 or timeout_seconds > 120):
        raise ValueError("native executable/timeout admission")
    before = hashlib.sha256(binary.read_bytes()).hexdigest()
    stdout = stderr = b""
    returncode: int | None = None
    failure = ""
    try:
        child = subprocess.run([str(binary)], input=request, capture_output=True,
                               check=False, timeout=timeout_seconds)
        stdout, stderr, returncode = child.stdout, child.stderr, child.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        failure = "native child timeout"
    except OSError as error:
        failure = f"native child spawn failure: {type(error).__name__}"
    try:
        after = hashlib.sha256(binary.read_bytes()).hexdigest()
    except OSError:
        after = "missing"
    if before != after:
        failure = failure or "native executable drift"
    if returncode != 0 or stderr:
        failure = failure or (f"native child failed: returncode={returncode}, "
                              f"stderr_bytes={len(stderr)}")
    if len(stdout) > MAX_RAW_BYTES:
        failure = failure or "native transcript oversized"
    receipt = {"binary": str(binary), "sha256_pre": before,
               "sha256_post": after, "returncode": returncode,
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
