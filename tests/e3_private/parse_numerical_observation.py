"""Convert only the compiled E3 frame's tagged stdout into reference inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from tests.e3_reference.transcript import canonical


WORD = re.compile(r"[0-9a-f]{8}\Z")
ROLE_NAMES = (
    "ET", "EP", "X", "N1", "QT", "KT", "VT", "QH", "KH", "VH", "AH",
    "AT", "AO", "R", "N2", "FU", "FG", "FD", "Y", "NF", "Z",
)


def _words(raw: str, count: int) -> list[str]:
    parts = raw.split(",")
    if len(parts) != count or any(not WORD.fullmatch(part) for part in parts):
        raise ValueError("numerical frame emitted malformed binary32 words")
    return parts


def _pair(raw: str, *, mask: bool = False) -> list[int] | list[bool]:
    parts = raw.split(",")
    if len(parts) != 2 or any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("numerical frame emitted malformed pair")
    values = [int(part) for part in parts]
    if mask:
        if any(value not in (0, 1) for value in values):
            raise ValueError("numerical frame emitted malformed bool mask")
        return [bool(value) for value in values]
    return values


def _integers(raw: str, count: int | None = None, maximum: int = (1 << 63) - 1) -> list[int]:
    parts = raw.split(",")
    if (count is not None and len(parts) != count) or not parts or any(
        not part or not part.isascii() or not part.isdecimal() or
        str(int(part)) != part or int(part) > maximum for part in parts
    ):
        raise ValueError("numerical frame emitted malformed integer words")
    return [int(part) for part in parts]


def parse(raw: bytes, expected_batches: int = 3) -> tuple[dict[str, object], bytes, dict[str, object]]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("numerical stdout is not ASCII") from error
    if not raw.endswith(b"\n") or lines[-1:] != ["E3-NUMERICAL-RUNTIME-PASS"]:
        raise ValueError("compiled numerical runtime did not complete")
    if expected_batches not in (2, 3):
        raise ValueError("numerical expected batch count is invalid")
    batches = [dict() for _ in range(expected_batches)]
    roles: list[str] = []
    cursor = 0
    def take(tag: str, ordinal: int | None = None) -> list[str]:
        nonlocal cursor
        if cursor >= len(lines) - 1:
            raise ValueError("numerical frame omitted an observation")
        fields = lines[cursor].split("\t")
        cursor += 1
        if fields[0] != tag or (ordinal is not None and
                (len(fields) < 2 or fields[1] != str(ordinal))):
            raise ValueError("numerical observation is missing, duplicate, or reordered")
        return fields

    for ordinal, batch in enumerate(batches):
        fields = take("E3ROW", ordinal)
        if len(fields) != 5:
            raise ValueError("numerical row line is malformed")
        batch["row"] = (_pair(fields[2]), _pair(fields[3]),
                        _pair(fields[4], mask=True))
        for name in ROLE_NAMES:
            fields = take("E3ROLE", ordinal)
            if len(fields) != 5 or fields[2] != name:
                raise ValueError("numerical role order differs from fixed schedule")
            shape = _integers(fields[3])
            if not 1 <= len(shape) <= 4 or 0 in shape:
                raise ValueError("numerical role shape is empty")
            count = 1
            for dimension in shape:
                count *= dimension
            _words(fields[4], count)
            roles.append("\t".join((str(ordinal), name, fields[3], fields[4])))
        fields = take("E3CE", ordinal)
        if len(fields) != 3:
            raise ValueError("numerical CE line is malformed")
        batch["ce"] = _words(fields[2], 2)
        fields = take("E3BATCH", ordinal)
        if len(fields) != 4:
            raise ValueError("numerical L3S line is malformed")
        batch["batch"] = (_words(fields[2], 4), _integers(fields[3], 1)[0])
        fields = take("E3STATE", ordinal)
        if len(fields) != 4:
            raise ValueError("numerical state line is malformed")
        batch["state"] = (_words(fields[2], 3), _pair(fields[3]))
    fields = take("E3FINAL")
    if len(fields) != 3:
        raise ValueError("missing or duplicate final publication")
    final = {"metrics_f32": _words(fields[1], 4), "counters": _pair(fields[2])}
    fields = take("E3RESULT")
    if len(fields) != 4 or cursor != len(lines) - 1:
        raise ValueError("numerical result is missing, duplicate, or reordered")
    values = _integers(fields[1], 6, (1 << 32) - 1)
    before = _integers(fields[2], maximum=255)
    after = _integers(fields[3], maximum=255)
    if len(before) < 168 or len(after) != len(before):
        raise ValueError("numerical cursor bytes are malformed")
    result = {"outputs": values, "cursor_before": bytes(before).hex(),
              "cursor_after": bytes(after).hex()}
    if [f"{word:08x}" for word in values[:4]] != final["metrics_f32"] or values[4:] != final["counters"]:
        raise ValueError("boxed outputs differ from native final publication")
    observed = []
    for ordinal, batch in enumerate(batches):
        if set(batch) != {"row", "ce", "batch", "state"}:
            raise ValueError("numerical frame omitted a batch field")
        ids, targets, mask = batch["row"]
        reduction, active = batch["batch"]
        state, counters = batch["state"]
        if active != sum(mask):
            raise ValueError("numerical frame active count differs from mask")
        z = roles[ordinal * len(ROLE_NAMES) + 20].split("\t")[3].split(",")
        observed.append({
            "ordinal": ordinal, "inputs": ids, "targets": targets, "mask": mask,
            "logits_f32": z, "ce_f32": batch["ce"],
            "reduction_f32": reduction[:3], "correct_f32": reduction[3],
            "state_f32": state, "counters": counters,
        })
    payload = {
        "kind": "observed-f32", "profile": "e3-private-n1-t2-v256-d4-v1",
        "batches": observed, "final": final,
    }
    return payload, ("\n".join(roles) + "\n").encode("ascii"), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--batches", required=True, type=int)
    args = parser.parse_args()
    payload, roles, result = parse(args.stdout.read_bytes(), args.batches)
    args.observed.write_bytes(canonical(payload))
    args.roles.write_bytes(roles)
    args.result.write_bytes(canonical(result))


if __name__ == "__main__":
    main()
