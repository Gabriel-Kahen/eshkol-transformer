"""Convert only the compiled E3 frame's tagged stdout into reference inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tests.e3_reference.transcript import canonical


WORD = re.compile(r"[0-9a-f]{8}\Z")
CURSOR = re.compile(r"E3ENDCURSOR\t([0-9a-f]{2,800})\Z")
CALLER_CURSOR = re.compile(r"E3CURSOR\t([0-9a-f]{2,800})\t([0-9a-f]{2,800})\Z")
RESULT = re.compile(r"E3RESULT\t(0|[1-9][0-9]*)(?:,(0|[1-9][0-9]*)){5}\Z")
CASE = re.compile(r"E3CASE\t([a-z][a-z0-9-]*)\Z")
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


def parse(raw: bytes, batch_count: int, *, padded_mutant: bool = False) -> tuple[dict[str, object], bytes]:
    if type(batch_count) is not int or batch_count not in (1, 2, 3):
        raise ValueError("numerical batch count must be one, two or three")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("numerical stdout is not ASCII") from error
    if not raw.endswith(b"\n") or lines[-1:] != ["E3-NUMERICAL-RUNTIME-PASS"]:
        raise ValueError("compiled numerical runtime did not complete")
    if len(lines) >= 3 and lines[-3].startswith("E3ENDCURSOR"):
        end = CURSOR.fullmatch(lines[-3])
        if end is None or len(end[1]) % 2:
            raise ValueError("compiled numerical end cursor is malformed")
        del lines[-3]
    batches = [dict() for _ in range(batch_count)]
    roles: list[str] = []
    final: dict[str, object] | None = None
    batch_tags = (("E3ROW",) + ("E3ROLE",) * len(ROLE_NAMES)
                  + ("E3CE", "E3BATCH", "E3STATE"))
    expected_order = [
        (tag, ordinal)
        for ordinal in range(batch_count)
        for tag in batch_tags
    ] + [("E3FINAL", None)]
    if len(lines) - 1 != len(expected_order):
        raise ValueError("numerical frame omitted or duplicated a record")
    for line, (expected_tag, expected_ordinal) in zip(lines[:-1], expected_order):
        fields = line.split("\t")
        tag = fields[0]
        if tag != expected_tag:
            raise ValueError("numerical frame record order or duplicate differs")
        if tag == "E3FINAL":
            if len(fields) != 3 or final is not None:
                raise ValueError("missing or duplicate final publication")
            final = {"metrics_f32": _words(fields[1], 4),
                     "counters": _pair(fields[2])}
            continue
        if tag not in {"E3ROW", "E3ROLE", "E3CE", "E3BATCH", "E3STATE"}:
            raise ValueError("unexpected compiled numerical stdout")
        if len(fields) < 3 or not fields[1].isascii() or not fields[1].isdecimal():
            raise ValueError("numerical batch ordinal is malformed")
        ordinal = int(fields[1])
        if ordinal != expected_ordinal:
            raise ValueError("numerical batch ordinal differs from record order")
        batch = batches[ordinal]
        if tag == "E3ROLE":
            if len(fields) != 5:
                raise ValueError("numerical role line is malformed")
            name, shape_raw, words_raw = fields[2:]
            role_index = len(roles) - ordinal * len(ROLE_NAMES)
            if role_index not in range(len(ROLE_NAMES)) or name != ROLE_NAMES[role_index]:
                raise ValueError("numerical role order differs from fixed schedule")
            shape = [int(value) for value in shape_raw.split(",")]
            count = 1
            for dimension in shape:
                if dimension < 1:
                    raise ValueError("numerical role shape is empty")
                count *= dimension
            _words(words_raw, count)
            roles.append("\t".join((str(ordinal), name, shape_raw, words_raw)))
            continue
        key = tag[2:].lower()
        if key in batch:
            raise ValueError("duplicate numerical batch field")
        if tag == "E3ROW":
            if len(fields) != 5:
                raise ValueError("numerical row line is malformed")
            batch[key] = (_pair(fields[2]), _pair(fields[3]),
                          _pair(fields[4], mask=True))
        elif tag == "E3CE":
            if len(fields) != 3:
                raise ValueError("numerical CE line is malformed")
            batch[key] = _words(fields[2], 2)
        elif tag == "E3BATCH":
            if len(fields) != 4 or not fields[3].isascii() or not fields[3].isdecimal():
                raise ValueError("numerical L3S line is malformed")
            batch[key] = (_words(fields[2], 4), int(fields[3]))
        elif tag == "E3STATE":
            if len(fields) != 4:
                raise ValueError("numerical state line is malformed")
            batch[key] = (_words(fields[2], 3), _pair(fields[3]))
    if final is None or len(roles) != batch_count * len(ROLE_NAMES):
        raise ValueError("numerical frame omitted final or roles")
    observed = []
    for ordinal, batch in enumerate(batches):
        if set(batch) != {"row", "ce", "batch", "state"}:
            raise ValueError("numerical frame omitted a batch field")
        ids, targets, mask = batch["row"]
        reduction, active = batch["batch"]
        state, counters = batch["state"]
        allowed_padded = (padded_mutant and batch_count == 3 and ordinal == 2
                          and mask == [True, False] and active == 2)
        if active != sum(mask) and not allowed_padded:
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
    return payload, ("\n".join(roles) + "\n").encode("ascii")


def parse_compiled(
    raw: bytes, batch_count: int, *, expected_case: str | None = None,
    padded_mutant: bool = False,
) -> tuple[dict[str, object], bytes, dict[str, object]]:
    """Bind the actual in-call end cursor and six caller outputs to one trace."""
    try:
        lines = raw.decode("ascii").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise ValueError("numerical stdout is not ASCII") from error
    if len(lines) < 7 or any(not line.endswith("\n") for line in lines):
        raise ValueError("compiled numerical result records are incomplete")
    tail = [line[:-1] for line in lines[-6:]]
    end_match = CURSOR.fullmatch(tail[0])
    case_match = CASE.fullmatch(tail[2])
    result_match = RESULT.fullmatch(tail[3])
    cursor_match = CALLER_CURSOR.fullmatch(tail[4])
    if (end_match is None or not tail[1].startswith("E3FINAL\t")
            or case_match is None or result_match is None
            or cursor_match is None or tail[5] != "E3-NUMERICAL-RUNTIME-PASS"):
        raise ValueError("compiled numerical result/cursor record order differs")
    case_id = case_match[1]
    if expected_case is not None and case_id != expected_case:
        raise ValueError("compiled numerical case identity differs")
    cursors = (cursor_match[1], end_match[1], cursor_match[2])
    if any(len(cursor) % 2 for cursor in cursors):
        raise ValueError("compiled numerical cursor has odd length")
    words = [int(part) for part in tail[3].split("\t", 1)[1].split(",")]
    if any(word >= 1 << 32 for word in words):
        raise ValueError("compiled numerical result is outside u32")
    observed, roles = parse("".join(lines[:-4]).encode("ascii")
                            + b"E3-NUMERICAL-RUNTIME-PASS\n",
                            batch_count, padded_mutant=padded_mutant)
    final = observed["final"]
    if words[:4] != [int(word, 16) for word in final["metrics_f32"]]:
        raise ValueError("compiled six-output metrics differ from observed frame")
    if words[4:] != final["counters"]:
        raise ValueError("compiled six-output counters differ from observed frame")
    return observed, roles, {
        "kind": "e3-section9-compiled-result-v1", "case_id": case_id,
        "outputs_u32": words, "cursor_before": cursors[0],
        "cursor_end": cursors[1], "cursor_after": cursors[2],
    }


def validate_result_against_case(
    result: dict[str, object], case: dict[str, object], observed: dict[str, object],
) -> None:
    """Compare actual caller/in-call bytes to the independently replayed D2 case."""
    d2 = case["d2"]
    if result["case_id"] != case["case_id"]:
        raise ValueError("compiled result case identity differs")
    if result["cursor_before"] != d2["start_cursor"]:
        raise ValueError("compiled caller start cursor differs from frozen D2")
    if result["cursor_end"] != d2["end_cursor"]:
        raise ValueError("compiled in-call end cursor differs from frozen D2")
    if result["cursor_after"] != d2["start_cursor"]:
        raise ValueError("compiled caller cursor was not restored")
    final = observed["final"]
    if result["outputs_u32"][:4] != [int(word, 16) for word in final["metrics_f32"]]:
        raise ValueError("compiled caller metrics differ from observed frame")
    if result["outputs_u32"][4:] != final["counters"]:
        raise ValueError("compiled caller counters differ from observed frame")


def _unique(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate compiled result field")
        result[key] = value
    return result


def decode_result(raw: bytes) -> dict[str, object]:
    try:
        result = json.loads(raw, object_pairs_hook=_unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("compiled result is not JSON") from error
    if (not isinstance(result, dict) or set(result) != {
            "kind", "case_id", "outputs_u32", "cursor_before", "cursor_end",
            "cursor_after"} or result["kind"] != "e3-section9-compiled-result-v1"
            or canonical(result) != raw):
        raise ValueError("compiled result schema or encoding differs")
    words = result["outputs_u32"]
    if (not isinstance(words, list) or len(words) != 6
            or any(type(word) is not int or not 0 <= word < 1 << 32 for word in words)):
        raise ValueError("compiled six outputs are malformed")
    for key in ("cursor_before", "cursor_end", "cursor_after"):
        cursor = result[key]
        if (not isinstance(cursor, str) or len(cursor) % 2 or not 2 <= len(cursor) <= 800
                or any(character not in "0123456789abcdef" for character in cursor)):
            raise ValueError("compiled cursor encoding is malformed")
    if not isinstance(result["case_id"], str) or CASE.fullmatch("E3CASE\t" + result["case_id"]) is None:
        raise ValueError("compiled case identity is malformed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--batch-count", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--padded-mutant", action="store_true")
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    raw = args.stdout.read_bytes()
    if args.result is not None or b"E3CASE\t" in raw:
        payload, roles, result = parse_compiled(raw, args.batch_count,
                                                expected_case=args.case,
                                                padded_mutant=args.padded_mutant)
    else:
        payload, roles = parse(raw, args.batch_count,
                               padded_mutant=args.padded_mutant)
        result = None
    args.observed.write_bytes(canonical(payload))
    args.roles.write_bytes(roles)
    if args.result is not None:
        if result is None:
            raise ValueError("compiled numerical result is absent")
        args.result.write_bytes(canonical(result))


if __name__ == "__main__":
    main()
