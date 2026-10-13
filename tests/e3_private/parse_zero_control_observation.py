"""Convert only the compiled zero-control E3 frame's tagged stdout into reference inputs."""

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
        raise ValueError("zero-control frame emitted malformed binary32 words")
    return parts


def _pair(raw: str, *, mask: bool = False) -> list[int] | list[bool]:
    parts = raw.split(",")
    if len(parts) != 2 or any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("zero-control frame emitted malformed pair")
    values = [int(part) for part in parts]
    if mask:
        if any(value not in (0, 1) for value in values):
            raise ValueError("zero-control frame emitted malformed bool mask")
        return [bool(value) for value in values]
    return values


def parse(raw: bytes) -> tuple[dict[str, object], bytes]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("zero-control stdout is not ASCII") from error
    if not raw.endswith(b"\n") or lines[-1:] != ["E3-ZERO-CONTROL-RUNTIME-PASS"]:
        raise ValueError("compiled zero-control runtime did not complete")
    batches = [dict() for _ in range(3)]
    roles: list[str] = []
    final: dict[str, object] | None = None
    for line in lines[:-1]:
        fields = line.split("\t")
        tag = fields[0]
        if tag == "E3FINAL":
            if len(fields) != 3 or final is not None:
                raise ValueError("missing or duplicate final publication")
            final = {"metrics_f32": _words(fields[1], 4),
                     "counters": _pair(fields[2])}
            continue
        if tag not in {"E3ROW", "E3ROLE", "E3CE", "E3BATCH", "E3STATE"}:
            raise ValueError("unexpected compiled zero-control stdout")
        if len(fields) < 3 or not fields[1].isascii() or not fields[1].isdecimal():
            raise ValueError("zero-control batch ordinal is malformed")
        ordinal = int(fields[1])
        if ordinal not in range(3):
            raise ValueError("zero-control batch ordinal is out of range")
        batch = batches[ordinal]
        if tag == "E3ROLE":
            if len(fields) != 5:
                raise ValueError("zero-control role line is malformed")
            name, shape_raw, words_raw = fields[2:]
            role_index = len(roles) - ordinal * len(ROLE_NAMES)
            if role_index not in range(len(ROLE_NAMES)) or name != ROLE_NAMES[role_index]:
                raise ValueError("zero-control role order differs from fixed schedule")
            shape = [int(value) for value in shape_raw.split(",")]
            count = 1
            for dimension in shape:
                if dimension < 1:
                    raise ValueError("zero-control role shape is empty")
                count *= dimension
            words = _words(words_raw, count)
            if any(word != "00000000" for word in words):
                raise ValueError("zero-control model emitted a nonzero role word")
            roles.append("\t".join((str(ordinal), name, shape_raw, words_raw)))
            continue
        key = tag[2:].lower()
        if key in batch:
            raise ValueError("duplicate zero-control batch field")
        if tag == "E3ROW":
            if len(fields) != 5:
                raise ValueError("zero-control row line is malformed")
            batch[key] = (_pair(fields[2]), _pair(fields[3]),
                          _pair(fields[4], mask=True))
        elif tag == "E3CE":
            if len(fields) != 3:
                raise ValueError("zero-control CE line is malformed")
            batch[key] = _words(fields[2], 2)
        elif tag == "E3BATCH":
            if len(fields) != 4 or not fields[3].isascii() or not fields[3].isdecimal():
                raise ValueError("zero-control L3S line is malformed")
            batch[key] = (_words(fields[2], 4), int(fields[3]))
        elif tag == "E3STATE":
            if len(fields) != 4:
                raise ValueError("zero-control state line is malformed")
            batch[key] = (_words(fields[2], 3), _pair(fields[3]))
    if final is None or len(roles) != 3 * len(ROLE_NAMES):
        raise ValueError("zero-control frame omitted final or roles")
    observed = []
    for ordinal, batch in enumerate(batches):
        if set(batch) != {"row", "ce", "batch", "state"}:
            raise ValueError("zero-control frame omitted a batch field")
        ids, targets, mask = batch["row"]
        reduction, active = batch["batch"]
        state, counters = batch["state"]
        if active != sum(mask):
            raise ValueError("zero-control frame active count differs from mask")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    args = parser.parse_args()
    payload, roles = parse(args.stdout.read_bytes())
    args.observed.write_bytes(canonical(payload))
    args.roles.write_bytes(roles)


if __name__ == "__main__":
    main()
