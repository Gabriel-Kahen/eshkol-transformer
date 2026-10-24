"""Strict receipt parser for producer and fresh CPU4 B250 parameter planes."""
import hashlib
import json
from pathlib import Path

SIZES = (64, 64, 64, 64, 128, 128, 16, 16, 16, 16, 4096, 16, 16, 64)
assert sum(SIZES) == 4768


def planes(stdout: bytes, prefix: str) -> tuple[bytes, ...]:
    label = prefix.encode() + b' '
    rows = [line for line in stdout.splitlines() if line.startswith(label)]
    if len(rows) != 14:
        raise ValueError(f'{prefix}: expected 14 complete rows, got {len(rows)}')
    result = []
    for index, (row, size) in enumerate(zip(rows, SIZES)):
        parts = row.split()
        if len(parts) != size + 3 or parts[0] != prefix.encode() or parts[1] != str(index).encode() or parts[2] != str(size).encode():
            raise ValueError(f'{prefix}: noncanonical row {index}')
        if any(not p.isdigit() or (len(p) > 1 and p[0] == 48) for p in parts[3:]):
            raise ValueError(f'{prefix}: noncanonical byte at {index}')
        values = tuple(int(p) for p in parts[3:])
        if any(v > 255 for v in values):
            raise ValueError(f'{prefix}: byte outside u8 at {index}')
        result.append(bytes(values))
    return tuple(result)


def seal(stdout: bytes, prefix: str, output: Path) -> dict:
    raw = planes(stdout, prefix)
    values = b''.join(raw)
    output.mkdir(mode=0o700)
    (output / 'producer.stdout').write_bytes(stdout)
    (output / 'planes.bin').write_bytes(values)
    meta = {'schema': 'cpu4-b250-live-plane-witness-v1',
            'source_stdout_sha256': hashlib.sha256(stdout).hexdigest(),
            'plane_lengths': list(SIZES),
            'plane_sha256': [hashlib.sha256(p).hexdigest() for p in raw],
            'values_bytes': len(values),
            'values_sha256': hashlib.sha256(values).hexdigest()}
    (output / 'planes.json').write_text(json.dumps(meta, sort_keys=True, indent=2) + '\n')
    return meta


def compare(producer: Path, fresh_stdout: bytes, prefix: str) -> None:
    meta = json.loads((producer / 'planes.json').read_text())
    expected = (producer / 'planes.bin').read_bytes()
    producer_stdout = (producer / 'producer.stdout').read_bytes()
    actual_planes = planes(fresh_stdout, prefix)
    if meta['schema'] != 'cpu4-b250-live-plane-witness-v1' or meta['plane_lengths'] != list(SIZES) or len(expected) != 4768:
        raise ValueError('producer witness format changed')
    if hashlib.sha256(expected).hexdigest() != meta['values_sha256']:
        raise ValueError('producer witness digest changed')
    if hashlib.sha256(producer_stdout).hexdigest() != meta['source_stdout_sha256'] or \
            b''.join(planes(producer_stdout, 'CPU4-LIVE-B250-PLANE')) != expected:
        raise ValueError('producer stdout witness changed')
    if [hashlib.sha256(p).hexdigest() for p in actual_planes] != meta['plane_sha256']:
        raise ValueError('fresh factory plane digest differs from live final B')
    if b''.join(actual_planes) != expected:
        raise ValueError('fresh factory plane bytes differ from live final B')
