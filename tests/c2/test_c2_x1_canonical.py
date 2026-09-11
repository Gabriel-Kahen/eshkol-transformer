#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import sys
import tempfile

driver = pathlib.Path(sys.argv[1])
root = pathlib.Path(__file__).resolve().parents[2]
golden = (root / "tests/x1/fixtures/resolved_minimal_v1.json").read_bytes()

def fp(data):
    return b"sha256:eshkol-config-json-v1:" + hashlib.sha256(data).hexdigest().encode()

def run(data, expected=None):
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td)
        (p / "x").write_bytes(data)
        (p / "f").write_bytes(fp(data) if expected is None else expected)
        text = subprocess.check_output([driver, p / "x", p / "f"], text=True)
        return tuple(map(int, text.strip().split(',')))

checks = 0
def expect(data, status, code=None, expected=None):
    global checks
    got = run(data, expected)
    assert got[0] == status, (status, got, data[:160])
    if code is not None: assert got[1] == code, (code, got)
    checks += 1
    return got

ok = expect(golden, 0)
assert ok[2:] == (256, 128, 64, 2, 4, 4, 16, 1729, 1)
expect(golden, 2, 6, b"sha256:eshkol-config-json-v1:" + b"0" * 64)
expect(golden, 2, 6, fp(golden).upper())
expect(golden, 2, 6, fp(golden) + b"\n")
expect(b"x" * 16385, 2, 2)
expect(golden[:-1] + b"\xff", 2, 3)

def replace(old, new, status=2, code=4):
    assert old in golden
    expect(golden.replace(old, new, 1), status, code)

replace(b'"format-version":[1,0]', b'"format-version":[2,0]', 3, 5)
replace(b'"config-schema-version":[1,0]', b'"config-schema-version":[1,1]', 3, 5)
replace(b'"required-features":[]', b'"required-features":["future"]', 3, 5)
replace(b'"format-version":[1,0]', b'"format-version":"2.0"')
replace(b'"model.device":"cpu"', b'"model.device":"gpu"', 8, 4)
replace(b'"model.dtype":"f32"', b'"model.dtype":"f64"', 7, 4)
replace(b'"checksum-algorithm":"sha256"', b'"checksum-algorithm":"sha1"')
replace(b'"model.context-length":128', b'"model.context-length":0', 2, 4)
replace(b'"model.hidden-size":64', b'"model.hidden-size":65', 6, 8)
replace(b'"model.kv-head-count":4', b'"model.kv-head-count":3', 6, 8)
replace(b'"run.seed":1729', b'"run.seed":01729')
replace(b'"run.seed":1729', b'"run.seed":9223372036854775808')
replace(b'"model.head-size":"derived"', b'"model.head-size":"default"', 2, 4)
replace(b'"model.kv-head-count":"default"', b'"model.kv-head-count":"derived"', 2, 4)
replace(b'"model.kv-head-count":4', b'"model.kv-head-count":2', 2, 7)
replace(b'"training.accumulation-steps":1', b'"training.accumulation-steps":2', 2, 7)
replace(b'"model.head-size":16', b'"model.head-size":8', 6, 8)
replace(b'"run.deterministic":true', b'"run.deterministic":false')

provenance_keys = (
    b"config-schema-major", b"config-schema-minor", b"model.context-length",
    b"model.device", b"model.dtype", b"model.head-size", b"model.hidden-size",
    b"model.kv-head-count", b"model.layer-count", b"model.query-head-count",
    b"model.vocabulary-size", b"run.deterministic", b"run.seed",
    b"training.accumulation-steps",
)
for key in provenance_keys:
    marker = b'"' + key + b'":"'
    start = golden.index(marker) + len(marker)
    end = golden.index(b'"', start)
    mutant = golden[:start] + b"bogus" + golden[end:]
    expect(mutant, 2, 4)

# Alternate allowed provenance remains identity-bearing and is accepted when its
# corresponding semantic constraint permits the resolved value.
valid = golden.replace(b'"model.context-length":"input"',
                       b'"model.context-length":"override"', 1)
expect(valid, 0)
valid = golden.replace(b'"model.kv-head-count":"default"',
                       b'"model.kv-head-count":"input"', 1).replace(
                           b'"model.kv-head-count":4', b'"model.kv-head-count":2', 1)
assert expect(valid, 0)[7] == 2
valid = golden.replace(b'"training.accumulation-steps":"default"',
                       b'"training.accumulation-steps":"input"', 1).replace(
                           b'"training.accumulation-steps":1',
                           b'"training.accumulation-steps":2', 1)
assert expect(valid, 0)[10] == 2

# Integrity is established before semantic taxonomy.
broken_version = golden.replace(b'"format-version":[1,0]',
                                b'"format-version":[2,0]', 1)
expect(broken_version, 2, 6, b"sha256:eshkol-config-json-v1:" + b"0" * 64)
expect(golden.replace(b',"format-version"', b', "format-version"', 1), 2, 4)
expect(golden.replace(b'}}\n', b'}}\r\n'), 2, 4)
expect(golden + b'\n', 2, 4)
for cut in (0, 1, 16, len(golden) - 1): expect(golden[:cut], 2)
for i in range(0, len(golden), 31):
    mutant = bytearray(golden); mutant[i] ^= 1
    got = run(bytes(mutant))
    assert got[0] in (2, 3, 7, 8), got
    checks += 1
print(f"c2 x1 canonical adversarial: PASS ({checks} checks)")
