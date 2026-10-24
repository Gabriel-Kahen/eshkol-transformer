#!/usr/bin/env python3
"""Independent canonical C4GRCV1 byte and SHA oracle; no native execution claim."""
import hashlib
from pathlib import Path
import struct

root = Path(__file__).resolve().parents[2]
source = (root / 'src/eshkol_transformer/cpu4_c4_p3_wire.inc').read_text()
header = (root / 'src/eshkol_transformer/cpu4_c4_capture_private.h').read_text()
fixture = (root / 'tests/g3c4/test_cpu4_p3_wire_native.c').read_text()
fp = b'sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704'
assert len(fp) == 96
values = bytes(4768)
fields = bytearray(192)
struct.pack_into('<IHH8IQ', fields, 0, 224, 1, 0, 0, 3, 4, 2, 1, 0, 0, 0, 2)
struct.pack_into('<4q', fields, 48, 3, 197, 3, 0)
struct.pack_into('<4I', fields, 80, 1, 1, 2, 0)
struct.pack_into('<6q', fields, 96, 0, 0x3f800000, 256, 0x3f800000, 2, 41)
struct.pack_into('<4q', fields, 144, 1, 41, 0, 0)
struct.pack_into('<4I', fields, 176, 14, 4768, 1, 0)
capture_digest = hashlib.sha256(b'cpu4-c4-capture-v1\0' + fields + values).digest()
capture_hex = 'cce6f6fb7768bde5a18404adcb2656a993e2f4cb3b5b0a9a5a27c526e00b8ee0'
assert capture_digest.hex() == capture_hex
c2 = bytes(range(32))
prefix = b'C4GRCV1\0' + struct.pack('<II', 400, 1) + fields + capture_digest + c2 + fp
assert len(prefix) == 368
outer_digest = hashlib.sha256(prefix).digest()
assert outer_digest.hex() == '86534e590e84fd563744af7f5ca346962c52702e3d605aee426bc6bd8ea113da'
wire = prefix + outer_digest
assert len(wire) == 400
assert hashlib.sha256(wire).hexdigest() == '9d4cd70114a1a244e192e4c19d1183d2510f007b2a1f405c258e97d414d20883'
assert 'et_cpu4_private_c4_p3_wire_encode_v1(' in source
assert 'et_cpu4_private_c4_p3_wire_decode_v1(' in source
assert 'ET_CPU4_C4_P3_WIRE_PRIVATE' in header
assert capture_hex in fixture and hashlib.sha256(wire).hexdigest() in fixture
assert 'cpu4_capture_record_valid(&payload, payload.record.digest)' in source
assert 'memcpy(out_record, &payload.record, sizeof(*out_record));' in source
print('C4GRCV1 independent byte/SHA oracle PASS: 192/224/368/400, values=4768')
