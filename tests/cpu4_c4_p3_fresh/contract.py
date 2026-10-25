"""Fixed B250/P3 FILE contract shared by the sealed driver and host negatives."""
import hashlib
import json
from pathlib import Path
import struct

from tests.cpu4_trained_g2.normal_driver import plan as trained_plan

OWNER = 'src/eshkol_transformer/g3c4_model_owner.c'
FEATURES = ('ET_CPU4_C4_HISTORY_PRIVATE', 'ET_CPU4_C4_P3_WIRE_PRIVATE',
            'ET_CPU4_C4_P3_RESUME_PRIVATE',
            'ET_CPU4_C4_P3_BOXED_BRIDGE_PRIVATE')
BRIDGES = ('native/cpu4_c4_p3_resume_ffi.c',
           'native/cpu4_c4_p3_wire_decode_ffi.c')
T1_FP = b'sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704'
REFERENCE_RESULT = b'CPU4-TRAINED-G2-RESULT second-eos 41 2 4 3 41 3 41'
REFERENCE_RNG = b'CPU4-TRAINED-G2-RNG 1 41 0 0'
RESUMED_RESULT = b'CPU4-P3-RESUMED 41 2 4 3 41 3 41 1 41 0 0'


def units(roster: dict, normal_overlay: dict) -> list[dict]:
    selected = trained_plan(roster, normal_overlay)
    owners = [unit for unit in selected if unit['source'] == OWNER]
    if len(owners) != 1 or any(feature in roster['common_defines'] for feature in FEATURES):
        raise ValueError('C4 P3 features require one private native owner')
    if any(feature in unit['defines'] for unit in selected for feature in FEATURES):
        raise ValueError('C4 P3 features already set outside overlay')
    owners[0]['defines'].extend(FEATURES)
    selected.extend({'source': source, 'defines': []} for source in BRIDGES)
    if len({unit['source'] for unit in selected}) != len(selected):
        raise ValueError('duplicate private C4 bridge translation unit')
    if sum(all(feature in unit['defines'] for feature in FEATURES)
           for unit in selected) != 1:
        raise ValueError('C4 P3 feature overlay is not sole-owner')
    if len(selected) != 39 or any(sum(unit['source'] == source for unit in selected) != 1
                                  for source in BRIDGES):
        raise ValueError('C4 P3 requires exactly two distinct boxed bridge units')
    return selected


def validate_wire(wire: bytes, values: bytes, c2_sha: bytes) -> None:
    if len(wire) != 400 or len(values) != 4768 or len(c2_sha) != 32:
        raise ValueError('C4 P3 declared extent changed')
    if (wire[:8] != b'C4GRCV1\0' or struct.unpack_from('<II', wire, 8) != (400, 1)
            or wire[240:272] != c2_sha or wire[272:368] != T1_FP):
        raise ValueError('C4 P3 outer identity changed')
    if hashlib.sha256(wire[:368]).digest() != wire[368:400]:
        raise ValueError('C4 P3 outer digest changed')
    fields = wire[16:208]
    if hashlib.sha256(b'cpu4-c4-capture-v1\0' + fields + values).digest() != wire[208:240]:
        raise ValueError('C4 P3 captured model digest changed')
    if (struct.unpack_from('<IHH8IQ', fields) !=
            (224, 1, 0, 0, 3, 4, 2, 1, 0, 0, 0, 2)
            or struct.unpack_from('<4q', fields, 48) != (3, 197, 3, 0)
            or struct.unpack_from('<4I', fields, 80) != (1, 1, 2, 0)
            or struct.unpack_from('<6q', fields, 96) !=
            (0, 0x3f800000, 256, 0x3f800000, 2, 41)
            or struct.unpack_from('<4q', fields, 144) != (1, 41, 0, 0)
            or struct.unpack_from('<4I', fields, 176) != (14, 4768, 1, 0)):
        raise ValueError('C4 P3 fixed trained profile changed')


def validate_reference(stdout: bytes) -> None:
    from tests.cpu4_trained_g2.check_output import check
    check(stdout.decode(), 'second-eos')
    if (stdout.splitlines().count(REFERENCE_RESULT) != 1 or
            stdout.splitlines().count(REFERENCE_RNG) != 1):
        raise ValueError('uninterrupted trained G2 reference changed')


def validate_resumed(stdout: bytes, reference: bytes) -> None:
    validate_reference(reference)
    reference_lines = reference.splitlines()
    expected = (b'CPU4-P3-RESUMED ' +
                REFERENCE_RESULT.split(b'second-eos ', 1)[1] + b' ' +
                REFERENCE_RNG.split(b'CPU4-TRAINED-G2-RNG ', 1)[1])
    if (reference_lines.count(REFERENCE_RESULT) != 1 or
            reference_lines.count(REFERENCE_RNG) != 1 or
            expected != RESUMED_RESULT):
        raise ValueError('uninterrupted reference cannot bind P3 result')
    lines = stdout.splitlines()
    if (lines.count(expected) != 1 or
            not lines[-1].startswith(b'CPU4-P3-FRESH-PASS resume ')):
        raise ValueError('fresh FILE resumed result differs from reference')


def normal_overlay(path: Path) -> dict:
    return json.loads(path.read_text())
