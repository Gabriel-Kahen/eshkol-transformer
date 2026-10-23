#!/usr/bin/env python3
"""Host-only source/layout checks for the gated native capture slice.

This intentionally does not compile or execute the native implementation.
"""
from pathlib import Path
import ctypes
import hashlib
import struct

root = Path(__file__).resolve().parents[2]
source = (root / 'src/eshkol_transformer/cpu4_c4_capture.inc').read_text()
header = (root / 'src/eshkol_transformer/cpu4_c4_capture_private.h').read_text()
native_fixture = (root / 'tests/g3c4/test_cpu4_capture_native.c').read_text()
reference = (root / 'native/c2_checkpoint_format.c').read_text()

def sha_body(text, start):
    return text[text.index(start):text.index('static uint16_t u16(', text.index(start))]

reference_sha = sha_body(reference, 'typedef struct sha256_state {')
capture_sha = source[source.index('typedef struct cpu4_sha256_state {'):
                     source.index('#define CPU4_CAPTURE_MAGIC')]
assert capture_sha.rstrip() == reference_sha.replace(
    'sha256_', 'cpu4_sha256_').replace('rotr32', 'cpu4_rotr32').rstrip()

class Record(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint32), ('major', ctypes.c_uint16),
                ('minor', ctypes.c_uint16),
                *[(n, ctypes.c_uint32) for n in ('flags', 'count', 'capacity',
                   'prompt', 'generated', 'manual', 'eos', 'reserved0')],
                ('epoch', ctypes.c_uint64), ('ids', ctypes.c_int64 * 4),
                ('origins', ctypes.c_uint32 * 4), ('policy', ctypes.c_int64 * 6),
                ('rng', ctypes.c_int64 * 4),
                *[(n, ctypes.c_uint32) for n in ('parameter_count',
                   'values_bytes', 'state', 'reserved1')],
                ('digest', ctypes.c_uint8 * 32)]

class Handle(ctypes.Structure):
    _fields_ = [('next', ctypes.c_void_p), ('magic', ctypes.c_uint64),
                *[(n, ctypes.c_uint32) for n in ('kind', 'state', 'busy', 'pad')]]

class Provenance(ctypes.Structure):
    _fields_ = [('next', ctypes.c_void_p), ('handle', ctypes.c_void_p),
                ('payload', ctypes.c_void_p), ('extent', ctypes.c_size_t),
                ('state', ctypes.c_uint32), ('reserved', ctypes.c_uint32),
                ('generator', ctypes.c_void_p), ('model', ctypes.c_void_p),
                ('cache', ctypes.c_void_p),
                ('parameters', ctypes.c_void_p * 14),
                ('handles', ctypes.c_void_p * 14),
                ('digest', ctypes.c_uint8 * 32)]

assert ctypes.sizeof(Record) == 224 and Record.digest.offset == 192
assert ctypes.sizeof(Handle) == 32
assert ctypes.sizeof(Provenance) == 320 and Provenance.digest.offset == 288
assert 'sizeof(cpu4_capture_payload) == 4992u' in source
assert 'sizeof(cpu4_capture_provenance) == 320u' in source
assert 'cpu4_capture_overlap(dst, bytes)' in source
assert 'cpu4_history_overlap(dst, bytes)' in source
assert 'cpu4_capture_binding_matches_pins(context, &pins)' in source
assert 'et_g3c4_prefill_binding_matches_pins(context)' not in source
assert '#ifdef ET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE\n    if (transport->kind' in source

for name in ('copy_record', 'copy_parameters'):
    function = source[source.index(f'et_cpu4_private_c4_capture_{name}_v1('):]
    assert function.index('cpu4_capture_tls_alias') < function.index(
        'et_g3c4_error_reset_internal')
assert 'cpu4_capture_tls_alias(dst, dst_bytes, sizeof(*dst))' in source
assert 'cpu4_capture_tls_alias(dst, dst_bytes, CPU4_CAPTURE_VALUES)' in source
assert 'cpu4_history_span(dst,fixed' in source
assert 'capture_fault_controls(owner)' in native_fixture
assert 'memset(&node->generator, 0, 31u * sizeof(void *))' in source
assert source.index('if (node->state == CPU4_CAPTURE_DEAD)') < source.index(
    'cpu4_capture_record_valid(node->payload, node->digest)')

# Independent canonical-format fixture; verifies normative field arithmetic.
record_prefix = struct.pack('<IHH8IQ4q4I6q4q4I',
                            224, 1, 0, 0, 2, 4, 1, 1, 0, 2, 0,
                            7, 11, 12, 0, 0, 1, 2, 0, 0,
                            1, 2, 3, 4, 5, 12,
                            6, 7, 8, 9, 14, 4768, 1, 0)
assert len(record_prefix) == 192
values = bytes(4768)
assert hashlib.sha256(b'cpu4-c4-capture-v1\0' + record_prefix + values).hexdigest() == (
    '4180dc8126bb49169af53afc5f1643be8e1cfce7b5879db27c2f312356e29d0d')
assert 'et_cpu4_private_c4_capture_create_v1' in header
assert 'et_cpu4_private_c4_capture_release_v1' in header
assert header.index('#ifdef ET_CPU4_C4_HISTORY_PRIVATE') < header.index(
    'typedef struct et_cpu4_c4_capture')
assert header.rfind('#endif') > header.index(
    'et_cpu4_private_c4_capture_release_v1')
assert '#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN' in native_fixture
assert 'second.count == 3u && second.cache_capacity == 4u' in native_fixture
assert 'et_g3c4_model_pins_begin_internal' in native_fixture
print('PASS: host-only capture source, SHA derivation, ABI layout, and canonical fixture')
