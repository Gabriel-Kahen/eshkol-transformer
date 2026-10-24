"""Strict finite-profile artifact checks; never executes checkpoint content."""
import hashlib
from pathlib import Path
import struct

A125_SHA = '499008516bce12c9ca07de19e97cee99ec373b0070cf13b34d035296889692cc'
C2_DOMAIN = b'eshkol-training-state-container-v1\0'
C1_DOMAIN = b'eshkol-checkpoint-container-v1\0'
METRICS = (940113418, 1082130432, 1065353490, 1065353216, 4, 1, 1000, 250, 249)
BASELINE = (1085315063, 1082130432, 1132021247, 0, 4, 1, 0, 0, 0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(path, updates):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == 25504,
            'fixed-profile regular C2 file required')
    data = path.read_bytes()
    require(data[:16] == bytes.fromhex('894553484b54524e53544154450d0a00'), 'C2 magic')
    require(struct.unpack_from('<HHIIIQ', data, 16) == (1, 0, 256, 1, 1, 0), 'C2 header')
    q = lambda offset: struct.unpack_from('<Q', data, offset)[0]
    require(q(40) == len(data) and q(48) == 256 and q(64) == 256+q(56)
            and q(64)+q(72)+32 == len(data), 'C2 lengths')
    require(data[-32:] == hashlib.sha256(C2_DOMAIN+data[:-32]).digest(), 'C2 checksum')
    require(struct.unpack_from('<IIII', data, 80) == (15, 14, 28, 43), 'C4 tensor/tie counts')
    require(tuple(q(x) for x in (96, 104, 112, 120)) == (10973, 2656, 9536, 1107),
            'fixed complete C1/O2/moments/X1 section sizes')
    require(tuple(q(x) for x in range(184, 240, 8)) ==
            (4*updates, updates, updates-1, 1, 1729, 0, 0), 'controls/RNG/counts')
    require(data[240:256] == b'\0'*16, 'C2 reserved bytes')
    model = data[q(64):q(64)+q(96)]
    require(len(model) >= 160 and model[:16] == bytes.fromhex('894553484b4f4c434b50540d0a1a0a00'), 'nested C1')
    require(model[-32:] == hashlib.sha256(C1_DOMAIN+model[:-32]).digest(), 'C1 checksum')
    offset, size = struct.unpack_from('<QQ', model, 64)
    # C1 repeats the tied 4096-byte embedding: 4768 unique + 4096 alias bytes.
    require(size == 8864 and offset+size+32 == len(model)
            and struct.unpack_from('<II', model, 80) == (15, 1), 'all values and tied alias required')
    moments = data[q(64)+q(96):q(64)+q(96)+q(112)]
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
            'updates': updates, 'serialized_values_with_alias_sha256': hashlib.sha256(model[offset:offset+size]).hexdigest(),
            'moments_sha256': hashlib.sha256(moments).hexdigest()}


def state(phase, updates):
    return f'CPU4-RESUME-STATE {phase} {4*updates} {updates} {updates-1} 4768 9536 {updates} 1065353216'


def words(prefix, values):
    return prefix+' '+' '.join(map(str, values))


def transcript(mode, text):
    if mode == 'produce':
        expected = [words('CPU4-HELDOUT-OBS baseline', BASELINE), state('a125', 125),
                    words('CPU4-HELDOUT-OBS post', METRICS), state('a250', 250),
                    words('CPU4-FRESH-METRICS produce', METRICS), 'CPU4-FRESH-PASS produce 250 0']
    elif mode == 'consume':
        expected = [state('b125', 125), 'CPU4-FRESH-RESTORED 25504 1192 2384', state('b250', 250),
                    words('CPU4-FRESH-METRICS consume', METRICS), 'CPU4-FRESH-PASS consume 125 0']
    else:
        require(mode in ('reject-corrupt', 'reject-identity'), 'unknown transcript mode')
        error = ('invalid-argument cpu4-checkpoint-factory-create' if mode == 'reject-identity'
                 else 'corrupt-data checkpoint-load')
        expected = [f'CPU4-FRESH-REJECT {mode} {error}', f'CPU4-FRESH-PASS {mode} 0 0']
    require(text == '\n'.join(expected)+'\n', 'exact records/metrics/steps/cleanup required: '+mode)


def compare(out, external):
    out, external = Path(out), Path(external)
    original = checkpoint(external, 125)
    require(original['sha256'] == A125_SHA, 'external accepted A125 identity')
    results = {}
    for name, updates in [('producer.a125.c2', 125), ('producer.a250.c2', 250),
                          ('consumer.restored125.c2', 125), ('consumer.b250.c2', 250)]:
        results[name] = checkpoint(out/name, updates)
    require((out/'producer.a125.c2').read_bytes() == external.read_bytes(), 'uninterrupted A125 reference')
    require((out/'consumer.restored125.c2').read_bytes() == external.read_bytes(), 'complete external A125 restore')
    require((out/'producer.a250.c2').read_bytes() == (out/'consumer.b250.c2').read_bytes(),
            'all A250/B250 planes, metadata and checksums must match')
    return results
