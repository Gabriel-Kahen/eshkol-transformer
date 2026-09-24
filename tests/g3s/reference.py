"""Independent G3-S development oracles; no production import or dependency.

The literal oracle rounds each arithmetic statement to IEEE binary32 and calls
this execution lane's expf, whose last bits are deliberately libm-scoped. The
separate Decimal(80) mathematical oracle does not call expf or reuse the literal
arithmetic. Mathematical probabilities use absolute tolerance 2e-5 (256-term
serial binary32 accumulation); discrete boundary choices have no tolerance.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from decimal import Decimal, localcontext
import json
from pathlib import Path
import struct

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
DOMAIN = 0x4733434154454731
_libm = ctypes.CDLL(ctypes.util.find_library('m'))
_libm.expf.argtypes = [ctypes.c_float]
_libm.expf.restype = ctypes.c_float


def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return float('-inf') if value < 0 else float('inf')


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def word(value):
    return struct.unpack('<f', struct.pack('<I', value))[0]


def philox(seed, low, high, domain=DOMAIN):
    # Independent integer description: 128-bit product halves and a tuple state.
    key = seed ^ domain
    keys = [key & MASK32, key >> 32]
    lanes = (low & MASK32, (low >> 32) & MASK32,
             high & MASK32, (high >> 32) & MASK32)
    for round_index in range(10):
        p0, p1 = lanes[0] * 0xd2511f53, lanes[2] * 0xcd9e8d57
        lanes = ((p1 >> 32) ^ lanes[1] ^ keys[0], p1 & MASK32,
                 (p0 >> 32) ^ lanes[3] ^ keys[1], p0 & MASK32)
        if round_index != 9:
            keys = [(keys[0] + 0x9e3779b9) & MASK32,
                    (keys[1] + 0xbb67ae85) & MASK32]
    return lanes


def sum32(values):
    total = 0.0
    for value in values:
        total = f32(total + value)
    return total


def literal(logits, temperature, k, p, uniform, mutation=None):
    maximum = max(logits)
    exponentials = [_libm.expf(f32(f32(x - maximum) / temperature)) for x in logits]
    total = sum32(exponentials if mutation != 'reverse_sum' else exponentials[::-1])
    if mutation == 'binary64_sum':
        total = sum(exponentials)
    a = [f32(x / total) for x in exponentials]
    order = sorted(range(256), key=lambda i: (-a[i], -i if mutation == 'reverse_tie' else i))
    if mutation == 'logit_sort':
        order = sorted(range(256), key=lambda i: (-logits[i], i))
    top = order[:k]
    ksum = sum32(a[i] for i in top)
    b = [f32(a[i] / ksum) for i in top]
    if mutation == 'skip_renormalize':
        b = [a[i] for i in top]
    prefix = k
    cumulative = 0.0
    for i, value in enumerate(b):
        cumulative = f32(cumulative + value)
        if cumulative >= p:
            prefix = i + 1
            break
    if mutation == 'p1_keep_all' and p == 1:
        prefix = k
    weight = sum32(b[:prefix])
    threshold = f32(uniform * weight)
    cumulative = 0.0
    token = None
    cdf = []
    for i, value in enumerate(b[:prefix]):
        cumulative = f32(cumulative + value)
        cdf.append(cumulative)
        crossing = cumulative >= threshold if mutation == 'nonstrict_cdf' else cumulative > threshold
        if token is None and crossing:
            token = top[i]
    if token is None:
        token = top[prefix-1] if mutation == 'last_tail' else next(top[i] for i in range(prefix-1, -1, -1) if b[i] > 0)
    return dict(a=a, order=order, b=b, prefix=prefix, weight=weight,
                threshold=threshold, cdf=cdf, token=token, total=total, ksum=ksum)


def mathematical(logits, temperature, k):
    with localcontext() as context:
        context.prec = 80
        x = [Decimal(v) for v in logits]
        maximum = max(x)
        exponentials = [((v - maximum) / Decimal(temperature)).exp() for v in x]
        total = sum(exponentials)
        probabilities = [v / total for v in exponentials]
        # Mathematical ranking/normalization is independent of f32 ties/order.
        ordered = sorted(probabilities, reverse=True)[:k]
        retained = sum(ordered)
        return ([float(v) for v in probabilities],
                [float(v / retained) for v in ordered])


def cases():
    result = []
    def add(name, logits, temperature=1.0, k=256, p=1.0, seed=0, low=0, high=0):
        result.append(dict(name=name, logits=list(map(f32, logits)), temperature=f32(temperature), k=k, p=f32(p), seed=seed, low=low, high=high))
    add('uniform', [0]*256)
    add('strict_cdf_equality', [0]*256, low=207528)
    add('uniform_max_uniform', [0]*256, low=3850303)
    add('uniform_zero_uniform', [0]*256, low=9773243)
    add('underflow_max_uniform', [0,-1,-90]+[-104]*253, low=3850303)
    add('underflow_zero_uniform', [0,-1,-90]+[-104]*253, low=9773243)
    add('signed_zero', [-0.0, 0.0]*128)
    add('k_one_high_id', [-5]*255+[3], k=1)
    add('equal_max_low_id', [2, 2]+[-1000]*254, p=0.5)
    add('p_equality', [0]*256, p=0.5)
    add('p_below_equality', [0]*256, p=word(0x3effffff))
    add('p_above_equality', [0]*256, p=word(0x3f000001))
    add('underflow_tail', [0, -1, -90]+[-104]*253)
    add('minimum_positive_temperature', [0]*256, temperature=word(1))
    add('minimum_temperature_nonzero_delta', [0]+[-word(1)]*255, temperature=word(1))
    add('maximum_finite_equal_logits', [word(0x7f7fffff)]*256)
    add('minimum_positive_p', [0]*256, p=word(1))
    add('maximum_temperature', list(range(-128,128)), temperature=word(0x7f7fffff))
    add('rounding_tie', [0, word(1)]+[-10]*254)
    add('distinct_key_counter_lanes', [0]*256, seed=0x123456789abcdef, low=0xfedcba9876543210, high=0x0123456789abcdef)
    add('carry_low', [0]*256, seed=13, low=MASK64, high=15)
    add('last_consumable', [0]*256, seed=(1<<63)-1, low=MASK64-1, high=MASK64)
    for salt in range(24):
        x = [f32((((i*37+salt*19)%127)-63)/7) for i in range(256)]
        add('asymmetric_%02d'%salt, x, temperature=[0.125,0.75,1,3][salt%4],
            k=[1,2,7,63,128,256][salt%6], p=[0.125,0.5,0.9,1][salt%4],
            seed=salt*971+19, low=salt*17, high=salt*3)
    # A geometric tail causes rounded cumulative b to hit one before k.
    add('p1_early', [0,-17]+[-1000]*254)
    # Full rounded cumulative is below one: the specified fallback keeps all k.
    add('p1_full_sum_fallback', [f32((i%11)/3) for i in range(256)], k=7)
    return result


def literal_fixed_checks():
    # Fixed integer vectors are literal checked-in data, never generated expectations.
    vectors = [
        (0,207528,0,(0x8f00006d,0x89ec72d0,0xc48478d4,0xf9cc0ffc)),
        (0,3850303,0,(0xffffff89,0x2d82719a,0x851fac5c,0x1a3ee311)),
        (0,9773243,0,(0x00000031,0x2fa183d2,0xcc48c00d,0xbe5e08a8)),
        (0,0,0,(0x8a3dbc56,0xf4ee7332,0x3d5889ed,0x84828c57)),
        (0x123456789abcdef,0xfedcba9876543210,0x0123456789abcdef,(1153277677, 4266973796, 821479019, 2652017347)),
        (1,0,0,(0x443838b2,0x1ae97503,0x6fe26bdc,0x25b76297)),
        (13,MASK64,15,(0x5605e713,0xf2230d7a,0xefc43fcf,0x8a979c82)),
        ((1<<63)-1,MASK64-1,MASK64,(0x7158b07c,0xafc9a6c1,0x55d85a0f,0xa02683ac)),
    ]
    assert philox(0,0,0,domain=0) == (0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8)
    for seed,low,high,expected in vectors:
        assert philox(seed,low,high) == expected, (philox(seed,low,high), expected)
    equal = literal([0.0]*256,1,256,0.5,0.5)
    assert equal['prefix'] == 128 and equal['token'] == 64
    assert bits(equal['a'][0]) == 0x3b800000
    assert literal([0.0]*256,1,256,1,0)['token'] == 0
    assert literal([0.0]*256,1,256,1,word(0x3f7fffff))['token'] == 255
    assert literal([0.0]*256,1,256,1,143/256)['token'] == 143
    early = literal([0,-17]+[-1000]*254,1,256,1,0.5)
    assert early['prefix'] == 1 and bits(early['weight']) == 0x3f800000
    fallback = literal([f32((i%11)/3) for i in range(256)],1,7,1,0.5)
    assert fallback['prefix'] == 7 and bits(fallback['weight']) == 0x3f7fffff
    assert [bits(v) for v in fallback['b']] == [0x3e124924]*7


def mutation_witnesses():
    # These reference variants establish that the fixture corpus distinguishes
    # the semantics; production mutation runs separately exercise the C checks.
    mutations = ('reverse_sum','binary64_sum','reverse_tie','logit_sort',
                 'skip_renormalize','p1_keep_all','nonstrict_cdf')
    witnesses = {}
    for mutation in mutations:
        for case in cases():
            args = (case['logits'],case['temperature'],case['k'],case['p'],0.5)
            if literal(*args) != literal(*args,mutation=mutation):
                witnesses[mutation] = case['name']
                break
        assert mutation in witnesses, mutation
    tagged = philox(0,0,0)
    assert tagged != philox(0,0,0,domain=0)
    assert (tagged[0] >> 8) != (tagged[0] & 0xffffff)
    assert (tagged[0] >> 8) != (tagged[1] >> 8)
    print('G3-S discriminating reference variants: '+json.dumps(witnesses,sort_keys=True))


def emit(path):
    rows = cases()
    lines = ['/* Generated by development-only independent reference.py. */',
             '#include <stdint.h>',
             'typedef struct { const char *name; uint32_t logits[256], temperature, p; int64_t k; uint64_t seed, low, high; uint32_t lanes[4], a[256], b[256], weight, threshold; unsigned order[256], prefix, token; double mathematical[256], mathematical_b[256]; } g3s_reference_case;',
             'static const g3s_reference_case g3s_cases[] = {']
    maximum_error = 0.0
    def nums(values): return '{'+','.join(str(v) for v in values)+'}'
    for case in rows:
        lanes = philox(case['seed'],case['low'],case['high'])
        uniform = f32((lanes[0] >> 8) * 2**-24)
        trace = literal(case['logits'],case['temperature'],case['k'],case['p'],uniform)
        oracle, oracle_b = mathematical(case['logits'],case['temperature'],case['k'])
        error = max(max(abs(x-y) for x,y in zip(trace['a'],oracle)),
                    max(abs(x-y) for x,y in zip(trace['b'],oracle_b)))
        assert error <= 2e-5, (case['name'],error)
        maximum_error = max(maximum_error,error)
        values = [json.dumps(case['name']), nums(map(bits,case['logits'])),str(bits(case['temperature'])),str(bits(case['p'])),str(case['k']),
                  str(case['seed'])+'ULL',str(case['low'])+'ULL',str(case['high'])+'ULL',nums(lanes),nums(map(bits,trace['a'])),
                  nums(list(map(bits,trace['b']))+[0]*(256-case['k'])),str(bits(trace['weight'])),str(bits(trace['threshold'])),
                  nums(trace['order']),str(trace['prefix']),str(trace['token']), nums(format(v,'.17g') for v in oracle),nums(format(v,'.17g') for v in oracle_b+[0.0]*(256-case['k']))]
        lines.append('{'+','.join(values)+'},')
    lines += ['};', 'static const uint32_t continuation_lanes[16][4] = {']
    for step in range(16):
        counter = ((4 << 64) | (MASK64-2)) + step
        lines.append(nums(philox(7,counter & MASK64,counter >> 64))+',')
    lines += ['};', '#define G3S_REFERENCE_COUNT (sizeof(g3s_cases)/sizeof(g3s_cases[0]))']
    path.write_text('\n'.join(lines)+'\n')
    print('G3-S independent Decimal80 reference: %d fixtures, max absolute probability error %.9g, tolerance 2e-5'%(len(rows),maximum_error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--header',type=Path,required=True)
    args = parser.parse_args()
    literal_fixed_checks()
    mutation_witnesses()
    emit(args.header)
