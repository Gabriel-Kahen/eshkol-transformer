#!/usr/bin/env python3
"""Validate and summarize the unchanged PA4-Wider paired benchmark output."""
import json
import re
import statistics
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text()
shapes = dict((int(i), list(map(int, dims.split(',')))) for i, dims in
              re.findall(r'^SHAPE (\d+) \[([\d,]+)\]$', raw, re.M))
assert len(shapes) == 6 and set(shapes) == set(range(6))
samples = {}
for line in raw.splitlines():
    if not line.startswith('SAMPLE '):
        continue
    _, shape, operation, scope, pair, order, ref, online = line.split()
    key = int(shape), operation, scope
    samples.setdefault(key, []).append((int(pair), order, int(ref), int(online)))
assert len(samples) == 36 and sum(map(len, samples.values())) == 432
assert raw.rstrip().endswith('PA4-Wider benchmark: PASS (432 pairs, 864 timed samples verified)')

results = []
all_ratios = []
for shape in range(6):
    for operation in ('forward', 'backward', 'joint'):
        for scope in ('dispatch', 'invoke'):
            pairs = samples[shape, operation, scope]
            assert len(pairs) == 12
            assert [(p, order) for p, order, _, _ in pairs] == [
                (p, 'AB' if p % 2 == 0 else 'BA') for p in range(12)]
            assert all(ref > 0 and online > 0 for _, _, ref, online in pairs)
            ref_ns = sorted(ref for _, _, ref, _ in pairs)[5] / 100
            online_ns = sorted(online for _, _, _, online in pairs)[5] / 100
            ratios = [online / ref for _, _, ref, online in pairs]
            all_ratios.extend(ratios)
            results.append({
                'shape_index': shape, 'shape': shapes[shape],
                'operation': operation, 'scope': scope,
                'reference_lower_median_ns': ref_ns,
                'online_lower_median_ns': online_ns,
                'online_over_reference_lower_median_ratio': online_ns / ref_ns,
                'pair_ratio_min': min(ratios), 'pair_ratio_max': max(ratios),
                'pair_ratio_median': statistics.median(ratios),
                'ab_ratio_min': min(ratios[::2]), 'ab_ratio_max': max(ratios[::2]),
                'ba_ratio_min': min(ratios[1::2]), 'ba_ratio_max': max(ratios[1::2]),
            })

print(json.dumps({'pairs': 432, 'timed_samples': 864,
                  'all_pair_ratio_min': min(all_ratios),
                  'all_pair_ratio_max': max(all_ratios),
                  'results': results}, indent=2))
