#!/usr/bin/env python3
"""Independent masked CE central differences through all 1192 CPU4 values."""
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cpu4_transport'))
import reference as model

TARGETS = (197, 3, 41, 7)
MASK = (1, 0, 1, 1)

def objective(p, seed=None):
    flat = model.forward(p)
    losses = []
    for t, target in enumerate(TARGETS):
        row = flat[t*256:(t+1)*256]
        maximum = max(row)
        losses.append(maximum + math.log(sum(math.exp(x-maximum) for x in row)) - row[target])
    return sum(w*x for w, x in zip(MASK, losses))

p = model.parameters()
logits = model.forward(p)
seed = []
for t, target in enumerate(TARGETS):
    row = logits[t*256:(t+1)*256]
    maximum = max(row)
    exps = [math.exp(x-maximum) for x in row]
    total = sum(exps)
    seed.extend(MASK[t]*(x/total - (j == target)) for j, x in enumerate(exps))
model.objective = objective
gradient = model.vjp(p, seed)
assert all(math.isfinite(x) for values in gradient for x in values)
assert all(max(map(abs, values)) > 1e-7 for values in gradient)
assert sum(MASK) == 3 and all(x == 0 for x in seed[256:512])
for ordinal, values in enumerate(p):
    j = len(values)-1
    model.near(gradient[ordinal][j], model.derivative(p, seed, values, j, step=5e-6))
text = model.header(p, seed, logits, gradient)
text += model.c_array('double', 'cpu4_objective_observation',
                      [objective(p), float(sum(MASK)), objective(p)/sum(MASK)], float.hex)
Path(sys.argv[1]).write_text(text)
print('CPU4 objective oracle PASS: masked CE, 1024 seeds, 1192 central differences,14 directional tails')
