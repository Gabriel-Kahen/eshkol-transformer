# Native E3 frame parity witness

This test drives the reviewed native E3 frame directly. It creates the accepted
seed-1729 M3 owner, runs all 21 forward roles, indexed cross entropy, L3S masked
reduction, accuracy, accumulation and four-value final publication, then compares
the observed transcript with the accepted mathematical reference.

Five cases cover packed single-shard and repartitioned inputs, a packed partial
tail, and two unpacked shard layouts. The strict comparator checks per-batch
logits, loss numerator and weight, correct and valid counts, global accumulation,
published counters and final metrics. A test-only role sidecar checks every native
role value and shape against the mathematical reference. The normal binary is run
twice to prove byte-identical output, and the same cases run under ASan and UBSan.

Before every successful case, the test exercises the real zero-weight path on the
same frame. Finalization must return invalid-state, leave all four destination
values and published counters unchanged, clean up successfully, and permit the
subsequent successful retry. The strict transcript schema describes successful
observations and therefore does not encode this negative path.

The fixture rows are generated at test time through the accepted D2 development
reference with pinned Python 3.14.6 and PyTorch 2.13.0+cpu. They are staged into
the native frame by the test. This witness does not execute a D2 borrower or
cursor, Eshkol wrapper, P1 mode, public evaluator, or full E3 traversal.

Run everything on a supported host from the repository root:

```bash
Q0_PYTHON=/absolute/path/to/pinned/python \
  scripts/test-e3-native-parity.sh all
```

The split modes allow reference preparation and strict verification on the pinned
development host while native execution runs in the supported compiler image:

```bash
Q0_PYTHON=/absolute/path/to/pinned/python \
  scripts/test-e3-native-parity.sh prepare /absolute/path/to/output
CC=clang-21 CXX=clang++-21 \
  scripts/test-e3-native-parity.sh native /absolute/path/to/output
Q0_PYTHON=/absolute/path/to/pinned/python \
  scripts/test-e3-native-parity.sh verify /absolute/path/to/output
```
