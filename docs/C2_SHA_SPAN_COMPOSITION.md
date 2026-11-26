# C2-owned C1 SHA span composition witness

Status: opt-in source candidate with one focused supported AOT pass at
`8cc9ca16`/tree `e8043fc1`. The sealed run is in
`/home/gabe/.codex/evidence/eshkol-transformer/c2-sha-span-composition-8cc9ca16-20261126.IjffC7/RESULTS.md`
(manifest SHA256 `44ac8897fe3d791adde447e944fb6caf4ca6bd28490289829596edafbf46c090`).
This branch follows the test-only native span helper from `e9cb6f52`. It changes no public C1/C2/CLI
entry, default C2 adapter callback, wire format, or shipped header.

The accepted C1 builder still receives a procedure callback on ordinary
paths. Its two SHA sites now pass an ordered list of existing bytevectors to
`c1-encode-digest-parts`. A procedure callback takes the original branch:
concatenate the list and invoke the one-argument callback, preserving its
complete-preimage contract and error delivery. Only the C2-private tagged
`(vector 'c1-private-span-sha procedure)` branch receives the individual spans.
The same digest length validation follows both branches.

The opt-in extension in `tests/c2/` passes C1's tensor domain, zero-digest
record, and payload directly to the macro-gated native helper, then passes
the container domain and unsigned container with an empty third span. It
uses the accepted C1 state encoder; it does not reimplement the checkpoint
layout. The appended witness checks exact domain bytes, each zero digest slot,
three tensor calls followed by one container call, byte parity against the
ordinary C1 output, original exception propagation, malformed digest error,
and C2 borrow cleanup.

Light checks:

```sh
bash scripts/test-c2-sha-spans-experiment.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/c2/test_c2_sha_span_composition.py --check
```

After an authorized supported compiler slot, the focused one-AOT recipe is
the host launcher with the reviewed exact commit and tree:

```sh
bash scripts/launch-c2-sha-span-composition.sh EXPECTED_HEAD EXPECTED_TREE
```

The host launcher acquires both shared build locks nonblocking, checks source
identity, runner hash and f31 image digest, then runs offline in the supported
Ubuntu 22.04/LLVM 21.1.8 image. The recipe stages the ordinary model-encode
witness with only the opt-in
  extension and appended assertions. It derives its native build/link commands
from the accepted model-encode gate, adds the experimental format-object
macro, and requires the same pinned toolchain. The exact `8cc9ca16` run passed
its focused AOT witness; moving the Python driver into `tests/c2/` here has
source/recipe and Q0 isolation checks, without a second AOT run. No RSS
improvement is claimed. Before integration, independently review the
new private C1 helper symbols and update exact source/symbol manifests where
required; run the supported C1/C2 package and joint operational gates for
byte identity, failure atomicity, ownership, and peak memory. The prior
native/scalar matched pair is one uncontrolled fixture and does not attribute
its 47,744 KiB RSS gap to SHA preimage buffers.
