# Held original-epoch handoff capsule candidate

`current_tree_candidate.tsv` is a new, unadmitted source recipe for the exact
private source at `793befcea413a1c76efeb374765aaf8484f59578`. It pins the
capsule/snapshot/lease bytes, this fixture, the inherited native aggregate
recipe, the existing allocator shim, and both historical TR3 lease manifests.
The historical manifests are evidence of earlier gates; their `lease_core_sha256`
remains `65e8f97c...` while this source is `28b1d5bd...`. Neither historical
manifest is an admission for the current tree.
The successor recipe carries the inherited `runner_role` in its strict
manifest and evidence output, and requires the fixture's 15-check pass marker.
The source test executes the rendered role expression and pass predicate.

The source-only preflight is:

```sh
python3 scripts/prepare-tr3-handoff-capsule-gate.py --verify-only
python3 -m unittest -v tests.tr3_handoff_capsule.test_runtime_candidate
```

After independent source review, render the held compiled recipe to an external
evidence directory with `--output /absolute/evidence/path/gate.sh`. The renderer
pins the current private source and produces a sibling of the reviewed joint
native aggregate build, with this fixture and an unadmitted result marker. A
later operator must inspect that generated script, use a bounded Docker slot,
record its exact compiler/container/runner/native archive, symbol closure,
stdout/stderr, resource use and SHA256SUMS, then obtain independent runtime
review. Rendering or source preflight alone produces no compiled evidence.

The fixture exercises exact train-mode capture, slot-4 handoff fence, public
mutation and trainer release denial, copied/foreign/stale capsule rejection,
changed copied/live controls, one-shot pre-eval abort, and two dependency
failure cuts including post-P1 release. Those cuts are synthetic dependency
faults. They do **not** prove allocator failure behavior.

The separate genuine allocator matrix remains held: extend the pinned
`tests/tr3_lease_failure/lease_failure_shim.cpp` mechanism to count only the
capsule capture window, then run fresh-process `object`, `all`, `promotion`,
`target`, and `handler` allocation prefixes until the first unhit ordinal.
Each hit must verify the same enrolled trainer/node, idle/train mode, exact
model/14-handle/O2/D2/T2 identity and native bits, unchanged cursor and
counters, zero leaked detached P1 owners, cleared guard and slot 4, and a
successful retry on the same owners. A failure after P1 publication must show
exactly one release. A provider outcome with uncertain custody must fail stop.
The existing lease shim is pinned as a source starting point, not represented
as a completed capsule allocator matrix. Eval transition, G3/O2 holds, C2
claim, trainer release, receipt, SAVE, and replay are outside this candidate.

## Separate capsule allocator-prefix source candidate

`allocator_prefix_candidate.tsv` pins a distinct five-class matrix over the
same accepted capsule production source. Its runner and archive identities
come from the historical **production-OFF allocator repair** manifest at
`de0b249`; they do not replace the inherited functional `81298` runner
in `current_tree_candidate.tsv`. The source preflight compares exact runtime
commit/tree and the complete `81298..de0b249` changed-path set, which is
limited to backend allocator code generation, runtime closure allocation, and
their tests. It also requires the capsule production directories to remain
byte-identical to the accepted `3ea8f80` source basis.

The fixture arms immediately inside an already installed guard, then calls
`tr3-handoff-capture-internal!` on an enrolled idle/train owner. Each ordinal
runs in a fresh process; `object`, `all`, `promotion`, and `target` sweep
through the first unhit ordinal, and `handler` exhausts live guards before
denying the exact handler allocation. Every hit checks original record/node
and fixed owner identities, cursor/RNG/counters, native parameter/gradient/O2
bits, cleared guard and slot 4, P1 live count and exact post-publication
release, then retries and aborts on the same owner. A provider outcome with
unaccounted native custody fails the case. The shell seals per-case output,
symbol closure, compiler inputs, and SHA256SUMS.

Source-only preflight:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/prepare-tr3-handoff-capsule-allocator-gate.py --verify-only
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.tr3_handoff_capsule.test_allocator_prefix_source
bash -n scripts/test-tr3-handoff-capsule-allocator-gate.sh
```

This is an **unadmitted source recipe**. The historical `de0b249` runner
and archive were removed from its evidence root
(`GENERATED_ARTIFACTS_REMOVED.md`); their hashes remain in the historical
manifest. A reviewed reconstruction of those exact bytes or a new pinned
runtime build, source/ABI review of the resulting functional-plus-allocator
pair, exact symbol closure, a bounded compiled five-class run, and
independent runtime evidence review are required before this matrix can be
admitted. No unavailable class or ordinal is counted as a pass.
