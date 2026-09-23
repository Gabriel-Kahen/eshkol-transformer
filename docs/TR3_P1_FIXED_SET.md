# TR3 P1 fixed-set leaf

This private leaf captures and revalidates the fixed M3 training set needed by
TR3. It adds no public name, registry, token, pin, or lifetime owner. The outer
TR3 lease remains responsible for authenticating the M3T entry and passing its
exact slot-8 handle vector.

## Private contract

The canonical P1 lexical surface remains append-only. Its accepted E3 prefix is
unchanged through slot 68, with these two source-private closures appended:

| Slot | Wrapper | Arguments | Result |
|---:|---|---|---|
| 69 | `tr3-p1-fixed-capture-internal` | model, supplied handles[14], modules[17], retained handles[14], carriers[14] | i64 status |
| 70 | `tr3-p1-fixed-recheck-internal` | model, modules[17], retained handles[14], carriers[14] | i64 status |

Capture requires the exact finalized CPU-f32 fixed profile, all 17 modules in
`train` mode, the genuine registered shells for all 14 supplied handles, 14
distinct carriers, and the head/token-embedding weight tie. It writes only the
three caller-owned output vectors. Recheck is read-only and validates the same
identity, topology, mode, provider, path, shape, dtype, device, carrier, and tie
facts.

Statuses are fixed:

| Status | Meaning |
|---:|---|
| 0 | exact fixed set accepted |
| 1 | malformed/aliased vectors, wrong shell kind, foreign model, or wrong supplied handle shell |
| 2 | protected comparison reentry, unfinalized module, or non-`train` mode |
| 3 | capture-time fixed-profile/topology/provider/path/shape mismatch |
| 4 | registered raw identity is malformed or the captured set drifted |

Rejected capture may have written a validated prefix into unpublished caller
scratch vectors. No rejected call publishes or retains those vectors.

## Fixed profile

The 17 module identities are root, `blocks`, block `0`, its `attention` node and
four projections, its `ffn` node and two projections, `norm1`, `norm2`, `head`,
`norm_final`, `position_embedding`, and `token_embedding`. Child counts,
parents, empty buffer lists, finalized state, and exact parameter-binding counts
are all checked.

The 14 retained handles are attention key/output/query/value weights `[4,4]`,
FFN down `[4,8]` and up `[8,4]` weights, norm1 and norm2 bias/weight `[4]`, head
weight `[256,4]`, final-norm bias/weight `[4]`, and position embedding weight
`[2,4]`. Head weight must be the same raw handle as token-embedding weight.

## Evidence

`scripts/check-tr3-p1-fixed.py` pins slots 69/70, exact unrolled capture counts,
the unchanged first 69 P1 closures and E3 helpers, the unchanged public surface,
the two-file private source closure, and the absence of shell construction,
public projection, native callbacks, exception construction, or predecessor
manifest edits.

`scripts/test-tr3-p1-fixed.sh` pins runtime commit
`81298b4a9608fb92eb6f351a2eabd8392da7d9ef`, tree
`7669312845a9d8d372006af52271045e69505813`, its production runner/archive
hashes, and the checked LLVM 21 container digest. It builds the native fixture
from repository source with the network disabled, compiles the genuine M3T
constructor at O0, and runs 55 focused checks plus 1,024 and 8,192
capture/recheck iterations with
arena poisoning and the f32 allocator armed to fail immediately. The test
requires exact zero deltas for the successful capture, successful recheck, full
loop, P1 live entries, and P1 tombstones, plus equal retained arena bytes across
both horizons. The frozen candidate measured 5,767,168 retained bytes at both
horizons. The unchanged E3 slots also pass their 749-check ordinary mode smoke
under the same runtime candidate. An independent Sol/high fresh review approved
the corrected leaf after checking status precedence, malformed-shape safety,
partial-write semantics, allocation evidence, and manifest/public-surface
preservation.

This leaf does not authenticate an M3T entry, install the O2 overlap hook, own a
lease registry/token, pin D2 borrows, or implement trainer lifetime. Those are
downstream TR3 responsibilities. The allocation evidence is candidate O0 AOT;
it makes no optimized-performance claim. Opaque P1 APIs intentionally provide
no way to inject malformed finalized raw topology or break the internal
head/token tie, so those unreachable corrupt-raw branches are pinned by source
checks rather than a raw mutation test seam. Full CI was outside this bounded
leaf task.
