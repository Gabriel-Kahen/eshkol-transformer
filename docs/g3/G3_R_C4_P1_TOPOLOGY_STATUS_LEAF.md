# G3-R C4 P1 topology status leaf

**Source candidate; compiled fixture not run.** This append-only private P1
slot 74 accepts a model shell and fourteen ordered handle shells. It returns
only a closed integer status: `0` means the exact P1 topology was observed,
`1` means an invalid supplied shell/vector or mismatched ordered handle,
`2` means construction/comparison is active or a module is unfinalized/not in
train mode, and `3`/`4` report malformed module graph or handle/provider
metadata. It neither returns a carrier nor calls `module-parameters`, a
finalizer, or construction mutation.

The check walks the rooted seventeen-module tree through the existing P1
fixed-profile reader, verifies canonical parent/child/binding identities,
CPU f32 I2 provider, train mode, fourteen distinct handles and carriers,
all paths and shapes, and the exact `token_embedding.weight` to
`head.weight` P1 handle tie. C4 position index 13 is `[4,4]`; the inherited
TR3 profile is `[2,4]`. It rechecks the module graph and handles before
returning. Slots 0–73 and the public feature-off module remain unchanged.

The source-only fixture covers exact C4, foreign root, malformed and swapped
supplied handles, a forged supplied head shell, the genuine C2 `[2,4]`
position shape, and eval/train mode. It also suppresses the constructor's
tie registration and requires that malformed construction to fail without a
live entry. The source checker pins the P1 tie equality branch, shape
distinction, read-only calls, slot mapping, and ordinary ABI. The forged head
shell does **not** corrupt a finalized P1 binding; a controlled changed-live-
binding negative remains for a later test-only authority. This leaf also
does not authenticate the native owner-to-P1 association, active C4 call or
lease exclusion, or a workspace. Those remain the separate obligations in
the [workspace authority contract](G3_R_C4_WORKSPACE_TOPOLOGY_AUTHORITY_CONTRACT.md).
