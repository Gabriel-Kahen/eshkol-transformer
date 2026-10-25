# G3-R C4 P1 topology status leaf

**Bounded compiled slot-74 status accepted.** This append-only private P1
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

The fixture covers exact C4, foreign root, malformed and swapped
supplied handles, a forged supplied head shell, the genuine C2 `[2,4]`
position shape, and eval/train mode. It also suppresses the constructor's
tie registration through a private default-I2 delegate and requires that
malformed construction to fail without a
live entry. The source checker pins the P1 tie equality branch, shape
distinction, read-only calls, slot mapping, and ordinary ABI. The forged head
shell does **not** corrupt a finalized P1 binding; a controlled changed-live-
binding negative remains for a later test-only authority. This leaf also
does not authenticate the native owner-to-P1 association, active C4 call or
lease exclusion, or a workspace. Those remain the separate obligations in
the [workspace authority contract](G3_R_C4_WORKSPACE_TOPOLOGY_AUTHORITY_CONTRACT.md).

The first compiled fixture attempt at `9772674e` stopped before IR because
its direct `set!` of the loaded I2 tied-registration function was undefined
to the compiler. Its sealed receipt is
`g3r-c4-p1-topology-9772674e-attempt1/SHA256SUMS-RUN` (SHA-256
`7d083207c3bc0870b8351074cfcd87afb9fb4a9294162c902ac8e67d1ce8fc09`).
The delegate repair preserves the missing-tie negative and production I2 call.
Its first compiled attempt timed out waiting for `heavy.lock`; the next
exact-source attempt reached runtime but failed with an uncaught user exception.
Two sealed diagnostic-only cuts localized that exception to the first slot-74
status call. The second trace passed the missing-tie rejection and positive C4
constructor, then stopped immediately after `S13-before-first-status`:
`g3r-c4-p1-topology-7bda2971-diagnostic-attempt4/SHA256SUMS-RUN`
(SHA-256 `a2768c42808e8cf7449ec90c4cd87b184d950c4bf3e1a750200f10844854bad8`).
The slot-74 guard had called the boolean
`protected-storage-comparison-active?` as a zero-argument function. This
source repair uses the existing boolean `cond` form and pins it with a
mutation test. The first full repaired-source cut remained **NONACCEPTING**;
subsequent diagnostic cuts passed slot-74 status controls and localized the
remaining exception to the fixture's C2 model construction. The shared I2
bridge in that recipe lacked `ET_M3T_PACKAGE_BUILD`, making its M3T
construction preflight report unavailable. The reviewed v2 recipe added that
flag alongside the C4 bridge flags without changing source or fixture.

The exact-source v2 receipt
`g3r-c4-p1-topology-0ed895a6-attempt2/SHA256SUMS-RUN` (SHA-256
`34d3341206289c6d9771888f633bdf900c90a8cba48619e96ebb459b21acd03f`)
is independently accepted for this bounded status leaf. Its 60 files verify;
launcher and Docker exited zero, normal/repeat each passed 11 checks with
byte-identical output, and the seeded model-authority control passed 25. The
source is `0ed895a65fef1549932b5cff5ea4654ea63af71c`, tree
`4b1676cdd9727093cf516aaa46fe3aca804dd2d6`; the independent result
review seal is `22f9156e53caed11e03f3dc28c3f0ff9b365b9bdee523de45a9fd16a24362dc1`.
Native owner association, slot 75, workspace admission, and trained-checkpoint
continuation remain separate gates.
