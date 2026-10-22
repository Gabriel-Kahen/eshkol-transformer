# Supported installed public G2 gate — staged, NOT RUN

This is a focused source port of existing gate `e6e6e0810fd5ce972821e5b37d0ff110bec3b0b8`
and installed caller `b3114a0325a33e8872eda9a586121639aeff4b26` (union `642d9163`).
It preserves reviewed public implementation `0a292f416a3add5ef511c84bb7d89922e4a690ab`:
CPU-f32 canonical C4, raw V256/P2, budget two, explicit byte EOS. There is no new
public API, model conversion, constructor seam or host-stub substitution.

Run on the pinned supported Ubuntu22.04/Clang21.1.8 f31 image with clean Eshkol
`fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`, canonical C7 runner digest
`fdc00685e6056b9c0027146b241a55fa72c47e0962c033bbd8be6426fad0104c`.
The old custom-6803 rejection and historical fe9/7dd attribution are not current
supported admission. Production toolchain/common/IR provenance checks stay strict.

## Exact commands inside the supported, pinned container

Acquire the team's global `heavy.lock` outside Docker. Bind immutable source at
`/workspace`, canonical compiler/build dependencies at their provenance paths,
and fresh one-use writable output at `/out`. Set `TMPDIR=/out/scratch`, retain
pre/post source/toolchain hashes and `verify_toolchain`, then select one command:

```bash
bash scripts/test-g3g-manual-public-g2.sh /out/gate normal
bash scripts/test-g3g-manual-public-g2.sh /out/gate sanitize
bash scripts/test-g3g-manual-public-g2.sh /out/gate all
```

Each command requires a fresh `/out/gate`. Normal and sanitize are separately
labeled G2 smoke gates; they explicitly leave P2/G1/C2 regression NOT RUN. `all`
runs normal G2 first, then SAN G2, compares their measured inventories/output,
and reuses the normal installed archive for the existing installed P2/G1/C2
regression. No duplicate ordinary build is required. Build timeout is 1200s,
caller compile timeout 900s, execution timeout 180s per first/repeat launch.
A bounded outer launcher owns CID cleanup, resource limits, exact source/compiler
pins and post-exit checksum sealing; root schedules it in CPU4 priority order.

The existing `build-g3g-manual.sh` and exact E1B tuple compile the actual owning
69-source Eshkol aggregate, including the real closed terminal coordinator and
reviewed typed owners. The installed caller requires only six public facades;
compile-only depfiles must prove their installed closure. The actual AOT caller
links the built archive with `--strict-types --no-stdlib -O 0`, then performs
first-EOS and two-token generation, all five accessors after generator close,
detached typed releases, and genuine `:rng` continuation after output release.

## Bounded builder delta

`G3G_MANUAL_SANITIZE=1` is accepted only for the exact installed manual tuple,
with no E3 sanitizer simultaneously selected. Separate G3 flag/relocation arrays
instrument generated private IR, both C bridges and ordinary native inputs.
All six reviewed provider builds select their existing `sanitize` mode.
E3 selectors, flag arrays, IR rewriting, import policy and strict M3CG provenance
are unchanged; integration must merge only these G3-specific hunks into shared
builder work rather than replacing the file.

Unlike the old gate, SAN must materialize and verify actual generated-private-IR
address checks using the existing checksum-pinned streaming helper and object
import verifier. Raw/derived IR, receipt and object ASan imports are retained.
The final object requires exact normal noninstrumentation imports, measured SAN
imports, unchanged public definitions and archive membership. No automatic
manifest repin is allowed. The public caller links the sanitizer runtime; this
does not claim instrumentation of the pinned third-party Eshkol runtime archive.
ASan/UBSan/LSan execution is fail-fast and retains first/repeat stdout/stderr.

## Remaining dependencies

Everything compiled remains **NOT RUN** for this successor. Independent runner/
builder/fixture source review precedes launch. The separate supported terminal
stage45 native-SAN/clone-lease release blocker remains unresolved and retained;
this runner neither fixes nor bypasses it. Parent-owned C4 training/SAVE/restore/
evaluation foundation and supported exact candidate build are separate gates.
A normal-only smoke does not establish SAN, installed coexistence, numerical
parity, full reproducibility or CPU-v1 completion.
