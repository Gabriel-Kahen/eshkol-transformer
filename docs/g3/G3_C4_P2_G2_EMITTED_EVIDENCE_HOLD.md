# P2/G2 terminal emitted evidence and remaining proof gate

Status: source-only successor to `5233f47c`. This change preserves the
production composer, zero-return tail, native ABI, and accepted terminal
contract. No compiled gate, Docker container, real one-use claim, or heavy
launcher ran for this source staging.

## Retained causal artifacts

`G3C4_TERMINAL_RETAIN_EMITTED=1` opts the exact terminal runner into a
two-step compile/link for terminal, feature-off, and all nine current
overlays in normal and sanitizer modes. The pinned compiler emits
`--emit-object --dump-ir --emit-depfile` to one raw object, bitcode, IR,
and depfile; the runner links **that object** into the executable. In the
sanitizer mode, an exact reversible IR rewrite adds ASan attributes and
Clang compiles the derived object; the runner links that derived object.
The generated files, native objects/depfiles/archive, linked executables,
overlays, and source-before/after hashes survive a successful opt-in run.
The ordinary runner retains its previous cleanup behavior.

The mechanical checker requires all 11 fixture artifacts, 17 native
objects, archive members, reviewed native/Eshkol depfile sets, unchanged
source hashes, exact ASan IR receipt, and an ASan-instrumented object. The
seven terminal FFI declarations in the current composer must match all
raw/ASan object imports exactly, including feature-off, which loads the
definitions but rejects admission at runtime. The
runner's `--wrap` set and allowed `__real_` references are exact. Other
raw imports must resolve in the native archive, pinned Eshkol runtime/static
archives, or a 59-name historical platform ceiling. Linked dynamic imports
are bounded by a 341-name **historical failed-run ceiling**; sanitizer
binaries additionally allow sanitizer-prefix imports. These historical
and prefix rules are provisional drift bounds, not an exact source-derived
imported-symbol or linked-provider closure. Current-head compiled imports
and selected providers still need independent review. Historical dependency
sets also require an exact current-head depfile run and review. Any drift
fails closed. The checker writes exact
per-artifact import and SHA256 inventories. The compiler command's
source-before/after identity is recorded by the runner.

## Successful-zero no-allocation remains open

The accepted contract permits legitimate allocations before native
terminal success and requires no allocation, callback, decode, registry
search, native readback, or recoverable validation after zero return
(`G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md`, “Zero return”).
The retained historical failed `terminal.raw.o.ll` places
`et_g3c4_private_p2g2_terminal_commit_v1` in the generated coordinator,
then emits tagged numeric and error branches with calls such as
`eshkol_make_exception_with_header`. Scanning all descendants or all
undefined symbols would mistake fallible/error paths for the successful
zero path. The current mechanical checker deliberately reports
`zero-tail proof HOLD`; runner opt-in writes
`emitted-acceptance.status` with the same limitation. A zero exit from
artifact capture is **not** emitted-code acceptance.

A separately reviewed verifier must bind the native return value to zero,
prove the successful branch through the preallocated barrier, captured
writes, `m3-call` busy-cell clear and original-shell return, and resolve
every reached direct/indirect call transitively across generated IR and
linked native/runtime code. It must reject allocation or recoverable
callbacks on that path, unknown indirect targets, a changed zero edge,
hidden helper calls, substituted IR/bitcode/object, missing depfiles, and
new imports. The generated bitcode/text IR identity and exact linked
runtime closure also require full review; nonempty artifacts and provider
membership alone do not prove either. The source-only negatives currently
cover malformed/extra depfiles, changed source, missing/empty/symlinked
artifacts, terminal FFI and provisional import drift, and altered ASan
derivation.

## Launcher and lifecycle HOLD

A new immutable recipe must pin this successor's exact head/tree, full
source closure, compiler commit/tree/runner/archives/provenance, image ID
checked live, and read-only `/workspace` plus nested real empty
`/workspace/.deps` mountpoint. Minimum available resources remain 64 GiB
disk and 4 GiB memory. A separately reviewed launcher must acquire the
sole-slot nofollow lock before any result write, retain its FD through
teardown, atomically consume one-use claim under that lock before mutable
preflight or child start, and never retry after claim. It must authenticate
the guest command and owned container identity without `--rm`, enforce a
finite outer wall clock (the repeat stage has none). Reviewed runner
instrumentation must record **each internal** compiler/link/test command
with exact argv, cwd, numeric exit, raw stdout and stderr; the launcher
must capture the top-level streams and prove signed child wait, both pipe
EOFs, descendant/named
writer quiescence, and owned-container absence. Preserve all failed
evidence. Uncertain teardown keeps the lock and evidence in quarantine;
only reviewed closure permits sealing. No executable launcher is included
here. The earlier 5233 source-only recipe and failed bundles remain intact.
