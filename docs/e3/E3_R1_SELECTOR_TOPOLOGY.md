# E3 §9 R1/R2 selector and tuple topology

Status: **source-only design**, based on `codex/e3-fault-symbol-audit` at
`6cd29ccd2568daedb1ef6b71697b71e3968897e2`. This refines the
[stage-cut inventory](E3_STAGE_CUT_INVENTORY.md) without adding a callable,
changing a package, or admitting an R1/R2 row. The selected route remains the
private bool evaluator. Every proposed fault is a test-only, recoverable
preinvoke or preflight denial; no fault enters a validated K1 provider invoke.

## Exact entry and ownership boundaries

`e3-evaluate-into!` resolves an authentic live frame before `m3-call` and
resets the first-error slot/stage within that call
(`native/e3_private_extension.esk:449-468`). A foreign or released frame is
therefore a **pre-entry** negative, not stage 0. `e3-frame-bind!` constructs the
native frame, 37 tensors, provider handles and epsilon before any evaluation
(`:154-210`, `src/eshkol_transformer/e3_frame.c:612-703`). Its allocator and
provider-discovery failures are setup cases, followed by a fresh-frame retry.

The entered-call sequence is P1 prepare (stage 1), D2 prepare (2), native
acquire and 14-pin admission (3), P1 enter (4), D2 read (5), staged
input/target/mask copies (6), 21 forward roles (7), CE (8), L3S reduce (9),
correct (10), accumulate (11), and finalize (12). The source sets each stage
before its operation (`e3_private_extension.esk:401-447,321-368`), and the
forward schedule calls roles 0 through 20 once per emitted batch
(`native/e3_forward_schedule.esk:4-25`). Native acquire first checks storage
and zeros six sum tensors, then begins the pin lease
(`e3_frame.c:721-747`); no invocation-owned allocation index is established
there. A frame-constructor allocation countdown cannot stand in for stage 3.

The E3 fault selector must be owned by the **authenticated native frame**,
never by a process-global countdown. A test driver first obtains the live
entry through `e3-frame-entry-live`, arms its native pointer while the frame is
idle, begins one test-only invocation scope, enters `e3-evaluate-into!`, and
ends that scope on either outcome. The closed request is
`(site, ordinal, occurrence)`; `site` is a reviewed enum, `ordinal` is a
site-specific role or copy index, and `occurrence` is the positive count of
that exact site **within one invocation**. No caller callback, string,
arbitrary error value or tensor mutation is admitted. The setter rejects an
unrecognized combination, zero or out-of-range occurrence, stale/foreign or
busy frame, and an existing armed or fired record before any model, cursor or
destination mutation. A separate idle-only reset is required after every
scenario; an unfired record must never silently arm the next invocation.

On reaching the selected site, the test-only code increments that site's
count, consumes the arm **before** returning one fixed native denial, and
records `reached`, `fired`, site, ordinal and occurrence for postcall readback.
The normal E3 wrapper captures the native domain/category/code before another
native call can clear it, maps the already-set stage through the existing
first-error slot, and uses the existing abort/restoration path. The driver
checks that exactly one fault fired and that retry after explicit reset uses
the same frame, dataset, model and destinations. A request whose occurrence
was never reached is a failed test, not a passing rollback case. Per-frame
storage also lets simultaneous test frames keep disjoint selectors; concurrent
use of the same frame remains rejected by the existing frame authority.

## Smallest nonduplicate first source slice

Implement only `site=forward-preinvoke`, `ordinal=0..20`,
`occurrence=1..8192`, under a new `ET_E3_FAULT_TESTING` guard in the native E3
frame. In `et_e3_private_forward_role_v1`, place the one-shot denial **after**
authentic frame, phase and `next_role` checks but **before** `e3_dispatch`
(`e3_frame.c:957-1060`). Role 1/occurrence 1 supplies a post-provider witness:
role 0 must have completed successfully first. Role 0/occurrence 2 additionally
requires one complete earlier batch. The fixed denial uses an existing E3
native status/category/code, which the stage-7 wrapper records as the first
mapped E1 error; the implementation review must freeze that literal status
before code is admitted. A source-only negative must prove an unarmed frame
has identical dispatch order and result. This slice adds no stage 3, D2,
copy, P1, CE, finalization, publication or resource proof by implication.

The test-only frame fields and C arm/begin/end/status/reset definitions belong to
`e3_frame.c` (the frame struct is defined there); declarations used by the
test bridge belong to `e3_frame_internal.h`, compiled only for the new tuple. The
driver's Eshkol externs and test scenario belong to a new
`tests/e3_private/fault_*` successor of the retention driver. Its driver root
loads `retention_driver_root.esk` and one fault extension; its bridge includes
`retention_bridge.c` and adds one boxed closed-run entry. This reuses the
retention tuple's authentic 17-mode, model/gradient, D2 cursor, output and
owner observations without changing the canonical private driver or installed
facade. It does **not** reuse D2's persistent read/batch selectors as an E3
occurrence counter.

The complete new lexical tuple needs its own root, bridge, private-renames and
public-exports files plus exact defined/undefined symbol, public-string,
archive-member, native-object, source-closure and native-source-closure
manifests. `scripts/e3-private-package-policy.sh` must recognize only the
four exact lexical inputs and the same six ordered include roots as retention
(`:237-259`); mixed members and ambient include roots reject. The E1B builder
must select this tuple explicitly for its generated P1/D2 stage, native test
defines, expected local symbols, export/string patterns and sanitizer import
policy (`scripts/build-e1b-consumer.sh:1265-1447,1710-1829,2334-2507`).
`ET_E3_FAULT_TESTING` may reach only the new tuple's E3 frame object. The
feature-off canonical/installed objects and their manifests must retain exact
symbol, string, source/object and export parity. A declaration in a header or
an `extern` alone does not establish a definition or an admitted tuple.

Focused source negatives precede compilation: exact mixed/partial tuple,
symlink and ambient-root rejection; duplicate E3 root; forged or wrong-arity
extern; absent feature-off symbol/string; one localized new definition and
matching undefined use; invalid/stale/foreign/busy/duplicate arm; zero,
over-cap, unreached, consumed and explicit-reset cases. The first compiled
failure/retry gate then compares exact pre/post cursor bytes, all 17 modes,
14 parameter and gradient images plus metadata/ties/RNG, six output words,
native first error and live/retired owner counts against a fault-free twin.
Normal, repeat, poisoned and ASan/UBSan/LSan resource horizons remain separate
required evidence. The full R1/R2 table still needs stage-specific P1, D2,
pin, copy, later read, CE/L3S/correct/accumulate/finalize and publication
preflight cuts; cleanup invariants and the postpublication commit tail remain
fail-stop, not recoverable selectors.

## Native prerequisite candidate

The guarded native frame candidate adds a forward-only arm/begin/end/readback/reset seam
under `ET_E3_FAULT_TESTING`; it is not selected by any package tuple yet. Its
fixed denial is native `(domain, category, code) = (0, 2, 2)` (`invalid-state`,
`phase`). The future noninstalled driver must call the authenticated idle-only
native `begin` immediately before one `e3-evaluate-into!` attempt and `end`
after either outcome, including a pre-acquire rejection. The evaluator and D2
prepare both call `et_e3_private_frame_check_v1` while the native frame is
idle; neither check changes selector state. Native finish also marks an
unfired entered arm missed. The preinvoke check follows frame, phase and next-role
admission and precedes operand setup and `e3_dispatch`. Idle readback selectors
0–4 return state, role, occurrence, reached and fired, respectively. States
are clear 0, armed 1, entered 2, fired 3 and missed 4; explicit idle reset is
required before another arm. A C state-machine probe calls both native frame
checks on a deliberately invalid-storage frame and verifies they cannot
expire an entered arm. Feature-off object comparison checks this prerequisite.
The noninstalled lexical tuple, bridge use, real successful D2 prepare,
compiled same-frame failure/retry matrix, and resource horizons remain
separate gates.

## Retention successor driver source candidate

`tests/e3_private/fault_driver_root.esk` loads the retained private root and
one fault extension; `fault_bridge.c` layers one boxed fault entry over the
retention bridge. The extension obtains the native pointer from an authentic
live E3 frame entry. Each selected call arms and begins the frame selector,
catches the synchronous evaluator outcome, then ends the selector scope before
readback or retry. A deliberately unreached role-20/occurrence-8192 control
exercises the successful end path; the selected failure path checks stage 7,
native `(0, 2, 2)`, exact reach/fired state, preserved cursor, 17 modes,
14 model/gradient images, RNG, six retained output words and live owners, then
explicitly resets and retries the same frame. The source accepts only the
closed forward role/occurrence scenario and reuses the existing retention
fixture. A guarded native raw counter observer reads the two published counts
after an acquired failure, when the normal committed-only counter accessor is
unavailable. Focused source checks and the native selector probe pass.

That first driver-only candidate had no E1B package selection. The
current-main source union below adds the noninstalled lexical tuple; compiled
D2 preparation and same-frame failure/retry remain unverified.

## Current-main source union

The noninstalled `fault_package` tuple and guarded selector are now composed
on main `94c4c1e` while preserving its zero tuple and canonical
source contracts. Focused source checks and the standalone native selector
probe pass; feature-off E3 preprocessing matches that main exactly. The
manifest inventories remain source candidates until an exact package build
and independent current-main review validate them. Real D2 failure/retry,
sanitizer behavior, and resource horizons remain open.
