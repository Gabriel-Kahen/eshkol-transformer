# Public-history and T1 one-root compiled union: inert gate plan

Status: **PLAN only**. Source basis is the narrow accepted union
`2117d9ac423ea9003713fbb575ec906a016adc46` (tree must be recorded by the
admitting reviewer). The separate strict public-capture source seal
`e84e9997…`, T1 B source seal `b08ba69a…`, and T1 inert-plan root seal
`39f627d…` are external review inputs: resolve and verify their **full**
digests, reviewed file inventories, and correspondence to the union before
admission. A short prefix is not a pin. Current-main broad lineage is HOLD.
No compiler, native object, loader, container, or runtime result follows from
this plan. The installed public ABI and production T1 lifetime are unchanged.

## Exact topology candidates, not launch authorization

1. The **installed public candidate** is the existing fixed tuple in
   `scripts/build-g3g-manual.sh`: `native/g3g_manual_package_root.esk`,
   `native/g3g_manual_package_bridge.c`, the matching private-renames and
   public-exports manifests, and ordered trusted include roots
   `internal/p1/lib`, `internal/c1/lib`, `internal/t1/lib`, `src`.
   `scripts/g3g-manual-package-policy.sh` rejects a mixed tuple. Its archive
   must contain exactly `g3g_manual_package.o`; its six provider objects and
   closure come from `scripts/g3g-manual-native-inputs.sh` and the checked
   `native/g3g_manual_package_*` manifests. The existing public runner
   candidate is `scripts/test-g3g-manual-public-g2.sh`, with its unchanged
   `scripts/test-g3g-manual-public.sh` P2/G1 regression. The public caller is
   `tests/g3g_manual/public_g2_runtime.esk`, which requires only installed
   facades and exercises genuine C4 first-EOS and two-byte G2 paths.
2. The **private history candidate** is one Eshkol root load,
   `native/g3g_manual_package_root.esk`, by
   `tests/g3c4/p2g2_public_history_private_runtime.esk`. The **private T1
   candidate** is the same single root loaded once by
   `tests/g3g_manual/private_t1_retirement_runtime.esk`; its native companion
   is `tests/t1/test_sealed_retire.c`. Compile each in a fresh, separate
   artifact/cache directory with one reviewed private native object/archive
   closure. Do not link either private caller to the installed package, load
   two package roots, or treat a second registry as the same issuer. The
   existing `scripts/test-g3g-manual-private.sh` provides a native-object and
   strict-AOT recipe candidate, but targets `private_fault_runtime.esk` and
   therefore is **not** itself a runner for these fixtures. A focused runner
   requires separate source/tests review before a heavy launch. Its T1-only
   build must gate `native/t1_i64_shell.c` with
   `ET_T1_SEALED_RELEASE_TEST_ONLY`, `ET_T1_I64_SHELL_TESTING`, and
   `ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE`, plus the exact I1 storage-query gate
   required by `test_sealed_retire.c`. Those macros and test entry points must
   be absent from the installed object and public archive. No second copy of
   `native/t1_i64_shell.c`, `native/i64_tensor.c`, or the provider closure may
   be linked into a private process.
3. The installed and private lanes are separate binaries and separate
   evidence. Their shared source union is proved by exact source bytes and
   one-root construction, not by cross-linking their objects. Any need to
   modify the package tuple, public exports, source API, or provider ABI
   returns to source review; this plan does not supply that API.

## Admission and negative preflight

An independent reviewer covers the full union source, fixtures, existing
source checks, proposed focused runner, manifest expectations, and this plan.
ROOT then seals a canonical source-admission record: full Git HEAD/tree,
clean tracked-byte/mode inventory, independent/root review seals, and the
three full external source/plan seals above. The record names the exact image
ID `sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`
only if its actual image inventory and inherited labels are independently
accepted. The historical failed F loader attempt is not reusable success.
The separately accepted F observation/attestation must first establish every
actual loader and transitive library path/hash; otherwise the compiled gate
stops before building.

The immutable launch decision must bind the actual selected compiler checkout
`6803dbe24fb65eaad2901767bf010787517946ef` / tree
`539d22b1b6a89c290aa13c91263982332b01b87b`, the exact external
fe9-to-6803 lock (`tests/g3c4/f_artifact_attest.py` pins candidate-lock SHA
`b798822b18535405092b201248d8e37682d5ddb9555b6080ee32131659bfbea5`),
the copied `eshkol-run`, runtime/static archives and cache receipts, the
resolved compiler C/C++ binaries and versions, the real Eshkol headers and
generated LLVM links, the six native provider source/object recipes, and all
selected tool/runtime libraries. Existing F pin-table digests are
`18764bbd7e634c0c01c3ca8f0337fa34f73d92b0631bffb23e25c3a1ef850cc7`
for tools and
`94a7c39142c213681da20ba640996d14504d4c697dcc88c925af35577dedb53c`
for 33 runtime rows. The gate records actual path, SHA-256, size, type,
resolved target and owner for every selected file, plus exact compiler argv,
environment, object depfiles, and provider/tool outputs. These historical
constants are candidate comparisons; they are not evidence that the present
image or files were observed. The canonical repository lock still names fe9;
no source checkout or lock is silently rewritten.

Before *each* build and run, reject a dirty/different source tree; missing,
extra, symlinked, special, or changed pin member; wrong image/labels or loader
closure; mismatched compiler/headers/runtime/lock; mixed package tuple;
changed depfile/source closure; different provider object or archive member;
wrong sanitizer instrumentation; duplicate root; unexpected test symbol in
production; and an occupied heavy slot. Exercise owned negative mutants for
those conditions and prove denial happens before any compiler or Docker
create. Recheck source, image, tool, runtime, header, provider, object and
output seals after each phase. A changed pin requires a new reviewed decision,
not a retry under the old one.

## Bounded proof matrix and stop rules

Run one heavy job at a time under a ROOT-owned, completion-triggered slot.
The reviewed launcher must predeclare per-phase wall time, CPU, memory/swap,
PID, file-size, output/log byte, writable-space and minimum-free-disk bounds;
use owned child groups and authenticated container IDs for cancellation and
cleanup. Preserve actual argv, exit/wait, stdout/stderr, object/archive and
resource receipts in a checksummed evidence directory. The F loader profile's
4 GiB, two CPU, 128 PID, 128 MiB file and 256 MiB `/tmp` bounds apply only
to that F phase; package compilation needs its own measured and reviewed
bounds. An incomplete cleanup, exceeded bound, missing receipt, or unexpected
stderr is failure and quarantine, never a passing partial result.

* **Private history:** strict AOT then normal, identical repeat, ASan and
  UBSan execution of the one-root fixture. Prove authentic `[0,255]` P2 input,
  first-EOS one-token and non-EOS two-token terminal output, all 16 history
  descriptor slots (including exact model/tokenizer/owner identity), unchanged
  native terminal IDs/cache/four RNG words, foreign/copied/one-use rejection,
  and wrong live input or output at capture **and** admission. At first and
  final terminal-snapshot cuts and the native input-provider cut, nested
  capture with both the same output and a different live output must deny
  before its own provider call while the outer capture alone issues. The
  original raised `#f` must propagate unchanged, roll back the guard, and
  permit certain same-owner retry. Add an explicit later denied-output
  capture/admit case and exact 64-input/64-history success versus 65th
  no-publication/no-owner-change cases before claiming the finite cap.
  Exercise precommit guard rollback and child-only post-issuance/postcommit
  fail-stop; a returned success from the impossible tail is a failure.
* **Private T1:** strict AOT one-root fixture and native C companion in
  normal, repeat, ASan and UBSan modes. Prove authentic sealed rank-one
  length-two T1/I1 pair, exact registry removal only after native success,
  borrow end and I1 storage deregistration, permanent stale-pointer tombstone
  and allocator-address non-reuse. Prove registered wrong-length refusal,
  absent/duplicate admission refusal with unchanged Eshkol/native counts,
  then C4-owned EOS/G2 history after T1 retirement. The post-borrow refusal
  child must die before publication while its parent remains intact.
  Intentionally retained wrong-length owners are excluded from leak claims;
  report their measured count and bytes separately. Production sealed T1
  remains process-lifetime retained.
* **Installed public:** run the fixed public G2 gate, normal/repeat and
  instrumented ASan/UBSan/LSan, plus unchanged P2/G1 regression. Verify
  authentic public first-EOS and two-byte G2 text, detached owner lifetimes,
  and public error behavior. Match package defined/export/undefined manifests,
  facade/depfile closure, archive members, native provider objects and their
  SHA-256 inventory. Compare normal versus instrumented noninstrumentation
  undefined symbols; require actual sanitizer imports in the latter. Inspect
  object, archive and executable symbol tables for localized private entries,
  no unresolved privileged authority, and no `et_t1_*test*`, I1 test hook,
  or other private fixture symbol in the production package or installed
  executable. Public G2 text alone does not prove private history or model
  parameter-bit equivalence.

The decision requires all three lanes' complete sealed receipts, independent
review of emitted objects and failures, and an exact tree/manifest recheck.
Any failed lane leaves the combined gate OPEN; no earlier source or partial
compiled proof is promoted. Broad current-main lineage, compiled union
acceptance, public paired SAVE/LOAD, model-bit capture, and data-only
continuation remain **OPEN/HOLD** until their own downstream contracts and
evidence exist.
