# G3-C4 P1 prepared construction split

This is the first implementation unit authorized by the
[model/context successor contract](G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md).
It does not implement the C4 I2 route, model authority, context, cache, active
call, sampler, or public API.

## Implemented boundary

The trusted P1 identity ledger now preserves native states 0 `OPEN`, 1
`SEALED`, and 2 `ABORTED`, and adds state 3 `PREPARED` through three private C
operations:

- `et_p1_private_construction_prepare_v1`
- `et_p1_private_construction_commit_prepared_v1`
- `et_p1_private_construction_abort_prepared_v1`

Prepare and prepared abort authenticate the exact active ledger and every
enrolled module/handle before mutation. Commit repeats that complete preflight,
then has an allocation-free, nonfailing mutation tail. The old seal remains an
OPEN-to-SEALED one-shot operation. The old abort remains OPEN-only and
idempotent only after ABORTED.

The canonical trusted P1 source appends two closures without changing slots
0..70:

- slot 71: prepare the exact construction tree, install canonical paths,
  finalize it, set and verify `eval`, then enter native and source PREPARED;
- slot 72: preflight PREPARED, commit native ownership, then publish source
  SEALED and clear the active construction in a guard-free tail.

Prepare reserves one cleanup-handler frame before installing its guard or
setting the reentrancy marker. Reserve failure aborts the still-OPEN admitted
construction directly; every later fallible prepare action runs under cleanup
that aborts it. The construction record and root are reread from rooted holders
after checked promotion. Both OPEN and PREPARED shells remain unpublished;
PREPARED also rejects the raw private handle access admitted during OPEN.
Native commit status failure exits with status 134 and cannot become a
recoverable transformer exception.

The two slot calls are exposed only through
`native/g3c4_p1_construction_extension.esk`. Public P1 provides, ABI version,
symbol manifest, package exports, and `tools/p1/module_surface.tsv` are
unchanged. Existing E3 slots 64..68, TR3 slots 69..70, the M3T transport source,
and the legacy slot-61 seal are pinned unchanged by structural checks.

## Verification

Checks available without the recovered supported Eshkol runtime:

- `python3 scripts/check-p1-prepared-split.py`
- strict Clang and GCC native construction builds
- exact trusted-symbol comparison and C++ signature compilation
- ASan/UBSan construction execution
- generated-root and whitespace checks

`scripts/test-p1-prepared.sh` is the focused supported-runtime gate. Its AOT
fixture covers prepared abort and seal, canonical `eval` publication, prepared
shell opacity, missing/changed schedule cleanup, unattached-module cleanup,
duplicate and legacy-operation rejection, next-construction baseline, repeated
arena-poisoned execution, and injected commit failure exiting 134.

The current CachyOS compatibility host lacks the pinned LLVM 21 toolchain, and
the previously recovered Eshkol runtime is not accepted for this source because
its emergency-rethrow behavior is unsupported. The AOT/runtime gate therefore
remains pending; no result from that runtime is claimed here.
