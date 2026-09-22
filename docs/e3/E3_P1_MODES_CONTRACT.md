# Exact proposed E3/P1 mode contract

Bounded source/design audit at E3-DESIGN branch; no edits/build/tests. This is a concrete proposed private contract for root/P1 acceptance, not evidence those slots exist. Shared guard/pin mechanics remain owned by task01a0c789-bd3f-7b21-b865-8b9365c7a282; E3 owns frame admission; this contract adds no guard or pin authority.

## 1. Actual compilation and chosen source mechanism

`scripts/build-m3.sh` invokes `scripts/build-e1b-consumer.sh` with the exact M3 root, bridge, rename/export files and include roots. That builder directly runs the Eshkol compiler with `--strict-types --no-stdlib --shared-lib --dump-ir --emit-depfile`; it does not C-preprocess Scheme. `native/m3_package_root.esk` loads M3T, which loads I2/T1; P1 is resolved to `internal/p1/lib/transformer/module.esk` through reviewed include roots. `p1-trusted-surface` is one `let` containing lexical raw-module authority and its final64-closure vector.

**Choose direct append of five closures64..68 in the existing canonical `internal/p1/lib/transformer/module.esk`.** Define their small private helpers and ledger inside that same `let`, immediately before its final vector. Preserve every0..63 closure and public `provide` list exactly. Put E3-only named wrappers in proposed `native/e3_p1_modes_extension.esk`, loaded only by proposed `native/e3_private_root.esk`; wrappers call the exact slots. No top-level P1 helper export, native P1 symbol, alternate module shadow, copied P1 source, lexical `(load ...)` assumption or invented compile-time conditional.

Consequences are explicit: canonical trusted vector length becomes69 for every inheritor; extra lexical closures/ledger initialization may affect private generated symbols, strings and fixed root retention. This requires an accepted P1 prerequisite and exact inherited manifest/retention review. “Preserve0..63” does not mean “predecessor artifact bytes unchanged.” If root forbids a69-slot canonical surface, stop and separately design a factory refactor; the current compiler recipe does not already provide a conditional lexical extension mechanism. Build policy must pin the updated canonical source and new E3 wrapper file; existing manifests cannot silently accept it.

## 2. Exact slots, arities and return convention

| Slot | Private wrapper in e3_p1_modes_extension.esk | Arity | Result |
|---:|---|---:|---|
| 64 | `p1-e3-mode-bind! model setup-box` | 2 | immediate i64 status; on0 publishes canonical token in setup-box[1] |
| 65 | `p1-e3-mode-prepare! token` | 1 | immediate i64 status |
| 66 | `p1-e3-mode-enter! token` | 1 | immediate i64 status |
| 67 | `p1-e3-mode-restore! token` | 1 | immediate i64 status |
| 68 | `p1-e3-mode-unbind! token` | 1 | immediate i64 status |

These are source-private Eshkol calls only, not boxed/native exports or installed operations. `setup-box` is the exact preallocated2-slot E3 control cell `[frame-key,#f]`. `frame-key` is the E3 root-owned canonical frame identity, already staged in the E3 ledger; it is neither a native address chosen by a caller nor a copyable tag as authority. The closed E3 frame-bind wrapper authenticates that exact staged frame/model pairing before slot64. P1 independently authenticates the P1 model, its fixed topology and ledger uniqueness. P1 deliberately does not implement a second native E3/shared-guard authenticator.

Bind occurs once per private frame, before evaluation; only it may allocate. Existing `raw-for-shell`/construction-check errors or allocation failure may raise during bind before modes/cursor/output are touched. Slots65..68 never call E1 error constructors, allocating `raw-for-shell` failure paths, public setters or `finalization-plan`; they return bounded statuses after nonallocating checks. E3 snapshots/maps the status at its outer safe error boundary, outside any mandatory cleanup tail.

Status table:0 OK;1 INVALID_IDENTITY (wrong/copy/foreign token, malformed setup box, unregistered model);2 INVALID_PHASE (duplicate bind, busy/dead/nonmatching lifecycle, protected-storage-comparison-active?);3 UNSUPPORTED_TOPOLOGY (authentic root differs from exact17 topology/provider profile);4 INVARIANT (invalid mode symbol, previously-bound canonical identity/topology unexpectedly changed, impossible slot/control invariant). Invalid phase/identity checks precede field dereference/mutation. E3 maps1 to invalid-argument,2 to invalid-state,3 to unsupported,4 to internal in private diagnostic domain `e3-p1-modes`. Final outward errors use the156-cell table: source-domain `e3-evaluation`, source-code=current stage, cause#f, preserving the mapped A0 category. No second outward P1 error form is constructed during cleanup. No raw record/value is embedded in error details. During an already admitted cleanup, nonzero status is an invariant defect requiring E3 fail-stop; it must not replace the first recoverable error or falsely claim rollback.

## 3. Token/ledger and root-region ownership

Add lexical `e3-mode-registry-root = (vector '())`. One setup allocates one inert token vector, one ledger record and three fixed vectors[17] (nodes, saved modes, child-list identities), plus one list cell. The token is the only returned value; raw-node/snapshot arrays stay lexical ledger-owned. Token contents are inert; every slot65..68 first exact-eq scans the ledger for its token, then reads the record. This is a mode-authority ledger, not a result/report registry.

Proposed ledger record exact slots:

| Slot | Type/meaning |
|---:|---|
| 0 | exact opaque token identity |
| 1 | lifecycle exact i64:0 idle,1 prepared,2 entered,3 dead |
| 2 | exact canonical E3 frame-key while live, #f dead |
| 3 | exact P1 root shell while live, #f dead |
| 4 | canonical raw P1 root while live, #f dead |
| 5 | fixed vector[17] canonical raw nodes; #f dead |
| 6 | fixed vector[17] saved `train`/`eval` symbols, initially all#f; #f dead |
| 7 | fixed vector[17] exact original child-list heads; #f dead |
| 8 | preallocated mode-check scratch/status immediate i64, initially0; reset0 |

All allocation, fixed-tree traversal and validation finishes before ledger publication. Publish staged ledger graph through the root holder's existing region write-barrier discipline, then reread the canonical first ledger record/token. Only that token is stored into setup-box[1]. The setup-box/frame-key must themselves already have stable root lifetime before native frame code retains Eshkol identities; never store a pre-promotion pointer in native control. If setup-box publication can allocate/promote, that is still setup: no model mutation occurs, and failed outer setup consumes cleanup of any published token through slot68.

Slots65..67 write only existing fixed slots with immediate phase/status and existing root-lifetime mode symbols. They allocate no list/vector/string/exception or closure-registration record. Post-preflight mode writes cannot trigger graph promotion because mode symbols and all targets are already root-owned. This still requires pinned optimized/poisoned AOT allocation-counter proof before claiming the tail infallible.

Unbind idle only clears slots2..7 and sets phase3; keep slots0/1/8 and the ledger cell as an inert identity tombstone. Exact dead unbind returns0; all other dead use returns2. No per-call enrollment or tombstone is created. One-time successful/failed frame-setup costs and released token/control retention are measured separately; repeated evaluation reuses the exact same token/arrays. This design makes no flat process-lifetime claim across arbitrarily many frame bindings and introduces no report quota. If a total frame-creation bound is required, root must add it explicitly; do not hide an unbounded setup history behind per-call flatness.

## 4. Exact fixed-node validation

Mode order/index0..16:

0 root;1 blocks;2 blocks/0;3 blocks/0/attention;4 blocks/0/attention/key;5 blocks/0/attention/output;6 blocks/0/attention/query;7 blocks/0/attention/value;8 blocks/0/ffn;9 blocks/0/ffn/down;10 blocks/0/ffn/up;11 blocks/0/norm1;12 blocks/0/norm2;13 head;14 norm_final;15 position_embedding;16 token_embedding.

Parent-index vector is `#(-1 0 1 2 3 3 3 3 2 8 8 2 2 0 0 0 0)`. Bind verifies every exact path, no extra/missing children, all17 distinct raw nodes, root parent#f, each child-parent edge, root count10=17, sealed flags7, and exact I2 provider identity through existing P1 authority. Every raw node is the actual13-slot private module vector; mode is slot1, children slot4, parent5, finalized7, node-count10. Child list items are3-slot `child-binding` records, name slot1 and raw-node slot2. The two tied parameter paths do not merge head/token_embedding module nodes. Outer E3 authenticates the genuine M3 constructor/profile/14 parameter bindings; tree shape alone is not M3 authority.

Capture child-list heads at setup. Each prepare rechecks token/root shell ledger identity, same canonical raw root, all17 node/parent/sealed identities and unchanged children heads, without generating paths/lists. Since P1 topology is sealed and only trusted same-aggregate code can mutate raw vectors, exact head/parent/node checks plus the original binding proof suffice; source-private malicious mutation remains outside the threat model. Test instrumentation may corrupt nodes to prove rejection before writes. Check every mode is exactly train or eval; preserve mixed modes.

## 5. State transitions and nonallocating tails

- **Bind**: admitted root+fresh setup-box/frame key, not previously enrolled; validate/allocate/root-publish; output canonical token; ledger idle. No module mode changes. On prepublication failure there is no token/authority. If token exists but later frame setup fails, unbind once before native frame teardown.
- **Prepare**: idle only. Check protected comparator inactive, exact bound identities/topology and all17 valid mode symbols first. Then copy all17 to preallocated saved array in index order and set prepared. No mutation of actual modes; any rejected admission leaves record/modes untouched.
- **Enter**: prepared only and E3/common guard owner has already acquired pins/published matching model/frame tokens. Revalidate all mode destinations and require every current mode equals its saved symbol (reject outside mode mutation). After all checks, write eval to all17 slot1 fields in order, then set entered. No callback/provider/native call/allocation after first mode write.
- **Restore**: prepared or entered only. Authenticate record and preflight all17 saved symbol/node/destination identities before writes. Prepared means no mode writes occurred: clear saved17 and return idle. Entered means require current modes all eval, restore all17 saved symbols in order, clear saved17 and return idle. E3 invokes this under retained model+outer exclusion before six-output publication. This function performs no cursor/result/pin/guard change. A nonzero preflight in an admitted cleanup is fail-stop upstream; after first restore write the fixed write/clear tail has no condition that can recoverably reject.
- **Unbind**: idle only; prepared/entered returns2 without writes. Clear roots, mark dead, leave inert token identity. E3 must unbind before native frame destroy; exact dead unbind idempotent. No automatic mode restoration or implicit cleanup during teardown.

Successful output publication and error staging remain entirely E3-owned. P1 mode helpers must not construct an error after mutation, clear shared exclusion, reset gradients or copy parameter bytes. E3 classifies the first caught error in the same region, stores only its mapped index/root-owned prebuilt shell, and raises outside the normally exited shared guard after cleanup; see E3_ERROR_LIFETIME_CONTRACT.md. No arbitrary caught graph is promoted during restoration.

## 6. Required exact-contract review/tests

1. Root/P1 accepts69-slot surface and unchanged first64, named wrapper file/root dependency, extra fixed allocation/manifest cost; all inherited public surfaces remain identical. No claim of unchanged binary hashes.
2. Mode bind from a short-lived setup region returns the canonical stable token; poisoned region exit followed by prepare/enter/restore works. Native frame never retains pre-promotion identities. Token copies/foreign roots/wrong setup boxes reject.
3. Bind alloc failure at each object/barrier stage: no mode/output/cursor mutation; exact root/native retention ledger reconciled and unbind when publication already happened. No general model teardown inferred.
4. All17 mixed modes restore exactly on success and failure after multiple batches. Corrupt/missing/extra/duplicate node, changed child head/parent, comparator reentry, repeat enter, restore twice, prepared unbind, dead-token reuse all fail before writes. Numeric statuses and fixed error mapping tested separately.
5. Allocation-disabled prepare/enter/restore/unbind with exact live/cumulative counters at1,024/8,192 calls; instrumented failure must detect allocating raw-for-shell/public-setter/planning paths. Shared exclusion retained; successful restore precedes every6 output write.

All filenames, slots and statuses above are **concrete proposal choices**, grounded in existing lexical source/recipe. No compiler conditional or already-merged restoration seam is asserted, and no shared-guard implementation is duplicated.

### Bind publication failure and prepared-restore clarification

Slot64 must install a local cleanup guard before root-ledger publication. If the root graph is published but writing canonical token to setup-box fails, use the already-read-back canonical record to clear slots2..7 and mark dead without allocating, then preserve/rethrow the original setup exception. Do not leave an inaccessible live record retaining the model/frame. A dead token/ledger cell is charged to failed setup retention even if no token reached the caller. The caller-visible setup-box[1] is changed only by the final token store; the reviewed write-barrier operation must establish its failure atomicity before this bind can be accepted.

Prepared restore also checks current17 modes still equal the saved17 before merely clearing the snapshot. Entered restore checks every current mode is eval. These checks catch forbidden intervening writes before either restore path changes state.
