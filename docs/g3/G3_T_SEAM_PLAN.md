# G3-T private seams: original planning hold

The successor [exact private-seam contract](G3_T_PRIVATE_CONTRACT.md) now
consolidates these questions and is accepted by
[decision 5752756205](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5752756205).
Production integration waits for both independently approved G3-N/S implementation
merges. The checklist below
records the original hold and is not a second competing contract.

The [binding decision](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5748532295)
accepts the C2 consumer/lifetime direction. This file preserves the separately
required private-seam work; it does **not** freeze C symbols, slot numbers, boxed
layouts, constructor identities or generic access authority. G3-T implementation
waits for G3-N/S API disposition and merged numerical dependencies. No production
code, build or new execution evidence is added here.

## Actual seams to preserve

- `native/m3_model_extension.esk` has aggregate-wide `m3-call-state`, and its
  `m3-call` boundary rejects reentrancy and clears that state on success/error.
- M3T also keeps an Eshkol model-entry active-frame field and native
  `owner.active`; workspace begin rejects an already-active model. These are
  distinct coordinated guards, not authority to call arbitrary private symbols.
- `src/eshkol_transformer/m3t_transport.c` admits native owner identities through
  its registry, holds 14 canonical parameter bindings and uses I2 scoped guards
  for fixed primitive dispatch. The fixed T2 workspace/graph path is not no-grad.
- `native/t1_wave1_root.esk` authenticates T1 shells against its own registry,
  copies IDs into decode staging, then calls its private semantic decoder.
  Successful T1 shells remain strongly retained. G3 must not enroll each output
  merely to obtain a decode path.

## Exact obligations for the later seam proposal

| Seam family | Fixed semantic boundary to freeze before implementation |
|---|---|
| Model admission | Genuine same-aggregate sealed M3 model; exact N1/V256/D4/H2/Dh2/L1/F8/C2 topology, 14 unique handles and tied identity; eval only. Reject wrong/dead/busy state before native payload access. There is currently no public model-destroy API; do not invent one. |
| Shared frame guard | One G3 frame must exclude M3 forward/VJP and M3T manual workspace activity, and those paths must exclude G3. Unify/check both Eshkol and native guard state in the successor source tuple; do not create an independent G3-only lock. Resolve the exact active-token representation and cleanup owner before freezing private slots. |
| Parameter pinning | Under that guard, validate all 14 bindings and acquire exact read-only I2 scopes before numerical use. Views remain stable through validation/invoke. Deduplicate tied roles onto their one parameter pin; no caller-selected parameter index/pointer escapes the closed schedule. Preallocated cleanup state drains scopes on every failure without allocation. |
| Cache binding | Capture canonical identities, value bits, topology/profile/eval and tokenizer identity. Decode checks against committed binding; exact restoration revalidates; prefill can replace binding. Persistent snapshots are G3-owned bytes, not VJP graphs or gradient destinations. |
| Input transport | Authenticate the accepted G3 input constructors only; borrowed CPU i64[1,P], P=1/2, and [1,1] decode. Native I1 storage rank does not imply broader K1 `storage.copy`. Copy before persistent retention; exclude output/length/T1/M3T kinds despite compatible metadata. |
| Role dispatch | Literal G3-N/G3-S/N2/N3K/A2 provider/operation/row table after N/S contracts merge. No arbitrary resolver or generic dispatch export. Explicit owned f32[1,256] last-logits copy; copy sampler i64[1] candidate into owned i64[1,1] decode scratch before model use. No descriptor relabel, backward graph or gradient slot/plan access. |
| Cache consumer | A2 owns separate f32[1,1,2,2,2] K/V, i64[1] lengths, bool[1,2] validity. Stage f32[1,2,A,2], A=P or1. Consume full-capacity views, materialize bool[1,A,2], i64[1,A] query and [1,2] key positions. Learned-position projected K is final attention-ready K with no RoPE; this narrow consumer interpretation is accepted and requires direct parity tests. |
| Typed RNG | Authenticate a distinct G3 snapshot, deep-copy its logical i64[4] words to generator ownership, and synchronously borrow them for G3-S. Neither inspected words nor initializer/dropout shells grant G3 authority. Independent snapshots survive source/output/generator release and use the accepted unary release. |
| T1 private decode | Authenticate retained baseline raw V256 tokenizer and genuine G3 i64[G], G=0/1. Borrow/copy IDs synchronously into bounded semantic staging, validate ranges and fill preallocated detached raw-byte output. Prepare that storage before prefill; do not call an allocating semantic decoder after commit. No permanent T1 enrollment, native-pointer API, public decoder widening or retained lease. |
| Publication/cleanup | Preallocate final output/list/empty-ID/RNG/text and cleanup capacity before prefill commit. Preflight all borrows/cache state before no-failure publication. End pins/views under the same guard; nonallocating cleanup must be concrete, not a promise to rely on finalizers. |

The final private proposal must name every fixed Eshkol/C entry, signature,
authentication identity, source inclusion guard, error transport field and native
symbol inventory. No such entry is called or implemented in this phase. Model
guard and tokenizer changes belong to one source-composed successor, preserving
predecessor public behavior; they must not be scattered across competing owners.

## Draw-dependent exhaustion admission

An authentic exhausted snapshot is valid at construction and for greedy, manual
prefill/decode and zero-budget generation. The four no-draw paths preserve it.
For the only positive categorical C2 budget (one), generator request admission
checks counter availability **before** replacing the old cache. A final draw may
publish the exhausted successor. G3-S's numeric `INVALID_ARGUMENT` exhaustion
must not be exposed as a new K1 `invalid-state` enum; G3-G performs authenticated
state admission and maps public `invalid-state` separately. No future multi-token
budget/EOS reservation rule is inferred from C2.

## Required proof and remaining freeze questions

Test cross-entry reentrancy: G3 while M3T holds a frame, M3/M3T during a G3 frame,
same-model and aggregate-wide guard behavior. Inject every pin/view/allocation and
cleanup failure and verify all guard cells, native active tokens, prefixes, RNG,
gradient values/counts and owner counts. Cover exact model restoration, fresh
prefill after mutation, stale/released input/output/RNG, wrong-kind/forged copied
identities, active-borrow release, independent accessor survival and no T1 registry
growth from output decode. Run public installed AOT plus native ASan/UBSan/LSan
and retained 1,024/8,192-loop accounting in the future implementation gate.

Remaining G3-T freeze work is the exact shared-guard/pin protocol and closed slot
inventory, staged-output/T1 decoder signature, authenticated RNG/tensor layout and
release state machine, and fixed provider-role wiring to merged N/S. These are
downstream API questions, not reasons to reopen the accepted C2, EOS, sampling,
ownership or publication choices.
