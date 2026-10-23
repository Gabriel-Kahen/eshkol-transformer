# Private CPU4 C4 history and detached capture

This source implements the coordinator-accepted proposal `214ec8` and its E3 delta review. It establishes no native execution, continuation, import, wire-format or checkpoint admission. Compiler/native/SAN gates remain **NOT RUN**. Calls are serialized, with fixed trusted providers and authoritative private allocation/registry links. Arbitrary native writes into those links or A2 internals are outside this contract.

Enable `ET_CPU4_C4_HISTORY_PRIVATE` only with the existing effective prerequisites:

```
ET_G3C4_GENERATOR_PRIVATE ET_G3C4_ACTIVE_CALL_PRIVATE
ET_G3C4_TOKEN_FRAME_PRIVATE ET_G3C4_TOKEN_FORWARD_PRIVATE
ET_G3C4_NATIVE_PINS_PRIVATE ET_G3C4_PREFILL3_PRIVATE
ET_G3C4_OUTPUT_DECODE_IDS_PRIVATE ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE
ET_I64_TENSOR_STORAGE_QUERY_PRIVATE ET_F32_TENSOR_STORAGE_QUERY_PRIVATE
```

The feature rejects an incomplete tuple. TOKEN_FRAME/TOKEN_FORWARD make the existing nested prefill dependencies explicit. Other route flags retain their existing prerequisites. OFF emits no history/capture storage or symbols and preserves the existing seven context layouts and 32-byte transport header. Host source projection checks that the original OFF token path is unchanged; native byte/layout validation remains pending.

## Custody and ABI

Context appends history80: ordered IDs/origins, prompt/generated/manual counts, generated-only EOS mask, READY state and commit epoch. Actual committed count equals A2 length; actual cache capacity is 2 or 4. Context provenance464 independently records original owner/14 parameters/handles/policy, committed cache/RNG/history and pending40. Pending40 holds exact returned transaction/frame/candidate pointers, expected epoch and operation/capacity. Total additional context custody is 544 bytes; context sizeof itself grows only80. Supported ON context extents are 6576/6584/8776/8784. All allocations precede publication; exact native acquisitions are enrolled immediately with fixed stores.

A2 cannot read-borrow while a transaction is active. Early preflight genuinely reads the idle cache before providers or sampling. Exact pending custody carries that proof across split calls; a staged transaction is checked through its genuine native view before publication. The existing terminal witness is used only by its old capacity4/length2-or3/append1 path; no general getter is invented. Every final native frontier retains its authenticated node locally through the native commit and fixed cache/RNG/history/mirror stores. Abort clears consumed pending custody; it never rolls back an already committed G2 prefix.

The coordinator-approved manual lifetime refinement keeps the same pending40: manual COMMITTED consumes/transfers the exact transaction/candidate, but retains the authoritative frame/op and updated committed epoch until finish actually frees the allocation. COMMITTED is drain-only: no provider, re-commit, prospective capacity check or epoch increment. It can drain a full capacity2 cache at epochMAX. Abort/re-commit reject COMMITTED without mutation; corrupted frame crosslinks reject until the exact original is restored. Only after real frame free do fixed stores clear frame/op. Capture still requires idle/drained and all pending40 zero.

The private header declares exactly four functions:

- `et_cpu4_private_c4_capture_create_v1(registered_generator)` returns an opaque capture or NULL.
- `et_cpu4_private_c4_capture_copy_record_v1(capture,dst,224)` copies the inert version1 record, aligned8.
- `et_cpu4_private_c4_capture_copy_parameters_v1(capture,dst,4768)` copies all14 canonical value planes, aligned4.
- `et_cpu4_private_c4_capture_release_v1(capture)` consumes only its own payload; exact repeated release succeeds.

The capture owns handle32 + private provenance320 + payload4992 (record224 + values4768): live5344, retired retained352. It is never in the old mixed transport registry. SHA256 covers the explicit little-endian record prefix and parameter words with `cpu4-c4-capture-v1\0`; the bounded private implementation derives from the existing C2 SHA source. Content SHA is neither pointer ownership nor checkpoint authority. Capturing pins the actual14 original identities/values and genuinely reads the cache; all leases close before publication. Capture adds no provider dispatch, numerical replay or RNG draw.

Detached copy/release never dereference weak source generator/model/cache/parameter addresses. Alias checks consult authoritative allocated ranges, including history/capture controls, live/dead capture handles, source contexts/models/manual frames and I1/F32/A2 storage. DEAD must have NULL payload/zero extent/cleared weak words and digest, plus exact header/reserved state, before any hash/free. Mutable handle/record/value/context corruption rejects without consumption and permits exact restoration/retry; authoritative allocation-link corruption is not an admitted control.

Unrelated noncontext legacy transport registry/type-allocation metadata remains trusted preexisting ABI bookkeeping. This feature makes no coherent arbitrary legacy type-tag corruption recovery claim and adds no extent registry. The specified NEW capture handle/record and CONTEXT policy/RNG/history/pointer/tag corruption controls remain admitted: full context spans use trusted history.context, and active or COMMITTED manual-frame spans use trusted history.pending.frame, independently of mutable context fields.

Destinations must denote caller-writable storage for the exact extent. Range/alignment/overflow and known-owner alias checks do not prove arbitrary virtual-memory writability; unmapped/dangling raw buffers are not safely recoverable inputs and no OS probes are performed. TLS destination overlap is rejected before diagnostic reset, even with invalid capture/length/alignment: fixed copy extent and safely representable declared extent are checked numerically, never dereferenced.

## Frontier and control inventory

| Actual source frontier (30f2 reference anchors) | History action / staged controls |
|---|---|
| Prefill1 5230; prefill2 5606; prefill3 4892 | Replace prompt IDs; epoch+1, RNG unchanged. Early target capacity and epoch preflight precedes discovery/A2 creation. Full prior history may be replaced. |
| Token publish4182 | Append selected generated ID and exact successor RNG; epoch+1. Begin checks before sample4032, last-logit begin before copy6036/dispatch6073. |
| P2G1 commit7989 | Same single append; later result publication is not another event. |
| G2 prefix7703 | Append first selected non-EOS2→3 and successor before reset. Subsequent abort preserves it. |
| Manual frame8130/8134 | Kind1 replaces prompt1/2; kind2 preserves original1 and appends one MANUAL→2; RNG unchanged. Candidate-only prefix6611 is not a context event. |
| G2 terminal8698 | First-EOS2→3 or final second append3→4. Final new checks precede the first I1 write as well as A2 commit. |
| Constructors/close | Initialize empty epoch0 before enrollment; clear only after actual cache consumption. Rejected close preserves custody; exact dead validates cleared history. |

APPEND requires a free actual slot and epoch room1 before effects. REPLACE checks its target, not old free space. Compound prompt plus first token checks room2; second G2 checks room1 separately. Pure G0/forward/sample-only queries authenticate provenance but impose no new append-capacity/epoch condition. No manual decode or generated-route capacity admission is widened.

Run host checks only:

```
python3 tests/g3c4/check-cpu4-history-source.py
python3 tests/g3c4/check-cpu4-capture-source.py
git diff --check
```

Repair source controls explicitly apply physical backslash/newline splicing before recognizing macro directives and reject the exact58f LINEAR/LAYOUT defect. Guard dispatch macros are selected outside continued replacement lists. Independent source controls reject the original missing temporary-prefix enrollment and mutable-frame alias scan. The staged manual native fixture checks detached captureA against active AND COMMITTED frameB while mutable frame/tag fields are hidden, sentinel/census preservation, exact restoration/cleanup, stage/commit prefix failure cuts with custody witnesses and retry, and full/MAX committed drain. Its extra linker-wrap requirements are in the fixture header; root owns the roster.

Future native fixture sources (NOT COMPILED/RUN): `test_cpu4_history_native.c`, `test_cpu4_history_manual_native.c`, `test_cpu4_capture_native.c`. They reuse actual terminal/manual predecessor sources and native providers. The history fixture also stages the separate hard replay3 witness: same model/context/original policy2 and committed RNG, actual prefix3 then second rejection/abort, no sampling replay, bitwise all physical K/V cells/mask/length/256 logits and RNG against genuine prefill3. It has not executed and proves no resumed authority by source inspection. The first adds effect-count linker wrappers listed in its header; the root owns the final common compiler/native roster. Controls cover real prompt/manual/prefix/terminal data movement, full-history replacement, fifth append/overflow before effects, prefix-preserving abort, mutable forgeries, leases, allocation faults, exact values, detached lifetime, TLS aliases and DEAD retry. The host checks validate source ordering with causal rejected mutations, source OFF projection, SHA derivation and LP64 arithmetic; they do not execute C.

New source dependencies for root manifest/MMD refresh: `cpu4_c4_history_types.h`, `cpu4_c4_history.inc`, `cpu4_c4_capture_private.h`, `cpu4_c4_capture.inc`, included from the existing owner/context sources. Root must compose the accepted O2/restore support fix `b0af97d1` before a common execution gate; this isolated branch remains based on `fd0db727`.

Future referenced-FILE binding must compare actual decoded14 values and authentic topology with captured bytes; digest alone is insufficient. All C2 image/access/staging owners close before O2 handoff. History3 replay/decode4, persisted sampler/history, resumed import and fresh-process continuation remain separate contracts.
