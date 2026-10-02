# G3-R data-only generation continuation contract

**Proposed prerequisite; no save, load or resumed-generation implementation is
accepted by this document.** The first bounded witness targets the existing
CPU-f32 `diagnostic-c2` generation profile with one batch row, raw V256 T1,
history length one or two and policy budget zero or one. It follows the
[G3 serialization requirement](../G3_GENERATION_PROPOSAL.md#serialization-and-acceptance-still-required)
and the accepted [C2 detached checkpoint](../C2_TRAINING_STATE.md). It adds no
public name, package export or C2 file-format change. A larger admitted
profile with repeated decode and later-token EOS remains necessary for full
G3-R acceptance; this bounded proof cannot substitute for it.

## Existing authority and dependency gap

C2 1.0 has a versioned, SHA-256-checked, data-only model/training-state image
and atomic publication. Its current K2 `tensor.f32/storage.copy` capability
admits rank zero/one and exact rank-two `[2,4]`, `[4,4]`, `[4,8]`, `[8,4]`
and `[256,4]`. These include the listed C4 parameter shapes; a generic
"rank-two unsupported" objection is false. C2's C4 position-table schema
witness proves storage admission, **not** restoration of a live C4 generator
model. C2's X1/dropout RNG and resolved training config neither serialize
nor authenticate G3-S's separate Philox state or generation policy.

The existing G3-T output RNG clone is an authenticated, independently live
four-word owner, and `g3t-generator-create/3` with `:rng` copies all four
words from such an owner. Seed construction supplies only the initial
counter. There is no accepted public or source-private G3-T factory that
reconstructs that owner from checked file words; the C4 result-RNG helper is
a different owner and not an import ABI. A reviewed typed reconstruction
entry, with exact signature and ownership chosen before code, must validate
version, nonnegative seed, two full counter words including the exhausted
sentinel, same-aggregate registry, liveness, and allocation rollback before
the existing `:rng` constructor can be used. It must not expose a raw
four-word vector as an owner.

The existing output RNG clone exposes a completed output's successor RNG,
not an arbitrary live generator/history save snapshot. A reviewed same-owner
save seam must capture the committed policy, complete ordered history,
generated/manual partition, terminal state and all four RNG words under
stable leases before writing either file. It must reject active calls and
forged, stale or cross-aggregate owners; reading a raw native pointer or
reconstructing history from detached output bytes alone is insufficient.

G3-M manual P2 prefill already authenticates owned T1 `[1,2]`, computes
last-row logits and a length-two A2 cache without a G3-S draw; manual P1
prefill and P1-to-position-one decode also exist in the private lineage.
These are replay primitives, not a restore compositor. The test-only
CLI3/G3-G checkpoint-to-generation aggregate already composes public C2
LOAD/TR3 restore with an exact M3T model and genuine P1/G1 T1 input in one
registry. It proves live-model restore reachability, not a typed G3 RNG
import, full-history replay or a G3-R transaction. A reviewed C2 LOAD to
exact live M3T model/eval/parameter identity, a same-aggregate T1 owner,
and an atomic candidate generator/cache replay transaction remain prerequisites.
No seeded C4 initializer, equal-seed model, copied cache pointer or
detached C2 training-state shell may stand in for restored model parameters.

## Canonical bounded continuation record

The proposed `G3RCV1` companion is exactly **288 bytes**, independently
SHA-256 checked, and binds one immutable canonical C2 1.0 image by the raw
SHA-256 digest of that image's complete bytes. It contains no path. The C2
image is staged and validated first; the companion is atomically published
last without replacing an existing pair. A crash before companion publication
may leave an unreachable C2 image, never a readable partial pair. A missing
or mismatched pair rejects before any live owner is changed. C2 already owns
its checked LOAD/SAVE and capability admission; the G3 companion parser,
whole-image SHA binding, pair publication and typed reconstruction are
**proposed work**, not existing C2 or G3 parser/SHA APIs. This is a proposed
G3 wire choice, not an extension to the accepted C2 schema.

| Offset | Bytes | Canonical field |
|---:|---:|---|
| 0 | 8 | ASCII `G3RCV1` followed by two zero bytes |
| 8 | 2+2 | little-endian major `1`, minor `0` |
| 12 | 4 | little-endian total length `288` |
| 16 | 4 | profile `1` = exact `diagnostic-c2` |
| 20 | 4 | algorithm ID `1` = exact `g3.philox4x32-10.categorical-f32.v1` |
| 24 | 4 | full committed history length, `1..2` |
| 28 | 4 | origin prompt length, `1..2` |
| 32 | 4 | prior generated count, `0..1` |
| 36 | 4 | prior manual append count, `0..1` |
| 40 | 48 | six little-endian i64 policy words: mode, exact binary32 temperature bits, top-k, exact binary32 top-p bits, requested `max_new`, EOS `-1` or byte ID |
| 88 | 32 | four little-endian i64 G3-S words: version `1`, nonnegative seed, counter-low, counter-high; preserve both counter bit patterns |
| 120 | 2 | committed raw-V256 history IDs in order, zero in unused slot |
| 122 | 6 | zero padding |
| 128 | 96 | exact ASCII T1 `sha256:eshkol-byte-tokenizer-v1:` fingerprint returned by the registered raw tokenizer |
| 224 | 32 | SHA-256 of the complete companion C2 image |
| 256 | 32 | SHA-256 of bytes `0..255` of this record |

All integers use fixed little-endian encoding; signed i64 words retain
two's-complement bits. The version/profile/algorithm IDs are exact;
binary32 controls occupy the low 32 bits of their i64 words with zero high
bits. `prompt + generated + manual = history` exactly. In this capacity-two
slice, a generated or manually appended byte requires an origin prompt of
length one; the two counts cannot both be one, and a generated count of one
requires `max_new=1`. No history byte outside `0..255` is representable. An
emitted EOS is included in history and generated count; EOS already in the
prompt or appended manually does not terminate generation. A saved generated
last byte equal to configured EOS retains its terminal meaning; a manual or
prompt match does not. No synthetic EOS appears at budget exhaustion.
Policy fields must pass the existing exact diagnostic-C2 generation bounds
and greedy constraints,
including `max_new` zero or one. The T1 fingerprint must equal both the
same-aggregate registered raw/empty-special V256 tokenizer and the C2
tokenizer field. The loaded C2 model must match the fixed M3T profile
identity and exact restored parameter bits; digest equality alone is not
permission to relabel another model. The record has no executable term,
cache bytes, pointer, registry ID, lease, graph or provider authority.

## Restore and replay transaction

Read the complete fixed record with exact EOF, length/version/reserved-byte
and SHA checks before interpreting fields. Validate the companion C2 file
through its own complete load/capability path, then its whole-file digest,
tokenizer field, model schema and rank/shape admission. Authenticate the
target runtime/provider, model profile and eval mode. The caller's policy
may lower C2's resource limits but cannot widen them; the G3 record itself
has an exact 288-byte limit and two-token history cap. The first bounded
profile must make no unsupported C4 restore or larger-context claim.

Construct fresh candidate model, tokenizer binding, typed G3 RNG, generator
and owned history input under one cleanup ledger. Use manual P1 or P2
prefill/replay as the history requires, then any admitted manual decode;
never call `generator-generate!` during reconstruction. Replay must run
no-grad, consume **zero** G3-S blocks and compare all four RNG words before
and after. Do not admit a new generation request merely because replay was
allowed: a saved full length-two history with policy `max_new=1` replays by
manual P2 prefill, then a new request rejects the full C2 context without
a draw. Commit the candidate ownership only after model, tokenizer, cache,
logits, history and RNG authentication succeeds; all fallible allocations,
borrows, provider calls and release preflights precede that commit. On any
failure, release only candidate owners in dependency order, preserve the
existing caller model/generator/cache/RNG and old outputs, and permit a
fresh valid retry. Held I1/A2/model leases and busy frames reject before
mutation. A defect after the first closed-tail mutation fails stop; no
partial resumed generator or stale registry edge escapes.

## Acceptance and remaining full-G3 dependency

Compare three independent processes: an uninterrupted admitted generation,
a producer that saves the exact model plus G3 record after a nonzero history,
and a fresh process that LOADs/reconstructs and continues. Equality means
exact restored parameter bits, unchanged gradient values/counts, tokenizer
fingerprint,
policy bits and all four RNG words; bitwise all-256 replayed last logits,
K/V and cache masks/lengths; then, where context permits, identical emitted
IDs, raw bytes including EOS, logical output lengths/cache lengths, next
logits and successor RNG. A saved P2 full history must reproduce its logits
and no-draw full-context rejection. Repeat with greedy, categorical,
singleton, carry, final consumable and exhausted counters; a saved seed
alone or a fresh equal-seed initializer is a negative control, never proof.
The actual resumed categorical ID and successor words must also match an
independent G3-S/Philox reference for the exact saved logits and counter.

Reject truncated/extended/torn records, bad SHA, version, reserved bytes,
wrong algorithm ID, policy bit patterns, history partitions, mismatched or
orphaned checkpoint/companion publication, tokenizer/model digest or profile,
wrong-kind C2 file, unsupported shape, stale model, forged/cross-owner RNG,
busy/borrowed owners and replay/provider/allocation cuts without partial
publication. Check exact dead/idempotent releases, detached old-output
survival, unchanged gradients and no graph allocation during replay;
feature-off and existing public C2/G3-G admissions stay unchanged. Supported
fresh-process normal/repeat and ASan+UBSan+LSan gates, source/symbol closure,
capability negatives and exact-head CI are required before implementation
acceptance. This C2-sized witness is only a prerequisite. Full G3-R still
requires a separately admitted larger profile with true repeated-token
manual replay, later-token failure/EOS, and full saved/restored continuation
equivalence; the present C4 P2/G2 prefix leaf and C2 storage witness do not
provide it.
