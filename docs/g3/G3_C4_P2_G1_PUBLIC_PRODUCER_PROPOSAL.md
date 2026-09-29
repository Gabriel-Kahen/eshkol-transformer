# G3-C4 fixed P2/G1 public profile and producer proposal

**Contract for independent review; no public C4 implementation is accepted.**
This is a separate profile proposal, not an expansion of the installed
`diagnostic-c2` admission. It depends on the reviewed private C4 shell,
[copy-out](G3_C4_P2_G1_COPYOUT_PROPOSAL.md), I2 overlap helper, and the
[public-profile gate](G3_C4_P2_G1_PUBLIC_PROFILE_GATE.md). The merged G3-G
public P1/G1, G0 and manual routes (#140/#142/#146) remain authoritative for
C2; the [generation proposal](../G3_GENERATION_PROPOSAL.md) still limits that
model's two learned positions to `P+G<=2`.

## Profile and authentic owners

Propose the separately named public profile **`diagnostic-c4-p2g1`**. It maps
only to the existing private `diagnostic-c4` model identity, with 14 unique
f32 parameters and learned positions `[4,4]`; it never rebrands an M3/C2 model
or changes its position tensor. This first public leaf fixes `N=1`, raw-byte
T1 V256 tokenizer with its actual fingerprint, `P=2`, `G=1`, CPU f32 model
execution, CPU i64 owned input/output IDs, and cache length exactly three.
No GPU, hidden transfer, alternative tokenizer, P1/P3 request, zero-budget
request, batch widening, or arbitrary C4 capacity is admitted here.

The eight-key public generation config keeps the accepted key names and
bit-exact sampling rules. The profile value is `diagnostic-c4-p2g1`,
`:max-new-tokens` is exactly `1`, and `:eos` is **`#f`** in this leaf. Greedy
retains temperature=1, top-k=256, top-p=1; categorical admits the already
reviewed private temperature/top-k/top-p ranges and seeded or logically
copied authenticated C4 RNG state. A byte EOS is a later semantic contract,
not a way to relax capacity. A full request is admitted before cache mutation;
successful output contains one selected byte, generated length one, cache
length three and the four successor RNG words.

An authentic C4 model must come from a separately reviewed public C4 model
authority. Today `g3c4-model-create-seeded-internal` is private, and the
installed public constructor accepts M3/C2 model owners. A public C4 model
constructor/import and its lifetime/release policy must be contracted and
integrated before this producer is callable. Likewise, today's public
arity-one `generation-input-create` creates a G3-T/C2 owner; the private
`generation-input-create-internal` creates a different C4 owner. The public
C4 input creation/admission seam, including its exact name, arity, and
release dispatch, requires a separate additive API decision. It must copy
an authentic same-aggregate sealed T1 `[2]` byte tensor to a C4-owned dense
I1 `[1,2]`, reject malformed shape/storage and IDs outside `[0,255]`, retain
no T1 borrow, and not make C2 input or output clones valid C4 prompts. No
current public call is asserted to create either C4 owner.
The proposed additive names, exact release dispatch and process-lifetime
model limitation are specified in the separate
[public C4 model/input authority contract](G3_C4_PUBLIC_MODEL_INPUT_AUTHORITY_PROPOSAL.md);
they remain unimplemented and subject to independent API review.

## One package and one producer transaction

Use **one source-composed registry-owning archive**. Add C4 source files to
the owning G3 package closure, never link a second copy of their module
registries. A future `generator-create` wrapper first tests exact C4 model
registry membership, including tombstoned entries. An authentic dead C4
model rejects as `invalid-state` before any C2 fallback; an authentic live
C4 model takes the C4 branch. Models absent from the C4 registry take the
existing C2 path byte-for-byte, including its config validation and
`P+G<=2` errors. For a live C4 model, validate the explicit public profile
and config, same-aggregate tokenizer, eval state and native C4 owner before
allocation.
Only this authenticated public C4 branch may translate
`diagnostic-c4-p2g1` into the existing private `diagnostic-c4` config;
passing the public symbol directly to `g3c4-generator-create-internal`
would be rejected. That translation grants no new native call ABI.
The C4 branch retains an authentic C4 registry member, not a C2 shell with
a changed tag. The eventual public `generator-generate!`
wrapper must dispatch by exact C4 generator registry membership (including
dead entries), then require an authentic C4 input owner; wrong-kind/forged/
cross-aggregate is `invalid-argument`, and a dead authentic owner is
`invalid-state`. Mutable public payload tags are never authority. Both
profile branches retain their own native owner and protected registry.
The C4 public dispatch and return are **not enabled by the private producer
alone**: an accepted exact-owner C4 output accessor/release adapter must
also be present so callers can use and release the returned output. G3-T
clone functions cannot read C4's cache length three.

The C4 producer must be a **production private coordinator**, not
`tests/g3c4/p2_g1_shell_native.c`'s `shell-test-prepare`, which hardcodes
`{0,255}`. After the full-request preflight it enters the reviewed sibling
`g3c4-with-p2g1-output-call-internal` scope and reserves the raw/staging
owners. Within that authenticated active call, it runs genuine P2 prefill
from the caller's owned `[1,2]` IDs, opens the accepted G3-S last-logit
sampling frame, and runs genuine T=1 token forward. It then invokes
`g3c4-p2g1-output-publish!` exactly once. That existing operation performs
native output prepare, ID copy, actual V256 T1 raw decode, native text
acceptance, mutable-carrier rechecks, frame prepare/commit and finish; the
coordinator must not repeat those steps. The shell contract owns the commit
cut: A2 cache, selected ID, RNG, native output and raw Eshkol shell become
live together;
its preflighted native-finish/shell tail performs no fallible work. The
private producer returns that exact output shell; it does not copy out or
create public accessor clones during the transaction.

Before commit, failed prefill/frame/sampling/decode/allocation must leave
no public output, no successor RNG, and no partial appended cache; abort
must preserve the documented pre-call or prompt-only state at each cut.
Held I1/A2 borrows reject before mutation, and an altered active child
edge follows the reviewed shell fail-stop rule rather than aborting another
owner. Retry from a valid state must match the independent G3-S/Philox
sampled-ID/RNG oracle and the same-provider P3 logits/K/V reference.
Output survives generator close. No public `generator-generate!` call may
return this C4 shell until an exact-owner public C4 accessor/release adapter
is accepted and composed into the same package. No C4 output may be passed
to C2 release or accessor functions; the C4 adapter must follow the
separate public-profile gate's fresh-clone, native-first release and error
requirements.

## Acceptance and remaining dependency

Review the public C4 model/input authority and lifetime decisions first,
then the private production coordinator. Review and integrate the exact-owner
C4 output accessor/release adapter before enabling public package dispatch
or return. Gate real T1 P2 prompts with at least two
distinct token pairs, greedy and categorical with nontrivial top-k/top-p,
selected ID/raw parity, committed cache length three, all successor RNG
words, two-call isolation and generator-close survival. Exercise malformed
I1 shape/descriptor/storage, `-1`/`256` IDs, forged/cross-registry/dead
owners, held I1/A2 borrows, allocation/decoder failures, abort and same-state
retry. Re-run C2 P1/G1, G0 and manual tests unchanged; prove one owning
archive, exact exports/closure, feature-off behavior and supported pinned
normal/repeat/sanitizer evidence before claiming public C4 production.

EOS stop policy, repeated `G>1` decode, `N>1`, cache/no-cache parity,
save/reload equivalence, a general public generator and CLI3 `generate`
remain separate dependencies. None is inferred from this one-token output.
