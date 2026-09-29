# G3-C4 public model and input authority proposal

**Independently reviewed additive contract; the model/input leaf has an
isolated package implementation candidate and is not merged.** This is the model/input prerequisite of the
[fixed P2/G1 producer](G3_C4_P2_G1_PUBLIC_PRODUCER_PROPOSAL.md). The merged
diagnostic-C2 G3-G facade and its `P+G<=2` admission remain unchanged. These
names are candidate exports only in the one G3-G package; they do not
constitute a public C4 generation claim.

## Closed proposed surface

| Name | Arity | Result and authority |
|---|---:|---|
| `diagnostic-c4-model-create-seeded` | 1 | New exact C4 model shell from one nonnegative signed-i64 seed; sealed, eval, CPU f32, 14 unique parameters including learned positions `[4,4]`. |
| `generation-c4-input-create` | 1 | New exact C4 input shell owning a dense CPU I1 `i64[1,2]` copy of one authentic same-aggregate sealed T1 `i64[2]` raw-byte prompt. |
| `generation-tensor-release!` | 1, existing | Extend exact-owner dispatch only for C4 input shells; return `#t` on successful or repeated exact C4 input release. |

The constructor is **seeded creation**, not an import of M3/C2 parameters,
an M3T initializer, checkpoint state or arbitrary f32 tensors. It delegates
to `g3c4-model-create-seeded-internal` with the seed unchanged. The native C4
owner and I2/P1 construction remain its sole authority; a public wrapper
neither fabricates a second shell nor changes private `diagnostic-c4` identity.
This adds no C4 `model-forward`, training, save/load, general module mutation
or public generator. The private constructor publishes a live eval-mode model
only after I2/P1 and native seal; failure leaves an unreturned dead tombstone
and unwinds I2 before native owner abort.

There is **no reviewed release for a sealed C4 model**. This leaf adds no
`model-release!` and claims no physical model teardown. The successful
model shell, 14 parameter handles and native owner remain rooted for process
lifetime, including after a future generator closes. This bounded diagnostic
retention is not permission to free a model through C2/M3 or I2 release.
A later release contract requires idle/retention checks and native/I2 teardown;
the constructor's pending-owner abort is not that operation. As A0 requires,
future generators retain caller-owned model/tokenizer references and never
destroy them on `generator-close!`.

## C4 input construction and release

`generation-c4-input-create` accepts only a genuine same-aggregate sealed T1
tensor. Before enrolling C4 input, authenticate T1 and require an exact
contiguous CPU i64 rank-one length **two** descriptor and byte IDs `0..255`.
The source-private `et_g3c4_private_input_from_t1_p2_v1` uses the reviewed
exact-pair T1 reader, then rejects P1 before any owned I1 allocation. It
copies the validated two byte IDs into independent I1 `[1,2]` and enrolls
one live C4 native input. The existing private constructor continues to
support P1/P2. The public wrapper never reads legacy T1 length or storage
before that authoritative native read. G3-T/C2 inputs, T1 shell copies,
output-ID clones, arbitrary
shape-matching I1 owners and raw pointers gain no C4 input authority. No T1
borrow, model/generator reference or aliased T1 storage survives the call.
The sealed T1 source cannot be written or normally released. Copy
independence is proved by distinct native I1 storage, continued T1 source
reachability and rejected post-seal write/abort attempts, not by promising
caller mutation of sealed T1.

The result is the exact `g3c4-registry` entry with kind `input`, profile
`diagnostic-c4`, and its own native I1 owner. Failure before publication
destroys any new native I1, tombstones the pending entry and returns no
shell; it cannot leave C2 provenance or a partial public value. Wrong-kind,
forged or cross-aggregate T1 is `invalid-argument`; genuine non-P2 length
or ID outside `[0,255]` is `shape-mismatch`. Preserve more precise accepted
T1/I1 dtype, device, layout, borrow or allocation errors rather than
inventing a CPU fallback or guessing a category. Re-raise with the public
operation name, bounded data-only details and `cause #f` through E1.

The installed arity-one `generation-input-create` continues to make only
G3-T/C2 owners. For public `generation-tensor-release!`, test exact C4
**transport and model** registry membership before the existing G3-T lookup.
A registered C4 transport `input` delegates to
`generation-tensor-release-internal!`, which destroys
native I1 first, then tombstones the shell. Exact already-dead C4 input
release is idempotent. A registered C4 transport wrong-kind value or an
authentic C4 model shell rejects `invalid-argument` without C2 fallback.
An active I1 borrow or other recoverable native rejection leaves the C4
input live and retryable, with no
partial tombstone. Unregistered values follow the current C2 path unchanged,
including its wrong-kind/dead/error order. Mutable shell tags and C4 native
pointers never authorize G3-T release.
The current `g3c4-registry-find` raises on an unregistered shell; a future
public branch selector therefore needs reviewed **nonthrowing exact-shell
membership probes** for both C4 registries in the one owning aggregate
before either branch's throwing validator. The model probe also serves the
future `generator-create` dispatch. A new public pointer/registry API is
not implied.

The copied input is independently caller-owned while its sealed T1 source
remains live. Releasing an idle C4 input does not close a model; a future
generator must likewise leave it caller-owned on close. A later generate
call with that exact dead input rejects `invalid-state` before cache/RNG
mutation. The future C4 producer authenticates exact C4 shell and live
native owner, then performs P2/G1 full-request admission. Generator-close
and dead-input generation witnesses belong to that later producer gate;
this leaf does not enable `generator-generate!`.

## Package and acceptance gate

Use one source-composed owning archive containing current G3-T public
facade and exactly one C4 model/input registry instance. Boxed E1 exports
and facade entries are added only after acceptance: the model constructor
belongs in `lib/transformer/diagnostic_transport.esk` beside
`diagnostic-model-create`, and the C4 input constructor belongs in
`lib/transformer/generation.esk`. Private C4 model/input functions remain
private archive members, not duplicate public exports. Registry membership,
including dead tombstones, precedes payload access. A registered dead C4
model must reject `invalid-state` at future `generator-create` dispatch,
never fall through to C2; unrelated models retain the C2 path.

Gate two distinct seeded C4 models and independently copied P2 prompts;
the `[4,4]` position/14-handle/tie/eval identity; bytes `0` and `255`;
continued T1 source reads, rejected post-seal write/abort, and disjoint
owned I1 storage with unchanged copied IDs. Cover wrong seed,
P1/P3/empty prompt and `-1`/`256` IDs. Malformed authentic T1
rank/extent/dtype/device/layout/storage needs an explicit **test-only
descriptor/storage corruption hook**, with fixture-owned restoration; a
forged T1 pointer alone does not prove this rejection. Also cover forged,
copied, dead and cross-registry shells, held I1 borrow,
wrong-kind/repeated release, constructor/allocation
failure cuts, preservation and same-authority retry. Re-run unchanged C2
input/release tests; prove one owner archive, exact exports/closure,
private-link negatives, fresh AOT callers, feature-off and supported pinned
fe9 normal/repeat/sanitizer gates. Input release after generator close and
dead-input generate rejection move to the later producer gate. No public
C4 result/generation follows until the production coordinator and output
accessor/release adapter are
reviewed and composed. EOS, loop, N>1, persistence and CLI3 remain separate.
