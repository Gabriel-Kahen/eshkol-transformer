# CPU4 frame-local training transport contract

Independently reviewed bounded implementation contract (four clarifications adopted; native span helper delta approved) on top of approved typed lease R2/063bc4c93.
All declarations are source-private under ET_CPU4_TRAINING_PRIVATE. No installed
exports or M3T owner conversion. This is the next executable same-owner training
substrate, not a new generic graph system.

```
void *et_cpu4_private_input_create_v1(void *lease, int64_t a, int64_t b, int64_t c, int64_t d);
int64_t et_cpu4_private_input_release_v1(void *input);
void *et_cpu4_private_logits_create_v1(void *lease);
int64_t et_cpu4_private_logits_copy_from_v1(void *logits, void *bytevector_header, int64_t byte_count);
int64_t et_cpu4_private_logits_copy_to_v1(void *logits, void *bytevector_header, int64_t byte_count);
int64_t et_cpu4_private_logits_release_v1(void *logits);
void *et_cpu4_private_workspace_create_v1(void *lease, void *input, int64_t mode);
int64_t et_cpu4_private_workspace_role_v1(void *workspace, int64_t family, int64_t index, int64_t reverse);
int64_t et_cpu4_private_workspace_copy_logits_v1(void *workspace, void *logits);
int64_t et_cpu4_private_workspace_vjp_begin_v1(void *workspace, void *upstream);
int64_t et_cpu4_private_workspace_check_primals_v1(void *workspace, int64_t mode);
int64_t et_cpu4_private_workspace_contribute_v1(void *workspace, int64_t mode, int64_t weight_bits, int64_t ordinal);
int64_t et_cpu4_private_workspace_release_v1(void *workspace);
int64_t et_cpu4_private_last_error_domain_v1(void);
int64_t et_cpu4_private_last_error_category_v1(void);
int64_t et_cpu4_private_last_error_code_v1(void);
```

Domains: 0 transport using ET_M3T categories/codes, 1 K1, 2 I1, 3 I2, 4 C4.
Capture downstream error triples immediately before cleanup. New transport has
its own registry; never enroll borrowed owner/lease in M3T registry. Controls
retain inert tombstones; payload tensors are released. Report control retention
per frame and payload live counters separately.

Input/logits/workspace own native child kinds1/4/2. Every create authenticates the
exact lease and registers a unique child in its active READ_FRAME before public
publication. All subsequent calls authenticate exact own registry identity and
current lease roster; foreign/copied/dead/mixed-lease objects reject. Children
cannot survive leave, because the native lease itself rejects it. No separate
public graph/control: a single-use workspace owns the immutable primal capture
and complete forward/VJP scratch for one frame. At most one live workspace exists per lease/READ frame, enforced by its native registry. It never resets or crosses a
frame; equivalent graph obligations remain in its own snapshots/ready bits.

Input/position storage is inline native i64[4] inside the authenticated controls, with dense kernel views constructed on the stack; no imported I1 carrier or I1 tensor lifetime exists. Input is copied once during create after validating all four IDs in [0,255].
Workspace create accepts mode0/1, same-lease input, snapshots fourteen canonical
parameters/identities/value tensors plus four IDs. It allocates slots with exact
shapes: token[1,4,4], stored heads[1,2,4,2], FFN[1,4,8], logits[1,4,256],
positions[1,4]={0,1,2,3}, keep[1,4,4]=16 ones, epsilon bits0x3727c5ac.
Parameter13 position is [4,4]. All create failures free unpublished allocations
and leave canonical bytes, gradients, owner and source input unchanged.

Each role dispatch consumes exactly one existing reviewed provider operation.
Families are closed0embedding,1linear,2norm,3GELU,4residual,5split,6merge,
7attention,8sum. Indices/reverse are exactly the existing M3T descriptors; all46
roles specialize T4 explicitly, split/merge request[1,4,2,2] but stored
heads[1,2,4,2], attention request[1,2,2,4,4,2]. Workspace phases are FORWARD -> REVERSE -> CONSUMED -> DEAD. All role/copy/seed/contribution calls reject CONSUMED; only release remains. Contribution requires both captured mode and supplied mode equal1. Forward/reverse ordering belongs
in Eshkol, preserving native/m3_schedule.esk's21/25 calls. No scalar numerical
fallback or native full-model schedule. Ready destinations publish only on
successful provider invocation. Reverse begins once after full forward logits.

Logits bytevector requires count4096 and matching length header4096; rejects
2048/4092/4100 before writes; 1024 binary32 words preserve explicit little endian.
Workspace copy_logits and vjp_begin require same-lease destination/upstream.
Seed copying rejects every nonfinite binary32 word before writes. Caller-provided bytevector storage must satisfy the existing source-private valid-header/readable-writable-span ABI; it is not an arbitrary native-pointer validator. Preflight the entire4104byte extent before reading the header, treating every nonzero I2 owned-span query result as rejection. Before reading a header or writing a payload, reject overlap with registered transport controls (including their inline i64 IDs/positions), canonical C4 owner/lease/child controls and all live/retained I2 owned spans. Reuse the existing ET_G3R_CANDIDATE_RETIRE_PRIVATE I2 owned-span query (already needed by CPU4 candidate retirement), plus one native C4 control-span query under ET_CPU4_TRAINING_PRIVATE. No application authority arises from the helper. Full VJP completion requires all fourteen unique gradients ready including tied
head+token sum and full position[4,4]. No caller-imported canonical roster.

Contribute permits mode1 only (canonical Eshkol binding separately authenticates
P1 train mode). It checks exact captured roster, current values byte-equal to
snapshots, constants including all four immutable captured IDs, all gradients ready, and unborrowed scratch. It prepares
one existing I2 gradient plan for all14 canonical parameter refs, supplied valid
nonnegative ordinal and uint32 weight bits, then commits/releases in a no-fail
tail and marks the workspace consumed. Failed preparation preserves all14
canonical gradient bytes/metadata. No canonical values, optimizer or scheduler
are modified. Eshkol leaves READ only after all children release, then enters
WRITE around the existing P1/O2 transaction. No graph survives an update.

Release preflights all owned tensor storage before child unregister/destruction;
retry after an active borrow ends succeeds, repeated exact dead release returns0.
Do not unregister the native child on failed payload-release preflight. Successful
child unregister precedes no-fail payload destruction/markdead; unexpected cleanup
defect aborts. Controls remain single-use to satisfy R2 child-identity constraints.

Next executable gate: supported CPU-f32 canonical C4 21forward/25reverse fixture
against independent Python binary64 full-model oracle and finite differences,
all14 canonical gradient contribution/identity preservation; malformed shape,
foreign/mixed/dead/phase/borrow/allocfail/stale/retry negatives; normal/repeat and
ASan/UBSan/LSan; feature-off closure unchanged. Then thin Eshkol training smoke
with actual P1/O2 and same-owner SAVE/restore/eval/generation. No caller may claim
that later integration from this transport-only gate.

Native-only support declaration, same serialized trust boundary:
`int64_t et_g3c4_training_span_disjoint_internal(void *lease, const void *span, size_t bytes);`
Authenticates the lease first and checks overflow/nonzero span and overlap against
all registered C4 owner and training-lease/child control extents. Returns0 iff
proved disjoint, no mutation. It does not claim general pointer writability.
