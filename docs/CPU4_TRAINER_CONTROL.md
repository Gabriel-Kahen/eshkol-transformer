# CPU4 private snapshot/restore control producer

`native/cpu4_trainer_control_extension.esk` adds a separate registered CPU4
trainer/control authority. It never enrolls an M3T owner or calls an old TR3
trainer authenticator. Load it into the same private identity aggregate after
X1, T2/D2, O2, `c2_d2_cursor_pair_extension.esk`, and the CPU4 P1/model binding
extensions. These are explicit source dependencies, not new public imports.
The file uses no stdlib `length`, captured bare externs, or callbacks.

The canonical record has exactly 22 slots:

| Slots | Meaning |
| --- | --- |
| 0–5 | CPU4 tag, exact self, trainer shell, state, active parent token, owner token |
| 6–9 | Resolved config, `[tokenizer-fingerprint,256,#t]`, C4 model, binding token |
| 10–14 | Exact modules17, handles14, carriers14, optimizer shell, genuine O2 record |
| 15–18 | Dataset shell, genuine D2 state, epoch-start cursor bytes, Philox RNG |
| 19–21 | Token count, completed updates, completed epochs |

The own registry additionally retains the genuine tokenizer shell/T2 entry.
Constructor values derive from the accepted binding; callers cannot supply a
parameter roster. Admission requires raw byte tokenization, a genuine idle
N1/T4/V256 D2 stream with a resumable initial position, exact O2 handle order,
zero O2 completed updates and absent gradients. Initial counters are zero and
RNG is `[philox4x32-10,1,config-seed,0,0]`. It rejects shared model, binding,
optimizer, dataset, or parameter handles with another CPU4 trainer, and rejects
another live O2 optimizer over the same handles. Failed construction leaves a
dead rooted identity tombstone; general trainer destruction is outside this leaf.
Constructor final admission proves native IDLE with an actual WRITE enter/leave
pair. Component roster checks alone do not claim to observe native phase.

Actual private signatures:

| Procedure | Contract |
| --- | --- |
| `cpu4-trainer-create-internal(binding,tokenizer,dataset,optimizer)` | Returns registered trainer shell |
| `cpu4-trainer-record-internal(trainer,operation)` | Authenticates own exact token/self and returns its canonical record |
| `cpu4-trainer-idle-check-internal(trainer,operation)` | Rechecks binding/P1, tokenizer, O2, D2 and control/cursor counters |
| `cpu4-trainer-control-begin-internal(trainer,kind)` | `kind` is `snapshot` or `restore`; returns registered active parent after actual native lease WRITE enter |
| `cpu4-trainer-control-end-internal!(parent)` | Checks complete idle component/control state, leaves native WRITE, and closes parent |
| `cpu4-trainer-controls-copy-internal(parent)` | Returns detached five-tuple for the same active trainer |
| `cpu4-trainer-controls-restore-stage-internal(parent,controls)` | Validates/copies controls under an active restore parent; returns registered stage |
| `cpu4-trainer-controls-restore-copy-internal(stage)` | Returns another detached copy for trusted component-plan construction |
| `cpu4-trainer-controls-restore-precommit-internal(stage)` | Rechecks exact active stage/parent and marks stage prepared |
| `cpu4-trainer-controls-restore-abort-internal!(stage)` | Consumes a staged/prepared control plan, preserving canonical controls |
| `cpu4-trainer-controls-restore-commit-internal!(stage)` | Requires prepared exact stage; commits only canonical slots 17–21, then consumes stage |

Parent and stage shells are own registered identities. Copied, foreign, stale,
or repeated committed stage tokens reject. Failed preflight leaves the parent
and trainer in the same state for retry. A pending stage must commit or abort
before parent end. Cleanup after failed component preparation must abort those
component plans, abort the control stage, then end the parent.

The detached controls are exactly:
`[RNG, [fingerprint,256], [current-cursor,epoch-start-cursor],
[canonical-config-UTF8,fingerprint], [tokens,updates,epochs]]`.
All mutable vectors, strings, and bytevectors are copied. Restore requires exact
resolved config bytes/fingerprint, fixed seed/zero RNG counters, raw tokenizer
identity, authenticated D2 cursor bytes and matching live dataset identity with
N1/T4. Counts and products must fit signed i64, with positive accumulation and
`contributions = updates * accumulation = epochs * (total-start) + current-start`;
`contributions <= tokens <= 4 * contributions`. The four-token upper bound is
also required to fit i64; larger contribution products reject explicitly.

G3 must independently authenticate the complete C2 image, all fourteen parameter
values, twenty-eight optimizer moments, optimizer/scheduler state and D2 restore
plan. A valid control tuple or registered stage alone does **not** authenticate
a full checkpoint. Stage controls before component plans hold busy participants;
run control precommit and all component-plan preflights before any canonical
write. Commit genuine I2/O2/D2 plans first, then call the control commit once.
After its exact local authority checks, that final tail has no dependency calls
or allocations. End the parent only after all component plans are finalized.

`step` explicitly rejects as unsupported. Full training needs a separate closed
handoff: native READ frame with children, atomic complete gradient contribution,
child drain/release and READ leave, followed by WRITE admission requiring genuine
complete gradients for O2 update. Snapshot/restore's absent-gradient boundary
cannot serve as that update admission. This leaf makes no training-run claim.

Cheap host checks: `python3 tests/cpu4_trainer_control/test_source_contract.py`.
They parse/check source structure and dependency names, not Eshkol execution.
Pinned-runtime constructor, failure/retry, snapshot/restore transaction, malformed
cursor/counter, retention and sanitizer execution remain the next compiled gate.
