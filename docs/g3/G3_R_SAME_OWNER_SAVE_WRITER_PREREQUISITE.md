# G3-R same-owner SAVE writer prerequisite

**Source-backed dependency, not an accepted writer or API.** This note is
based on main `09f7ed52`, the native owner/test sibling `411d25ef`, and the
private Eshkol copyout candidate `97aec234`. The latter two descend from
`f20cd277` independently; they have not been composed or admitted on main.
The follow-on compiled aggregate has focused candidate evidence; independent
review and current-main union remain gates. This note leaves the
[G3RCV1 data-only contract](G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md),
[record codec](G3_R_RECORD_CODEC_LEAF.md), and pinned G3-R file-pair LOAD
gate unchanged.

## Existing callable boundaries

- `g3r-committed-save-snapshot!(generator, destination)` in
  `native/g3r_committed_save_snapshot_extension.esk` checks the registered
  G3-T generator, its M3T model and raw V256 T1 root. Its native helper pins
  14 parameters and borrows the A2 cache while checking a 136-byte same-owner
  copy. It returns detached policy[6], RNG[4], ordered raw history, prompt,
  generated and manual counts, and generated-EOS classification. It releases
  those leases before return. It supplies no C2 owner or file authority.
- `c2-training-state-compose-internal(request)` in
  `native/c2_training_state_extension.esk` consumes exact P1 and O2 state
  children and validated controls to create a detached C2 owner. The
  candidate copyout neither supplies nor authenticates these children.
- `c2-checkpoint-save-internal!(state, path, policy, options)` in
  `native/c2_checkpoint_save_extension.esk` borrows that detached owner,
  assembles and validates a C2 image, ends the borrow, then publishes its
  bytes through `c1-native-atomic-write`. The private
  `c2-save-with-owner-borrow(state, limits, subpolicy, build, publish)`
  exposes a synchronous image callback, but does not bind the C2 owner to a
  G3-T generator. Neither entry returns the encoded image or its whole-image
  SHA-256 to the caller of ordinary SAVE.
- `g3r-record-encode!(destination, policy, rng, history, prompt, generated,
  manual, fingerprint, c2-digest)` accepts an independently authenticated
  32-byte digest of **the complete immutable C2 image**. It produces only an
  in-memory, data-only 288-byte companion with no executable term. The pinned
  separate file-pair gate at
  `999d9a25` defines `g3r-file-pair-admit-internal` to read and validate an
  existing pair for LOAD; that function is absent from `97aec234` and is not
  a SAVE publisher.

Thus `snapshot -> codec -> two ordinary writes` would join independently
acquired states and could hash different C2 bytes from those published. A
matching tokenizer string, model seed, detached output, caller-supplied
history, C2 trailer checksum, or path reread cannot prove same-owner model
bits or whole-image digest. The 136-byte copy is staging, not G3RCV1 bytes.

## Required reviewed composition contract

Before choosing a writer signature, identify one owning aggregate that
retains the exact live G3-T generator, M3T model, raw V256 T1 tokenizer,
resolved X1/config and dataset/cursor controls, O2 optimizer and P1 capsule.
The bridge must authenticate their registry identities and active edges,
not merely equal values. It must hold the model/parameter, optimizer and
generator/cache state stable from the same-owner snapshot through a bounded
P1/O2/control capture. It must compare the 14 exact parameter identities and
bits represented in that capture with the pinned model image, plus the
tokenizer fingerprint, policy, four RNG words and ordered partition. The
existing copyout releases its leases before return, so simply calling it
before C2 capture does not establish this joint stable interval. A new
coordinated hold or equivalently reviewed identity/version revalidation is
needed. The source-private capture must either build a detached C2 owner
from those authenticated children or return a typed, checked capture that
`c2-training-state-compose-internal` can consume. A successful detached
owner must remain immutable during encoding. Lease order, one-way ownership
transfer, abort cleanup and post-transfer fail-stop behavior need independent
review. A historical divergent branch has
`g3r-save-c2-capture-internal(generator, model, resolved, tokenizer, dataset,
optimizer, capsule)` returning a detached C2 owner, G3 words and epoch. It is
absent from main and this candidate lineage, and does not itself supply C2
image bytes, digest or pair publication. Its `g3r_same_owner_save*` sources
are candidate design evidence, not an API in this lineage.

The writer then needs a single-image publication boundary. A proposed
feature-local entry may accept the exact captured C2 owner, both validated
paths, policy and the detached G3 fields, and return success only after:

1. C2's existing measure/encode/parse checks produce one immutable image.
   Compute ordinary SHA-256 over those exact complete bytes while they are
   retained; do not reopen a path or substitute the C2 trailer digest.
2. Encode a fresh exact 288-byte G3RCV1 record with that digest, the copied
   raw T1 fingerprint, policy/RNG/history/partition and terminal derivation.
   Decode/check the staged record before any file mutation. The current
   codec's nine-argument signature and 288-byte layout remain unchanged.
3. Publish the C2 image to a new path with the existing no-overwrite atomic
   writer, then publish the companion last with the same no-overwrite rule.
   Reject equal paths, preexisting targets and invalid paths under the
   checked I/O contract. The current atomic writer has an unsynchronized
   directory-mutation boundary and trusts intermediate path components; the
   coordinator must state that environmental boundary. Its status
   distinguishes failure before rename from failure after rename, where a
   complete new file may be visible without proven directory durability.
   Failure before companion publication may leave an unreachable C2 image;
   no failure may replace an existing pair, mutate the live generator or
   advertise successful SAVE.
   A committed companion failure needs explicit recovery/admission handling,
   not a claim that the pair is absent.

The writer interface and whether the image callback may safely retain/copy
C2 bytes across C2's regional scratch boundary are **unselected**. A reviewed
signature must state byte ownership and lifetime, digest output ownership,
path reservation/order and the precise result/cleanup record. The existing
`c2-save-with-owner-borrow` callback is a possible source seam, not proof that
its regional bytevector may escape. No implementation should claim this
composition by calling ordinary C2 SAVE and hashing the path afterward.

## Minimum focused gate for a successor

Use one real owning aggregate to compare the captured C2 parameter/optimizer
bits and controls with the same live generator and its snapshot. Cover P1,
P2, manual append, G1 completed and generated EOS; reject idle aborted G1,
busy or borrowed owners, changed model/T1/policy, cross-aggregate handles,
malformed partition and changed C2 capture. Test SHA against an independent
oracle on the exact published C2 bytes, then use the accepted file-pair LOAD
admission to reject a same-size valid C2 substitution. Inject capture,
encode, hash, path, first-write and companion-write cuts; prove no false
success, no replaced file, no leaked authority or active lease. Retry after
a post-rename orphan or committed failure must use fresh pair paths or an
explicitly reviewed recovery/admission path. Run feature-off closure,
compiled aggregate, normal/repeat and
sanitizer gates on one exact composed source head before acceptance. A
successful pair still does not prove cold restore or resumed generation.
