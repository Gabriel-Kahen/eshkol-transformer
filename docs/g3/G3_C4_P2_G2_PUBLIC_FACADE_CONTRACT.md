# Public C4 P2/G2 generation facade contract

Status: **source-only proposal, not an approved API contract or admitted public
route**. Source basis is `c7e6ff3c8a3d860af6df88219f41b68743e9f070` /
tree `865f40584938a93bcf8c9bbd9049ebd648eb5cd1`. The private same-call
terminal source is a candidate; its current-main compiled, repeat, sanitizer,
emitted-tail and predecessor gates remain open. This contract does not authorize
installed dispatch or imply G3-R SAVE/continuation support.

## Existing symbols and missing bridge

The public names and arities already exist in `lib/transformer/generation.esk`:
`generator-create(model, tokenizer, config)`, `generator-generate!(generator,
input)`, the five `generation-output-*` readers, and the tensor, RNG, output
and generator release operations. The installed diagnostic manual aggregate
`g3g_manual_package_root.esk` loads
`g3g_manual_public_extension.esk`,
`g3c4_public_generation_dispatch_extension.esk` and
`g3c4_public_output_adapter_extension.esk`; its public C4 route admits only
P2/G1, budget one and EOS disabled. The separate older C2/G0 aggregate
`g3g_g0_package_root.esk` has no C4 route. Neither loads the private
P2/G2 terminal candidate. The C4 public result constructor
`et_g3c4_private_result_tensor_create_v1(kind, value)` creates only scalar
`i64[1]`; IDs, lengths and cache lengths are fixed to ID 0..255, 1 and 3.
The authentic public C4 model and owned P2 input constructors are also
manual-package-specific: `diagnostic-c4-model-create-seeded` is in
`native/g3g_manual_facades/transformer/diagnostic_transport.esk`, and
`generation-c4-input-create` is in
`native/g3g_manual_facades/transformer/generation.esk`. A G2 public route
must compose those exact constructors in the same installed aggregate; the
ordinary `lib/transformer/generation.esk` and older C2/G0 root cannot
provide a public C4 P2 input.

The actual private terminal operations are
`g3c4-p2g2-terminal-generate-internal(generator,input)`, its
`g3c4-p2g2-terminal-snapshot-internal(output)` eight-word detached read,
`g3c4-p2g2-terminal-ids-clone-internal(output)`,
`g3c4-p2g2-terminal-ids-copy-internal(clone)` and the separate terminal and
clone release operations,
`g3c4-p2g2-terminal-release-internal!` and
`g3c4-p2g2-terminal-ids-release-internal!`. The native clone/copy/release ABI is
`et_g3c4_private_p2g2_terminal_ids_clone_v1`,
`et_g3c4_private_p2g2_terminal_ids_clone_copy_v1` and
`et_g3c4_private_p2g2_terminal_ids_clone_release_v1`. Its private kind-9
logical I1 owner is independent of its terminal source, but its one-element
Eshkol list wrapper is **not** a public tensor owner. The private snapshot
returns `(id0,id1,logical-count,cache-length,rng0,rng1,rng2,rng3)`;
`id1=0` is an unused physical slot on first EOS and must never be emitted.

## Proposed public admission and result

The new explicit public config profile is `diagnostic-c4-p2g2`. The public
facade preserves the existing constructor order: authenticate C4 model first,
parse the complete flat eight-pair config, translate the public profile to
private `diagnostic-c4` and normalize its policy, then authenticate the
tokenizer in `g3c4-generator-create-with-source`. A foreign tokenizer with a
malformed config must therefore retain the config error precedence. It admits
exactly CPU-f32/V256, same-aggregate raw T1 tokenizer,
an owned P2 input of two byte IDs, batch one, budget two, supported
greedy/categorical policy and `:eos` of `#f` or a byte ID. Seed or authentic
detached C4 result RNG remains exclusive. No other profile obtains budget
two through this route. Preflight authenticates both owners and the exact
native request before the private terminal call. All sampling, T1 decode,
prefix commit, EOS branching and second draw stay inside that accepted
same-call coordinator; the facade supplies no sampled IDs or cache state.

`generator-generate!` returns the original private terminal shell as the
public output identity once the route is installed. It contains no live
call, generator, model or tokenizer ownership edge after success. A first
draw equal to EOS returns logical G1: IDs `i64[1]`, one text byte, length
`1`, cache length `3`, no second draw and no visible second slot. Otherwise
the second draw returns logical G2: IDs `i64[2]`, two text bytes, length `2`,
cache length `4`. A second-draw EOS is included as the second ID/text byte.
The authenticated terminal snapshot supplies the four final RNG words.

Each public accessor first authenticates the exact terminal output and
fresh detached snapshot, checks the selected logical IDs against the private
clone, and checks the copied text against the terminal's authenticated T1
bytes before publication. `generation-output-ids` creates a **new public
rank-one CPU I1 owner** with exact logical shape `[1]` or `[2]` and copied
IDs; `generation-output-lengths` and
`generation-output-cache-lengths` each create a new public scalar `i64[1]`
owner holding the authenticated 1/2 or 3/4 value. `generation-output-rng`
creates a new opaque, immutable, separately releasable C4 RNG owner from
the snapshot words. `generation-output-text` returns a fresh singleton
list containing a detached raw bytevector of length 1 or 2. Every call
returns independent storage. The result tensor/RNG owner and returned text
retain only data, never a terminal, clone, call, generator, model,
tokenizer, cache or input authority. They remain readable/releasable after
terminal output release and generator close. Releasing one result leaves
other results and the output unchanged; a second exact-owner release is
idempotent. A caller mutation of returned text cannot change later reads.

The private kind-9 clone may be used as an authenticated source during an
accessor, followed by checked copy and release on both success and every
recoverable failure. No failure may leave a live unpublished clone or partial
public owner. It must not be returned as
the public IDs owner or cast to one. A reviewed native public length-2 I1
constructor/adapter and scalar result constructors for values 2 and 4 are
therefore prerequisites. Their implementation must prove exact kind, shape,
storage, lease, error, and release semantics; the current scalar constructor
does not meet this contract. Snapshot and copy must complete before source
release; the public result must survive both terminal and private clone
release. No accessor may manufacture a public result by relabeling a C2
owner or publishing the capacity-two native buffer on G1.

## Dispatch, failure and acceptance gate

The existing public facade must classify membership in both C4 registries,
including dead shells, before considering the C2 path. Authentic wrong-kind
and forged identities return `invalid-argument` before any liveness check;
dead authentic owners on read or generate return `invalid-state`. An exact
dead result/output owner on its matching release remains idempotent. Full
request validation precedes the irreversible
terminal call. Use the existing public E1 error boundary with the invoked
operation name, bounded data-only details and no cause. On accessor failure,
discard only unpublished detached allocations; the terminal output and
previous results remain live. An authentic held I1 lease rejects release
without mutation and permits retry after the lease ends.

Acceptance requires, in dependency order: accepted current-main compiled
private terminal gate; reviewed public length-2 typed-result ABI and Eshkol
adapter; same-registry public C4 dispatch and package closure; normal,
repeat and sanitizer runs of genuine P2/G2 greedy/categorical, first EOS,
second EOS and non-EOS, independent Philox/T1/ID/cache references, G1
unused-slot exclusion, source-release persistence, lease/retry, malformed
owner/config/shape, failure cuts, feature-off and unchanged C2/P2-G1 cases.
Only an installed-package runtime and symbol/source-closure gate can admit
the public facade. No G3-R SAVE or data-only continuation format is defined
here.
