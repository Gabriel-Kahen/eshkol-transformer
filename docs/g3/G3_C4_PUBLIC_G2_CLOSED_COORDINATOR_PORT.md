# Public G2 closed-coordinator port (source admission only)

The accepted public target is `diagnostic-c4-p2g2`: CPU-f32 canonical C4 model,
raw V256 tokenizer, P2 prompt, budget two and explicit integer EOS 0..255.
First EOS returns one ID; otherwise the terminal output owns two logical IDs.
There are no new public names or arities. P2/G1 and C2 public routes remain.
This ports the reviewed `cdf5f5e` public adapter onto terminal union
`4506bc8f89bfe8083f5c47d71dcad74d968222c5` and the reviewed typed stack
`2d607e2a` + `43756db16` + `136076ce5` + `666d54eb`.

## Closed ticket composition

The original two-argument `g3c4-p2g2-terminal-commit-private!` remains callable.
Its lexical core accepts `(generator input enrollment)`; the private route uses
`#f` and fresh dummy marker carriers. The only additional initializer is
`g3c4-p2g2-terminal-enrollment-route-private(stage, roots7)`. The owning public
dispatch consumes it exactly once, before installing its callable producer.
The installer seals before returning a three-argument route `(generator input
ticket)`, captures the original secret bridge and seven distinct singleton root
vectors, and rejects a second installation with exit 134. There is no ambient
current-ticket setter. Roots are ticket/output-row/output-anchor/output-binding,
then generator-row/generator-anchor/generator-binding. The bridge enrolls all
pending nodes before calling the route; bind changes existing fields, not heads.

The route captures the seven exact pending node heads before any stage. The
core invokes only `admit`, `bind`, and `verify`, all before native commit. Bind
receives the original pending output, live record and original-authority closure.
Verify binds the selected count and IDs; it makes no final-RNG assumption.
After the last callable stage, core rebuilds the authentic 31-slot precommit
frame and checks every captured slot, root head, ticket21/row12/anchor12 identity,
marker phase, policy word, output/native/source binding and live authority.
Original live closure and node identities are captured before bind. Both public
rows are checked field by field. Only native commit follows these final checks.

On native zero, the original irreversible barrier is first. Three fixed stores
set captured ticket slot 12, row slot 8 and anchor slot 8 to `live`. The original
closed fixed-write tail follows. No callback, allocation or recoverable validator
runs there. A postzero exception exits 134. Before zero, the original coordinator
guard completes native abort/drain before propagating failure. Only then can the
outer public guard restore its own exact pending root heads and mark its ticket
and rows dead; mismatched cleanup or failed abort exits 134.

## Detached results and continuation

IDs, length and cache owners delegate to the reviewed logical-clone and typed
wrapper stack: native kinds 10/11/12, dense CPU-i64 rank one `[1]` or `[2]`.
Public provenance is independent of the private typed authority. Public release
maps the opaque internal typed kind to the existing operation kind and invokes
its authentic private release; held-I1 rejection preserves authority. Text is a
fresh logical bytevector. RNG remains an independently owned scalar RNG record.
Detached snapshots are header-aware, 64-byte copies with extent 72; all four RNG
words are decoded after terminal publication, including signed words.

The existing generator `:rng` selector now authenticates G2 public result rows as
well as the unchanged P2/G1 path. It checks genuine native RNG-kind/liveness,
compares all four exact native words with the captured public snapshot, and
reauthenticates the same row. The existing constructor copies that native owner;
it does not reseed or convert it. The result RNG survives output/generator release.
Private, forged, cross-native, wrong-kind, mutated and dead sources reject.

## Evidence and next gate

Source checks and host controls execute the actual coordinator, typed wrapper,
public publication/release and RNG selector/constructor bodies with explicitly
stubbed native providers. They are control-flow evidence only. Native owner and
typed fixture pass strict C11 syntax checks. Supported Eshkol AOT compilation,
current native runtime, sanitizers, installed coexistence and numerical behavior
are **NOT RUN** for this port. The exact installed export/string/rename/defined
manifests remain unchanged and must be checked by the supported build; no relaxed
allowlist is authorized. The staged installed fixture covers all five accessors,
three release routes, EOS/non-EOS and detached RNG continuation. The single next
executable gate is independent review of the frozen source/tests, followed by the
orchestrator-scheduled supported installed build under the shared heavy lock.
