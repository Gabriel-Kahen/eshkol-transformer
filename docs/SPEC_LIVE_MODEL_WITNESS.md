# Greedy speculative decoding with live G3-C4 model forwards

`make test-spec-live-model` runs an opt-in native C11 witness on the private
G3-C4 CPU-f32 forward stack. Each scenario starts with one prompt token,
uses a 256-token vocabulary and a proposal horizon of one or two, and has
four total positions. The model has one layer, two K/V heads, and head width
two. Each model owner has its own paged pool and uninterrupted A2 control.

The draft greedily chooses each proposal from its own live logits. Both
models forward the proposal, and SPEC5 verifies the resulting target rows.
The two paged sequences truncate to the prompt plus the verified accepted
prefix. Both models then forward the verifier's correction or bonus and
append their respective new K/V. If capacity remains, both forward one more
greedy continuation token. The G3-C4 frame accepts each selected token
through the preceding live row whose argmax produced it. Only the initial
prompt uses a one-hot token carrier. SPEC5 sees the real target logit rows.

Before each paged forward, the witness reads the retained PKV prefix by head
and position. It stages those exact binary32 values in a fresh private A2
cache as `[1,2,length,2]` and runs the accepted attention over that cache.
After the G3-C4 frame publishes, an A2 read borrow supplies the new
`[1,2,2]` per-token K/V that the witness appends to PKV. The test uses two
slots per physical block to release the rejected tail block at position 2.
It verifies copy-on-write isolates the one-token prompt anchor and checks
pool accounting through final cleanup.

The scenarios exercise rejection after one accepted proposal, a bonus plus
continuation after one accepted proposal, and full acceptance of two
proposals. An independent target-only greedy loop checks every emitted and
continued token. At rollback, correction or bonus, and continuation, both
models' paged-prefix K/V and live logits match their uninterrupted A2
controls bit for bit. The target control's four rows also match the
independent G3-C4 full-prefix forward. All greedy G3-C4 frames preserve their
RNG state, and every SPEC5 call consumes zero draws.

The focused gate passed optimized C11 and ASan/UBSan/LSan with identical
stdout in `eshkol-r0-debug:20260928` (Ubuntu 22.04, Clang 21.1.8).
The target uses seed 1729 and prompt token 71. Draft seed 38 agrees on the
first proposal and diverges on the second. Full two-proposal acceptance
uses a separately initialized draft with the target's seed.

| Case | Proposed | Accepted | Emitted | Continuation forwards per model |
| --- | ---: | ---: | ---: | ---: |
| Late rejection | 2 | 1 | 2 | 1 |
| Bonus and continuation | 1 | 1 | 2 | 1 |
| Full acceptance | 2 | 2 | 3 | 0 |

The gate exercises the private CPU-f32 model and its A2 attention. PKV
owns rollback and storage, and the witness copies K/V between the two
managers on each forward. Public generation, direct paged attention,
stochastic verification, EOS, production memory behavior, accelerator use,
and throughput remain outside this gate.
