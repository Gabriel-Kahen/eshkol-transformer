# Greedy speculative decoding with live G3-C4 model forwards

`make test-spec-live-model` runs an opt-in native C11 witness that calls
[`et_spec_paged_round_greedy_v1`](SPEC_PAGED_ROUND.md) with real draft and
target outputs from the private G3-C4 CPU-f32 model. Each case starts with
one prompt token and has four total positions, a 256-token vocabulary, and
a proposal horizon `K` of one or two. Each model has one layer, two K/V heads
of width two, a separate paged pool, and an uninterrupted A2 control. Each
pool has four physical blocks with two slots each and supports two live
sequences.

The draft greedily proposes tokens from its own live logits. Both private
candidate producers start from their model's paged parent and forward those
tokens. The request carries each producer's contiguous binary32 K and V
spans of shape `[K,1,2,2]` and the target logits of shape `[K+1,256]`. The
target row at index zero follows the prompt. Subsequent rows follow each
candidate in order. G3-C4 selects each proposal from the draft row that
produced it. Only the initial prompt uses a one-hot token carrier.

The coordinator forks both parents, stages the candidates, verifies them,
and truncates its children. The callback reads the retained child prefixes
by head and position and checks that only the live prefix blocks remain.
It stages the copied K/V as `[1,2,length,2]` in the private A2 caches of two
scratch generators created before each attempt. It forwards the correction
or bonus using the preceding target logit row, returns one `[1,2,2]` K/V
token per model, and saves the new logits in caller-owned scratch. The
witness checks that the G3-C4 transport registry head stays unchanged across
the coordinator call, then closes both scratch generators. On success the
coordinator returns both child sequence IDs with the new K/V appended.
Where capacity permits, the witness uses the returned children and the new
target logits for a greedy continuation in both models.

The independent controls drive each model using the target's greedy token
at every position. Both models' K/V and logits match the controls bit for bit
at the accepted prefix, correction or bonus, and continuation. The target's
four control rows also match an independent G3-C4 full-prefix forward. All
greedy G3-C4 frames preserve their RNG words, and SPEC5 reports zero draws.
The prompt anchors retain their K/V through copy-on-write.

In the late rejection case, the callback finishes both real forwards and
returns an injected error. The witness checks the result bytes, both parent
block tables and K/V, and all physical block refcounts before retrying the
same request successfully. Cumulative copy-on-write telemetry counts both
attempts. All cases verify pool accounting through final cleanup.

The target uses seed 1729 and prompt token 71. Draft seed 38 agrees on the
first proposal and diverges on the second. Full two-proposal acceptance uses
a separately initialized draft with the target seed.

| Case | Proposed | Accepted | Emitted | Continuation forwards per model |
| --- | ---: | ---: | ---: | ---: |
| Late rejection | 2 | 1 | 2 | 1 |
| Bonus and continuation | 1 | 1 | 2 | 1 |
| Full acceptance | 2 | 2 | 3 | 0 |

The gate compares stdout across two optimized C11 runs and two runs with
ASan, UBSan, and leak detection. The pinned runtime is Ubuntu 22.04.5 with
Clang 21.1.8 in `eshkol-r0-debug:20260928`
(`sha256:68e2e292a255c9b42206e2298d12770aea7594ff9b82219460cb27ee28dbdc99`).

The witness uses the in-process C round ABI and copies K/V between PKV and
A2 for private attention. Public generation, direct paged attention,
stochastic verification, EOS, production memory behavior, accelerator use,
and throughput remain outside this gate.
