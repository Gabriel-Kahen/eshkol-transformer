# Opt-in native constrained beam step

Include `eshkol_transformer/constrained_beam_abi.h` and link
`native/constrained_beam.c` with the native [BEAM5](BEAM_SEARCH_DECODING.md)
and [CD5](CONSTRAINED_DECODING_FSA.md) cores. Initialize the caller-owned beam
with `et_beam_init_v1` and build the CD5 vocabulary index using the candidate
producer's token bytes and EOS ID. Keep that index alive and use it throughout
the search. An application can build against the three native sources with the
same floating-point flags as the gate.

```sh
clang-21 -std=c11 -O2 -ffp-contract=off -fexcess-precision=standard \
  -fno-fast-math -I include application.c native/constrained_beam.c \
  native/beam_search.c native/constrained_decode.c -lm -o application
```

Call `et_constrained_beam_step_v1` with dense row-major CPU F32 logits. Supply
one row per active beam, in the current order returned by BEAM5, and one column
per vocabulary token ID. The vocabulary size and EOS ID must match across the
beam and index. Re-read the active hypotheses after each successful step to
prepare the next rows. Use `et_beam_finalize_v1` after BEAM5 reports completion.

The DFA begins at its start state on each hypothesis's generated suffix. Prompt
tokens remain in the BEAM5 repetition and n-gram histories. Each active suffix
must remain a legal prefix, and any retained completion must end in admissible
EOS. CD5 excludes forbidden candidates and permits EOS only from accepting
states. BEAM5 processes the remaining logits and computes its existing
binary64 per-row log-softmax before adding each candidate's log probability to
its parent. Beam order, tie-breaking, EOS ranking, Wu length penalty and
stopping follow BEAM5. A budget-limited unfinished hypothesis denotes a
reachable prefix and may require more tokens to satisfy the grammar.

The step is atomic across the full caller-owned beam object, including its
scratch bytes. An invalid or completed beam or an inadmissible history returns
`ET_BEAM_ERR_STATE`. Invalid geometry returns `ET_BEAM_ERR_INVALID_ARG` and
beam/logit overlap returns `ET_BEAM_ERR_ALIASING`. Nonfinite raw values are
rejected even on forbidden columns. If any row has no candidate after CD5 or
BEAM5 masks, the step returns `ET_BEAM_ERR_NO_VALID_TOKENS`. Input storage and
the index are borrowed for the call.

The native leaf follows BEAM5's limits of 8 beams, 4096 vocabulary tokens, and
256 total tokens per hypothesis. It stages a fixed-size binary64 matrix, one
F32 row, and one BEAM5 state on the calling thread's stack. The optimized
pinned x86-64 build requires 378,808 bytes (about 370 KiB) of automatic storage for the wrapper
frame, plus the called functions' frames. The leaf does not allocate on the heap. It consumes caller-provided
logits and has no model, tokenizer, G3, CLI or AOT dispatch integration.

Run `CC=clang-21 CXX=clang++-21 make test-constrained-beam` on the repository's
Ubuntu 22.04 / Clang 21.1.8 image
`sha256:68e2e292a255c9b42206e2298d12770aea7594ff9b82219460cb27ee28dbdc99`.
The gate checks the C++ header and native object symbols, then runs optimized
and sanitizer tests for cross-row branching, normalized scores, processor
ordering, completion, failure atomicity, and the maximum supported geometry.
