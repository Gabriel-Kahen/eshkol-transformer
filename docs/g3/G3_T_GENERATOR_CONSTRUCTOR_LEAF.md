# G3-T seeded generator constructor leaf

This source-private precursor follows the accepted [G3-T contract](G3_T_PRIVATE_CONTRACT.md),
the [M3T admission leaf](G3_T_MODEL_ADMISSION_LEAF.md), and the
[authentic native context leaf](G3_T_NATIVE_CONTEXT_LEAF.md). Load
`g3t_generator_constructor_extension.esk` after `m3_package_root.esk`,
`m3_call_adapters.esk`, and `g3t_model_admission_extension.esk` in the same
trusted source aggregate. It installs no public generation API or package.

`g3t-generator-create(model, tokenizer, config)` enters the shared `m3-call`
guard once. It accepts exactly eight flat option pairs: `:profile` set to
`diagnostic-c2`, `:sampling` set to `greedy` or `categorical`, `:temperature-bits`,
`:top-k`, `:top-p-bits`, `:max-new-tokens`, `:eos`, and exactly one of `:seed`
or `:rng`. The later private RNG-input leaf authenticates a live same-aggregate
kind-8 owner and copies its words into the generator. Policy values are checked as exact i64
and binary32 bit patterns, including greedy's exact 1/256/1 settings. The
result is a detached seven-slot C2 policy vector; no source option list or
raw tokenizer pointer enters native code.

The constructor authenticates the exact live M3T model entry using
`g3t-model-entry-live` and the exact same-aggregate T1 registry shell. It
requires the baseline raw tokenizer with V256, empty specials/prefix/suffix,
and the canonical T1 byte-tokenizer fingerprint. Before native allocation it
roots a pending 12-slot G3-T entry in `g3t-registry` and reads back the
canonical entry. Native `generator_seed` receives only the authenticated
M3T owner pointer and normalized scalars; the RNG route passes the admitted
kind-8 pointer to native `generator_rng`. Failure closes any created native
context and leaves only an inert Eshkol tombstone; success publishes a live
generator with model/tokenizer/policy roots. Native error domain/category/code
are snapped before cleanup and mapped to bounded E1 details under the invoking
private operation. Its four extra helpers and the predecessor model-admission
helper are listed in
`native/g3t_generator_local_symbols.txt` for future localization.

`g3t-generator-close!(generator)` authenticates the exact G3-T shell and its
idle links, asks native close to reject busy or borrowed context, then clears
slots 3..11. Exact dead close is idempotent. Closing does not require the model
to remain in eval mode, so a mode change does not strand an idle generator.

The pinned witness uses a genuine M3T C2 model and T1 byte tokenizer. It
covers greedy and categorical policies, foreign/copied identities, strict T1,
wrong profile/mode/policy,
same-model active frame, native and A2 allocation failure/retry, region-rooted
publication, busy close, tombstones, and post-close reuse in normal/repeat/
sanitizer modes. The source/AOT witness checks native E1 category, operation,
domain, and code mapping alongside rejection and guard cleanup;
it does not claim a public error envelope. The Eshkol registry is not yet
packaged or installed. Frame transcript, prefill/decode/sample, result
publication, and public generation remain separate dependencies.
