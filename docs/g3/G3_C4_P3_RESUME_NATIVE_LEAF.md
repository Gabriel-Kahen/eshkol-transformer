# Private C4 P3 to G2 numerical resume prerequisite

This source-private leaf adds `et_cpu4_private_c4_p3_resume_prefix_v1` under
`ET_CPU4_C4_P3_RESUME_PRIVATE`. It requires the existing C4 P2/G2 second-frame,
P3 prefill, and committed-history features. The entry accepts a registered
active kind-2/budget-2 generator, its clean pending two-ID output, three raw
V256 IDs and a caller-owned 256-logit destination.

The trusted caller must authenticate the persisted C2 image, exact restored
model bits, live tokenizer, checked four-word RNG, and the saved prompt/G1
history before calling this native entry. The native entry does not parse a
file or prove those external authorities. It validates native ownership and
output state, rejects a first generated ID equal to EOS, then performs the
existing no-draw three-token prefill. It records the first two IDs as prompt
and the third as generated, leaving history epoch two, cache length three,
retained next logits and the existing second-frame path ready. The ordinary
`et_g3c4_private_prefill3_v1` continues to reject budget two.

`tests/g3c4/test_cpu4_p3_resume_native.c` compares exact P3 logits, full
cache K/V/mask/length, history, generated IDs and successor RNG against an
uninterrupted same-owner two-token control. It injects a provider failure
inside P3 attention, verifies unchanged cache/history/RNG/output and no
sampling draw, then retries successfully. The test constructs its RNG in
controlled native state; it is not a checked FILE import or a trained-B250
fresh-process test. Those remain separate compiled gates.
