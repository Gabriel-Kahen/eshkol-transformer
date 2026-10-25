# Trained-C2 first-token EOS reference fixture

`tests/g3r/trained_c2_eos_reference.esk` is a source-staged private reference
for the exact CLI3-trained context-two C2 image used by the G3-R pair producer.
Its compiled gate must pass the same C2 file path to both programs and compare
the printed complete `c2_sha256` with the pair receipt. It reads that image,
restores it through CLI3/TR3 into a live M3T model, rereads the file to reject
in-place changes, releases the trainer, then enters eval mode.

The fixture runs two fresh G3-T generators over the same restored model,
tokenizer, prompt, categorical policy and seed. The first genuinely samples
with no EOS setting. Its emitted ID becomes the second generator's EOS setting.
The fixture compares raw output bytes, committed history, all four RNG words,
history partitions and the generated-EOS terminal latch, then prints the C2
digest, prompt, ID and successor RNG words. An independently compiled run must
also compare the output with the future fresh-process continuation receipt.

This is a **source-only prerequisite** until a pinned compiled normal/repeat
and sanitizer gate succeeds. It proves at most first-token EOS for one generated
token after P1 on the trained C2. The `diagnostic-c2` constructor caps context
at two and generation budget at one. Repeated-token generation and later-token
EOS require a separately admitted larger profile and cannot be inferred from
this fixture or the C4 P2/G2 test model.
