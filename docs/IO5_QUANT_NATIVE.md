# IO5 native codec and independent Q8_0 decoder witness

The C11 codec ABI, implementation, and native tests are carried from the
reviewed standalone candidate `29ae61cd`. The wire converter now uses the
merged M3 and decoder mappings or bare C1 state; it rejects the unmerged MOD5
profile. See [the wire contract](IO5_QUANTIZED_INTERCHANGE.md).

`scripts/test-io5-quant.sh` checks deterministic `.a` and `.so` rebuilds,
C++17 linkage, exhaustive F16/BF16 bit patterns, Q8_0 numeric and malformed
inputs, zero heap allocations, and ASan/UBSan/LSan. The codec is shape agnostic;
the wire converter checks GGML row alignment and format metadata.

`scripts/test-io5-q8-ggml-decoder.sh` uses an explicitly selected GGML prefix
and SHA-256-checked source archive at commit
`417a024b4fb6bc3116d41af7c067166a039bdf12`. It checks the installed
headers against that archive and records the library hash. The witness creates
a test-only GGUF tensor with GGML shape `(64, 3)` and exact raw Q8_0 bytes.
GGML's reader must recover the type, shape, and all 204 raw bytes; its native
Q8_0 decoder then matches all 192 Eshkol decoded F32 bit patterns. Six finite
scales cover signed zero, minimum subnormal, minimum normal, one, and maximum
finite F16, with signed byte values including -128 and 127. This tests decoder
compatibility; encoder bit identity and live model execution remain unmeasured.
