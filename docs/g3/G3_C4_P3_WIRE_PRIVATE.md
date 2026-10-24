# Private C4 P3 checked data record

`ET_CPU4_C4_P3_WIRE_PRIVATE` exposes two source-private C entries in
`cpu4_c4_capture_private.h`:

```c
int64_t et_cpu4_private_c4_p3_wire_encode_v1(
    const et_cpu4_c4_capture_record_v1 *record, const void *values,
    uint64_t values_bytes, const uint8_t c2_sha[32], const uint8_t t1_fp[96],
    void *wire, uint64_t wire_bytes);
int64_t et_cpu4_private_c4_p3_wire_decode_v1(
    const void *wire, uint64_t wire_bytes, const void *values,
    uint64_t values_bytes, const uint8_t expected_c2_sha[32],
    const uint8_t expected_t1_fp[96],
    et_cpu4_c4_capture_record_v1 *out_record);
```

Both calls require exact `values_bytes=4768` and `wire_bytes=400`, valid
nonoverlapping declared spans, and preserve their destination bytes on rejection.
The wire is `C4GRCV1\0` at 0, little-endian length 400 at 8, profile 1 at 12,
canonical little-endian capture fields 192 plus capture digest 32 at 16,
whole B250 C2 SHA-256 at 240, exact raw T1 fingerprint 96 at 272, and SHA-256
over bytes 0–367 at 368. The capture digest covers the canonical 192 fields
and the separately supplied 4,768 parameter bytes. It uses the existing
`cpu4-c4-capture-v1\0` domain. The wire carries neither parameter bytes nor a
persisted registry identity.

The only accepted record is P3 with IDs `[3,197,3,0]`, origins `[1,1,2,0]`,
history counts `(3,2,1,0)`, cache capacity 4, epoch 2, EOS mask 0, greedy
policy with two-token budget and EOS 41 or 255, and four-word RNG `(1,41,0,0)`.
The zero EOS mask describes the generated first token at absolute index 2;
EOS41 can become terminal only at the next generated token, absolute index 3
(mask bit 8). The data codec checks field semantics and hashes. It does not
issue a live model, T1, history or RNG owner, rebuild K/V/logits, or authorize
the P3→G2 numerical adapter. Eshkol must reauthenticate those current-process
owners after exact B250 FILE factory restore before using a decoded record.

The focused C unit compiles the actual codec body with an independent OpenSSL
SHA shim and fixed synthetic capture-validator stub; its Python oracle derives
the exact 400 bytes separately. This is codec behavior evidence only. A full
CPU4 aggregate/native and fresh-process continuation gate remains required.
