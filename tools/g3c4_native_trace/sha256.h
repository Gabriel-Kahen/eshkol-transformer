#ifndef G3C4_SHA256_H
#define G3C4_SHA256_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    uint8_t block[64];
    size_t used;
} sha256_state;

void sha256_init(sha256_state *s);
void sha256_update(sha256_state *s, const void *data, size_t length);
void sha256_final(sha256_state *s, uint8_t out[32]);
#endif
