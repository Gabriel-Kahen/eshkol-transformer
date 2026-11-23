#ifndef ESHKOL_TRANSFORMER_DIST5_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_DIST5_ESHKOL_BRIDGE_H

#include "eshkol_transformer/dist5_collectives_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque per-rank handle owning the DIST5 Unix-domain-socket communicator and
 * the discovered MOD5 K1 runtime. Keep alive across all calls, then close once
 * via et_dist5_aot_close(). */
typedef struct et_dist5_aot_handle et_dist5_aot_handle_t;

/* Bootstrap a rank process into a DIST5 collective group via an AF_UNIX
 * rendezvous socket in sock_dir (required when world_size > 1) and discover
 * the MOD5 K1 runtime. Returns ET_DIST5_OK on success or an explicit
 * ET_DIST5_ERR_* status on failure, with *out set to NULL and zero leaked
 * descriptors or allocations. */
int32_t et_dist5_aot_open(const char *sock_dir, int64_t rank,
                          int64_t world_size, int64_t timeout_ms,
                          et_dist5_aot_handle_t **out);
void et_dist5_aot_close(et_dist5_aot_handle_t *handle);

/* Handle metadata. Return -1 when handle is NULL. */
int64_t et_dist5_aot_rank(const et_dist5_aot_handle_t *handle);
int64_t et_dist5_aot_world_size(const et_dist5_aot_handle_t *handle);

/* Synchronous collective wrappers. Buffer arguments accept void * so
 * misaligned-pointer negative tests never invoke undefined pointer casts in
 * C11 or C++17. Invalid arguments, mismatches, timeouts, peer exits, and
 * non-finite operands fail explicitly with ET_DIST5_ERR_* and zero silent
 * fallbacks. */
int32_t et_dist5_aot_barrier(et_dist5_aot_handle_t *handle, int64_t tag,
                             int64_t timeout_ms);
int32_t et_dist5_aot_broadcast_f32(et_dist5_aot_handle_t *handle, void *buf,
                                   int64_t count, int64_t root_rank,
                                   int64_t tag, int64_t timeout_ms);
int32_t et_dist5_aot_allreduce_f32(et_dist5_aot_handle_t *handle,
                                   const void *sendbuf, void *recvbuf,
                                   int64_t count, int32_t op, int64_t tag,
                                   int64_t timeout_ms);
int32_t et_dist5_aot_allgather_f32(et_dist5_aot_handle_t *handle,
                                   const void *sendbuf, int64_t send_count,
                                   void *recvbuf, int64_t tag,
                                   int64_t timeout_ms);
int32_t et_dist5_aot_reduce_scatter_f32(et_dist5_aot_handle_t *handle,
                                        const void *sendbuf, void *recvbuf,
                                        int64_t recv_count, int32_t op,
                                        int64_t tag, int64_t timeout_ms);
int32_t et_dist5_aot_sync_gradients_f32(et_dist5_aot_handle_t *handle,
                                        const void *local, void *mean,
                                        int64_t count, int64_t tag,
                                        int64_t timeout_ms);

/* Binary32 buffer, exact bit, deterministic f32 accumulation, and SHA-256
 * digest helpers with explicit null, alignment, and bounds checks. */
void *et_dist5_aot_alloc_f32(int64_t count);
void et_dist5_aot_free_f32(void *buf);
int32_t et_dist5_aot_store_f32(void *buf, int64_t capacity, int64_t index,
                               double value);
double et_dist5_aot_load_f32(const void *buf, int64_t capacity, int64_t index);
int32_t et_dist5_aot_store_f32_bits(void *buf, int64_t capacity, int64_t index,
                                    int64_t bits);
int64_t et_dist5_aot_load_f32_bits(const void *buf, int64_t capacity,
                                   int64_t index);
int32_t et_dist5_aot_equal_bits_f32(const void *a, const void *b,
                                    int64_t count);
int32_t et_dist5_aot_sha256_hex_f32(const void *buf, int64_t count,
                                    char *hex_out, int64_t hex_capacity);
int32_t et_dist5_aot_read_byte(const void *buf, int64_t capacity,
                               int64_t index);
int32_t et_dist5_aot_f32_add_inplace(void *dst, const void *src, int64_t count);
int32_t et_dist5_aot_f32_div_scalar_inplace(void *buf, int64_t count,
                                            double divisor);

/* MOD5 K1 kernel dispatch over the handle's discovered K1 runtime.
 * Returns 0 on success or an explicit K1 error category.
 * RMSNorm: x, dy, y, dx [1,8,16]; gamma, dgamma [16]; epsilon scalar.
 * SwiGLU: all views [1,8,32].
 * Attention: q, dy, y, dq [1,4,8,4]; k, v, dk, dv [1,Hkv,8,4];
 * qpos, kpos [1,8]; mask [1,8,8]; slopes [4]. Hkv is 1 or 2. */
int32_t et_dist5_aot_mod5_rms_forward(const et_dist5_aot_handle_t *handle,
                                      void *x, void *gamma, void *epsilon,
                                      void *y);
int32_t et_dist5_aot_mod5_rms_backward(const et_dist5_aot_handle_t *handle,
                                       void *x, void *gamma, void *epsilon,
                                       void *dy, void *dx, void *dgamma);
int32_t et_dist5_aot_mod5_swiglu_forward(const et_dist5_aot_handle_t *handle,
                                         void *gate, void *up, void *y);
int32_t et_dist5_aot_mod5_swiglu_backward(const et_dist5_aot_handle_t *handle,
                                          void *gate, void *up, void *dy,
                                          void *dgate, void *dup);
int32_t et_dist5_aot_mod5_attention_forward(const et_dist5_aot_handle_t *handle,
                                            int64_t kv_heads, void *q, void *k,
                                            void *v, void *qpos, void *kpos,
                                            void *mask, void *slopes,
                                            int64_t window, void *y);
int32_t et_dist5_aot_mod5_attention_backward(
    const et_dist5_aot_handle_t *handle, int64_t kv_heads, void *q, void *k,
    void *v, void *qpos, void *kpos, void *mask, void *slopes, int64_t window,
    void *dy, void *dq, void *dk, void *dv);

#ifdef __cplusplus
}
#endif

#endif
