#ifndef ESHKOL_TRANSFORMER_DIST5_COLLECTIVES_ABI_H
#define ESHKOL_TRANSFORMER_DIST5_COLLECTIVES_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_DIST5_ABI_MAJOR 1u
#define ET_DIST5_ABI_MINOR 0u
#define ET_DIST5_MAX_WORLD_SIZE 64u
#define ET_DIST5_MAX_BUFFER_BYTES (64u * 1024u * 1024u)

typedef struct et_dist5_comm et_dist5_comm_t;
typedef int32_t et_dist5_status_t;
typedef uint32_t et_dist5_dtype_t;
typedef uint32_t et_dist5_op_t;

enum {
  ET_DIST5_OK = 0,
  ET_DIST5_ERR_INVALID_ARG,
  ET_DIST5_ERR_MISMATCH,
  ET_DIST5_ERR_PEER_DISCONNECTED,
  ET_DIST5_ERR_TIMEOUT,
  ET_DIST5_ERR_NONFINITE,
  ET_DIST5_ERR_IO,
  ET_DIST5_ERR_ALLOC
};

enum { ET_DIST5_F32 = 1, ET_DIST5_I64 };
enum { ET_DIST5_SUM = 1, ET_DIST5_MEAN, ET_DIST5_MIN, ET_DIST5_MAX };

/*
 * Rank 0 supplies connected AF_UNIX SOCK_STREAM endpoints in peer rank order
 * (1 through world_size - 1). Each other rank supplies its endpoint to rank 0.
 * World size 1 supplies zero descriptors. Descriptors are duplicated
 * on success. Close the originals, including unused inherited endpoints, before
 * the first collective. ABI major must match; ABI minor must be supported.
 */
et_dist5_status_t et_dist5_comm_create(uint32_t abi_major, uint32_t abi_minor,
                                       uint32_t rank, uint32_t world_size,
                                       const int *fds, size_t fd_count,
                                       et_dist5_comm_t **out);
void et_dist5_comm_destroy(et_dist5_comm_t *comm);

/*
 * Use one caller per communicator. All ranks issue matching calls in the same
 * order. A tag is an application-supplied identifier and may be reused.
 * Timeouts are positive milliseconds. Counts are positive element counts; each
 * full operand/result (including gathered or scattered operands) is
 * byte-capped. Exact in-place allreduce and gradient sync are supported.
 * Separate input and output spans for allgather and reduce_scatter must be
 * disjoint.
 *
 * Each call writes its local output only on success. A transport error or
 * mismatch makes the communicator unusable; destroy and re-bootstrap it.
 */
et_dist5_status_t et_dist5_barrier(et_dist5_comm_t *comm, uint64_t tag,
                                   int timeout_ms);
et_dist5_status_t et_dist5_broadcast(et_dist5_comm_t *comm, void *buf,
                                     size_t count, et_dist5_dtype_t dtype,
                                     uint32_t root_rank, uint64_t tag,
                                     int timeout_ms);
et_dist5_status_t et_dist5_allreduce(et_dist5_comm_t *comm, const void *sendbuf,
                                     void *recvbuf, size_t count,
                                     et_dist5_dtype_t dtype, et_dist5_op_t op,
                                     uint64_t tag, int timeout_ms);
et_dist5_status_t et_dist5_allgather(et_dist5_comm_t *comm, const void *sendbuf,
                                     size_t send_count, void *recvbuf,
                                     et_dist5_dtype_t dtype, uint64_t tag,
                                     int timeout_ms);
et_dist5_status_t et_dist5_reduce_scatter(et_dist5_comm_t *comm,
                                          const void *sendbuf, void *recvbuf,
                                          size_t recv_count,
                                          et_dist5_dtype_t dtype,
                                          et_dist5_op_t op, uint64_t tag,
                                          int timeout_ms);

/* Coordinated finite preflight and F32 MEAN. Update weights only on success. */
et_dist5_status_t et_dist5_sync_gradients_f32(et_dist5_comm_t *comm,
                                              const float *local, float *mean,
                                              size_t count, uint64_t tag,
                                              int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif
