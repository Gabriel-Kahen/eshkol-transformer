#ifndef ESHKOL_TRANSFORMER_AC4_CHECKPOINTING_ABI_H
#define ESHKOL_TRANSFORMER_AC4_CHECKPOINTING_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_AC4_ABI_MAJOR 1u
#define ET_AC4_ABI_MINOR 0u
#define ET_AC4_MAX_LAYERS 16u

typedef uint32_t et_ac4_status;
enum {
  ET_AC4_OK = 0,
  ET_AC4_INVALID_ARGUMENT = 1,
  ET_AC4_SHAPE_MISMATCH = 2,
  ET_AC4_UNSUPPORTED = 3,
  ET_AC4_ALIASING = 4,
  ET_AC4_ACTIVATION_LIMIT = 5,
  ET_AC4_WORKSPACE_TOO_SMALL = 6,
  ET_AC4_NUMERIC_ERROR = 7,
  ET_AC4_PROVIDER_ERROR = 8
};

typedef uint32_t et_ac4_policy;
enum {
  ET_AC4_POLICY_NONE = 0,          /* Keep every layer's tape and input. */
  ET_AC4_POLICY_EVERY_LAYER = 1,   /* Keep every layer input. */
  ET_AC4_POLICY_SQRT_SEGMENTS = 2, /* Keep boundaries and cache one segment. */
  ET_AC4_POLICY_EVERY_K = 3        /* Keep boundaries; replay prefixes. */
};

typedef uint32_t et_ac4_mode;
enum { ET_AC4_FORWARD_ONLY = 0, ET_AC4_FORWARD_BACKWARD = 1 };

typedef struct et_ac4_shape_v1 {
  uint32_t layers;
  uint32_t batch;
  uint32_t tokens;
  uint32_t query_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
} et_ac4_shape_v1;

typedef struct et_ac4_config_v1 {
  et_ac4_shape_v1 shape;
  et_ac4_mode mode;
  et_ac4_policy policy;
  uint32_t checkpoint_every; /* 1..layers for EVERY_K; zero otherwise. */
  float epsilon;             /* Finite and strictly positive. */
  float dropout_rate;        /* Finite in [0, 1), applied to both branches. */
  uint64_t seed;
  uint64_t step;
  size_t activation_limit_bytes; /* Zero disables the cap. */
} et_ac4_config_v1;

typedef struct et_ac4_const_span_v1 {
  const float *data;
  size_t count;
} et_ac4_const_span_v1;

typedef struct et_ac4_span_v1 {
  float *data;
  size_t count;
} et_ac4_span_v1;

typedef struct et_ac4_io_v1 {
  et_ac4_const_span_v1 input;
  et_ac4_const_span_v1 target;
  et_ac4_const_span_v1 parameters;
  et_ac4_span_v1 output;
  et_ac4_span_v1 d_input;      /* {NULL, 0} for FORWARD_ONLY. */
  et_ac4_span_v1 d_parameters; /* {NULL, 0} for FORWARD_ONLY. */
  float *workspace;
  size_t workspace_bytes;
} et_ac4_io_v1;

typedef struct et_ac4_memory_v1 {
  size_t activation_floats; /* Floats per residual stream. */
  size_t parameter_floats_per_layer;
  size_t parameter_floats;      /* Floats in the complete stack. */
  size_t activation_peak_bytes; /* Predicted by plan; observed by run. */
  size_t gradient_staging_bytes;
  size_t workspace_bytes;
  uint32_t retained_checkpoints;  /* Excludes input owned by the caller. */
  uint32_t retained_tapes;        /* During the original forward pass. */
  uint32_t rematerialized_layers; /* Replay forwards, counting repeats. */
} et_ac4_memory_v1;

typedef struct et_ac4_result_v1 {
  double loss; /* Half mean squared error against target. */
  et_ac4_memory_v1 memory;
} et_ac4_result_v1;

/*
 * CPU binary32 only. Input, target, output and d_input are dense [N,T,D],
 * D = query_heads * head_dim. Verified (N,T,Hq,Hkv,Dh) rows are (1,2,2,1,2),
 * (2,3,4,1,2), (2,3,4,2,2), and (2,3,2,2,2).
 *
 * Each layer packs gamma1[D], Wq[D,D], Wk[V,D], Wv[V,D], Wo[D,D], gamma2[D],
 * Wgate[D,D], Wup[D,D], Wdown[D,D] in this order, V = kv_heads * head_dim.
 * Matrices are output-major and have no biases. Parameter gradients use the
 * same packing. Attention is causal with positions 0..T-1.
 *
 * Plan validates config and returns the workspace requirement even when the
 * configured activation cap is lower. Forward-only uses a rolling stream
 * for all policies. All structs and payload spans are naturally aligned and
 * pairwise disjoint, including read-only spans and the full workspace span.
 * Counts must exactly match the plan. Workspace storage may be reused across
 * calls and its contents are scratch, including on failure. Plan and run
 * leave their other outputs byte-for-byte unchanged on any failure.
 */
et_ac4_status et_ac4_plan_v1(const et_ac4_config_v1 *config,
                             et_ac4_memory_v1 *memory);
et_ac4_status et_ac4_run_v1(const et_ac4_config_v1 *config,
                            const et_ac4_io_v1 *io, et_ac4_result_v1 *result);

#ifdef __cplusplus
}
#endif

#endif
