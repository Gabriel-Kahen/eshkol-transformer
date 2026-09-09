/* Generated from the strict Q0 fixture; do not edit by hand. */
#ifndef ESHKOL_TRANSFORMER_N2_REFERENCE_VECTORS_H
#define ESHKOL_TRANSFORMER_N2_REFERENCE_VECTORS_H

#include <stddef.h>
#include <stdint.h>

#define ET_N2_REFERENCE_FIXTURE_SHA256 \
  "a32e065db6d7654600121696ba680cb15654c6712696e37fdb4a5cfb51abca03"
#define ET_N2_REFERENCE_TENSOR_COUNT UINT32_C(82)
#define ET_N2_REFERENCE_CASE_COUNT UINT32_C(26)

enum et_n2_reference_dtype {
  ET_N2_REFERENCE_F32 = 1,
  ET_N2_REFERENCE_I64 = 2,
};

enum et_n2_reference_role {
  ET_N2_REFERENCE_INPUT = 1,
  ET_N2_REFERENCE_EXPECTED = 2,
  ET_N2_REFERENCE_ANALYTIC_GRADIENT = 3,
};

struct et_n2_reference_tensor {
  const char *name;
  uint32_t dtype;
  uint32_t role;
  uint32_t rank;
  const int64_t *shape;
  uint64_t element_count;
  const void *words;
};

struct et_n2_reference_case {
  const char *name;
  const char *operation;
  const char *kind;
  uint32_t input_count;
  const uint16_t *inputs;
  uint32_t output_count;
  const uint16_t *outputs;
};

static const int64_t et_n2_ref_activation_gelu_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(8),
};
static const uint32_t et_n2_ref_activation_gelu_gradient_dx_words[] = {
  UINT32_C(0x00000000), UINT32_C(0xbb43b7aa), UINT32_C(0x3d2aa150), UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0xbf8acc9a), UINT32_C(0x3fd05d50), UINT32_C(0xc0000000),
};

static const int64_t et_n2_ref_activation_gelu_output_forward_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(8),
};
static const uint32_t et_n2_ref_activation_gelu_output_forward_words[] = {
  UINT32_C(0x80000000), UINT32_C(0xbb84b340), UINT32_C(0xbe227686), UINT32_C(0x80000000), UINT32_C(0x00000000), UINT32_C(0x3eb103af), UINT32_C(0x3ffa2d0c), UINT32_C(0x41a00000),
};

static const int64_t et_n2_ref_activation_input_upstream_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(8),
};
static const uint32_t et_n2_ref_activation_input_upstream_words[] = {
  UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0xbf000000), UINT32_C(0x3f400000), UINT32_C(0x3f800000), UINT32_C(0xbfa00000), UINT32_C(0x3fc00000), UINT32_C(0xc0000000),
};

static const int64_t et_n2_ref_activation_input_x_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(8),
};
static const uint32_t et_n2_ref_activation_input_x_words[] = {
  UINT32_C(0xc1a00000), UINT32_C(0xc0400000), UINT32_C(0xbf800000), UINT32_C(0x80000000), UINT32_C(0x00000000), UINT32_C(0x3f000000), UINT32_C(0x40000000), UINT32_C(0x41a00000),
};

static const int64_t et_n2_ref_activation_relu_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(8),
};
static const uint32_t et_n2_ref_activation_relu_gradient_dx_words[] = {
  UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0xbfa00000), UINT32_C(0x3fc00000), UINT32_C(0xc0000000),
};

static const int64_t et_n2_ref_activation_relu_output_forward_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(8),
};
static const uint32_t et_n2_ref_activation_relu_output_forward_words[] = {
  UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x3f000000), UINT32_C(0x40000000), UINT32_C(0x41a00000),
};

static const int64_t et_n2_ref_dropout_eval_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_eval_gradient_dx_words[] = {
  UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0x3e000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_dropout_eval_input_upstream_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_eval_input_upstream_words[] = {
  UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0x3e000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_dropout_eval_input_x_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_eval_input_x_words[] = {
  UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a),
  UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd),
};

static const int64_t et_n2_ref_dropout_eval_output_forward_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_eval_output_forward_words[] = {
  UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a),
  UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd),
};

static const int64_t et_n2_ref_dropout_train_p0_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_train_p0_gradient_dx_words[] = {
  UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0x3e000000), UINT32_C(0x3e800000),
};

static const uint32_t et_n2_ref_dropout_train_p0_input_probability_words[] = {
  UINT32_C(0x00000000),
};

static const int64_t et_n2_ref_dropout_train_p0_input_state_shape[] = {
  INT64_C(4),
};
static const int64_t et_n2_ref_dropout_train_p0_input_state_words[] = {
  INT64_C(1), INT64_C(2999170649027065890), -INT64_C(8817193942522041720), INT64_C(247824715720788526),
};

static const int64_t et_n2_ref_dropout_train_p0_input_upstream_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_train_p0_input_upstream_words[] = {
  UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0x3e000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_dropout_train_p0_input_x_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_train_p0_input_x_words[] = {
  UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a),
  UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd),
};

static const int64_t et_n2_ref_dropout_train_p0_output_forward_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_train_p0_output_forward_words[] = {
  UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a),
  UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd),
};

static const int64_t et_n2_ref_dropout_train_p0_output_next_state_shape[] = {
  INT64_C(4),
};
static const int64_t et_n2_ref_dropout_train_p0_output_next_state_words[] = {
  INT64_C(1), INT64_C(2999170649027065890), -INT64_C(8817193942522041720), INT64_C(247824715720788526),
};

static const int64_t et_n2_ref_dropout_train_p25_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_train_p25_gradient_dx_words[] = {
  UINT32_C(0x3e2aaaab), UINT32_C(0x3eaaaaab), UINT32_C(0x3f000000), UINT32_C(0x00000000), UINT32_C(0x3e2aaaab), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x3f2aaaab),
  UINT32_C(0x3e2aaaab), UINT32_C(0x3eaaaaab),
};

static const uint32_t et_n2_ref_dropout_train_p25_input_probability_words[] = {
  UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_dropout_train_p25_input_state_shape[] = {
  INT64_C(4),
};
static const int64_t et_n2_ref_dropout_train_p25_input_state_words[] = {
  INT64_C(1), INT64_C(2999170649027065890), -INT64_C(8817193942522041720), INT64_C(247824715720788526),
};

static const int64_t et_n2_ref_dropout_train_p25_input_upstream_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_train_p25_input_upstream_words[] = {
  UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0x3e000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_dropout_train_p25_input_x_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_train_p25_input_x_words[] = {
  UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a),
  UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd),
};

static const int64_t et_n2_ref_dropout_train_p25_output_forward_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(5),
};
static const uint32_t et_n2_ref_dropout_train_p25_output_forward_words[] = {
  UINT32_C(0xbf4cccce), UINT32_C(0xbf088889), UINT32_C(0xbe888889), UINT32_C(0x00000000), UINT32_C(0x3e888889), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0xbf4cccce),
  UINT32_C(0xbf088889), UINT32_C(0xbe888889),
};

static const int64_t et_n2_ref_dropout_train_p25_output_next_state_shape[] = {
  INT64_C(4),
};
static const int64_t et_n2_ref_dropout_train_p25_output_next_state_words[] = {
  INT64_C(1), INT64_C(2999170649027065890), -INT64_C(8817193942522041717), INT64_C(247824715720788526),
};

static const int64_t et_n2_ref_embedding_repeated_gradient_dweight_shape[] = {
  INT64_C(5), INT64_C(6),
};
static const uint32_t et_n2_ref_embedding_repeated_gradient_dweight_words[] = {
  UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4cccce), UINT32_C(0x3e4cccce),
  UINT32_C(0xbf4ccccd), UINT32_C(0xbecccccd), UINT32_C(0x00000000), UINT32_C(0x3ecccccd), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000),
  UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x3eccccce), UINT32_C(0xbf19999a), UINT32_C(0xbe4ccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3f19999a), UINT32_C(0xbeccccce),
  UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd),
};

static const int64_t et_n2_ref_embedding_repeated_input_ids_shape[] = {
  INT64_C(2), INT64_C(3),
};
static const int64_t et_n2_ref_embedding_repeated_input_ids_words[] = {
  INT64_C(1), INT64_C(3), INT64_C(1), INT64_C(4),
  INT64_C(0), INT64_C(3),
};

static const int64_t et_n2_ref_embedding_repeated_input_upstream_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(6),
};
static const uint32_t et_n2_ref_embedding_repeated_input_upstream_words[] = {
  UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a),
  UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbecccccd),
  UINT32_C(0xbe4ccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd),
  UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbecccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x00000000),
  UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a),
};

static const int64_t et_n2_ref_embedding_repeated_input_weight_shape[] = {
  INT64_C(5), INT64_C(6),
};
static const uint32_t et_n2_ref_embedding_repeated_input_weight_words[] = {
  UINT32_C(0xbf200000), UINT32_C(0xbf000000), UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000),
  UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0x3f200000), UINT32_C(0xbf200000), UINT32_C(0xbf000000), UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000),
  UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0x3f200000), UINT32_C(0xbf200000), UINT32_C(0xbf000000),
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_embedding_repeated_output_forward_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(6),
};
static const uint32_t et_n2_ref_embedding_repeated_output_forward_words[] = {
  UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000), UINT32_C(0x3f200000), UINT32_C(0xbf200000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000),
  UINT32_C(0x3f000000), UINT32_C(0x3f200000), UINT32_C(0xbf200000), UINT32_C(0xbf000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0x3f200000), UINT32_C(0xbf200000), UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000),
  UINT32_C(0xbf200000), UINT32_C(0xbf000000), UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000),
  UINT32_C(0x3f000000), UINT32_C(0x3f200000), UINT32_C(0xbf200000), UINT32_C(0xbf000000),
};

static const int64_t et_n2_ref_layer_norm_affine_gradient_dbeta_shape[] = {
  INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_affine_gradient_dbeta_words[] = {
  UINT32_C(0xbeb33333), UINT32_C(0x3eb33332), UINT32_C(0x00000000), UINT32_C(0x3f333333),
};

static const int64_t et_n2_ref_layer_norm_affine_gradient_dgamma_shape[] = {
  INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_affine_gradient_dgamma_words[] = {
  UINT32_C(0x3f8090a0), UINT32_C(0xbecb33c5), UINT32_C(0x3e09c152), UINT32_C(0x3f430792),
};

static const int64_t et_n2_ref_layer_norm_affine_gradient_dx_shape[] = {
  INT64_C(2), INT64_C(2), INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_affine_gradient_dx_words[] = {
  UINT32_C(0xbe6a3ea6), UINT32_C(0x3e0f15d2), UINT32_C(0xbdfb8d3e), UINT32_C(0x3e58ef72), UINT32_C(0x4314b9d3), UINT32_C(0xc30dcef2), UINT32_C(0x41260504), UINT32_C(0xc18a5980),
  UINT32_C(0x3bfb9580), UINT32_C(0x3d2b567c), UINT32_C(0xbdfca183), UINT32_C(0x3d973cec), UINT32_C(0x3c6e6440), UINT32_C(0xbb2e8e00), UINT32_C(0xbda6a513), UINT32_C(0x3d8e4cfb),
};

static const int64_t et_n2_ref_layer_norm_affine_input_beta_shape[] = {
  INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_affine_input_beta_words[] = {
  UINT32_C(0xbe4ccccd), UINT32_C(0x3dcccccd), UINT32_C(0x3e99999a), UINT32_C(0xbecccccd),
};

static const uint32_t et_n2_ref_layer_norm_affine_input_epsilon_words[] = {
  UINT32_C(0x3727c5ac),
};

static const int64_t et_n2_ref_layer_norm_affine_input_gamma_shape[] = {
  INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_affine_input_gamma_words[] = {
  UINT32_C(0x3f400000), UINT32_C(0xbfa00000), UINT32_C(0x3f000000), UINT32_C(0x3fc00000),
};

static const int64_t et_n2_ref_layer_norm_affine_input_upstream_shape[] = {
  INT64_C(2), INT64_C(2), INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_affine_input_upstream_words[] = {
  UINT32_C(0xbeb33333), UINT32_C(0xbe333333), UINT32_C(0x00000000), UINT32_C(0x3e333333), UINT32_C(0x3eb33333), UINT32_C(0x3f066666), UINT32_C(0xbeb33333), UINT32_C(0xbe333333),
  UINT32_C(0x00000000), UINT32_C(0x3e333333), UINT32_C(0x3eb33333), UINT32_C(0x3f066666), UINT32_C(0xbeb33333), UINT32_C(0xbe333333), UINT32_C(0x00000000), UINT32_C(0x3e333333),
};

static const int64_t et_n2_ref_layer_norm_affine_input_x_shape[] = {
  INT64_C(2), INT64_C(2), INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_affine_input_x_words[] = {
  UINT32_C(0xbf800000), UINT32_C(0x3f000000), UINT32_C(0x40000000), UINT32_C(0xbe800000), UINT32_C(0x40400000), UINT32_C(0x40400000), UINT32_C(0x40400000), UINT32_C(0x40400000),
  UINT32_C(0x3e000000), UINT32_C(0xbf400000), UINT32_C(0x3fa00000), UINT32_C(0x40200000), UINT32_C(0xc0000000), UINT32_C(0x3f800000), UINT32_C(0x3e800000), UINT32_C(0x3f400000),
};

static const int64_t et_n2_ref_layer_norm_affine_output_forward_shape[] = {
  INT64_C(2), INT64_C(2), INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_affine_output_forward_words[] = {
  UINT32_C(0xbf8b3032), UINT32_C(0xbde3eacf), UINT32_C(0x3f87c2ea), UINT32_C(0xbf948fb6), UINT32_C(0xbe4ccccd), UINT32_C(0x3dcccccd), UINT32_C(0x3e99999a), UINT32_C(0xbecccccd),
  UINT32_C(0xbf1a8431), UINT32_C(0x3fd5b164), UINT32_C(0x3efbff1e), UINT32_C(0x3fdb63f5), UINT32_C(0xbfbb820f), UINT32_C(0xbf743f2a), UINT32_C(0x3ecf91c1), UINT32_C(0x3f0c764a),
};

static const int64_t et_n2_ref_layer_norm_d1_gradient_dbeta_shape[] = {
  INT64_C(1),
};
static const uint32_t et_n2_ref_layer_norm_d1_gradient_dbeta_words[] = {
  UINT32_C(0x3f000000),
};

static const int64_t et_n2_ref_layer_norm_d1_gradient_dgamma_shape[] = {
  INT64_C(1),
};
static const uint32_t et_n2_ref_layer_norm_d1_gradient_dgamma_words[] = {
  UINT32_C(0x00000000),
};

static const int64_t et_n2_ref_layer_norm_d1_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(1),
};
static const uint32_t et_n2_ref_layer_norm_d1_gradient_dx_words[] = {
  UINT32_C(0x00000000),
};

static const int64_t et_n2_ref_layer_norm_d1_input_beta_shape[] = {
  INT64_C(1),
};
static const uint32_t et_n2_ref_layer_norm_d1_input_beta_words[] = {
  UINT32_C(0xbec00000),
};

static const uint32_t et_n2_ref_layer_norm_d1_input_epsilon_words[] = {
  UINT32_C(0x3727c5ac),
};

static const int64_t et_n2_ref_layer_norm_d1_input_gamma_shape[] = {
  INT64_C(1),
};
static const uint32_t et_n2_ref_layer_norm_d1_input_gamma_words[] = {
  UINT32_C(0x3fa00000),
};

static const int64_t et_n2_ref_layer_norm_d1_input_upstream_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(1),
};
static const uint32_t et_n2_ref_layer_norm_d1_input_upstream_words[] = {
  UINT32_C(0x3f000000),
};

static const int64_t et_n2_ref_layer_norm_d1_input_x_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(1),
};
static const uint32_t et_n2_ref_layer_norm_d1_input_x_words[] = {
  UINT32_C(0x40000000),
};

static const int64_t et_n2_ref_layer_norm_d1_output_forward_shape[] = {
  INT64_C(1), INT64_C(1), INT64_C(1),
};
static const uint32_t et_n2_ref_layer_norm_d1_output_forward_words[] = {
  UINT32_C(0xbec00000),
};

static const int64_t et_n2_ref_layer_norm_near_constant_gradient_dbeta_shape[] = {
  INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_near_constant_gradient_dbeta_words[] = {
  UINT32_C(0xbe800000), UINT32_C(0x3f000000), UINT32_C(0x40100000), UINT32_C(0xc0200000),
};

static const int64_t et_n2_ref_layer_norm_near_constant_gradient_dgamma_shape[] = {
  INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_near_constant_gradient_dgamma_words[] = {
  UINT32_C(0x391e1d1e), UINT32_C(0x3aed2bad), UINT32_C(0xbb856892), UINT32_C(0xbb31e0c2),
};

static const int64_t et_n2_ref_layer_norm_near_constant_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_near_constant_gradient_dx_words[] = {
  UINT32_C(0xc2e349f1), UINT32_C(0xc25967d8), UINT32_C(0x43d1fe9a), UINT32_C(0xc37bfe44), UINT32_C(0xc2a7ff02), UINT32_C(0xc3230dec), UINT32_C(0x43a5866e), UINT32_C(0xc2a7fede),
};

static const int64_t et_n2_ref_layer_norm_near_constant_input_beta_shape[] = {
  INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_near_constant_input_beta_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0xbe4ccccd), UINT32_C(0x3e99999a), UINT32_C(0xbecccccd),
};

static const uint32_t et_n2_ref_layer_norm_near_constant_input_epsilon_words[] = {
  UINT32_C(0x3727c5ac),
};

static const int64_t et_n2_ref_layer_norm_near_constant_input_gamma_shape[] = {
  INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_near_constant_input_gamma_words[] = {
  UINT32_C(0x3f400000), UINT32_C(0xbf000000), UINT32_C(0x3fa00000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_layer_norm_near_constant_input_upstream_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_near_constant_input_upstream_words[] = {
  UINT32_C(0x3e800000), UINT32_C(0xbf400000), UINT32_C(0x3fc00000), UINT32_C(0xbf800000), UINT32_C(0xbf000000), UINT32_C(0x3fa00000), UINT32_C(0x3f400000), UINT32_C(0xbfc00000),
};

static const int64_t et_n2_ref_layer_norm_near_constant_input_x_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_near_constant_input_x_words[] = {
  UINT32_C(0x3f800000), UINT32_C(0x3f800020), UINT32_C(0x3f7fffc0), UINT32_C(0x3f800040), UINT32_C(0xc0000000), UINT32_C(0xbfffffc0), UINT32_C(0xc0000010), UINT32_C(0xbfffffe0),
};

static const int64_t et_n2_ref_layer_norm_near_constant_output_forward_shape[] = {
  INT64_C(1), INT64_C(2), INT64_C(4),
};
static const uint32_t et_n2_ref_layer_norm_near_constant_output_forward_words[] = {
  UINT32_C(0x3dcbdfa1), UINT32_C(0xbe4d1bdc), UINT32_C(0x3e987123), UINT32_C(0xbecc9182), UINT32_C(0x3dcbdfa1), UINT32_C(0xbe4db9f9), UINT32_C(0x3e987123), UINT32_C(0xbeccb909),
};

static const int64_t et_n2_ref_linear_bias_gradient_dbias_shape[] = {
  INT64_C(5),
};
static const uint32_t et_n2_ref_linear_bias_gradient_dbias_words[] = {
  UINT32_C(0x3e000000), UINT32_C(0xbe000000), UINT32_C(0x3f200000), UINT32_C(0x3ec00000), UINT32_C(0x3e000000),
};

static const int64_t et_n2_ref_linear_bias_gradient_dweight_shape[] = {
  INT64_C(5), INT64_C(4),
};
static const uint32_t et_n2_ref_linear_bias_gradient_dweight_words[] = {
  UINT32_C(0x3e533334), UINT32_C(0x3ec99999), UINT32_C(0x3e79999b), UINT32_C(0x3e866667), UINT32_C(0x3eb66667), UINT32_C(0x3eacccce), UINT32_C(0xbc9999a0), UINT32_C(0xbd199999),
  UINT32_C(0x3eb66666), UINT32_C(0x3e900000), UINT32_C(0xbe066668), UINT32_C(0xbd199998), UINT32_C(0x3eb66667), UINT32_C(0x3d99999a), UINT32_C(0xbf0b3334), UINT32_C(0xbef9999a),
  UINT32_C(0xbdc00000), UINT32_C(0xbf14ccce), UINT32_C(0xbd666664), UINT32_C(0xbd19999a),
};

static const int64_t et_n2_ref_linear_bias_gradient_dx_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_linear_bias_gradient_dx_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3d19999a), UINT32_C(0xbcccccce), UINT32_C(0x3db33333), UINT32_C(0xbe400000), UINT32_C(0xbe0ccccd), UINT32_C(0x3e333334), UINT32_C(0xbd199999),
  UINT32_C(0x3ccccccc), UINT32_C(0x3db33334), UINT32_C(0xbccccccc), UINT32_C(0x3d19999a), UINT32_C(0xbe266667), UINT32_C(0xbe400000), UINT32_C(0xbe000000), UINT32_C(0x3de66667),
  UINT32_C(0x3d4ccccc), UINT32_C(0x3e0ccccd), UINT32_C(0xbe000000), UINT32_C(0xbe59999a), UINT32_C(0x3d800000), UINT32_C(0x3d800000), UINT32_C(0xbccccccf), UINT32_C(0x3d800000),
};

static const int64_t et_n2_ref_linear_bias_input_bias_shape[] = {
  INT64_C(5),
};
static const uint32_t et_n2_ref_linear_bias_input_bias_words[] = {
  UINT32_C(0xbe99999a), UINT32_C(0xbdcccccd), UINT32_C(0x00000000), UINT32_C(0x3e4ccccd), UINT32_C(0x3ecccccd),
};

static const int64_t et_n2_ref_linear_bias_input_upstream_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(5),
};
static const uint32_t et_n2_ref_linear_bias_input_upstream_words[] = {
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_linear_bias_input_weight_shape[] = {
  INT64_C(5), INT64_C(4),
};
static const uint32_t et_n2_ref_linear_bias_input_weight_words[] = {
  UINT32_C(0xbe99999a), UINT32_C(0xbe4ccccd), UINT32_C(0xbdcccccd), UINT32_C(0x00000000), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0xbe99999a),
  UINT32_C(0xbe4ccccd), UINT32_C(0xbdcccccd), UINT32_C(0x00000000), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0xbe99999a), UINT32_C(0xbe4ccccd),
  UINT32_C(0xbdcccccd), UINT32_C(0x00000000), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd),
};

static const int64_t et_n2_ref_linear_bias_input_x_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_linear_bias_input_x_words[] = {
  UINT32_C(0xbf19999a), UINT32_C(0xbee66667), UINT32_C(0xbe99999a), UINT32_C(0xbe19999a), UINT32_C(0x00000000), UINT32_C(0x3e19999a), UINT32_C(0x3e99999a), UINT32_C(0x3ee66667),
  UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbee66667), UINT32_C(0xbe99999a), UINT32_C(0xbe19999a), UINT32_C(0x00000000), UINT32_C(0x3e19999a), UINT32_C(0x3e99999a),
  UINT32_C(0x3ee66667), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbee66667), UINT32_C(0xbe99999a), UINT32_C(0xbe19999a), UINT32_C(0x00000000), UINT32_C(0x3e19999a),
};

static const int64_t et_n2_ref_linear_bias_output_forward_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(5),
};
static const uint32_t et_n2_ref_linear_bias_output_forward_words[] = {
  UINT32_C(0x00000000), UINT32_C(0xbe970a3e), UINT32_C(0x3e19999a), UINT32_C(0x3d851eb8), UINT32_C(0x3ecccccd), UINT32_C(0xbeb851ec), UINT32_C(0xbdeb851f), UINT32_C(0x3cf5c290),
  UINT32_C(0x3d851eb8), UINT32_C(0x3f051eb8), UINT32_C(0xbea147ae), UINT32_C(0xbe51eb86), UINT32_C(0xbdb851ec), UINT32_C(0x3eab851e), UINT32_C(0x3e70a3d7), UINT32_C(0xbe8a3d71),
  UINT32_C(0xbe23d70a), UINT32_C(0x3d75c290), UINT32_C(0x3d851eb8), UINT32_C(0x3efae148), UINT32_C(0xbefd70a4), UINT32_C(0x3ca3d704), UINT32_C(0xbe47ae15), UINT32_C(0x3f3d70a4),
  UINT32_C(0x3e51eb85), UINT32_C(0xbe3851ec), UINT32_C(0xbe51eb86), UINT32_C(0x3db851ec), UINT32_C(0x3d851eb8), UINT32_C(0x3eeb851f),
};

static const int64_t et_n2_ref_linear_no_bias_gradient_dweight_shape[] = {
  INT64_C(5), INT64_C(4),
};
static const uint32_t et_n2_ref_linear_no_bias_gradient_dweight_words[] = {
  UINT32_C(0x3e533334), UINT32_C(0x3ec99999), UINT32_C(0x3e79999b), UINT32_C(0x3e866667), UINT32_C(0x3eb66667), UINT32_C(0x3eacccce), UINT32_C(0xbc9999a0), UINT32_C(0xbd199999),
  UINT32_C(0x3eb66666), UINT32_C(0x3e900000), UINT32_C(0xbe066668), UINT32_C(0xbd199998), UINT32_C(0x3eb66667), UINT32_C(0x3d99999a), UINT32_C(0xbf0b3334), UINT32_C(0xbef9999a),
  UINT32_C(0xbdc00000), UINT32_C(0xbf14ccce), UINT32_C(0xbd666664), UINT32_C(0xbd19999a),
};

static const int64_t et_n2_ref_linear_no_bias_gradient_dx_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_linear_no_bias_gradient_dx_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3d19999a), UINT32_C(0xbcccccce), UINT32_C(0x3db33333), UINT32_C(0xbe400000), UINT32_C(0xbe0ccccd), UINT32_C(0x3e333334), UINT32_C(0xbd199999),
  UINT32_C(0x3ccccccc), UINT32_C(0x3db33334), UINT32_C(0xbccccccc), UINT32_C(0x3d19999a), UINT32_C(0xbe266667), UINT32_C(0xbe400000), UINT32_C(0xbe000000), UINT32_C(0x3de66667),
  UINT32_C(0x3d4ccccc), UINT32_C(0x3e0ccccd), UINT32_C(0xbe000000), UINT32_C(0xbe59999a), UINT32_C(0x3d800000), UINT32_C(0x3d800000), UINT32_C(0xbccccccf), UINT32_C(0x3d800000),
};

static const int64_t et_n2_ref_linear_no_bias_input_upstream_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(5),
};
static const uint32_t et_n2_ref_linear_no_bias_input_upstream_words[] = {
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0x3f000000),
  UINT32_C(0xbec00000), UINT32_C(0xbe800000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_linear_no_bias_input_weight_shape[] = {
  INT64_C(5), INT64_C(4),
};
static const uint32_t et_n2_ref_linear_no_bias_input_weight_words[] = {
  UINT32_C(0xbe99999a), UINT32_C(0xbe4ccccd), UINT32_C(0xbdcccccd), UINT32_C(0x00000000), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0xbe99999a),
  UINT32_C(0xbe4ccccd), UINT32_C(0xbdcccccd), UINT32_C(0x00000000), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0xbe99999a), UINT32_C(0xbe4ccccd),
  UINT32_C(0xbdcccccd), UINT32_C(0x00000000), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd),
};

static const int64_t et_n2_ref_linear_no_bias_input_x_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_linear_no_bias_input_x_words[] = {
  UINT32_C(0xbf19999a), UINT32_C(0xbee66667), UINT32_C(0xbe99999a), UINT32_C(0xbe19999a), UINT32_C(0x00000000), UINT32_C(0x3e19999a), UINT32_C(0x3e99999a), UINT32_C(0x3ee66667),
  UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbee66667), UINT32_C(0xbe99999a), UINT32_C(0xbe19999a), UINT32_C(0x00000000), UINT32_C(0x3e19999a), UINT32_C(0x3e99999a),
  UINT32_C(0x3ee66667), UINT32_C(0x3f19999a), UINT32_C(0xbf19999a), UINT32_C(0xbee66667), UINT32_C(0xbe99999a), UINT32_C(0xbe19999a), UINT32_C(0x00000000), UINT32_C(0x3e19999a),
};

static const int64_t et_n2_ref_linear_no_bias_output_forward_shape[] = {
  INT64_C(2), INT64_C(3), INT64_C(5),
};
static const uint32_t et_n2_ref_linear_no_bias_output_forward_words[] = {
  UINT32_C(0x3e99999a), UINT32_C(0xbe47ae15), UINT32_C(0x3e19999a), UINT32_C(0xbe0a3d71), UINT32_C(0x00000000), UINT32_C(0xbd75c290), UINT32_C(0xbc75c290), UINT32_C(0x3cf5c290),
  UINT32_C(0xbe0a3d71), UINT32_C(0x3df5c290), UINT32_C(0xbc75c290), UINT32_C(0xbdd70a3e), UINT32_C(0xbdb851ec), UINT32_C(0x3e0a3d70), UINT32_C(0xbe28f5c3), UINT32_C(0x3cf5c290),
  UINT32_C(0xbd75c290), UINT32_C(0x3d75c290), UINT32_C(0xbe0a3d71), UINT32_C(0x3db851ec), UINT32_C(0xbe47ae15), UINT32_C(0x3df5c28e), UINT32_C(0xbe47ae15), UINT32_C(0x3f0a3d71),
  UINT32_C(0xbe47ae15), UINT32_C(0x3df5c290), UINT32_C(0xbdd70a3e), UINT32_C(0x3db851ec), UINT32_C(0xbe0a3d71), UINT32_C(0x3d75c290),
};

static const int64_t et_n2_ref_residual_distinct_gradient_dbranch_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_distinct_gradient_dbranch_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
};

static const int64_t et_n2_ref_residual_distinct_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_distinct_gradient_dx_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
};

static const int64_t et_n2_ref_residual_distinct_input_branch_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_distinct_input_branch_words[] = {
  UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000), UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0xbe000000), UINT32_C(0x00000000), UINT32_C(0x3e000000),
  UINT32_C(0x3e800000), UINT32_C(0x3ec00000), UINT32_C(0xbe000000), UINT32_C(0x00000000),
};

static const int64_t et_n2_ref_residual_distinct_input_upstream_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_distinct_input_upstream_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
};

static const int64_t et_n2_ref_residual_distinct_input_x_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_distinct_input_x_words[] = {
  UINT32_C(0xbf400000), UINT32_C(0xbf000000), UINT32_C(0xbe800000), UINT32_C(0x00000000), UINT32_C(0x3e800000), UINT32_C(0x3f000000), UINT32_C(0x3f400000), UINT32_C(0xbf400000),
  UINT32_C(0xbf000000), UINT32_C(0xbe800000), UINT32_C(0x00000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_residual_distinct_output_forward_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_distinct_output_forward_words[] = {
  UINT32_C(0xbf600000), UINT32_C(0xbf000000), UINT32_C(0xbe000000), UINT32_C(0x3e800000), UINT32_C(0x3f200000), UINT32_C(0x3ec00000), UINT32_C(0x3f400000), UINT32_C(0xbf200000),
  UINT32_C(0xbe800000), UINT32_C(0x3e000000), UINT32_C(0xbe000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_residual_repeated_gradient_dbranch_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_repeated_gradient_dbranch_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
};

static const int64_t et_n2_ref_residual_repeated_gradient_dx_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_repeated_gradient_dx_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
};

static const int64_t et_n2_ref_residual_repeated_input_branch_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_repeated_input_branch_words[] = {
  UINT32_C(0xbf400000), UINT32_C(0xbf000000), UINT32_C(0xbe800000), UINT32_C(0x00000000), UINT32_C(0x3e800000), UINT32_C(0x3f000000), UINT32_C(0x3f400000), UINT32_C(0xbf400000),
  UINT32_C(0xbf000000), UINT32_C(0xbe800000), UINT32_C(0x00000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_residual_repeated_input_upstream_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_repeated_input_upstream_words[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd), UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e99999a), UINT32_C(0x3ecccccd),
};

static const int64_t et_n2_ref_residual_repeated_input_x_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_repeated_input_x_words[] = {
  UINT32_C(0xbf400000), UINT32_C(0xbf000000), UINT32_C(0xbe800000), UINT32_C(0x00000000), UINT32_C(0x3e800000), UINT32_C(0x3f000000), UINT32_C(0x3f400000), UINT32_C(0xbf400000),
  UINT32_C(0xbf000000), UINT32_C(0xbe800000), UINT32_C(0x00000000), UINT32_C(0x3e800000),
};

static const int64_t et_n2_ref_residual_repeated_output_forward_shape[] = {
  INT64_C(1), INT64_C(3), INT64_C(4),
};
static const uint32_t et_n2_ref_residual_repeated_output_forward_words[] = {
  UINT32_C(0xbfc00000), UINT32_C(0xbf800000), UINT32_C(0xbf000000), UINT32_C(0x00000000), UINT32_C(0x3f000000), UINT32_C(0x3f800000), UINT32_C(0x3fc00000), UINT32_C(0xbfc00000),
  UINT32_C(0xbf800000), UINT32_C(0xbf000000), UINT32_C(0x00000000), UINT32_C(0x3f000000),
};

static const struct et_n2_reference_tensor et_n2_reference_tensors[] = {
  {
    "activation.gelu.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_activation_gelu_gradient_dx_shape,
    UINT64_C(8), et_n2_ref_activation_gelu_gradient_dx_words,
  },
  {
    "activation.gelu.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_activation_gelu_output_forward_shape,
    UINT64_C(8), et_n2_ref_activation_gelu_output_forward_words,
  },
  {
    "activation.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_activation_input_upstream_shape,
    UINT64_C(8), et_n2_ref_activation_input_upstream_words,
  },
  {
    "activation.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_activation_input_x_shape,
    UINT64_C(8), et_n2_ref_activation_input_x_words,
  },
  {
    "activation.relu.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_activation_relu_gradient_dx_shape,
    UINT64_C(8), et_n2_ref_activation_relu_gradient_dx_words,
  },
  {
    "activation.relu.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_activation_relu_output_forward_shape,
    UINT64_C(8), et_n2_ref_activation_relu_output_forward_words,
  },
  {
    "dropout.eval.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_dropout_eval_gradient_dx_shape,
    UINT64_C(10), et_n2_ref_dropout_eval_gradient_dx_words,
  },
  {
    "dropout.eval.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_dropout_eval_input_upstream_shape,
    UINT64_C(10), et_n2_ref_dropout_eval_input_upstream_words,
  },
  {
    "dropout.eval.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_dropout_eval_input_x_shape,
    UINT64_C(10), et_n2_ref_dropout_eval_input_x_words,
  },
  {
    "dropout.eval.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_dropout_eval_output_forward_shape,
    UINT64_C(10), et_n2_ref_dropout_eval_output_forward_words,
  },
  {
    "dropout.train.p0.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_dropout_train_p0_gradient_dx_shape,
    UINT64_C(10), et_n2_ref_dropout_train_p0_gradient_dx_words,
  },
  {
    "dropout.train.p0.input.probability", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(0), NULL,
    UINT64_C(1), et_n2_ref_dropout_train_p0_input_probability_words,
  },
  {
    "dropout.train.p0.input.state", ET_N2_REFERENCE_I64,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_dropout_train_p0_input_state_shape,
    UINT64_C(4), et_n2_ref_dropout_train_p0_input_state_words,
  },
  {
    "dropout.train.p0.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_dropout_train_p0_input_upstream_shape,
    UINT64_C(10), et_n2_ref_dropout_train_p0_input_upstream_words,
  },
  {
    "dropout.train.p0.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_dropout_train_p0_input_x_shape,
    UINT64_C(10), et_n2_ref_dropout_train_p0_input_x_words,
  },
  {
    "dropout.train.p0.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_dropout_train_p0_output_forward_shape,
    UINT64_C(10), et_n2_ref_dropout_train_p0_output_forward_words,
  },
  {
    "dropout.train.p0.output.next_state", ET_N2_REFERENCE_I64,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(1), et_n2_ref_dropout_train_p0_output_next_state_shape,
    UINT64_C(4), et_n2_ref_dropout_train_p0_output_next_state_words,
  },
  {
    "dropout.train.p25.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_dropout_train_p25_gradient_dx_shape,
    UINT64_C(10), et_n2_ref_dropout_train_p25_gradient_dx_words,
  },
  {
    "dropout.train.p25.input.probability", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(0), NULL,
    UINT64_C(1), et_n2_ref_dropout_train_p25_input_probability_words,
  },
  {
    "dropout.train.p25.input.state", ET_N2_REFERENCE_I64,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_dropout_train_p25_input_state_shape,
    UINT64_C(4), et_n2_ref_dropout_train_p25_input_state_words,
  },
  {
    "dropout.train.p25.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_dropout_train_p25_input_upstream_shape,
    UINT64_C(10), et_n2_ref_dropout_train_p25_input_upstream_words,
  },
  {
    "dropout.train.p25.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_dropout_train_p25_input_x_shape,
    UINT64_C(10), et_n2_ref_dropout_train_p25_input_x_words,
  },
  {
    "dropout.train.p25.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_dropout_train_p25_output_forward_shape,
    UINT64_C(10), et_n2_ref_dropout_train_p25_output_forward_words,
  },
  {
    "dropout.train.p25.output.next_state", ET_N2_REFERENCE_I64,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(1), et_n2_ref_dropout_train_p25_output_next_state_shape,
    UINT64_C(4), et_n2_ref_dropout_train_p25_output_next_state_words,
  },
  {
    "embedding.repeated.gradient.dweight", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(2), et_n2_ref_embedding_repeated_gradient_dweight_shape,
    UINT64_C(30), et_n2_ref_embedding_repeated_gradient_dweight_words,
  },
  {
    "embedding.repeated.input.ids", ET_N2_REFERENCE_I64,
    ET_N2_REFERENCE_INPUT, UINT32_C(2), et_n2_ref_embedding_repeated_input_ids_shape,
    UINT64_C(6), et_n2_ref_embedding_repeated_input_ids_words,
  },
  {
    "embedding.repeated.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_embedding_repeated_input_upstream_shape,
    UINT64_C(36), et_n2_ref_embedding_repeated_input_upstream_words,
  },
  {
    "embedding.repeated.input.weight", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(2), et_n2_ref_embedding_repeated_input_weight_shape,
    UINT64_C(30), et_n2_ref_embedding_repeated_input_weight_words,
  },
  {
    "embedding.repeated.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_embedding_repeated_output_forward_shape,
    UINT64_C(36), et_n2_ref_embedding_repeated_output_forward_words,
  },
  {
    "layer_norm.affine.gradient.dbeta", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(1), et_n2_ref_layer_norm_affine_gradient_dbeta_shape,
    UINT64_C(4), et_n2_ref_layer_norm_affine_gradient_dbeta_words,
  },
  {
    "layer_norm.affine.gradient.dgamma", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(1), et_n2_ref_layer_norm_affine_gradient_dgamma_shape,
    UINT64_C(4), et_n2_ref_layer_norm_affine_gradient_dgamma_words,
  },
  {
    "layer_norm.affine.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_layer_norm_affine_gradient_dx_shape,
    UINT64_C(16), et_n2_ref_layer_norm_affine_gradient_dx_words,
  },
  {
    "layer_norm.affine.input.beta", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_layer_norm_affine_input_beta_shape,
    UINT64_C(4), et_n2_ref_layer_norm_affine_input_beta_words,
  },
  {
    "layer_norm.affine.input.epsilon", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(0), NULL,
    UINT64_C(1), et_n2_ref_layer_norm_affine_input_epsilon_words,
  },
  {
    "layer_norm.affine.input.gamma", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_layer_norm_affine_input_gamma_shape,
    UINT64_C(4), et_n2_ref_layer_norm_affine_input_gamma_words,
  },
  {
    "layer_norm.affine.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_layer_norm_affine_input_upstream_shape,
    UINT64_C(16), et_n2_ref_layer_norm_affine_input_upstream_words,
  },
  {
    "layer_norm.affine.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_layer_norm_affine_input_x_shape,
    UINT64_C(16), et_n2_ref_layer_norm_affine_input_x_words,
  },
  {
    "layer_norm.affine.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_layer_norm_affine_output_forward_shape,
    UINT64_C(16), et_n2_ref_layer_norm_affine_output_forward_words,
  },
  {
    "layer_norm.d1.gradient.dbeta", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(1), et_n2_ref_layer_norm_d1_gradient_dbeta_shape,
    UINT64_C(1), et_n2_ref_layer_norm_d1_gradient_dbeta_words,
  },
  {
    "layer_norm.d1.gradient.dgamma", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(1), et_n2_ref_layer_norm_d1_gradient_dgamma_shape,
    UINT64_C(1), et_n2_ref_layer_norm_d1_gradient_dgamma_words,
  },
  {
    "layer_norm.d1.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_layer_norm_d1_gradient_dx_shape,
    UINT64_C(1), et_n2_ref_layer_norm_d1_gradient_dx_words,
  },
  {
    "layer_norm.d1.input.beta", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_layer_norm_d1_input_beta_shape,
    UINT64_C(1), et_n2_ref_layer_norm_d1_input_beta_words,
  },
  {
    "layer_norm.d1.input.epsilon", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(0), NULL,
    UINT64_C(1), et_n2_ref_layer_norm_d1_input_epsilon_words,
  },
  {
    "layer_norm.d1.input.gamma", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_layer_norm_d1_input_gamma_shape,
    UINT64_C(1), et_n2_ref_layer_norm_d1_input_gamma_words,
  },
  {
    "layer_norm.d1.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_layer_norm_d1_input_upstream_shape,
    UINT64_C(1), et_n2_ref_layer_norm_d1_input_upstream_words,
  },
  {
    "layer_norm.d1.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_layer_norm_d1_input_x_shape,
    UINT64_C(1), et_n2_ref_layer_norm_d1_input_x_words,
  },
  {
    "layer_norm.d1.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_layer_norm_d1_output_forward_shape,
    UINT64_C(1), et_n2_ref_layer_norm_d1_output_forward_words,
  },
  {
    "layer_norm.near_constant.gradient.dbeta", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(1), et_n2_ref_layer_norm_near_constant_gradient_dbeta_shape,
    UINT64_C(4), et_n2_ref_layer_norm_near_constant_gradient_dbeta_words,
  },
  {
    "layer_norm.near_constant.gradient.dgamma", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(1), et_n2_ref_layer_norm_near_constant_gradient_dgamma_shape,
    UINT64_C(4), et_n2_ref_layer_norm_near_constant_gradient_dgamma_words,
  },
  {
    "layer_norm.near_constant.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_layer_norm_near_constant_gradient_dx_shape,
    UINT64_C(8), et_n2_ref_layer_norm_near_constant_gradient_dx_words,
  },
  {
    "layer_norm.near_constant.input.beta", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_layer_norm_near_constant_input_beta_shape,
    UINT64_C(4), et_n2_ref_layer_norm_near_constant_input_beta_words,
  },
  {
    "layer_norm.near_constant.input.epsilon", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(0), NULL,
    UINT64_C(1), et_n2_ref_layer_norm_near_constant_input_epsilon_words,
  },
  {
    "layer_norm.near_constant.input.gamma", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_layer_norm_near_constant_input_gamma_shape,
    UINT64_C(4), et_n2_ref_layer_norm_near_constant_input_gamma_words,
  },
  {
    "layer_norm.near_constant.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_layer_norm_near_constant_input_upstream_shape,
    UINT64_C(8), et_n2_ref_layer_norm_near_constant_input_upstream_words,
  },
  {
    "layer_norm.near_constant.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_layer_norm_near_constant_input_x_shape,
    UINT64_C(8), et_n2_ref_layer_norm_near_constant_input_x_words,
  },
  {
    "layer_norm.near_constant.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_layer_norm_near_constant_output_forward_shape,
    UINT64_C(8), et_n2_ref_layer_norm_near_constant_output_forward_words,
  },
  {
    "linear.bias.gradient.dbias", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(1), et_n2_ref_linear_bias_gradient_dbias_shape,
    UINT64_C(5), et_n2_ref_linear_bias_gradient_dbias_words,
  },
  {
    "linear.bias.gradient.dweight", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(2), et_n2_ref_linear_bias_gradient_dweight_shape,
    UINT64_C(20), et_n2_ref_linear_bias_gradient_dweight_words,
  },
  {
    "linear.bias.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_linear_bias_gradient_dx_shape,
    UINT64_C(24), et_n2_ref_linear_bias_gradient_dx_words,
  },
  {
    "linear.bias.input.bias", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(1), et_n2_ref_linear_bias_input_bias_shape,
    UINT64_C(5), et_n2_ref_linear_bias_input_bias_words,
  },
  {
    "linear.bias.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_linear_bias_input_upstream_shape,
    UINT64_C(30), et_n2_ref_linear_bias_input_upstream_words,
  },
  {
    "linear.bias.input.weight", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(2), et_n2_ref_linear_bias_input_weight_shape,
    UINT64_C(20), et_n2_ref_linear_bias_input_weight_words,
  },
  {
    "linear.bias.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_linear_bias_input_x_shape,
    UINT64_C(24), et_n2_ref_linear_bias_input_x_words,
  },
  {
    "linear.bias.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_linear_bias_output_forward_shape,
    UINT64_C(30), et_n2_ref_linear_bias_output_forward_words,
  },
  {
    "linear.no_bias.gradient.dweight", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(2), et_n2_ref_linear_no_bias_gradient_dweight_shape,
    UINT64_C(20), et_n2_ref_linear_no_bias_gradient_dweight_words,
  },
  {
    "linear.no_bias.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_linear_no_bias_gradient_dx_shape,
    UINT64_C(24), et_n2_ref_linear_no_bias_gradient_dx_words,
  },
  {
    "linear.no_bias.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_linear_no_bias_input_upstream_shape,
    UINT64_C(30), et_n2_ref_linear_no_bias_input_upstream_words,
  },
  {
    "linear.no_bias.input.weight", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(2), et_n2_ref_linear_no_bias_input_weight_shape,
    UINT64_C(20), et_n2_ref_linear_no_bias_input_weight_words,
  },
  {
    "linear.no_bias.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_linear_no_bias_input_x_shape,
    UINT64_C(24), et_n2_ref_linear_no_bias_input_x_words,
  },
  {
    "linear.no_bias.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_linear_no_bias_output_forward_shape,
    UINT64_C(30), et_n2_ref_linear_no_bias_output_forward_words,
  },
  {
    "residual.distinct.gradient.dbranch", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_residual_distinct_gradient_dbranch_shape,
    UINT64_C(12), et_n2_ref_residual_distinct_gradient_dbranch_words,
  },
  {
    "residual.distinct.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_residual_distinct_gradient_dx_shape,
    UINT64_C(12), et_n2_ref_residual_distinct_gradient_dx_words,
  },
  {
    "residual.distinct.input.branch", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_residual_distinct_input_branch_shape,
    UINT64_C(12), et_n2_ref_residual_distinct_input_branch_words,
  },
  {
    "residual.distinct.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_residual_distinct_input_upstream_shape,
    UINT64_C(12), et_n2_ref_residual_distinct_input_upstream_words,
  },
  {
    "residual.distinct.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_residual_distinct_input_x_shape,
    UINT64_C(12), et_n2_ref_residual_distinct_input_x_words,
  },
  {
    "residual.distinct.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_residual_distinct_output_forward_shape,
    UINT64_C(12), et_n2_ref_residual_distinct_output_forward_words,
  },
  {
    "residual.repeated.gradient.dbranch", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_residual_repeated_gradient_dbranch_shape,
    UINT64_C(12), et_n2_ref_residual_repeated_gradient_dbranch_words,
  },
  {
    "residual.repeated.gradient.dx", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_ANALYTIC_GRADIENT, UINT32_C(3), et_n2_ref_residual_repeated_gradient_dx_shape,
    UINT64_C(12), et_n2_ref_residual_repeated_gradient_dx_words,
  },
  {
    "residual.repeated.input.branch", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_residual_repeated_input_branch_shape,
    UINT64_C(12), et_n2_ref_residual_repeated_input_branch_words,
  },
  {
    "residual.repeated.input.upstream", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_residual_repeated_input_upstream_shape,
    UINT64_C(12), et_n2_ref_residual_repeated_input_upstream_words,
  },
  {
    "residual.repeated.input.x", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_INPUT, UINT32_C(3), et_n2_ref_residual_repeated_input_x_shape,
    UINT64_C(12), et_n2_ref_residual_repeated_input_x_words,
  },
  {
    "residual.repeated.output.forward", ET_N2_REFERENCE_F32,
    ET_N2_REFERENCE_EXPECTED, UINT32_C(3), et_n2_ref_residual_repeated_output_forward_shape,
    UINT64_C(12), et_n2_ref_residual_repeated_output_forward_words,
  },
};

static const uint16_t et_n2_ref_case_dropout_eval_forward_inputs[] = {
  UINT16_C(8),
};
static const uint16_t et_n2_ref_case_dropout_eval_forward_outputs[] = {
  UINT16_C(9),
};

static const uint16_t et_n2_ref_case_dropout_eval_gradient_inputs[] = {
  UINT16_C(7),
};
static const uint16_t et_n2_ref_case_dropout_eval_gradient_outputs[] = {
  UINT16_C(6),
};

static const uint16_t et_n2_ref_case_dropout_train_p0_forward_inputs[] = {
  UINT16_C(14), UINT16_C(11), UINT16_C(12),
};
static const uint16_t et_n2_ref_case_dropout_train_p0_forward_outputs[] = {
  UINT16_C(15), UINT16_C(16),
};

static const uint16_t et_n2_ref_case_dropout_train_p0_gradient_inputs[] = {
  UINT16_C(13), UINT16_C(11), UINT16_C(12),
};
static const uint16_t et_n2_ref_case_dropout_train_p0_gradient_outputs[] = {
  UINT16_C(10),
};

static const uint16_t et_n2_ref_case_dropout_train_p25_forward_inputs[] = {
  UINT16_C(21), UINT16_C(18), UINT16_C(19),
};
static const uint16_t et_n2_ref_case_dropout_train_p25_forward_outputs[] = {
  UINT16_C(22), UINT16_C(23),
};

static const uint16_t et_n2_ref_case_dropout_train_p25_gradient_inputs[] = {
  UINT16_C(20), UINT16_C(18), UINT16_C(19),
};
static const uint16_t et_n2_ref_case_dropout_train_p25_gradient_outputs[] = {
  UINT16_C(17),
};

static const uint16_t et_n2_ref_case_embedding_repeated_forward_inputs[] = {
  UINT16_C(25), UINT16_C(27),
};
static const uint16_t et_n2_ref_case_embedding_repeated_forward_outputs[] = {
  UINT16_C(28),
};

static const uint16_t et_n2_ref_case_embedding_repeated_gradient_inputs[] = {
  UINT16_C(25), UINT16_C(26),
};
static const uint16_t et_n2_ref_case_embedding_repeated_gradient_outputs[] = {
  UINT16_C(24),
};

static const uint16_t et_n2_ref_case_gelu_exact_forward_inputs[] = {
  UINT16_C(3),
};
static const uint16_t et_n2_ref_case_gelu_exact_forward_outputs[] = {
  UINT16_C(1),
};

static const uint16_t et_n2_ref_case_gelu_exact_gradient_inputs[] = {
  UINT16_C(3), UINT16_C(2),
};
static const uint16_t et_n2_ref_case_gelu_exact_gradient_outputs[] = {
  UINT16_C(0),
};

static const uint16_t et_n2_ref_case_layer_norm_affine_forward_inputs[] = {
  UINT16_C(36), UINT16_C(34), UINT16_C(32), UINT16_C(33),
};
static const uint16_t et_n2_ref_case_layer_norm_affine_forward_outputs[] = {
  UINT16_C(37),
};

static const uint16_t et_n2_ref_case_layer_norm_affine_gradient_inputs[] = {
  UINT16_C(36), UINT16_C(34), UINT16_C(33), UINT16_C(35),
};
static const uint16_t et_n2_ref_case_layer_norm_affine_gradient_outputs[] = {
  UINT16_C(31), UINT16_C(30), UINT16_C(29),
};

static const uint16_t et_n2_ref_case_layer_norm_d1_forward_inputs[] = {
  UINT16_C(45), UINT16_C(43), UINT16_C(41), UINT16_C(42),
};
static const uint16_t et_n2_ref_case_layer_norm_d1_forward_outputs[] = {
  UINT16_C(46),
};

static const uint16_t et_n2_ref_case_layer_norm_d1_gradient_inputs[] = {
  UINT16_C(45), UINT16_C(43), UINT16_C(42), UINT16_C(44),
};
static const uint16_t et_n2_ref_case_layer_norm_d1_gradient_outputs[] = {
  UINT16_C(40), UINT16_C(39), UINT16_C(38),
};

static const uint16_t et_n2_ref_case_layer_norm_near_constant_forward_inputs[] = {
  UINT16_C(54), UINT16_C(52), UINT16_C(50), UINT16_C(51),
};
static const uint16_t et_n2_ref_case_layer_norm_near_constant_forward_outputs[] = {
  UINT16_C(55),
};

static const uint16_t et_n2_ref_case_layer_norm_near_constant_gradient_inputs[] = {
  UINT16_C(54), UINT16_C(52), UINT16_C(51), UINT16_C(53),
};
static const uint16_t et_n2_ref_case_layer_norm_near_constant_gradient_outputs[] = {
  UINT16_C(49), UINT16_C(48), UINT16_C(47),
};

static const uint16_t et_n2_ref_case_linear_bias_forward_inputs[] = {
  UINT16_C(62), UINT16_C(61), UINT16_C(59),
};
static const uint16_t et_n2_ref_case_linear_bias_forward_outputs[] = {
  UINT16_C(63),
};

static const uint16_t et_n2_ref_case_linear_bias_gradient_inputs[] = {
  UINT16_C(62), UINT16_C(61), UINT16_C(60),
};
static const uint16_t et_n2_ref_case_linear_bias_gradient_outputs[] = {
  UINT16_C(58), UINT16_C(57), UINT16_C(56),
};

static const uint16_t et_n2_ref_case_linear_no_bias_forward_inputs[] = {
  UINT16_C(68), UINT16_C(67),
};
static const uint16_t et_n2_ref_case_linear_no_bias_forward_outputs[] = {
  UINT16_C(69),
};

static const uint16_t et_n2_ref_case_linear_no_bias_gradient_inputs[] = {
  UINT16_C(68), UINT16_C(67), UINT16_C(66),
};
static const uint16_t et_n2_ref_case_linear_no_bias_gradient_outputs[] = {
  UINT16_C(65), UINT16_C(64),
};

static const uint16_t et_n2_ref_case_relu_zero_forward_inputs[] = {
  UINT16_C(3),
};
static const uint16_t et_n2_ref_case_relu_zero_forward_outputs[] = {
  UINT16_C(5),
};

static const uint16_t et_n2_ref_case_relu_zero_gradient_inputs[] = {
  UINT16_C(3), UINT16_C(2),
};
static const uint16_t et_n2_ref_case_relu_zero_gradient_outputs[] = {
  UINT16_C(4),
};

static const uint16_t et_n2_ref_case_residual_distinct_forward_inputs[] = {
  UINT16_C(74), UINT16_C(72),
};
static const uint16_t et_n2_ref_case_residual_distinct_forward_outputs[] = {
  UINT16_C(75),
};

static const uint16_t et_n2_ref_case_residual_distinct_gradient_inputs[] = {
  UINT16_C(73),
};
static const uint16_t et_n2_ref_case_residual_distinct_gradient_outputs[] = {
  UINT16_C(71), UINT16_C(70),
};

static const uint16_t et_n2_ref_case_residual_repeated_forward_inputs[] = {
  UINT16_C(80), UINT16_C(78),
};
static const uint16_t et_n2_ref_case_residual_repeated_forward_outputs[] = {
  UINT16_C(81),
};

static const uint16_t et_n2_ref_case_residual_repeated_gradient_inputs[] = {
  UINT16_C(79),
};
static const uint16_t et_n2_ref_case_residual_repeated_gradient_outputs[] = {
  UINT16_C(77), UINT16_C(76),
};

static const struct et_n2_reference_case et_n2_reference_cases[] = {
  {
    "dropout.eval.forward", "dropout.eval.forward", "parity",
    UINT32_C(1), et_n2_ref_case_dropout_eval_forward_inputs,
    UINT32_C(1), et_n2_ref_case_dropout_eval_forward_outputs,
  },
  {
    "dropout.eval.gradient", "dropout.eval.backward", "gradient",
    UINT32_C(1), et_n2_ref_case_dropout_eval_gradient_inputs,
    UINT32_C(1), et_n2_ref_case_dropout_eval_gradient_outputs,
  },
  {
    "dropout.train.p0.forward", "dropout.train.forward", "known_value",
    UINT32_C(3), et_n2_ref_case_dropout_train_p0_forward_inputs,
    UINT32_C(2), et_n2_ref_case_dropout_train_p0_forward_outputs,
  },
  {
    "dropout.train.p0.gradient", "dropout.train.backward", "gradient",
    UINT32_C(3), et_n2_ref_case_dropout_train_p0_gradient_inputs,
    UINT32_C(1), et_n2_ref_case_dropout_train_p0_gradient_outputs,
  },
  {
    "dropout.train.p25.forward", "dropout.train.forward", "known_value",
    UINT32_C(3), et_n2_ref_case_dropout_train_p25_forward_inputs,
    UINT32_C(2), et_n2_ref_case_dropout_train_p25_forward_outputs,
  },
  {
    "dropout.train.p25.gradient", "dropout.train.backward", "gradient",
    UINT32_C(3), et_n2_ref_case_dropout_train_p25_gradient_inputs,
    UINT32_C(1), et_n2_ref_case_dropout_train_p25_gradient_outputs,
  },
  {
    "embedding.repeated.forward", "embedding.forward", "repeated_input",
    UINT32_C(2), et_n2_ref_case_embedding_repeated_forward_inputs,
    UINT32_C(1), et_n2_ref_case_embedding_repeated_forward_outputs,
  },
  {
    "embedding.repeated.gradient", "embedding.backward", "gradient",
    UINT32_C(2), et_n2_ref_case_embedding_repeated_gradient_inputs,
    UINT32_C(1), et_n2_ref_case_embedding_repeated_gradient_outputs,
  },
  {
    "gelu.exact.forward", "gelu.forward", "parity",
    UINT32_C(1), et_n2_ref_case_gelu_exact_forward_inputs,
    UINT32_C(1), et_n2_ref_case_gelu_exact_forward_outputs,
  },
  {
    "gelu.exact.gradient", "gelu.backward", "gradient",
    UINT32_C(2), et_n2_ref_case_gelu_exact_gradient_inputs,
    UINT32_C(1), et_n2_ref_case_gelu_exact_gradient_outputs,
  },
  {
    "layer_norm.affine.forward", "layer-norm.forward", "parity",
    UINT32_C(4), et_n2_ref_case_layer_norm_affine_forward_inputs,
    UINT32_C(1), et_n2_ref_case_layer_norm_affine_forward_outputs,
  },
  {
    "layer_norm.affine.gradient", "layer-norm.backward", "gradient",
    UINT32_C(4), et_n2_ref_case_layer_norm_affine_gradient_inputs,
    UINT32_C(3), et_n2_ref_case_layer_norm_affine_gradient_outputs,
  },
  {
    "layer_norm.d1.forward", "layer-norm.forward", "boundary",
    UINT32_C(4), et_n2_ref_case_layer_norm_d1_forward_inputs,
    UINT32_C(1), et_n2_ref_case_layer_norm_d1_forward_outputs,
  },
  {
    "layer_norm.d1.gradient", "layer-norm.backward", "gradient",
    UINT32_C(4), et_n2_ref_case_layer_norm_d1_gradient_inputs,
    UINT32_C(3), et_n2_ref_case_layer_norm_d1_gradient_outputs,
  },
  {
    "layer_norm.near_constant.forward", "layer-norm.forward", "boundary",
    UINT32_C(4), et_n2_ref_case_layer_norm_near_constant_forward_inputs,
    UINT32_C(1), et_n2_ref_case_layer_norm_near_constant_forward_outputs,
  },
  {
    "layer_norm.near_constant.gradient", "layer-norm.backward", "gradient",
    UINT32_C(4), et_n2_ref_case_layer_norm_near_constant_gradient_inputs,
    UINT32_C(3), et_n2_ref_case_layer_norm_near_constant_gradient_outputs,
  },
  {
    "linear.bias.forward", "linear.forward", "parity",
    UINT32_C(3), et_n2_ref_case_linear_bias_forward_inputs,
    UINT32_C(1), et_n2_ref_case_linear_bias_forward_outputs,
  },
  {
    "linear.bias.gradient", "linear.backward", "gradient",
    UINT32_C(3), et_n2_ref_case_linear_bias_gradient_inputs,
    UINT32_C(3), et_n2_ref_case_linear_bias_gradient_outputs,
  },
  {
    "linear.no_bias.forward", "linear.forward-no-bias", "parity",
    UINT32_C(2), et_n2_ref_case_linear_no_bias_forward_inputs,
    UINT32_C(1), et_n2_ref_case_linear_no_bias_forward_outputs,
  },
  {
    "linear.no_bias.gradient", "linear.backward-no-bias", "gradient",
    UINT32_C(3), et_n2_ref_case_linear_no_bias_gradient_inputs,
    UINT32_C(2), et_n2_ref_case_linear_no_bias_gradient_outputs,
  },
  {
    "relu.zero.forward", "relu.forward", "special_value",
    UINT32_C(1), et_n2_ref_case_relu_zero_forward_inputs,
    UINT32_C(1), et_n2_ref_case_relu_zero_forward_outputs,
  },
  {
    "relu.zero.gradient", "relu.backward", "gradient",
    UINT32_C(2), et_n2_ref_case_relu_zero_gradient_inputs,
    UINT32_C(1), et_n2_ref_case_relu_zero_gradient_outputs,
  },
  {
    "residual.distinct.forward", "residual.forward", "parity",
    UINT32_C(2), et_n2_ref_case_residual_distinct_forward_inputs,
    UINT32_C(1), et_n2_ref_case_residual_distinct_forward_outputs,
  },
  {
    "residual.distinct.gradient", "residual.backward", "gradient",
    UINT32_C(1), et_n2_ref_case_residual_distinct_gradient_inputs,
    UINT32_C(2), et_n2_ref_case_residual_distinct_gradient_outputs,
  },
  {
    "residual.repeated.forward", "residual.forward", "repeated_input",
    UINT32_C(2), et_n2_ref_case_residual_repeated_forward_inputs,
    UINT32_C(1), et_n2_ref_case_residual_repeated_forward_outputs,
  },
  {
    "residual.repeated.gradient", "residual.backward", "gradient",
    UINT32_C(1), et_n2_ref_case_residual_repeated_gradient_inputs,
    UINT32_C(2), et_n2_ref_case_residual_repeated_gradient_outputs,
  },
};

#endif
