/* Candidate exact outputs for the supported Ubuntu 22.04 / Clang 21.1.8 lane.
 * Acceptance requires the supported CI run; local compatibility is not proof.
 */
#ifndef ESHKOL_TRANSFORMER_N2_BIT_GOLDENS_H
#define ESHKOL_TRANSFORMER_N2_BIT_GOLDENS_H

#include <stdint.h>

#define ET_N2_BIT_GOLDEN_PLATFORM "ubuntu-22.04-x86_64-clang-21.1.8"

/* FNV-1a over raw output bytes; these supplement the expanded word arrays. */
static const uint64_t et_n2_hash_embedding_y = UINT64_C(0x870f312d455893e5);
static const uint64_t et_n2_hash_embedding_dweight =
    UINT64_C(0x88cd727caf4e5226);
static const uint64_t et_n2_hash_linear_bias_y = UINT64_C(0x54bd75e336cc8e78);
static const uint64_t et_n2_hash_linear_dx = UINT64_C(0x7c99345e8946d725);
static const uint64_t et_n2_hash_linear_dbias = UINT64_C(0xb4f5897be651e015);
static const uint64_t et_n2_hash_layer_norm_dgamma =
    UINT64_C(0x368be4c5d00b500e);
static const uint64_t et_n2_hash_layer_norm_dbeta =
    UINT64_C(0xce23c66eb7b3b3b0);
static const uint64_t et_n2_hash_layer_norm_d1_y = UINT64_C(0x4a99487f9ba4408b);
static const uint64_t et_n2_hash_layer_norm_d1_dx =
    UINT64_C(0x4d25767f9dce13f5);
static const uint64_t et_n2_hash_layer_norm_d1_dgamma =
    UINT64_C(0x4d25767f9dce13f5);
static const uint64_t et_n2_hash_layer_norm_d1_dbeta =
    UINT64_C(0x4a98c77f9ba36558);
static const uint64_t et_n2_hash_layer_norm_constant_y =
    UINT64_C(0x569a0c7d0e134376);
static const uint64_t et_n2_hash_layer_norm_constant_dx =
    UINT64_C(0x6850589435bdc72c);
static const uint64_t et_n2_hash_layer_norm_constant_dgamma =
    UINT64_C(0x5467b0da1d106495);
static const uint64_t et_n2_hash_layer_norm_constant_dbeta =
    UINT64_C(0xf55a4ac33599de05);
static const uint64_t et_n2_hash_relu_y = UINT64_C(0x3dfac65d44f7ebdb);
static const uint64_t et_n2_hash_relu_dx = UINT64_C(0x6743863e190155e8);
static const uint64_t et_n2_hash_residual_y = UINT64_C(0xe00583dab6bee48b);
static const uint64_t et_n2_hash_residual_dx = UINT64_C(0xdaec424841b198f4);
static const uint64_t et_n2_hash_residual_dbranch =
    UINT64_C(0xdaec424841b198f4);
static const uint64_t et_n2_hash_dropout_eval_y = UINT64_C(0xfd27690d4f163970);
static const uint64_t et_n2_hash_dropout_eval_dx = UINT64_C(0x72fd0ba4cb3381c5);
static const uint64_t et_n2_hash_dropout_pos_zero_y =
    UINT64_C(0xfd27690d4f163970);
static const uint64_t et_n2_hash_dropout_pos_zero_state =
    UINT64_C(0x29df073a676b745a);
static const uint64_t et_n2_hash_dropout_pos_zero_dx =
    UINT64_C(0x72fd0ba4cb3381c5);
static const uint64_t et_n2_hash_dropout_neg_zero_y =
    UINT64_C(0xfd27690d4f163970);
static const uint64_t et_n2_hash_dropout_neg_zero_state =
    UINT64_C(0x29df073a676b745a);
static const uint64_t et_n2_hash_dropout_neg_zero_dx =
    UINT64_C(0x72fd0ba4cb3381c5);

static const uint32_t et_n2_bits_linear_y[30] = {
    UINT32_C(0xe0021ab1), UINT32_C(0x5f2d78ec), UINT32_C(0xdfad78ec),
    UINT32_C(0x5fad78ec), UINT32_C(0xdf2d78ec), UINT32_C(0x60021ab1),
    UINT32_C(0xdf2d78ec), UINT32_C(0x5fad78ec), UINT32_C(0xdfad78ec),
    UINT32_C(0x5f2d78ec), UINT32_C(0xbf480000), UINT32_C(0x3d000000),
    UINT32_C(0xbf300000), UINT32_C(0x3f480000), UINT32_C(0xbf180000),
    UINT32_C(0xbed00000), UINT32_C(0x3d000000), UINT32_C(0xbe400000),
    UINT32_C(0x3d000000), UINT32_C(0x3d000000), UINT32_C(0xbf600000),
    UINT32_C(0x3ed00000), UINT32_C(0xbf380000), UINT32_C(0x3fb80000),
    UINT32_C(0xbf100000), UINT32_C(0xbfb80000), UINT32_C(0x3eb00000),
    UINT32_C(0xbf700000), UINT32_C(0x3f580000), UINT32_C(0xbee00000)};
static const uint32_t et_n2_bits_linear_dweight[20] = {
    UINT32_C(0x41200000), UINT32_C(0xbf400000), UINT32_C(0xbfc00000),
    UINT32_C(0x00000000), UINT32_C(0xc0200000), UINT32_C(0x3e400000),
    UINT32_C(0x3ec00000), UINT32_C(0x00000000), UINT32_C(0x00000000),
    UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000),
    UINT32_C(0x40200000), UINT32_C(0xbe400000), UINT32_C(0xbec00000),
    UINT32_C(0x00000000), UINT32_C(0x40a00000), UINT32_C(0xbec00000),
    UINT32_C(0xbf400000), UINT32_C(0x00000000)};
static const uint32_t et_n2_bits_layer_norm_y[16] = {
    UINT32_C(0xbe58f3b9), UINT32_C(0x3da463c0), UINT32_C(0x3e8d7526),
    UINT32_C(0xbea866d9), UINT32_C(0x3f5c5439), UINT32_C(0x3dccccc2),
    UINT32_C(0xbed0704c), UINT32_C(0xbeccccca), UINT32_C(0xbf0ce9b9),
    UINT32_C(0x3fdc780c), UINT32_C(0x3ef6a2bc), UINT32_C(0x3fd9f05d),
    UINT32_C(0xbe9f9cc4), UINT32_C(0xbdb0facb), UINT32_C(0x3d9c8e3a),
    UINT32_C(0x3e8ab149)};
static const uint32_t et_n2_bits_layer_norm_dx[16] = {
    UINT32_C(0xc2c86e7a), UINT32_C(0x424f4b7b), UINT32_C(0xc189a987),
    UINT32_C(0x4283332f), UINT32_C(0x32b62693), UINT32_C(0xb3225a0a),
    UINT32_C(0x32b62693), UINT32_C(0xb19e6454), UINT32_C(0xbc5f19e3),
    UINT32_C(0x3d617b30), UINT32_C(0xbdf2f1c6), UINT32_C(0x3d9e176e),
    UINT32_C(0xc2b8ef4f), UINT32_C(0x423b3f75), UINT32_C(0xc120066f),
    UINT32_C(0x425e9ca8)};
static const uint32_t et_n2_bits_gelu_y[8] = {
    UINT32_C(0x80000000), UINT32_C(0xbb84b340), UINT32_C(0xbe227686),
    UINT32_C(0x80000000), UINT32_C(0x00000000), UINT32_C(0x3eb103af),
    UINT32_C(0x3ffa2d0c), UINT32_C(0x41200000)};
static const uint32_t et_n2_bits_gelu_dx[8] = {
    UINT32_C(0x9b688e14), UINT32_C(0xbbc3b7af), UINT32_C(0x3d7ff1f5),
    UINT32_C(0x3f000000), UINT32_C(0xbf200000), UINT32_C(0x3fa68f1f),
    UINT32_C(0xc00ae8e0), UINT32_C(0x40200000)};
static const uint32_t et_n2_bits_dropout_y[10] = {
    UINT32_C(0xbf4cccce), UINT32_C(0xbf088889), UINT32_C(0xbe888889),
    UINT32_C(0x00000000), UINT32_C(0x3e888889), UINT32_C(0x00000000),
    UINT32_C(0x00000000), UINT32_C(0xbf4cccce), UINT32_C(0xbf088889),
    UINT32_C(0xbe888889)};
static const int64_t et_n2_bits_dropout_next[4] = {
    INT64_C(1), INT64_C(2999170649027065890), -INT64_C(8817193942522041717),
    INT64_C(247824715720788526)};
static const uint32_t et_n2_bits_dropout_dx[10] = {
    UINT32_C(0x3e2aaaab), UINT32_C(0x3eaaaaab), UINT32_C(0x3f000000),
    UINT32_C(0x00000000), UINT32_C(0x3e2aaaab), UINT32_C(0x00000000),
    UINT32_C(0x00000000), UINT32_C(0x3f2aaaab), UINT32_C(0x3e2aaaab),
    UINT32_C(0x3eaaaaab)};

#endif
