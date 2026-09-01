#include "e1b_error_consumer_bridge.h"

/* Canonical Wave 2 is built from trusted sources, never localized archives. */
#include "x1_config_consumer_bridge.c"
#include "p1_package_bridge.c"
#include "d1_e1b_package_bridge.c"

#define ET_T1_DECLARE_UNARY(name)                                           \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t value)
#define ET_T1_DECLARE_BINARY(name)                                          \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t left,             \
                                     eshkol_tagged_value_t right)
#define ET_T1_DECLARE_TERNARY(name)                                         \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t first,            \
                                     eshkol_tagged_value_t second,           \
                                     eshkol_tagged_value_t third)

ET_T1_DECLARE_UNARY(et_e1b_private_t1_tokenizer_byte_cabi_v1);
ET_T1_DECLARE_BINARY(et_e1b_private_t1_tokenizer_load_cabi_v1);
ET_T1_DECLARE_TERNARY(et_e1b_private_t1_tokenizer_save_cabi_v1);
ET_T1_DECLARE_BINARY(et_e1b_private_t1_tokenizer_encode_cabi_v1);
ET_T1_DECLARE_BINARY(et_e1b_private_t1_tokenizer_decode_cabi_v1);
ET_T1_DECLARE_UNARY(et_e1b_private_t1_tokenizer_vocab_size_cabi_v1);
ET_T1_DECLARE_UNARY(et_e1b_private_t1_tokenizer_fingerprint_cabi_v1);
ET_T1_DECLARE_BINARY(et_e1b_private_t1_tokenizer_special_token_id_cabi_v1);

extern eshkol_tagged_value_t et_e1b_private_c1_persistence_policy_cabi_v1(
    eshkol_tagged_value_t max_file, eshkol_tagged_value_t max_metadata,
    eshkol_tagged_value_t max_tensor, eshkol_tagged_value_t max_tensors,
    eshkol_tagged_value_t device);

#define ET_T1_PUBLIC_UNARY(name, target)                                    \
  void name(void *input, void *output) {                                    \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));      \
  }

#define ET_T1_PUBLIC_BINARY(name, target)                                   \
  void name(void *left, void *right, void *output) {                        \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) =                                          \
        target(*et_e1b_box_value_v1(left), *et_e1b_box_value_v1(right));     \
  }

ET_T1_PUBLIC_UNARY(et_e1b_public_t1_tokenizer_byte_v1,
                   et_e1b_private_t1_tokenizer_byte_cabi_v1)
ET_T1_PUBLIC_BINARY(et_e1b_public_t1_tokenizer_load_v1,
                    et_e1b_private_t1_tokenizer_load_cabi_v1)
ET_T1_PUBLIC_BINARY(et_e1b_public_t1_tokenizer_encode_v1,
                    et_e1b_private_t1_tokenizer_encode_cabi_v1)
ET_T1_PUBLIC_BINARY(et_e1b_public_t1_tokenizer_decode_v1,
                    et_e1b_private_t1_tokenizer_decode_cabi_v1)
ET_T1_PUBLIC_UNARY(et_e1b_public_t1_tokenizer_vocab_size_v1,
                   et_e1b_private_t1_tokenizer_vocab_size_cabi_v1)
ET_T1_PUBLIC_UNARY(et_e1b_public_t1_tokenizer_fingerprint_v1,
                   et_e1b_private_t1_tokenizer_fingerprint_cabi_v1)
ET_T1_PUBLIC_BINARY(et_e1b_public_t1_tokenizer_special_token_id_v1,
                    et_e1b_private_t1_tokenizer_special_token_id_cabi_v1)

void et_e1b_public_t1_tokenizer_save_v1(void *tokenizer, void *path,
                                        void *policy, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_t1_tokenizer_save_cabi_v1(
      *et_e1b_box_value_v1(tokenizer), *et_e1b_box_value_v1(path),
      *et_e1b_box_value_v1(policy));
}

void et_e1b_public_c1_persistence_policy_v1(
    void *max_file, void *max_metadata, void *max_tensor, void *max_tensors,
    void *device, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_c1_persistence_policy_cabi_v1(
      *et_e1b_box_value_v1(max_file), *et_e1b_box_value_v1(max_metadata),
      *et_e1b_box_value_v1(max_tensor), *et_e1b_box_value_v1(max_tensors),
      *et_e1b_box_value_v1(device));
}
