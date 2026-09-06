#include "o2_optimizer_internal.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#define ET_O2_BUILDER_MAGIC UINT64_C(0x45544f324255494c)
#define ET_O2_OPTIMIZER_MAGIC UINT64_C(0x45544f324f505449)
#define ET_O2_STATE_MAGIC UINT64_C(0x45544f3253544154)
#define ET_O2_HANDLE_MAGIC UINT64_C(0x45544f3248414e44)
#define ET_O2_BORROW_MAGIC UINT64_C(0x45544f32424f5252)

_Static_assert(ET_O2_MAX_PARAMETERS <= ET_F32_PARAMETER_MAX_BATCH / 3u,
               "O2 step exceeds the I2 transaction ceiling");
_Static_assert(ET_O2_MAX_PARAMETERS <= SIZE_MAX / 3u,
               "O2 entry multipliers overflow size_t");

typedef struct et_o2_options {
  uint32_t learning_rate_bits;
  uint32_t beta1_bits;
  uint32_t beta2_bits;
  uint32_t epsilon_bits;
  uint32_t weight_decay_bits;
} et_o2_options;

typedef struct et_o2_config {
  uint32_t clip_kind;
  uint32_t clip_max_bits;
  uint32_t schedule_kind;
  uint32_t minimum_ratio_bits;
  uint64_t warmup_updates;
  uint64_t total_updates;
} et_o2_config;

typedef struct et_o2_entry {
  et_f32_parameter *parameter;
  const void *p1_handle;
  const et_f32_tensor *storage_owner;
  et_f32_tensor *exp_avg;
  et_f32_tensor *exp_avg_sq;
  et_o2_options options;
  uint32_t initialized;
} et_o2_entry;

typedef struct et_o2_state_entry {
  et_f32_tensor *exp_avg;
  et_f32_tensor *exp_avg_sq;
  et_o2_options options;
} et_o2_state_entry;

struct et_o2_optimizer_builder {
  uint64_t magic;
  et_o2_optimizer_builder *registry_next;
  size_t count;
  et_o2_config config;
  et_o2_entry *entries;
};

struct et_o2_optimizer {
  uint64_t magic;
  et_o2_optimizer *registry_next;
  size_t count;
  et_o2_config config;
  et_o2_entry *entries;
  uint64_t completed_updates;
  uint32_t provider_major;
  uint32_t provider_minor;
  char provider_id[sizeof(ET_O2_PROVIDER_ID)];
  uint32_t busy;
};

struct et_o2_optimizer_state_handle {
  uint64_t magic;
  et_o2_optimizer_state_handle *registry_next;
  et_o2_optimizer_state *owner;
  size_t index;
  uint32_t moment_kind;
  uint32_t live;
};

struct et_o2_optimizer_state {
  uint64_t magic;
  et_o2_optimizer_state *registry_next;
  size_t count;
  et_o2_config config;
  et_o2_state_entry *entries;
  et_o2_optimizer_state_handle **handles;
  uint64_t completed_updates;
  uint64_t owned_clone_count;
  uint32_t provider_major;
  uint32_t provider_minor;
  char provider_id[sizeof(ET_O2_PROVIDER_ID)];
  uint32_t lifecycle;
  uint32_t active_borrows;
};

struct et_o2_optimizer_state_borrow {
  uint64_t magic;
  et_o2_optimizer_state_borrow *registry_next;
  et_o2_optimizer_state *owner;
  et_o2_optimizer_state_handle *handle;
  et_f32_tensor_borrow *i2_borrow;
};

static et_o2_optimizer_builder *et_o2_builders;
static et_o2_optimizer *et_o2_optimizers;
static et_o2_optimizer_state *et_o2_states;
static et_o2_optimizer_state_handle *et_o2_handles;
static et_o2_optimizer_state_borrow *et_o2_borrows;
static et_o2_optimizer_state_borrow *et_o2_retired_borrows;
static size_t et_o2_owned_state_clones;

#ifdef ET_O2_TESTING
static size_t et_o2_allocation_limit = SIZE_MAX;
static size_t et_o2_successful_allocations;
static size_t et_o2_release_limit = SIZE_MAX;
static size_t et_o2_successful_releases;
#endif

static void *et_o2_calloc(size_t count, size_t size) {
  void *allocation;
  if (size != 0u && count > SIZE_MAX / size) {
    return NULL;
  }
#ifdef ET_O2_TESTING
  if (et_o2_successful_allocations >= et_o2_allocation_limit) {
    return NULL;
  }
#endif
  allocation = calloc(count, size);
#ifdef ET_O2_TESTING
  if (allocation != NULL) {
    et_o2_successful_allocations++;
  }
#endif
  return allocation;
}

static void et_o2_error_clear(et_o2_error_v1 *error) {
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
}

static int32_t et_o2_fail(et_o2_error_v1 *error, uint32_t category,
                          uint32_t code, const char *operation,
                          const char *message) {
  if (error != NULL) {
    et_o2_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s", operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return (int32_t)category;
}

static int32_t et_o2_success(et_o2_error_v1 *error) {
  et_o2_error_clear(error);
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_from_i2(const et_f32_tensor_error *source,
                             et_o2_error_v1 *error, const char *operation,
                             uint32_t default_category) {
  uint32_t category = default_category;
  if (source != NULL) {
    if (source->category == ET_F32_TENSOR_ERROR_SHAPE_MISMATCH) {
      category = ET_O2_STATUS_SHAPE_MISMATCH;
    } else if (source->category == ET_F32_TENSOR_ERROR_NONCONTIGUOUS) {
      category = ET_O2_STATUS_NONCONTIGUOUS;
    } else if (source->category == ET_F32_TENSOR_ERROR_INTERNAL) {
      category = ET_O2_STATUS_INTERNAL;
    }
  }
  return et_o2_fail(error, category,
                    source != NULL && source->code != 0u
                        ? source->code
                        : ET_O2_CODE_INVALID_HANDLE,
                    operation, "I2 dense CPU-f32 operation failed");
}

static et_o2_optimizer_builder *et_o2_find_builder(const void *candidate) {
  et_o2_optimizer_builder *cursor = et_o2_builders;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor->magic == ET_O2_BUILDER_MAGIC ? cursor : NULL;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_o2_optimizer *et_o2_find_optimizer(const void *candidate) {
  et_o2_optimizer *cursor = et_o2_optimizers;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor->magic == ET_O2_OPTIMIZER_MAGIC ? cursor : NULL;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_o2_optimizer_state *et_o2_find_state(const void *candidate) {
  et_o2_optimizer_state *cursor = et_o2_states;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor->magic == ET_O2_STATE_MAGIC ? cursor : NULL;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_o2_optimizer_state_handle *et_o2_find_handle(const void *candidate) {
  et_o2_optimizer_state_handle *cursor = et_o2_handles;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor->magic == ET_O2_HANDLE_MAGIC ? cursor : NULL;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_o2_optimizer_state_borrow *et_o2_find_borrow(const void *candidate) {
  et_o2_optimizer_state_borrow *cursor = et_o2_borrows;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor->magic == ET_O2_BORROW_MAGIC ? cursor : NULL;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static void et_o2_unlink_builder(et_o2_optimizer_builder *builder) {
  et_o2_optimizer_builder **link = &et_o2_builders;
  while (*link != NULL && *link != builder) {
    link = &(*link)->registry_next;
  }
  if (*link == builder) {
    *link = builder->registry_next;
  }
}

static void et_o2_unlink_borrow(et_o2_optimizer_state_borrow *borrow) {
  et_o2_optimizer_state_borrow **link = &et_o2_borrows;
  while (*link != NULL && *link != borrow) {
    link = &(*link)->registry_next;
  }
  if (*link == borrow) {
    *link = borrow->registry_next;
  }
}

static int et_o2_bits_finite(uint32_t bits) {
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int et_o2_bits_nonnegative_finite(uint32_t bits) {
  return (bits & UINT32_C(0x80000000)) == 0u && et_o2_bits_finite(bits);
}

static int et_o2_bits_positive_finite(uint32_t bits) {
  return et_o2_bits_nonnegative_finite(bits) &&
         (bits & UINT32_C(0x7fffffff)) != 0u;
}

static float et_o2_float_from_bits(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t et_o2_bits_from_float(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int et_o2_float_environment(void) {
  return (_mm_getcsr() & UINT32_C(0x0000ffc0)) == UINT32_C(0x00001f80);
}

static uint32_t et_o2_binary(uint32_t left_bits, uint32_t right_bits,
                             uint32_t operation) {
  const __m128 left = _mm_set_ss(et_o2_float_from_bits(left_bits));
  const __m128 right = _mm_set_ss(et_o2_float_from_bits(right_bits));
  __m128 result;
  if (operation == 0u) {
    result = _mm_add_ss(left, right);
  } else if (operation == 1u) {
    result = _mm_sub_ss(left, right);
  } else if (operation == 2u) {
    result = _mm_mul_ss(left, right);
  } else {
    result = _mm_div_ss(left, right);
  }
  return et_o2_bits_from_float(_mm_cvtss_f32(result));
}

static uint32_t et_o2_add(uint32_t left, uint32_t right) {
  return et_o2_binary(left, right, 0u);
}

static uint32_t et_o2_sub(uint32_t left, uint32_t right) {
  return et_o2_binary(left, right, 1u);
}

static uint32_t et_o2_mul(uint32_t left, uint32_t right) {
  return et_o2_binary(left, right, 2u);
}

static uint32_t et_o2_div(uint32_t left, uint32_t right) {
  return et_o2_binary(left, right, 3u);
}

static uint32_t et_o2_sqrt(uint32_t value) {
  __m128 operand = _mm_set_ss(et_o2_float_from_bits(value));
  return et_o2_bits_from_float(_mm_cvtss_f32(_mm_sqrt_ss(operand)));
}

static uint32_t et_o2_pow_u64(uint32_t base, uint64_t exponent) {
  uint32_t result = UINT32_C(0x3f800000);
  while (exponent != 0u) {
    if ((exponent & 1u) != 0u) {
      result = et_o2_mul(result, base);
    }
    exponent >>= 1u;
    if (exponent != 0u) {
      base = et_o2_mul(base, base);
    }
  }
  return result;
}

typedef struct et_o2_u256 {
  uint64_t limb[4];
} et_o2_u256;

static et_o2_u256 et_o2_u256_from_u64(uint64_t value) {
  et_o2_u256 result = {{value, 0u, 0u, 0u}};
  return result;
}

static int et_o2_u256_compare(const et_o2_u256 *left, const et_o2_u256 *right) {
  size_t index;
  for (index = 4u; index > 0u; --index) {
    if (left->limb[index - 1u] < right->limb[index - 1u]) {
      return -1;
    }
    if (left->limb[index - 1u] > right->limb[index - 1u]) {
      return 1;
    }
  }
  return 0;
}

static void et_o2_u256_shift_left(et_o2_u256 *value, unsigned shift) {
  const unsigned words = shift / 64u;
  const unsigned bits = shift % 64u;
  et_o2_u256 source = *value;
  size_t destination;
  memset(value, 0, sizeof(*value));
  for (destination = words; destination < 4u; ++destination) {
    const size_t source_index = destination - words;
    value->limb[destination] |= source.limb[source_index] << bits;
    if (bits != 0u && source_index > 0u) {
      value->limb[destination] |=
          source.limb[source_index - 1u] >> (64u - bits);
    }
  }
}

static void et_o2_u256_add(et_o2_u256 *left, const et_o2_u256 *right) {
  uint64_t carry = 0u;
  size_t index;
  for (index = 0u; index < 4u; ++index) {
    uint64_t first = left->limb[index] + right->limb[index];
    uint64_t first_carry = first < left->limb[index] ? 1u : 0u;
    uint64_t result = first + carry;
    uint64_t second_carry = result < first ? 1u : 0u;
    left->limb[index] = result;
    carry = first_carry | second_carry;
  }
}

static void et_o2_u256_subtract(et_o2_u256 *left, const et_o2_u256 *right) {
  uint64_t borrow = 0u;
  size_t index;
  for (index = 0u; index < 4u; ++index) {
    uint64_t first = left->limb[index] - right->limb[index];
    uint64_t first_borrow = left->limb[index] < right->limb[index] ? 1u : 0u;
    uint64_t result = first - borrow;
    uint64_t second_borrow = first < borrow ? 1u : 0u;
    left->limb[index] = result;
    borrow = first_borrow | second_borrow;
  }
}

static int et_o2_u256_bit(const et_o2_u256 *value, unsigned bit) {
  return (int)((value->limb[bit / 64u] >> (bit % 64u)) & 1u);
}

static unsigned et_o2_u256_bit_length(const et_o2_u256 *value) {
  size_t index;
  for (index = 4u; index > 0u; --index) {
    if (value->limb[index - 1u] != 0u) {
      return (unsigned)((index - 1u) * 64u + 64u -
                        (unsigned)__builtin_clzll(value->limb[index - 1u]));
    }
  }
  return 0u;
}

static et_o2_u256 et_o2_u256_multiply_u32(uint64_t value, uint32_t multiplier) {
  const uint64_t low_product = (value & UINT64_C(0xffffffff)) * multiplier;
  const uint64_t high_product = (value >> 32u) * multiplier;
  const uint64_t shifted_high = high_product << 32u;
  et_o2_u256 result = {{0u, 0u, 0u, 0u}};
  result.limb[0] = low_product + shifted_high;
  result.limb[1] =
      (high_product >> 32u) + (result.limb[0] < low_product ? 1u : 0u);
  return result;
}

static uint32_t et_o2_u256_divide_small(const et_o2_u256 *numerator,
                                        const et_o2_u256 *denominator,
                                        et_o2_u256 *remainder) {
  const unsigned bits = et_o2_u256_bit_length(numerator);
  uint32_t quotient = 0u;
  unsigned index;
  memset(remainder, 0, sizeof(*remainder));
  for (index = bits; index > 0u; --index) {
    et_o2_u256_shift_left(remainder, 1u);
    remainder->limb[0] |= (uint64_t)et_o2_u256_bit(numerator, index - 1u);
    quotient <<= 1u;
    if (et_o2_u256_compare(remainder, denominator) >= 0) {
      et_o2_u256_subtract(remainder, denominator);
      quotient |= 1u;
    }
  }
  return quotient;
}

static uint32_t et_o2_round_rational_f32(et_o2_u256 numerator,
                                         et_o2_u256 denominator) {
  et_o2_u256 scaled;
  et_o2_u256 remainder;
  et_o2_u256 twice_remainder;
  uint32_t significand;
  unsigned normalization_shift = 0u;
  unsigned exponent_field;
  if (et_o2_u256_bit_length(&numerator) == 0u) {
    return 0u;
  }
  scaled = numerator;
  while (et_o2_u256_compare(&scaled, &denominator) < 0) {
    et_o2_u256_shift_left(&scaled, 1u);
    normalization_shift++;
  }
  if (normalization_shift <= 126u) {
    scaled = numerator;
    et_o2_u256_shift_left(&scaled, normalization_shift + 23u);
    significand = et_o2_u256_divide_small(&scaled, &denominator, &remainder);
    twice_remainder = remainder;
    et_o2_u256_shift_left(&twice_remainder, 1u);
    if (et_o2_u256_compare(&twice_remainder, &denominator) > 0 ||
        (et_o2_u256_compare(&twice_remainder, &denominator) == 0 &&
         (significand & 1u) != 0u)) {
      significand++;
    }
    exponent_field = 127u - normalization_shift;
    if (significand == UINT32_C(0x01000000)) {
      significand >>= 1u;
      exponent_field++;
    }
    return (exponent_field << 23u) | (significand - UINT32_C(0x00800000));
  }
  scaled = numerator;
  et_o2_u256_shift_left(&scaled, 149u);
  significand = et_o2_u256_divide_small(&scaled, &denominator, &remainder);
  twice_remainder = remainder;
  et_o2_u256_shift_left(&twice_remainder, 1u);
  if (et_o2_u256_compare(&twice_remainder, &denominator) > 0 ||
      (et_o2_u256_compare(&twice_remainder, &denominator) == 0 &&
       (significand & 1u) != 0u)) {
    significand++;
  }
  return significand;
}

static void et_o2_ratio_from_schedule(const et_o2_config *config,
                                      uint64_t update, et_o2_u256 *numerator,
                                      et_o2_u256 *denominator) {
  if (config->warmup_updates > 0u && update <= config->warmup_updates) {
    *numerator = et_o2_u256_from_u64(update);
    *denominator = et_o2_u256_from_u64(config->warmup_updates);
  } else {
    const uint64_t elapsed = update - config->warmup_updates;
    const uint64_t duration = config->total_updates - config->warmup_updates;
    const uint32_t exponent =
        (config->minimum_ratio_bits >> 23u) & UINT32_C(0xff);
    const uint32_t fraction = config->minimum_ratio_bits & UINT32_C(0x007fffff);
    const unsigned denominator_power =
        exponent == 0u ? 149u : (unsigned)(150u - exponent);
    const uint32_t ratio_numerator =
        exponent == 0u ? fraction : UINT32_C(0x00800000) | fraction;
    et_o2_u256 ratio_term = et_o2_u256_multiply_u32(elapsed, ratio_numerator);
    *numerator = et_o2_u256_from_u64(duration - elapsed);
    et_o2_u256_shift_left(numerator, denominator_power);
    et_o2_u256_add(numerator, &ratio_term);
    *denominator = et_o2_u256_from_u64(duration);
    et_o2_u256_shift_left(denominator, denominator_power);
  }
}

static int et_o2_options_valid(const et_o2_options *options) {
  const uint32_t one = UINT32_C(0x3f800000);
  return et_o2_bits_nonnegative_finite(options->learning_rate_bits) &&
         et_o2_bits_nonnegative_finite(options->beta1_bits) &&
         et_o2_float_from_bits(options->beta1_bits) <
             et_o2_float_from_bits(one) &&
         et_o2_bits_nonnegative_finite(options->beta2_bits) &&
         et_o2_float_from_bits(options->beta2_bits) <
             et_o2_float_from_bits(one) &&
         et_o2_bits_positive_finite(options->epsilon_bits) &&
         et_o2_bits_nonnegative_finite(options->weight_decay_bits);
}

static int et_o2_config_valid(const et_o2_config *config) {
  if (config->clip_kind == ET_O2_CLIP_NONE) {
    if (config->clip_max_bits != 0u) {
      return 0;
    }
  } else if (config->clip_kind == ET_O2_CLIP_GLOBAL_L2) {
    if (!et_o2_bits_positive_finite(config->clip_max_bits)) {
      return 0;
    }
  } else {
    return 0;
  }
  if (config->schedule_kind == ET_O2_SCHEDULE_CONSTANT) {
    return config->warmup_updates == 0u && config->total_updates == 0u &&
           config->minimum_ratio_bits == UINT32_C(0x3f800000);
  }
  return config->schedule_kind == ET_O2_SCHEDULE_LINEAR &&
         config->total_updates > 0u &&
         config->total_updates <= (uint64_t)INT64_MAX &&
         config->warmup_updates < config->total_updates &&
         et_o2_bits_nonnegative_finite(config->minimum_ratio_bits) &&
         et_o2_float_from_bits(config->minimum_ratio_bits) <= 1.0f;
}

static int et_o2_provider_valid(const et_o2_optimizer_state *state) {
  return state->provider_major == ET_O2_PROVIDER_ABI_MAJOR &&
         state->provider_minor == ET_O2_PROVIDER_ABI_MINOR &&
         memcmp(state->provider_id, ET_O2_PROVIDER_ID,
                sizeof(ET_O2_PROVIDER_ID)) == 0;
}

static int et_o2_optimizer_provider_valid(const et_o2_optimizer *optimizer) {
  return optimizer->provider_major == ET_O2_PROVIDER_ABI_MAJOR &&
         optimizer->provider_minor == ET_O2_PROVIDER_ABI_MINOR &&
         memcmp(optimizer->provider_id, ET_O2_PROVIDER_ID,
                sizeof(ET_O2_PROVIDER_ID)) == 0;
}

static int32_t et_o2_require_optimizer(const et_o2_optimizer *candidate,
                                       const char *operation,
                                       et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  if (optimizer == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_INVALID_HANDLE, operation,
                      "optimizer is foreign, stale, or invalid");
  }
  if (!et_o2_optimizer_provider_valid(optimizer)) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_PROVIDER_MISMATCH, operation,
                      "optimizer provider invariant is broken");
  }
  if (optimizer->count == 0u || optimizer->count > ET_O2_MAX_PARAMETERS ||
      optimizer->entries == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT,
                      operation, "optimizer receiver bounds are invalid");
  }
  if (optimizer->busy != 0u) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_ACTIVE_BORROW, operation,
                      "optimizer is already active");
  }
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_require_state_live(const et_o2_optimizer_state *candidate,
                                        const char *operation,
                                        et_o2_error_v1 *error) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  if (state == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, operation,
                      "optimizer state is malformed or unregistered");
  }
  if (state->lifecycle != ET_O2_OPTIMIZER_STATE_LIVE) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED,
                      operation, "optimizer state is not live");
  }
  if (!et_o2_provider_valid(state)) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_PROVIDER_MISMATCH, operation,
                      "optimizer state provider invariant is broken");
  }
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_tensor_zero(et_f32_tensor *tensor, et_o2_error_v1 *error,
                                 const char *operation) {
  et_f32_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_f32_tensor_error i2_error;
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_tensor_borrow_begin_v1(tensor, &borrow, &i2_error) != 0 ||
      et_f32_tensor_borrow_view_v1(borrow, &view, &i2_error) != 0) {
    if (borrow != NULL) {
      (void)et_f32_tensor_borrow_end_v1(&borrow, NULL);
    }
    return et_o2_from_i2(&i2_error, error, operation,
                         ET_O2_STATUS_INVALID_STATE);
  }
  if (view->byte_length != 0u) {
    memset(view->data, 0, view->byte_length);
  }
  if (et_f32_tensor_borrow_end_v1(&borrow, &i2_error) != 0) {
    return et_o2_from_i2(&i2_error, error, operation, ET_O2_STATUS_INTERNAL);
  }
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_create_zero_moment(et_f32_parameter *parameter,
                                        et_f32_tensor **moment,
                                        et_o2_error_v1 *error) {
  const et_f32_tensor *value = NULL;
  et_f32_tensor_error i2_error;
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_parameter_value_tensor_v1(parameter, &value, &i2_error) != 0 ||
      et_f32_tensor_clone_v1(value, moment, &i2_error) != 0) {
    return et_o2_from_i2(&i2_error, error, "optimizer-create",
                         ET_O2_STATUS_INVALID_STATE);
  }
  if (et_o2_tensor_zero(*moment, error, "optimizer-create") != 0) {
    (void)et_f32_tensor_destroy_v1(moment, NULL);
    return (int32_t)(error != NULL ? error->category : ET_O2_STATUS_INTERNAL);
  }
  return ET_O2_STATUS_OK;
}

static void et_o2_destroy_entry_moments(et_o2_entry *entries, size_t count) {
  size_t index;
  if (entries == NULL) {
    return;
  }
  for (index = 0u; index < count; ++index) {
    if (entries[index].exp_avg_sq != NULL) {
      (void)et_f32_tensor_destroy_v1(&entries[index].exp_avg_sq, NULL);
    }
    if (entries[index].exp_avg != NULL) {
      (void)et_f32_tensor_destroy_v1(&entries[index].exp_avg, NULL);
    }
  }
}

int32_t et_o2_optimizer_builder_create_v1(
    size_t parameter_count, uint32_t clip_kind, uint32_t clip_max_bits,
    uint32_t schedule_kind, uint64_t warmup_updates, uint64_t total_updates,
    uint32_t minimum_ratio_bits, et_o2_optimizer_builder **output,
    et_o2_error_v1 *error) {
  et_o2_optimizer_builder *builder;
  et_o2_config config;
  if (output == NULL || *output != NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-create",
                      "builder output must be a nonnull empty slot");
  }
  config.clip_kind = clip_kind;
  config.clip_max_bits = clip_max_bits;
  config.schedule_kind = schedule_kind;
  config.minimum_ratio_bits = minimum_ratio_bits;
  config.warmup_updates = warmup_updates;
  config.total_updates = total_updates;
  if (parameter_count == 0u || parameter_count > ET_O2_MAX_PARAMETERS ||
      !et_o2_config_valid(&config)) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_OPTION, "optimizer-create",
                      "optimizer count or schedule/clip options are invalid");
  }
  builder = (et_o2_optimizer_builder *)et_o2_calloc(1u, sizeof(*builder));
  if (builder != NULL) {
    builder->entries =
        (et_o2_entry *)et_o2_calloc(parameter_count, sizeof(*builder->entries));
  }
  if (builder == NULL || builder->entries == NULL) {
    free(builder);
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_ALLOCATION_FAILED, "optimizer-create",
                      "cannot allocate optimizer builder");
  }
  builder->magic = ET_O2_BUILDER_MAGIC;
  builder->count = parameter_count;
  builder->config = config;
  builder->registry_next = et_o2_builders;
  et_o2_builders = builder;
  *output = builder;
  return et_o2_success(error);
}

int32_t et_o2_optimizer_builder_set_v1(
    et_o2_optimizer_builder *candidate, size_t index,
    et_f32_parameter *parameter, const void *p1_handle,
    uint32_t learning_rate_bits, uint32_t beta1_bits, uint32_t beta2_bits,
    uint32_t epsilon_bits, uint32_t weight_decay_bits, et_o2_error_v1 *error) {
  et_o2_optimizer_builder *builder = et_o2_find_builder(candidate);
  et_o2_entry *entry;
  et_o2_options options;
  et_f32_tensor_error i2_error;
  const et_f32_tensor *storage_owner;
  size_t previous;
  if (builder == NULL || index >= (builder != NULL ? builder->count : 0u) ||
      parameter == NULL || p1_handle == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, "optimizer-create",
                      "builder entry operands are invalid");
  }
  entry = &builder->entries[index];
  if (entry->initialized != 0u) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_DUPLICATE_PARAMETER, "optimizer-create",
                      "builder entry is already initialized");
  }
  options.learning_rate_bits = learning_rate_bits;
  options.beta1_bits = beta1_bits;
  options.beta2_bits = beta2_bits;
  options.epsilon_bits = epsilon_bits;
  options.weight_decay_bits = weight_decay_bits;
  if (!et_o2_options_valid(&options)) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_OPTION, "optimizer-create",
                      "AdamW options are invalid");
  }
  storage_owner = et_f32_parameter_canonical_owner_v1(parameter);
  if (storage_owner == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_INVALID_HANDLE, "optimizer-create",
                      "parameter storage is foreign or stale");
  }
  for (previous = 0u; previous < builder->count; ++previous) {
    if (builder->entries[previous].initialized != 0u &&
        (builder->entries[previous].parameter == parameter ||
         builder->entries[previous].p1_handle == p1_handle ||
         builder->entries[previous].storage_owner == storage_owner)) {
      return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                        ET_O2_CODE_DUPLICATE_PARAMETER, "optimizer-create",
                        "parameter storage is already assigned to a group");
    }
  }
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_parameter_validate_identity_v1(parameter, p1_handle, &i2_error) !=
      0) {
    return et_o2_from_i2(&i2_error, error, "optimizer-create",
                         ET_O2_STATUS_INVALID_STATE);
  }
  entry->parameter = parameter;
  entry->p1_handle = p1_handle;
  entry->storage_owner = storage_owner;
  entry->options = options;
  if (et_o2_create_zero_moment(parameter, &entry->exp_avg, error) != 0 ||
      et_o2_create_zero_moment(parameter, &entry->exp_avg_sq, error) != 0) {
    if (entry->exp_avg_sq != NULL) {
      (void)et_f32_tensor_destroy_v1(&entry->exp_avg_sq, NULL);
    }
    if (entry->exp_avg != NULL) {
      (void)et_f32_tensor_destroy_v1(&entry->exp_avg, NULL);
    }
    memset(entry, 0, sizeof(*entry));
    return (int32_t)(error != NULL ? error->category : ET_O2_STATUS_INTERNAL);
  }
  entry->initialized = 1u;
  return et_o2_success(error);
}

int32_t et_o2_optimizer_builder_finish_v1(et_o2_optimizer_builder **slot,
                                          et_o2_optimizer **output,
                                          et_o2_error_v1 *error) {
  et_o2_optimizer_builder *builder;
  et_o2_optimizer *optimizer;
  size_t left;
  size_t right;
  if (slot == NULL || output == NULL || *output != NULL ||
      (builder = et_o2_find_builder(slot != NULL ? *slot : NULL)) == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, "optimizer-create",
                      "builder finish operands are invalid");
  }
  for (left = 0u; left < builder->count; ++left) {
    if (builder->entries[left].initialized == 0u) {
      return et_o2_fail(error, ET_O2_STATUS_SHAPE_MISMATCH,
                        ET_O2_CODE_MISSING_PARAMETER, "optimizer-create",
                        "optimizer builder has a missing entry");
    }
    for (right = left + 1u; right < builder->count; ++right) {
      if (builder->entries[right].initialized != 0u &&
          (builder->entries[left].parameter ==
               builder->entries[right].parameter ||
           builder->entries[left].p1_handle ==
               builder->entries[right].p1_handle ||
           et_f32_tensor_storage_owner_identical_v1(
               et_f32_parameter_canonical_owner_v1(
                   builder->entries[left].parameter),
               et_f32_parameter_canonical_owner_v1(
                   builder->entries[right].parameter)) == 1)) {
        return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                          ET_O2_CODE_DUPLICATE_PARAMETER, "optimizer-create",
                          "optimizer contains a duplicate parameter identity");
      }
    }
  }
  optimizer = (et_o2_optimizer *)et_o2_calloc(1u, sizeof(*optimizer));
  if (optimizer == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_ALLOCATION_FAILED, "optimizer-create",
                      "cannot publish optimizer receiver");
  }
  optimizer->magic = ET_O2_OPTIMIZER_MAGIC;
  optimizer->count = builder->count;
  optimizer->config = builder->config;
  optimizer->entries = builder->entries;
  optimizer->provider_major = ET_O2_PROVIDER_ABI_MAJOR;
  optimizer->provider_minor = ET_O2_PROVIDER_ABI_MINOR;
  memcpy(optimizer->provider_id, ET_O2_PROVIDER_ID, sizeof(ET_O2_PROVIDER_ID));
  optimizer->registry_next = et_o2_optimizers;
  et_o2_optimizers = optimizer;
  builder->entries = NULL;
  et_o2_unlink_builder(builder);
  builder->magic = 0u;
  free(builder);
  *slot = NULL;
  *output = optimizer;
  return et_o2_success(error);
}

int32_t et_o2_optimizer_builder_abort_v1(et_o2_optimizer_builder **slot,
                                         et_o2_error_v1 *error) {
  et_o2_optimizer_builder *builder;
  if (slot == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-create",
                      "builder slot is null");
  }
  if (*slot == NULL) {
    return et_o2_success(error);
  }
  builder = et_o2_find_builder(*slot);
  if (builder == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, "optimizer-create",
                      "builder is foreign or stale");
  }
  et_o2_unlink_builder(builder);
  et_o2_destroy_entry_moments(builder->entries, builder->count);
  free(builder->entries);
  builder->magic = 0u;
  free(builder);
  *slot = NULL;
  return et_o2_success(error);
}

static int32_t et_o2_validate_absent_gradients(et_o2_optimizer *optimizer,
                                               const char *operation,
                                               et_o2_error_v1 *error) {
  size_t index;
  et_f32_tensor_error i2_error;
  for (index = 0u; index < optimizer->count; ++index) {
    et_f32_gradient_metadata_v1 metadata;
    memset(&i2_error, 0, sizeof(i2_error));
    if (et_f32_parameter_validate_identity_v1(
            optimizer->entries[index].parameter,
            optimizer->entries[index].p1_handle, &i2_error) != 0) {
      return et_o2_from_i2(&i2_error, error, operation,
                           ET_O2_STATUS_INVALID_STATE);
    }
    memset(&metadata, 0, sizeof(metadata));
    metadata.struct_size = sizeof(metadata);
    if (et_f32_parameter_gradient_metadata_v1(
            optimizer->entries[index].parameter, &metadata, &i2_error) != 0) {
      return et_o2_from_i2(&i2_error, error, operation,
                           ET_O2_STATUS_INVALID_STATE);
    }
    if (metadata.state != ET_F32_GRADIENT_ABSENT ||
        metadata.contribution_count != 0u ||
        metadata.normalization_weight_bits != 0u) {
      return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                        ET_O2_CODE_GRADIENT_METADATA, operation,
                        "optimizer operation requires absent gradients");
    }
  }
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_schedule_factor(const et_o2_config *config,
                                     uint64_t update, uint32_t *factor,
                                     et_o2_error_v1 *error) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_u256 numerator;
  et_o2_u256 denominator;
  uint32_t result;
  if (!et_o2_float_environment()) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_FLOAT_ENVIRONMENT, "optimizer-step!",
                      "binary32 environment is unsupported");
  }
  if (config->schedule_kind == ET_O2_SCHEDULE_CONSTANT) {
    *factor = one;
    return ET_O2_STATUS_OK;
  }
  if (update > config->total_updates) {
    result = config->minimum_ratio_bits;
  } else {
    et_o2_ratio_from_schedule(config, update, &numerator, &denominator);
    result = et_o2_round_rational_f32(numerator, denominator);
  }
  if (!et_o2_bits_nonnegative_finite(result) ||
      et_o2_float_from_bits(result) > 1.0f) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE,
                      "optimizer-step!", "schedule factor is invalid");
  }
  *factor = result;
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_borrow_tensor(et_f32_tensor *tensor,
                                   et_f32_tensor_borrow **borrow,
                                   const et_kernel_tensor_view_v1 **view,
                                   et_o2_error_v1 *error,
                                   const char *operation) {
  et_f32_tensor_error i2_error;
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_tensor_borrow_begin_v1(tensor, borrow, &i2_error) != 0 ||
      et_f32_tensor_borrow_view_v1(*borrow, view, &i2_error) != 0) {
    if (*borrow != NULL) {
      (void)et_f32_tensor_borrow_end_v1(borrow, NULL);
    }
    return et_o2_from_i2(&i2_error, error, operation,
                         ET_O2_STATUS_INVALID_STATE);
  }
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_validate_state_moment(et_f32_tensor *tensor,
                                           int require_nonnegative,
                                           et_o2_error_v1 *error) {
  et_f32_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_f32_tensor_error i2_error;
  size_t index;
  int32_t result;
  if ((result = et_o2_borrow_tensor(tensor, &borrow, &view, error,
                                    "optimizer-load-state!")) != 0) {
    return result;
  }
  if (view->struct_size < sizeof(*view) || view->dtype == NULL ||
      strcmp(view->dtype, "f32") != 0 || view->device == NULL ||
      strcmp(view->device, "cpu") != 0 ||
      view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u || view->byte_length % sizeof(uint32_t) != 0u ||
      (view->byte_length != 0u && view->data == NULL)) {
    result = et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                        ET_O2_CODE_INVALID_MOMENT, "optimizer-load-state!",
                        "optimizer moment metadata is malformed");
    goto cleanup;
  }
  for (index = 0u; index < view->byte_length / sizeof(uint32_t); ++index) {
    uint32_t bits;
    memcpy(&bits, (const unsigned char *)view->data + index * sizeof(bits),
           sizeof(bits));
    if (!et_o2_bits_finite(bits)) {
      result = et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                          ET_O2_CODE_NONFINITE, "optimizer-load-state!",
                          "optimizer moment contains a nonfinite value");
      goto cleanup;
    }
    if (require_nonnegative != 0 && (bits & UINT32_C(0x80000000)) != 0u &&
        (bits & UINT32_C(0x7fffffff)) != 0u) {
      result = et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                          ET_O2_CODE_INVALID_MOMENT, "optimizer-load-state!",
                          "second moment contains a negative value");
      goto cleanup;
    }
  }
  result = ET_O2_STATUS_OK;

cleanup:
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_tensor_borrow_end_v1(&borrow, &i2_error) != 0) {
    return et_o2_from_i2(&i2_error, error, "optimizer-load-state!",
                         ET_O2_STATUS_INTERNAL);
  }
  return result;
}

static int32_t et_o2_validate_state_for_load(et_o2_optimizer_state *state,
                                             et_o2_error_v1 *error) {
  size_t entry_count;
  size_t index;
  int32_t result;
  if (!et_o2_provider_valid(state)) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_PROVIDER_MISMATCH, "optimizer-load-state!",
                      "optimizer state provider invariant is broken");
  }
  if (state->count == 0u || state->count > ET_O2_MAX_PARAMETERS ||
      state->entries == NULL || state->handles == NULL ||
      state->owned_clone_count != (uint64_t)(state->count * 2u) ||
      state->owned_clone_count > (uint64_t)et_o2_owned_state_clones) {
    return et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                      ET_O2_CODE_OWNER_CONFLICT, "optimizer-load-state!",
                      "optimizer state ownership metadata is inconsistent");
  }
  if (!et_o2_config_valid(&state->config)) {
    return et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                      ET_O2_CODE_INVALID_OPTION, "optimizer-load-state!",
                      "optimizer state configuration is invalid");
  }
  if (state->completed_updates > (uint64_t)INT64_MAX) {
    return et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                      ET_O2_CODE_COUNTER_OVERFLOW, "optimizer-load-state!",
                      "optimizer state counter is outside signed-i64 range");
  }
  entry_count = state->count * 2u;
  for (index = 0u; index < state->count; ++index) {
    if (!et_o2_options_valid(&state->entries[index].options)) {
      return et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                        ET_O2_CODE_INVALID_OPTION, "optimizer-load-state!",
                        "optimizer state group options are invalid");
    }
  }
  for (index = 0u; index < entry_count; ++index) {
    et_o2_optimizer_state_handle *handle = state->handles[index];
    if (et_o2_find_handle(handle) != handle || handle->live == 0u ||
        handle->owner != state || handle->index != index / 2u ||
        handle->moment_kind != (uint32_t)(index % 2u)) {
      return et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                        ET_O2_CODE_OWNER_CONFLICT, "optimizer-load-state!",
                        "optimizer state handle metadata is inconsistent");
    }
  }
  for (index = 0u; index < state->count; ++index) {
    if (et_f32_tensor_is_live_v1(state->entries[index].exp_avg) != 1 ||
        et_f32_tensor_is_live_v1(state->entries[index].exp_avg_sq) != 1) {
      return et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                        ET_O2_CODE_OWNER_CONFLICT, "optimizer-load-state!",
                        "optimizer state moment owner is invalid");
    }
    result =
        et_o2_validate_state_moment(state->entries[index].exp_avg, 0, error);
    if (result != 0) {
      return result;
    }
    result =
        et_o2_validate_state_moment(state->entries[index].exp_avg_sq, 1, error);
    if (result != 0) {
      return result;
    }
  }
  for (index = 0u; index < entry_count; ++index) {
    const et_f32_tensor *left = (index % 2u) == 0u
                                    ? state->entries[index / 2u].exp_avg
                                    : state->entries[index / 2u].exp_avg_sq;
    size_t right_index;
    for (right_index = index + 1u; right_index < entry_count; ++right_index) {
      const et_f32_tensor *right =
          (right_index % 2u) == 0u
              ? state->entries[right_index / 2u].exp_avg
              : state->entries[right_index / 2u].exp_avg_sq;
      if (left == right) {
        return et_o2_fail(error, ET_O2_STATUS_CORRUPT_DATA,
                          ET_O2_CODE_OWNER_CONFLICT, "optimizer-load-state!",
                          "optimizer state moments are not independent");
      }
    }
  }
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_global_norm(et_o2_optimizer *optimizer,
                                 uint32_t weight_bits, uint32_t *norm_bits,
                                 et_o2_error_v1 *error) {
  uint32_t scale = 0u;
  uint32_t sumsq = UINT32_C(0x3f800000);
  size_t entry_index;
  for (entry_index = 0u; entry_index < optimizer->count; ++entry_index) {
    et_f32_tensor_borrow *borrow = NULL;
    const et_kernel_tensor_view_v1 *view = NULL;
    et_f32_tensor_error i2_error;
    size_t element;
    memset(&i2_error, 0, sizeof(i2_error));
    if (et_f32_parameter_gradient_borrow_begin_v1(
            optimizer->entries[entry_index].parameter, &borrow, &i2_error) !=
            0 ||
        et_f32_tensor_borrow_view_v1(borrow, &view, &i2_error) != 0) {
      if (borrow != NULL) {
        (void)et_f32_tensor_borrow_end_v1(&borrow, NULL);
      }
      return et_o2_from_i2(&i2_error, error, "optimizer-step!",
                           ET_O2_STATUS_INVALID_STATE);
    }
    for (element = 0u; element < view->byte_length / sizeof(float); ++element) {
      uint32_t bits;
      uint32_t absolute;
      memcpy(&bits, (const unsigned char *)view->data + element * sizeof(bits),
             sizeof(bits));
      bits = et_o2_div(bits, weight_bits);
      if (!et_o2_bits_finite(bits)) {
        (void)et_f32_tensor_borrow_end_v1(&borrow, NULL);
        return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_NONFINITE, "optimizer-step!",
                          "normalized gradient is nonfinite");
      }
      absolute = bits & UINT32_C(0x7fffffff);
      if (absolute != 0u) {
        if (scale == 0u ||
            et_o2_float_from_bits(scale) < et_o2_float_from_bits(absolute)) {
          uint32_t ratio = scale == 0u ? 0u : et_o2_div(scale, absolute);
          sumsq = et_o2_add(UINT32_C(0x3f800000),
                            et_o2_mul(sumsq, et_o2_mul(ratio, ratio)));
          scale = absolute;
        } else {
          uint32_t ratio = et_o2_div(absolute, scale);
          sumsq = et_o2_add(sumsq, et_o2_mul(ratio, ratio));
        }
        if (!et_o2_bits_finite(sumsq)) {
          (void)et_f32_tensor_borrow_end_v1(&borrow, NULL);
          return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                            ET_O2_CODE_NONFINITE, "optimizer-step!",
                            "gradient norm is unrepresentable");
        }
      }
    }
    if (et_f32_tensor_borrow_end_v1(&borrow, NULL) != 0) {
      return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                        ET_O2_CODE_PROVIDER_DEFECT, "optimizer-step!",
                        "gradient borrow could not be closed");
    }
  }
  if (scale == 0u) {
    *norm_bits = 0u;
  } else {
    *norm_bits = et_o2_mul(scale, et_o2_sqrt(sumsq));
    if (!et_o2_bits_finite(*norm_bits)) {
      return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE,
                        "optimizer-step!", "gradient norm is unrepresentable");
    }
  }
  return ET_O2_STATUS_OK;
}

static void et_o2_release_borrow(et_f32_tensor_borrow **borrow) {
  if (*borrow != NULL) {
    (void)et_f32_tensor_borrow_end_v1(borrow, NULL);
  }
}

static int32_t et_o2_stage_entry(et_o2_entry *entry, uint64_t update,
                                 uint32_t schedule_factor, uint32_t weight_bits,
                                 uint32_t clip_factor,
                                 et_f32_tensor **parameter_stage,
                                 et_f32_tensor **exp_avg_stage,
                                 et_f32_tensor **exp_avg_sq_stage,
                                 et_o2_error_v1 *error) {
  const et_f32_tensor *parameter_value = NULL;
  et_f32_tensor_error i2_error;
  et_f32_tensor_borrow *borrows[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  const et_kernel_tensor_view_v1 *views[7] = {NULL, NULL, NULL, NULL,
                                              NULL, NULL, NULL};
  uint32_t beta1_power;
  uint32_t beta2_power;
  uint32_t bias1;
  uint32_t bias2;
  uint32_t learning_rate;
  size_t element_count;
  size_t index;
  int32_t result = ET_O2_STATUS_INTERNAL;
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_parameter_value_tensor_v1(entry->parameter, &parameter_value,
                                       &i2_error) != 0 ||
      et_f32_tensor_clone_v1(parameter_value, parameter_stage, &i2_error) !=
          0 ||
      et_f32_tensor_clone_v1(entry->exp_avg, exp_avg_stage, &i2_error) != 0 ||
      et_f32_tensor_clone_v1(entry->exp_avg_sq, exp_avg_sq_stage, &i2_error) !=
          0) {
    return et_o2_from_i2(&i2_error, error, "optimizer-step!",
                         ET_O2_STATUS_INTERNAL);
  }
  if (et_o2_borrow_tensor((et_f32_tensor *)parameter_value, &borrows[0],
                          &views[0], error, "optimizer-step!") != 0 ||
      et_f32_parameter_gradient_borrow_begin_v1(entry->parameter, &borrows[1],
                                                &i2_error) != 0 ||
      et_f32_tensor_borrow_view_v1(borrows[1], &views[1], &i2_error) != 0 ||
      et_o2_borrow_tensor(entry->exp_avg, &borrows[2], &views[2], error,
                          "optimizer-step!") != 0 ||
      et_o2_borrow_tensor(entry->exp_avg_sq, &borrows[3], &views[3], error,
                          "optimizer-step!") != 0 ||
      et_o2_borrow_tensor(*parameter_stage, &borrows[4], &views[4], error,
                          "optimizer-step!") != 0 ||
      et_o2_borrow_tensor(*exp_avg_stage, &borrows[5], &views[5], error,
                          "optimizer-step!") != 0 ||
      et_o2_borrow_tensor(*exp_avg_sq_stage, &borrows[6], &views[6], error,
                          "optimizer-step!") != 0) {
    if (error != NULL && error->category == 0u) {
      (void)et_o2_from_i2(&i2_error, error, "optimizer-step!",
                          ET_O2_STATUS_INVALID_STATE);
    }
    result = (int32_t)(error != NULL && error->category != 0u
                           ? error->category
                           : ET_O2_STATUS_INVALID_STATE);
    goto cleanup;
  }
  for (index = 1u; index < 7u; ++index) {
    if (views[index]->byte_length != views[0]->byte_length) {
      result =
          et_o2_fail(error, ET_O2_STATUS_SHAPE_MISMATCH,
                     ET_O2_CODE_INVALID_HANDLE, "optimizer-step!",
                     "parameter, gradient, moment, or stage shape differs");
      goto cleanup;
    }
  }
  beta1_power = et_o2_pow_u64(entry->options.beta1_bits, update);
  beta2_power = et_o2_pow_u64(entry->options.beta2_bits, update);
  bias1 = et_o2_sub(UINT32_C(0x3f800000), beta1_power);
  bias2 = et_o2_sub(UINT32_C(0x3f800000), beta2_power);
  learning_rate = et_o2_mul(entry->options.learning_rate_bits, schedule_factor);
  if (!et_o2_bits_positive_finite(bias1) ||
      !et_o2_bits_positive_finite(bias2) ||
      !et_o2_bits_nonnegative_finite(learning_rate)) {
    result =
        et_o2_fail(error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE,
                   "optimizer-step!",
                   "bias correction or effective learning rate is invalid");
    goto cleanup;
  }
  element_count = views[0]->byte_length / sizeof(uint32_t);
  for (index = 0u; index < element_count; ++index) {
    uint32_t parameter_bits;
    uint32_t gradient_bits;
    uint32_t exp_avg_bits;
    uint32_t exp_avg_sq_bits;
    uint32_t gradient;
    uint32_t next_avg;
    uint32_t next_avg_sq;
    uint32_t avg_hat;
    uint32_t avg_sq_hat;
    uint32_t denominator;
    uint32_t decay_factor;
    uint32_t next_parameter;
    memcpy(&parameter_bits,
           (const unsigned char *)views[0]->data + index * sizeof(uint32_t),
           sizeof(parameter_bits));
    memcpy(&gradient_bits,
           (const unsigned char *)views[1]->data + index * sizeof(uint32_t),
           sizeof(gradient_bits));
    memcpy(&exp_avg_bits,
           (const unsigned char *)views[2]->data + index * sizeof(uint32_t),
           sizeof(exp_avg_bits));
    memcpy(&exp_avg_sq_bits,
           (const unsigned char *)views[3]->data + index * sizeof(uint32_t),
           sizeof(exp_avg_sq_bits));
    if (!et_o2_bits_finite(parameter_bits) ||
        !et_o2_bits_finite(gradient_bits) || !et_o2_bits_finite(exp_avg_bits) ||
        !et_o2_bits_finite(exp_avg_sq_bits)) {
      result = et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_NONFINITE, "optimizer-step!",
                          "parameter, gradient, or moment is nonfinite");
      goto cleanup;
    }
    gradient = et_o2_mul(et_o2_div(gradient_bits, weight_bits), clip_factor);
    next_avg = et_o2_add(
        et_o2_mul(entry->options.beta1_bits, exp_avg_bits),
        et_o2_mul(et_o2_sub(UINT32_C(0x3f800000), entry->options.beta1_bits),
                  gradient));
    next_avg_sq = et_o2_add(
        et_o2_mul(entry->options.beta2_bits, exp_avg_sq_bits),
        et_o2_mul(et_o2_sub(UINT32_C(0x3f800000), entry->options.beta2_bits),
                  et_o2_mul(gradient, gradient)));
    avg_hat = et_o2_div(next_avg, bias1);
    avg_sq_hat = et_o2_div(next_avg_sq, bias2);
    denominator =
        et_o2_add(et_o2_sqrt(avg_sq_hat), entry->options.epsilon_bits);
    decay_factor =
        et_o2_sub(UINT32_C(0x3f800000),
                  et_o2_mul(learning_rate, entry->options.weight_decay_bits));
    next_parameter =
        et_o2_sub(et_o2_mul(parameter_bits, decay_factor),
                  et_o2_div(et_o2_mul(learning_rate, avg_hat), denominator));
    if (!et_o2_bits_finite(gradient) || !et_o2_bits_finite(next_avg) ||
        !et_o2_bits_finite(next_avg_sq) || !et_o2_bits_finite(avg_hat) ||
        !et_o2_bits_nonnegative_finite(avg_sq_hat) ||
        !et_o2_bits_positive_finite(denominator) ||
        !et_o2_bits_finite(decay_factor) ||
        !et_o2_bits_finite(next_parameter)) {
      result =
          et_o2_fail(error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE,
                     "optimizer-step!", "AdamW produced a nonfinite value");
      goto cleanup;
    }
    memcpy((unsigned char *)views[4]->data + index * sizeof(uint32_t),
           &next_parameter, sizeof(next_parameter));
    memcpy((unsigned char *)views[5]->data + index * sizeof(uint32_t),
           &next_avg, sizeof(next_avg));
    memcpy((unsigned char *)views[6]->data + index * sizeof(uint32_t),
           &next_avg_sq, sizeof(next_avg_sq));
  }
  result = ET_O2_STATUS_OK;

cleanup:
  for (index = 7u; index > 0u; --index) {
    et_o2_release_borrow(&borrows[index - 1u]);
  }
  return result;
}

int32_t et_o2_optimizer_step_v1(et_o2_optimizer *candidate,
                                et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  et_f32_tensor **stages = NULL;
  et_f32_tensor_copy_assignment_v1 *assignments = NULL;
  et_f32_tensor_copy_plan *plan = NULL;
  et_f32_tensor_error i2_error;
  uint64_t expected_count = 0u;
  uint32_t expected_weight = 0u;
  uint32_t norm = 0u;
  uint32_t clip_factor = UINT32_C(0x3f800000);
  uint32_t schedule_factor = 0u;
  uint64_t update;
  size_t index;
  int32_t result;
  if ((result = et_o2_require_optimizer(candidate, "optimizer-step!", error)) !=
      0) {
    return result;
  }
  if (optimizer->completed_updates >= (uint64_t)INT64_MAX) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_COUNTER_OVERFLOW, "optimizer-step!",
                      "completed-update counter would overflow");
  }
  if (!et_o2_float_environment()) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_FLOAT_ENVIRONMENT, "optimizer-step!",
                      "binary32 environment is unsupported");
  }
  optimizer->busy = 1u;
  for (index = 0u; index < optimizer->count; ++index) {
    et_f32_gradient_metadata_v1 metadata;
    int32_t finite = 0;
    memset(&metadata, 0, sizeof(metadata));
    metadata.struct_size = sizeof(metadata);
    memset(&i2_error, 0, sizeof(i2_error));
    if (et_f32_parameter_validate_identity_v1(
            optimizer->entries[index].parameter,
            optimizer->entries[index].p1_handle, &i2_error) != 0 ||
        et_f32_parameter_gradient_metadata_v1(
            optimizer->entries[index].parameter, &metadata, &i2_error) != 0) {
      result = et_o2_from_i2(&i2_error, error, "optimizer-step!",
                             ET_O2_STATUS_INVALID_STATE);
      goto cleanup;
    }
    if (metadata.state != ET_F32_GRADIENT_PRESENT) {
      result = et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_GRADIENT_ABSENT, "optimizer-step!",
                          "every bound parameter requires a present gradient");
      goto cleanup;
    }
    if (metadata.contribution_count == 0u ||
        metadata.contribution_count > (uint64_t)INT64_MAX ||
        !et_o2_bits_positive_finite(metadata.normalization_weight_bits) ||
        (index != 0u &&
         (metadata.contribution_count != expected_count ||
          metadata.normalization_weight_bits != expected_weight))) {
      result = et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_GRADIENT_METADATA, "optimizer-step!",
                          "gradient accumulation metadata is inconsistent");
      goto cleanup;
    }
    if (et_f32_parameter_gradient_finite_v1(optimizer->entries[index].parameter,
                                            &finite, &i2_error) != 0 ||
        finite != 1) {
      result =
          et_o2_fail(error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE,
                     "optimizer-step!", "gradient contains a nonfinite value");
      goto cleanup;
    }
    expected_count = metadata.contribution_count;
    expected_weight = metadata.normalization_weight_bits;
  }
  update = optimizer->completed_updates + 1u;
  if ((result = et_o2_schedule_factor(&optimizer->config, update,
                                      &schedule_factor, error)) != 0 ||
      (result = et_o2_global_norm(optimizer, expected_weight, &norm, error)) !=
          0) {
    goto cleanup;
  }
  if (optimizer->config.clip_kind == ET_O2_CLIP_GLOBAL_L2 &&
      et_o2_float_from_bits(norm) >
          et_o2_float_from_bits(optimizer->config.clip_max_bits)) {
    clip_factor = et_o2_div(optimizer->config.clip_max_bits, norm);
    if (!et_o2_bits_positive_finite(clip_factor)) {
      result =
          et_o2_fail(error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE,
                     "optimizer-step!", "gradient clip factor is invalid");
      goto cleanup;
    }
  }
  stages =
      (et_f32_tensor **)et_o2_calloc(optimizer->count * 3u, sizeof(*stages));
  assignments = (et_f32_tensor_copy_assignment_v1 *)et_o2_calloc(
      optimizer->count * 3u, sizeof(*assignments));
  if (stages == NULL || assignments == NULL) {
    result =
        et_o2_fail(error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_ALLOCATION_FAILED,
                   "optimizer-step!", "cannot allocate AdamW staging batch");
    goto cleanup;
  }
  for (index = 0u; index < optimizer->count; ++index) {
    const et_f32_tensor *parameter_value = NULL;
    memset(&i2_error, 0, sizeof(i2_error));
    if ((result = et_o2_stage_entry(
             &optimizer->entries[index], update, schedule_factor,
             expected_weight, clip_factor, &stages[index * 3u],
             &stages[index * 3u + 1u], &stages[index * 3u + 2u], error)) != 0 ||
        et_f32_parameter_value_tensor_v1(optimizer->entries[index].parameter,
                                         &parameter_value, &i2_error) != 0) {
      if (result == 0) {
        result = et_o2_from_i2(&i2_error, error, "optimizer-step!",
                               ET_O2_STATUS_INVALID_STATE);
      }
      goto cleanup;
    }
    assignments[index * 3u].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    assignments[index * 3u].destination = (et_f32_tensor *)parameter_value;
    assignments[index * 3u].source = stages[index * 3u];
    assignments[index * 3u + 1u].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    assignments[index * 3u + 1u].destination =
        optimizer->entries[index].exp_avg;
    assignments[index * 3u + 1u].source = stages[index * 3u + 1u];
    assignments[index * 3u + 2u].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    assignments[index * 3u + 2u].destination =
        optimizer->entries[index].exp_avg_sq;
    assignments[index * 3u + 2u].source = stages[index * 3u + 2u];
  }
  if (et_f32_tensor_copy_plan_prepare_v1(optimizer->count * 3u, assignments,
                                         &plan, &i2_error) != 0) {
    result = et_o2_from_i2(&i2_error, error, "optimizer-step!",
                           ET_O2_STATUS_INVALID_STATE);
    goto cleanup;
  }
  if (et_f32_tensor_copy_plan_commit_v1(plan, &i2_error) != 0) {
    result = et_o2_from_i2(&i2_error, error, "optimizer-step!",
                           ET_O2_STATUS_INTERNAL);
    goto cleanup;
  }
  optimizer->completed_updates = update;
  result = et_o2_success(error);

cleanup:
  if (plan != NULL) {
    (void)et_f32_tensor_copy_plan_release_v1(&plan, NULL);
  }
  if (stages != NULL) {
    for (index = 0u; index < optimizer->count * 3u; ++index) {
      if (stages[index] != NULL) {
        (void)et_f32_tensor_destroy_v1(&stages[index], NULL);
      }
    }
  }
  free(assignments);
  free(stages);
  optimizer->busy = 0u;
  return result;
}

int32_t et_o2_optimizer_zero_grad_v1(et_o2_optimizer *candidate,
                                     et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  et_f32_parameter **parameters = NULL;
  et_f32_gradient_reset_plan *plan = NULL;
  et_f32_tensor_error i2_error;
  size_t index;
  int32_t result;
  if ((result = et_o2_require_optimizer(candidate, "optimizer-zero-grad!",
                                        error)) != 0) {
    return result;
  }
  optimizer->busy = 1u;
  parameters =
      (et_f32_parameter **)et_o2_calloc(optimizer->count, sizeof(*parameters));
  if (parameters == NULL) {
    result = et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                        ET_O2_CODE_ALLOCATION_FAILED, "optimizer-zero-grad!",
                        "cannot allocate gradient reset batch");
    goto cleanup;
  }
  for (index = 0u; index < optimizer->count; ++index) {
    memset(&i2_error, 0, sizeof(i2_error));
    if (et_f32_parameter_validate_identity_v1(
            optimizer->entries[index].parameter,
            optimizer->entries[index].p1_handle, &i2_error) != 0) {
      result = et_o2_from_i2(&i2_error, error, "optimizer-zero-grad!",
                             ET_O2_STATUS_INVALID_STATE);
      goto cleanup;
    }
    parameters[index] = optimizer->entries[index].parameter;
  }
  if (et_f32_gradient_reset_plan_prepare_v1(optimizer->count, parameters, &plan,
                                            &i2_error) != 0 ||
      et_f32_gradient_reset_plan_commit_v1(plan, &i2_error) != 0) {
    result = et_o2_from_i2(&i2_error, error, "optimizer-zero-grad!",
                           ET_O2_STATUS_INVALID_STATE);
    goto cleanup;
  }
  result = et_o2_success(error);

cleanup:
  if (plan != NULL) {
    (void)et_f32_gradient_reset_plan_release_v1(&plan, NULL);
  }
  free(parameters);
  optimizer->busy = 0u;
  return result;
}

int32_t et_o2_optimizer_completed_updates_v1(const et_o2_optimizer *candidate,
                                             uint64_t *completed_updates,
                                             et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  int32_t result;
  if (completed_updates == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-state",
                      "counter output is null");
  }
  if ((result = et_o2_require_optimizer(candidate, "optimizer-state", error)) !=
      0) {
    return result;
  }
  *completed_updates = optimizer->completed_updates;
  return et_o2_success(error);
}

int32_t
et_o2_optimizer_schedule_factor_bits_v1(const et_o2_optimizer *candidate,
                                        uint32_t *factor_bits,
                                        et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  uint64_t update;
  int32_t result;
  if (factor_bits == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-state",
                      "schedule output is null");
  }
  if ((result = et_o2_require_optimizer(candidate, "optimizer-state", error)) !=
      0) {
    return result;
  }
  update = optimizer->completed_updates == (uint64_t)INT64_MAX
               ? optimizer->completed_updates
               : optimizer->completed_updates + 1u;
  if (et_o2_schedule_factor(&optimizer->config, update, factor_bits, error) !=
      0) {
    return (int32_t)(error != NULL ? error->category
                                   : ET_O2_STATUS_INVALID_STATE);
  }
  return et_o2_success(error);
}

static void et_o2_release_owned_state_prefix(et_o2_optimizer_state *state,
                                             size_t entry_count) {
  size_t index;
  for (index = 0u; index < entry_count; ++index) {
    if (state->entries[index].exp_avg_sq != NULL) {
      (void)et_f32_owned_tensor_release_v1(state->entries[index].exp_avg_sq,
                                           NULL);
      state->entries[index].exp_avg_sq = NULL;
      et_o2_owned_state_clones--;
    }
    if (state->entries[index].exp_avg != NULL) {
      (void)et_f32_owned_tensor_release_v1(state->entries[index].exp_avg, NULL);
      state->entries[index].exp_avg = NULL;
      et_o2_owned_state_clones--;
    }
  }
}

int32_t et_o2_optimizer_state_snapshot_v1(et_o2_optimizer *candidate,
                                          et_o2_optimizer_state **output,
                                          et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  et_o2_optimizer_state *state = NULL;
  size_t index;
  size_t handle_index;
  int32_t result;
  et_f32_tensor_error i2_error;
  if (output == NULL || *output != NULL ||
      (result = et_o2_require_optimizer(candidate, "optimizer-state", error)) !=
          0) {
    if (output == NULL || (output != NULL && *output != NULL)) {
      return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                        ET_O2_CODE_NULL_ARGUMENT, "optimizer-state",
                        "state output must be a nonnull empty slot");
    }
    return result;
  }
  optimizer->busy = 1u;
  if ((result = et_o2_validate_absent_gradients(optimizer, "optimizer-state",
                                                error)) != 0) {
    goto cleanup;
  }
  state = (et_o2_optimizer_state *)et_o2_calloc(1u, sizeof(*state));
  if (state != NULL) {
    state->entries = (et_o2_state_entry *)et_o2_calloc(optimizer->count,
                                                       sizeof(*state->entries));
    state->handles = (et_o2_optimizer_state_handle **)et_o2_calloc(
        optimizer->count * 2u, sizeof(*state->handles));
  }
  if (state == NULL || state->entries == NULL || state->handles == NULL) {
    result =
        et_o2_fail(error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_ALLOCATION_FAILED,
                   "optimizer-state", "cannot allocate optimizer state owner");
    goto cleanup;
  }
  state->count = optimizer->count;
  state->config = optimizer->config;
  state->completed_updates = optimizer->completed_updates;
  state->provider_major = ET_O2_PROVIDER_ABI_MAJOR;
  state->provider_minor = ET_O2_PROVIDER_ABI_MINOR;
  memcpy(state->provider_id, ET_O2_PROVIDER_ID, sizeof(ET_O2_PROVIDER_ID));
  for (index = 0u; index < optimizer->count; ++index) {
    state->entries[index].options = optimizer->entries[index].options;
    memset(&i2_error, 0, sizeof(i2_error));
    if (et_f32_owned_tensor_clone_v1(optimizer->entries[index].exp_avg,
                                     &state->entries[index].exp_avg,
                                     &i2_error) != 0) {
      result = et_o2_from_i2(&i2_error, error, "optimizer-state",
                             ET_O2_STATUS_INTERNAL);
      goto cleanup;
    }
    state->owned_clone_count++;
    et_o2_owned_state_clones++;
    if (et_f32_owned_tensor_clone_v1(optimizer->entries[index].exp_avg_sq,
                                     &state->entries[index].exp_avg_sq,
                                     &i2_error) != 0) {
      result = et_o2_from_i2(&i2_error, error, "optimizer-state",
                             ET_O2_STATUS_INTERNAL);
      goto cleanup;
    }
    state->owned_clone_count++;
    et_o2_owned_state_clones++;
  }
  for (handle_index = 0u; handle_index < optimizer->count * 2u;
       ++handle_index) {
    et_o2_optimizer_state_handle *handle =
        (et_o2_optimizer_state_handle *)et_o2_calloc(1u, sizeof(*handle));
    if (handle == NULL) {
      result = et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                          ET_O2_CODE_ALLOCATION_FAILED, "optimizer-state",
                          "cannot allocate state-backed moment handle");
      goto cleanup;
    }
    handle->magic = ET_O2_HANDLE_MAGIC;
    handle->owner = state;
    handle->index = handle_index / 2u;
    handle->moment_kind = (uint32_t)(handle_index % 2u);
    handle->live = 1u;
    state->handles[handle_index] = handle;
  }
  state->magic = ET_O2_STATE_MAGIC;
  state->lifecycle = ET_O2_OPTIMIZER_STATE_LIVE;
  state->registry_next = et_o2_states;
  et_o2_states = state;
  for (handle_index = 0u; handle_index < optimizer->count * 2u;
       ++handle_index) {
    state->handles[handle_index]->registry_next = et_o2_handles;
    et_o2_handles = state->handles[handle_index];
  }
  *output = state;
  state = NULL;
  result = et_o2_success(error);

cleanup:
  if (state != NULL) {
    if (state->handles != NULL) {
      for (handle_index = 0u; handle_index < optimizer->count * 2u;
           ++handle_index) {
        free(state->handles[handle_index]);
      }
    }
    if (state->entries != NULL) {
      et_o2_release_owned_state_prefix(state, optimizer->count);
    }
    free(state->handles);
    free(state->entries);
    free(state);
  }
  optimizer->busy = 0u;
  return result;
}

int32_t et_o2_optimizer_load_state_v1(et_o2_optimizer *optimizer_candidate,
                                      et_o2_optimizer_state *state_candidate,
                                      et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(optimizer_candidate);
  et_o2_optimizer_state *state = et_o2_find_state(state_candidate);
  et_f32_tensor_copy_assignment_v1 *assignments = NULL;
  et_f32_tensor_copy_plan *plan = NULL;
  et_f32_tensor_error i2_error;
  size_t index;
  int32_t result;
  if ((result = et_o2_require_optimizer(optimizer_candidate,
                                        "optimizer-load-state!", error)) != 0 ||
      (result = et_o2_require_state_live(
           state_candidate, "optimizer-load-state!", error)) != 0) {
    return result;
  }
  if (state->active_borrows != 0u) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_ACTIVE_BORROW, "optimizer-load-state!",
                      "optimizer state has an active borrow");
  }
  optimizer->busy = 1u;
  state->active_borrows++;
  if ((result = et_o2_validate_state_for_load(state, error)) != 0) {
    goto cleanup;
  }
  if (state->count != optimizer->count) {
    result = et_o2_fail(error, ET_O2_STATUS_SHAPE_MISMATCH,
                        ET_O2_CODE_INVALID_HANDLE, "optimizer-load-state!",
                        "optimizer state parameter count differs");
    goto cleanup;
  }
  if ((result = et_o2_validate_absent_gradients(
           optimizer, "optimizer-load-state!", error)) != 0) {
    goto cleanup;
  }
  assignments = (et_f32_tensor_copy_assignment_v1 *)et_o2_calloc(
      optimizer->count * 2u, sizeof(*assignments));
  if (assignments == NULL) {
    result = et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                        ET_O2_CODE_ALLOCATION_FAILED, "optimizer-load-state!",
                        "cannot allocate optimizer load batch");
    goto cleanup;
  }
  for (index = 0u; index < optimizer->count; ++index) {
    assignments[index * 2u].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    assignments[index * 2u].destination = optimizer->entries[index].exp_avg;
    assignments[index * 2u].source = state->entries[index].exp_avg;
    assignments[index * 2u + 1u].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    assignments[index * 2u + 1u].destination =
        optimizer->entries[index].exp_avg_sq;
    assignments[index * 2u + 1u].source = state->entries[index].exp_avg_sq;
  }
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_tensor_copy_plan_prepare_v1(optimizer->count * 2u, assignments,
                                         &plan, &i2_error) != 0) {
    result = et_o2_from_i2(&i2_error, error, "optimizer-load-state!",
                           ET_O2_STATUS_SHAPE_MISMATCH);
    goto cleanup;
  }
  if (et_f32_tensor_copy_plan_commit_v1(plan, &i2_error) != 0) {
    result = et_o2_from_i2(&i2_error, error, "optimizer-load-state!",
                           ET_O2_STATUS_INTERNAL);
    goto cleanup;
  }
  optimizer->config = state->config;
  optimizer->completed_updates = state->completed_updates;
  for (index = 0u; index < optimizer->count; ++index) {
    optimizer->entries[index].options = state->entries[index].options;
  }
  result = et_o2_success(error);

cleanup:
  if (plan != NULL) {
    (void)et_f32_tensor_copy_plan_release_v1(&plan, NULL);
  }
  free(assignments);
  state->active_borrows--;
  optimizer->busy = 0u;
  return result;
}

int32_t
et_o2_optimizer_state_entry_count_v1(const et_o2_optimizer_state *candidate,
                                     size_t *entry_count,
                                     et_o2_error_v1 *error) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  int32_t result;
  if ((result = et_o2_require_state_live(candidate, "optimizer-state",
                                         error)) != 0) {
    return result;
  }
  if (entry_count == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-state",
                      "state entry-count output is null");
  }
  *entry_count = state->count;
  return et_o2_success(error);
}

int32_t et_o2_optimizer_state_completed_updates_v1(
    const et_o2_optimizer_state *candidate, uint64_t *completed_updates,
    et_o2_error_v1 *error) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  int32_t result;
  if ((result = et_o2_require_state_live(candidate, "optimizer-state",
                                         error)) != 0) {
    return result;
  }
  if (completed_updates == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-state",
                      "state counter output is null");
  }
  *completed_updates = state->completed_updates;
  return et_o2_success(error);
}

int32_t
et_o2_optimizer_state_option_bits_v1(const et_o2_optimizer_state *candidate,
                                     size_t index, uint32_t option,
                                     uint32_t *bits, et_o2_error_v1 *error) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  int32_t result;
  if ((result = et_o2_require_state_live(candidate, "optimizer-state",
                                         error)) != 0) {
    return result;
  }
  if (bits == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-state",
                      "state option output is null");
  }
  if (index >= state->count || option > ET_O2_OPTION_WEIGHT_DECAY) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_OPTION, "optimizer-state",
                      "state option index is invalid");
  }
  if (option == ET_O2_OPTION_LEARNING_RATE) {
    *bits = state->entries[index].options.learning_rate_bits;
  } else if (option == ET_O2_OPTION_BETA1) {
    *bits = state->entries[index].options.beta1_bits;
  } else if (option == ET_O2_OPTION_BETA2) {
    *bits = state->entries[index].options.beta2_bits;
  } else if (option == ET_O2_OPTION_EPSILON) {
    *bits = state->entries[index].options.epsilon_bits;
  } else {
    *bits = state->entries[index].options.weight_decay_bits;
  }
  return et_o2_success(error);
}

int32_t et_o2_optimizer_state_moment_handle_v1(
    et_o2_optimizer_state *candidate, size_t index, uint32_t moment_kind,
    et_o2_optimizer_state_handle **output, et_o2_error_v1 *error) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  int32_t result;
  if ((result = et_o2_require_state_live(candidate, "optimizer-state",
                                         error)) != 0) {
    return result;
  }
  if (output == NULL || *output != NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-state",
                      "moment handle output must be a nonnull empty slot");
  }
  if (index >= state->count || moment_kind > ET_O2_MOMENT_EXP_AVG_SQ) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_OPTION, "optimizer-state",
                      "moment handle selector is invalid");
  }
  *output = state->handles[index * 2u + moment_kind];
  return et_o2_success(error);
}

int32_t et_o2_optimizer_state_borrow_begin_v1(
    et_o2_optimizer_state *state_candidate,
    et_o2_optimizer_state_handle *handle_candidate,
    et_o2_optimizer_state_borrow **output,
    const et_kernel_tensor_view_v1 **view, et_o2_error_v1 *error) {
  et_o2_optimizer_state *state = et_o2_find_state(state_candidate);
  et_o2_optimizer_state_handle *handle = et_o2_find_handle(handle_candidate);
  et_o2_optimizer_state_borrow *borrow;
  et_f32_tensor *tensor;
  et_f32_tensor_error i2_error;
  int32_t result;
  if ((result = et_o2_require_state_live(
           state_candidate, "optimizer-state-borrow", error)) != 0) {
    return result;
  }
  if (output == NULL || view == NULL || *output != NULL || *view != NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-state-borrow",
                      "borrow outputs must be nonnull empty slots");
  }
  if (handle == NULL || handle->live == 0u || handle->owner != state ||
      handle->index >= state->count ||
      handle->moment_kind > ET_O2_MOMENT_EXP_AVG_SQ) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, "optimizer-state-borrow",
                      "state-backed moment handle does not match its owner");
  }
  borrow = (et_o2_optimizer_state_borrow *)et_o2_calloc(1u, sizeof(*borrow));
  if (borrow == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_ALLOCATION_FAILED, "optimizer-state-borrow",
                      "cannot allocate state borrow shell");
  }
  tensor = handle->moment_kind == ET_O2_MOMENT_EXP_AVG
               ? state->entries[handle->index].exp_avg
               : state->entries[handle->index].exp_avg_sq;
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_tensor_borrow_begin_v1(tensor, &borrow->i2_borrow, &i2_error) !=
          0 ||
      et_f32_tensor_borrow_view_v1(borrow->i2_borrow, view, &i2_error) != 0) {
    if (borrow->i2_borrow != NULL) {
      (void)et_f32_tensor_borrow_end_v1(&borrow->i2_borrow, NULL);
    }
    free(borrow);
    return et_o2_from_i2(&i2_error, error, "optimizer-state-borrow",
                         ET_O2_STATUS_INVALID_STATE);
  }
  borrow->magic = ET_O2_BORROW_MAGIC;
  borrow->owner = state;
  borrow->handle = handle;
  borrow->registry_next = et_o2_borrows;
  et_o2_borrows = borrow;
  state->active_borrows++;
  *output = borrow;
  return et_o2_success(error);
}

int32_t et_o2_optimizer_state_borrow_end_v1(et_o2_optimizer_state_borrow **slot,
                                            et_o2_error_v1 *error) {
  et_o2_optimizer_state_borrow *borrow;
  et_f32_tensor_error i2_error;
  if (slot == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, "optimizer-state-borrow",
                      "borrow slot is null");
  }
  if (*slot == NULL) {
    return et_o2_success(error);
  }
  borrow = et_o2_find_borrow(*slot);
  if (borrow == NULL || borrow->owner == NULL ||
      borrow->owner->active_borrows == 0u) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, "optimizer-state-borrow",
                      "borrow is foreign or stale");
  }
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_tensor_borrow_end_v1(&borrow->i2_borrow, &i2_error) != 0) {
    return et_o2_from_i2(&i2_error, error, "optimizer-state-borrow",
                         ET_O2_STATUS_INTERNAL);
  }
  borrow->owner->active_borrows--;
  et_o2_unlink_borrow(borrow);
  borrow->magic = 0u;
  borrow->owner = NULL;
  borrow->handle = NULL;
  borrow->registry_next = et_o2_retired_borrows;
  et_o2_retired_borrows = borrow;
  *slot = NULL;
  return et_o2_success(error);
}

int32_t
et_o2_optimizer_state_lifecycle_v1(const et_o2_optimizer_state *candidate,
                                   uint32_t *lifecycle, et_o2_error_v1 *error) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  if (state == NULL || lifecycle == NULL) {
    return et_o2_fail(
        error, ET_O2_STATUS_INVALID_ARGUMENT,
        state == NULL ? ET_O2_CODE_INVALID_HANDLE : ET_O2_CODE_NULL_ARGUMENT,
        "optimizer-state-release!", "state lifecycle operands are invalid");
  }
  if (state->lifecycle != ET_O2_OPTIMIZER_STATE_LIVE &&
      state->lifecycle != ET_O2_OPTIMIZER_STATE_RELEASING &&
      state->lifecycle != ET_O2_OPTIMIZER_STATE_DEAD) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT,
                      "optimizer-state-release!",
                      "state lifecycle invariant is broken");
  }
  *lifecycle = state->lifecycle;
  return et_o2_success(error);
}

int32_t et_o2_optimizer_state_release_v1(et_o2_optimizer_state *candidate,
                                         et_o2_error_v1 *error) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  size_t entry_count;
  size_t index;
  int provider_defect = 0;
  if (state == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, "optimizer-state-release!",
                      "optimizer state is malformed or unregistered");
  }
  if (state->lifecycle == ET_O2_OPTIMIZER_STATE_DEAD) {
    return et_o2_success(error);
  }
  if (state->lifecycle != ET_O2_OPTIMIZER_STATE_LIVE ||
      state->active_borrows != 0u) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_ACTIVE_BORROW, "optimizer-state-release!",
                      "optimizer state is busy or releasing");
  }
  if (!et_o2_provider_valid(state)) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_PROVIDER_MISMATCH, "optimizer-state-release!",
                      "optimizer state provider invariant is broken");
  }
  if (state->count == 0u || state->count > ET_O2_MAX_PARAMETERS ||
      state->entries == NULL || state->handles == NULL ||
      state->owned_clone_count != (uint64_t)(state->count * 2u) ||
      state->owned_clone_count > (uint64_t)et_o2_owned_state_clones) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_OWNER_CONFLICT, "optimizer-state-release!",
                      "optimizer state ownership ledger is invalid");
  }
  entry_count = state->count * 2u;
  for (index = 0u; index < entry_count; ++index) {
    et_o2_optimizer_state_handle *handle = state->handles[index];
    if (et_o2_find_handle(handle) != handle || handle->live == 0u ||
        handle->owner != state || handle->index != index / 2u ||
        handle->moment_kind != (uint32_t)(index % 2u)) {
      return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                        ET_O2_CODE_OWNER_CONFLICT, "optimizer-state-release!",
                        "state-backed handle ledger is invalid");
    }
  }
  for (index = 0u; index < state->count; ++index) {
    if (et_f32_tensor_is_live_v1(state->entries[index].exp_avg) != 1 ||
        et_f32_tensor_is_live_v1(state->entries[index].exp_avg_sq) != 1 ||
        state->entries[index].exp_avg == state->entries[index].exp_avg_sq) {
      return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                        ET_O2_CODE_OWNER_CONFLICT, "optimizer-state-release!",
                        "owned moment ledger is invalid");
    }
  }
  for (index = 0u; index < entry_count; ++index) {
    const et_f32_tensor *left = (index % 2u) == 0u
                                    ? state->entries[index / 2u].exp_avg
                                    : state->entries[index / 2u].exp_avg_sq;
    size_t right_index;
    for (right_index = index + 1u; right_index < entry_count; ++right_index) {
      const et_f32_tensor *right =
          (right_index % 2u) == 0u
              ? state->entries[right_index / 2u].exp_avg
              : state->entries[right_index / 2u].exp_avg_sq;
      if (left == right) {
        return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_OWNER_CONFLICT, "optimizer-state-release!",
                          "owned moment ledger contains a duplicate owner");
      }
    }
  }
  state->lifecycle = ET_O2_OPTIMIZER_STATE_RELEASING;
  for (index = 0u; index < entry_count; ++index) {
    state->handles[index]->live = 0u;
  }
  for (index = 0u; index < state->count; ++index) {
    et_f32_tensor *owned[2] = {state->entries[index].exp_avg,
                               state->entries[index].exp_avg_sq};
    size_t moment;
    for (moment = 0u; moment < 2u; ++moment) {
      int32_t status = et_f32_owned_tensor_release_v1(owned[moment], NULL);
#ifdef ET_O2_TESTING
      if (et_o2_successful_releases >= et_o2_release_limit) {
        provider_defect = 1;
      } else {
        et_o2_successful_releases++;
      }
#endif
      if (status != 0) {
        provider_defect = 1;
      }
      et_o2_owned_state_clones--;
      state->owned_clone_count--;
    }
    state->entries[index].exp_avg = NULL;
    state->entries[index].exp_avg_sq = NULL;
  }
  for (index = 0u; index < entry_count; ++index) {
    state->handles[index]->owner = NULL;
  }
  free(state->handles);
  free(state->entries);
  state->handles = NULL;
  state->entries = NULL;
  state->count = 0u;
  state->active_borrows = 0u;
  memset(&state->config, 0, sizeof(state->config));
  state->completed_updates = 0u;
  state->provider_major = 0u;
  state->provider_minor = 0u;
  memset(state->provider_id, 0, sizeof(state->provider_id));
  state->lifecycle = ET_O2_OPTIMIZER_STATE_DEAD;
  if (provider_defect != 0) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT,
                      "optimizer-state-release!",
                      "owned moment release violated its admitted contract");
  }
  return et_o2_success(error);
}

#ifdef ET_O2_TESTING
void et_o2_test_fail_alloc_after_v1(size_t allowed) {
  et_o2_allocation_limit = allowed;
  et_o2_successful_allocations = 0u;
}

void et_o2_test_fail_release_after_v1(size_t allowed) {
  et_o2_release_limit = allowed;
  et_o2_successful_releases = 0u;
}

void et_o2_test_reset_failpoints_v1(void) {
  et_o2_allocation_limit = SIZE_MAX;
  et_o2_successful_allocations = 0u;
  et_o2_release_limit = SIZE_MAX;
  et_o2_successful_releases = 0u;
}

void et_o2_test_live_counts_snapshot_v1(et_o2_test_live_counts_v1 *counts) {
  et_o2_optimizer_builder *builder;
  et_o2_optimizer *optimizer;
  et_o2_optimizer_state *state;
  et_o2_optimizer_state_handle *handle;
  et_o2_optimizer_state_borrow *borrow;
  if (counts == NULL || counts->struct_size != sizeof(*counts)) {
    return;
  }
  memset((unsigned char *)counts + sizeof(counts->struct_size), 0,
         sizeof(*counts) - sizeof(counts->struct_size));
  for (builder = et_o2_builders; builder != NULL;
       builder = builder->registry_next) {
    counts->builders++;
  }
  for (optimizer = et_o2_optimizers; optimizer != NULL;
       optimizer = optimizer->registry_next) {
    counts->optimizers++;
  }
  for (state = et_o2_states; state != NULL; state = state->registry_next) {
    if (state->lifecycle == ET_O2_OPTIMIZER_STATE_DEAD) {
      counts->dead_states++;
    } else {
      counts->live_states++;
    }
  }
  for (handle = et_o2_handles; handle != NULL; handle = handle->registry_next) {
    if (handle->live != 0u) {
      counts->live_state_handles++;
    } else {
      counts->dead_state_handles++;
    }
  }
  for (borrow = et_o2_borrows; borrow != NULL; borrow = borrow->registry_next) {
    counts->state_borrows++;
  }
  counts->owned_state_clones = et_o2_owned_state_clones;
}

int32_t
et_o2_test_optimizer_set_completed_updates_v1(et_o2_optimizer *candidate,
                                              uint64_t completed_updates) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  if (optimizer == NULL || optimizer->busy != 0u) {
    return -1;
  }
  optimizer->completed_updates = completed_updates;
  return 0;
}

int32_t
et_o2_test_state_set_provider_version_v1(et_o2_optimizer_state *candidate,
                                         uint32_t major, uint32_t minor) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  if (state == NULL || state->lifecycle != ET_O2_OPTIMIZER_STATE_LIVE ||
      state->active_borrows != 0u) {
    return -1;
  }
  state->provider_major = major;
  state->provider_minor = minor;
  return 0;
}

int32_t
et_o2_test_state_set_owned_clone_count_v1(et_o2_optimizer_state *candidate,
                                          uint64_t owned_clone_count) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  if (state == NULL || state->lifecycle != ET_O2_OPTIMIZER_STATE_LIVE ||
      state->active_borrows != 0u) {
    return -1;
  }
  state->owned_clone_count = owned_clone_count;
  return 0;
}

int32_t et_o2_test_state_set_config_v1(
    et_o2_optimizer_state *candidate, uint32_t clip_kind,
    uint32_t clip_max_bits, uint32_t schedule_kind, uint64_t warmup_updates,
    uint64_t total_updates, uint32_t minimum_ratio_bits) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  if (state == NULL || state->lifecycle != ET_O2_OPTIMIZER_STATE_LIVE ||
      state->active_borrows != 0u) {
    return -1;
  }
  state->config.clip_kind = clip_kind;
  state->config.clip_max_bits = clip_max_bits;
  state->config.schedule_kind = schedule_kind;
  state->config.warmup_updates = warmup_updates;
  state->config.total_updates = total_updates;
  state->config.minimum_ratio_bits = minimum_ratio_bits;
  return 0;
}

int32_t et_o2_test_state_set_option_bits_v1(et_o2_optimizer_state *candidate,
                                            size_t index, uint32_t option,
                                            uint32_t bits) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  et_o2_options *options;
  if (state == NULL || state->lifecycle != ET_O2_OPTIMIZER_STATE_LIVE ||
      state->active_borrows != 0u || index >= state->count ||
      option > ET_O2_OPTION_WEIGHT_DECAY) {
    return -1;
  }
  options = &state->entries[index].options;
  if (option == ET_O2_OPTION_LEARNING_RATE) {
    options->learning_rate_bits = bits;
  } else if (option == ET_O2_OPTION_BETA1) {
    options->beta1_bits = bits;
  } else if (option == ET_O2_OPTION_BETA2) {
    options->beta2_bits = bits;
  } else if (option == ET_O2_OPTION_EPSILON) {
    options->epsilon_bits = bits;
  } else {
    options->weight_decay_bits = bits;
  }
  return 0;
}

int32_t
et_o2_test_state_set_completed_updates_v1(et_o2_optimizer_state *candidate,
                                          uint64_t completed_updates) {
  et_o2_optimizer_state *state = et_o2_find_state(candidate);
  if (state == NULL || state->lifecycle != ET_O2_OPTIMIZER_STATE_LIVE ||
      state->active_borrows != 0u) {
    return -1;
  }
  state->completed_updates = completed_updates;
  return 0;
}

int32_t et_o2_test_optimizer_moment_bits_v1(const et_o2_optimizer *candidate,
                                            size_t index, uint32_t moment_kind,
                                            uint32_t *bits, size_t count,
                                            et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  et_f32_tensor *moment;
  et_f32_tensor_error i2_error;
  if (optimizer == NULL || index >= optimizer->count ||
      moment_kind > ET_O2_MOMENT_EXP_AVG_SQ || bits == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, "optimizer-test-moment",
                      "moment inspection operands are invalid");
  }
  moment = moment_kind == ET_O2_MOMENT_EXP_AVG
               ? optimizer->entries[index].exp_avg
               : optimizer->entries[index].exp_avg_sq;
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_tensor_copy_bits_to_v1(moment, bits, count, &i2_error) != 0) {
    return et_o2_from_i2(&i2_error, error, "optimizer-test-moment",
                         ET_O2_STATUS_INVALID_ARGUMENT);
  }
  return et_o2_success(error);
}
#endif
