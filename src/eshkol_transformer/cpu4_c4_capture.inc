#include "cpu4_c4_capture_private.h"

/* Feature-local bounded SHA-256, derived from native/c2_checkpoint_format.c. */
typedef struct cpu4_sha256_state {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} cpu4_sha256_state;

static uint32_t cpu4_rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void cpu4_sha256_block(cpu4_sha256_state *state, const uint8_t block[64]) {
  static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  uint32_t a, b, c, d, e, f, g, h;
  size_t i;
  for (i = 0; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) |
           ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 = cpu4_rotr32(w[i - 15u], 7u) ^ cpu4_rotr32(w[i - 15u], 18u) ^
                        (w[i - 15u] >> 3u);
    const uint32_t s1 = cpu4_rotr32(w[i - 2u], 17u) ^ cpu4_rotr32(w[i - 2u], 19u) ^
                        (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  a = state->h[0]; b = state->h[1]; c = state->h[2]; d = state->h[3];
  e = state->h[4]; f = state->h[5]; g = state->h[6]; h = state->h[7];
  for (i = 0; i < 64u; ++i) {
    const uint32_t s1 = cpu4_rotr32(e, 6u) ^ cpu4_rotr32(e, 11u) ^ cpu4_rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = cpu4_rotr32(a, 2u) ^ cpu4_rotr32(a, 13u) ^ cpu4_rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
  state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

static void cpu4_sha256_init(cpu4_sha256_state *state) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u,
                                      0x3c6ef372u, 0xa54ff53au,
                                      0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
  memcpy(state->h, initial, sizeof(initial));
  state->length = 0u;
  state->used = 0u;
}

static void cpu4_sha256_update(cpu4_sha256_state *state, const uint8_t *bytes,
                          size_t length) {
  state->length += (uint64_t)length;
  while (length != 0u) {
    size_t take = 64u - state->used;
    if (take > length) take = length;
    memcpy(state->block + state->used, bytes, take);
    state->used += take; bytes += take; length -= take;
    if (state->used == 64u) {
      cpu4_sha256_block(state, state->block);
      state->used = 0u;
    }
  }
}

static void cpu4_sha256_final(cpu4_sha256_state *state, uint8_t digest[32]) {
  uint64_t bits = state->length * UINT64_C(8);
  size_t i;
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    cpu4_sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (i = 0; i < 8u; ++i) state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  cpu4_sha256_block(state, state->block);
  for (i = 0; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}


#define CPU4_CAPTURE_MAGIC UINT64_C(0x4350344334434150)
enum { CPU4_CAPTURE_KIND = 13, CPU4_CAPTURE_LIVE = 1,
       CPU4_CAPTURE_DEAD = 2, CPU4_CAPTURE_VALUES = 4768 };

struct et_cpu4_c4_capture {
  struct et_cpu4_c4_capture *next; /* Always NULL: independent provenance registry. */
  uint64_t magic;
  uint32_t kind, state, busy, padding;
};

typedef struct cpu4_capture_payload {
  et_cpu4_c4_capture_record_v1 record;
  unsigned char values[CPU4_CAPTURE_VALUES];
} cpu4_capture_payload;

typedef struct cpu4_capture_provenance {
  struct cpu4_capture_provenance *next;
  et_cpu4_c4_capture *handle;
  cpu4_capture_payload *payload;
  size_t extent;
  uint32_t state, reserved;
  const void *generator, *model, *cache; /* Weak identity words only. */
  const void *parameters[14], *handles[14];
  unsigned char digest[32];
} cpu4_capture_provenance;

_Static_assert(sizeof(et_cpu4_c4_capture) == 32u, "capture handle extent");
_Static_assert(sizeof(cpu4_capture_payload) == 4992u, "capture payload extent");
_Static_assert(sizeof(cpu4_capture_provenance) == 320u, "capture provenance extent");
_Static_assert(offsetof(cpu4_capture_provenance, generator) == 40u,
               "capture weak identity offset");
_Static_assert(offsetof(cpu4_capture_provenance, digest) == 288u,
               "capture private seal offset");

static cpu4_capture_provenance *cpu4_captures;

/* Registry links are authoritative; no detached path follows weak source words. */
static int cpu4_capture_overlap(const void *pointer, size_t bytes) {
  const cpu4_capture_provenance *node;
  for (node = cpu4_captures; node != NULL; node = node->next) {
    if (et_g3c4_ranges_overlap(pointer, bytes, node, sizeof(*node)) ||
        et_g3c4_ranges_overlap(pointer, bytes, node->handle,
                                sizeof(*node->handle)) ||
        (node->payload != NULL &&
         et_g3c4_ranges_overlap(pointer, bytes, node->payload,
                                 sizeof(*node->payload))))
      return 1;
  }
  return 0;
}

static void cpu4_capture_le16(unsigned char *p, uint16_t v) {
  p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8u);
}
static void cpu4_capture_le32(unsigned char *p, uint32_t v) {
  for (unsigned i = 0; i < 4u; ++i) p[i] = (unsigned char)(v >> (8u * i));
}
static void cpu4_capture_le64(unsigned char *p, uint64_t v) {
  for (unsigned i = 0; i < 8u; ++i) p[i] = (unsigned char)(v >> (8u * i));
}

/* Hash input is field encoded; host padding and host byte order do not enter. */
static void cpu4_capture_digest(const cpu4_capture_payload *payload,
                                unsigned char digest[32]) {
  static const unsigned char domain[] = "cpu4-c4-capture-v1";
  const et_cpu4_c4_capture_record_v1 *r = &payload->record;
  unsigned char fields[192], word[4];
  cpu4_sha256_state sha;
  memset(fields, 0, sizeof(fields));
  cpu4_capture_le32(fields, r->size);
  cpu4_capture_le16(fields + 4u, r->major);
  cpu4_capture_le16(fields + 6u, r->minor);
  cpu4_capture_le32(fields + 8u, r->flags);
  cpu4_capture_le32(fields + 12u, r->count);
  cpu4_capture_le32(fields + 16u, r->cache_capacity);
  cpu4_capture_le32(fields + 20u, r->prompt_count);
  cpu4_capture_le32(fields + 24u, r->generated_count);
  cpu4_capture_le32(fields + 28u, r->manual_count);
  cpu4_capture_le32(fields + 32u, r->generated_eos_mask);
  cpu4_capture_le32(fields + 36u, r->reserved0);
  cpu4_capture_le64(fields + 40u, r->commit_epoch);
  for (unsigned i = 0; i < 4u; ++i) {
    cpu4_capture_le64(fields + 48u + 8u*i, (uint64_t)r->ids[i]);
    cpu4_capture_le32(fields + 80u + 4u*i, r->origins[i]);
  }
  for (unsigned i = 0; i < 6u; ++i)
    cpu4_capture_le64(fields + 96u + 8u*i, (uint64_t)r->policy[i]);
  for (unsigned i = 0; i < 4u; ++i)
    cpu4_capture_le64(fields + 144u + 8u*i, (uint64_t)r->rng[i]);
  cpu4_capture_le32(fields + 176u, r->parameter_count);
  cpu4_capture_le32(fields + 180u, r->values_bytes);
  cpu4_capture_le32(fields + 184u, r->state);
  cpu4_capture_le32(fields + 188u, r->reserved1);
  cpu4_sha256_init(&sha);
  cpu4_sha256_update(&sha, domain, sizeof(domain));
  cpu4_sha256_update(&sha, fields, sizeof(fields));
  for (size_t offset = 0u; offset < sizeof(payload->values); offset += 4u) {
    uint32_t bits;
    memcpy(&bits, payload->values + offset, 4u);
    cpu4_capture_le32(word, bits);
    cpu4_sha256_update(&sha, word, sizeof(word));
  }
  cpu4_sha256_final(&sha, digest);
}

static int cpu4_capture_record_valid(const cpu4_capture_payload *payload,
                                     const unsigned char seal[32]) {
  const et_cpu4_c4_capture_record_v1 *r = &payload->record;
  unsigned char digest[32];
  uint32_t prompt = 0u, generated = 0u, manual = 0u, mask = 0u;
  if (r->size != 224u || r->major != 1u || r->minor != 0u ||
      r->flags != 0u || r->reserved0 != 0u || r->reserved1 != 0u ||
      r->state != 1u || r->parameter_count != 14u ||
      r->values_bytes != CPU4_CAPTURE_VALUES ||
      (r->cache_capacity != 2u && r->cache_capacity != 4u) ||
      r->count > r->cache_capacity || r->count > 4u)
    return 0;
  for (unsigned i = 0u; i < 4u; ++i) {
    if (i >= r->count) {
      if (r->ids[i] != 0 || r->origins[i] != 0u) return 0;
      continue;
    }
    if (r->ids[i] < 0 || r->ids[i] > 255) return 0;
    switch (r->origins[i]) {
      case 1u: ++prompt; break;
      case 2u: ++generated;
        if (r->ids[i] == r->policy[5]) mask |= 1u << i;
        break;
      case 3u: ++manual; break;
      default: return 0;
    }
  }
  if (prompt != r->prompt_count || generated != r->generated_count ||
      manual != r->manual_count || mask != r->generated_eos_mask)
    return 0;
  cpu4_capture_digest(payload, digest);
  return memcmp(digest, r->digest, sizeof(digest)) == 0 &&
         memcmp(digest, seal, sizeof(digest)) == 0;
}

static cpu4_capture_provenance *cpu4_capture_admit(
    const et_cpu4_c4_capture *candidate, int allow_dead) {
  cpu4_capture_provenance *node;
  const et_cpu4_c4_capture *h;
  for (node = cpu4_captures; node != NULL; node = node->next)
    if (node->handle == candidate) break;
  if (node == NULL) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  h = node->handle;
  if (h->next != NULL || h->magic != CPU4_CAPTURE_MAGIC ||
      h->kind != CPU4_CAPTURE_KIND || h->busy != 0u || h->padding != 0u ||
      node->reserved != 0u || node->state != h->state ||
      (node->state != CPU4_CAPTURE_LIVE && node->state != CPU4_CAPTURE_DEAD)) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  if (node->state == CPU4_CAPTURE_DEAD) {
    unsigned char zero[32] = {0};
    const void *weak[31] = {0};
    if (node->payload != NULL || node->extent != 0u ||
        memcmp(&node->generator, weak, sizeof(weak)) != 0 ||
        memcmp(node->digest, zero, sizeof(zero)) != 0) {
      (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
      return NULL;
    }
    if (!allow_dead) {
      (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
      return NULL;
    }
    return node;
  }
  if (node->payload == NULL || node->extent != sizeof(*node->payload) ||
      !cpu4_capture_record_valid(node->payload, node->digest)) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  return node;
}

static int64_t cpu4_capture_copy_alias(const void *dst, size_t bytes) {
  const et_g3c4_transport_header_internal *transport;
  const et_g3c4_model_owner_internal *model;
  const cpu4_history_node *history;
#ifdef ET_CPU4_TRAINING_PRIVATE
  if (cpu4_control_span_overlaps_internal(dst, bytes))
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
#endif
  if (cpu4_capture_overlap(dst, bytes) || cpu4_history_overlap(dst, bytes))
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  /* The trusted history crosslink preserves the full context allocation span
     even if a mutable transport kind word was corrupted. */
  for (history = cpu4_history_registry; history != NULL;
       history = history->next) {
    if (et_g3c4_ranges_overlap(dst, bytes, history->context,
                                sizeof(*history->context)))
      return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
#ifdef ET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE
    /* Authoritative ownership survives corruption of context->manual_frame. */
    if (history->pending.frame != NULL &&
        et_g3c4_ranges_overlap(dst, bytes, history->pending.frame,
                                sizeof(et_g3c4_manual_frame_internal)))
      return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
#endif
  }
  for (transport = et_g3c4_transport_registry; transport != NULL;
       transport = transport->registry_next) {
    if (et_g3c4_ranges_overlap(dst, bytes, transport,
                                et_g3c4_transport_record_bytes(transport)))
      return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  }
  for (model = et_g3c4_owner_registry; model != NULL;
       model = model->registry_next)
    if (et_g3c4_ranges_overlap(dst, bytes, model, sizeof(*model)))
      return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  if (et_i64_tensor_private_storage_overlap_v1(dst, bytes) != 0 ||
      et_f32_tensor_private_storage_overlap_v1(dst, bytes) != 0 ||
      et_a2_kv_cache_private_storage_overlap_v1(dst, bytes) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  return 0;
}

static int cpu4_capture_tls_alias(const void *dst, uint64_t bytes, size_t fixed) {
  const uintptr_t p = (uintptr_t)dst;
  const uintptr_t tls = (uintptr_t)&et_g3c4_error_state;
  if (p >= tls && p < tls + sizeof(et_g3c4_error_state)) return 1;
  if (cpu4_history_span(dst,fixed,&et_g3c4_error_state,
                         sizeof(et_g3c4_error_state))) return 1;
  return bytes <= SIZE_MAX && et_g3c4_range_valid(dst, (size_t)bytes) &&
      et_g3c4_ranges_overlap(dst, (size_t)bytes,
                              &et_g3c4_error_state,
                              sizeof(et_g3c4_error_state));
}

/* Capture owns a local read pin set; idle context->pins must remain untouched. */
static int cpu4_capture_binding_matches_pins(
    const et_g3c4_context_internal *context,
    const et_g3c4_model_pins_internal *pins) {
  size_t offset = 0u;
  if (context->prefill_binding_ready != 1u ||
      context->prefill_binding_reserved != 0u) return 0;
  for (size_t i = 0u; i < 14u; ++i) {
    const et_kernel_tensor_view_v1 *view = &pins->views[i];
    if (context->prefill_binding_identities[i] != pins->identities[i] ||
        view->data == NULL ||
        view->byte_length > sizeof(context->prefill_binding_values) - offset ||
        memcmp(context->prefill_binding_values + offset,
               view->data, view->byte_length) != 0)
      return 0;
    offset += view->byte_length;
  }
  return offset == sizeof(context->prefill_binding_values);
}

et_cpu4_c4_capture *et_cpu4_private_c4_capture_create_v1(
    void *registered_generator) {
  et_g3c4_context_internal *context;
  cpu4_history_node *history;
  et_g3c4_model_pins_internal pins = {0};
  et_f32_tensor_error error;
  et_cpu4_c4_capture *handle = NULL;
  cpu4_capture_provenance *node = NULL;
  cpu4_capture_payload *payload = NULL;
  size_t offset = 0u;
  uint32_t capacity = 0u;
#ifdef ET_G3C4_OUTPUT_RESERVATION_PRIVATE
  et_g3c4_output_internal *pending_output = NULL;
#endif
#ifdef ET_G3C4_LOGITS_RESERVATION_PRIVATE
  et_g3c4_logits_internal *pending_logits = NULL;
#endif
  et_g3c4_error_reset_internal();
  context = et_g3c4_admit_idle_call(registered_generator);
  if (context == NULL) return NULL;
  history = cpu4_history_auth(context);
  if (history == NULL) return NULL;
  {
    unsigned char zero[40] = {0};
    _Static_assert(sizeof(history->pending) == 40u, "pending proof extent");
    if (memcmp(&history->pending, zero, sizeof(zero)) != 0) {
      (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
      return NULL;
    }
  }
#ifdef ET_G3C4_OUTPUT_RESERVATION_PRIVATE
  if (et_g3c4_pending_output_lookup(context, &pending_output) != 0)
    return NULL;
  if (pending_output != NULL) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
#endif
#ifdef ET_G3C4_LOGITS_RESERVATION_PRIVATE
  if (et_g3c4_pending_logits_lookup(context, &pending_logits) != 0)
    return NULL;
  if (pending_logits != NULL) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
#endif
  if (cpu4_history_cache_check(context, history, &capacity) != 0)
    return NULL;
  handle = et_g3c4_allocate(sizeof(*handle));
  node = et_g3c4_allocate(sizeof(*node));
  payload = et_g3c4_allocate(sizeof(*payload));
  if (handle == NULL || node == NULL || payload == NULL) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    goto fail;
  }
  if (et_g3c4_capture_f32(et_g3c4_model_pins_begin_internal(
          context->owner->parameters, context->owner->handles,
          &pins, &error), &error) != 0)
    goto fail;
  for (size_t i = 0u; i < 14u; ++i) {
    const et_kernel_tensor_view_v1 *view = &pins.views[i];
    size_t bytes = et_g3c4_parameter_elements(i) * sizeof(float);
    if (pins.parameters[i] != history->parameters[i] ||
        pins.identities[i] != history->handles[i] ||
        view->data == NULL || view->byte_length != bytes ||
        bytes > sizeof(payload->values) - offset) {
      (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
      goto fail;
    }
    memcpy(payload->values + offset, view->data, bytes);
    offset += bytes;
  }
  if (offset != sizeof(payload->values) ||
      (context->cpu4_history.count != 0u &&
       !cpu4_capture_binding_matches_pins(context, &pins)) ||
      (context->cpu4_history.count == 0u &&
       context->prefill_binding_ready != 0u)) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_STALE_BINDING);
    goto fail;
  }
  if (et_g3c4_capture_f32(
          et_g3c4_model_pins_check_internal(&pins, &error), &error) != 0)
    goto fail;
  if (cpu4_history_cache_check(context, history, &capacity) != 0 ||
      cpu4_history_auth(context) != history)
    goto fail;

  et_cpu4_c4_capture_record_v1 *r = &payload->record;
  const cpu4_history_record *h = &context->cpu4_history;
  r->size = sizeof(*r); r->major = 1u; r->minor = 0u;
  r->count = h->count; r->cache_capacity = capacity;
  r->prompt_count = h->prompt_count;
  r->generated_count = h->generated_count;
  r->manual_count = h->manual_count;
  r->generated_eos_mask = h->generated_eos_mask;
  r->commit_epoch = h->commit_epoch;
  memcpy(r->ids, h->ids, sizeof(r->ids));
  memcpy(r->origins, h->origin, sizeof(r->origins));
  memcpy(r->policy, history->policy, sizeof(r->policy));
  memcpy(r->rng, history->rng, sizeof(r->rng));
  r->parameter_count = 14u;
  r->values_bytes = CPU4_CAPTURE_VALUES;
  r->state = 1u;
  cpu4_capture_digest(payload, r->digest);
  if (!cpu4_capture_record_valid(payload, r->digest)) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    goto fail;
  }
  et_g3c4_model_pins_end_internal(&pins);
  handle->magic = CPU4_CAPTURE_MAGIC;
  handle->kind = CPU4_CAPTURE_KIND;
  handle->state = CPU4_CAPTURE_LIVE;
  node->handle = handle;
  node->payload = payload;
  node->extent = sizeof(*payload);
  node->state = CPU4_CAPTURE_LIVE;
  node->generator = context;
  node->model = context->owner;
  node->cache = context->cache;
  for (size_t i = 0u; i < 14u; ++i) {
    node->parameters[i] = history->parameters[i];
    node->handles[i] = history->handles[i];
  }
  memcpy(node->digest, r->digest, sizeof(node->digest));
  /* The only publication: all leases are closed and nothing can fail below. */
  node->next = cpu4_captures;
  cpu4_captures = node;
  return handle;
fail:
  if (pins.self == &pins) et_g3c4_model_pins_end_internal(&pins);
  if (payload != NULL) { memset(payload, 0, sizeof(*payload)); free(payload); }
  free(node);
  free(handle);
  return NULL;
}

int64_t et_cpu4_private_c4_capture_copy_record_v1(
    const et_cpu4_c4_capture *capture,
    et_cpu4_c4_capture_record_v1 *dst, uint64_t dst_bytes) {
  cpu4_capture_provenance *node;
  if (cpu4_capture_tls_alias(dst, dst_bytes, sizeof(*dst))) return ET_G3C4_INVALID_ARGUMENT;
  et_g3c4_error_reset_internal();
  node = cpu4_capture_admit(capture, 0);
  if (node == NULL) return et_g3c4_error_state.category;
  if (dst_bytes != sizeof(*dst) ||
      !et_g3c4_range_valid(dst, sizeof(*dst)) ||
      (uintptr_t)dst % _Alignof(et_cpu4_c4_capture_record_v1) != 0u)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  if (cpu4_capture_copy_alias(dst, sizeof(*dst)) != 0)
    return et_g3c4_error_state.category;
  if (cpu4_capture_admit(capture, 0) != node ||
      cpu4_capture_copy_alias(dst, sizeof(*dst)) != 0)
    return et_g3c4_error_state.category;
  memcpy(dst, &node->payload->record, sizeof(*dst));
  return 0;
}

int64_t et_cpu4_private_c4_capture_copy_parameters_v1(
    const et_cpu4_c4_capture *capture, void *dst, uint64_t dst_bytes) {
  cpu4_capture_provenance *node;
  if (cpu4_capture_tls_alias(dst, dst_bytes, CPU4_CAPTURE_VALUES)) return ET_G3C4_INVALID_ARGUMENT;
  et_g3c4_error_reset_internal();
  node = cpu4_capture_admit(capture, 0);
  if (node == NULL) return et_g3c4_error_state.category;
  if (dst_bytes != CPU4_CAPTURE_VALUES ||
      !et_g3c4_range_valid(dst, CPU4_CAPTURE_VALUES) ||
      (uintptr_t)dst % _Alignof(float) != 0u)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  if (cpu4_capture_copy_alias(dst, CPU4_CAPTURE_VALUES) != 0)
    return et_g3c4_error_state.category;
  if (cpu4_capture_admit(capture, 0) != node ||
      cpu4_capture_copy_alias(dst, CPU4_CAPTURE_VALUES) != 0)
    return et_g3c4_error_state.category;
  memcpy(dst, node->payload->values, CPU4_CAPTURE_VALUES);
  return 0;
}

int64_t et_cpu4_private_c4_capture_release_v1(et_cpu4_c4_capture *capture) {
  cpu4_capture_provenance *node;
  et_g3c4_error_reset_internal();
  node = cpu4_capture_admit(capture, 1);
  if (node == NULL) return et_g3c4_error_state.category;
  if (node->state == CPU4_CAPTURE_DEAD) return 0;
  /* Prevalidated fixed tail; detached source identity words are never followed. */
  memset(node->payload, 0, sizeof(*node->payload));
  free(node->payload);
  node->payload = NULL;
  node->extent = 0u;
  memset(&node->generator, 0, 31u * sizeof(void *));
  memset(node->digest, 0, sizeof(node->digest));
  node->state = CPU4_CAPTURE_DEAD;
  capture->state = CPU4_CAPTURE_DEAD;
  return 0;
}

#ifdef ET_CPU4_C4_P3_WIRE_PRIVATE
#include "cpu4_c4_p3_wire.inc"
#endif
