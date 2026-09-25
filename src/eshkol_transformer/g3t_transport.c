/* Source-private G3-T transport composition. M3T's native registry and owner
 * are included once; no C4 identity or transport source enters this TU. */
#include "m3_model.c"
#include "g3t_transport.h"
#include "m3_call_pins.h"
#include "eshkol_transformer/a2_kv_cache.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { G3T_GENERATOR = 1, G3T_LIVE = 1, G3T_DEAD = -1 };
enum { G3T_ARGUMENT = 1, G3T_STATE = 2, G3T_INTERNAL = 5 };
enum { G3T_IDENTITY = 1, G3T_LIFECYCLE = 2, G3T_CONFIG = 3,
       G3T_TOPOLOGY = 6, G3T_ALLOCATION = 7, G3T_INVARIANT = 10 };
typedef struct g3t_context {
  struct g3t_context *next;
  int kind, state, busy, call_kind;
  owner *model;
  et_a2_kv_cache *cache;
  int64_t policy[6], rng[4];
  et_g3t_model_pins_internal pins;
} g3t_context;
static g3t_context *g3t_registry;
static int64_t g3t_error_domain, g3t_error_category, g3t_error_code;

static void g3t_clear(void) {
  g3t_error_domain = g3t_error_category = g3t_error_code = 0;
}
static int64_t g3t_fail(int64_t domain, int64_t category, int64_t code) {
  g3t_error_domain = domain;
  g3t_error_category = category;
  g3t_error_code = code;
  return category;
}
static int64_t g3t_bad(int64_t category, int64_t code) {
  return g3t_fail(0, category, code);
}
static int64_t g3t_f32_failure(const et_f32_tensor_error *error) {
  return g3t_fail(3, error->category, error->code);
}
static int64_t g3t_a2_failure(const et_kernel_error *error) {
  return g3t_fail(1, error->category, error->code);
}
int64_t et_g3t_private_last_error_domain_v1(void) { return g3t_error_domain; }
int64_t et_g3t_private_last_error_category_v1(void) { return g3t_error_category; }
int64_t et_g3t_private_last_error_code_v1(void) { return g3t_error_code; }

#ifdef ET_G3T_TESTING
#ifndef ET_A2_KV_CACHE_TESTING
#error "G3-T allocation witness requires A2 cache testing hooks"
#endif
static uint64_t g3t_allocation_limit = UINT64_MAX, g3t_allocations;
int64_t et_g3t_test_fail_alloc_after_v1(uint64_t count) {
  g3t_allocation_limit = count;
  g3t_allocations = 0;
  return 0;
}
int64_t et_g3t_test_fail_a2_after_v1(uint64_t count) {
  et_a2_kv_cache_test_fail_alloc_after_v1((size_t)count);
  return 0;
}
uint64_t et_g3t_test_live_contexts_v1(void) {
  uint64_t count = 0;
  for (g3t_context *c = g3t_registry; c; c = c->next)
    if (c->state == G3T_LIVE) ++count;
  return count;
}
#endif
static g3t_context *g3t_allocate(void) {
#ifdef ET_G3T_TESTING
  if (g3t_allocations >= g3t_allocation_limit) return NULL;
#endif
  g3t_context *c = calloc(1, sizeof(*c));
#ifdef ET_G3T_TESTING
  if (c) ++g3t_allocations;
#endif
  return c;
}
/* Search by address before reading any candidate bytes. Dead headers remain
 * enrolled, so an old pointer cannot gain authority through address reuse. */
static g3t_context *g3t_admit(void *candidate, int allow_dead) {
  g3t_context *c = g3t_registry;
  while (c && c != candidate) c = c->next;
  if (!c || c->kind != G3T_GENERATOR) {
    g3t_bad(G3T_ARGUMENT, G3T_IDENTITY);
    return NULL;
  }
  if (!allow_dead && c->state != G3T_LIVE) {
    g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    return NULL;
  }
  return c;
}
static owner *g3t_model(void *candidate) {
  record *r = registry;
  while (r && r != candidate) r = r->next;
  if (!r || r->kind != OWNER) {
    g3t_bad(G3T_ARGUMENT, G3T_IDENTITY);
    return NULL;
  }
  owner *o = (owner *)r;
  record *original = registry, *successor = registry;
  while (original && original != (record *)o->original) original = original->next;
  while (successor && successor != (record *)o->successor) successor = successor->next;
  if (o->r.state != 1 || o->initialized != 14 || !original || !successor ||
      original->kind != INITIALIZER || successor->kind != INITIALIZER ||
      original->state != 1 || successor->state != 1) {
    g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    return NULL;
  }
  for (size_t i = 0; i < 14; ++i) {
    if (!o->p[i] || !o->handles[i]) {
      g3t_bad(G3T_STATE, G3T_TOPOLOGY);
      return NULL;
    }
    for (size_t j = 0; j < i; ++j)
      if (o->p[i] == o->p[j] || o->handles[i] == o->handles[j]) {
        g3t_bad(G3T_STATE, G3T_TOPOLOGY);
        return NULL;
      }
    et_f32_tensor_error error;
    if (et_f32_parameter_validate_identity_v1(o->p[i], o->handles[i], &error)) {
      g3t_f32_failure(&error);
      return NULL;
    }
  }
  return o;
}
static int g3t_positive_f32_bits(int64_t candidate) {
  if (candidate < 0 || (uint64_t)candidate > UINT32_MAX) return 0;
  uint32_t bits = (uint32_t)candidate;
  return bits != 0 && (bits & UINT32_C(0x80000000)) == 0 &&
         (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}
static int g3t_policy_valid(int64_t mode, int64_t temperature_bits, int64_t k,
                            int64_t p_bits, int64_t max_new, int64_t eos) {
  if ((mode != 0 && mode != 1) ||
      !g3t_positive_f32_bits(temperature_bits) ||
      !g3t_positive_f32_bits(p_bits) ||
      p_bits > INT64_C(0x3f800000) || k < 1 || k > 256 ||
      max_new < 0 || max_new > 1 || eos < -1 || eos > 255) return 0;
  return mode != 0 ||
         (temperature_bits == INT64_C(0x3f800000) && k == 256 &&
          p_bits == INT64_C(0x3f800000));
}
void *et_g3t_private_generator_seed_v1(
    void *model_owner, int64_t seed, int64_t mode, int64_t temperature_bits,
    int64_t k, int64_t p_bits, int64_t max_new, int64_t eos) {
  g3t_clear();
  owner *o = g3t_model(model_owner);
  if (!o) return NULL;
  if (seed < 0 || !g3t_policy_valid(mode, temperature_bits, k, p_bits,
                                    max_new, eos)) {
    g3t_bad(G3T_ARGUMENT, G3T_CONFIG);
    return NULL;
  }
  if (o->active) {
    g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    return NULL;
  }
  g3t_context *c = g3t_allocate();
  if (!c) {
    g3t_bad(G3T_INTERNAL, G3T_ALLOCATION);
    return NULL;
  }
  et_kernel_error error;
  if (et_a2_kv_cache_create_v1(1, 1, 2, 2, 2, &c->cache, &error)) {
    g3t_a2_failure(&error);
    free(c);
    return NULL;
  }
  c->kind = G3T_GENERATOR;
  c->state = G3T_LIVE;
  c->model = o;
  c->policy[0] = mode;
  c->policy[1] = temperature_bits;
  c->policy[2] = k;
  c->policy[3] = p_bits;
  c->policy[4] = max_new;
  c->policy[5] = eos;
  c->rng[0] = 1;
  c->rng[1] = seed;
  c->next = g3t_registry;
  g3t_registry = c;
  return c;
}
int64_t et_g3t_private_generator_close_v1(void *candidate) {
  g3t_clear();
  g3t_context *c = g3t_admit(candidate, 1);
  if (!c) return g3t_error_category;
  if (c->state == G3T_DEAD) return 0;
  owner *o = g3t_model(c->model);
  if (!o) return g3t_error_category;
  if (c->busy || o->active || c->pins.held_mask) return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  et_kernel_error error;
  if (et_a2_kv_cache_destroy_v1(&c->cache, &error)) return g3t_a2_failure(&error);
  memset(c->policy, 0, sizeof(c->policy));
  memset(c->rng, 0, sizeof(c->rng));
  c->model = NULL;
  c->state = G3T_DEAD;
  return 0;
}
int64_t et_g3t_private_call_acquire_v1(void *candidate, int64_t call_kind) {
  g3t_clear();
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return g3t_error_category;
  if (call_kind < 0 || call_kind > 2) return g3t_bad(G3T_ARGUMENT, G3T_CONFIG);
  owner *o = g3t_model(c->model);
  if (!o) return g3t_error_category;
  if (c->busy || o->active || c->pins.held_mask || !c->cache)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  et_f32_tensor_error error;
  if (et_g3t_model_pins_begin_internal(o->p, (const void *const *)o->handles,
                                       &c->pins, &error)) return g3t_f32_failure(&error);
  c->call_kind = (int)call_kind;
  c->busy = 1;
  o->active = c;
  return 0;
}
int64_t et_g3t_private_call_abort_v1(void *candidate) {
  g3t_clear();
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return g3t_error_category;
  if (!c->busy) return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  owner *o = g3t_model(c->model);
  if (!o || o->active != c) abort();
  et_f32_tensor_error error;
  if (et_g3t_model_pins_check_internal(&c->pins, &error)) abort();
  et_g3t_model_pins_end_internal(&c->pins);
  o->active = NULL;
  c->busy = 0;
  c->call_kind = 0;
  return 0;
}
#ifdef ET_G3T_TESTING
int64_t et_g3t_test_pin_count_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return -1;
  if (!c->busy) return c->pins.held_mask ? -1 : 0;
  et_f32_tensor_error error;
  return et_g3t_model_pins_check_internal(&c->pins, &error) ? -1 : 14;
}
int64_t et_g3t_test_cache_empty_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return 0;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL, *lengths = NULL;
  const et_kernel_tensor_view_v1 *mask = NULL;
  et_kernel_error error;
  if (et_a2_kv_cache_read_borrow_begin_v1(c->cache, &borrow, &error)) return 0;
  int ok = !et_a2_kv_cache_read_borrow_layer_v1(
      borrow, 0, &keys, &values, &lengths, &mask, &error) &&
      keys && values && lengths && mask && ((const int64_t *)lengths->data)[0] == 0;
  if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error)) abort();
  return ok;
}
#endif
