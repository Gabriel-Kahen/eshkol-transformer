/* Private status-only C4 owner/context exclusion fixture. */
#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3R_C4_CONTEXT_EXCLUSION_PRIVATE 1

#ifdef ET_G3R_C4_CONTEXT_EXCLUSION_A2_TU
#include "../../native/a2_kv_cache.c"
#else
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include <stdio.h>

#define CHECK(x) do { checks++; if (!(x)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
#define OK(x) CHECK((x) == 0)

static int checks;
static unsigned char first_ids[14], second_ids[14];
static et_g3c4_model_owner_internal *first_owner, *second_owner;

static et_g3c4_model_owner_internal *make_owner(
    unsigned char ids[14], int64_t seed) {
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(seed);
  CHECK(owner != NULL);
  for (int64_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_bind_v1(owner, i, &ids[(size_t)i]));
  for (int64_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, i));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

static void status(const void *candidate, int64_t category, int64_t code) {
  et_g3c4_model_owner_internal first_before = *first_owner;
  et_g3c4_model_owner_internal second_before = *second_owner;
  et_g3c4_transport_header_internal *registry_before =
      et_g3c4_transport_registry;
  int64_t got = et_g3c4_private_no_live_owner_context_status_v1(candidate);
  CHECK(got == category);
  CHECK(et_g3c4_private_last_error_category_v1() == category);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
  CHECK(et_g3c4_private_last_error_domain_v1() ==
        (category == 0 ? 0 : ET_G3C4_DOMAIN_C4));
  CHECK(!memcmp(first_owner, &first_before, sizeof(first_before)));
  CHECK(!memcmp(second_owner, &second_before, sizeof(second_before)));
  CHECK(et_g3c4_transport_registry == registry_before);
}

int main(void) {
  int foreign = 0;
  et_g3c4_model_owner_internal *first = make_owner(first_ids, 1729);
  et_g3c4_model_owner_internal *second = make_owner(second_ids, 1730);
  first_owner = first;
  second_owner = second;
  et_g3c4_model_owner_internal first_before = *first;
  et_g3c4_model_owner_internal second_before = *second;
  status(NULL, ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  status(&foreign, ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  status(first, 0, 0);

  /* Mixed transport records must be validated and skipped by kind. */
  et_g3c4_rng_internal *rng =
      (et_g3c4_rng_internal *)et_g3c4_private_rng_seed_v1(9);
  CHECK(rng != NULL);
  status(first, 0, 0);
  uint64_t rng_magic = rng->transport.magic;
  rng->transport.magic = 0;
  status(first, ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
  rng->transport.magic = rng_magic;
  status(first, 0, 0);

  et_g3c4_context_internal *other =
      (et_g3c4_context_internal *)et_g3c4_private_generator_seed_v1(
          second, 4, 0, INT64_C(0x3f800000), 256,
          INT64_C(0x3f800000), 1, -1);
  CHECK(other != NULL);
  status(first, 0, 0);
  status(second, ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);

  et_g3c4_context_internal *same =
      (et_g3c4_context_internal *)et_g3c4_private_generator_seed_v1(
          first, 5, 0, INT64_C(0x3f800000), 256,
          INT64_C(0x3f800000), 1, -1);
  CHECK(same != NULL);
  CHECK(first->active == NULL && same->cache != NULL);
  et_g3c4_context_internal same_before = *same;
  status(first, ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  CHECK(!memcmp(same, &same_before, sizeof(same_before)));
  CHECK(!memcmp(first, &first_before, sizeof(first_before)));
  CHECK(!memcmp(second, &second_before, sizeof(second_before)));

  OK(et_g3c4_private_call_acquire_v1(same, 2, 1));
  CHECK(first->active == same);
  status(first, ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_call_finish_v1(same));
  CHECK(first->active == NULL);
  status(first, ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_generator_close_v1(same));
  CHECK(same->transport.state == ET_G3C4_CONTEXT_DEAD);
  status(first, 0, 0);
  status(second, ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_generator_close_v1(other));
  status(second, 0, 0);
  OK(et_g3c4_private_rng_release_v1(rng));
  status(first, 0, 0);

  /* A live non-generator context also retains a cache and excludes create. */
  et_g3c4_context_internal *base =
      (et_g3c4_context_internal *)et_g3c4_private_context_create_v1(first);
  CHECK(base != NULL && base->cache != NULL && first->active == NULL);
  status(first, ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_context_close_v1(base));
  status(first, 0, 0);

  first->active = &foreign;
  status(first, ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  et_g3c4_rng_internal *edge_rng =
      (et_g3c4_rng_internal *)et_g3c4_private_rng_seed_v1(11);
  CHECK(edge_rng != NULL);
  et_g3c4_model_owner_internal edge_before = *first;
  et_g3c4_transport_header_internal *edge_registry =
      et_g3c4_transport_registry;
  size_t edge_allocations = et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_context_create_v1(first) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_LIFECYCLE);
  CHECK(et_g3c4_private_generator_seed_v1(
      first, 12, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), 1, -1) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_LIFECYCLE);
  CHECK(et_g3c4_private_generator_rng_v1(
      first, edge_rng, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), 1, -1) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_LIFECYCLE);
  CHECK(!memcmp(first, &edge_before, sizeof(edge_before)));
  CHECK(et_g3c4_transport_registry == edge_registry);
  CHECK(et_g3c4_context_successful_allocations == edge_allocations);
  first->active = NULL;
  OK(et_g3c4_private_rng_release_v1(edge_rng));
  CHECK(!memcmp(first, &first_before, sizeof(first_before)));
  CHECK(!memcmp(second, &second_before, sizeof(second_before)));
  printf("G3-R C4 native context exclusion PASS checks=%d\n", checks);
  return 0;
}
#endif
