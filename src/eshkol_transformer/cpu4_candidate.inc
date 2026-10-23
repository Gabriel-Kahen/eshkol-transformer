/* Source-private CPU4 producer; included after C4 context definitions. */
#include "../../native/p1_identity_internal.h"
#include "../../native/o2_optimizer_internal.h"
#include "cpu4_candidate_internal.h"

#ifdef ET_CPU4_CANDIDATE_TESTING
static int cpu4_candidate_fail_after;
#define CPU4_CANDIDATE_FAULT(stage) do { \
  if (cpu4_candidate_fail_after == (stage)) _Exit(134); \
} while (0)
#else
#define CPU4_CANDIDATE_FAULT(stage) ((void)0)
#endif

static et_cpu4_candidate_internal *cpu4_candidate_admit(void *pointer) {
  et_cpu4_candidate_internal *candidate = cpu4_candidates;
  while (candidate != NULL && candidate != pointer) candidate = candidate->next;
  if (candidate == NULL || candidate->origin_pid != (int64_t)getpid() ||
      candidate->owner == NULL ||
      et_g3c4_admit_owner(candidate->owner, 1) != candidate->owner ||
      candidate->owner->candidate != candidate) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  return candidate;
}

static int cpu4_candidate_dead_exact(const et_cpu4_candidate_internal *candidate) {
  const et_g3c4_model_owner_internal *owner = candidate->owner;
  if (owner->state != ET_G3C4_OWNER_ABORTED || owner->initialized != 0u ||
      owner->successor_ready != 0 || owner->active != NULL ||
      owner->training_lease != NULL || owner->live_contexts != 0u ||
      candidate->optimizer != NULL || candidate->generator != NULL)
    return 0;
  for (size_t i = 0u; i < 14u; ++i)
    if (owner->parameters[i] != NULL || owner->handles[i] != NULL ||
        et_f32_parameter_is_live_v1(candidate->parameters[i]) != 0 ||
        (candidate->handles[i] != NULL &&
         et_p1_public_token_live_v1(candidate->handles[i]) != 0)) return 0;
  return 1;
}

void *et_cpu4_private_candidate_begin_v1(int64_t seed) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = ET_R_ALLOC(ET_R_C4,1,et_g3c4_allocate(sizeof(*candidate)),1u,sizeof(*candidate));
  if (candidate == NULL) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    return NULL;
  }
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(seed);
  if (owner == NULL) { ET_R_FREE(ET_R_C4,1,candidate,1u,sizeof(*candidate)); return NULL; }
  /* All allocations precede this exact typed enrollment. */
  candidate->owner = owner;
  memcpy(candidate->parameters, owner->parameters, sizeof(candidate->parameters));
  candidate->origin_pid = (int64_t)getpid();
  candidate->state = CPU4_CANDIDATE_CONSTRUCTING;
  candidate->next = cpu4_candidates;
  cpu4_candidates = candidate;
  owner->candidate = candidate;
  return candidate;
}

void *et_cpu4_private_candidate_owner_v1(void *pointer) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = cpu4_candidate_admit(pointer);
  if (candidate == NULL) return NULL;
  if (candidate->state == CPU4_CANDIDATE_DEAD) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  return candidate->owner;
}

int64_t et_cpu4_private_candidate_bind_graph_v1(
    void *pointer, void *context, void *construction) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = cpu4_candidate_admit(pointer);
  if (candidate == NULL) return et_g3c4_error_state.category;
  if (candidate->state != CPU4_CANDIDATE_CONSTRUCTING ||
      candidate->owner != staged_c4_owner ||
      candidate->owner->state != ET_G3C4_OWNER_OPEN ||
      candidate->p1_context != NULL || candidate->p1_scope != NULL)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (et_p1_private_candidate_state_preflight_v1(context, construction, 0) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  for (const et_cpu4_candidate_internal *other = cpu4_candidates;
       other != NULL; other = other->next)
    if (other != candidate && other->p1_scope == construction)
      return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  candidate->p1_context = context;
  candidate->p1_scope = construction;
  return 0;
}

int64_t et_cpu4_private_candidate_abort_open_v1(void *pointer) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = cpu4_candidate_admit(pointer);
  if (candidate == NULL) return et_g3c4_error_state.category;
  if (candidate->state == CPU4_CANDIDATE_DEAD)
    return cpu4_candidate_dead_exact(candidate) ? 0 :
        et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (candidate->state != CPU4_CANDIDATE_CONSTRUCTING ||
      candidate->owner != staged_c4_owner ||
      (candidate->owner->state != ET_G3C4_OWNER_OPEN &&
       candidate->owner->state != ET_G3C4_OWNER_PREPARED))
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (candidate->p1_scope != NULL &&
      et_p1_private_candidate_state_preflight_v1(
          candidate->p1_context, candidate->p1_scope, 2) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (memcmp(candidate->parameters, candidate->owner->parameters,
             sizeof(candidate->parameters)) != 0 ||
      memcmp(candidate->handles, candidate->owner->handles,
             sizeof(candidate->handles)) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_IDENTITY);
  /* Prove all children before the existing no-fail staged-owner abort. */
  for (size_t i = 0u; i < 14u; ++i) {
    et_f32_tensor_error error;
    if (et_g3c4_capture_f32(et_cpu4_parameter_destroy_preflight_internal(
            candidate->owner->parameters[i], &error), &error))
      return et_g3c4_error_state.category;
  }
  candidate->abort_authorized = 1;
  int64_t status = et_g3c4_private_model_owner_abort_v1(candidate->owner);
  candidate->abort_authorized = 0;
  if (status == 0) candidate->state = CPU4_CANDIDATE_DEAD;
  return status;
}

static int64_t cpu4_candidate_roster(et_cpu4_candidate_internal *candidate) {
  et_g3c4_model_owner_internal *owner = candidate->owner;
  if (owner->state != ET_G3C4_OWNER_SEALED || owner->initialized != 14u ||
      !owner->successor_ready || owner->training_lease != NULL ||
      owner->active != NULL || candidate->p1_scope == NULL ||
      candidate->p1_context == NULL)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  for (size_t i = 0u; i < 14u; ++i) {
    et_f32_tensor_error error;
    if (owner->parameters[i] != candidate->parameters[i] ||
        owner->handles[i] != candidate->handles[i])
      return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_IDENTITY);
    if (owner->parameters[i] == NULL || owner->handles[i] == NULL ||
        et_g3c4_capture_f32(et_f32_parameter_validate_identity_v1(
            owner->parameters[i], owner->handles[i], &error), &error)) {
      if (et_g3c4_error_state.category == 0)
        (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_IDENTITY);
      return et_g3c4_error_state.category;
    }
    if (et_p1_private_candidate_handle_preflight_v1(
            candidate->p1_context, candidate->p1_scope, owner->handles[i]) != 0)
      return et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    if (!et_g3c4_parameter_shape_valid(owner, i))
      return et_g3c4_fail(ET_G3C4_SHAPE_MISMATCH, ET_G3C4_CODE_SHAPE);
    if (et_g3c4_capture_f32(et_cpu4_parameter_destroy_preflight_internal(
            owner->parameters[i], &error), &error))
      return et_g3c4_error_state.category;
    for (size_t j = 0u; j < i; ++j)
      if (owner->parameters[i] == owner->parameters[j] ||
          owner->handles[i] == owner->handles[j] ||
          et_f32_parameter_canonical_owner_v1(owner->parameters[i]) ==
              et_f32_parameter_canonical_owner_v1(owner->parameters[j]))
        return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_TOPOLOGY);
  }
  if (et_p1_private_candidate_graph_preflight_v1(
          candidate->p1_context, candidate->p1_scope) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_IDENTITY);
  return 0;
}

static int64_t cpu4_candidate_generator_preflight(
    et_cpu4_candidate_internal *candidate, void *generator) {
  et_g3c4_model_owner_internal *owner = candidate->owner;
  if (generator == NULL) {
    if (owner->live_contexts == 0u) return 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
#ifdef ET_G3C4_GENERATOR_PRIVATE
  et_g3c4_context_internal *context = et_g3c4_admit_context(generator);
  if (context == NULL) return et_g3c4_error_state.category;
  if (owner->live_contexts != 1u || context->owner != owner ||
      context->generator_kind != 1u || context->generator_ready != 1u ||
      ET_G3C4_CONTEXT_STATE(context) != ET_G3C4_CONTEXT_LIVE ||
      ET_G3C4_CONTEXT_BUSY(context) != ET_G3C4_CALL_IDLE ||
      context->cache == NULL || context->call_kind != 0 ||
      context->budget != 0 || context->acquired_mask != 0u ||
      !et_g3c4_pins_idle(&context->pins) ||
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
      !et_g3c4_token_frame_idle(context) ||
#endif
      !et_g3c4_valid_generator_policy(
          context->generator_policy[0], context->generator_policy[1],
          context->generator_policy[2], context->generator_policy[3],
          context->generator_policy[4], context->generator_policy[5]) ||
      context->generator_rng_words[0] != 1 || context->generator_rng_words[1] < 0)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  et_kernel_error error;
  return et_g3c4_capture_kernel(
      et_cpu4_kv_cache_destroy_preflight_internal(context->cache, &error), &error);
#else
  return et_g3c4_fail(ET_G3C4_UNSUPPORTED, ET_G3C4_CODE_CAPABILITY);
#endif
}

int64_t et_cpu4_private_candidate_adopt_v1(void *pointer, void *generator) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = cpu4_candidate_admit(pointer);
  if (candidate == NULL) return et_g3c4_error_state.category;
  if (candidate->state != CPU4_CANDIDATE_UNPUBLISHED)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  int64_t status = cpu4_candidate_roster(candidate);
  if (status != 0) return status;
  status = cpu4_candidate_generator_preflight(candidate, generator);
  if (status != 0) return status;
  candidate->state = CPU4_CANDIDATE_ADOPTED;
  return 0;
}

static int64_t cpu4_candidate_retire_preflight(
    et_cpu4_candidate_internal *candidate, et_o2_optimizer *optimizer,
    void *generator) {
  int64_t status = cpu4_candidate_roster(candidate);
  if (status != 0) return status;
  status = cpu4_candidate_generator_preflight(candidate, generator);
  if (status != 0) return status;
  et_o2_error_v1 error;
  if (et_cpu4_optimizer_roster_preflight_internal(optimizer,
          candidate->owner->parameters, candidate->owner->handles, &error) != 0)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  return 0;
}

int64_t et_cpu4_private_candidate_retire_prepare_v1(
    void *pointer, void *optimizer, void *generator) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = cpu4_candidate_admit(pointer);
  if (candidate == NULL) return et_g3c4_error_state.category;
  if (candidate->state != CPU4_CANDIDATE_UNPUBLISHED)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  int64_t status = cpu4_candidate_retire_preflight(candidate, optimizer, generator);
  if (status != 0) return status;
  candidate->optimizer = optimizer;
  candidate->generator = generator;
  candidate->state = CPU4_CANDIDATE_PREPARED;
  return 0;
}

int64_t et_cpu4_private_candidate_retire_cancel_v1(void *pointer) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = cpu4_candidate_admit(pointer);
  if (candidate == NULL) return et_g3c4_error_state.category;
  if (candidate->state != CPU4_CANDIDATE_PREPARED)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  candidate->optimizer = NULL;
  candidate->generator = NULL;
  candidate->state = CPU4_CANDIDATE_UNPUBLISHED;
  return 0;
}

int64_t et_cpu4_private_candidate_retire_graph_commit_v1(void *pointer) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = cpu4_candidate_admit(pointer);
  if (candidate == NULL) return et_g3c4_error_state.category;
  if (candidate->state != CPU4_CANDIDATE_PREPARED)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  int64_t status = cpu4_candidate_retire_preflight(
      candidate, candidate->optimizer, candidate->generator);
  if (status != 0) return status;
  int mutated = 0;
#ifdef ET_G3C4_GENERATOR_PRIVATE
  if (candidate->generator != NULL) {
    status = et_g3c4_private_generator_close_v1(candidate->generator);
    if (status != 0) return status; /* close failed before destroying its cache */
    mutated = 1;
    CPU4_CANDIDATE_FAULT(1);
  }
#endif
  if (candidate->optimizer != NULL) {
    et_o2_error_v1 error;
    if (et_o2_private_candidate_retire_commit_v1(candidate->optimizer, &error)) {
      if (mutated) _Exit(134);
      return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    }
    mutated = 1;
    CPU4_CANDIDATE_FAULT(2);
  }
  if (et_p1_private_candidate_graph_revoke_v1(
          candidate->p1_context, candidate->p1_scope) != 0) {
    if (mutated) _Exit(134);
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_IDENTITY);
  }
  CPU4_CANDIDATE_FAULT(3);
  candidate->state = CPU4_CANDIDATE_REVOKED;
  return 0;
}

int64_t et_cpu4_private_candidate_retire_payload_commit_v1(void *pointer) {
  et_g3c4_error_reset_internal();
  et_cpu4_candidate_internal *candidate = cpu4_candidate_admit(pointer);
  if (candidate == NULL) return et_g3c4_error_state.category;
  if (candidate->state == CPU4_CANDIDATE_DEAD)
    return cpu4_candidate_dead_exact(candidate) ? 0 :
        et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (candidate->state != CPU4_CANDIDATE_REVOKED)
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  et_g3c4_model_owner_internal *owner = candidate->owner;
  if (owner->active != NULL || owner->training_lease != NULL ||
      owner->live_contexts != 0u) _Exit(134);
  for (size_t i = 0u; i < 14u; ++i) {
    et_f32_tensor_error error;
    if (owner->parameters[i] != candidate->parameters[i] ||
        owner->handles[i] != candidate->handles[i]) _Exit(134);
    if (et_cpu4_parameter_destroy_preflight_internal(owner->parameters[i], &error))
      _Exit(134);
  }
  for (size_t i = 0u; i < 14u; ++i) {
    if (et_f32_parameter_destroy_v1(&owner->parameters[i], NULL)) _Exit(134);
    CPU4_CANDIDATE_FAULT(4 + (int)i);
  }
  memset(owner->handles, 0, sizeof(owner->handles));
  owner->initialized = 0u;
  owner->successor_ready = 0;
  owner->state = ET_G3C4_OWNER_ABORTED;
  candidate->optimizer = NULL;
  candidate->generator = NULL;
  candidate->state = CPU4_CANDIDATE_DEAD;
  return 0;
}

#undef CPU4_CANDIDATE_FAULT
