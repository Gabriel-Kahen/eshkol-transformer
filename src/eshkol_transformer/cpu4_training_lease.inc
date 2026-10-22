/* Included only by the canonical C4 owner TU under ET_CPU4_TRAINING_PRIVATE. */
enum { CPU4_LEASE_IDLE, CPU4_LEASE_READ, CPU4_LEASE_WRITE,
       CPU4_LEASE_PREPARED, CPU4_LEASE_DEAD };
#define CPU4_LEASE_MAGIC UINT64_C(0x435055344c454153)
typedef struct cpu4_training_child {
  struct cpu4_training_child *next;
  const void *identity;
  uint64_t ordinal;
  int64_t kind;
  int live;
} cpu4_training_child;
struct et_g3c4_training_lease_internal {
  struct et_g3c4_training_lease_internal *next;
  uint64_t magic;
  et_g3c4_model_owner_internal *owner;
  et_f32_parameter *parameters[14];
  const void *identities[14];
  cpu4_training_child *children;
  uint64_t ordinal;
  int phase;
};
static et_g3c4_training_lease_internal *cpu4_training_leases;

static int cpu4_lease_error(int category, int code) {
  return (int)et_g3c4_fail(category, code);
}
static et_g3c4_training_lease_internal *cpu4_admit_lease(void *candidate) {
  et_g3c4_training_lease_internal *lease = cpu4_training_leases;
  while (lease != NULL && lease != candidate) lease = lease->next;
  if (lease == NULL || lease->magic != CPU4_LEASE_MAGIC) {
    (void)cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  return lease;
}
static int cpu4_lease_roster(et_g3c4_training_lease_internal *lease) {
  if (lease->phase == CPU4_LEASE_DEAD)
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  et_g3c4_model_owner_internal *owner = et_g3c4_admit_owner(lease->owner, 0);
  if (owner == NULL) return (int)et_g3c4_error_state.category;
  if (owner->state != ET_G3C4_OWNER_SEALED ||
      owner->training_lease != lease || owner->active != NULL ||
      owner->live_contexts != 0u)
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  for (size_t i = 0; i < 14u; ++i) {
    et_f32_tensor_error error;
    if (owner->parameters[i] != lease->parameters[i] ||
        owner->handles[i] != lease->identities[i])
      return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_IDENTITY);
    if (et_g3c4_capture_f32(et_f32_parameter_validate_identity_v1(
            owner->parameters[i], owner->handles[i], &error), &error))
      return (int)et_g3c4_error_state.category;
    if (!et_g3c4_parameter_shape_valid(owner, i))
      return cpu4_lease_error(ET_G3C4_SHAPE_MISMATCH, ET_G3C4_CODE_SHAPE);
  }
  return 0;
}
static int cpu4_children_idle(const et_g3c4_training_lease_internal *lease) {
  for (const cpu4_training_child *child = lease->children;
       child != NULL; child = child->next)
    if (child->live) return 0;
  return 1;
}
static int cpu4_storage_idle(et_g3c4_training_lease_internal *lease) {
  et_f32_tensor_error error;
  return et_g3c4_owner_idle_preflight(lease->owner, &error);
}
#ifdef ET_G3C4_CONTEXT_PRIVATE
static int cpu4_context_available(et_g3c4_model_owner_internal *owner) {
  return owner->training_lease == NULL && owner->live_contexts != SIZE_MAX;
}
static void cpu4_context_published(et_g3c4_model_owner_internal *owner) {
  if (!cpu4_context_available(owner)) abort();
  owner->live_contexts++;
}
static int cpu4_context_close_ready(et_g3c4_model_owner_internal *owner) {
  if (owner->training_lease != NULL || owner->live_contexts == 0u)
    return cpu4_lease_error(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
  return 0;
}
#endif

void *et_g3c4_private_training_lease_create_v1(void *candidate) {
  et_g3c4_error_reset_internal();
  et_g3c4_model_owner_internal *owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return NULL;
  if (owner->state != ET_G3C4_OWNER_SEALED || owner->active != NULL ||
      owner->training_lease != NULL || owner->live_contexts != 0u) {
    (void)cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  et_f32_tensor_error error;
  if (et_g3c4_owner_idle_preflight(owner, &error)) return NULL;
  for (size_t i = 0; i < 14u; ++i)
    if (!et_g3c4_parameter_shape_valid(owner, i)) {
      (void)cpu4_lease_error(ET_G3C4_SHAPE_MISMATCH, ET_G3C4_CODE_SHAPE);
      return NULL;
    }
  et_g3c4_training_lease_internal *lease = et_g3c4_allocate(sizeof(*lease));
  if (lease == NULL) {
    (void)cpu4_lease_error(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    return NULL;
  }
  lease->owner = owner;
  lease->magic = CPU4_LEASE_MAGIC;
  memcpy(lease->parameters, owner->parameters, sizeof(lease->parameters));
  memcpy(lease->identities, owner->handles, sizeof(lease->identities));
  lease->next = cpu4_training_leases;
  cpu4_training_leases = lease;
  owner->training_lease = lease;
  return lease;
}

static void *cpu4_lease_member(void *candidate, int64_t index, int identity) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL || cpu4_lease_roster(lease)) return NULL;
  if (index < 0 || index >= 14) {
    (void)cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return NULL;
  }
  return identity ? (void *)lease->identities[index] : lease->parameters[index];
}
void *et_g3c4_private_training_lease_parameter_v1(void *lease, int64_t index) {
  return cpu4_lease_member(lease, index, 0);
}
void *et_g3c4_private_training_lease_identity_v1(void *lease, int64_t index) {
  return cpu4_lease_member(lease, index, 1);
}
int64_t et_g3c4_private_training_lease_check_v1(void *candidate) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  return lease == NULL ? et_g3c4_error_state.category : cpu4_lease_roster(lease);
}
int64_t et_g3c4_private_training_lease_enter_v1(void *candidate, int64_t phase) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL || cpu4_lease_roster(lease)) return et_g3c4_error_state.category;
  if (phase != CPU4_LEASE_READ && phase != CPU4_LEASE_WRITE)
    return cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
  if (lease->phase != CPU4_LEASE_IDLE || !cpu4_children_idle(lease) ||
      lease->ordinal == UINT64_MAX)
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (cpu4_storage_idle(lease)) return et_g3c4_error_state.category;
  lease->ordinal++;
  lease->phase = (int)phase;
  return 0;
}
int64_t et_g3c4_private_training_lease_leave_v1(void *candidate) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL || cpu4_lease_roster(lease)) return et_g3c4_error_state.category;
  if ((lease->phase != CPU4_LEASE_READ && lease->phase != CPU4_LEASE_WRITE) ||
      !cpu4_children_idle(lease))
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (cpu4_storage_idle(lease)) return et_g3c4_error_state.category;
  lease->phase = CPU4_LEASE_IDLE;
  return 0;
}
int64_t et_g3c4_training_child_acquire_internal(
    void *candidate, const void *identity, int64_t kind) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL || cpu4_lease_roster(lease)) return et_g3c4_error_state.category;
  if (identity == NULL || kind < 1 || kind > 5)
    return cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
  if (lease->phase != CPU4_LEASE_READ)
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  for (et_g3c4_training_lease_internal *other = cpu4_training_leases;
       other != NULL; other = other->next)
    for (cpu4_training_child *child = other->children; child != NULL; child = child->next)
      if (child->identity == identity)
        return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_IDENTITY);
  cpu4_training_child *child = et_g3c4_allocate(sizeof(*child));
  if (child == NULL) return cpu4_lease_error(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
  child->identity = identity;
  child->kind = kind;
  child->ordinal = lease->ordinal;
  child->live = 1;
  child->next = lease->children;
  lease->children = child;
  return 0;
}
int64_t et_g3c4_training_child_release_internal(
    void *candidate, const void *identity, int64_t kind) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL || cpu4_lease_roster(lease)) return et_g3c4_error_state.category;
  if (lease->phase != CPU4_LEASE_READ)
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  for (cpu4_training_child *child = lease->children; child != NULL; child = child->next)
    if (child->identity == identity && child->kind == kind &&
        child->ordinal == lease->ordinal && child->live) {
      child->live = 0;
      return 0;
    }
  return cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
}
int64_t et_g3c4_private_training_lease_prepare_release_v1(void *candidate) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL || cpu4_lease_roster(lease)) return et_g3c4_error_state.category;
  if (lease->phase != CPU4_LEASE_IDLE || !cpu4_children_idle(lease))
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (cpu4_storage_idle(lease)) return et_g3c4_error_state.category;
  lease->phase = CPU4_LEASE_PREPARED;
  return 0;
}
int64_t et_g3c4_private_training_lease_cancel_release_v1(void *candidate) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL || cpu4_lease_roster(lease)) return et_g3c4_error_state.category;
  if (lease->phase != CPU4_LEASE_PREPARED)
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  lease->phase = CPU4_LEASE_IDLE;
  return 0;
}
int64_t et_g3c4_private_training_lease_commit_release_v1(void *candidate) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL) return et_g3c4_error_state.category;
  if (lease->phase == CPU4_LEASE_DEAD) return 0;
  if (lease->phase != CPU4_LEASE_PREPARED || lease->owner->training_lease != lease ||
      !cpu4_children_idle(lease))
    return cpu4_lease_error(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  lease->owner->training_lease = NULL;
  lease->phase = CPU4_LEASE_DEAD;
  return 0;
}

static int cpu4_control_overlap(const void *span, size_t bytes,
                               const void *control, size_t extent) {
  uintptr_t a = (uintptr_t)span, b = (uintptr_t)control;
  return b > UINTPTR_MAX - extent || (a < b + extent && b < a + bytes);
}
int64_t et_g3c4_training_span_disjoint_internal(
    void *candidate, const void *span, size_t bytes) {
  et_g3c4_error_reset_internal();
  et_g3c4_training_lease_internal *lease = cpu4_admit_lease(candidate);
  if (lease == NULL || cpu4_lease_roster(lease)) return et_g3c4_error_state.category;
  if (span == NULL || bytes == 0 || (uintptr_t)span > UINTPTR_MAX - bytes)
    return cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  for (et_g3c4_model_owner_internal *owner = et_g3c4_owner_registry;
       owner != NULL; owner = owner->registry_next)
    if (cpu4_control_overlap(span, bytes, owner, sizeof(*owner)))
      return cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  for (et_g3c4_training_lease_internal *item = cpu4_training_leases;
       item != NULL; item = item->next) {
    if (cpu4_control_overlap(span, bytes, item, sizeof(*item)))
      return cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
    for (cpu4_training_child *child = item->children; child != NULL; child = child->next)
      if (cpu4_control_overlap(span, bytes, child, sizeof(*child)))
        return cpu4_lease_error(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_ALIAS);
  }
  return 0;
}
