/* Same-TU private feature. Registry/allocation links are trusted; calls serial. */
#ifdef ET_CPU4_C4_HISTORY_PRIVATE
typedef struct cpu4_history_pending {
  et_a2_kv_cache_transaction *transaction;
  void *frame;
  et_a2_kv_cache *candidate;
  uint64_t epoch;
  uint32_t operation, capacity;
} cpu4_history_pending;
typedef struct cpu4_history_node {
  struct cpu4_history_node *next;
  et_g3c4_context_internal *context;
  et_g3c4_model_owner_internal *original_owner;
  et_a2_kv_cache *current_cache;
  uint32_t state, kind;
  int64_t policy[6], rng[4];
  cpu4_history_record history;
  et_f32_parameter *parameters[14];
  const void *handles[14];
  cpu4_history_pending pending;
} cpu4_history_node;
_Static_assert(sizeof(cpu4_history_pending) == 40u, "CPU4 pending extent");
_Static_assert(offsetof(cpu4_history_node, pending) == 424u,
               "CPU4 pending offset");
_Static_assert(sizeof(cpu4_history_node) == 464u, "CPU4 provenance extent");
static cpu4_history_node *cpu4_history_registry;
#ifdef ET_CPU4_C4_HISTORY_TESTING
static size_t cpu4_history_allocation_limit = SIZE_MAX;
static size_t cpu4_history_successful_allocations;
#endif
static void *cpu4_history_calloc(size_t n) {
#ifdef ET_CPU4_C4_HISTORY_TESTING
  if (cpu4_history_successful_allocations >= cpu4_history_allocation_limit)
    return NULL;
#endif
  void *p = calloc(1u,n);
#ifdef ET_CPU4_C4_HISTORY_TESTING
  if (p != NULL) ++cpu4_history_successful_allocations;
#endif
  return p;
}
static int cpu4_capture_overlap(const void *, size_t);
static int cpu4_history_overlap(const void *, size_t);
static int cpu4_history_fail(void) {
  return (int)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
}
static int cpu4_history_output_alias(const void *p,size_t n) {
  return cpu4_capture_overlap(p,n) || cpu4_history_overlap(p,n);
}
static int cpu4_history_zero(const void *p, size_t n) {
  const unsigned char *b = p;
  for (size_t i = 0; i < n; ++i) if (b[i] != 0u) return 0;
  return 1;
}
static int cpu4_history_span(const void *a, size_t n, const void *b, size_t m) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return n != 0u && m != 0u &&
      (x > UINTPTR_MAX - (n - 1u) || y > UINTPTR_MAX - (m - 1u) ||
       (x <= y ? y - x < n : x - y < m));
}
static int cpu4_history_overlap(const void *p, size_t n) {
  for (cpu4_history_node *h = cpu4_history_registry; h; h = h->next)
    if (cpu4_history_span(p, n, h, sizeof(*h))) return 1;
  return 0;
}
static cpu4_history_node *cpu4_history_find(const void *c) {
  cpu4_history_node *found = NULL;
  for (cpu4_history_node *h = cpu4_history_registry; h; h = h->next)
    if (h->context == c) {
      if (found != NULL) return NULL;
      found = h;
    }
  return found;
}
static int cpu4_history_record_valid(const cpu4_history_record *r,
                                      int64_t eos) {
  uint32_t p = 0, g = 0, m = 0, mask = 0;
  if (r->state != 1u || r->count > 4u) return 0;
  for (uint32_t i = 0; i < 4u; ++i) {
    if (i >= r->count) {
      if (r->ids[i] != 0 || r->origin[i] != 0u) return 0;
      continue;
    }
    if (r->ids[i] < 0 || r->ids[i] > 255) return 0;
    if (r->origin[i] == 1u) ++p;
    else if (r->origin[i] == 2u) {
      ++g;
      if (r->ids[i] == eos) mask |= UINT32_C(1) << i;
    } else if (r->origin[i] == 3u) ++m;
    else return 0;
  }
  return p == r->prompt_count && g == r->generated_count &&
      m == r->manual_count && mask == r->generated_eos_mask;
}
static cpu4_history_node *cpu4_history_auth(
    const et_g3c4_context_internal *c) {
  cpu4_history_node *h = cpu4_history_find(c);
  if (!h || h->kind != c->generator_kind ||
      h->state != ET_G3C4_CONTEXT_STATE(c)) goto bad;
  if (h->state == ET_G3C4_CONTEXT_DEAD) {
    if (h->original_owner != NULL || h->current_cache != NULL ||
        !cpu4_history_zero(h->policy, sizeof(h->policy)) ||
        !cpu4_history_zero(h->rng, sizeof(h->rng)) ||
        !cpu4_history_zero(&h->history, sizeof(h->history)) ||
        !cpu4_history_zero(h->parameters, sizeof(h->parameters)) ||
        !cpu4_history_zero(h->handles, sizeof(h->handles)) ||
        !cpu4_history_zero(&h->pending, sizeof(h->pending)) ||
        !cpu4_history_zero(&c->cpu4_history, sizeof(c->cpu4_history))) goto bad;
    return h;
  }
  if (h->state != ET_G3C4_CONTEXT_LIVE || h->original_owner != c->owner ||
      h->current_cache != c->cache || c->owner == NULL ||
      memcmp(c->generator_policy, h->policy, sizeof(h->policy)) ||
      memcmp(c->generator_rng_words, h->rng, sizeof(h->rng)) ||
      memcmp(&c->cpu4_history, &h->history, sizeof(h->history)) ||
      !cpu4_history_record_valid(&h->history, h->policy[5]) ||
      memcmp(c->owner->parameters, h->parameters, sizeof(h->parameters)) ||
      memcmp(c->owner->handles, h->handles, sizeof(h->handles))) goto bad;
  if (h->pending.operation == 0u) {
    if (!cpu4_history_zero(&h->pending, sizeof(h->pending))) goto bad;
  } else {
    if (h->pending.operation > 4u ||
        h->pending.epoch != h->history.commit_epoch ||
        (h->pending.capacity != 2u && h->pending.capacity != 4u)) goto bad;
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
    if (h->pending.operation == 2u &&
        h->pending.transaction != c->token_frame_transaction) goto bad;
#endif
#ifdef ET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE
    if (h->pending.operation >= 3u && h->pending.frame != c->manual_frame)
      goto bad;
#ifdef ET_G3C4_MANUAL_A2_PRIVATE
    if (h->pending.operation >= 3u && h->pending.frame != NULL &&
        (h->pending.candidate != c->manual_frame->a2_candidate ||
         h->pending.transaction != c->manual_frame->a2_transaction)) goto bad;
#endif
#endif
  }
  return h;
bad:
  (void)cpu4_history_fail();
  return NULL;
}
static int cpu4_history_cache_check(const et_g3c4_context_internal *c,
                                    cpu4_history_node *h, uint32_t *capacity) {
  et_a2_kv_cache_read_borrow *b = NULL;
  const et_kernel_tensor_view_v1 *k = NULL, *v = NULL, *l = NULL, *m = NULL;
  et_kernel_error e;
  et_g3c4_error_state_internal first;
  int bad = 0;
  if (et_g3c4_capture_kernel(et_a2_kv_cache_read_borrow_begin_v1(
          c->cache, &b, &e), &e) != 0) return (int)et_g3c4_error_state.category;
  if (et_g3c4_capture_kernel(et_a2_kv_cache_read_borrow_layer_v1(
          b, 0u, &k, &v, &l, &m, &e), &e) != 0) goto fail;
  if (!k || !v || !l || !m || !k->shape || !v->shape || !l->shape || !m->shape ||
      !k->data || !v->data || !l->data || !m->data ||
      k->struct_size != sizeof(*k) || v->struct_size != sizeof(*v) ||
      l->struct_size != sizeof(*l) || m->struct_size != sizeof(*m) ||
      !k->dtype || !v->dtype || !l->dtype || !m->dtype ||
      strcmp(k->dtype,"f32") || strcmp(v->dtype,"f32") ||
      strcmp(l->dtype,"i64") || strcmp(m->dtype,"bool") ||
      !k->device || !v->device || !l->device || !m->device ||
      strcmp(k->device,"cpu") || strcmp(v->device,"cpu") ||
      strcmp(l->device,"cpu") || strcmp(m->device,"cpu") ||
      k->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      v->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      l->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      m->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      k->offset_bytes || v->offset_bytes || l->offset_bytes || m->offset_bytes ||
      k->rank != 4u || v->rank != 4u || l->rank != 1u || m->rank != 2u ||
      k->shape[0] != 1u || k->shape[1] != 2u || k->shape[3] != 2u ||
      (k->shape[2] != 2u && k->shape[2] != 4u) ||
      memcmp(k->shape, v->shape, 4u*sizeof(uint64_t)) ||
      k->byte_length != k->shape[2]*4u*sizeof(float) ||
      v->byte_length != k->byte_length || l->shape[0] != 1u ||
      l->byte_length != sizeof(int64_t) || m->shape[0] != 1u ||
      m->shape[1] != k->shape[2] || m->byte_length != k->shape[2] ||
      *(const int64_t *)l->data != (int64_t)h->history.count ||
      h->history.count > k->shape[2]) bad = 1;
  if (!bad) for (uint32_t i=0; i < (uint32_t)k->shape[2]; ++i)
    if (((const uint8_t *)m->data)[i] != (uint8_t)(i<h->history.count)) bad=1;
  if (bad) { (void)cpu4_history_fail(); goto fail; }
  *capacity = (uint32_t)k->shape[2];
  if (et_a2_kv_cache_read_borrow_end_v1(&b,&e) != 0) abort();
  return 0;
fail:
  first=et_g3c4_error_snapshot_internal();
  if (et_a2_kv_cache_read_borrow_end_v1(&b,&e) != 0) abort();
  et_g3c4_error_restore_internal(first);
  return (int)et_g3c4_error_state.category;
}
/* op0 authenticates a query; 1/3 replace, 2/4 append. Never arms custody. */
static cpu4_history_node *cpu4_history_preflight(
    const et_g3c4_context_internal *c, uint32_t op, uint32_t n,
    uint32_t target_capacity, uint64_t epochs) {
  cpu4_history_node *h=cpu4_history_auth(c);
  uint32_t cap=0;
  if (!h || h->state != ET_G3C4_CONTEXT_LIVE) return NULL;
  if (h->pending.operation == 2u && h->pending.transaction != NULL)
    cap=h->pending.capacity; /* exact transaction owns the pre-read cache */
  else if (cpu4_history_cache_check(c,h,&cap) != 0) return NULL;
  if (!op) return h; /* queries retain their old full/MAX admission */
  if (h->pending.operation != 0u && h->pending.operation != op) {
    (void)cpu4_history_fail(); return NULL;
  }
  if ((op == 2u || op == 4u) && !et_g3c4_prefill_binding_matches_pins(c)) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE,ET_G3C4_CODE_STALE_BINDING);
    return NULL;
  }
  if (h->history.commit_epoch > UINT64_MAX-epochs) goto bad;
  if (op == 1u || op == 3u) {
    if (!n || n>target_capacity || target_capacity>4u) goto bad;
  } else if (op == 2u || op == 4u) {
    if (h->history.count>=cap || h->history.count>=4u ||
        (op==4u && h->history.count!=1u) || (op==2u && cap!=4u)) goto bad;
  } else goto bad;
  return h;
bad:
  (void)et_g3c4_fail(ET_G3C4_INVALID_STATE,ET_G3C4_CODE_LIFECYCLE);
  return NULL;
}
static int cpu4_history_staged_check(cpu4_history_node *h,
    et_a2_kv_cache_transaction *transaction, uint32_t count, uint32_t capacity) {
  et_a2_kv_cache_transaction_view *view=NULL;
  const et_kernel_tensor_view_v1 *k=NULL,*v=NULL,*l=NULL,*m=NULL;
  et_kernel_error e;
  et_g3c4_error_state_internal first;
  if (h->pending.transaction != transaction || transaction == NULL)
    return cpu4_history_fail();
  if (et_g3c4_capture_kernel(et_a2_kv_cache_transaction_view_begin_v1(
      transaction,0u,&view,&e),&e) != 0) return (int)et_g3c4_error_state.category;
  if (et_g3c4_capture_kernel(et_a2_kv_cache_transaction_view_tensors_v1(
      view,&k,&v,&l,&m,&e),&e) != 0) goto fail;
  if (!k || !v || !l || !m || !k->shape || !v->shape || !l->shape || !m->shape ||
      !k->data || !v->data || !l->data || !m->data ||
      k->struct_size != sizeof(*k) || v->struct_size != sizeof(*v) ||
      l->struct_size != sizeof(*l) || m->struct_size != sizeof(*m) ||
      k->rank!=4u || v->rank!=4u || l->rank!=1u || m->rank!=2u ||
      !k->dtype || !v->dtype || !l->dtype || !m->dtype ||
      strcmp(k->dtype,"f32") || strcmp(v->dtype,"f32") ||
      strcmp(l->dtype,"i64") || strcmp(m->dtype,"bool") ||
      !k->device || !v->device || !l->device || !m->device ||
      strcmp(k->device,"cpu") || strcmp(v->device,"cpu") ||
      strcmp(l->device,"cpu") || strcmp(m->device,"cpu") ||
      k->layout!=ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      v->layout!=ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      l->layout!=ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      m->layout!=ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      k->offset_bytes || v->offset_bytes || l->offset_bytes || m->offset_bytes ||
      k->shape[0]!=1u || k->shape[1]!=2u || k->shape[2]!=capacity ||
      k->shape[3]!=2u || memcmp(k->shape,v->shape,4u*sizeof(uint64_t)) ||
      k->byte_length!=capacity*4u*sizeof(float) || v->byte_length!=k->byte_length ||
      l->shape[0]!=1u || l->byte_length!=sizeof(int64_t) ||
      *(const int64_t *)l->data!=(int64_t)count || count>capacity ||
      m->shape[0]!=1u || m->shape[1]!=capacity || m->byte_length!=capacity) {
    (void)cpu4_history_fail(); goto fail;
  }
  for (uint32_t i=0;i<capacity;++i)
    if (((const uint8_t *)m->data)[i]!=(uint8_t)(i<count)) {
      (void)cpu4_history_fail(); goto fail;
    }
  if (et_a2_kv_cache_transaction_view_end_v1(&view,&e)!=0) abort();
  return 0;
fail:
  first=et_g3c4_error_snapshot_internal();
  if (et_a2_kv_cache_transaction_view_end_v1(&view,&e)!=0) abort();
  et_g3c4_error_restore_internal(first);
  return (int)et_g3c4_error_state.category;
}
static int cpu4_history_effect(const et_g3c4_context_internal *c) {
  cpu4_history_node *h=cpu4_history_auth(c);
  if (!h) return (int)et_g3c4_error_state.category;
#ifdef ET_G3C4_MANUAL_FRAME_COMMIT_PRIVATE
  if (h->pending.frame != NULL &&
      ((const et_g3c4_manual_frame_internal *)h->pending.frame)->publication_state == 3u)
    return (int)et_g3c4_fail(ET_G3C4_INVALID_STATE,ET_G3C4_CODE_LIFECYCLE);
#endif
  if (!h->pending.operation) return 0;
  return cpu4_history_preflight(c,h->pending.operation,1u,
      h->pending.capacity,1u) == NULL ? (int)et_g3c4_error_state.category : 0;
}
static void cpu4_history_start(cpu4_history_node *h, uint32_t op, uint32_t cap) {
  h->pending.epoch=h->history.commit_epoch;
  h->pending.operation=op; h->pending.capacity=cap;
}
static void cpu4_history_pending_clear(cpu4_history_node *h) {
  memset(&h->pending,0,sizeof(h->pending));
}
static void cpu4_history_publish(cpu4_history_node *h,
    et_g3c4_context_internal *c, const int64_t *ids, uint32_t n,
    uint32_t origin, int replace) {
  cpu4_history_record *r=&c->cpu4_history;
  if (replace) {
    uint64_t epoch=r->commit_epoch;
    memset(r,0,sizeof(*r)); r->state=1u; r->commit_epoch=epoch;
  }
  for (uint32_t i=0;i<n;++i) {
    uint32_t at=r->count++;
    r->ids[at]=ids[i]; r->origin[at]=origin;
    if (origin==1u) ++r->prompt_count;
    else if (origin==2u) {
      ++r->generated_count;
      if (ids[i]==h->policy[5]) r->generated_eos_mask|=UINT32_C(1)<<at;
    } else ++r->manual_count;
  }
  ++r->commit_epoch;
  h->history=*r; h->current_cache=c->cache;
  memcpy(h->rng,c->generator_rng_words,sizeof(h->rng));
  if (h->pending.operation >= 3u && h->pending.frame != NULL) {
    /* COMMITTED still owns its frame until finish actually frees it. */
    h->pending.transaction=NULL;
    h->pending.candidate=NULL;
    h->pending.epoch=r->commit_epoch;
  } else cpu4_history_pending_clear(h);
}
static cpu4_history_node *cpu4_history_allocate(void) {
  cpu4_history_node *h=cpu4_history_calloc(sizeof(*h));
  if (!h) (void)et_g3c4_fail(ET_G3C4_INTERNAL,ET_G3C4_CODE_ALLOCATION);
  return h;
}
static void cpu4_history_enroll(cpu4_history_node *h,
                                et_g3c4_context_internal *c) {
  h->context=c; h->original_owner=c->owner; h->current_cache=c->cache;
  h->state=ET_G3C4_CONTEXT_LIVE; h->kind=c->generator_kind;
  memcpy(h->policy,c->generator_policy,sizeof(h->policy));
  memcpy(h->rng,c->generator_rng_words,sizeof(h->rng));
  memcpy(h->parameters,c->owner->parameters,sizeof(h->parameters));
  memcpy(h->handles,c->owner->handles,sizeof(h->handles));
  c->cpu4_history.state=1u; h->history=c->cpu4_history;
  h->next=cpu4_history_registry; cpu4_history_registry=h;
}
static void cpu4_history_close(cpu4_history_node *h,
                               et_g3c4_context_internal *c) {
  memset(&c->cpu4_history,0,sizeof(c->cpu4_history));
  h->original_owner=NULL; h->current_cache=NULL;
  memset(h->policy,0,sizeof(h->policy)); memset(h->rng,0,sizeof(h->rng));
  memset(&h->history,0,sizeof(h->history));
  memset(h->parameters,0,sizeof(h->parameters));
  memset(h->handles,0,sizeof(h->handles));
  cpu4_history_pending_clear(h); h->state=ET_G3C4_CONTEXT_DEAD;
}
#endif
