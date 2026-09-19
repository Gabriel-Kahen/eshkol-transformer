/* Fixed-profile graph ownership and transaction transport only. Every numerical
 * forward/reverse schedule lives in Eshkol. M3T's standalone source is unchanged. */
#include "m3_model.h"
static void *m3t_workspace_create_original(void *);
static int64_t m3t_workspace_begin_original(void *, void *, int64_t);
static int64_t m3t_workspace_reset_original(void *);
static int64_t m3t_workspace_release_original(void *);
#define et_m3t_private_workspace_create_v1 m3t_workspace_create_original
#define et_m3t_private_workspace_begin_v1 m3t_workspace_begin_original
#define et_m3t_private_workspace_reset_v1 m3t_workspace_reset_original
#define et_m3t_private_workspace_release_v1 m3t_workspace_release_original
#include "m3t_transport.c"
#undef et_m3t_private_workspace_create_v1
#undef et_m3t_private_workspace_begin_v1
#undef et_m3t_private_workspace_reset_v1
#undef et_m3t_private_workspace_release_v1

enum { M3_GRAPH = 6, M3_LOGITS = 7 };
typedef struct m3_graph {
  record r;
  owner *model;
  initializer *original, *successor;
  et_f32_parameter *parameters[14];
  void *handles[14];
  et_f32_tensor *saved[14], *logits;
  et_i64_tensor *ids;
  uint32_t epsilon;
  int64_t positions[2];
  unsigned char keep[4];
  int mode, root_live;
  size_t leases;
  workspace *active;
} m3_graph;
typedef struct m3_logits {
  record r;
  m3_graph *graph;
  et_f32_tensor *tensor;
} m3_logits;
typedef struct m3_workspace {
  struct m3_workspace *next;
  workspace *value;
  m3_graph *graph;
  int ever_begun, owned;
} m3_workspace;
static m3_workspace *m3_workspaces;

/* An impossible cleanup defect terminates instead of misreporting committed
 * state as a retryable rejection. Tests exercise these tails with allocation off. */
static void m3_infallible(int rc) { if (rc) abort(); }
static void m3_guard_end(et_f32_scoped_guard_internal *g) {
  m3_infallible(et_f32_tensor_scoped_end_internal(g));
}
static void m3_destroy_f32(et_f32_tensor **t) {
  et_f32_tensor_error e;
  m3_infallible(et_f32_tensor_destroy_v1(t, &e));
}
static void m3_destroy_i64(et_i64_tensor **t) {
  et_i64_tensor_error e;
  m3_infallible(et_i64_tensor_destroy_v1(t, &e));
}
static m3_workspace *m3_find_workspace(workspace *w) {
  m3_workspace *s = m3_workspaces;
  while (s && s->value != w) s = s->next;
  return s;
}
static m3_workspace *m3_admit_workspace(void *p, int enrollment) {
  workspace *w = admit(p, WORKSPACE, 0);
  if (!w) return NULL;
  m3_workspace *s = m3_find_workspace(w);
  if (!s || (!s->owned && (!enrollment || s->ever_begun || w->r.state))) {
    bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_LIFECYCLE);
    return NULL;
  }
  return s;
}
static int m3_i64_unborrowed(et_i64_tensor *t) {
  et_i64_tensor_error e;
  return ierror(et_m3_private_i64_unborrowed_v1(t, &e), &e);
}
static int m3_workspace_eligible(m3_workspace *s) {
  workspace *w = s->value;
  if (w->busy || (w->r.state && w->model->active != w))
    return (int)bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_REENTRANCY);
  if (s->graph && s->graph->active != w)
    return (int)bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_LIFECYCLE);
  et_f32_scoped_guard_internal guards[14 + SLOT_COUNT + 1] = {0};
  size_t n = 0; int rc = 0;
  for (size_t i = 0; i < 14 && !rc; ++i) rc = guard_begin(w->saved[i], &guards[n++]);
  for (size_t i = 0; i < SLOT_COUNT && !rc; ++i) rc = guard_begin(w->slots[i], &guards[n++]);
  if (!rc) rc = guard_begin(w->epsilon, &guards[n++]);
  if (!rc) rc = m3_i64_unborrowed(w->ids);
  if (!rc) rc = m3_i64_unborrowed(w->positions);
  while (n) m3_guard_end(&guards[--n]);
  return rc;
}
static void m3_reset_fields(m3_workspace *s) {
  workspace *w = s->value;
  if (s->graph) s->graph->active = NULL;
  s->graph = NULL;
  if (w->model->active == w) w->model->active = NULL;
  w->r.state = 0; w->busy = 0;
  memset(w->ready, 0, sizeof(w->ready));
}

void *et_m3t_private_workspace_create_v1(void *p) {
  clear();
  /* Stage history before workspace publication, including public workspaces. */
  owner *o = admit(p, OWNER, 0);
  if (!o) return NULL;
  m3_workspace *s = allocate(sizeof(*s));
  if (!s) return NULL;
  workspace *w = m3t_workspace_create_original(p);
  if (!w) { free(s); return NULL; }
  s->value = w; s->next = m3_workspaces; m3_workspaces = s;
  return w;
}
int64_t et_m3t_private_workspace_begin_v1(void *p, void *ip, int64_t mode) {
  int64_t rc = m3t_workspace_begin_original(p, ip, mode);
  if (!rc) {
    m3_workspace *s = m3_find_workspace(p);
    if (!s || s->graph) abort();
    s->ever_begun = 1;
  }
  return rc;
}
int64_t et_m3_private_workspace_reset_v1(void *p) {
  clear();
  m3_workspace *s = m3_admit_workspace(p, 1);
  if (!s) return error_category;
  if (m3_workspace_eligible(s)) return error_category;
  s->owned = 1;
  m3_reset_fields(s);
  return 0;
}
int64_t et_m3t_private_workspace_reset_v1(void *p) {
  clear();
  workspace *w = admit(p, WORKSPACE, 0);
  if (!w) return error_category;
  m3_workspace *s = m3_find_workspace(w);
  if (s && s->owned)
    return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_LIFECYCLE);
  return m3t_workspace_reset_original(p);
}
int64_t et_m3t_private_workspace_release_v1(void *p) {
  clear();
  workspace *w = admit(p, WORKSPACE, 1);
  if (!w) return error_category;
  if (w->r.state < 0) return 0;
  m3_workspace *s = m3_find_workspace(w);
  if (!s || !s->owned) return m3t_workspace_release_original(p);
  if (m3_workspace_eligible(s)) return error_category;
  m3_reset_fields(s); w->r.state = -1;
  for (size_t i = 0; i < 14; ++i) m3_destroy_f32(&w->saved[i]);
  for (size_t i = 0; i < SLOT_COUNT; ++i) m3_destroy_f32(&w->slots[i]);
  m3_destroy_f32(&w->epsilon);
  m3_destroy_i64(&w->ids); m3_destroy_i64(&w->positions);
  return 0;
}

static int m3_graph_storage(m3_graph *g) {
  et_f32_scoped_guard_internal guards[15] = {0};
  size_t n = 0; int rc = 0;
  for (size_t i = 0; i < 14 && !rc; ++i) rc = guard_begin(g->saved[i], &guards[n++]);
  if (!rc) rc = guard_begin(g->logits, &guards[n++]);
  if (!rc) rc = m3_i64_unborrowed(g->ids);
  while (n) m3_guard_end(&guards[--n]);
  return rc;
}
static int m3_graph_constants(m3_graph *g) {
  if (g->epsilon != UINT32_C(0x3727c5ac) || g->positions[0] != 0 ||
      g->positions[1] != 1 || memcmp(g->keep, "\1\1\1\1", 4) ||
      (g->mode != 0 && g->mode != 1))
    return (int)bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_STALE);
  return 0;
}
static void m3_destroy_graph_payload(m3_graph *g) {
  for (size_t i = 0; i < 14; ++i) m3_destroy_f32(&g->saved[i]);
  m3_destroy_f32(&g->logits); m3_destroy_i64(&g->ids);
  g->model = NULL; g->original = NULL; g->successor = NULL;
  memset(g->parameters, 0, sizeof(g->parameters));
  memset(g->handles, 0, sizeof(g->handles));
}
static void m3_drop_graph(m3_graph *g) {
  --g->leases;
  if (!g->leases) { g->r.state = -1; m3_destroy_graph_payload(g); }
}
void *et_m3_private_graph_capture_v1(void *p) {
  clear();
  m3_workspace *s = m3_admit_workspace(p, 0);
  if (!s) return NULL;
  workspace *w = s->value;
  if (w->r.state != 1 || !w->ready[Z] || s->graph) {
    bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_LIFECYCLE); return NULL;
  }
  if (m3_workspace_eligible(s)) return NULL;
  m3_graph *g = allocate(sizeof(*g));
  if (!g) return NULL;
  g->model = w->model; g->original = w->model->original;
  g->successor = w->model->successor; g->mode = w->mode;
  memcpy(g->parameters, w->model->p, sizeof(g->parameters));
  memcpy(g->handles, w->model->handles, sizeof(g->handles));
  et_f32_tensor_error fe; et_i64_tensor_error ie;
  for (size_t i = 0; i < 14; ++i)
    if (f32_error(et_f32_tensor_clone_v1(w->saved[i], &g->saved[i], &fe), &fe)) goto failed;
  if (f32_error(et_f32_tensor_clone_v1(w->slots[Z], &g->logits, &fe), &fe)) goto failed;
  g->ids = new_ids(); if (!g->ids) goto failed;
  int64_t ids[2];
  if (ierror(et_i64_tensor_copy_to_v1(w->ids, ids, 2, &ie), &ie) ||
      ierror(et_i64_tensor_copy_from_v1(g->ids, ids, 2, &ie), &ie) ||
      ierror(et_i64_tensor_copy_to_v1(w->positions, g->positions, 2, &ie), &ie) ||
      f32_error(et_f32_tensor_copy_bits_to_v1(w->epsilon, &g->epsilon, 1, &fe), &fe)) goto failed;
  memcpy(g->keep, w->keep, 4);
  if (m3_graph_constants(g)) goto failed;
  g->leases = 1; g->root_live = 1;
  enroll(&g->r, M3_GRAPH);
  return g;
failed:
  m3_destroy_graph_payload(g); free(g); return NULL;
}
int64_t et_m3_private_graph_restore_v1(void *p, void *gp) {
  clear();
  m3_workspace *s = m3_admit_workspace(p, 0);
  if (!s) return error_category;
  m3_graph *g = admit(gp, M3_GRAPH, 0);
  if (!g) return error_category;
  workspace *w = s->value;
  if (w->r.state || w->model->active || g->active || w->model != g->model || s->graph)
    return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_LIFECYCLE);
  if (m3_graph_constants(g) || m3_graph_storage(g) || m3_workspace_eligible(s)) return error_category;
  for (size_t i = 0; i < 14; ++i) if (copy_tensor(w->saved[i], g->saved[i])) return error_category;
  int64_t ids[2]; et_i64_tensor_error ie; et_f32_tensor_error fe;
  if (ierror(et_i64_tensor_copy_to_v1(g->ids, ids, 2, &ie), &ie) ||
      ierror(et_i64_tensor_copy_from_v1(w->ids, ids, 2, &ie), &ie) ||
      ierror(et_i64_tensor_copy_from_v1(w->positions, g->positions, 2, &ie), &ie) ||
      f32_error(et_f32_tensor_copy_bits_from_v1(w->epsilon, &g->epsilon, 1, &fe), &fe)) return error_category;
  memcpy(w->keep, g->keep, 4);
  memset(w->ready, 0, sizeof(w->ready));
  w->mode = g->mode; w->r.state = 1; w->model->active = w;
  s->ever_begun = 1; s->graph = g; g->active = w;
  return 0;
}
int64_t et_m3_private_graph_check_primals_v1(void *gp, int64_t mode) {
  clear();
  m3_graph *g = admit(gp, M3_GRAPH, 0);
  if (!g) return error_category;
  if (mode != 0 && mode != 1) return bad(ET_M3T_INVALID_ARGUMENT, ET_M3T_CODE_SELECTOR);
  if (g->mode != mode || g->model->original != g->original ||
      g->model->successor != g->successor ||
      memcmp(g->parameters, g->model->p, sizeof(g->parameters)) ||
      memcmp(g->handles, g->model->handles, sizeof(g->handles)))
    return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_STALE);
  if (m3_graph_constants(g) || m3_graph_storage(g)) return error_category;
  for (size_t i = 0; i < 14; ++i) {
    et_f32_tensor_error e;
    if (f32_error(et_f32_parameter_validate_identity_v1(g->parameters[i], g->handles[i], &e), &e)) return error_category;
    et_f32_tensor *v = value(g->model, i); if (!v) return error_category;
    et_f32_scoped_guard_internal a = {0}, b = {0};
    int rc = guard_begin(v, &a);
    if (!rc) rc = guard_begin(g->saved[i], &b);
    if (!rc && (a.view.rank != b.view.rank || a.view.byte_length != b.view.byte_length ||
        memcmp(a.view.shape, b.view.shape, a.view.rank * sizeof(uint64_t)) ||
        memcmp(a.view.data, b.view.data, a.view.byte_length)))
      rc = (int)bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_STALE);
    m3_guard_end(&b); m3_guard_end(&a);
    if (rc) return rc;
  }
  return 0;
}
int64_t et_m3_private_graph_release_v1(void *gp) {
  clear();
  m3_graph *g = admit(gp, M3_GRAPH, 1);
  if (!g) return error_category;
  if (!g->root_live) return 0;
  if (g->active) return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_REENTRANCY);
  if (m3_graph_storage(g)) return error_category;
  g->root_live = 0; m3_drop_graph(g); return 0;
}
void *et_m3_private_model_logits_create_v1(void *gp) {
  clear();
  m3_graph *g = admit(gp, M3_GRAPH, 0);
  if (!g) return NULL;
  if (g->active || g->leases == SIZE_MAX) {
    bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_REENTRANCY); return NULL;
  }
  m3_logits *l = allocate(sizeof(*l)); if (!l) return NULL;
  et_f32_tensor_error e;
  if (f32_error(et_f32_tensor_clone_v1(g->logits, &l->tensor, &e), &e)) { free(l); return NULL; }
  l->graph = g; ++g->leases; enroll(&l->r, M3_LOGITS); return l;
}
int64_t et_m3_private_model_logits_copy_to_v1(void *lp, void *header, int64_t count) {
  clear();
  m3_logits *l = admit(lp, M3_LOGITS, 0); if (!l) return error_category;
  unsigned char *bytes = byte_payload(header, count); if (!bytes) return error_category;
  /* The accepted profile is x86-64 little-endian; I2 also admits output aliases. */
  et_f32_tensor_error e;
  return f32_error(et_f32_tensor_copy_bits_to_v1(l->tensor, (uint32_t *)(void *)bytes, 512, &e), &e);
}
int64_t et_m3_private_model_logits_release_v1(void *lp) {
  clear();
  m3_logits *l = admit(lp, M3_LOGITS, 1); if (!l) return error_category;
  if (l->r.state < 0) return 0;
  m3_graph *g = l->graph;
  if (g->active) return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_REENTRANCY);
  et_f32_scoped_guard_internal a = {0}; int rc = guard_begin(l->tensor, &a);
  m3_guard_end(&a); if (rc) return rc;
  if (m3_graph_storage(g)) return error_category;
  l->r.state = -1; l->graph = NULL;
  m3_destroy_f32(&l->tensor); m3_drop_graph(g); return 0;
}

#ifdef ET_M3_TESTING
static owner *m3_last_owner;
#endif
int64_t et_m3_private_workspace_contribute_v1(void *p, void *gp,
    int64_t mode, int64_t weight_bits, int64_t ordinal) {
  clear();
  m3_workspace *s = m3_admit_workspace(p, 0); if (!s) return error_category;
  m3_graph *g = admit(gp, M3_GRAPH, 0); if (!g) return error_category;
  workspace *w = s->value;
  if (s->graph != g || g->active != w || w->model != g->model || w->r.state != 2)
    return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_LIFECYCLE);
  if (weight_bits < 0 || (uint64_t)weight_bits > UINT32_MAX || ordinal < 0)
    return bad(ET_M3T_INVALID_ARGUMENT, ET_M3T_CODE_SELECTOR);
  if (m3_workspace_eligible(s)) return error_category;
  if (et_m3_private_graph_check_primals_v1(g, mode) ||
      et_m3t_private_workspace_check_primals_v1(w, mode)) return error_category;
  int64_t ids[2], saved_ids[2]; et_i64_tensor_error ie;
  if (ierror(et_i64_tensor_copy_to_v1(w->ids, ids, 2, &ie), &ie) ||
      ierror(et_i64_tensor_copy_to_v1(g->ids, saved_ids, 2, &ie), &ie)) return error_category;
  if (memcmp(ids, saved_ids, sizeof(ids))) return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_STALE);
  et_f32_gradient_contribution_v1 rows[14];
  for (size_t i = 0; i < 14; ++i) {
    if (!w->ready[GKEY + i]) return bad(ET_M3T_INVALID_STATE, ET_M3T_CODE_LIFECYCLE);
    rows[i] = (et_f32_gradient_contribution_v1){sizeof(rows[i]), g->parameters[i],
      w->slots[GKEY + i], (uint64_t)ordinal};
  }
  et_f32_gradient_plan *plan = NULL; et_f32_tensor_error e;
  w->busy = 1;
  int rc = et_f32_gradient_plan_prepare_v1(14, rows, (uint32_t)weight_bits, &plan, &e);
  if (rc) { w->busy = 0; return f32_error(rc, &e); }
  /* Valid disjoint stack plan/error slots, all source pins owned by this call,
   * no callback/reentry and all cleanup eligibility checked before prepare. */
  m3_infallible(et_f32_gradient_plan_commit_v1(plan, &e));
  m3_infallible(et_f32_gradient_plan_release_v1(&plan, &e));
#ifdef ET_M3_TESTING
  m3_last_owner = g->model;
#endif
  m3_reset_fields(s);
  return 0;
}

#ifdef ET_M3_TESTING
#include <stdio.h>
extern et_f32_tensor *et_m3t_test_parameter_gradient(et_f32_parameter *);
extern int64_t et_m3_test_i64_counts_v1(int64_t);
#ifdef ET_M3_PACKAGE_TESTING
extern void et_i2_test_live_builder_counts_v1(size_t *, size_t *, size_t *);
extern void et_i2_test_retired_builder_counts_v1(size_t *, size_t *, size_t *, size_t *);
#endif
static int64_t m3_test_copy(int64_t index, void *header, int64_t count, int gradient) {
  clear();
  if (!m3_last_owner || index < 0 || index >= 14 || !header || count < 0)
    return bad(ET_M3T_INVALID_ARGUMENT, ET_M3T_CODE_SELECTOR);
  et_f32_tensor *t = gradient ? et_m3t_test_parameter_gradient(m3_last_owner->p[index]) : value(m3_last_owner, (size_t)index);
  size_t bytes = 0; et_f32_tensor_error e; int64_t encoded;
  if (!t || f32_error(et_f32_tensor_byte_length_v1(t, &bytes, &e), &e)) return error_category;
  if ((uint64_t)count != bytes) return bad(ET_M3T_SHAPE_MISMATCH, ET_M3T_CODE_SHAPE);
  memcpy(&encoded, header, 8);
  if (encoded != count) return bad(ET_M3T_SHAPE_MISMATCH, ET_M3T_CODE_SHAPE);
  return f32_error(et_f32_tensor_copy_bits_to_v1(t, (uint32_t *)(void *)((unsigned char *)header + 8), bytes / 4, &e), &e);
}
int64_t et_m3_test_last_gradient_copy_v1(int64_t i, void *h, int64_t n) {
  return m3_test_copy(i, h, n, 1);
}
int64_t et_m3_test_last_parameter_copy_v1(int64_t i, void *h, int64_t n) {
  return m3_test_copy(i, h, n, 0);
}
int64_t et_m3_test_last_gradient_metadata_v1(int64_t i, int64_t field) {
  clear();
  if (!m3_last_owner || i < 0 || i >= 14 || field < 0 || field > 2) {
    bad(ET_M3T_INVALID_ARGUMENT, ET_M3T_CODE_SELECTOR); return -1;
  }
  et_f32_gradient_metadata_v1 m = {.struct_size = sizeof(m)}; et_f32_tensor_error e;
  if (f32_error(et_f32_parameter_gradient_metadata_v1(m3_last_owner->p[i], &m, &e), &e)) return -1;
  return field == 0 ? (int64_t)m.state : field == 1 ? (int64_t)m.contribution_count : (int64_t)m.normalization_weight_bits;
}
static void m3_count_tensor(et_f32_tensor *t, size_t *count, size_t *payload, size_t *metadata) {
  if (!t) return;
  size_t bytes = 0, rank = 0; et_f32_tensor_error e;
  m3_infallible(et_f32_tensor_byte_length_v1(t, &bytes, &e));
  m3_infallible(et_f32_tensor_rank_v1(t, &rank, &e));
  ++*count; *payload += bytes; *metadata += rank * (sizeof(uint64_t) + sizeof(size_t));
}
int64_t et_m3_test_report_counts_v1(void) {
  et_f32_test_live_counts_v1 live = {.struct_size = sizeof(live)};
  et_f32_test_retired_counts_v1 retired = {.struct_size = sizeof(retired)};
  et_f32_test_live_counts_snapshot_v1(&live); et_f32_test_retired_counts_snapshot_v1(&retired);
  size_t graphs = 0, logits_count = 0, graph_shells = 0, logits_shells = 0, frames = 0, workspace_shells = 0;
  size_t tensors = 0, payload = 0, metadata = 0;
  size_t transport_controls = 0;
  for (record *r = registry; r; r = r->next) {
    if (r->kind == INITIALIZER) transport_controls += sizeof(initializer);
    if (r->kind == OWNER) transport_controls += sizeof(owner);
    if (r->kind == INPUT) transport_controls += sizeof(input);
    if (r->kind == LOGITS) transport_controls += sizeof(logits);
    if (r->kind == WORKSPACE) transport_controls += sizeof(workspace);
    if (r->kind == M3_GRAPH) { ++graph_shells; if (r->state >= 0) ++graphs; }
    if (r->kind == M3_LOGITS) { ++logits_shells; if (r->state >= 0) ++logits_count; }
    if (r->state < 0) continue;
    if (r->kind == OWNER) {
      owner *o = (owner *)r;
      for (size_t i = 0; i < 14; ++i) if (o->p[i]) {
        m3_count_tensor(value(o, i), &tensors, &payload, &metadata);
        m3_count_tensor(et_m3t_test_parameter_gradient(o->p[i]), &tensors, &payload, &metadata);
      }
    } else if (r->kind == WORKSPACE) {
      workspace *w = (workspace *)r;
      for (size_t i = 0; i < 14; ++i) m3_count_tensor(w->saved[i], &tensors, &payload, &metadata);
      for (size_t i = 0; i < SLOT_COUNT; ++i) m3_count_tensor(w->slots[i], &tensors, &payload, &metadata);
      m3_count_tensor(w->epsilon, &tensors, &payload, &metadata);
    } else if (r->kind == LOGITS) {
      m3_count_tensor(((logits *)r)->tensor, &tensors, &payload, &metadata);
    } else if (r->kind == M3_LOGITS) {
      m3_count_tensor(((m3_logits *)r)->tensor, &tensors, &payload, &metadata);
    } else if (r->kind == M3_GRAPH) {
      m3_graph *g = (m3_graph *)r;
      for (size_t i = 0; i < 14; ++i) m3_count_tensor(g->saved[i], &tensors, &payload, &metadata);
      m3_count_tensor(g->logits, &tensors, &payload, &metadata);
    }
  }
  for (m3_workspace *s = m3_workspaces; s; s = s->next) { ++workspace_shells; if (s->value->r.state > 0) ++frames; }
  size_t live_copy = 0, live_reset = 0, live_decode = 0;
  size_t dead_copy = 0, dead_reset = 0, dead_decode = 0, builder_bytes = 0;
  int observed = 0;
#ifdef ET_M3_PACKAGE_TESTING
  et_i2_test_live_builder_counts_v1(&live_copy, &live_reset, &live_decode);
  et_i2_test_retired_builder_counts_v1(&dead_copy, &dead_reset, &dead_decode, &builder_bytes);
  observed = 1;
#endif
  printf("M3 native graphs=%zu logits=%zu frames=%zu graph-data=%zu "
      "m3-controls=%zu m3t-controls=%zu m3-f32-tensors=%zu m3-f32-payload=%zu m3-f32-metadata=%zu "
      "i2-tensors=%zu i2-parameters=%zu i2-owned-clones=%zu i2-borrows=%zu i2-copy-plans=%zu i2-gradient-plans=%zu "
      "i2-reset-plans=%zu i2-retired=%zu i1-tensors=%lld i1-borrows=%lld "
      "i1-payload=%lld i1-controls=%lld i1-metadata=%lld "
      "i2-retired-tensors=%zu i2-retired-parameters=%zu i2-retired-borrows=%zu "
      "i2-retired-copy-plans=%zu i2-retired-gradient-plans=%zu i2-retired-reset-plans=%zu "
      "bridge-observed=%d bridge-live-copy=%zu bridge-live-reset=%zu bridge-live-decode=%zu "
      "bridge-retired-copy=%zu bridge-retired-reset=%zu bridge-retired-decode=%zu bridge-retired-bytes=%zu\n",
      graphs, logits_count, frames,
      graphs * 6800 + logits_count * 2048,
      graph_shells * sizeof(m3_graph) + logits_shells * sizeof(m3_logits) + workspace_shells * sizeof(m3_workspace),
      transport_controls, tensors, payload, metadata, live.tensors, live.parameters, live.owned_clones, live.borrows, live.copy_plans,
      live.gradient_plans, live.reset_plans, retired.retained_control_bytes,
      (long long)et_m3_test_i64_counts_v1(0), (long long)et_m3_test_i64_counts_v1(1),
      (long long)et_m3_test_i64_counts_v1(2), (long long)et_m3_test_i64_counts_v1(3),
      (long long)et_m3_test_i64_counts_v1(4), retired.tensors, retired.parameters, retired.borrows,
      retired.copy_plans, retired.gradient_plans, retired.reset_plans,
      observed, live_copy, live_reset, live_decode, dead_copy, dead_reset, dead_decode, builder_bytes);
  return 0;
}
#endif
