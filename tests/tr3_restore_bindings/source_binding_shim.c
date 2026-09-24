#include <stdint.h>
#include <stdlib.h>

static int optimizer_identity;
static int plan_identity;
static int builder_identity;
static int parameter_identity;
static int handle_identity;
static int stage_identities[64];
static int64_t last_category;
static int64_t request_entries;
static int64_t fail_prepare;
static void *last_authority;

void *et_tr3_c_binding_test_optimizer_v1(void) { return &optimizer_identity; }
void *et_tr3_c_binding_test_parameter_v1(void) { return &parameter_identity; }
void *et_tr3_c_binding_test_handle_v1(void) { return &handle_identity; }
void *et_tr3_c_binding_test_stage_v1(int64_t index) {
  if (index < 0 || index >= 64) return NULL;
  return &stage_identities[index];
}
void et_tr3_c_binding_test_fail_prepare_v1(int64_t enabled) {
  fail_prepare = enabled;
}
int64_t et_tr3_c_binding_test_exact_authority_v1(void) {
  return last_authority == &plan_identity ? 1 : 0;
}

int64_t et_tr3_c_private_restore_last_category_v1(void) {
  return last_category;
}
int64_t et_tr3_c_private_restore_request_initialize_v1(
    void *request, int64_t target, int64_t receiver, int64_t clip_kind,
    int64_t clip_bits, int64_t schedule_kind, int64_t minimum_bits,
    int64_t warmup, int64_t total) {
  (void)target;
  (void)receiver;
  (void)clip_kind;
  (void)clip_bits;
  (void)schedule_kind;
  (void)minimum_bits;
  (void)warmup;
  (void)total;
  if (request == NULL) return -1;
  request_entries = 0;
  last_category = 0;
  return 0;
}
int64_t et_tr3_c_private_restore_request_append14_v1(
    void *request, int64_t learning_rate, int64_t beta1, int64_t beta2,
    int64_t epsilon, int64_t weight_decay, void *avg, void *sq) {
  (void)learning_rate;
  (void)beta1;
  (void)beta2;
  (void)epsilon;
  (void)weight_decay;
  if (request == NULL || avg == NULL || sq == NULL || request_entries >= 14) {
    last_category = 1;
    return -1;
  }
  request_entries++;
  return 0;
}
void *et_tr3_c_private_o2_restore_prepare_v1(void *optimizer, void *request) {
  if (fail_prepare || optimizer != &optimizer_identity || request == NULL ||
      request_entries != 14) {
    last_category = 2;
    return NULL;
  }
  return &plan_identity;
}
int64_t et_tr3_c_private_o2_restore_append_v1(void *plan, void *builder) {
  return plan == &plan_identity && builder == &builder_identity ? 0 : -1;
}
int64_t et_tr3_c_private_o2_restore_check_v1(void *plan) {
  return plan == &plan_identity ? 0 : -1;
}
void et_tr3_c_private_o2_restore_commit_native_v1(void *plan) {
  if (plan != &plan_identity) abort();
}
void et_tr3_c_private_o2_restore_finalize_v1(void *plan) {
  if (plan != &plan_identity) abort();
}
int64_t et_tr3_c_private_o2_restore_abort_v1(void *plan) {
  return plan == &plan_identity ? 0 : -1;
}
void *et_tr3_c_private_i2_restore_create_v1(void *authority) {
  last_authority = authority;
  return authority == &plan_identity ? &builder_identity : NULL;
}
int64_t et_tr3_c_private_i2_restore_append_parameter_v1(
    void *builder, void *authority, void *parameter, void *handle,
    void *source) {
  last_authority = authority;
  return builder == &builder_identity && authority == &plan_identity &&
                 parameter == &parameter_identity &&
                 handle == &handle_identity && source != NULL
             ? 0
             : -1;
}
int64_t et_tr3_c_private_i2_restore_prepare_v1(void *builder,
                                               void *authority) {
  last_authority = authority;
  return builder == &builder_identity && authority == &plan_identity ? 0 : -1;
}
int64_t et_tr3_c_private_i2_restore_commit_checked_v1(void *builder,
                                                      void *authority) {
  return builder == &builder_identity && authority == &plan_identity ? 0 : -1;
}
int64_t et_tr3_c_private_i2_restore_abort_checked_v1(void *builder,
                                                     void *authority) {
  last_authority = authority;
  return builder == &builder_identity && authority == &plan_identity ? 0 : -1;
}
void et_tr3_c_private_i2_owned_release_checked_v1(void *owned) {
  if (owned == NULL) abort();
}
