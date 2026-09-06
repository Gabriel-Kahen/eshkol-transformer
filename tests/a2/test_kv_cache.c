#include "eshkol_transformer/a2_kv_cache.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int checks, failures;
#define CHECK(x) do { checks++; if (!(x)) { failures++; fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); } } while (0)

static et_kernel_tensor_view_v1 make_view(void *data, size_t bytes,
                                          const char *dtype, size_t rank,
                                          const uint64_t *shape) {
  et_kernel_tensor_view_v1 view = {
      sizeof(view), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, rank, shape};
  return view;
}

static void expect_failure(int32_t result, const et_kernel_error *error,
                           et_kernel_error_category category) {
  CHECK(result == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code != ET_KERNEL_CODE_OK);
  CHECK(error->operation[0] != '\0');
}

static void expect_success(int32_t result, const et_kernel_error *error) {
  CHECK(result == 0);
  CHECK(error->category == ET_KERNEL_ERROR_NONE);
  CHECK(error->code == ET_KERNEL_CODE_OK);
}

static void *exclusive_end_overflow(size_t bytes) {
  return (void *)(uintptr_t)(UINTPTR_MAX - (bytes - 1u));
}

static void test_abi_and_create_failures(void) {
  et_kernel_error error;
  et_a2_kv_cache *cache = NULL;
  size_t i;
  CHECK(et_a2_kv_cache_abi_major_v1() == 1);
  CHECK(et_a2_kv_cache_abi_minor_v1() == 0);
  expect_success(et_a2_kv_cache_abi_require_v1(1u, 0u, &error), &error);
  expect_failure(et_a2_kv_cache_abi_require_v1(2u, 0u, &error), &error,
                 ET_KERNEL_ERROR_VERSION_MISMATCH);
  CHECK(error.code == ET_KERNEL_CODE_ABI_MAJOR_MISMATCH);
  CHECK(strcmp(error.operation,"a2-kv-cache.abi-require") == 0);
  CHECK(strcmp(error.message,
               "requested ABI major does not match supported major 1") == 0);
  expect_failure(et_a2_kv_cache_abi_require_v1(1u, 1u, &error), &error,
                 ET_KERNEL_ERROR_VERSION_MISMATCH);
  CHECK(error.code == ET_KERNEL_CODE_UNKNOWN_REQUIRED_FEATURE);
  CHECK(strcmp(error.message,
               "requested minimum ABI minor exceeds supported minor 0") == 0);
  expect_failure(et_a2_kv_cache_abi_require_v1(2u, 1u, &error), &error,
                 ET_KERNEL_ERROR_VERSION_MISMATCH);
  CHECK(error.code == ET_KERNEL_CODE_ABI_MAJOR_MISMATCH);
  CHECK(et_a2_kv_cache_abi_require_v1(1u,1u,NULL) ==
        ET_KERNEL_ERROR_VERSION_MISMATCH);
  expect_failure(et_a2_kv_cache_create_v1(0,1,1,1,1,&cache,&error), &error,
                 ET_KERNEL_ERROR_SHAPE_MISMATCH);
  CHECK(cache == NULL);
  expect_failure(et_a2_kv_cache_create_v1(UINT64_MAX,2,2,2,2,&cache,&error),
                 &error, ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(cache == NULL);
  for (i = 0u; i < 5u; i++) {
    et_a2_kv_cache_test_fail_alloc_after_v1(i);
    expect_failure(et_a2_kv_cache_create_v1(2,2,2,4,2,&cache,&error), &error,
                   ET_KERNEL_ERROR_INTERNAL);
    CHECK(cache == NULL);
  }
  et_a2_kv_cache_test_reset_allocator_v1();
  expect_success(et_a2_kv_cache_create_v1(1,1,1,1,1,&cache,&error), &error);
  expect_success(et_a2_kv_cache_destroy_v1(&cache,&error), &error);
  CHECK(cache == NULL);
  expect_failure(et_a2_kv_cache_destroy_v1(&cache,&error), &error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
}

static void check_initial_and_layer(et_a2_kv_cache *cache,
                                    const float *expected_k,
                                    const float *expected_v,
                                    const int64_t expected_lengths[2]) {
  et_kernel_error error;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *k = NULL, *v = NULL, *lengths = NULL;
  const et_kernel_tensor_view_v1 *mask = NULL;
  size_t i;
  expect_success(et_a2_kv_cache_read_borrow_begin_v1(cache,&borrow,&error),&error);
  expect_success(et_a2_kv_cache_read_borrow_layer_v1(
      borrow,0,&k,&v,&lengths,&mask,&error),&error);
  CHECK(k->rank == 4u && k->shape[0] == 2u && k->shape[1] == 2u);
  CHECK(k->shape[2] == 4u && k->shape[3] == 2u);
  CHECK(k->byte_length == 32u * sizeof(float));
  CHECK(v->data != k->data);
  CHECK(lengths->rank == 1u && lengths->shape[0] == 2u);
  CHECK(mask->rank == 2u && mask->shape[0] == 2u && mask->shape[1] == 4u);
  CHECK(strcmp(mask->dtype,"bool") == 0 && strcmp(mask->device,"cpu") == 0);
  CHECK(mask->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR);
  CHECK(mask->offset_bytes == 0u && mask->byte_length == 8u);
  CHECK(((const int64_t *)lengths->data)[0] == expected_lengths[0]);
  CHECK(((const int64_t *)lengths->data)[1] == expected_lengths[1]);
  for (i=0;i<8u;i++)
    CHECK(((const uint8_t *)mask->data)[i] ==
          (uint8_t)((i % 4u) < (size_t)expected_lengths[i / 4u]));
  for (i=0;i<32u;i++) {
    CHECK(((const float *)k->data)[i] == expected_k[i]);
    CHECK(((const float *)v->data)[i] == expected_v[i]);
  }
  k=v=lengths=mask=NULL;
  expect_failure(et_a2_kv_cache_read_borrow_layer_v1(
      borrow,2,&k,&v,&lengths,&mask,&error),&error,ET_KERNEL_ERROR_SHAPE_MISMATCH);
  expect_success(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error),&error);
}

static void test_transaction_and_views(void) {
  et_kernel_error error;
  et_a2_kv_cache *cache = NULL;
  et_a2_kv_cache_transaction *txn = NULL, *stale_txn;
  et_a2_kv_cache_transaction_view *view = NULL, *stale_view;
  et_a2_kv_cache_read_borrow *borrow = NULL, *stale_borrow;
  const et_kernel_tensor_view_v1 *k = NULL, *v = NULL, *lengths = NULL;
  const et_kernel_tensor_view_v1 *mask = NULL;
  uint64_t count_shape[1] = {2}, source_shape[4] = {2,2,2,2};
  int64_t counts_data[2] = {2,0};
  et_kernel_tensor_view_v1 counts = make_view(
      counts_data,sizeof(counts_data),"i64",1,count_shape);
  float source_k[16], source_v[16], before_k[16], before_v[16];
  float expected_k[32] = {0}, expected_v[32] = {0};
  et_kernel_tensor_view_v1 keys = make_view(
      source_k,sizeof(source_k),"f32",4,source_shape);
  et_kernel_tensor_view_v1 values = make_view(
      source_v,sizeof(source_v),"f32",4,source_shape);
  int64_t expected_lengths[2] = {0,0};
  size_t i;
  for(i=0;i<16u;i++){source_k[i]=(float)i+1.0f;source_v[i]=(float)i+101.0f;}
  memcpy(before_k,source_k,sizeof(source_k));memcpy(before_v,source_v,sizeof(source_v));
  expect_success(et_a2_kv_cache_create_v1(2,2,2,4,2,&cache,&error),&error);
  check_initial_and_layer(cache,expected_k,expected_v,expected_lengths);

  /* A committed borrow excludes transaction begin and destruction. */
  expect_success(et_a2_kv_cache_read_borrow_begin_v1(cache,&borrow,&error),&error);
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,2,&counts,&txn,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_failure(et_a2_kv_cache_destroy_v1(&cache,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
  stale_borrow=borrow;
  expect_success(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error),&error);
  expect_failure(et_a2_kv_cache_read_borrow_end_v1(&stale_borrow,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);

  expect_success(et_a2_kv_cache_transaction_begin_v1(
      cache,2,&counts,&txn,&error),&error);
  source_k[8]=NAN; /* Entire row 1 is unused because append_counts[1] is zero. */
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(isnan(source_k[8]));
  expect_failure(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&view,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(view==NULL);
  source_k[8]=before_k[8];
  source_v[8]=NAN;
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(isnan(source_v[8]));
  source_v[8]=before_v[8];
  expect_success(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error);
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&view,&error),&error);
  expect_success(et_a2_kv_cache_transaction_view_tensors_v1(
      view,&k,&v,&lengths,&mask,&error),&error);
  CHECK(((const int64_t *)lengths->data)[0] == 2);
  CHECK(((const int64_t *)lengths->data)[1] == 0);
  CHECK(((const float *)k->data)[0] == 1.0f);
  CHECK(((const float *)k->data)[2] == 3.0f);
  CHECK(((const float *)k->data)[16] == 0.0f);
  CHECK(mask->rank == 2u && mask->shape[0] == 2u && mask->shape[1] == 4u);
  CHECK(mask->byte_length == 8u && strcmp(mask->dtype,"bool") == 0);
  CHECK(memcmp(mask->data,"\1\1\0\0\0\0\0\0",8u) == 0);
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,1,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_failure(et_a2_kv_cache_transaction_commit_v1(&txn,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_failure(et_a2_kv_cache_transaction_abort_v1(&txn,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_failure(et_a2_kv_cache_destroy_v1(&cache,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
  stale_view=view;
  expect_success(et_a2_kv_cache_transaction_view_end_v1(&view,&error),&error);
  expect_failure(et_a2_kv_cache_transaction_view_end_v1(&stale_view,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_failure(et_a2_kv_cache_transaction_commit_v1(&txn,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,1,&keys,&values,&error),&error);
  CHECK(memcmp(source_k,before_k,sizeof(source_k))==0);
  CHECK(memcmp(source_v,before_v,sizeof(source_v))==0);
  stale_txn=txn;
  expect_success(et_a2_kv_cache_transaction_commit_v1(&txn,&error),&error);
  CHECK(txn==NULL);
  expect_failure(et_a2_kv_cache_transaction_abort_v1(&stale_txn,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expected_lengths[0]=2;
  for(i=0;i<4u;i++){expected_k[i]=source_k[i];expected_v[i]=source_v[i];}
  for(i=4u;i<8u;i++){expected_k[i+4u]=source_k[i];expected_v[i+4u]=source_v[i];}
  check_initial_and_layer(cache,expected_k,expected_v,expected_lengths);

  /* Exact-width and exact-capacity append, then abort scrubs staged bytes. */
  counts_data[0]=2;counts_data[1]=2;
  for(i=0;i<16u;i++){source_k[i]=(float)i+201.0f;source_v[i]=(float)i+301.0f;}
  expect_success(et_a2_kv_cache_transaction_begin_v1(
      cache,2,&counts,&txn,&error),&error);
  expect_success(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error);
  expect_success(et_a2_kv_cache_transaction_abort_v1(&txn,&error),&error);
  check_initial_and_layer(cache,expected_k,expected_v,expected_lengths);

  /* Admission boundaries: all zero, count>A, capacity one-over, negative. */
  counts_data[0]=0;counts_data[1]=0;
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,2,&counts,&txn,&error),&error,ET_KERNEL_ERROR_SHAPE_MISMATCH);
  counts_data[0]=3;
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,2,&counts,&txn,&error),&error,ET_KERNEL_ERROR_SHAPE_MISMATCH);
  counts_data[0]=2;counts_data[1]=0;
  expect_success(et_a2_kv_cache_transaction_begin_v1(
      cache,2,&counts,&txn,&error),&error);
  expect_success(et_a2_kv_cache_transaction_stage_layer_v1(txn,0,&keys,&values,&error),&error);
  expect_success(et_a2_kv_cache_transaction_stage_layer_v1(txn,1,&keys,&values,&error),&error);
  expect_success(et_a2_kv_cache_transaction_commit_v1(&txn,&error),&error);
  expected_lengths[0]=4;
  for(i=0u;i<4u;i++) { expected_k[i+4u]=source_k[i]; expected_v[i+4u]=source_v[i]; }
  for(i=4u;i<8u;i++) { expected_k[i+8u]=source_k[i]; expected_v[i+8u]=source_v[i]; }
  check_initial_and_layer(cache,expected_k,expected_v,expected_lengths);
  counts_data[0]=1;counts_data[1]=0;
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,&txn,&error),&error,ET_KERNEL_ERROR_SHAPE_MISMATCH);
  counts_data[0]=-1;
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,&txn,&error),&error,ET_KERNEL_ERROR_SHAPE_MISMATCH);
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,0,&counts,&txn,&error),&error,ET_KERNEL_ERROR_SHAPE_MISMATCH);

  /* Wrong-kind and forged handles reject before dereference. */
  txn=(et_a2_kv_cache_transaction *)(void *)cache;
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  txn=(et_a2_kv_cache_transaction *)(uintptr_t)0x12345u;
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  txn=NULL;
  expect_success(et_a2_kv_cache_destroy_v1(&cache,&error),&error);
}

static void test_validation_atomicity_and_failpoints(void) {
  et_kernel_error error;
  et_a2_kv_cache *cache=NULL;
  et_a2_kv_cache *other=NULL;
  et_a2_kv_cache_transaction *txn=NULL;
  et_a2_kv_cache_transaction_view *view=NULL;
  et_a2_kv_cache_read_borrow *borrow=NULL;
  uint64_t cs[1]={1}, ss[4]={1,1,1,1}, bad_ss[4]={1,1,2,1};
  int64_t count_data[1]={1}; float kd[1]={7},vd[1]={9};
  et_kernel_tensor_view_v1 counts=make_view(count_data,sizeof(count_data),"i64",1,cs);
  et_kernel_tensor_view_v1 keys=make_view(kd,sizeof(kd),"f32",4,ss);
  et_kernel_tensor_view_v1 values=make_view(vd,sizeof(vd),"f32",4,ss);
  const et_kernel_tensor_view_v1 *retained_k=NULL,*retained_v=NULL;
  const et_kernel_tensor_view_v1 *retained_lengths=NULL,*retained_mask=NULL;
  et_kernel_error *retained_error;
  float retained_storage_before[2];
  size_t i;
  expect_success(et_a2_kv_cache_create_v1(1,1,1,2,1,&cache,&error),&error);
  CHECK(et_a2_kv_cache_abi_require_v1(
      1u,0u,(et_kernel_error *)(void *)cache)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_read_borrow_begin_v1(cache,&borrow,&error),&error);
  expect_success(et_a2_kv_cache_read_borrow_layer_v1(
      borrow,0,&retained_k,&retained_v,&retained_lengths,&retained_mask,&error),
      &error);
  retained_error=(et_kernel_error *)retained_k->data;
  memcpy(retained_storage_before,retained_k->data,sizeof(retained_storage_before));
  {
    const et_kernel_tensor_view_v1 *rk=NULL,*rv=NULL,*rl=NULL,*rm=NULL;
    CHECK(et_a2_kv_cache_read_borrow_layer_v1(
        borrow,0,&rk,&rv,&rl,&rm,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(rk==NULL&&rv==NULL&&rl==NULL&&rm==NULL);
  }
  CHECK(et_a2_kv_cache_read_borrow_end_v1(
      &borrow,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(borrow!=NULL);
  CHECK(memcmp(retained_error,retained_storage_before,
               sizeof(retained_storage_before))==0);
  expect_success(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error),&error);

  CHECK(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,&txn,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(txn==NULL);
  CHECK(memcmp(retained_error,retained_storage_before,
               sizeof(retained_storage_before))==0);
  for(i=0;i<5u;i++) {
    et_a2_kv_cache_test_fail_alloc_after_v1(i);
    expect_failure(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&counts,&txn,&error),&error,ET_KERNEL_ERROR_INTERNAL);
    CHECK(txn==NULL);
  }
  et_a2_kv_cache_test_reset_allocator_v1();
  {
    struct enlarged_view {
      et_kernel_tensor_view_v1 prefix;
      uint64_t future_extension;
    } enlarged = {counts, UINT64_C(0xfeedface)};
    enlarged.prefix.struct_size = sizeof(enlarged);
    expect_success(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&enlarged.prefix,&txn,&error),&error);
    expect_success(et_a2_kv_cache_transaction_abort_v1(&txn,&error),&error);
    CHECK(enlarged.future_extension == UINT64_C(0xfeedface));
  }
  {
    int64_t aliased_counts_data[1] = {0};
    uint64_t aliased_shape[1] = {1};
    et_kernel_tensor_view_v1 aliased_counts = make_view(
        aliased_counts_data,sizeof(aliased_counts_data),"i64",1,aliased_shape);
    CHECK(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&aliased_counts,
        (et_a2_kv_cache_transaction **)(void *)&aliased_counts_data[0],
        &error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(aliased_counts_data[0]==0);
  }
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      cache,1,(const et_kernel_tensor_view_v1 *)(const void *)cache,
      &txn,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_transaction_begin_v1(cache,1,&counts,&txn,&error),&error);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(retained_error,retained_storage_before,
               sizeof(retained_storage_before))==0);
  expect_failure(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&view,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(view==NULL);
  keys.shape=bad_ss;
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_SHAPE_MISMATCH);
  keys.shape=ss; kd[0]=NAN;
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  kd[0]=7;
  values.data=kd;
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  values.data=vd;
  expect_success(et_a2_kv_cache_create_v1(1,1,1,2,1,&other,&error),&error);
  expect_success(et_a2_kv_cache_read_borrow_begin_v1(other,&borrow,&error),&error);
  {
    const et_kernel_tensor_view_v1 *other_k=NULL,*other_v=NULL,*other_l=NULL,*other_m=NULL;
    expect_success(et_a2_kv_cache_read_borrow_layer_v1(
        borrow,0,&other_k,&other_v,&other_l,&other_m,&error),&error);
    keys.data=other_k->data;
  }
  expect_success(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error),&error);
  expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  keys.data=kd;
  expect_success(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error);
  memcpy(retained_storage_before,retained_error,sizeof(retained_storage_before));
  CHECK(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&view,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(view==NULL);
  CHECK(memcmp(retained_error,retained_storage_before,
               sizeof(retained_storage_before))==0);
  CHECK(et_a2_kv_cache_transaction_commit_v1(
      &txn,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(txn!=NULL);
  CHECK(et_a2_kv_cache_transaction_abort_v1(
      &txn,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(txn!=NULL);
  expect_success(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&view,&error),&error);
  {
    const et_kernel_tensor_view_v1 *vk=NULL,*vv=NULL,*vl=NULL,*vm=NULL;
    CHECK(et_a2_kv_cache_transaction_view_tensors_v1(
        view,&vk,&vv,&vl,&vm,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(vk==NULL&&vv==NULL&&vl==NULL&&vm==NULL);
  }
  CHECK(et_a2_kv_cache_transaction_view_end_v1(
      &view,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(view!=NULL);
  expect_success(et_a2_kv_cache_transaction_view_end_v1(&view,&error),&error);
  et_a2_kv_cache_test_fail_alloc_after_v1(0);
  expect_failure(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&view,&error),&error,ET_KERNEL_ERROR_INTERNAL);
  CHECK(view==NULL);
  et_a2_kv_cache_test_reset_allocator_v1();
  expect_success(et_a2_kv_cache_transaction_abort_v1(&txn,&error),&error);
  memcpy(retained_storage_before,retained_error,sizeof(retained_storage_before));
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      cache,&borrow,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(borrow==NULL);
  CHECK(memcmp(retained_error,retained_storage_before,
               sizeof(retained_storage_before))==0);
  for(i=0u;i<3u;i++) {
    et_a2_kv_cache_test_fail_alloc_after_v1(i);
    expect_failure(et_a2_kv_cache_read_borrow_begin_v1(
        cache,&borrow,&error),&error,ET_KERNEL_ERROR_INTERNAL);
    CHECK(borrow==NULL);
  }
  et_a2_kv_cache_test_reset_allocator_v1();

  /* Error/source and output-slot aliases reject without writes. */
  {
    union { et_kernel_error error; int64_t align; } alias;
    et_kernel_tensor_view_v1 alias_counts=make_view(
        &alias.error,sizeof(int64_t),"i64",1,cs);
    memset(&alias,0,sizeof(alias));
    CHECK(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&alias_counts,&txn,&alias.error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(txn==NULL);
  }
  expect_success(et_a2_kv_cache_read_borrow_begin_v1(cache,&borrow,&error),&error);
  {
    const et_kernel_tensor_view_v1 *same=NULL;
    CHECK(et_a2_kv_cache_read_borrow_layer_v1(
        borrow,0,&same,&same,&same,&same,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(same==NULL);
  }
  expect_success(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error),&error);
  expect_success(et_a2_kv_cache_destroy_v1(&other,&error),&error);
  memcpy(retained_storage_before,retained_error,sizeof(retained_storage_before));
  CHECK(et_a2_kv_cache_destroy_v1(
      &cache,retained_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(cache!=NULL);
  CHECK(memcmp(retained_error,retained_storage_before,
               sizeof(retained_storage_before))==0);
  expect_success(et_a2_kv_cache_destroy_v1(&cache,&error),&error);
}

static void test_output_and_text_error_alias_atomicity(void) {
  union error_record {
    et_kernel_error error;
    uint64_t alignment;
    unsigned char bytes[sizeof(et_kernel_error)];
  } record, before;
  et_kernel_error error;
  et_a2_kv_cache *cache=NULL,*created=NULL;
  et_a2_kv_cache_transaction *txn=NULL;
  et_a2_kv_cache_read_borrow *borrow=NULL;
  uint64_t count_shape[1]={1};
  int64_t count_data[1]={1};
  et_kernel_tensor_view_v1 counts=make_view(
      count_data,sizeof(count_data),"i64",1,count_shape);
  int32_t result;

  /* Creator output/error overlap must not clear or format the error record. */
  memset(&record,0xa5,sizeof(record)); before=record;
  result=et_a2_kv_cache_create_v1(
      1,1,1,1,1,(et_a2_kv_cache **)(void *)&record.error.operation[0],
      &record.error);
  CHECK(result==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&record,&before,sizeof(record))==0);
  CHECK(created==NULL);

  expect_success(et_a2_kv_cache_create_v1(1,1,1,2,1,&cache,&error),&error);
  memset(&record,0x5a,sizeof(record)); before=record;
  result=et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,
      (et_a2_kv_cache_transaction **)(void *)&record.error.message[0],
      &record.error);
  CHECK(result==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&record,&before,sizeof(record))==0);
  CHECK(txn==NULL);

  /* Destroy/end handle slots containing real handles are likewise preserved. */
  memset(&record,0x3c,sizeof(record));
  memcpy(&record.error.operation[0],&cache,sizeof(cache)); before=record;
  result=et_a2_kv_cache_destroy_v1(
      (et_a2_kv_cache **)(void *)&record.error.operation[0],&record.error);
  CHECK(result==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&record,&before,sizeof(record))==0);
  expect_success(et_a2_kv_cache_read_borrow_begin_v1(cache,&borrow,&error),&error);
  memset(&record,0xc3,sizeof(record));
  memcpy(&record.error.operation[0],&borrow,sizeof(borrow)); before=record;
  result=et_a2_kv_cache_read_borrow_end_v1(
      (et_a2_kv_cache_read_borrow **)(void *)&record.error.operation[0],
      &record.error);
  CHECK(result==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&record,&before,sizeof(record))==0);

  /* Each of the four accessor slots is audited independently. */
  for(size_t alias_index=0u;alias_index<4u;alias_index++) {
    const et_kernel_tensor_view_v1 *k=NULL,*v=NULL,*lengths=NULL,*mask=NULL;
    const et_kernel_tensor_view_v1 **slots[4]={&k,&v,&lengths,&mask};
    memset(&record,(int)(0x70u+alias_index),sizeof(record)); before=record;
    slots[alias_index]=(const et_kernel_tensor_view_v1 **)(void *)
        &record.error.message[16u];
    result=et_a2_kv_cache_read_borrow_layer_v1(
        borrow,0,slots[0],slots[1],slots[2],slots[3],&record.error);
    CHECK(result==ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(memcmp(&record,&before,sizeof(record))==0);
    CHECK(k==NULL&&v==NULL&&lengths==NULL&&mask==NULL);
  }
  expect_success(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error),&error);

  /* Dtype/device text overlapping error is rejected before reading or writing. */
  memset(&record,0x91,sizeof(record));
  memcpy(&record.error.message[0],"i64",4u);
  counts.dtype=&record.error.message[0]; before=record;
  result=et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,&txn,&record.error);
  CHECK(result==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&record,&before,sizeof(record))==0);
  CHECK(txn==NULL);
  counts.dtype="i64";

  memset(&record,0x19,sizeof(record));
  memcpy(&record.error.message[0],"cpu",4u);
  counts.device=&record.error.message[0]; before=record;
  result=et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,&txn,&record.error);
  CHECK(result==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&record,&before,sizeof(record))==0);
  CHECK(txn==NULL);
  counts.device="cpu";

  /* A maximum logical text span with no NUL is bounded and diagnosed. */
  {
    static char no_nul[ET_KERNEL_MAX_SYMBOL_BYTES+1u];
    memset(no_nul,'i',sizeof(no_nul)); counts.dtype=no_nul;
    expect_failure(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&counts,&txn,&error),&error,ET_KERNEL_ERROR_DTYPE_MISMATCH);
    CHECK(txn==NULL);
    counts.dtype="i64";
  }
  expect_success(et_a2_kv_cache_destroy_v1(&cache,&error),&error);
}

static void test_output_contract_and_overflowing_spans(void) {
  et_kernel_error error;
  et_kernel_error *high_error=(et_kernel_error *)
      exclusive_end_overflow(sizeof(et_kernel_error));
  et_a2_kv_cache **high_cache_slot=(et_a2_kv_cache **)
      exclusive_end_overflow(sizeof(void *));
  et_a2_kv_cache_transaction **high_txn_slot=(et_a2_kv_cache_transaction **)
      exclusive_end_overflow(sizeof(void *));
  et_a2_kv_cache_transaction_view **high_view_slot=
      (et_a2_kv_cache_transaction_view **)
      exclusive_end_overflow(sizeof(void *));
  et_a2_kv_cache_read_borrow **high_borrow_slot=
      (et_a2_kv_cache_read_borrow **)exclusive_end_overflow(sizeof(void *));
  const et_kernel_tensor_view_v1 **high_descriptor_slot=
      (const et_kernel_tensor_view_v1 **)
      exclusive_end_overflow(sizeof(void *));
  const et_kernel_tensor_view_v1 *high_descriptor=
      (const et_kernel_tensor_view_v1 *)
      exclusive_end_overflow(sizeof(et_kernel_tensor_view_v1));
  et_a2_kv_cache *cache=NULL,*created=NULL,*cache_sentinel;
  et_a2_kv_cache_transaction *txn=NULL,*txn_sentinel;
  et_a2_kv_cache_transaction_view *lease=NULL,*view_sentinel;
  et_a2_kv_cache_read_borrow *borrow=NULL,*borrow_sentinel;
  const et_kernel_tensor_view_v1 *k=NULL,*v=NULL,*lengths=NULL,*mask=NULL;
  uint64_t one_shape[1]={1},source_shape[4]={1,1,1,1};
  int64_t count_data[1]={1};
  float key_data[1]={2.0f},value_data[1]={3.0f};
  _Alignas(void *) unsigned char dtype_slot_storage[2u*sizeof(void *)];
  _Alignas(void *) unsigned char device_slot_storage[2u*sizeof(void *)];
  unsigned char dtype_slot_before[sizeof(dtype_slot_storage)];
  unsigned char device_slot_before[sizeof(device_slot_storage)];
  et_kernel_tensor_view_v1 counts=make_view(
      count_data,sizeof(count_data),"i64",1,one_shape);
  et_kernel_tensor_view_v1 keys=make_view(
      key_data,sizeof(key_data),"f32",4,source_shape);
  et_kernel_tensor_view_v1 values=make_view(
      value_data,sizeof(value_data),"f32",4,source_shape);

  /* Error and creator-output spans whose exclusive ends overflow are raw. */
  CHECK(et_a2_kv_cache_abi_require_v1(1,0,high_error)==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_a2_kv_cache_create_v1(1,1,1,1,1,&created,high_error)==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(created==NULL);
  CHECK(et_a2_kv_cache_create_v1(1,1,1,1,1,high_cache_slot,&error)==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* NULL-on-entry is required and every failed creator preserves its slot. */
  cache_sentinel=(et_a2_kv_cache *)(uintptr_t)0x1111u;
  created=cache_sentinel;
  expect_failure(et_a2_kv_cache_create_v1(1,1,1,1,1,&created,&error),&error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(created==cache_sentinel);
  expect_success(et_a2_kv_cache_create_v1(1,1,1,2,1,&cache,&error),&error);
  CHECK(et_a2_kv_cache_destroy_v1(&cache,high_error)==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(cache!=NULL);

  txn_sentinel=(et_a2_kv_cache_transaction *)(uintptr_t)0x2222u;
  txn=txn_sentinel;
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,&txn,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(txn==txn_sentinel);
  txn=NULL;
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,&txn,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(txn==NULL);
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,high_txn_slot,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      cache,1,high_descriptor,&txn,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(txn==NULL);
  CHECK(sizeof(void *)>=4u);
  memset(dtype_slot_storage,0,sizeof(dtype_slot_storage));
  memcpy(dtype_slot_storage+sizeof(void *)-3u,"i64",3u);
  memcpy(dtype_slot_before,dtype_slot_storage,sizeof(dtype_slot_storage));
  counts.dtype=(const char *)dtype_slot_storage+sizeof(void *)-3u;
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,
      (et_a2_kv_cache_transaction **)(void *)(dtype_slot_storage+sizeof(void *)),
      &error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(dtype_slot_storage,dtype_slot_before,
               sizeof(dtype_slot_storage))==0);
  counts.dtype="i64";
  memset(device_slot_storage,0,sizeof(device_slot_storage));
  memcpy(device_slot_storage+sizeof(void *)-3u,"cpu",3u);
  memcpy(device_slot_before,device_slot_storage,sizeof(device_slot_storage));
  counts.device=(const char *)device_slot_storage+sizeof(void *)-3u;
  expect_failure(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,
      (et_a2_kv_cache_transaction **)(void *)(device_slot_storage+sizeof(void *)),
      &error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(device_slot_storage,device_slot_before,
               sizeof(device_slot_storage))==0);
  counts.device="cpu";
  {
    _Alignas(void *) unsigned char misaligned_storage[sizeof(void *)+1u]={0};
    expect_failure(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&counts,
        (et_a2_kv_cache_transaction **)(void *)(misaligned_storage+1u),&error),
        &error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(txn==NULL);
  }
  {
    et_kernel_tensor_view_v1 malformed=counts;
    malformed.shape=(const uint64_t *)exclusive_end_overflow(sizeof(uint64_t));
    CHECK(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&malformed,&txn,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(txn==NULL);
    malformed=counts;
    malformed.data=exclusive_end_overflow(sizeof(int64_t));
    expect_failure(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&malformed,&txn,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(txn==NULL);
    malformed=counts;
    malformed.dtype=(const char *)exclusive_end_overflow(
        ET_KERNEL_MAX_SYMBOL_BYTES+1u);
    expect_failure(et_a2_kv_cache_transaction_begin_v1(
        cache,1,&malformed,&txn,&error),&error,ET_KERNEL_ERROR_DTYPE_MISMATCH);
    CHECK(txn==NULL);
  }
  expect_success(et_a2_kv_cache_transaction_begin_v1(
      cache,1,&counts,&txn,&error),&error);

  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,high_descriptor,&values,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,high_descriptor,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  {
    et_kernel_tensor_view_v1 malformed=keys;
    malformed.data=exclusive_end_overflow(sizeof(float));
    expect_failure(et_a2_kv_cache_transaction_stage_layer_v1(
        txn,0,&malformed,&values,&error),&error,
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  }
  view_sentinel=(et_a2_kv_cache_transaction_view *)(uintptr_t)0x3333u;
  lease=view_sentinel;
  expect_failure(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&lease,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(lease==view_sentinel);
  lease=NULL;
  expect_failure(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&lease,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(lease==NULL); /* Overflow/error failures above left the layer unstaged. */
  expect_success(et_a2_kv_cache_transaction_stage_layer_v1(
      txn,0,&keys,&values,&error),&error);
  CHECK(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&lease,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(lease==NULL);
  CHECK(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,high_view_slot,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_transaction_view_begin_v1(
      txn,0,&lease,&error),&error);

  CHECK(et_a2_kv_cache_transaction_view_tensors_v1(
      lease,&k,&v,&lengths,&mask,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(k==NULL&&v==NULL&&lengths==NULL&&mask==NULL);
  for(size_t index=0u;index<4u;index++) {
    const et_kernel_tensor_view_v1 *a=NULL,*b=NULL,*c=NULL,*d=NULL;
    const et_kernel_tensor_view_v1 **slots[4]={&a,&b,&c,&d};
    slots[index]=high_descriptor_slot;
    CHECK(et_a2_kv_cache_transaction_view_tensors_v1(
        lease,slots[0],slots[1],slots[2],slots[3],&error)==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(a==NULL&&b==NULL&&c==NULL&&d==NULL);
  }
  k=(const et_kernel_tensor_view_v1 *)(uintptr_t)0x4444u;
  expect_failure(et_a2_kv_cache_transaction_view_tensors_v1(
      lease,&k,&v,&lengths,&mask,&error),&error,
      ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(k==(const et_kernel_tensor_view_v1 *)(uintptr_t)0x4444u);
  CHECK(v==NULL&&lengths==NULL&&mask==NULL);
  CHECK(et_a2_kv_cache_transaction_view_end_v1(
      &lease,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(lease!=NULL);
  CHECK(et_a2_kv_cache_transaction_view_end_v1(
      high_view_slot,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_transaction_view_end_v1(&lease,&error),&error);
  CHECK(et_a2_kv_cache_transaction_commit_v1(
      &txn,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(txn!=NULL);
  CHECK(et_a2_kv_cache_transaction_abort_v1(
      &txn,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(txn!=NULL);
  CHECK(et_a2_kv_cache_transaction_commit_v1(
      high_txn_slot,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_a2_kv_cache_transaction_abort_v1(
      high_txn_slot,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_transaction_commit_v1(&txn,&error),&error);

  borrow_sentinel=(et_a2_kv_cache_read_borrow *)(uintptr_t)0x5555u;
  borrow=borrow_sentinel;
  expect_failure(et_a2_kv_cache_read_borrow_begin_v1(
      cache,&borrow,&error),&error,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(borrow==borrow_sentinel);
  borrow=NULL;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      cache,&borrow,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(borrow==NULL);
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      cache,high_borrow_slot,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_read_borrow_begin_v1(cache,&borrow,&error),&error);
  k=(const et_kernel_tensor_view_v1 *)(uintptr_t)0x6666u;
  v=lengths=mask=NULL;
  expect_failure(et_a2_kv_cache_read_borrow_layer_v1(
      borrow,0,&k,&v,&lengths,&mask,&error),&error,
      ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(k==(const et_kernel_tensor_view_v1 *)(uintptr_t)0x6666u);
  CHECK(v==NULL&&lengths==NULL&&mask==NULL);
  k=v=lengths=mask=NULL;
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      borrow,0,&k,&v,&lengths,&mask,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(k==NULL&&v==NULL&&lengths==NULL&&mask==NULL);
  for(size_t index=0u;index<4u;index++) {
    const et_kernel_tensor_view_v1 *a=NULL,*b=NULL,*c=NULL,*d=NULL;
    const et_kernel_tensor_view_v1 **slots[4]={&a,&b,&c,&d};
    slots[index]=high_descriptor_slot;
    CHECK(et_a2_kv_cache_read_borrow_layer_v1(
        borrow,0,slots[0],slots[1],slots[2],slots[3],&error)==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(a==NULL&&b==NULL&&c==NULL&&d==NULL);
  }
  CHECK(et_a2_kv_cache_read_borrow_end_v1(
      &borrow,high_error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(borrow!=NULL);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(
      high_borrow_slot,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  expect_success(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error),&error);
  CHECK(et_a2_kv_cache_destroy_v1(
      high_cache_slot,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(cache!=NULL);
  expect_success(et_a2_kv_cache_destroy_v1(&cache,&error),&error);
  CHECK(cache==NULL);
}

int main(void) {
  test_abi_and_create_failures();
  test_transaction_and_views();
  test_validation_atomicity_and_failpoints();
  test_output_and_text_error_alias_atomicity();
  test_output_contract_and_overflowing_spans();
  if(failures){fprintf(stderr,"A2 KV CACHE FAIL: %d of %d checks\n",failures,checks);return 1;}
  printf("A2 KV CACHE PASS: %d ABI, transaction, lease, atomicity, and adversarial checks\n",checks);
  return 0;
}
