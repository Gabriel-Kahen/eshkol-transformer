/* Noninstalled observations only; production constructor/lease/frame unchanged. */
#include "../../src/eshkol_transformer/g3c4_model_owner_internal.h"
#include <stdlib.h>
#include <string.h>

typedef struct image {
  void *parameters[14];
  const void *identities[14];
  uint32_t words[14][2][1024];
  size_t counts[14][2];
  et_f32_gradient_metadata_v1 metadata[14];
} image;

void *et_cpu4_consumer_destination_create(void) {
  et_f32_tensor *tensor=NULL; et_f32_tensor_error error;
  if (et_f32_tensor_create_v1(0,NULL,&tensor,&error)) return NULL;
  return tensor;
}
int64_t et_cpu4_consumer_destination_destroy(void *value) {
  et_f32_tensor *tensor=value; et_f32_tensor_error error;
  return et_f32_tensor_destroy_v1(&tensor,&error);
}
int64_t et_cpu4_consumer_destination_bits(void *value) {
  uint32_t bits=0; et_f32_tensor_error error;
  return et_f32_tensor_copy_bits_to_v1(value,&bits,1,&error) ? -1 : (int64_t)bits;
}
static int capture(void *lease,image *out) {
  memset(out,0,sizeof(*out));
  for (int64_t i=0;i<14;++i) {
    et_f32_parameter *parameter=et_g3c4_private_training_lease_parameter_v1(lease,i);
    et_f32_tensor_error error;
    if (!parameter) return 1;
    out->parameters[i]=parameter;
    out->identities[i]=et_g3c4_private_training_lease_identity_v1(lease,i);
    out->metadata[i].struct_size=sizeof(out->metadata[i]);
    if (et_f32_parameter_gradient_metadata_v1(parameter,&out->metadata[i],&error)) return 1;
    for (size_t gradient=0;gradient<2;++gradient) {
      if (gradient && out->metadata[i].state==ET_F32_GRADIENT_ABSENT) continue;
      et_f32_tensor *snapshot=NULL;
      int status=gradient ? et_f32_parameter_gradient_snapshot_v1(parameter,&snapshot,&error)
                          : et_f32_parameter_value_snapshot_v1(parameter,&snapshot,&error);
      if (status) return 1;
      status=et_f32_tensor_element_count_v1(snapshot,&out->counts[i][gradient],&error);
      if (!status && out->counts[i][gradient]<=1024)
        status=et_f32_tensor_copy_bits_to_v1(snapshot,out->words[i][gradient],out->counts[i][gradient],&error);
      else status=1;
      if (et_f32_tensor_destroy_v1(&snapshot,&error)) return 1;
      if (status) return 1;
    }
  }
  return 0;
}
void *et_cpu4_consumer_image_create(void *lease) {
  image *out=malloc(sizeof(*out));
  if (out && capture(lease,out)) { free(out);out=NULL; }
  return out;
}
int64_t et_cpu4_consumer_image_equal(void *lease,void *expected) {
  image *actual=malloc(sizeof(*actual));
  if (!actual || !expected) {free(actual);return 0;}
  int same=!capture(lease,actual) && !memcmp(actual,expected,sizeof(*actual));
  free(actual);return same;
}
void et_cpu4_consumer_image_destroy(void *value) { free(value); }

/* Real public I2 accumulation after canonical model construction/commitseal. */
int64_t et_cpu4_consumer_seed_gradients(void *lease) {
  for (int64_t i=0;i<14;++i) {
    et_f32_parameter *parameter=et_g3c4_private_training_lease_parameter_v1(lease,i);
    et_f32_tensor *numerator=NULL; et_f32_gradient_plan *plan=NULL;
    et_f32_tensor_error error; size_t count=0; uint32_t words[1024]={0};
    if (!parameter || et_f32_parameter_value_snapshot_v1(parameter,&numerator,&error)) return 1;
    int status=et_f32_tensor_element_count_v1(numerator,&count,&error);
    if (!status && count<=1024) status=et_f32_tensor_copy_bits_from_v1(numerator,words,count,&error);
    else status=1;
    et_f32_gradient_contribution_v1 contribution={sizeof(contribution),parameter,numerator,0};
    if (!status) status=et_f32_gradient_plan_prepare_v1(1,&contribution,UINT32_C(0x3f800000),&plan,&error);
    if (!status) status=et_f32_gradient_plan_commit_v1(plan,&error);
    if (plan && et_f32_gradient_plan_release_v1(&plan,&error)) status=1;
    if (et_f32_tensor_destroy_v1(&numerator,&error)) status=1;
    if (status) return 1;
  }
  return 0;
}
void *et_cpu4_consumer_borrow_begin(void *value) {
  et_f32_tensor_borrow *borrow=NULL; et_f32_tensor_error error;
  return et_f32_tensor_borrow_begin_v1(value,&borrow,&error) ? NULL : borrow;
}
int64_t et_cpu4_consumer_borrow_end(void *value) {
  et_f32_tensor_borrow *borrow=value; et_f32_tensor_error error;
  return et_f32_tensor_borrow_end_v1(&borrow,&error);
}
int64_t et_cpu4_consumer_view_word(void *value,int64_t index) {
  const et_kernel_tensor_view_v1 *view=value; int64_t word=-1;
  if (!view || index<0 || index>=4 || view->rank!=2 ||
      !view->shape || view->shape[0]!=1 || view->shape[1]!=4 ||
      view->byte_length!=32 || !view->dtype || strcmp(view->dtype,"i64") ||
      !view->data || view->offset_bytes!=0) return -1;
  memcpy(&word,(const unsigned char *)view->data+8*(size_t)index,8);
  return word;
}
