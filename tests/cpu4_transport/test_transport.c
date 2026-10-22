#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_CPU4_TRAINING_PRIVATE 1
#define ET_CPU4_TRANSPORT_TESTING 1
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include "../../src/eshkol_transformer/cpu4_training_transport.c"
#include "reference.h"
#include <stdio.h>

extern void et_m3t_test_k1_fail_after(size_t);
extern void et_m3t_test_k1_allocator_reset(void);
static size_t checks;
#define CHECK(x) do {++checks;if(!(x)){fprintf(stderr,"FAIL line%d: %s (d%lld c%lld e%lld)\n",__LINE__,#x,(long long)c4t_domain,(long long)c4t_category,(long long)c4t_code);exit(1);}}while(0)
#define OK(x) CHECK((x)==0)
#define NO(x) CHECK((x)!=0)
static unsigned char identities[2][14],foreign;
static double max_logits_error,max_gradient_error;
static size_t parameter_words(size_t i) {return cpu4_ref_parameter_offsets[i+1]-cpu4_ref_parameter_offsets[i];}
static et_g3c4_model_owner_internal *model(size_t index) {
 et_g3c4_model_owner_internal *o=et_g3c4_private_model_owner_create_seeded_v1(1729);CHECK(o);
 for(int64_t i=0;i<14;i++) {
  OK(et_g3c4_private_model_owner_bind_v1(o,i,&identities[index][i]));
  OK(et_g3c4_private_model_owner_initialize_v1(o,i));
 }
 OK(et_g3c4_private_model_owner_prepare_seal_v1(o));OK(et_g3c4_private_model_owner_commit_seal_v1(o));
 for(size_t i=0;i<14;i++) {et_f32_tensor_error e;
  OK(et_f32_tensor_copy_bits_from_v1(o->parameters[i]->value,cpu4_ref_parameter_bits+cpu4_ref_parameter_offsets[i],parameter_words(i),&e));
 }
 return o;
}
typedef struct state {
 uint32_t values[1192],gradients[1192];et_f32_gradient_metadata_v1 metadata[14];
 et_f32_parameter *parameters[14];const void *handles[14];
} state;
static void capture(et_g3c4_model_owner_internal *o,state *s) {
 memset(s,0,sizeof(*s));memcpy(s->parameters,o->parameters,sizeof(s->parameters));memcpy(s->handles,o->handles,sizeof(s->handles));
 for(size_t i=0;i<14;i++) {et_f32_tensor_error e;size_t off=cpu4_ref_parameter_offsets[i],n=parameter_words(i);
  OK(et_f32_tensor_copy_bits_to_v1(o->parameters[i]->value,s->values+off,n,&e));
  OK(et_f32_tensor_copy_bits_to_v1(o->parameters[i]->gradient,s->gradients+off,n,&e));
  s->metadata[i].struct_size=sizeof(s->metadata[i]);OK(et_f32_parameter_gradient_metadata_v1(o->parameters[i],&s->metadata[i],&e));
 }
}
static void unchanged(et_g3c4_model_owner_internal *o,const state *s) {state after;capture(o,&after);CHECK(!memcmp(s,&after,sizeof(after)));}
static void near(float actual,double expected,double atol,double rtol,double *maximum) {
 double error=fabs((double)actual-expected);if(error>*maximum)*maximum=error;
 CHECK(isfinite(actual)&&error<=atol+rtol*fabs(expected));
}
static const int forward_roles[21][2]={
 {EMBEDDING,0},{EMBEDDING,1},{RESIDUAL,0},{NORM,0},{LINEAR,0},{LINEAR,1},{LINEAR,2},
 {SPLIT,0},{SPLIT,1},{SPLIT,2},{ATTENTION,0},{MERGE,0},{LINEAR,3},{RESIDUAL,1},
 {NORM,1},{LINEAR,4},{GELU,0},{LINEAR,5},{RESIDUAL,2},{NORM,2},{LINEAR,6}};
static const int reverse_roles[25][2]={
 {LINEAR,6},{NORM,2},{RESIDUAL,2},{LINEAR,5},{GELU,0},{LINEAR,4},{NORM,1},{SUM,2},
 {RESIDUAL,1},{LINEAR,3},{MERGE,0},{ATTENTION,0},{SPLIT,0},{SPLIT,1},{SPLIT,2},
 {LINEAR,0},{LINEAR,1},{LINEAR,2},{SUM,0},{NORM,0},{SUM,1},{RESIDUAL,0},
 {EMBEDDING,0},{EMBEDDING,1},{SUM,3}};
static void forward(c4t_workspace *w) {
 for(size_t i=0;i<21;i++)OK(et_cpu4_private_workspace_role_v1(w,forward_roles[i][0],forward_roles[i][1],0));
}
static void reverse(c4t_workspace *w) {
 for(size_t i=0;i<25;i++)OK(et_cpu4_private_workspace_role_v1(w,reverse_roles[i][0],reverse_roles[i][1],1));
}
typedef struct bytevector {int64_t length;unsigned char bytes[4096];} bytevector;
static void fill(bytevector *v,const uint32_t *bits) {
 v->length=4096;for(size_t i=0;i<1024;i++)for(size_t j=0;j<4;j++)v->bytes[i*4+j]=(unsigned char)(bits[i]>>(8*j));
}
static void compare_gradients(c4t_workspace *w) {
 for(size_t i=0;i<14;i++) {et_f32_tensor_error e;uint32_t bits[1024];size_t n=parameter_words(i),off=cpu4_ref_parameter_offsets[i];
  CHECK(w->ready[GKEY+i]);OK(et_f32_tensor_copy_bits_to_v1(w->slots[GKEY+i],bits,n,&e));
  for(size_t j=0;j<n;j++){float f;memcpy(&f,&bits[j],4);near(f,cpu4_ref_gradients[off+j],CPU4_REF_GRADIENT_ATOL,CPU4_REF_GRADIENT_RTOL,&max_gradient_error);}
 }
}
static void release_frame(void *lease,void *input,void *output,void *upstream,void *workspace) {
 OK(et_cpu4_private_input_release_v1(input));OK(et_cpu4_private_input_release_v1(input));
 OK(et_cpu4_private_logits_release_v1(output));OK(et_cpu4_private_logits_release_v1(upstream));
 OK(et_cpu4_private_workspace_release_v1(workspace));OK(et_cpu4_private_workspace_release_v1(workspace));
 OK(et_g3c4_private_training_lease_leave_v1(lease));
}
int main(void) {
 et_g3c4_model_owner_internal *a=model(0),*b=model(1);
 void *lease=et_g3c4_private_training_lease_create_v1(a),*other=et_g3c4_private_training_lease_create_v1(b);CHECK(lease&&other);
 state before;capture(a,&before);
 et_f32_test_live_counts_v1 initial={.struct_size=sizeof(initial)};et_f32_test_live_counts_snapshot_v1(&initial);
 CHECK(et_cpu4_private_input_create_v1(lease,3,197,3,41)==NULL);
 OK(et_g3c4_private_training_lease_enter_v1(lease,1));OK(et_g3c4_private_training_lease_enter_v1(other,1));
 CHECK(et_cpu4_private_input_create_v1(&foreign,3,197,3,41)==NULL);
 CHECK(et_cpu4_private_input_create_v1(lease,3,197,3,256)==NULL);
 c4t_alloc_remaining=0;CHECK(et_cpu4_private_input_create_v1(lease,3,197,3,41)==NULL);c4t_alloc_remaining=SIZE_MAX;
 c4t_input *input=et_cpu4_private_input_create_v1(lease,3,197,3,41),*mixed=et_cpu4_private_input_create_v1(other,3,197,3,41);CHECK(input&&mixed);
 c4t_input copied=*input;NO(et_cpu4_private_input_release_v1(&copied));
 c4t_logits *output=et_cpu4_private_logits_create_v1(lease),*upstream=et_cpu4_private_logits_create_v1(lease),*mixed_logits=et_cpu4_private_logits_create_v1(other);CHECK(output&&upstream&&mixed_logits);
 CHECK(et_cpu4_private_workspace_create_v1(lease,mixed,1)==NULL);
 CHECK(et_cpu4_private_workspace_create_v1(lease,input,2)==NULL);
 et_m3t_test_k1_fail_after(0);CHECK(et_cpu4_private_workspace_create_v1(lease,input,1)==NULL);et_m3t_test_k1_allocator_reset();unchanged(a,&before);
 for(size_t cut=0;cut<5;cut++) {
  size_t allowed=(size_t[]){0,1,7,60,120}[cut];
  et_f32_tensor_test_fail_alloc_after_v1(allowed);CHECK(et_cpu4_private_workspace_create_v1(lease,input,1)==NULL);et_f32_tensor_test_reset_allocator_v1();unchanged(a,&before);
 }
 et_g3c4_allocation_limit=et_g3c4_successful_allocations;
 CHECK(et_cpu4_private_workspace_create_v1(lease,input,1)==NULL);et_g3c4_allocation_limit=SIZE_MAX;unchanged(a,&before);
 c4t_workspace *w=et_cpu4_private_workspace_create_v1(lease,input,1);CHECK(w);
 CHECK(et_cpu4_private_workspace_create_v1(lease,input,1)==NULL);
 NO(et_g3c4_private_training_lease_leave_v1(lease));
 NO(et_cpu4_private_workspace_role_v1(&foreign,0,0,0));NO(et_cpu4_private_workspace_role_v1(w,99,0,0));
 NO(et_cpu4_private_workspace_role_v1(w,0,0,2));NO(et_cpu4_private_workspace_role_v1(w,LINEAR,6,0));
 NO(et_cpu4_private_workspace_copy_logits_v1(w,output));NO(et_cpu4_private_workspace_vjp_begin_v1(w,upstream));
 bytevector bytes;fill(&bytes,cpu4_ref_upstream_bits);bytevector preserved=bytes;
 for(size_t i=0;i<3;i++){int64_t count=(int64_t[]){2048,4092,4100}[i];NO(et_cpu4_private_logits_copy_from_v1(upstream,&bytes,count));CHECK(!memcmp(&bytes,&preserved,sizeof(bytes)));}
 bytes.length=4092;NO(et_cpu4_private_logits_copy_to_v1(output,&bytes,4096));CHECK(bytes.length==4092);bytes=preserved;
 NO(et_cpu4_private_logits_copy_to_v1(output,w,4096));NO(et_cpu4_private_logits_copy_to_v1(output,a,4096));
 NO(et_cpu4_private_logits_copy_to_v1(output,lease,4096));NO(et_cpu4_private_logits_copy_to_v1(output,a->parameters[10]->value->data,4096));
 bytes.bytes[2]=0x80;bytes.bytes[3]=0x7f;NO(et_cpu4_private_logits_copy_from_v1(upstream,&bytes,4096));bytes=preserved;
 et_f32_tensor_error e;et_f32_scoped_guard_internal guard={0};OK(et_f32_tensor_scoped_begin_internal(upstream->tensor,&guard,&e));
 NO(et_cpu4_private_logits_copy_from_v1(upstream,&bytes,4096));NO(et_cpu4_private_logits_release_v1(upstream));OK(et_f32_tensor_scoped_end_internal(&guard));
 OK(et_cpu4_private_logits_copy_from_v1(upstream,&bytes,4096));
 w->ids[3]++;NO(et_cpu4_private_workspace_check_primals_v1(w,1));w->ids[3]--;
 w->positions[3]--;NO(et_cpu4_private_workspace_check_primals_v1(w,1));w->positions[3]++;
 w->keep[15]=0;NO(et_cpu4_private_workspace_check_primals_v1(w,1));w->keep[15]=1;
 uint32_t saved;OK(et_f32_tensor_copy_bits_to_v1(w->epsilon,&saved,1,&e));uint32_t wrong=saved+1;
 OK(et_f32_tensor_copy_bits_from_v1(w->epsilon,&wrong,1,&e));NO(et_cpu4_private_workspace_check_primals_v1(w,1));OK(et_f32_tensor_copy_bits_from_v1(w->epsilon,&saved,1,&e));
 a->parameters[13]->value->data[15]+=1;NO(et_cpu4_private_workspace_role_v1(w,EMBEDDING,0,0));CHECK(!w->ready[ET]);
 OK(et_f32_tensor_copy_bits_from_v1(a->parameters[13]->value,cpu4_ref_parameter_bits+cpu4_ref_parameter_offsets[13],16,&e));
 forward(w);NO(et_cpu4_private_workspace_role_v1(w,LINEAR,6,0));
 NO(et_cpu4_private_workspace_copy_logits_v1(w,mixed_logits));OK(et_cpu4_private_workspace_copy_logits_v1(w,output));
 OK(et_cpu4_private_logits_copy_to_v1(output,&bytes,4096));
 for(size_t i=0;i<1024;i++){uint32_t u=(uint32_t)bytes.bytes[4*i]|((uint32_t)bytes.bytes[4*i+1]<<8)|((uint32_t)bytes.bytes[4*i+2]<<16)|((uint32_t)bytes.bytes[4*i+3]<<24);float f;memcpy(&f,&u,4);near(f,cpu4_ref_logits[i],CPU4_REF_FORWARD_ATOL,CPU4_REF_FORWARD_RTOL,&max_logits_error);}
 NO(et_cpu4_private_workspace_vjp_begin_v1(w,mixed_logits));OK(et_cpu4_private_workspace_vjp_begin_v1(w,upstream));
 NO(et_cpu4_private_workspace_vjp_begin_v1(w,upstream));NO(et_cpu4_private_workspace_contribute_v1(w,1,0x3f800000,0));
 reverse(w);compare_gradients(w);unchanged(a,&before);
 OK(et_f32_tensor_scoped_begin_internal(w->slots[GPOS],&guard,&e));NO(et_cpu4_private_workspace_release_v1(w));NO(et_cpu4_private_workspace_contribute_v1(w,1,0x3f800000,0));OK(et_f32_tensor_scoped_end_internal(&guard));
 et_f32_tensor_test_fail_alloc_after_v1(0);NO(et_cpu4_private_workspace_contribute_v1(w,1,0x3f800000,0));et_f32_tensor_test_reset_allocator_v1();unchanged(a,&before);
 NO(et_cpu4_private_workspace_contribute_v1(w,0,0x3f800000,0));NO(et_cpu4_private_workspace_contribute_v1(w,1,-1,0));NO(et_cpu4_private_workspace_contribute_v1(w,1,0x3f800000,1));unchanged(a,&before);
 OK(et_cpu4_private_workspace_contribute_v1(w,1,0x3f800000,0));CHECK(w->r.state==C4T_CONSUMED);
 NO(et_cpu4_private_workspace_contribute_v1(w,1,0x3f800000,1));NO(et_cpu4_private_workspace_copy_logits_v1(w,output));NO(et_cpu4_private_workspace_role_v1(w,EMBEDDING,0,0));NO(et_cpu4_private_workspace_vjp_begin_v1(w,upstream));
 state after;capture(a,&after);CHECK(!memcmp(after.values,before.values,sizeof(after.values)));CHECK(!memcmp(after.parameters,before.parameters,sizeof(after.parameters)));CHECK(!memcmp(after.handles,before.handles,sizeof(after.handles)));
 for(size_t i=0;i<14;i++){CHECK(after.metadata[i].state==ET_F32_GRADIENT_PRESENT&&after.metadata[i].contribution_count==1&&after.metadata[i].normalization_weight_bits==0x3f800000);for(size_t j=cpu4_ref_parameter_offsets[i];j<cpu4_ref_parameter_offsets[i+1];j++){float f;memcpy(&f,&after.gradients[j],4);near(f,cpu4_ref_gradients[j],CPU4_REF_GRADIENT_ATOL,CPU4_REF_GRADIENT_RTOL,&max_gradient_error);}}
 release_frame(lease,input,output,upstream,w);
 OK(et_cpu4_private_input_release_v1(mixed));OK(et_cpu4_private_logits_release_v1(mixed_logits));OK(et_g3c4_private_training_lease_leave_v1(other));
 /* Eval captures cannot contribute even if caller supplies train mode. */
 OK(et_g3c4_private_training_lease_enter_v1(lease,1));input=et_cpu4_private_input_create_v1(lease,3,197,3,41);CHECK(input);
 w=et_cpu4_private_workspace_create_v1(lease,input,0);CHECK(w);NO(et_cpu4_private_workspace_contribute_v1(w,1,0x3f800000,1));OK(et_cpu4_private_workspace_release_v1(w));OK(et_cpu4_private_input_release_v1(input));OK(et_g3c4_private_training_lease_leave_v1(lease));
 et_f32_test_live_counts_v1 final={.struct_size=sizeof(final)};et_f32_test_live_counts_snapshot_v1(&final);CHECK(!memcmp(&initial,&final,sizeof(final)));
 size_t controls=0,control_bytes=0;for(c4t_record *r=c4t_registry;r;r=r->next){CHECK(r->state==C4T_DEAD);controls++;control_bytes+=r->extent;}
 size_t child_controls=0;
 for(et_g3c4_training_lease_internal *item=cpu4_training_leases;item;item=item->next)
  for(cpu4_training_child *child=item->children;child;child=child->next){CHECK(!child->live);child_controls++;}
 OK(et_g3c4_private_training_lease_prepare_release_v1(lease));OK(et_g3c4_private_training_lease_commit_release_v1(lease));
 OK(et_g3c4_private_training_lease_prepare_release_v1(other));OK(et_g3c4_private_training_lease_commit_release_v1(other));
 printf("CPU4-TRANSPORT-PASS checks=%zu parameters=14 values=1192 logits=1024 max-logits-error=%.9g max-gradient-error=%.9g payload-tensors-baseline=%zu retained-controls=%zu retained-control-bytes=%zu retained-lease-child-controls=%zu retained-lease-child-bytes=%zu\n",checks,max_logits_error,max_gradient_error,initial.tensors,controls,control_bytes,child_controls,child_controls*sizeof(cpu4_training_child));
 return 0;
}
