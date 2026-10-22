#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_CPU4_TRAINING_PRIVATE 1
#define ET_CPU4_OBJECTIVE_PRIVATE 1
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include "../../src/eshkol_transformer/cpu4_training_transport.c"
#include "reference.h"
#include <stdio.h>
extern void et_m3t_test_k1_fail_after(size_t);
extern void et_m3t_test_k1_allocator_reset(void);
static size_t checks;
#define CHECK(x) do {++checks;if(!(x)){fprintf(stderr,"FAIL line%d: %s d%lld c%lld e%lld\n",__LINE__,#x,(long long)c4t_domain,(long long)c4t_category,(long long)c4t_code);exit(1);}}while(0)
#define OK(x) CHECK((x)==0)
#define NO(x) CHECK((x)!=0)
static unsigned char identities[2][14];
static size_t words(size_t i) {return cpu4_ref_parameter_offsets[i+1]-cpu4_ref_parameter_offsets[i];}
static void *model(size_t ordinal) {
 et_g3c4_model_owner_internal *owner=et_g3c4_private_model_owner_create_seeded_v1(1729);CHECK(owner);
 for(int64_t i=0;i<14;i++)OK(et_g3c4_private_model_owner_bind_v1(owner,i,&identities[ordinal][i]));
 for(int64_t i=0;i<14;i++)OK(et_g3c4_private_model_owner_initialize_v1(owner,i));
 OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
 for(size_t i=0;i<14;i++) {et_f32_tensor_error e;
  OK(et_f32_tensor_copy_bits_from_v1(owner->parameters[i]->value,cpu4_ref_parameter_bits+cpu4_ref_parameter_offsets[i],words(i),&e));
 }
 return owner;
}
static const int fwd[21][2]={{0,0},{0,1},{4,0},{2,0},{1,0},{1,1},{1,2},{5,0},{5,1},{5,2},{7,0},{6,0},{1,3},{4,1},{2,1},{1,4},{3,0},{1,5},{4,2},{2,2},{1,6}};
static const int rev[25][2]={{1,6},{2,2},{4,2},{1,5},{3,0},{1,4},{2,1},{8,2},{4,1},{1,3},{6,0},{7,0},{5,0},{5,1},{5,2},{1,0},{1,1},{1,2},{8,0},{2,0},{8,1},{4,0},{0,0},{0,1},{8,3}};
static float number(const unsigned char *bytes) {uint32_t bits=(uint32_t)bytes[0]|(uint32_t)bytes[1]<<8|(uint32_t)bytes[2]<<16|(uint32_t)bytes[3]<<24;float value;memcpy(&value,&bits,4);return value;}
static double maximum;
static void near(double actual,double expected) {double error=fabs(actual-expected);if(error>maximum)maximum=error;CHECK(isfinite(actual)&&error<=3e-4+5e-4*fabs(expected));}
int main(void) {
 et_g3c4_model_owner_internal *owner=model(0),*other=model(1);
 void *lease=et_g3c4_private_training_lease_create_v1(owner),*mixed=et_g3c4_private_training_lease_create_v1(other);CHECK(lease&&mixed);
 et_f32_test_live_counts_v1 baseline={.struct_size=sizeof(baseline)};et_f32_test_live_counts_snapshot_v1(&baseline);
 OK(et_g3c4_private_training_lease_enter_v1(lease,1));OK(et_g3c4_private_training_lease_enter_v1(mixed,1));
 void *input=et_cpu4_private_input_create_v1(lease,3,197,3,41);CHECK(input);
 void *workspace=et_cpu4_private_workspace_create_v1(lease,input,1);CHECK(workspace);
 void *logits=et_cpu4_private_logits_create_v1(lease),*seed=et_cpu4_private_logits_create_v1(lease),*mixed_seed=et_cpu4_private_logits_create_v1(mixed);CHECK(logits&&seed&&mixed_seed);
 for(size_t i=0;i<21;i++)OK(et_cpu4_private_workspace_role_v1(workspace,fwd[i][0],fwd[i][1],0));
 OK(et_cpu4_private_workspace_copy_logits_v1(workspace,logits));
 struct {int64_t length;int64_t values[4];} targets={32,{197,3,41,7}},copied={32,{0,0,0,0}};
 struct {int64_t length;unsigned char values[4];} mask={4,{1,0,1,1}};
 struct {int64_t length;unsigned char values[12];} observation={12,{0}};
 unsigned char observation_before[sizeof(observation)];
 for(size_t i=0;i<12;i++)observation.values[i]=(unsigned char)(0xa1u+i);
 memcpy(observation_before,&observation,sizeof(observation));
 et_kernel_tensor_view_v1 view=c4o_tensor(targets.values,32,"i64",2,c4t_ids_shape);
 OK(et_cpu4_private_batch_copy_v1(lease,&view,&copied,1));CHECK(!memcmp(copied.values,targets.values,32));
 view.dtype="f32";NO(et_cpu4_private_batch_copy_v1(lease,&view,&copied,1));CHECK(c4t_domain==1&&c4t_category==ET_KERNEL_ERROR_DTYPE_MISMATCH);view.dtype="i64";
 copied.values[0]=99;targets.values[3]=256;NO(et_cpu4_private_batch_copy_v1(lease,&view,&copied,0));CHECK(copied.values[0]==99);targets.values[3]=7;
 NO(et_cpu4_private_batch_copy_v1(lease,&view,lease,1));NO(et_cpu4_private_batch_copy_v1(lease,&view,workspace,1));NO(et_cpu4_private_batch_copy_v1(lease,&view,owner->parameters[10]->value->data,1));
 c4o_bytevector before={.length=4096},after={.length=4096};OK(et_cpu4_private_logits_copy_to_v1(seed,&before,4096));
 #define UNCHANGED() do {OK(et_cpu4_private_logits_copy_to_v1(seed,&after,4096));CHECK(!memcmp(&before,&after,sizeof(before)));CHECK(!memcmp(observation_before,&observation,sizeof(observation)));}while(0)
 NO(et_cpu4_private_objective_prepare_v1(logits,mixed_seed,&targets,&mask,&observation));UNCHANGED();
 NO(et_cpu4_private_objective_prepare_v1(logits,logits,&targets,&mask,&observation));UNCHANGED();
 targets.values[0]=256;NO(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,&observation));UNCHANGED();targets.values[0]=197;
 mask.values[1]=2;NO(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,&observation));UNCHANGED();mask.values[1]=0;
 mask.values[0]=mask.values[2]=mask.values[3]=0;NO(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,&observation));UNCHANGED();mask.values[0]=mask.values[2]=mask.values[3]=1;
 NO(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,lease));UNCHANGED();
 NO(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,workspace));UNCHANGED();
 observation.length=8;NO(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,&observation));observation.length=12;UNCHANGED();
 et_m3t_test_k1_fail_after(0);NO(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,&observation));et_m3t_test_k1_allocator_reset();UNCHANGED();
 et_f32_tensor_error error;et_f32_scoped_guard_internal guard={0};OK(et_f32_tensor_scoped_begin_internal(((c4t_logits *)seed)->tensor,&guard,&error));
 NO(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,&observation));OK(et_f32_tensor_scoped_end_internal(&guard));UNCHANGED();
 OK(et_cpu4_private_objective_prepare_v1(logits,seed,&targets,&mask,&observation));
 for(size_t i=0;i<3;i++)near(number(observation.values+4*i),cpu4_objective_observation[i]);
 CHECK(number(observation.values+4)==3);
 OK(et_cpu4_private_logits_copy_to_v1(seed,&after,4096));for(size_t i=0;i<1024;i++){float expected;uint32_t bits=cpu4_ref_upstream_bits[i];memcpy(&expected,&bits,4);near(number(after.payload+4*i),expected);}
 OK(et_cpu4_private_workspace_vjp_begin_v1(workspace,seed));
 for(size_t i=0;i<25;i++)OK(et_cpu4_private_workspace_role_v1(workspace,rev[i][0],rev[i][1],1));
 OK(et_cpu4_private_workspace_contribute_v1(workspace,1,0x40400000,0));
 for(size_t i=0;i<14;i++){uint32_t values[1024];
  et_f32_gradient_metadata_v1 metadata={.struct_size=sizeof(metadata)};
  OK(et_f32_parameter_gradient_metadata_v1(owner->parameters[i],&metadata,&error));
  CHECK(metadata.state==ET_F32_GRADIENT_PRESENT&&metadata.contribution_count==1&&metadata.normalization_weight_bits==0x40400000);
  OK(et_f32_tensor_copy_bits_to_v1(owner->parameters[i]->value,values,words(i),&error));
  CHECK(!memcmp(values,cpu4_ref_parameter_bits+cpu4_ref_parameter_offsets[i],4*words(i)));
  OK(et_f32_tensor_copy_bits_to_v1(owner->parameters[i]->gradient,values,words(i),&error));for(size_t j=0;j<words(i);j++){float actual;memcpy(&actual,values+j,4);near(actual,cpu4_ref_gradients[cpu4_ref_parameter_offsets[i]+j]);}}
 OK(et_cpu4_private_workspace_release_v1(workspace));OK(et_cpu4_private_input_release_v1(input));OK(et_cpu4_private_logits_release_v1(logits));OK(et_cpu4_private_logits_release_v1(seed));OK(et_cpu4_private_logits_release_v1(mixed_seed));
 OK(et_g3c4_private_training_lease_leave_v1(lease));OK(et_g3c4_private_training_lease_leave_v1(mixed));
 OK(et_g3c4_private_training_lease_prepare_release_v1(lease));OK(et_g3c4_private_training_lease_commit_release_v1(lease));OK(et_g3c4_private_training_lease_prepare_release_v1(mixed));OK(et_g3c4_private_training_lease_commit_release_v1(mixed));
 et_f32_test_live_counts_v1 final={.struct_size=sizeof(final)};et_f32_test_live_counts_snapshot_v1(&final);CHECK(!memcmp(&baseline,&final,sizeof(final)));
 printf("CPU4-OBJECTIVE-PASS checks=%zu values=1192 seeds=1024 mask-weight=3 max-error=%.9g\n",checks,maximum);
 return 0;
}
