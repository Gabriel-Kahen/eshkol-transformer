#define ET_M3T_TESTING 1
/* This focused native test can observe private slots; it is never packaged. */
#include "../../src/eshkol_transformer/m3t_transport.c"
#include <stdio.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d %s [domain=%lld category=%lld code=%lld]\n",__FILE__,__LINE__,#x,(long long)error_domain,(long long)error_category,(long long)error_code);exit(1);}}while(0)
#define OK(x) CHECK((x)==0)
static unsigned char handles[14];
static owner *make_owner(initializer *s) {
 owner *o=et_m3t_private_owner_create_v1(s);CHECK(o);
 for(int i=0;i<14;i++)OK(et_m3t_private_owner_bind_v1(o,i,&handles[i]));
 for(int i=0;i<14;i++)OK(et_m3t_private_owner_initialize_v1(o,i));
 CHECK(et_m3t_private_owner_successor_v1(o)==o->successor);
 OK(et_m3t_private_owner_seal_v1(o));OK(et_m3t_private_initializer_seal_v1(o->successor));return o;
}
static void forward(workspace *w) {
 OK(et_m3t_private_embedding_v1(w,0,0));OK(et_m3t_private_embedding_v1(w,1,0));
 OK(et_m3t_private_residual_v1(w,0,0));OK(et_m3t_private_layer_norm_v1(w,0,0));
 OK(et_m3t_private_linear_v1(w,0,0));OK(et_m3t_private_linear_v1(w,1,0));OK(et_m3t_private_linear_v1(w,2,0));
 OK(et_m3t_private_heads_split_v1(w,0,0));OK(et_m3t_private_heads_split_v1(w,1,0));OK(et_m3t_private_heads_split_v1(w,2,0));
 OK(et_m3t_private_attention_v1(w,0));OK(et_m3t_private_heads_merge_v1(w,0));
 OK(et_m3t_private_linear_v1(w,3,0));OK(et_m3t_private_residual_v1(w,1,0));
 OK(et_m3t_private_layer_norm_v1(w,1,0));OK(et_m3t_private_linear_v1(w,4,0));
 OK(et_m3t_private_gelu_v1(w,0));OK(et_m3t_private_linear_v1(w,5,0));OK(et_m3t_private_residual_v1(w,2,0));
 OK(et_m3t_private_layer_norm_v1(w,2,0));OK(et_m3t_private_linear_v1(w,6,0));
}
static void reverse(workspace *w) {
 OK(et_m3t_private_linear_v1(w,6,1));OK(et_m3t_private_layer_norm_v1(w,2,1));OK(et_m3t_private_residual_v1(w,2,1));
 OK(et_m3t_private_linear_v1(w,5,1));OK(et_m3t_private_gelu_v1(w,1));OK(et_m3t_private_linear_v1(w,4,1));
 OK(et_m3t_private_layer_norm_v1(w,1,1));OK(et_m3t_private_sum_v1(w,2));OK(et_m3t_private_residual_v1(w,1,1));
 OK(et_m3t_private_linear_v1(w,3,1));OK(et_m3t_private_heads_merge_v1(w,1));OK(et_m3t_private_attention_v1(w,1));
 OK(et_m3t_private_heads_split_v1(w,0,1));OK(et_m3t_private_heads_split_v1(w,1,1));OK(et_m3t_private_heads_split_v1(w,2,1));
 OK(et_m3t_private_linear_v1(w,0,1));OK(et_m3t_private_linear_v1(w,1,1));OK(et_m3t_private_linear_v1(w,2,1));
 OK(et_m3t_private_sum_v1(w,0));OK(et_m3t_private_layer_norm_v1(w,0,1));OK(et_m3t_private_sum_v1(w,1));
 OK(et_m3t_private_residual_v1(w,0,1));OK(et_m3t_private_embedding_v1(w,0,1));OK(et_m3t_private_embedding_v1(w,1,1));OK(et_m3t_private_sum_v1(w,3));
}
#include "role_oracle.h"
#include "failure_cases.h"
int main(void) {
 initializer *s=et_m3t_private_initializer_create_v1(1729);CHECK(s);OK(et_m3t_private_initializer_seal_v1(s));
 discovery_failures(s);
 owner *o=make_owner(s);CHECK(o->successor->words[2]==290);CHECK(s->words[2]==0);
 input *i=et_m3t_private_input_create_v1();CHECK(i);OK(et_m3t_private_input_copy_v1(i,65,66));
 logits *l=et_m3t_private_logits_create_v1(),*u=et_m3t_private_logits_create_v1();CHECK(l&&u);
 struct {int64_t n;unsigned char bytes[2048];} bytes={2048,{0}};
 for(size_t n=0;n<512;n++){uint32_t bit=n%2?UINT32_C(0xbf000000):UINT32_C(0x3f800000);memcpy(bytes.bytes+4*n,&bit,4);}
 OK(et_m3t_private_logits_copy_from_v1(u,&bytes,2048));
 memory_loop(o,i,u,l);
 ingress_and_initializer(s,o,i);
 lifecycle_negatives(o,i,u,l);
 workspace *w=et_m3t_private_workspace_create_v1(o);CHECK(w);
 oracle *a=calloc(1,sizeof(*a));CHECK(a);oracle_begin(a,o,u);oracle_forward(a);
 OK(et_m3t_private_workspace_begin_v1(w,i,1));
 /* Current owner bytes and input can change after begin; dispatch must consume
  * the independent saved frame, and exact restoration must pass check. */
 uint32_t primal[8],changed[8];et_f32_tensor_error e;
 OK(et_f32_tensor_copy_bits_to_v1(value(o,13),primal,8,&e));memcpy(changed,primal,sizeof(primal));changed[0]^=UINT32_C(0x80000000);
 OK(et_f32_tensor_copy_bits_from_v1(value(o,13),changed,8,&e));OK(et_m3t_private_input_copy_v1(i,7,8));
 forward(w);oracle_equal(a,w,ET,Z);CHECK(et_m3t_private_workspace_check_primals_v1(w,1));
 OK(et_f32_tensor_copy_bits_from_v1(value(o,13),primal,8,&e));OK(et_m3t_private_workspace_check_primals_v1(w,1));
 OK(et_m3t_private_input_copy_v1(i,65,66));
 OK(et_m3t_private_workspace_copy_logits_v1(w,l));OK(et_m3t_private_workspace_vjp_begin_v1(w,u));
 memset(bytes.bytes,0,sizeof(bytes.bytes));OK(et_m3t_private_logits_copy_from_v1(u,&bytes,2048));
 reverse(w);oracle_reverse(a);oracle_equal(a,w,DZ,GTOKEN);free(a);
 for(int n=0;n<14;n++)CHECK(et_m3t_private_workspace_gradient_v1(w,n));
 OK(et_m3t_private_workspace_check_primals_v1(w,1));
 uint32_t saved_logits[512],unchanged_logits[512];OK(et_f32_tensor_copy_bits_to_v1(l->tensor,saved_logits,512,&e));
 OK(et_m3t_private_workspace_reset_v1(w));OK(et_m3t_private_input_copy_v1(i,7,8));
 OK(et_m3t_private_workspace_begin_v1(w,i,0));forward(w);
 OK(et_f32_tensor_copy_bits_to_v1(l->tensor,unchanged_logits,512,&e));CHECK(!memcmp(saved_logits,unchanged_logits,sizeof(saved_logits)));
 OK(et_m3t_private_workspace_reset_v1(w));
 OK(et_m3t_private_workspace_release_v1(w));OK(et_m3t_private_workspace_release_v1(w));
 OK(et_m3t_private_input_release_v1(i));OK(et_m3t_private_logits_release_v1(l));OK(et_m3t_private_logits_release_v1(u));
 construction_failures(s);workspace_failures(o);
 puts("m3t native transport and independent role wiring passed");return 0;
}
