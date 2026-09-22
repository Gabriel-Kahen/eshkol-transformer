/* Normal canonical object only. Paint measures a conservative *total* including
 * callback plus external libc/libm frames. The compiler call-graph proof separately
 * excludes caller/K1 and external frames, per the provider's storage contract. */
#include "eshkol_transformer/g3s_sampling_abi.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdalign.h>
static alignas(64) unsigned char stack_space[131072];
extern void g3s_test_on_stack(void *,void (*)(void *),void *);
static float logits[256],temperature=1,top_p=1;
static int64_t top_k=256,rng[4]={1,7,0,0},token,state[4];
static uint64_t row[]={1,256},one[]={1},four[]={4};
static et_kernel_tensor_view_v1 in[5],out[2];
static et_kernel_request_v1 req;
static et_kernel_call_v1 call;
static et_kernel_error error;
static int invoke,rc;
static const et_kernel_provider_v1 *provider;
static void callback(void *unused) { (void)unused; if(invoke) {provider->invoke_call(&call);rc=0;} else rc=provider->validate_call(&call,&error); }
static et_kernel_tensor_view_v1 view(void *p,size_t bytes,const char *dt,size_t rank,const uint64_t *sh) {et_kernel_tensor_view_v1 v={sizeof(v),p,bytes,dt,"cpu",1,0,rank,sh};return v;}
int main(void) {
 provider=et_g3s_kernel_provider_v1();size_t peak=0;
 for(int cat=0;cat<2;++cat) for(int mode=0;mode<6;++mode) {
  temperature=1;top_k=256;top_p=1;rng[0]=1;rng[2]=rng[3]=0;
  for(size_t i=0;i<256;++i) logits[i]=(float)(i%31)*0.125f;
  in[0]=view(logits,sizeof(logits),"f32",2,row);
  if(cat) {in[1]=view(&temperature,4,"f32",0,NULL);in[2]=view(&top_k,8,"i64",0,NULL);in[3]=view(&top_p,4,"f32",0,NULL);in[4]=view(rng,32,"i64",1,four);}else in[1]=view(rng,32,"i64",1,four);
  out[0]=view(&token,8,"i64",1,one);out[1]=view(state,32,"i64",1,four);
  req=(et_kernel_request_v1){sizeof(req),cat?"g3s.categorical.forward":"g3s.greedy.forward","f32","cpu",2,row,1,{0}};
  size_t n=cat?5:2;call=(et_kernel_call_v1){sizeof(call),cat?"g3s.categorical":"g3s.greedy",&req,n,sizeof(in[0]),n*sizeof(in[0]),in,2,sizeof(out[0]),sizeof(out),out};
  if(mode==1) logits[255]=-FLT_MAX;
  if(mode==2) {top_k=1;top_p=0.01f;}
  if(mode==3) {logits[0]=FLT_MAX;logits[255]=-FLT_MAX;}
  if(mode==4) rng[0]=2;
  if(mode==5) call.input_count=0;
  for(invoke=0;invoke<2;++invoke) {
   if(invoke && provider->validate_call(&call,&error)!=0) continue;
   /* Warm dynamic linker/libm independently before a measured normal call. */
   callback(NULL);
   memset(stack_space,0xa5,sizeof(stack_space));
   g3s_test_on_stack(stack_space+sizeof(stack_space),callback,NULL);
   size_t first=0;while(first<sizeof(stack_space)&&stack_space[first]==0xa5) ++first;
   size_t used=sizeof(stack_space)-first;if(used>peak) peak=used;
   if(first==0 || used>16384) {fprintf(stderr,"G3S stack ceiling exceeded: %zu\n",used);return 1;}
   if(mode<3 && rc!=0) {fprintf(stderr,"G3S stack case unexpectedly rejected: %s\n",error.message);return 1;}
  }
 }
 printf("G3S normal runtime conservative stack high-water: %zu bytes <=16384 (includes measurement callback and external libc/libm; provider-only compiler proof excludes caller/K1 and external frames)\n",peak);return 0;
}
