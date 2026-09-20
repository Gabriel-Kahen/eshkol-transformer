#include "test_common.h"
static void print_values(const char *name,const char *field,const float *values,size_t count) {
  printf("%s.%s",name,field);for(size_t i=0;i<count;i++)printf(" %08x",bits(values[i]));putchar('\n');
}
static void arithmetic(et_kernel_runtime *r) {
  float loss[]={2,10},mask[]={0.5f,2},out[3],s[2];
  reduce(r,loss,mask,0,out);CHECK(out[0]==21 && out[1]==2.5f && bits(out[2])==0x41066666u);
  seed(r,mask,0,0,s);CHECK(s[0]==0.5f && s[1]==2);
  seed(r,mask,0,1,s);CHECK(bits(s[0])==0x3e4ccccdu && bits(s[1])==0x3f4ccccdu);
  mask[0]=3;mask[1]=4;seed(r,mask,0,1,s);CHECK(bits(s[0])==0x3edb6db7u);
  loss[0]=0x1p-24f;loss[1]=0x1.000002p0f;mask[0]=1;mask[1]=0x1.000002p0f;
  reduce(r,loss,mask,0,out);CHECK(bits(out[0])==0x3f800002u);
  /* Pure two-term traversal reversal is equivalent, unlike pairing reversal. */
  float reversed_loss[]={loss[1],loss[0]},reversed_mask[]={mask[1],mask[0]},other[3];
  reduce(r,reversed_loss,reversed_mask,0,other);CHECK(memcmp(out,other,12)==0);
}
static void composition(et_kernel_runtime *r,et_kernel_runtime *l2,int emit) {
  const char *names[]={"ones","weighted","zero","scaled"};
  float masks[][2]={{1,1},{0.5f,2},{0,1},{1,4}};
  float logits[512],loss[2],out[3],ns[2],ms[2],ng[512],mg[512],max_num_fd=0,max_mean_fd=0;logits_init(logits);l2_forward(l2,logits,loss);
  for(size_t c=0;c<4;c++) {
    reduce(r,loss,masks[c],0,out);seed(r,masks[c],0,0,ns);seed(r,masks[c],0,1,ms);
    l2_backward(l2,logits,ns,ng);l2_backward(l2,logits,ms,mg);
    if(c==0 || c==2) {
      unsigned char bm[]={c==0?1:0,1};float bo[3],bs[2];
      reduce(r,loss,bm,1,bo);CHECK(memcmp(out,bo,sizeof(bo))==0);
      seed(r,bm,1,0,bs);CHECK(memcmp(ns,bs,sizeof(bs))==0);
      seed(r,bm,1,1,bs);CHECK(memcmp(ms,bs,sizeof(bs))==0);
    }
    const size_t positions[]={0,3,127,255,256,256+201,400,511};
    for(size_t p=0;p<sizeof(positions)/sizeof(positions[0]);p++) {
      size_t i=positions[p];float original=logits[i],high[3],low[3],tmp[2];
      logits[i]=original+0.03125f;l2_forward(l2,logits,tmp);reduce(r,tmp,masks[c],0,high);
      logits[i]=original-0.03125f;l2_forward(l2,logits,tmp);reduce(r,tmp,masks[c],0,low);logits[i]=original;
      float num_error=fabsf((high[0]-low[0])/0.0625f-ng[i]),mean_error=fabsf((high[2]-low[2])/0.0625f-mg[i]);
      CHECK(num_error<0.00016f);CHECK(mean_error<0.00008f);
      max_num_fd=fmaxf(max_num_fd,num_error);max_mean_fd=fmaxf(max_mean_fd,mean_error);
    }
    if(c==2)for(size_t i=0;i<256;i++)CHECK(ng[i]==0 && mg[i]==0);
    if(emit) {
      print_values(names[c],"loss",loss,2);print_values(names[c],"reduction",out,3);
      print_values(names[c],"numerator_seed",ns,2);print_values(names[c],"mean_seed",ms,2);
      print_values(names[c],"numerator_gradient",ng,512);print_values(names[c],"mean_gradient",mg,512);
    }
  }
  if(!emit)printf("L3S native FD PASS: maximum numerator error %.9g, mean error %.9g\n",max_num_fd,max_mean_fd);
}
int main(int argc,char **argv) {
  CHECK(argc==1 || (argc==2 && strcmp(argv[1],"--emit")==0));environment();
  et_kernel_runtime *r=runtime(et_l3s_kernel_provider_v1()),*l2=runtime(et_l2_indexed_cross_entropy_provider_v1());
  arithmetic(r);composition(r,l2,argc==2);et_kernel_runtime_destroy(l2);et_kernel_runtime_destroy(r);
  if(argc==1)puts("L3S NUMERICAL PASS: exact arithmetic, real L2 composition and finite differences");return 0;
}
