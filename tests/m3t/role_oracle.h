/* Independent literal provider wiring. No production role descriptor is read.
 * Each oracle buffer is separate from transport storage. These comparisons prove
 * transport wiring; reviewed provider suites own primitive gradient accuracy. */
typedef struct oracle {
 float s[SLOT_COUNT][1024],p[14][1024],eps;
 int64_t ids[2],pos[2];unsigned char keep[4];
} oracle;
static size_t oracle_shape(int s,const uint64_t **shape) {
 static const uint64_t small[]={1,2,4},heads[]={1,2,2,2},wide[]={1,2,8},z[]={1,2,256};
 static const uint64_t weights[14][2]={{4,4},{4,4},{4,4},{4,4},{4,8},{8,4},{4,0},{4,0},{4,0},{4,0},{256,4},{4,0},{4,0},{2,4}};
 if(s>=GKEY&&s<=GPOS){*shape=weights[s-GKEY];return (*shape)[1]?2:1;}
 if(s==GHEAD||s==GTOKEN){*shape=weights[10];return 2;}
 if(s==Z||s==DZ){*shape=z;return 3;}
 if(s==FU||s==FG||s==DFU||s==DFG){*shape=wide;return 3;}
 if(s==QH||s==KH||s==VH||s==AH||s==DQH||s==DKH||s==DVH||s==DAH){*shape=heads;return 4;}
 *shape=small;return 3;
}
static et_kernel_tensor_view_v1 ov(oracle *a,int ref) {
 static const uint64_t ids[]={1,2},mask[]={1,2,2};
 if(ref==IDS)return view(a->ids,16,"i64",2,ids);
 if(ref==POS)return view(a->pos,16,"i64",2,ids);
 if(ref==KEEP)return view(a->keep,4,"bool",3,mask);
 if(ref==EPS)return view(&a->eps,4,"f32",0,NULL);
 const uint64_t *shape;size_t rank=oracle_shape(ref>=P0?GKEY+ref-P0:ref,&shape),count=1;
 for(size_t i=0;i<rank;i++)count*=shape[i];
 return view(ref>=P0?a->p[ref-P0]:a->s[ref],count*4,"f32",rank,shape);
}
static void oracle_call(oracle *a,int rt,const char *cap,const char *op,size_t rank,
 const uint64_t *shape,size_t ni,const int *inputs,size_t no,const int *outputs) {
 et_kernel_tensor_view_v1 in[7],out[3];
 for(size_t i=0;i<ni;i++)in[i]=ov(a,inputs[i]);
 for(size_t i=0;i<no;i++)out[i]=ov(a,outputs[i]);
 OK(dispatch(rt,cap,op,rank,shape,in,ni,out,no));
}
#define ORACLE(rt,cap,op,shape,ins,outs) do { \
 const uint64_t rs[]=shape;const int ri[]=ins,ro[]=outs; \
 oracle_call(a,rt,cap,op,sizeof(rs)/sizeof(*rs),rs,sizeof(ri)/sizeof(*ri),ri,sizeof(ro)/sizeof(*ro),ro); \
} while(0)
#define A(...) {__VA_ARGS__}
static void oracle_forward(oracle *a) {
 ORACLE(0,"n3k.embedding-forward","n3k.embedding.forward",A(1,2,256,4),A(IDS,P0+10),A(ET));
 ORACLE(0,"n3k.embedding-forward","n3k.embedding.forward",A(1,2,2,4),A(POS,P0+13),A(EP));
 ORACLE(0,"n3k.residual","n3k.residual.forward",A(1,2,4),A(ET,EP),A(X));
 ORACLE(1,"kernel.norm","layer-norm.forward",A(1,2,4),A(X,P0+7,P0+6,EPS),A(N1));
 ORACLE(0,"n3k.linear","n3k.linear.forward-no-bias",A(1,2,4,4),A(N1,P0+2),A(QT));
 ORACLE(0,"n3k.linear","n3k.linear.forward-no-bias",A(1,2,4,4),A(N1,P0+0),A(KT));
 ORACLE(0,"n3k.linear","n3k.linear.forward-no-bias",A(1,2,4,4),A(N1,P0+3),A(VT));
 ORACLE(0,"n3k.head-layout","n3k.heads.split.forward",A(1,2,2,2),A(QT),A(QH));
 ORACLE(0,"n3k.head-layout","n3k.heads.split.forward",A(1,2,2,2),A(KT),A(KH));
 ORACLE(0,"n3k.head-layout","n3k.heads.split.forward",A(1,2,2,2),A(VT),A(VH));
 ORACLE(2,"kernel.causal-attention","causal-attention.forward",A(1,2,2,2,2,2),A(QH,KH,VH,POS,POS,KEEP),A(AH));
 ORACLE(0,"n3k.head-layout","n3k.heads.merge.forward",A(1,2,2,2),A(AH),A(AT));
 ORACLE(0,"n3k.linear","n3k.linear.forward-no-bias",A(1,2,4,4),A(AT,P0+1),A(AO));
 ORACLE(0,"n3k.residual","n3k.residual.forward",A(1,2,4),A(X,AO),A(R));
 ORACLE(1,"kernel.norm","layer-norm.forward",A(1,2,4),A(R,P0+9,P0+8,EPS),A(N2));
 ORACLE(0,"n3k.linear","n3k.linear.forward-no-bias",A(1,2,4,8),A(N2,P0+5),A(FU));
 ORACLE(0,"n3k.gelu","n3k.gelu.forward",A(1,2,8),A(FU),A(FG));
 ORACLE(0,"n3k.linear","n3k.linear.forward-no-bias",A(1,2,8,4),A(FG,P0+4),A(FD));
 ORACLE(0,"n3k.residual","n3k.residual.forward",A(1,2,4),A(R,FD),A(Y));
 ORACLE(1,"kernel.norm","layer-norm.forward",A(1,2,4),A(Y,P0+12,P0+11,EPS),A(NF));
 ORACLE(0,"n3k.linear","n3k.linear.forward-no-bias",A(1,2,4,256),A(NF,P0+10),A(Z));
}
static void oracle_reverse(oracle *a) {
 ORACLE(0,"n3k.linear","n3k.linear.backward-no-bias",A(1,2,4,256),A(NF,P0+10,DZ),A(DNF,GHEAD));
 ORACLE(1,"kernel.norm","layer-norm.backward",A(1,2,4),A(Y,P0+12,EPS,DNF),A(DY,GFW,GFB));
 ORACLE(0,"n3k.residual","n3k.residual.backward",A(1,2,4),A(DY),A(DRSKIP,DFD));
 ORACLE(0,"n3k.linear","n3k.linear.backward-no-bias",A(1,2,8,4),A(FG,P0+4,DFD),A(DFG,GDOWN));
 ORACLE(0,"n3k.gelu","n3k.gelu.backward",A(1,2,8),A(FU,DFG),A(DFU));
 ORACLE(0,"n3k.linear","n3k.linear.backward-no-bias",A(1,2,4,8),A(N2,P0+5,DFU),A(DN2,GUP));
 ORACLE(1,"kernel.norm","layer-norm.backward",A(1,2,4),A(R,P0+9,EPS,DN2),A(DRFFN,GN2W,GN2B));
 ORACLE(0,"n3k.activation-sum","n3k.sum2.forward",A(1,2,4),A(DRSKIP,DRFFN),A(DR));
 ORACLE(0,"n3k.residual","n3k.residual.backward",A(1,2,4),A(DR),A(DXSKIP,DAO));
 ORACLE(0,"n3k.linear","n3k.linear.backward-no-bias",A(1,2,4,4),A(AT,P0+1,DAO),A(DAT,GOUTPUT));
 ORACLE(0,"n3k.head-layout","n3k.heads.merge.backward",A(1,2,2,2),A(DAT),A(DAH));
 ORACLE(2,"kernel.causal-attention","causal-attention.backward",A(1,2,2,2,2,2),A(QH,KH,VH,POS,POS,KEEP,DAH),A(DQH,DKH,DVH));
 ORACLE(0,"n3k.head-layout","n3k.heads.split.backward",A(1,2,2,2),A(DQH),A(DQT));
 ORACLE(0,"n3k.head-layout","n3k.heads.split.backward",A(1,2,2,2),A(DKH),A(DKT));
 ORACLE(0,"n3k.head-layout","n3k.heads.split.backward",A(1,2,2,2),A(DVH),A(DVT));
 ORACLE(0,"n3k.linear","n3k.linear.backward-no-bias",A(1,2,4,4),A(N1,P0+2,DQT),A(DN1Q,GQUERY));
 ORACLE(0,"n3k.linear","n3k.linear.backward-no-bias",A(1,2,4,4),A(N1,P0+0,DKT),A(DN1K,GKEY));
 ORACLE(0,"n3k.linear","n3k.linear.backward-no-bias",A(1,2,4,4),A(N1,P0+3,DVT),A(DN1V,GVALUE));
 ORACLE(0,"n3k.activation-sum","n3k.sum3.forward",A(1,2,4),A(DN1Q,DN1K,DN1V),A(DN1));
 ORACLE(1,"kernel.norm","layer-norm.backward",A(1,2,4),A(X,P0+7,EPS,DN1),A(DXATTENTION,GN1W,GN1B));
 ORACLE(0,"n3k.activation-sum","n3k.sum2.forward",A(1,2,4),A(DXSKIP,DXATTENTION),A(DX));
 ORACLE(0,"n3k.residual","n3k.residual.backward",A(1,2,4),A(DX),A(DET,DEP));
 ORACLE(0,"n3k.embedding-backward","n3k.embedding.backward",A(1,2,256,4),A(IDS,DET),A(GTOKEN));
 ORACLE(0,"n3k.embedding-backward","n3k.embedding.backward",A(1,2,2,4),A(POS,DEP),A(GPOS));
 ORACLE(0,"n3k.tied-sum","n3k.sum2.forward",A(256,4),A(GHEAD,GTOKEN),A(GTIED));
}
#undef A
#undef ORACLE
static void oracle_begin(oracle *a,owner *o,logits *u) {
 memset(a,0,sizeof(*a));a->ids[0]=65;a->ids[1]=66;a->pos[1]=1;memset(a->keep,1,4);
 uint32_t eps=UINT32_C(0x3727c5ac);memcpy(&a->eps,&eps,4);et_f32_tensor_error e;
 for(size_t i=0;i<14;i++) {et_kernel_tensor_view_v1 v=ov(a,P0+(int)i);uint32_t bits[1024];
  OK(et_f32_tensor_copy_bits_to_v1(value(o,i),bits,v.byte_length/4,&e));memcpy(v.data,bits,v.byte_length);
 }
 uint32_t bits[512];OK(et_f32_tensor_copy_bits_to_v1(u->tensor,bits,512,&e));memcpy(a->s[DZ],bits,sizeof(bits));
}
static void oracle_equal(oracle *a,workspace *w,int first,int last) {
 for(int i=first;i<=last;i++) {
  et_kernel_tensor_view_v1 expected=ov(a,i);uint32_t actual[1024];et_f32_tensor_error e;
  OK(et_f32_tensor_copy_bits_to_v1(w->slots[i],actual,expected.byte_length/4,&e));
  if(memcmp(actual,expected.data,expected.byte_length))fprintf(stderr,"oracle mismatch slot %d\n",i);
  CHECK(!memcmp(actual,expected.data,expected.byte_length));
 }
}
