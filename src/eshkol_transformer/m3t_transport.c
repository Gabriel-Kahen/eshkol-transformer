#include "m3t_transport.h"
#include "m3t_f32_scoped.h"
#include "../../native/f32_parameter_internal.h"
#include "../../native/t1_i64_shell.h"
#include "eshkol_transformer/i64_tensor.h"
#include "eshkol_transformer/n3k_primitives_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#include "eshkol_transformer/a2_attention_abi.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* All registries are private, serialized identity admission. Released control
 * shells remain inert and reachable, preventing address reuse into authority. */
enum { INITIALIZER=1, OWNER, INPUT, LOGITS, WORKSPACE };
typedef struct record { struct record *next; int kind, state; } record;
static record *registry;
static int64_t error_domain, error_category, error_code;
static int64_t fail(int domain, int category, int code) {
  error_domain=domain;error_category=category;error_code=code;return category;
}
static int64_t bad(int category,int code) { return fail(0,category,code); }
static void clear(void) { error_domain=error_category=error_code=0; }
int64_t et_m3t_private_last_error_domain_v1(void) {return error_domain;}
int64_t et_m3t_private_last_error_category_v1(void) {return error_category;}
int64_t et_m3t_private_last_error_code_v1(void) {return error_code;}
static int f32_error(int rc,const et_f32_tensor_error *e) {return rc?(int)fail(3,e->category,e->code):0;}
static int ierror(int rc,const et_i64_tensor_error *e) {return rc?(int)fail(2,e->category,e->code):0;}
static int kerror(int rc,const et_kernel_error *e) {return rc?(int)fail(1,e->category,e->code):0;}
#ifdef ET_M3T_TESTING
static size_t alloc_remaining=SIZE_MAX;
void et_m3t_test_fail_alloc_after(size_t n) {alloc_remaining=n;}
#endif
static void *allocate(size_t n) {
#ifdef ET_M3T_TESTING
  if (!alloc_remaining) {bad(ET_M3T_INTERNAL,ET_M3T_CODE_ALLOCATION);return NULL;}
  if (alloc_remaining!=SIZE_MAX) --alloc_remaining;
#endif
  void *p=calloc(1,n);if(!p)bad(ET_M3T_INTERNAL,ET_M3T_CODE_ALLOCATION);return p;
}
static void enroll(record *r,int kind) {r->kind=kind;r->next=registry;registry=r;}
static void *admit(void *p,int kind,int allow_dead) {
  record *r;for(r=registry;r && r!=(record*)p;r=r->next) {}
  if(!r || r->kind!=kind) {bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);return NULL;}
  if(r->state<0 && !allow_dead) {bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);return NULL;}
  return r;
}
typedef struct initializer {record r; int64_t words[4];} initializer;
typedef struct owner {
  record r; initializer *original,*successor; et_f32_parameter *p[14];
  void *handles[14]; size_t initialized; void *active;
} owner;
static owner *staged_owner;
typedef struct input {record r;et_i64_tensor *tensor;} input;
typedef struct logits {record r;et_f32_tensor *tensor;} logits;
static const uint64_t parameter_shape[14][2] = {
 {4,4},{4,4},{4,4},{4,4},{4,8},{8,4},{4,0},{4,0},{4,0},{4,0},
 {256,4},{4,0},{4,0},{2,4}};
static size_t parameter_rank(size_t i) {return parameter_shape[i][1]?2:1;}
static const uint64_t token_shape[]={1,2,4},head_shape[]={1,2,2,2},
 wide_shape[]={1,2,8},logits_shape[]={1,2,256},ids_shape[]={1,2};
static et_kernel_runtime *runtimes[3];
static const et_kernel_provider_v1 *resolver(void *ctx,const char *symbol) {
 return strcmp(symbol,ET_KERNEL_PROVIDER_SYMBOL_V1)?NULL:ctx;
}
static int discover(void) {
 if(runtimes[0])return 0;
 const et_kernel_provider_v1 *p[]={et_n3k_kernel_provider_v1(),et_n2_kernel_provider_v1(),et_a2_kernel_provider_v1()};
 et_kernel_runtime *staged[3]={0};et_kernel_error e;
 for(size_t i=0;i<3;i++) {
  if(kerror(et_kernel_runtime_discover(resolver,(void*)p[i],&staged[i],&e),&e))goto failed;
  for(size_t j=0;j<i;j++)for(size_t k=0;k<p[i]->capability_count;k++)
   for(size_t l=0;l<p[j]->capability_count;l++) {
    const et_kernel_capability_v1 *a=(const void*)((const unsigned char*)p[i]->capabilities+k*p[i]->capability_stride);
    const et_kernel_capability_v1 *b=(const void*)((const unsigned char*)p[j]->capabilities+l*p[j]->capability_stride);
    if(!strcmp(a->name,b->name)){bad(ET_M3T_INTERNAL,ET_M3T_CODE_TOPOLOGY);goto failed;}
   }
 }
 memcpy(runtimes,staged,sizeof(staged));return 0;
failed:for(size_t i=0;i<3;i++)et_kernel_runtime_destroy(staged[i]);return (int)error_category;
}
static et_f32_tensor *new_tensor(size_t rank,const uint64_t *shape) {
 et_f32_tensor *t=NULL;et_f32_tensor_error e;
 if(f32_error(et_f32_tensor_create_v1(rank,shape,&t,&e),&e))return NULL;return t;
}
static et_i64_tensor *new_ids(void) {
 et_i64_tensor *t=NULL;et_i64_tensor_error e;
 if(ierror(et_i64_tensor_create_v1(2,ids_shape,&t,&e),&e))return NULL;return t;
}
static void destroy_f32(et_f32_tensor **t) {et_f32_tensor_error e;(void)et_f32_tensor_destroy_v1(t,&e);}
static void destroy_i64(et_i64_tensor **t) {et_i64_tensor_error e;(void)et_i64_tensor_destroy_v1(t,&e);}
static et_f32_tensor *value(owner *o,size_t i) {
 const et_f32_tensor *t=NULL;et_f32_tensor_error e;
 if(f32_error(et_f32_parameter_value_tensor_v1(o->p[i],&t,&e),&e))return NULL;
 return (et_f32_tensor*)t;
}
static int guard_begin(et_f32_tensor *t,et_f32_scoped_guard_internal *g) {
 et_f32_tensor_error e;return f32_error(et_f32_tensor_scoped_begin_internal(t,g,&e),&e);
}
static int copy_tensor(et_f32_tensor *dst,et_f32_tensor *src) {
 et_f32_scoped_guard_internal a={0},b={0};int rc=guard_begin(src,&a);
 if(!rc)rc=guard_begin(dst,&b);
 if(!rc && (a.view.rank!=b.view.rank || a.view.byte_length!=b.view.byte_length ||
    memcmp(a.view.shape,b.view.shape,a.view.rank*sizeof(uint64_t))))
  rc=(int)bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);
 if(!rc)memcpy(b.view.data,a.view.data,a.view.byte_length);
 et_f32_tensor_scoped_end_internal(&b);et_f32_tensor_scoped_end_internal(&a);return rc;
}
void *et_m3t_private_initializer_create_v1(int64_t seed) {
 clear();if(seed<0) {bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);return NULL;}
 initializer *s=allocate(sizeof(*s));if(!s)return NULL;
 s->words[0]=1;s->words[1]=seed;enroll(&s->r,INITIALIZER);return s;
}
int64_t et_m3t_private_initializer_word_v1(void *p,int64_t i) {
 clear();initializer *s=admit(p,INITIALIZER,0);if(!s)return 0;
 if(i<0||i>3){bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);return 0;}return s->words[i];
}
int64_t et_m3t_private_initializer_seal_v1(void *p) {
 clear();initializer *s=admit(p,INITIALIZER,0);if(!s)return error_category;
 if(s->r.state!=0)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 s->r.state=1;return 0;
}
int64_t et_m3t_private_initializer_abort_v1(void *p) {
 clear();initializer *s=admit(p,INITIALIZER,1);if(!s)return error_category;
 if(s->r.state==1)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 s->r.state=-1;return 0;
}
static int preflight_capabilities(void);
void *et_m3t_private_owner_create_v1(void *p) {
 clear();initializer *s=admit(p,INITIALIZER,0);if(!s)return NULL;
 if(staged_owner){bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);return NULL;}
 if(s->r.state!=1){bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);return NULL;}
 uint64_t low,high;memcpy(&low,&s->words[2],8);memcpy(&high,&s->words[3],8);
 if(high==UINT64_MAX && low>UINT64_MAX-290) {bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_COUNTER);return NULL;}
 if(discover() || preflight_capabilities())return NULL;
 owner *o=allocate(sizeof(*o));if(!o)return NULL;
 o->successor=allocate(sizeof(*o->successor));if(!o->successor){free(o);return NULL;}
 memcpy(o->successor->words,s->words,sizeof(s->words));o->original=s;
 for(size_t i=0;i<14;i++) {
  et_f32_tensor *t=new_tensor(parameter_rank(i),parameter_shape[i]);et_f32_tensor_error e;
  if(!t)goto failed;
  int rc=et_f32_parameter_create_v1(t,&o->p[i],&e);destroy_f32(&t);
  if(f32_error(rc,&e))goto failed;
 }
 enroll(&o->successor->r,INITIALIZER);enroll(&o->r,OWNER);staged_owner=o;return o;
failed:
 for(size_t i=0;i<14;i++)if(o->p[i]){et_f32_tensor_error e;et_f32_parameter_destroy_v1(&o->p[i],&e);}
 free(o->successor);free(o);return NULL;
}
void *et_m3t_private_owner_parameter_v1(void *p,int64_t i) {
 clear();owner *o=admit(p,OWNER,0);if(!o)return NULL;
 if(i<0||i>=14){bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);return NULL;}
 return o->p[i];
}
int64_t et_m3t_private_owner_bind_v1(void *p,int64_t i,void *handle) {
 clear();owner *o=admit(p,OWNER,0);if(!o)return error_category;
 if(o->r.state!=0)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 if(i<0||i>=14||!handle)return bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
 if(o->handles[i])return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 for(size_t j=0;j<14;j++)if(o->handles[j]==handle)return bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_TOPOLOGY);
 et_f32_tensor_error e;
 if(f32_error(et_f32_parameter_bind_identity_v1(o->p[i],handle,&e),&e))return error_category;
 o->handles[i]=handle;return 0;
}
static et_kernel_tensor_view_v1 view(void *p,size_t bytes,const char *dtype,size_t rank,const uint64_t *shape) {
 return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),p,bytes,dtype,"cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,rank,shape};
}
static int dispatch(int runtime,const char *cap,const char *op,size_t rank,const uint64_t *shape,
 et_kernel_tensor_view_v1 *in,size_t ni,et_kernel_tensor_view_v1 *out,size_t no) {
 et_kernel_request_v1 request={sizeof(request),op,"f32","cpu",rank,shape,1,{0}};
 et_kernel_call_v1 call={sizeof(call),cap,&request,ni,sizeof(*in),ni*sizeof(*in),in,no,sizeof(*out),no*sizeof(*out),out};
 et_kernel_error e;return kerror(et_kernel_runtime_dispatch(runtimes[runtime],&call,&e),&e);
}
int64_t et_m3t_private_owner_initialize_v1(void *p,int64_t i) {
 clear();owner *o=admit(p,OWNER,0);if(!o)return error_category;
 if(o->r.state!=0)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 if(i<0||i>=14)return bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
 if((size_t)i!=o->initialized)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 for(size_t j=0;j<14;j++)if(!o->handles[j])return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_TOPOLOGY);
 et_f32_tensor *t=value(o,(size_t)i);if(!t)return error_category;
 int rc=0;
 if(parameter_rank((size_t)i)==1) {
  uint32_t bits[4];uint32_t bit=(i==7||i==9||i==12)?UINT32_C(0x3f800000):0;
  for(size_t j=0;j<4;j++)bits[j]=bit;
  et_f32_tensor_error e;rc=f32_error(et_f32_tensor_copy_bits_from_v1(t,bits,4,&e),&e);
 } else {
  et_f32_scoped_guard_internal g={0};rc=guard_begin(t,&g);
  if(!rc) {
   const uint64_t four[]={4};int64_t next[4]={0};
   et_kernel_tensor_view_v1 in[]={view(o->successor->words,32,"i64",1,four)};
   et_kernel_tensor_view_v1 out[]={g.view,view(next,32,"i64",1,four)};
   rc=dispatch(0,"n3k.matrix-init","n3k.matrix-init.uniform",2,parameter_shape[i],in,1,out,2);
   if(!rc)memcpy(o->successor->words,next,32);
  }
  et_f32_tensor_scoped_end_internal(&g);
 }
 if(!rc)o->initialized++;return rc;
}
static int owner_ready(owner *o) {
 if(o->r.state!=0||o->initialized!=14)return (int)bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 for(size_t i=0;i<14;i++)if(!o->handles[i])return (int)bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_TOPOLOGY);
 return 0;
}
void *et_m3t_private_owner_successor_v1(void *p) {
 clear();owner *o=admit(p,OWNER,0);if(!o)return NULL;
 if(o->r.state==0 && owner_ready(o))return NULL;return o->successor;
}
int64_t et_m3t_private_owner_seal_v1(void *p) {
 clear();owner *o=admit(p,OWNER,0);if(!o)return error_category;
 if(owner_ready(o))return error_category;o->r.state=1;staged_owner=NULL;return 0;
}
static int32_t preflight_error(et_f32_tensor_error *error) {
 if(error){memset(error,0,sizeof(*error));error->category=ET_F32_TENSOR_ERROR_INVALID_STATE;
  error->code=ET_F32_TENSOR_CODE_INVALID_HANDLE;}
 return ET_F32_TENSOR_ERROR_INVALID_STATE;
}
int32_t et_m3t_construction_parameter_preflight_internal(void *parameter,
    const void *exact_handle, et_f32_tensor_error *error) {
 owner *found=NULL;size_t index=0;
 for(record *r=registry;r;r=r->next)if(r->kind==OWNER && r->state==0) {
  owner *o=(owner*)r;
  for(size_t i=0;i<14;i++)if(o->p[i]==parameter){found=o;index=i;}
 }
 if(!found || found!=staged_owner || (found->handles[index] && found->handles[index]!=exact_handle))return preflight_error(error);
 et_f32_parameter *p=parameter;const void *actual=NULL;et_f32_tensor_error local;
 if(!et_f32_parameter_is_live_v1(p))return preflight_error(error);
 int32_t rc=et_f32_parameter_identity_v1(p,&actual,&local);
 /* For a registered live parameter, identity's sole remaining rejection is
  * genuinely unbound. A NULL expected handle never bypasses a real binding. */
 if(rc && (local.category!=ET_F32_TENSOR_ERROR_INVALID_STATE ||
           local.code!=ET_F32_TENSOR_CODE_INVALID_HANDLE)) {
  if(error)*error=local;return rc;
 }
 if(actual!=exact_handle)return preflight_error(error);
 /* Even a partial P1 ledger must prove the entire fixed native owner can be
  * destroyed. Unregistered carriers are still owned exclusively by this owner. */
 for(size_t i=0;i<14;i++) {
  if(found->handles[i]) {
   rc=et_f32_parameter_validate_identity_v1(found->p[i],found->handles[i],error);
   if(rc)return rc;
  }
  rc=et_m3t_f32_parameter_preflight_internal(found->p[i],error);
  if(rc)return rc;
 }
 return 0;
}
int64_t et_m3t_private_owner_abort_v1(void *p) {
 clear();owner *o=admit(p,OWNER,1);if(!o)return error_category;
 if(o->r.state<0)return 0;
 if(o->r.state!=0)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 /* Repeat the fixed collection preflight before any destructive step. */
 for(size_t i=0;i<14;i++) {
  et_f32_tensor_error e;
  const void *actual=NULL;
  if(!et_f32_parameter_is_live_v1(o->p[i]))return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
  (void)et_f32_parameter_identity_v1(o->p[i],&actual,&e);
  if(f32_error(et_m3t_construction_parameter_preflight_internal(o->p[i],actual,&e),&e))return error_category;
 }
 for(size_t i=0;i<14;i++)if(o->p[i]) {
  et_f32_tensor_error e;if(f32_error(et_f32_parameter_destroy_v1(&o->p[i],&e),&e))return error_category;
 }
 o->successor->r.state=-1;o->r.state=-1;staged_owner=NULL;return 0;
}
void *et_m3t_private_input_create_v1(void) {
 clear();input *i=allocate(sizeof(*i));if(!i)return NULL;
 i->tensor=new_ids();if(!i->tensor){free(i);return NULL;}
 enroll(&i->r,INPUT);return i;
}
int64_t et_m3t_private_input_copy_v1(void *p,int64_t first,int64_t second) {
 clear();input *i=admit(p,INPUT,0);if(!i)return error_category;
 if(first<0||first>=256||second<0||second>=256)return bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);
 int64_t words[]={first,second};et_i64_tensor_error e;
 return ierror(et_i64_tensor_copy_from_v1(i->tensor,words,2,&e),&e);
}
int64_t et_m3t_private_input_copy_tokenizer_v1(void *p,void *t) {
 clear();input *i=admit(p,INPUT,0);if(!i)return error_category;
 int64_t n=et_t1_i64_shell_length_v1(t);
 if(et_t1_i64_shell_last_status_v1())return bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);
 if(n!=2)return bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);
 int64_t a=et_t1_i64_shell_read_v1(t,0);
 if(et_t1_i64_shell_last_status_v1())return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 int64_t b=et_t1_i64_shell_read_v1(t,1);
 if(et_t1_i64_shell_last_status_v1())return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 return et_m3t_private_input_copy_v1(i,a,b);
}
int64_t et_m3t_private_input_release_v1(void *p) {
 clear();input *i=admit(p,INPUT,1);if(!i)return error_category;
 if(i->r.state<0)return 0;et_i64_tensor_error e;
 if(ierror(et_i64_tensor_destroy_v1(&i->tensor,&e),&e))return error_category;
 i->r.state=-1;return 0;
}
void *et_m3t_private_logits_create_v1(void) {
 clear();logits *l=allocate(sizeof(*l));if(!l)return NULL;
 l->tensor=new_tensor(3,logits_shape);if(!l->tensor){free(l);return NULL;}
 enroll(&l->r,LOGITS);return l;
}
static unsigned char *byte_payload(void *header,int64_t count) {
 if(count!=2048||!header || (uintptr_t)header>UINTPTR_MAX-2056) {
  bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);return NULL;
 }
 int64_t encoded;memcpy(&encoded,header,8);
 if(encoded!=2048){bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);return NULL;}
 return (unsigned char*)header+8;
}
int64_t et_m3t_private_logits_copy_from_v1(void *p,void *header,int64_t count) {
 clear();logits *l=admit(p,LOGITS,0);if(!l)return error_category;
 unsigned char *bytes=byte_payload(header,count);if(!bytes)return error_category;
 uint32_t bits[512];for(size_t i=0;i<512;i++)bits[i]=(uint32_t)bytes[4*i]|((uint32_t)bytes[4*i+1]<<8)|((uint32_t)bytes[4*i+2]<<16)|((uint32_t)bytes[4*i+3]<<24);
 et_f32_tensor_error e;return f32_error(et_f32_tensor_copy_bits_from_v1(l->tensor,bits,512,&e),&e);
}
int64_t et_m3t_private_logits_copy_to_v1(void *p,void *header,int64_t count) {
 clear();logits *l=admit(p,LOGITS,0);if(!l)return error_category;
 unsigned char *bytes=byte_payload(header,count);if(!bytes)return error_category;
 uint32_t bits[512];et_f32_tensor_error e;
 if(f32_error(et_f32_tensor_copy_bits_to_v1(l->tensor,bits,512,&e),&e))return error_category;
 for(size_t i=0;i<512;i++)for(size_t j=0;j<4;j++)bytes[4*i+j]=(unsigned char)(bits[i]>>(8*j));
 return 0;
}
int64_t et_m3t_private_logits_release_v1(void *p) {
 clear();logits *l=admit(p,LOGITS,1);if(!l)return error_category;
 if(l->r.state<0)return 0;et_f32_tensor_error e;
 if(f32_error(et_f32_tensor_destroy_v1(&l->tensor,&e),&e))return error_category;
 l->r.state=-1;return 0;
}
/* Token-major [1,2,4], heads [1,2,2,2], FFN [1,2,8], logits
 * [1,2,256]; final gradient slots use the canonical parameter shapes. */
enum slot {
 ET,EP,X,N1,QT,KT,VT,QH,KH,VH,AH,AT,AO,R,N2,FU,FG,FD,Y,NF,Z,
 DZ,DNF,DY,DRSKIP,DFD,DFG,DFU,DN2,DRFFN,DR,DXSKIP,DAO,DAT,DAH,
 DQH,DKH,DVH,DQT,DKT,DVT,DN1Q,DN1K,DN1V,DN1,DXATTENTION,DX,DET,DEP,
 GKEY,GOUTPUT,GQUERY,GVALUE,GDOWN,GUP,GN1B,GN1W,GN2B,GN2W,GTIED,GFB,GFW,GPOS,
 GHEAD,GTOKEN,SLOT_COUNT
};
typedef struct workspace {
 record r;owner *model;et_f32_tensor *saved[14],*slots[SLOT_COUNT],*epsilon;
 et_i64_tensor *ids,*positions;unsigned char keep[4];unsigned char ready[SLOT_COUNT];int mode,busy;
} workspace;
static size_t slot_shape(size_t i,const uint64_t **shape) {
 if(i>=GKEY && i<=GPOS){*shape=parameter_shape[i-GKEY];return parameter_rank(i-GKEY);}
 if(i==GHEAD||i==GTOKEN){*shape=parameter_shape[10];return 2;}
 if(i==Z||i==DZ){*shape=logits_shape;return 3;}
 if(i==FU||i==FG||i==DFU||i==DFG){*shape=wide_shape;return 3;}
 if(i==QH||i==KH||i==VH||i==AH||i==DAH||i==DQH||i==DKH||i==DVH){*shape=head_shape;return 4;}
 *shape=token_shape;return 3;
}
static void destroy_workspace(workspace *w) {
 for(size_t i=0;i<14;i++)destroy_f32(&w->saved[i]);
 for(size_t i=0;i<SLOT_COUNT;i++)destroy_f32(&w->slots[i]);
 destroy_f32(&w->epsilon);destroy_i64(&w->ids);destroy_i64(&w->positions);
}
void *et_m3t_private_workspace_create_v1(void *p) {
 clear();owner *o=admit(p,OWNER,0);if(!o)return NULL;
 if(o->r.state!=1){bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);return NULL;}
 workspace *w=allocate(sizeof(*w));if(!w)return NULL;w->model=o;
 for(size_t i=0;i<14;i++)if(!(w->saved[i]=new_tensor(parameter_rank(i),parameter_shape[i])))goto failed;
 for(size_t i=0;i<SLOT_COUNT;i++) {
  const uint64_t *shape;size_t rank=slot_shape(i,&shape);
  if(!(w->slots[i]=new_tensor(rank,shape)))goto failed;
 }
 w->epsilon=new_tensor(0,NULL);if(!w->epsilon)goto failed;
 uint32_t eps=UINT32_C(0x3727c5ac);et_f32_tensor_error fe;
 if(f32_error(et_f32_tensor_copy_bits_from_v1(w->epsilon,&eps,1,&fe),&fe))goto failed;
 w->ids=new_ids();if(!w->ids)goto failed;w->positions=new_ids();if(!w->positions)goto failed;
 int64_t positions[]={0,1};et_i64_tensor_error ie;
 if(ierror(et_i64_tensor_copy_from_v1(w->positions,positions,2,&ie),&ie))goto failed;
 memset(w->keep,1,4);enroll(&w->r,WORKSPACE);return w;
failed:destroy_workspace(w);free(w);return NULL;
}
static int active(workspace *w) {
 if(w->busy)return (int)bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_REENTRANCY);
 if(w->r.state<1 || w->model->active!=w)return (int)bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 return 0;
}
int64_t et_m3t_private_workspace_begin_v1(void *p,void *ip,int64_t mode) {
 clear();workspace *w=admit(p,WORKSPACE,0);if(!w)return error_category;
 input *i=admit(ip,INPUT,0);if(!i)return error_category;
 if(w->busy)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_REENTRANCY);
 if(w->r.state || w->model->active)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 if(mode!=0&&mode!=1)return bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
 int64_t ids[2];et_i64_tensor_error e;
 if(ierror(et_i64_tensor_copy_to_v1(i->tensor,ids,2,&e),&e))return error_category;
 if(ids[0]<0||ids[0]>=256||ids[1]<0||ids[1]>=256)return bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);
 for(size_t n=0;n<14;n++) {
  et_f32_tensor_error fe;
  if(f32_error(et_f32_parameter_validate_identity_v1(w->model->p[n],w->model->handles[n],&fe),&fe))return error_category;
  et_f32_tensor *v=value(w->model,n);if(!v || copy_tensor(w->saved[n],v))return error_category;
 }
 if(ierror(et_i64_tensor_copy_from_v1(w->ids,ids,2,&e),&e))return error_category;
 memset(w->ready,0,sizeof(w->ready));w->mode=(int)mode;w->r.state=1;w->model->active=w;return 0;
}
int64_t et_m3t_private_workspace_vjp_begin_v1(void *p,void *lp) {
 clear();workspace *w=admit(p,WORKSPACE,0);if(!w)return error_category;
 logits *l=admit(lp,LOGITS,0);if(!l)return error_category;
 if(active(w))return error_category;
 if(w->r.state!=1||!w->ready[Z])return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 if(copy_tensor(w->slots[DZ],l->tensor))return error_category;
 w->ready[DZ]=1;w->r.state=2;return 0;
}
int64_t et_m3t_private_workspace_copy_logits_v1(void *p,void *lp) {
 clear();workspace *w=admit(p,WORKSPACE,0);if(!w)return error_category;
 logits *l=admit(lp,LOGITS,0);if(!l)return error_category;
 if(active(w))return error_category;
 if(!w->ready[Z])return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 return copy_tensor(l->tensor,w->slots[Z]);
}
int64_t et_m3t_private_workspace_check_primals_v1(void *p,int64_t mode) {
 clear();workspace *w=admit(p,WORKSPACE,0);if(!w)return error_category;
 if(active(w))return error_category;
 if(mode!=0&&mode!=1)return bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
 if(w->mode!=mode)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_STALE);
 for(size_t i=0;i<14;i++) {
  et_f32_tensor_error e;int32_t same=0;
  if(f32_error(et_f32_parameter_validate_identity_v1(w->model->p[i],w->model->handles[i],&e),&e))return error_category;
  et_f32_tensor *v=value(w->model,i);if(!v)return error_category;
  et_f32_scoped_guard_internal a={0},b={0};int rc=guard_begin(v,&a);
  if(!rc)rc=guard_begin(w->saved[i],&b);
  if(!rc)same=a.view.rank==b.view.rank && a.view.byte_length==b.view.byte_length &&
   !memcmp(a.view.shape,b.view.shape,a.view.rank*sizeof(uint64_t)) && !memcmp(a.view.data,b.view.data,a.view.byte_length);
  et_f32_tensor_scoped_end_internal(&b);et_f32_tensor_scoped_end_internal(&a);
  if(rc)return rc;if(!same)return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_STALE);
 }
 uint32_t eps;et_f32_tensor_error e;
 if(f32_error(et_f32_tensor_copy_bits_to_v1(w->epsilon,&eps,1,&e),&e))return error_category;
 int64_t pos[2];et_i64_tensor_error ie;
 if(ierror(et_i64_tensor_copy_to_v1(w->positions,pos,2,&ie),&ie))return error_category;
 if(eps!=UINT32_C(0x3727c5ac)||pos[0]!=0||pos[1]!=1 || memcmp(w->keep,"\1\1\1\1",4))return bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_STALE);
 return 0;
}
static int workspace_unborrowed(workspace *w) {
 if(w->busy)return (int)bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_REENTRANCY);
 et_f32_scoped_guard_internal guards[14+SLOT_COUNT+1]={0};size_t n=0;int rc=0;
 for(size_t i=0;i<14 && !rc;i++)rc=guard_begin(w->saved[i],&guards[n++]);
 for(size_t i=0;i<SLOT_COUNT && !rc;i++)rc=guard_begin(w->slots[i],&guards[n++]);
 if(!rc)rc=guard_begin(w->epsilon,&guards[n++]);
 et_i64_tensor_borrow *a=NULL,*b=NULL;et_i64_tensor_error e;
 if(!rc)rc=ierror(et_i64_tensor_borrow_begin_v1(w->ids,&a,&e),&e);
 if(!rc)rc=ierror(et_i64_tensor_borrow_begin_v1(w->positions,&b,&e),&e);
 et_i64_tensor_borrow_end_v1(&b,&e);et_i64_tensor_borrow_end_v1(&a,&e);
 while(n)et_f32_tensor_scoped_end_internal(&guards[--n]);return rc;
}
int64_t et_m3t_private_workspace_reset_v1(void *p) {
 clear();workspace *w=admit(p,WORKSPACE,0);if(!w)return error_category;
 if(workspace_unborrowed(w))return error_category;
 if(w->model->active==w)w->model->active=NULL;
 w->r.state=0;memset(w->ready,0,sizeof(w->ready));return 0;
}
int64_t et_m3t_private_workspace_release_v1(void *p) {
 clear();workspace *w=admit(p,WORKSPACE,1);if(!w)return error_category;
 if(w->r.state<0)return 0;
 if(workspace_unborrowed(w))return error_category;
 if(w->model->active==w)w->model->active=NULL;
 w->r.state=-1;memset(w->ready,0,sizeof(w->ready));destroy_workspace(w);return 0;
}
void *et_m3t_private_workspace_gradient_v1(void *p,int64_t i) {
 clear();workspace *w=admit(p,WORKSPACE,0);if(!w)return NULL;
 if(active(w))return NULL;
 if(i<0||i>=14){bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);return NULL;}
 if(w->r.state!=2){bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);return NULL;}
 for(size_t n=GKEY;n<=GPOS;n++)if(!w->ready[n]){bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);return NULL;}
 return w->slots[GKEY+i];
}
/* Closed role descriptors are wiring, not a schedule. Only one descriptor is
 * consumed per public call; destinations become ready only after dispatch. */
enum { P0=SLOT_COUNT, EPS=P0+14, IDS, POS, KEEP };
enum family { EMBEDDING,LINEAR,NORM,GELU,RESIDUAL,SPLIT,MERGE,ATTENTION,SUM };
typedef struct role {
 int family,index,reverse,runtime;const char *cap,*op;
 size_t rank;uint64_t shape[6];size_t ni,no;int in[7],out[3];
} role;
static const role roles[]={
 {EMBEDDING,0,0,0,"n3k.embedding-forward","n3k.embedding.forward",4,{1,2,256,4},2,1,{IDS,P0+10},{ET}},
 {EMBEDDING,0,1,0,"n3k.embedding-backward","n3k.embedding.backward",4,{1,2,256,4},2,1,{IDS,DET},{GTOKEN}},
 {EMBEDDING,1,0,0,"n3k.embedding-forward","n3k.embedding.forward",4,{1,2,2,4},2,1,{POS,P0+13},{EP}},
 {EMBEDDING,1,1,0,"n3k.embedding-backward","n3k.embedding.backward",4,{1,2,2,4},2,1,{POS,DEP},{GPOS}},
 {LINEAR,0,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,2,4,4},2,1,{N1,P0+2},{QT}},
 {LINEAR,0,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,2,4,4},3,2,{N1,P0+2,DQT},{DN1Q,GQUERY}},
 {LINEAR,1,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,2,4,4},2,1,{N1,P0+0},{KT}},
 {LINEAR,1,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,2,4,4},3,2,{N1,P0+0,DKT},{DN1K,GKEY}},
 {LINEAR,2,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,2,4,4},2,1,{N1,P0+3},{VT}},
 {LINEAR,2,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,2,4,4},3,2,{N1,P0+3,DVT},{DN1V,GVALUE}},
 {LINEAR,3,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,2,4,4},2,1,{AT,P0+1},{AO}},
 {LINEAR,3,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,2,4,4},3,2,{AT,P0+1,DAO},{DAT,GOUTPUT}},
 {LINEAR,4,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,2,4,8},2,1,{N2,P0+5},{FU}},
 {LINEAR,4,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,2,4,8},3,2,{N2,P0+5,DFU},{DN2,GUP}},
 {LINEAR,5,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,2,8,4},2,1,{FG,P0+4},{FD}},
 {LINEAR,5,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,2,8,4},3,2,{FG,P0+4,DFD},{DFG,GDOWN}},
 {LINEAR,6,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,2,4,256},2,1,{NF,P0+10},{Z}},
 {LINEAR,6,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,2,4,256},3,2,{NF,P0+10,DZ},{DNF,GHEAD}},
 {NORM,0,0,1,"kernel.norm","layer-norm.forward",3,{1,2,4},4,1,{X,P0+7,P0+6,EPS},{N1}},
 {NORM,0,1,1,"kernel.norm","layer-norm.backward",3,{1,2,4},4,3,{X,P0+7,EPS,DN1},{DXATTENTION,GN1W,GN1B}},
 {NORM,1,0,1,"kernel.norm","layer-norm.forward",3,{1,2,4},4,1,{R,P0+9,P0+8,EPS},{N2}},
 {NORM,1,1,1,"kernel.norm","layer-norm.backward",3,{1,2,4},4,3,{R,P0+9,EPS,DN2},{DRFFN,GN2W,GN2B}},
 {NORM,2,0,1,"kernel.norm","layer-norm.forward",3,{1,2,4},4,1,{Y,P0+12,P0+11,EPS},{NF}},
 {NORM,2,1,1,"kernel.norm","layer-norm.backward",3,{1,2,4},4,3,{Y,P0+12,EPS,DNF},{DY,GFW,GFB}},
 {GELU,0,0,0,"n3k.gelu","n3k.gelu.forward",3,{1,2,8},1,1,{FU},{FG}},
 {GELU,0,1,0,"n3k.gelu","n3k.gelu.backward",3,{1,2,8},2,1,{FU,DFG},{DFU}},
 {RESIDUAL,0,0,0,"n3k.residual","n3k.residual.forward",3,{1,2,4},2,1,{ET,EP},{X}},
 {RESIDUAL,0,1,0,"n3k.residual","n3k.residual.backward",3,{1,2,4},1,2,{DX},{DET,DEP}},
 {RESIDUAL,1,0,0,"n3k.residual","n3k.residual.forward",3,{1,2,4},2,1,{X,AO},{R}},
 {RESIDUAL,1,1,0,"n3k.residual","n3k.residual.backward",3,{1,2,4},1,2,{DR},{DXSKIP,DAO}},
 {RESIDUAL,2,0,0,"n3k.residual","n3k.residual.forward",3,{1,2,4},2,1,{R,FD},{Y}},
 {RESIDUAL,2,1,0,"n3k.residual","n3k.residual.backward",3,{1,2,4},1,2,{DY},{DRSKIP,DFD}},
 {SPLIT,0,0,0,"n3k.head-layout","n3k.heads.split.forward",4,{1,2,2,2},1,1,{QT},{QH}},
 {SPLIT,0,1,0,"n3k.head-layout","n3k.heads.split.backward",4,{1,2,2,2},1,1,{DQH},{DQT}},
 {SPLIT,1,0,0,"n3k.head-layout","n3k.heads.split.forward",4,{1,2,2,2},1,1,{KT},{KH}},
 {SPLIT,1,1,0,"n3k.head-layout","n3k.heads.split.backward",4,{1,2,2,2},1,1,{DKH},{DKT}},
 {SPLIT,2,0,0,"n3k.head-layout","n3k.heads.split.forward",4,{1,2,2,2},1,1,{VT},{VH}},
 {SPLIT,2,1,0,"n3k.head-layout","n3k.heads.split.backward",4,{1,2,2,2},1,1,{DVH},{DVT}},
 {MERGE,0,0,0,"n3k.head-layout","n3k.heads.merge.forward",4,{1,2,2,2},1,1,{AH},{AT}},
 {MERGE,0,1,0,"n3k.head-layout","n3k.heads.merge.backward",4,{1,2,2,2},1,1,{DAT},{DAH}},
 {ATTENTION,0,0,2,"kernel.causal-attention","causal-attention.forward",6,{1,2,2,2,2,2},6,1,{QH,KH,VH,POS,POS,KEEP},{AH}},
 {ATTENTION,0,1,2,"kernel.causal-attention","causal-attention.backward",6,{1,2,2,2,2,2},7,3,{QH,KH,VH,POS,POS,KEEP,DAH},{DQH,DKH,DVH}},
 {SUM,0,1,0,"n3k.activation-sum","n3k.sum3.forward",3,{1,2,4},3,1,{DN1Q,DN1K,DN1V},{DN1}},
 {SUM,1,1,0,"n3k.activation-sum","n3k.sum2.forward",3,{1,2,4},2,1,{DXSKIP,DXATTENTION},{DX}},
 {SUM,2,1,0,"n3k.activation-sum","n3k.sum2.forward",3,{1,2,4},2,1,{DRSKIP,DRFFN},{DR}},
 {SUM,3,1,0,"n3k.tied-sum","n3k.sum2.forward",2,{256,4},2,1,{GHEAD,GTOKEN},{GTIED}},
};
static int require_row(int runtime,const char *cap,const char *op,size_t rank,const uint64_t *shape) {
 et_kernel_request_v1 req={sizeof(req),op,"f32","cpu",rank,shape,1,{0}};
 const et_kernel_capability_v1 *entry=NULL;et_kernel_error e;
 return kerror(et_kernel_runtime_capability_require(runtimes[runtime],cap,&req,&entry,&e),&e);
}
static int preflight_capabilities(void) {
 for(size_t i=0;i<sizeof(roles)/sizeof(*roles);i++) {
  const role *r=&roles[i];if(require_row(r->runtime,r->cap,r->op,r->rank,r->shape))return (int)error_category;
 }
 for(size_t i=0;i<14;i++)if(parameter_rank(i)==2 &&
  require_row(0,"n3k.matrix-init","n3k.matrix-init.uniform",2,parameter_shape[i]))return (int)error_category;
 return 0;
}
static et_f32_tensor *f32_source(workspace *w,int ref) {
 if(ref<SLOT_COUNT)return w->slots[ref];
 if(ref<EPS)return w->saved[ref-P0];
 return w->epsilon;
}
static int execute(workspace *w,const role *r) {
 if(active(w))return (int)error_category;
 if(w->r.state!=(r->reverse?2:1))return (int)bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 for(size_t i=0;i<r->ni;i++)if(r->in[i]<SLOT_COUNT && !w->ready[r->in[i]])
  return (int)bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 for(size_t i=0;i<r->no;i++)if(w->ready[r->out[i]])return (int)bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 w->busy=1;
 et_f32_scoped_guard_internal guards[10]={0};et_f32_tensor *owners[10]={0};size_t ng=0;
 et_i64_tensor_borrow *leases[2]={0};const et_kernel_tensor_view_v1 *iv[2]={0};
 int acquisition[12];size_t acquired=0;
 et_kernel_tensor_view_v1 in[7],out[3];et_i64_tensor_error ie;int rc=0;
 const uint64_t keep_shape[]={1,2,2};
 for(size_t n=0;n<r->ni+r->no && !rc;n++) {
  int is_output=n>=r->ni;int ref=is_output?r->out[n-r->ni]:r->in[n];
  et_kernel_tensor_view_v1 *dest=is_output?&out[n-r->ni]:&in[n];
  if(ref==KEEP) {*dest=view(w->keep,4,"bool",3,keep_shape);continue;}
  if(ref==IDS||ref==POS) {
   size_t j=ref==IDS?0:1;
   if(!leases[j]) {
    rc=ierror(et_i64_tensor_borrow_begin_v1(j?w->positions:w->ids,&leases[j],&ie),&ie);
    if(!rc){acquisition[acquired++]=10+(int)j;rc=ierror(et_i64_tensor_borrow_view_v1(leases[j],&iv[j],&ie),&ie);}
   }
   if(!rc)*dest=*iv[j];continue;
  }
  et_f32_tensor *t=f32_source(w,ref);size_t j=0;
  while(j<ng && owners[j]!=t)j++;
  if(j==ng) {owners[ng]=t;rc=guard_begin(t,&guards[ng]);if(!rc)acquisition[acquired++]=(int)ng;ng++;}
  if(!rc)*dest=guards[j].view;
 }
 if(!rc)rc=dispatch(r->runtime,r->cap,r->op,r->rank,r->shape,in,r->ni,out,r->no);
 /* Cleanup cannot overwrite the captured error triple. Guard end does not
  * allocate. I1 end frees rather than retaining lease control shells. */
 while(acquired){int a=acquisition[--acquired];
  if(a>=10)et_i64_tensor_borrow_end_v1(&leases[a-10],&ie);
  else et_f32_tensor_scoped_end_internal(&guards[a]);
 }
 if(!rc)for(size_t i=0;i<r->no;i++)w->ready[r->out[i]]=1;
 w->busy=0;return rc;
}
static int64_t primitive(void *p,int family,int64_t index,int64_t reverse) {
 clear();workspace *w=admit(p,WORKSPACE,0);if(!w)return error_category;
 if(active(w))return error_category;
 if(reverse!=0 && reverse!=1)return bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
 for(size_t i=0;i<sizeof(roles)/sizeof(*roles);i++)if(roles[i].family==family && roles[i].index==index && roles[i].reverse==reverse)
  return execute(w,&roles[i]);
 return bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
}
#define ROLE_FUNCTION(name,family) \
 int64_t et_m3t_private_##name##_v1(void *w,int64_t i,int64_t b) {return primitive(w,family,i,b);}
ROLE_FUNCTION(embedding,EMBEDDING)
ROLE_FUNCTION(linear,LINEAR)
ROLE_FUNCTION(layer_norm,NORM)
ROLE_FUNCTION(residual,RESIDUAL)
ROLE_FUNCTION(heads_split,SPLIT)
#undef ROLE_FUNCTION
int64_t et_m3t_private_gelu_v1(void *w,int64_t b) {return primitive(w,GELU,0,b);}
int64_t et_m3t_private_heads_merge_v1(void *w,int64_t b) {return primitive(w,MERGE,0,b);}
int64_t et_m3t_private_attention_v1(void *w,int64_t b) {return primitive(w,ATTENTION,0,b);}
int64_t et_m3t_private_sum_v1(void *w,int64_t i) {return primitive(w,SUM,i,1);}
