/* Frame-local transport only. Eshkol owns the 21/25 numerical schedule. */
#include "cpu4_training_transport.h"
#if defined(ET_CPU4_TRAINING_PRIVATE)
#if !defined(ET_G3R_CANDIDATE_RETIRE_PRIVATE)
#error "CPU4 byte ingress requires reviewed I2 owned-span provenance"
#endif
#include "m3t_transport.h"
#include "m3t_f32_scoped.h"
#include "g3c4_model_owner_internal.h"
#include "eshkol_transformer/n3k_primitives_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#include "eshkol_transformer/a2_attention_abi.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

enum { C4T_INPUT=1, C4T_LOGITS, C4T_WORKSPACE };
enum { C4T_FORWARD=1, C4T_REVERSE, C4T_CONSUMED, C4T_DEAD };
typedef struct c4t_record {
 struct c4t_record *next; void *lease; size_t extent; int kind,state;
} c4t_record;
static c4t_record *c4t_registry;
static int64_t c4t_domain,c4t_category,c4t_code;
static void c4t_clear(void) {c4t_domain=c4t_category=c4t_code=0;}
static int c4t_fail(int domain,int category,int code) {
 c4t_domain=domain;c4t_category=category;c4t_code=code;return category;
}
static int c4t_bad(int category,int code) {return c4t_fail(0,category,code);}
static int c4t_ferror(int rc,const et_f32_tensor_error *e) {
 return rc?c4t_fail(3,e->category,e->code):0;
}
static int c4t_kerror(int rc,const et_kernel_error *e) {
 return rc?c4t_fail(1,e->category,e->code):0;
}
static int c4t_cerror(int64_t rc) {
 if(!rc)return 0;
 int domain=(int)et_g3c4_private_last_error_domain_v1();
 return c4t_fail(domain?domain:4,(int)et_g3c4_private_last_error_category_v1(),
                   (int)et_g3c4_private_last_error_code_v1());
}
int64_t et_cpu4_private_last_error_domain_v1(void) {return c4t_domain;}
int64_t et_cpu4_private_last_error_category_v1(void) {return c4t_category;}
int64_t et_cpu4_private_last_error_code_v1(void) {return c4t_code;}
#ifdef ET_CPU4_TRANSPORT_TESTING
static size_t c4t_alloc_remaining=SIZE_MAX;
#endif
static void *c4t_allocate(size_t bytes) {
#ifdef ET_CPU4_TRANSPORT_TESTING
 if(!c4t_alloc_remaining){c4t_bad(ET_M3T_INTERNAL,ET_M3T_CODE_ALLOCATION);return NULL;}
 if(c4t_alloc_remaining!=SIZE_MAX)--c4t_alloc_remaining;
#endif
 void *p=calloc(1,bytes);if(!p)c4t_bad(ET_M3T_INTERNAL,ET_M3T_CODE_ALLOCATION);return p;
}
static c4t_record *c4t_admit(void *p,int kind,int dead) {
 c4t_record *r=c4t_registry;while(r && r!=p)r=r->next;
 if(!r || r->kind!=kind){c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);return NULL;}
 if(r->state==C4T_DEAD){if(dead)return r;c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);return NULL;}
 if(c4t_cerror(et_g3c4_private_training_lease_check_v1(r->lease)))return NULL;
 return r;
}
static int c4t_publish(c4t_record *r,void *lease,int kind,size_t extent,int child) {
 if(c4t_cerror(et_g3c4_training_child_acquire_internal(lease,r,child)))return (int)c4t_category;
 r->lease=lease;r->kind=kind;r->extent=extent;r->state=C4T_FORWARD;
 r->next=c4t_registry;c4t_registry=r;return 0;
}
static int c4t_overlap(const void *a,size_t an,const void *b,size_t bn) {
 uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
 return y>UINTPTR_MAX-bn || (x<y+bn && y<x+an);
}
static unsigned char *c4t_bytes(c4t_record *r,void *header,int64_t bytes) {
 if(bytes!=4096 || !header || (uintptr_t)header>UINTPTR_MAX-4104) {
  c4t_bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);return NULL;
 }
 if(c4t_cerror(et_g3c4_training_span_disjoint_internal(r->lease,header,4104)))return NULL;
 for(c4t_record *item=c4t_registry;item;item=item->next)
  if(c4t_overlap(header,4104,item,item->extent)) {
   c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);return NULL;
  }
 if(et_f32_tensor_private_owned_span_overlap_v1(header,4104)) {
  c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);return NULL;
 }
 int64_t length;memcpy(&length,header,8);
 if(length!=4096){c4t_bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);return NULL;}
 return (unsigned char *)header+8;
}
static const uint64_t c4t_parameters[14][2]={
 {4,4},{4,4},{4,4},{4,4},{4,8},{8,4},{4,0},{4,0},{4,0},{4,0},
 {256,4},{4,0},{4,0},{4,4}};
static const uint64_t c4t_tokens[]={1,4,4},c4t_heads[]={1,2,4,2},
 c4t_wide[]={1,4,8},c4t_logits_shape[]={1,4,256},c4t_ids_shape[]={1,4};
static size_t c4t_rank(size_t i){return c4t_parameters[i][1]?2:1;}
static et_f32_tensor *c4t_new(size_t rank,const uint64_t *shape) {
 et_f32_tensor *t=NULL;et_f32_tensor_error e;
 return c4t_ferror(et_f32_tensor_create_v1(rank,shape,&t,&e),&e)?NULL:t;
}
static void c4t_destroy(et_f32_tensor **t) {
 et_f32_tensor_error e;if(et_f32_tensor_destroy_v1(t,&e))abort();
}
static int c4t_guard(et_f32_tensor *t,et_f32_scoped_guard_internal *g) {
 et_f32_tensor_error e;return c4t_ferror(et_f32_tensor_scoped_begin_internal(t,g,&e),&e);
}
static void c4t_end(et_f32_scoped_guard_internal *g) {
 if(et_f32_tensor_scoped_end_internal(g))abort();
}
static int c4t_copy(et_f32_tensor *dst,et_f32_tensor *src) {
 et_f32_scoped_guard_internal a={0},b={0};int rc=c4t_guard(src,&a);
 if(!rc)rc=c4t_guard(dst,&b);
 if(!rc && (a.view.rank!=b.view.rank || a.view.byte_length!=b.view.byte_length ||
    (a.view.rank && memcmp(a.view.shape,b.view.shape,a.view.rank*sizeof(uint64_t)))))
  rc=c4t_bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);
 if(!rc)memcpy(b.view.data,a.view.data,a.view.byte_length);
 c4t_end(&b);c4t_end(&a);return rc;
}
typedef struct c4t_input {c4t_record r;int64_t ids[4];} c4t_input;
typedef struct c4t_logits {c4t_record r;et_f32_tensor *tensor;} c4t_logits;
void *et_cpu4_private_input_create_v1(void *lease,int64_t a,int64_t b,int64_t c,int64_t d) {
 c4t_clear();if(c4t_cerror(et_g3c4_private_training_lease_check_v1(lease)))return NULL;
 int64_t ids[]={a,b,c,d};
 for(size_t i=0;i<4;i++)if(ids[i]<0||ids[i]>=256) {
  c4t_bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);return NULL;
 }
 c4t_input *input=c4t_allocate(sizeof(*input));if(!input)return NULL;
 memcpy(input->ids,ids,sizeof(ids));
 if(c4t_publish(&input->r,lease,C4T_INPUT,sizeof(*input),1)){free(input);return NULL;}
 return input;
}
int64_t et_cpu4_private_input_release_v1(void *p) {
 c4t_clear();c4t_record *r=c4t_admit(p,C4T_INPUT,1);if(!r)return c4t_category;
 if(r->state==C4T_DEAD)return 0;
 if(c4t_cerror(et_g3c4_training_child_release_internal(r->lease,r,1)))return c4t_category;
 r->state=C4T_DEAD;return 0;
}
void *et_cpu4_private_logits_create_v1(void *lease) {
 c4t_clear();if(c4t_cerror(et_g3c4_private_training_lease_check_v1(lease)))return NULL;
 c4t_logits *l=c4t_allocate(sizeof(*l));if(!l)return NULL;
 l->tensor=c4t_new(3,c4t_logits_shape);if(!l->tensor){free(l);return NULL;}
 if(c4t_publish(&l->r,lease,C4T_LOGITS,sizeof(*l),4)){c4t_destroy(&l->tensor);free(l);return NULL;}
 return l;
}
static int c4t_finite(const uint32_t bits[1024]) {
 for(size_t i=0;i<1024;i++)if((bits[i]&UINT32_C(0x7f800000))==UINT32_C(0x7f800000))
  return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SHAPE);
 return 0;
}
int64_t et_cpu4_private_logits_copy_from_v1(void *p,void *header,int64_t count) {
 c4t_clear();c4t_logits *l=(void *)c4t_admit(p,C4T_LOGITS,0);if(!l)return c4t_category;
 unsigned char *bytes=c4t_bytes(&l->r,header,count);if(!bytes)return c4t_category;
 uint32_t bits[1024];
 for(size_t i=0;i<1024;i++)bits[i]=(uint32_t)bytes[4*i]|((uint32_t)bytes[4*i+1]<<8)|((uint32_t)bytes[4*i+2]<<16)|((uint32_t)bytes[4*i+3]<<24);
 if(c4t_finite(bits))return c4t_category;
 et_f32_tensor_error e;return c4t_ferror(et_f32_tensor_copy_bits_from_v1(l->tensor,bits,1024,&e),&e);
}
int64_t et_cpu4_private_logits_copy_to_v1(void *p,void *header,int64_t count) {
 c4t_clear();c4t_logits *l=(void *)c4t_admit(p,C4T_LOGITS,0);if(!l)return c4t_category;
 unsigned char *bytes=c4t_bytes(&l->r,header,count);if(!bytes)return c4t_category;
 uint32_t bits[1024];et_f32_tensor_error e;
 et_f32_scoped_guard_internal g={0};int rc=c4t_guard(l->tensor,&g);c4t_end(&g);
 if(rc)return rc;
 if(c4t_ferror(et_f32_tensor_copy_bits_to_v1(l->tensor,bits,1024,&e),&e)||c4t_finite(bits))return c4t_category;
 for(size_t i=0;i<1024;i++)for(size_t j=0;j<4;j++)bytes[4*i+j]=(unsigned char)(bits[i]>>(8*j));
 return 0;
}
int64_t et_cpu4_private_logits_release_v1(void *p) {
 c4t_clear();c4t_logits *l=(void *)c4t_admit(p,C4T_LOGITS,1);if(!l)return c4t_category;
 if(l->r.state==C4T_DEAD)return 0;
 et_f32_tensor_error e;
 if(c4t_ferror(et_f32_tensor_private_destroy_ready_v1(l->tensor,&e),&e))return c4t_category;
 if(c4t_cerror(et_g3c4_training_child_release_internal(l->r.lease,l,4)))return c4t_category;
 c4t_destroy(&l->tensor);l->r.state=C4T_DEAD;return 0;
}
enum slot {
 ET,EP,X,N1,QT,KT,VT,QH,KH,VH,AH,AT,AO,R,N2,FU,FG,FD,Y,NF,Z,
 DZ,DNF,DY,DRSKIP,DFD,DFG,DFU,DN2,DRFFN,DR,DXSKIP,DAO,DAT,DAH,
 DQH,DKH,DVH,DQT,DKT,DVT,DN1Q,DN1K,DN1V,DN1,DXATTENTION,DX,DET,DEP,
 GKEY,GOUTPUT,GQUERY,GVALUE,GDOWN,GUP,GN1B,GN1W,GN2B,GN2W,GTIED,GFB,GFW,GPOS,
 GHEAD,GTOKEN,SLOT_COUNT
};
typedef struct c4t_workspace {
 c4t_record r;et_f32_parameter *parameters[14];const void *identities[14];
 et_f32_tensor *saved[14],*slots[SLOT_COUNT],*epsilon;
 int64_t ids[4],captured_ids[4],positions[4];unsigned char keep[16],ready[SLOT_COUNT];
 int mode,busy;
} c4t_workspace;
static size_t c4t_slot_shape(size_t i,const uint64_t **shape) {
 if(i>=GKEY && i<=GPOS){*shape=c4t_parameters[i-GKEY];return c4t_rank(i-GKEY);}
 if(i==GHEAD||i==GTOKEN){*shape=c4t_parameters[10];return 2;}
 if(i==Z||i==DZ){*shape=c4t_logits_shape;return 3;}
 if(i==FU||i==FG||i==DFU||i==DFG){*shape=c4t_wide;return 3;}
 if(i==QH||i==KH||i==VH||i==AH||i==DAH||i==DQH||i==DKH||i==DVH){*shape=c4t_heads;return 4;}
 *shape=c4t_tokens;return 3;
}
static void c4t_destroy_workspace(c4t_workspace *w) {
 for(size_t i=0;i<14;i++)c4t_destroy(&w->saved[i]);
 for(size_t i=0;i<SLOT_COUNT;i++)c4t_destroy(&w->slots[i]);
 c4t_destroy(&w->epsilon);
}
static et_f32_tensor *c4t_value(et_f32_parameter *p) {
 const et_f32_tensor *value=NULL;et_f32_tensor_error e;
 return c4t_ferror(et_f32_parameter_value_tensor_v1(p,&value,&e),&e)?NULL:(et_f32_tensor *)value;
}
static int c4t_preflight_capabilities(void);
void *et_cpu4_private_workspace_create_v1(void *lease,void *ip,int64_t mode) {
 c4t_clear();if(c4t_cerror(et_g3c4_private_training_lease_check_v1(lease)))return NULL;
 c4t_input *input=(void *)c4t_admit(ip,C4T_INPUT,0);if(!input)return NULL;
 if(input->r.lease!=lease || (mode!=0&&mode!=1)) {
  c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);return NULL;
 }
 for(c4t_record *item=c4t_registry;item;item=item->next)
  if(item->kind==C4T_WORKSPACE && item->lease==lease && item->state!=C4T_DEAD) {
   c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);return NULL;
  }
 if(c4t_preflight_capabilities())return NULL;
 c4t_workspace *w=c4t_allocate(sizeof(*w));if(!w)return NULL;
 w->mode=(int)mode;memcpy(w->ids,input->ids,sizeof(w->ids));memcpy(w->captured_ids,input->ids,sizeof(w->ids));
 for(size_t i=0;i<4;i++) {if(w->ids[i]<0||w->ids[i]>=256) {
   c4t_bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);goto failed;}w->positions[i]=(int64_t)i;}
 memset(w->keep,1,sizeof(w->keep));
 for(int64_t i=0;i<14;i++) {
  w->parameters[i]=et_g3c4_private_training_lease_parameter_v1(lease,i);
  w->identities[i]=et_g3c4_private_training_lease_identity_v1(lease,i);
  if(!w->parameters[i]||!w->identities[i]) {c4t_cerror(1);goto failed;}
  w->saved[i]=c4t_new(c4t_rank((size_t)i),c4t_parameters[i]);if(!w->saved[i])goto failed;
  et_f32_tensor *value=c4t_value(w->parameters[i]);if(!value||c4t_copy(w->saved[i],value))goto failed;
 }
 for(size_t i=0;i<SLOT_COUNT;i++) {
  const uint64_t *shape;size_t rank=c4t_slot_shape(i,&shape);
  if(!(w->slots[i]=c4t_new(rank,shape)))goto failed;
 }
 w->epsilon=c4t_new(0,NULL);if(!w->epsilon)goto failed;
 uint32_t epsilon=UINT32_C(0x3727c5ac);et_f32_tensor_error e;
 if(c4t_ferror(et_f32_tensor_copy_bits_from_v1(w->epsilon,&epsilon,1,&e),&e))goto failed;
 if(c4t_publish(&w->r,lease,C4T_WORKSPACE,sizeof(*w),2))goto failed;
 return w;
failed:c4t_destroy_workspace(w);free(w);return NULL;
}
static int c4t_active(c4t_workspace *w) {
 if(w->busy)return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_REENTRANCY);
 if(w->r.state!=C4T_FORWARD && w->r.state!=C4T_REVERSE)
  return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 return 0;
}
static int c4t_primals(c4t_workspace *w,int64_t mode) {
 if(c4t_active(w))return (int)c4t_category;
 if(mode!=w->mode || (mode!=0&&mode!=1))return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_STALE);
 for(int64_t i=0;i<14;i++) {
  void *p=et_g3c4_private_training_lease_parameter_v1(w->r.lease,i);
  void *id=et_g3c4_private_training_lease_identity_v1(w->r.lease,i);
  if(!p||!id){c4t_cerror(1);return (int)c4t_category;}
  if(p!=w->parameters[i]||id!=w->identities[i])return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_IDENTITY);
  et_f32_tensor *value=c4t_value(p);if(!value)return (int)c4t_category;
  et_f32_scoped_guard_internal a={0},b={0};int rc=c4t_guard(value,&a);
  if(!rc)rc=c4t_guard(w->saved[i],&b);
  int same=!rc && a.view.rank==b.view.rank && a.view.byte_length==b.view.byte_length &&
   !memcmp(a.view.shape,b.view.shape,a.view.rank*sizeof(uint64_t)) && !memcmp(a.view.data,b.view.data,a.view.byte_length);
  c4t_end(&b);c4t_end(&a);if(rc)return rc;
  if(!same)return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_STALE);
 }
 uint32_t epsilon;et_f32_tensor_error e;
 if(c4t_ferror(et_f32_tensor_copy_bits_to_v1(w->epsilon,&epsilon,1,&e),&e))return (int)c4t_category;
 if(epsilon!=UINT32_C(0x3727c5ac)||memcmp(w->ids,w->captured_ids,sizeof(w->ids)))
  return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_STALE);
 for(size_t i=0;i<4;i++)if(w->positions[i]!=(int64_t)i||w->ids[i]<0||w->ids[i]>=256)
  return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_STALE);
 for(size_t i=0;i<16;i++)if(w->keep[i]!=1)return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_STALE);
 return 0;
}
int64_t et_cpu4_private_workspace_check_primals_v1(void *p,int64_t mode) {
 c4t_clear();c4t_workspace *w=(void *)c4t_admit(p,C4T_WORKSPACE,0);if(!w)return c4t_category;
 return c4t_primals(w,mode);
}
static int c4t_storage_ready(c4t_workspace *w) {
 if(w->busy)return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_REENTRANCY);
 et_f32_tensor_error e;
 for(size_t i=0;i<14;i++)if(c4t_ferror(et_f32_tensor_private_destroy_ready_v1(w->saved[i],&e),&e))return (int)c4t_category;
 for(size_t i=0;i<SLOT_COUNT;i++)if(c4t_ferror(et_f32_tensor_private_destroy_ready_v1(w->slots[i],&e),&e))return (int)c4t_category;
 return c4t_ferror(et_f32_tensor_private_destroy_ready_v1(w->epsilon,&e),&e);
}
int64_t et_cpu4_private_workspace_release_v1(void *p) {
 c4t_clear();c4t_workspace *w=(void *)c4t_admit(p,C4T_WORKSPACE,1);if(!w)return c4t_category;
 if(w->r.state==C4T_DEAD)return 0;
 if(c4t_storage_ready(w))return c4t_category;
 if(c4t_cerror(et_g3c4_training_child_release_internal(w->r.lease,w,2)))return c4t_category;
 c4t_destroy_workspace(w);w->r.state=C4T_DEAD;return 0;
}
int64_t et_cpu4_private_workspace_copy_logits_v1(void *p,void *lp) {
 c4t_clear();c4t_workspace *w=(void *)c4t_admit(p,C4T_WORKSPACE,0);if(!w)return c4t_category;
 c4t_logits *l=(void *)c4t_admit(lp,C4T_LOGITS,0);if(!l)return c4t_category;
 if(l->r.lease!=w->r.lease)return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);
 if(c4t_primals(w,w->mode))return c4t_category;
 if(!w->ready[Z])return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 return c4t_copy(l->tensor,w->slots[Z]);
}
int64_t et_cpu4_private_workspace_vjp_begin_v1(void *p,void *lp) {
 c4t_clear();c4t_workspace *w=(void *)c4t_admit(p,C4T_WORKSPACE,0);if(!w)return c4t_category;
 c4t_logits *l=(void *)c4t_admit(lp,C4T_LOGITS,0);if(!l)return c4t_category;
 if(l->r.lease!=w->r.lease)return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);
 if(c4t_primals(w,w->mode))return c4t_category;
 if(w->r.state!=C4T_FORWARD||!w->ready[Z])return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 if(c4t_copy(w->slots[DZ],l->tensor))return c4t_category;
 w->ready[DZ]=1;w->r.state=C4T_REVERSE;return 0;
}
int64_t et_cpu4_private_workspace_contribute_v1(void *p,int64_t mode,int64_t weight_bits,int64_t ordinal) {
 c4t_clear();c4t_workspace *w=(void *)c4t_admit(p,C4T_WORKSPACE,0);if(!w)return c4t_category;
 if(w->mode!=1||mode!=1||w->r.state!=C4T_REVERSE||weight_bits<0||
    (uint64_t)weight_bits>UINT32_MAX||ordinal<0)return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
 if(c4t_primals(w,mode)||c4t_storage_ready(w))return c4t_category;
 et_f32_gradient_contribution_v1 rows[14];
 for(size_t i=0;i<14;i++) {
  if(!w->ready[GKEY+i])return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
  rows[i]=(et_f32_gradient_contribution_v1){sizeof(rows[i]),w->parameters[i],w->slots[GKEY+i],(uint64_t)ordinal};
 }
 et_f32_gradient_plan *plan=NULL;et_f32_tensor_error e;w->busy=1;
 int rc=et_f32_gradient_plan_prepare_v1(14,rows,(uint32_t)weight_bits,&plan,&e);
 if(rc){w->busy=0;return c4t_ferror(rc,&e);}
 if(et_f32_gradient_plan_commit_v1(plan,&e)||et_f32_gradient_plan_release_v1(&plan,&e))abort();
 w->busy=0;w->r.state=C4T_CONSUMED;return 0;
}
enum { P0=SLOT_COUNT, EPS=P0+14, IDS, POS, KEEP };
enum family { EMBEDDING,LINEAR,NORM,GELU,RESIDUAL,SPLIT,MERGE,ATTENTION,SUM };
typedef struct role {
 int family,index,reverse,runtime;const char *cap,*op;
 size_t rank;uint64_t shape[6];size_t ni,no;int in[7],out[3];
} role;
static const role roles[]={
 {EMBEDDING,0,0,0,"n3k.embedding-forward","n3k.embedding.forward",4,{1,4,256,4},2,1,{IDS,P0+10},{ET}},
 {EMBEDDING,0,1,0,"n3k.embedding-backward","n3k.embedding.backward",4,{1,4,256,4},2,1,{IDS,DET},{GTOKEN}},
 {EMBEDDING,1,0,0,"n3k.embedding-forward","n3k.embedding.forward",4,{1,4,4,4},2,1,{POS,P0+13},{EP}},
 {EMBEDDING,1,1,0,"n3k.embedding-backward","n3k.embedding.backward",4,{1,4,4,4},2,1,{POS,DEP},{GPOS}},
 {LINEAR,0,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,4,4,4},2,1,{N1,P0+2},{QT}},
 {LINEAR,0,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,4,4,4},3,2,{N1,P0+2,DQT},{DN1Q,GQUERY}},
 {LINEAR,1,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,4,4,4},2,1,{N1,P0+0},{KT}},
 {LINEAR,1,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,4,4,4},3,2,{N1,P0+0,DKT},{DN1K,GKEY}},
 {LINEAR,2,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,4,4,4},2,1,{N1,P0+3},{VT}},
 {LINEAR,2,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,4,4,4},3,2,{N1,P0+3,DVT},{DN1V,GVALUE}},
 {LINEAR,3,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,4,4,4},2,1,{AT,P0+1},{AO}},
 {LINEAR,3,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,4,4,4},3,2,{AT,P0+1,DAO},{DAT,GOUTPUT}},
 {LINEAR,4,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,4,4,8},2,1,{N2,P0+5},{FU}},
 {LINEAR,4,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,4,4,8},3,2,{N2,P0+5,DFU},{DN2,GUP}},
 {LINEAR,5,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,4,8,4},2,1,{FG,P0+4},{FD}},
 {LINEAR,5,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,4,8,4},3,2,{FG,P0+4,DFD},{DFG,GDOWN}},
 {LINEAR,6,0,0,"n3k.linear","n3k.linear.forward-no-bias",4,{1,4,4,256},2,1,{NF,P0+10},{Z}},
 {LINEAR,6,1,0,"n3k.linear","n3k.linear.backward-no-bias",4,{1,4,4,256},3,2,{NF,P0+10,DZ},{DNF,GHEAD}},
 {NORM,0,0,1,"kernel.norm","layer-norm.forward",3,{1,4,4},4,1,{X,P0+7,P0+6,EPS},{N1}},
 {NORM,0,1,1,"kernel.norm","layer-norm.backward",3,{1,4,4},4,3,{X,P0+7,EPS,DN1},{DXATTENTION,GN1W,GN1B}},
 {NORM,1,0,1,"kernel.norm","layer-norm.forward",3,{1,4,4},4,1,{R,P0+9,P0+8,EPS},{N2}},
 {NORM,1,1,1,"kernel.norm","layer-norm.backward",3,{1,4,4},4,3,{R,P0+9,EPS,DN2},{DRFFN,GN2W,GN2B}},
 {NORM,2,0,1,"kernel.norm","layer-norm.forward",3,{1,4,4},4,1,{Y,P0+12,P0+11,EPS},{NF}},
 {NORM,2,1,1,"kernel.norm","layer-norm.backward",3,{1,4,4},4,3,{Y,P0+12,EPS,DNF},{DY,GFW,GFB}},
 {GELU,0,0,0,"n3k.gelu","n3k.gelu.forward",3,{1,4,8},1,1,{FU},{FG}},
 {GELU,0,1,0,"n3k.gelu","n3k.gelu.backward",3,{1,4,8},2,1,{FU,DFG},{DFU}},
 {RESIDUAL,0,0,0,"n3k.residual","n3k.residual.forward",3,{1,4,4},2,1,{ET,EP},{X}},
 {RESIDUAL,0,1,0,"n3k.residual","n3k.residual.backward",3,{1,4,4},1,2,{DX},{DET,DEP}},
 {RESIDUAL,1,0,0,"n3k.residual","n3k.residual.forward",3,{1,4,4},2,1,{X,AO},{R}},
 {RESIDUAL,1,1,0,"n3k.residual","n3k.residual.backward",3,{1,4,4},1,2,{DR},{DXSKIP,DAO}},
 {RESIDUAL,2,0,0,"n3k.residual","n3k.residual.forward",3,{1,4,4},2,1,{R,FD},{Y}},
 {RESIDUAL,2,1,0,"n3k.residual","n3k.residual.backward",3,{1,4,4},1,2,{DY},{DRSKIP,DFD}},
 {SPLIT,0,0,0,"n3k.head-layout","n3k.heads.split.forward",4,{1,4,2,2},1,1,{QT},{QH}},
 {SPLIT,0,1,0,"n3k.head-layout","n3k.heads.split.backward",4,{1,4,2,2},1,1,{DQH},{DQT}},
 {SPLIT,1,0,0,"n3k.head-layout","n3k.heads.split.forward",4,{1,4,2,2},1,1,{KT},{KH}},
 {SPLIT,1,1,0,"n3k.head-layout","n3k.heads.split.backward",4,{1,4,2,2},1,1,{DKH},{DKT}},
 {SPLIT,2,0,0,"n3k.head-layout","n3k.heads.split.forward",4,{1,4,2,2},1,1,{VT},{VH}},
 {SPLIT,2,1,0,"n3k.head-layout","n3k.heads.split.backward",4,{1,4,2,2},1,1,{DVH},{DVT}},
 {MERGE,0,0,0,"n3k.head-layout","n3k.heads.merge.forward",4,{1,4,2,2},1,1,{AH},{AT}},
 {MERGE,0,1,0,"n3k.head-layout","n3k.heads.merge.backward",4,{1,4,2,2},1,1,{DAT},{DAH}},
 {ATTENTION,0,0,2,"kernel.causal-attention","causal-attention.forward",6,{1,2,2,4,4,2},6,1,{QH,KH,VH,POS,POS,KEEP},{AH}},
 {ATTENTION,0,1,2,"kernel.causal-attention","causal-attention.backward",6,{1,2,2,4,4,2},7,3,{QH,KH,VH,POS,POS,KEEP,DAH},{DQH,DKH,DVH}},
 {SUM,0,1,0,"n3k.activation-sum","n3k.sum3.forward",3,{1,4,4},3,1,{DN1Q,DN1K,DN1V},{DN1}},
 {SUM,1,1,0,"n3k.activation-sum","n3k.sum2.forward",3,{1,4,4},2,1,{DXSKIP,DXATTENTION},{DX}},
 {SUM,2,1,0,"n3k.activation-sum","n3k.sum2.forward",3,{1,4,4},2,1,{DRSKIP,DRFFN},{DR}},
 {SUM,3,1,0,"n3k.tied-sum","n3k.sum2.forward",2,{256,4},2,1,{GHEAD,GTOKEN},{GTIED}},
};
static et_kernel_runtime *c4t_runtimes[3];
static const et_kernel_provider_v1 *c4t_resolver(void *ctx,const char *symbol) {
 return strcmp(symbol,ET_KERNEL_PROVIDER_SYMBOL_V1)?NULL:ctx;
}
static int c4t_preflight_capabilities(void) {
 if(!c4t_runtimes[0]) {
  const et_kernel_provider_v1 *providers[]={et_n3k_kernel_provider_v1(),et_n2_kernel_provider_v1(),et_a2_kernel_provider_v1()};
  et_kernel_runtime *staged[3]={0};et_kernel_error e;
  for(size_t i=0;i<3;i++)if(c4t_kerror(et_kernel_runtime_discover(c4t_resolver,(void *)providers[i],&staged[i],&e),&e)) {
   for(size_t j=0;j<3;j++)et_kernel_runtime_destroy(staged[j]);return (int)c4t_category;
  }
  memcpy(c4t_runtimes,staged,sizeof(staged));
 }
 for(size_t i=0;i<sizeof(roles)/sizeof(*roles);i++) {
  const role *r=&roles[i];et_kernel_request_v1 req={sizeof(req),r->op,"f32","cpu",r->rank,r->shape,1,{0}};
  const et_kernel_capability_v1 *cap=NULL;et_kernel_error e;
  if(c4t_kerror(et_kernel_runtime_capability_require(c4t_runtimes[r->runtime],r->cap,&req,&cap,&e),&e))return (int)c4t_category;
 }
 return 0;
}
static et_kernel_tensor_view_v1 c4t_view(void *data,size_t bytes,const char *dtype,size_t rank,const uint64_t *shape) {
 return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),data,bytes,dtype,"cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,rank,shape};
}
static et_f32_tensor *c4t_source(c4t_workspace *w,int ref) {
 if(ref<SLOT_COUNT)return w->slots[ref];if(ref<EPS)return w->saved[ref-P0];return w->epsilon;
}
static int c4t_execute(c4t_workspace *w,const role *r) {
 if(c4t_primals(w,w->mode))return (int)c4t_category;
 if(w->r.state!=(r->reverse?C4T_REVERSE:C4T_FORWARD))return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 for(size_t i=0;i<r->ni;i++)if(r->in[i]<SLOT_COUNT&&!w->ready[r->in[i]])return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 for(size_t i=0;i<r->no;i++)if(w->ready[r->out[i]])return c4t_bad(ET_M3T_INVALID_STATE,ET_M3T_CODE_LIFECYCLE);
 w->busy=1;et_f32_scoped_guard_internal guards[10]={0};et_f32_tensor *owners[10]={0};size_t ng=0;
 et_kernel_tensor_view_v1 in[7],out[3];int rc=0;const uint64_t keep_shape[]={1,4,4};
 for(size_t n=0;n<r->ni+r->no&&!rc;n++) {
  int output=n>=r->ni,ref=output?r->out[n-r->ni]:r->in[n];
  et_kernel_tensor_view_v1 *dest=output?&out[n-r->ni]:&in[n];
  if(ref==KEEP){*dest=c4t_view(w->keep,16,"bool",3,keep_shape);continue;}
  if(ref==IDS||ref==POS){*dest=c4t_view(ref==IDS?w->ids:w->positions,32,"i64",2,c4t_ids_shape);continue;}
  et_f32_tensor *tensor=c4t_source(w,ref);size_t j=0;while(j<ng&&owners[j]!=tensor)j++;
  if(j==ng){owners[ng]=tensor;rc=c4t_guard(tensor,&guards[ng]);ng++;}
  if(!rc)*dest=guards[j].view;
 }
 if(!rc) {
  et_kernel_request_v1 req={sizeof(req),r->op,"f32","cpu",r->rank,r->shape,1,{0}};
  et_kernel_call_v1 call={sizeof(call),r->cap,&req,r->ni,sizeof(*in),r->ni*sizeof(*in),in,r->no,sizeof(*out),r->no*sizeof(*out),out};
  et_kernel_error e;rc=c4t_kerror(et_kernel_runtime_dispatch(c4t_runtimes[r->runtime],&call,&e),&e);
 }
 while(ng)c4t_end(&guards[--ng]);
 if(!rc)for(size_t i=0;i<r->no;i++)w->ready[r->out[i]]=1;
 w->busy=0;return rc;
}
int64_t et_cpu4_private_workspace_role_v1(void *p,int64_t family,int64_t index,int64_t reverse) {
 c4t_clear();c4t_workspace *w=(void *)c4t_admit(p,C4T_WORKSPACE,0);if(!w)return c4t_category;
 if(reverse!=0&&reverse!=1)return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
 for(size_t i=0;i<sizeof(roles)/sizeof(*roles);i++)if(roles[i].family==family&&roles[i].index==index&&roles[i].reverse==reverse)
  return c4t_execute(w,&roles[i]);
 return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
}
#ifdef ET_CPU4_OBJECTIVE_PRIVATE
#include "cpu4_training_objective.inc"
#endif

#endif
