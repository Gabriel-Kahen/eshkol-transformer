"""Emit fixed concrete owning visitors; no native or Scheme execution."""
from tests.cpuv1_resource.scheme_schema import SCHEMAS,EXTRA_DEFS

CANDIDATE='native/cpu4_candidate_extension.esk'
CANDIDATE_DEFS='''(define (cpuv1-resource-candidate-visit packet row)
  (if (and (cpuv1-resource-vector? row 12)
           (cpuv1-resource-vector? (vector-ref row 0) 1)
           (eq? (vector-ref (vector-ref row 0) 0) (vector-ref row 0))
           (not (vector-ref row 11))
           (or (eq? (vector-ref row 1) 'unpublished)
               (eq? (vector-ref row 1) 'adopted)
               (eq? (vector-ref row 1) 'dead))
           (cpuv1-resource-vector? (vector-ref row 4) 12)
           (cpuv1-resource-vector? (vector-ref row 5) 11)
           (cpuv1-resource-vector? (vector-ref row 6) 6)
           (eq? (cpuv1-resource-model-find (vector-ref (vector-ref row 4) 0))
                (vector-ref row 4))
           (cpuv1-resource-member? (vector-ref row 5)
             (vector-ref i2-construction-registry-root 0) 0)
           (if (eq? (vector-ref row 1) 'dead)
               (eq? (vector-ref (vector-ref row 4) 2) 'dead)
               (and (eq? (vector-ref (vector-ref row 4) 3) (vector-ref row 3))
                    (eq? (vector-ref (vector-ref row 4) 2) 'live))))
      (cpuv1-resource-state! packet 64
        (if (eq? (vector-ref row 1) 'dead) 2 1) 4
        (cond ((eq? (vector-ref row 1) 'unpublished) 1)
              ((eq? (vector-ref row 1) 'adopted) 2) (else 3)))
      (cpuv1-resource-reject! packet)))
'''

def visitor_name(i):return 'cpuv1-resource-row-'+str(i)

def definitions(path,rows):
    answer=EXTRA_DEFS.get(path,'')
    for row in rows:
        i=row['id'];s=SCHEMAS[i]
        if s['mode']=='registry':
            answer+=f'''(define ({visitor_name(i)} packet row)
  (if {s['check']}
      (cpuv1-resource-state! packet {i} {s['state']} {s['links']} {s['raw']})
      (cpuv1-resource-reject! packet)))
'''
    return answer


def typed_call(row):
    i=row['id'];name=row['name'];s=SCHEMAS[i];mode=s['mode']
    head=f'(vector-ref {name} 0)' if s.get('holder') else name
    if mode=='registry':return f'(cpuv1-resource-walk! packet {i} {head} {visitor_name(i)} 0)'
    if mode=='quiescent-empty':return f'(if (null? {head}) (cpuv1-resource-seal! packet {i} 0) (cpuv1-resource-reject! packet))'
    if mode in ('constant','procedure','boolean-holder'):
        valid=f'(and (integer? {name}) (>= {name} 0))' if mode=='constant' else f'(procedure? {name})'
        scalar=name if mode=='constant' else '0'
        if mode=='boolean-holder':valid=f'(and (cpuv1-resource-vector? {name} 1) (boolean? (vector-ref {name} 0)))';scalar=f'(if (vector-ref {name} 0) 1 0)'
        return f'''(if {valid}
  (begin (cpuv1-resource-scalar! packet {i} 0 {scalar})
         (cpuv1-resource-state! packet {i} 6 0 0)
         (cpuv1-resource-seal! packet {i} 1))
  (cpuv1-resource-reject! packet))'''
    if mode=='p1-shells':return '(cpuv1-resource-seal! packet 3 (cpuv1-resource-spine-count packet (shell-registry-current) 0))'
    if mode=='p1-index':return '(begin (cpuv1-resource-state! packet 4 6 1 0) (cpuv1-resource-seal! packet 4 1))'
    if mode=='p1-active':return '''(begin
  (cpuv1-resource-active-list packet (vector-ref active-record-root 0) shell-identity-index 0)
  (cpuv1-resource-active-list packet (vector-ref active-record-root 1) shell-identity-index 0)
  (cpuv1-resource-active-list packet (vector-ref active-record-root 4) shell-identity-index 0)
  (cpuv1-resource-active-list packet (vector-ref active-record-root 5) shell-identity-index 0)
  (cpuv1-resource-seal! packet 62
    (+ (cpuv1-resource-spine-count packet (vector-ref active-record-root 0) 0)
       (cpuv1-resource-spine-count packet (vector-ref active-record-root 1) 0)
       (cpuv1-resource-spine-count packet (vector-ref active-record-root 4) 0)
       (cpuv1-resource-spine-count packet (vector-ref active-record-root 5) 0))))'''
    if mode=='session':
        checks={59:f'(and (cpuv1-resource-vector? {name} 3) (not (vector-ref {name} 1)) (or (eq? (vector-ref {name} 2) \'new) (eq? (vector-ref {name} 2) \'baseline) (eq? (vector-ref {name} 2) \'done)))',60:f'(and (cpuv1-resource-vector? {name} 2) (or (eq? (vector-ref {name} 0) \'new) (eq? (vector-ref {name} 0) \'running) (eq? (vector-ref {name} 0) \'done)))',61:f'(and (cpuv1-resource-vector? {name} 1) (or (not (vector-ref {name} 0)) (cpuv1-resource-vector? (vector-ref {name} 0) 16)))'}
        states={59:('(vector-ref '+name+' 2)',['new','baseline','done']),
                60:('(vector-ref '+name+' 0)',['new','running','done']),
                61:('(vector-ref (vector-ref '+name+' 0) 0)',['training','handoff','adopted','released','done'])}
        value,names=states[i]
        phase='(cond '+' '.join(f"((eq? {value} '{n}) {j})" for j,n in enumerate(names,1))+' (else 15))'
        if i==61:phase=f'(if (not (vector-ref {name} 0)) 0 {phase})'
        return f'''(if {checks[i]}
  (begin (cpuv1-resource-state! packet {i} 6 0 {phase})
         (cpuv1-resource-seal! packet {i} 1))
  (cpuv1-resource-reject! packet))'''
    raise ValueError('unclassified root '+str(row))
