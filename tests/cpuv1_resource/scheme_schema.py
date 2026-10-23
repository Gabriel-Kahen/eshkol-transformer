"""Concrete read-only owning-source schemas for the frozen successful route.

Expressions run inside the owning unit. States are actual source lifetime words,
not inferred counts. Transaction-only roots must be empty at quiescent samples.
All byte ownership stays in whole-arena census. No observer allocates a spine.
"""

def v(slot): return f'(vector-ref row {slot})'
def vector(n): return f'(cpuv1-resource-vector? row {n})'
def eq(a,b): return f'(eq? {a} {b})'
def call(name,slot=0): return eq(f'({name} {v(slot)})','row')
def state(slot): return f'(cpuv1-resource-state-code {v(slot)})'

SCHEMAS={}
def registry(i,n,slot,check,links=1,holder=True,**kw):
    SCHEMAS[i]=dict(mode='registry',holder=holder,check=f'(and {vector(n)} {check})',state=state(slot),links=links,**kw)
def permanent(i,check,holder=False,links=1):
    SCHEMAS[i]=dict(mode='registry',holder=holder,check=check,state='1',links=links)
for i in [0,1,11,12,13,14]: SCHEMAS[i]=dict(mode='constant')
SCHEMAS[10]=dict(mode='procedure')
SCHEMAS[51]=dict(mode='boolean-holder')
# P1 registries: final records are the exact canonical native-index records.
permanent(2,f'(and {vector(3)} (cpuv1-resource-vector? {v(2)} 11) (let ((record (shell-entry {v(1)}))) (and record (eq? (vector-ref record 2) {v(2)}))))',True,2)
SCHEMAS[3]=dict(mode='p1-shells')
SCHEMAS[4]=dict(mode='p1-index')
registry(5,5,1,call('construction-record-resource'),2) # local bounded wrapper generated below
SCHEMAS[6]=dict(mode='registry',holder=True,check=f'(and {vector(11)} (eq? (e3-mode-record {v(0)}) row) (not (vector-ref e3-mode-registry-root 1)))',state=f'(if (= {v(1)} 3) 2 (if (= {v(1)} 0) 1 5))',links=1)
SCHEMAS[62]=dict(mode='p1-active')
registry(63,6,5,f'(and (eq? (construction-record-resource {v(0)}) {v(1)}) (or (eq? {v(5)} \'live) (eq? {v(5)} \'retired)))',2)
SCHEMAS[63]['state']=f'(if (eq? {v(5)} \'retired) 2 1)'
# Tokenizer registry rows are permanent [shell, core] lists.
for i,prefix in [(7,'t1'),(8,'t2')]:
    permanent(i,f'(and (pair? row) (pair? (cdr row)) (null? (cdr (cdr row))) (eq? ({prefix}-private-entry (car row)) row))')
registry(9,7,2,call('c2-load-find-stage'),1,False)
registry(15,9,7,call('c2-training-state-find'),1,False)
# E3 numeric state is 0 staging, 1 idle, 2/3 acquired/evaluating, 4 unbound.
SCHEMAS[25]=dict(mode='registry',holder=True,check=f'(and {vector(17)} {call("e3-frame-entry-find")} (integer? {v(1)}) (<= 0 {v(1)}) (<= {v(1)} 4) (if (= {v(1)} 4) (and (not {v(3)}) (not {v(4)})) (and {v(3)} (eq? {v(9)} (cpu4-model-binding-native-lease-internal {v(3)})))))',state=f'(cond ((= {v(1)} 4) 2) ((= {v(1)} 0) 3) ((= {v(1)} 1) 1) (else 5))',links=2)
registry(26,12,2,call('cpuv1-resource-g3c4-find'),1)
registry(27,12,2,call('cpuv1-resource-model-find'),1)
# P2/G1 records persist; their lifetime comes from the referenced output entry.
SCHEMAS[28]=dict(mode='registry',holder=True,check=f'(and {vector(5)} (cpuv1-resource-vector? {v(1)} 12) (eq? (cpuv1-resource-g3c4-find (vector-ref {v(1)} 0)) {v(1)}))',state=f'(cpuv1-resource-state-code (vector-ref {v(1)} 2))',links=1)
# These scoped maps cannot survive a successful region return.
for i in [29,30,31,32]:SCHEMAS[i]=dict(mode='quiescent-empty',holder=True)
# Retained authority closures expose only immutable shell plus read-only checks.
for i in [33,35]:
    SCHEMAS[i]=dict(mode='registry',holder=True,check='(and (procedure? row) (let ((entry (cpuv1-resource-g3c4-find (row \'shell #f)))) (and entry (if (eq? (vector-ref entry 2) \'dead) (row \'dead entry) (row \'check entry)))))',state='(cpuv1-resource-state-code (vector-ref (cpuv1-resource-g3c4-find (row \'shell #f)) 2))',links=2)
SCHEMAS[34]=dict(mode='registry',holder=True,check=f'(and {vector(16)} (cpuv1-resource-vector? {v(1)} 12) (procedure? {v(10)}) (eq? (cpuv1-resource-g3c4-find {v(2)}) {v(1)}) (cpuv1-resource-member? {v(10)} (vector-ref g3c4-p2g2-terminal-live-originals 0) 0) (if (eq? {v(0)} \'dead) ({v(10)} \'dead {v(1)}) ({v(10)} \'check row)))',state=state(0),links=3)
SCHEMAS[36]=dict(mode='registry',holder=True,check='(let ((entry (cpuv1-resource-g3c4-find row))) (and entry (eq? (vector-ref entry 1) \'terminal-output) (eq? (vector-ref entry 2) \'dead)))',state='2',links=1)
registry(37,10,9,eq(f'(cpuv1-resource-g3c4-find {v(0)})',v(1)),2)
SCHEMAS[38]=dict(mode='registry',holder=True,check=f'(and {vector(2)} (procedure? {v(1)}) (let ((entry (cpuv1-resource-g3c4-find {v(0)}))) (and entry (let ((owner (cpuv1-resource-find-vector {v(0)} (vector-ref g3c4-public-g2-results 0) 0 0))) (and owner ({v(1)} owner entry))))))',state='1',links=2)
registry(39,10,9,f'(and {eq(f"(cpuv1-resource-g3c4-find {v(0)})",v(1))} (eq? {v(9)} \'dead))',2)
for i in [40,41]:registry(i,8,7,eq(f'(cpuv1-resource-g3c4-find {v(0)})',v(1)),2)
SCHEMAS[42]=dict(mode='registry',holder=True,check=f'(and {vector(2)} (procedure? {v(1)}) (let ((owner (cpuv1-resource-find-vector {v(0)} (vector-ref g3c4-public-g2-generator-rows 0) 0 0)) (anchor (cpuv1-resource-find-vector {v(0)} (vector-ref g3c4-public-g2-generator-anchors 0) 0 0))) (and owner anchor (eq? (vector-ref owner 1) (cpuv1-resource-g3c4-find {v(0)})) (or (eq? (vector-ref owner 7) \'dead) ({v(1)} owner anchor (vector-ref owner 1))))))',state='(cpuv1-resource-state-code (vector-ref (cpuv1-resource-find-vector (vector-ref row 0) (vector-ref g3c4-public-g2-generator-rows 0) 0 0) 7))',links=3)
registry(43,21,12,eq(f'(cpuv1-resource-g3c4-find {v(5)})',v(6)),2)
for i in [44,45]:registry(i,12,8,eq(f'(cpuv1-resource-g3c4-find {v(0)})',v(1)),2)
SCHEMAS[46]=dict(mode='registry',holder=True,check=f'(and {vector(2)} (procedure? {v(1)}) (cpuv1-resource-find-vector {v(0)} (vector-ref g3c4-public-g2-output-rows 0) 0 0) (cpuv1-resource-g3c4-find {v(0)}))',state='1',links=2)
SCHEMAS[47]=dict(mode='registry',holder=True,check=f'(and {vector(5)} (cpuv1-resource-vector? {v(0)} 12) (eq? (cpuv1-resource-g3c4-find {v(4)}) {v(0)}))',state=f'(cpuv1-resource-state-code (vector-ref {v(0)} 2))',links=1)
# Source-present legacy generation/M3T are route-unused under frozen CPU4.
for i in [48,52]:SCHEMAS[i]=dict(mode='quiescent-empty',holder=True)
permanent(49,f'(and {vector(1)} {v(0)} (i2-module? {v(0)}))',True)
registry(50,11,3,f'(and (eq? {v(0)} \'i2-construction) (not (vector-ref i2-construction-registry-root 1)))',1)
registry(53,8,6,eq(f'(o2-find-identity {v(0)} o2-states)','row'),1,False)
for i in [54,56]:permanent(i,f'(and {vector(6)} {v(0)})')
for i in [55,57]:permanent(i,f'(and {vector(1)} {v(0)})')
permanent(58,'(and (pair? row) (pair? (cdr row)) (pair? (cdr (cdr row))) (pair? (cdr (cdr (cdr row)))) (null? (cdr (cdr (cdr (cdr row))))) (eq? (x1-private-entry (car row)) row))')
for i in [59,60,61]:SCHEMAS[i]=dict(mode='session')

# Pure bounded scan, inserted only where the original aggregate exposes the
# actual owning registry. It never calls the raising public admission helpers.
EXTRA_DEFS={
 'internal/p1/lib/transformer/module.esk': '''(define (construction-record-resource identity)
  (cpuv1-resource-find-vector identity (vector-ref construction-registry-root 0) 0 0))
''',
 'native/g3c4_call_entry_extension.esk': '''(define (cpuv1-resource-g3c4-find identity)
  (cpuv1-resource-find-vector identity (vector-ref g3c4-registry 0) 0 0))
''',
 'native/g3c4_model_extension.esk': '''(define (cpuv1-resource-model-find identity)
  (cpuv1-resource-find-vector identity (vector-ref g3c4-model-registry 0) 0 0))
'''}

def tagged(tag,n,extra):
    return f'(and {vector(n)} (eq? {v(0)} {tag}) (eq? {v(1)} row) {extra})'
def false_slots(*slots):return ' '.join(f'(not {v(i)})' for i in slots)
def find(root,key,slot):return f'(cpuv1-resource-find-vector {key} (vector-ref {root} 0) {slot} 0)'
def owned(key):return find('cpu4-owned-candidates',key,2)
def trainer(record):return f'(cpuv1-resource-trainer-record? {record})'
# Exact CPU4 custody states. Closed records keep historical lease references.
SCHEMAS[16]=dict(mode='registry',holder=True,check=tagged('cpu4-owned-tag',15,f'(and {false_slots(12,14)} (cpuv1-resource-vector? {v(13)} 14) (cpuv1-resource-candidate-row {v(5)}))'),state=state(3),links=2)
SCHEMAS[17]=dict(mode='registry',holder=True,check=tagged("'cpu4-checkpoint",17,f'(and (eq? {v(3)} \'dead) {false_slots(4,9,10,13,14)} (cpuv1-resource-vector? {v(5)} 14) (cpuv1-resource-all-cells-clear? {v(5)} 0) (cpuv1-resource-vector? {v(7)} 1) (not (vector-ref {v(7)} 0)) (or (not {v(15)}) (cpuv1-resource-member? {v(15)} (vector-ref cpu4-control-stages 0) 0)) (or (not {v(16)}) (cpuv1-resource-member? {v(16)} (vector-ref cpu4-control-parents 0) 0)))'),state='2',links=3)
SCHEMAS[18]=dict(mode='registry',holder=True,check=f'(and {vector(9)} (eq? {v(5)} \'dead) {false_slots(0,1,2,3,4)} (or (not {v(7)}) (cpuv1-resource-member? {v(7)} (vector-ref cpu4-control-parents 0) 0)))',state='2',links=1)
SCHEMAS[19]=dict(mode='registry',holder=True,check=tagged('cpu4-checkpoint-factory-tag',19,f'(and {false_slots(10,11,18)} (or (and (eq? {v(3)} \'dead) {false_slots(13,14,15)}) (and (eq? {v(3)} \'ready) {owned(v(14))})) (or (not {v(12)}) (cpuv1-resource-member? {v(12)} (vector-ref cpu4-checkpoint-parents 0) 0)))'),state=f'(if (eq? {v(3)} \'ready) 1 2)',links=2)
SCHEMAS[20]=dict(mode='registry',holder=True,check=tagged('cpu4-handoff-tag',24,f'(and (eq? {v(3)} \'adopted) {false_slots(18,23)} (eq? {owned(v(4))} {v(5)}) (eq? (vector-ref {v(5)} 3) \'dead) (eq? (cpuv1-resource-model-find {v(6)}) {v(7)}) (cpuv1-resource-vector? {v(15)} 2) (eq? (vector-ref {v(15)} 0) {v(6)}) (eq? (vector-ref {v(15)} 1) {v(14)}) (let ((entry (cpuv1-resource-g3c4-find {v(14)}))) (and entry (or (eq? (vector-ref entry 2) \'dead) (and (eq? (vector-ref entry 5) {v(7)}) (eq? (vector-ref entry 6) {v(8)}))))))'),state='1',links=4)
SCHEMAS[21]=dict(mode='registry',holder=True,check=f'(and {vector(12)} (eq? {find("cpu4-model-bindings",v(0),0)} row) (or (and (eq? {v(1)} \'live) (eq? (cpuv1-resource-model-find {v(2)}) {v(3)}) (eq? (vector-ref {v(3)} 3) {v(4)}) {v(5)} (cpuv1-resource-vector? {v(9)} 17) (cpuv1-resource-vector? {v(10)} 14) (cpuv1-resource-vector? {v(11)} 14)) (and (or (eq? {v(1)} \'released) (eq? {v(1)} \'dead)) (not {v(5)}))))',state=state(1),links=2)
SCHEMAS[22]=dict(mode='registry',holder=True,check=tagged('cpu4-control-parent-tag',8,f'(and (or (eq? {v(5)} \'closed) (eq? {v(5)} \'dead)) (or (eq? {v(4)} \'snapshot) (eq? {v(4)} \'restore)) (not {v(7)}) {trainer(v(3))})'),state=f'(if (eq? {v(5)} \'closed) 2 (cpuv1-resource-state-code {v(5)}))',links=2)
SCHEMAS[23]=dict(mode='registry',holder=True,check=tagged('cpu4-control-stage-tag',6,f'(and (or (eq? {v(4)} \'aborted) (eq? {v(4)} \'committed)) (not {v(5)}) {find("cpu4-control-parents",v(3),2)})'),state='2',links=2)
SCHEMAS[24]=dict(mode='registry',holder=True,check=tagged('cpu4-step-tag',20,f'(and (not (vector-ref cpu4-step-frames 1)) (or (eq? {v(4)} \'committed) (eq? {v(4)} \'aborted) (eq? {v(4)} \'dead)) {false_slots(11,15,16,18)} (cpuv1-resource-vector? {v(3)} 3) (cpuv1-resource-member? {v(3)} (vector-ref cpu4-trainers 0) 0) (eq? (vector-ref {v(3)} 0) {v(2)}) {trainer(v(2))} (cpuv1-resource-vector? {v(17)} 4))'),state='2',links=3)
EXTRA_DEFS['native/cpu4_trainer_control_extension.esk']='''(define (cpuv1-resource-trainer-record? record)
  (and (cpuv1-resource-vector? record 22)
       (eq? (vector-ref record 0) cpu4-trainer-tag)
       (eq? (vector-ref record 1) record)
       (not (vector-ref cpu4-trainers 1))
       (let ((row (cpuv1-resource-find-vector record (vector-ref cpu4-trainers 0) 0 0)))
         (and row (cpuv1-resource-vector? row 3)))))
'''
# Values with no lifecycle in source are explicit permanent identity references.
# No inferred destructor state is attributed to policy/config/tokenizer shells.
SCHEMAS[58]['check']=SCHEMAS[58]['check'].replace('x1-private-entry','x1-config-entry')
EXTRA_DEFS['native/cpu4_model_binding_extension.esk']='''(define (cpuv1-resource-binding-roster? row index)
  (if (= index 14) #t
      (let ((handle (vector-ref (vector-ref row 10) index))
            (carrier (vector-ref (vector-ref row 11) index))
            (native (i2-native-g3c4-owner-parameter (vector-ref row 4) index))
            (leased (cpu4-native-lease-parameter (vector-ref row 5) index))
            (identity (cpu4-native-lease-identity (vector-ref row 5) index)))
        (and (not (null? native)) (not (null? leased))
             (cpuv1-resource-vector? carrier 5)
             (eq? handle (vector-ref (vector-ref (vector-ref row 3) 8) index))
             (eq? (vector-ref carrier 0) i2-carrier-tag)
             (eq? (vector-ref carrier 1) 'parameter)
             (eq? (vector-ref carrier 2) native)
             (eq? (vector-ref carrier 3) handle)
             (eq? leased native) (eq? identity handle)
             (cpuv1-resource-binding-roster? row (+ index 1))))))
'''
SCHEMAS[21]['check']=SCHEMAS[21]['check'].replace('(cpuv1-resource-vector? (vector-ref row 11) 14))','(cpuv1-resource-vector? (vector-ref row 11) 14) (cpuv1-resource-vector? (vector-ref (vector-ref row 3) 8) 14) (cpuv1-resource-binding-roster? row 0))')
# Raw source-phase counts accompany broad lifetime buckets. Index0 is STATIC;
# names are packet metadata, never an inferred native allocation state.
RAW_NAMES={}
def raw(i,slot,names):
    RAW_NAMES[i]=['STATIC']+names
    value=f'(vector-ref row {slot})'
    SCHEMAS[i]['raw']='(cond '+ ' '.join(f"((eq? {value} '{name}) {j})" for j,name in enumerate(names,1))+' (else 15))'
for i,slot,names in [
 (5,1,['open','sealed','prepared','aborted']), (9,2,['live','consuming','dead']),
 (15,7,['constructing','live','releasing','dead']), (16,3,['open','dead','payload-retired']),
 (17,3,['constructing','source-staged','source-closed','sealed','committing','cleanup-blocked','dead']),
 (18,5,['constructing','cleanup-blocked','dead']),
 (19,3,['dead','loading','constructing','preparing','committing','ready','cleanup-blocked']),
 (20,3,['training','detached','optimizer-retired','eval','generator-owned','adopted']),
 (21,1,['pending','dead','live','released']), (22,5,['pending','dead','active','closed']),
 (23,4,['pending','staged','prepared','aborted','committed']),
 (24,4,['pending','dead','active','prepared','committed','aborted']),
 (26,2,['pending','live','dead']), (27,2,['pending','live','dead']),
 (34,0,['unpublished','live','dead']), (37,9,['pending','live','dead']),
 (39,9,['pending','live','dead']), (40,7,['pending','live','dead']),
 (41,7,['pending','live','dead']), (43,12,['pending','live','dead']),
 (44,8,['pending','live','dead']), (45,8,['pending','live','dead']),
 (50,3,['open','sealed','aborted']), (53,6,['constructing','live','releasing','dead']),
 (63,5,['live','retired'])]:raw(i,slot,names)
for i in [6,25]:
    RAW_NAMES[i]=['STATIC']+['source-'+str(n) for n in range(5)]
    SCHEMAS[i]['raw']='(+ (vector-ref row 1) 1)'
for i in [28,33,35,38,42,46,47]:
    RAW_NAMES[i]=['STATIC','LIVE','DEAD','PENDING','PREPARED','BUSY']
    SCHEMAS[i]['raw']=SCHEMAS[i]['state']
for i in SCHEMAS:
    if i not in RAW_NAMES:RAW_NAMES[i]=['STATIC','PERMANENT-REFERENCE']
    SCHEMAS[i].setdefault('raw','1' if SCHEMAS[i]['mode']=='registry' else '0')
RAW_NAMES[64]=['STATIC','unpublished','adopted','dead']
# Terminal live rows are scrubbed after release; closure authority persists in
# the POSITIONALLY PAIRED originals root, not in now-cleared row10/shell slots.
EXTRA_DEFS['native/g3c4_p2g2_terminal_live_authority_extension.esk']='''(define (cpuv1-resource-terminal-original wanted rows originals seen)
  (cond ((or (null? rows) (null? originals)) #f)
        ((or (not (pair? rows)) (not (pair? originals)) (>= seen 16777216)) #f)
        ((eq? wanted (car rows)) (car originals))
        (else (cpuv1-resource-terminal-original wanted (cdr rows) (cdr originals) (+ seen 1)))))
'''
SCHEMAS[34]['check']='''(and (cpuv1-resource-vector? row 16)
  (let ((original (cpuv1-resource-terminal-original row
                    (vector-ref g3c4-p2g2-terminal-live-anchors 0)
                    (vector-ref g3c4-p2g2-terminal-live-originals 0) 0)))
    (and (procedure? original)
         (cpuv1-resource-vector? (original 'entry #f) 12)
         (eq? (cpuv1-resource-g3c4-find (original 'shell #f)) (original 'entry #f))
         (if (eq? (vector-ref row 0) 'dead)
             (original 'dead (original 'entry #f))
             (and (eq? (vector-ref row 1) (original 'entry #f))
                  (eq? (vector-ref row 10) original) (original 'check row))))))'''
SCHEMAS[25]['check']=SCHEMAS[25]['check'].replace('(cpu4-model-binding-native-lease-internal (vector-ref row 3))', '(vector-ref (cpuv1-resource-find-vector (vector-ref row 3) (vector-ref cpu4-model-bindings 0) 0 0) 5)')
RAW_NAMES[3]=['STATIC','native-live','native-dead']

RAW_NAMES[59]=['STATIC','new','baseline','done']
RAW_NAMES[60]=['STATIC','new','running','done']
RAW_NAMES[61]=['unset','training','handoff','adopted','released','done']
SCHEMAS[25]['check']='''(and (cpuv1-resource-vector? row 17)
 (eq? (e3-frame-entry-find (vector-ref row 0)) row)
 (integer? (vector-ref row 1)) (<= 0 (vector-ref row 1)) (<= (vector-ref row 1) 4)
 (if (= (vector-ref row 1) 4)
     (and (not (vector-ref row 3)) (not (vector-ref row 4)))
     (let ((binding (cpuv1-resource-find-vector (vector-ref row 3)
                      (vector-ref cpu4-model-bindings 0) 0 0)))
       (and (cpuv1-resource-vector? binding 12)
            (eq? (vector-ref row 9) (vector-ref binding 5))))))'''

# F1: admitted provider native identities have eight callback tokens outside
# shell-registry/index. Authenticate each exact slot rather than count as shell.
SCHEMAS[2]['check']=SCHEMAS[2]['check'][:-1]+" (cpuv1-resource-vector? (vector-ref (vector-ref row 2) 10) 8) (= (cpuv1-resource-count-vector (vector-ref row 1) (vector-ref tensor-provider-registry-root 0) 1 0 0) 1) (= (cpuv1-resource-provider-callbacks! packet native-context (vector-ref row 1) (vector-ref (vector-ref row 2) 10) 0) 0))"
