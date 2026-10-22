;;; Actual factory/guard Scheme with explicit nonnative reader/child doubles.
;;; No Eshkol, C2 file decoding or native execution admission is claimed.
(use-modules (srfi srfi-1) (srfi srfi-34) (rnrs bytevectors))
(define checks 0)
(define (check label value) (if value (set! checks (+ checks 1)) (error label)))
(define (forms path)
  (call-with-input-file path
    (lambda (port) (let read-next ((out '()))
      (let ((form (read port)))
        (if (eof-object? form) (reverse out) (read-next (cons form out))))))))
(define (definition name source)
  (find (lambda (f) (and (pair? f) (eq? (car f) 'define)
                        (eq? (if (pair? (cadr f)) (caadr f) (cadr f)) name))) source))
(eval (definition 'cpu4-checkpoint-require-nonregional!
                   (forms "native/cpu4_checkpoint_extension.esk")) (current-module))
(for-each (lambda (f) (eval f (current-module)))
          (forms "native/cpu4_checkpoint_factory_extension.esk"))
(define region-stack '())
(define (cpu4-checkpoint-native-current-region) (if (null? region-stack) '() (car region-stack)))
(define events '())
(define (event value) (set! events (append events (list value))))
(define fault (vector 'factory-fault))
(define cut 'none)
(define image-release-blocked? #f)
(define external-state? #f)
(define post-consumed-release? #f)
(define images '())
(define stages '())
(define children '())
(define source-canonical "source-config")
(define resolved (vector source-canonical "source-fingerprint" 1729))
(define tokenizer (vector 'caller-tokenizer 'live))
(define dataset (vector 'caller-dataset 'live))
(define report (vector 'authentic-report-double))
(define policy (vector 'policy-double))
(define (cpu4-binding-fail category operation message) (raise (vector category operation)))
(define cpu4-control-fail cpu4-binding-fail)
(define (cpu4-binding-fail-stop code) (throw 'fail-stop code))
(define (fault-at phase) (if (eq? cut phase) (raise fault)))
(define (live-images) (count (lambda (state) (eq? (vector-ref state 7) 'live)) images))
(define (live-children) (count (lambda (row) (eq? (vector-ref row 3) 'open)) children))
(define (caller-live?) (and (eq? (vector-ref tokenizer 1) 'live)
                           (eq? (vector-ref dataset 1) 'live)))
(define (c2-checkpoint-load-stage-internal path policy)
  (event 'read-stage) (fault-at 'stage)
  (let ((stage (vector 'stage 'live))) (set! stages (cons stage stages)) stage))
(define (c2-checkpoint-load-capability-descriptors-internal stage operation)
  (event 'descriptors) (fault-at 'descriptors) '(fixed-f32))
(define (c2-public-capability-admit-all supplied descriptors)
  (event 'admit) (check "genuine caller report forwarded" (eq? supplied report)) (fault-at 'admit))
(define (c2-checkpoint-load-reconstruct-internal stage operation)
  (event 'reconstruct) (fault-at 'reconstruct)
  (let* ((image (vector 'loaded-C2)) (state (make-vector 9 #f)))
    (vector-set! state 0 image) (vector-set! state 7 'live)
    (set! images (cons state images)) (vector-set! stage 1 'dead) image))
(define (c2-checkpoint-load-staging-dead-internal? stage) (eq? (vector-ref stage 1) 'dead))
(define (c2-checkpoint-load-staging-release-internal! stage)
  (event 'stage-release) (vector-set! stage 1 'dead) #t)
(define (c2-training-state-find image) (find (lambda (state) (eq? image (vector-ref state 0))) images))
(define (c2-training-state-release-internal! image operation)
  (let ((state (c2-training-state-find image)))
    (check "release only exact owned reconstructed shell" state)
    (if image-release-blocked? (begin (event 'image-blocked) (raise 'image-release-blocked)))
    (if (not (eq? (vector-ref state 7) 'dead))
        (begin (event 'image-release) (vector-set! state 7 'dead)))
    (if post-consumed-release? (begin (set! post-consumed-release? #f) (raise 'post-consumed-release)))
    #t))
(define (cpu4-checkpoint-parent trainer) (let ((parent (make-vector 17 #f)))
  (vector-set! parent 2 trainer) (vector-set! parent 3 'constructing) parent))
(define (cpu4-checkpoint-stage-source! parent image)
  (event 'source-stage) (vector-set! parent 4 'borrow)
  (fault-at 'source-access)
  (vector-set! parent 8
    (vector (vector 'philox4x32-10 1 1729 0 0) (vector "raw" 256)
            (vector '#vu8(1) '#vu8(2))
            (vector (string->utf8 source-canonical) "source-fingerprint")
            (vector 0 0 0)))
  (vector-set! parent 4 #f) (vector-set! parent 3 'source-closed)
  (event 'source-access-end)
  (vector 'topology (vector '(checkpoint-optimizer-config) 'paths 'aliases 0)))
(define (cpu4-checkpoint-abort! parent)
  (if (vector-ref parent 4) (begin (event 'source-access-end) (vector-set! parent 4 #f)))
  (event 'staging-drain) (vector-set! parent 3 'dead) #t)
(define (cpu4-binding-profile! config operation)
  (check "authentic supplied resolved config forwarded" (eq? config resolved)))
(eval (definition 'd2-core-bytes=? (forms "internal/d2/lib/d2_semantic_core.esk")) (current-module))
(define (x1-private-config-canonical config) (vector-ref config 0))
(define (x1-private-config-fingerprint config) (vector-ref config 1))
(define (x1-private-config-ref config key) (vector-ref config 2))
(define (cpu4-owned-candidate-create-internal config)
  (event 'candidate) (check "owned C2 released before ANY child acquisition" (= (live-images) 0))
  (fault-at 'candidate)
  (let ((row (make-vector 15 #f)))
    (vector-set! row 3 'open) (set! children (cons row children)) row))
(define (cpu4-owned-optimizer-create-internal! child config)
  (event 'optimizer)
  (check "optimizer config derives from authenticated detached source"
         (equal? config '(checkpoint-optimizer-config)))
  (vector-set! child 6 'owned-optimizer) (fault-at 'optimizer) 'owned-optimizer)
(define (cpu4-owned-binding-create-internal! child)
  (event 'binding) (vector-set! child 9 'owned-binding) (fault-at 'binding) 'owned-binding)
(define (cpu4-owned-trainer-create-internal! child supplied-tokenizer supplied-dataset)
  (event 'trainer)
  (check "caller T2/D2 forwarded by exact identity"
         (and (eq? supplied-tokenizer tokenizer) (eq? supplied-dataset dataset)))
  (fault-at 'trainer)
  (let ((trainer (vector 'trainer child))) (vector-set! child 10 trainer) trainer))
(define (cpu4-trainer-record-internal trainer operation)
  (event 'record) (fault-at 'record) (vector trainer))
(define (cpu4-checkpoint-prepare! record parent detached)
  (event 'prepare) (fault-at 'prepare) (vector-set! parent 3 'sealed))
(define (cpu4-checkpoint-commit! parent)
  (event 'commit) (vector-set! parent 3 'committing) (fault-at 'postwrite)
  (vector-set! parent 3 'dead) #t)
(define (cpu4-owned-candidate-retire-internal! child)
  (event 'child-preflight)
  (check "owned image/access/staging never survives to global O2 preflight" (= (live-images) 0))
  (if external-state? (raise 'external-live-state))
  (vector-set! child 3 'dead) (event 'child-retired) #t)
(define (create recovery)
  (cpu4-checkpoint-factory-create-internal! "checkpoint.c2" policy report resolved tokenizer dataset recovery))

;; Guard rejection precedes recovery publication, file access, registry creation
;; and cleanup/retry effects. Region pointer nesting here is an explicit double.
(check "host baseline query is NULL" (null? (cpu4-checkpoint-native-current-region)))
(for-each
  (lambda (depth)
    (set! region-stack (make-list depth (vector 'region)))
    (let ((recovery (vector #f)) (before (vector-ref cpu4-checkpoint-factories 0)))
      (set! events '())
      (check "regional create rejects unsupported before any side effect"
        (guard (caught (#t (eq? (vector-ref caught 0) 'unsupported))) (create recovery) #f))
      (check "regional create leaves recovery/file/registry/owners unchanged"
        (and (not (vector-ref recovery 0)) (eq? before (vector-ref cpu4-checkpoint-factories 0))
             (null? events) (= (live-images) 0) (= (live-children) 0)))))
  '(1 2))
(set! region-stack '())

;; Every normal constructor cut returns cleanup authority and preserves caller
;; resources. Optimizer/binding cuts include partially published child custody.
(for-each
  (lambda (phase)
    (set! cut phase) (set! events '())
    (let ((recovery (vector #f)))
      (check "constructor cut raises" (guard (caught (#t #t)) (create recovery) #f))
      (check "recovery ticket enrolled before every acquisition cut" (vector-ref recovery 0))
      (cpu4-checkpoint-factory-retire-internal! (vector-ref recovery 0))
      (check "constructor cut drains source/children and preserves caller T2/D2"
        (and (= (live-images) 0) (= (live-children) 0) (caller-live?)))))
  '(stage descriptors admit reconstruct source-access candidate optimizer binding trainer record prepare))
(set! cut 'none)
(set! events '())
(define ready-recovery (vector #f))
(define trainer (create ready-recovery))
(check "factory returns genuine trainer on original child"
  (and (eq? (car (vector->list trainer)) 'trainer) (= (live-children) 1) (= (live-images) 0)))
(check "success path stages/ends source and releases image before child creation"
  (and (< (list-index (lambda (x) (eq? x 'source-access-end)) events)
          (list-index (lambda (x) (eq? x 'image-release)) events))
       (< (list-index (lambda (x) (eq? x 'image-release)) events)
          (list-index (lambda (x) (eq? x 'candidate)) events))))
(set! external-state? #t)
(check "external global state blocks owned cleanup without losing custody"
  (guard (caught (#t (eq? caught 'external-live-state)))
    (cpu4-checkpoint-factory-retire-internal! (vector-ref ready-recovery 0)) #f))
(check "blocked cleanup retains same exact owned child/trainer"
  (let ((row (cpu4-checkpoint-factory-row (vector-ref ready-recovery 0) 'test)))
    (and (eq? (vector-ref row 15) trainer) (vector-ref row 14)
         (eq? (vector-ref row 3) 'cleanup-blocked) (= (live-children) 1))))
(set! region-stack (list (vector 'outer) (vector 'inner)))
(let ((before events))
  (check "regional retry rejects without touching held authority"
    (guard (caught (#t (eq? (vector-ref caught 0) 'unsupported)))
      (cpu4-checkpoint-factory-retire-internal! (vector-ref ready-recovery 0)) #f))
  (check "regional retry ownership unchanged" (and (eq? before events) (= (live-children) 1))))
(set! region-stack '()) (set! external-state? #f)
(cpu4-checkpoint-factory-retire-internal! (vector-ref ready-recovery 0))
(check "nonregional explicit retry retires exact child and preserves T2/D2"
  (and (= (live-children) 0) (caller-live?)))

;; A pre-transfer C2 release rejection must retain the source shell. It blocks
;; child creation/retirement until a later nonregional explicit retry drains it.
(set! image-release-blocked? #t) (set! events '())
(define image-recovery (vector #f))
(check "pre-transfer image release failure is reported"
  (guard (caught (#t (eq? caught 'image-release-blocked))) (create image-recovery) #f))
(check "blocked owned image remains rooted and no child/global preflight occurs"
  (let ((row (cpu4-checkpoint-factory-row (vector-ref image-recovery 0) 'test)))
    (and (vector-ref row 11) (= (live-images) 1) (= (live-children) 0)
         (not (memq 'child-preflight events)) (eq? (vector-ref row 3) 'cleanup-blocked))))
(set! image-release-blocked? #f)
(cpu4-checkpoint-factory-retire-internal! (vector-ref image-recovery 0))
(check "owned image retry releases exact image before closing recovery"
  (and (= (live-images) 0) (= (live-children) 0) (caller-live?)))

(set! post-consumed-release? #t)
(define dead-image-recovery (vector #f))
(check "post-consumption image defect reported without retrying dead owner"
  (guard (caught (#t (eq? caught 'post-consumed-release))) (create dead-image-recovery) #f))
(check "verified dead image clears rooted authority despite release exception"
  (let ((row (cpu4-checkpoint-factory-row (vector-ref dead-image-recovery 0) 'test)))
    (and (not (vector-ref row 11)) (= (live-images) 0) (eq? (vector-ref row 3) 'dead))))

(vector-set! resolved 0 "different-config")
(define wrong-recovery (vector #f))
(check "different canonical config rejects without seeded substitute child"
  (guard (caught (#t #t)) (create wrong-recovery) #f))
(check "config rejection owns no model/image and preserves caller data"
  (and (= (live-images) 0) (= (live-children) 0) (caller-live?)))
(vector-set! resolved 0 source-canonical)

;; Original construction error and exact child survive an external global
;; O2-state blocker encountered only while cleanup drains its owned resources.
(set! cut 'prepare) (set! external-state? #t)
(define partial-recovery (vector #f))
(check "partial construction retains original failure across blocked cleanup"
  (guard (caught (#t (eq? caught fault))) (create partial-recovery) #f))
(check "partial cleanup retains exact ticket and source custody already drained"
  (let ((row (cpu4-checkpoint-factory-row (vector-ref partial-recovery 0) 'test)))
    (and (eq? (vector-ref row 16) fault)
         (eq? (vector-ref row 17) 'external-live-state)
         (not (vector-ref row 11)) (vector-ref row 14)
         (eq? (vector-ref (vector-ref row 12) 3) 'dead)
         (caller-live?))))
(set! cut 'none) (set! external-state? #f)
(cpu4-checkpoint-factory-retire-internal! (vector-ref partial-recovery 0))
(check "external-state removal permits exact partial-child retry"
  (and (= (live-images) 0) (= (live-children) 0) (caller-live?)))

;; Exercise the actual existing-trainer entry guards against held caller cells.
(for-each
  (lambda (name) (eval (definition name (forms "native/cpu4_checkpoint_extension.esk")) (current-module)))
  '(cpu4-trainer-snapshot-internal cpu4-trainer-restore-internal!))
(set! region-stack (list (vector 'outer) (vector 'inner)))
(set! events '())
(define snapshot-output (vector #f))
(check "existing snapshot guard rejects before output/owner effects"
  (guard (caught (#t (eq? (vector-ref caught 0) 'unsupported)))
    (cpu4-trainer-snapshot-internal 'caller-trainer snapshot-output) #f))
(check "existing restore guard rejects before source/receiver effects"
  (guard (caught (#t (eq? (vector-ref caught 0) 'unsupported)))
    (cpu4-trainer-restore-internal! 'caller-trainer 'caller-checkpoint) #f))
(check "checkpoint regional negatives leave caller output and owners unchanged"
  (and (null? events) (not (vector-ref snapshot-output 0)) (caller-live?)))
(set! region-stack '())

;; Last: a sealed write-tail exception cannot return a recoverable partial model.
(set! cut 'postwrite) (set! events '())
(check "unexpected postwrite close/write defect fail-stops"
  (catch 'fail-stop
    (lambda () (create (vector #f)) #f)
    (lambda (key code) (= code 134))))
(check "postwrite defect never attempts ordinary child retirement"
  (not (memq 'child-preflight events)))
(display "CPU4 fresh checkpoint actual factory/guard host PASS: checks=")
(display checks) (newline)
