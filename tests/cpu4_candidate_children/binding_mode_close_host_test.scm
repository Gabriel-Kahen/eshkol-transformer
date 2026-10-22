;;; Actual owned helper plus extracted public G2 auth/close; native calls are doubles.
(load "host_test.scm")
(define public-path (or (getenv "CPU4_PUBLIC_G2_SOURCE")
                        (error "CPU4_PUBLIC_G2_SOURCE must name the reviewed public G2 source")))
(define (forms path)
  (call-with-input-file path
    (lambda (port) (let loop ((out '()))
      (let ((form (read port)))
        (if (eof-object? form) (reverse out) (loop (cons form out))))))))
(define (load-definition name source)
  (let ((form (find (lambda (form)
                     (and (pair? form) (eq? (car form) 'define)
                          (eq? (if (pair? (cadr form)) (caadr form) (cadr form)) name))) source)))
    (if (not form) (error "required actual source definition missing" name))
    (eval form (current-module))))
(define public-source (forms public-path))
(define call-source (forms "native/g3c4_call_entry_extension.esk"))
(define-syntax m3-call (syntax-rules () ((_ operation body ...) (begin body ...))))
(define-syntax g3c4-public-boundary (syntax-rules () ((_ operation body ...) (begin body ...))))
(define bitwise-and logand)
(define g3c4-fail cpu4-binding-fail)
(define g3c4-model-profile 'diagnostic-c4)
(define g3c4-p2g2-ownership-gate (vector #t))
(define g3c4-model-registry (vector '()))
(define g3c4-registry (vector '()))
(define (m3t-exact-i64? value) (and (integer? value) (exact? value)))
(define (module-mode-internal model) (vector-ref (vector-ref model 1) 0))
(define tokenizer (vector 'generation-tokenizer))
(define core (vector 'core 'raw '() '() '() #f "test-fingerprint"))
(define g3c4-baseline-tokenizer-fingerprint "test-fingerprint")
(define tokenizer-entry (list tokenizer core))
(define (t1-private-entry value) (and (eq? value tokenizer) tokenizer-entry))
(define (t1-core-tokenizer? value) (eq? value core))
(define (t1-tokenizer-core-vocab-size value) 256)
(define (g3c4-native-check status operation)
  (if (not (= status 0)) (cpu4-binding-fail 'invalid-state operation "native close rejected")) #t)
(define (g3c4-public-transport-member shell) (g3c4-registry-find shell 'test))
(define native-closes 0)
(define (g3c4-native-generator-close native)
  (check 'close-keeps-original-native-context (eq? native live-context))
  (vector-set! native 2 'closed)
  (set! live-context #f) (set! native-closes (+ native-closes 1)) 0)
(for-each (lambda (name) (load-definition name call-source))
  '(g3c4-entry-dead! g3c4-dead-entry? g3c4-registry-find g3c4-model-entry-live
    g3c4-baseline-tokenizer-entry g3c4-require-model-eval g3c4-positive-finite-f32-bits?
    g3c4-policy? g3c4-generator-entry-live g3c4-generator-entry-call-ready
    g3c4-generator-close-internal!))
(for-each (lambda (name) (load-definition name public-source))
  '(g3c4-public-g2-generator-rows g3c4-public-g2-generator-anchors
    g3c4-public-g2-generator-bindings g3c4-public-g2-root-find
    g3c4-public-g2-profile-original g3c4-public-g2-profile-live
    g3c4-public-g2-generator-create g3c4-public-generator-close))
;;; Lower source constructor/native context creation is an explicit double.
(define (g3c4-generator-create-with-source model tok operation source-thunk source-check)
  (let* ((policy (vector-ref (source-thunk) 0))
         (shell (vector 'generator)) (native (vector 'context model 'live))
         (entry (vector shell 'generator 'live native #f
                        (g3c4-model-entry-live model operation #f) tok policy
                        #f #f #f g3c4-model-profile)))
    (set! live-context native)
    (vector-set! g3c4-registry 0 (cons entry (vector-ref g3c4-registry 0)))
    shell))
(define before-close-checks checks)
(for-each
  (lambda (restore-eval?)
    (reset!) (set! native-closes 0)
    (let* ((t (ticket)) (opt (cpu4-owned-optimizer-create-internal! t 'config))
           (r (row t)) (model (cpu4-owned-model-internal t))
           (raw-bits (vector-ref model 2)) (bits (vector->list raw-bits))
           (model-entry (vector model 'c4-model 'live (vector 'owner) 'seeded
                                (vector 1 1729 0 0) (vector 1 1729 292 0)
                                #f (list->vector handles) #f #f g3c4-model-profile))
           (policy (vector g3c4-model-profile 0 1065353216 256 1065353216 2 255)))
      (vector-set! g3c4-model-registry 0 (list model-entry))
      (let* ((generator (g3c4-public-g2-generator-create model tokenizer policy
                            (vector policy 'seed 1729) #f #f))
             (entry (g3c4-generator-entry-live generator 'test))
             (context live-context)
             (profile (g3c4-public-g2-profile-live generator 'test))
             (registry o2-optimizers) (record (vector-ref r 7)) (native-o2 (vector-ref r 8)))
        (check 'owned-bind-rejects-live-context
          (eq? (guard (caught (#t caught)) (cpu4-owned-binding-create-internal! t) #f) context-error))
        (check 'late-context-rejection-retains-all17-train (all-mode? model 'train))
        (check 'rejection-clears-busy-with-no-binding (and (not (vector-ref r 12)) (not (vector-ref r 9))))
        (check 'context-tokenizer-policy-preserved
          (and (eq? context live-context) (eq? (vector-ref context 2) 'live)
               (eq? tokenizer (vector-ref entry 6)) (eq? policy (vector-ref entry 7))
               (eq? profile (g3c4-public-g2-profile-live generator 'test))))
        (check 'actual-call-ready-rejects-train
          (reject? (lambda () (g3c4-generator-entry-call-ready generator 'test))))
        (check 'forged-public-close-rejects
          (reject? (lambda () (g3c4-public-generator-close (vector 'forged)))))
        (vector-set! policy 6 254)
        (check 'changed-public-policy-close-rejects
          (reject? (lambda () (g3c4-public-generator-close generator))))
        (check 'failed-close-does-not-release-context
          (and (= native-closes 0) (eq? context live-context) (eq? (vector-ref context 2) 'live)))
        (vector-set! policy 6 255)
        (if restore-eval?
            (begin
              (module-eval! model)
              (check 'explicit-eval-restores-actual-call-ready
                (eq? entry (g3c4-generator-entry-call-ready generator 'test)))))
        (check 'actual-public-close-allows-train-or-eval (g3c4-public-generator-close generator))
        (check 'same-native-context-closed-once
          (and (= native-closes 1) (not live-context) (eq? (vector-ref context 2) 'closed)))
        (check 'close-did-not-change-mode
          (all-mode? model (if restore-eval? 'eval 'train)))
        (check 'public-close-idempotent (g3c4-public-generator-close generator))
        (check 'no-second-native-close (= native-closes 1))
        (let ((binding (cpu4-owned-binding-create-internal! t)))
          (check 'close-then-exact-same-owned-binding-retry
            (and (eq? opt (vector-ref r 6)) (eq? record (vector-ref r 7))
                 (eq? native-o2 (vector-ref r 8)) (eq? registry o2-optimizers)
                 (eq? binding (vector-ref r 9)) (not (vector-ref r 12))
                 (eq? model (cpu4-owned-model-internal t)) (all-mode? model 'train))))
        (check 'model-bits-unchanged-after-close-retry
          (and (eq? raw-bits (vector-ref model 2)) (equal? bits (vector->list raw-bits)))))))
  '(#f #t))
(format #t "CPU4 owned binding public-close host PASS: ~a additional checks; source-extracted auth/close, native doubles~%"
        (- checks before-close-checks))
