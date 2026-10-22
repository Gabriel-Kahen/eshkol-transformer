;;; Execute actual Scheme constructors with explicit nonnative acquisition/fault
;;; doubles. Primitive stores are instrumented to inject a post-pointer fault;
;;; all constructor control flow is read from the production source. Native
;;; admission and Eshkol primitive failure behavior are not claimed.
(use-modules (srfi srfi-1) (srfi srfi-34) (rnrs bytevectors))
(define checks 0)
(define (check label good) (if good (set! checks (+ checks 1)) (error label)))
(define original-vector-set! vector-set!)
(define inject-store? #f)
;; Simulate a wrapper fault after the first fixed native-pointer publication.
;; Production vector-set! is non-failing; this makes its guarded custody visible.
(define (test-fixed-store! target index value)
    (original-vector-set! target index value)
    (if (and inject-store? (= index 2) (memq value '(native-plan native-builder)))
        (begin (set! inject-store? #f) (raise 'after-pointer-store))))

(define (instrument-stores form)
  (cond ((not (pair? form)) form)
        ((eq? (car form) 'quote) form)
        (else
          (cons (if (eq? (car form) 'vector-set!) 'test-fixed-store!
                    (instrument-stores (car form)))
                (map instrument-stores (cdr form))))))
(call-with-input-file "native/tr3_c_restore_bindings_extension.esk"
  (lambda (port)
    (let read-next ()
      (let ((form (read port)))
        (if (not (eof-object? form))
            (begin (if (and (pair? form) (eq? (car form) 'define))
                       (eval (instrument-stores form) (current-module)))
                   (read-next)))))))
(define (o2-fail category operation message) (raise (vector category)))
(define (tr3-lease-fail-stop code) (raise (vector 'fail-stop code)))
(define (o2-member-identity? item items) (memq item items))
(define (o2-exact-nonnegative-i64? value) (and (integer? value) (>= value 0)))
(define (i2-bounded-list-length values limit)
  (let count ((remaining values) (n 0))
    (cond ((null? remaining) n) ((or (not (pair? remaining)) (= n limit)) #f)
          (else (count (cdr remaining) (+ n 1))))))
(define optimizer (vector 'optimizer 'native-optimizer))
(define paths (map (lambda (i) (list (number->string i))) (iota 14)))
(define (o2-require-optimizer operation shell) optimizer)
(define (o2-parse-config operation config)
  (vector 'canonical (vector 0 0) (vector 0 0 0 0)
          (list (vector paths 0 0 0 0 0))))
(define (o2-path-member? path paths) (member path paths))
(define (i2-carrier-preflight value kind operation) value)
(define (tr3-c-restore-native-request-initialize . args) 0)
(define (tr3-c-restore-native-request-append14 . args) 0)
(define (tr3-c-restore-native-last-category) 7)
(define mode 'success)
(define native-owned #f)
(define abort-count 0)
(define (tr3-c-restore-native-o2-prepare optimizer request)
  (let ((row (car tr3-c-restore-o2-ledgers)))
    (check "O2 exact self row globally rooted before native acquisition"
      (and (eq? row (vector-ref row 1)) (not (vector-ref row 2))
           (eq? (vector-ref row 7) 'constructing)
           (= (vector-length (vector-ref row 6)) 2)
           (= (vector-length (vector-ref (vector-ref row 6) 0)) 28)
           (= (bytevector-length (vector-ref (vector-ref row 6) 1)) 728))))
  (cond ((eq? mode 'before) (raise 'before-acquire))
        ((eq? mode 'null) '())
        (else (set! native-owned 'native-plan) native-owned)))
(define (tr3-c-restore-native-i2-create owner)
  (let ((row (car tr3-c-restore-i2-ledgers)))
    (check "I2 exact self row globally rooted before native acquisition"
      (and (eq? row (vector-ref row 1)) (not (vector-ref row 2))
           (eq? (vector-ref row 5) 'constructing)
           (eq? owner 'native-plan))))
  (cond ((eq? mode 'before) (raise 'before-acquire))
        ((eq? mode 'null) '())
        (else (set! native-owned 'native-builder) native-owned)))
(define (tr3-c-restore-native-i2-abort builder owner)
  (check "unpublished I2 cleanup keeps exact O2 authority"
    (and (eq? builder 'native-builder) (eq? owner 'native-plan)))
  (set! abort-count (+ abort-count 1))
  (set! native-owned #f) 0)
(define (make-request)
  (vector 'optimizer (vector 'config paths '() 0)
          (map (lambda (i) (vector 'buffer 'owned-clone i)) (iota 28))
          0 0 'test #f))
(for-each
  (lambda (failure)
    (set! mode failure)
    (let* ((request (make-request)) (stages (vector-ref request 2)))
      (check "O2 acquisition failure is recoverable"
        (guard (caught (#t #t)) (tr3-c-restore-o2-prepare-internal request) #f))
      (check "failed O2 constructor globally retains inert exact tombstone"
        (and (eq? (vector-ref (car tr3-c-restore-o2-ledgers) 7) 'aborted)
             (not (vector-ref request 6))))
      (check "no-plan failure preserves every caller-owned stage"
        (every (lambda (row) (integer? (vector-ref row 2))) stages))))
  '(before null))
(set! mode 'success)
(define request (make-request))
(define stages (vector-ref request 2))
(tr3-c-restore-o2-prepare-internal request)
(define o2-ledger (vector-ref request 6))
(check "successful O2 plan is exactly globally rooted published owner"
  (and (eq? o2-ledger (car tr3-c-restore-o2-ledgers))
       (eq? (vector-ref o2-ledger 2) 'native-plan)
       (eq? (vector-ref o2-ledger 7) 'prepared)))
(check "all28 caller raw pointers clear only after plan ownership exists"
  (every (lambda (row) (null? (vector-ref row 2))) stages))
(set! inject-store? #t)
(define after-request (make-request))
(check "O2 after-pointer fault preserves original exception"
  (guard (caught (#t (eq? caught 'after-pointer-store)))
    (tr3-c-restore-o2-prepare-internal after-request) #f))
(check "O2 after-pointer fault transfers exact cleanup through existing output"
  (and (eq? (vector-ref after-request 6) (car tr3-c-restore-o2-ledgers))
       (eq? (vector-ref (vector-ref after-request 6) 7) 'prepared)
       (every (lambda (row) (null? (vector-ref row 2))) (vector-ref after-request 2))))
(define o2-abort-blocked? #t)
(define o2-destroy-count 0)
(define (tr3-c-restore-native-o2-abort plan)
  (check "published O2 cleanup retains exact native plan" (eq? plan 'native-plan))
  (if o2-abort-blocked? 7
      (begin (set! o2-destroy-count (+ o2-destroy-count 1)) 0)))
(define published-after-fault (vector-ref after-request 6))
(check "published O2 owner preserves abort-blocked retry authority"
  (guard (caught (#t
                   (and (eq? (vector-ref published-after-fault 2) 'native-plan)
                        (eq? (vector-ref published-after-fault 7) 'prepared))))
    (tr3-c-restore-o2-abort-internal! published-after-fault) #f))
(set! o2-abort-blocked? #f)
(tr3-c-restore-o2-abort-internal! published-after-fault)
(check "published O2 retry destroys once and consumes exact ledger"
  (and (= o2-destroy-count 1)
       (null? (vector-ref published-after-fault 2))
       (eq? (vector-ref published-after-fault 7) 'aborted)))
(for-each
  (lambda (failure)
    (set! mode failure)
    (check "I2 no-native acquisition failure is recoverable"
      (guard (caught (#t #t)) (tr3-c-restore-i2-create-internal o2-ledger) #f))
    (check "failed I2 constructor globally retains inert exact tombstone"
      (and (eq? (vector-ref (car tr3-c-restore-i2-ledgers) 5) 'aborted)
           (null? (vector-ref (car tr3-c-restore-i2-ledgers) 2))))
    (check "I2 failure never consumes caller O2 owner"
      (and (eq? (vector-ref o2-ledger 2) 'native-plan)
           (eq? (vector-ref o2-ledger 7) 'prepared))))
  '(before null))
(set! mode 'success)
(define i2-ledger (tr3-c-restore-i2-create-internal o2-ledger))
(check "I2 successful prefix published by fixed stores"
  (and (eq? i2-ledger (car tr3-c-restore-i2-ledgers))
       (eq? (vector-ref i2-ledger 2) 'native-builder)
       (eq? (vector-ref i2-ledger 5) 'prefix)))
(set! inject-store? #t)
(check "I2 after-pointer failure preserves original exception"
  (guard (caught (#t (eq? caught 'after-pointer-store)))
    (tr3-c-restore-i2-create-internal o2-ledger) #f))
(check "I2 acquired empty prefix aborts exactly once, O2 remains intact"
  (and (= abort-count 1)
       (eq? (vector-ref (car tr3-c-restore-i2-ledgers) 5) 'aborted)
       (null? (vector-ref (car tr3-c-restore-i2-ledgers) 2))
       (eq? (vector-ref o2-ledger 7) 'prepared)))
(display "Restore binding instrumented source construction/rooting host PASS: checks=")
(display checks) (newline)
