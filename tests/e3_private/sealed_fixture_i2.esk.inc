;;; Generated zero-fixture tuple only. The ordinary I2 seal still clears its
;;; open-construction rollback cells; this route retains its exact carrier and
;;; module-membership cells for one private sealed-graph retirement.
(define (i2-candidate-construction-seal-internal! value)
  (let* ((record
           (i2-construction-open-route
             value 'm3t-candidate 'i2-candidate-construction-seal-internal!))
         (root (module-construction-seal-internal! (vector-ref record 1))))
    (vector-set! record 3 'sealed)
    ;; Slot 10 is the exact published I2 membership head. A later module
    ;; publication must be rejected before this fixture can retire.
    (vector-set! record 10 (i2-modules-current))
    (vector-set! i2-construction-registry-root 1 #f)
    root))

(define (i2-candidate-construction-preflight! record triples)
  (let ((operation 'g3r-candidate-retire)
        (membership-count (i2-bounded-list-length (vector-ref record 4)
                                                  i2-max-batch))
        (carrier-count (i2-bounded-list-length (vector-ref record 5) 14)))
    (if (or (not (eq? (vector-ref record 9) 'm3t-candidate))
            (not (eq? (vector-ref record 3) 'sealed))
            (not (eq? (i2-modules-current) (vector-ref record 10)))
            (not membership-count)
            (not (= i2-module-count
                    (+ (vector-ref record 7) membership-count)))
            (not carrier-count)
            (not (= carrier-count 14))
            (not (= (vector-length triples) 14)))
        (i2-fail 'invalid-state operation
                 "candidate I2 membership or carrier ledger changed"))
    (let members ((expected (vector-ref record 4))
                  (actual (i2-modules-current)))
      (cond ((null? expected)
             (if (not (eq? actual (vector-ref record 6)))
                 (i2-fail 'invalid-state operation
                          "candidate I2 membership tail changed")))
            ((or (null? actual) (not (eq? (car expected) (car actual))))
             (i2-fail 'invalid-state operation
                      "candidate I2 membership prefix changed"))
            (else (members (cdr expected) (cdr actual)))))
    (let check ((index 0))
      (if (= index 14) #t
          (let* ((triple (vector-ref triples index))
                 (carrier (vector-ref triple 2)))
            (if (or (not (i2-identity-member? carrier (vector-ref record 5)))
                    (not (eq? (vector-ref carrier 2) (vector-ref triple 0)))
                    (not (eq? (vector-ref carrier 3) (vector-ref triple 1))))
                (i2-fail 'invalid-state operation
                         "candidate I2 indexed carrier changed"))
            (i2-carrier-preflight carrier 'parameter operation)
            (check (+ index 1)))))
    #t))

(define (i2-candidate-construction-unlink! record)
  ;; P1 token revocation has already crossed the destructive boundary.
  ;; The exact head and count were authenticated in preflight; this tail
  ;; only restores the preconstruction I2 root and invalidates its carriers.
  (vector-set! i2-module-registry-root 0 (vector-ref record 6))
  (set! i2-module-count (vector-ref record 7))
  (let loop ((remaining (vector-ref record 5)))
    (if (null? remaining) #t
        (begin
          (vector-set! (car remaining) 2 '())
          (vector-set! (car remaining) 3 '())
          (loop (cdr remaining)))))
  (vector-set! record 3 'retired)
  (vector-set! record 2 #f)
  (vector-set! record 4 '())
  (vector-set! record 5 '())
  (vector-set! record 6 '())
  (vector-set! record 8 #f)
  (vector-set! record 10 #f)
  #t)
