;;; Generated zero-fixture tuple only. The ledger is rooted before any native
;;; construction and never obtained from caller-editable returned metadata.
(define m3t-candidate-root (vector '()))
(define (m3t-candidate-ledger model operation)
  (let loop ((remaining (vector-ref m3t-candidate-root 0)))
    (cond ((null? remaining)
           (m3t-fail 'invalid-argument operation "not a candidate fixture"))
          ((eq? (vector-ref (car remaining) 1) (m3t-find model))
           (car remaining))
          (else (loop (cdr remaining))))))

(define (m3t-candidate-retire! model)
  (let* ((operation 'g3r-candidate-retire)
         (entry (m3t-entry model 'model operation #t))
         (ledger (m3t-candidate-ledger model operation))
         (original (vector-ref ledger 2))
         (successor (vector-ref ledger 3))
         (native (vector-ref ledger 4))
         (construction (vector-ref ledger 5))
         (i2-record (vector-ref ledger 6))
         (triples (vector-ref ledger 7))
         (handles (vector-ref entry 8)))
    (if (or (not (eq? (vector-ref ledger 0) 'sealed))
            (not (eq? entry (vector-ref ledger 1)))
            (not (eq? original (vector-ref entry 5)))
            (not (eq? successor (vector-ref entry 6)))
            (not (eq? native (vector-ref entry 3)))
            (not (eq? (vector-ref original 2) 'live))
            (not (eq? (vector-ref successor 2) 'live))
            (vector-ref entry 10))
        (m3t-fail 'invalid-state operation "candidate model is in use"))
    (let workspace ((remaining (vector-ref m3t-registry 0)))
      (if (null? remaining) #t
          (begin
            (if (and (eq? (vector-ref (car remaining) 1) 'workspace)
                     (eq? (vector-ref (car remaining) 4) entry)
                     (eq? (vector-ref (car remaining) 2) 'live))
                (m3t-fail 'invalid-state operation
                          "candidate has a live workspace"))
            (workspace (cdr remaining)))))
    (let frame ((remaining (vector-ref e3-frame-registry-root 0)))
      (if (null? remaining) #t
          (begin
            (if (and (eq? (vector-ref (car remaining) 3) entry)
                     (not (= (vector-ref (car remaining) 1) 4)))
                (m3t-fail 'invalid-state operation
                          "candidate has an unreleased E3 frame"))
            (frame (cdr remaining)))))
    (m3t-validate-schedule (vector-ref ledger 9) handles operation)
    (let check ((index 0))
      (if (= index 14) #t
          (let ((triple (vector-ref triples index)))
            (if (or (not (vector? triple))
                    (not (= (vector-length triple) 3))
                    (not (eq? (vector-ref triple 1) (vector-ref handles index))))
                (m3t-fail 'invalid-state operation
                          "candidate indexed enrollment changed"))
            (m3t-check
              (m3t-native-candidate-binding-preflight
                native index (vector-ref triple 0) (vector-ref triple 1))
              operation)
            (check (+ index 1)))))
    (i2-candidate-construction-preflight! i2-record triples)
    (module-candidate-construction-preflight-internal!
      construction (vector-ref ledger 10) handles (vector-ref ledger 9))
    ;; Both native tickets are the final fallible operations. After this point
    ;; no Scheme query, callback, allocation or user code may run.
    (m3t-check
      (m3t-native-candidate-initializer-retire-preflight
        (vector-ref successor 3)) operation)
    (if (not (= (m3t-native-candidate-retire-preflight
                  native (vector-ref original 3) (vector-ref successor 3)) 0))
        (m3t-candidate-invariant-fail))
    (module-candidate-construction-revoke-internal! construction)
    (i2-candidate-construction-unlink! i2-record)
    (if (not (= (m3t-native-candidate-retire-commit native) 0))
        (e3-native-invariant-fail))
    (if (not (= (m3t-native-candidate-initializer-retire-commit
                  (vector-ref successor 3)) 0))
        (e3-native-invariant-fail))
    (vector-set! entry 2 'dead)
    (vector-set! successor 2 'dead)
    (vector-set! ledger 0 'retired)
    #t))
