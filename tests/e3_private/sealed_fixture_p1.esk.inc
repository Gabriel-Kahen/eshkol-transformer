;;; Generated zero-fixture tuple only. These operations are absent from the
;;; ordinary trusted P1 source and never exported by a production package.
(define (g3r-construction-preflight identity root handles schedule)
  (let* ((operation 'g3r-candidate-retire)
         (record (construction-record operation identity)))
    (if (or (not (eq? (vector-ref record 1) 'sealed))
            (not (eq? (vector-ref record 3) root))
            (not (vector? handles))
            (not (= (vector-length handles) 14))
            (not (vector? schedule))
            (not (= (vector-length schedule) 2))
            (not (construction-schedule-equal?
                   (construction-rows record operation) schedule)))
        (fail 'invalid-state operation "candidate P1 construction changed" '()))
    (let check ((index 0))
      (if (= index 14) #t
          (let find ((entries (shell-registry-current)))
            (cond ((null? entries)
                   (fail 'invalid-state operation "candidate P1 handle is missing" '()))
                  ((and (= (vector-length (car entries)) 4)
                        (eq? (vector-ref (car entries) 1) 'handle)
                        (eq? (vector-ref (car entries) 0) (vector-ref handles index))
                        (eq? (vector-ref (car entries) 3) identity)
                        (not (eq? (vector-ref (car entries) 2)
                                  dead-construction-raw)))
                   (check (+ index 1)))
                  (else (find (cdr entries)))))))
    (native-check operation
      (p1-native-candidate-graph-preflight native-context identity))
    #t))

(define (g3r-construction-revoke! identity)
  ;; All Eshkol/native membership scans have completed before the destructive
  ;; boundary. An unexpected native status is a contract defect, never a
  ;; recoverable error after any token has been revoked.
  (let ((record (construction-record 'g3r-candidate-retire identity))
        (entries (shell-registry-current)))
    (if (not (= (p1-native-candidate-graph-revoke native-context identity) 0))
        (e3-native-invariant-fail))
    (let loop ((remaining entries))
      (if (null? remaining) #t
          (let ((entry (car remaining)))
            (if (and (= (vector-length entry) 4)
                     (eq? (vector-ref entry 3) identity)
                     (or (eq? (vector-ref entry 1) 'module)
                         (eq? (vector-ref entry 1) 'handle)))
                (begin
                  (if (eq? (vector-ref entry 1) 'module)
                      (active-record-unlink! 1 (vector-ref entry 2)))
                  (vector-set! entry 2 dead-construction-raw)))
            (loop (cdr remaining)))))
    (vector-set! record 1 'retired)
    (vector-set! record 3 #f)
    (vector-set! record 4 #f))
  #t)
