;;; Development-only Scheme execution of the source-private publisher with
;;; controlled C2/record/I/O callbacks. No compiled Eshkol acceptance claim.
(use-modules ((rnrs bytevectors) #:prefix r:))

(define (make-bytevector length fill) (r:make-bytevector length fill))
(define (bytevector? value) (r:bytevector? value))
(define (bytevector-length value) (r:bytevector-length value))
(define (string->utf8 value) (r:string->utf8 value))
(define (utf8->string value) (r:utf8->string value))
(define (bytevector-copy source start end)
  (let ((result (make-bytevector (- end start) 0)))
    (r:bytevector-copy! source start result 0 (- end start))
    result))

(define g3r-record-size 288)
(define g3r-save-tokenizer-fingerprint
  "sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704")
(define writes '())
(define on-callback (lambda () #t))
(define failure #f)

(define (c2-save-fail-raw category message details cause)
  (throw 'expected-failure category message))
(define (c1-native-fail operation message status)
  (throw 'expected-failure operation message status))
(define (c1-valid-file-path? path)
  (and (string? path) (> (string-length path) 0)))
(define (file-exists? path) #f)
(define (c2-policy-limits-internal operation policy) 'limits)
(define (c2-policy-c1-subpolicy-internal operation policy) 'subpolicy)
(define (c2-save-effective-model-policy limits subpolicy) subpolicy)
(define c2-save-stage-components-internal 'stage)
(define (c2-save-with-owner-borrow state limits subpolicy build publish)
  (on-callback)
  (publish (string->utf8 "validated-C2-image")))
(define (g3r-record-fields? policy rng history prompt generated manual
                            fingerprint digest)
  (and (vector? policy) (= (vector-length policy) 6)
       (vector? rng) (= (vector-length rng) 4)
       (bytevector? history) (= (bytevector-length history) 1)
       (= prompt 1) (= generated 0) (= manual 0)
       (bytevector? fingerprint) (= (bytevector-length fingerprint) 96)
       (bytevector? digest) (= (bytevector-length digest) 32)))
(define (c1-sha256 bytes) (make-bytevector 32 (bytevector-length bytes)))
(define (g3r-record-encode! destination policy rng history prompt generated
                            manual fingerprint digest)
  (if (eq? failure 'encode) (throw 'expected-failure 'encode))
  destination)
(define (g3r-record-decode! record digest destination)
  (if (eq? failure 'decode) (throw 'expected-failure 'decode))
  destination)
(define (c1-native-atomic-write path bytes expected-length overwrite)
  (set! writes (append writes (list (string-copy path))))
  (cond ((and (eq? failure 'first-write) (= (length writes) 1)) 1)
        ((and (eq? failure 'second-write) (= (length writes) 2)) 1)
        (else 0)))

(load (cadr (command-line)))

(define fields
  (vector (vector 0 1065353216 256 1065353216 0 -1)
          (vector 1 7 0 0) (string->utf8 "A") 1 0 0
          (string->utf8 g3r-save-tokenizer-fingerprint)))
(define checks 0)
(define (check label value)
  (if value (set! checks (+ checks 1))
      (error (string-append "publisher path test failed: " label))))
(define (expect-failure thunk)
  (catch 'expected-failure
    (lambda () (thunk) #f)
    (lambda args #t)))

(define (mutation-case which)
  (let ((c2-original (string-copy "/tmp/c2-target"))
        (companion-original (string-copy "/tmp/g3-target")))
    (set! writes '())
    (set! failure #f)
    (set! on-callback
      (lambda ()
        (if (eq? which 'c2)
            (string-set! c2-original 5 #\X)
            (string-set! companion-original 5 #\X))))
    (check "callback succeeds"
      (g3r-c2-exact-image-pair-publish-internal!
        'c2-owner c2-original companion-original 'policy fields))
    (check (string-append "writes retain both preflight path snapshots: "
                          (symbol->string which))
      (equal? writes '("/tmp/c2-target" "/tmp/g3-target")))
    (check "original caller path actually changed"
      (if (eq? which 'c2)
          (not (string=? c2-original "/tmp/c2-target"))
          (not (string=? companion-original "/tmp/g3-target"))))))

(mutation-case 'c2)
(mutation-case 'companion)

(set! on-callback (lambda () #t))
(set! failure 'encode)
(set! writes '())
(check "encode failure propagates"
  (expect-failure
    (lambda ()
      (g3r-c2-exact-image-pair-publish-internal!
        'c2-owner "/tmp/c2-target" "/tmp/g3-target" 'policy fields))))
(check "encode failure never writes" (null? writes))

(set! failure 'decode)
(set! writes '())
(check "decode failure propagates"
  (expect-failure
    (lambda ()
      (g3r-c2-exact-image-pair-publish-internal!
        'c2-owner "/tmp/c2-target" "/tmp/g3-target" 'policy fields))))
(check "decode failure never writes" (null? writes))

(set! failure 'first-write)
(set! writes '())
(check "first write failure propagates"
  (expect-failure
    (lambda ()
      (g3r-c2-exact-image-pair-publish-internal!
        'c2-owner "/tmp/c2-target" "/tmp/g3-target" 'policy fields))))
(check "first write failure stops before companion"
  (equal? writes '("/tmp/c2-target")))

(set! failure 'second-write)
(set! writes '())
(check "second write failure propagates"
  (expect-failure
    (lambda ()
      (g3r-c2-exact-image-pair-publish-internal!
        'c2-owner "/tmp/c2-target" "/tmp/g3-target" 'policy fields))))
(check "second write follows C2"
  (equal? writes '("/tmp/c2-target" "/tmp/g3-target")))

(display "G3-R path snapshot source execution PASS: checks=")
(display checks)
(newline)
