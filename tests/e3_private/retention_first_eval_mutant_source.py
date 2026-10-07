"""Exact test-only derivation of the first-call retention mutant driver."""

from pathlib import Path
import argparse


HERE = Path(__file__).resolve().parent
BASE = HERE / "retention_driver.esk"
MUTANT = HERE / "retention_first_eval_mutant_driver.esk"

PATCHES = (
    (
        "(extern i64 e3-test-seed-gradient ptr\n"
        "        :real et_e3_test_seed_gradient_v1)\n",
        "(extern i64 e3-test-seed-gradient ptr\n"
        "        :real et_e3_test_seed_gradient_v1)\n"
        "(extern i64 e3-test-first-eval-corrupt ptr i64\n"
        "        :real et_e3_test_first_eval_corrupt_v1)\n",
    ),
    (
        "(define e3-test-retention-root (vector #f))\n",
        "(define e3-test-retention-root (vector #f))\n"
        "(define e3-test-retention-first-mutation (vector -1))\n",
    ),
    (
        "    ;; Keep the evaluator's region-byte measurement unchanged. Every completed\n"
        "    ;; call, including the first warmup, checks the original pre-call image.\n",
        "    ;; Mutation is after the first evaluator's measured region, before its\n"
        "    ;; original-image comparison; later calls cannot select this hook.\n"
        "    (let ((mutation (vector-ref e3-test-retention-first-mutation 0)))\n"
        "      (if (>= mutation 0)\n"
        "          (begin\n"
        "            (e3-test-retention-check 'pre-mutation-model-image\n"
        "              (e3-test-retention-model=? model model-image))\n"
        "            (let ((owner (vector-ref\n"
        "                           (m3t-entry model 'model 'e3-test-run #t) 3)))\n"
        "              (e3-test-retention-check 'mutation-applied\n"
        "                (= (e3-test-first-eval-corrupt owner mutation) 0))\n"
        "              (e3-test-retention-check 'mutation-readback\n"
        "                (cond\n"
        "                  ((= mutation 0)\n"
        "                   (and (= (e3-test-gradient-metadata owner 0 0) 1)\n"
        "                        (= (e3-test-model-word owner 0 1 0) 1056964608)))\n"
        "                  ((= mutation 1)\n"
        "                   (= (e3-test-model-word owner 0 1 0) 1056964609))\n"
        "                  ((= mutation 2)\n"
        "                   (= (e3-test-gradient-metadata owner 0 1) 2))\n"
        "                  (else #f))))\n"
        "            (vector-set! e3-test-retention-first-mutation 0 -1))))\n"
        "    ;; Keep the evaluator's region-byte measurement unchanged. Every completed\n"
        "    ;; call, including the first warmup, checks the original pre-call image.\n",
    ),
    (
        "(define (e3-test-retention-prepare horizon present? corpus-path)\n",
        "(define (e3-test-retention-prepare horizon present? corpus-path mutation)\n",
    ),
    (
        "  (e3-test-retention-check 'gradient-choice (boolean? present?))\n",
        "  (e3-test-retention-check 'gradient-choice (boolean? present?))\n"
        "  (e3-test-retention-check 'mutation-choice\n"
        "    (or (and (not present?) (= mutation 0))\n"
        "        (and present? (or (= mutation 1) (= mutation 2)))))\n",
    ),
    (
        "      ;; Warm up lazy setup, then measure a separate steady-state complete call.\n",
        "      (vector-set! e3-test-retention-first-mutation 0 mutation)\n"
        "      ;; Warm up lazy setup, then measure a separate steady-state complete call.\n",
    ),
    (
        "(define (e3-test-retention-run phase horizon present? corpus-path)\n"
        "  (cond ((eq? phase 'prepare)\n"
        "         (e3-test-retention-prepare horizon present? corpus-path))\n",
        "(define (e3-test-retention-run phase horizon present? corpus-path mutation)\n"
        "  (cond ((eq? phase 'prepare)\n"
        "         (e3-test-retention-prepare horizon present? corpus-path mutation))\n",
    ),
    (
        "    (and (vector? scenario) (= (vector-length scenario) 4)\n",
        "    (and (vector? scenario) (= (vector-length scenario) 5)\n",
    ),
    (
        "                         (vector-ref scenario 3) corpus-path))\n",
        "                         (vector-ref scenario 3) corpus-path\n"
        "                         (vector-ref scenario 4)))\n",
    ),
)


def transform(source: str, *, reverse: bool = False) -> str:
    for before, after in (reversed(PATCHES) if reverse else PATCHES):
        old, new = (after, before) if reverse else (before, after)
        if source.count(old) != 1:
            raise ValueError("mutant derivation did not find one exact source hunk")
        source = source.replace(old, new, 1)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = transform(BASE.read_text())
    if args.check:
        if MUTANT.read_text() != expected or transform(expected, reverse=True) != BASE.read_text():
            raise SystemExit("first-evaluation mutant source differs from exact baseline patch")
    else:
        MUTANT.write_text(expected)


if __name__ == "__main__":
    main()
