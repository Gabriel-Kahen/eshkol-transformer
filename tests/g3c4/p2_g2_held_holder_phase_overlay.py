"""Build a test-only Eshkol overlay for the held-holder crash trace."""
from pathlib import Path
import sys


SOURCE = Path("native/g3c4_p2g2_terminal_held_holder_extension.esk")
ROOT_SOURCE = Path("native/g3c4_p2g2_terminal_held_holder_root.esk")

# Each replacement must be unique in the production source.  No production
# module is edited or linked against the diagnostic fd-4 symbol.
MARKERS = (
    ("(input-entry (g3c4-p2g2-prefix-preflight generator input))",
     "(input-entry (begin (held-phase 30)\n"
     "                       (let ((value (g3c4-p2g2-prefix-preflight generator input)))\n"
     "                         (held-phase 31) value)))"),
    ("(holder-original (g3c4-p2g2-holder-available-check holder operation))",
     "(holder-original (begin (held-phase 32)\n"
     "                           (let ((value (g3c4-p2g2-holder-available-check holder operation)))\n"
     "                             (held-phase 33) value)))"),
    ("(generator-entry (g3c4-generator-entry-call-ready generator operation))",
     "(generator-entry (begin (held-phase 34)\n"
     "                           (let ((value (g3c4-generator-entry-call-ready generator operation)))\n"
     "                             (held-phase 35) value)))"),
    ('"P2/G2 idle owner changed before call"))\n        (guard (caught\n',
     '"P2/G2 idle owner changed before call"))\n'
     "        (held-phase 36)\n        (guard (caught\n"),
    ("          (g3c4-native-check (g3c4-native-call-acquire native 2 2)\n"
     "                             operation)",
     "          (held-phase 37)\n"
     "          (g3c4-native-check (g3c4-native-call-acquire native 2 2)\n"
     "                             operation)\n"
     "          (held-phase 38)"),
    ("(output (g3c4-p2g2-protected-reserve call))",
     "(output (begin (held-phase 39)\n"
     "                         (let ((value (g3c4-p2g2-protected-reserve call)))\n"
     "                           (held-phase 40) value)))"),
    ("            (g3c4-p2g2-holder-bind!\n"
     "              holder native (vector-ref output 3) operation)",
     "            (held-phase 41)\n"
     "            (g3c4-p2g2-holder-bind!\n"
     "              holder native (vector-ref output 3) operation)\n"
     "            (held-phase 42)"),
    ("            (g3c4-native-check\n"
     "              (g3c4-p2g2-native-eos-first\n"
     "                native native-input first-staging 16) operation)",
     "            (held-phase 43)\n"
     "            (g3c4-native-check\n"
     "              (begin (held-phase 44)\n"
     "                (let ((status (g3c4-p2g2-native-eos-first\n"
     "                                native native-input first-staging 16)))\n"
     "                  (held-phase 45) status)) operation)\n"
     "            (held-phase 46)"),
    ("            (g3c4-p2g2-terminal-active-check\n"
     "              call input input-entry native-input terminal operation)\n"
     "            (if (= (vector-ref holder 4) 2)",
     "            (held-phase 47)\n"
     "            (g3c4-p2g2-terminal-active-check\n"
     "              call input input-entry native-input terminal operation)\n"
     "            (held-phase 48)\n"
     "            (held-phase 49)\n"
     "            (if (= (vector-ref holder 4) 2)"),
    ("                  holder native (vector-ref output 3) operation))\n"
     "            (let* ((id0 (g3c4-p2g2-terminal-selected\n"
     "                          first-staging operation))",
     "                  holder native (vector-ref output 3) operation))\n"
     "            (held-phase 50)\n"
     "            (let* ((id0 (begin (held-phase 51)\n"
     "                          (let ((value (g3c4-p2g2-terminal-selected\n"
     "                                         first-staging operation)))\n"
     "                            (held-phase 52) value)))"),
    ("              (t1-private-g3-decode-raw-into!\n"
     "                tokenizer first-staging first)\n"
     "              (g3c4-p2g2-terminal-active-check\n"
     "                call input input-entry native-input terminal operation)",
     "              (held-phase 53)\n"
     "              (t1-private-g3-decode-raw-into!\n"
     "                tokenizer first-staging first)\n"
     "              (held-phase 54)\n"
     "              (held-phase 55)\n"
     "              (g3c4-p2g2-terminal-active-check\n"
     "                call input input-entry native-input terminal operation)\n"
     "              (held-phase 56)"),
)


def instrument(source: str) -> str:
    entry = source.index("(define (g3c4-p2g2-terminal-generate-held-internal")
    head, body = source[:entry], source[entry:]
    for before, after in MARKERS:
        if body.count(before) != 1:
            raise ValueError(f"expected one phase anchor: {before}")
        body = body.replace(before, after, 1)
    return head + body


def main() -> None:
    root, destination = Path(sys.argv[1]), Path(sys.argv[2])
    destination.mkdir(parents=True, exist_ok=True)
    # Nested loads search the requiring file's directory before -I. The
    # feature root must live beside the instrumented extension.
    (destination / ROOT_SOURCE.name).write_bytes((root / ROOT_SOURCE).read_bytes())
    (destination / SOURCE.name).write_text(instrument((root / SOURCE).read_text()))


if __name__ == "__main__":
    main()
