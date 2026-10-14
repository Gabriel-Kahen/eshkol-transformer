"""Build a test-only Eshkol overlay for the held-holder crash trace."""
from pathlib import Path
import sys


SOURCE = Path("native/g3c4_p2g2_terminal_held_holder_extension.esk")

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
    (destination / SOURCE.name).write_text(instrument((root / SOURCE).read_text()))


if __name__ == "__main__":
    main()
