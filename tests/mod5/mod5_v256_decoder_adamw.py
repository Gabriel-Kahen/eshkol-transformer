"""Generate the bounded live decoder witness and verify its AdamW evidence."""

from itertools import accumulate
import math
from pathlib import Path
import struct
import sys

from tests.eval5 import mod5_reference_oracle as oracle, mod5_v256_fixture as evaluation
from tests.eval5.build_mod5_v256_witness import digest_literal, one
from tests.mod5 import mod5_v256_c2_fixture as training
from tests.mod5.mod5_v256_gradient import HEAD, EMBEDDING, loss, section, source_for, tied_model


TAIL = Path(__file__).with_name("mod5_v256_decoder_adamw_tail.esk")
SIZES = tuple(math.prod(shape) for shape in oracle.SHAPES_256)
SUCCESS = "MOD5 V256 decoder AdamW PASS rows=6 updates=6"
# Same exact CPU f32 settings as the source-generated optimizer group.
OPTION_BITS = ("3c800000", "3f000000", "3f400000", "35800000", "3c800000")


def generate(directory):
    record = training.build(directory / "training")
    evaluation.build(directory / "evaluation")
    untied = (directory / "evaluation/model.etcp").read_bytes()
    tied = tied_model(untied)
    with (directory / "tied.etcp").open("xb") as stream:
        stream.write(tied)

    gradient = source_for(directory / "evaluation", evaluation.sha(tied))
    # Keep the pinned decoder and checked C1 loader. O2 extends the D2 root's
    # existing I2 identity universe so there is only one P1 provider registry.
    marker = ";;; Uses the fixed EVAL5 T1, D1 and C1 inputs and real MOD5-C backward."
    if gradient.count(marker) != 1:
        raise ValueError("expected one gradient witness boundary")
    prefix = one(gradient.split(marker, 1)[0], '(load "i2_wave2_root.esk")',
                 '(load "d2_wave2_root.esk")\n(load "o2_wave2_extension.esk")')
    # D2 supplies its own T2 tokenizer and live rows. The EVAL5/T1 source
    # helpers belong to the other aggregate and are unnecessary here.
    prefix = one(prefix, section(prefix, "(define expected-train-sha ",
                                "(define (load-checked-weights "), "")
    tied_code = section(gradient, "(define (canonicalize-tied! ", "(define (gradient-witness ")
    tail = one(TAIL.read_text(), "@@TIED_SHA@@", digest_literal(evaluation.sha(tied)))
    tail = one(tail, "@@TOKENIZER_SHA@@", digest_literal(record["tokenizer_artifact_sha256"]))
    tail = one(tail, "@@OPTIONS@@", " ".join(f'"{value}"' for value in OPTION_BITS))
    with (directory / "witness.esk").open("x") as stream:
        stream.write(prefix + tied_code + tail)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def words(data):
    return tuple(value for (value,) in struct.iter_unpack("<f", data))


def pack(values):
    return b"".join(struct.pack("<f", value) for value in values)


def adamw(before, grad, avg, avg_sq, update):
    """Independent binary32 recurrence for constant-schedule, unclipped O2."""
    lr, beta1, beta2, epsilon, decay = (
        struct.unpack(">f", bytes.fromhex(value))[0] for value in OPTION_BITS)
    # The chosen betas are exact dyadics through the sixth step.
    bias1, bias2 = f32(1 - beta1 ** update), f32(1 - beta2 ** update)
    m1 = f32(f32(beta1 * avg) + f32((1 - beta1) * grad))
    m2 = f32(f32(beta2 * avg_sq) + f32((1 - beta2) * f32(grad * grad)))
    corrected = f32(m1 / bias1)
    denominator = f32(f32(math.sqrt(f32(m2 / bias2))) + epsilon)
    parameter = f32(f32(before * f32(1 - f32(lr * decay)))
                    - f32(f32(lr * corrected) / denominator))
    return parameter, m1, m2


def roster(values, tied):
    sizes = SIZES[:EMBEDDING] if tied else SIZES
    offsets = tuple(accumulate(sizes, initial=0))
    shaped = [values[start:end] for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    if tied:
        shaped.append(shaped[HEAD])
    return shaped


def check_words(observed, expected, label):
    if observed != expected:
        for index, (left, right) in enumerate(zip(words(observed), words(expected), strict=True)):
            if pack((left,)) != pack((right,)):
                raise ValueError(f"{label}, word {index}: actual {left!r}, expected {right!r}")
        raise ValueError(f"{label}: noncanonical f32 byte sequence")


def verify(directory, variant, transcript, bit_trace):
    if variant not in ("untied", "tied"):
        raise ValueError("unknown decoder/AdamW variant")
    tied = variant == "tied"
    count = EMBEDDING if tied else len(SIZES)
    fixture = training.verify(directory / "training")
    evaluation.verify(directory / "evaluation")
    initial = (directory / "evaluation/model.etcp").read_bytes()
    model = (directory / "tied.etcp").read_bytes() if tied else initial
    if tied and model != tied_model(initial):
        raise ValueError("tied initial model differs from the canonical alias fixture")

    rows = transcript.read_text().splitlines()
    if len(rows) != 15 or rows[0] != f"PROFILE {variant} {count}" or rows[-1] != SUCCESS:
        raise ValueError("decoder/AdamW transcript inventory differs")
    stride = 4 * sum(SIZES[:count])
    bits = bit_trace.read_bytes()
    if len(bits) != 25 * stride:
        raise ValueError("weight, gradient or moment trace extent differs")
    pre = words(bits[:stride])
    initial_weights = oracle.weights_from_c1(model)
    check_words(bits[:stride], pack(value for row in initial_weights[:count] for value in row),
                "initial native model differs from C1")
    average = square = (0.0,) * len(pre)

    for update, doc in enumerate(training.DOCUMENTS, start=1):
        cursor = fixture["oracle_cursor_sha256_by_next_ordinal"][update - 1]
        if rows[2 * update - 1] != f"CURSOR {update - 1} {cursor}":
            raise ValueError(f"update {update}: incorrect live D2 cursor")
        fields = rows[2 * update].split()
        expected_row = ["ROW", str(update - 1), doc[:-1].hex(), doc[1:].hex(), "8"]
        if len(fields) != 7 or fields[:5] != expected_row:
            raise ValueError(f"update {update}: incorrect live D2 tokens or loss mask")
        offset = (1 + 4 * (update - 1)) * stride
        transferred, post_bits, avg_bits, sq_bits = (
            bits[offset + plane * stride:offset + (plane + 1) * stride] for plane in range(4))
        gradient, post = words(transferred), words(post_bits)
        moments, squares = words(avg_bits), words(sq_bits)
        expected = tuple(zip(*(adamw(*element, update) for element in zip(
            pre, gradient, average, square, strict=True)), strict=True))
        for label, observed, predicted in zip(
                ("weights", "exp_avg", "exp_avg_sq"),
                (post_bits, avg_bits, sq_bits), expected, strict=True):
            check_words(observed, pack(predicted), f"update {update} {label}")
        if (pack(pre) == post_bits or not any(gradient) or not any(moments)
                or not any(value > 0 for value in squares)):
            raise ValueError(f"update {update}: degenerate gradient, state or parameter change")
        for label, value, weights in zip(("pre", "post"), map(float, fields[5:]),
                                         (pre, post), strict=True):
            expected_loss = loss(roster(weights, tied), doc)
            if not math.isclose(value, expected_loss, rel_tol=5e-6, abs_tol=3e-6):
                raise ValueError(f"update {update} {label}: CE {value!r} != {expected_loss!r}")
        if float(fields[5]) == float(fields[6]):
            raise ValueError(f"update {update}: refreshed decoder loss did not change")
        pre, average, square = post, moments, squares

    if rows[-2] != f"CURSOR 6 {fixture['oracle_cursor_sha256_by_next_ordinal'][-1]}":
        raise ValueError("final D2 cursor differs")
    print(f"{variant}: six live rows, {count} physical destinations, exact AdamW words and CE oracle PASS")


def main(argv):
    if len(argv) == 3 and argv[1] == "build":
        generate(Path(argv[2]))
    elif len(argv) == 6 and argv[1] == "verify":
        verify(Path(argv[2]), argv[3], Path(argv[4]), Path(argv[5]))
    else:
        raise SystemExit("usage: python3 -m tests.mod5.mod5_v256_decoder_adamw "
                         "build DIR | verify DIR tied|untied TRANSCRIPT BITS")


if __name__ == "__main__":
    main(sys.argv)
