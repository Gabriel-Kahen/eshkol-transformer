from __future__ import annotations

import hashlib
from pathlib import Path
import random
import struct
import unittest

from tests.t2.generate_fixture import DOCUMENTS, SPECIALS, fixture_bytes
from tests.t2.reference import (
    CHECKSUM_DOMAIN,
    IDENTITY_DOMAIN,
    MAX_STREAM_BYTES,
    MAX_STREAM_TOKEN_IDS,
    Core,
    FormatError,
    Merge,
    Special,
    StreamDecoder,
    StreamEncoder,
    V1_HEADER,
    decode_artifact,
    decode_ids,
    encode_artifact,
    encode_bytes,
    fingerprint,
    pack_i64,
    payload,
    train,
)


FIXTURE = Path(__file__).with_name("fixtures") / "bpe_tokenizer_v1.tsv"


def configured(policy: str = "raw") -> Core:
    return train(DOCUMENTS, 8, 2, policy, SPECIALS, ("bos",), ("eos",))


def resign(raw: bytes) -> bytes:
    start = raw.rfind(b"checksum\t")
    if start < 0:
        raise AssertionError("mutation lost checksum")
    body = raw[:start]
    digest = hashlib.sha256(CHECKSUM_DOMAIN + body).hexdigest().encode()
    return body + b"checksum\t" + digest + b"\n"


def resize_payload(raw: bytes) -> bytes:
    start = raw.index(b"payload-bytes\t")
    end = raw.index(b"\n", start)
    payload_start = end + 1
    payload_end = raw.index(b"checksum-algorithm\t", payload_start)
    return raw[:start] + f"payload-bytes\t{payload_end - payload_start}".encode() + raw[end:]


class TrainingTests(unittest.TestCase):
    def test_tie_break_is_unsigned_pair_order(self) -> None:
        core = train(((b"abac",),), 1, 1)
        self.assertEqual(core.merges, (Merge(0, 256, ord("a"), ord("b")),))

    def test_document_order_and_chunk_partition_are_nonsemantic(self) -> None:
        one = train(DOCUMENTS, 8, 2, "raw", SPECIALS, ("bos",), ("eos",))
        reordered = train(tuple(reversed(DOCUMENTS)), 8, 2, "raw", SPECIALS, ("bos",), ("eos",))
        repartitioned = train(
            ((b"banana banana",), (b"ban", b"d", b"ana"), (b"b", b"a", b"n", b"a", b"n", b"a")),
            8, 2, "raw", SPECIALS, ("bos",), ("eos",),
        )
        self.assertEqual(one, reordered)
        self.assertEqual(one, repartitioned)
        self.assertEqual(encode_artifact(one), encode_artifact(repartitioned))

    def test_document_boundary_never_contributes_pair(self) -> None:
        self.assertEqual(train(((b"a",), (b"b",)), 1, 1).merges, ())
        self.assertEqual(train(((b"a", b"b"),), 1, 1).merges[0].left_id, ord("a"))

    def test_replacement_is_left_to_right_without_overlap(self) -> None:
        core = train(((b"aaaa",),), 1, 1)
        self.assertEqual(encode_bytes(core, b"aaaa"), (256, 256))
        self.assertEqual(encode_bytes(core, b"aaa"), (256, ord("a")))

    def test_maximum_is_bound_and_minimum_stops_training(self) -> None:
        self.assertEqual(train(((b"abc",),), 256, 2).merges, ())
        self.assertEqual(len(train(((b"abc",),), 1, 1).merges), 1)

    def test_training_admission_and_special_rules(self) -> None:
        self.assertEqual(len(train(((b"x" * 65_536,),), 0, 1).merges), 0)
        bad_calls = (
            lambda: train((), -1, 1),
            lambda: train((), 257, 1),
            lambda: train((), 1, 0),
            lambda: train(((b"x" * 65_537,),), 1, 1),
            lambda: train(((b"x",),), 1, 1, "replace"),
            lambda: train(((b"x",),), 1, 1, "raw", (("z", "omit"), ("a", "omit"))),
            lambda: train(((b"x",),), 1, 1, "raw", (("x", "error"),), ("x",), ()),
        )
        for call in bad_calls:
            with self.assertRaisesRegex(FormatError, "invalid-argument"):
                call()

    def test_strict_validates_complete_documents_not_chunks(self) -> None:
        core = train(((b"\xe2", b"\x82\xac"),), 1, 1, "strict")
        self.assertEqual(decode_ids(core, encode_bytes(core, "€")), "€".encode())
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            train(((b"\xe2",),), 1, 1, "strict")


class SemanticsTests(unittest.TestCase):
    def test_all_bytes_and_merge_round_trips(self) -> None:
        core = configured()
        cases = (b"", bytes(range(256)), b"banana banana", b"\0\xffbanana\x80")
        for source in cases:
            ids = encode_bytes(core, source)
            self.assertEqual(ids[0], core.prefix[0])
            self.assertEqual(ids[-1], core.suffix[0])
            self.assertEqual(decode_ids(core, ids), source)

    def test_special_spelling_is_never_recognized(self) -> None:
        core = configured()
        source = b"bos eos invalid"
        self.assertEqual(decode_ids(core, encode_bytes(core, source)), source)

    def test_error_special_unknown_and_bad_id_reject(self) -> None:
        core = configured()
        error_id = next(item.token_id for item in core.specials if item.decode == "error")
        for ids in ((error_id,), (core.specials[-1].token_id + 1,), (-1,), (True,), ("1",)):
            with self.assertRaisesRegex(FormatError, "invalid-argument"):
                decode_ids(core, ids)

    def test_raw_and_strict_utf8_policy(self) -> None:
        raw = configured("raw")
        self.assertEqual(decode_ids(raw, encode_bytes(raw, b"\xff\x80")), b"\xff\x80")
        strict = configured("strict")
        for malformed in (b"\x80", b"\xc0\x80", b"\xed\xa0\x80", b"\xe2\x82"):
            with self.assertRaisesRegex(FormatError, "invalid-argument"):
                encode_bytes(strict, malformed)
            with self.assertRaisesRegex(FormatError, "invalid-argument"):
                decode_ids(strict, tuple(malformed))


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.core = configured()
        self.raw = encode_artifact(self.core)

    def test_fixture_parse_reserialize_checksum_and_fingerprint(self) -> None:
        self.assertEqual(FIXTURE.read_bytes(), fixture_bytes())
        self.assertEqual(self.raw, fixture_bytes())
        parsed = decode_artifact(self.raw)
        self.assertEqual(parsed.core, self.core)
        self.assertEqual(encode_artifact(parsed.core), self.raw)
        checksum_input = self.raw[:self.raw.rfind(b"checksum\t")]
        expected_checksum = hashlib.sha256(CHECKSUM_DOMAIN + checksum_input).hexdigest().encode()
        self.assertEqual(self.raw.split(b"checksum\t", 1)[1].strip(), expected_checksum)
        digest = hashlib.sha256(IDENTITY_DOMAIN + V1_HEADER + payload(self.core)).hexdigest()
        self.assertEqual(parsed.fingerprint, f"sha256:eshkol-bpe-tokenizer-v1:{digest}")
        self.assertEqual(parsed.fingerprint, fingerprint(self.core))
        self.assertEqual(
            parsed.fingerprint,
            "sha256:eshkol-bpe-tokenizer-v1:1866ebedd76bf7d0e8e111ab25603a99aee927ea341e1305e4bb2c145648eb72",
        )

    def test_repeated_serialization_is_byte_deterministic(self) -> None:
        self.assertEqual({encode_artifact(configured()) for _ in range(20)}, {self.raw})

    def test_policy_limits_are_exact(self) -> None:
        payload_size = len(payload(self.core))
        decode_artifact(self.raw, max_file=len(self.raw), max_metadata=payload_size)
        with self.assertRaises(FormatError):
            decode_artifact(self.raw, max_file=len(self.raw) - 1)
        with self.assertRaises(FormatError):
            decode_artifact(self.raw, max_metadata=payload_size - 1)

    def test_truncation_trailing_nonascii_crlf_and_checksum_reject(self) -> None:
        bad = [
            b"", self.raw[:-1], self.raw + b"\n", b"\xef\xbb\xbf" + self.raw,
            self.raw.replace(b"\n", b"\r\n", 1),
            self.raw.replace(b"byte-bpe", b"byte-bp\xff", 1),
            self.raw.replace(b"merge\t0\t256\t97\t110", b"merge\t0\t256\t97\t111", 1),
        ]
        for source in bad:
            with self.assertRaises(FormatError):
                decode_artifact(source)
        for length in range(len(self.raw)):
            with self.assertRaises(FormatError):
                decode_artifact(self.raw[:length])

    def test_deep_merge_mutations_reject_even_when_resigned(self) -> None:
        first = self.core.merges[0]
        record = f"merge\t{first.rank}\t{first.result_id}\t{first.left_id}\t{first.right_id}".encode()
        mutations = (
            record.replace(b"merge\t0\t256", b"merge\t1\t256"),
            record.replace(b"\t256\t", b"\t257\t", 1),
            record.rsplit(b"\t", 1)[0] + f"\t{first.result_id}".encode(),
        )
        for replacement in mutations:
            mutated = resize_payload(self.raw.replace(record, replacement, 1))
            with self.assertRaises(FormatError):
                decode_artifact(resign(mutated))

    def test_version_algorithm_limits_and_canonical_integer_categories(self) -> None:
        cases = (
            (b"version\t1\t0", b"version\t2\t0", "version-mismatch"),
            (b"version\t1\t0", b"version\t1\t1", "version-mismatch"),
            (b"version\t1\t0", b"version\t01\t0", "corrupt-data"),
            (b"required-features\t0", b"required-features\t1", "unsupported"),
            (b"checksum-algorithm\tsha256", b"checksum-algorithm\tsha512", "unsupported"),
            (b"limit-merges\t256", b"limit-merges\t255", "corrupt-data"),
        )
        for old, new, category in cases:
            mutated = resize_payload(self.raw.replace(old, new, 1))
            with self.assertRaisesRegex(FormatError, category):
                decode_artifact(resign(mutated))


class StreamingTests(unittest.TestCase):
    @staticmethod
    def stream_encode(core: Core, chunks: tuple[bytes, ...]) -> tuple[int, ...]:
        state = StreamEncoder(core)
        output: list[int] = []
        for chunk in chunks:
            output.extend(state.push(chunk))
        output.extend(state.finish())
        return tuple(output)

    @staticmethod
    def stream_decode(core: Core, chunks: tuple[tuple[int, ...], ...]) -> bytes:
        state = StreamDecoder(core)
        output = bytearray()
        for chunk in chunks:
            output.extend(state.push(pack_i64(chunk)))
        output.extend(state.finish())
        return bytes(output)

    def test_every_input_boundary_and_empty_chunks_match_whole(self) -> None:
        core = configured()
        source = b"banana bandana banana\0banana"
        expected = encode_bytes(core, source)
        for boundary in range(len(source) + 1):
            actual = self.stream_encode(core, (b"", source[:boundary], b"", source[boundary:], b""))
            self.assertEqual(actual, expected)
        for boundary in range(len(expected) + 1):
            actual = self.stream_decode(core, ((), expected[:boundary], (), expected[boundary:], ()))
            self.assertEqual(actual, source)

    def test_seeded_adversarial_chunking_matches_whole(self) -> None:
        rng = random.Random(0x5432)
        core = configured()
        for _ in range(128):
            source = bytes(rng.randrange(256) for _ in range(rng.randrange(96)))
            cuts = sorted(rng.randrange(len(source) + 1) for _ in range(12))
            chunks = tuple(source[a:b] for a, b in zip((0, *cuts), (*cuts, len(source))))
            encoded = self.stream_encode(core, chunks)
            self.assertEqual(encoded, encode_bytes(core, source))
            token_cuts = sorted(rng.randrange(len(encoded) + 1) for _ in range(12))
            token_chunks = tuple(encoded[a:b] for a, b in zip((0, *token_cuts), (*token_cuts, len(encoded))))
            self.assertEqual(self.stream_decode(core, token_chunks), source)

    def test_strict_utf8_scalars_may_cross_stream_boundaries(self) -> None:
        core = configured("strict")
        source = "a€\U0001f642z".encode()
        state = StreamEncoder(core)
        output = list(state.push(b"a\xe2"))
        # The complete ASCII byte has entered the rank-stage transducer, but a
        # merge stage may retain it until the next complete token arrives.
        self.assertEqual(tuple(output), core.prefix)
        self.assertEqual(state.push(b"\x82"), ())
        output.extend(state.push(b"\xac" + source[4:]))
        output.extend(state.finish())
        encoded = tuple(output)
        self.assertEqual(encoded, encode_bytes(core, source))
        token_chunks = tuple((token_id,) for token_id in encoded)
        self.assertEqual(self.stream_decode(core, token_chunks), source)

    def test_strict_four_byte_scalar_matches_at_every_split(self) -> None:
        core = Core(utf8_policy="strict")
        source = "a€\U0001f642z".encode()
        expected = encode_bytes(core, source)
        for split in range(len(source) + 1):
            with self.subTest(direction="encode", split=split):
                self.assertEqual(
                    self.stream_encode(core, (source[:split], source[split:])),
                    expected,
                )
            with self.subTest(direction="decode", split=split):
                self.assertEqual(
                    self.stream_decode(core, (expected[:split], expected[split:])),
                    source,
                )

    def test_strict_four_byte_utf8_boundaries(self) -> None:
        core = Core(utf8_policy="strict")
        valid = (
            b"\xf0\x90\x80\x80",  # U+10000: lowest F0 scalar.
            b"\xf4\x8f\xbf\xbf",  # U+10FFFF: highest Unicode scalar.
        )
        invalid = (
            b"\xf0\x8f\xbf\xbf",  # Below U+10000: overlong F0 encoding.
            b"\xf4\x90\x80\x80",  # Above U+10FFFF.
        )
        for source in valid:
            expected = tuple(source)
            for split in range(len(source) + 1):
                with self.subTest(
                    kind="valid", direction="encode", source=source, split=split
                ):
                    self.assertEqual(
                        self.stream_encode(core, (source[:split], source[split:])),
                        expected,
                    )
                with self.subTest(
                    kind="valid", direction="decode", source=source, split=split
                ):
                    self.assertEqual(
                        self.stream_decode(core, (expected[:split], expected[split:])),
                        source,
                    )
        for source in invalid:
            ids = tuple(source)
            for split in range(len(source) + 1):
                with self.subTest(
                    kind="invalid", direction="encode", source=source, split=split
                ):
                    with self.assertRaisesRegex(FormatError, "invalid-argument"):
                        self.stream_encode(core, (source[:split], source[split:]))
                with self.subTest(
                    kind="invalid", direction="decode", source=source, split=split
                ):
                    with self.assertRaisesRegex(FormatError, "invalid-argument"):
                        self.stream_decode(core, (ids[:split], ids[split:]))

    def test_empty_logical_stream_gets_prefix_and_suffix_once(self) -> None:
        core = configured()
        self.assertEqual(self.stream_encode(core, ()), encode_bytes(core, b""))
        self.assertEqual(self.stream_encode(core, (b"", b"")), encode_bytes(core, b""))

    def test_malformed_staging_limit_and_failed_state_are_terminal(self) -> None:
        core = configured()
        decoder = StreamDecoder(core)
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            decoder.push(b"\0")
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            decoder.finish()

        decoder = StreamDecoder(core)
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            decoder.push(struct.pack("<q", -1))
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            decoder.push(b"")

        encoder = StreamEncoder(Core())
        self.assertEqual(len(encoder.push(b"x" * MAX_STREAM_BYTES)), MAX_STREAM_BYTES)
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            encoder.push(b"x")
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            encoder.finish()

        decoder = StreamDecoder(Core())
        self.assertEqual(len(decoder.push(pack_i64((ord("x"),) * MAX_STREAM_BYTES))), MAX_STREAM_BYTES)
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            decoder.push(pack_i64((ord("x"),)))
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            decoder.finish()

        decoder = StreamDecoder(configured())
        omit_id = configured().prefix[0]
        self.assertEqual(
            decoder.push(pack_i64((omit_id,) * MAX_STREAM_TOKEN_IDS)), b""
        )
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            decoder.push(pack_i64((omit_id,)))

    def test_feed_and_double_finish_reject(self) -> None:
        encoder = StreamEncoder(configured())
        encoder.finish()
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            encoder.push(b"")
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            encoder.finish()
        decoder = StreamDecoder(configured())
        decoder.finish()
        with self.assertRaisesRegex(FormatError, "invalid-argument"):
            decoder.push(b"")


if __name__ == "__main__":
    unittest.main()
