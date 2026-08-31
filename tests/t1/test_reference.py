from __future__ import annotations

import hashlib
import locale
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from tests.t1.reference import (
    CHECKSUM_DOMAIN,
    FormatError,
    Identity,
    MAX_FILE,
    Special,
    decode_artifact,
    decode_ids,
    encode_artifact,
    encode_bytes,
    fingerprint,
)


FIXTURE = Path(__file__).with_name("fixtures") / "byte_tokenizer_v1.tsv"


def configured(policy: str = "raw") -> Identity:
    return Identity(
        policy,
        (
            Special(256, "bos", "omit"),
            Special(257, "eos", "omit"),
            Special(258, "invalid", "error"),
        ),
        (256,),
        (257,),
    )


def resign(raw: bytes) -> bytes:
    marker = b"checksum\t"
    start = raw.rfind(marker)
    if start < 0:
        raise AssertionError("test mutation lost checksum record")
    body = raw[:start]
    digest = hashlib.sha256(CHECKSUM_DOMAIN + body).hexdigest().encode()
    return body + marker + digest + b"\n"


class SemanticsTests(unittest.TestCase):
    def test_all_256_raw_bytes_round_trip(self) -> None:
        source = bytes(range(256))
        ids = encode_bytes(Identity(), source)
        self.assertEqual(ids, tuple(range(256)))
        self.assertEqual(decode_ids(Identity(), ids), source)

    def test_empty_and_embedded_nul(self) -> None:
        for source in (b"", b"\0", b"a\0b\0", b"\0" * 32):
            self.assertEqual(decode_ids(Identity(), encode_bytes(Identity(), source)), source)

    def test_valid_utf8_is_byte_exact_without_normalization(self) -> None:
        corpora = (
            "ASCII",
            "caf\u00e9 cafe\u0301",
            "\u03bb\u0416\u05d0\u0634",
            "\u6f22\u5b57\U0001f642",
            "\r\n\t\0",
        )
        tokenizer = Identity("strict")
        for text in corpora:
            source = text.encode("utf-8")
            self.assertEqual(decode_ids(tokenizer, encode_bytes(tokenizer, source)), source)

    def test_raw_preserves_representative_malformed_utf8(self) -> None:
        malformed = (
            b"\x80",
            b"\xc0\x80",
            b"\xe0\x80\x80",
            b"\xed\xa0\x80",
            b"\xf4\x90\x80\x80",
            b"\xf5\x80\x80\x80",
            b"\xe2\x82",
            b"\xff",
        )
        for source in malformed:
            self.assertEqual(decode_ids(Identity("raw"), encode_bytes(Identity("raw"), source)), source)

    def test_strict_rejects_malformed_utf8_on_encode_and_decode(self) -> None:
        malformed = (b"\x80", b"\xc0\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x82", b"\xff")
        for source in malformed:
            with self.assertRaisesRegex(FormatError, "invalid-argument"):
                encode_bytes(Identity("strict"), source)
            with self.assertRaisesRegex(FormatError, "invalid-argument"):
                decode_ids(Identity("strict"), tuple(source))

    def test_special_insertion_order_repetition_and_omission(self) -> None:
        tokenizer = configured()
        ids = encode_bytes(tokenizer, b"abc")
        self.assertEqual(ids, (256, 97, 98, 99, 257))
        self.assertEqual(decode_ids(tokenizer, ids), b"abc")

    def test_same_special_may_overlap_prefix_and_suffix(self) -> None:
        tokenizer = Identity(
            "raw", (Special(256, "boundary", "omit"),),
            (256, 256), (256, 256),
        )
        ids = (256, 256, 120, 256, 256)
        self.assertEqual(encode_bytes(tokenizer, b"x"), ids)
        self.assertEqual(decode_ids(tokenizer, ids), b"x")

    def test_special_spelling_is_not_recognized(self) -> None:
        tokenizer = configured()
        source = b"<bos>bos decode.error eos"
        self.assertEqual(encode_bytes(tokenizer, source)[1:-1], tuple(source))

    def test_error_special_and_unknown_ids_reject(self) -> None:
        tokenizer = configured()
        for ids in ((258,), (259,), (-1,), (1 << 63,), (True,), ("1",)):
            with self.assertRaisesRegex(FormatError, "invalid-argument"):
                decode_ids(tokenizer, ids)  # type: ignore[arg-type]

    def test_special_validation_collisions_gaps_and_bad_insertions(self) -> None:
        bad = (
            Identity("raw", (Special(255, "x", "omit"),)),
            Identity("raw", (Special(257, "x", "omit"),)),
            Identity("raw", (Special(256, "x", "omit"), Special(256, "y", "omit"))),
            Identity("raw", (Special(256, "x", "omit"), Special(257, "x", "omit"))),
            Identity("raw", (Special(256, "X", "omit"),)),
            Identity("raw", (Special(256, "x", "replace"),)),
            Identity("raw", (Special(256, "x", "error"),), (256,), ()),
            Identity("raw", (Special(256, "x", "omit"),), (257,), ()),
        )
        for tokenizer in bad:
            with self.assertRaises(FormatError):
                encode_artifact(tokenizer)


class FormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = configured()
        self.raw = encode_artifact(self.identity)

    def test_fixture_and_fingerprint_are_frozen(self) -> None:
        self.assertEqual(FIXTURE.read_bytes(), self.raw)
        self.assertEqual(
            fingerprint(self.identity),
            "sha256:eshkol-byte-tokenizer-v1:f4c24c4680961301963fb01723297c7a841bfa54241729d24e689401750221b8",
        )

    def test_parse_reserialize_and_repeat_determinism(self) -> None:
        for _ in range(20):
            parsed = decode_artifact(self.raw)
            self.assertEqual(parsed.identity, self.identity)
            self.assertEqual(encode_artifact(parsed.identity), self.raw)
            self.assertEqual(parsed.fingerprint, fingerprint(self.identity))

    def test_construction_order_does_not_change_bytes(self) -> None:
        reversed_identity = Identity("raw", tuple(reversed(self.identity.specials)), self.identity.prefix, self.identity.suffix)
        self.assertEqual(encode_artifact(reversed_identity), self.raw)

    def test_higher_minor_inert_optional_field_is_canonical(self) -> None:
        raw = encode_artifact(self.identity, minor=1, optional=(("comment", b"00ff"),))
        parsed = decode_artifact(raw)
        self.assertEqual(parsed.minor, 1)
        self.assertEqual(parsed.optional_fields, (("comment", b"00ff"),))
        self.assertNotEqual(parsed.fingerprint, fingerprint(self.identity))

    def test_unknown_required_feature_is_unsupported(self) -> None:
        raw = encode_artifact(self.identity, minor=1, required=("new-semantics",))
        with self.assertRaisesRegex(FormatError, "unsupported"):
            decode_artifact(raw)

    def test_version_1_0_rejects_optional_fields(self) -> None:
        raw = encode_artifact(self.identity, minor=0, optional=(("comment", b"00"),))
        with self.assertRaisesRegex(FormatError, "corrupt-data"):
            decode_artifact(raw)

    def test_policy_limits_use_minimum_of_policy_and_hard_limit(self) -> None:
        decode_artifact(self.raw, max_file=len(self.raw), max_metadata=MAX_FILE)
        with self.assertRaises(FormatError):
            decode_artifact(self.raw, max_file=len(self.raw) - 1)
        payload_size = len(self.raw.split(b"payload-bytes\t", 1)[1].split(b"\n", 1)[1].split(b"checksum-algorithm", 1)[0])
        decode_artifact(self.raw, max_metadata=payload_size)
        with self.assertRaises(FormatError):
            decode_artifact(self.raw, max_metadata=payload_size - 1)

    def test_exact_special_and_insertion_count_limits_round_trip(self) -> None:
        specials = tuple(Special(256 + index, f"s{index}", "omit") for index in range(4096))
        identity = Identity("raw", specials, (256,) * 4096, (4351,) * 4096)
        raw = encode_artifact(identity)
        self.assertLessEqual(len(raw), MAX_FILE)
        self.assertEqual(decode_artifact(raw).identity, identity)

    def test_exact_name_and_optional_value_limits(self) -> None:
        name = "a" + "x" * 63
        identity = Identity("raw", (Special(256, name, "omit"),))
        self.assertEqual(decode_artifact(encode_artifact(identity)).identity, identity)
        higher = encode_artifact(identity, minor=1, optional=(("opaque", b"ab" * 64),))
        self.assertEqual(decode_artifact(higher).optional_fields, (("opaque", b"ab" * 64),))
        with self.assertRaises(FormatError):
            encode_artifact(Identity("raw", (Special(256, name + "x", "omit"),)))
        with self.assertRaises(FormatError):
            encode_artifact(identity, minor=1, optional=(("opaque", b"ab" * 65),))

    def test_corruption_truncation_trailing_nonascii_crlf_bom(self) -> None:
        bad = [
            b"",
            self.raw[:-1],
            self.raw + b"\n",
            self.raw + b"x",
            b"\xef\xbb\xbf" + self.raw,
            self.raw.replace(b"\n", b"\r\n", 1),
            self.raw.replace(b"raw", b"r\xffw", 1),
            self.raw.replace(b"kind\tbyte", b"kind\tbytf", 1),
        ]
        for raw in bad:
            with self.assertRaises(FormatError):
                decode_artifact(raw)

    def test_every_truncation_is_rejected(self) -> None:
        for length in range(len(self.raw)):
            with self.assertRaises(FormatError):
                decode_artifact(self.raw[:length])

    def test_line_cap_is_exactly_256(self) -> None:
        mutated = self.raw.replace(b"normalization\tnone", b"normalization\t" + b"x" * 243, 1)
        payload_start = mutated.index(b"kind\tbyte\n")
        checksum_start = mutated.index(b"checksum-algorithm")
        mutated = re_sub_payload_size(mutated, checksum_start - payload_start)
        with self.assertRaisesRegex(FormatError, "overlong line"):
            decode_artifact(resign(mutated))

    def test_version_feature_reserved_and_algorithm_categories(self) -> None:
        cases = (
            (resign(self.raw.replace(b"version\t1\t0", b"version\t2\t0", 1)), "version-mismatch"),
            (resign(self.raw.replace(b"version\t1\t0", b"version\t01\t0", 1)), "corrupt-data"),
            (resign(self.raw.replace(b"checksum-algorithm\tsha256", b"checksum-algorithm\tsha512", 1)), "unsupported"),
            (resign(self.raw.replace(b"limit-prefix\t4096", b"limit-prefix\t4095", 1)), "corrupt-data"),
        )
        for raw, category in cases:
            with self.assertRaisesRegex(FormatError, category):
                decode_artifact(raw)

    def test_deep_semantic_mutations_with_recomputed_checksum_reject(self) -> None:
        replacements = (
            (b"normalization\tnone", b"normalization\tnfc"),
            (b"byte-ids\t0\t255", b"byte-ids\t1\t256"),
            (b"special\t256\tbos\tomit", b"special\t255\tbos\tomit"),
            (b"special\t257\teos\tomit", b"special\t257\tbos\tomit"),
            (b"special\t258\tinvalid\terror", b"special\t258\tinvalid\tomitx"),
            (b"prefix\t0\t256", b"prefix\t1\t256"),
            (b"suffix\t0\t257", b"suffix\t0\t259"),
        )
        for old, new in replacements:
            mutated = self.raw.replace(old, new, 1)
            payload_start = mutated.index(b"kind\tbyte\n")
            checksum_start = mutated.index(b"checksum-algorithm")
            observed = checksum_start - payload_start
            mutated = re_sub_payload_size(mutated, observed)
            with self.assertRaises(FormatError):
                decode_artifact(resign(mutated))

        # Changing an insertion to another configured `omit` special is a
        # valid semantic change when the checksum and payload length are
        # recomputed.  It must not be confused with an overlap/collision.
        changed = self.raw.replace(b"prefix\t0\t256", b"prefix\t0\t257", 1)
        payload_start = changed.index(b"kind\tbyte\n")
        checksum_start = changed.index(b"checksum-algorithm")
        changed = re_sub_payload_size(changed, checksum_start - payload_start)
        self.assertEqual(decode_artifact(resign(changed)).identity.prefix, (257,))

    def test_seeded_integrity_mutations_all_reject(self) -> None:
        rng = random.Random(0x5431)
        for _ in range(256):
            raw = bytearray(self.raw)
            index = rng.randrange(0, len(raw) - 66)
            raw[index] ^= 1 << rng.randrange(8)
            with self.assertRaises(FormatError):
                decode_artifact(bytes(raw))

    def test_fresh_process_locale_timezone_hostile_environment(self) -> None:
        code = (
            "from tests.t1.reference import *; "
            "x=Identity('raw',(Special(256,'bos','omit'),Special(257,'eos','omit'),"
            "Special(258,'invalid','error')),(256,),(257,)); "
            "import hashlib; print(hashlib.sha256(encode_artifact(x)).hexdigest()); print(fingerprint(x))"
        )
        outputs = []
        for locale_name, timezone in (("C", "UTC"), ("C.UTF-8", "Pacific/Honolulu")):
            env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(Path.cwd()), "LC_ALL": locale_name, "TZ": timezone, "HOME": "/definitely/not-used", "PYTHONDONTWRITEBYTECODE": "1"}
            outputs.append(subprocess.check_output([sys.executable, "-c", code], env=env))
        self.assertEqual(outputs[0], outputs[1])


def re_sub_payload_size(raw: bytes, observed: int) -> bytes:
    start = raw.index(b"payload-bytes\t")
    end = raw.index(b"\n", start)
    return raw[:start] + f"payload-bytes\t{observed}".encode() + raw[end:]


if __name__ == "__main__":
    unittest.main()
