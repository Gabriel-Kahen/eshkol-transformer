from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.t1.generate_fixture import fixture_identity
from tests.t1.tokenizer_format import (
    MAX_ARTIFACT_BYTES,
    MAX_INSERTIONS,
    MAX_SPECIALS,
    SpecialToken,
    TokenizerFormatError,
    TokenizerIdentity,
    decode_artifact,
    encode_artifact,
    fingerprint,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "byte_tokenizer_v1.tsv"
CHECKSUM_DOMAIN = b"eshkol-byte-tokenizer-checksum-v1\n"
IDENTITY_DOMAIN = b"sha256:eshkol-byte-tokenizer-v1\n"


def _resign(raw: bytes) -> bytes:
    checksum = raw.rfind(b"checksum\t")
    if checksum < 0:
        raise AssertionError("test mutation removed checksum record")
    body = raw[:checksum]
    digest = hashlib.sha256(CHECKSUM_DOMAIN + body).hexdigest().encode("ascii")
    return body + b"checksum\t" + digest + b"\n"


def _replace_payload_line(raw: bytes, old: bytes, new: bytes) -> bytes:
    if raw.count(old) != 1:
        raise AssertionError(f"test mutation did not find unique line {old!r}")
    changed = raw.replace(old, new, 1)
    header_end = changed.index(b"\n", changed.index(b"payload-bytes\t")) + 1
    algorithm = changed.index(b"checksum-algorithm\t", header_end)
    payload_size = algorithm - header_end
    payload_line_start = changed.index(b"payload-bytes\t")
    payload_line_end = changed.index(b"\n", payload_line_start) + 1
    changed = (
        changed[:payload_line_start]
        + f"payload-bytes\t{payload_size}\n".encode("ascii")
        + changed[payload_line_end:]
    )
    return _resign(changed)


class CanonicalTokenizerFormatTests(unittest.TestCase):
    def test_frozen_fixture_round_trips_and_has_stable_fingerprint(self) -> None:
        raw = FIXTURE.read_bytes()
        parsed = decode_artifact(raw)
        self.assertEqual(parsed.identity, fixture_identity())
        self.assertEqual(encode_artifact(parsed.identity), raw)
        self.assertEqual(parsed.fingerprint, fingerprint(fixture_identity()))
        self.assertEqual(
            parsed.fingerprint,
            "sha256:eshkol-byte-tokenizer-v1:"
            "f4c24c4680961301963fb01723297c7a841bfa54241729d24e689401750221b8",
        )

    def test_fresh_process_generations_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated = []
            for name in ("first.tsv", "second.tsv"):
                output = Path(temporary) / name
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "tests.t1.generate_fixture",
                        "--output",
                        str(output),
                    ],
                    cwd=HERE.parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                generated.append(output.read_bytes())
        self.assertEqual(generated[0], generated[1])
        self.assertEqual(generated[0], FIXTURE.read_bytes())

    def test_special_construction_order_is_canonicalized(self) -> None:
        expected = fixture_identity()
        reversed_specials = TokenizerIdentity(
            expected.utf8_policy,
            tuple(reversed(expected.specials)),
            expected.prefix,
            expected.suffix,
        )
        self.assertEqual(encode_artifact(reversed_specials), encode_artifact(expected))
        self.assertEqual(fingerprint(reversed_specials), fingerprint(expected))

    def test_empty_and_strict_identities_are_expressible(self) -> None:
        for policy in ("raw", "strict"):
            with self.subTest(policy=policy):
                identity = TokenizerIdentity(policy)
                parsed = decode_artifact(encode_artifact(identity))
                self.assertEqual(parsed.identity, identity)
                self.assertRegex(
                    parsed.fingerprint,
                    r"\Asha256:eshkol-byte-tokenizer-v1:[0-9a-f]{64}\Z",
                )

    def test_identity_changes_affect_fingerprint(self) -> None:
        source = fixture_identity()
        variants = (
            TokenizerIdentity("strict", source.specials, source.prefix, source.suffix),
            TokenizerIdentity("raw", source.specials, (), source.suffix),
            TokenizerIdentity("raw", source.specials, source.prefix, (257, 257)),
            TokenizerIdentity(
                "raw",
                source.specials[:-1] + (SpecialToken(258, "replacement", "error"),),
                source.prefix,
                source.suffix,
            ),
        )
        original = fingerprint(source)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(fingerprint(variant), original)

    def test_checksum_payload_and_declared_size_corruption_are_rejected(self) -> None:
        raw = FIXTURE.read_bytes()
        corruptions = (
            raw.replace(b"normalization\tnone", b"normalization\tnope", 1),
            raw[:-66] + b"0" * 64 + b"\n",
            raw.replace(b"payload-bytes\t", b"payload-bytes\t0", 1),
        )
        for corrupted in corruptions:
            with self.subTest(corrupted=corrupted[:50]):
                with self.assertRaises(TokenizerFormatError):
                    decode_artifact(corrupted)

    def test_version_and_required_feature_categories_are_explicit(self) -> None:
        raw = FIXTURE.read_bytes()
        version = _resign(raw.replace(b"version\t1\t0\n", b"version\t2\t0\n", 1))
        with self.assertRaises(TokenizerFormatError) as caught:
            decode_artifact(version)
        self.assertEqual(caught.exception.category, "version-mismatch")

        features = _resign(raw.replace(
            b"required-features\t0\noptional-fields\t0\n",
            b"required-features\t1\nrequired-feature\tfuture\n"
            b"optional-fields\t0\n",
            1,
        ))
        with self.assertRaises(TokenizerFormatError) as caught:
            decode_artifact(features)
        self.assertEqual(caught.exception.category, "unsupported")

    def test_higher_minor_with_sorted_optional_fields_is_accepted_inertly(self) -> None:
        raw = FIXTURE.read_bytes()
        higher = raw.replace(b"version\t1\t0\n", b"version\t1\t7\n", 1)
        higher = higher.replace(
            b"optional-fields\t0\n",
            b"optional-fields\t2\n"
            b"optional-field\talpha\t00\n"
            b"optional-field\tzed\tdeadbeef\n",
            1,
        )
        higher = _resign(higher)
        parsed = decode_artifact(higher)
        self.assertEqual(parsed.version_minor, 7)
        self.assertEqual(
            parsed.optional_fields,
            (("alpha", b"00"), ("zed", b"deadbeef")),
        )
        self.assertEqual(parsed.identity, fixture_identity())
        self.assertNotEqual(encode_artifact(parsed.identity), higher)

    def test_feature_and_optional_records_require_canonical_order_and_uniqueness(self) -> None:
        raw = FIXTURE.read_bytes().replace(b"version\t1\t0\n", b"version\t1\t1\n", 1)
        mutations = (
            raw.replace(
                b"required-features\t0\n",
                b"required-features\t2\n"
                b"required-feature\tzeta\nrequired-feature\talpha\n",
                1,
            ),
            raw.replace(
                b"required-features\t0\n",
                b"required-features\t2\n"
                b"required-feature\talpha\nrequired-feature\talpha\n",
                1,
            ),
            raw.replace(
                b"optional-fields\t0\n",
                b"optional-fields\t2\n"
                b"optional-field\tzeta\t00\noptional-field\talpha\t02\n",
                1,
            ),
            raw.replace(
                b"optional-fields\t0\n",
                b"optional-fields\t2\n"
                b"optional-field\talpha\t00\noptional-field\talpha\t02\n",
                1,
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation[:180]):
                with self.assertRaises(TokenizerFormatError) as caught:
                    decode_artifact(_resign(mutation))
                self.assertEqual(caught.exception.category, "corrupt-data")

    def test_optional_values_are_even_bounded_lowercase_hex(self) -> None:
        raw = FIXTURE.read_bytes().replace(b"version\t1\t0\n", b"version\t1\t1\n", 1)
        for value in (b"", b"0", b"AA", b"g0", b"00" * 65):
            with self.subTest(value=value[:20]):
                mutation = raw.replace(
                    b"optional-fields\t0\n",
                    b"optional-fields\t1\noptional-field\talpha\t" + value + b"\n",
                    1,
                )
                with self.assertRaises(TokenizerFormatError):
                    decode_artifact(_resign(mutation))

    def test_declared_limit_records_are_exact_and_in_identity_header(self) -> None:
        raw = FIXTURE.read_bytes()
        limits = (
            (b"limit-file-bytes\t1048576\n", b"limit-file-bytes\t1048575\n"),
            (b"limit-specials\t4096\n", b"limit-specials\t4095\n"),
            (b"limit-name-bytes\t64\n", b"limit-name-bytes\t63\n"),
            (b"limit-header-items\t4096\n", b"limit-header-items\t4095\n"),
            (
                b"limit-optional-value-bytes\t64\n",
                b"limit-optional-value-bytes\t63\n",
            ),
            (b"limit-prefix\t4096\n", b"limit-prefix\t4095\n"),
            (b"limit-suffix\t4096\n", b"limit-suffix\t4095\n"),
        )
        for old, new in limits:
            with self.subTest(limit=old):
                with self.assertRaises(TokenizerFormatError):
                    decode_artifact(_resign(raw.replace(old, new, 1)))

    def test_fingerprint_uses_algorithm_canonicalization_domain_and_full_header(self) -> None:
        raw = FIXTURE.read_bytes()
        parsed = decode_artifact(raw)
        payload_marker = raw.index(b"payload-bytes\t")
        header = raw[:payload_marker]
        expected = hashlib.sha256(
            IDENTITY_DOMAIN + header + parsed.payload
        ).hexdigest()
        old_domain = hashlib.sha256(
            b"eshkol-byte-tokenizer-identity-v1\n" + header + parsed.payload
        ).hexdigest()
        self.assertEqual(
            parsed.fingerprint,
            "sha256:eshkol-byte-tokenizer-v1:" + expected,
        )
        self.assertNotEqual(expected, old_domain)

    def test_source_size_is_checked_before_conversion(self) -> None:
        class OversizeWithoutSafeConversion:
            def __len__(self) -> int:
                return MAX_ARTIFACT_BYTES + 1

            def __bytes__(self) -> bytes:
                raise AssertionError("bytes() must not run for an oversized source")

        with self.assertRaisesRegex(TokenizerFormatError, "artifact exceeds"):
            decode_artifact(OversizeWithoutSafeConversion())  # type: ignore[arg-type]

    def test_noncanonical_container_bytes_are_rejected(self) -> None:
        raw = FIXTURE.read_bytes()
        mutations = (
            b"\xef\xbb\xbf" + raw,
            raw.replace(b"\n", b"\r\n", 1),
            raw + b"\n",
            raw[:-1],
            raw.replace(b"payload-bytes\t", b"payload-bytes\t0", 1),
            raw.replace(b"checksum-algorithm\tsha256", b"checksum-algorithm\tsha512", 1),
            raw.replace(b"checksum\t", b"checksum\tA", 1),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation[:40]):
                with self.assertRaises(TokenizerFormatError):
                    decode_artifact(mutation)

    def test_every_truncation_of_frozen_fixture_is_rejected(self) -> None:
        raw = FIXTURE.read_bytes()
        for length in range(len(raw)):
            with self.subTest(length=length):
                with self.assertRaises(TokenizerFormatError):
                    decode_artifact(raw[:length])

    def test_malformed_payload_records_are_rejected_after_valid_checksum(self) -> None:
        raw = FIXTURE.read_bytes()
        mutations = (
            (b"normalization\tnone\n", b"normalization\tnfc\n"),
            (b"utf8-policy\traw\n", b"utf8-policy\treplace\n"),
            (b"byte-ids\t0\t255\n", b"byte-ids\t1\t256\n"),
            (b"special\t256\tbos\tomit\n", b"special\t0256\tbos\tomit\n"),
            (b"special\t257\teos\tomit\n", b"special\t259\teos\tomit\n"),
            (b"special\t257\teos\tomit\n", b"special\t257\tbos\tomit\n"),
            (b"special\t258\tinvalid\terror\n", b"special\t258\tBad\terror\n"),
            (b"special\t258\tinvalid\terror\n", b"special\t258\tinvalid\temit\n"),
            (b"prefix\t0\t256\n", b"prefix\t1\t256\n"),
            (b"prefix\t0\t256\n", b"prefix\t0\t999\n"),
            (b"prefix\t0\t256\n", b"prefix\t0\t258\n"),
            (b"suffix-count\t1\n", b"suffix-count\t0\n"),
            (b"suffix\t0\t257\n", b"unknown\t0\t257\n"),
        )
        for old, new in mutations:
            with self.subTest(new=new):
                with self.assertRaises(TokenizerFormatError) as caught:
                    decode_artifact(_replace_payload_line(raw, old, new))
                self.assertEqual(caught.exception.category, "corrupt-data")

    def test_builder_rejects_collisions_gaps_bad_references_and_limits(self) -> None:
        bad = (
            TokenizerIdentity("replace"),
            TokenizerIdentity("raw", (SpecialToken(255, "collision", "omit"),)),
            TokenizerIdentity("raw", (SpecialToken(257, "gap", "omit"),)),
            TokenizerIdentity(
                "raw",
                (SpecialToken(256, "same", "omit"), SpecialToken(257, "same", "omit")),
            ),
            TokenizerIdentity("raw", (SpecialToken(256, "Bad", "omit"),)),
            TokenizerIdentity("raw", (SpecialToken(256, "x", "emit"),)),
            TokenizerIdentity("raw", (SpecialToken(256, "x", "omit"),), (257,), ()),
            TokenizerIdentity("raw", (SpecialToken(256, "x", "error"),), (256,), ()),
            TokenizerIdentity(
                "raw",
                tuple(SpecialToken(256 + index, f"s{index}", "omit") for index in range(MAX_SPECIALS + 1)),
            ),
            TokenizerIdentity(
                "raw",
                (SpecialToken(256, "x", "omit"),),
                (256,) * (MAX_INSERTIONS + 1),
                (),
            ),
        )
        for identity in bad:
            with self.subTest(identity=str(identity)[:120]):
                with self.assertRaises(TokenizerFormatError):
                    encode_artifact(identity)

    def test_artifact_and_line_caps_are_enforced(self) -> None:
        with self.assertRaisesRegex(TokenizerFormatError, "exceeds"):
            decode_artifact(b"x" * (MAX_ARTIFACT_BYTES + 1))
        raw = FIXTURE.read_bytes()
        overlong = _replace_payload_line(
            raw,
            b"normalization\tnone\n",
            b"normalization\t" + b"x" * 300 + b"\n",
        )
        with self.assertRaisesRegex(TokenizerFormatError, "line exceeds"):
            decode_artifact(overlong)


if __name__ == "__main__":
    unittest.main()
