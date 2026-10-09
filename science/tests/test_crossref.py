from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx

from medai_science.sources.crossref import CrossrefClient, CrossrefMetadataCache, build_retention_payload, build_unresolved_reference_query, normalize_work, select_retained_reference


def _work(doi: str = "10.1234/example.1") -> dict:
    return {
        "DOI": doi,
        "title": ["A biomedical research article"],
        "author": [{"given": "Ada", "family": "Researcher", "ORCID": "https://orcid.org/0000-0000-0000-0000"}],
        "container-title": ["Journal of Examples"],
        "publisher": "Example publisher",
        "type": "journal-article",
        "published-print": {"date-parts": [[2024, 3, 2]]},
        "reference": [{"key": "ref-1", "DOI": "10.9999/cited", "article-title": "Cited work", "year": "2020", "unstructured": "Citation metadata only"}],
        "references-count": 1,
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/", "start": {"date-parts": [[2024, 3, 2]]}}],
    }


class CrossrefTests(unittest.TestCase):
    def test_unresolved_reference_query_uses_only_bounded_bibliographic_fields(self):
        self.assertEqual(
            build_unresolved_reference_query({
                "article-title": "A sufficiently descriptive paper title",
                "author": "Example Author", "year": "2020", "journal-title": "Research Journal",
                "unstructured": "must never be queried",
            }),
            {"query": "A sufficiently descriptive paper title Example Author 2020 Research Journal", "title": "A sufficiently descriptive paper title"},
        )
        for invalid in (
            {}, {"article-title": "short"}, {"article-title": "This is a valid title", "year": True},
            {"article-title": "This is a valid title", "year": "20x0"},
            {"article-title": "This is a valid title", "author": "x" * 129},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_unresolved_reference_query(invalid)

    def test_select_retained_reference_binds_session_artifact_record_index_and_hashes(self):
        work = normalize_work(_work())
        work["references"] = [{"article-title": "A sufficiently descriptive unresolved citation", "author": "Example Author", "year": "2020", "reference_index": 7}]
        payload = build_retention_payload([work])
        artifact = {
            "id": "crossref-artifact", "session_id": "RES-1", "kind": "crossref_citation_metadata",
            "source": "Crossref REST API", "payload": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        }
        artifact["sha256"] = hashlib.sha256(artifact["payload"].encode("utf-8")).hexdigest()
        selected = select_retained_reference(artifact, session_id="RES-1", artifact_id="crossref-artifact", citing_doi=work["doi"], reference_index=7)
        self.assertEqual(selected["reference"]["article-title"], "A sufficiently descriptive unresolved citation")
        self.assertEqual(selected["source_artifact_sha256"], artifact["sha256"])
        self.assertIsNone(selected["source_cited_doi"])
        for kwargs in (
            {"session_id": "RES-2", "artifact_id": "crossref-artifact", "citing_doi": work["doi"], "reference_index": 7},
            {"session_id": "RES-1", "artifact_id": "crossref-artifact", "citing_doi": work["doi"], "reference_index": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_retained_reference(artifact, **kwargs)
        tampered = {**artifact, "payload": artifact["payload"].replace("unresolved citation", "altered citation")}
        tampered["sha256"] = hashlib.sha256(tampered["payload"].encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "records do not match"):
            select_retained_reference(tampered, session_id="RES-1", artifact_id="crossref-artifact", citing_doi=work["doi"], reference_index=7)

    def test_doi_lookup_normalizes_bibliography_and_citation_edges_without_evidence_polarity(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("/works/10.1234/example.1", request.url.path)
            self.assertEqual(request.url.params.get("mailto"), "research@example.org")
            return httpx.Response(200, json={"status": "ok", "message": _work()})

        result = asyncio.run(CrossrefClient(mailto="research@example.org", transport=httpx.MockTransport(handler)).get_work("https://doi.org/10.1234/example.1"))
        self.assertEqual(result["doi"], "10.1234/example.1")
        self.assertEqual(result["published_date"], "2024-03-02")
        self.assertEqual(result["references"][0]["DOI"], "10.9999/cited")
        self.assertNotIn("unstructured", result["references"][0])
        self.assertNotIn("polarity", result["references"][0])
        self.assertIn("does not establish support", result["reference_semantics"])
        self.assertEqual(len(result["metadata_sha256"]), 64)
        self.assertNotIn("abstract", result)

    def test_search_is_bounded_and_uses_polite_pool_contact(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params.get("query.bibliographic"), "APOE microglia")
            self.assertEqual(request.url.params.get("rows"), "2")
            self.assertEqual(request.url.params.get("mailto"), "research@example.org")
            return httpx.Response(200, json={"message": {"items": [_work(), _work("10.1234/example.2"), _work("10.1234/example.3")]}})

        result = asyncio.run(CrossrefClient(mailto="research@example.org", transport=httpx.MockTransport(handler)).search_bibliographic("APOE microglia", limit=2))
        self.assertEqual(len(result), 2)

    def test_rejects_patient_identifying_search_and_doi_response_mismatch(self):
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            asyncio.run(CrossrefClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))).search_bibliographic("patient id: ABC1234"))

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": _work("10.9999/wrong")})

        with self.assertRaisesRegex(ValueError, "does not match"):
            asyncio.run(CrossrefClient(transport=httpx.MockTransport(handler)).get_work("10.1234/example.1"))

    def test_rejects_duplicate_json_keys_and_nonstandard_numeric_constants(self):
        for body in (
            b'{"message":{"DOI":"10.1234/example.1","DOI":"10.9999/other"}}',
            b'{"message":{"DOI":"10.1234/example.1","score":NaN}}',
            b'{"message":{"DOI":"10.1234/example.1","score":Infinity}}',
        ):
            with self.subTest(body=body):
                transport = httpx.MockTransport(lambda _: httpx.Response(200, content=body))
                with self.assertRaisesRegex(ValueError, "invalid JSON"):
                    asyncio.run(CrossrefClient(transport=transport).get_work("10.1234/example.1"))

    def test_reference_count_is_capped_and_doi_validation_is_strict(self):
        work = _work()
        work["reference"] = [{"DOI": f"10.9999/ref{i}"} for i in range(250)]
        normalized = normalize_work(work)
        self.assertEqual(len(normalized["references"]), 200)
        self.assertTrue(normalized["references_truncated"])
        with self.assertRaises(ValueError):
            CrossrefClient.validate_doi("patient id: 123456")

    def test_invalid_crossref_date_parts_are_not_misreported_as_publication_dates(self):
        work = _work()
        work["published-print"] = {"date-parts": [[True, 3, 2]]}
        work["published-online"] = {"date-parts": [[2024, 2, 30]]}
        work["issued"] = {"date-parts": [[2023, 9]]}
        self.assertEqual(normalize_work(work)["published_date"], "2023-09")

        work["issued"] = {"date-parts": [[2023, 13, 1]]}
        self.assertIsNone(normalize_work(work)["published_date"])

    def test_artifact_retention_strips_free_text_and_enforces_medai_payload_limit(self):
        records = []
        for index in range(5):
            row = normalize_work(_work(f"10.1234/example.{index}"))
            row["references"] = [{"DOI": "10.9999/cited", "article-title": "title " * 80, "unstructured": "sensitive copyright text " * 100, "reference_index": j} for j in range(150)]
            records.append(row)
        payload = build_retention_payload(records)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.assertLessEqual(len(encoded), 240_000)
        self.assertNotIn("sensitive copyright text", encoded.decode())
        self.assertEqual(len(payload["complete_result_sha256"]), 64)
        canonical_records = json.dumps(payload["records"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(payload["retained_records_sha256"], hashlib.sha256(canonical_records).hexdigest())
        for record in payload["records"]:
            self.assertEqual(record["references_retained_count"], len(record["references"]))
            self.assertEqual(record["references_retention_truncated"], record["references_retained_count"] < 150)

    def test_metadata_cache_reuses_only_hash_keyed_normalized_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crossref-cache.sqlite"
            requests = 0

            def handler(_: httpx.Request) -> httpx.Response:
                nonlocal requests
                requests += 1
                return httpx.Response(200, json={"message": {"items": [_work()]}})

            client = CrossrefClient(transport=httpx.MockTransport(handler), cache_path=str(path))
            first = asyncio.run(client.search_bibliographic("APOE microglia", limit=1))
            second = asyncio.run(client.search_bibliographic("APOE microglia", limit=1))
            self.assertEqual(requests, 1)
            self.assertEqual(first, second)
            self.assertEqual(first[0]["license_metadata"][0]["URL"], "https://creativecommons.org/licenses/by/4.0/")
            with sqlite3.connect(path) as connection:
                rows = connection.execute("SELECT cache_key,payload_json,payload_sha256 FROM crossref_metadata_cache").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertNotIn("APOE microglia", rows[0][0] + rows[0][1])
            self.assertNotIn("Citation metadata only", rows[0][1])
            self.assertEqual(hashlib.sha256(rows[0][1].encode()).hexdigest(), rows[0][2])

    def test_metadata_cache_refuses_unlicensed_free_text_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = CrossrefMetadataCache(Path(folder) / "crossref-cache.sqlite")
            key = cache.key("doi", {"doi": "10.1234/example.1"})
            record = normalize_work(_work())
            record["abstract"] = "full text must not enter the metadata cache"
            with self.assertRaisesRegex(ValueError, "metadata-only"):
                cache.put(key, record, now=100)

            record = normalize_work(_work())
            record["references"][0]["unstructured"] = "copied citation prose"
            with self.assertRaisesRegex(ValueError, "metadata-only"):
                cache.put(key, record, now=100)
            self.assertIsNone(cache.get(key, now=101))

    def test_metadata_cache_discards_records_with_added_free_text_on_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crossref-cache.sqlite"
            cache = CrossrefMetadataCache(path)
            key = cache.key("doi", {"doi": "10.1234/example.1"})
            record = normalize_work(_work())
            self.assertTrue(cache.put(key, record, now=100))
            record["abstract"] = "must be rejected even with a matching cache digest"
            encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "UPDATE crossref_metadata_cache SET payload_json=?,payload_sha256=? WHERE cache_key=?",
                    (encoded, hashlib.sha256(encoded.encode()).hexdigest(), key),
                )
            self.assertIsNone(cache.get(key, now=101))
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM crossref_metadata_cache").fetchone()[0], 0)

    def test_metadata_cache_expiry_and_corrupt_digest_fail_closed_to_provider(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crossref-cache.sqlite"
            cache = CrossrefMetadataCache(path, ttl_seconds=60)
            key = cache.key("doi", {"doi": "10.1234/example.1"})
            cached = normalize_work(_work(), retrieved_at="2026-01-01T00:00:00Z")
            self.assertTrue(cache.put(key, cached, now=100))
            self.assertIsNone(cache.get(key, now=160))
            self.assertTrue(cache.put(key, cached, now=100))
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE crossref_metadata_cache SET payload_sha256=? WHERE cache_key=?", ("0" * 64, key))
            self.assertIsNone(cache.get(key, now=101))
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM crossref_metadata_cache").fetchone()[0], 0)
            self.assertTrue(cache.put(key, cached, now=100))
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE crossref_metadata_cache SET stored_at=102 WHERE cache_key=?", (key,))
            self.assertIsNone(cache.get(key, now=101))
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM crossref_metadata_cache").fetchone()[0], 0)

    def test_metadata_cache_rejects_expiry_beyond_configured_ttl(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crossref-cache.sqlite"
            cache = CrossrefMetadataCache(path, ttl_seconds=60)
            key = cache.key("doi", {"doi": "10.1234/example.1"})
            cached = normalize_work(_work(), retrieved_at="2026-01-01T00:00:00Z")
            self.assertTrue(cache.put(key, cached, now=100))
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE crossref_metadata_cache SET expires_at=10000 WHERE cache_key=?", (key,))
            self.assertIsNone(cache.get(key, now=101))
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM crossref_metadata_cache").fetchone()[0], 0)

    def test_metadata_cache_clock_arguments_require_nonnegative_integers(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = CrossrefMetadataCache(Path(folder) / "crossref-cache.sqlite")
            key = cache.key("doi", {"doi": "10.1234/example.1"})
            row = normalize_work(_work(), retrieved_at="2026-01-01T00:00:00Z")
            for now in (True, 1.5, -1):
                with self.subTest(now=now), self.assertRaisesRegex(ValueError, "timestamp"):
                    cache.put(key, row, now=now)
                with self.subTest(read_now=now), self.assertRaisesRegex(ValueError, "timestamp"):
                    cache.get(key, now=now)

    def test_any_cache_operation_prunes_corrupt_timestamps_for_other_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = CrossrefMetadataCache(Path(folder) / "crossref-cache.sqlite", ttl_seconds=60)
            valid_key = cache.key("doi", {"doi": "10.1234/valid.1"})
            corrupt_key = cache.key("doi", {"doi": "10.1234/corrupt.1"})
            valid = normalize_work(_work("10.1234/valid.1"), retrieved_at="2026-01-01T00:00:00Z")
            corrupt = normalize_work(_work("10.1234/corrupt.1"), retrieved_at="2026-01-01T00:00:00Z")
            self.assertTrue(cache.put(valid_key, valid, now=100))
            self.assertTrue(cache.put(corrupt_key, corrupt, now=100))
            with sqlite3.connect(cache.path) as connection:
                connection.execute("UPDATE crossref_metadata_cache SET expires_at=10000 WHERE cache_key=?", (corrupt_key,))
            self.assertIsNotNone(cache.get(valid_key, now=101))
            with sqlite3.connect(cache.path) as connection:
                self.assertEqual(connection.execute("SELECT cache_key FROM crossref_metadata_cache").fetchall(), [(valid_key,)])

    def test_metadata_cache_bounds_payload_and_rejects_symlink_database(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = CrossrefMetadataCache(Path(folder) / "cache.sqlite")
            key = cache.key("doi", {"doi": "10.1234/example.1"})
            self.assertFalse(cache.put(key, {"text": "x" * (256 * 1024 + 1)}))
            real_path = Path(folder) / "real.sqlite"
            CrossrefMetadataCache(real_path)
            symlink = Path(folder) / "alias.sqlite"
            symlink.symlink_to(real_path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                CrossrefMetadataCache(symlink)

    def test_metadata_cache_evicts_oldest_entry_and_uses_private_file_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.sqlite"
            cache = CrossrefMetadataCache(path)
            rows = [normalize_work(_work(f"10.1234/cache-{index}"), retrieved_at="2026-01-01T00:00:00Z") for index in range(3)]
            keys = [cache.key("doi", {"doi": row["doi"]}) for row in rows]
            with patch("medai_science.sources.crossref._CACHE_MAX_ENTRIES", 2):
                for index, (key, row) in enumerate(zip(keys, rows)):
                    self.assertTrue(cache.put(key, row, now=100 + index))
            self.assertIsNone(cache.get(keys[0], now=102))
            self.assertIsNotNone(cache.get(keys[2], now=102))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
