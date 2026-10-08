from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medai_science.sources import pubmed_cache
from medai_science.sources.pubmed_cache import PubMedSearchCache, search_key


class PubMedSearchCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "search-cache.sqlite"
        self.cache = PubMedSearchCache(self.path, ttl_seconds=60)

    def test_cache_stores_only_query_digest_and_bounded_identifier_coverage(self):
        query = 'APOE[Title/Abstract] AND "Alzheimer Disease"[MeSH Terms]'
        pmids = ["123", "456"]
        coverage = {"total_hits": 7, "requested_limit": 2}
        self.cache.put(query, 2, pmids, coverage, now=100)

        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT cache_key,pmids_json,total_hits,requested_limit FROM pubmed_search_cache").fetchall()
        self.assertEqual(rows, [(search_key(query, 2), '["123","456"]', 7, 2)])
        self.assertNotIn(query, self.path.read_bytes().decode("latin-1"))

        found = self.cache.get(query, 2, now=159)
        self.assertEqual(found[0], pmids)
        self.assertEqual(found[1]["total_hits"], 7)
        self.assertTrue(found[1]["search_truncated"])
        self.assertEqual(found[1]["cache"], "hit")
        self.assertIsNone(self.cache.get(query, 2, now=160))

    def test_invalid_cache_rows_are_removed_and_limits_are_enforced(self):
        query = "TREM2"
        self.cache.put(query, 2, ["1"], {"total_hits": 1}, now=100)
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE pubmed_search_cache SET pmids_json='[1]'")
        self.assertIsNone(self.cache.get(query, 2, now=101))
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM pubmed_search_cache").fetchone()[0], 0)
        with self.assertRaisesRegex(ValueError, "validated bounded"):
            self.cache.put(query, 1, ["1", "2"], {"total_hits": 2})
        with self.assertRaisesRegex(ValueError, "TTL"):
            PubMedSearchCache(self.path, ttl_seconds=3601)

    def test_read_prunes_expired_entries_for_other_queries(self):
        self.cache.put("expired query", 1, ["1"], {"total_hits": 1}, now=100)
        self.cache.put("current query", 1, ["2"], {"total_hits": 1}, now=150)
        self.assertEqual(self.cache.get("current query", 1, now=160)[0], ["2"])
        with sqlite3.connect(self.path) as connection:
            retained = connection.execute("SELECT cache_key FROM pubmed_search_cache").fetchall()
        self.assertEqual(retained, [(search_key("current query", 1),)])

    def test_symlink_cache_path_is_rejected(self):
        target = Path(self.directory.name) / "target.sqlite"
        target.touch()
        link = Path(self.directory.name) / "link.sqlite"
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            PubMedSearchCache(link)

    def test_cache_prunes_oldest_entries_to_a_fixed_retention_limit(self):
        with patch.object(pubmed_cache, "_MAX_CACHE_ENTRIES", 3):
            for number in range(5):
                query = f"gene{number}"
                self.cache.put(query, 1, [str(number + 1)], {"total_hits": 1}, now=100 + number)
            with sqlite3.connect(self.path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM pubmed_search_cache").fetchone()[0]
            self.assertEqual(count, 3)
            self.assertIsNone(self.cache.get("gene0", 1, now=104))
            self.assertIsNone(self.cache.get("gene1", 1, now=104))
            self.assertEqual(self.cache.get("gene4", 1, now=104)[0], ["5"])


if __name__ == "__main__":
    unittest.main()
