from __future__ import annotations

import asyncio
import gzip
import unittest

import httpx

from medai_science.sources.bounded_http import event_loop_lock, read_bounded_response


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes):
        self.chunks = chunks
        self.consumed = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def aclose(self):
        return None


class BoundedHttpTests(unittest.TestCase):
    def test_async_rate_limit_locks_are_lazy_and_event_loop_scoped(self):
        loops = [asyncio.new_event_loop(), asyncio.new_event_loop()]

        async def lock_for(namespace):
            return event_loop_lock(namespace)

        try:
            first = loops[0].run_until_complete(lock_for("fixture"))
            same_loop = loops[0].run_until_complete(lock_for("fixture"))
            other_namespace = loops[0].run_until_complete(lock_for("other"))
            second = loops[1].run_until_complete(lock_for("fixture"))
            self.assertIs(first, same_loop)
            self.assertIsNot(first, other_namespace)
            self.assertIsNot(first, second)
        finally:
            for loop in loops:
                loop.close()

    def test_reads_exact_limit_and_rejects_actual_stream_overflow_without_length(self):
        async def run(stream):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))) as client:
                async with client.stream("GET", "https://example.org/data") as response:
                    return await read_bounded_response(response, 5, "fixture")

        self.assertEqual(asyncio.run(run(ChunkStream(b"ab", b"cde"))), b"abcde")
        with self.assertRaisesRegex(ValueError, "response exceeds"):
            asyncio.run(run(ChunkStream(b"abc", b"def")))

    def test_stops_before_consuming_chunks_after_the_first_limit_excess(self):
        stream = ChunkStream(b"12345", b"oversized", b"unreachable")

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))) as client:
                async with client.stream("GET", "https://example.org/data") as response:
                    return await read_bounded_response(response, 5, "fixture")

        with self.assertRaisesRegex(ValueError, "response exceeds"):
            asyncio.run(run())
        self.assertEqual(stream.consumed, 2)

    def test_caps_decoded_gzip_bytes_even_when_compressed_body_fits(self):
        compressed = gzip.compress(b"A" * 100_000)
        self.assertLess(len(compressed), 1_000)

        async def run():
            stream = ChunkStream(compressed)
            transport = httpx.MockTransport(lambda _: httpx.Response(
                200,
                headers={"content-encoding": "gzip", "content-length": str(len(compressed))},
                stream=stream,
            ))
            async with httpx.AsyncClient(transport=transport) as client:
                async with client.stream("GET", "https://example.org/data") as response:
                    return await read_bounded_response(response, 1_000, "fixture")

        with self.assertRaisesRegex(ValueError, "response exceeds"):
            asyncio.run(run())

    def test_rejects_oversized_content_length_before_reading_body(self):
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"content-length": "100"}, stream=ChunkStream(b"ok")))) as client:
                async with client.stream("GET", "https://example.org/data") as response:
                    return await read_bounded_response(response, 5, "fixture")

        with self.assertRaisesRegex(ValueError, "response exceeds"):
            asyncio.run(run())

    def test_rejects_bad_limits_and_http_errors(self):
        async def read(response, limit):
            return await read_bounded_response(response, limit, "fixture")

        with self.assertRaisesRegex(ValueError, "positive integer"):
            asyncio.run(read(httpx.Response(200, content=b"ok"), 0))
        with self.assertRaises(httpx.HTTPStatusError):
            asyncio.run(read(httpx.Response(503, request=httpx.Request("GET", "https://example.org/data"), content=b"unavailable"), 5))


if __name__ == "__main__":
    unittest.main()
