from __future__ import annotations

import asyncio
import unittest

import httpx

from medai_science.cancellation import ResearchSessionCancelled, run_cancel_aware, run_cancel_aware_thread


class ResearchOperationCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_worker_receives_cancellation_signal_and_is_joined(self):
        checks = 0
        stopped = False

        async def check(_session_id: str) -> bool:
            nonlocal checks
            checks += 1
            return checks >= 2

        def worker(*, cancel_event):
            nonlocal stopped
            while not cancel_event.wait(0.001):
                pass
            stopped = True
            return "cancelled"

        with self.assertRaisesRegex(ResearchSessionCancelled, "container stopped"):
            await run_cancel_aware_thread("RES-test", worker, cancellation_check=check, poll_seconds=0.005)
        self.assertTrue(stopped)

    async def test_preexisting_cancellation_does_not_start_blocking_worker(self):
        started = False

        async def check(_session_id: str) -> bool:
            return True

        def worker(*, cancel_event):
            nonlocal started
            started = True
            return "unexpected"

        with self.assertRaisesRegex(ResearchSessionCancelled, "was not started"):
            await run_cancel_aware_thread("RES-test", worker, cancellation_check=check)
        self.assertFalse(started)

    async def test_preexisting_cancellation_prevents_provider_request_from_starting(self):
        started = False

        async def check(_session_id: str) -> bool:
            return True

        async def provider_request() -> str:
            nonlocal started
            started = True
            return "unexpected"

        with self.assertRaisesRegex(ResearchSessionCancelled, "was not started"):
            await run_cancel_aware("RES-test", provider_request(), cancellation_check=check)
        self.assertFalse(started)

    async def test_persisted_session_cancellation_cancels_in_flight_operation(self):
        request_cancelled = asyncio.Event()
        checks = 0

        async def check(session_id: str) -> bool:
            nonlocal checks
            self.assertEqual(session_id, "RES-test")
            checks += 1
            return checks >= 2

        class HangingProvider(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                try:
                    await asyncio.Event().wait()
                    raise AssertionError("provider request should be cancelled")
                finally:
                    request_cancelled.set()

        async def provider_request() -> httpx.Response:
            async with httpx.AsyncClient(transport=HangingProvider()) as client:
                return await client.get("https://provider.example.test/records")

        with self.assertRaisesRegex(ResearchSessionCancelled, "in-flight provider operation cancelled"):
            await run_cancel_aware("RES-test", provider_request(), cancellation_check=check, poll_seconds=0.001)
        self.assertTrue(request_cancelled.is_set())
        self.assertGreaterEqual(checks, 2)

    async def test_finished_operation_returns_without_waiting_for_cancellation_poll(self):
        async def check(_session_id: str) -> bool:
            return False

        result = await run_cancel_aware(
            "RES-test", asyncio.sleep(0, result={"records": []}), cancellation_check=check, poll_seconds=1
        )
        self.assertEqual(result, {"records": []})


if __name__ == "__main__":
    unittest.main()
