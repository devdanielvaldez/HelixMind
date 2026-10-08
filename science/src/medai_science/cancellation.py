"""Durable session cancellation for in-flight scientific source operations."""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, Callable


class ResearchSessionCancelled(RuntimeError):
    """A persisted session cancellation interrupted a running source operation."""


async def run_cancel_aware(
    session_id: str,
    operation: Awaitable[Any],
    *,
    cancellation_check: Callable[[str], Awaitable[bool]],
    poll_seconds: float = 1.0,
) -> Any:
    """Cancel a provider coroutine when the durable session flag is set."""
    if poll_seconds <= 0:
        raise ValueError("cancellation poll interval must be positive")
    try:
        already_cancelled = await cancellation_check(session_id)
    except asyncio.CancelledError:
        if asyncio.iscoroutine(operation):
            operation.close()
        elif isinstance(operation, asyncio.Future):
            operation.cancel()
        raise
    except Exception:
        already_cancelled = False
    if already_cancelled:
        if asyncio.iscoroutine(operation):
            operation.close()
        elif isinstance(operation, asyncio.Future):
            operation.cancel()
        raise ResearchSessionCancelled(
            "research session cancellation requested; provider operation was not started"
        )
    source_task = asyncio.ensure_future(operation)

    async def watch_cancellation() -> None:
        while True:
            await asyncio.sleep(poll_seconds)
            try:
                if await cancellation_check(session_id):
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                # A transient control-plane read failure must not fabricate cancellation.
                continue

    watcher = asyncio.create_task(watch_cancellation())
    try:
        done, _ = await asyncio.wait({source_task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            source_task.cancel()
            await asyncio.gather(source_task, return_exceptions=True)
            raise ResearchSessionCancelled(
                "research session cancellation requested; in-flight provider operation cancelled"
            )
        return await source_task
    except BaseException:
        if not source_task.done():
            source_task.cancel()
            await asyncio.gather(source_task, return_exceptions=True)
        raise
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


async def run_cancel_aware_thread(
    session_id: str,
    function: Callable[..., Any],
    *args: Any,
    cancellation_check: Callable[[str], Awaitable[bool]],
    poll_seconds: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Run a bounded blocking operation in a thread and signal cooperative cancellation."""
    if poll_seconds <= 0:
        raise ValueError("cancellation poll interval must be positive")
    try:
        already_cancelled = await cancellation_check(session_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        already_cancelled = False
    if already_cancelled:
        raise ResearchSessionCancelled("research session cancellation requested; sandbox operation was not started")

    cancel_event = threading.Event()
    operation = asyncio.create_task(asyncio.to_thread(function, *args, cancel_event=cancel_event, **kwargs))
    try:
        while True:
            try:
                return await asyncio.wait_for(asyncio.shield(operation), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
            if operation.done():
                return await operation
            try:
                cancelled = await cancellation_check(session_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A temporary API read failure cannot fabricate cancellation.
                continue
            if cancelled:
                cancel_event.set()
                await asyncio.gather(operation, return_exceptions=True)
                raise ResearchSessionCancelled("research session cancellation requested; sandbox container stopped")
    except asyncio.CancelledError:
        cancel_event.set()
        await asyncio.shield(asyncio.gather(operation, return_exceptions=True))
        raise
