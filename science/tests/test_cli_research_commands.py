from __future__ import annotations

import unittest
from unittest.mock import Mock

from medai_science.research_cli import cancel_session, get_session_status, list_sessions, validate_session_id, watch_session


class ResearchApiCliTests(unittest.TestCase):
    def test_list_reads_tenant_scoped_sessions_from_api(self):
        request = Mock(return_value={"items": [{"id": "RES-1"}]})
        result = list_sessions(request)
        request.assert_called_once_with("/v1/research", method="GET")
        self.assertEqual(result["items"][0]["id"], "RES-1")

    def test_status_encodes_validated_session_id(self):
        request = Mock(return_value={"terminal": False, "tasks": [], "task_counts": {}})
        result = get_session_status(request, "RES-2026.1")
        request.assert_called_once_with("/v1/research/RES-2026.1/status", method="GET")
        self.assertFalse(result["terminal"])

    def test_cancel_posts_to_session_control_endpoint(self):
        request = Mock(return_value={"status": "cancellation_requested"})
        result = cancel_session(request, "RES-2026.1")
        request.assert_called_once_with("/v1/research/RES-2026.1/control", method="POST")
        self.assertEqual(result["status"], "cancellation_requested")

    def test_cancel_rejects_invalid_session_before_api_request(self):
        request = Mock()
        with self.assertRaisesRegex(ValueError, "unsupported characters"):
            cancel_session(request, "../foreign")
        request.assert_not_called()

    def test_watch_polls_until_terminal_and_sleeps_between_reads(self):
        statuses = iter((
            {"terminal": False, "tasks": [], "task_counts": {"running": 1}},
            {"terminal": True, "tasks": [], "task_counts": {"needs_review": 1}},
        ))
        request = Mock(side_effect=lambda *_args, **_kwargs: next(statuses))
        sleep = Mock()
        echo = Mock()
        watch_session(request, "RES-1", 10, emit=echo, sleep=sleep)
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(10)
        self.assertEqual(echo.call_count, 2)

    def test_invalid_session_id_is_rejected_before_api_request(self):
        with self.assertRaisesRegex(ValueError, "unsupported characters"):
            validate_session_id("../../other")


if __name__ == "__main__":
    unittest.main()
