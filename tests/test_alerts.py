"""Tests for the crowd alert connector (task 4)."""

import pytest

from people_group.alerts import (
    AlertDeliveryError,
    CrowdAlertConnector,
    HttpTransport,
    InMemoryTransport,
)
from people_group.events import Event, EventType
from people_group.models import ThresholdLevel


def critical_event(**overrides) -> Event:
    defaults = dict(
        type=EventType.CROWD_ALERT,
        region="plaza",
        message="Crowd alert: 6 people in 2 group(s) at plaza",
        people_count=6,
        group_count=2,
        level=ThresholdLevel.CRITICAL,
        frame_id=4,
    )
    defaults.update(overrides)
    return Event(**defaults)


class TestSendAlert:
    def test_posts_alert_payload(self):
        transport = InMemoryTransport()
        connector = CrowdAlertConnector(
            "https://platform.example.com/api/crowd", api_key="secret-key", transport=transport
        )
        assert connector.send_alert(critical_event()) is True

        request = transport.requests[0]
        assert request.url == "https://platform.example.com/api/crowd"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["X-Api-Key"] == "secret-key"
        assert request.headers["X-Event-Type"] == "crowd_alert"

        payload = request.json()
        assert payload["kind"] == "crowd_alert"
        assert payload["event"]["region"] == "plaza"
        assert payload["event"]["people_count"] == 6
        assert payload["event"]["level"] == "critical"

    def test_no_api_key_header_when_unset(self):
        transport = InMemoryTransport()
        CrowdAlertConnector("https://x.example/alerts", transport=transport).send_alert(critical_event())
        assert "X-Api-Key" not in transport.requests[0].headers

    def test_retry_on_server_error_then_success(self):
        transport = InMemoryTransport(statuses=[500, 200])
        connector = CrowdAlertConnector("https://x.example/alerts", transport=transport, retry_backoff=0)
        assert connector.send_alert(critical_event()) is True
        assert len(transport.requests) == 2

    def test_gives_up_after_max_retries(self):
        transport = InMemoryTransport(statuses=[500, 500, 500])
        connector = CrowdAlertConnector(
            "https://x.example/alerts", transport=transport, max_retries=2, retry_backoff=0
        )
        assert connector.send_alert(critical_event()) is False
        assert len(transport.requests) == 3  # initial attempt + 2 retries

    def test_client_error_is_not_retried(self):
        transport = InMemoryTransport(statuses=[400])
        connector = CrowdAlertConnector("https://x.example/alerts", transport=transport)
        assert connector.send_alert(critical_event()) is False
        assert len(transport.requests) == 1

    def test_network_failure_is_retried_and_reported(self):
        class FlakyTransport:
            def __init__(self):
                self.calls = 0

            def send(self, url, headers, body):
                self.calls += 1
                if self.calls == 1:
                    raise AlertDeliveryError("connection refused")
                return 200

        transport = FlakyTransport()
        connector = CrowdAlertConnector("https://x.example/alerts", transport=transport, retry_backoff=0)
        assert connector.send_alert(critical_event()) is True
        assert transport.calls == 2


class TestConnectionProbe:
    def test_probe_sends_connectivity_payload(self):
        transport = InMemoryTransport()
        connector = CrowdAlertConnector("https://x.example/alerts", transport=transport)
        assert connector.test_connection() is True
        payload = transport.requests[0].json()
        assert payload["kind"] == "connectivity_probe"
        assert transport.requests[0].headers["X-Event-Type"] == "connectivity_probe"

    def test_probe_reports_failure(self):
        transport = InMemoryTransport(statuses=[403])
        connector = CrowdAlertConnector("https://x.example/alerts", transport=transport)
        assert connector.test_connection() is False


class TestConnectorValidation:
    def test_empty_endpoint_rejected(self):
        with pytest.raises(ValueError, match="endpoint"):
            CrowdAlertConnector("  ")

    def test_negative_retries_rejected(self):
        with pytest.raises(ValueError, match="max_retries"):
            CrowdAlertConnector("https://x.example/alerts", max_retries=-1)

    def test_http_transport_is_default(self):
        connector = CrowdAlertConnector("https://x.example/alerts")
        assert isinstance(connector._transport, HttpTransport)
