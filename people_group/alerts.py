"""Crowd alert delivery to the platform (task 4).

The connector posts a JSON alert for a critical event to a platform
endpoint. The transport is pluggable so tests (and the demo) can run without
a network.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from .events import Event

USER_AGENT = "people-group/0.1.0"


class AlertDeliveryError(RuntimeError):
    """Raised when the transport fails at the network level."""


@runtime_checkable
class AlertTransport(Protocol):
    """Sends a JSON body to a URL and returns the HTTP status code."""

    def send(self, url: str, headers: dict, body: bytes) -> int:
        ...


class HttpTransport:
    """Real HTTP POST via urllib (no external dependencies)."""

    def __init__(self, timeout: float = 5.0) -> None:
        self._timeout = timeout

    def send(self, url: str, headers: dict, body: bytes) -> int:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            return int(exc.code)  # 4xx/5xx are completed responses
        except urllib.error.URLError as exc:
            raise AlertDeliveryError(f"network error posting to {url}: {exc.reason}") from exc


@dataclass
class RecordedRequest:
    url: str
    headers: dict
    body: bytes

    def json(self) -> dict:
        return json.loads(self.body.decode("utf-8"))


class InMemoryTransport:
    """Test/demo transport: records requests and replays scripted statuses."""

    def __init__(self, statuses: Sequence[int] = ()) -> None:
        self.requests: List[RecordedRequest] = []
        self._statuses: List[int] = list(statuses)

    def send(self, url: str, headers: dict, body: bytes) -> int:
        self.requests.append(RecordedRequest(url=url, headers=headers, body=body))
        if self._statuses:
            return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return 200


class CrowdAlertConnector:
    """Delivers crowd alerts to the platform endpoint.

    * 2xx  -> delivered
    * 4xx  -> rejected (not retried; fix the request)
    * 5xx / network error -> retried up to ``max_retries`` times with backoff
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        transport: Optional[AlertTransport] = None,
        max_retries: int = 2,
        retry_backoff: float = 0.05,
        source: str = "people-group",
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must not be empty")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        self._endpoint = endpoint
        self._api_key = api_key
        self._transport: AlertTransport = transport if transport is not None else HttpTransport()
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._source = source

    @property
    def endpoint(self) -> str:
        return self._endpoint

    # -- payloads ----------------------------------------------------------
    def build_payload(self, event: Event) -> dict:
        return {
            "kind": "crowd_alert",
            "source": self._source,
            "event": event.to_dict(),
        }

    def _headers(self, event_type: str) -> dict:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "X-Source": self._source,
            "X-Event-Type": event_type,
        }
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        return headers

    # -- sending -------------------------------------------------------------
    def _post_with_retries(self, payload: dict, event_type: str) -> Optional[int]:
        body = json.dumps(payload).encode("utf-8")
        headers = self._headers(event_type)
        status: Optional[int] = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                time.sleep(self._retry_backoff * (2 ** (attempt - 1)))
            try:
                status = self._transport.send(self._endpoint, headers, body)
            except AlertDeliveryError:
                status = None
                continue
            if 200 <= status < 300:
                return status
            if 400 <= status < 500:
                return status  # client error: retrying will not help
        return status

    def send_alert(self, event: Event) -> bool:
        """Send a crowd alert; True when the platform acknowledged (2xx)."""
        status = self._post_with_retries(self.build_payload(event), event.type.value)
        return status is not None and 200 <= status < 300

    def test_connection(self) -> bool:
        """Send a connectivity probe to verify the platform endpoint is reachable."""
        payload = {
            "kind": "connectivity_probe",
            "source": self._source,
            "timestamp": time.time(),
        }
        status = self._post_with_retries(payload, "connectivity_probe")
        return status is not None and 200 <= status < 300
