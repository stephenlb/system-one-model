"""Dependency-free HTTP client for a deployed truetype Space or API."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TrueTypeError(RuntimeError):
    """A request failed or the server returned an invalid response."""


class TrueTypeClient:
    """Call a hosted System One replica without installing model dependencies.

    ``endpoint`` is the Space's direct ``https://<owner>-<space>.hf.space`` URL,
    or a self-hosted API base URL. Supply ``token`` for a protected endpoint.
    """

    def __init__(self, endpoint: str, *, token: str | None = None, timeout: float = 120) -> None:
        endpoint = endpoint.rstrip("/")
        if not endpoint.startswith(("https://", "http://")):
            raise ValueError("endpoint must be an http:// or https:// URL")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoint = endpoint
        self.token = token
        self.timeout = timeout

    def system_one(
        self,
        *,
        state: str | dict | list,
        questions: dict[str, dict],
        model: str = "gemma-4-12b",
        temperature: float = 0.7,
    ) -> dict:
        """Return the API response with ``answers``, ``model``, and ``usage``."""
        if not questions:
            raise ValueError("questions must not be empty")
        payload = json.dumps({
            "state": state,
            "questions": questions,
            "model": model,
            "temperature": temperature,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            f"{self.endpoint}/v1/systemone", data=payload, headers=headers, method="POST"
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (ValueError, AttributeError):
                pass
            raise TrueTypeError(f"API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise TrueTypeError(f"Could not reach {self.endpoint}: {exc.reason}") from exc
        except (ValueError, UnicodeError) as exc:
            raise TrueTypeError("API returned invalid JSON") from exc
