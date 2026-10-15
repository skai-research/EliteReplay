"""Shared LiteLLM transport for hosted annotations (local rollouts stay local).

Credentials are read from the environment, never from experiment YAMLs. HTTP
errors are deliberately summarized without response bodies or request headers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
import shlex
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_BASE_URL = "https://labeler.example.invalid/v1"
DEFAULT_MODEL = "qwen3-coder-30b-a3b-instruct"


def base_url(value=None):
    value = (value or os.environ.get("LITELLM_BASE_URL")
             or os.environ.get("API_ENDPOINT") or DEFAULT_BASE_URL).rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LiteLLM endpoint must be an HTTPS URL without credentials, query or fragment")
    return value if value.endswith("/v1") else value + "/v1"


def require_api_key():
    key = os.environ.get("LITELLM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("LITELLM_API_KEY is not set; export it before launching hosted annotation")
    return key


def load_environment(profile):
    """Read only literal LiteLLM assignments; never execute the shell profile."""
    allowed = {"LITELLM_API_KEY", "LITELLM_BASE_URL", "API_ENDPOINT"}
    values = {}
    for line in Path(profile).expanduser().read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not match or match[1] not in allowed:
            continue
        parts = shlex.split(match[2], comments=True)
        if len(parts) != 1 or any(character in parts[0] for character in ("$", "`", "\n")):
            raise ValueError(f"{match[1]} must be a literal assignment; shell expansion is not executed")
        values[match[1]] = parts[0]
    for name, value in values.items():
        if not os.environ.get(name):
            os.environ[name] = value
    require_api_key()
    base_url()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LiteLLMError(RuntimeError):
    def __init__(self, message, *, status=None):
        super().__init__(message)
        self.status = status


class LiteLLMClient:
    """Chat Completions with bounded transient retries and optional usage log."""
    def __init__(self, *, endpoint=None, attempts=5, timeout=600, usage_path=None):
        self.base_url = base_url(endpoint)
        self.attempts = int(attempts)
        self.timeout = timeout
        if self.attempts < 1 or self.timeout <= 0:
            raise ValueError("LiteLLM attempts and timeout must be positive")
        self.usage_path = Path(usage_path) if usage_path else None
        self._lock = threading.Lock()

    def complete(self, *, model, messages, max_tokens, temperature=0, top_p=1):
        key = require_api_key()
        body = {"model": model, "messages": list(messages), "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p}
        # Do not forward seed: the selected hosted backend does not promise
        # support. Deterministic local request IDs and caches still apply.
        for attempt in range(self.attempts):
            request = Request(self.base_url + "/chat/completions",
                data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            try:
                with build_opener(_NoRedirect).open(request, timeout=self.timeout) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                if not isinstance(choice["message"].get("content"), str):
                    raise LiteLLMError("LiteLLM returned no text completion")
                usage = result.get("usage", {})
                if self.usage_path:
                    record = {"time": time.time(), "provider": "litellm", "base_url": self.base_url,
                        "requested_model": model, "response_model": result.get("model"),
                        "request_id": result.get("id"), "attempt": attempt + 1,
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "finish_reason": choice.get("finish_reason")}
                    with self._lock:
                        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
                        with self.usage_path.open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                return result
            except HTTPError as error:
                status = error.code
                error.close()
                if status not in (408, 429, 500, 502, 503, 504) or attempt + 1 == self.attempts:
                    raise LiteLLMError(f"LiteLLM Chat Completions failed (HTTP {status})", status=status) from None
            except (URLError, TimeoutError):
                if attempt + 1 == self.attempts:
                    raise LiteLLMError("LiteLLM connection failed after bounded retries") from None
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                raise LiteLLMError("LiteLLM returned an invalid Chat Completions response") from None
            time.sleep(min(30, 2**attempt + random.random()))
