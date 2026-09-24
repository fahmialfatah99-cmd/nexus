"""HTTP transport + Server-Sent-Events parser (pure stdlib).

Two hard requirements drove this design:

1. **Injectability** -- every provider talks through :class:`Transport`, so the
   whole stack (including streaming tool calls) is unit-testable offline with
   :class:`MockTransport`.
2. **Correct SSE** -- naive ``line.split("data:")`` parsers break on multi-line
   data fields, CRLF line endings, comments, events split across TCP chunks and
   on a UTF-8 multi-byte character that lands on a chunk boundary. The parser
   below implements the WHATWG event-stream algorithm properly.
"""

from __future__ import annotations

import gzip
import http.client
import io
import json
import os
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from ..core.errors import (
    AuthError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TimeoutError_,
)
from ..core.logging_ import get_logger

USER_AGENT = "nexus-cli/1.0 (+https://github.com/nexus)"

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class SSEEvent:
    """One dispatched server-sent event."""

    data: str = ""
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None

    def json(self) -> Any:
        return json.loads(self.data)


@dataclass
class Response:
    status: int
    headers: Dict[str, str]
    body: bytes = b""
    url: str = ""

    def json(self) -> Any:
        return json.loads(self.text())

    def text(self) -> str:
        return decode_body(self.body, self.headers)

    def header(self, name: str) -> Optional[str]:
        low = name.lower()
        for k, v in self.headers.items():
            if k.lower() == low:
                return v
        return None


def decode_body(body: bytes, headers: Dict[str, str]) -> str:
    enc = "utf-8"
    for k, v in headers.items():
        if k.lower() == "content-type" and "charset=" in v.lower():
            enc = v.lower().split("charset=")[-1].split(";")[0].strip().strip('"') or "utf-8"
    try:
        return body.decode(enc, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class StreamHandle:
    """Incremental byte source for streaming responses."""

    def __init__(self, chunks: Iterable[bytes], close: Optional[Callable[[], None]] = None) -> None:
        self._chunks = iter(chunks)
        self._close = close
        self.consumed = 0

    def __iter__(self) -> Iterator[bytes]:
        for c in self._chunks:
            self.consumed += len(c)
            yield c

    def close(self) -> None:
        if self._close is not None:
            try:
                self._close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._close = None

    def __enter__(self) -> "StreamHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# SSE parser
# --------------------------------------------------------------------------- #
class SSEParser:
    """Streaming Server-Sent-Events parser.

    Feed arbitrary byte chunks via :meth:`feed`; fully-formed events are yielded
    as they complete. Call :meth:`close` at end-of-stream to flush a trailing
    event that was not terminated by a blank line (some servers do this).
    """

    def __init__(self) -> None:
        self._buf = b""
        self._data: List[str] = []
        self._event: Optional[str] = None
        self._id: Optional[str] = None
        self._retry: Optional[int] = None
        self._first = True

    def feed(self, chunk: bytes) -> Iterator[SSEEvent]:
        self._buf += chunk
        while True:
            _line, sep, rest = self._take_line()
            if not sep:
                # Incomplete line (or a trailing CR whose LF has not arrived
                # yet): leave the buffer untouched and wait for more bytes.
                break
            self._buf = rest
            line = _line
            ev = self._process_line(line)
            if ev is not None:
                yield ev

    def _take_line(self) -> Tuple[bytes, bytes, bytes]:
        """Return (line_without_terminator, terminator, remainder)."""
        buf = self._buf
        i = 0
        n = len(buf)
        while i < n:
            b = buf[i : i + 1]
            if b == b"\n":
                return buf[:i], b"\n", buf[i + 1 :]
            if b == b"\r":
                if i + 1 < n:
                    if buf[i + 1 : i + 2] == b"\n":
                        return buf[:i], b"\r\n", buf[i + 2 :]
                    return buf[:i], b"\r", buf[i + 1 :]
                # Trailing CR: we cannot know yet whether LF follows.
                return buf, b"", b""
            i += 1
        return buf, b"", b""

    def _process_line(self, raw: bytes) -> Optional[SSEEvent]:
        if self._first:
            self._first = False
            if raw.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
                raw = raw[3:]
        if not raw:
            return self._dispatch()
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("utf-8", errors="replace")
        if line.startswith(":"):  # comment / keep-alive
            return None
        if ":" in line:
            name, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
        else:
            name, value = line, ""
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value or None
        elif name == "id":
            if "\x00" not in value:
                self._id = value or None
        elif name == "retry":
            if value.isdigit():
                self._retry = int(value)
        return None

    def _dispatch(self) -> Optional[SSEEvent]:
        if not self._data:
            self._event = None
            return None
        ev = SSEEvent(data="\n".join(self._data), event=self._event, id=self._id, retry=self._retry)
        self._data = []
        self._event = None
        return ev

    def close(self) -> Optional[SSEEvent]:
        if self._buf:
            rest, self._buf = self._buf, b""
            self._process_line(rest)
        return self._dispatch()


def iter_sse(chunks: Iterable[bytes]) -> Iterator[SSEEvent]:
    parser = SSEParser()
    for chunk in chunks:
        for ev in parser.feed(chunk):
            yield ev
    tail = parser.close()
    if tail is not None:
        yield tail


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #
@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.6
    max_delay: float = 20.0
    jitter: float = 0.25
    respect_retry_after: bool = True

    def delay_for(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None and self.respect_retry_after and retry_after > 0:
            return min(retry_after, self.max_delay)
        exp = min(self.base_delay * (2 ** max(0, attempt - 1)), self.max_delay)
        return exp * (1 + random.uniform(-self.jitter, self.jitter))


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
class Transport:
    """Abstract transport. Providers only ever use this interface."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Any] = None,
        data: Optional[bytes] = None,
        timeout: float = 120.0,
        stream: bool = False,
    ) -> Tuple[Response, Optional[StreamHandle]]:
        raise NotImplementedError


class HTTPTransport(Transport):
    """urllib based transport with retries, gzip and streaming support."""

    def __init__(self, policy: Optional[RetryPolicy] = None, ssl_context: Optional[ssl.SSLContext] = None,
                 proxy: Optional[str] = None) -> None:
        self.policy = policy or RetryPolicy()
        self.ssl_context = ssl_context
        self.proxy = proxy or os_environ_proxy()
        self.log = get_logger()

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Any] = None,
        data: Optional[bytes] = None,
        timeout: float = 120.0,
        stream: bool = False,
    ) -> Tuple[Response, Optional[StreamHandle]]:
        hdrs = {
            "User-Agent": USER_AGENT,
            "Accept": "text/event-stream" if stream else "application/json",
            "Accept-Encoding": "gzip",
        }
        if headers:
            hdrs.update({k: v for k, v in headers.items() if v is not None})
        payload: Optional[bytes] = None
        if json_body is not None:
            payload = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif data is not None:
            payload = data

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.policy.max_attempts + 1):
            req = urllib.request.Request(url, data=payload, headers=hdrs, method=method.upper())
            opener_args: List[Any] = []
            if self.ssl_context is not None:
                opener_args.append(urllib.request.HTTPSHandler(context=self.ssl_context))
            if self.proxy:
                opener_args.append(urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy}))
            opener = urllib.request.build_opener(*opener_args) if opener_args else urllib.request.build_opener()
            try:
                resp = opener.open(req, timeout=timeout)
            except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a body
                body = _safe_read(exc)
                retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                # Safe to retry: no bytes have been handed to the caller yet.
                if exc.code in RETRYABLE_STATUS and attempt < self.policy.max_attempts:
                    delay = self.policy.delay_for(attempt, retry_after)
                    self.log.warning("http retry", status=exc.code, attempt=attempt, delay=round(delay, 2), url=url)
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise http_error(exc.code, body, dict(exc.headers or {}), url) from exc
            except (socket.timeout, TimeoutError) as exc:
                if attempt < self.policy.max_attempts:
                    delay = self.policy.delay_for(attempt)
                    self.log.warning("timeout retry", attempt=attempt, delay=round(delay, 2), url=url)
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise TimeoutError_(f"Request to {url} timed out after {timeout}s") from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                if attempt < self.policy.max_attempts and _is_retryable_reason(reason):
                    delay = self.policy.delay_for(attempt)
                    self.log.warning("network retry", attempt=attempt, delay=round(delay, 2), reason=str(reason))
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise NetworkError(f"Network error contacting {url}: {reason}") from exc
            except http.client.HTTPException as exc:
                # IncompleteRead / BadStatusLine / RemoteDisconnected: the server
                # hung up or sent a malformed stream. Retryable, and it must
                # surface as a NetworkError (so failover can kick in) rather than
                # as an "unexpected provider error".
                if attempt < self.policy.max_attempts:
                    delay = self.policy.delay_for(attempt)
                    self.log.warning("http protocol retry", attempt=attempt, delay=round(delay, 2),
                                     error=f"{type(exc).__name__}: {exc}")
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise NetworkError(f"Connection broke while talking to {url}: "
                                   f"{type(exc).__name__}: {exc}") from exc
            except OSError as exc:
                if attempt < self.policy.max_attempts:
                    time.sleep(self.policy.delay_for(attempt))
                    last_exc = exc
                    continue
                raise NetworkError(f"Network error contacting {url}: {exc}") from exc

            headers_out = {k: v for k, v in resp.headers.items()}
            if stream:
                handle = StreamHandle(_decompress_chunks(resp, headers_out), close=resp.close)
                return Response(status=resp.status, headers=headers_out, url=url), handle
            try:
                raw = resp.read()
            except (http.client.HTTPException, ValueError) as exc:
                resp.close()
                if attempt < self.policy.max_attempts:
                    time.sleep(self.policy.delay_for(attempt))
                    last_exc = exc
                    continue
                raise NetworkError(f"Connection broke while reading {url}: "
                                   f"{type(exc).__name__}: {exc}") from exc
            body = _maybe_gunzip(raw, headers_out)
            resp.close()
            return Response(status=resp.status, headers=headers_out, body=body, url=url), None

        # Exhausted retries on a transport-level condition.
        raise NetworkError(f"Failed to reach {url} after {self.policy.max_attempts} attempts: {last_exc}")


class NexusHTTPError(ProviderError):
    """Marker base so callers can catch transport-mapped errors uniformly."""


def os_environ_proxy() -> Optional[str]:
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(var)
        if v:
            return v
    return None


def _safe_read(fp) -> bytes:
    try:
        return fp.read()
    except Exception:
        return b""


def _maybe_gunzip(raw: bytes, headers: Dict[str, str]) -> bytes:
    enc = ""
    for k, v in headers.items():
        if k.lower() == "content-encoding":
            enc = v.lower()
    if "gzip" in enc and raw[:2] == b"\x1f\x8b":
        try:
            return gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        except OSError:
            return raw
    if "deflate" in enc and raw:
        import zlib

        try:
            return zlib.decompress(raw)
        except zlib.error:
            try:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
            except zlib.error:
                return raw
    return raw


def _decompress_chunks(resp, headers: Dict[str, str]) -> Iterator[bytes]:
    enc = ""
    for k, v in headers.items():
        if k.lower() == "content-encoding":
            enc = v.lower()
    if "gzip" not in enc:
        while True:
            try:
                chunk = resp.read(8192)
            except (socket.timeout, OSError):
                break
            except (http.client.HTTPException, ValueError) as exc:
                raise NetworkError(f"Connection broke mid-stream: {type(exc).__name__}: {exc}") from exc
            if not chunk:
                break
            yield chunk
        return
    import zlib

    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    while True:
        try:
            chunk = resp.read(4096)
        except (socket.timeout, OSError):
            break
        except (http.client.HTTPException, ValueError) as exc:
            raise NetworkError(f"Connection broke mid-stream: {type(exc).__name__}: {exc}") from exc
        if not chunk:
            break
        out = dec.decompress(chunk)
        if out:
            yield out
    tail = dec.flush()
    if tail:
        yield tail


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        if dt is not None:
            return max(0.0, (dt.timestamp() - time.time()))
    except Exception:
        pass
    return None


def _is_retryable_reason(reason: Any) -> bool:
    text = str(reason).lower()
    if isinstance(reason, (socket.timeout, TimeoutError, ConnectionError)):
        return True
    return any(k in text for k in ("timed out", "temporary failure", "connection reset", "broken pipe",
                                   "remote end closed", "eof occurred", "try again", "unreachable"))


def http_error(status: int, body: bytes, headers: Dict[str, str], url: str) -> NexusHTTPError:
    text = decode_body(body, headers).strip()
    detail = _extract_error_message(text) or (text[:600] if text else f"HTTP {status}")
    if status in (401, 403):
        return AuthError(f"Authentication failed ({status}) for {url}: {detail}")
    if status == 404:
        return ModelNotFoundError(f"Not found ({status}) for {url}: {detail}")
    if status == 429:
        return RateLimitError(f"Rate limited ({status}): {detail}", retry_after=_parse_retry_after(headers.get("Retry-After")))
    if status >= 500 or status in (408, 425):
        return ProviderError(f"Provider error ({status}): {detail}", status_code=status, retryable=True)
    if status == 400 and _looks_like_context_overflow(text):
        return ProviderError(f"Request rejected ({status}): {detail}", status_code=status, retryable=False,
                             hint="The conversation is too large for this model. Try /compact.")
    return ProviderError(f"Request failed ({status}): {detail}", status_code=status, retryable=False)


def _extract_error_message(text: str) -> Optional[str]:
    if not text.startswith("{") and not text.startswith("["):
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    for path in (("error", "message"), ("message",), ("error",), ("detail",), ("error", "error", "message")):
        cur: Any = obj
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and cur.strip():
            return cur.strip()
        if ok and isinstance(cur, dict) and cur.get("message"):
            return str(cur["message"])
    return None


def _looks_like_context_overflow(text: str) -> bool:
    low = text.lower()
    return "context" in low and any(k in low for k in ("length", "window", "exceed", "too long", "too many tokens"))


# --------------------------------------------------------------------------- #
# Test/offline transport
# --------------------------------------------------------------------------- #
@dataclass
class MockTransport(Transport):
    """Deterministic transport used by the test-suite and ``nexus --demo``.

    ``responses`` is a list of either :class:`Response`, ``(status, json_obj)``
    tuples, or callables ``(request) -> Response``. SSE payloads can be given as
    ``sse=[...]`` which are chunked exactly like a real server would.
    """

    responses: List[Any] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    chunk_size: int = 24

    def add(self, *args, **kwargs) -> "MockTransport":
        self.responses.append((args, kwargs))
        return self

    def queue(self, item: Any) -> "MockTransport":
        self.responses.append(item)
        return self

    def request(self, method, url, *, headers=None, json_body=None, data=None, timeout=120.0, stream=False):
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "json": json_body, "stream": stream})
        if not self.responses:
            raise ProviderError("MockTransport exhausted (no queued response)", provider="mock")
        item = self.responses.pop(0)
        if callable(item):
            item = item({"method": method, "url": url, "json": json_body, "headers": headers or {}})
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
            status, payload = item
            body = json.dumps(payload).encode()
            return Response(status=status, headers={"Content-Type": "application/json"}, body=body, url=url), None
        if isinstance(item, dict) and "sse" in item:
            raw = b"".join(_encode_sse(ev) for ev in item["sse"])
            chunks = [raw[i : i + self.chunk_size] for i in range(0, len(raw), self.chunk_size)] or [b""]
            return Response(status=item.get("status", 200), headers={"Content-Type": "text/event-stream"},
                            url=url), StreamHandle(iter(chunks))
        if isinstance(item, Response):
            return item, None
        raise ProviderError(f"Unsupported MockTransport response: {item!r}")


def _encode_sse(ev: Any) -> bytes:
    if isinstance(ev, str):
        if ev == "[DONE]":
            return b"data: [DONE]\n\n"
        return f"data: {ev}\n\n".encode()
    if isinstance(ev, dict):
        if ev.get("event"):
            return f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n".encode()
        return f"data: {json.dumps(ev.get('data', ev))}\n\n".encode()
    raise TypeError(f"cannot encode SSE event: {ev!r}")
