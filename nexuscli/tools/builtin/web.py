"""Network tools: ``web_fetch`` and ``web_search``.

Security note that most agent CLIs get wrong: ``web_fetch`` performs **SSRF
screening**. Model-supplied URLs are resolved and rejected when they point at
loopback/link-local/private/metadata addresses, unless the user explicitly
opted in with ``--allow-private-network``. Without this, a prompt-injected page
can make the agent read ``http://169.254.169.254/`` or scan the LAN.
"""

from __future__ import annotations

import gzip
import html
import io
import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.errors import NetworkError
from ..base import ConfirmationRequest, Tool, ToolContext, ToolResult, clip

MAX_FETCH_BYTES = 4_000_000
ALLOWED_SCHEMES = {"http", "https"}
BLOCKED_HOSTS = {"metadata.google.internal", "metadata"}
USER_AGENT = "Mozilla/5.0 (compatible; NexusCLI/1.0; +https://nexus.local)"


def screen_url(url: str, *, allow_private: bool = False) -> Tuple[str, Optional[str]]:
    """Validate a URL. Returns (normalised_url, error_or_None)."""
    if not url or not url.strip():
        return url, "empty URL"
    url = url.strip()
    if "//" not in url and not url.startswith(("http:", "https:")):
        url = "https://" + url
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:
        return url, f"unparseable URL: {exc}"
    if parsed.scheme not in ALLOWED_SCHEMES:
        return url, f"scheme '{parsed.scheme}' is not allowed (http/https only)"
    host = (parsed.hostname or "").lower()
    if not host:
        return url, "URL has no host"
    if host in BLOCKED_HOSTS or host.endswith(".internal") or host.endswith(".local"):
        if not allow_private:
            return url, f"host '{host}' is blocked (internal/metadata name)"
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return url, f"DNS resolution failed for '{host}': {exc}"
    if not allow_private:
        for info in infos:
            ip_str = info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                    or ip.is_multicast or ip.is_unspecified):
                return url, (f"host '{host}' resolves to {ip_str}, which is a private/reserved "
                             "address. Refusing to fetch (SSRF protection). Use --allow-private-network to override.")
    return urllib.parse.urlunparse(parsed), None


class _TextExtractor(HTMLParser):
    """HTML -> readable text. Keeps links, list markers and table cell separation."""

    SKIP = {"script", "style", "noscript", "template", "svg", "head", "iframe", "canvas"}
    BLOCK = {"p", "div", "section", "article", "header", "footer", "nav", "aside", "main",
             "br", "hr", "ul", "ol", "table", "thead", "tbody", "tr", "blockquote", "pre"}
    HEADING = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip_depth = 0
        self._href: Optional[str] = None
        self._link_text: List[str] = []
        self._in_pre = False
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in self.HEADING:
            self.parts.append("\n\n" + self.HEADING[tag])
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "td" or tag == "th":
            self.parts.append(" | ")
        elif tag == "pre":
            self._in_pre = True
            self.parts.append("\n```\n")
        elif tag == "code" and not self._in_pre:
            self.parts.append("`")
        elif tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self._href = v
                    self._link_text = []
                    break
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag == "pre":
            self._in_pre = False
            self.parts.append("\n```\n")
        elif tag == "code" and not self._in_pre:
            self.parts.append("`")
        elif tag == "a" and self._href is not None:
            text = "".join(self._link_text).strip()
            self.parts.append(f"{text} ({self._href})" if text and text != self._href else self._href)
            self._href = None
        elif tag in self.HEADING or tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        # <title> lives inside <head>, which is otherwise skipped.
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        if self._href is not None:
            self._link_text.append(data)
            return
        self.parts.append(data if self._in_pre else data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = html.unescape(raw)
        lines = []
        for line in raw.split("\n"):
            stripped = line.strip() if not self._in_pre else line.rstrip()
            lines.append(stripped)
        out = "\n".join(lines)
        while "\n\n\n" in out:
            out = out.replace("\n\n\n", "\n\n")
        return out.strip()


def html_to_text(markup: str) -> Tuple[str, str]:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        # Malformed HTML: fall back to tag stripping rather than failing.
        import re

        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", markup, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return html.unescape(re.sub(r"[ \t]+", " ", text)).strip(), ""
    return parser.text(), parser.title.strip()


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a URL and return its content as readable text (HTML is converted to markdown-ish text; "
        "JSON is pretty-printed). Use `raw=true` for untouched bytes-as-text. Private/internal addresses "
        "are refused unless the user enabled them. For search results use web_search."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 500, "default": 20000},
            "raw": {"type": "boolean", "default": False, "description": "Do not convert HTML to text."},
            "timeout": {"type": "number", "default": 30},
            "headers": {"type": "object", "description": "Extra request headers."},
        },
        "required": ["url"],
    }
    category = "web"
    read_only = True
    needs_network = True

    def confirmation(self, args, ctx):
        return ConfirmationRequest(title=f"Fetch {args.get('url')}", risk="normal",
                                   key=f"web_fetch:{urllib.parse.urlparse(str(args.get('url'))).hostname}")

    def run(self, args, ctx):
        allow_private = bool(getattr(ctx.permissions, "allow_private_network", False)) if ctx.permissions else False
        url, err = screen_url(args["url"], allow_private=allow_private)
        if err:
            return ToolResult.fail(err)
        timeout = float(args.get("timeout") or 30)
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
                   "Accept-Encoding": "gzip"}
        for k, v in (args.get("headers") or {}).items():
            headers[str(k)] = str(v)
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                final_url = resp.geturl()
                status = getattr(resp, "status", 200)
                ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                encoding = resp.headers.get_content_charset() or "utf-8"
                content_encoding = (resp.headers.get("Content-Encoding") or "").lower()
                data = resp.read(MAX_FETCH_BYTES + 1)
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read(4096)
            except Exception:
                pass
            return ToolResult.fail(f"HTTP {exc.code} {exc.reason} for {url}\n{body.decode('utf-8', 'replace')[:1000]}")
        except urllib.error.URLError as exc:
            return ToolResult.fail(f"Network error fetching {url}: {exc.reason}")
        except (socket.timeout, TimeoutError):
            return ToolResult.fail(f"Timed out after {timeout:.0f}s fetching {url}")
        except OSError as exc:
            return ToolResult.fail(f"Fetch failed: {exc}")
        truncated = len(data) > MAX_FETCH_BYTES
        if truncated:
            data = data[:MAX_FETCH_BYTES]
        if content_encoding == "gzip" and data[:2] == b"\x1f\x8b":
            try:
                data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()
            except OSError:
                pass
        try:
            text = data.decode(encoding, errors="replace")
        except LookupError:
            text = data.decode("utf-8", errors="replace")
        title = ""
        if not args.get("raw"):
            if "html" in ctype or text.lstrip()[:1] == "<":
                text, title = html_to_text(text)
            elif "json" in ctype or text.lstrip()[:1] in ("{", "["):
                try:
                    text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
        limit = min(int(args.get("max_chars") or 20000), ctx.max_output_chars)
        head = f"URL: {final_url}\nStatus: {status}\nType: {ctype or 'unknown'}"
        if title:
            head += f"\nTitle: {title}"
        if truncated:
            head += f"\n(truncated to {MAX_FETCH_BYTES // 1000}KB)"
        return ToolResult(content=f"{head}\n\n{clip(text, limit)}",
                          data={"url": final_url, "status": status, "content_type": ctype, "title": title,
                                "chars": len(text)})


SEARCH_PROVIDERS = ("brave", "tavily", "serper", "exa", "searxng", "duckduckgo")


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web and return titles, URLs and snippets. Requires a configured search backend "
        "(brave/tavily/serper/exa/searxng) -- see `nexus config search`. Results are links, not facts: "
        "open the promising ones with web_fetch before relying on them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 6},
            "freshness": {"type": "string", "enum": ["any", "day", "week", "month", "year"], "default": "any"},
        },
        "required": ["query"],
    }
    category = "web"
    read_only = True
    needs_network = True

    def confirmation(self, args, ctx):
        return ConfirmationRequest(title=f"Search: {args.get('query')}", risk="normal", key="web_search:*")

    def run(self, args, ctx):
        cfg = (getattr(ctx.config, "search", {}) if ctx.config else {}) or {}
        backend = (cfg.get("backend") or _auto_backend(cfg)).lower()
        if backend not in SEARCH_PROVIDERS:
            return ToolResult.fail(
                f"No search backend configured. Supported: {', '.join(SEARCH_PROVIDERS)}. "
                "Set one with `nexus config set search.backend brave` plus its API key."
            )
        try:
            results = _search(backend, args["query"], int(args.get("max_results") or 6),
                              args.get("freshness", "any"), cfg, ctx)
        except NetworkError as exc:
            return ToolResult.fail(str(exc))
        except Exception as exc:  # never let a flaky search backend kill the loop
            ctx.log.warning("web_search failed", backend=backend, error=str(exc))
            return ToolResult.fail(f"Search backend '{backend}' failed: {exc}")
        if not results:
            return ToolResult(content=f"No results for '{args['query']}' via {backend}.", data={"results": []})
        lines = [f"{i}. {r['title']}\n   {r['url']}\n   {r.get('snippet', '')}".rstrip()
                 for i, r in enumerate(results, 1)]
        return ToolResult(content="\n\n".join(lines), data={"results": results, "backend": backend})


def _auto_backend(cfg: Dict[str, Any]) -> str:
    import os

    for name, var in (("brave", "BRAVE_API_KEY"), ("tavily", "TAVILY_API_KEY"), ("serper", "SERPER_API_KEY"),
                      ("exa", "EXA_API_KEY"), ("searxng", "SEARXNG_URL")):
        if os.environ.get(var) or cfg.get(f"{name}_api_key") or cfg.get(var.lower()):
            return name
    return ""


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float = 25) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    return _read_json(req, timeout)


def _get_json(url: str, headers: Dict[str, str], timeout: float = 25) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    return _read_json(req, timeout)


def _read_json(req: urllib.request.Request, timeout: float) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(2_000_000)
            if resp.headers.get("Content-Encoding", "").lower() == "gzip" and raw[:2] == b"\x1f\x8b":
                raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
            return json.loads(raw.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        raise NetworkError(f"{exc.code} {exc.reason} from {req.full_url}") from exc
    except urllib.error.URLError as exc:
        raise NetworkError(f"Network error contacting {req.full_url}: {exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise NetworkError(f"Timeout contacting {req.full_url}") from exc
    except json.JSONDecodeError as exc:
        raise NetworkError(f"Non-JSON response from {req.full_url}") from exc


def _search(backend: str, query: str, limit: int, freshness: str, cfg: Dict[str, Any], ctx: ToolContext) -> List[Dict[str, str]]:
    import os

    def key(name: str) -> str:
        v = cfg.get(f"{name}_api_key") or os.environ.get(f"{name.upper()}_API_KEY") or ""
        if not v:
            raise NetworkError(f"No API key for search backend '{name}'. Set {name.upper()}_API_KEY.")
        return str(v)

    out: List[Dict[str, str]] = []
    if backend == "brave":
        params = {"q": query, "count": str(min(limit, 20))}
        if freshness != "any":
            params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}[freshness]
        data = _get_json("https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(params),
                         {"X-Subscription-Token": key("brave"), "Accept": "application/json"})
        for item in (data.get("web") or {}).get("results") or []:
            out.append({"title": item.get("title", ""), "url": item.get("url", ""),
                        "snippet": item.get("description", "")})
    elif backend == "tavily":
        data = _post_json("https://api.tavily.com/search",
                          {"api_key": key("tavily"), "query": query, "max_results": limit,
                           "search_depth": "basic", "include_answer": False},
                          {})
        for item in data.get("results") or []:
            out.append({"title": item.get("title", ""), "url": item.get("url", ""),
                        "snippet": item.get("content", "")})
    elif backend == "serper":
        data = _post_json("https://google.serper.dev/search", {"q": query, "num": limit},
                          {"X-API-KEY": key("serper")})
        for item in data.get("organic") or []:
            out.append({"title": item.get("title", ""), "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")})
    elif backend == "exa":
        data = _post_json("https://api.exa.ai/search",
                          {"query": query, "numResults": limit, "contents": ["text"]},
                          {"x-api-key": key("exa")})
        for item in data.get("results") or []:
            out.append({"title": item.get("title", ""), "url": item.get("url", ""),
                        "snippet": (item.get("text") or "")[:400]})
    elif backend == "searxng":
        base = (cfg.get("searxng_url") or os.environ.get("SEARXNG_URL") or "").rstrip("/")
        if not base:
            raise NetworkError("SEARXNG_URL is not set.")
        data = _get_json(f"{base}/search?format=json&" + urllib.parse.urlencode({"q": query}),
                         {"Accept": "application/json"})
        for item in (data.get("results") or [])[:limit]:
            out.append({"title": item.get("title", ""), "url": item.get("url", ""),
                        "snippet": item.get("content", "")})
    elif backend == "duckduckgo":
        data = _get_json("https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}), {"Accept": "application/json"})
        if data.get("AbstractText"):
            out.append({"title": data.get("Heading", query), "url": data.get("AbstractURL", ""),
                        "snippet": data.get("AbstractText", "")})
        for topic in (data.get("RelatedTopics") or [])[:limit]:
            if isinstance(topic, dict) and topic.get("FirstURL"):
                out.append({"title": topic.get("Text", "")[:80], "url": topic["FirstURL"],
                            "snippet": topic.get("Text", "")})
    else:
        raise NetworkError(f"Unsupported search backend '{backend}'.")
    return out[:limit]


def build_tools() -> List[Tool]:
    return [WebFetchTool(), WebSearchTool()]
