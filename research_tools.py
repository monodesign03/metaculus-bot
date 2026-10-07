"""
Research sources that need no LLM: the pages a question resolves on, similar
questions Metaculus has already resolved, and Manifold prediction markets.

Published FutureEval analyses link these to better scores: scraping the
links in the resolution criteria (the winning bot had "link assistants"), and
checking similar questions and markets.

Everything here is synchronous (requests); the bot runs it in worker threads.
Every public function returns "" on any failure: research is best-effort and
must never stop a forecast.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Iterable

import requests

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0 Safari/537.36"
)
READER_PREFIX = "https://r.jina.ai/"  # renders JS-heavy pages and PDFs as text
METACULUS_POSTS_URL = "https://www.metaculus.com/api/posts/"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
MIN_USEFUL_PAGE_CHARS = 400
MAX_PAGE_BYTES = 3_000_000

_URL = re.compile(r"https?://[^\s<>\"'\]\}|]+")
_SKIPPED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".mp4", ".zip")
_STOPWORDS = set(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    either few for from further had has have having he her here hers him his how if in into
    is it its itself just least less many may might more most much must no nor not now of
    off on once only or other our out over own per same she should since so some such than
    that the their them then there these they this those through to too under until up upon
    very was we were what when where whether which while who whom why will with within
    would you your yes question resolve resolves resolved resolution according reported
    report reports announce announced official officially least greater higher lower date
    dates day days week weeks month months year years january february march april june july
    august september october november december monday tuesday wednesday thursday friday
    saturday sunday next last time end start
    """.split()
)


# ------------------------------------------------------------------ text ----
def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n[…truncated]"


def _clean_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line and (not cleaned or not cleaned[-1]):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe", "nav", "footer"}
    _BLOCK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
        "article", "table", "ul", "ol", "header", "main", "blockquote", "pre", "dd", "dt",
    }
    _CELL = {"td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self.skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")
        elif tag in self._CELL:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as error:  # malformed HTML: keep what was parsed
        logger.debug(f"HTML parse problem: {error}")
    return _clean_whitespace("".join(parser.parts))


def _terms(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-']{3,}|\d[\d,.]*", text or "")
    return {w.lower().strip(",.") for w in words if w.lower() not in _STOPWORDS}


def select_relevant_passages(text: str, query: str, limit: int) -> str:
    """Keeps the start of the page plus the paragraphs that share the most
    words and numbers with the question, in their original order."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    terms = _terms(query)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    head = text[: min(600, limit // 4)]
    scored = []
    for index, paragraph in enumerate(paragraphs):
        lowered = paragraph.lower()
        score = sum(1 for term in terms if term in lowered)
        if re.search(r"\d", paragraph):
            score += 0.5
        if score > 0:
            scored.append((score, index))
    budget = limit - len(head) - 10
    chosen: set[int] = set()
    for _, index in sorted(scored, key=lambda item: (-item[0], item[1])):
        paragraph = paragraphs[index][:600]
        if len(paragraph) + 1 > budget:
            continue
        chosen.add(index)
        budget -= len(paragraph) + 1
        if budget < 80:
            break
    body = "\n".join(paragraphs[i][:600] for i in sorted(chosen))
    return f"{head}\n[…]\n{body}".strip()


def top_keywords(text: str, count: int) -> list[str]:
    """Content words from a question title: names first (any capital letter,
    e.g. "iPhone"), then numbers that are not years, then other words."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9&\-']*", text or "")
    names, numbers, others = [], [], []
    for token in tokens:
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        if token.isdigit():
            if not (len(token) == 4 and 1900 <= int(token) <= 2100):
                numbers.append(token)
        elif len(token) > 2:
            (names if any(c.isupper() for c in token) else others).append(token)
    ordered: list[str] = []
    for token in names + numbers + others:
        if token.lower() not in {o.lower() for o in ordered}:
            ordered.append(token)
    return ordered[:count]


# ------------------------------------------------------------------ urls ----
def _balance_parentheses(url: str) -> str:
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
    return url


def extract_urls(texts: Iterable[str | None], limit: int) -> list[str]:
    found: list[str] = []
    for text in texts:
        for match in _URL.findall(text or ""):
            url = _balance_parentheses(match.rstrip(".,;:!?*_'\""))
            if url.lower().endswith(_SKIPPED_EXTENSIONS):
                continue
            if url not in found:
                found.append(url)
    return found[:limit]


# ----------------------------------------------------------------- pages ----
def _decode(raw: bytes, declared: str | None) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(declared or "latin-1", errors="replace")


def _fetch_direct(url: str, timeout: float) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.8",
        },
        timeout=timeout,
        stream=True,
    )
    response.raise_for_status()
    raw = response.raw.read(MAX_PAGE_BYTES, decode_content=True)
    content_type = response.headers.get("content-type", "").lower()
    body = _decode(raw, response.encoding)
    if "json" in content_type:
        try:
            return json.dumps(json.loads(body), indent=1, ensure_ascii=False)
        except ValueError:
            return body
    if "html" in content_type or body.lstrip()[:1] == "<":
        return html_to_text(body)
    if content_type.startswith("text/") or "csv" in content_type:
        return body
    return ""  # PDFs and other binaries: left to the reader service


def _fetch_via_reader(url: str, timeout: float) -> str:
    response = requests.get(
        READER_PREFIX + url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/plain"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def fetch_page_text(url: str, timeout: float = 20) -> str:
    text = ""
    try:
        text = _fetch_direct(url, timeout)
    except Exception as error:
        logger.info(f"Direct fetch failed for {url}: {error}")
    if len(text.strip()) < MIN_USEFUL_PAGE_CHARS:
        try:
            text = _fetch_via_reader(url, timeout * 2) or text
        except Exception as error:
            logger.info(f"Reader fetch failed for {url}: {error}")
    return _clean_whitespace(text)


def resolution_source_digest(
    urls: list[str], query: str, per_page_limit: int
) -> str:
    sections = []
    for url in urls:
        text = fetch_page_text(url)
        if len(text) < 40:
            continue
        excerpt = select_relevant_passages(text, query, per_page_limit)
        sections.append(f"### {url}\n{excerpt}")
    return "\n\n".join(sections)


# --------------------------------------------------------------- metaculus ----
def _format_resolution(value) -> str:
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value[:10] if re.match(r"\d{4}-\d{2}-\d{2}T", value) else value
        return f"{number:.6g}"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def _format_resolved_post(post: dict) -> str | None:
    question = post.get("question")
    if not isinstance(question, dict):
        return None
    resolution = question.get("resolution")
    if resolution in (None, "", "annulled", "ambiguous"):
        return None
    title = (post.get("title") or question.get("title") or "").strip()
    if not title:
        return None
    resolved_on = str(
        question.get("actual_resolve_time") or question.get("scheduled_resolve_time") or ""
    )[:10]
    question_type = question.get("type", "")
    return f"- {title} → {_format_resolution(resolution)} ({question_type}, resolved {resolved_on})"


def similar_resolved_questions(
    post_id: int | None,
    title: str,
    token: str | None,
    limit: int = 8,
    timeout: float = 20,
) -> str:
    """Resolved Metaculus questions most similar to this one (by Metaculus'
    own similarity ranking, falling back to its search)."""
    headers = {"Authorization": f"Token {token}"} if token else {}
    attempts: list[dict] = []
    if post_id:
        attempts.append({"similar_to_post_id": post_id})
    if title:
        attempts.append({"search": title})
    for extra in attempts:
        params = {"statuses": "resolved", "limit": limit * 2, **extra}
        try:
            response = requests.get(
                METACULUS_POSTS_URL, params=params, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as error:
            logger.info(f"Metaculus similar-question lookup failed ({extra}): {error}")
            continue
        lines = [line for line in map(_format_resolved_post, results) if line]
        if lines:
            return "\n".join(lines[:limit])
    return ""


# ---------------------------------------------------------------- markets ----
def _millis_to_date(value) -> str:
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return "?"


def manifold_markets(
    title: str, binary_only: bool, limit: int = 5, timeout: float = 15
) -> str:
    term = " ".join(top_keywords(title, 6))
    if not term:
        return ""
    params = {"term": term, "limit": 20, "filter": "open"}
    if binary_only:
        params["contractType"] = "BINARY"
    try:
        response = requests.get(
            MANIFOLD_SEARCH_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        markets = response.json()
    except Exception as error:
        logger.info(f"Manifold search failed: {error}")
        return ""
    lines = []
    for market in markets if isinstance(markets, list) else []:
        if not isinstance(market, dict) or (market.get("uniqueBettorCount") or 0) < 10:
            continue
        outcome_type = market.get("outcomeType")
        if outcome_type == "BINARY" and isinstance(market.get("probability"), (int, float)):
            current = f"{market['probability']:.0%}"
        elif outcome_type == "PSEUDO_NUMERIC" and isinstance(market.get("value"), (int, float)):
            current = f"expected value {market['value']:.4g}"
        else:
            continue
        lines.append(
            f"- {market.get('question', '').strip()}: {current} "
            f"({market.get('uniqueBettorCount')} traders, closes "
            f"{_millis_to_date(market.get('closeTime'))}) {market.get('url', '')}"
        )
        if len(lines) >= limit:
            break
    return "\n".join(lines)
