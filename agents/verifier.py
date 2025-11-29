# agents/verifier.py
"""
TruthPulse verifier - cleaned and user-friendly output.

This file is based on your previous implementation with improvements:
- Robust claim extraction
- Cleaner human-readable formatter (format_clean_report)
- No debug injection into final message
- Aviation/defense targeted final filtering when claim indicates those domains
- Quote extraction helper retained
"""

import os
import time
import asyncio
import json
import textwrap
import re
import pickle
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urljoin, urlparse
from html import unescape
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from langdetect import detect
import numpy as np

# Optional NewsAPI
try:
    from newsapi import NewsApiClient
except Exception:
    NewsApiClient = None

# ---------------------------
# Config / env
# ---------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
VERBOSE = os.getenv("VERBOSE", "0") == "1"
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.20"))
EMBED_CACHE_FILE = os.getenv("EMBEDDINGS_CACHE_FILE", "./emb_cache.pkl")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

# ---------------------------
# Simple in-memory TTL cache
# ---------------------------
_CACHE: Dict[str, Dict[str, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    itm = _CACHE.get(key)
    if not itm:
        return None
    if time.time() - itm["ts"] > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return itm["value"]


def _cache_set(key: str, value: Any) -> None:
    _CACHE[key] = {"ts": time.time(), "value": value}


# ---------------------------
# Embeddings cache (file-backed)
# ---------------------------
_EMBED_CACHE: Dict[str, List[float]] = {}
_EMBED_CACHE_LOADED = False


def _load_embed_cache():
    global _EMBED_CACHE_LOADED, _EMBED_CACHE
    if _EMBED_CACHE_LOADED:
        return
    try:
        if os.path.exists(EMBED_CACHE_FILE):
            with open(EMBED_CACHE_FILE, "rb") as f:
                _EMBED_CACHE = pickle.load(f) or {}
    except Exception:
        _EMBED_CACHE = {}
    _EMBED_CACHE_LOADED = True


def _save_embed_cache():
    try:
        with open(EMBED_CACHE_FILE, "wb") as f:
            pickle.dump(_EMBED_CACHE, f)
    except Exception:
        pass


def _embed_cache_key(text: str) -> str:
    return str(hash(text.strip()[:1024]))


# ---------------------------
# Defence RSS feeds (India-focused)
# ---------------------------
DEFENCE_RSS_FEEDS = [
    "https://www.aninews.in/rss/ani_topstories.xml",
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://www.hindustantimes.com/rss/topnews/rssfeed.xml",
    "https://indianexpress.com/section/india/feed/"
]


async def fetch_defence_rss(max_results_per_feed: int = 5) -> List[Dict[str, Any]]:
    cache_key = f"defence_rss::{max_results_per_feed}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    results: List[Dict[str, Any]] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TruthPulse/1.0)"}
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
            for feed in DEFENCE_RSS_FEEDS:
                try:
                    r = await client.get(feed, follow_redirects=True)
                    if r.status_code != 200:
                        continue
                    soup = BeautifulSoup(r.text, "xml")
                    items = soup.find_all("item")[:max_results_per_feed]
                    for it in items:
                        title = (it.title.text if it.title else "").strip()
                        link = (it.link.text if it.link else "") or (it.find("guid").text if it.find("guid") else "")
                        pub = it.find("pubDate").text if it.find("pubDate") else ""
                        desc = (it.description.text if it.description else "")[:400]
                        results.append({
                            "type": "defence_rss",
                            "title": title,
                            "url": link,
                            "snippet": desc,
                            "date": pub,
                            "source_feed": feed
                        })
                except Exception:
                    continue
    except Exception:
        # if httpx fails entirely, return empty list
        pass
    # dedupe
    out, seen = [], set()
    for r in results:
        key = (r.get("url") or r.get("title"))[:400]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    _cache_set(cache_key, out)
    return out


# ---------------------------
# Other fetchers
# (kept similar to previous, each wrapped in try/except)
# ---------------------------

async def fetch_google_factchecks(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if not GOOGLE_API_KEY:
        return []
    cache_key = f"gfc::{query}::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": GOOGLE_API_KEY, "pageSize": max_results}
    results = []
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(url, params=params)
            if r.status_code != 200:
                _cache_set(cache_key, results)
                return results
            data = r.json()
    except Exception:
        _cache_set(cache_key, results)
        return results
    for it in data.get("claims", []):
        cr = (it.get("claimReview") or [])
        first = cr[0] if cr else {}
        pub = (first.get("publisher") or {}) or {}
        results.append({
            "type": "google_factcheck",
            "title": first.get("title") or it.get("text", "")[:250],
            "url": first.get("url") or "",
            "review_date": first.get("reviewDate") or "",
            "review_author": pub.get("name") or "",
            "rating": first.get("textualRating") or "",
            "text": it.get("text", "")
        })
    _cache_set(cache_key, results)
    return results


async def fetch_pib_factcheck(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    cache_key = f"pib::{query}::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    base = "https://pib.gov.in"
    search_url = f"https://pib.gov.in/SearchRelease.aspx?Term={quote_plus(query)}"
    results = []
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            r = await client.get(search_url)
            if r.status_code != 200:
                _cache_set(cache_key, results)
                return results
            soup = BeautifulSoup(r.text, "lxml")
            anchors = soup.select("a[href*='Release']")[:max_results]
            if not anchors:
                anchors = soup.select(".searchresults a")[:max_results]
            for a in anchors:
                href = a.get("href")
                title = a.get_text(strip=True) or a.get("title", "")
                if href:
                    full = urljoin(base, href)
                    results.append({"type": "pib", "title": title, "url": full})
    except Exception:
        pass
    _cache_set(cache_key, results)
    return results


async def fetch_factly(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    cache_key = f"factly::{query}::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    url = f"https://factly.in/?s={quote_plus(query)}"
    results = []
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(url)
            if r.status_code != 200:
                _cache_set(cache_key, results)
                return results
            soup = BeautifulSoup(r.text, "lxml")
            posts = soup.select("article h2 a")[:max_results]
            for p in posts:
                results.append({"type": "factly", "title": p.get_text(strip=True), "url": p.get("href")})
    except Exception:
        pass
    _cache_set(cache_key, results)
    return results


async def fetch_boom_live(max_results: int = 3) -> List[Dict[str, Any]]:
    cache_key = f"boom::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    url = "https://www.boomlive.in/rss/feed/fact-check"
    results = []
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(url)
            if r.status_code != 200:
                _cache_set(cache_key, results)
                return results
            soup = BeautifulSoup(r.text, "xml")
            items = soup.find_all("item")[:max_results]
            for it in items:
                results.append({"type": "boomlive", "title": it.title.text, "url": it.link.text})
    except Exception:
        pass
    _cache_set(cache_key, results)
    return results


async def fetch_afp(max_results: int = 5) -> List[Dict[str, Any]]:
    cache_key = f"afp::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    url = "https://factcheck.afp.com/rss"
    results = []
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(url)
            if r.status_code != 200:
                _cache_set(cache_key, results)
                return results
            soup = BeautifulSoup(r.text, "xml")
            items = soup.find_all("item")[:max_results]
            for it in items:
                results.append({"type": "afp", "title": it.title.text, "url": it.link.text})
    except Exception:
        pass
    _cache_set(cache_key, results)
    return results


async def fetch_politifact(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    cache_key = f"politifact::{query}::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    url = f"https://www.politifact.com/api/v2/search/?q={quote_plus(query)}"
    results = []
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                for item in (data or [])[:max_results]:
                    results.append({"type": "politifact", "title": item.get("title", ""), "url": "https://www.politifact.com" + item.get("url", "")})
    except Exception:
        pass
    _cache_set(cache_key, results)
    return results


async def fetch_reuters(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    cache_key = f"reuters::{query}::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    base = "https://www.reuters.com"
    search_url = f"https://www.reuters.com/site-search/?query={quote_plus(query)}"
    results = []
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(search_url)
            if r.status_code != 200:
                _cache_set(cache_key, results)
                return results
            soup = BeautifulSoup(r.text, "lxml")
            anchors = soup.select("a[href^='/']")[:max_results*3]
            seen = set()
            for a in anchors:
                href = a.get("href")
                title = a.get_text(strip=True)
                if not href or not title:
                    continue
                if href.startswith("/pictures") or href.startswith("/markets"):
                    continue
                full = urljoin(base, href)
                if full in seen:
                    continue
                seen.add(full)
                results.append({"type": "reuters", "title": title, "url": full})
                if len(results) >= max_results:
                    break
    except Exception:
        pass
    _cache_set(cache_key, results)
    return results


async def fetch_google_news(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    cache_key = f"gnews::{query}::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    results = []
    search_url = f"https://news.google.com/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        async with httpx.AsyncClient(timeout=12.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(search_url)
            if r.status_code != 200:
                _cache_set(cache_key, results)
                return results
            soup = BeautifulSoup(r.text, "lxml")
            anchors = soup.select("article a[href]")[:max_results*2]
            seen = set()
            for a in anchors:
                href = a.get("href")
                if not href:
                    continue
                if href.startswith("./"):
                    href = href.replace("./", "https://news.google.com/")
                if href.startswith("/"):
                    href = "https://news.google.com" + href
                title = a.get_text(strip=True)
                if not title or title in seen:
                    continue
                seen.add(title)
                parent = a.find_parent("article")
                source = ""
                snippet = ""
                pub_date = ""
                if parent:
                    src_el = parent.select_one("time")
                    if src_el and src_el.has_attr("datetime"):
                        pub_date = src_el["datetime"]
                    source_el = parent.select_one("div[role='heading']") or parent.select_one(".SVJrMe")
                    if source_el:
                        source = source_el.get_text(strip=True)
                    snippet_el = parent.select_one(".xBbh9")
                    if snippet_el:
                        snippet = snippet_el.get_text(strip=True)
                results.append({"type": "google_news", "title": title, "url": href, "snippet": snippet, "source": source, "date": pub_date})
                if len(results) >= max_results:
                    break
    except Exception:
        pass
    _cache_set(cache_key, results)
    return results


async def fetch_newsapi(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    if not NEWSAPI_KEY or NewsApiClient is None:
        return []
    cache_key = f"newsapi::{query}::{max_results}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    results = []
    try:
        client = NewsApiClient(api_key=NEWSAPI_KEY)
        resp = client.get_everything(q=query, language="en", sort_by="relevancy", page_size=max_results)
        articles = resp.get("articles", []) or []
        for a in articles[:max_results]:
            results.append({"type": "newsapi", "title": a.get("title", ""), "url": a.get("url", ""), "snippet": a.get("description", ""), "source": a.get("source", {}).get("name", "")})
    except Exception:
        pass
    _cache_set(cache_key, results)
    return results


# ---------------------------
# Embeddings + similarity
# ---------------------------

async def embed_text(text: str) -> Optional[np.ndarray]:
    if not OPENAI_KEY:
        return None
    _load_embed_cache()
    key = _embed_cache_key(text)
    if key in _EMBED_CACHE:
        try:
            return np.array(_EMBED_CACHE[key], dtype=float)
        except Exception:
            pass
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        def sync_call():
            return client.embeddings.create(input=text, model="text-embedding-3-small")
        resp = await asyncio.to_thread(sync_call)
        vec = resp.data[0].embedding
        _EMBED_CACHE[key] = list(vec)
        try:
            _save_embed_cache()
        except Exception:
            pass
        return np.array(vec, dtype=float)
    except Exception:
        return None


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    try:
        if a is None or b is None:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except Exception:
        return 0.0


# ---------------------------
# Claim extraction (robust)
# ---------------------------

async def extract_claim(raw_text: str) -> str:
    """
    Clean, safe claim extractor.
    - Remove raw ChatCompletion dumps and wrappers if present.
    - If it looks clean already, return it.
    - Otherwise attempt OpenAI extraction (safe fallback).
    """
    if not raw_text:
        return raw_text

    # quick sanitizers: remove likely ChatCompletion artifacts
    junk_patterns = [
        r"Choice\([^)]*\)",
        r"ChatCompletionMessage\([^)]*\)",
        r"finish_reason\s*=\s*'[^']*'",
        r"role\s*=\s*'assistant'",
        r"function_call\=[\s\S]*"
    ]
    cleaned = raw_text
    for pat in junk_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()
    # If cleaned is already a good sentence (>=4 words), return
    if len(cleaned.split()) >= 4 and len(cleaned) < 1000:
        return cleaned[:500]

    # Otherwise try using OpenAI to extract a single concise claim
    if not OPENAI_KEY:
        return raw_text[:500]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        system_prompt = "Extract the single factual claim from the user's input. Output only the claim sentence, nothing else."
        def sync_call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_text}
                ],
                temperature=0.0,
                max_tokens=80
            )
        resp = await asyncio.to_thread(sync_call)
        # robust extraction logic for different response shapes
        try:
            choices = getattr(resp, "choices", None) or resp.get("choices", [])
            first = choices[0]
            msg = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
            if isinstance(msg, dict):
                text = msg.get("content", "")
            else:
                text = getattr(first, "text", None) or str(first)
            text = text.strip()
            # sanitize again
            for pat in junk_patterns:
                text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
            if text and len(text.split()) >= 3:
                return text[:500]
        except Exception:
            pass
    except Exception:
        pass

    # fallback to original raw text (trimmed)
    return raw_text[:500]


# ---------------------------
# Improved structured synthesis requiring evidence mapping (keeps original prompt)
# ---------------------------

async def call_openai_synthesis_structured(claim: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not OPENAI_KEY:
        return {"verdict": "Not Found", "explanation": "No OPENAI_KEY set.", "supported_facts": [], "unsupported_facts": [], "missing_evidence": [], "sources": [], "confidence": 0.0}
    prompt_sources = []
    for s in sources[:12]:
        prompt_sources.append({
            "title": s.get("title") or s.get("review_title") or "",
            "url": s.get("url") or s.get("review_url") or "",
            "snippet": (s.get("snippet") or s.get("text") or "")[:320]
        })
    system_msg = textwrap.dedent("""
        You are TruthPulse, a rigorous fact-check assistant. Output MUST be valid JSON only. 
        Given a claim and candidate sources, do the following:
        1) Break the claim into sub-facts (short sentences) as needed.
        2) For each sub-fact, determine if it's SUPPORTED, CONTRADICTED, or NOT FOUND in the sources.
        3) For each SUPPORTED or CONTRADICTED sub-fact, list an evidence array with objects {title, url, publisher(optional), date(optional), quote} where 'quote' is a short sentence excerpt from the source that directly supports/contradicts the sub-fact. If the exact quote isn't present, use the best short paraphrase and mark it as 'paraphrase'.
        4) Provide a 'missing_evidence' array listing what authoritative document or statement would definitively confirm or deny.
        5) Return a single verdict in [\"Verified\",\"False\",\"Misleading\",\"Partially True\",\"Not Found\"], a brief explanation (1-2 sentences), and a numeric confidence between 0.0 and 1.0.
        Output JSON schema example:
        {
          "verdict":"Partially True",
          "explanation":"Short explanation",
          "supported_facts":[ {"fact":"...", "evidence":[{"title":"...","url":"...","quote":"..."}] }, ...],
          "unsupported_facts":[ {...} ],
          "missing_evidence":[ "..." ],
          "sources":[ {...} ],
          "confidence":0.72
        }
        Be conservative: only mark as SUPPORTED if a source explicitly states (or quotes an authority) the sub-fact. If a source is only related context, mark that sub-fact as MISSING.
    """).strip()
    user_msg = f"Claim: {claim}\n\nCandidate sources:\n"
    if prompt_sources:
        for p in prompt_sources:
            user_msg += f"- {p['title']} | {p['url']} | {p['snippet']}\n"
    else:
        user_msg += "(no matched sources)\n"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        def sync_call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1000,
                temperature=0.0,
            )
        resp = await asyncio.to_thread(sync_call)
        # extract text robustly
        def extract_text(r):
            try:
                if isinstance(r, dict):
                    ch = r.get("choices", [])
                    if ch:
                        first = ch[0]
                        msg = first.get("message") or first.get("text") or {}
                        if isinstance(msg, dict):
                            return msg.get("content", "")
                        return str(msg)
                choices = getattr(r, "choices", None)
                if choices:
                    first = choices[0]
                    msg = getattr(first, "message", None)
                    if msg:
                        c = getattr(msg, "content", None)
                        if c:
                            return c
                    txt = getattr(first, "text", None)
                    if txt:
                        return txt
            except Exception:
                pass
            return str(r)
        raw = extract_text(resp).strip()
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"(\{[\s\S]*\})", raw)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = None
        if not isinstance(parsed, dict):
            # fallback: return raw text as explanation
            return {"verdict": "Not Found", "explanation": raw[:800], "supported_facts": [], "unsupported_facts": [], "missing_evidence": [], "sources": [], "confidence": 0.0}
        # sanitize and defaults
        verdict = parsed.get("verdict", "Not Found")
        explanation = parsed.get("explanation", "") or ""
        supported = parsed.get("supported_facts", []) or []
        unsupported = parsed.get("unsupported_facts", []) or []
        missing = parsed.get("missing_evidence", []) or []
        out_sources = parsed.get("sources", []) or []
        confidence = parsed.get("confidence", parsed.get("Confidence", 0.0))
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.0
        # normalize sources
        norm_sources = []
        for s in out_sources:
            if isinstance(s, str):
                parts = [p.strip() for p in s.split("|")]
                if len(parts) >= 2 and parts[1].startswith("http"):
                    norm_sources.append({"title": parts[0], "url": parts[1], "publisher": "", "date": "", "snippet": ""})
                else:
                    norm_sources.append({"title": s, "url": "", "publisher": "", "date": "", "snippet": ""})
            elif isinstance(s, dict):
                norm_sources.append({
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "publisher": s.get("publisher", s.get("review_author", "")),
                    "date": s.get("date", s.get("review_date", "")),
                    "snippet": s.get("snippet", ""),
                    "rating": s.get("rating", "")
                })
        return {
            "verdict": verdict,
            "explanation": explanation,
            "supported_facts": supported,
            "unsupported_facts": unsupported,
            "missing_evidence": missing,
            "sources": norm_sources,
            "confidence": confidence
        }
    except Exception as exc:
        return {"verdict": "Not Found", "explanation": f"Error during synthesis: {exc}", "supported_facts": [], "unsupported_facts": [], "missing_evidence": [], "sources": [], "confidence": 0.0}


# ---------------------------
# Helper: fetch publish date and extract quote/sentence from URL
# ---------------------------

async def fetch_publish_date(url: str) -> Optional[str]:
    if not url or not url.startswith("http"):
        return None
    try:
        async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "Mozilla/5.0 (TruthPulse/1.0)"}) as client:
            r = await client.get(url, follow_redirects=True)
            if r.status_code != 200:
                return None
            soup = BeautifulSoup(r.text, "lxml")
            meta_keys = [
                ("meta", {"property": "article:published_time"}),
                ("meta", {"property": "og:article:published_time"}),
                ("meta", {"name": "pubdate"}),
                ("meta", {"name": "publication_date"}),
                ("meta", {"name": "article:published_time"}),
                ("time", {}),
            ]
            for tag, attrs in meta_keys:
                try:
                    if tag == "time":
                        t = soup.find("time")
                        if t and t.has_attr("datetime"):
                            return t["datetime"]
                        if t and t.text:
                            return t.text.strip()
                    else:
                        m = soup.find(tag, attrs=attrs)
                        if m and m.has_attr("content"):
                            return m["content"].strip()
                        if m and m.has_attr("value"):
                            return m["value"].strip()
                except Exception:
                    continue
            for m in soup.find_all("meta"):
                name = m.get("name", "").lower()
                if "date" in name or "pub" in name:
                    if m.get("content"):
                        return m.get("content").strip()
            return None
    except Exception:
        return None


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    return [p.strip() for p in parts if p.strip()]


async def extract_quote_from_url(url: str, phrases: List[str], max_chars: int = 240) -> Optional[str]:
    if not url or not url.startswith("http"):
        return None
    cache_key = f"quote::{url}::{'|'.join(phrases)[:120]}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        headers = {"User-Agent": "Mozilla/5.0 (TruthPulse/1.0)"}
        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            r = await client.get(url, follow_redirects=True)
            if r.status_code != 200:
                _cache_set(cache_key, None)
                return None
            text = r.text
            text = unescape(text)
            soup = BeautifulSoup(text, "lxml")
            body_text = ""
            article = soup.find("article")
            if article:
                body_text = article.get_text(separator=" ", strip=True)
            if not body_text:
                candidates = soup.select("div[class*='article'], div[class*='content'], div[class*='story'], section[class*='article']")
                for c in candidates:
                    body_text = c.get_text(separator=" ", strip=True)
                    if body_text and len(body_text) > 160:
                        break
            if not body_text:
                ps = soup.find_all("p")
                body_text = " ".join([p.get_text(strip=True) for p in ps[:12]])
            if not body_text:
                _cache_set(cache_key, None)
                return None
            sentences = split_into_sentences(body_text)
            low_phrases = [p.lower() for p in phrases if p and len(p) > 2]
            for sent in sentences:
                low = sent.lower()
                if any(p in low for p in low_phrases):
                    res = sent.strip()
                    if len(res) > max_chars:
                        res = res[:max_chars].rsplit(" ", 1)[0] + "..."
                    _cache_set(cache_key, res)
                    return res
            for sent in sentences:
                if len(sent) > 40:
                    res = sent.strip()
                    if len(res) > max_chars:
                        res = res[:max_chars].rsplit(" ", 1)[0] + "..."
                    _cache_set(cache_key, res)
                    return res
            res = body_text.strip()[:max_chars].rsplit(" ", 1)[0] + "..."
            _cache_set(cache_key, res)
            return res
    except Exception:
        _cache_set(cache_key, None)
        return None


# ---------------------------
# Authority weights & keywords
# ---------------------------

AUTHORITY_WEIGHTS = {
    "google_factcheck": 1.0,
    "afp": 1.0,
    "reuters": 0.95,
    "pib": 0.9,
    "boomlive": 0.85,
    "factly": 0.8,
    "politifact": 0.85,
    "newsapi": 0.9,
    "google_news": 0.7,
    "defence_rss": 0.92,
    "other": 0.5
}


def authority_weight(src_type: str) -> float:
    return AUTHORITY_WEIGHTS.get(src_type.lower(), AUTHORITY_WEIGHTS["other"])


REQ_KEYWORDS = [
    "india", "indian", "modi", "government", "defence", "defense",
    "army", "navy", "air force", "iaf", "drdo", "isro", "border", "loc", "ladakh",
    "pakistan", "china", "russia", "meeting", "talks", "minister",
    "emergency", "crisis", "attack", "terror", "terrorist", "blast", "bomb",
    "disaster", "earthquake", "flood", "cyclone", "explosion", "rescue", "evacuation", "alert",
    "missile", "s-500", "s-400", "brahmos", "agni", "prithvi", "tejas",
    "fighter", "jet", "warship", "submarine", "radar",
    "intelligence", "raw", "ib", "threat", "infiltration", "espionage", "ceasefire", "hostile", "mobilisation",
    "airbus", "a320", "airline", "flight", "flights", "dgca", "faa", "easa", "airport", "delay", "cancellation"
]


def contains_required_keywords(title: str) -> bool:
    if not title:
        return False
    t = title.lower()
    for k in REQ_KEYWORDS:
        if k in t:
            return True
    return False


# ---------------------------
# Clean, human-friendly formatter (final output)
# ---------------------------

def _strip_json_like_blocks(text: str) -> str:
    if not text:
        return text
    # remove large {...} JSON blocks likely inserted by LLM
    def repl(m):
        inner = m.group(0)
        return "" if len(inner) > 200 else inner
    text = re.sub(r"\{[\s\S]*?\}", repl, text)
    # remove any debug markers
    text = re.sub(r"---\s*DEBUG[\s\S]*?---\s*end debug\s*---?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text.strip()


def _shorten_url(url: str, max_len: int = 60) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url)
        domain = p.netloc or url
        path = p.path or ""
        if path and path != "/":
            display = domain + path
        else:
            display = domain
        if len(display) > max_len:
            display = display[:max_len].rsplit("/", 1)[0] + "/…"
        return display
    except Exception:
        if len(url) > max_len:
            return url[:max_len-3] + "..."
        return url


def _normalize_date(d: Optional[str]) -> str:
    if not d:
        return "unknown"
    try:
        dt = None
        try:
            dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
        except Exception:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(d)
        if dt:
            return dt.strftime("%b %d, %Y")
    except Exception:
        pass
    s = d.strip()
    return s[:20] + ("..." if len(s) > 20 else "")


def format_clean_report(
    claim: str,
    verdict: str,
    explanation: str,
    sources: List[Dict[str, Any]],
    confidence: float,
    language: str,
    supported: List[Dict[str, Any]],
    unsupported: List[Dict[str, Any]],
    missing: List[str]
) -> str:
    clean_expl = _strip_json_like_blocks(explanation or "")
    if not clean_expl:
        clean_expl = "No concise explanation available."

    # Helper: turn fact-items into unique ordered short strings
    def top_facts_list(items):
        if not items:
            return ["—"]
        seen = set()
        out = []
        for it in items:
            # it may be dict {"fact": "..."} or string
            fact_text = it.get("fact") if isinstance(it, dict) else str(it)
            fact_text = " ".join(fact_text.splitlines()).strip()
            if not fact_text:
                continue
            # Normalize whitespace and lower for dedupe key
            key = re.sub(r"\s+", " ", fact_text).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(fact_text[:200])
            if len(out) >= 3:
                break
        if not out:
            return ["—"]
        return out

    supported_list = top_facts_list(supported)
    unsupported_list = top_facts_list(unsupported)
    missing_list = (missing or [])[:4]

    # choose top 3 sources by semantic_score (stable)
    def score_key(s):
        return s.get("semantic_score", 0.0)
    sorted_srcs = sorted(sources, key=score_key, reverse=True)[:3]

    src_lines = []
    for s in sorted_srcs:
        title = s.get("title") or s.get("name") or "Source"
        url = s.get("url") or ""
        pub = s.get("publisher") or s.get("source") or s.get("review_author") or "unknown"
        date = _normalize_date(s.get("date") or s.get("review_date") or s.get("pubDate") or "")
        src_lines.append({
            "title": title if len(title) <= 140 else title[:137] + "...",
            "url": url,
            "display": _shorten_url(url),
            "pub": pub,
            "date": date
        })

    if confidence >= 0.8:
        conf_emoji = "🟩"
        conf_label = "High"
    elif confidence >= 0.5:
        conf_emoji = "🟨"
        conf_label = "Medium"
    else:
        conf_emoji = "🟥"
        conf_label = "Low"

    lines = []
    lines.append("🧾 TruthPulse — Fact Check Report\n")
    lines.append(f"📌 Claim:\n\"{claim}\"\n")
    lines.append(f"🟦 Verdict: {verdict}")
    lines.append(f"Short reason: {clean_expl}\n")

    lines.append("📰 Summary:")
    if supported_list and supported_list != ["—"]:
        lines.append("Supported findings:")
        for f in supported_list:
            lines.append(f"• {f}")
    if unsupported_list and unsupported_list != ["—"]:
        lines.append("Not supported / contradicted:")
        for f in unsupported_list:
            lines.append(f"• {f}")
    if (not supported_list or supported_list == ["—"]) and (not unsupported_list or unsupported_list == ["—"]):
        lines.append("• No concrete supporting or contradicting facts were found in top sources.\n")

    if missing_list:
        lines.append("\nWhat would confirm/deny this claim:")
        for m in missing_list:
            lines.append(f"• {m}")

    lines.append("\n🔗 Top sources:")
    if not src_lines:
        lines.append("• No relevant sources found.")
    else:
        for s in src_lines:
            lines.append(f"• {s['title']}")
            lines.append(f"  ({s['pub']}, {s['date']})")
            if s['display']:
                lines.append(f"  {s['display']}")
            else:
                lines.append(f"  {s['url']}")
    lines.append(f"\n📊 Confidence: {conf_emoji} {confidence:.2f} ({conf_label})")
    lines.append("\nNote: For urgent or life-safety claims, verify with official agencies (DGCA, FAA, EASA, PIB, MoD).")

    out = "\n".join(lines)
    out = re.sub(r"\n{2,}", "\n\n", out).strip()
    return out



# ---------------------------
# normalize_sources helper
# ---------------------------

def normalize_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for s in sources:
        key = (s.get("url") or s.get("title") or "")[:300]
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ---------------------------
# Main orchestrator: verify_claim
# ---------------------------

async def verify_claim(text_or_url: str) -> str:
    if not text_or_url or not text_or_url.strip():
        return "Please send a text or link to verify."
    user_lang = await detect_language(text_or_url)
    try:
        claim = await extract_claim(text_or_url.strip())
    except Exception:
        claim = text_or_url.strip()[:500]
    cache_key = f"verify::{claim}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # make variants
    variants = [claim]
    words = [w for w in re.split(r"\W+", claim) if len(w) > 3]
    if words:
        variants.append(" ".join(words[:6]))
    # attempt Hindi variant
    if OPENAI_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            def sync_call():
                return client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Translate the following sentence into Hindi. Output only the translation."},
                        {"role": "user", "content": claim}
                    ],
                    max_tokens=120,
                    temperature=0.0
                )
            resp = await asyncio.to_thread(sync_call)
            try:
                choices = getattr(resp, "choices", None) or resp.get("choices", [])
                first = choices[0]
                msg = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
                htext = ""
                if isinstance(msg, dict):
                    htext = msg.get("content", "")
                else:
                    htext = getattr(first, "text", None) or str(first)
                htext = htext.strip()
                if htext and len(htext) > 3:
                    variants.append(htext)
            except Exception:
                pass
        except Exception:
            pass
    variants = list(dict.fromkeys(variants))[:3]

    # build fetch tasks
    fetch_tasks = []
    fetch_tasks.append(asyncio.create_task(fetch_defence_rss()))
    fetch_tasks.append(asyncio.create_task(fetch_boom_live()))
    fetch_tasks.append(asyncio.create_task(fetch_afp()))
    for v in variants:
        fetch_tasks.append(asyncio.create_task(fetch_google_factchecks(v)))
        fetch_tasks.append(asyncio.create_task(fetch_pib_factcheck(v)))
        fetch_tasks.append(asyncio.create_task(fetch_factly(v)))
        fetch_tasks.append(asyncio.create_task(fetch_politifact(v)))
        fetch_tasks.append(asyncio.create_task(fetch_reuters(v)))
        fetch_tasks.append(asyncio.create_task(fetch_google_news(v)))
        if NEWSAPI_KEY:
            fetch_tasks.append(asyncio.create_task(fetch_newsapi(v)))
    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    sources: List[Dict[str, Any]] = []
    for res in results:
        if isinstance(res, list):
            sources.extend(res)
    sources = normalize_sources(sources)
    raw_fetched = list(sources)

    # semantic scoring
    query_vec = await embed_text(claim)
    final_sources: List[Dict[str, Any]] = []
    if query_vec is not None and sources:
        scored = []
        for s in sources:
            title = s.get("title") or s.get("review_title") or s.get("text", "") or s.get("snippet", "")
            src_vec = await embed_text(title[:800])
            score = cosine_similarity(query_vec, src_vec) if src_vec is not None else 0.0
            s["semantic_score"] = float(score)
            s["authority"] = authority_weight(s.get("type", "other"))
            scored.append(s)
        filtered = [s for s in scored if s.get("semantic_score", 0.0) >= SEMANTIC_THRESHOLD and contains_required_keywords(s.get("title", ""))]
        filtered.sort(key=lambda x: (x.get("semantic_score", 0.0) * x.get("authority", 0.5)), reverse=True)
        final_sources = filtered

    # additional domain narrowing: if claim mentions aviation/airbus/dgca -> keep only aviation-related titles
    claim_lower = claim.lower()
    aviation_terms = ["airbus", "a320", "airline", "flight", "flights", "dgca", "faa", "easa", "airport", "delay", "cancellation", "software"]
    if any(t in claim_lower for t in aviation_terms) and final_sources:
        def aviation_filter(s):
            title = (s.get("title") or "").lower()
            snippet = (s.get("snippet") or "").lower()
            for t in aviation_terms:
                if t in title or t in snippet:
                    return True
            return False
        av = [s for s in final_sources if aviation_filter(s)]
        if av:
            final_sources = av

    # fallback to defence_rss if none
    explanation = ""
    if not final_sources:
        defence_items = [s for s in sources if s.get("type") == "defence_rss" and contains_required_keywords(s.get("title", ""))]
        if defence_items:
            final_sources = defence_items[:6]
            explanation = "Related defence news items found from RSS feeds; these are not direct fact-checks."

    # compute prior_score
    if final_sources:
        num = sum((s.get("semantic_score", 0.0) * s.get("authority", 0.5)) for s in final_sources)
        den = sum((s.get("authority", 0.5)) for s in final_sources) or 1.0
        prior_score = float(num / den)
    else:
        prior_score = 0.0

    # LLM synthesis requiring evidence mapping
    synth = await call_openai_synthesis_structured(claim, final_sources)
    verdict = synth.get("verdict", "Not Found")
    explanation_llm = synth.get("explanation", "") or ""
    supported = synth.get("supported_facts", []) or []
    unsupported = synth.get("unsupported_facts", []) or []
    missing = synth.get("missing_evidence", []) or []
    synth_sources = synth.get("sources", []) or []
    llm_conf = float(synth.get("confidence", 0.0))

    # If LLM returned sources, use them; else convert final_sources for display
    display_sources = synth_sources if synth_sources else []
    if not display_sources and final_sources:
        for s in final_sources:
            display_sources.append({
                "title": s.get("title") or s.get("review_title") or "",
                "url": s.get("url") or s.get("review_url") or "",
                "publisher": s.get("source") or s.get("review_author") or "",
                "date": s.get("date") or s.get("review_date") or "",
                "snippet": s.get("snippet") or s.get("text") or "",
                "rating": s.get("rating", ""),
                "semantic_score": s.get("semantic_score", 0.0)
            })

    # Enrich evidence quotes (best-effort)
    phrase_tokenizer = lambda s: [w for w in re.split(r"\W+", s) if len(w) > 3][:12]
    async_tasks = []
    for fact_entry in (supported + unsupported):
        evs = fact_entry.get("evidence", []) or []
        for ev in evs:
            if ev.get("quote"):
                continue
            url = ev.get("url") or ""
            phrases = phrase_tokenizer(fact_entry.get("fact") or "")
            if url:
                async_tasks.append((ev, url, phrases))
    async def _run_quote_tasks(tasks):
        sem = asyncio.Semaphore(6)
        async def _task(ev, url, phrases):
            async with sem:
                q = await extract_quote_from_url(url, phrases)
                if q:
                    ev["quote"] = q
                else:
                    ev["quote"] = ""
        return await asyncio.gather(*( _task(ev, url, phrases) for (ev, url, phrases) in tasks ), return_exceptions=True)
    if async_tasks:
        await _run_quote_tasks(async_tasks)

    # If no supported/unsupported facts returned by LLM, create a synthetic mapping from display_sources
    if not supported and not unsupported:
        synth_supported = []
        phrases = phrase_tokenizer(claim)
        for s in display_sources[:6]:
            url = s.get("url","")
            title = s.get("title","")
            quote = None
            if url:
                quote = await extract_quote_from_url(url, phrases)
            ev = {"title": title, "url": url, "quote": quote or s.get("snippet","")[:240]}
            synth_supported.append({"fact": claim, "evidence": [ev]})
        supported = synth_supported

    # contradiction detection & combined confidence
    contradictions = False
    ratings = {}
    for s in display_sources:
        r = (s.get("rating") or "").lower()
        if r:
            ratings.setdefault(r, 0)
            ratings[r] += 1
    if len(ratings.keys()) > 1:
        contradictions = True
    combined = 0.6 * llm_conf + 0.4 * prior_score
    if contradictions:
        combined *= 0.75
        if explanation_llm:
            explanation_llm = explanation_llm + " Note: conflicting findings among sources."
        else:
            explanation_llm = "Conflicting findings among sources."
    explanation_final = explanation_llm or explanation or ("No verified sources found for this claim." if not display_sources else "")
    combined = max(0.0, min(1.0, combined))
    if not display_sources:
        combined = 0.0

    # Final formatted output (clean, human-friendly)
    final_text = format_clean_report(claim, verdict, explanation_final, display_sources, combined, user_lang, supported, unsupported, missing)
    if user_lang in ["hi", "hinglish"]:
        final_text = await translate_output(final_text, target_lang=user_lang)
    _cache_set(cache_key, final_text)
    return final_text


# ---------------------------
# Language detect & translate (reuse previous)
# ---------------------------

async def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "hi":
            return "hi"
        if lang == "en":
            hindi_words = ["hai", "nahi", "kaise", "kyu", "kya", "haan", "mera", "tera"]
            if any(w in text.lower() for w in hindi_words):
                return "hinglish"
            return "en"
    except Exception:
        return "en"
    return "en"


async def translate_output(text: str, target_lang: str) -> str:
    if target_lang == "en" or not OPENAI_KEY:
        return text
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        def sync_call():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Translate the following text into {target_lang}. Keep formatting identical."},
                    {"role": "user", "content": text}
                ],
                max_tokens=800,
                temperature=0.2
            )
        resp = await asyncio.to_thread(sync_call)
        try:
            choices = getattr(resp, "choices", None) or resp.get("choices", [])
            first = choices[0]
            msg = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
            if isinstance(msg, dict):
                return msg.get("content", "").strip()
            return str(first)
        except Exception:
            return str(resp)
    except Exception:
        return text
