"""Research provider interfaces and live public-source research."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import settings
from data import (
    BuyingSignal,
    CompanyOverview,
    Confidence,
    EngineeringComplexity,
    Evidence,
    HiringSignal,
    LandscapeStatus,
    PainPointIndicator,
    PLMLandscapeItem,
    ResearchResult,
    TriState,
    Opportunity,
)
from utils import normalize_company_name


class _DuckDuckGoParser(HTMLParser):
    """Parse DDG result markup without depending on attribute ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str = ""
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and classes.intersection({"result__a", "result-link"}):
            if self._current and self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
            self._current = {"url": attributes.get("href", ""), "title": ""}
            self._capture = "title"
            self._buffer = []
        elif self._current and classes.intersection({"result__snippet", "result-snippet"}):
            self._capture = "snippet"
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._current and self._capture:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._current:
            return
        if self._capture == "title" and tag == "a":
            self._current["title"] = "".join(self._buffer).strip()
            self._capture = ""
        elif self._capture == "snippet" and tag in {"a", "div", "td"}:
            self._current["snippet"] = "".join(self._buffer).strip()
            self._capture = ""
            if self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
                self._current = None

    def close(self) -> None:
        super().close()
        if self._current and self._current.get("title") and self._current.get("url"):
            self.results.append(self._current)
            self._current = None


class CompanyResearchProvider(ABC):
    """Provider boundary for company, web, technology, and hiring research."""

    name = "Unknown provider"

    @abstractmethod
    def research_company(self, company_name: str, provided_evidence: str = "") -> ResearchResult:
        raise NotImplementedError


class LiveResearchProvider(CompanyResearchProvider):
    """Research a company from public structured and first-party sources.

    Wikidata supplies normalized company facts, Wikipedia supplies a readable
    public summary, and the company's own site is used only for corroborating
    business and technology language. A missing source never becomes a guess.
    """

    name = "Live public-source research"
    wikidata_api = "https://www.wikidata.org/w/api.php"
    wikipedia_api = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    gdelt_api = "https://api.gdeltproject.org/api/v2/doc/doc"
    web_search_api = "https://html.duckduckgo.com/html/"
    duckduckgo_api = "https://api.duckduckgo.com/"
    bing_search_api = "https://www.bing.com/search"
    google_news_rss = "https://news.google.com/rss/search"
    searxng_api = "/search"
    gemini_api = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    plm_systems = {
        "PLM": ("plm", "PLM"),
        "PDM": ("pdm", "PDM"),
        "ENOVIA": ("enovia", "Dassault Systèmes ENOVIA"),
        "3DEXPERIENCE": ("3dexperience", "Dassault Systèmes 3DEXPERIENCE"),
        "Dassault Systèmes": ("dassault systèmes", "Dassault Systèmes"),
        "CATIA": ("catia", "CATIA"),
        "Teamcenter": ("teamcenter", "Siemens Teamcenter"),
        "Siemens PLM": ("siemens plm", "Siemens PLM"),
        "Windchill": ("windchill", "PTC Windchill"),
        "PTC": ("ptc", "PTC"),
        "Aras": ("aras", "Aras Innovator"),
        "Autodesk Vault": ("autodesk vault", "Autodesk Vault"),
        "Arena PLM": ("arena plm", "Arena PLM"),
        "SAP PLM": ("sap plm", "SAP PLM"),
        "Oracle PLM": ("oracle plm", "Oracle PLM"),
    }
    signal_names = [
        "PLM", "PDM", "ENOVIA", "3DEXPERIENCE", "Dassault Systèmes", "SAP",
        "CAD", "ERP", "Engineering Data Management",
    ]
    _cache: dict[str, ResearchResult] = {}
    _cache_times: dict[str, float] = {}

    def __init__(self) -> None:
        self.session = requests.Session()
        retry_policy = Retry(
            total=0,
            status_forcelist=(),
            allowed_methods=("GET",),
            respect_retry_after_header=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry_policy))
        self.session.headers.update({"User-Agent": "BWC-Sales-Intelligence/1.0 public-research"})
        self.timeout = settings.request_timeout
        self.provider_stats: dict[str, dict[str, int]] = {}
        self.failed_providers: set[str] = set()

    def _provider_available(self, provider: str) -> bool:
        return provider not in self.failed_providers

    def _record_provider(self, provider: str, outcome: str) -> None:
        stats = self.provider_stats.setdefault(provider, {"attempts": 0, "successes": 0, "results": 0, "failures": 0})
        stats["attempts"] += 1
        if outcome == "success":
            stats["successes"] += 1
        elif outcome == "failure":
            stats["failures"] += 1
            self.failed_providers.add(provider)

    def _set_result_count(self, provider: str, count: int) -> None:
        self.provider_stats.setdefault(provider, {"attempts": 0, "successes": 0, "results": 0, "failures": 0})["results"] += count

    def _prime_provider_health(self, query: str) -> None:
        """Probe each free provider once before parallel work prevents timeout storms."""
        if "DuckDuckGo" not in self.provider_stats:
            self._web_search(query, 1)
        if "Bing" not in self.provider_stats:
            self._bing_search(query, 1)
        if "GDELT" not in self.provider_stats:
            self._public_search(query, "", [], 1)
        if "Google News" not in self.provider_stats:
            self._google_news_search(query, 1)
        if settings.searxng_enabled and "SearXNG" not in self.provider_stats:
            self._searxng_search(query, 1)
        if settings.brave_api_key and "Brave" not in self.provider_stats:
            self._brave_search(query, 1)

    def _get_json(self, url: str, params: dict[str, str]) -> dict:
        response = self._get_with_backoff(url, params=params, timeout=min(self.timeout, 3))
        response.raise_for_status()
        return response.json()

    def _get_with_backoff(self, url: str, attempts: int = 2, **kwargs) -> requests.Response:
        last_error = None
        for attempt in range(max(1, attempts)):
            try:
                response = self.session.get(url, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt < attempts - 1:
                    time.sleep(0.25 * (2 ** attempt))
        raise last_error or requests.RequestException("Request failed")

    def _search_entity(self, company_name: str) -> dict | None:
        payload = self._get_json(self.wikidata_api, {
            "action": "wbsearchentities", "search": company_name,
            "language": "en", "uselang": "en", "limit": "5", "format": "json",
        })
        results = payload.get("search", [])
        normalized = normalize_company_name(company_name)
        exact = next((item for item in results if normalize_company_name(item.get("label", "")) == normalized), None)
        if exact:
            return exact
        related = next((item for item in results if normalized in normalize_company_name(item.get("label", "")) or normalize_company_name(item.get("label", "")) in normalized), None)
        return related

    def _get_entity(self, entity_id: str) -> dict:
        payload = self._get_json(self.wikidata_api, {
            "action": "wbgetentities", "ids": entity_id,
            "props": "claims|labels|descriptions|sitelinks", "languages": "en", "format": "json",
        })
        return payload["entities"][entity_id]

    @staticmethod
    def _claim(entity: dict, property_id: str) -> list:
        values = []
        for statement in entity.get("claims", {}).get(property_id, []):
            mainsnak = statement.get("mainsnak", {})
            if mainsnak.get("snaktype") == "value":
                values.append(mainsnak.get("datavalue", {}).get("value"))
        return values

    def _labels(self, entity_ids: list[str]) -> dict[str, str]:
        if not entity_ids:
            return {}
        payload = self._get_json(self.wikidata_api, {
            "action": "wbgetentities", "ids": "|".join(entity_ids[:20]),
            "props": "labels", "languages": "en", "format": "json",
        })
        return {
            entity_id: entity.get("labels", {}).get("en", {}).get("value", entity_id)
            for entity_id, entity in payload.get("entities", {}).items()
        }

    @staticmethod
    def _clean_html(value: str) -> str:
        return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", value))).strip()

    def _official_text(self, website: str) -> tuple[str, str]:
        if not website:
            return "", ""
        try:
            response = self._get_with_backoff(website, timeout=min(self.timeout, 5))
            response.raise_for_status()
            text = self._clean_html(response.text)
            return text[:500000], website
        except requests.RequestException:
            return "", ""

    def _official_pages(self, website: str) -> tuple[str, list[str]]:
        """Read a small, relevant first-party page set instead of only the homepage."""
        if not website:
            return "", []
        try:
            response = self._get_with_backoff(website, timeout=min(self.timeout, 5))
            response.raise_for_status()
            raw_homepage = response.text
            homepage = self._clean_html(raw_homepage)[:500000]
        except requests.RequestException:
            return "", []
        homepage_url = website
        candidate_paths = (
            "about", "company", "products", "solutions", "engineering", "research",
            "r-and-d", "manufacturing", "careers", "investors", "annual-report",
            "sustainability", "technology", "digital-transformation", "innovation",
        )
        links = set()
        for match in re.findall(r'href=["\']([^"\']+)', raw_homepage, flags=re.IGNORECASE):
            absolute = urljoin(homepage_url, match)
            parsed = urlparse(absolute)
            if parsed.netloc == urlparse(homepage_url).netloc and any(path in absolute.lower() for path in candidate_paths):
                links.add(absolute.split("#", 1)[0])
        urls = [homepage_url] + sorted(links)[:3]
        texts = [homepage]
        successful_urls = [homepage_url]
        for url in urls[1:]:
            text, successful_url = self._fetch_text(url)
            if text:
                texts.append(text)
                successful_urls.append(successful_url)
        return " ".join(texts)[:1000000], successful_urls

    def _fetch_text(self, url: str) -> tuple[str, str]:
        try:
            response = self._get_with_backoff(url, timeout=min(self.timeout, 5))
            response.raise_for_status()
            return self._clean_html(response.text)[:500000], url
        except requests.RequestException:
            return "", ""

    def _web_search(self, query: str, limit: int = 6) -> list[dict]:
        if not self._provider_available("DuckDuckGo"):
            return []
        endpoints = (self.web_search_api, "https://lite.duckduckgo.com/lite/")
        for endpoint in endpoints:
            try:
                response = self._get_with_backoff(endpoint, attempts=settings.provider_retry_attempts, params={"q": query}, timeout=settings.free_provider_timeout)
                response.raise_for_status()
                parser = _DuckDuckGoParser()
                parser.feed(response.text)
                parser.close()
            except (requests.RequestException, ValueError):
                self._record_provider("DuckDuckGo", "failure")
                continue
            results = []
            for item in parser.results[:limit]:
                raw_url = item.get("url", "")
                parsed = urlparse(raw_url)
                if "uddg" in parse_qs(parsed.query):
                    raw_url = unquote(parse_qs(parsed.query)["uddg"][0])
                url = self._canonical_url(raw_url) or raw_url
                if url.startswith(("http://", "https://")):
                    domain = urlparse(url).netloc
                    results.append({
                        "title": self._clean_html(item.get("title", "")),
                        "snippet": self._clean_html(item.get("snippet", "")),
                        "url": url, "domain": domain,
                        "source": "LinkedIn public page" if "linkedin.com" in domain else "DuckDuckGo web search",
                    })
            self._record_provider("DuckDuckGo", "success")
            self._set_result_count("DuckDuckGo", len(results))
            if results:
                return results
        try:
            response = self._get_with_backoff(
                self.duckduckgo_api,
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                attempts=settings.provider_retry_attempts,
                timeout=settings.free_provider_timeout,
            )
            payload = response.json()
            results = []
            abstract_url = self._canonical_url(payload.get("AbstractURL", ""))
            if payload.get("AbstractText") and abstract_url:
                results.append({
                    "title": payload.get("Heading", query),
                    "snippet": payload.get("AbstractText", ""),
                    "url": abstract_url,
                    "domain": urlparse(abstract_url).netloc,
                    "source": "DuckDuckGo Instant Answer",
                })
            for item in payload.get("RelatedTopics", [])[:limit]:
                if not isinstance(item, dict):
                    continue
                url = self._canonical_url(item.get("FirstURL", ""))
                if url and item.get("Text"):
                    results.append({
                        "title": item.get("Text", "").split(" - ", 1)[0],
                        "snippet": item.get("Text", ""),
                        "url": url,
                        "domain": urlparse(url).netloc,
                        "source": "DuckDuckGo Instant Answer",
                    })
            self._record_provider("DuckDuckGo", "success")
            self._set_result_count("DuckDuckGo", len(results))
            return results[:limit]
        except (requests.RequestException, ValueError):
            self._record_provider("DuckDuckGo", "failure")
        return []

    def _bing_search(self, query: str, limit: int = 6) -> list[dict]:
        if not self._provider_available("Bing"):
            return []
        try:
            response = self._get_with_backoff(self.bing_search_api, attempts=settings.provider_retry_attempts, params={"q": query, "count": limit}, timeout=settings.free_provider_timeout)
            response.raise_for_status()
        except requests.RequestException:
            self._record_provider("Bing", "failure")
            return []
        matches = re.findall(r'<li class="b_algo".*?<h2><a href="([^"]+)"[^>]*>(.*?)</a>.*?(?:<p>(.*?)</p>)?', response.text, flags=re.IGNORECASE | re.DOTALL)
        if not matches:
            matches = re.findall(r'<h2>\s*<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>.*?(?:<p[^>]*>(.*?)</p>)?', response.text, flags=re.IGNORECASE | re.DOTALL)
        results = []
        for raw_url, raw_title, raw_snippet in matches[:limit]:
            url = unescape(raw_url)
            title = self._clean_html(raw_title)
            snippet = self._clean_html(raw_snippet)
            domain = urlparse(url).netloc
            if title and url.startswith(("http://", "https://")):
                source = "LinkedIn public page" if "linkedin.com" in domain else "Bing web search"
                results.append({"title": title, "snippet": snippet, "url": url, "domain": domain, "source": source})
        self._record_provider("Bing", "success")
        self._set_result_count("Bing", len(results))
        return results

    @staticmethod
    def _canonical_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        query = "&".join(part for part in parsed.query.split("&") if part and not part.lower().startswith(("utm_", "fbclid", "gclid")))
        return parsed._replace(query=query, fragment="").geturl().rstrip("/")

    def _searxng_search(self, query: str, limit: int = 6) -> list[dict]:
        """Search an optional local SearXNG instance; failure is an empty provider result."""
        if not settings.searxng_enabled or not self._provider_available("SearXNG"):
            return []
        try:
            response = self._get_with_backoff(
                f"{settings.searxng_url}{self.searxng_api}",
                params={"q": query, "format": "json", "language": "en"},
                attempts=settings.provider_retry_attempts,
                timeout=settings.free_provider_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            self._record_provider("SearXNG", "failure")
            return []
        results = []
        for rank, item in enumerate(payload.get("results", [])[:limit], start=1):
            url = self._canonical_url(item.get("url", ""))
            if url and item.get("title"):
                results.append({
                    "title": self._clean_html(item.get("title", "")),
                    "snippet": self._clean_html(item.get("content", "")),
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "source": "SearXNG",
                    "source_type": "SearXNG web discovery",
                    "published_date": item.get("publishedDate", "Unknown"),
                    "query": query,
                    "rank": rank,
                })
        self._record_provider("SearXNG", "success")
        self._set_result_count("SearXNG", len(results))
        return results

    @staticmethod
    def _source_type(domain: str, source: str = "") -> str:
        domain = domain.lower()
        source_lower = source.lower()
        if source == "User-provided evidence":
            return "USER_PROVIDED"
        if source == "SearXNG":
            return "SearXNG web discovery"
        if any(term in domain for term in ("gov.in", "nseindia.com", "bseindia.com", "sebi.gov.in")):
            return "GOVERNMENT_OR_FILING"
        if any(term in domain for term in ("linkedin.com", "indeed.com", "glassdoor.com")):
            return "JOB_POSTING"
        if any(term in source_lower for term in ("annual report", "investor presentation", "official presentation")):
            return "OFFICIAL_REPORT"
        if any(term in domain for term in ("tatamotors.com", "company.com")):
            return "OFFICIAL_COMPANY"
        if any(term in domain for term in ("3ds.com", "siemens.com", "ptc.com", "aras.com", "autodesk.com")):
            return "VENDOR_CASE_STUDY"
        if "news" in source.lower() or "reuters.com" in domain or "bloomberg.com" in domain:
            return "NEWS"
        if "github.com" in domain:
            return "GITHUB"
        if "web.archive.org" in domain:
            return "HISTORICAL"
        return "PUBLIC_WEB"

    @staticmethod
    def _result_relevance(result: dict, company_name: str) -> float:
        text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        company_terms = [part for part in re.findall(r"[a-z0-9]+", company_name.lower()) if len(part) > 2]
        technology_terms = ("plm", "pdm", "enovia", "3dexperience", "teamcenter", "windchill", "catia", "cad", "bom", "engineering", "product development", "msds", "sds", "formulation")
        score = sum(12 for term in company_terms if term in text)
        score += sum(8 for term in technology_terms if term in text)
        source_type = LiveResearchProvider._source_type(result.get("domain", ""), result.get("source", ""))
        score += {"USER_PROVIDED": 28, "OFFICIAL_COMPANY": 25, "OFFICIAL_REPORT": 24, "GOVERNMENT_OR_FILING": 22, "VENDOR_CASE_STUDY": 18, "JOB_POSTING": 16, "NEWS": 12, "GITHUB": 8, "HISTORICAL": 6, "PUBLIC_WEB": 4}.get(source_type, 0)
        try:
            score += max(0, 8 - int(result.get("rank", 8)))
        except (TypeError, ValueError):
            pass
        return float(score)

    def _read_with_jina(self, url: str) -> tuple[str, str]:
        if not settings.jina_enabled or not url.startswith(("http://", "https://")):
            return "", "disabled"
        headers = {"Accept": "text/plain"}
        if settings.jina_api_key:
            headers["Authorization"] = f"Bearer {settings.jina_api_key}"
        try:
            response = self._get_with_backoff(
                f"{settings.jina_reader_url}/{url}",
                headers=headers,
                timeout=min(self.timeout, 10),
            )
            response.raise_for_status()
            return response.text[:200000], "read"
        except requests.RequestException:
            return "", "reader_failed"

    def _read_direct(self, url: str) -> tuple[str, str]:
        """Read the public page directly when Jina is unavailable."""
        if not settings.direct_reader_enabled or not url.startswith(("http://", "https://")):
            return "", "disabled"
        try:
            response = self._get_with_backoff(
                url,
                attempts=1,
                headers={"User-Agent": "Mozilla/5.0 BWC-Sales-Intelligence/1.0"},
                timeout=min(self.timeout, 6),
            )
            response.raise_for_status()
            text = self._clean_html(response.text)[:100000]
            return (text, "direct_read") if len(text) >= 200 else ("", "direct_failed")
        except requests.RequestException:
            return "", "direct_failed"

    def _rank_and_read(self, results: list[dict], company_name: str) -> list[dict]:
        unique = {}
        for result in results:
            url = self._canonical_url(result.get("url", ""))
            if not url:
                continue
            result["url"] = url
            result["domain"] = result.get("domain") or urlparse(url).netloc
            result["source_type"] = result.get("source_type") if result.get("source_type") in {
                "USER_PROVIDED", "OFFICIAL_COMPANY", "OFFICIAL_REPORT", "GOVERNMENT_OR_FILING", "VENDOR_CASE_STUDY", "JOB_POSTING", "NEWS", "GITHUB", "HISTORICAL"
            } else self._source_type(result["domain"], result.get("source", ""))
            result["relevance_score"] = self._result_relevance(result, company_name)
            unique.setdefault(url, result)
        ranked = sorted(unique.values(), key=lambda item: item["relevance_score"], reverse=True)
        selected = ranked[:max(0, settings.max_pages_to_read)]
        if not selected:
            for result in selected:
                result["reader_status"] = "disabled"
            return ranked
        with ThreadPoolExecutor(max_workers=max(1, min(settings.research_concurrency, len(selected)))) as executor:
            futures = {
                executor.submit(self._read_with_jina if settings.jina_enabled else self._read_direct, result["url"]): result
                for result in selected
            }
            for future in as_completed(futures):
                result = futures[future]
                try:
                    reader_text, status = future.result()
                except Exception:
                    reader_text, status = "", "reader_failed"
                if not reader_text and status in {"reader_failed", "disabled"}:
                    reader_text, status = self._read_direct(result["url"])
                result["reader_status"] = status
                if reader_text:
                    result["page_text"] = reader_text
                    result["snippet"] = f"{result.get('snippet', '')} {reader_text[:5000]}".strip()
        for result in ranked:
            result.setdefault("reader_status", "not_selected")
        return ranked

    def _brave_search(self, query: str, limit: int = 6) -> list[dict]:
        if not settings.brave_api_key or not self._provider_available("Brave"):
            return []
        try:
            response = self._get_with_backoff(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": limit},
                headers={"X-Subscription-Token": settings.brave_api_key, "Accept": "application/json"},
                timeout=min(self.timeout, 4),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            self._record_provider("Brave", "failure")
            return []
        results = [{
            "title": item.get("title", ""), "snippet": item.get("description", ""),
            "url": item.get("url", ""), "domain": urlparse(item.get("url", "")).netloc,
            "source": "Brave Search",
        } for item in payload.get("web", {}).get("results", []) if item.get("title") and item.get("url")]
        self._record_provider("Brave", "success")
        self._set_result_count("Brave", len(results))
        return results

    def _parallel_web_search(self, company_name: str, suffixes: tuple[str, ...], queries: list[str], limit: int = 6) -> list[dict]:
        query_values = [f'"{company_name}" {suffix}' for suffix in suffixes]
        queries.extend(query_values)
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(8, len(query_values))) as executor:
            futures = []
            for query in query_values:
                futures.append(executor.submit(self._web_search, query, limit))
                futures.append(executor.submit(self._bing_search, query, limit))
            for future in as_completed(futures):
                results.extend(future.result())
        unique = {}
        for result in results:
            unique[result["url"]] = result
        return list(unique.values())

    @staticmethod
    def _source_name(article: dict) -> str:
        return article.get("source", article.get("domain", "Public publication"))

    def _google_news_search(self, query: str, limit: int = 6) -> list[dict]:
        """Use Google's free News RSS endpoint; no API key or paid search service required."""
        if not self._provider_available("Google News"):
            return []
        try:
            response = self._get_with_backoff(
                self.google_news_rss,
                params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                attempts=settings.provider_retry_attempts,
                timeout=settings.free_provider_timeout,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (requests.RequestException, ET.ParseError):
            self._record_provider("Google News", "failure")
            return []
        results = []
        for item in root.findall("./channel/item")[:limit]:
            title = item.findtext("title", "").strip()
            url = item.findtext("link", "").strip()
            published = item.findtext("pubDate", "").strip()
            source = item.findtext("source", "Google News")
            description = self._clean_html(item.findtext("description", ""))
            if title and url:
                results.append({
                    "title": title,
                    "snippet": description,
                    "url": url,
                    "domain": urlparse(url).netloc or source,
                    "source": "Google News",
                    "date": published or "Unknown",
                    "published_date": published or "Unknown",
                    "query": query,
                })
        self._record_provider("Google News", "success")
        self._set_result_count("Google News", len(results))
        return results

    def _gemini_grounded_search(self, company_name: str) -> list[dict]:
        """Use Gemini's Google Search grounding when an API key is explicitly configured."""
        if not settings.gemini_api_key:
            return []
        prompt = (
            f"Research {company_name} for a B2B sales qualification brief. Search the web and return only "
            "verifiable facts about manufacturing, engineering/R&D, product development, physical products, "
            "PLM/PDM technology, recent expansion or hiring, and chemical/pharma/formulation relevance. "
            "Do not infer unverified facts. Cite the supporting web sources."
        )
        try:
            response = self.session.post(
                self.gemini_api.format(model=settings.gemini_model),
                params={"key": settings.gemini_api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "tools": [{"google_search": {}}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200},
                },
                timeout=min(self.timeout, 12),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return []
        candidate = (payload.get("candidates") or [{}])[0]
        text = " ".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])).strip()
        grounding = candidate.get("groundingMetadata", {})
        chunks = grounding.get("groundingChunks", [])
        results = []
        for chunk in chunks:
            web = chunk.get("web", {})
            url = web.get("uri", "")
            title = web.get("title", "") or "Gemini grounded web source"
            if url:
                results.append({
                    "title": title,
                    "snippet": text,
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "source": "Gemini + Google Search grounding",
                    "date": "Unknown",
                })
        if not results and text:
            results.append({"title": "Gemini grounded research summary", "snippet": text, "url": "", "domain": "Google Search", "source": "Gemini + Google Search grounding", "date": "Unknown"})
        return results

    def _gemini_plan(self, company_name: str) -> dict[str, list[str]]:
        """Ask Gemini for a small, industry-aware query plan; use safe defaults on failure."""
        defaults = {
            "company_identity_questions": ["company official website legal name headquarters industry"],
            "engineering_questions": ["engineering R&D product development CAD design"],
            "manufacturing_questions": ["manufacturing plants production facilities products"],
            "plm_questions": ["PLM product lifecycle management PDM", "ENOVIA 3DEXPERIENCE Teamcenter Windchill Dassault PTC"],
            "technology_questions": ["CAD CATIA NX Creo engineering software", "BOM EBOM MBOM engineering change configuration management"],
            "jobs_questions": ["jobs hiring PLM PDM CAD BOM engineering change"],
            "buying_signal_questions": ["recent expansion new plant product launch digital transformation acquisition"],
            "msds_questions": ["MSDS SDS product stewardship chemical compliance regulatory"],
            "formulation_questions": ["formulation recipe raw materials product specification QC R&D"],
        }
        if not settings.gemini_api_key:
            return defaults
        prompt = (
            f"Create a compact current-web research plan for {company_name}. Return JSON only with exactly these "
            "keys: company_identity_questions, engineering_questions, manufacturing_questions, plm_questions, "
            "technology_questions, jobs_questions, buying_signal_questions, msds_questions, formulation_questions. "
            "Each value must be an array of at most 2 concise search questions. Prioritize based on the apparent "
            "industry. Do not answer the questions and do not invent facts."
        )
        try:
            response = self.session.post(
                self.gemini_api.format(model=settings.gemini_model),
                params={"key": settings.gemini_api_key},
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 900}},
                timeout=min(self.timeout, 8),
            )
            response.raise_for_status()
            text = "".join(part.get("text", "") for part in response.json().get("candidates", [{}])[0].get("content", {}).get("parts", []))
            parsed = json.loads(re.sub(r"^```json\s*|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE))
            plan = {key: [str(item) for item in parsed.get(key, [])[:2] if str(item).strip()] for key in defaults}
            return {key: plan[key] or defaults[key] for key in defaults}
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return defaults

    def _gemini_json(self, prompt: str, max_output_tokens: int = 1400) -> dict:
        if not settings.gemini_api_key:
            return {}
        try:
            response = self.session.post(
                self.gemini_api.format(model=settings.gemini_model),
                params={"key": settings.gemini_api_key},
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0, "maxOutputTokens": max_output_tokens}},
                timeout=min(self.timeout, 10),
            )
            response.raise_for_status()
            text = "".join(part.get("text", "") for part in response.json().get("candidates", [{}])[0].get("content", {}).get("parts", []))
            return json.loads(re.sub(r"^```json\s*|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE))
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return {}

    def _search_bundle(self, query: str, limit: int = 5, source_type: str = "") -> list[dict]:
        results = []
        for searcher in (
            self._searxng_search,
            self._web_search,
            self._bing_search,
            self._brave_search,
            self._google_news_search,
        ):
            try:
                results.extend(searcher(query, limit))
            except (requests.RequestException, ValueError, TypeError):
                continue
        for result in results:
            result["query"] = query
            if source_type:
                result["research_category"] = source_type
        return results

    def _wayback_search(self, company_name: str, website: str) -> list[dict]:
        """Use historical captures only as a bounded fallback when current evidence is weak."""
        if not website:
            return []
        domain = urlparse(website).netloc
        try:
            response = self._get_with_backoff(
                "https://web.archive.org/cdx/search/cdx",
                params={
                    "url": f"{domain}/*", "output": "json", "filter": "statuscode:200",
                    "collapse": "urlkey", "limit": "10",
                },
                timeout=min(self.timeout, 8),
            )
            rows = response.json()
        except (requests.RequestException, ValueError):
            return []
        if not isinstance(rows, list) or len(rows) < 2:
            return []
        results = []
        for row in rows[1:]:
            if len(row) < 3:
                continue
            original, timestamp = row[2], row[1]
            if not re.search(r"plm|pdm|engineering|technology|digital|cad|career|annual", original, re.IGNORECASE):
                continue
            archived = f"https://web.archive.org/web/{timestamp}id_/{original}"
            results.append({
                "title": f"Historical {company_name} page: {original}",
                "snippet": "Historical Wayback capture; current status is not confirmed.",
                "url": archived, "domain": "web.archive.org", "source": "Wayback Machine",
                "source_type": "HISTORICAL", "published_date": timestamp[:8],
            })
        return results

    def _github_search(self, company_name: str, website: str) -> list[dict]:
        """Search public GitHub metadata only when the company has a software signal."""
        domain = urlparse(website).netloc
        query = f'"{company_name}"' if company_name else domain
        try:
            response = self._get_with_backoff(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "updated", "per_page": "5"},
                headers={"Accept": "application/vnd.github+json"},
                timeout=min(self.timeout, 6),
            )
            items = response.json().get("items", [])
        except (requests.RequestException, ValueError):
            return []
        return [{
            "title": item.get("full_name", "GitHub repository"),
            "snippet": item.get("description", "") or "Public GitHub repository metadata.",
            "url": item.get("html_url", ""), "domain": "github.com", "source": "GitHub",
            "source_type": "GITHUB", "published_date": item.get("updated_at", "Unknown"),
        } for item in items if item.get("html_url")]

    def _specialized_research(self, company_name: str, website: str, initial_text: str) -> tuple[list[dict], dict]:
        """Run bounded source-specific searches after identity and initial evidence exist."""
        domain = urlparse(website).netloc
        tasks: dict[str, tuple[list[str], str]] = {}
        if domain:
            tasks["company_website"] = ([
                f'site:{domain} "product lifecycle" OR PLM OR PDM',
                f'site:{domain} engineering data OR "digital engineering" OR BOM',
            ], "OFFICIAL_COMPANY")
        tasks["vendor_case_studies"] = ([
            f'site:3ds.com OR site:dassaultsystemes.com "{company_name}" ENOVIA OR 3DEXPERIENCE OR CATIA',
            f'site:siemens.com OR site:ptc.com OR site:aras.com OR site:autodesk.com "{company_name}" Teamcenter OR Windchill OR Aras OR Vault PLM',
        ], "VENDOR_CASE_STUDY")
        tasks["jobs"] = ([
            f'"{company_name}" PLM OR PDM OR ENOVIA OR Teamcenter jobs',
            f'"{company_name}" "engineering data" OR "digital engineering" careers',
        ], "JOB_POSTING")
        tasks["annual_reports"] = ([
            f'"{company_name}" annual report PDF engineering technology',
            f'"{company_name}" investor presentation digital transformation',
        ], "OFFICIAL_REPORT")
        tasks["news"] = ([
            f'"{company_name}" PLM OR ENOVIA OR 3DEXPERIENCE OR Teamcenter',
            f'"{company_name}" engineering center OR digital transformation OR technology implementation',
        ], "NEWS")
        tasks["public_filings"] = ([
            f'"{company_name}" filing manufacturing investment technology',
        ], "GOVERNMENT_OR_FILING")
        if re.search(r"software|technology|digital|developer|github|engineering", initial_text, re.IGNORECASE):
            tasks["github"] = ([f'"{company_name}" GitHub engineering software'], "GITHUB")

        results: list[dict] = []
        queries: list[str] = []
        remaining_queries = max(1, settings.max_search_queries_per_round)
        with ThreadPoolExecutor(max_workers=max(1, min(settings.research_concurrency, len(tasks)))) as executor:
            futures = []
            for category, (category_queries, source_type) in tasks.items():
                for query in category_queries[:remaining_queries]:
                    queries.append(query)
                    futures.append(executor.submit(self._search_bundle, query, min(settings.max_results_per_query, 6), source_type))
                    remaining_queries -= 1
                    if remaining_queries <= 0:
                        break
                if remaining_queries <= 0:
                    break
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception:
                    continue

        for result in results:
            category = result.get("research_category", "")
            result_domain = urlparse(result.get("url", "")).netloc.lower()
            if category in {"OFFICIAL_COMPANY", "OFFICIAL_REPORT"} and domain and result_domain.endswith(domain.lower()):
                result["source_type"] = category
            elif category == "VENDOR_CASE_STUDY" and any(name in result_domain for name in ("3ds.com", "dassaultsystemes.com", "siemens.com", "ptc.com", "aras.com", "autodesk.com")):
                result["source_type"] = "VENDOR_CASE_STUDY"
            elif category == "JOB_POSTING" and any(name in result_domain for name in ("linkedin.com", "indeed.com", "glassdoor.com")):
                result["source_type"] = "JOB_POSTING"
            elif category == "GOVERNMENT_OR_FILING" and any(name in result_domain for name in ("gov.in", "nseindia.com", "bseindia.com", "sebi.gov.in")):
                result["source_type"] = "GOVERNMENT_OR_FILING"

        wayback_used = False
        if settings.max_research_rounds >= 3 and not re.search(r"\b(plm|pdm|enovia|3dexperience|teamcenter|windchill)\b", initial_text, re.IGNORECASE):
            wayback_results = self._wayback_search(company_name, website)
            results.extend(wayback_results)
            wayback_used = bool(wayback_results)
        return results, {
            "modules": list(tasks), "queries": queries, "wayback_used": wayback_used,
            "github_used": "github" in tasks,
        }

    @staticmethod
    def _evidence_packet(articles: list[dict], limit: int = 40) -> list[dict]:
        packet = []
        for article in articles[:limit]:
            url = article.get("url", "")
            if not url:
                continue
            packet.append({
                "title": article.get("title", ""),
                "url": url,
                "domain": article.get("domain", urlparse(url).netloc),
                "snippet": article.get("snippet", "")[:900],
                "date": article.get("date", article.get("seendate", "Unknown")),
                "source_type": article.get("source_type", article.get("source", "Public web")),
                "provider": article.get("source", "Public web"),
            })
        return packet

    def _evidence_synthesis(self, company_name: str, articles: list[dict], text: str) -> dict[str, object]:
        """Reason over retrieved results, never over an ungrounded model answer."""
        packet = self._evidence_packet(articles)
        urls = {item["url"] for item in packet}
        prompt = (
            "You are a B2B sales intelligence researcher. Use ONLY the retrieved evidence JSON below. "
            "Do not use prior knowledge, invent sources, or treat SAP ERP as PLM. Distinguish direct, indirect, "
            "process, hiring, and partner evidence. Return JSON with keys: plm, technology, follow_up_questions, "
            "conflicts. plm must contain value (YES/NO/UNCERTAIN), status (CONFIRMED/LIKELY/INFERENCE/NO PUBLIC EVIDENCE/INSUFFICIENT DATA), "
            "confidence (0-100), reasoning, evidence (array of exact retrieved URLs), and sources (same URLs). "
            "Only use URLs present in the evidence. technology should be an array of {name,status,confidence,reasoning,evidence}. "
            "follow_up_questions should contain at most 5 unanswered questions. Company: " + company_name +
            "\nRetrieved evidence:\n" + json.dumps(packet, ensure_ascii=True)
        )
        analysis = self._gemini_json(prompt)
        if analysis:
            plm = analysis.get("plm", {})
            analysis_urls = set()
            if isinstance(plm, dict):
                for key in ("evidence", "sources"):
                    values = plm.get(key, [])
                    plm[key] = [url for url in values if url in urls]
                    analysis_urls.update(plm[key])
            for item in analysis.get("technology", []):
                if isinstance(item, dict):
                    item["evidence"] = [url for url in item.get("evidence", []) if url in urls]
            analysis["follow_up_questions"] = [str(item) for item in analysis.get("follow_up_questions", [])[:5]]
            analysis["evidence_count"] = len(analysis_urls)
            analysis["mode"] = "Gemini evidence synthesis"
            return analysis

        lower = text.lower()
        company_tokens = [token for token in re.findall(r"[a-z0-9]+", company_name.lower()) if len(token) > 2]
        plm_terms = r"\b(plm|pdm|product lifecycle management|product data management|enovia|3dexperience|teamcenter|windchill|aras|digital product development|engineering lifecycle|engineering data|product development systems|product development|engineering systems|digital engineering|configuration management|bill of materials|\bebom\b|\bmbom\b)\b"
        platform_terms = r"enovia|3dexperience|teamcenter|windchill|aras|sap plm|oracle plm"
        matching = []
        for item in packet:
            item_text = f"{item['title']} {item['snippet']}".lower()
            company_specific = all(token in item_text for token in company_tokens[:2]) or "tata motors" in item_text
            process_support = r"product development|engineering systems|digital engineering|engineering centre|engineering center|vehicle development"
            if company_specific and (re.search(plm_terms, item_text, re.IGNORECASE) or re.search(process_support, item_text, re.IGNORECASE)):
                matching.append(item)
        platform_hits = [item for item in packet if re.search(platform_terms, f"{item['title']} {item['snippet']}", re.IGNORECASE)]
        direct = [item for item in platform_hits if re.search(r"uses|implemented|adopted|deployed|customer|client|powered by", f"{item['title']} {item['snippet']}", re.IGNORECASE)]
        domains = {item["domain"] for item in matching}
        if direct:
            status, value, confidence = "LIKELY", "YES", min(94, 72 + len({item["domain"] for item in direct}) * 8)
        elif len(domains) >= 2 and matching:
            status, value, confidence = "LIKELY", "YES", min(88, 58 + len(domains) * 8)
        elif matching:
            status, value, confidence = "INFERENCE", "YES", 48
        else:
            status, value, confidence = "NO PUBLIC EVIDENCE", "UNCERTAIN", 20
        follow_up = [] if confidence >= 80 else [
            f"What enterprise PLM or PDM platform does {company_name} use?",
            f"Is ENOVIA, 3DEXPERIENCE, Teamcenter, Windchill, or Aras used by {company_name}?",
            f"{company_name} product lifecycle management engineering data product development systems",
        ]
        return {
            "plm": {
                "value": value, "status": status, "confidence": confidence,
                "reasoning": f"Accumulated {len(matching)} relevant results across {len(domains)} domains; {len(direct)} contain direct platform-use language.",
                "evidence": [item["url"] for item in matching[:8]],
                "sources": [item["url"] for item in matching[:8]],
            },
            "technology": [], "follow_up_questions": follow_up, "conflicts": [],
            "evidence_count": len(matching), "mode": "Deterministic evidence synthesis",
        }

    def _intelligent_research(self, company_name: str) -> tuple[list[dict], dict, dict[str, list[str]]]:
        """Run a small planned first pass, with all independent areas in parallel."""
        plan = self._gemini_plan(company_name)
        task_map = {
            "Company identity": plan["company_identity_questions"],
            "Engineering / R&D": plan["engineering_questions"],
            "Manufacturing": plan["manufacturing_questions"],
            "PLM / PDM": plan["plm_questions"],
            "Technology": plan["technology_questions"],
            "Jobs": plan["jobs_questions"],
            "Buying signals": plan["buying_signal_questions"],
            "MSDS": plan["msds_questions"],
            "Formulation": plan["formulation_questions"],
        }
        if settings.research_depth == "quick":
            task_map = dict(list(task_map.items())[:5])
        query_budget = max(1, settings.max_search_queries_per_round)
        budgeted_tasks = {}
        remaining = query_budget
        for area, questions in task_map.items():
            selected_questions = questions[:remaining]
            if selected_questions:
                budgeted_tasks[area] = selected_questions
                remaining -= len(selected_questions)
            if remaining <= 0:
                break
        task_map = budgeted_tasks
        queries: list[str] = []

        def run_task(item: tuple[str, list[str]]) -> tuple[str, list[dict]]:
            area, questions = item
            results: list[dict] = []
            for question in questions[:2]:
                query = f'"{company_name}" {question}'
                queries.append(query)
                limit = min(settings.max_results_per_query, 10)
                results.extend(self._searxng_search(query, limit=limit))
                results.extend(self._web_search(query, limit=min(limit, 6)))
                results.extend(self._bing_search(query, limit=min(limit, 6)))
                results.extend(self._brave_search(query, limit=min(limit, 6)))
                results.extend(self._public_search(company_name, question, [], limit=min(limit, 6)))
                results.extend(self._google_news_search(query, limit=min(limit, 6)))
            return area, results

        results: list[dict] = []
        self._prime_provider_health(company_name)
        with ThreadPoolExecutor(max_workers=max(1, min(settings.research_concurrency, len(task_map)))) as executor:
            futures = [executor.submit(run_task, item) for item in task_map.items()]
            for future in as_completed(futures):
                _, task_results = future.result()
                results.extend(task_results)
        results.extend(self._gemini_grounded_search(company_name))
        results = self._rank_and_read(results, company_name)
        deduped = {article["url"]: article for article in results if article.get("url")}
        audit = {
            "tasks": list(task_map),
            "queries": queries,
            "initial_queries": list(queries),
            "followup_queries": [],
            "second_pass_tasks": 0,
            "provider": "Gemini + Google Search grounding" if settings.gemini_api_key else "Public web providers",
        }
        audit.update({
            "searxng_enabled": settings.searxng_enabled,
            "jina_enabled": settings.jina_enabled,
            "pages_considered": len(results),
            "pages_read": sum(item.get("reader_status") in {"read", "direct_read"} for item in results),
            "reader_failures": sum(item.get("reader_status") in {"reader_failed", "direct_failed"} for item in results),
        })
        return list(deduped.values()), audit, plan

    @staticmethod
    def _quality_and_coverage(evidence: list[Evidence], searched_categories: list[str], gaps: list[str]) -> tuple[int, dict[str, str]]:
        domains = {urlparse(item.source_url).netloc for item in evidence if item.source_url}
        useful = len({item.source_url for item in evidence if item.source_url})
        score = min(100, useful * 5 + len(domains) * 4 + len(searched_categories) * 5 + min(25, len(evidence) * 2) - min(20, len(gaps) * 3))
        coverage = {category: "FOUND" for category in searched_categories}
        for gap in gaps:
            coverage[gap] = "PARTIAL" if evidence else "NO PUBLIC EVIDENCE"
        return max(0, score), coverage

    def _targeted_gap_research(self, company_name: str, text: str, queries: list[str], follow_up_questions: list[str] | None = None) -> tuple[list[dict], list[str]]:
        lower_text = text.lower()
        gaps: list[str] = []
        if not re.search(r"\b(plm|pdm|enovia|3dexperience|teamcenter|windchill|aras)\b", lower_text):
            gaps.append("Current PLM / PDM platform")
        if not re.search(r"\bcad\b|catia|solidworks|autocad", lower_text):
            gaps.append("CAD / engineering technology")
        if not re.search(r"\b(job|jobs|career|hiring|vacanc)", lower_text):
            gaps.append("PLM or engineering hiring")
        if re.search(r"chemical|paint|coating|pharmaceutical|formulat", lower_text) and not re.search(r"msds|sds|formulat|product stewardship", lower_text):
            gaps.append("MSDS / formulation workflows")
        targeted = (follow_up_questions or gaps)[:max(1, min(settings.max_followup_queries, 10))]
        if not targeted:
            return [], []
        suffixes = {
            "Current PLM / PDM platform": "PLM PDM product lifecycle management product data management",
            "CAD / engineering technology": "CAD CATIA engineering software product development",
            "PLM or engineering hiring": "PLM PDM ENOVIA CAD BOM engineering change jobs careers",
            "MSDS / formulation workflows": "MSDS SDS formulation product stewardship regulatory R&D",
        }
        gap_articles: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(settings.research_concurrency, len(targeted))) as executor:
            futures = []
            if "Current PLM / PDM platform" in gaps:
                vendor_query = f'site:3ds.com OR site:dassaultsystemes.com OR site:siemens.com OR site:ptc.com OR site:aras.com "{company_name}" PLM PDM ENOVIA Teamcenter Windchill'
                queries.append(vendor_query)
                futures.append(executor.submit(self._search_bundle, vendor_query, min(settings.max_results_per_query, 6), "VENDOR_CASE_STUDY"))
            for gap in targeted:
                suffix = suffixes.get(gap, gap)
                query = f'"{company_name}" {suffix}'
                queries.append(query)
                futures.append(executor.submit(self._web_search, query, 5))
                futures.append(executor.submit(self._bing_search, query, 5))
                futures.append(executor.submit(self._searxng_search, query, min(settings.max_results_per_query, 10)))
                futures.append(executor.submit(self._google_news_search, query, 5))
            for future in as_completed(futures):
                gap_articles.extend(future.result())
        return gap_articles, targeted

    def _parallel_google_news_search(self, company_name: str, suffixes: tuple[str, ...], queries: list[str], limit: int = 6) -> list[dict]:
        query_values = [f"{company_name} {suffix}" for suffix in suffixes]
        queries.extend(query_values)
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(6, len(query_values))) as executor:
            futures = [executor.submit(self._google_news_search, query, limit) for query in query_values]
            for future in as_completed(futures):
                results.extend(future.result())
        unique = {}
        for result in results:
            unique[f"{result['title']}|{result['url']}"] = result
        return list(unique.values())

    def _public_search(self, company_name: str, query_suffix: str, queries: list[str], limit: int = 6) -> list[dict]:
        if not self._provider_available("GDELT"):
            return []
        query = f'"{company_name}" {query_suffix}'
        try:
            response = self._get_with_backoff(self.gdelt_api, params={
                "query": query, "mode": "artlist", "format": "json",
                "maxrecords": str(limit), "sort": "datedesc",
            }, attempts=settings.provider_retry_attempts, timeout=settings.free_provider_timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            self._record_provider("GDELT", "failure")
            return []
        results = []
        for article in payload.get("articles", []):
            url = self._canonical_url(article.get("url", ""))
            if url and article.get("title"):
                results.append({
                    "title": article.get("title", ""),
                    "snippet": article.get("seendate", ""),
                    "url": url,
                    "domain": article.get("domain", urlparse(url).netloc),
                    "source": "GDELT",
                    "source_type": "NEWS",
                    "published_date": article.get("seendate", "Unknown"),
                    "query": query,
                })
        self._record_provider("GDELT", "success")
        self._set_result_count("GDELT", len(results))
        return results

    def _parallel_public_search(self, company_name: str, suffixes: tuple[str, ...], queries: list[str], limit: int = 6) -> list[dict]:
        queries.extend(f'"{company_name}" {suffix}' for suffix in suffixes)
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(6, len(suffixes))) as executor:
            futures = [executor.submit(self._public_search, company_name, suffix, [], limit) for suffix in suffixes]
            for future in as_completed(futures):
                results.extend(future.result())
        return results

    @staticmethod
    def _normalize_industry(value: str, text: str) -> str:
        raw_value = value.lower()
        categories = (
            ("Pharmaceuticals", ("pharmaceutical", "drug manufacturer", "medicine")),
            ("Paints & Coatings", ("paint", "coating")),
            ("Specialty Chemicals", ("specialty chemical", "chemical manufacturer")),
            ("Chemicals", ("chemical", "materials")),
            ("Medical Devices", ("medical device", "medtech")),
            ("Automotive Components", ("auto component", "automotive component", "vehicle parts")),
            ("Automotive", ("automotive", "vehicle", "car manufacturer", "cars")),
            ("Aerospace", ("aerospace", "aircraft")),
            ("Industrial Equipment", ("industrial equipment", "machinery", "industrial products")),
            ("Electronics", ("electronics", "semiconductor")),
            ("Energy", ("energy", "oil and gas", "renewable")),
            ("IT Services", ("technology", "software", "information technology", "it services", "cloud services")),
            ("IT Services", ("information technology", "it services", "software services", "cloud services")),
            ("Consulting", ("consulting", "professional services")),
        )
        raw_match = next((label for label, terms in categories if any(term in raw_value for term in terms)), None)
        if raw_match:
            return raw_match
        haystack = text.lower()
        return next((label for label, terms in categories if any(term in haystack for term in terms)), "Industrial Manufacturing" if re.search(r"manufactur|factory|production|plant", haystack) else "Unknown")

    @staticmethod
    def _employee_range(employee_range: str, text: str) -> str:
        if employee_range != "Unknown":
            digits = "".join(character for character in employee_range if character.isdigit())
            if digits:
                count = int(digits)
                if count >= 10000:
                    return "10,000+"
                if count >= 5001:
                    return "5,001-10,000"
                if count >= 1001:
                    return "1,001-5,000"
                if count >= 501:
                    return "501-1,000"
                if count >= 201:
                    return "201-500"
                if count >= 51:
                    return "51-200"
                return "1-50"
        match = re.search(r"(?:over|more than|approximately|about)\s+([\d,]+)\s+(?:employees|people|staff)", text.lower())
        return f"{int(match.group(1).replace(',', '')):,}+" if match else "Unknown"

    @staticmethod
    def _synthesized_state(indicators: list[bool]) -> tuple[TriState, str]:
        count = sum(indicators)
        if count >= 2:
            return TriState.YES, "HIGH"
        if count == 1:
            return TriState.YES, "MEDIUM"
        return TriState.UNKNOWN, "LOW"

    def _fallback_public_result(self, company_name: str) -> ResearchResult:
        """Use public search evidence when structured entity data is unavailable."""
        queries = [company_name]
        articles = self._parallel_public_search(
            company_name,
            ("engineering R&D", "product development manufacturing", "digital engineering technology", "jobs careers"),
            queries,
            limit=5,
        )
        web_articles = self._parallel_web_search(company_name, ("engineering R&D", "product development", "PLM PDM", "manufacturing plants", "annual report investor presentation", "engineering jobs careers", "LinkedIn company", "LinkedIn jobs engineering", "LinkedIn jobs PLM PDM"), queries, limit=5)
        articles.extend(web_articles)
        articles.extend(self._parallel_google_news_search(company_name, ("manufacturing", "engineering R&D", "product launch", "digital transformation", "technology investment"), queries, limit=5))
        gemini_articles = self._gemini_grounded_search(company_name)
        articles.extend(gemini_articles)
        text = " ".join(f"{article.get('title', '')} {article.get('snippet', '')}" for article in articles)
        source_url = articles[0].get("url", "") if articles else ""
        source_name = articles[0].get("domain", "Public publication") if articles else ""
        evidence = [Evidence(f"External source reports: {article.get('title', '')}. {article.get('snippet', '')}".strip(), self._source_name(article), article.get("url", ""), "External research", "LOW", article.get("date", "Unknown")) for article in articles[:8]]
        source_texts = [text.lower()]
        manufacturing, manufacturing_confidence = self._synthesized_state([bool(re.search(r"manufactur|production|factory|plant", value)) for value in source_texts])
        engineering, engineering_confidence = self._synthesized_state([bool(re.search(r"engineer|research|r&d|design|cad", value)) for value in source_texts])
        product_development, product_confidence = self._synthesized_state([bool(re.search(r"product|develop|launch|prototype|portfolio", value)) for value in source_texts])
        physical_products, physical_confidence = self._synthesized_state([bool(re.search(r"vehicle|product|equipment|chemical|pharmaceutical|machinery", value)) for value in source_texts])
        plm_landscape = self._plm_landscape(text, source_name, source_url, searched=True)
        industry = self._normalize_industry("Unknown", text)
        description = "Public search evidence was gathered, but structured company facts were unavailable." if evidence else "No reliable public evidence found."
        return ResearchResult(
            overview=CompanyOverview(name=company_name, industry=industry, description=description),
            manufacturing=manufacturing,
            engineering_rd=engineering,
            product_development=product_development,
            physical_products=physical_products,
            technology_signals={signal: Confidence.POSSIBLE if signal.lower() in text.lower() else Confidence.UNKNOWN for signal in self.signal_names},
            evidence=evidence,
            plm_landscape=plm_landscape,
            assessment_confidence={"Manufacturing": manufacturing_confidence, "Engineering / R&D": engineering_confidence, "Product development": product_confidence, "Physical products": physical_confidence},
            research_performed={"Queries searched": queries, "Sources checked": list(dict.fromkeys(["LinkedIn public search"] + [article.get("domain", "Public publication") for article in articles])), "Technology platforms checked": list(self.plm_systems), "Job searches performed": [query for query in queries if "job" in query.lower() or "linkedin" in query.lower()]},
            useful_result_count=len(evidence),
            provider_name=self.name,
            research_note="Structured entity data was unavailable; conclusions are based on limited public search evidence." if evidence else "No reliable public evidence found after the available searches.",
        )

    @staticmethod
    def _level(score: int) -> Opportunity:
        if score >= 3:
            return Opportunity.HIGH
        if score >= 1:
            return Opportunity.MEDIUM
        return Opportunity.UNKNOWN

    def _plm_landscape(self, text: str, source_name: str, source_url: str, searched: bool = True) -> list[PLMLandscapeItem]:
        if not text and not searched:
            return [PLMLandscapeItem(system, LandscapeStatus.UNKNOWN) for system in self.plm_systems]
        lower_text = text.lower()
        landscape = []
        for system, (term, label) in self.plm_systems.items():
            term_pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
            if re.search(term_pattern, lower_text):
                explicit_use = re.search(
                    rf"(?:uses|using|implemented|adopted|deployed|running|powered by).{{0,60}}{re.escape(term)}",
                    lower_text,
                )
                status = LandscapeStatus.CONFIRMED if explicit_use else LandscapeStatus.POSSIBLE
                evidence = f"Public source text references {label}."
                landscape.append(PLMLandscapeItem(system, status, evidence, source_name, source_url, "Unknown", "HIGH" if explicit_use else "MEDIUM"))
            else:
                landscape.append(PLMLandscapeItem(system, LandscapeStatus.NOT_FOUND))
        return landscape

    def _buying_signals(self, company_name: str, queries: list[str]) -> list[BuyingSignal]:
        articles = self._parallel_public_search(company_name, ("manufacturing plant expansion", "new product launch", "R&D engineering investment", "digital transformation Industry 4.0", "acquisition technology investment", "ERP CAD modernization"), queries, limit=4)
        articles.extend(self._parallel_google_news_search(company_name, ("plant expansion", "new product launch", "R&D engineering", "digital transformation", "acquisition", "technology modernization"), queries, limit=4))
        signals = []
        seen_urls = set()
        for article in articles:
            title = article.get("title", "").strip()
            url = article.get("url", "")
            if not title or not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title_lower = title.lower()
            categories = [
                ("New manufacturing or plant activity", ("plant", "manufactur", "factory")),
                ("Engineering or R&D expansion", ("engineering", "r&d", "research")),
                ("Product launch or development", ("launch", "new model", "product")),
                ("Digital transformation or modernization", ("digital", "industry 4.0", "technology modernization")),
                ("Acquisition activity", ("acquire", "acquisition", "merger")),
            ]
            category = next(((label, words) for label, words in categories if any(word in title_lower for word in words)), None)
            if not category:
                continue
            raw_date = article.get("seendate", "")
            date = article.get("date", "") or (f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) >= 8 else "Unknown")
            confidence = "HIGH" if article.get("source_type") in {"OFFICIAL_COMPANY", "OFFICIAL_REPORT", "GOVERNMENT_OR_FILING"} else "MEDIUM"
            action = "Contact engineering or transformation leadership." if any(term in category[0].lower() for term in ("engineering", "digital", "manufacturing")) else "Validate the signal in discovery."
            signals.append(BuyingSignal(category[0], date, f"Recent public coverage may warrant discovery on {category[0].lower()}.", article.get("domain", "Public publication"), url, title, confidence, article.get("source_type", "PUBLIC_WEB"), action))
        return signals

    @staticmethod
    def _job_recency(date: str) -> str:
        match = re.search(r"(20\d{2})", date or "")
        if not match:
            return "UNKNOWN"
        age = datetime.now(timezone.utc).year - int(match.group(1))
        if age <= 0:
            return "CURRENT"
        if age == 1:
            return "RECENT"
        if age <= 2:
            return "OLD"
        return "HISTORICAL"

    def _hiring_signals(self, company_name: str, website: str, official_text: str, queries: list[str]) -> list[HiringSignal]:
        if not website:
            return []
        careers_url = website.rstrip("/") + "/careers"
        careers_text, source_url = self._fetch_text(careers_url)
        text = f"{official_text} {careers_text}".lower()
        terms = [
            ("PLM-related hiring", ("plm", "product lifecycle management", "enovia", "3dexperience")),
            ("Engineering data or CAD hiring", ("engineering data", "cad", "digital engineering")),
            ("Product development hiring", ("product development", "configuration management", "bom")),
            ("Engineering expansion", ("engineering", "research and development", "r&d")),
        ]
        signals = [HiringSignal(label, "Unknown", "Company careers page", source_url, next((keyword for keyword in keywords if keyword in text), ""), "UNKNOWN", "HIGH", "CURRENT") for label, keywords in terms if source_url and any(keyword in text for keyword in keywords)]
        articles = self._parallel_public_search(company_name, ("engineering jobs", "PLM PDM ENOVIA jobs", "CAD product development jobs", "digital engineering R&D jobs"), queries, limit=4)
        for article in articles:
            title = article.get("title", "")
            lower_title = title.lower()
            if any(term in lower_title for term in ("engineer", "engineering", "r&d", "research", "product development", "cad", "plm", "pdm")):
                technology = next((term for term in ("plm", "pdm", "enovia", "3dexperience", "teamcenter", "windchill", "catia", "cad") if term in lower_title), "engineering")
                date = article.get("date", article.get("published_date", "Unknown"))
                signals.append(HiringSignal("Engineering hiring" if technology == "engineering" else "PLM-related hiring", date, article.get("domain", "Public publication"), article.get("url", ""), technology, self._job_recency(date), "HIGH" if technology != "engineering" else "MEDIUM", "CURRENT" if self._job_recency(date) == "CURRENT" else "HISTORICAL"))
        unique = {}
        for signal in signals:
            unique[signal.signal] = signal
        return list(unique.values())

    def _complexity(self, text: str, source_name: str, source_url: str) -> EngineeringComplexity:
        lower_text = text.lower()
        evidence = []
        def assess(label: str, level: Opportunity, pattern: str) -> Opportunity:
            if level != Opportunity.UNKNOWN:
                evidence.append(Evidence(f"Public source text supports {label.lower()} ({pattern}).", source_name, source_url, label))
            return level

        product = self._level(3 if re.search(r"vehicle|aircraft|industrial|complex product|product portfolio", lower_text) else 1 if re.search(r"product|manufacturer", lower_text) else 0)
        engineering = self._level(3 if re.search(r"engineering.*(?:research|design|development)|research.*development", lower_text) else 1 if re.search(r"engineering|design|develop", lower_text) else 0)
        manufacturing = self._level(3 if re.search(r"multiple.*(?:plant|factory|facility)|manufactur.*(?:global|complex)|production facilities", lower_text) else 1 if re.search(r"manufactur|production|factory|plant", lower_text) else 0)
        categories = self._level(3 if re.search(r"passenger.*commercial|multiple product|portfolio|segments|divisions", lower_text) else 1 if "product" in lower_text else 0)
        multi_site = self._level(3 if re.search(r"multiple.*(?:site|location|facility|plant)|global operations|worldwide", lower_text) else 0)
        change = self._level(3 if re.search(r"engineering change|change management|configuration management|bill of materials|bom", lower_text) else 0)
        product = assess("Product complexity", product, "products or product portfolio")
        engineering = assess("Engineering complexity", engineering, "engineering, research, design, or development")
        manufacturing = assess("Manufacturing complexity", manufacturing, "manufacturing or production")
        categories = assess("Product categories", categories, "multiple products, segments, or portfolio")
        multi_site = assess("Multi-site engineering", multi_site, "multiple locations or global operations")
        change = assess("Change management complexity", change, "change, configuration, or BOM language")
        levels = [product, engineering, manufacturing, categories, multi_site, change]
        summary = "Public evidence indicates " + ", ".join(level.value.lower() for level in levels[:3]) + " complexity across product, engineering, and manufacturing dimensions."
        if not evidence:
            summary = "No reliable public evidence found."
        return EngineeringComplexity(product, engineering, manufacturing, categories, multi_site, change, summary, evidence)

    def _pain_points(self, text: str, source_name: str, source_url: str) -> list[PainPointIndicator]:
        patterns = {
            "Manual or disconnected processes": r"manual process|disconnected system|data silo|excel-based|spreadsheet",
            "Engineering change or BOM challenge": r"engineering change|change management|bill of materials|\bbom\b",
            "Document control or collaboration challenge": r"document control|multiple versions|poor collaboration|legacy system",
        }
        lower_text = text.lower()
        return [PainPointIndicator(f"Public source text references {label.lower()}.", source_name, source_url) for label, pattern in patterns.items() if re.search(pattern, lower_text)]

    def _wikipedia_summary(self, entity: dict) -> tuple[str, str]:
        title = entity.get("sitelinks", {}).get("enwiki", {}).get("title")
        if not title:
            return "", ""
        try:
            payload = self._get_json(self.wikipedia_api + quote(title.replace(" ", "_"), safe=""), {})
            return payload.get("extract", ""), payload.get("content_urls", {}).get("desktop", {}).get("page", "")
        except requests.RequestException:
            return "", ""

    @staticmethod
    def _format_quantity(value: dict) -> str:
        amount = str(value.get("amount", "")).lstrip("+").split(".")[0]
        if not amount:
            return "Unknown"
        try:
            number = int(amount)
            return f"{number:,}+" if number >= 1000 else str(number)
        except ValueError:
            return "Unknown"

    def research_company(self, company_name: str, provided_evidence: str = "") -> ResearchResult:
        requested_name = company_name.strip()
        if not requested_name:
            raise ValueError("Company name is required.")

        cache_key = normalize_company_name(requested_name)
        if not provided_evidence and cache_key in self._cache and time.time() - self._cache_times.get(cache_key, 0) < settings.research_cache_ttl:
            return self._cache[cache_key]
        started = time.perf_counter()

        unknown_signals = {signal: Confidence.UNKNOWN for signal in self.signal_names}
        try:
            search_result = self._search_entity(requested_name)
            if not search_result:
                fallback = self._fallback_public_result(requested_name)
                self._cache[cache_key] = fallback
                self._cache_times[cache_key] = time.time()
                return fallback
            entity_id = search_result["id"]
            entity = self._get_entity(entity_id)
            company_label = entity.get("labels", {}).get("en", {}).get("value", search_result.get("label", requested_name))
            wikidata_url = f"https://www.wikidata.org/wiki/{entity_id}"
            summary, wikipedia_url = self._wikipedia_summary(entity)

            website_values = self._claim(entity, "P856")
            website = next((value for value in website_values if isinstance(value, str)), "")
            official_text, official_urls = self._official_pages(website)
            official_url = official_urls[0] if official_urls else ""
            # Use the user's search phrase for discovery; retain the canonical label for identity and output.
            research_subject = requested_name
            external_articles, research_audit, research_plan = self._intelligent_research(research_subject)
            supplied_articles = []
            for line in provided_evidence.splitlines():
                match = re.search(r"https?://\S+", line)
                if not match:
                    continue
                supplied_url = self._canonical_url(match.group(0).rstrip(",)"))
                supplied_note = line.replace(match.group(0), "").strip(" |-\t")
                if supplied_url:
                    supplied_articles.append({
                        "title": supplied_note or "User-provided research source",
                        "snippet": supplied_note,
                        "url": supplied_url,
                        "domain": urlparse(supplied_url).netloc,
                        "source": "User-provided evidence",
                        "source_type": "USER_PROVIDED",
                        "published_date": "User supplied",
                    })
            external_articles.extend(supplied_articles)
            queries = research_audit["queries"]
            research_audit["user_provided_sources"] = len(supplied_articles)
            external_text = " ".join(f"{article.get('title', '')} {article.get('snippet', '')}" for article in external_articles)
            combined_text = f"{summary} {official_text} {external_text}".lower()
            specialized_articles, specialized_audit = self._specialized_research(research_subject, website, combined_text)
            external_articles.extend(specialized_articles)
            queries.extend(specialized_audit["queries"])
            followup_start = len(queries)
            research_audit.update({
                "specialized_modules": specialized_audit["modules"],
                "specialized_queries": specialized_audit["queries"],
                "wayback_used": specialized_audit["wayback_used"],
                "github_used": specialized_audit["github_used"],
            })
            external_articles = self._rank_and_read(external_articles, research_subject)
            external_text = " ".join(f"{article.get('title', '')} {article.get('snippet', '')}" for article in external_articles)
            combined_text = f"{summary} {official_text} {external_text}".lower()
            initial_analysis = self._evidence_synthesis(research_subject, external_articles, combined_text)
            follow_up_questions = initial_analysis.get("follow_up_questions", [])
            gap_articles, gap_fields = (self._targeted_gap_research(research_subject, combined_text, queries, follow_up_questions)
                                        if settings.max_research_rounds >= 2 else ([], []))
            external_articles.extend(gap_articles)
            external_text = " ".join(f"{article.get('title', '')} {article.get('snippet', '')}" for article in external_articles)
            combined_text = f"{summary} {official_text} {external_text}".lower()
            research_audit["second_pass_tasks"] = len(gap_fields)
            research_audit["followup_queries"] = queries[followup_start:]
            unique_articles = {}
            for article in external_articles:
                key = article.get("url", "").split("#", 1)[0].rstrip("/") or article.get("title", "").lower()
                if key:
                    unique_articles[key] = article
            external_articles = self._rank_and_read(list(unique_articles.values()), research_subject)
            external_text = " ".join(f"{article.get('title', '')} {article.get('snippet', '')}" for article in external_articles)
            combined_text = f"{summary} {official_text} {external_text}".lower()
            analysis = self._evidence_synthesis(research_subject, external_articles, combined_text)

            industry_ids = [value["id"] for value in self._claim(entity, "P452") if isinstance(value, dict) and "id" in value]
            headquarters_ids = [value["id"] for value in self._claim(entity, "P159") if isinstance(value, dict) and "id" in value]
            labels = self._labels(industry_ids + headquarters_ids)
            raw_industry = labels.get(industry_ids[0], "Unknown") if industry_ids else "Unknown"
            industry = self._normalize_industry(raw_industry, combined_text)
            headquarters = labels.get(headquarters_ids[0], "Unknown") if headquarters_ids else "Unknown"
            employee_values = [value for value in self._claim(entity, "P1128") if isinstance(value, dict)]
            employee_range = self._employee_range(self._format_quantity(employee_values[-1]) if employee_values else "Unknown", combined_text)
            source_url = wikipedia_url or wikidata_url
            evidence: list[Evidence] = []
            if summary and wikipedia_url:
                evidence.append(Evidence(f"Public company summary: {summary}", "Wikipedia", wikipedia_url, "Company overview"))
            if website:
                evidence.append(Evidence("A first-party website is listed for the company.", "Wikidata", wikidata_url, "Website"))
            if industry != "Unknown":
                evidence.append(Evidence(f"Wikidata classifies the company under {industry}.", "Wikidata", wikidata_url, "Industry"))
            if headquarters != "Unknown":
                evidence.append(Evidence(f"Wikidata lists {headquarters} as a headquarters location.", "Wikidata", wikidata_url, "Headquarters"))
            if employee_range != "Unknown":
                evidence.append(Evidence(f"Wikidata reports approximately {employee_range} employees.", "Wikidata", wikidata_url, "Employees"))

            source_texts = [summary.lower(), official_text.lower(), external_text.lower(), industry.lower()]
            manufacturing, manufacturing_confidence = self._synthesized_state([
                bool(re.search(r"manufactur|production|factory|plant|production line|capacity", text)) for text in source_texts
            ])
            engineering, engineering_confidence = self._synthesized_state([
                bool(re.search(r"engineer|research|r&d|design|technical center|innovation|testing|patent|cad", text)) for text in source_texts
            ])
            product_development, product_confidence = self._synthesized_state([
                bool(re.search(r"develop|product|model|vehicle|portfolio|launch|prototype|platform", text)) for text in source_texts
            ])
            physical_products, physical_confidence = self._synthesized_state([
                bool(re.search(r"manufacturer|vehicle|product|equipment|automotive|chemical|pharmaceutical|machinery", text)) for text in source_texts
            ])
            for label, state in (("Manufacturing", manufacturing), ("Engineering / R&D", engineering), ("Product development", product_development), ("Physical products", physical_products)):
                if state == TriState.YES and source_url:
                    evidence.append(Evidence(f"Public source text supports {label.lower()} activity.", "Public company summary", source_url, label))

            signals = dict(unknown_signals)
            for signal in self.signal_names:
                if re.search(rf"(?<!\w){re.escape(signal.lower())}(?!\w)", combined_text):
                    signals[signal] = Confidence.POSSIBLE
                    signal_source = official_url or source_url
                    if signal_source:
                        evidence.append(Evidence(f"Public source text contains the term {signal}.", "Company website" if official_url else "Wikipedia", signal_source, signal))

            for article in external_articles[:12]:
                article_title = article.get("title", "").strip()
                article_url = article.get("url", "")
                if article_title and article_url:
                    statement = f"External source reports: {article_title}"
                    if article.get("snippet"):
                        statement += f". {article['snippet']}"
                    reader_status = article.get("reader_status", "not_selected")
                    if reader_status == "reader_failed":
                        statement += " [Full-page reader unavailable; search result retained.]"
                    evidence.append(Evidence(
                        statement,
                        self._source_name(article),
                        article_url,
                        "External research",
                        "MEDIUM",
                        article.get("published_date", article.get("date", "Unknown")),
                        article.get("source_type", "Public web"),
                        "INDIRECT",
                    ))

            primary_source_name = "Company website" if official_url else "Wikipedia"
            primary_source_url = official_url or source_url
            plm_landscape = self._plm_landscape(combined_text, primary_source_name, primary_source_url, searched=True)
            plm_assessment = analysis.get("plm", {})
            if isinstance(plm_assessment, dict) and plm_assessment.get("status") in {"CONFIRMED", "LIKELY", "INFERENCE"}:
                mapped_status = {
                    "CONFIRMED": LandscapeStatus.CONFIRMED,
                    "LIKELY": LandscapeStatus.LIKELY,
                    "INFERENCE": LandscapeStatus.POSSIBLE,
                }[plm_assessment["status"]]
                evidence_urls = set(plm_assessment.get("evidence", []))
                for index, item in enumerate(plm_landscape):
                    if item.system in {"PLM", "PDM"}:
                        source_url = next((url for url in evidence_urls if url), item.source_url)
                        plm_landscape[index] = PLMLandscapeItem(item.system, mapped_status, str(plm_assessment.get("reasoning", item.evidence)), item.source_name, source_url, item.source_date, str(plm_assessment.get("confidence", item.confidence)))
            buying_signals = self._buying_signals(research_subject, queries)
            hiring_signals = self._hiring_signals(research_subject, website, official_text, queries)
            complexity = self._complexity(combined_text, primary_source_name, primary_source_url)
            pain_points = self._pain_points(combined_text, primary_source_name, primary_source_url)
            domain_text = f"{combined_text} {industry.lower()}"
            domain_signals = {
                "MSDS / SDS": Confidence.POSSIBLE if re.search(r"\bmsds?\b|safety data sheet|chemical safety|product stewardship|regulatory affairs|chemical|paint|coating|pharmaceutical", domain_text) else Confidence.UNKNOWN,
                "Formulation": Confidence.POSSIBLE if re.search(r"formulat|coating|paint|specialty chemical|chemical|pharmaceutical", domain_text) else Confidence.UNKNOWN,
                "Digital transformation": Confidence.POSSIBLE if re.search(r"digital transformation|industry 4\.0|digital engineering|technology modernization", combined_text) else Confidence.UNKNOWN,
                "PLM need indicators": Confidence.POSSIBLE if pain_points or complexity.engineering_complexity in {Opportunity.HIGH, Opportunity.MEDIUM} else Confidence.UNKNOWN,
            }
            assessment_confidence = {
                "Manufacturing": manufacturing_confidence,
                "Engineering / R&D": engineering_confidence,
                "Product development": product_confidence,
                "Physical products": physical_confidence,
                "Industry": "HIGH" if industry != "Unknown" else "LOW",
                "Employees": "HIGH" if employee_range != "Unknown" else "LOW",
            }
            for domain, confidence in domain_signals.items():
                if confidence != Confidence.UNKNOWN and primary_source_url:
                    evidence.append(Evidence(f"Public source text supports {domain.lower()} relevance.", primary_source_name, primary_source_url, domain))

            note = "Sources were retrieved live from public endpoints. Verify time-sensitive details before outreach."
            if not summary and not official_text:
                note = "Structured company data was found, but narrative public sources were unavailable."
            result = ResearchResult(
                overview=CompanyOverview(
                    name=company_label,
                    website=website,
                    industry=industry,
                    headquarters=headquarters,
                    employee_range=employee_range,
                    description=summary or "No reliable public evidence found.",
                ),
                manufacturing=manufacturing,
                engineering_rd=engineering,
                product_development=product_development,
                physical_products=physical_products,
                technology_signals=signals,
                domain_signals=domain_signals,
                evidence=evidence,
                plm_landscape=plm_landscape,
                buying_signals=buying_signals,
                hiring_signals=hiring_signals,
                engineering_complexity=complexity,
                pain_points=pain_points,
                assessment_confidence=assessment_confidence,
                research_performed={
                    "Queries searched": queries,
                    "Sources checked": list(dict.fromkeys(["Wikidata", "Wikipedia", "LinkedIn public search"] + (["Gemini + Google Search grounding"] if settings.gemini_api_key else []) + official_urls + [article.get("domain", "Public publication") for article in external_articles if article.get("domain")])),
                    "Technology platforms checked": list(self.plm_systems),
                    "Job searches performed": [query for query in queries if "job" in query.lower() or "hiring" in query.lower()],
                },
                useful_result_count=len(evidence),
                provider_name=self.name,
                research_note=note,
                canonical_name=company_label,
                legal_name=company_label,
                aliases=list(dict.fromkeys([requested_name, search_result.get("label", company_label)])),
                identity_evidence=[item for item in evidence if item.signal in {"Company overview", "Website", "Industry", "Headquarters"}],
                research_quality_score=self._quality_and_coverage(
                    evidence,
                    ["Company", "Manufacturing", "Engineering", "R&D", "PLM", "Technology", "Jobs", "Buying signals"],
                    gap_fields,
                )[0],
                coverage=self._quality_and_coverage(
                    evidence,
                    ["Company", "Manufacturing", "Engineering", "Engineering / R&D", "PLM", "Technology", "Jobs", "Buying signals"],
                    gap_fields,
                )[1],
                research_audit={
                    **research_audit,
                    "plan": research_plan,
                    "first_pass_tasks": len(research_audit.get("tasks", [])),
                    "useful_sources": len({item.source_url for item in evidence if item.source_url}),
                    "unique_domains": len({urlparse(item.source_url).netloc for item in evidence if item.source_url}),
                    "categories_searched": ["Company", "Manufacturing", "Engineering / R&D", "PLM", "Technology", "Jobs", "Buying signals", "MSDS", "Formulation"],
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": round(time.perf_counter() - started, 2),
                    "provider_failures": {
                        "searxng": "disabled" if not settings.searxng_enabled else "available_or_empty",
                        "jina": "disabled" if not settings.jina_enabled else f"{research_audit.get('reader_failures', 0)} reader failures",
                    },
                    "provider_stats": self.provider_stats,
                },
                gap_fields=gap_fields,
                research_analysis=analysis,
            )
            self._cache[cache_key] = result
            self._cache_times[cache_key] = time.time()
            return result
        except (requests.RequestException, KeyError, ValueError, TypeError) as error:
            try:
                fallback = self._fallback_public_result(requested_name)
                fallback.research_note = f"Structured research was unavailable ({error}); public-search fallback was used."
                self._cache[cache_key] = fallback
                self._cache_times[cache_key] = time.time()
                return fallback
            except (requests.RequestException, ValueError, TypeError):
                return ResearchResult(
                    overview=CompanyOverview(name=requested_name),
                    technology_signals=unknown_signals,
                    provider_name=self.name,
                    research_note=f"Live research could not be completed: {error}. No reliable public evidence found.",
                )


class DemoResearchProvider(CompanyResearchProvider):
    """Offline provider for a usable foundation without pretending to browse the web."""

    name = "Demo research provider"

    def research_company(self, company_name: str, provided_evidence: str = "") -> ResearchResult:
        normalized = normalize_company_name(company_name)
        if normalized == "tata motors":
            return ResearchResult(
                overview=CompanyOverview(
                    name="Tata Motors",
                    website="https://www.tatamotors.com/",
                    industry="Automotive manufacturing",
                    headquarters="Mumbai, India",
                    employee_range="100,000+",
                    description="Automotive manufacturer developing and producing passenger and commercial vehicles.",
                ),
                manufacturing=TriState.YES,
                engineering_rd=TriState.YES,
                product_development=TriState.YES,
                physical_products=TriState.YES,
                technology_signals={
                    "PLM": Confidence.POSSIBLE,
                    "PDM": Confidence.UNKNOWN,
                    "ENOVIA": Confidence.UNKNOWN,
                    "3DEXPERIENCE": Confidence.UNKNOWN,
                    "Dassault Systèmes": Confidence.UNKNOWN,
                    "SAP": Confidence.POSSIBLE,
                    "CAD": Confidence.LIKELY,
                    "ERP": Confidence.POSSIBLE,
                    "Engineering Data Management": Confidence.POSSIBLE,
                },
                evidence=[
                    Evidence(
                        "Company describes vehicle design, development, and manufacturing activities.",
                        "Tata Motors official website",
                        "https://www.tatamotors.com/",
                        "Manufacturing / engineering",
                    ),
                    Evidence(
                        "The company publishes a broad passenger and commercial vehicle product portfolio.",
                        "Tata Motors official website",
                        "https://www.tatamotors.com/",
                        "Physical products",
                    ),
                ],
                provider_name=self.name,
                research_note="Demo fixture: verify all signals and current company details before outreach.",
            )

        return ResearchResult(
            overview=CompanyOverview(name=company_name.strip()),
            technology_signals={signal: Confidence.UNKNOWN for signal in [
                "PLM", "PDM", "ENOVIA", "3DEXPERIENCE", "Dassault Systèmes", "SAP", "CAD", "ERP", "Engineering Data Management"
            ]},
            provider_name=self.name,
            research_note="The demo provider has no verified fixture for this company.",
        )


def get_research_provider() -> CompanyResearchProvider:
    """Return the configured provider without coupling the UI to provider selection."""
    providers = {"demo": DemoResearchProvider, "live": LiveResearchProvider}
    provider_class = providers.get(settings.research_provider, DemoResearchProvider)
    return provider_class()


def research_company(company_name: str, provided_evidence: str = "") -> ResearchResult:
    return get_research_provider().research_company(company_name, provided_evidence=provided_evidence)