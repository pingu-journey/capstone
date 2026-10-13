import hashlib
import json
import os

from tavily import TavilyClient

from config import OUTPUT_DIR
from tools.artifacts import write_json
from tools.services import ConfigurationError, check_service_error
from threading import RLock


_CACHE_ENABLED = True
_CACHE_LOCK = RLock()


def set_cache_enabled(enabled: bool) -> None:
    global _CACHE_ENABLED
    _CACHE_ENABLED = enabled


def search(query: str, max_results: int = 5) -> list[dict]:
    # One cache writer/client call at a time; atomic replacement prevents partial reads.
    with _CACHE_LOCK:
        return _search(query, max_results)


def _search(query: str, max_results: int) -> list[dict]:
    cache_dir = OUTPUT_DIR / "cache" / "search"
    key = query if max_results == 5 else f"{query}|{max_results}"
    cache_file = cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
    if _CACHE_ENABLED and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ConfigurationError("TAVILY_API_KEY를 .env에 설정하세요.")

    try:
        raw = TavilyClient(api_key=api_key).search(query, max_results=max_results)
    except Exception as error:
        check_service_error(error)
        raise
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "published_date": item.get("published_date"),
        }
        for item in raw.get("results", [])
    ]
    if _CACHE_ENABLED:
        write_json(cache_file, results)
    return results
