import hashlib
import json
import os

from tavily import TavilyClient

from config import OUTPUT_DIR


_CACHE_ENABLED = True


def set_cache_enabled(enabled: bool) -> None:
    global _CACHE_ENABLED
    _CACHE_ENABLED = enabled


def search(query: str, max_results: int = 5) -> list[dict]:
    cache_dir = OUTPUT_DIR / "cache" / "search"
    cache_file = cache_dir / f"{hashlib.sha1(query.encode()).hexdigest()}.json"
    if _CACHE_ENABLED and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY를 .env에 설정하세요.")

    raw = TavilyClient(api_key=api_key).search(query, max_results=max_results)
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
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return results
