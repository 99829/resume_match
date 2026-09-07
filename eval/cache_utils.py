"""
Lightweight disk cache for parse_jd / parse_resume outputs.
Keeps eval runs stable across LLM nondeterminism — delete the relevant
cache file if you want to re-test parsing after a prompt change.
"""
import hashlib
import json
import pathlib

_CACHE_DIR = pathlib.Path(__file__).parent / ".parse_cache"
_CACHE_DIR.mkdir(exist_ok=True)


def cached_parse(text: str, parse_fn, prefix: str) -> dict:
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    cache_path = _CACHE_DIR / f"{prefix}_{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    result = parse_fn(text)
    cache_path.write_text(json.dumps(result, indent=2))
    return result