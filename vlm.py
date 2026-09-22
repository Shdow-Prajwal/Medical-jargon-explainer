import io
import json
import re
import hashlib
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import ollama
from PIL import Image
from pydantic import BaseModel
from config import get_settings
from cache import vlm_cache, explain_cache
import structlog

logger = structlog.get_logger()

settings = get_settings()
MODEL = settings.vlm_model

# Simple circuit breaker
_cb_failures = 0
_cb_lock = Lock()
_cb_open_until = 0


def _record_success():
    global _cb_failures
    with _cb_lock:
        _cb_failures = 0


def _record_failure():
    global _cb_failures, _cb_open_until
    with _cb_lock:
        _cb_failures += 1
        if _cb_failures >= settings.ollama_cb_threshold:
            _cb_open_until = __import__("time").time() + settings.ollama_cb_reset_seconds
            logger.warning("circuit_breaker.opened", threshold=settings.ollama_cb_threshold, reset_seconds=settings.ollama_cb_reset_seconds)


def _circuit_allows() -> bool:
    with _cb_lock:
        if __import__("time").time() < _cb_open_until:
            return False
        return True


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, (ollama.ResponseError, ConnectionError, TimeoutError))


class Finding(BaseModel):
    name: str
    value: str
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None
    page: int
    bbox: list[int] = []


class Extraction(BaseModel):
    answer: str
    findings: list[Finding]


def _shrink(path, max_side=1400):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _page_hash(image_path: str) -> str:
    # Hash file content for cache key stability
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(settings.ollama_max_retries),
    retry=retry_if_exception_type((ollama.ResponseError, ConnectionError, TimeoutError)),
    reraise=True,
)
def _chat_with_retry(**kwargs):
    if not _circuit_allows():
        raise RuntimeError("Circuit breaker open")
    try:
        resp = ollama.chat(**kwargs)
        _record_success()
        return resp
    except Exception as e:
        _record_failure()
        raise


def extract(question, pages):
    # Cache key
    page_hashes = [_page_hash(p["image_path"]) for p in pages]
    cached = vlm_cache.get(MODEL, question, page_hashes)
    if cached:
        logger.info("vlm.cache_hit", question=question[:50])
        return Extraction.model_validate(cached)

    msgs = [{
        "role": "user",
        "content": (
            f"Question: {question}\n"
            f"Pages are in order: {[p['page'] for p in pages]}.\n"
            "Answer the question in 1-3 sentences in the 'answer' field, using only what is "
            "visible on the pages. If the pages do not contain the answer, say exactly: "
            "'This is not on the pages I read.' Do not give medical advice or diagnoses.\n"
            "Then list any specific test values you used in 'findings'. "
            "Copy the reference range printed next to each value into reference_range, "
            "exactly as written. Leave it empty if none is printed."
        ),
        "images": [_shrink(p["image_path"]) for p in pages],
    }]

    r = _chat_with_retry(
        model=MODEL,
        messages=msgs,
        format=Extraction.model_json_schema(),
        options={"num_ctx": 8192, "temperature": 0},
        keep_alive="10m",
    )

    logger.info("vlm.timings", **{k: round(r[k] / 1e9, 1) for k in
            ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")},
        output_tokens=r["eval_count"])

    content = r["message"]["content"]
    logger.debug("vlm.raw_response", content=content[:500])

    if not content.strip():
        thinking = r["message"].get("thinking")
        raise RuntimeError(
            "Model returned empty content"
            + (" (it put its output in the 'thinking' field)" if thinking else "")
        )

    extraction = Extraction.model_validate_json(content)
    # Cache the result (as dict for JSON-serializable)
    vlm_cache.set(MODEL, question, page_hashes, extraction.model_dump())
    return extraction


def check_range(value, ref):
    if not ref:
        return "unknown"
    value, ref = value.replace(",", ""), ref.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", value)
    if not m:
        return "unknown"
    v = float(m.group())

    r = re.match(r"\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)", ref)
    if r:
        lo, hi = float(r[1]), float(r[2])
        return "low" if v < lo else "high" if v > hi else "normal"
    r = re.match(r"\s*[<≤]\s*(\d+(?:\.\d+)?)", ref)
    if r:
        return "high" if v > float(r[1]) else "normal"
    r = re.match(r"\s*[>≥]\s*(\d+(?:\.\d+)?)", ref)
    if r:
        return "low" if v < float(r[1]) else "normal"
    return "unknown"


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(settings.ollama_max_retries),
    retry=retry_if_exception_type((ollama.ResponseError, ConnectionError, TimeoutError)),
    reraise=True,
)
def _explain_chat(names):
    if not _circuit_allows():
        raise RuntimeError("Circuit breaker open")
    try:
        r = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": (
                "For each medical test name, write one plain-language sentence on what it measures. "
                "No advice, no diagnosis, no mention of any values. "
                f"Return a JSON object mapping each name to its sentence. Names: {names}"
            )}],
            format="json",
            options={"num_ctx": 4096, "temperature": 0},
            keep_alive="10m",
        )
        _record_success()
        return r
    except Exception as e:
        _record_failure()
        raise


def explain(findings):
    names = list({f.name for f in findings})[:15]
    if not names:
        return {}

    cached = explain_cache.get(names)
    if cached:
        logger.info("explain.cache_hit", names=names)
        return cached

    r = _explain_chat(names)
    try:
        result = json.loads(r["message"]["content"])
    except Exception:
        result = {}

    explain_cache.set(names, result)
    return result