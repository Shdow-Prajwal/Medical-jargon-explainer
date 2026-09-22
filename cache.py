import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Optional
import diskcache
from config import get_settings


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def clear_prefix(self, prefix: str) -> None:
        ...


class DiskCacheBackend(CacheBackend):
    def __init__(self):
        settings = get_settings()
        self._cache = diskcache.Cache(
            directory=settings.cache_dir,
            size_limit=settings.cache_size_limit_bytes,
        )

    def _make_key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        self._cache.set(key, value, expire=ttl)

    def delete(self, key: str) -> None:
        self._cache.delete(key)

    def clear_prefix(self, prefix: str) -> None:
        # diskcache doesn't have prefix delete; iterate keys
        for key in list(self._cache):
            if key.startswith(prefix):
                self._cache.delete(key)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# High-level helpers
class EmbeddingCache:
    def __init__(self, backend: CacheBackend):
        self._backend = backend
        self._ns = "emb"
        self._ttl = 7 * 24 * 3600  # 7 days

    def key(self, model: str, text: str) -> str:
        return self._make_key(self._ns, f"{model}:{_hash(text)}")

    def get(self, model: str, text: str):
        return self._backend.get(self.key(model, text))

    def set(self, model: str, text: str, embeddings):
        self._backend.set(self.key(model, text), embeddings, self._ttl)

    def _make_key(self, ns: str, suffix: str) -> str:
        return f"{ns}:{suffix}"


class VLMCache:
    def __init__(self, backend: CacheBackend):
        self._backend = backend
        self._ns = "vlm"
        self._ttl = 24 * 3600  # 24h

    def key(self, model_version: str, question: str, page_hashes: list[str]) -> str:
        combined = question + "|" + "|".join(sorted(page_hashes))
        return self._make_key(self._ns, f"{model_version}:{_hash(combined)}")

    def get(self, model_version: str, question: str, page_hashes: list[str]):
        return self._backend.get(self.key(model_version, question, page_hashes))

    def set(self, model_version: str, question: str, page_hashes: list[str], value):
        self._backend.set(self.key(model_version, question, page_hashes), value, self._ttl)

    def _make_key(self, ns: str, suffix: str) -> str:
        return f"{ns}:{suffix}"


class ExplainCache:
    def __init__(self, backend: CacheBackend):
        self._backend = backend
        self._ns = "explain"
        self._ttl = 7 * 24 * 3600

    def key(self, names: list[str]) -> str:
        return self._make_key(self._ns, _hash("|".join(sorted(names))))

    def get(self, names: list[str]):
        return self._backend.get(self.key(names))

    def set(self, names: list[str], value):
        self._backend.set(self.key(names), value, self._ttl)

    def _make_key(self, ns: str, suffix: str) -> str:
        return f"{ns}:{suffix}"


# Singleton instances
_backend = DiskCacheBackend()
embedding_cache = EmbeddingCache(_backend)
vlm_cache = VLMCache(_backend)
explain_cache = ExplainCache(_backend)