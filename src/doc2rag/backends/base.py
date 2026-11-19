from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from doc2rag.models import Block


class BackendNotAvailable(RuntimeError):
    pass


class Backend(ABC):
    name: str = ""
    extensions: frozenset[str] = frozenset()

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @abstractmethod
    def parse(self, path: Path, artifacts_dir: Path, **options: Any) -> list[Block]:
        """Parse ``path`` into blocks; backends may write intermediate files to ``artifacts_dir``."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions


_REGISTRY: dict[str, type[Backend]] = {}
_INSTANCES: dict[str, Backend] = {}


def register_backend(cls: type[Backend]) -> type[Backend]:
    _REGISTRY[cls.name] = cls
    return cls


def backend_names() -> list[str]:
    return list(_REGISTRY)


def backend_class(name: str) -> type[Backend]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown backend {name!r}; available: {backend_names()}") from None


def get_backend(name: str) -> Backend:
    """Return a cached backend instance (model-heavy backends load once per process)."""
    if name not in _INSTANCES:
        from doc2rag.certs import maybe_use_system_certs

        cls = backend_class(name)
        if not cls.is_available():
            raise BackendNotAvailable(f"Backend {name!r} is not installed. Try: pip install \"doc2rag[{name}]\"")
        maybe_use_system_certs()
        _INSTANCES[name] = cls()
    return _INSTANCES[name]
