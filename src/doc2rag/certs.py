"""Opt-in: trust the operating system's certificate store (corporate TLS proxies)."""

from __future__ import annotations

import logging
import os

log = logging.getLogger("doc2rag")


def use_system_certs() -> bool:
    """Make Python's ``ssl`` (requests/httpx/huggingface_hub) trust the OS certificate store.

    Needed only behind TLS-inspecting proxies (Zscaler, Netskope, ...) when Docling or
    MinerU download their models on first use. Requires ``pip install "doc2rag[certs]"``.
    Returns ``False`` if ``truststore`` is not installed.

    Note: this affects the current process only. MinerU runs as a subprocess; behind
    such a proxy, point ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` at your corporate CA
    bundle, or pre-download models and run offline.
    """
    try:
        import truststore
    except ImportError:
        log.warning("truststore not installed; run: pip install \"doc2rag[certs]\"")
        return False
    truststore.inject_into_ssl()
    return True


def maybe_use_system_certs() -> None:
    if os.environ.get("DOC2RAG_SYSTEM_CERTS", "").lower() in ("1", "true", "yes"):
        use_system_certs()
