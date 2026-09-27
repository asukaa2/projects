"""
i18n/i18n.py
============

Modern internationalization helper for the RVC WebUI.

Features
--------
* **Fallback chain.** Lookups go: requested locale -> parent locale
  (e.g. ``pt_BR`` -> ``pt``) -> ``en_US`` -> the source key. This lets
  us ship ~99 locale files even when only a few have full translations;
  untranslated keys gracefully fall back to English rather than to the
  Chinese source string.
* **Code normalization.** Accepts ``en_US``, ``en-US``, ``en`` (the
  latter resolves to the first matching file, e.g. ``en_US.json``).
* **Language registry.** Each locale has a human-readable display name
  (in English) and a script tag, used by the WebUI's language dropdown.
  See :mod:`i18n.locale.registry`.
* **Lazy + cached loading.** Locale files are read on first use and
  cached at the module level. Switching languages at runtime is cheap.
* **Thread-safe.** A lock guards the cache so concurrent Gradio threads
  don't double-load the same JSON.
* **Plural support.** ``i18n.n(key, count)`` returns ``key_plural`` or
  ``key_singular`` based on the count, with the same fallback chain.
* **Format string support.** Since ``__call__`` returns a ``str``, you
  can pass it straight to ``.format(...)`` or use it as an f-string
  template::

      i18n("Hello, {name}!").format(name=user)

* **CLI helpers.** :func:`available_languages` returns the full
  language registry; :func:`list_available_locales` returns only the
  locale files actually present on disk.

Migration notes
---------------
The previous ``I18nAuto`` class is preserved as a thin wrapper around
the new :class:`I18n` engine, so existing call sites that do::

    from i18n.i18n import I18nAuto
    i18n = I18nAuto()
    gr.Button(i18n("Some text"))

keep working unchanged. New code should prefer the module-level
:func:`i18n` callable and the :func:`set_language` /
:func:`get_language` helpers.
"""

from __future__ import annotations

import json
import locale as _locale_mod
import logging
import os
import threading
from importlib import resources
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The fallback locale used when the requested one is missing a key.
DEFAULT_LOCALE = "en_US"

#: The locale whose keys are the canonical source strings (Chinese).
SOURCE_LOCALE = "zh_CN"

#: Filesystem directory where the JSON locale files live.
_LOCALE_DIR = "i18n/locale"

#: Cache of loaded locale dicts, keyed by locale code (e.g. "en_US").
#:
#: The cache is module-level so that switching languages at runtime
#: doesn't re-read the same file twice. The lock below guards access
#: for thread safety (Gradio dispatches handlers in a thread pool).
_LOCALE_CACHE: Dict[str, Dict[str, str]] = {}
_CACHE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

def _load_registry() -> Dict[str, Dict[str, str]]:
    """Load the language registry lazily.

    The registry lives in ``i18n/locale/registry.py`` as a dict named
    ``LANGUAGES``. We import it on first use rather than at module
    import time so that the core ``i18n`` module stays lightweight.
    """
    try:
        from i18n.locale.registry import LANGUAGES  # type: ignore
        return LANGUAGES
    except Exception:  # pragma: no cover - defensive
        logger.warning("i18n.locale.registry could not be imported; "
                       "falling back to empty registry.")
        return {}


def available_languages() -> Dict[str, Dict[str, str]]:
    """Return the full language registry.

    The result is a dict ``{locale_code: {"name": str, "native": str,
    "script": str}}``. See ``i18n/locale/registry.py`` for the source.
    """
    return _load_registry()


# ---------------------------------------------------------------------------
# Locale code normalization
# ---------------------------------------------------------------------------

def _normalize_locale_code(code: Optional[str]) -> str:
    """Normalize a user-supplied locale code.

    Accepts ``en_US``, ``en-US``, ``en`` (resolves to the most-likely
    matching file on disk, e.g. ``en`` -> ``en_US``). Returns the empty
    string if the input is None or empty.

    The two-letter -> four-letter mapping prefers the most common
    variant (``en`` -> ``en_US``, ``zh`` -> ``zh_CN``, ``pt`` ->
    ``pt_BR``, etc.). If none of the preferred candidates exist on
    disk, the first matching stem alphabetically is used.
    """
    if not code:
        return ""
    code = code.strip().replace("-", "_")
    # ``en`` -> pick the most-likely matching file on disk.
    if "_" not in code and code not in _files_on_disk():
        # Order of preference for the language prefix -> region.
        preferred_regions = (
            "US", "CN", "GB", "TW", "HK", "SG", "FR", "DE", "ES",
            "PT", "BR", "RU", "JP", "KR", "IT", "TR", "IN", "ID",
            "MX", "CA", "AU",
        )
        candidates = [
            f"{code}_{r}.json"
            for r in preferred_regions
            if f"{code}_{r}.json" in _files_on_disk()
        ]
        if candidates:
            return candidates[0][:-5]  # strip ".json"
        # Fallback: alphabetically first matching file.
        for f in sorted(_files_on_disk()):
            stem = f[:-5]  # strip ".json"
            if stem.split("_")[0] == code:
                return stem
    return code


def _files_on_disk() -> List[str]:
    """Return the list of locale JSON files present on disk."""
    try:
        files = sorted(os.listdir(_LOCALE_DIR))
    except (FileNotFoundError, NotADirectoryError):
        try:
            # Fallback to importlib.resources (works inside installed
            # packages where the filesystem layout may differ).
            files = sorted(
                f.name for f in resources.files("i18n.locale").iterdir()
                if str(f).endswith(".json")
            )
        except Exception:
            files = []
    return [f for f in files if f.endswith(".json")]


def list_available_locales() -> List[str]:
    """Return locale codes that have a corresponding JSON file on disk."""
    out = []
    for f in _files_on_disk():
        if f.endswith(".json") and not f.startswith("registry"):
            out.append(f[:-5])
    return sorted(out)


# ---------------------------------------------------------------------------
# Locale file loading
# ---------------------------------------------------------------------------

def _load_locale_file(locale_code: str) -> Dict[str, str]:
    """Load and cache a single locale JSON file."""
    if not locale_code:
        return {}
    with _CACHE_LOCK:
        cached = _LOCALE_CACHE.get(locale_code)
        if cached is not None:
            return cached
        path = os.path.join(_LOCALE_DIR, f"{locale_code}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Could not load locale %r (%s); treating as empty.",
                            locale_code, exc)
            data = {}
        _LOCALE_CACHE[locale_code] = data
        return data


def _clear_cache() -> None:
    """Clear the locale cache (used by tests)."""
    with _CACHE_LOCK:
        _LOCALE_CACHE.clear()


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

def _fallback_chain(locale_code: str) -> Iterable[str]:
    """Yield locale codes in fallback order.

    For ``pt_BR``: yields ``pt_BR``, ``pt``, ``en_US``.
    For ``en``: yields ``en_US``.
    For ``zh``: yields ``zh_CN``.
    For an unknown code with no prefix: yields the code, then ``en_US``.
    """
    if not locale_code:
        yield DEFAULT_LOCALE
        return

    seen: set = set()
    code = locale_code
    while code and code not in seen:
        seen.add(code)
        yield code
        # Strip the region suffix (``pt_BR`` -> ``pt``) for the next hop.
        if "_" in code:
            code = code.split("_", 1)[0]
        else:
            # ``en`` -> ``en_US``, ``zh`` -> ``zh_CN``.
            short = code
            for candidate in (short + "_" + c for c in ("US", "CN", "GB", "TW", "HK", "SG")):
                if os.path.isfile(os.path.join(_LOCALE_DIR, candidate + ".json")):
                    code = candidate
                    break
            else:
                break
    if DEFAULT_LOCALE not in seen:
        yield DEFAULT_LOCALE


def _lookup(key: str, locale_code: str) -> Optional[str]:
    """Walk the fallback chain for ``locale_code`` looking for ``key``.

    Returns the first non-empty translation found, or ``None`` if none
    of the locales in the chain contain the key.
    """
    for code in _fallback_chain(locale_code):
        data = _load_locale_file(code)
        if key in data and data[key]:
            return data[key]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class I18n:
    """Stateful internationalization helper bound to a single locale.

    Instances are cheap to create and safe to share across threads.
    The locale file is loaded lazily on first lookup.
    """

    def __init__(self, language: Optional[str] = None):
        self.set_language(language)

    # ------------------------------------------------------------------
    # Language management
    # ------------------------------------------------------------------

    def set_language(self, language: Optional[str] = None) -> None:
        """Switch to a different language at runtime.

        If ``language`` is ``None``, ``"Auto"`` or empty, the system
        locale is detected via :func:`locale.getdefaultlocale`.
        """
        if language in (None, "", "Auto", "auto"):
            try:
                sys_lang = _locale_mod.getdefaultlocale()[0]
            except Exception:  # pragma: no cover - defensive
                sys_lang = None
            language = sys_lang or DEFAULT_LOCALE
        self.language = _normalize_locale_code(language) or DEFAULT_LOCALE

    def get_language(self) -> str:
        return self.language

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def __call__(self, key: str) -> str:
        """Return the translation for ``key`` in the current locale.

        Falls back through the parent locale, ``en_US``, and finally
        returns the source ``key`` if no translation is found anywhere
        in the chain.
        """
        if not isinstance(key, str):
            # Mirror the previous behavior: pass non-string keys through
            # untouched (e.g. ints used as identifiers).
            return key
        result = _lookup(key, self.language)
        if result is not None:
            return result
        # Final fallback: return the source key so that even totally
        # untranslated locales show *something* legible (the Chinese
        # source string, which is what the original code did).
        return key

    def n(self, key: str, count: int) -> str:
        """Plural-aware lookup.

        Looks up ``key_plural`` if ``count != 1`` else ``key_singular``,
        with the same fallback chain as :meth:`__call__`. If neither is
        found, falls back to ``key`` itself.
        """
        suffix = "plural" if count != 1 else "singular"
        plural_key = f"{key}_{suffix}"
        result = _lookup(plural_key, self.language)
        if result is not None:
            return result
        # Try the base key as a last resort.
        return self(key)

    def has(self, key: str) -> bool:
        """True if a translation exists for ``key`` in the current locale
        (or any ancestor in the fallback chain)."""
        return _lookup(key, self.language) is not None

    def __repr__(self) -> str:
        reg = available_languages()
        native = reg.get(self.language, {}).get("native", self.language)
        return f"I18n(locale={self.language}, native={native!r})"


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

#: The default module-level :class:`I18n` instance, lazily initialized.
#:
#: Code that wants a single shared translator (the common case) should
#: use the module-level :func:`i18n` / :func:`set_language` helpers,
#: which delegate to this instance. Code that wants independent
#: locale state (e.g. tests) should construct its own :class:`I18n`.
_default_i18n: Optional[I18n] = None
_default_lock = threading.Lock()


def _get_default() -> I18n:
    """Return (and lazily construct) the shared default :class:`I18n`."""
    global _default_i18n
    if _default_i18n is None:
        with _default_lock:
            if _default_i18n is None:
                _default_i18n = I18n()
    return _default_i18n


def i18n(key: str) -> str:
    """Translate ``key`` using the shared default :class:`I18n` instance."""
    return _get_default()(key)


def set_language(language: Optional[str]) -> None:
    """Change the language of the shared default :class:`I18n`."""
    _get_default().set_language(language)


def get_language() -> str:
    """Return the locale code of the shared default :class:`I18n`."""
    return _get_default().get_language()


# ---------------------------------------------------------------------------
# Backwards compatibility: I18nAuto
# ---------------------------------------------------------------------------

class I18nAuto(I18n):
    """Legacy alias for :class:`I18n`.

    Older code (and the upstream RVC project) instantiate this class
    directly. The new implementation is a subclass of :class:`I18n`
    so existing call sites keep working without modification.
    """

    def __init__(self, language=None):
        super().__init__(language)


def load_language_list(language: str) -> Dict[str, str]:
    """Load a single locale file by code (kept for back-compat).

    The previous module exposed this as a top-level function; we keep
    it so any external code that imports it directly still works.
    """
    return _load_locale_file(_normalize_locale_code(language))


__all__ = [
    "I18n",
    "I18nAuto",
    "i18n",
    "set_language",
    "get_language",
    "load_language_list",
    "available_languages",
    "list_available_locales",
    "DEFAULT_LOCALE",
    "SOURCE_LOCALE",
]
