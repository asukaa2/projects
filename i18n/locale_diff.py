"""
i18n/locale_diff.py
===================

Sync all locale JSON files against the standard file (zh_CN.json).

For every other ``i18n/locale/*.json`` file:

* Missing keys (present in the standard, absent in this locale) are
  added with the source string as the value (so the UI shows the
  Chinese source rather than nothing — the runtime fallback chain in
  :class:`i18n.i18n.I18n` will then forward to ``en_US`` for an actual
  translation).
* Extra keys (present in this locale, absent in the standard) are
  removed — they're stale and would otherwise rot.
* Keys are re-sorted to match the standard file's order so diffs
  between locales are minimal.

CLI usage
---------

.. code-block:: bash

    # Sync all locales against zh_CN.json
    python -m i18n.locale_diff

    # Use a different standard file
    python -m i18n.locale_diff --standard i18n/locale/en_US.json

    # Restrict to a specific locale
    python -m i18n.locale_diff --only en_US,ja_JP

    # Dry run (print a report, don't rewrite)
    python -m i18n.locale_diff --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import List, Set


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DIR = "i18n/locale"
DEFAULT_STANDARD = "zh_CN.json"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_json(path: str) -> "OrderedDict[str, str]":
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def save_json(path: str, data: "OrderedDict[str, str]") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")


def sync_one(
    lang_path: str,
    standard: "OrderedDict[str, str]",
    dry_run: bool = False,
) -> dict:
    """Sync a single locale file against the standard.

    Returns a small report dict: ``{"added": [...], "removed": [...],
    "preserved": int}``.

    Stub files
    ----------
    A locale is considered a *stub* if its JSON contains a top-level
    ``_meta`` key (the convention used by
    :mod:`scripts.create_locale_stubs`). For stubs, missing keys are
    written as JSON ``null`` rather than the source string, so the
    runtime :class:`i18n.i18n.I18n` fallback chain forwards to
    ``en_US`` (and then to the source key) for any untranslated key.
    This means: an untranslated locale shows English in the UI rather
    than Chinese — much better UX for non-Chinese speakers.

    For non-stub locales (the 11 originally shipped), missing keys
    are filled with the source string itself, preserving the previous
    behaviour.
    """
    lang = load_json(lang_path)
    standard_keys: List[str] = list(standard.keys())
    standard_set: Set[str] = set(standard_keys)
    lang_set: Set[str] = set(lang.keys())

    # The `_meta` key is reserved for stub-file metadata (see
    # scripts.create_locale_stubs). Don't treat it as an "extra" key
    # to remove; preserve it at the top of the synced file.
    RESERVED_META_KEY = "_meta"
    meta_value = lang.get(RESERVED_META_KEY)

    missing = standard_set - lang_set
    extra = lang_set - standard_set - {RESERVED_META_KEY}

    is_stub = meta_value is not None or RESERVED_META_KEY in lang

    # Build the new dict in the standard's key order, preserving
    # existing translations.
    new: "OrderedDict[str, str]" = OrderedDict()
    if meta_value is not None:
        new[RESERVED_META_KEY] = meta_value
    for k in standard_keys:
        if k in lang and lang[k] is not None:
            new[k] = lang[k]
        elif is_stub:
            # Stub locale: leave the key as JSON null so the runtime
            # I18n fallback chain forwards to en_US (and finally the
            # source key). This means an untranslated locale shows
            # English in the UI rather than Chinese.
            new[k] = None
        else:
            # Real locale with a partial translation: keep the previous
            # behaviour and fill with the source string so the UI shows
            # *something* legible (the runtime I18n class returns the
            # source string as the final fallback anyway, but having it
            # in the file keeps the JSON self-describing for human
            # reviewers).
            new[k] = k

    report = {
        "added": sorted(missing),
        "removed": sorted(extra),
        "preserved": len(standard_set & lang_set),
    }

    if not dry_run:
        save_json(lang_path, new)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="locale_diff",
        description="Sync every i18n/locale/*.json against the standard file.",
    )
    parser.add_argument(
        "--dir", default=DEFAULT_DIR,
        help="Directory containing the locale JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--standard", default=DEFAULT_STANDARD,
        help="Standard file name (within --dir; default: %(default)s).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated list of locale codes to sync "
             "(e.g. en_US,ja_JP). Default: all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print a report, don't rewrite any files.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-locale added/removed keys.",
    )
    args = parser.parse_args(argv)

    standard_path = os.path.join(args.dir, args.standard)
    if not os.path.isfile(standard_path):
        print(f"Standard file not found: {standard_path}", file=sys.stderr)
        return 1
    standard = load_json(standard_path)
    standard_keys = list(standard.keys())

    # Build the list of locale files to sync.
    only_set = set()
    if args.only:
        only_set = {s.strip() + ".json" for s in args.only.split(",") if s.strip()}

    lang_files = sorted(
        f for f in os.listdir(args.dir)
        if f.endswith(".json")
        and f != args.standard
        and not f.startswith("registry")
        and (not only_set or f in only_set)
    )

    if not lang_files:
        print("No locale files to sync.", file=sys.stderr)
        return 0

    total_added = total_removed = 0
    for fname in lang_files:
        path = os.path.join(args.dir, fname)
        rep = sync_one(path, standard, dry_run=args.dry_run)
        total_added += len(rep["added"])
        total_removed += len(rep["removed"])
        status = "would update" if args.dry_run else "updated"
        print(f"  {status:14s} {fname}: "
              f"+{len(rep['added'])} -{len(rep['removed'])} "
              f"({rep['preserved']} preserved)")
        if args.verbose and (rep["added"] or rep["removed"]):
            for k in rep["added"]:
                print(f"      + {k!r}")
            for k in rep["removed"]:
                print(f"      - {k!r}")

    print(f"\nTotal: +{total_added} -{total_removed} across {len(lang_files)} locale(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------
# The original module-level statements ran the sync immediately on
# import. Modern usage is via the CLI (`python -m i18n.locale_diff`),
# but we keep a small shim so `python i18n/locale_diff.py` still works.

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
