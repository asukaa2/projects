"""
i18n/scan_i18n.py
================

Modern AST-based scanner for i18n() calls.

For every ``.py`` file under the project (or under a directory given
on the command line), this script:

1. Parses the source into an AST.
2. Finds every call to one of the configured translation function names
   (``i18n`` by default; ``_`` and ``gettext`` can be added via
   ``--function``).
3. Extracts the first *constant string* argument from each call. Only
   literal strings are extracted — f-strings, string concatenations,
   and variable references are reported but not added to the key set
   (the runtime :class:`~i18n.i18n.I18n` falls back to the source key
   when a translation is missing, so this is safe).
4. Compares the extracted keys against the standard locale file
   (``i18n/locale/zh_CN.json`` by default — zh_CN is the canonical
   source language whose keys *are* the source strings).
5. Reports unused and missing keys, and (unless ``--no-write`` is
   given) rewrites the standard file so its keys exactly match the
   code.

The previous implementation used :class:`ast.Str`, which is deprecated
in Python 3.8+ (it was merged into :class:`ast.Constant`); this
rewrite uses :class:`ast.Constant` and works on Python 3.8 through
3.13+.

CLI usage
---------

.. code-block:: bash

    # Scan the whole project, update zh_CN.json with the extracted keys
    python -m i18n.scan_i18n

    # Just print a report, don't rewrite anything
    python -m i18n.scan_i18n --no-write

    # Also treat _() and gettext() as translation functions
    python -m i18n.scan_i18n --function _ --function gettext

    # Restrict the scan to a subdirectory
    python -m i18n.scan_i18n infer/
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
from collections import OrderedDict
from typing import Iterable, List, Set, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: The locale file whose keys are the canonical source strings.
DEFAULT_STANDARD_FILE = "i18n/locale/zh_CN.json"

#: Translation function names recognized by default. Extend with
#: ``--function`` on the CLI.
DEFAULT_FUNCTION_NAMES: Tuple[str, ...] = ("i18n",)


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------

def extract_i18n_strings(
    tree: ast.AST,
    function_names: Set[str],
) -> Tuple[List[str], List[Tuple[str, int, str]]]:
    """Walk an AST and collect strings from translation calls.

    Parameters
    ----------
    tree:
        The parsed AST of a single Python file.
    function_names:
        Names to treat as translation functions (e.g. ``{"i18n"}``).

    Returns
    -------
    (keys, dynamic_calls)
        ``keys`` is the list of constant-string first-arguments found
        in translation calls (may contain duplicates — callers
        usually dedup). ``dynamic_calls`` is a list of
        ``(filename, lineno, repr)`` for calls whose first argument
        was *not* a constant string, so reviewers can spot
        accidentally-parameterized keys.
    """
    keys: List[str] = []
    dynamic: List[Tuple[str, int, str]] = []  # filled by caller with file name

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Direct name calls:  i18n("...")
        if isinstance(func, ast.Name) and func.id in function_names:
            pass
        # Attribute calls:  self.i18n("...") or i18n.i18n("...")
        elif isinstance(func, ast.Attribute) and func.attr in function_names:
            pass
        else:
            continue

        if not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            keys.append(arg0.value)
        else:
            # Dynamic first-arg (f-string, variable, etc.) — record so
            # reviewers can spot accidentally-parameterized keys.
            try:
                snippet = ast.unparse(arg0)
            except Exception:
                snippet = f"<{type(arg0).__name__}>"
            dynamic.append(("<unknown>", node.lineno, snippet))

    return keys, dynamic


def scan_file(
    path: str,
    function_names: Set[str],
) -> Tuple[List[str], List[Tuple[str, int, str]]]:
    """Scan a single .py file. Returns the same shape as
    :func:`extract_i18n_strings`, but with the filename filled in on
    the dynamic_calls tuples.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
    except (OSError, UnicodeDecodeError):
        return [], []
    if not any(fn + "(" in code for fn in function_names):
        # Fast path: skip files that don't even mention the function name.
        return [], []
    try:
        tree = ast.parse(code, filename=path)
    except SyntaxError:
        return [], []
    keys, dynamic = extract_i18n_strings(tree, function_names)
    dynamic = [(path, line, snippet) for _f, line, snippet in dynamic]
    return keys, dynamic


# ---------------------------------------------------------------------------
# Standard file management
# ---------------------------------------------------------------------------

def load_standard(standard_file: str) -> "OrderedDict[str, str]":
    """Load the standard locale file (zh_CN.json)."""
    with open(standard_file, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def write_standard(
    standard_file: str,
    keys: Iterable[str],
    existing: "OrderedDict[str, str]",
) -> None:
    """Rewrite the standard locale file.

    New keys are added with the key itself as the value (the source
    string IS the key in this scheme). Existing translations are
    preserved. The output is sorted alphabetically by key for
    reproducible diffs.
    """
    out: "OrderedDict[str, str]" = OrderedDict()
    # Preserve existing values for keys that still appear in the code.
    for k in sorted(set(keys)):
        if k in existing:
            out[k] = existing[k]
        else:
            out[k] = k  # new key — value = source string
    with open(standard_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scan_i18n",
        description="Scan .py files for i18n() calls and sync the "
                    "standard locale file.",
    )
    parser.add_argument(
        "path", nargs="?", default=".",
        help="File or directory to scan (default: current dir).",
    )
    parser.add_argument(
        "--standard", default=DEFAULT_STANDARD_FILE,
        help="Standard locale file to sync (default: %(default)s).",
    )
    parser.add_argument(
        "--function", action="append", default=None,
        help="Translation function name (default: i18n). May be given "
             "multiple times to scan multiple names (e.g. _, gettext).",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Just print the report; do not rewrite the standard file.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-file counts and dynamic-arg warnings.",
    )
    args = parser.parse_args(argv)

    function_names: Set[str] = set(args.function or list(DEFAULT_FUNCTION_NAMES))

    # Build the file list.
    if os.path.isfile(args.path):
        py_files = [args.path]
    else:
        py_files = sorted(
            glob.glob(os.path.join(args.path, "**", "*.py"), recursive=True)
        )

    all_keys: List[str] = []
    all_dynamic: List[Tuple[str, int, str]] = []
    per_file: dict[str, int] = {}
    for path in py_files:
        # Skip the scanner itself.
        if os.path.normpath(path) == os.path.normpath(__file__):
            continue
        keys, dynamic = scan_file(path, function_names)
        if keys or dynamic:
            per_file[path] = len(keys)
        all_keys.extend(keys)
        all_dynamic.extend(dynamic)

    if args.verbose:
        print(f"Scanned {len(py_files)} .py files; "
              f"{sum(per_file.values())} i18n() calls; "
              f"{len(set(all_keys))} unique keys.")
        for path, count in sorted(per_file.items()):
            print(f"  {count:>4d}  {path}")
        if all_dynamic:
            print(f"\nDynamic first-args (cannot be auto-extracted): {len(all_dynamic)}")
            for path, line, snippet in all_dynamic[:20]:
                print(f"  {path}:{line}: {snippet}")
            if len(all_dynamic) > 20:
                print(f"  ... and {len(all_dynamic) - 20} more")

    # Compare against the standard file.
    standard = load_standard(args.standard)
    standard_keys: Set[str] = set(standard.keys())
    code_keys: Set[str] = set(all_keys)

    missing = code_keys - standard_keys
    unused = standard_keys - code_keys
    print(f"\nStandard file: {args.standard}")
    print(f"  code keys:  {len(code_keys)}")
    print(f"  std  keys:  {len(standard_keys)}")
    print(f"  missing (in code, not in std): {len(missing)}")
    print(f"  unused  (in std, not in code): {len(unused)}")
    if missing and args.verbose:
        print("  missing keys:")
        for k in sorted(missing)[:50]:
            print(f"    + {k!r}")
    if unused and args.verbose:
        print("  unused keys:")
        for k in sorted(unused)[:50]:
            print(f"    - {k!r}")

    if not args.no_write:
        write_standard(args.standard, all_keys, standard)
        print(f"\nUpdated {args.standard} "
              f"(preserved {len(standard_keys & code_keys)} existing, "
              f"added {len(missing)}, dropped {len(unused)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Legacy entry points (kept so `python i18n/scan_i18n.py` still works)
# ---------------------------------------------------------------------------

# The module-level statements below reproduce the old behavior of
# running the scan immediately on import. They are guarded by a
# ``__main__`` check above; running the file as a script now uses the
# argparse-based CLI. To run the scan with default settings from a
# script, do ``python -m i18n.scan_i18n``.
