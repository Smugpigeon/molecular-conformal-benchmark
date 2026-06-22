"""rdkit <-> pandas compatibility shim.

rdkit 2024.03's ``Chem.PandasTools.InstallPandasTools()`` (run at import time)
reads ``pandas.io.formats.format.get_adjustment``. pandas 2.2 relocated that
symbol to ``pandas.io.formats.printing``. rdkit normally detects the pandas
version and adapts, but that detection silently breaks when ``pkg_resources``
is unavailable (e.g. after setuptools >= 81), falling back to the legacy path
and raising ``AttributeError`` on ``import unimol_tools`` (which pulls in
PandasTools transitively).

Calling :func:`patch_rdkit_pandas_compat` before importing ``unimol_tools``
aliases the symbol back so the import succeeds regardless of the installed
setuptools version. Idempotent and dependency-light.

See CLAUDE.md §11 (pitfalls table) for the war story.
"""

from __future__ import annotations


def patch_rdkit_pandas_compat() -> None:
    """Alias ``get_adjustment`` into ``pandas.io.formats.format`` if missing.

    No-op when the attribute already exists (older pandas, or already patched).
    Best-effort: any failure is swallowed so the genuine downstream import
    error (if any) surfaces with its own traceback.
    """
    try:
        import pandas.io.formats.format as _f
        import pandas.io.formats.printing as _p

        if not hasattr(_f, "get_adjustment") and hasattr(_p, "get_adjustment"):
            _f.get_adjustment = _p.get_adjustment  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001  (shim must never mask the real import error)
        pass
