"""Operator-facing CLIs.

These are thin, importable wrappers around the existing ops scripts under
``scripts/``. Console-script entrypoints declared in ``pyproject.toml`` point
at the :func:`main` functions here so ``pip install .`` gives operators real
commands instead of path-dependent ``python scripts/…`` invocations.
"""

from __future__ import annotations
