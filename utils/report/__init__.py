"""Static and interactive reporting entry points.

``from utils.report import run`` historically referred to the former flat
``utils/report.py`` module.  Resolve those attributes lazily while keeping this
directory available as a normal package.
"""

from importlib import import_module


def __getattr__(name: str):
    return getattr(import_module("utils.report.report"), name)
