"""MSIEOF utility package with lazy compatibility aliases.

New code should import from the responsibility-based subpackages, for example
``utils.pipeline.arrow_sr`` or ``utils.base.chemical_graph``.  Historical flat
imports remain available during the migration period without eagerly loading
PyTorch, Ultralytics, MolScribe, or RDKit.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


class _LazyModule(ModuleType):
    """Load the target module only when one of its attributes is accessed."""

    def __init__(self, name: str, target: str):
        super().__init__(name)
        self._target = target
        self._real: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._real is None:
            self._real = importlib.import_module(self._target)
        return self._real

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._load(), name)

    def __dir__(self):
        return dir(self._load())


_ALIASES = {
    # Production pipeline.
    "extract": "utils.pipeline.extract",
    "fragment_cleanup": "utils.pipeline.fragment_cleanup",
    "fragment_cleanup": "utils.pipeline.fragment_cleanup",
    "post_segment": "utils.pipeline.post_segment",
    "arrow_sr": "utils.pipeline.arrow_sr",
    "img2smiles": "utils.pipeline.img2smiles",
    "resolution_classfier": "utils.pipeline.resolution_classifier",
    "resolution_classifier": "utils.pipeline.resolution_classifier",
    # Shared libraries.
    "image_loader": "utils.base.image_loader",
    "image_utils": "utils.base.image_utils",
    "viz_utils": "utils.base.viz_utils",
    "chem_utils": "utils.base.chem_utils",
    "chemical_graph": "utils.base.chemical_graph",
    "smiles_canon": "utils.base.smiles_canon",
    # Evaluation and benchmarks.
    "evaluate_page2smiles": "utils.evaluation.evaluate_page2smiles",
    "evaluate_page2smiles_full": "utils.evaluation.evaluate_page2smiles_full",
    "build_pagesmiles_jsonl": "utils.evaluation.build_pagesmiles_jsonl",
    "run_quick_test": "utils.evaluation.run_quick_test",
    "tableB_detection": "utils.evaluation.benchmarks.table_b_detection",
    "tableB_end2end": "utils.evaluation.benchmarks.table_b_end2end",
    "_verify_decimer_e2e": "utils.evaluation.benchmarks.verify_decimer_e2e",
    # Experiments, reports, data tools, and maintenance.
    "ablation_unified": "utils.experiments.ablation_unified",
    "ablation_v2": "utils.experiments.ablation_v2",
    "arrow_protection": "utils.experiments.arrow_protection",
    "report_interactive": "utils.report.report_interactive",
    "visualize_hits": "utils.report.visualize_hits",
    "render_tables_png": "utils.report.render_tables_png",
    "download_images": "utils.data_tools.download_images",
    "find_lowres_arrow": "utils.data_tools.find_lowres_arrow",
    "fix_classifier_pickle": "utils.maintenance.fix_classifier_pickle",
    "verify_fragment_cleanup_migration": (
        "utils.maintenance.verify_fragment_cleanup_migration"
    ),
}


def _register_aliases() -> None:
    for old_name, target in _ALIASES.items():
        qualified = f"{__name__}.{old_name}"
        proxy = _LazyModule(qualified, target)
        sys.modules[qualified] = proxy
        globals()[old_name] = proxy


_register_aliases()
del _register_aliases
