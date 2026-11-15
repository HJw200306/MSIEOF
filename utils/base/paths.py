"""Central project paths shared by relocated utility modules."""

from pathlib import Path


UTILS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = UTILS_ROOT.parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIS_DIR = PROJECT_ROOT / "vis"


__all__ = [
    "DATA_DIR",
    "MODEL_DIR",
    "OUTPUTS_DIR",
    "PROJECT_ROOT",
    "UTILS_ROOT",
    "VIS_DIR",
]
