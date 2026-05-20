import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))
