"""CLI wrapper for the NIH training pipeline."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nih_pipeline.pipeline import main


if __name__ == "__main__":
    main()
