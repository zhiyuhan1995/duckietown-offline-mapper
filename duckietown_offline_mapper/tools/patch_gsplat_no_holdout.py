#!/usr/bin/env python3
"""Patch gsplat's COLMAP example dataset so test_every <= 0 means train-all.

The Duckietown 3DGS QA run uses every third video frame for training and does
not need a validation holdout during long render-quality checks. The upstream
gsplat example treats test_every as a modulo divisor, so test_every=0 needs a
small local patch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
"""

NEW = """        if self.parser.test_every <= 0:
            if split == "train":
                self.indices = indices
            else:
                self.indices = indices[:0]
        elif split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]
"""


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return False
    if OLD not in text:
        raise RuntimeError(f"Could not find the expected Dataset split block in {path}")
    path.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gsplat-colmap-py",
        default="external/gsplat-v1.3.0/examples/datasets/colmap.py",
        help="Path to gsplat examples/datasets/colmap.py",
    )
    return parser.parse_args()


def main() -> None:
    path = Path(parse_args().gsplat_colmap_py)
    changed = patch(path)
    print(f"{'patched' if changed else 'already patched'}: {path}")


if __name__ == "__main__":
    main()
