"""Resize NIH PNG images into a smaller grayscale training cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def resize_one(task: tuple[str, str, int]) -> str:
    src_str, dst_str, size = task
    src = Path(src_str)
    dst = Path(dst_str)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        try:
            with Image.open(dst) as existing:
                if existing.size == (size, size) and existing.mode == "L":
                    return "skipped"
        except Exception:
            pass

    with Image.open(src) as image:
        resized = image.convert("L").resize((size, size), resample=Image.Resampling.LANCZOS)
        resized.save(dst, format="PNG", optimize=True)
    return "written"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize NIH image set before training.")
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()
    files = sorted(src_dir.glob("*.png"))
    tasks = [(str(src), str(dst_dir / src.name), args.size) for src in files]

    written = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for idx, result in enumerate(pool.map(resize_one, tasks, chunksize=64), start=1):
            if result == "written":
                written += 1
            else:
                skipped += 1
            if idx % 2000 == 0 or idx == len(tasks):
                print(f"{idx}/{len(tasks)} processed | written={written} skipped={skipped}", flush=True)

    print(f"done | total={len(tasks)} written={written} skipped={skipped} dst={dst_dir}", flush=True)


if __name__ == "__main__":
    main()
