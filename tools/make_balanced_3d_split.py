from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class Sample:
    stem: str
    source_split: str
    main: Path
    detail1: Path
    detail2: Path
    label: Path
    is_background: bool
    area: float
    center_x: float
    center_y: float
    stratum: tuple


def find_image(directory: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing image for {stem} in {directory}")


def quantile_bins(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    cuts = []
    for i in range(1, bins):
        idx = round((len(values) - 1) * i / bins)
        cuts.append(values[idx])
    return cuts


def bin_value(value: float, cuts: list[float]) -> int:
    for i, cut in enumerate(cuts):
        if value <= cut:
            return i
    return len(cuts)


def parse_label(label: Path) -> tuple[bool, float, float, float]:
    lines = [line.strip() for line in label.read_text(errors="ignore").splitlines() if line.strip()]
    if not lines:
        return True, 0.0, 0.0, 0.0
    parts = lines[0].split()
    if len(parts) != 5:
        raise ValueError(f"invalid label columns: {label}")
    _, x, y, w, h = parts
    x, y, w, h = float(x), float(y), float(w), float(h)
    return False, w * h, x, y


def collect_samples(source: Path) -> list[Sample]:
    raw = []
    object_areas, object_xs, object_ys = [], [], []
    for split in ("train", "val", "test"):
        main_dir = source / "images" / split
        detail1_dir = main_dir / "detail1"
        detail2_dir = main_dir / "detail2"
        label_dir = source / "labels" / split
        for main in sorted(p for p in main_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            stem = main.stem
            label = label_dir / f"{stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"missing label for {stem}")
            detail1 = find_image(detail1_dir, stem)
            detail2 = find_image(detail2_dir, stem)
            is_background, area, center_x, center_y = parse_label(label)
            raw.append((stem, split, main, detail1, detail2, label, is_background, area, center_x, center_y))
            if not is_background:
                object_areas.append(area)
                object_xs.append(center_x)
                object_ys.append(center_y)

    area_cuts = quantile_bins(object_areas, 5)
    x_cuts = quantile_bins(object_xs, 4)
    y_cuts = quantile_bins(object_ys, 3)

    samples = []
    for stem, split, main, detail1, detail2, label, is_background, area, center_x, center_y in raw:
        if is_background:
            stratum = ("bg",)
        else:
            stratum = (
                "obj",
                bin_value(area, area_cuts),
                bin_value(center_x, x_cuts),
                bin_value(center_y, y_cuts),
            )
        samples.append(Sample(stem, split, main, detail1, detail2, label, is_background, area, center_x, center_y, stratum))
    return samples


def assign_splits(samples: list[Sample], seed: int) -> dict[str, str]:
    targets = {"train": 1280, "val": 159, "test": 161}
    ratio = {k: v / len(samples) for k, v in targets.items()}
    groups: dict[tuple, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.stratum, []).append(sample)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    counts = {k: 0 for k in targets}

    for group_samples in groups.values():
        rng.shuffle(group_samples)
        n = len(group_samples)
        want_val = round(n * ratio["val"])
        want_test = round(n * ratio["test"])
        want_train = n - want_val - want_test
        split_plan = ["val"] * want_val + ["test"] * want_test + ["train"] * want_train
        rng.shuffle(split_plan)
        for sample, split in zip(group_samples, split_plan):
            assignment[sample.stem] = split
            counts[split] += 1

    for split in ("val", "test", "train"):
        while counts[split] > targets[split]:
            receiver = next(k for k in ("train", "val", "test") if counts[k] < targets[k])
            donor = next(stem for stem, s in assignment.items() if s == split)
            assignment[donor] = receiver
            counts[split] -= 1
            counts[receiver] += 1

    for split in ("val", "test", "train"):
        while counts[split] < targets[split]:
            donor_split = next(k for k in ("train", "val", "test") if counts[k] > targets[k])
            donor = next(stem for stem, s in assignment.items() if s == donor_split)
            assignment[donor] = split
            counts[donor_split] -= 1
            counts[split] += 1

    return assignment


def copy_sample(sample: Sample, split: str, dest: Path) -> None:
    image_dir = dest / "images" / split
    detail1_dir = image_dir / "detail1"
    detail2_dir = image_dir / "detail2"
    label_dir = dest / "labels" / split
    for directory in (image_dir, detail1_dir, detail2_dir, label_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(sample.main, image_dir / sample.main.name)
    shutil.copy2(sample.detail1, detail1_dir / sample.detail1.name)
    shutil.copy2(sample.detail2, detail2_dir / sample.detail2.name)
    shutil.copy2(sample.label, label_dir / sample.label.name)


def summarize(samples: list[Sample], assignment: dict[str, str]) -> dict:
    summary = {}
    for split in ("train", "val", "test"):
        subset = [sample for sample in samples if assignment[sample.stem] == split]
        objects = [sample for sample in subset if not sample.is_background]
        areas = [sample.area for sample in objects]
        xs = [sample.center_x for sample in objects]
        ys = [sample.center_y for sample in objects]
        summary[split] = {
            "images": len(subset),
            "backgrounds": sum(sample.is_background for sample in subset),
            "objects": len(objects),
            "mean_area": sum(areas) / len(areas) if areas else 0.0,
            "mean_center_x": sum(xs) / len(xs) if xs else 0.0,
            "mean_center_y": sum(ys) / len(ys) if ys else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a copy-only balanced split for triple-input YOLO data.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.source = args.source.resolve()
    args.dest = args.dest.resolve()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if args.dest.exists() and any(args.dest.iterdir()):
        raise RuntimeError(f"destination is not empty: {args.dest}")

    samples = collect_samples(args.source)
    if len(samples) != 1600:
        raise RuntimeError(f"expected 1600 samples, got {len(samples)}")

    assignment = assign_splits(samples, args.seed)
    for sample in samples:
        copy_sample(sample, assignment[sample.stem], args.dest)

    summary = summarize(samples, assignment)
    (args.dest / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.dest / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "source_split", "new_split", "background", "area", "center_x", "center_y"])
        for sample in sorted(samples, key=lambda x: x.stem):
            writer.writerow([
                sample.stem,
                sample.source_split,
                assignment[sample.stem],
                int(sample.is_background),
                sample.area,
                sample.center_x,
                sample.center_y,
            ])

    data_yaml = "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: object\n"
    (args.dest / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
