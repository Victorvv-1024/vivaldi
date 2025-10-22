import json
import os
from pathlib import Path
from typing import Iterable

from obb_anns.obb_anns import OBBAnns

DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "deepscore"
LABELS_DIR = DATASET_ROOT / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def convert_box_to_yolo(bbox: Iterable[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def convert_split(json_path: Path) -> list[str]:
    print(f"Converting {json_path.name}...")
    obb = OBBAnns(str(json_path))
    obb.load_annotations()
    obb.set_annotation_set_filter(["deepscores"])
    img_indices = list(range(len(obb.img_info)))
    imgs, anns = obb.get_img_ann_pair(idxs=img_indices)

    missing = []
    for img_info, ann_df in zip(imgs, anns):
        label_path = LABELS_DIR / f"{Path(img_info['filename']).stem}.txt"
        width, height = img_info["width"], img_info["height"]
        lines = []
        if not ann_df.empty:
            for cat_ids, bbox in zip(ann_df["cat_id"], ann_df["a_bbox"]):
                if not cat_ids or bbox is None:
                    continue
                # cat_ids entries are lists of stringified ints; pick the deepscores id (first element)
                class_id = int(cat_ids[0]) - 1
                lines.append(f"{class_id} {convert_box_to_yolo(bbox, width, height)}\n")
        if lines:
            label_path.write_text("".join(lines))
        else:
            # ensure an empty file exists so downstream scripts can open it
            label_path.touch()
    return [img["filename"] for img in imgs]


def main() -> None:
    split_lists: dict[str, list[str]] = {"train": [], "test": []}
    json_files = sorted(DATASET_ROOT.glob("deepscores-complete-*_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No DeepScores JSON files found under {DATASET_ROOT}")

    for json_path in json_files:
        filenames = convert_split(json_path)
        if json_path.stem.endswith("_train"):
            split_lists.setdefault("train", []).extend(filenames)
        elif json_path.stem.endswith("_test"):
            split_lists.setdefault("test", []).extend(filenames)

    for split_name, filenames in split_lists.items():
        if not filenames:
            continue
        list_path = DATASET_ROOT / f"deepscores_{split_name}.txt"
        with list_path.open("w") as f:
            for filename in filenames:
                f.write(f"./images/{filename}\n")
        print(f"Wrote {len(filenames)} entries to {list_path}")


if __name__ == "__main__":
    main()
