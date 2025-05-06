import json
import sys
from os import path
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

from preprocessing.deplot_wrapper import deploit_table
from preprocessing.image_utils import MAX_PIXELS, resize_and_pad
from preprocessing.ocr_utils import extract_ocr_boxes
from preprocessing.prompt_builder import build_prompt

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import OUTPUT_PATH
from helpers.data_load import load_datasets, load_real_image_path

OCR_LANGUAGE = "eng"
OCR_CONFIDENCE_THRESHOLD = 60


def process_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    max_pixels: int = MAX_PIXELS,
    lang: str = "eng",
    conf_thr: int = 60,
    split_name: str = "train",
):

    output_dir.mkdir(parents=True, exist_ok=True)
    out_images = Path(path.join(output_dir, "images"))
    out_images.mkdir(exist_ok=True, parents=True)

    records = []

    for i, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        image_file = load_real_image_path(row["image_file"], **{split_name: True})
        question = row["question"]
        answer = row.get("answer", "")  # optional gold answer
        qa_pair_type = row.get("qa_pair_type", "graph")

        # Load full-res image
        img_full = Image.open(image_file).convert("RGB")
        orig_w, orig_h = img_full.size

        # OCR
        boxes = extract_ocr_boxes(img_full, lang=lang, conf_thr=conf_thr)

        # Resize & pad
        img_sq, pad_x, pad_y = resize_and_pad(img_full.copy(), max_pixels=max_pixels)
        canvas_w, canvas_h = img_sq.size
        scale_x = (canvas_w - 2 * pad_x) / orig_w
        scale_y = (canvas_h - 2 * pad_y) / orig_h

        # Transform + normalise boxes
        boxes_px: List[Tuple[int, int, int, int]] = []
        ocr_norm: List[Tuple[float, float, float, float, str]] = []
        for b in boxes:
            b2 = b.transform(scale_x, scale_y, pad_x, pad_y)
            x1, y1, x2, y2 = b2.normalised(canvas_w, canvas_h)
            boxes_px.append((b2.x, b2.y, b2.x + b2.w, b2.y + b2.h))
            ocr_norm.append((x1, y1, x2, y2, b2.text))

        # DePlot table
        table_str = deploit_table(img_sq)

        # Build prompt
        user_prompt = build_prompt(question, qa_pair_type, ocr_norm, table_str)

        # Save square image
        rel_img_path = Path(path.join("images", f"{row['instance_id']}.png"))
        img_sq.save(path.join(output_dir, rel_img_path))

        # Record JSONL entry
        records.append(
            {
                "image": str(rel_img_path),
                "user": user_prompt,
                "assistant": answer,
            }
        )

    # Write JSONL
    out_jsonl = Path(path.join(output_dir, "chart_vqa_instructions.jsonl"))
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(records)} instruction samples to {out_jsonl}")


def main():
    datasets = load_datasets(train=True, validation=False, test=False)

    for split_name, df in datasets.items():
        print(f"\n=== Processing {split_name} split ({len(df)} rows) ===")
        out_dir = Path(path.join(OUTPUT_PATH, split_name))
        process_dataset(
            df,
            out_dir,
            max_pixels=MAX_PIXELS,
            lang=OCR_LANGUAGE,
            conf_thr=OCR_CONFIDENCE_THRESHOLD,
            split_name=split_name,
        )

    print("\nAll splits processed.")


if __name__ == "__main__":
    main()
