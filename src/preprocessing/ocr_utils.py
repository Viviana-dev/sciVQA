import sys
from os import path

import pytesseract
from PIL import Image, ImageDraw, ImageFont

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.data_load import load_datasets, load_real_image_path
from preprocessing.bbox import BBox


def extract_ocr_boxes(img: Image.Image, lang: str = "eng", conf_thr: int = 60) -> list[BBox]:
    """Run Tesseract OCR and return cleaned BBoxes."""

    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

    boxes: list[BBox] = []
    n = len(data["text"])
    for i in range(n):
        if int(data["conf"][i]) < conf_thr:
            continue
        text = " ".join(data["text"][i].strip().split())
        if not text:
            continue
        box = BBox(
            x=data["left"][i],
            y=data["top"][i],
            w=data["width"][i],
            h=data["height"][i],
            text=text,
            conf=int(data["conf"][i]),
        )
        boxes.append(box)

    # Deduplicate identical strings that overlap heavily (keeps 1st occurrence)
    uniq = {}
    for b in boxes:
        if b.text.lower() not in uniq:
            uniq[b.text.lower()] = b
    return list(uniq.values())


def visualize_ocr_boxes(
    img_path: str,
    lang: str = "eng",
    conf_thr: int = 60,
    color: str = "red",
    show: bool = True,
    save_path: str | None = None,
) -> list[BBox]:
    """Run Tesseract on *img_path* and draw the resulting bounding boxes.

    Parameters
    ----------
    img_path : str
        Path to the input image.
    lang : str, default "eng"
        OCR language(s) recognised by Tesseract.
    conf_thr : int, default 60
        Minimum confidence required to keep a token.
    color : str, default "red"
        Outline/text colour for the visualisation.
    show : bool, default True
        If ``True`` opens the annotated image with the default viewer via
        :pymeth:`PIL.Image.Image.show`.
    save_path : str | None
        If given, the annotated image is written to this path.

    Returns
    -------
    list[BBox]
        The list of deduplicated OCR bounding boxes.
    """
    img = Image.open(img_path).convert("RGB")
    boxes = extract_ocr_boxes(img, lang=lang, conf_thr=conf_thr)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except IOError:  # On headless servers a default bitmap font is still OK
        font = None

    for b in boxes:
        # Draw rectangle
        draw.rectangle(
            [(b.x, b.y), (b.x + b.w, b.y + b.h)],
            outline=color,
            width=2,
        )
        # Draw label slightly above the top‑left corner
        label_y = b.y - 10 if b.y - 10 > 0 else b.y + 2
        draw.text((b.x, label_y), b.text, fill=color, font=font)

    if save_path:
        img.save(save_path)
    if show:
        img.show()

    return boxes


if __name__ == "__main__":
    train_dataset = load_datasets(train=True, test=False, validation=False)
    random_dataset = train_dataset.sample(1)
    for _, row in random_dataset.iterrows():
        img_path = load_real_image_path(row["image_file"], train=True)
        print(f"Image path: {img_path}")
        visualize_ocr_boxes(img_path, lang="eng", conf_thr=60, color="red", show=True)
