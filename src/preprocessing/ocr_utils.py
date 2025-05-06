import sys
from os import path

from PIL import Image, ImageDraw, ImageFont
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.data_load import load_datasets, load_real_image_path
from preprocessing.bbox import BBox

_surya_recognizer: RecognitionPredictor | None = None
_surya_detector: DetectionPredictor | None = None


def _get_surya_predictors() -> tuple[RecognitionPredictor, DetectionPredictor]:
    """Return cached Surya recognition & detection predictors (GPU if avail)."""
    global _surya_recognizer, _surya_detector
    if _surya_recognizer is None:
        _surya_recognizer = RecognitionPredictor()
    if _surya_detector is None:
        _surya_detector = DetectionPredictor()
    return _surya_recognizer, _surya_detector


def extract_ocr_boxes(img: Image.Image, lang: str = "en", conf_thr: int = 51) -> list[BBox]:
    """Run Surya OCR and return cleaned BBoxes."""

    recognizer, detector = _get_surya_predictors()

    # Surya accepts a list of images and list of language lists
    predictions = recognizer([img], [[lang] if lang else None], detector)
    page = predictions[0]  # single‑image batch

    boxes: list[BBox] = []
    for line in page.text_lines:
        conf_f = line.confidence
        if conf_f * 100 < conf_thr:
            continue
        text = line.text

        if not text:
            continue

        # line["bbox"] is (x1, y1, x2, y2)
        x1, y1, x2, y2 = line.bbox
        boxes.append(
            BBox(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                text=text,
                conf=int(conf_f * 100),
            )
        )

    # Deduplicate identical strings that overlap heavily (keeps 1st occurrence)
    uniq = {}
    for b in boxes:
        if b.text.lower() not in uniq:
            uniq[b.text.lower()] = b
    return list(uniq.values())


def visualize_ocr_boxes(
    img: Image.Image,
    boxes: list[BBox],
    color: str = "red",
    show: bool = True,
    save_path: str | None = None,
) -> list[BBox]:
    """Run Surya on *img_path* and draw the resulting bounding boxes.

    Parameters
    ----------
    img : PIL.Image.Image
        The image to annotate.  It is converted to RGB if not already.
    boxes : list[BBox]
        The list of OCR bounding boxes to draw.  Each box must have
        attributes *x*, *y*, *w*, *h*, and *text*.
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
    img = img.convert("RGB")

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
    _get_surya_predictors()  # warm‑up once
    train_dataset = load_datasets(train=True, test=False, validation=False)
    random_dataset = train_dataset.sample(1)
    for _, row in random_dataset.iterrows():
        img_path = load_real_image_path(row["image_file"], train=True)
        img = Image.open(img_path).convert("RGB")
        boxes = extract_ocr_boxes(img, lang="en", conf_thr=60)
        print(f"Image path: {img_path}")
        visualize_ocr_boxes(img, boxes=boxes, color="red", show=True)
