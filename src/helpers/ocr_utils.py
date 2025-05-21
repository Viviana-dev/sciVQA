import sys
from os import path
from typing import Literal

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.settings import Settings

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.bbox import BBox
from helpers.dataset_utils import load_real_image_path

_surya_recognizer: RecognitionPredictor | None = None
_surya_detector: DetectionPredictor | None = None

Settings.DISABLE_TQDM = True


def _get_surya_predictors() -> tuple[RecognitionPredictor, DetectionPredictor]:
    """Return cached Surya recognition & detection predictors (GPU if available).

    Returns
    -------
    tuple[RecognitionPredictor, DetectionPredictor]
        The Surya recognition and detection predictors.
    """

    global _surya_recognizer, _surya_detector
    if _surya_recognizer is None:
        _surya_recognizer = RecognitionPredictor()
    if _surya_detector is None:
        _surya_detector = DetectionPredictor()
    return _surya_recognizer, _surya_detector


def extract_ocr_boxes(
    df: pd.DataFrame, conf_thr: int = 51, split: Literal["train", "test", "validation"] = "train"
) -> list[list[BBox]]:
    """Run Surya OCR on all images in the DataFrame and return a list of BBox for each Image.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the image paths.
    conf_thr : int, default 51
        The confidence threshold for filtering OCR results.
    split : str, default "train"
        The split of the dataset to load images from. Can be "train", "test", or "validation".

    Returns
    -------
    list[list[BBox]]
        A list of lists of BBox objects, one for each image in the DataFrame.
        Each BBox object contains the bounding box coordinates, text, and confidence score.
    """

    recognizer, detector = _get_surya_predictors()

    image_paths = [load_real_image_path(row["image_file"], **{split: True}) for _, row in df.iterrows()]
    images = [Image.open(p).convert("RGB") for p in image_paths]

    predictions = recognizer(
        images=images, det_predictor=detector, detection_batch_size=128, recognition_batch_size=1024, math_mode=False
    )

    all_boxes: list[BBox] = []

    for page in predictions:
        boxes_per_image: list[BBox] = []
        for line in page.text_lines:
            conf_f = line.confidence
            if conf_f * 100 < conf_thr:
                continue
            text = line.text
            if not text:
                continue

            x1, y1, x2, y2 = line.bbox
            boxes_per_image.append(
                BBox(
                    x=int(x1),
                    y=int(y1),
                    w=int(x2 - x1),
                    h=int(y2 - y1),
                    text=text,
                    conf=int(conf_f * 100),
                )
            )
        all_boxes.append(boxes_per_image)

    return all_boxes


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
