import sys
from os import path

import torch
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.dataset_utils import load_datasets, load_real_image_path
from helpers.ocr_utils import visualize_ocr_boxes

# Cached DePlot objects
#
# Loading Google's DePlot model (≈1 GB) and its processor is expensive,
# both in terms of time (several seconds) and VRAM/CPU RAM.  To avoid
# paying that cost every time `deploit_table()` is called we keep one
# instance of each at module scope.  They are initialised lazily on the
# first invocation and reused for the lifetime of the Python process.
_deplot_processor: AutoProcessor | None = None
_deplot_model: Pix2StructForConditionalGeneration | None = None


def deploit_table(img: Image.Image, device: str | torch.device | None = None) -> str:
    """Run Google's DePlot model to extract a markdown table from a chart image.

    Parameters
    ----------
    img : PIL.Image.Image
        Square-padded chart image (RGB).
    device : str or torch.device, optional
        Where to place the model tensors.  If *None*, uses "cuda" when
        available, otherwise "cpu".

    Returns
    -------
    str
        Markdown-formatted table (or an empty string if DePlot cannot decode
        one).
    """

    global _deplot_processor, _deplot_model

    # Lazy‑load to avoid startup penalty when --deplot is not required
    if _deplot_processor is None or _deplot_model is None:
        _deplot_processor = AutoProcessor.from_pretrained("google/deplot")
        _deplot_model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
        _deplot_model.eval()
        if torch.cuda.is_available():
            _deplot_model.to("cuda")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    prompt = "generate underlying data table of the figure below:"
    inputs = _deplot_processor(images=img, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = _deplot_model.generate(**inputs, max_new_tokens=512)

    table_md = _deplot_processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    return table_md


if __name__ == "__main__":
    train_dataset = load_datasets(train=True, test=False, validation=False)
    random_dataset = train_dataset.sample(1)
    for _, row in random_dataset.iterrows():
        img_path = load_real_image_path(row["image_file"], train=True)
        print(f"Image path: {img_path}")
        visualize_ocr_boxes(img_path, lang="eng", conf_thr=60, color="red", show=True)
        image = Image.open(img_path).convert("RGB")
        table = deploit_table(image)
        print(f"DePlot table:\n{table}")
