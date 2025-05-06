import math

from PIL import Image, ImageOps

MAX_PIXELS = 2000000  # 2Mpx -> downscale images larger than this


def resize_and_pad(
    img: Image.Image, max_pixels: int = MAX_PIXELS, bg_color: str = "white"
) -> tuple[Image.Image, int, int]:
    """Resize the image so that width x height ≤ *max_pixels* (keeping aspect
    ratio) and then pad each side up to the next multiple of 28 px.

    The result is **not forced to be square**; both dimensions are simply
    rounded to a clean ViT patch grid.

    Returns
    -------
    PIL.Image.Image
        The resized-and-padded image.
    int, int
        Padding added to the **left** and **top** edges, respectively.
    """
    orig_w, orig_h = img.size

    # --- Optional down‑scaling ------------------------------------
    if orig_w * orig_h > max_pixels:
        scale = math.sqrt(max_pixels / (orig_w * orig_h))
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)

    # --- Pad to multiples of 28 -----------------------------------
    pad_w = (-img.width) % 28
    pad_h = (-img.height) % 28
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    padded = ImageOps.expand(
        img,
        border=(pad_left, pad_top, pad_w - pad_left, pad_h - pad_top),
        fill=bg_color,
    )
    assert padded.width % 28 == 0 and padded.height % 28 == 0
    return padded, pad_left, pad_top
