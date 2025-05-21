from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BBox:
    """Axis-aligned bounding box produced by OCR.

    All coordinates are given in the pixel space of the *original* image unless
    noted otherwise.

    Attributes
    ----------
    x, y : int
        Top-left corner of the box in pixels.
    w, h : int
        Width and height of the box in pixels.
    text : str
        Text recognised inside the box.
    conf : int
        Tesseract confidence score for the recognised text (0-100).

    The helper methods convert the box into different coordinate systems that
    are useful during preprocessing.
    """

    x: int  # left (px)
    y: int  # top (px)
    w: int  # width (px)
    h: int  # height (px)
    text: str  # recognised string
    conf: int  # OCR confidence 0-100

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def transform(self, scale_x: float, scale_y: float, pad_x: int, pad_y: int) -> BBox:
        """Return a new :class:`BBox` expressed in the coordinate system of a
        resized *and* padded copy of the image.

        Parameters
        ----------
        scale_x, scale_y : float
            Multiplicative scaling factors applied to the original coordinates
            in the x and y directions respectively.
        pad_x, pad_y : int
            Amount of horizontal and vertical padding (in pixels) that was
            added to the left and top edges when the image was padded to form a
            square.

        Returns
        -------
        BBox
            A **copy** of this bounding box in the resized-padded coordinate
            space. The original instance remains unchanged.
        """
        return BBox(
            x=int(self.x * scale_x + pad_x),
            y=int(self.y * scale_y + pad_y),
            w=int(self.w * scale_x),
            h=int(self.h * scale_y),
            text=self.text,
            conf=self.conf,
        )

    def normalised(self, canvas_w: int, canvas_h: int) -> tuple[float, float, float, float]:
        """Return the box in relative coordinates inside a rectangular canvas.

        Parameters
        ----------
        canvas_w, canvas_h : int
            Width and height of the padded image.

        Returns
        -------
        tuple[float, float, float, float]
            Normalised (x1, y1, x2, y2) in the range 0-1.
        """
        return (
            round(self.x / canvas_w, 4),
            round(self.y / canvas_h, 4),
            round((self.x + self.w) / canvas_w, 4),
            round((self.y + self.h) / canvas_h, 4),
        )

    def to_xy(self) -> tuple[float, float, float, float, str, int]:
        """Convert a BBox to (x1, y1, x2, y2) format.

        Returns
        -------
        tuple[float, float, float, float, str, int]
            The bounding box in (x1, y1, x2, y2, text, conf) format.

        """
        return (
            self.x,
            self.y,
            self.x + self.w,
            self.y + self.h,
            self.text,
            self.conf,
        )
