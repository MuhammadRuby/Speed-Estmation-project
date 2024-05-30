from typing import Dict, List

import cv2
import numpy as np

from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.tools.detections import Detections

class SpeedEstimator:
    def __init__(self):
        self.positions = []
        self.speed = 0

    def update(self, player_position: Point, bottle_positions: List[np.ndarray]) -> bool:
        if len(bottle_positions) == 2:
            self.positions.append(player_position)
            if len(self.positions) > 2:
                return True
        return False

class SpeedAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: Color = Color.white(),
        text_thickness: float = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_padding: int = 10,
    ):
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: float = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, speed_estimator: SpeedEstimator) -> np.ndarray:
        speed_text = f"Speed: {speed_estimator.speed:.2f} m/s"

        (text_width, text_height), _ = cv2.getTextSize(
            speed_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        text_x = int(frame.shape[1] - text_width - 10)
        text_y = int(text_height + 10)

        cv2.rectangle(
            frame,
            (text_x - self.text_padding, text_y - text_height - self.text_padding),
            (text_x + text_width + self.text_padding, text_y + self.text_padding),
            self.color.as_bgr(),
            -1,
        )

        cv2.putText(
            frame,
            speed_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        return frame
