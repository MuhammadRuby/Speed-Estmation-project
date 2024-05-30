from typing import Dict
import cv2
import numpy as np
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect
from supervision.tools.detections import Detections
import time

class SpeedEstimator:
    def __init__(self):
        self.bottles = []
        self.person_id = None
        self.start_time = None
        self.end_time = None
        self.speed = 0.0

    def update(self, detections: Detections, keypoints: list):
        # Identify bottles and person
        for xyxy, _, class_id, tracker_id in detections:
            if class_id == 39:  # Bottle
                self.bottles.append(xyxy)
            elif class_id == 0 and self.person_id is None:  # Person
                self.person_id = tracker_id
        
        # If we have identified both bottles and the person, estimate speed
        if len(self.bottles) >= 2 and self.person_id is not None:
            self.bottles = sorted(self.bottles, key=lambda x: x[0])  # Sort bottles by x coordinate
            start_bottle = self.bottles[0]
            end_bottle = self.bottles[1]

            if self.start_time is None:
                # Check if the person crosses the start line
                if self._crosses_line(keypoints, start_bottle):
                    self.start_time = time.time()
            
            if self.start_time is not None:
                # Check if the person crosses the end line
                if self._crosses_line(keypoints, end_bottle):
                    self.end_time = time.time()
                    time_diff = self.end_time - self.start_time
                    self.speed = 10.0 / time_diff  # Distance is 10 meters

    def _crosses_line(self, keypoints: list, line: np.ndarray) -> bool:
        left_knee = keypoints[0][14]
        right_knee = keypoints[0][15]
        return line[0] <= left_knee[0] <= line[2] and line[0] <= right_knee[0] <= line[2]

def draw_speed(frame: np.ndarray, speed_estimator: SpeedEstimator) -> np.ndarray:
    if speed_estimator.speed > 0:
        speed_text = f"Speed: {speed_estimator.speed:.2f} m/s"
        cv2.putText(frame, speed_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return frame
