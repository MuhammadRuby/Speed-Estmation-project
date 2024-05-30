from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from utils import SpeedEstimator, SpeedAnnotator
from tqdm.notebook import tqdm
import numpy as np
import cv2
from ultralytics import YOLO
import time

MODEL_POSE = "yolov8x-pose.pt"
model_pose = YOLO(MODEL_POSE)

MODEL = "yolov8x.pt"
model = YOLO(MODEL)

# dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# get class id ball and bottle
CLASS_ID = [32, 39]

SOURCE_VIDEO_PATH = f"dataset/running.mp4"
TARGET_VIDEO_PATH = f"dataset/running_result.mp4"

# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.5)

# create SpeedEstimator instance
speed_estimator = SpeedEstimator()

# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    start_time = None
    end_time = None
    distance_covered = 10  # meters (5 meters to the mark and 5 meters back)
    
    for frame in tqdm(generator, total=video_info.total_frames):
        results_poses = model_pose.track(frame, persist=True)
        annotated_frame = results_poses[0].plot()
        
        results_objects = model.track(frame, persist=True, conf=0.1)
        tracker_ids = results_objects[0].boxes.id.int().cpu().numpy() if results_objects[0].boxes.id is not None else None
        detections = Detections(
            xyxy=results_objects[0].boxes.xyxy.cpu().numpy(),
            confidence=results_objects[0].boxes.conf.cpu().numpy(),
            class_id=results_objects[0].boxes.cls.cpu().numpy().astype(int),
            tracker_id=tracker_ids
        )

        # filter detections for ball and bottle
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # Get keypoints for the person
        keypoints = results_poses[0].keypoints.xy.int().cpu().tolist()
        if len(keypoints) > 0:
            player_position = keypoints[0][0]  # Assuming keypoint[0][0] is the player's center point
        
        # Detect bottles and calculate speed
        bottles_detected = [detections.xyxy[i] for i, class_id in enumerate(detections.class_id) if CLASS_NAMES_DICT[class_id] == "bottle"]
        if len(bottles_detected) == 2:
            bottle_positions = [bottle for bottle in bottles_detected]
            if start_time is None:
                start_time = time.time()
            
            if speed_estimator.update(player_position, bottle_positions):
                end_time = time.time()
                time_taken = end_time - start_time
                if time_taken > 0:
                    speed = distance_covered / time_taken  # meters per second
                    speed_estimator.speed = speed
                    start_time = None  # Reset start_time for the next run

        # annotate and display frame
        labels = [
            f"id:{track_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for i, (xyxy, confidence, class_id, track_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id))
        ]
        annotated_frame = box_annotator.annotate(frame=annotated_frame, detections=detections, labels=labels)
        speed_annotator = SpeedAnnotator()
        speed_annotator.annotate(annotated_frame, speed_estimator)

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        sink.write_frame(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
cv2.destroyAllWindows()
