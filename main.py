from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from utils import SpeedEstimator, draw_speed
from tqdm.notebook import tqdm
import numpy as np
import cv2
from ultralytics import YOLO

MODEL_POSE = "yolov8x-pose.pt"
model_pose = YOLO(MODEL_POSE)

MODEL = "yolov8x.pt"
model = YOLO(MODEL)

# dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# get class id for ball and bottle
CLASS_ID_BALL = [32]
CLASS_ID_BOTTLE = [39]

SOURCE_VIDEO_PATH = "dataset/running.mp4"
TARGET_VIDEO_PATH = "dataset/running_result.mp4"

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
    # loop over video frames
    for frame in tqdm(generator, total=video_info.total_frames):
        results_poses = model_pose.track(frame, persist=True)
        annotated_frame = results_poses[0].plot()
        keypoints = results_poses[0].keypoints.xy.int().cpu().tolist()
        bboxes = results_poses[0].boxes.xyxy.cpu().numpy()

        results_objects = model.track(frame, persist=True, conf=0.1)
        tracker_ids = results_objects[0].boxes.id.int().cpu().numpy() if results_objects[0].boxes.id is not None else None
        detections = Detections(
            xyxy=results_objects[0].boxes.xyxy.cpu().numpy(),
            confidence=results_objects[0].boxes.conf.cpu().numpy(),
            class_id=results_objects[0].boxes.cls.cpu().numpy().astype(int),
            tracker_id=tracker_ids
        )

        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID_BOTTLE + CLASS_ID_BALL for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # update speed estimator
        speed_estimator.update(detections=detections, keypoints=keypoints)

        # annotate and display frame
        labels = [
            f"id:{track_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, track_id
            in detections
        ]
        annotated_frame = box_annotator.annotate(frame=annotated_frame, detections=detections, labels=labels)
        
        # draw speed analysis
        annotated_frame = draw_speed(frame=annotated_frame, speed_estimator=speed_estimator)

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        sink.write_frame(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
cv2.destroyAllWindows()
