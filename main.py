import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import torch
import constants
from time import sleep

# CONFIG = SoccerFieldConfiguration()

def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:

    video = cv2.VideoCapture(source_video_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {source_video_path}")
    player_detection_model = YOLO(constants.PLAYER_DETECTION_MODEL_PATH).to(device=device)
    while True:
        success, frame = video.read()
        if not success:
            print("End of video reached or failed to retrieve frame.")
            break
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        print(detections)
        frame = constants.BOX_ANNOTATOR.annotate(frame, detections)
        frame = constants.BOX_LABEL_ANNOTATOR.annotate(frame, detections)
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def main(source_video_path: str, target_video_path: str, mode: constants.Mode) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    frame_generator = run_player_detection(source_video_path, device=device)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=False)
    parser.add_argument('--mode', type=constants.Mode, default=constants.Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        mode=args.mode
    )