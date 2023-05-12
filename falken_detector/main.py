# by Richi Rod AKA @richionline / falken20
# ./falken_detector/main.py

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# The mediapipe object detector requires a trained model
# More information and trained models:
# https://developers.google.com/mediapipe/solutions/vision/object_detector/index#models
model_path = './trained_model/efficientdet_lite0.tflite'


BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def detect_image() -> None:
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=5,
        running_mode=VisionRunningMode.IMAGE
    )

    with ObjectDetector.create_from_options(options) as detector:
        pass
