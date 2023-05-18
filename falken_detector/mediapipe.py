# by Richi Rod AKA @richionline / falken20
# ./falken_detector/mediapipe.py

# MediaPipe Solutions provides a suite of libraries and tools for you to quickly apply artificial
# intelligence (AI) and machine learning (ML) techniques in your applications.

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


# The mediapipe object detector requires a trained model
# More information and trained models:
# https://developers.google.com/mediapipe/solutions/vision/object_detector/index#models
MODEL_PATH = './trained_model/efficientdet_lite0.tflite'
IMAGE_FILE = "https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg"

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


def detect_image() -> None:

    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        max_results=5,
        running_mode=vision.RunningMode.IMAGE,
        score_threshold=0.5
    )

    detector = vision.ObjectDetector.create_from_options(options)

    # Load the input image
    image = cv2.imread(IMAGE_FILE)
    cv2.imshow(image)

    # Detect objects in the input image
    detection_result = detector.detect(image)

    # Process the detection result, in this case, visualize it
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow(rgb_annotated_image)


if __name__ == '__main__':
    detect_image()
