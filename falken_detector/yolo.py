# by Richi Rod AKA @richionline / falken20
# ./falken_detector/yolo.py

# YOLO is based on a deep learning architecture that uses a single neural network for detecting
# objects within an image. The YOLO algorithm breaks an image down into smaller regions.

from imageai.Detection import VideoObjectDetection


# STEP 1: Creat an Object of the VideoObjectDetection Class
vid_obj_detect = VideoObjectDetection()


# STEP 2: Set and Load the YOLO Model
#  You need to call the setModelTypeAsYOLOv3() method since you’ll be using the YOLO algorithm
# for detecting objects from videos
# https://imageai.readthedocs.io/en/latest/index.html
vid_obj_detect.setModelTypeAsTinyYOLOv3()

# The Yolo model that the imageai library uses for object detection is available at the following:
# https://bit.ly/2UqlRGD
# We'll use yolo.h5 model
# vid_obj_detect.setModelPath(r"./falken_detector/datasets/yolo.h5")
vid_obj_detect.setModelPath("https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5")
vid_obj_detect.loadModel()


# STEP 3: Detect Objects from Videos
input_file_path = r"./falken_detector/resources/input_video.mp4"
output_file_path = r"./falken_detector/resources/"

detected_vid_obj = vid_obj_detect.detectObjectsFromVideo(
    input_file_path=input_file_path,
    output_file_path=output_file_path,
    frames_per_seconds=15,  # Specifies the number of frames per second for the output video
    log_progress=True,
    return_detected_frame=True
)
