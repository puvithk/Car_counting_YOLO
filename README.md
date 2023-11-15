# Car_counting_YOLO
# Vehicle Detection and Counting using YOLO and OpenCV

This Python script uses the YOLO (You Only Look Once) object detection model to detect vehicles in a video stream. It then counts the number of vehicles that cross a predefined line in the video.

## Dependencies

The script uses the following libraries:
- `ultralytics` for the YOLO model
- `cv2` for video processing
- `cvzone` for drawing on images
- `sort` for tracking objects
- `numpy` for numerical operations

## How it Works

1. The script first loads the YOLO model and a list of class names for object detection.
2. It initializes a SORT (Simple Online and Realtime Tracking) tracker for tracking vehicles.
3. It opens a video file and starts processing the frames one by one.
4. For each frame, it applies a mask and then runs the YOLO model to detect objects.
5. It filters the detections to only keep those that are vehicles and have a high confidence score.
6. It feeds the detections to the SORT tracker, which returns the tracked objects.
7. For each tracked object, it checks if the object's center has crossed the predefined line. If it has, it increments the vehicle count.
8. It draws the bounding boxes of the tracked objects and the vehicle count on the frame and displays it.

## Usage

To run the script, simply execute it with a Python interpreter. Make sure the paths to the YOLO weights file, the video file, and the mask image are correctly set in the script.

Please note that this script is intended for educational purposes and may not be suitable for real-world vehicle counting applications without further tuning and optimization. It's also important to note that the accuracy of the vehicle count depends on the quality of the video and the performance of the YOLO model and SORT tracker. 

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).