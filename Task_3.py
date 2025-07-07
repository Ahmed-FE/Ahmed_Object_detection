import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from helper_functions import Text_on_Frame,detection_Frame,track
from datetime import datetime,timedelta
date_format="%Y%m%d_%H%M%S"      # date format provided in the assignment
FrameRate = 1 # 1 Hz frame rate
# start date of camera frames
current_datetime=datetime.strptime("20250203_102557", date_format)
# end date of the camera frames
end_date=datetime.strptime("20250203_102843", date_format)
# the file path to the frames
file_path=r"C:\Users\seka_\Desktop\Task_CMMI\assignment_data\North"
fps=4
file_name="20250203_102740.png"
frame_path = os.path.join(file_path, file_name)
frame = cv2.imread(frame_path)
height, width, layers = frame.shape
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_path = r"C:\Users\seka_\Desktop\Task_CMMI\North_Camera_Detection_video.mp4"
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
model = YOLO('yolov10n.pt')  # using yolov10m
while current_datetime <= end_date:
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
    print(' frame number ' + timestamp)
    photo_path = os.path.join(file_path, timestamp + ".png")
    frame_North = cv2.imread(photo_path)
    detection_xywh_North, detection_xyxy_North = detection_Frame(frame_North, model)  # detect the objects in the frame
    ### use deep sort to track the objects in the image
    tracking_ids_north, boxes_north = track(detection_xywh_North, frame_North)
    Text_on_Frame(tracking_ids_north, boxes_north, frame_North)
    video.write(frame_North)
    current_datetime += timedelta(seconds=FrameRate)

video.release()
cv2.destroyAllWindows()


