# Import dependencies
from datetime import datetime,timedelta
from ultralytics import YOLO
import os
import json
date_format="%Y%m%d_%H%M%S"      # date format provided in the assignment
FrameRate = 1 # 1 Hz frame rate
# start date of camera frames
current_datetime=datetime.strptime("20250203_102557", date_format)
# end date of the camera frames
end_date=datetime.strptime("20250203_102843", date_format)
# the file path to the frames
file_path=r"C:\Users\seka_\Desktop\Task_CMMI\assignment_data\North"
# The YOLO model used in the process
model=YOLO('YOLOv10n.pt')
j=0
bounding_boxes=[]
class_label=[]
time_stamp=[]
confidence_score=[]
while current_datetime <= end_date:
    # Time stamp to be added to the Json file and used in creating detection
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
    photo_path=os.path.join(file_path,timestamp+".png")
    # perform the prediction for class boat which has ID =8
    results = model.predict(photo_path,classes=[8])  # classes can be defined here with using keyword classes and class ID for boat =8 and person = 0
    for r in results:
        img_bounding_boxes = []
        for i, box in enumerate(r.boxes):
            # Extract bounding box coordinates (xyxy format)
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            # Extract confidence score
            confidence = box.conf[0].item()
            # Extract class label
            class_id = int(box.cls[0].item())
            label = r.names[class_id]
            bounding_box=[x_min,y_min,x_max,y_max] # bounding box of the detcted object
            bounding_boxes.append(bounding_box)
            img_bounding_boxes.append(bounding_box)
            time_stamp.append(timestamp)
            class_label.append(label)
            confidence_score.append(confidence)
            print(
                f"Object: {label}, Confidence: {confidence:.2f}, Bounding Box: [{x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f}]")
    j+=1
    current_datetime += timedelta(seconds=FrameRate)
output_dictionary={'Time Stamp': time_stamp,'Class Label':class_label,'bounding boxes':bounding_boxes,'Confidence Score':confidence_score}
filename = r"C:\Users\seka_\Desktop\Task_CMMI\detections_cam_North.json"
with open(filename, 'w') as json_file:
  json.dump(output_dictionary, json_file, indent=4) # indent=4 for pretty-printing

