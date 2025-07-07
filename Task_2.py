import math
import cv2
import os
from ultralytics import YOLO
from datetime import datetime,timedelta
from helper_functions import compute_disparity,detection_Frame,track,masked_depth_map,distance_for_detected_object_in_a_frame,find_distance_lon_lat
import json
date_format="%Y%m%d_%H%M%S"      # date format provided in the assignment
FrameRate = 1 # 1 Hz frame rate
# start date of camera frames
current_datetime=datetime.strptime("20250203_102557", date_format)
# end date of the camera frames
end_date=datetime.strptime("20250203_102843", date_format)
"20250203_102843"
# the file path to the frames
file_path=r"C:\Users\seka_\Desktop\Task_CMMI\assignment_data\North"
### calculate the baseline between the two cameras using the longitude and latitude of both cameras
#lat_North = 34.9778490
#lat_RP = 34.9778320
#lon_North = 33.9478417
#lon_RP = 33.9477566
#d_known=find_distance_lon_lat(lat_North,lon_North,lat_RP,lon_RP)
d_known=7.980595
# the camera is 3.52 m from sea level
h=3.52
horizontal_distance=d_known  # the horizontal distance between the camera and the boat at this frame
d=math.sqrt(h**2+horizontal_distance**2)

d_scaled=0.00411601  # the relative distance of the center of the object in the image
Scaling_factor=d/d_scaled  # the scaling factor to correct the distance to real world coordinate
## the Yolo model to perform prediction
model = YOLO('yolov10n.pt')  # using yolov10m
time_stamp=[]
depth_output_from_midas=[]
depth_map_pixel_wise=[]
masked_depth_maps=[]
j=1
while current_datetime <= end_date:

    # Time stamp to be added to the Json file and used in creating detection
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
    print(' frame number ' + timestamp )
    photo_path=os.path.join(file_path,timestamp+".png")
    frame_North=cv2.imread(photo_path)
    ### calculate the depth map using Midas deepl learning model
    # https://pytorch.org/hub/intelisl_midas_v2/
    depth_map = compute_disparity(frame_North)

    #depth_output_from_midas.append(depth_map.tolist())

    detection_xywh_North, detection_xyxy_North = detection_Frame(frame_North, model)  # detect the objects in the frame
    ### use deep sort to track the objects in the image
    tracking_ids_north, boxes_north = track(detection_xywh_North, frame_North)

    # run this line to save your depth map to .npy file
    #np.save(r"C:\Users\seka_\Desktop\Task_CMMI\depth_map.npy", depth_map)
    depth_map[depth_map == 0] = .00001
    depth_map = 1 / depth_map
    time_stamp.append(timestamp)
    # for calibration of the distance
    # I used the image 20250203_102811 and calculated the actual depth from
    # longitude and lattitude of the boat and camera
    # distance scaled was found from the depth distance to be .006279468

    depth_map = Scaling_factor * depth_map

    depth_map[depth_map > 200] = 200
    depth_map_pixel_wise.append(depth_map.tolist())
    ################# run all of that to create
    masked_map=masked_depth_map(depth_map,detection_xyxy_North)
    ### run this line if you want to compute a distance for an object in a frame ##########################3
    #list_distances=distance_for_detected_object_in_a_frame(detection_xyxy_North,depth_map,frame_North)
    #masked_depth_maps.append(masked_map.tolist())
    current_datetime += timedelta(seconds=FrameRate)
    j += 1
    if j==9:
        break

output_dictionary={'Time Stamp': time_stamp,'depth_frame ' : depth_map_pixel_wise}
filename = r"C:\Users\seka_\Desktop\Task_CMMI\depth_mask_North.json"
with open(filename, 'w') as json_file:
  json.dump(output_dictionary, json_file, indent=4) # indent=4 for pretty-printing
