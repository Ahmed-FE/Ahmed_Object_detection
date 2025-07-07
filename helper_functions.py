import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import haversine as hs
from haversine import Unit
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
def compute_disparity(frame):
    # download the MIdas Model
    midas = torch.hub.load('intel-isl/MiDaS', "DPT_Large")
    midas.to('cpu')
    midas.eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.dpt_transform
    #frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #########################################
    batch = transform(frame).to('cpu')
    with torch.no_grad():
        prediction = midas(batch)
        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1),
                                                           size=frame.shape[:2],
                                                           mode='bicubic',
                                                           align_corners=False).squeeze()
        output = prediction.cpu().numpy()
        output = (((output - np.min(output)) / np.max(output)) * 255 )# normalize
        return output

def detection_Frame(Frame,Model):
    result =Model.predict(Frame,classes=[8])
    for r in result :
        detections=[]
        detection_xyxy=[]
        classes=r.boxes.cls.numpy()
        conf=r.boxes.conf.numpy()
        xyxy=r.boxes.xyxy.numpy()
        for i ,item in enumerate (xyxy):
            xmin,ymin,xmax,ymax=item
            w=xmax-xmin
            h=ymax-ymin
            detections.append(([xmin,ymin,w,h],conf[i], classes[i]))
            detection_xyxy.append(([xmin, ymin, xmax, ymax], conf[i], classes[i]))
    return detections,detection_xyxy

def track(detections, frame):
    object_tracker = DeepSort(
        max_age=50,
        n_init=2,
        nms_max_overlap=0.3,
        max_cosine_distance=0.8,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    )
    tracks = object_tracker.update_tracks(detections, frame=frame)
    tracking_ids = []
    boxes = []
    for track in tracks:
        tracking_ids.append(track.track_id)
        ltrb = track.to_ltrb()
        boxes.append(ltrb)

    return tracking_ids, boxes

def find_distance_lon_lat(lat_North,lon_North,lat_RP,lon_RP):

    camera1_coords = (lat_North, lon_North)  # North Camera coordinate(latitude, longitude)
    RP_coords = (lat_RP, lon_RP)  # South Camera coordinate (latitude, longitude)
    # calibration for the relative calculated distance
    d = hs.haversine(camera1_coords, RP_coords, unit=Unit.METERS)  # base line distance
    return d


def masked_depth_map(depth_map,detections):
    height, width = depth_map.shape
    masked_image = np.zeros((height, width), dtype=np.uint8)
    for detection in detections:
        box=detection[0]
        x1, y1, x2, y2 = box # Example coordinates
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y1)
        masked_image[y1:y2, x1:x2]=depth_map[y1:y2, x1:x2]
    return masked_image


def distance_for_detected_object_in_a_frame (detection_xyxy,depth_map,frame):
    list_distances=[]
    for detection in detection_xyxy:
        xmin, ymin, xmax, ymax = detection[0]
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        center_x = int(((xmax - xmin) / 2) + xmin)
        center_y = int(((ymax - ymin) / 2) + ymin)

        if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
            Real_depth = depth_map[center_y, center_x]
            print(Real_depth)

            # Convert disparity to depth (avoid division by zero)
            if Real_depth > 0:
                depth = (Real_depth)
                print(f"Object at ({center_x}, {center_y}) has estimated depth: {depth:.2f} meters")
                list_distances.append(([center_x,center_y],depth))

            # Optionally, draw bounding box and display depth on image
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            if 'depth' in locals():  # Check if depth was calculated
                cv2.putText(frame, f"{depth:.2f}m", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
    #####################3
    ### You can regenerate my plot with any frame using this line of code #############################
    # plt.figure(figsize=(10, 8))
    # plt.imshow(frame_North)
    # plt.figure(figsize=(10, 8))
    # heatmap = sns.heatmap(depth_map, norm=mcolors.LogNorm(), cmap='viridis')
    # colorbar = heatmap.collections[0].colorbar
    # colorbar.set_label('Log Scale Colorbar')
    # colorbar.ax.set_ylabel('Logarithmic Scale', rotation=270, labelpad=20)

    # Display the heatmap
    # plt.title('Seaborn Heatmap with Logarithmic-Scale Colorbar')
    # plt.show()

    return list_distances

def Text_on_Frame(tracking_ids,boxes,frame):
    for tracking_id, bounding_box in zip(tracking_ids, boxes):
        cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), (int(
            bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
        cv2.putText(frame, f"boat number {str(tracking_id)}", (int(bounding_box[0]), int(
            bounding_box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)