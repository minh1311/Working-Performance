import cv2
from datetime import datetime, time, timedelta
import shapely.geometry as geometry
import numpy as np


class WorkZone:
    def __init__(self, zone_id, polygon):
        self.zone_id = zone_id
        self.points = polygon
        self.total_time_in = timedelta(hours=0, minutes= 0)
        self.session_time_in = timedelta(hours=0, minutes= 0)

        self.status = "out"
        self.time_set = datetime.now()
    def check_in_box(self, center_points):
        return geometry.Polygon(self.points).contains(geometry.Point(center_points))

    def run(self, track_id, xyxy, score,center_point, img):
        self.draw_per_in_box( track_id, xyxy, score,center_point, img)
        

        if self.status == "out":
            if self.check_in_box(center_point):
                self.status = "in"
                self.time_set = datetime.now()
                cv2.putText(img, (str)(track_id)+"-----", (int(xyxy[0] ), int(xyxy[1])) , 1,2,(0,0,255),2) 


        if self.status == "in":
            if self.check_in_box(center_point):
                self.session_time_in = datetime.now() - self.time_set
                cv2.putText(img, (str)(track_id)+"-----"+(str)(self.session_time_in)+"===="+(str)(self.total_time_in+self.session_time_in), (int(xyxy[0] ), int(xyxy[1])) , 1,2,(0,0,255),2) 

            else:
                self.total_time_in += self.session_time_in
                self.status = "out"
                self.session_time_in = timedelta(hours=0, minutes=0)
                cv2.putText(img, (str)(track_id)+"-----", (int(xyxy[0] ), int(xyxy[1])) , 1,2,(0,0,255),2) 


        

    def draw_per_in_box(self, track_id, xyxy, score,center_point, img):
        top_left = (int(xyxy[0] ), int(xyxy[1]))  # (x1, y1)
        bottom_right = (int(xyxy[2] ), int(xyxy[3]) )  # (x2, y2)


        cv2.rectangle(img, top_left, bottom_right, (255, 255, 0), 2)
        cv2.circle(img,  center_point, radius=1, color=(0, 0, 255), thickness=1)

    def draw_polygon(self, img):
        cv2.polylines(img, [np.array(self.points)], isClosed=True, color=(0, 255, 0), thickness=2)
 
    
def setPolygons(polygons_list):
    workzones = []
    for i in range(len(polygons_list)):
        workzones.append(WorkZone(i,polygons_list[i]))

    return workzones
