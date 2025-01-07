from datetime import datetime, time, timedelta
import cv2
import numpy as np

class Person:
    def __init__(self,track_id: int, zone_id: int,xyxy, score: float, center_box:tuple):
        self.track_id = track_id
        self.zone_id  = zone_id
        self.xyxy = xyxy
        self.score = score
        self.center_box_list = [center_box]

        self.total_time_in = timedelta(hours=0, minutes= 0)
        self.session_time_in = timedelta(hours=0, minutes= 0)
        self.total_time_out = timedelta(hours=0, minutes= 0)
        self.session_time_out = timedelta(hours=0, minutes= 0)

        self.status = "out"
        self.time_set = datetime.now()
       

    def update(self, xyxy, score, center_box):
        self.xyxy = xyxy
        self.score = score
        self.center_box_list.append(center_box)

    def getTrackId(self):
        return self.track_id

    def run(self, polygons):


        for polygon in polygons:
            if polygon.getZoneId() == self.zone_id:
                if self.status == "out":
                    if polygon.check_in_box(self.center_box_list[-1]) :
                        self.status = "in"
                        self.time_set = datetime.now()
                        
                if self.status == "in":
                    if polygon.check_in_box(self.center_box_list[-1]) :
                        self.session_time_in = datetime.now() - self.time_set

                    else:
                        self.total_time_in += self.session_time_in
                        self.status = "out"
                        self.session_time_in = timedelta(hours=0, minutes= 0)

    def draw(self, img):
        # print(self.xyxy, len(self.xyxy), type(self.xyxy))
        top_left = (int(self.xyxy[0] ), int(self.xyxy[1]))  # (x1, y1)
        bottom_right = (int(self.xyxy[2] ), int(self.xyxy[3]) )  # (x2, y2)


        cv2.rectangle(img, top_left, bottom_right, (255, 255, 0), 2)
        cv2.putText(img, (str)(self.track_id)+"-----"+(str)(self.session_time_in)+"===="+(str)(self.total_time_in+self.session_time_in), top_left, 1,2,(0,0,255),2) 
        for center_box in self.center_box_list:
            cv2.circle(img,  center_box, radius=1, color=(0, 0, 255), thickness=1)
