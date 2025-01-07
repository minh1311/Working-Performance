from datetime import datetime, time, timedelta
import cv2
import shapely.geometry as geometry
import numpy as np

class Polygon:
    def __init__(self, zone_id, polygon):
        self.zone_id = zone_id
        self.points = polygon

        

    def getZoneId(self):
        return self.zone_id

    def check_in_box(self, center_points):
        return geometry.Polygon(self.points).contains(geometry.Point(center_points))
    
    def draw_polygon(self, img):
        cv2.polylines(img, [np.array(self.points)], isClosed=True, color=(0, 255, 0), thickness=2)

def setPolygons(polygons_list):
    polygons = []
    for i in range(len(polygons_list)):
        polygons.append(Polygon(i,polygons_list[i]))

    return polygons
