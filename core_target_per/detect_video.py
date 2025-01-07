import cv2
import numpy as np
import sys
from detector import *
from bytetrack.tracker.byte_tracker import BYTETracker
from bytetrack.mc_bytetrack import dict_dot_notation
# Danh sách để lưu các polygon
polygons = [[(579, 478), (825, 469), (812, 730), (582, 740)], [(913, 534), (1221, 528), (1242, 756), (898, 746)], [(816, 360), (1064, 368), (1074, 499), (843, 509)]]
current_polygon = []
# polygon config video quan mt
# [[(579, 478), (825, 469), (812, 730), (582, 740)], [(913, 534), (1221, 528), (1242, 756), (898, 746)], [(816, 360), (1064, 368), (1074, 499), (843, 509)]]


tracker = BYTETracker(args=dict_dot_notation({
                                    'track_thresh': 0.3,
                                    'track_buffer': 50,
                                    'match_thresh': 0.8,
                                    'mot20': False,
                                }),
                                frame_rate=20,
                            )

def getCenterPoint(xyxy):
    return ((int)((xyxy[2]-xyxy[0])/2 + xyxy[0]), (int)((xyxy[3]-xyxy[1])/2 + xyxy[1]))


def main():
    person_detections=[]
    for box, score, class_id in zip(boxes, scores, class_ids):

                    
        if int(class_id) == 67:
            input_track = np.append(np.append(box, score),class_id)
            
            # person_detections = person_detections.tolist()
            print(type(person_detections))
            person_detections.append(input_track)
            
    
    person_detections = np.array(person_detections)


    output_stracks = []
    if person_detections is not None and len(person_detections) != 0:
        height_frame, width_frame = frame.shape[0], frame.shape[1]

        output_stracks = tracker.update(person_detections, (height_frame, width_frame), (height_frame, width_frame))


    person_track = []
    
    for strack in output_stracks:
        track_id = strack.track_id
        tlwh = strack.tlwh
        score = strack.score
        xyxy = np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
        
        # cv2.putText(frame_draw, str(track_id), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)
        # cv2.rectangle(frame_draw, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2, 8)
        
        person_track.append([track_id, xyxy, score,getCenterPoint(xyxy)])

# return boxes, scores, classes
# print(len(outputs[0][0]))

        # top_left = (int(tlwh[0] ), int(tlwh[1]))  # (x1, y1)
        # bottom_right = (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))  # (x2, y2)

        # cv2.putText(frame,"{} - {}".format(track_id, score), top_left, 2,2,(255,255,0),1,1)


#         # Vẽ hình chữ nhật
        top_left = (int(tlwh[0] ), int(tlwh[1]))  # (x1, y1)
        bottom_right = (int(tlwh[0]) + int(tlwh[2] ), int(tlwh[1])+int(tlwh[3]) )  # (x2, y2)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), 2)
        cv2.putText(frame, "-----"+(str)(track_id), top_left, 1,1,(0,0,255))
        print(getCenterPoint(xyxy))
        cv2.circle(frame, getCenterPoint(xyxy), radius=5, color=(0, 0, 255), thickness=1)

    
            
            

if __name__ == "__main__":
    
    detector = YOLOV("./assets/yolo11m.onnx", 0.3, 0.3, "gpu")
    video_path = sys.argv[1]
    
    cap = cv2.VideoCapture(video_path)
    
 
    
    while cap.isOpened():
        print(polygons)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Vẽ tất cả các polygon
        for polygon in polygons:
            cv2.polylines(frame, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

  
        # Dự đoán
        boxes, scores, class_ids = detector.predict(frame)

        main()
        # Hiển thị video
        cv2.imshow("Video", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
