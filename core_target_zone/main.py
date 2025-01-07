
import sys
from detector import *
from bytetrack.tracker.byte_tracker import BYTETracker
from bytetrack.mc_bytetrack import dict_dot_notation
from WorkZone import *
from config import *

def getCenterPoint(xyxy):
    return ((int)((xyxy[2]-xyxy[0])/2 + xyxy[0]), (int)((xyxy[3]-xyxy[1])/2 + xyxy[1]))


if __name__ == "__main__":
    detector = YOLOV(model_path_detect, conf_threshold, iou_threshold, device)

    tracker = BYTETracker(args=dict_dot_notation({
                                        'track_thresh': track_thresh,
                                        'track_buffer': track_buffer,
                                        'match_thresh': match_thresh,
                                        'mot20': False,
                                    }),
                                    frame_rate=frame_rate,
                                )
    video_path = sys.argv[1]
    
    cap = cv2.VideoCapture(video_path)

    output_file = 'demo_WorkPerformance.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    height, width, _ = 1080,1920,0
    video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))
    
    polygons = setPolygons(polygons_list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, class_ids = detector.predict(frame)
        person_detections = []

        for box, score, class_id in zip(boxes, scores, class_ids):

                        
            if int(class_id) == 0:
                input_track = np.append(np.append(box, score),class_id)
                
                # person_detections = person_detections.tolist()
                person_detections.append(input_track)

                # if score > track_thresh:
                #     score = 0.85


                # top_left = (int(box[0] - box[2]/2 ), int(box[1] - box[3]/2 ))  # (x1, y1)
                # bottom_right = (int(box[0] + box[2]/2 ), int(box[1] + box[3]/2 ) )  # (x2, y2)

                
                
        
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
            # top_left = (int(xyxy[0] ), int(xyxy[1] ))  # (x1, y1)
            # bottom_right = (int(xyxy[2]), int(xyxy[3]) )  # (x2, y2)

            # cv2.putText(frame, (str)(track_id)+"---"+(str)(f"{score:.2f}"), top_left , 1,2,(0,0,255),2) 
            # cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), 2)

            center_point = getCenterPoint(xyxy)
            person_track.append([track_id, xyxy, score,center_point])
            for polygon in polygons:
                polygon.run(track_id, xyxy, score,center_point, frame)



        # for plg in polygons:
        #     plg.draw_polygon(frame)

        
        
        frame = cv2.resize(frame,(1920,1080))
        cv2.imshow("Video", frame)
        video_writer.write(frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
