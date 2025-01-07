from core_target_zone.config import * 

import sys
from detector import *
from bytetrack.tracker.byte_tracker import BYTETracker
from bytetrack.mc_bytetrack import dict_dot_notation
from Polygon import *
from Person import Person




persons = []



def getCenterPoint(xyxy):
    return ((int)((xyxy[2]-xyxy[0])/2 + xyxy[0]), (int)((xyxy[3]-xyxy[1])/2 + xyxy[1]))

# def main():
    
            

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
    
    polygons = setPolygons(polygons_list)
    

    output_file = 'demo_WorkPerformance.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    height, width, _ = 1080,1920,0
    video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height))
        
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        print("====================================================================================")
        # Vẽ tất cả các polygon
        

        start_time = datetime.now()
        # Dự đoán
        boxes, scores, class_ids = detector.predict(frame)
        current_time = datetime.now()
        # main()
        person_detections=[]
        for box, score, class_id in zip(boxes, scores, class_ids):

                        
            if int(class_id) == 0:
                input_track = np.append(np.append(box, score),class_id)
                
                # person_detections = person_detections.tolist()
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
            person_track.append([track_id, xyxy, score,getCenterPoint(xyxy)])

        for track_id, xyxy, score, center_box in person_track:
            create_per = False
            for per in persons:
                if track_id == per.getTrackId():

                    per.update(xyxy, score, center_box)
                    per.run(polygons)
                    per.draw(frame)
                    create_per = True
                    break
            if create_per == False:
                new_per = Person(track_id, 0, xyxy, score, center_box)
                persons.append(new_per)
                new_per.draw(img=frame)

        print("model", current_time-start_time, "--- full pipeline:", datetime.now()-start_time)

        for plg in polygons:
            plg.draw_polygon(frame)
        # Hiển thị video

        frame = cv2.resize(frame,(1920,1080))
        cv2.imshow("Video", frame)
        video_writer.write(frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
