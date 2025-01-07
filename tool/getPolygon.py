import cv2
import numpy as np
import sys
from bytetrack.tracker.byte_tracker import BYTETracker

# Danh sách để lưu các polygon
polygons = []
current_polygon = []
# polygon config video quan mt
# [[(579, 478), (825, 469), (812, 730), (582, 740)], [(913, 534), (1221, 528), (1242, 756), (898, 746)], [(816, 360), (1064, 368), (1074, 499), (843, 509)]]

# Hàm để xử lý sự kiện chuột
def draw_polygon(event, x, y, flags, param):
    global current_polygon, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))  # Thêm tọa độ vào polygon hiện tại

    elif event == cv2.EVENT_RBUTTONDOWN:  # Nhấn chuột phải để hoàn thành polygon
        if len(current_polygon) > 0:
            polygons.append(current_polygon)  # Thêm polygon hiện tại vào danh sách
            current_polygon = []  # Reset polygon hiện tại

if __name__ == "__main__":
    
  
    video_path = sys.argv[1]
    
    cap = cv2.VideoCapture(video_path)
    
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', draw_polygon)
    
    while cap.isOpened():
        print(polygons)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Vẽ tất cả các polygon
        for polygon in polygons:
            cv2.polylines(frame, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Vẽ polygon hiện tại nếu có
        if len(current_polygon) > 0:
            cv2.polylines(frame, [np.array(current_polygon)], isClosed=False, color=(255, 0, 0), thickness=2)

        # Hiển thị video
        # frame = cv2.resize(frame,(1920,1080))
        cv2.imshow("Video", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
