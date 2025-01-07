from detector import *


if __name__ == '__main__':
    model_path = './assets/yolo11m.onnx'
    # img_path = '/home/minhnh/Pictures/Examples-of-Human-Nature-scaled.jpg'
    img_path = sys.argv[1]
    img = cv2.imread(img_path)

    yolo11 = YOLOV(model_path, 0.3, 0.4, "gpu")
    yolo11.predict(img)
    cv2.imshow("Output", img)
        
    cv2.waitKey(0)
