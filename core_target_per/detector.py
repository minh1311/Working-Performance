
import onnxruntime as ort

import numpy as np
import cv2

from utils import xywh2xyxy, multiclass_nms, nms
import sys

import cv2
coco_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


class YOLOV:
    def __init__(self, model_path, conf_threshold, iou_threshold, device = "cpu" ):
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.provider = ["CPUExecutionProvider","CUDAExecutionProvider"] if device == "gpu" else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers = self.provider)


        # self.session = ort.InferenceSession(model_path, providers = ort.get_available_providers())


        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape

        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]
        
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img.astype(np.float32, copy=False)
        
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor
    
    def postprocess(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)

        predictions = predictions[scores > self.conf_threshold, :]

        scores = scores[scores > self.conf_threshold]


        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes

        indices = nms(boxes, scores, self.iou_threshold)
        # indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    
    
    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    # def draw_detections(self, image,boxes, scores,
    #                            class_ids, draw_scores=True, mask_alpha=0.4):

    #     return draw_detections(image, boxes, scores,
    #                            class_ids, mask_alpha)

    # def get_input_details(self):
    #     model_inputs = self.session.get_inputs()
    #     self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    #     self.input_shape = model_inputs[0].shape
    #     self.input_height = self.input_shape[2]
    #     self.input_width = self.input_shape[3]

    # def get_output_details(self):
    #     model_outputs = self.session.get_outputs()
    #     self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


    def predict(self, image):
    
        input_tensor = self.preprocess(image)
        
        outputs = self.session.run(self.output_names, {self.input_names[0]:input_tensor})
        
        boxes, scores, class_ids = self.postprocess(outputs)
        
        return boxes, scores, class_ids


    
    



