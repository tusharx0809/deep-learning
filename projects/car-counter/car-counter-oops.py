from ultralytics import YOLO
import cv2
import math
from sort import *

class VehicleDetector:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]
        
        self.font_settings = {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'scale': 0.75,
            'thickness': 2,
            'color': (255, 0, 255)
        }
        
        resolutions_mask_limits ={
            '1280x720':  ['D:/college/Python/tensorflow-basics/object -detection/mask.png',[400,297,673,297]],
            '1920x1080': ['D:/college/Python/tensorflow-basics/object -detection/mask2.png',[10,900,2000,900]]
        }
        if not self.cap.isOpened():
            print('No Video')
        else:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = str(w)+'x'+str(h)
            
        if resolution in resolutions_mask_limits:
            self.mask = cv2.imread(resolutions_mask_limits[resolution][0])
            self.limits = resolutions_mask_limits[resolution][1]
            
        self.total_count = []
        
    def apply_mask(self, frame):
        if self.mask is not None:
            return cv2.bitwise_and(frame, self.mask)
        return frame

    def process_detections(self, results, frame):
        detections = np.empty((0, 5))
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = self.class_names[cls]
                
                if current_class in ['car','motorbike','bus','truck'] and conf > 0.3:
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))
        
        return detections, frame

    def track_vehicles(self, detections, frame):
        results_tracker = self.tracker.update(detections)
        cv2.line(frame,(self.limits[0],self.limits[1]),(self.limits[2],self.limits[3]), (0,0,255),5)
        
        for result in results_tracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(frame, 
                       f'Vehicle {int(id)}',    
                       (x1, y1-10),
                       self.font_settings['font'],
                       self.font_settings['scale'],
                       self.font_settings['color'],
                       self.font_settings['thickness'])
            cx, cy = (x1+x2)//2, (y1+y2)//2 #centre point
            cv2.circle(frame, (cx, cy),5,(255,0,255),cv2.FILLED)
            
            if self.limits[0] < cx < self.limits[2] and self.limits[1]-15 < cy < self.limits[1]+15:
                if self.total_count.count(id) == 0:
                    self.total_count.append(id)
                    cv2.line(frame,(self.limits[0],self.limits[1]),(self.limits[2],self.limits[3]), (0,255,0),5)
            cv2.putText(frame,f'Count {len(self.total_count)}',(10,30), 
                        self.font_settings['font'], 
                        self.font_settings['scale'], 
                        self.font_settings['color'], 
                        self.font_settings['thickness'])
        return frame

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break
                
            # Apply mask to the frame
            masked_frame = self.apply_mask(frame)
            
            # Get model predictions
            results = self.model(masked_frame, stream=True)
            
            # Process detections
            detections, frame = self.process_detections(results, frame)
            
            # Track vehicles
            frame = self.track_vehicles(detections, frame)
            
            # Display results
            cv2.imshow("Image", frame)
            #cv2.imshow("ImageRegion", masked_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    # Initialize detector with paths
    detector = VehicleDetector(
        model_path='D:/college/Python/tensorflow-basics/object -detection/yolo-weights/yolov10x.pt',
        video_path='D:/college/Python/tensorflow-basics/object -detection/videos/traffic.mp4',
    )
    
    # Run the detection
    detector.run()


if __name__ == "__main__":
    main()
    
    
