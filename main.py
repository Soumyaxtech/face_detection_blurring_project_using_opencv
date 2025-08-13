import cv2
import mediapipe as mp;

# read image.........................

img = cv2.imread(r"C:\Users\Soumyajit Koley\OneDrive\Documents\face_detection 1.jpg")

# detect faces.........................

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection =0,min_detection_confidence =0.5) as face_detection:
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    output = face_detection.process(img_rgb)
    
    for detection in output.detections:
        
        location_data = detection.location_data
        bbox = location_data.relative_bounding_box
        
        x1,y1,w,h = bbox.xmin, bbox.ymin, bbox.width, bbox.height