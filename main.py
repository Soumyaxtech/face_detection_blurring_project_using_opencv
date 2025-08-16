import cv2
import mediapipe as mp;

# read image.........................

image = cv2.imread(r"C:\Users\Soumyajit Koley\OneDrive\Documents\face_detection 1.jpg")

img = cv2.resize(image,(500,500))
H, W, _ = img.shape

# detect faces.........................

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection =0,min_detection_confidence =0.5) as face_detection:
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    output = face_detection.process(img_rgb)
    
    if output.detections is not None:
        for detection in output.detections:
        
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
        
            x1,y1,w,h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            
            x1 = int (x1*W)
            y1 = int (y1*H) # creating proper boundry using coordinate
            w = int (w*W)
            h = int (h*H)
            
            #img = cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),10)
            
            img[x1:x1+w, y1:y1+h, :]= cv2.blur(img[x1:x1+w, y1:y1+h, :],(50,50))
    
            cv2.imshow("frame",img)
            cv2.waitKey(0)

# blur faces....................................

