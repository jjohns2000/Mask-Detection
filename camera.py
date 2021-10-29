import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np


faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('train_model.h5')
class VideoCamera(object):
    def __init__(self):
       
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        
        self.video.release()
    def get_frame(self):
       
        ret, frame = self.video.read()
        
        resize = cv2.resize(frame, (300,200))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        print("no. of faces = ",len(faces))
        print("faces = ",faces)
        faces_list=[]
        preds=[]
        for i in range(len(faces)):
            x, y, w, h = faces[i]
            print("x,y,w,h",x,y,w,h)
            face_frame = frame[y:y+h,x:x+w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame =  preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list)>0:
                preds = model.predict(faces_list[-1])
            for pred in preds:
                (mask, withoutMask) = pred
        
            if mask> withoutMask:
                label = "Mask"
            else:
                label = "No Mask"
        
            if label == "Mask":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)  
        
        
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x*2, y- 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
 
            cv2.rectangle(frame, (x*2, y), (x +int(4.5* w), y + int(3.2*h)),color, 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
