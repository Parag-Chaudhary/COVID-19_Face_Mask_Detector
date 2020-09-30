# Initialising Modules
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np

# Loading the model
model = load_model('Mask_Detector.model')
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Starting the camera
cap = cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while True:
    ret, frame = cap.read()
    color = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(color,1.3,5)
    
# Defining the box around the face
    for (x,y,w,h) in faces:

        face_img= color[y:y+w,x:x+h]
        resized= cv2.resize(face_img,(224,224))
        img_arr= img_to_array(resized)
        pre_process= preprocess_input(img_arr)
        reshaped= np.reshape(pre_process,(1,224,224,3))

 # Predict using model
        result = model.predict(reshaped)

        labels = np.argmax(result,axis=1)[0]
        percentage= np.round(np.max(result,axis=1)*100,2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[labels],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[labels],-1)
        cv2.putText(frame,labels_dict[labels],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(frame,str(percentage),(x+130,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

# Press q to Quit
    cv2.imshow('CAMERA', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
