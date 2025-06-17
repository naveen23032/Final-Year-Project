Final year Project Program
import face_recognition as fr 
import numpy as np 
import cv2 
import os 
from time import sleep
from PIL import Image
import RPi.GPIO as GPIO
buzzer=14
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer,GPIO.OUT)

# Get the absolute path to the 'Known_Faces' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
faces_path = os.path.join(current_dir, "Known_Faces")

def get_face_encodings():
    face_names = os.listdir(faces_path)
    face_encodings = []

    for i, name in enumerate(face_names):
        face = fr.load_image_file(os.path.join(faces_path, name))
        face_encodings.append(fr.face_encodings(face)[0])

        face_names[i] = name.split(".")[0] 
    
    return face_encodings, face_names

face_encodings, face_names = get_face_encodings()

video = cv2.VideoCapture(0)  # Adjust the path to your video file

scl = 2

while True:
#     video = cv2.VideoCapture("/home/pi/Desktop/project file/Known_Faces/Jack Ma.jpg")
    success, frame = video.read()

    if not success:
#         print("Error reading frame")
        break

    resized_frame = cv2.resize(frame, (int(frame.shape[1] / scl), int(frame.shape[0] / scl)))

    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(rgb_frame)
    unknown_encodings = fr.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(unknown_encodings, face_locations):
        result = fr.compare_faces(face_encodings, face_encoding, 0.4)

        if True in result:
            name = face_names[result.index(True)]
            print(name)
            GPIO.output(buzzer,GPIO.LOW)
            top, right, bottom, left = face_location

            cv2.rectangle(frame, (left * scl, top * scl), (right * scl, bottom * scl), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left * scl, bottom * scl + 20), font, 0.8, (255, 255, 255), 1)
        else:
            name = "Unknown"
            print(name)
            GPIO.output(buzzer,GPIO.HIGH)
            sleep(2)
            GPIO.output(buzzer,GPIO.LOW)
            
            top, right, bottom, left = face_location
            
            cv2.rectangle(frame, (left * scl, top * scl), (right * scl, bottom * scl), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left * scl, bottom * scl + 20), font, 0.8, (255, 255, 255), 1)
        
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite("image.jpg", frame)
        break
video.release()
cv2.destroyAllWindows()
