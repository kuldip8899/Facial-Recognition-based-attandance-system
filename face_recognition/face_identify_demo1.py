#impot nexessary library
#import keras
from keras.engine import Model
#for model
from keras import models
from keras import layers
from keras.layers import Input
#for preprocessing
from keras.preprocessing import image
#feature extraction layers with VGGFace
from keras_vggface.vggface import VGGFace
#alias for the namespace will be created
import numpy as np
from keras_vggface import utils
#for distance calculation
import scipy.spatial
import scipy as sp
import cv2
import os
#pattern matching
import glob
#saving object by serializaion
import pickle

#define function
#load file (pickle)
def load_stuff(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff

#Singleton class for real time face identification
class FaceIdentify(object):
#use haarcascade_frontalface_alt xml file
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
#define function for check precompute_features
    def __new__(cls, precompute_features_file=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceIdentify, cls).__new__(cls)
        return cls.instance
#define function for load stuff file
    def __init__(self, precompute_features_file=None):
        self.face_size = 224
        self.precompute_features_map = load_stuff(precompute_features_file)
#load vggface model
        print("Loading VGG Face model...")
# pooling: None, avg or max
        self.model = VGGFace(model='resnet50',
                             include_top=False,
                             input_shape=(224, 224, 3),
                             pooling='avg')  
        print("Loading VGG Face model done")
#for label(required set all parameter) 
    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
#set textsize
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
#for rectangle
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
#for text
		cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

#define crop_face function 
    def crop_face(self, imgarray, section, margin=20, size=224):
       
#imgarray: full image
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
#face detected area (x, y, w, h)
        (x, y, w, h) = section
#add some margin to the face detected area to include a full head
        margin = int(min(w, h) * margin / 100)
#set values
		x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
#check margin values
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
#the result image resolution with be (size x size)
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
#return resized image in numpy array with shape (size x size x 3)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

#define face identify function using threeshold values
    def identify_face(self, features, threshold=100):
        distances = []
#match featuers with precompute_features_map file
        for person in self.precompute_features_map:
            person_features = person.get("features")
#use euclidean distance
            distance = sp.spatial.distance.euclidean(person_features, features)
            distances.append(distance)
#get min distance value
        min_distance_value = min(distances)
#get min distance index
        min_distance_index = distances.index(min_distance_value)
#check min_distance_value and threshold values
        if min_distance_value < threshold:
#print person name
            print(self.precompute_features_map[min_distance_index].get("name"))
            return self.precompute_features_map[min_distance_index].get("name")
        else:
#print unknown
            return "Unknown"
#define function detect face
    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

# 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
# infinite loop, break by key ESC
        while True:
            if not video_capture.isOpened():
                sleep(5)
#Capture frame-by-frame
            ret, frame = video_capture.read()
#change colorspaces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64)
            )
# placeholder for cropped faces
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=10, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                face_imgs[i, :, :, :] = face_img
            if len(face_imgs) > 0:
# generate features for each face
                features_faces = self.model.predict(face_imgs)
                predicted_names = [self.identify_face(features_face) for features_face in features_faces]
# draw results
            for i, face in enumerate(faces):
                label = "{}".format(predicted_names[i])
                self.draw_label(frame, (face[0], face[1]), label)

            cv2.imshow('TraComo Faces', frame)
# ESC key press			
            if cv2.waitKey(5) == 27:  
                break
# When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


#define main function
def main():
    face = FaceIdentify(precompute_features_file="./data/precompute_features.pickle")
    face.detect_face()

if __name__ == "__main__":
    main()
