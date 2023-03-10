#import necessary library
import cv2
import os
import numpy as np
import glob
import pickle
#for face feature extraction (PRETRAINED)
from keras_vggface.vggface import VGGFace
#for preprocessing
from keras.preprocessing import image
import utils
#train extremely deep neural networks
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf



#define function
def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


#Singleton class to extraction face images from video files
class FaceExtractor(object):

#xml file (haarcascade)
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"

#define function using default values for extract
    def __new__(cls, weight_file=None, face_size=224):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceExtractor, cls).__new__(cls)
        return cls.instance

    def __init__(self, face_size=224):
        self.face_size = face_size
#cropping face in video file.
    def crop_face(self, imgarray, section, margin=20, size=224):
      
# imgarray: full image
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
			
#section: face detected area (x, y, w, h)
        (x, y, w, h) = section
#add some margin to the face detected area to include a full head
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
#check condition
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
#cropped img array
        cropped = imgarray[y_a: y_b, x_a: x_b]
	
#the result image resolution with be (size x size)
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

# resized image in numpy array with shape
        resized_img = np.array(resized_img)
		
#return resized image
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

#define function for face extract
    def extract_faces(self, video_file, save_folder):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        cap = cv2.VideoCapture(video_file)
		#for length
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		#for width
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		#for height
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		#for fps(frame per second)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        print("length: {}, w x h: {} x {}, fps: {}".format(length, width, height, fps))
# infinite loop, break by key ESC
        frame_counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
			#loop
            if ret:
                frame_counter = frame_counter + 1
				#changing colorspaces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				#use different parameter(scale factor,minneighbors,minsize)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(64, 64)
                )
                # only keep the biggest face as the main subject
                face = None
				# Get the largest face as main face
                if len(faces) > 1:  
				# area = w * h
                    face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3])) 
                elif len(faces) == 1:
                    face = faces[0]
                if face is not None:
                    face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                    (x, y, w, h) = cropped
				#draw a rectangle on any image
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                #display an image in a window
					cv2.imshow('Faces', frame)
				#png type
                    imgfile = os.path.basename(video_file).replace(".","_") +"_"+ str(frame_counter) + ".png"
                    imgfile = os.path.join(save_folder, imgfile)
                #save an image to any storage device
					cv2.imwrite(imgfile, face_img)
            # ESC key press
			if cv2.waitKey(5) == 27:  
                break
            # If the number of captured frames is equal to the total number of frames,
            # we stop
			if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()

#main function.
def main():
	#use vggaface and model resnet50 
	# pooling: None, avg or max
    resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                pooling='avg') 
    #define function for image2x
	#define path
	def image2x(image_path):
	#load image
        img = image.load_img(image_path, target_size=(224, 224))
	#image array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    #x = utils.preprocess_input(x, version=1)  # or version=2
        return x
	#define mean function
    def cal_mean_feature(image_folder):
        face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))
	#Yield successive n-sized chunks from l.
        def chunks(l, n):
            
            for i in range(0, len(l), n):
                yield l[i:i + n]
		#batch_size
        batch_size = 32
        face_images_chunks = chunks(face_images, batch_size)
        fvecs = None
        for face_images_chunk in face_images_chunks:
            images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
            #prediction
			batch_fvecs = resnet50_features.predict(images)
            if fvecs is None:
                fvecs = batch_fvecs
            else:
                fvecs = np.append(fvecs, batch_fvecs, axis=0)
        #return nparray
		return np.array(fvecs).sum(axis=0) / len(fvecs)
	#face folder by name
    FACE_IMAGES_FOLDER = "./data/face_images"
    #viedeo folder
	VIDEOS_FOLDER = "./data/videos"
    extractor = FaceExtractor()
	folders = list(glob.iglob(os.path.join(VIDEOS_FOLDER, '*')))
    os.makedirs(FACE_IMAGES_FOLDER, exist_ok=True)
    names = [os.path.basename(folder) for folder in folders]
    #for name of person
	for i, folder in enumerate(folders):
        name = names[i]
        videos = list(glob.iglob(os.path.join(folder, '*.*')))
        save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
        print(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for video in videos:
            extractor.extract_faces(video, save_folder)

    precompute_features = []
    for i, folder in enumerate(folders):
        name = names[i]
        save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
        mean_features = cal_mean_feature(image_folder=save_folder)
        precompute_features.append({"name": name, "features": mean_features})
	#generate pickel file
    pickle_stuff("./data/precompute_features.pickle", precompute_features)


if __name__ == "__main__":
    main()

