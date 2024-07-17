#!/usr/bin/python
"""

$ pip3 install scikit-learn
$ pip3 install numpy
$ pip3 install opencv-contrib-python

"""

import cv2
import math
import re
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import threading
from checkinimutil import FPS
from checkinimutil import WebCamVideoStream
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="att"
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_frame: frame to do the prediction on.
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.

    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        print(name)
        #This prints as Bilal_Nayyar_EMPID_GCS001
        #splitting it below
        #empid=re.sub(r'^.*?_EMPID_', ' ', name)
        #print(empid)

        #sep = '_'
        #firstname = name.split(sep, 1)[0]
        #print(firstname)

        #sep = '_'
        #lastname = name.split(sep, 2)[1]
        #print(lastname)

        #fullname = firstname+' '+lastname
        #print(fullname)
        #END splitting it below
        #if name != 'unknown':
        mycursor = mydb.cursor()
        sql = "INSERT INTO tmp_attd1 (name, bio, mach) VALUES (%s, %s, %s)"
        val = (name, 'Clock-In', '107')
        mycursor.execute(sql, val)
        mydb.commit()
        print(mycursor.rowcount, "record inserted.")

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage

#DRK start
# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()

# Start the camera
#camera = cv2.VideoCapture('rtsp://admin:Admin12345@172.16.0.107/Channels/101')
camera = cv2.VideoCapture(0)
#camera = WebCamVideoStream(src=0).start()
# Start the cleaning thread
cam_cleaner = CameraBufferCleanerThread(camera)

# Use the frame whenever you want
while True:
    frame = cam_cleaner.last_frame
    if cam_cleaner.last_frame is not None:
        #cv2.imshow('The last frame', cam_cleaner.last_frame)
        predictions1 = predict(cam_cleaner.last_frame, model_path="model_final.clf")
        frame = show_prediction_labels_on_image(frame, predictions1)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('camera', frame)
    if ord('q') == cv2.waitKey(10):
        camera.release()
        cv2.destroyAllWindows()
        exit(0)
#DRK End

####
####

