# import the necessary package
from __future__ import print_function

import cv2
from sklearn import neighbors
import imutils
import argparse

import dlib
from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import rect_to_bb

import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np


from sklearn.manifold import TSNE
from checkinimutil import FPS
from checkinimutil import WebCamVideoStream
import math
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import re
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="att"
)

def predict_knn(X_frame, knn_clf=None, model_path=None, distance_threshold=0.75):
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
    # Bilal 
    #face_locations = face_recognition.face_locations(X_frame, number_of_times_to_upsample=2)
    #faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=face_locations)
    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print(are_matches)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def prediction(face_aligned,svc,threshold=0.1):
    face_encodings=np.zeros((1,128))
    try:
        x_face_locations=face_recognition.face_locations(face_aligned)
        print(x_face_locations)
        faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
        #print("dhkkshk",faces_encodings)
        if(len(faces_encodings)==0):
            return ([-1],[0])

    except:
        print("error")

        return ([-1],[0])

    prob=svc.predict_proba(faces_encodings)
    result=np.where(prob[0]==np.amax(prob[0]))
    print(prob[0][result[0]])
    if(prob[0][result[0]]<=threshold):
        return ([-1],prob[0][result[0]])

    return (result[0],prob[0][result[0]])

def predict(X_frame, knn_clf=None,encoder_file = None,model_svc=None,model_knn_path=None,  distance_threshold=0.75):
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
    if knn_clf is None and model_knn_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    """if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)"""
    with open(model_svc, 'rb') as f:
            svc = pickle.load(f)
    encoder=LabelEncoder()
    encoder.classes_ = np.load(encoder_file)

    X_face_locations = face_recognition.face_locations(X_frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame,0)
    name_predict = []
    for face in faces:
        print("INFO : inside for loop")
        print(face)
        (x,y,w,h) = face_utils.rect_to_bb(face)

        #face_aligned = fa.align(frame,gray_frame,face)
        try:
            
            faceOrig = imutils.resize(frame[y:y + h, x:x + w], width=512)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            #plt.imshow(faceOrig)
            #plt.show()

            print("hello")
            (pred,prob)=prediction(faceOrig,svc)
            print(pred)
        except:
            print("error 2")
            pred = [-1]
        if(pred!=[-1]):
            print(np.ravel([pred]))

            person_name=encoder.inverse_transform(np.ravel([pred]))[0]

            pred=person_name
            name_predict.append(person_name)
            print(name_predict)
        else:
                print("erroe")
                person_name="unknown"
                name_predict.append(person_name)
                ttt = predict_knn(X_frame, model_path=model_knn_path)
                
                return ttt

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []
    # Bilal 
    #face_locations = face_recognition.face_locations(X_frame, number_of_times_to_upsample=2)
    #faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=face_locations)
    # Find encodings for faces in the test image
    #faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
    # Use the KNN model to find the best matches for the test face
    #closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    #print(closest_distances)
    #are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    #print(are_matches)
    # Predict classes and remove classifications that aren't within the threshold
    return [(name, loc)  for name, loc in zip(name_predict, X_face_locations)]


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
        top *= 1
        right *= 1
        bottom *= 1
        left *= 1
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        #To check the type for example it was bytes for : print(type(name))
        print(name)
        name = name.decode('utf-8')
        if name != 'unknown':
            """
            #Getting Empid by decoding to utf-8 and splitting it below
            name = name.decode('utf-8')
            empid=re.sub(r'^.*?_EMPID_', ' ', name)
            print(empid)
        
            #Getting the First Name
            sep = '_'
            firstname = name.split(sep, 1)[0]
            print(firstname)

            #Getting the Last Name
            sep = '_'
            lastname = name.split(sep, 2)[1]
            print(lastname)

            #Getting the Full Name
            fullname = firstname+' '+lastname
            print(fullname)
            #END splitting it below
            #if name != 'unknown':
            """
            #The following checks to make sure that there are no multiple entries in the DB.
            try:
               mycursorfetchname = mydb.cursor()
            except: 
                  mydb = mysql.connector.connect(
                                                host="localhost",
                                                user="root",
                                                password="",
                                                database="att"
                                                )
                  mycursorfetchname = mydb.cursor()

            mycursorfetchname.execute("SELECT name FROM `tmp_attd1` WHERE mach='126' ORDER BY id DESC LIMIT 1")
            dataname = mycursorfetchname.fetchall()
            for row in dataname:
                #print("Data Name is = ", row[0], )
                fetchedname = row[0]
                #print(fetchedname)

            if name != fetchedname :
            
                try:

                    mycursor = mydb.cursor()
                except:
                      mydb = mysql.connector.connect(
                                                    host="localhost",
                                                    user="root",
                                                    password="",
                                                    database="att"
                                                    )
                      mycursor = mydb.cursor()
                sql = "INSERT INTO tmp_attd1 (name, bio, mach, location_id, compute_id,loc_code) VALUES (%s, %s, %s, %s, %s, %s)"
                val = (name, 'Clock-Out', '126', '3', '13','SHED_KHI' )
                mycursor.execute(sql, val)
                mydb.commit()
                print(mycursor.rowcount, "record inserted.")

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=999900,
            help="# of frames to loop over FPS test")
    ap.add_argument("-d", "--display", type=int, default=1,
            help="whether or not frames should be displayed")
    args = vars(ap.parse_args())
    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
    #svc_save_path="svc_model.sav"




   
    # grab a pointer to the video stream and initialize the FPS counter
    print ("[INFO] sampling frames from webcam")
    #stream = WebCamVideoStream(src="rtsp://admin:Admin12345@172.16.0.102/Channels/1").start()
    stream = WebCamVideoStream(src=0).start()
    fps = FPS().start()

    # loop over some frames
    while 44 < args["num_frames"]:
        # grab the frame from the stream and resize it to have a maximum 
        # width of 400 pixels
        frame = stream.read()
        frame = imutils.resize(frame, width=640)
        #frame = frame[50:720, 180:500]


        # check to see if the frame should be displayed on screen
        if args["display"] > 0:
            predictions1 = predict(frame, model_svc="svc_model4.sav", encoder_file='classes4.npy', model_knn_path="model_final88.clf")
            frame = show_prediction_labels_on_image(frame, predictions1)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xff
        if ord('q') == cv2.waitKey(10):
            stream.release()
            cv2.destroyAllWindows()
            exit(0)

        # update the fps counter
        fps.update()

    # stop the timer and display the information
    fps.stop()
    print ("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
    print ("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    stream.stop()
    cv2.destroyAllWindows()
exec(open('both.model_checkout.py').read())
