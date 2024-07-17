# import the necessary package
from __future__ import print_function

import cv2
from sklearn import neighbors
import imutils
import argparse
from cam1imutil import FPS
from cam1imutil import WebCamVideoStream
import math
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.4):
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
        print(name)

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=800,
            help="# of frames to loop over FPS test")
    ap.add_argument("-d", "--display", type=int, default=1,
            help="whether or not frames should be displayed")
    args = vars(ap.parse_args())

    # grab a pointer to the video stream and initialize the FPS counter
    print ("[INFO] sampling frames from webcam")
    #stream = WebCamVideoStream(src="rtsp://admin:pass1234@172.16.0.112/Channels/102").start()
    stream = WebCamVideoStream(src="0").start()
    fps = FPS().start()

    # loop over some frames
    while fps._numFrames < args["num_frames"]:
        # grab the frame from the stream and resize it to have a maximum 
        # width of 400 pixels
        frame = stream.read()
        frame = imutils.resize(frame, width=400)

        # check to see if the frame should be displayed on screen
        if args["display"] > 0:
            predictions1 = predict(frame, model_path="model_final.clf")
            frame = show_prediction_labels_on_image(frame, predictions1)
            #cv2.imshow("Frame", frame)
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



