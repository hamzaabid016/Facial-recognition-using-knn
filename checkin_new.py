import cv2
import imutils
from checkinimutil import FPS
from checkinimutil import WebCamVideoStream
import numpy as np
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import mysql.connector


class FaceRecognizer:
    def __init__(self, model_path=None, distance_threshold=0.43):
        self.knn_clf = None
        self.model_path = model_path
        self.distance_threshold = distance_threshold

        self.mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="attendance_data"
        )

        if self.model_path:
            with open(self.model_path, 'rb') as f:
                self.knn_clf = pickle.load(f)

    def predict(self, X_frame):
        X_face_locations = face_recognition.face_locations(X_frame)

        if len(X_face_locations) == 0:
            return []

        faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= self.distance_threshold for i in range(len(X_face_locations))]

        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    def show_prediction_labels_on_image(self, frame, predictions):
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)

        for name, (top, right, bottom, left) in predictions:
            top *= 1
            right *= 1
            bottom *= 1
            left *= 1

            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            name = name.encode("UTF-8")
            name = name.decode('utf-8')

            if name != 'unknown':
                mycursorfetchname = self.mydb.cursor()
                mycursorfetchname.execute("SELECT name FROM `tmp_attd12` WHERE mach='107' ORDER BY id DESC LIMIT 1")
                dataname = mycursorfetchname.fetchall()
                for row in dataname:
                    fetchedname = row[0]

                if name != fetchedname:
                    mycursor = self.mydb.cursor()
                    sql = "INSERT INTO tmp_attd12 (name, bio, mach, location_id, compute_id,loc_code) VALUES (%s, %s, %s, %s, %s, %s)"
                    val = (name, 'Clock-In', '107', '1', '4', 'Office')
                    mycursor.execute(sql, val)
                    self.mydb.commit()
                    print(mycursor.rowcount, "record inserted.")

        del draw
        opencvimage = np.array(pil_image)
        return opencvimage


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-frames", type=int, default=99999999900, help="# of frames to loop over FPS test")
    ap.add_argument("-d", "--display", type=int, default=1, help="whether or not frames should be displayed")
    args = vars(ap.parse_args())

    print("[INFO] sampling frames from webcam")
    stream = WebCamVideoStream(src=0).start()
    fps = FPS().start()

    recognizer = FaceRecognizer(model_path="final.clf")

    while fps._numFrames < args["num_frames"]:
        frame = stream.read()
        frame = imutils.resize(frame, width=640)

        if args["display"] > 0:
            predictions1 = recognizer.predict(frame)
            frame = recognizer.show_prediction_labels_on_image(frame, predictions1)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                stream.release()
                cv2.destroyAllWindows()
                exit(0)

        fps.update()

    fps.stop()
    print("[INFO] elapsed time : {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    stream.stop()
    cv2.destroyAllWindows()
