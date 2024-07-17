from PIL import Image, ImageDraw
import pandas as pd
import glob
import face_recognition
import cv2
import sys
from sklearn import neighbors
import os
import os.path
import pickle
import concurrent.futures
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np

class FaceRecognitionTrainer:
    def __init__(self):
        self.X = []
        self.y = []
        self.count = 0

    def train(self, class_dir):
        xx = []
        yy = []
        class_dir2 = class_dir.replace("\\", '/')
        print("sdhbsd", class_dir2)
        for img_path in image_files_in_folder(class_dir2):
            print(img_path)
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            print("ddfsdsfsf", len(face_bounding_boxes))
            verbose = True
            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                xx.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                yy.append(class_dir2)
                self.count += 1
        return xx, yy

    def test(self, f):
        for i in f:
            for j in zip(i.result()[0], i.result()[1]):
                self.X.append(j[0])
                self.y.append(j[1].split('/')[-1])

        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', weights='distance')
        knn_clf.fit(self.X, self.y)

        # Save the trained KNN classifier
        with open("final.clf", 'wb') as f:
            pickle.dump(knn_clf, f)

        print("check for training")

    def main(self, path):
        fff = []
        print("========gh=================")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            print(path)
            for sub_path in glob.glob(path + "/*"):
                print("gggggg--------", sub_path)
                tt = executor.submit(self.train, sub_path)
                fff.append(tt)

        self.test(fff)

if __name__ == '__main__':
    path = 'push/'
    fr_trainer = FaceRecognitionTrainer()
    fr_trainer.main(path)
