from PIL import Image,ImageDraw
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
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import LabelEncoder
from face_recognition.face_recognition_cli import image_files_in_folder

X = []
y = []
count = 0
def train(class_dir):
    global count
    #global X
    #global y
    xx = []
    yy =[]
    class_dir2 = class_dir.replace("\\", '/')
    print("sdhbsd", class_dir2)                            
    # Loop through each training image for the current person
    for img_path in image_files_in_folder(class_dir2):
        print(img_path)
        image = face_recognition.load_image_file(img_path)
        face_bounding_boxes = face_recognition.face_locations(image)
        #print(len(face_bounding_boxes))
        print("ddfsdsfsf", len(face_bounding_boxes))
        verbose = True
        if len(face_bounding_boxes) != 1:
            print("dasd", len(face_bounding_boxes))
            # If there are no people (or too many people) in a training image, skip the image.
            if verbose:
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
        else:
            # Add face encoding for current image to the training set
            #print("dasd", len(face_bounding_boxes))
            #X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            xx.append((face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]).tolist())
            yy.append( class_dir2)
            #y.append(class_dir)
            #print(class_dir)
            #print(xx)
            count += 1
            #print("hhhhhhhh0", count)
    return xx, yy
def test(f):
    global X, y
    for i in f:
        for j in zip(i.result()[0], i.result()[1]):
            X.append(j[0])
            #print(j[1])
            y.append(j[1].split('/')[-1])
    #print("aa", f)
    #print(y)
    #print(X)
    targets=np.array(y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y=encoder.transform(y)
 
    X1=np.array(X)
    print("shape: "+ str(X1.shape))
    np.save('classes4.npy', encoder.classes_)
    svc = SVC(kernel='linear',probability=True)
    svc.fit(X1,y)
    svc_save_path="svc_model4.sav"
    with open(svc_save_path, 'wb') as f:
       pickle.dump(svc,f)
   
    print(y)
    print(type(X1))
    print("check for training")
def main(path):
    fff = []
    
    print("========gh=================")
    #Names_Num(path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        print(path)
        
        for sub_path in glob.glob(path+"/*"):
                print("gggggg--------", sub_path)
                tt = executor.submit(train, sub_path)
                fff.append(tt)
        #fff = concurrent.futures.wait(fff, 2)
        #print("ghjhk", fff)
    test(fff)
    #t = [i.result() for i in fff]
    #print(t)
    #print(tt.result())
path ='push/'
if __name__ =='__main__':

    main(path)
