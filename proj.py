#!/usr/bin/python
import cv2, os
import numpy as np
from PIL import Image
import csv
import pandas as pd
import pdb as pdb
from sklearn.cross_validation import train_test_split


# Different recognizers
recognizer_LBPH = cv2.face.createLBPHFaceRecognizer()
recognizer_Fisher = cv2.face.createFisherFaceRecognizer()
recognizer_Eigen = cv2.face.createEigenFaceRecognizer()

# haar cascades for recognizing different angles
cascade_paths = ['haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml']

face_alt = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_profile = cv2.CascadeClassifier("haarcascade_profileface.xml")

settings = {
    'minNeighbors': 2, 
    'minSize': (40,40)
}


train = pd.read_csv('driver_imgs_list.csv')
mask = np.random.choice([False, True], len(train), p=[0.75, 0.25])
train = train[mask]
#train = train[0:500]

predict_images = []

scaled_size = (100,75)
for i, image in train.iterrows():
    image_path = './train/' + image.classname + '/' + image.img
    predict_image_pil = Image.open(image_path).convert('L') #greyscale
    predict_image_pil.thumbnail(scaled_size, Image.ANTIALIAS)
    predict_image = np.array(predict_image_pil, 'uint8') #to array
    predict_images.append(predict_image)
    
images, predict_images, labels, predict_labels = train_test_split( predict_images, train['classname'], test_size=0.2, random_state=42)


# Extract relevant data
training_data = images
training_labels = labels
prediction_data = predict_images
prediction_labels = predict_labels

training_labels = map(lambda each:int(each.strip("c")), training_labels)
prediction_labels = map(lambda each:int(each.strip("c")), prediction_labels)

train_X=[]
train_Y=[]
test_X=[]
test_Y=[]

num_detect = 0
num_not_detect = 0

print 'detecting faces from training data'
for j, image in enumerate(training_data):

    #detect using different classifiers
    face = face_alt.detectMultiScale(image, **settings)
    face2 = face_default.detectMultiScale(image, **settings)
    face3 = face_profile.detectMultiScale(image, **settings)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face2) == 1:
        facefeatures = face2
    elif len(face3) == 1:
        facefeatures = face3
    else:
        facefeatures = ""
        num_not_detect = num_not_detect + 1

    for (x, y, w, h) in facefeatures:
        num_detect = num_detect + 1
        train_X.append(image)#[y: y + h, x: x + w])
        train_Y.append(training_labels[j])
        cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
        #cv2.waitKey(100)

percentage = num_detect/float((num_detect + num_not_detect))
print 'detected {} faces, out of {} ({:0.2f})'.format(num_detect, num_detect + num_not_detect, percentage)


# print 'predicting using LBPH'
# correct = 0
# incorrect = 0 
# mis=[0,0,0,0,0,0,0]
# recognizer_LBPH.train(train_X,np.array(train_Y))

# for i, image in enumerate(prediction_data):

#     pred, conf = recognizer_LBPH.predict(image)

#     if pred == prediction_labels[i]:
#         correct += 1
#     else:
#         incorrect += 1
#         #mis[prediction_labels[i]] += 1
#         cv2.imwrite("difficult\\%s_%s_%s.jpg" %(prediction_labels[i], pred, i), image) #<-- this one is new
# print 'accuracy using LBPH: {}%'.format((100*correct)/(correct + incorrect))

print 'predicting using Fisher'
correct = 0
incorrect = 0 
recognizer_Fisher.train(train_X,np.array(train_Y))

for i, image in enumerate(prediction_data):
    pred, conf = recognizer_Fisher.predict(image)

    if pred == prediction_labels[i]:
        correct += 1
    else:
        incorrect += 1
print 'accuracy using Fisher: {}%'.format((100*correct)/(correct + incorrect))


# print 'predicting using Eigen'
# correct = 0
# incorrect = 0 
# recognizer_Eigen.train(train_X,np.array(train_Y))
# for i, image in enumerate(prediction_data):

#     pred, conf = recognizer_Eigen.predict(image)

#     if pred == prediction_labels[i]:
#         correct += 1
#     else:
#         incorrect += 1
# print 'accuracy using Eigen: {}%'.format((100*correct)/(correct + incorrect))






