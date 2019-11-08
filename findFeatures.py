import argparse as ap
import os

import cv2
import joblib
import numpy as np
from scipy.cluster.vq import *
from sklearn import preprocessing

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required=True)
args = parser.parse_args()

# Get the training classes names and store them in a list
train_path = args.trainingSet  # train_path = "dataset/train/"
training_names = os.listdir(train_path)
numWords = 1000

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths.append(image_path)

# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

for i, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    print(f"Extract SIFT of {training_names[i]} image, {i} of {len(image_paths)} images")
    kpts, des = sift.detectAndCompute(im, None)  # des: all candidate keypoints in the image
    # rootsift
    # rs = RootSIFT()
    # des = rs.compute(kpts, des)
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
# downsampling = 1
# descriptors = des_list[0][1][::downsampling,:]
# for image_path, descriptor in des_list[1:]:
#     descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]  # descriptors: all candidate keypoints
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
print(f"Start k-means: {numWords} words, {descriptors.shape[0]} key points")
voc, variance = kmeans(descriptors, numWords, 1)  # voc: all 1000 keypoints (1000, 182)

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), numWords), "float32")  # each row is the hist of a image
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# Perform L2 normalization
im_features = im_features * idf
im_features = preprocessing.normalize(im_features, norm='l2')

ifid = [[] for _ in range(numWords)]

for i, im_feature in enumerate(im_features):
    for j, feature in enumerate(im_feature):
        if feature != 0:
            ifid[j].append((i, feature))

joblib.dump((ifid, image_paths, idf, numWords, voc), "bof.pkl", compress=3)
