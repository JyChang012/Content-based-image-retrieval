import argparse as ap
import os

import cv2
import joblib
import numpy as np
from scipy.cluster.vq import *
from sklearn import preprocessing
from tqdm import tqdm

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required=True)
args = parser.parse_args(
    # '-t dataset/training/'.split(' ')
)

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

for img_idx, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    print(f"Extract SIFT of {training_names[img_idx]} image, {img_idx} of {len(image_paths)} images")
    kpts, des = sift.detectAndCompute(im, None)  # des: all candidate keypoints in the image
    # rootsift
    # rs = RootSIFT()
    # des = rs.compute(kpts, des)
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]  # descriptors: all candidate keypoints
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  # memory explodes!

# Perform k-means clustering
print(f"Start k-means: {numWords} words, {descriptors.shape[0]} key points")
voc, variance = kmeans(descriptors, numWords, 1)  # voc: all 1000 keypoints (1000, 182)

# Calculate the histogram of features
imgs_features = np.zeros((len(image_paths), numWords), "float32")  # each row is the hist of a image
for img_idx in range(len(image_paths)):
    words, distance = vq(des_list[img_idx][1], voc)
    for w in words:
        imgs_features[img_idx][w] += 1

del des_list

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((imgs_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

# Perform L2 normalization
imgs_features = imgs_features * idf
imgs_features = preprocessing.normalize(imgs_features, norm='l2')

inverted_file_idx = [[] for _ in range(numWords)]

for img_idx, img_features in enumerate(tqdm(imgs_features)):
    for feature_idx, feature_val in enumerate(img_features):
        if feature_val != 0:
            inverted_file_idx[feature_idx].append((img_idx, feature_val))

joblib.dump((inverted_file_idx, image_paths, idf, numWords, voc), "bof.pkl", compress=3)
