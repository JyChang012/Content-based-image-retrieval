import argparse as ap

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.vq import *
from sklearn import preprocessing

# import numpy as np

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required=True)
args = parser.parse_args()

# Get query image path
image_path = args.image

# Load the classifier, class names, scaler, number of clusters and vocabulary 
inverted_file_idx, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
kpts, des = sift.detectAndCompute(im, None)

# rootsift
# rs = RootSIFT()
# des = rs.compute(kpts, des)

des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

test_features = np.zeros(numWords, "float32")
words, distance = vq(descriptors, voc)
for w in words:
    test_features[w] += 1

# Perform Tf-Idf vectorization and L2 normalization
test_features = test_features * idf
test_features = preprocessing.normalize(test_features.reshape(1, -1), norm='l2').flatten()

# recover histograms of candidate images
candidates_bow = dict()
for i, feature in enumerate(test_features):
    if feature != 0:
        for candidate, val in inverted_file_idx[i]:
            if candidate not in candidates_bow:
                candidates_bow[candidate] = np.zeros(numWords)
            candidates_bow[candidate][i] += val

# sort according to similarity, return indices of images
rank_ID = sorted(candidates_bow, key=lambda idx: candidates_bow[idx] @ test_features, reverse=True)

# Visualize the results
plt.figure()
plt.gray()
plt.subplot(5, 4, 1)
plt.imshow(im[:, :, ::-1])
plt.title('query image')
plt.axis('off')

for i, ID in enumerate(rank_ID[:16]):
    img = Image.open(image_paths[ID])
    plt.gray()
    plt.subplot(5, 4, i + 5)
    plt.imshow(img)
    plt.axis('off')

plt.savefig('result.svg')
plt.show()
