import argparse as ap

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import *
from sklearn import preprocessing
from tqdm import tqdm
import os

# import numpy as np

MIN_MATCH_COUNT = 10
Min_INLIERS_COUNT = 10

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required=True)
args = parser.parse_args(
    # '-i dataset/training/radcliffe_camera_000397.jpg'.split(' ')
)

# Get query image path
image_path = args.image

# Load the classifier, class names, scaler, number of clusters and vocabulary 
voc_tree, image_paths, idf, numWords, voc = joblib.load("bof_tree.pkl")

# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
query_kpts, query_des = sift.detectAndCompute(im, None)

des_list.append((image_path, query_des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

# voc_tree.init_tf_idf(idf)
rank_ID = voc_tree.search_nearest(descriptors)

# Visualize the results
plt.figure(
    tight_layout=True
)
plt.gray()
plt.subplot(5, 4, 1)
plt.imshow(im[:, :, ::-1])
plt.title('query image')
plt.axis('off')

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

for i, ID in enumerate(rank_ID[:16]):
    # img = Image.open(image_paths[ID])
    candidate_img = cv2.imread(image_paths[ID])
    kpts, des = sift.detectAndCompute(candidate_img, None)

    matches = flann.knnMatch(query_des, des, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    plt.gray()
    plt.subplot(5, 4, i + 5)

    plt.imshow(cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([query_kpts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.)
        # if not np.sum(mask) / mask.shape[0] > .4:r
        if not np.sum(mask) > Min_INLIERS_COUNT:
            plt.title('verification failed')
    else:
        plt.title('insufficient matching points', fontdict=dict(size=6))

plt.savefig(f'result_{os.path.split(args.image)[-1]}.svg')
plt.show()
