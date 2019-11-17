import argparse as ap

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.vq import *
from sklearn import preprocessing
from tqdm import tqdm
import os

# import numpy as np

MIN_MATCH_COUNT = 10
Min_INLIERS_COUNT = 10
P_FACTOR = .2
N_FACTOR = .2

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required=True)
parser.add_argument('-f', '--feedback', help='Whether to enable feedback', action='store_true')

args = parser.parse_args(
    '-i dataset/testing/trinity_000033.jpg -f'.split(' ')  # radcliffe_camera_000397.jpg
)

# Get query image path
image_path = args.image

# Load the classifier, class names, scaler, number of clusters and vocabulary 
inverted_file_idx, image_paths, idf, numWords, voc = joblib.load("pkl/bof.pkl")

# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
query_kpts, query_des = sift.detectAndCompute(im, None)

# rootsift
# rs = RootSIFT()
# des = rs.compute(kpts, des)

des_list.append((image_path, query_des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

test_features = np.zeros(numWords, "float32")
words, distance = vq(descriptors, voc)
for w in words:
    test_features[w] += 1

# Perform Tf-Idf vectorization and L2 normalization
test_features = test_features * idf
test_features = preprocessing.normalize(test_features.reshape(1, -1), norm='l2').flatten()


def search(query_features):
    # recover histograms of candidate images
    candidates_bow = dict()
    for i, feature in enumerate(tqdm(query_features)):
        if feature != 0:
            for candidate, val in inverted_file_idx[i]:
                if candidate not in candidates_bow:
                    candidates_bow[candidate] = np.zeros(numWords)
                candidates_bow[candidate][i] += val

    # sort according to similarity, return indices of images
    rank_ID = sorted(candidates_bow, key=lambda idx: candidates_bow[idx] @ query_features, reverse=True)

    # Visualize the results
    plt.figure(
        tight_layout=True
    )
    plt.gray()
    plt.subplot(5, 4, 1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # BGR2RGB
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
            if np.sum(mask) > Min_INLIERS_COUNT:
                plt.title(f'{i}: succeeded', fontdict=dict(size=6))
            else:
                plt.title(f'{i}: verification failed', fontdict=dict(size=6))

        else:
            plt.title(f'{i}: insufficient matching points', fontdict=dict(size=6))

    plt.savefig(f'result_{os.path.split(args.image)[-1]}.svg')
    plt.show()

    return candidates_bow, rank_ID


candidates_bow, rank_ID = search(test_features)
if args.feedback:
    while input('Do you want to continue to give feedback?\n') in ['yes', 'y']:
        positives = input('Please enter the indices of your desirable images, starting from 0 and split by space!\n')
        positives = [int(num) for num in positives.split(' ')]
        positives = map(rank_ID.__getitem__, positives)
        positives = set(positives)
        negatives = set(rank_ID) - positives

        positives = np.vstack([candidates_bow[i] for i in positives])
        negatives = np.vstack([candidates_bow[i] for i in negatives])
        test_features = test_features + P_FACTOR * np.average(positives, axis=0) - \
                        N_FACTOR * np.average(negatives, axis=0)
        test_features = preprocessing.normalize(test_features.reshape(1, -1), norm='l2').flatten()
        candidates_bow, rank_ID = search(test_features)
        print('This is the new result!')


