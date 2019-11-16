import os

import cv2
import joblib
import numpy as np
from scipy.cluster.vq import *
from sklearn import preprocessing
from tqdm import tqdm


class VocabularyTree:

    def __init__(self, k=3, depth=20):
        self.k = k
        self.depth = depth
        self.nodes = []
        self.centroids = None

    def train(self, data):
        self._train(self.nodes, data, current_depth=1)

    def _train(self, node, data, current_depth):
        words, _ = kmeans(data, self.k)
        if current_depth == self.depth:
            node.children = [VocabularyTreeNode(centroid, True, i) for i, centroid in enumerate(words)]
            self.centroids = node.children
        else:
            mask, _ = vq(data, words)
            for word_id, centroid in enumerate(words):
                child = VocabularyTreeNode(centroid, False)
                self._train(child, data[mask == word_id], current_depth+1)
                node.children.append(child)

    def create_inv_idx(self, imgs_features):
        for img_idx, img_features in enumerate(tqdm(imgs_features)):
            for feature_idx, feature_val in enumerate(img_features):
                if feature_val != 0:
                    self.centroids[feature_idx].inv_file_idx.append((img_idx, feature_val))

    def search_nearest(self, test_features):
        pass

class VocabularyTreeNode:

    def __init__(self, centroid, is_leave, word_id=None):
        self.centroid = centroid
        self.is_leave = is_leave
        if is_leave:
            self.inv_file_idx = []
            self.word_id = word_id
        else:
            self.children = []
