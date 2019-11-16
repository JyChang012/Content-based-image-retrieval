import os

import cv2
import joblib
import numpy as np
from scipy.cluster.vq import *
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.cluster import KMeans


class VocabularyTree:

    def __init__(self, k=3, depth=20):
        self.k = k
        self.depth = depth
        self.root_node = VocabularyTreeNode.get_root_node()
        self.leaves = []
        self.idf = None

    @property
    def voc(self):
        if self.leaves:
            return np.vstack([leaf.centroid for leaf in self.leaves])
        else:
            raise RuntimeError("Haven't initialized yet!")

    def train(self, all_features):
        self._train(self.root_node, all_features, current_depth=1)

    def _train(self, node, features, current_depth):
        print(f'start k-means of depth {current_depth}')
        k_means = KMeans(self.k)
        # words, _ = kmeans(features, self.k)
        k_means.fit(features)
        words = k_means.cluster_centers_
        if current_depth == self.depth:
            node.children = [VocabularyTreeNode(centroid, True) for centroid in words]
            self.leaves += node.children
        else:
            mask, _ = vq(features, words)
            for word_id, centroid in enumerate(words):
                child = VocabularyTreeNode(centroid, False)
                self._train(child, features[mask == word_id], current_depth + 1)
                node.children.append(child)

    def create_inv_idx(self, imgs_features):
        for img_idx, img_features in enumerate(tqdm(imgs_features)):
            for feature_idx, feature_val in enumerate(img_features):
                if feature_val != 0:
                    self.leaves[feature_idx].inv_file_idx.append((img_idx, feature_val))

        nbr_occurences = np.sum((imgs_features > 0) * 1, axis=0)
        idf = np.array(np.log((1.0 * imgs_features.shape[0] + 1) / (1.0 * nbr_occurences + 1)), 'float32')
        self.init_tf_idf(idf)

    def search_nearest(self, test_descriptors):
        print('start retrieving image')
        candidate = dict()
        for feature in tqdm(test_descriptors):
            leaf = self._search_nearest(self.root_node, feature)
            for img_id, feature_val in leaf.inv_file_idx:
                if img_id not in candidate:
                    candidate[img_id] = 0.
                candidate[img_id] += feature_val * leaf.weight
        rank_id = sorted(candidate, key=candidate.get, reverse=True)
        return rank_id

    def _search_nearest(self, node, feature):
        if node.is_leaf:
            return node
        else:
            centroids = node.centroids_array
            mask, _ = vq(feature[np.newaxis], centroids)
            return self._search_nearest(node.children[mask[0]], feature)

    def init_tf_idf(self, idf):
        self.idf = idf
        for weight, leaf in zip(idf, self.leaves):
            leaf.weight = weight
            # leaf.word_id = word_id


class VocabularyTreeNode:

    def __init__(self, centroid, is_leaf, is_root=False):
        self.centroid = centroid
        self.is_leaf = is_leaf
        self.is_root = is_root
        # self.word_id = None
        self.weight = None
        if is_leaf:
            self.inv_file_idx = []
        else:
            self.children = []

    @property
    def centroids_array(self):
        return np.vstack([child.centroid for child in self.children])

    @classmethod
    def get_root_node(cls):
        return cls(None, False, True)
