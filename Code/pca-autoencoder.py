import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model


def construct_observation_matrix(x_train, x_val):
    """ build training or validation set

     :param training set
     :return: observation matrix for pca
     """

    for img in len(x_train):
        test_im = x_train[img]
        x_train[img] = cv2.cvtColor(x_train[img], cv2.COLOR_BGR2GRAY)

        print('process image {}'.format(img))

    return x_train


def calculate_pc(x_train_observation_matrix, principal_components):
    # centering data is not necessary using svd
    # svd calculates eigenvector

    # mean_val = np.mean(x_train_observation_matrix, axis=1)
    u, s, v = np.linalg.svd(x_train_observation_matrix)
    x_reduced = np.dot(u[:, :principal_components], np.diag(s[:principal_components]))
    return x_reduced

#
# def autoencoder():
# # fully concected input and output layer
