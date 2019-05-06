import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model




def construct_observation_matrix(x_train,x_val):
    """ build training or validation set

     :param training set
     :return: observation matrix for pca
     """
    for observation in x_train:


def calculate_pc(x_train_observation_matrix,principal_components):
    #centering data is not necessary using svd
    #svd calculates eigenvector

    #mean_val = np.mean(x_train_observation_matrix, axis=1)
    u,s,v = np.linalg.svd(x_train_observation_matrix)
    x_reduced = np.dot(u[:,:principal_components],np.diag(s[:principal_components]))
    return x_reduced