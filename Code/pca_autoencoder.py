import numpy as np

from keras.layers import Dense, InputLayer, Flatten, Reshape
from keras.models import Sequential


def construct_observation_matrix(x_train):
    """ build training or validation set

     :param training set
     :return: observation matrix for pca
     """
    np.reshape(x_train, (1489, -1, 3))
    x_train_vector = np.empty((1489, 784, 3))
    for (i, img) in enumerate(x_train):
        x_train_vector[i] = np.reshape(img, (784, 3))
    print(x_train.shape)
    return x_train_vector


def calculate_pca(x_train_observation_matrix, principal_components):
    # centering data is not necessary using svd
    # svd calculates eigenvector
    # calculate for 3 channels
    # mean_val = np.mean(x_train_observation_matrix, axis=1)
    x_red = x_train_observation_matrix[:, :, 0]
    x_blue = x_train_observation_matrix[:, :, 1]
    x_green = x_train_observation_matrix[:, :, 2]
    ur, sr, vr = np.linalg.svd(x_red)
    ub, sb, vb = np.linalg.svd(x_blue)
    ug, sg, vg = np.linalg.svd(x_green)
    x_reduced_red = np.dot(ur[:, :principal_components], np.diag(sr[:principal_components]))
    x_reduced_blue = np.dot(ub[:, :principal_components], np.diag(sb[:principal_components]))
    x_reduced_green = np.dot(ug[:, :principal_components], np.diag(sg[:principal_components]))
    return x_reduced_red, x_reduced_blue, x_reduced_green


def linear_autoencoder(im_shape, code_size, x_train, x_val):
    autoencoder = Sequential()

    # Input Layer
    autoencoder.add(InputLayer(im_shape))

    # Image -> vector (easier to work with and to keep in memory)
    autoencoder.add(Flatten())

    # Code layer
    autoencoder.add(Dense(code_size))

    # Vector -> image
    autoencoder.add(Dense(np.prod(im_shape)))

    # Output layer
    autoencoder.add(Reshape(im_shape))

    autoencoder.compile(optimizer='adam', loss='mse',
                        metrics=['mse', 'mae'])  # here we can play with the optimizer and loss function
    autoencoder.summary()
    # actual training
    history = autoencoder.fit(x=x_train, y=x_train, epochs=100, validation_data=[x_val, x_val], verbose=1)

    return history
