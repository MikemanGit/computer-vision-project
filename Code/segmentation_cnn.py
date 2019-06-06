import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

from keras import Model

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from keras.layers import Input, Conv2D, MaxPooling2D, add, Conv2DTranspose, Dropout


# just for testing purposes
def show_mask(image):
    grayscale = rgb2gray(image)
    thresh = threshold_otsu(grayscale)
    binary_im = grayscale > thresh

    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(grayscale, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(grayscale.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')

    ax[2].imshow(binary_im, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')

    plt.show()


def create_simple_binary_mask(x_train):
    seg_y_train = np.empty((x_train.shape[0], 224, 224, 1))
    for (i, image) in enumerate(x_train):
        grayscale = rgb2gray(image)
        binary_im = grayscale > threshold_otsu(grayscale)
        seg_y_train[i] = np.reshape(binary_im, (224, 224, 1))
    return seg_y_train.astype(int)


# dice coefficient used from https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def create_segmentation_cnn(im_shape):
    # create network based on VGG16
    input_img = Input(shape=im_shape)

    # adjust params for rgb img
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(
        input_img)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(conv1)
    pool1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(
        pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(conv2)
    pool2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool2')(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(
        pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(conv3)
    pool3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool3')(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(
        pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(conv4)
    pool4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool4')(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(
        pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(conv5)
    pool5 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool5')(conv5)

    # We append a 1 × 1 convolution with channel dimension 21 to predict scores for each of the PASCAL classes
    # (including background) at each of the coarse output locations, followed by a deconvolution layer to bilinearly
    # upsample the coarse outputs to pixel - dense outputs
    # just going to use a 1x1 conv with channel dimension of 2 since it is either foreground or background.

    fc1 = Conv2D(4096, (7, 7), activation='relu', padding='same')(pool5)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Conv2D(4096, (1, 1), activation='relu', padding='same')(fc1)
    fc2 = Dropout(0.5)(fc2)

    fnc8_upsample1 = Conv2D(1, (1, 1), activation='relu', padding='same',
                            kernel_initializer='random_normal')(fc2)
    fnc8_upsample1 = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same')(fnc8_upsample1)

    pool4_skip = Conv2D(1, (1, 1), activation='relu', padding='same',
                        kernel_initializer='random_normal')(pool4)

    fnc8_upsample1 = add([fnc8_upsample1, pool4_skip])

    fnc8_upsample2 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='same')(fnc8_upsample1)

    pool3_skip = Conv2D(1, (1, 1), activation='relu', padding='same',
                        kernel_initializer='random_normal')(pool3)

    fnc8_upsample2 = add([fnc8_upsample2, pool3_skip])

    output = Conv2DTranspose(1, kernel_size=(16, 16), strides=(8, 8), padding='same', name='output')(fnc8_upsample2)

    seg = Model(input_img, output)

    # fix loss functie + metric

    seg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae', dice_coef])
    return seg
