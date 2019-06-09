import pickle

import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K

from keras import Model
from keras.applications.vgg16 import VGG16

from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from keras.layers import Input, Conv2D, MaxPooling2D, add, Conv2DTranspose, Dropout, Add


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


"""An implementation of the Intersection over Union (IoU) metric for Keras."""


# https://gist.github.com/Kautenja/69d306c587ccdf464c45d28c1545e580
def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels


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

    output = Conv2DTranspose(3, kernel_size=(16, 16), strides=(8, 8), padding='same', name='output')(fnc8_upsample2)

    seg = Model(input_img, output)

    seg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae', dice_coef, mean_iou])
    return seg


def create_pretrained_segmentation_cnn(im_shape):
    # load the vgg16 pretrained network but do not include the fully connected layers
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=im_shape)
    model_vgg16_conv.trainable = False

    fc1 = Conv2D(4096, (7, 7), activation='relu', padding='same')(model_vgg16_conv.output)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Conv2D(4096, (1, 1), activation='relu', padding='same')(fc1)
    fc2 = Dropout(0.5)(fc2)

    fnc8_upsample1 = Conv2D(1, (1, 1), activation='relu', padding='same',
                            kernel_initializer='random_normal')(fc2)
    fnc8_upsample1 = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same')(fnc8_upsample1)

    pool4 = model_vgg16_conv.get_layer('block4_pool').output
    pool4 = Conv2D(1, (1, 1), activation='relu', padding='same',
                   kernel_initializer='random_normal')(pool4)

    skip_connection1 = Add()([fnc8_upsample1, pool4])

    fnc8_upsample2 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='same')(skip_connection1)

    pool3 = model_vgg16_conv.get_layer('block3_pool').output
    pool3 = Conv2D(1, (1, 1), activation='relu', padding='same',
                   kernel_initializer='random_normal')(pool3)

    skip_connection2 = Add()([fnc8_upsample2, pool3])

    output = Conv2DTranspose(3, kernel_size=(16, 16), strides=(8, 8), padding='same', name='output')(skip_connection2)

    seg = Model(model_vgg16_conv.input, output)

    # fix loss functie + metric

    seg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae', dice_coef, mean_iou])
    return seg


# testing purpose
def show_seg_image(x_train, y_train, im_idx):
    if x_train.size <= im_idx <= 0:
        print('invalid index')
    else:
        plt.subplot(1, 2, 1)
        io.imshow(x_train[im_idx])
        plt.title('original image')
        plt.subplot(1, 2, 2)
        plt.imshow(y_train[im_idx])
        plt.title('segmented image')
        plt.show()


def seg_image(cnn, image):
    segmented = cnn.predict(np.expand_dims(image, axis=0))[0]

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title('segmented image')

    plt.show()


def visualize_training_history(filepath, activation, loss):
    # retrieve:
    f = open(filepath, 'rb')
    history = pickle.load(f)
    f.close()

    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy {}'.format(activation))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss {}'.format(loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for mse
    plt.plot(history['mean_squared_error'])
    plt.plot(history['val_mean_squared_error'])
    plt.title('mean squared error')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for mean iou
    plt.plot(history['mean_iou'])
    plt.plot(history['val_mean_iou'])
    plt.title('intersection Over Union')
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for dice score
    plt.plot(history['dice_coef'])
    plt.plot(history['val_dice_coef'])
    plt.title('Dice Score')
    plt.ylabel('dice score')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
