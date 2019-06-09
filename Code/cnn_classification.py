import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.models import Model, Sequential


def create_classifier(cnn):
    # create new model
    classifier = Sequential()

    # get the pretrained layers
    trained_layers = [layers for layers in cnn.layers[:12]]

    # freeze and add the layers to the model
    for layer in trained_layers:
        layer.trainable = False
        classifier.add(layer)
    # initialize with or without random weights
    classifier.add(Flatten())
    classifier.add(Dense(5, activation='softmax'))  # softmax -> multiclass, multilabel
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae'])
    return classifier


def create_random_classifier(im_shape):
    # create network with random initializing layers
    input_img = Input(shape=im_shape)
    cnn = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(input_img)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling2D((2, 2), padding='same')(cnn)
    cnn = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling2D((2, 2), padding='same')(cnn)
    cnn = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(cnn)
    cnn = BatchNormalization()(cnn)
    encoded = MaxPooling2D((2, 2), padding='same')(cnn)

    classifier = Flatten()(encoded)
    classifier = Dense(5, activation='softmax', kernel_initializer='random_normal')(
        classifier)  # softmax -> multiclass, multilabel
    classifier = Model(input_img, classifier)

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae'])
    return classifier


def predict_class(cnn, im, label):
    prediction = np.squeeze(cnn.predict(np.expand_dims(im, axis=0)))
    print('the image belongs to class {} and is predicted as class {}'.format(label, np.argmax(prediction)))
