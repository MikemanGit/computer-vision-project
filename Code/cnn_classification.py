from keras.layers import Input,InputLayer, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model, Sequential


def create_classifier(im_shape, cnn):
    # create new model
    classifier = Sequential()

    classifier.add(InputLayer(im_shape))
    # get the pretrained layers
    trained_layers = [layers for layers in cnn.layers[:7]]

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
    #create network with random initializing layers
    input_img = Input(shape=im_shape)

    # adjust params for rgb img
    cnn = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(input_img)
    cnn = MaxPooling2D((2, 2), padding='same')(cnn)
    cnn = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(cnn)
    cnn = MaxPooling2D((2, 2), padding='same')(cnn)
    cnn = Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal')(cnn)
    encoded = MaxPooling2D((2, 2), padding='same')(cnn)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    # initialize with or without random weights
    classifier = Flatten()(encoded)
    classifier = Dense(5, activation='softmax', kernel_initializer='random_normal')(classifier)  # softmax -> multiclass, multilabel
    classifier = Model(input_img, classifier)

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse', 'mae'])
    return classifier
