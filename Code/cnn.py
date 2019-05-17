from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K


def create_cnn(im_shape):
    input_img = Input(shape=im_shape)

    cnn = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    cnn = MaxPooling2D((2, 2), padding='same')(cnn)
    cnn = Conv2D(8, (3, 3), activation='relu', padding='same')(cnn)
    cnn = MaxPooling2D((2, 2), padding='same')(cnn)
    cnn = Conv2D(8, (3, 3), activation='relu', padding='same')(cnn)
    encoded = MaxPooling2D((2, 2), padding='same')(cnn)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    cnn = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    cnn = UpSampling2D((2, 2))(cnn)
    cnn = Conv2D(8, (3, 3), activation='relu', padding='same')(cnn)
    cnn = UpSampling2D((2, 2))(cnn)
    cnn = Conv2D(16, (3, 3), activation='relu')(cnn)
    cnn = UpSampling2D((2, 2))(cnn)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(cnn)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder