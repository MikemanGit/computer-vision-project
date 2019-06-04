from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from skimage import io
from matplotlib import pyplot as plt


def create_cnn(im_shape):
    input_img = Input(shape=im_shape)

    # adjust params for rgb img
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
    # sigmoid or softmax activation
    # dropout layer to avoid overfitting
    decoded = Conv2D(3, (3, 3), activation='softmax', padding='same')(cnn)

    autoencoder = Model(input_img, decoded)
    # think of good loss  function for classification
    # binary crossentropy good enough ?
    autoencoder.compile(optimizer='adamax', loss='mse',
                        metrics=['acc', 'mse', 'mae'])
    return autoencoder


def train_cnn(cnn, x_train, y_train, epochs, validation_data):
    cnn.summary()
    cnn.fit(x=x_train, y=y_train, epochs=epochs, validation_data=validation_data, verbose=1)
    return cnn


def show_image(x_train, y_train, im_idx):
    if x_train.size <= im_idx <= 0:
        print('invalid index')
    else:
        io.imshow(x_train[im_idx])
        plt.show()
        print('image {} has label {}'.format(im_idx, y_train[im_idx]))


def graph_summary(cnn):
    # summarize history for accuracy
    plt.plot(cnn.history.history['acc'])
    plt.plot(cnn.history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(cnn.history.history['loss'])
    plt.plot(cnn.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
