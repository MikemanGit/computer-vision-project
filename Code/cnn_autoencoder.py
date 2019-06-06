from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model
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
    cnn = Conv2D(16, (3, 3), activation='relu', padding='same')(cnn)
    cnn = UpSampling2D((2, 2))(cnn)
    # sigmoid or softmax activation
    # dropout layer to avoid overfitting
    decoded = Conv2D(3, (3, 3), activation='softmax', padding='same')(cnn)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adamax', loss='mse',
                        metrics=['acc', 'mse', 'mae'])
    return autoencoder


def train_cnn(cnn, x_train, y_train, epochs, validation_data):
    cnn.summary()

    #early stoppping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    cnn.fit(x=x_train, y=y_train, epochs=epochs, validation_data=validation_data, verbose=1, callbacks=[es])
    return cnn


def show_image(x_train, y_train, im_idx):
    if x_train.size <= im_idx <= 0:
        print('invalid index')
    else:
        io.imshow(x_train[im_idx])
        plt.show()
        print('image {} has label {}'.format(im_idx, y_train[im_idx]))


def visualize_train_res(cnn, model_name, activation='', loss=''):
    plot_model(cnn, '{}.png'.format(model_name), show_shapes=True, show_layer_names=True)
    # summarize history for accuracy
    plt.plot(cnn.history.history['acc'])
    plt.plot(cnn.history.history['val_acc'])
    plt.title('model accuracy {}'.format(activation))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(cnn.history.history['loss'])
    plt.plot(cnn.history.history['val_loss'])
    plt.title('model loss {}'.format(loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def show_reconstruction_image(cnn, im):
    # show original and reconstructed image of the autoencoder
    reconstructed = cnn.predict(im)

    plt.subplot(1, 2, 1)
    plt.plot(im)
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.plot(reconstructed)
    plt.title('reconstructed image')

    plt.show()
