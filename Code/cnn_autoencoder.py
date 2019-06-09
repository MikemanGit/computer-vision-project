import numpy as np
import pickle

from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
from keras.utils.vis_utils import plot_model
from skimage import io
from matplotlib import pyplot as plt


def create_cnn(im_shape):
    input_img = Input(shape=im_shape)

    # adjust params for rgb img
    cnn = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling2D((2, 2), padding='same')(cnn)
    cnn = Conv2D(64, (3, 3), activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling2D((2, 2), padding='same')(cnn)
    cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    encoded = MaxPooling2D((2, 2), padding='same')(cnn)

    cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    cnn = BatchNormalization()(cnn)
    cnn = UpSampling2D((2, 2))(cnn)
    cnn = Conv2D(64, (3, 3), activation='relu', padding='same')(cnn)
    cnn = UpSampling2D((2, 2))(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu', padding='same')(cnn)
    cnn = UpSampling2D((2, 2))(cnn)
    cnn = Conv2D(3, (3, 3), padding='same')(cnn)
    decoded = BatchNormalization()(cnn)
    # sigmoid or softmax activation -> softmax
    # dropout layer to avoid overfitting
    # decoded = Conv2D(3, (3, 3), activation='softmax', padding='same')(cnn)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adamh', loss='mse',
                        metrics=['acc', 'mse', 'mae'])
    return autoencoder


def train_cnn(cnn, filename, x_train, y_train, epochs, validation_data):
    cnn.summary()

    # early stoppping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    train_hist = cnn.fit(x=x_train, y=y_train, epochs=epochs, validation_data=validation_data, verbose=1,
                         callbacks=[es])
    # save:
    f = open(
        '/Users/Michael/PycharmProjects/School/computer-vision-project/Models/Training History/history{}.pckl'.format(
            filename),
        'wb')
    pickle.dump(train_hist.history, f)
    f.close()
    return cnn


def show_image(x_train, y_train, im_idx):
    if x_train.size <= im_idx <= 0:
        print('invalid index')
    else:
        io.imshow(x_train[im_idx])
        plt.show()
        print('image {} has label {}'.format(im_idx, y_train[im_idx]))


def visualize_train_res(cnn, model_name, activation='', loss=''):
    plot_model(cnn,
               '/Users/Michael/PycharmProjects/School/computer-vision-project/Models/Figures/{}.png'.format(model_name),
               show_shapes=True, show_layer_names=True)
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
    reconstructed = cnn.predict(np.expand_dims(im, axis=0))[0]

    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.title('reconstructed image')

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
