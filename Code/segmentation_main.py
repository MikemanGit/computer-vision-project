import os
import time

import numpy as np
from lxml import etree
from skimage import io
from skimage.transform import resize
from keras.models import save_model, load_model

from Code.cnn_autoencoder import train_cnn
from Code.segmentation_cnn import show_seg_image, visualize_training_history, create_segmentation_cnn, seg_image, \
    create_pretrained_segmentation_cnn, dice_coef, mean_iou

# enable gpu support
import plaidml.keras

plaidml.keras.install_backend()

# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog',
          'bird']  # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = "/Users/Michael/PycharmProjects/School/computer-vision-project/VOCdevkit"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 64  # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)

# step1 - build list of filtered filenames
annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
annotation_files = os.listdir(annotation_folder)
filtered_filenames = []
for a_f in annotation_files:
    tree = etree.parse(os.path.join(annotation_folder, a_f))
    if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
        filtered_filenames.append(a_f[:-4])

# step2 - build (x,y) for TRAIN/VAL (segmentation)
classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Segmentation/")
classes_files = os.listdir(classes_folder)
train_files = [os.path.join(classes_folder, 'train.txt')]
val_files = [os.path.join(classes_folder, 'val.txt')]


def build_segmentation_dataset(list_of_files):
    """ build training or validation set for segmentation task

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, image_size, image_size, 3)
    """
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    seg_image_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationClass")
    image_filenames = [os.path.join(image_folder, file) for f in lines for file in os.listdir(image_folder) if
                       f in file]
    seg_image_filenames = [os.path.join(seg_image_folder, file) for f in lines for file in os.listdir(seg_image_folder)
                           if
                           f in file]
    x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
        'float32')
    y = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in seg_image_filenames]).astype(
        'float32')
    return x, y


x_train, y_train = build_segmentation_dataset(train_files)
print('%i training images from %i classes' % (x_train.shape[0], y_train.shape[1]))
x_val, y_val = build_segmentation_dataset(val_files)
print('%i validation images from %i classes' % (x_val.shape[0], y_train.shape[1]))
show_seg_image(x_train, y_train, 100)

# seg_net = create_pretrained_segmentation_cnn(im_shape=(64, 64, 3))
# trained_seg_net = train_cnn(cnn=seg_net, filename='segnet', x_train=x_train, y_train=y_train, epochs=50,
#                            validation_data=(x_val, y_val))
# save_model(model=trained_seg_net,
#           filepath='/Users/Michael/PycharmProjects/School/computer-vision-project/Models/segnet.h5')


trained_seg_net = load_model(filepath='/Users/Michael/PycharmProjects/School/computer-vision-project/Models/segnet.h5',
                             custom_objects={'dice_coef': dice_coef, 'mean_iou': mean_iou})

training_history = visualize_training_history(
    filepath='/Users/Michael/PycharmProjects/School/computer-vision-project/Models/Training History/history_segnet.pckl',
    activation='softmax', loss='binary_crossentropy')

seg_image(trained_seg_net, x_train[0])
