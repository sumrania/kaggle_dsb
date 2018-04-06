# source activate tensorflow_p27
# pip install scikit-image
# pip install opencv-python

import os, sys, warnings, random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

from SegDataGenerator import SegDataGenerator
from my_utils import plots, LRFinder

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Should image be larger? What's the range of image sizes in dataset (see exploration kernels)
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1

data_path = '../data/dataset_256x256.npz'

# Note: this assumes that dataset is already solved to npz (by data_preprocessing notebook)
def load_saved_data(data_path, image_size=(256, 256)):
    print('Reading from previously loaded data.')
    npzfile = np.load(data_path)
    return npzfile['X_train'], npzfile['Y_train'], npzfile['C_train'], npzfile['W_train'], npzfile['X_test']

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def build_unet(): 
    # Build U-Net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    # Standard U-net model
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

    return model
#   outputs = Conv2D(1, (1, 1), activation=None) (c9)

# # work-around for keras' output vs label dim checking - pad output with a layer of garbage
# # putting constant of zeros in might be better...
# hack_image = Conv2D(1, (1, 1))(inputs)
# outputs_hack = concatenate([outputs, hack_image], axis=3)

# model = Model(inputs=[inputs], outputs=[outputs_hack])

# # remove sigmoid activation on last layer if using this
# def pixelwise_weighted_cross_entropy_loss(y_true, y_pred):
    
#     pred = tf.gather(y_pred, [0], axis=3)
#     mask = tf.gather(y_true, [0], axis=3)
#     weights = tf.gather(y_true, [1], axis=3)
#     loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=mask, 
#                                    logits=pred,
#                                    weights=weights)
    # return loss

    # model.compile(optimizer='adam', loss=pixelwise_weighted_cross_entropy_loss, metrics=[mean_iou])

  # model.summary()

model_path = 'models'
batch_size = 4

# trainGenerator = SegDataGenerator(validation_split=0.2, 
#                                    horizontal_flip=False, vertical_flip=False,
#                                    featurewise_center=False, featurewise_std_normalization=False,
#                                    elastic_transform=True, rotation_right=False)
                                 
# trainGenerator.fit(X_train)

trainGenerator = SegDataGenerator(validation_split=0.2, width_shift_range=0.02,
                                   height_shift_range=0.02, zoom_range=0.1,
                                   horizontal_flip=True, vertical_flip=True,
                                   featurewise_center=False, featurewise_std_normalization=False,
                                   elastic_transform=True, rotation_right=True)


# train_data = trainGenerator.flow_from_directory(data_path, subset='training', batch_size=batch_size,
#                                                class_mode='segmentation', color_mode='rgb',
#                                                use_contour=False, label_bw=True)
# val_data = trainGenerator.flow_from_directory(data_path, subset='validation', batch_size=batch_size,
#                                               class_mode='segmentation', color_mode='rgb', 
#                                               use_contour=False, label_bw=True)


train_data = trainGenerator.flow_from_directory(data_path, subset='training', batch_size=batch_size,
                                               class_mode='segmentation', color_mode='grayscale',
                                               use_weights=False, label_bw=True)
val_data = trainGenerator.flow_from_directory(data_path, subset='validation', batch_size=batch_size,
                                              class_mode='segmentation', color_mode='grayscale', 
                                              use_weights=False, label_bw=True)

print("made data generators!")

epochs=5
steps_per_epoch=250
validation_step=10

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
checkpoint = ModelCheckpoint(model_path+'weight.{epoch:02d}.hdf5', monitor='val_loss',
                             mode='min', period=1)
earlystopper = EarlyStopping(patience=5, verbose=1)

model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, 
                    callbacks=[earlystopper, checkpoint], validation_data=val_data, 
                    validation_steps=validation_step, shuffle=True)

if __name___ == "__main__":
    pass