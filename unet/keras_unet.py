# source activate tensorflow_p27
# pip install scikit-image opencv-python

import os, sys, warnings, random
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf

from SegDataGenerator import SegDataGenerator, rgb2gray
from my_utils import plots, load_saved_data, LRFinder, CyclicLR

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

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

# remove sigmoid activation on last layer if using this
def pixelwise_weighted_cross_entropy_loss(y_true, y_pred):
    
    pred = tf.gather(y_pred, [0], axis=3)
    mask = tf.gather(y_true, [0], axis=3)
    weights = tf.gather(y_true, [1], axis=3)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=mask, logits=pred, weights=weights)
    return loss

def ConvBlock(inputs, num_kernels, kernel_shape=(3,3), p_dropout=0.1):
    conv = Conv2D(num_kernels, kernel_shape, activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    conv = Dropout(p_dropout) (conv)
    conv = Conv2D(num_kernels, kernel_shape, activation='elu', kernel_initializer='he_normal', padding='same') (conv)
    conv = BatchNormalization() (conv)
    return conv

def build_unet(lr, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, use_weights=False):
    print('Building U-net model')

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = ConvBlock(inputs, 16, (3,3), 0.1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = ConvBlock(p1, 32, (3,3), 0.1)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = ConvBlock(p2, 64, (3,3), 0.1)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = ConvBlock(p3, 128, (3,3), 0.1)
    p4 = MaxPooling2D((2, 2)) (c4)

    c5 = ConvBlock(p4, 256, (3,3), 0.3)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = ConvBlock(u6, 128, (3,3), 0.2)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = ConvBlock(u7, 64, (3,3), 0.2)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = ConvBlock(u8, 32, (3,3), 0.1)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = ConvBlock(u9, 16, (3,3), 0.1)

    # opt = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    opt = optimizers.Adam(lr=lr, decay=0.0) # TODO use this

    if not use_weights: # Standard U-net model
        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[mean_iou])

    else: # U-net with pixelwise weights on loss
        outputs = Conv2D(1, (1, 1), activation=None) (c9) # No activation because it's included in loss function

        # work-around for keras' output vs label dim checking - pad output with a layer of garbage
        padding_layer = Conv2D(1, (1, 1))(inputs)
        outputs_padded = concatenate([outputs, padding_layer], axis=3)

        model = Model(inputs=[inputs], outputs=[outputs_padded])
        # TODO figure out how to get this metric to work - keras checks input vs output dimensions
        model.compile(optimizer=opt, loss=pixelwise_weighted_cross_entropy_loss) #, metrics=[mean_iou])


    # model.summary()
    return model



def build_data_generators(data_path, batch_size, use_weights=False):

    # trainGenerator = SegDataGenerator(validation_split=0.2, width_shift_range=0.02,
    #                                    height_shift_range=0.02, zoom_range=0.1,
    #                                    horizontal_flip=True, vertical_flip=True,
    #                                    featurewise_center=False, featurewise_std_normalization=False,
    #                                    elastic_transform=True, rotation_right=True)
    trainGenerator = SegDataGenerator(validation_split=0.2,
                                      horizontal_flip=True, vertical_flip=True,
                                      elastic_transform=True, rotation_right=True)
    # trainGenerator.fit(X_train) # if normalization is on - TODO try this

    color_mode = 'rgb' if RGB else 'grayscale'
    train_data = trainGenerator.flow_from_directory(data_path, subset='training', batch_size=batch_size,
                                                   class_mode='segmentation', color_mode=color_mode,
                                                   use_weights=use_weights, use_contour=False, label_bw=True)
    val_data = trainGenerator.flow_from_directory(data_path, subset='validation', batch_size=batch_size,
                                                  class_mode='segmentation', color_mode=color_mode, 
                                                  use_weights=use_weights, use_contour=False, label_bw=True)

    return train_data, val_data


if __name__ == "__main__":

    # Should image be larger? What's the range of image sizes in dataset (see exploration kernels)
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    RGB = True
    IMG_CHANNELS = 3 if RGB else 1

    EPOCHS = 30
    BATCH_SIZE = 16
    STEPS_PER_EPOCH = 250
    VALIDATION_STEPS = 10

    LEARNING_RATE = 1e-4
    USE_WEIGHTS = True

    data_path = '../data/dataset_256x256.npz'
    save_path = 'models/'
    model_name = 'unet_rgb_weights'

    print(model_name)
    print('RGB: {}, USE_WEIGHTS: {}, lr: {}'.format(RGB, USE_WEIGHTS, LEARNING_RATE))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = build_unet(LEARNING_RATE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, USE_WEIGHTS)
    # model = model.load_weights('models/unet_baseline_12.hdf5') # TODO try loading

    train_data, val_data = build_data_generators(data_path, BATCH_SIZE, use_weights=USE_WEIGHTS)

    # lr_finder = LRFinder(model)
    # lr_finder.find_generator(train_data, start_lr=1e-6, end_lr=1, num_batches=300, epochs=1)
    # lr_finder.plot_loss(n_skip_beginning=0, n_skip_end=0)
    # plt.savefig('lr_finder_loss.png')
    # lr_finder.plot_loss_change(sma=20, n_skip_beginning=0, n_skip_end=0, y_lim=(-0.01, 0.01))
    # plt.savefig('lr_finder_loss_change.png')
    # import pdb; pdb.set_trace()

    checkpoint = ModelCheckpoint(save_path+model_name+'_{epoch:02d}.hdf5', monitor='val_loss',
                                 mode='min', period=1, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    cyclic_lr = CyclicLR(base_lr=1e-4, max_lr=1e-3, step_size=2*STEPS_PER_EPOCH,
                         mode='triangular')
    tensorboard = TensorBoard(log_dir='/tmp/unet')
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

    callbacks = [checkpoint, earlystopper]
    print('Callbacks: ', callbacks)

    print('Start training...')
    model.fit_generator(train_data, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=val_data,
                        validation_steps=VALIDATION_STEPS, shuffle=True)

    # See viz_model.ipynb for visualizations and run-length encoding

    ### Find optimal learning rate

    # print('Running learning rate range finder')
    # X_train, Y_train, C_train, W_train, X_test = load_saved_data(data_path, image_size=(IMG_HEIGHT, IMG_WIDTH))

    # hist = model.history.history
    # plt.plot(hist['val_loss'])

