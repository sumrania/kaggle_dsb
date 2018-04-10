# source activate tensorflow_p27
# pip install scikit-image opencv-python keras

import os, sys, warnings, random
import numpy as np
import matplotlib.pyplot as plt

import keras
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

# TODO try this
def jaccard_dice_coeff_loss(y_true, y_pred):
    """
    Dice loss based on Jaccard dice score coefficent.
    """
    IMG_HEIGHT = IMG_WIDTH = 256 # TODO make sure these are same as in main
    axis = np.arange(1,len([IMG_HEIGHT,IMG_WIDTH,1])+1)
    offset = 1e-5

    corr = tf.reduce_sum(y_true * y_pred, axis=axis)
    l2_pred = tf.reduce_sum(tf.square(y_pred), axis=axis)
    l2_true = tf.reduce_sum(tf.square(y_true), axis=axis)
    dice_coeff = (2. * corr + 1e-5) / (l2_true + l2_pred + 1e-5)
    loss = tf.subtract(1., tf.reduce_mean(dice_coeff))
    return loss

# remove sigmoid activation on last layer if using this
def pixelwise_weighted_cross_entropy_loss(y_true, y_pred):
    pred = tf.gather(y_pred, [0], axis=3)
    mask = tf.gather(y_true, [0], axis=3)
    weights = tf.gather(y_true, [1], axis=3)

    pred = tf.Print(pred, ["pred: ", tf.shape(pred), pred])
    mask = tf.Print(mask, ["mask: ", tf.shape(mask), mask])
    weights = tf.Print(weights, ["weights: ", tf.shape(weights), weights])

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
    # c1 = ConvBlock(inputs, 64, (3,3), 0.1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = ConvBlock(p1, 32, (3,3), 0.1)
    # c2 = ConvBlock(p1, 128, (3,3), 0.1)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = ConvBlock(p2, 64, (3,3), 0.1)
    # c3 = ConvBlock(p2, 256, (3,3), 0.1)
    # c3 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = ConvBlock(p3, 128, (3,3), 0.1)
    # c4 = ConvBlock(p3, 512, (3,3), 0.1)
    # c4 = Conv2D(512, (3,3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
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
        padding_layer = tf.zeros_like(outputs)
        outputs_padded = concatenate([outputs, padding_layer], axis=3)
        
        outputs = tf.Print(outputs, ["outputs: ", tf.shape(outputs), outputs])
        outputs_padded = tf.Print(outputs_padded, ["outputs_padded: ", tf.shape(outputs_padded), outputs_padded])

        model = Model(inputs=[inputs], outputs=[outputs_padded])
        # TODO figure out how to get this metric to work - keras checks input vs output dimensions
        model.compile(optimizer=opt, loss=pixelwise_weighted_cross_entropy_loss) #, metrics=[mean_iou])

    # model.summary()
    return model



def build_data_generators(data_path, batch_size, target_size, use_weights=False):

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
                                                   use_weights=use_weights, use_contour=False, label_bw=True,
                                                   target_size=target_size)
    val_data = trainGenerator.flow_from_directory(data_path, subset='validation', batch_size=batch_size,
                                                  class_mode='segmentation', color_mode=color_mode, 
                                                  use_weights=use_weights, use_contour=False, label_bw=True, 
                                                  target_size=target_size)

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

    data_path = '../data/dataset_fixed_256x256.npz'
    save_path = 'models/'
    model_name = 'rgb_batchnorm_fixed_256_weights'

    if not os.path.exists(save_path): 
        os.makedirs(save_path)

    print(model_name)
    print('RGB: {}, USE_WEIGHTS: {}, lr: {}'.format(RGB, USE_WEIGHTS, LEARNING_RATE))

    model = build_unet(LEARNING_RATE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, USE_WEIGHTS)
    # model = model.load_weights('models/unet_baseline_12.hdf5') # TODO try loading
    # model = load_model('models/unet_rgb_batchnorm_fixed_384_24.hdf5')

    # Copy first 10 conv layers from here
    # From: https://stackoverflow.com/questions/43294367/how-can-i-load-the-weights-only-for-some-layers#43294368
    # index matches found manually, using code at bottom
    # print('Initializing "encoder" with pre-trained VGG16 weights')
    # vgg16 = keras.applications.vgg16.VGG16()
    # vgglayeridx_to_unetlayeridx = { 1:1, 2:3, 4:6, 5:8, 7:11, 8:13, 9:15, 11:17, 12:19, 13:21}
    # for idx in vgglayeridx_to_unetlayeridx.keys():
    #     vgg_idx, unet_idx = idx, vgglayeridx_to_unetlayeridx[idx]
    #     model.layers[unet_idx].set_weights(vgg16.layers[vgg_idx].get_weights())

    train_data, val_data = build_data_generators(data_path, BATCH_SIZE, target_size=(IMG_HEIGHT,IMG_WIDTH), use_weights=USE_WEIGHTS)

    # lr_finder = LRFinder(model)
    # lr_finder.find_generator(train_data, start_lr=1e-6, end_lr=1, num_batches=300, epochs=1)
    # lr_finder.plot_loss(n_skip_beginning=0, n_skip_end=0)
    # plt.savefig('lr_finder_loss.png')
    # lr_finder.plot_loss_change(sma=20, n_skip_beginning=0, n_skip_end=0, y_lim=(-0.01, 0.01))
    # plt.savefig('lr_finder_loss_change.png')
    # import pdb; pdb.set_trace()

    checkpoint = ModelCheckpoint(save_path+model_name+'_{epoch:02d}.hdf5', monitor='val_loss',
                                 mode='min', period=1, save_weights_only=False)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    cyclic_lr = CyclicLR(base_lr=1e-4, max_lr=1e-3, step_size=2*STEPS_PER_EPOCH,
                         mode='triangular')
    tensorboard = TensorBoard(log_dir='/tmp/unet')
    plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    callbacks = [checkpoint, earlystopper, plateau, tensorboard]
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


    # VGG to U-net index matching
    # for i, o in enumerate(vgg16.layers): 
    #     if(isinstance(o, keras.layers.convolutional.Conv2D)): 
    #         print(i)
    #         print(o.get_weights()[0].shape)

    # for i, o in enumerate(model.layers): 
    #     if(isinstance(o, keras.layers.convolutional.Conv2D)): 
    #         print(i)
    #         print(o.get_weights()[0].shape)
