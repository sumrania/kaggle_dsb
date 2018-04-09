import os
import sys

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import keras
from keras.models import Model, load_model
from skimage.transform import resize
from skimage.io import imread
from skimage.morphology import label

from my_utils import load_saved_data, plots
from keras_unet import build_unet, mean_iou
from SegDataGenerator import SegDataGenerator, rgb2gray
from calc_score import *

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

if __name__=="__main__":

    RGB = True
    USE_WEIGHTS = False

    IMG_HEIGHT = IMG_WIDTH= 256
    IMG_CHANNELS = 3 if RGB else 1

    npz_save_name = 'unet_predictions.npz'
    csv_save_name = 'unet_submission.csv'

    ### Load model with weights, or entire model 
    # Entire model is preferred (even though filesizes are much larger), since architectures might change
    print('Loading model')

    # weights_path = '../unet/models/unet_rgb_batchnorm_24.hdf5'
    # model = build_unet(0.0, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, USE_WEIGHTS)
    # model.load_weights(weights_path)

    model_path = 'models/unet_rgb_vgg16_batchnorm_28.hdf5'
    model = load_model(model_path, custom_objects={'mean_iou': mean_iou})

    data_path = '../data/dataset_256x256.npz'
    X_train, Y_train, C_train, W_train, X_test = load_saved_data(data_path, image_size=(IMG_HEIGHT, IMG_WIDTH))

    # TODO convert to grayscale
    if not RGB:
        X_train_gray = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
        X_test_gray = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        for i in range(X_train.shape[0]):
            X_train_gray[i,:,:,:] = rgb2gray(X_train[i,:,:,:])
        for i in range(X_test.shape[0]):
            X_test_gray[i,:,:,:] = rgb2gray(X_test[i,:,:,:])    
        X_train = X_train_gray
        X_test = X_test_gray
        
    # train/val split
    split = 0.8
    X_train, Y_train = X_train[:int(X_train.shape[0]*split)], Y_train[:int(X_train.shape[0]*split)]
    X_val, Y_val = X_train[int(X_train.shape[0]*split):], Y_train[int(X_train.shape[0]*split):]

    # Make predictions!
    preds_train = model.predict(X_train, verbose=1)
    preds_val = model.predict(X_val, verbose=1)
    preds_test = model.predict(X_test, verbose=1)

    # Drop garbage layer if using weights
    if USE_WEIGHTS:
        def sigmoid(x): return 1/(1+np.exp(-x))
        preds_train = sigmoid(preds_train[:,:,:,0])
        preds_val = sigmoid(preds_val[:,:,:,0])
        preds_test = sigmoid(preds_test[:,:,:,0])

    # Threshold predictions - TODO figure out what happens without this
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    TEST_PATH = '../data/stage1_test/'
    test_ids = next(os.walk(TEST_PATH))[1]

    # Create list of upsampled test masks
    sizes_test = []
    print('Getting test image sizes ...')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        pred = preds_test[i] 
        preds_test_upsampled.append(resize(np.squeeze(pred), 
                                           (sizes_test[i][0], sizes_test[i][1]), 
                                           mode='constant', preserve_range=True))

    print('Saving to npz at: ', npz_save_name)
    np.savez_compressed(npz_save_name, preds_train=preds_train, preds_val=preds_val, preds_test=preds_test)
    print('Done!')

    # Create submission file
    new_test_ids = []
    rles = []
    for n, id_ in tqdm(enumerate(test_ids)):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    sub.to_csv(csv_save_name, index=False)
    print('Saved submission file to ', csv_save_name)
