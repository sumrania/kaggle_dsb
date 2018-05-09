import os
import numpy as np
import SegDataGenerator as gen
import scipy.misc as sci

batch_size = 1
data_path = '../../data/dataset_256x256.npz'
pretrain_path = '../data/pretrain/'


if not os.path.exists(pretrain_path):
    os.makedirs(pretrain_path)

trainGenerator = gen.SegDataGenerator(validation_split=0.2, width_shift_range=0.2,
                                   height_shift_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, vertical_flip=True,
                                   elastic_transform=True, rotation_range=50)
train_data = trainGenerator.flow_from_directory(data_path, subset='training', batch_size=batch_size,
                                               class_mode='segmentation', color_mode='grayscale',
                                               use_contour=True, label_bw=True)

# generate pre-train data
for i in range(1000):
    image, label = next(train_data)
    seg = label['segmentation']
    image = np.squeeze(image, axis=0)
    seg = np.squeeze(seg, axis=0)
    image = np.tile(image, [1,1,3])
    seg = np.tile(seg, [1,1,3])
    pair = np.concatenate((seg, image), axis=1)
    sci.imsave(pretrain_path + str(i) + '.png', pair)

print('save 1000 pretrain data')
