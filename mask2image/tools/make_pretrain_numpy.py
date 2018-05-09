import numpy as np
import scipy.misc as sci
import glob

data_path = '../pretrain_test/images/'
dst_path = '../data/pretrain_256x256.npz'

image_list = []
num_image = len(glob.glob(data_path + '*.png')) // 3
X_train = np.zeros((num_image, 256, 256, 3))
Y_train = np.zeros((num_image, 256, 256, 1))
i = 0
for filename in glob.glob(data_path + '*.png'):
    image_list.append(filename)
image_list.sort()
# print(image_list)
for filename in image_list:
    type = filename.split('-')[-1]
    image = sci.imread(filename)
    print(filename)
    if type == 'inputs.png':
        X_train[i,:,:,:] = image
    elif type == 'outputs.png':
        Y_train[i,:,:,:] = np.mean(image, axis=2, keepdims=True)
        i += 1
np.savez(dst_path, X_train, Y_train)
print('finish saving image pairs to npz')
