import os
import numpy as np
import scipy.misc as sci

data_path = '../data/dataset_256x256.npz'
dst_path = '../data/pink_raw_256x256.npz'
train_path = '../data/train/'
val_path = '../data/val/'
validation_split = 0.2

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(val_path):
    os.makedirs(val_path)

file = np.load(data_path)
X_train = file['X_train']
Y_train = file['Y_train']

print(X_train.shape)
print(Y_train.shape)
X_train_pink = np.zeros(X_train.shape)
Y_train_pink = np.zeros(Y_train.shape)

num_image = X_train.shape[0]
val_start = int(num_image * (1-validation_split))
cnt = 0
for i in range(num_image):
    image = X_train[i]
    if np.mean(image) < 100:
        continue
    label = Y_train[i]
    X_train_pink[cnt,:,:,:] = image
    Y_train_pink[cnt,:,:,:] = label
    cnt += 1
    label = np.tile(label, [1,1,3])
    # print(label.shape)
    pair = np.concatenate((label, image), axis=1)
    if i < val_start:
        sci.imsave(train_path + str(i) + '.png', pair)
    else:
        sci.imsave(val_path + str(i - val_start) + '.png', pair)

X_train_pink = X_train_pink[:cnt,:,:,:]
Y_train_pink = Y_train_pink[:cnt,:,:,:]
np.savez(dst_path, X_train_pink, Y_train_pink)
print('saved all image pairs!')
