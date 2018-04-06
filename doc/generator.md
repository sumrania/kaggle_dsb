# SegDataGenerator Instruction

SegDataGenerator is a segmentation image generator that based on original keras ImageGenerator. 


## Define the Generator for Data Augmentation

This should include all the data augmentation parameters. Add validation split, elastic trasnformation and right angle rotation([0,90,180,270]) to the original one. The label(segmentation mask and contour mask) will do the same trasnformation with the image.

### validation_split:
split some data from the [X_train, Y_train] as the validation dataset. Will keep the validation data from the back. The default value is 0. The value should be [0,1]

### rotation_right:
Use right angle rotation or not. The default value is False.

### elatic_transform:
Use elastic transformation or not. The default value is False.
Other default param: elastic_alpha=512, elastic_sigma=20.48, elastic_alpha_affine=20.48,

### Example:
trainGenerator = gen.SegDataGenerator(validation_split=0.2, width_shift_range=0.02,
                                   height_shift_range=0.02, zoom_range=0.1,
                                   horizontal_flip=True, vertical_flip=True,
                                   samplewise_center=False, samplewise_std_normalization=False,
                                   elastic_transform=True, rotation_right=True)


## Use flow_from_directory to Get Batches

Read images and labels from the given path. Add subset, use_contour and label_bw.

### directory and subset:
For this task, the directory should be the the path to the .npz file. The subset should be one of the 'training', 'validation' and 'testing'.

### use_contour:
Can choose whether use contour or not. The default is False.

### label_bw:
The segmentation mask now has possibility for each pixel. Set label_bw=True can trasnfer both segmentation and contour mask to 0-1 mask.

### Example:
train_data = trainGenerator.flow_from_directory(data_path, subset='training', batch_size=batch_size,
                                               class_mode='segmentation', color_mode='grayscale',
                                               use_contour=False, label_bw=True)

## Standard Usage for Data Augmentation

trainGenerator = gen.SegDataGenerator(validation_split=0.2, width_shift_range=0.02,
                                   height_shift_range=0.02, zoom_range=0.1,
                                   horizontal_flip=True, vertical_flip=True,
                                   samplewise_center=True, samplewise_std_normalization=True,
                                   elastic_transform=True, rotation_right=True)
                                   
train_data = trainGenerator.flow_from_directory(data_path, subset='training', batch_size=batch_size,
                                               class_mode='segmentation', color_mode='grayscale',
                                               use_contour=False, label_bw=True)

### note:
Use vizGenerator to visualizae the data augmentation results.
width_shift_range & height_shift_range & zoom_range may cause the generated image with stretched cells.

