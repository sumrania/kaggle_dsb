# Common Pre-processing Steps

inspired from https://www.kaggle.com/raoulma/nuclei-dsb-2018-tensorflow-u-net-score-0-352

## Preprocessing

- Resize Image and Mask (both train and test) to 256x256
  - Maximum number of images are of this size (334/671: 49%)
- Normalize Image and Mask
  - Mean subtraction and divide by std
  - Divide by max value
  - Divide by 1 if max value per axis is less than mean of image
- Convert images to grayscale
- Invert images with light background
- Maybe threshold images maybe using Otsu method used by [STKBailey](https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies)
- Transform y_label probabilities to either 0 or 1 (?)

## Data Augmentation for both Images and Masks

- rotation_range = 90.
- width_shift_range = 0.02 
- height_shift_range = 0.02
- zoom_range = 0.10
- horizontal_flip=True
- vertical_flip=True


