Google collab folder with notebooks: https://drive.google.com/open?id=184SCoxmH8RFsYquz0o43MYfwrCx9o2vm

* see `keras_unet_kernel` and `unet_with_deformation`

### To-do

##### Preprocessing

* Write function to add weights for overlapping/touching cells
* try centering and normalization (feature-wise)
 * https://www.kaggle.com/hexietufts/easy-to-use-keras-imagedatagenerator

* Elastic deformation
    * ImageDataGenerator makes it easy to do standard data augmentation/pre-processing, but not easy to do an elastic deformation (with same random seed) for both an image and its label. Solution: make CustomImageDataGenerator contain ImageDataGenerator? or superclass? 
    * figure out what necessary interfaces are for ImageDataGenerator with elastic deformation

* Figure out how to implement weights and separations for cell borders
    * Want segmentation mask to have background pixels at borders
    * Weights of 1 to 10: 1 for normal , 5 for close ones, 10 for background pixels that were added
    * `ndimage.binary_opening`? like https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies
    * `cv2.findContours`

* Get contours (for DCAN)

##### Testing

* Test U-net with elastic deformation data augmentation

* Add batchnorm:
    * Replace dropout? test with/without
    * batchnorm: conv->relu->batchnorm->dropout
    * https://tuatini.me/practical-image-segmentation-with-unet/

* Should padding be valid? or same? In paper I think it's valid - but then this adds added complexity of 
* Identify failure cases - high validation loss
    * Color clusters by label number

##### Misc

* Organize work so far in single notebook
* understand run length encoding, evaluation metric

---

* Implement learning rate finder, cyclic learning rate for keras

### Pre-processing

* Do we need a standard image height and width for training? (see keras u-net kernel)
    * "select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size" for U-net
* normalize images to [-1, 1]
* To train the network to identify borders between touching cells: 
    * Insert background pixels between all touching objects in training images
    * Assign individual loss weight to each pixel (high at borders)
* Get contours? 

### Data augmentation

* original U-net paper makes elastic deformations

```
Especially random elastic deformations of the training samples seem to be the key concept to train a segmentation network with very few annotated images. We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid. The displacements are sampled from a Gaussian distribution with 10 pixels standard deviation. Per-pixel displacements are then computed using bicubic interpolation. Drop-out layers at the end of the contracting path perform further implicit data augmentation.
```

From "Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis"
```
However, our best results were deformations were created by first generating random displacement fields, that is ∆x(x,y) = rand(-1,+1) and ∆x(x,y)=rand(-1,+1), where rand(-1,+1) is a random number between -1 and +1, generated with a uniform distribution. The fields ∆x and ∆y are then convolved with a Gaussian of standard deviation sigma (in pixels). The displacement fields are then multiplied by a scaling factor alpha that controls the intensity of the deformation. 
```

* Other standard data augmentation methods:
    * cropping
    * Affine transformations (w/ bilinear interpolation)
        * mirroring, rotating
        * stretch vertically and horizontally
    * shade with a hue
    * add noise
    * RGB to grayscale?

What is the set of semantically correct augmentations that we can apply to microscope images? 

### Other

* train/val split, cross validation method
* standard function for evaluating models? Model class that has predict() function, with whatever else it needs to do to load weights. Evaluator will use test set to compute goodness metrics. Can we periodically rotate out a new testing/validation set from our training data? Having all data on repo would be great for this 

* https://github.com/matterport/Mask_RCNN
* larger images perform better in medical imaging
* in the last 12 months NasNets have shown significant improvement (50%) over older models
* The more classes you train the model on properly, the better results you can expect. (auxiliary tasks?)
* https://yanirseroussi.com/2014/08/24/how-to-almost-win-kaggle-competitions/
* Set up a local validation environment - cross validation?
* Try a bunch of approaches, create ensembles of the various approaches
* Data contains 650 images
* Examine data, and know it well! No need to generalize past this specific dataset, for kaggle competition
* what should our model look for? How big of a receptive field does it need for that?
* fast.ai: data reading/augmentation, NN training tricks
* scikit-learn: standard data analysis algorithms
* https://www.kaggle.com/c/data-science-bowl-2018/discussion/48130
* Kernels are a good starting point. From there, we’ll have to look closer at models/papers to understand pros/cons, understand the data, etc.
* read about xgboost

* Multi-level features and concatenations work well for image segmentation because we need high-resolution contextual information as well as fine feature-level information
* Another common issue in medical image segmentation: separating overlapping objects
    * U-net paper augments the mask label to strongly emphasize learning the boundaries between touching cells.
    * DCAN separates the network into two heads at the output (multi-task learning): segmentation and contour probabilities, and has separate mask labels for each head. These are then fused at the end for final segmentation. This is a much more explicit way of dealing with objects in contact

* In DCAN, what are the "auxiliary tasks that encourage gradient flow"?

##### Traditional CV methods

Most kernels demonstrating solutions using traditional methods did something like the following pipeline: 

* Threshold image with Otsu method (assume pixel values are distributed in bimodal distribution, find threshold in middle of the two peaks)
* Dilate/erode image to get rid of noise and separate touching components

### References

##### To read

* Tiramisu (basically a deeper U-net)
    * https://twitter.com/jeremyphoward/status/962890507130032129

##### Old

* [Depthnet](https://arxiv.org/abs/1608.06993), [Unet](https://arxiv.org/abs/1505.04597)
* other medical stuff (see Jeremy Howard’s tweets)
* https://www.kaggle.com/c/data-science-bowl-2018/discussion/49692
* https://www.sciencedirect.com/science/article/pii/S1361841516302043

* https://twitter.com/alexandrecadrin
* http://blog.kaggle.com/2016/04/28/yelp-restaurant-photo-classification-winners-interview-1st-place-dmitrii-tsybulevskii/

* https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/
* https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/
* [Fast R-CNN](http://ieeexplore.ieee.org/document/7410526/)
* [Notes of Broad Institute guy](https://www.kaggle.com/c/data-science-bowl-2018/discussion/48130)
* https://yanirseroussi.com/2014/08/24/how-to-almost-win-kaggle-competitions/
* https://blog.insightdatascience.com/heart-disease-diagnosis-with-deep-learning-c2d92c27e730
* https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html

_CheXNet_

* https://www.youtube.com/watch?v=QmIM24JDE3A
* https://lukeoakdenrayner.wordpress.com/2018/01/24/chexnet-an-in-depth-review/
* https://medium.com/@judywawira/are-computers-better-than-doctors-2e07a05ae7ea

