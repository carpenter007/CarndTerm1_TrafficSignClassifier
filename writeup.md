# **Traffic Sign Recognition** 

---
This description refers to the [project code](https://github.com/carpenter007/CarndTerm1_TrafficSignClassifier) on GitHub.

**Building a Traffic Sign Recognition Project**

The goal of this project was the following:
* Load the data set of german traffic signs (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images from web
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupResources/Histogram_raw.png "Histogram of the training data"
[image2]: ./writeupResources/classifierExamples.png "Example of each classifier"
[image3]: ./writeupResources/preprocessingImage.png "Preprocessing a random image"
[image4]: ./writeupResources/Histogram_atLeastMeanSizeForAll.png "Histogram: at least mean size for all classifiers"
[image5]: ./writeupResources/Histogram_sameSizeForAll.png "Histogram: Same size for all classifiers"
[image6]: ./writeupResources/NewSigns.png "new signs from the web"
[image7]: ./writeupResources/predictions.png "highest 5 predictions"



### Data Set Summary & Exploration

I'm working with the data set provided from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) which is a subset of the German Traffic Sign Benchmark data set [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).


I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3, which means that the width and the height of each image is 32 pixels. The depth is 3 and represents a RGB image
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. 

The numpy command 'unique' is a nice method to get the number of images for each classifier.

```sh
import numpy as np
...
unique, counts = np.unique(y_train, return_counts=True)
```

It is a histogram which shows the distribution of different classes in the dataset given. You can easily notice that this distribution is pretty uneven (many classes have way below number of samples than mean). Its a good information when thinking about pre-processing the data set. 

![alt text][image1]


To get an insight how the data is looking, I used the Matplot library to visualize images from each classifier. 

![alt text][image2]



### Design and Test a Model Architecture

To get better predictions and train / validation accuracy we had to preprocess the image data.
I defined several functions to preprocess the data.

**Normalization of the image data:**
- Norming the data by the streching the 'min'-'max' data range of each RGB image to the maximum range (0 to 255) (only if no histogram equilization was performed)
- Normalized the data to a new min-max range (e.g. -0.5 to 0.5) to produce a range on which our gradients don't go out of control.

**Equilize Histogram:**
I got better results when performing histogram equilization on each image instead of norming the RGB data to the maximum range (0 to 255). Therefore I convert the RGB image to YUV color space. Then I equalize the histogram of the Y channel (brightness range now is from 0 to 255). At the end I convert the YUV image back to RGB format.

**Grayscale:**
Grayscaling the images before processing them through the CovNet will allow the network to focus on more relevant attributes like shapes, lines and edges at all.

**Tranforming:**
Transforming training images helps to make the NN more robust. Transforming could be: 
- Shifting the traffic sign image a few pixels in horizontal or vertical direction (-2 to 2 pixels)
- Rotating the image (-15 to 15 degrees)
- Scale the image (ratio of 0.9 to 1.1)
The ranges are choosen by adoption from the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) provided by LeCun

**Augmentation:**
Since the number of images per class varies a lot, I process augmentation for classes with less images. This will avoid overfitting the weights and biases for detecting classes with higher number of images while weak numbered classifiers leading to an innocent bias towards majority classes.

**Benchmarking of these functions**
- Normalization (scaling the image data and normalizing to lower numbers) increased the validation accuracy the most. I have not faced any negative side effects with this transformation.
- Equilizing the histogram of a YUV converted image has also improven the results. But the improvement is swallowed if the image is grayscaled and normalized afterwards.
- Grayscaling the image had some side effects indeed in my network. Grayscaling improved the validation accuracy and the test accuracy. But when it came to new traffic signs from the web, the grayscaling of the training data droped the accuracy from 80% to 40%. It seems like the colors in the image (when using histogram equilizing) can lead to an stable prediction, even if the image differs in all other features.
- Generating fake data had the expected effect, as decribed above. It prevents overfitting through the transformation and it avoids preferences to majority classes. But I've realized that the augmentation lead to a drop of the validation accuracy very fast, if the number of fake data is to big. First I generated data for every classifier until it has as many images as the classifier with the maximum number of images. The network was underfitting very hard. The result was quite interesting. While the validation accuracy dropped from 9.3 to 6.0, the test accuracy increased from 9.0 to 9.3. Generating images to the mean number of images per classifier lead to a more balanced result. 
![alt text][image4] ![alt text][image5]


Here is an example of a random traffic sign image when passing all the preprocessing steps:

1. Original image
2. Histogram equilazion
3. Random transforming
4. Grayscaling

![alt text][image3]

5. Normalizing the image is not shown here

**Last but not least: Shuffling the image data (training + validation data set) before processing it to the network**



My final model consisted of the following layers:

**Layer 1:** Convolutional. Input = 32x32x'dim'. Output = 28x28x6.
Relu Activation function
Max Pooling with 2x2 size. Input = 28x28x6. Output = 14x14x6.

**Layer 2:** Convolutional. Output = 10x10x16.
Relu Activation function
Pooling. Input = 10x10x16. Output = 5x5x16.
Flatten. Input = 5x5x16. Output = 400.

**Layer 3:** Fully Connected. Input = 400. Output = 120.
Relu Activation function
Dropout with 70% keeping rate

**Layer 4:** Fully Connected. Input = 120. Output = 84. 
Relu Activation function
Dropout with 70% keeping rate

**Layer 5:** Fully Connected. Input = 84. Output = 43 (n_classes).

All in all, the model is similar to the LeNet-5. In layer 3 and 4 I drop out units to prevent the system from overfitting

### Training the model

To update weights iterative based in training data  I used the very performant Adaptive Moment Estimation optimizer: Adam. Therefore I had to set following hyper parameters:

**learning rate** which describes the proportion that weights are updated:  **0.001**

**mu** which describes the mean parameter in the truncated_normal method: **0**

**sigma** which stands for the standard derivation (also for truncated_normal): **0.1**

**dropout** which is (1-keep probability): **0.5**

**batch size: 128**

**epochs: 30**

**learning rate: 0.0009**


My final model results were:
* training set accuracy of 
* validation set accuracy of 0.980
* test set accuracy of 0.942

I started with the LeNet-5 architecture which performed nice on the MNIST data set before. When I saw that just a good normalization of the data leads to an accuracy of 0.9, I was pretty sure that the LeNet-5 architectur will be able to reach the goal of at least 0.93 accuracy.

By mixing different preprocessing methods with different hyperparameter settings I was able to improve the validation accuracy to >9.3. But with increasing epochs the architecture was very fast to overfit.
Therefore I added some Dropouts into the architecture.


### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]

For humans the Stop sign may is the hardest to detect. But for the model the Slippery sign made trouble.

Here are the results of the prediction.
The image on the left side is the image which is tested. The next images in the row show the next 5 highest predictions.

![alt text][image7]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Compared to the test accuracy, this is worse than expected.

While the "no crossing" sign is detected very certain, the others are not that confident. I even realized, that when training the model again, with the same parameters and test data the prediction of some signs can swap to the wrong sign.

The predictions on new images were much more solid when I used colored images for the training set. Especially the Stop sign was predicted with 100% since the red color is a good identifer, even if there are branches in the picture.

The slippery road sign proably is not recognized because of the bigger size and position of the image which is even out of the range of the augmented jittered data used to train the model.
