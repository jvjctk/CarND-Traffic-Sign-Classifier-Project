# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jvjctk/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

![alt text][./images/imageshow.jpg]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][./images/visualization.jpg]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I shuffled data using 'sklearn.utils'. I did not converted to gray scale as I decided to use 3 color channels. Then I splitted data into train and validation set using same 'sklearn.utils'. I found mean and standard deviation of each set and normalized data using this mean and standard deviation. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x3 RGB image   							| 
| Convolution			| 1x1 stride, VALID padding, outputs 28x28x7 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 16x16x7 					|
| Convolution			| 1x1 stride, VALID padding, outputs 10x10x13 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x13					|
| Flatten				| outputs 325 									|
| Fully Connected		| input = 325, output = 153 					|
| RELU					|												|
| Fully Connected		| input = 700, output = 43 						|

 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an edited version of classic LeNet. Accurancy is found to be improved with the number of epochs. So I prefer to use 60 epochs. I took learning rate as .001 and batch size as 100 what I found as standard value. Entropy was found using  'tf.nn.softmax_cross_entropy_with_logits' which is followed by 'tf.reduce_mean' to find out loss. Optimizer used is 'tf.train.AdamOptimizer'

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.3
* test set accuracy of 91.1

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried a classic version of LeNet-5 with 5 layers. It was working good for my images. However I tried to modified the network and I reduced 1 fully connected layer. It also gives us almost same resuts but increased calculation speed 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web: ![alt text][./images/newimages.jpg]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| General caution  		| General caution   					| 
| No passing   			| No passing 							|
| Roundabout mandatory	| Roundabout mandatory					|
| 60 km/h	      		| 60 km/h				 				|
| Stop  				| Stop      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of images from web. However this varies for defferent epochs

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a General caution sign (probability of 0.9999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| .99         			| General caution   						| 
| .00     				| Traffic signals 							|
| .00					| Pedestrians								|
| .00	      			| Right-of-way at the next intersection 	|
| .00				    | Road narrows on the right				 	|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


