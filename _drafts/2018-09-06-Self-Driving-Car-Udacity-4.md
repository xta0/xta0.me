---
updated: "2018-09-10"
layout: post
list_title: Autonomous Driving | Computer Vision 
title: Computer Vision 
categories: [AI,Autonomous-Driving]
mathjax: true
---

<img class="md-img-center" src="{{site.baseurl}}/assets/images/2018/09/cv-1.png">

> Course Notes from Udacity Self-Driving Car Nanodegree Program


### Image Classification Pipeline

An image classifier is an algorithm that takes in an image as input and outputs a label or “class” that identifies that image. For example, a traffic sign classifier will look at different of roads and be able to identify whether that road contains humans, cars, bikes and so on. Distinguishing and classifying each image based on its contents.

There are many types of classifiers, used to recognize specific objects or even behaviors — like whether a person is walking or running — but they all involve a similar series of steps...

1. First, a computer receives visual input from an imaging device like a camera. This is typically captured as an image or a sequence of images.
2. Each image is then sent through some pre-processing steps whose purpose is to standardize each image. Common pre-processing steps include resizing an image, or rotating it, to change its shape or transforming the image from one color to another - like from color to grayscale. Only by standardizing each image, for example: making them the same size, can you then compare them and further analyze them in the same way.
3. Next, we extract features. Features are what help us define certain objects, and they are usually information about object shape or color. For example, some features that distinguish a car from a bicycle are that a car is usually a much larger shape and that it has 4 wheels instead of two. The shape and wheels would be distinguishing features for a car. And we’ll talk more about features later in this lesson.
4. And then, finally, these features are fed into a classification model! This step looks at any features from the previous step and predicts whether, say, this image is of a car or a pedestrian or a bike, and so on.

### Pre-processing

Pre-processing images is all about standardizing input images so that you can move further along the pipeline and analyze images in the same way.

Really common pre-processing steps include:

1. Changing how an image looks spatially, by using geometric transforms which can scale an image, rotate it, or even change how far away an object appears, and
2. Changing color schemes, like choosing to use grayscale images over color images.

### HSV Conversion

In the code example, I used HSV space to help detect a green screen background under different lighting conditions. OpenCV provides a function hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV) that converts images from one color space to another.

After this conversion, I plotted the individual color channels; it was easy to see that the Hue channel remained fairly constant under different lighting conditions.

### Classifier

- day vs night
- training image source : [AMOS dataset](http://cs.uky.edu/~jacobs/datasets/amos/)
- feature extraction

### Feature Extraction

- Average Brightness

Here were the steps we took to extract the average brightness of an image.

1. Convert the image to HSV color space (the Value channel is an approximation for brightness)
2. Sum up all the values of the pixels in the Value channel
3. Divide that brightness sum by the area of the image, which is just the width times the height.

This gave us one value: the average brightness or the average Value of that image.

- High Pass Filter

## Resources

- [Image Kernels](http://setosa.io/ev/image-kernels/)
- [Roberts Cross Edge Detector](http://homepages.inf.ed.ac.uk/rbf/HIPR2/roberts.htm)
- [CNN explained](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)