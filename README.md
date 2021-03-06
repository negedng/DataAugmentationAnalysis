# Data augmentation analysis for image classification
Team Project for the Foundations of Machine Learning (COMP61011) course at the University of Manchester, UK.
## About
The main goal of this project is to analyse different data augmentation methods for image classification and their performance on different Machine Learning models.
## Overview of the project
The chart below summarizes the tasks of the project
![Project workflow](/res/tasks_workflow.png)
## Datasets
We are using 3 openly avaible image classification datasets.
 - MNIST (handwritten digits): http://yann.lecun.com/exdb/mnist/
 - CIFAR 10 (images): https://www.cs.toronto.edu/~kriz/cifar.html
 - Fashion-MNIST (clothes): https://github.com/zalandoresearch/fashion-mnist
## Creating balanced datasets
From the chosen datasets we have to create balanced sets. We reduce the data to the same number of classes by choosing the classes with the most available examples. Then we create balanced data by elliminating examples of each classes to the number of samples in the smallest class.
## Creating unbalanced datasets
We choose a class to perform data augmentation on it. We reduce the number of samples in the selected class to a given percentage. We select the remaining samples in multiple different ways for data validation.
## Data augmentation methods
We are using multiple data augmentation methods which are popular for image classification.
 - Rotate and crop
 - Elastic Distortion
 - Shear
 - All the above
![Data augmentation examples](/res/augmentTypes.png)

We are using the [Augmentor](https://github.com/mdbloice/Augmentor) package for the first two methods. 
## Machine Learning models
We are using Machine Learning models described in the lecture
 - Logistic Regression
 - Support Vector Machine
 - k-Nearest Neighbours
## Performance analysis and Results
We compare the different models and different methods. Results will be available here with the lecturer's approval.
## Team members
 - Gergely Dániel Németh
 - Hanliang Rao
