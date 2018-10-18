# Data augmentation analysis for image classification
Team Project for the Foundations of Machine Learning (COMP61011) course at the University of Manchester, UK.
## About
The main goal of this project is to analyse different data augmentation methods for image classification and their performance on different Machine Learning models.
## Overview of the project
The chart below summarizes the tasks of the project
![Project workflow](/res/tasks_workflow.png)
## Datasets
We are using 3 openly avaible image classification datasets.
## Creating balanced datasets
From the chosen datasets we have to create balanced sets. We reduce the data to the same number of classes (`nclasses`) by choosing the classes with the most available examples. Then we create balanced data by elliminating examples of each classes to the number of samples in the smallest class.
## Creating unbalanced datasets
We choose a class to perform data augmentation on it. We reduce the number of samples in the `selected_class` to a given percentage (`reduce_perc`). We select the remaining samples in multiple different ways for data validation.
## Data augmentation methods
We are using multiple data augmentation methods which are popular for image classification
## Machine Learning models
We are using Machine Learning models described in the lecture
## Performance analysis and Results
We compare the different models and different methods. Results will available here with the lecturer's approval.
## Team members
 - Gergely Dániel Németh
 - Hanliang Rao
