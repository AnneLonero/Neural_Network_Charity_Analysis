# Neural_Network_Charity_Analysis

## Overview
The purpose of this project is analyzing the dataset containing more than 34,000 organizations that have received funding from Alphabet Soup (a philanthopic non-profit organization) over the years to identify impact of each donation and vet the potential recipients. We used deep-learning neural networks with TensorFlow and Python to analyze and classify the success of charitable donations.

The following methods were also used in the analysis:
* Preprocess data for neural network model
* Compile, train and evaluate the model
* Optimize the model for higher accuracy

## Results

### Data Preprocessing
* Column `IS_SUCCESSFUL` is considered as target for our deep learning neural network. This column cointains binary data indicates whether or not the charity donation was used effectively.
* Columns `APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT` are considered features for our model. We later encoded these features, splitting them into training and testing datasets and standardized them to feed our learning model.
* Two columns `EIN` and `NAME` are identification information that we don't need for our model and were removed from the input data.

### Compliling, Training and Evaluating the Model
* In the initial deep-learning neural network, we selected two hidden layers with 8 and 5 neurons for each layer, and `ReLU` for activation function to speed up the training process. However, we used `Sigmoid` as activation for output layer since output is a binary classification.

![Initial deep-learning model](https://user-images.githubusercontent.com/114631804/231018555-bda5d9fd-c560-47e1-bbfa-83f375b54021.png)


* The model shows the accuracy is 0.726, which is almost 72%. This is pretty low number and we would considered this model has NOT been able to achieve the target performance.

![Initial model testing result](https://user-images.githubusercontent.com/114631804/231018588-b75f244b-10db-4355-800d-22e4b5c57a3d.png)


* To attempt to increase model performance, we increased values in the `APPLICATION_TYPE` buckets by categorized any `APPLICATION_TYPE` that has less then 1000 applications to "Other" category rather than 500 in the initial model. This change decrease our number of columns from 44 to 41. As a result, our accuracy slightly improved to 0.7278.

![Initial model testing result](https://user-images.githubusercontent.com/114631804/231018624-c8c47440-f2f0-42ab-95bb-85665e815826.png)

We also added more neurons to each hidden layers, 16 and 10 and later on added another hidden layer with 5 neurons. However, the result was not significantly improved.

## Summary
Our deep-learning neural network did not reach our target of 75% accuracy. We can potentially apply other methods to optimize the performance of our model. We can also use supervised machine learning model to generate classified output to compare the performance against our neural network model.
