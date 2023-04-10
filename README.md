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

!image.png


## Summary