# Neural_Network_Charity_Analysis

## Overview 

Neural networks (also known as artificial neural networks, or ANN) are a set of algorithms that are modeled after the human brain. They are an advanced form of machine learning that recognizes patterns and features in input data and provides a clear quantitative output. In its simplest form, a neural network contains layers of neurons, which perform individual computations. These computations are connected and weighed against one another until the neurons reach the final layer, which returns a numerical result, or an encoded categorical result.

The purpose of the analysis was with the “Deep Learning Neural Networks”, some features were used in the provided dataset to create a binary classifier to analyze the success of charitable donations.

The following methods  were used for the analysis:
* Preprocessing the data for the neural network model
* Compile, train and evaluate the model
* Optimize the model


## Results 

### Data Preprocessing

Using Pandas and Scikit-Learn’s StandardScaler(), the dataset was preprocessed in order to compile, train, and evaluate the neural network model later on.

* The [charity_data.csv](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/charity_data.csv) dataset was read into a Pandas DataFrame. 

* The following preprocessing steps were performed:
    * The "EIN" and "NAME" columns had been dropped.
    * The columns with more than 10 unique values were grouped together.
    * The categorical variables were encoded using the one-hot encoding.
    * The preprocessed data was split into features and target arrays.
    * The preprocessed data was split into training and testing datasets.
    * The numerical values had been standardized using the `StandardScaler()` module.

* The **“IS_SUCCESSFUL”** column was considered the **target** variable for the model. 

* Variables that were considered **features** for the model were every column except for IS_SUCCESSFUL which is the target variable and the ones that will be dropped.

* Variables that were neither targets nor features for the dataset were columns that were dropped  “EIN” and “NAME”  because those had little impact on the outcome. 

### Compiling, Training and Evaluating the Model
