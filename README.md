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

Using Pandas and Scikit-Learn’s `StandardScaler()`, the dataset was preprocessed in order to compile, train, and evaluate the neural network model later on.

* The [charity_data.csv](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/charity_data.csv) dataset was read into a Pandas DataFrame. 

* The following preprocessing steps were performed:
    * The "EIN" and "NAME" columns had been dropped.
    * The columns with more than 10 unique values were grouped together.
    * The categorical variables were encoded using the one-hot encoding.
    * The preprocessed data was split into features and target arrays.
    * The preprocessed data was split into training and testing datasets.
    * The numerical values had been standardized using the `StandardScaler()` module.
    * The results were saved and export to an HDF5 file. [AlphabetSoupCharity.h5](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5)
    

* The **“IS_SUCCESSFUL”** column was considered the **target** variable for the model. 

* Variables that were considered **features** for the model were every column except for IS_SUCCESSFUL which is the target variable and the ones that will be dropped.

* Variables that were neither targets nor features for the dataset were columns that were dropped  “EIN” and “NAME”  because those had little impact on the outcome. 

### Compiling, Training and Evaluating the Model

* The neural network model using Tensorflow Keras contains working code that performs the following steps:
    * The number of layers, the number of neurons per layer, and activation function were defined. 
    * An output layer with an activation function was created.
    * The structure of the model and the loss and accuracy of the model was displayed. 
    * The model's weights were saved every 5 epochs.
    [Image_3](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/Images/Image_3.png)
    * The results were saved to an HDF5 file.  [AlphabetSoupCharity_Optimization.h5](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5)

* For the **input layer**, the number of input features must be equal to the number of variables in the feature DataFrame. Because of that the `len(X_train[0])` was used. 
* In the **hidden layers**, the deep learning model structure should be slightly different—two hidden layers were set to the model. The first hidden layers had 80 neurons and the second 30 neurons. All of the hidden layers used the `relu` activation function to identify nonlinear characteristics from the input values.
* In the **output layer**, the same parameters were used from the basic neural network including the sigmoid activation function. The **“sigmoid activation”** function would help to predict the probability that an employee is at risk for attrition.  [Image_1](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/Images/image_1.png)

* The model was not able to reach the target of 75% accuracy.  The accuracy of the model was 68% and this was not a satisfying performance to help predict the outcome of the charity donations. [Image_2.png](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/Images/Image_2.png)

* To increase the performance of the model;

    * At the first attempt, in addition to the “EIN” and “NAME“ columns “USE_CASE” column was removed. However, model accuracy only went up to 73%.  [Attempt_1](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/Images/Attempt_1.png)
    * At the second attempt for optimization, an additional hidden layer was added to the model and the number of neurons was changed. The first hidden layer was set to 100 neurons and the second hidden layer was set to 50 neurons. The additional third hidden layer was set to 20 neurons. But the model accuracy dropped to 72.5%. [Attempt_2](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/Images/Attempt_2.png)
    * For the third attempt, the activation function of the output layer was changed to the `tanh` activation function.  However, the accuracy of the model stayed the same with the second attempt, 72.5%. [Attempt_3](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/Images/Attempt_3.png)

## Summary

The initial neural network accuracy score was **68%**. After the optimization attempts, the accuracy went up to **72.5%**, however, it did not reach the target of 75% accuracy. Considering the result of the first attempt,  removing more features, or simply adding more data to the dataset might increase accuracy. 

Because of the binary classification,  a supervised machine learning model can be used, such as the `Random Forest Classifier` to combine a multitude of decision trees to generate a classified output and evaluate its performance against our deep learning model.


## Resources

* **Data Source**: 
    * [charity_data.csv](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/charity_data.csv) 
    * [AlphabetSoupCharity.h5](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.h5) 
    * [AlphabetSoupCharity_Optimzation.h5](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5)
* **Data Tools**: 
    * [AlphabetSoupCharity.ipynb](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)
    
    * [AlphabetSoupCharity_Optimzation.ipynb](https://github.com/duygusimsek/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimzation.ipynb)
* **Software**: 
    * [Python 3.10.2](https://www.python.org/downloads)
    * [Jupiter Notebook 6.3.0](https://jupyter.org/)
    * Pandas
    * Anaconda 
