# Installation Instructions
-----------------------------------------------------------------------

1. Create a virtual environment with python 3.7. For example if you are using conda, use statement: `conda create env -n some_name python=3.7`

2. Activate environment. For example with conda, use statement `conda activate some_name`

3. Install `poetry` [https://python-poetry.org/docs/#installation]

4. Run `poetry install`

# Overview - Organizing code for model selection and model evaluation
-----------------------------------------------------------------------
Data scientists generally spend a lot of time training different models before narrowing down on the model that best suits their data. Here I have provided a code structure that makes it easy to implement and evaluate many models using my wrapper function. This simplifies the job of a data scientist and ML engineer to a great extent. To walk throught the code, I used a toy dataset and various regression models to demonstrate while providing a quick background for regression.


# Background
-----------------------------------------------------------------------
Regression:

The goal of regression is to predict a continuous response variable given some input variables. There are different regression models available. Some are meaningful and relevant for the problem and some not so much. We are interested in learning which models are the best for your problem and dataset. How do we do this? We need to identify relevant models from the non-relevant ones. For example, linear regression for predicting a continuous variable is a relevant model while logistic regression is not. Logistic regression is used for classification.

Model Selection:

How do we say one model is better than another? There are many measures that are used, one of which is r-square. One caveat to call to attention, r-square is not the best metric to evaluate multiple variable regression, but it is the most looked at. If r-square is low there is no linear relationship. We also need to look at the r-square for test and training data. Test r-square is more reliable because training r-square is a result of the data used to build r-square. Test r-square is also an indicator of model generalization, i.e. how well does the model perform on un-seen data.

Writing code to explore different models is a time-consuming process, not to mention the bulky code it creates. Therefore, it needs automation.

Here is one such tool which automates all required functions, a wrapper function that creates an architecture for the code, enables less cumbersome and more manageable code that is also easier to debug!

Data:

For explaining the steps of the code we will be using the automobile mpg dataset to predict the miles per gallon of the car given other features of the car such as engine size, age, horsepower, weight and acceleration of car.

The dataset has been cleaned and preprocessed for the purpose of the tutorial. Data Preprocessing is a topic on its own and if not done correctly can result in misleading conclusions. This topic is not covered here.

Relevant regression models under consideration: Linear, Ridge, Lasso, Elastic Net, SVR, SGD, KNN, DT (we will try to cover as many as possible in the time frame)

Model Workflow in real-life:

Model workflow in real-life is process oriented. Sourcing data -> Exploratory data analysis -> data pre-processing -> training and evaluating best models and choosing best models for your data.

It all starts with sourcing data, is there enough data for the model to start detecting patterns? Sourcing data and curating features is the first step (not covered here).

Exploratory data analysis is where you understand the data trends/patterns and outliers in the through visualizations. It provides the context needed to develop appropriate model and interpret results correctly.

Data preparation and pre-processing is massaging the data to make it suitable for model algorithms. This is a common requirement for many machine learning algorithms implemented in sklearn, they might behave badly if the individual features deviate from model assumptions.

Train and test splits are performed on data as the goal of ML is to find prediction on un-seen data. The ML model learns patterns from the data that is used to train it and needs to be evaluated on un-seen data to learn if it also generalizes well. The test hold out data serves this purpose of model validation.

Typically many models are tried on the datasets and the best performing model is chosen for production and deployment. There are many evaluation/ model scoring metrics which allow for comparison between models. The model with best score is considered the best model.

Regularization:

Regularization is a technique that is used commonly for model selection although not always necessary. It controls for overfitted models and improves performance for un-seen data. In this process we try to penalize the coefficients of variables by reducing their magnitude. As a result coefficients of some variables are much higher than the coefficients of other variables, which control the predictions. The popular methods for regularization in linear models are Lasso, Ridge and ElasticNet regressions which follow L1, L2 and combination of both methodologies.

Model Validation:

The idea of building models works on a constructive feedback principle. You build a model, get feedback from metrics, make improvements and continue until you achieve a desirable accuracy. Evaluation metrics explain the performance of a model. An important aspect of evaluation metrics is their capability to discriminate among model results. After you are finished building your model, these metrics will help you in evaluating your model’s accuracy.

RMSE is the square root of average of squared errors. While this is a good metric that gives a sense of how close the predicted values are to actual value. There is no baseline to say a specific value is considered good and no baseline for comparison.

R-square metric is the proportion of that variation of sum of squared deviations from the mean that is explained by the model. In other words, how good is our regression model compared to a very simple model that just predicts the mean value of target from the train set as predictions? How much of variation of the target is explained by the regression? If it is 0, your model isn’t explaining anything, if it is closer to 1 your model is getting better at predicting the target.

R-square doesn’t consider the number of variables in the model. So it doesn’t penalize complex models, rather complex models tend to have r-square closer to 1. Hence this metric should be used with caution. For the purpose of our dummy data, we will consider using r-square for evaluation.

Adjusted r-square takes the number of features into account and penalizes more complex models. This is generally a better metric for multi-variate regression.

Now you can jump into the python code to see the API.
