Explanation of Key Sections:
Data Preparation and Feature Engineering:

1- Load the dataset.
Create a sales difference column and generate supervised data.
Split the data into training and testing sets.

2- Scaling:
Use Min-Max scaling to scale the features between -1 and 1.

3- Training and Predictions:
Train a Linear Regression model.
Make predictions and inverse transform the results to the original scale.

4- Evaluation:
Calculate MSE, MAE, and R2 score to evaluate the model’s performance.

5- Visualization:
Plot the actual sales versus predicted sales for visual comparison.

# Sales Prediction with Linear Regression

This project demonstrates how to use linear regression to predict monthly sales based on historical sales data. The project includes data preprocessing, feature engineering, model training, evaluation, and visualization.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to predict future sales using a linear regression model. We use historical sales data, preprocess it, create a supervised learning dataset, and train a linear regression model to make predictions. The project also includes evaluation metrics and visualizations to compare the predicted sales against the actual sales.

## Prerequisites

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

You can install the required packages using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib
```

# The script will perform the following steps:

* Load and preprocess the data
* Create a supervised learning problem
* Split the data into training and testing sets
* Train a linear regression model
* Evaluate the model using MSE, MAE, and R2 score
* Plot the actual vs. predicted sales
* 
