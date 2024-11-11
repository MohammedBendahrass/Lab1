# Lab1
The main purpose behind this lab is to get familiar with Pytorch library to do  Classification and Regression tasks by establishing DNN/MLP architectures.
# NYSE Data Analysis and Deep Learning Model Development

This repository contains two notebooks, nyse-prices.ipynb and nyse-fundamentals.ipynb, dedicated to analyzing New York Stock Exchange (NYSE) data and implementing machine learning and deep neural network (DNN) models for regression and classification tasks using PyTorch.

## Objective

The primary objective of this project is to build proficiency in using the PyTorch library to develop models for classification and regression tasks, specifically by creating Deep Neural Network (DNN) architectures. We apply these methods to financial data from the NYSE to predict stock-related metrics and uncover patterns.

## Dataset

The dataset used is available on Kaggle: [New York Stock Exchange Data](https://www.kaggle.com/datasets/dgawlik/nyse). It contains historical stock price data and various fundamental indicators for companies listed on the NYSE. This data serves as the basis for both exploratory data analysis and model development.

## Project Workflow
### Exploratory Data Analysis (EDA)
Conducted EDA techniques to understand, clean, and visualize the dataset, identifying significant trends, patterns, and any preprocessing requirements.

### Deep Neural Network for Regression

  - Designed a DNN model using PyTorch to perform a regression task on stock prices or other continuous target variables.
  - Utilized Multi-Layer Perceptron (MLP) architectures to handle the complexities of stock price prediction.

### Hyperparameter Tuning with GridSearch

  - Employed the GridSearchCV tool from sklearn to determine optimal hyperparameters, including learning rate, optimizer choice, epochs, and model architecture.
  - Identified the best combination of parameters to improve model efficiency and accuracy.

### Model Training Visualization

  - Plotted Loss vs. Epochs and Accuracy vs. Epochs graphs for both training and test datasets.
  - Interpreted these plots to understand model performance, convergence, and areas for improvement.

### Regularization Techniques

  - Implemented regularization methods such as dropout and weight decay to enhance model generalization and reduce overfitting.
  - Compared results with and without regularization to highlight its impact on model performance.

## Notebooks Summary

### nyse-prices.ipynb

  - Goal: Perform EDA on NYSE price data and develop a DNN regression model.
  - Analysis: Visualized price trends, computed statistical summaries, and engineered features for deeper insights.
  - Model: Developed a regression model to predict stock price movements based on historical data.

### nyse-fundamentals.ipynb

  - Goal: Analyze fundamental financial data for companies on the NYSE and build a DNN regression model.
  - Analysis: Examined and visualized key financial metrics (e.g., revenue, net income) to assess company performance.
  - Model: Built a regression model using financial indicators to predict metrics related to financial health.





