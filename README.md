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

### nyse-prices-adjusted.ipynb

  - Goal: Perform EDA on NYSE price data and develop a DNN regression model.
  - Analysis: Visualized price trends, computed statistical summaries, and engineered features for deeper insights.
  - Model: Developed a regression model to predict stock price movements based on historical data.

### nyse-fundamentals.ipynb

  - Goal: Analyze fundamental financial data for companies on the NYSE and build a DNN regression model.
  - Analysis: Examined and visualized key financial metrics (e.g., revenue, net income) to assess company performance.
  - Model: Built a regression model using financial indicators to predict metrics related to financial health.

## Requirements

  - Python 3.x
  - PyTorch
  - Scikit-Learn
  - Pandas
  - Matplotlib
  - Seaborn

## Usage

  - Clone this repository and open the notebooks.
  - Follow the workflow in each notebook to replicate the analysis and modeling steps.
  - Modify parameters as needed to explore specific models or data subsets.


# Part Tow Multi Class Classification

# Predictive Maintenance - Deep Learning with PyTorch

This repository contains a Jupyter notebook, predictive-maintenance.ipynb, focused on predictive maintenance using deep learning techniques with PyTorch. The notebook guides you through building and optimizing a deep neural network for a multi-class classification problem.

## Objective

The primary goal of this lab is to become proficient with the PyTorch library for performing classification and regression tasks. The notebook demonstrates building deep neural networks (DNNs) and multi-layer perceptrons (MLPs) to solve a predictive maintenance problem.

## Dataset
The dataset used for this project is available on Kaggle: [Machine Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification). It contains machine sensor data with the goal of predicting maintenance needs.

## Notebook Workflow

### 1. Data Preprocessing

  - Applied data cleaning techniques and standardized/normalized the dataset for improved model performance.

### 2. Exploratory Data Analysis (EDA)

  - Used various EDA techniques to gain insights into the dataset and visualize relationships between features.

### 3. Data Augmentation

  - Performed data augmentation to balance the dataset and improve the model's ability to generalize.

### 4. Model Architecture

  - Established a deep neural network (DNN) using PyTorch for the multi-class classification task, targeting failure type prediction.

### 5. Hyperparameter Tuning

  - Used the GridSearch tool from the sklearn library to identify optimal hyperparameters such as learning rate, optimizer, number of epochs, and model architecture for better model efficiency.

### 6. Training Visualization

  - Visualized training and validation results, specifically Loss vs. Epochs and Accuracy vs. Epochs, with interpretations of each graph.

### 7. Model Evaluation

  - Calculated performance metrics including accuracy, sensitivity, and F1 score on both training and testing datasets.

### 8. Regularization Techniques
  
  - Applied various regularization methods to the model and compared their results with the initial model performance.

## Results

The notebook showcases the effectiveness of the DNN/MLP model in handling multi-class classification for predictive maintenance and provides insights into the benefits of regularization, data augmentation, and hyperparameter optimization.

## Requirements

  - PyTorch
  - scikit-learn
  - Pandas
  - Matplotlib





