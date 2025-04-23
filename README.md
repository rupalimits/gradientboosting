# Employee Attrition Prediction using Gradient Boosting Models

## Problem Statement

Employee attrition poses a significant challenge for organizations, impacting productivity, team morale, and overall operational costs. Understanding the factors leading to employee turnover and predicting attrition in advance can help HR departments take proactive measures to retain talent.

This project aims to build a machine learning pipeline that can predict whether an employee is likely to leave the organization, based on a variety of personal and professional attributes.

## Project Objective (Target)

The goal is to predict the likelihood of employee attrition (Yes/No) using historical HR data. The task is framed as a binary classification problem where the target variable is Attrition.

## Dataset

Source: IBM HR Analytics Employee Attrition & Performance dataset (Kaggle Link)
Total Records: 1,470
Target Column: Attrition (Yes or No)
Feature Types: Mix of categorical, ordinal, and numerical features (e.g., Age, MonthlyIncome, JobRole, YearsAtCompany, WorkLifeBalance, etc.)

## Approach

1. Data Cleaning and Preprocessing:
  a. Handled missing values (none present in the dataset)
  b. Encoded categorical features using one-hot encoding and label encoding
  c. Scaled numerical features using standardization

2. Model Selection and Training: Evaluated and compared the performance of three advanced gradient boosting algorithms:
  a. XGBoost
  b. CatBoost
  c. LightGBM
  d. Hyperparameter Tuning: Used Hyperopt library for this.

## Evaluation Metrics

To ensure robust performance analysis, the following metrics were used:
Accuracy, Precision, Recall, F1-Score, ROC-AUC Score, Confusion Matrix

## Best Model: XGBoost
Accuracy: 89%, F1-Score: 87%, ROC-AUC: 0.91

Outperformed CatBoost and LightGBM on both precision and recall, indicating better balance and generalization across classes.

## Tech Stack
Python, Jupyter Notebook, Scikit-learn, XGBoost, LightGBM, CatBoost, Matplotlib & Seaborn for visualizations
