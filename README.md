# customer-churn-predictor

A machine learning web application that predicts whether a customer will churn based on their service usage and demographic information.

## Live Demo

https://customer-churn-predictor-p9oqvm3mqm7vcrwwt8n37p.streamlit.app/

## Project Overview

This project predicts customer churn in the telecommunications industry. It uses a Gradient Boosting Classifier trained on the IBM Telco Customer Churn dataset. The model considers factors like contract type, tenure, monthly charges, and service subscriptions to determine churn probability.

The application provides:
- Interactive input form for customer details
- Real-time churn prediction with probability scores
- Risk assessment with actionable recommendations
- Model performance metrics display
- Clean, professional user interface

## Dataset

The dataset used is the IBM Telco Customer Churn dataset, which contains:
- 7,043 customer records
- 21 features including demographic, account, and service information
- Target variable: Churn (Yes/No)

## Model Performance

- Accuracy: 81.48%
- Precision: 67.19%
- Recall: 81.21%
- F1-Score: 65.77%

## Technologies Used

- Python 3.10
- Streamlit for web interface
- Scikit-learn for machine learning
- Pandas and NumPy for data processing
- Pickle for model serialization
- matplotlib and seaborn for visualizations

## Features

- One-hot encoding for categorical variables
- Feature engineering including tenure grouping
- Multiple feature selection techniques
- Gradient Boosting Classifier with optimized parameters
- Interactive web interface with risk assessment

## Installation

1. Clone the repository
