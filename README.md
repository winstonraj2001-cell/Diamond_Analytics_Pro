# 💎 Diamond Analytics Pro

An AI-powered Data Science dashboard for diamond price prediction, customer segmentation, market analytics, business intelligence, and interactive data visualization.

## 📌 Project Overview

Diamond Analytics Pro is an end-to-end Machine Learning and Data Science project developed using Python and Streamlit.

The project combines machine learning, feature engineering, clustering, exploratory data analysis, and business intelligence into a single interactive dashboard.

The main objective is to predict diamond prices using physical and quality-related characteristics and provide meaningful analytical insights for business decision-making.

## 🎯 Project Objectives

- Predict diamond prices using Machine Learning
- Analyze diamond characteristics and quality
- Perform K-Means customer/product segmentation
- Analyze market and pricing patterns
- Generate business intelligence insights
- Provide interactive data visualizations
- Evaluate Machine Learning model performance
- Build a professional Streamlit dashboard

## 🤖 Machine Learning

### Diamond Price Prediction

**Algorithm:** XGBoost Regression

**Task:** Regression

**Target Variable:** `price_inr`

The model predicts the estimated diamond price in Indian Rupees based on diamond characteristics and engineered features.

### Features

The prediction system uses diamond attributes such as:

- Carat
- Depth
- Table
- X
- Y
- Z
- Cut
- Color
- Clarity
- Calculated Volume
- Dimension Ratio
- Other engineered/model features

The final trained model uses **28 features**.

## 🎯 K-Means Segmentation

K-Means clustering is used to group diamonds into meaningful segments based on their characteristics.

The clustering workflow includes:

- Feature selection
- Data preprocessing
- Feature scaling
- K-Means clustering
- Cluster encoding
- Segment analysis
- Visualization

The resulting clusters help identify different groups of diamonds based on their characteristics and pricing patterns.

## 📊 Dashboard Modules

The Streamlit dashboard contains multiple analytical sections:

### 💎 Executive Command Center

Provides an overall project summary and key diamond analytics.

### 💰 AI Price Prediction

Allows users to enter diamond characteristics and obtain an estimated diamond price using the trained XGBoost Regression model.

### 🎯 K-Means Segmentation

Displays diamond clusters and segmentation results.

### 📊 Market Analytics

Provides interactive analysis of diamond prices, characteristics, quality, and market patterns.

### 💼 Business Intelligence

Provides data-driven business insights based on the analyzed diamond dataset.

### ⚙️ Model Performance

Displays information related to the Machine Learning model and its performance.

### 📁 Data Explorer

Allows exploration of the underlying diamond dataset.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly
- Joblib
- Machine Learning
- Data Visualization
- Feature Engineering
- K-Means Clustering

## 📂 Project Structure

```text
Diamond_Analytics_Pro/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│
├── models/
│   ├── best_diamond_model.pkl
│   ├── diamond_cluster_encoder.pkl
│   ├── diamond_cluster_scaler.pkl
│   ├── diamond_encoder.pkl
│   ├── diamond_feature_information.pkl
│   ├── diamond_kmeans_model.pkl
│   └── diamond_scaler.pkl
│
├── utils/
│   ├── __init__.py
│   ├── clustering_utils.py
│   ├── insights.py
│   ├── model_loader.py
│   └── prediction_utils.py
│
└── notebooks/
