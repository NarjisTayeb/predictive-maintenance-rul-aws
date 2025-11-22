Predictive Maintenance with AWS – Remaining Useful Life (RUL) Prediction

This project develops a complete predictive maintenance system to estimate the Remaining Useful Life (RUL) of aircraft engines using the NASA CMAPSS turbofan engine dataset.
It compares multiple machine learning and deep learning models and includes an AWS-ready deployment structure.

🚀 Project Overview

The goal of this project is to predict when an aircraft engine is likely to fail by estimating its RUL based on multivariate sensor data.
This allows maintenance teams to perform repairs before failure, reducing downtime and increasing reliability.

🔧 Tech Stack

Python (Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras)

AWS S3 – data storage & model artifacts

AWS SageMaker – model training & deployment architecture

Jupyter Notebook for experimentation

Matplotlib / Seaborn for visualization

🧹 Data Preprocessing

Missing value handling

Sensor normalization

RUL labeling

Window generation for LSTM models

Feature engineering (cycle-based degradation indicators)

🧠 Models Implemented
1️⃣ Random Forest Regressor

Best-performing model

MAE ≈ 8.7, RMSE ≈ 12.9

2️⃣ XGBoost Regressor

Strong performance close to RF

MAE ≈ 8.86, RMSE ≈ 12.93

3️⃣ LSTM Network

Time-series deep learning model

Performed well but not better than RF/XGBoost

Higher RMSE due to dataset complexity

📊 Results Summary
Model	MAE	RMSE
Random Forest	8.77	12.98
XGBoost	8.86	12.93
LSTM	19.28	38.29

➡️ Random Forest was chosen for deployment due to best accuracy + lowest complexity.

☁️ AWS Deployment (Architecture Ready)

Although the endpoint was not deployed to avoid AWS charges, the project includes full AWS deployment preparation:

Trained model saved as model.joblib

Packaged into model.tar.gz

Custom inference script included: inference.py

Designed for deployment on AWS SageMaker with an SKLearn container

Artifacts stored in Amazon S3

📁 Repository Structure
├── predictive_maintenance.ipynb      # Full analysis, modeling, evaluation
├── model.joblib                      # Trained Random Forest model
├── model.tar.gz                      # Packaged model for SageMaker
├── inference.py                      # Inference script for AWS deployment
├── README.md                         # Project documentation
└── data/                             # (optional) NASA CMAPSS sample files

📌 How to Use

Clone repository

Install dependencies

Run the notebook to reproduce results

Use model.joblib for inference

(Optional) Deploy on AWS SageMaker using the provided model.tar.gz + inference.py

⭐ Key Skills Demonstrated

Predictive Maintenance

Machine Learning (Regression Models)

Time-Series Modeling

Deep Learning (LSTM)

AWS Deployment Architecture

Feature Engineering

Model Evaluation

Data Preprocessing
