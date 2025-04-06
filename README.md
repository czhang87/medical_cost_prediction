Here's a sample `README.md` for the project available at the provided GitHub link:

---

# Medical Cost Prediction

This project aims to predict medical insurance costs based on various demographic and health-related features using machine learning models. The dataset used for this analysis includes columns such as age, sex, BMI, number of children, smoking status, and region.

The goal of the project is to build and evaluate models that can predict the medical insurance charges for individuals.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Project Overview

This repository contains a Jupyter notebook (`medical_cost_prediction.ipynb`) that walks through the following key steps:

1. **Exploratory Data Analysis (EDA)**: Visualization of data distributions and relationships between the features and target variable (`charges`).

2. **Data Preprocessing**: Splitting the dataset into training and testing sets, encoding categorical features, and selecting the best features.

3. **Model Training**: Using TPOT for AutoML that automates the pipelines of random forest and gradient boosting and trains models for predicting medical insurance charges. 

4. **Model Evaluation**: Evaluating models using appropriate metrics like mean absolute error (MAE), root mean square error (RMSE), and R-squared to determine the best-performing model.

5. **MLflow**: Using the MLflow to autolog the experiments and register models

## Dataset

The dataset used for this project is based on medical insurance costs data. It contains the following columns:

- **age**: Age of the individual
- **sex**: Gender of the individual (male/female)
- **bmi**: Body mass index (BMI) of the individual
- **children**: Number of children/dependents covered by the insurance
- **smoker**: Whether the individual smokes or not (yes/no)
- **region**: Geographic region of the individual (northeast, southeast, southwest, northwest)
- **charges**: The medical insurance charges for the individual (target variable)

You can find more about the dataset [here](https://www.kaggle.com/datasets/mirichoi0218/insurance).

## Installation

To run this project on your local machine, you'll need the following dependencies:

- Python 3.10.4
- Jupyter Notebook
- Libraries: `tpot`, `mlflow`, `torch`, `numpy`, `seaborn`

You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/czhang87/medical_cost_prediction.git
cd medical_cost_prediction
```

2. Open the `medical_cost_prediction.ipynb` Jupyter notebook:

```bash
jupyter notebook medical_cost_prediction.ipynb
```

3. Run each cell in the notebook to explore the dataset, preprocess the data, train the models, and evaluate their performance.

## Model Evaluation

In this notebook, two models are evaluated for predicting the target variable (`charges`), including:

- Random Forest Regressor
- Gradient Boosting Regressor

For each model, we evaluate performance using:

- **Mean Squared Error (MSE)**
- **Root Mean Square Error (RMSE)**
- **R-squared (R²)**

This helps in understanding the model's ability to explain the variance in the target variable and its prediction accuracy.

## License

This project is licensed under the MIT License.

---
