# Data Analytics in Healthcare: Predicting Claims Demand and Segmenting Providers in North Carolina

## Overview
This project focuses on leveraging data analytics to predict the demand for healthcare claims and segment healthcare providers in North Carolina. By analyzing historical claims data, we aim to provide actionable insights for healthcare providers and policymakers.

## Objectives
1. Predict the demand of claims for each healthcare provider.
2. Segment health providers and link underlying characteristics to the number of claims.
3. Use the segments and predict the claim for each one.

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributors](#contributors)

## Dataset
- **Source:** CAPSTONE project for IE Business school in collaboration with Accenture.
- **Description:** Historical claims data from North Carolina's health insurance market (Years 2018, 2019, 2020).
- **Preprocessing:** Data cleaning and preprocessing steps to ensure data quality.

## Methodology
### Data Collection and Preprocessing
- Cleaned and preprocessed the data to handle missing values and ensure consistency.

### Exploratory Data Analysis (EDA)
- Conducted EDA to understand data distributions and identify key trends.
- Visualized data using histograms, scatter plots, and other graphical representations.

### Feature Engineering
- Created new features using ratios and interactions to enhance model performance.

### Predictive Modeling
- Selected models: linear regression, elastic net, and boosting.
- Trained and validated models using metrics like RMSE, MAE, MAPE, and R-squared.
- Predicted demand for healthcare claims.

### Provider Segmentation
- Applied clustering techniques (K-means, GMM, Birch, among others) to segment healthcare providers.
- Analyzed segments to link characteristics to the number of claims.

## Results
- **Predictive Modeling:** The final predictions are done using a weighted average of the predictions of the 2 approaches utilized during the development of models.
- **Segmentation Analysis:** Three clusters were detected. Each of them had their own important characteristics related to the composition of claims and geographic information.

## Usage
### Prerequisites
- Python 3.9
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/AxelResnik/CAPSTONE.git
    ```

### Running the Project
1. Examples of how to use the code can be found in the "notebooks" file.

## Project Structure
```
CAPSTONE/
├── notebooks/
│   ├── EDA_notebook/
│   ├── clustering_notebook/
│   ├── prediction_notebook/
├── models/
│   ├── preprocess.py
│   ├── eda.py
│   ├── train_model.py
├── scripts/
│   ├── prediction_preprocess.py
├── README.md
├── .gitignore
```

## Contributors
- [Ariel Ruben Pajuelo Muñoz](https://github.com/pachacutexx)
- [Axel Resnik](https://github.com/axelresnik)
- [Harel Ben David](https://github.com/harelbendavid)
- [Iran Yexalen Benitez Calzada](https://github.com/iran-benitez)
- [Majd Ennaby](https://github.com/majdennaby)
