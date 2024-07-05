# Data Analytics in Healthcare: Predicting Claims Demand and Segmenting Providers in North Carolina

## Overview
This project focuses on leveraging data analytics to predict the demand for healthcare claims and segment healthcare providers in North Carolina. By analyzing historical claims data, we aim to provide actionable insights for healthcare providers and policymakers.

## Objectives
1. Predict the demand of claims for each healthcare provider.
2. Segment health providers and link underlying characteristics to the number of claims.

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

## Dataset
- **Source:** CAPTONE project for ie Business school in collaboration with Accenture.
- **Description:** Historical claims data from North Carolina's health insurance market (Years 2018, 2019, 2020).
- **Preprocessing:** Data cleaning and preprocessing steps to ensure data quality.

## Methodology
### Data Collection and Preprocessing
- Collected data from [source].
- Cleaned and preprocessed the data to handle missing values and ensure consistency.

### Exploratory Data Analysis (EDA)
- Conducted EDA to understand data distributions and identify key trends.
- Visualized data using histograms, scatter plots, and other graphical representations.

### Feature Engineering
- Created new features using ratios and interactions to enhance model performance.

### Predictive Modeling
- Selected models: linear regression, decision trees, etc.
- Trained and validated models using metrics like RMSE and R-squared.
- Predicted demand for healthcare claims.

### Provider Segmentation
- Applied clustering techniques (e.g., K-means) to segment healthcare providers.
- Analyzed segments to link characteristics to the number of claims.

## Results
- **Predictive Modeling:** Key findings and model performance metrics.
- **Segmentation Analysis:** Characteristics of identified segments and their relation to claim numbers.

## Usage
### Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/AxelResnik/CAPSTONE.git
    ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project
1. Preprocess the data:
    ```bash
    python preprocess.py
    ```
2. Perform exploratory data analysis:
    ```bash
    python eda.py
    ```
3. Train predictive models:
    ```bash
    python prediction_benchmarks.py
    ```
4. Segment healthcare providers:
    ```bash
    python segment_providers.py
    ```

## Project Structure
```
healthcare-claims-prediction/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── eda.py
│   ├── train_model.py
│   ├── segment_providers.py
├── results/
│   ├── figures/
│   ├── tables/
├── README.md
├── requirements.txt
```

## Contributors
- [Ariel Ruben Pajuelo Muñoz](https://github.com/pachacutexx)
- [Axel Resnik](https://github.com/axelresnik)
- [Harel Ben David](https://github.com/harelbendavid)
- [Iran Yexalen Benitez Calzada](https://github.com/iran-benitez)
- [Majd Ennaby](https://github.com/majdennaby)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
