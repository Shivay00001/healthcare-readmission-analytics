# Healthcare Analytics: Hospital Readmission Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Production-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

Hospital readmissions are a critical metric for healthcare quality. High readmission rates indicate potential gaps in patient care and lead to significant financial penalties for hospitals. This project implements a machine learning solution to predict the likelihood of a diabetic patient being readmitted within 30 days of discharge.

By identifying high-risk patients early, healthcare providers can intervene with targeted care plans, reducing costs and improving patient outcomes.

## 🚀 Key Features

- **Synthetic Clinical Data Generation**: A robust data generator creates realistic patient records, including demographics, admission details, diagnoses (ICD-10 categories), and lab results (HbA1c, Glucose).
- **Comprehensive Preprocessing**: Handling of missing values, categorical encoding (One-Hot), and numerical scaling.
- **Predictive Modeling**: Utilizes **Random Forest Classifier** to predict readmission risk.
- **Model Explainability**: Feature importance analysis to identify top risk factors (e.g., Number of Inpatient Visits, Age, HbA1c levels).

## 🛠️ Tech Stack

## 🛠️ Tech Stack & Tools

This project leverages a comprehensive stack of modern data analysis and engineering tools:

### Core Programming & Libraries

- **Python**: Primary language for data processing and modeling.
- **Scikit-learn**: Machine learning implementation (Random Forest, Logic Regression).
- **Matplotlib & Seaborn**: Advanced data visualization.
- **R Programming**: Used for statistical validation and ancillary analysis.

### Data Visualization & BI

- **PowerBI**: Interactive dashboards for hospital KPIs.
- **Tableau**: Patient risk heatmaps and readmission trends.
- **Qlik Sense**: Associative data exploration.
- **Excel**: Data pivot tables and stakeholder reporting.

### Data Engineering & Big Data

- **SQL**: Data extraction and querying.
- **Apache**: Big data processing framework support.
- **Talend**: ETL pipelines for data ingestion.
- **Google BigQuery**: Data warehousing solution.
- **Splunk**: Log analysis and operational intelligence.

### Analytics & Statistical Software

- **Google Analytics**: Web traffic and patient portal engagement (simulated).
- **SAS**: Advanced statistical analysis for clinical trials.

## 📂 Project Structure

```
healthcare-readmission/
├── data_generator.py    # Generates synthetic healthcare dataset
├── model_pipeline.py    # Preprocessing, Training, and Evaluation pipeline
├── healthcare_dataset.csv # (Generated) The dataset
├── readmission_model.joblib # (Generated) The trained model
├── feature_importance.png # (Generated) Feature importance plot
└── README.md            # Project documentation
```

## 📊 Methodology

1. **Data Simulation**: We simulate 10,000+ patient records with probabilistic distributions based on real-world healthcare statistics.
2. **Feature Engineering**:
    - **Derived Features**: `time_in_hospital`, `num_procedures`, `num_medications`.
    - **Interaction Terms**: Combining `age` and `comorbidities` to capture high-risk groups.
3. **Model Training**:
    - A Random Forest Classifier is trained on 80% of the data.
    - Class imbalance is handled via class weighting.
4. **Evaluation**:
    - **ROC-AUC Score**: Measures the model's ability to distinguish between readmitted and non-readmitted patients.
    - **Confusion Matrix**: Visualizes true positives vs. false positives.

## 📈 Results

*Note: Results vary slightly due to stochastic data generation.*

- **Accuracy**: ~85%
- **ROC-AUC**: ~0.82
- **Key Drivers**:
  - Number of previous inpatient visits.
  - Discharge to Skilled Nursing Facility (SNF).
  - High HbA1c levels (>8%).

## 💻 How to Run

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/healthcare-readmission.git
    cd healthcare-readmission
    ```

2. **Install dependencies**:

    ```bash
    pip install pandas numpy scikit-learn faker matplotlib seaborn
    ```

3. **Generate Data**:

    ```bash
    python data_generator.py
    ```

4. **Train Model**:

    ```bash
    python model_pipeline.py
    ```

5. **View Results**: Check the console output for metrics and open `feature_importance.png` for visualizations.

## 📜 License

This project is licensed under the MIT License.
