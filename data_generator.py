import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()
np.random.seed(42)
random.seed(42)

def generate_healthcare_data(n_samples=10000):
    print(f"Generating {n_samples} synthetic patient records...")
    
    # --- Demographics ---
    data = []
    for _ in range(n_samples):
        patient_id = fake.uuid4()
        gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
        age = np.random.randint(18, 90)
        race = np.random.choice(['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'], 
                                p=[0.7, 0.1, 0.05, 0.1, 0.05])
        data.append([patient_id, gender, age, race])
    
    df = pd.DataFrame(data, columns=['patient_id', 'gender', 'age', 'race'])
    
    # --- Admission Details ---
    # Admission types: Emergency, Urgent, Elective
    df['admission_type_id'] = np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples, p=[0.6, 0.2, 0.2])
    
    # Discharge disposition: Home, SNF, Hospice, etc.
    df['discharge_disposition_id'] = np.random.choice(['Home', 'SNF', 'Home with Health', 'Hospice'], n_samples, p=[0.7, 0.15, 0.1, 0.05])
    
    # Admission source: Physician Referral, Emergency Room, etc.
    df['admission_source_id'] = np.random.choice(['Physician Referral', 'Emergency Room', 'Transfer'], n_samples, p=[0.4, 0.5, 0.1])
    
    # Time in hospital (days)
    df['time_in_hospital'] = np.random.poisson(4, n_samples) + 1  # Minimum 1 day
    
    # --- Clinical Features ---
    # Number of lab procedures
    df['num_lab_procedures'] = np.random.normal(40, 15, n_samples).astype(int)
    df['num_lab_procedures'] = df['num_lab_procedures'].clip(lower=1)
    
    # Number of procedures
    df['num_procedures'] = np.random.randint(0, 7, n_samples)
    
    # Number of medications
    df['num_medications'] = np.random.randint(1, 40, n_samples)
    
    # Number of diagnoses (ICD-9/10 simulated count)
    df['number_diagnoses'] = np.random.randint(1, 10, n_samples)
    
    # --- Diagnostic Features (Simulated) ---
    # Diagnosis 1, 2, 3 (Simulate categories like 'Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury')
    diag_categories = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
    df['diag_1'] = np.random.choice(diag_categories, n_samples)
    df['diag_2'] = np.random.choice(diag_categories, n_samples)
    df['diag_3'] = np.random.choice(diag_categories, n_samples)
    
    # --- Diabetes Specifics ---
    # Glucose Serum test result
    df['max_glu_serum'] = np.random.choice(['None', 'Norm', '>200', '>300'], n_samples, p=[0.9, 0.05, 0.03, 0.02])
    
    # A1C test result
    df['A1Cresult'] = np.random.choice(['None', 'Norm', '>7', '>8'], n_samples, p=[0.8, 0.1, 0.05, 0.05])
    
    # Medication changes
    df['change'] = np.random.choice(['No', 'Ch'], n_samples, p=[0.6, 0.4])
    
    # Diabetes meds
    df['diabetesMed'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3])
    
    # --- Historical Data ---
    # Number of outpatient vists in preceding year
    df['number_outpatient'] = np.random.poisson(0.5, n_samples)
    
    # Number of emergency visits in preceding year
    df['number_emergency'] = np.random.poisson(0.2, n_samples)
    
    # Number of inpatient visits in preceding year
    df['number_inpatient'] = np.random.poisson(0.3, n_samples)
    
    # --- Target Variable Generation (Logic-based for realism) ---
    # Calculate a risk score to determine readmission probability
    # Base probability
    prob = 0.1 
    
    # Risk factors
    prob += np.where(df['age'] > 70, 0.15, 0)
    prob += np.where(df['age'] < 30, -0.05, 0)
    prob += np.where(df['number_inpatient'] > 0, 0.2, 0)
    prob += np.where(df['number_emergency'] > 0, 0.1, 0)
    prob += np.where(df['discharge_disposition_id'] == 'SNF', 0.1, 0) # Discharged to Skilled Nursing Facility
    prob += np.where(df['A1Cresult'].isin(['>7', '>8']), 0.1, 0)
    prob += np.where(df['diag_1'] == 'Diabetes', 0.05, 0)
    prob += np.where(df['time_in_hospital'] > 7, 0.05, 0)
    
    # Interactions
    prob += np.where((df['age'] > 60) & (df['num_medications'] > 20), 0.1, 0)
    
    # Clamp probability
    prob = np.clip(prob, 0, 0.8)
    
    # Assign target based on probability
    df['readmitted'] = np.random.binomial(1, prob)
    # Convert manually to binary/string if needed, but keeping as int 0/1 is good for modeling
    
    # --- Post-processing ---
    # Add some noise/randomness
    noise_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[noise_indices, 'readmitted'] = 1 - df.loc[noise_indices, 'readmitted']

    print(f"Data generation complete. Class distribution: \n{df['readmitted'].value_counts(normalize=True)}")
    
    return df

if __name__ == "__main__":
    df = generate_healthcare_data(15000)
    output_file = "healthcare_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
