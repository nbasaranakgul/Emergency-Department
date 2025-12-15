# Emergency Department Readmission Prediction - Part 1: Data Preprocessing

## Overview

This notebook represents the first part of a comprehensive machine learning project focused on predicting Emergency Department (ED) readmissions within 30 days. This preprocessing stage prepares the data for subsequent logistic regression modeling by performing extensive data cleaning, feature engineering, and transformation operations.

## Project Context

The goal of this project is to predict the risk of patients returning to the Emergency Department after discharge. While this focuses on ED visits rather than traditional inpatient admissions, the principles and prediction procedures are similar to standard readmission prediction workflows.

## Dataset

- **Source**: Kaggle dataset - Epic Sample
- **Size**: 2,001 patient visits × 21 variables
- **Target Variable**: `RETURN` (0 = no readmission, 1 = readmission within 30 days)

### Key Variables

**Demographics:**
- Age, Sex, Race, Ethnicity

**Visit Information:**
- ED arrival and departure information
- Hospital name
- Arrival Emergency Severity Index (ESI)
- ED disposition
- Length of stay (arrival to departure time in minutes)

**Clinical & Financial:**
- Risk of mortality
- Severity of illness
- Insurance/financial class
- Visit charges

## Preprocessing Steps

### 1. Environment Setup
- Google Colab environment configuration
- Package imports: pandas, numpy, seaborn, matplotlib, missingno, scikit-learn
- Dataset download from Kaggle using API

### 2. Data Understanding
- Initial data exploration and structure review
- Descriptive statistics and data type assessment
- Identification of data fields and their meanings

### 3. Data Quality Assessment
- Missing value analysis using visualization techniques
- Distribution analysis of numeric variables
- Categorical variable frequency analysis

### 4. Data Cleaning

**Missing Value Handling:**
- Removed records with missing target variable (`RETURN`)
- Dropped columns with excessive missing values (Risk of Mortality, Severity of Illness indicators)
- Labeled missing values in key categorical variables as "Missing"

**Category Consolidation:**
- **Insurance/Financial Class**: Grouped rare categories (Medicare Replacement Plan, Military, Worker's Comp, Out of State Medicaid) into "Other"
- **ED Disposition**: Consolidated 16 categories into 7 meaningful groups:
  - Admit (includes psychiatric admissions and L&D transfers)
  - Discharge
  - Observation
  - Left Early (includes AMA and incomplete treatments)
  - LWBS after Triage / LWBS before Triage
  - Other (includes deceased, transfers, errors, elopement)

### 5. Feature Engineering

**Temporal Feature Extraction:**
- Separated date and time components from arrival and discharge timestamps
- Created season variables (Winter, Spring, Summer, Fall) for arrival and discharge
- Generated time-of-day categories:
  - Night-AM (00:00-07:00)
  - Morning (07:01-12:00)
  - Afternoon (12:01-16:00)
  - Evening (16:01-19:00)
  - Night-PM (19:01-23:59)

**Data Reduction:**
- Removed redundant columns (INDEX, PAT_ENC_CSN_ID, original timestamp fields)

### 6. Data Transformation

**Normalization:**
- Applied MinMaxScaler to continuous variables (AGE, ARRIVE_DEPART_MIN, CHARGES)
- Scaled values to range [0, 1] to ensure comparable scales with binary features

**Encoding:**
- Created dummy variables for all categorical features
- Generated binary (0/1) columns for each category
- Maintained interpretability for logistic regression modeling

### 7. Output Generation

Two clean datasets were produced:

1. **Epic-clean.csv**: Processed data with categorical features intact
2. **Epic-dummy.csv**: Fully processed data with dummy variables, ready for modeling

## Technical Requirements

```python
pandas
numpy
seaborn
matplotlib
missingno
scikit-learn
```

## Usage

This notebook is designed to run in Google Colab:

1. Mount Google Drive for data storage
2. Configure Kaggle API credentials
3. Run cells sequentially from top to bottom
4. Download the output CSV files for use in modeling

## Key Insights from Exploratory Analysis

- Patients who return within 30 days tend to be older (average age ~53 vs. ~44)
- Readmitted patients have significantly higher charges (~$10,000 vs. ~$3,000)
- The dataset has a class imbalance with fewer readmission cases
- Certain disposition types and insurance classes show different readmission patterns

## Data Preprocessing Decisions

- **Conservative missing value approach**: Only dropped variables with extensive missingness; preserved records with partial information
- **Clinical relevance in categorization**: Grouped similar clinical outcomes (e.g., different admission types)
- **Temporal granularity**: Created both seasonal and time-of-day features to capture potential cyclical patterns
- **Scaling strategy**: Used MinMaxScaler to maintain interpretability while ensuring feature comparability

## Next Steps

The cleaned and preprocessed data from this notebook will be used in:
- **Part 2**: Logistic regression model development
- **Part 3**: Model evaluation and validation
- **Part 4**: Feature importance analysis and clinical interpretation

## File Structure

```
├── Epic_Sample.csv          # Raw data (from Kaggle)
├── Epic-clean.csv           # Cleaned data with categories
├── Epic-dummy.csv           # Model-ready data with dummy variables
└── Data_Prep_Notebook.ipynb # This preprocessing notebook
```

## Notes

- The target variable `RETURN` indicates ED readmission within 30 days (not traditional inpatient readmission)
- All datetime processing preserves both date and time information for temporal analysis
- Dummy variable creation uses pandas' get_dummies() with appropriate naming conventions

## Author

Part of a healthcare analytics and machine learning project for Emergency Department readmission prediction.

## License

Dataset is provided through Kaggle under their terms of use.
