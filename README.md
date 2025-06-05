# Lam8da_2025
Lambda team focuses on customer churn as a project for a Telecom company using the dataset to find patterns and make useful predictions for the business to growth
# Telco Customer Churn Analysis

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Exploratory Data Analysis (EDA) & Visualizations](#exploratory-data-analysis-eda--visualizations)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview
Welcome to the **Telco Customer Churn Analysis**, a capstone project by the Lam8da_2025 team! This initiative applies advanced data science techniques to analyze customer churn in a telecommunications dataset. By investigating demographics, service subscriptions, and billing patterns, we aim to identify churn drivers and inform retention strategies. Currently, our work includes data cleaning and exploratory visualizations, with predictive modeling and recommendations planned for future phases.

## Problem Statement
Customer churn represents a critical challenge for telecommunications providers, resulting in lost revenue and elevated acquisition costs. The Telco Customer Churn dataset, comprising 7,043 rows and 21 columns, exhibits class imbalance and intricate relationships among features like tenure, charges, and contract types. Our objective is to pinpoint key factors driving churn—such as high costs or short-term contracts—and deliver actionable insights to enhance customer retention and business stability.

## Features
- **Churn Insights**: Detailed visualizations of churn behavior and trends.
- **Key Variables**: Analysis of tenure, monthly charges, and contract types.
- **Visual Tools**: Professional charts, including histograms, boxplots, and heatmaps.
- **Data Preparation**: Robust cleaning and preprocessing for reliable analysis.
- **Future Scope**: Predictive models and retention strategies in development.

## Installation
Set up the Telco Customer Churn Analysis project locally with these steps:

1. **Clone the Repository**
   - Execute the following command to download the project files:
     ```bash
     git clone https://github.com/myname1sace/Lam8da_2025.git
     ```
2. **Navigate to the Project Directory**
   - Move into the project folder:
     ```bash
     cd Lam8da_2025
     ```
3. **Install Dependencies**
   - Ensure Python 3.7 or higher is installed.
   - Install required packages listed in the repository:
     ```bash
     pip install -r requirements.txt
     ```
   - If `requirements.txt` is unavailable, manually install core libraries:
     ```bash
     pip install pandas seaborn matplotlib missingno
     ```
4. **Set Up Environment Variables**
   - Create a `.env` file in the root directory.
   - Add configuration details, such as the dataset path:
     ```
     DATA_PATH=your_dataset_path/telco_customer_churn.csv
     ```
5. **Run the Application**
   - Launch the analysis script or notebook:
     ```bash
     python main.py
     ```
   - Note: Check the repository for specific scripts (e.g., `preprocess.py`, `eda.ipynb`) or documentation for execution details.

## Usage
- **Data Preparation**: Execute preprocessing scripts to clean the dataset.
- **Exploratory Analysis**: Run EDA code to generate and review visualizations.
- **Example Workflow**:
  - Load the dataset using the path specified in `.env`.
  - Run preprocessing and EDA scripts (e.g., via `python eda.py` or a Jupyter notebook).
  - Examine outputs like churn distribution or contract type analysis.
- **Status**: Current progress includes visualization; modeling and insights to be added in future updates.

## Data Cleaning and Preprocessing
The Telco Customer Churn dataset underwent meticulous cleaning and preprocessing to ensure data quality for analysis and future modeling. Below are the detailed steps:

### 1. Importing the Dataset
- **Action**: Loaded the dataset using the `pandas` library.
- **Code**: `df = pandas.read_csv('telco_customer_churn.csv')`
- **Details**: The dataset includes 7,043 rows and 21 columns, encompassing customer demographics (e.g., age, gender), services (e.g., internet type), account details (e.g., tenure, charges), and churn status.

### 2. Initial Inspection
- **Action**: Examined structure and summary statistics.
- **Code**:
  - `df.info()`: Checked data types and non-null counts.
  - `df.describe()`: Reviewed numerical distributions.
- **Finding**: Identified `TotalCharges` as an object type, requiring conversion to numeric for analysis.

### 3. Data Type Conversion
- **Action**: Converted `TotalCharges` to a numeric format.
- **Code**: `df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')`
- **Result**: Introduced 11 missing values (`NaN`) due to non-numeric entries in the column.

### 4. Handling Missing Values
- **Action**: Analyzed and addressed missing data in `TotalCharges`.
- **Observation**: All 11 rows with missing values had `tenure` equal to 0, indicating new customers.
- **Decision**: As these rows comprised less than 0.2% of the dataset, they were removed.
- **Code**: `df = df.dropna()`
- **Outcome**: Eliminated missing values to ensure data integrity.

### 5. Mapping Binary Values
- **Action**: Standardized the `SeniorCitizen` column for consistency.
- **Observation**: Originally encoded as `0` and `1`.
- **Code**: `df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})`
- **Purpose**: Aligned with other binary categorical variables (e.g., 'Yes'/'No' format) for uniformity.

### 6. Ensuring `TotalCharges` Consistency
- **Action**: Validated and corrected `TotalCharges` values.
- **Code**: 
  - Created a new column: `df['TotalChargesNew'] = df['MonthlyCharges'] * df['tenure']`
- **Observation**: Detected minor discrepancies between original `TotalCharges` and calculated values, likely due to rounding or data entry errors.
- **Decision**: Dropped the original column and renamed the new one.
- **Code**: 
  - `df = df.drop('TotalCharges', axis=1)`
  - `df = df.rename(columns={'TotalChargesNew': 'TotalCharges'})`
- **Outcome**: Ensured consistent, accurate charge data.

### 7. Final Dataset Check
- **Action**: Verified data quality post-processing.
- **Code**:
  - `df.isna().sum()`: Confirmed no missing values remained.
  - `df.info()`: Validated column data types and structure.
- **Result**: Dataset is now clean, properly typed, and ready for analysis and modeling.

- **Outcome**: A refined dataset, free of missing values and inconsistencies, prepared for subsequent steps.

## Exploratory Data Analysis (EDA) & Visualizations
Exploratory Data Analysis (EDA) was conducted to examine the Telco Customer Churn dataset, uncover churn patterns, identify potential predictors, and assess relationships between numerical and categorical features. Visualizations and statistics supported hypothesis development for future modeling.

### 1. Churn Distribution
- **Objective**: Assess the proportion of customers who churned versus those who stayed, and check for class imbalance.
- **Code**:
```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  sns.countplot(data=df, x='Churn', palette='Set2')
  plt.title("Churn Distribution")
  plt.xlabel("Churn")
  plt.ylabel("Count")
  plt.show()
```
- **Interpretation**: The bar chart revealed a class imbalance, with more customers staying than churning. This suggests the need for resampling techniques (e.g., SMOTE, undersampling) or model adjustments to prevent bias in future predictions.
### 2. Customer Tenure Distribution
- **Objective**: Examine customer tenure duration and identify trends in churn timing.
- **Code**:
```python
 sns.histplot(data=df, x='tenure', bins=30, kde=True, color='skyblue')
 plt.title("Distribution of Customer Tenure")
 plt.xlabel("Tenure (months)")
 plt.ylabel("Number of Customers")
 plt.show()
```
- **Interpretation**: The histogram showed a peak at short tenures, especially within the first year, with fewer long-term customers. This indicates early churn is prevalent, emphasizing the need for improved early retention strategies.
### 3. Monthly Charges Distribution
- **Objective**: Analyze the spread of monthly charges and explore pricing’s impact on churn.
- **Code**:
```python
 sns.histplot(df['MonthlyCharges'], kde=True, color='salmon')
 plt.title("Distribution of Monthly Charges")
 plt.xlabel("Monthly Charges")
 plt.ylabel("Frequency")
 plt.show()
```
- **Interpretation**: Charges were right-skewed, with most customers paying $20–$80. A smaller group faced higher costs, possibly for premium services. Elevated bills may contribute to dissatisfaction and churn risk.
### 4. Churn by Contract Type
- **Objective**: Evaluate how contract duration influences churn behavior.
- **Code**:
```python
 sns.countplot(data=df, x='Contract', hue='Churn', palette='Set1')
 plt.title("Churn by Contract Type")
 plt.xlabel("Contract Type")
 plt.ylabel("Count")
 plt.xticks(rotation=45)
 plt.show()
```
- **Interpretation**: Month-to-month contracts showed a significantly higher churn rate than one-year or two-year contracts. Longer commitments, possibly with cancellation fees, bolster retention, suggesting a strategy to promote extended contracts.
### 5. Monthly Charges by Churn
- **Objective**: Compare monthly charges between churned and retained customers to assess pricing’s role.
- **Code**:
```python
 sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='pastel')
 plt.title("Monthly Charges by Churn")
 plt.xlabel("Churn")
 plt.ylabel("Monthly Charges")
 plt.show()
```
- **Interpretation**: Churned customers had higher average charges, supporting the hypothesis that cost drives dissatisfaction. Outliers among churners highlight specific segments needing targeted retention efforts.
### 6. Internet Service vs. Churn
- **Objective**: Investigate the impact of internet service type on churn likelihood.
- **Code**:
```python
 sns.countplot(data=df, x='InternetService', hue='Churn', palette='Set3')
 plt.title("Internet Service Type vs. Churn")
 plt.xlabel("Internet Service")
 plt.ylabel("Count")
 plt.show()
```
- **Interpretation**: Fiber optic users exhibited higher churn than DSL or no-internet customers. This may reflect higher expectations or pricing issues, indicating a need to address service quality or cost for fiber optic subscribers.
### 7. Heatmap of Correlations
- **Objective**: Explore linear relationships between numerical variables and churn-related metrics.
- **Code**:
```python
 corr = df.corr(numeric_only=True)
 plt.figure(figsize=(10, 8))
 sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
 plt.title("Correlation Heatmap")
 plt.show()
```
- **Interpretation**: A strong positive correlation existed between MonthlyCharges and TotalCharges. Tenure showed a moderate positive link to TotalCharges and a weak negative link to churn. Though churn is categorical, trends suggest shorter tenure and higher charges relate to churn risk.
### 8. Missing Data Visualization
- **Objective**: Visually assess missing data to guide cleaning or imputation decisions.
- **Code**:
```python
 import missingno as msno
 msno.matrix(df)
 plt.show()
 msno.heatmap(df)
 plt.show()
```
- **Interpretation**: Missing values were limited to TotalCharges, tied to zero-tenure (new) customers. These were deemed valid and addressed by removal, ensuring minimal impact on analysis.
### 9. Pair Plot of Key Numerical Features
- **Objective**: Visualize relationships and clusters among tenure, charges, and churn.
- **Code**:
```python
 sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], hue='Churn', palette='husl')
 plt.suptitle("Pair Plot of Numerical Features", y=1.02)
 plt.show()
```
- **Interpretation**: Churned customers clustered at lower tenure and moderate-to-high monthly charges. Retained customers spread across longer tenures, reinforcing links between early exits, high costs, and churn.
## Key EDA Findings
Below is a summary of critical insights from our exploratory data analysis of the Telco Customer Churn dataset:

| Feature             | Observation                          | Implication for Churn                     |
|---------------------|--------------------------------------|-------------------------------------------|
| Churn Distribution  | More customers stay than churn        | Class imbalance; may need SMOTE or adjustments |
| Tenure              | Peak at short tenures (<12 months)   | Early churn prevalent; focus on early retention |
| Monthly Charges     | Right-skewed, most pay $20–$80       | High charges may drive dissatisfaction     |
| Contract Type       | Month-to-month has highest churn      | Longer contracts improve retention        |
| Internet Service    | Fiber optic users churn more         | Address service quality or pricing issues  |

### Contributing
We welcome contributions to enhance this project:
1. Fork the repository.
2. Create a branch: git checkout -b feature/your-feature-name
3. Commit changes: git commit -m "Describe your changes"
4. Push to the branch: git push origin feature/your-feature-name
5. Submit a pull request for review. Adhere to PEP 8 standards and include tests for new additions.
### Team
Member 1: [Role, Data Analyst]
Member 2: [Role, Visualization Expert]
Member 3: [Role, Preprocessing Lead]
Member 4: [Role, Project Manager]
Member 5:
Member 6:
### Organization: 
Techcrush Institute[] (Lam8da_2025)
### License
This project is licensed under the MIT License. See the LICENSE file for details.
ContactEmail: [your-group-email@example.com]
### GitHub Issues: 
https://github.com/myname1sace/Lam8da_2025/issues
### Acknowledgments
Gratitude to our capstone advisors for their guidance.Thanks to open-source libraries: pandas, seaborn, matplotlib, missingno.
