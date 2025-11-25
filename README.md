# TrustGate.AI
Gatekeeper for model quality, production guardrails, trust, bias, risk, and anomaly detection, 
  .
 README. Scanned on 2025-11-06 18:08._

## Overview
# Case Study : Customer Loan Purchase Prediction

This case study will help us to understand the stages in the AI/ML project lifecycle with loans data set to predict whether potential customer's will be targeted for loans. We will focus on the following stages namely -

- Preprocess the data.
- Handle missing values.
- Perform feature engineering.
- Machine learning classification model.
- Build and evaluate classification models.
- Provide insights based on the model's performance.

#1. Import Libraries/Dataset

  a.  Download the dataset.

  b.  Import the required libraries.

## Project Structure
```
+---.idea
�   �   .gitignore
�   �   AIML.iml
�   �   misc.xml
�   �   modules.xml
�   �   workspace.xml
�   �   
�   +---inspectionProfiles
�           profiles_settings.xml
�           
+---Loan Pediction Modal Evaluation
�   �   Customer_Financial_Info.csv
�   �   Loan Approval Prediction.py
�   �   loan_approval_best_model.joblib
�   �   Loan_Approval_test.csv
�   �   Loan_Approval_train.csv
�   �   model_comparison_summary.csv
�   �   model_test.py
�   �   model_validation.py
�   �   
�   +---__pycache__
�           model_validation.cpython-39.pyc
�           
+---Reinforcement Learning
    �   dino_run.py
    �   
    +---aigym_env
            setup.py

```

## Data
ata/file paths:

- `/content/Customer_Financial_Info.csv`

## Environment & Requirements
Install dependencies (adjust versions as needed):
```bash
pip install matplotlib numpy pandas scikit-learn seaborn
```

## How to Run
1. Run  Loan Approval Prediction.py
   - `loan_approval_best_model.joblib file generated.`
2. Run model_validation.
3. Run the model_test

## Modeling (auto-detected)
- Algorithms referenced: DecisionTreeClassifier, KNeighborsClassifier, LogisticRegression, RandomForestClassifier
- Metrics computed: classification_report, roc_auc_score
- Target variable: _Not detected from the notebook code_
- Random state(s): 42

## Notes
- For production use, consider promoting and consuming it.
