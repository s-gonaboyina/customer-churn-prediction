# Customer Churn Prediction

Logistic regression model that predicts which telecom customers are likely to churn, built for IN503 (Introduction to Machine Learning), Unit 10.

## Project description

The Telco Customer Churn dataset has 7,043 customers and 21 columns (demographics, services, account details, and the target `Churn`). The notebook covers the full workflow: exploratory analysis, cleaning (including the hidden blank values in `TotalCharges`), a leakage-safe `ColumnTransformer` + `Pipeline`, one engineered feature (`total_services`), `GridSearchCV` tuning of the regularization parameter `C`, evaluation (accuracy, precision, recall, F1, confusion matrix, ROC/AUC), coefficient interpretation, and a saved model with a `predict_churn()` function.

## Files

- `Gonaboyina_Sesha_IN503_Unit10_Assignment.ipynb`: the full notebook
- `customer_churn_model.pkl`: trained pipeline (created by running the notebook, section 6.1)
- `requirements.txt`: dependencies

## How to run

### Google Colab (recommended)

1. Upload the notebook to Colab (File > Upload notebook).
2. Run all cells (Runtime > Run all). The first cell installs `kagglehub`, which downloads `blastchar/telco-customer-churn`.
3. If the download fails (some networks block it), download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), upload it to the Colab session files, and run again. The notebook falls back to that local file automatically.
4. Download `customer_churn_model.pkl` from the Colab file panel and add it to the repository.

### Local

```bash
pip install -r requirements.txt
jupyter notebook Gonaboyina_Sesha_IN503_Unit10_Assignment.ipynb
```

### Using the saved model

```python
import joblib, pandas as pd
model = joblib.load('customer_churn_model.pkl')
customer = pd.DataFrame([{ ... 19 raw columns, see notebook section 6.3 ... }])
# Add the total_services column first (add_total_services() in the notebook), then:
model.predict_proba(customer)[:, 1]
```

The `predict_churn()` function in the notebook does the feature engineering, validation, and thresholding for you.

## Key findings summary

- **Data quality:** `TotalCharges` was stored as text because 11 new customers (tenure 0) had blank values. They were converted to NaN and dropped, leaving 7,032 rows.
- **Class imbalance:** about 73% stayed and 27% churned, so accuracy is compared with an always-"No" baseline and recall/F1 for the churn class are reported.
- **Contract type:** month-to-month customers churn most.
- **Internet service:** fiber optic customers churn more than DSL or no-internet customers.
- **Tenure:** new customers are the highest risk.
- **Pricing:** higher monthly charges go along with higher churn.
- **Model results:** Best C = 100. On the test set: accuracy 0.80, precision 0.63, recall 0.56, F1 0.60 (churn class), ROC AUC 0.83. Top five drivers: MonthlyCharges, tenure, InternetService_Fiber optic, InternetService_DSL, total_services.

## Recommendations

Move month-to-month customers to longer contracts, run an onboarding program for the first 3 to 6 months, review fiber pricing and quality, promote add-on services, and use churn probability (with a tuned threshold) to prioritize retention outreach.

## Dependencies

Python 3.9+, pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, kagglehub. See `requirements.txt`.

## Deployment idea

Serve the pickled pipeline from a FastAPI endpoint in a Docker container (pin the scikit-learn version), score customers nightly in batch for the CRM, and monitor drift and actual churn to decide when to retrain.
