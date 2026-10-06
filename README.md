# Telecom Customer Churn Intelligence Dashboard

A business-intelligence project that turns raw telecom customer data into decisions:
**Data → Insight → Decision → Action.** It shows who is leaving, why, how much revenue is at risk, and what management should do about it.

## Problem Statement
A telecom company loses about 1 in 4 customers. Leadership needs to know which customers churn, what drives it, which active customers are most likely to leave next, and which retention actions to prioritise.

## Dataset
- **Name:** Telco Customer Churn (IBM sample data), 7,043 customers, 21 columns
- **Source (Kaggle):** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Public mirror used as automatic fallback:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Target column: `Churn` (Yes/No)

## Tech Stack
| Layer | Tools |
|---|---|
| Backend / ML | Python, pandas, NumPy, scikit-learn (Gradient Boosting classifier) |
| Frontend | Streamlit, Plotly |

## Dashboard Pages
1. **Executive Overview**: KPIs (customers, churn rate, monthly revenue, revenue lost, ARPU) and churn trends by tenure and contract.
2. **Churn Drivers & Segments**: model feature importance, segment explorer, contract × internet heatmap, model performance.
3. **Risk, Opportunity & Action**: at-risk active customers, top-15 contact list, recommended actions, and a single-customer what-if scorer.

## Model Performance (20% hold-out test set)
Accuracy 80.3% · ROC-AUC 0.843 · F1 (churn) 0.581

## Key Findings
- Overall churn rate is 26.5%.
- Month-to-month contracts churn at 42.7% vs 2.8% on two-year contracts.
- Customers in their first 12 months churn at 47.4%.
- Electronic-check payers churn at 45.3%; fiber-optic customers at 41.9%.

## How to Run
```bash
pip install -r requirements.txt
streamlit run SnehaTanty_ChurnDashboard.py
```
The app loads `Telco-Customer-Churn.csv` from the same folder if present; otherwise it downloads the dataset from the public mirror above. You can also upload a CSV with the same columns from the sidebar.

## Files
- `SnehaTanty_ChurnDashboard.py`: complete application (data, model, dashboard in one file)
- `requirements.txt`: dependencies
- `README.md`: this file
- Project report (Word/PDF): documentation with outputs
