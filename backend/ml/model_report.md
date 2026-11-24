# SafePay AI model report

Training data: synthetic (50,000 transactions). Rows: 50,000; fraud: 1,500 (3.00%).

Best model: **Logistic Regression**

| Metric | Value |
|---|---:|
| Precision | 0.9707 |
| Recall | 0.9933 |
| F1 | 0.9819 |
| ROC-AUC | 1.0000 |

Confusion matrix (rows actual [genuine, fraud], columns predicted [genuine, fraud]): `[[9691, 9], [2, 298]]`

Candidate ROC-AUC: `{'Logistic Regression': 0.999985910652921, 'Random Forest': 0.9999742268041236, 'XGBoost': 0.9999711340206185}`
