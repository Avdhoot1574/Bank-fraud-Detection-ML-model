# Bank Fraud Detection Model
Project By : Avdhoot Nakod<br>
Linkedin Profile : https://www.linkedin.com/in/avdhoot-nakod-b869ba268/ <br>
## Overview
This project implements a **Bank Fraud Detection Model** using **Machine Learning** techniques. The model is built using Python and essential libraries, and it predicts fraudulent transactions based on multiple features in the dataset.

---

## Dataset
The dataset consists of 100 sample entries with the following features:
1. **TransactionID** - Unique identifier for transactions.
2. **Amount** - Transaction amount.
3. **TransactionType** - Type of transaction (Online, POS, ATM, Wire Transfer).
4. **Location** - Location where the transaction occurred.
5. **Time** - Time of transaction (Morning, Afternoon, Evening, Night).
6. **AccountAge** - Age of the account in days.
7. **IsInternational** - Indicates if the transaction is international (1 or 0).
8. **IsHighRiskCountry** - Indicates if the country is high risk (1 or 0).
9. **PreviousFraudHistory** - Previous fraud history (1 or 0).
10. **FraudDetected** - Target label (1 for fraud, 0 for non-fraud).

---

## Requirements
- Python 3.8 or later
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - joblib

Install dependencies using:
```
pip install pandas numpy scikit-learn joblib
```

---

## Model Implementation
The model uses **RandomForestClassifier** for classification. It performs the following steps:
1. Loads and preprocesses the dataset.
2. Encodes categorical features.
3. Splits data into training and testing sets.
4. Trains a **Random Forest Classifier**.
5. Evaluates the model using accuracy, confusion matrix, and classification report.
6. Saves the model using **joblib**.

---

## Running the Model
1. Ensure the dataset (bank_fraud_sample.csv) is in the project directory.
2. Run the Python script:
```
python bank_fraud_model.py
```
3. View the evaluation metrics in the terminal.

---

## Output
- **Accuracy**, **Confusion Matrix**, and **Classification Report** are displayed.
- The trained model is saved as `bank_fraud_model.pkl`.

---

## Future Enhancements
- Feature engineering for better performance.
- Implementation of anomaly detection techniques.
- Integration with web applications for real-time fraud detection.

---

## License
This project is licensed under the **MIT License**.

---

## Author
Avdhoot

