import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    r2_score,
    mean_squared_error
)

# ------------------------------------------------
# 1. Reload the original cleaned dataset
# ------------------------------------------------

df_final = pd.read_csv("/customer_churn_clean (1).csv")

# Convert TotalCharges to numeric
df_final["TotalCharges"] = pd.to_numeric(
    df_final["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
df_final = df_final.dropna().copy()

# IMPORTANT:
# Customer ID should NOT be used as a predictor
df_final = df_final.drop(columns=["customerID"])

# Make sure gender is treated as categorical
df_final["gender"] = df_final["gender"].astype(str)

print("Final dataset shape:", df_final.shape)
print("\nColumns:")
print(df_final.columns.tolist())

# ------------------------------------------------
# 2. Encode categorical variables
# ------------------------------------------------

df_final["Churn"] = df_final["Churn"].map({
    "Yes": 1,
    "No": 0
})

categorical_cols_final = df_final.select_dtypes(
    include="object"
).columns.tolist()

encoders_final = {}

df_encoded_final = df_final.copy()

for col in categorical_cols_final:
    le = LabelEncoder()
    df_encoded_final[col] = le.fit_transform(
        df_encoded_final[col]
    )
    encoders_final[col] = le

print("\nCategorical columns:")
print(categorical_cols_final)

# ------------------------------------------------
# 3. LOGISTIC REGRESSION
# ------------------------------------------------

features_clf_final = [
    c for c in df_encoded_final.columns
    if c != "Churn"
]

X = df_encoded_final[features_clf_final]
y = df_encoded_final["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

clf_model = LogisticRegression(
    max_iter=2000
)

clf_model.fit(X_train, y_train)

y_pred = clf_model.predict(X_test)
y_prob = clf_model.predict_proba(X_test)[:, 1]

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------
# 4. Logistic regression coefficient table
# ------------------------------------------------

coef_table_final = pd.DataFrame({
    "feature": features_clf_final,
    "coefficient": clf_model.coef_[0]
})

coef_table_final["absolute_coefficient"] = (
    coef_table_final["coefficient"].abs()
)

coef_table_final = coef_table_final.sort_values(
    "absolute_coefficient",
    ascending=False
)

print("\nTop churn drivers:")
display(coef_table_final.head(10))

# ------------------------------------------------
# 5. LINEAR REGRESSION
# Predict Monthly Charges
# ------------------------------------------------

features_reg_final = [
    c for c in df_encoded_final.columns
    if c not in ["MonthlyCharges", "Churn"]
]

Xr = df_encoded_final[features_reg_final]
yr = df_encoded_final["MonthlyCharges"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr,
    yr,
    test_size=0.20,
    random_state=42
)

reg_model = LinearRegression()

reg_model.fit(Xr_train, yr_train)

yr_pred = reg_model.predict(Xr_test)

r2_final = r2_score(yr_test, yr_pred)

rmse_final = np.sqrt(
    mean_squared_error(yr_test, yr_pred)
)

mae_final = np.mean(
    np.abs(yr_test - yr_pred)
)

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

print("R²   :", round(r2_final, 4))
print("RMSE :", round(rmse_final, 4))
print("MAE  :", round(mae_final, 4))

# ------------------------------------------------
# 6. SAVE EVERYTHING FOR STREAMLIT
# ------------------------------------------------

bundle_final = {
    "clf_model": clf_model,
    "reg_model": reg_model,
    "encoders": encoders_final,
    "features_clf": features_clf_final,
    "features_reg": features_reg_final,
    "categorical_cols": categorical_cols_final,
    "coef_table": coef_table_final
}

joblib.dump(
    bundle_final,
    "churn_model.pkl"
)

print("\n================================")
print("SUCCESS: churn_model.pkl created")
print("================================")
