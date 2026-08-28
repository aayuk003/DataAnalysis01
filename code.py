# ==========================================
# DATA PREPROCESSING PIPELINE
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# 1. LOAD DATASET
df = pd.read_csv("dataset.csv")

print("Original Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# 2. REMOVE DUPLICATES
df = df.drop_duplicates()


# 3. HANDLE MISSING VALUES
# Numerical columns -> Median
numeric_cols = df.select_dtypes(include=np.number).columns

num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])


# Categorical columns -> Most frequent value
categorical_cols = df.select_dtypes(include="object").columns

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])


# 4. ENCODE CATEGORICAL DATA
# Convert categorical columns into numerical values

encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))


# 5. SEPARATE FEATURES AND TARGET
# CHANGE THIS COLUMN NAME according to your dataset

TARGET_COLUMN = "target"

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]


# 6. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 7. FEATURE SCALING
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 8. FINAL SHAPE
print("\nPreprocessing Complete!")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# 9. CHECK FOR MISSING VALUES
print("\nMissing values:")
print(pd.DataFrame(X_train).isnull().sum().sum())


# 10. SAVE PROCESSED DATA
train_data = pd.DataFrame(X_train)
train_data["target"] = y_train.reset_index(drop=True)

test_data = pd.DataFrame(X_test)
test_data["target"] = y_test.reset_index(drop=True)

train_data.to_csv("processed_train.csv", index=False)
test_data.to_csv("processed_test.csv", index=False)

print("\nSaved:")
print("processed_train.csv")
print("processed_test.csv")
