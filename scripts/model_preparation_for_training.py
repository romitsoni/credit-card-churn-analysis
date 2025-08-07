import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prepare_data_for_modeling(df=None):
    print("\n" + "=" * 80)
    print("section 7: data preparation for modeling")
    print("=" * 80)

    # Load dataset if not provided
    if df is None:
        df = pd.read_csv("customer_features_enriched.csv")
    # step 1: inspect categorical columns by data type
    print("7.1 feature preparation:")
    print("-" * 30)
    print("object-type columns (typically categorical):")
    print(df.dtypes[df.dtypes == 'object'])

    print("\ncategory-type columns (explicitly categorized):")
    print(df.dtypes[df.dtypes == 'category'])

    #columns for the one hot encoding
    categorical_features = ['Gender', 'AgeBand', 'TenureBand', 'ActivityLevel']

    # step 3: apply one-hot encoding (drop_first=True to avoid dummy variable trap)
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # step 4: print feature count before and after encoding
    print(f"\noriginal number of features: {len(df.columns)}")
    print(f"number of features after one-hot encoding: {len(df_encoded.columns)}")

    # step 5: display new columns created by encoding
    new_columns = [col for col in df_encoded.columns if col not in df.columns and col != 'Churn']
    print("\none-hot encoded columns added:")
    for col in new_columns:
        print(" -", col)

    # step 6: prepare feature matrix x and target vector y
    feature_columns = [col for col in df_encoded.columns if col not in ['CustomerID', 'Churn']]
    X = df_encoded[feature_columns]
    y = df_encoded['Churn']

    # step 7: final shapes
    print(f"\nfinal feature matrix shape (x): {X.shape}")
    print(f"target variable shape (y): {y.shape}")

    numerical_features = X.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()

    # 3. apply MinMaxScaler to numerical features
    scaler = MinMaxScaler()
    X_scaled = X.copy()
    X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

    # 4. print scaled feature names and preview
    print(f"scaled {len(numerical_features)} numerical features using minmaxscaler:")
    print(numerical_features)

    print("\nsample of scaled dataframe (X_scaled):")
    print(X_scaled.head())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train churn rate: {y_train.mean():.3f}")
    print(f"Test churn rate: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test