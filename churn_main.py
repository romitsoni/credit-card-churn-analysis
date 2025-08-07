import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


# configuring display options
from scripts.config import set_display_options
set_display_options()



# Step 1 Loading the dataset
from scripts.data_loader import load_data
df = load_data("data/raw/exl_credit_card_churn_data.csv")



#step 2 cleaning , handling missing values, visualization
from scripts.data_cleaner import (
    explore_data,
    handle_missing_values,
    identify_and_impute_missing_values,
    visualize_missing_values_before,
    visualize_missing_values_after
)

explore_data(df, "eda_summary.md")
visualize_missing_values_before(df)
df_cleaned = handle_missing_values(df)
df_final = identify_and_impute_missing_values(df_cleaned)
visualize_missing_values_after(df_final)
df = df_final



#Step 3 Outliers detection and handling
from scripts.data_outliers_removal import detect_and_handle_outliers
df = detect_and_handle_outliers(df)


#Step 4 Feature Engineering
from scripts.feature_engineering import create_features
df = create_features(df)



#Step 5 EDA Analysis
from scripts.eda_analysis import perform_eda, generate_profiling_reports, analyze_engineered_features
df = perform_eda(df)
df = analyze_engineered_features(df)
generate_profiling_reports()


# Step 6 Model preparation(data preparation)
from scripts.model_preparation_for_training import prepare_data_for_modeling
X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)

#Step 7 Model Training and evaluation
from scripts.model_training_evaluation import train_and_evaluate_models
results = train_and_evaluate_models(X_train, X_test, y_train, y_test)



# Step 8 Model comparison and insights
from scripts.model_comparison import compare_models_and_insights
best_model_name, best_model = compare_models_and_insights(results, df)



