import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def detect_and_handle_outliers(df):
    print("\n" + "=" * 80)
    print("Outliers Detection and Handling")
    print("=" * 80)

    print(df.shape)
    print(df.head())
    # Select numerical columns for outlier detection (excluding ID and target)
    numerical_cols = df.select_dtypes(include=['float64', 'int64', 'Int64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['CustomerID', 'Churn']]

    # Columns where we will apply IQR capping
    outlier_columns = [col for col in numerical_cols if col in ['Age', 'Balance', 'EstimatedSalary','NumOfProducts']]
    print(f"Analyzing outliers in columns: {outlier_columns}")

    # Method: IQR capping
    print("\n3.1 IQR METHOD WITH CAPPING:")
    print("-" * 35)

    # Copy of original before capping for visual comparison
    df_before_capping = df.copy()

    # Apply IQR capping and visualize
    for col in outlier_columns:
        df, lower, upper = handle_outliers_iqr(df, col)

        plt.figure(figsize=(12, 5))
        plt.suptitle(f"Boxplot After Log Transformation for {col}", fontsize=14)

        if col in ['Balance', 'EstimatedSalary']:
            # Apply log1p transformation to avoid log(0)
            log_before = np.log1p(df_before_capping[col])
            log_after = np.log1p(df[col])

            # Boxplot before
            plt.subplot(1, 2, 1)
            sns.boxplot(y=log_before, color='salmon')
            plt.title(f"log({col}) - Before Capping")

            # Boxplot after
            plt.subplot(1, 2, 2)
            sns.boxplot(y=log_after, color='lightgreen')
            plt.title(f"log({col}) - After Capping")

        else:
            # Boxplot before
            plt.subplot(1, 2, 1)
            sns.boxplot(y=df_before_capping[col], color='salmon')
            plt.title(f"{col} - Before Capping")

            # Boxplot after
            plt.subplot(1, 2, 2)
            sns.boxplot(y=df[col], color='lightgreen')
            plt.title(f"{col} - After Capping")

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

        # Histogram comparison (Zoomed in view)
        plt.figure(figsize=(12, 5))
        plt.suptitle(f"Histogram for {col} Before and After Capping", fontsize=14)

        if col in ['Balance', 'EstimatedSalary']:
            # Use log scale for zoomed-in histograms too
            sns.histplot(np.log1p(df_before_capping[col]), bins=30, color='red', label='Before Capping', kde=False,
                         alpha=0.6)
            sns.histplot(np.log1p(df[col]), bins=30, color='green', label='After Capping', kde=False, alpha=0.6)
            plt.xlabel(f"log({col} + 1)")
        else:
            sns.histplot(df_before_capping[col], bins=30, color='red', label='Before Capping', kde=False, alpha=0.6)
            sns.histplot(df[col], bins=30, color='green', label='After Capping', kde=False, alpha=0.6)
            plt.xlabel(col)

        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Final shape after capping
    print(f"\nDataset shape after IQR capping: {df.shape}")

    # DEBUG: Numerical columns and their min-max
    print("\nDEBUGGING: NUMERICAL COLUMNS & THEIR MIN-MAX VALUES")
    print("=" * 60)
    numerical_cols = df.select_dtypes(include=['float64', 'int64', 'Int64']).columns.tolist()
    print(f"Numerical columns:\n{numerical_cols}\n")

    for col in numerical_cols:
        print(f"{col}: Min = {df[col].min()}, Max = {df[col].max()}")

    print(df.info())

    # Drop top 3 highest EstimatedSalary rows (outlier rows after capping)
    top3_indices = df['EstimatedSalary'].nlargest(3).index
    print("\nTop 3 rows with highest EstimatedSalary:")
    print(df.loc[top3_indices])

    df = df.drop(index=top3_indices).reset_index(drop=True)
    print("\nDropped top 3 EstimatedSalary rows.")
    print("Final shape:", df.shape)

    df.to_csv("data/processed/customer_outliers_handled.csv", index=False)
    print("Saved outlier-handled dataset to: customer_outliers_handled.csv")
    return df


def handle_outliers_iqr(df, column):
    df = df[df['Age'] <= 100]
    df = df[df['NumOfProducts'] != 10]
    df = df[df['NumOfProducts'] != 0]
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column}: {len(outliers)} outliers detected")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

    df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
    return df, lower_bound, upper_bound