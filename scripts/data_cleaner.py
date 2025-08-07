import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def explore_data(df, output_file="eda_summary.md"):
    with open(output_file, "w") as f:
        # Instead of print(), use file.write()
        f.write("1. Original dataset\n")
        f.write("-" * 50 + "\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Columns: {list(df.columns)}\n\n")

        f.write("Database info:\n")
        df.info(buf=f)  # info() accepts a buffer argument

        f.write("\n\n2. First 10 records of the dataset\n")
        f.write("-" * 50 + "\n")
        f.write(df.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("3. Missing values analysis\n")
        f.write("-" * 50 + "\n")
        missing_values = df.isnull().sum()
        f.write("Missing values per column:\n")
        for col, missing in missing_values.items():
            if missing > 0:
                percent = (missing / len(df)) * 100
                f.write(f"{col}: {missing} ({percent:.2f}%)\n")

        f.write(f"\nTotal missing values: {df.isnull().sum().sum()}\n")


def handle_missing_values(df):
    # Make a copy to avoid modifying original
    df = df.copy()

    # SECTION 2: Data Cleaning and the Missing value imputation

    print("\n" + "=" * 80)
    print("SECTION 2: DATA CLEANING AND MISSING VALUE IMPUTATION")
    print("=" * 80)
    print("Checking for string 'nan' values in each column...\n")
    for col in df.columns:
        str_col = df[col].astype(str).str.lower().str.strip()
        if 'nan' in str_col.unique():
            print(f"Column '{col}' contains string 'nan' values.")

    # Reset index just in case
    df = df.reset_index(drop=True)
    # handling inconsistencies in the data

    print("\n[Before] Unique values in Gender column:")
    print(df['Gender'].unique())
    print("Missing values in Gender column:", df['Gender'].isnull().sum())

    df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
    df['Gender'] = df['Gender'].replace('nan', np.nan)
    print("Missing values in Gender column:", df['Gender'].isnull().sum())

    print("[After] Unique values in Gender column:")
    print(df['Gender'].unique())

    print("\nNumber of rows with negative Age:", (df['Age'] < 0).sum())
    print("Number of rows with negative EstimatedSalary:", (df['EstimatedSalary'] < 0).sum())

    df.loc[df['EstimatedSalary'] < 0, 'EstimatedSalary'] = np.nan
    df.loc[df['Age'] < 0, 'Age'] = np.nan
    df['Age'] = df['Age'].astype('Int64')

    print("\n[Before] Total rows:", df.shape[0])
    print("Duplicate rows count:", df.duplicated().sum())

    # Check how many duplicate rows are present
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Confirm removal
    print(f"New shape after removing duplicates: {df.shape}")

    print("\nNumber of rows with negative Balance:", (df['Balance'] < 0).sum())

    # replacing negative balance to Nan values
    df.loc[df['Balance'] < 0, 'Balance'] = np.nan

    # ====== FIXED HASCRCARD CLEANING ======
    print("\n[Before] Unique values in HasCrCard column:")
    print(df['HasCrCard'].unique())

    # Convert to lowercase strings first
    df['HasCrCard'] = df['HasCrCard'].astype(str).str.lower().str.strip()
    print(f"After string conversion: {df['HasCrCard'].unique()}")

    # Mapping including all variations
    hascrcard_mapping = {
        '1': 1, '1.0': 1,
        '0': 0, '0.0': 0,
        'yes': 1,
        'nan': np.nan
    }

    df['HasCrCard'] = df['HasCrCard'].map(hascrcard_mapping)
    df['HasCrCard'] = df['HasCrCard'].astype('Int64')

    print("[After] Unique values in HasCrCard column:")
    print(df['HasCrCard'].unique())
    print(f"HasCrCard value counts: {df['HasCrCard'].value_counts(dropna=False)}")

    #isActiveMember Cleaning
    print("\n[Before] Unique values in IsActiveMember column:")
    print(df['IsActiveMember'].unique())

    # Convert to string and lowercase first
    df['IsActiveMember'] = df['IsActiveMember'].astype(str).str.lower().str.strip()
    print(f"After string conversion: {df['IsActiveMember'].unique()}")

    # Create a mapping for possible values
    activemember_mapping = {
        '1': 1, '1.0': 1,
        '0': 0, '0.0': 0,
        'no': 0, '-1': 0, '-1.0': 0,
        'nan': np.nan
    }

    df['IsActiveMember'] = df['IsActiveMember'].map(activemember_mapping)
    df['IsActiveMember'] = df['IsActiveMember'].astype('Int64')

    print("[After] Unique values in IsActiveMember column:")
    print(df['IsActiveMember'].unique())
    print(f"IsActiveMember value counts: {df['IsActiveMember'].value_counts(dropna=False)}")

    #churn data cleaning
    print("\n[Before] Unique values in Churn column:")
    print(df['Churn'].unique())

    # Convert to string and lowercase first - THIS IS THE KEY FIX
    df['Churn'] = df['Churn'].astype(str).str.lower().str.strip()
    print(f"After string conversion: {df['Churn'].unique()}")

    # Create mapping for known values - include all variations
    churn_mapping = {
        '1': 1, '1.0': 1,
        '0': 0, '0.0': 0,
        '2': 0, '2.0': 0,
        'maybe': 0,
        'nan': np.nan
    }

    # Apply mapping
    df['Churn'] = df['Churn'].map(churn_mapping)
    print(f"After mapping, before Int64 conversion: {df['Churn'].unique()}")

    # Convert to  integer type
    df['Churn'] = df['Churn'].astype('Int64')

    print("[After] Unique values in Churn column:")
    print(df['Churn'].unique())
    print(f"Churn value counts after cleaning: {df['Churn'].value_counts(dropna=False)}")

    #NumOfProducts Cleaning
    print("\n[Before] Unique values in NumOfProducts column:")
    print(df['NumOfProducts'].unique())
    df['NumOfProducts'] = df['NumOfProducts'].astype('Int64')

    print("[After] Unique values in NumOfProducts column:")
    print(df['NumOfProducts'].unique())

    print("\n" + "=" * 50)
    print("TENURE PROCESSING")
    print("=" * 50)
    print("Age unique values:", df['Age'].unique())
    print("Tenure unique values:", df['Tenure'].unique())

    df['Tenure'] = df['Tenure'].astype('Int64')

    print("\nFinal data info after all cleaning:")
    print(df.info())

    print(f"\nFinal gender data: {df['Gender'].unique()}")

    #final validataion
    print("\n" + "=" * 60)
    print("final validataion of the data")
    print("=" * 60)
    print(f"Gender unique values: {sorted(df['Gender'].dropna().unique())}")
    print(f"HasCrCard unique values: {sorted(df['HasCrCard'].dropna().unique())}")
    print(f"IsActiveMember unique values: {sorted(df['IsActiveMember'].dropna().unique())}")
    print(f"Churn unique values: {sorted(df['Churn'].dropna().unique())}")
    print(f"NumOfProducts unique values: {sorted(df['NumOfProducts'].dropna().unique())}")

    #handling the label
    churn_unique = df['Churn'].dropna().unique()
    expected_churn = [0, 1]

    print(f"\n CHURN VALIDATION:")
    print(f"Expected: {expected_churn}")
    print(f"Actual: {sorted(churn_unique)}")

    if set(churn_unique).issubset(set(expected_churn)):
        print(" CHURN CLEANING SUCCESSFUL!")
    else:
        print(" CHURN CLEANING FAILED!")
        print("Unexpected values found in Churn column")

    # Check for any remaining unmapped values
    print("\nChecking for unmapped values:")
    for col in ['Gender', 'HasCrCard', 'IsActiveMember', 'Churn']:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f" {col}: {null_count} null values")
            else:
                print(f" {col}: No null values")

    return df


def visualize_missing_values_before(df):
    """Visualize missing values before imputation"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Barplot of missing values (before imputation)
    plt.figure(figsize=(10, 5))

    # Count missing values per column
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]  # Keep only columns with missing values

    if len(missing_counts) > 0:
        # Create barplot
        sns.barplot(x=missing_counts.index, y=missing_counts.values, palette="Reds_r")
        plt.title("Barplot: Count of Missing Values (Before Imputation)")
        plt.xticks(rotation=45)
        plt.ylabel("Missing Count")

        # Annotate each bar with count
        for index, value in enumerate(missing_counts.values):
            plt.text(index, value, str(value), ha='center', va='bottom', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Barplot: Count of Missing Values (Before Imputation)")

    plt.tight_layout()
    plt.show()


def identify_and_impute_missing_values(df):
    # Make a copy to avoid modifying original
    df = df.copy()

    print("\n" + "=" * 80)
    print("SECTION 3: IDENTIFY NUMERICAL & CATEGORICAL COLUMNS")
    print("=" * 80)

    # 1. Include all numeric types, including pandas nullable Int64
    numerical_cols = df.select_dtypes(include=['float64', 'int64', 'Int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    print(f"\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print("@" * 50)

    # 2. Remove target column from numerical_cols if present
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')

    # 3. Check columns with missing values
    numerical_missing = [col for col in numerical_cols if df[col].isnull().sum() > 0]
    categorical_missing = [col for col in categorical_cols if df[col].isnull().sum() > 0]

    print(f"\nNumerical columns with missing values: {numerical_missing}")
    print(f"Categorical columns with missing values: {categorical_missing}")

    # Find columns that contain the string 'nan' (not actual np.nan)
    print("Checking for string 'nan' values in each column...\n")
    for col in df.columns:
        str_col = df[col].astype(str).str.lower().str.strip()
        if 'nan' in str_col.unique():
            print(f"Column '{col}' contains string 'nan' values.")

    print(df.isnull().sum())

    # 3.1 BEFORE IMPUTATION SUMMARY
    print("Missing values before imputation:\n")
    print(df.isna().sum())

    #categorical and numerical value seperation

    # Create backup DataFrames
    df_median = df.copy()
    df_mode = df.copy()

    # Updated column groups - filter to only existing columns
    all_numerical_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    all_categorical_cols = ['Gender', 'HasCrCard', 'IsActiveMember']

    # Filter to only existing columns
    numerical_cols = [col for col in all_numerical_cols if col in df.columns]
    categorical_cols = [col for col in all_categorical_cols if col in df.columns]


    # Impute NUMERICAL columns with MEDIAN
    print("\n Imputing numerical columns with MEDIAN:")
    for col in numerical_cols:
        if df_median[col].isnull().sum() > 0:  # Only impute if there are missing values
            median_val = df_median[col].median()
            df_median[col] = df_median[col].fillna(int(median_val))  # cast to int
            print(f"Imputed {col} with median: {median_val:.2f}")
        else:
            print(f"No missing values in {col}, skipping imputation")


    # Impute CATEGORICAL columns with MODE
    print("\nImputing categorical columns with MODE:")
    for col in categorical_cols:
        if df_mode[col].isnull().sum() > 0:  # Only impute if there are missing values
            mode_values = df_mode[col].mode()
            if len(mode_values) > 0:
                mode_val = mode_values[0]
                df_mode[col] = df_mode[col].fillna(mode_val)
                print(f"Imputed {col} with mode: '{mode_val}'")
            else:
                print(f"No mode found for {col}, skipping imputation")
        else:
            print(f"No missing values in {col}, skipping imputation")

    # Merge both into final cleaned DataFrame
    df_cleaned = df.copy()

    # Apply numerical imputation
    if numerical_cols:
        df_cleaned[numerical_cols] = df_median[numerical_cols]

    # Apply categorical imputation
    if categorical_cols:
        df_cleaned[categorical_cols] = df_mode[categorical_cols]

    # Final check
    print("\nFinal NaN count (should all be 0 for imputed columns):")
    print(df_cleaned.isna().sum())
    print(df_cleaned.info())
    print("\n")
    print(df_cleaned.head(25))

    # Drop rows where Churn is null
    if 'Churn' in df_cleaned.columns:
        initial_shape = df_cleaned.shape
        df_cleaned = df_cleaned.dropna(subset=['Churn'])
        dropped_rows = initial_shape[0] - df_cleaned.shape[0]
        print(f"Dropped {dropped_rows} rows where Churn was null")

    # Reset index after dropping
    df_cleaned.reset_index(drop=True, inplace=True)

    # Final check
    print("\nAfter dropping null Churns:")
    print(df_cleaned.isna().sum())
    print(f"Final shape: {df_cleaned.shape}")
    df_cleaned.to_csv("data/processed/customer_cleaned_imputed.csv", index=False)

    return df_cleaned


def visualize_missing_values_after(df):

    # Count missing values
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts == 0]  # Show only those with 0 missing

    # Create barplot (all bars at 0)
    plt.figure(figsize=(10, 4))

    if len(missing_counts) > 0:
        sns.barplot(x=missing_counts.index, y=missing_counts.values, color='green')
        plt.title("Missing Values after Imputation")
        plt.ylabel("Missing Count")
        plt.xticks(rotation=45)

        # Annotate with 0
        for index, value in enumerate(missing_counts.values):
            plt.text(index, value + 0.01, str(value), ha='center', va='bottom', fontsize=9)

        plt.ylim(0, 1)  # Ensure it's visually clear all are 0
    else:
        plt.text(0.5, 0.5, 'No columns with 0 missing values to display', ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.title("Missing Values after Imputation")

    plt.tight_layout()
    plt.show()
