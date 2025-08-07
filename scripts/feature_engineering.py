import pandas as pd
import numpy as np


def create_features(df=None):

    # FEATURE ENGINEERING

    print("\n" + "=" * 80)
    print("SECTION 5: FEATURE ENGINEERING")
    print("=" * 80)

    # Load cleaned dataset
    if df is None:
        df = pd.read_csv("customer_outliers_handled.csv")

    print("5.1 CREATING NEW FEATURES:")
    print("-" * 35)

    # Feature 1: Balance per Product (Balance efficiency)
    df['BalancePerProduct'] = df['Balance'] / (df['NumOfProducts'] + 1)
    print("Created BalancePerProduct: Balance efficiency metric")

    # Feature 2 defining  Age Band
    df['AgeBand'] = pd.cut(df['Age'],
                           bins=[0, 29, 39, 49, 59, float('inf')],
                           labels=['18-29', '30-39', '40-49', '50-59', '60+'])
    print(" Created AgeBand: Age group categorization")

    # Feature 3 creating Tenure Band
    df['TenureBand'] = pd.cut(df['Tenure'],
                              bins=[-1, 2, 5, float('inf')],
                              labels=['New (0-2 years)', 'Medium (3-5 years)', 'Long (6+ years)'])
    print(" Created TenureBand: Customer relationship length")

    # Feature 4 creating the Activity Level (CrCard + IsActiveMember)
    df['ActivityLevel'] = (df['HasCrCard'] + df['IsActiveMember']).map({0: 'Low', 1: 'Medium', 2: 'High'})
    print("Created ActivityLevel: Customer engagement metric")

    # Feature 5 :- Salary per Product that is spending capacity
    df['SalaryPerProduct'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
    print("Created SalaryPerProduct: Spending capacity per product")

    # Feature 6: Balance to Salary Ratio
    df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    print(" Created BalanceToSalaryRatio: Financial health indicator")

    # Feature 7  High Value Customer (balance or salary in top 25%)
    balance_thresh = df['Balance'].quantile(0.75)
    salary_thresh = df['EstimatedSalary'].quantile(0.75)
    df['HighValueCustomer'] = ((df['Balance'] > balance_thresh) | (df['EstimatedSalary'] > salary_thresh)).astype(int)
    print(f"Created HighValueCustomer: {df['HighValueCustomer'].sum()} high-value customers identified")

    # Feature 8: Customer Risk Score (based on multiple churn factors)
    df['CustomerRiskScore'] = (
            (df['Age'] < 35).astype(int) * 0.2 +
            (df['Tenure'] < 3).astype(int) * 0.3 +
            (df['NumOfProducts'] == 1).astype(int) * 0.2 +
            (df['IsActiveMember'] == 0).astype(int) * 0.3
    )
    print(" Created CustomerRiskScore Composite churn risk indicator")


    # Feature 9: HasZeroBalance (flag for 0 balance)
    df['HasZeroBalance'] = (df['Balance'] == 0).astype(int)
    print("Created HasZeroBalance: Flag for zero balance customers")


    # Feature 10: IsSingleProduct (simple user flag)
    df['IsSingleProduct'] = (df['NumOfProducts'] == 1).astype(int)
    print("Created IsSingleProduct: Flag for single product users")

    # Feature 11 & 12: Log-transformed Balance & Salary
    df['LogBalance'] = np.log1p(df['Balance'])
    df['LogSalary'] = np.log1p(df['EstimatedSalary'])
    print("Created LogBalance and LogSalary: Log-transformed financial features")

    # Summary
    print(f"\n5.2 FEATURE ENGINEERING SUMMARY:")
    print("-" * 40)
    print(f"Original features: {len(df.columns) - 12}")
    print(f"New numerical features: 6")
    print(f"New categorical features: 3")
    print(f"Total features after engineering: {len(df.columns)}")

    # Display sample of new features
    new_features = ['BalancePerProduct', 'AgeBand', 'TenureBand', 'ActivityLevel',
                    'SalaryPerProduct', 'BalanceToSalaryRatio', 'HighValueCustomer',
                    'CustomerRiskScore', 'HasZeroBalance', 'IsSingleProduct',
                    'LogBalance', 'LogSalary']

    print(f"\nSample of new features:")
    print(df[new_features].head(10))

    # Save enriched dataset
    df.to_csv("data/processed/customer_features_enriched.csv", index=False)
    print("Saved feature-engineered dataset to: customer_features_enriched.csv")
    print(df.head(10))

    return df