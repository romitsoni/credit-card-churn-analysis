import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import os

def generate_profiling_reports():
    try:
        # Ensure the output directory exists
        os.makedirs("reports", exist_ok=True)

        # File paths
        raw_file_path = "data/raw/exl_credit_card_churn_data.csv"
        processed_file_path = "data/processed/customer_features_enriched.csv"

        # Load datasets
        raw_df = pd.read_csv(raw_file_path)
        processed_df = pd.read_csv(processed_file_path)

        # Generate raw data profile
        raw_profile = ProfileReport(raw_df, title="Raw Dataset Profile Report", explorative=True)
        raw_profile.to_file("reports/raw_data_profile.html")

        # Generate processed data profile
        processed_profile = ProfileReport(processed_df, title="Processed Dataset Profile Report", explorative=True)
        processed_profile.to_file("reports/processed_data_profile.html")

        print("\n" + "=" * 80)
        print("AUTO-PROFILING REPORTS GENERATED SUCCESSFULLY")
        print("Raw report saved to       → reports/raw_data_profile.html")
        print("Processed report saved to → reports/processed_data_profile.html")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"[ERROR] Failed to generate profiling reports: {e}")





def perform_eda(df=None):

    print("\n" + "=" * 80)
    print("SECTION 6: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Load dataset if not provided
    if df is None:
        df = pd.read_csv("customer_features_enriched.csv")

    print("6.1 CHURN DISTRIBUTION ANALYSIS:")
    print("-" * 40)

    # Basic churn statistics
    churn_counts = df['Churn'].value_counts()
    churn_percentages = df['Churn'].value_counts(normalize=True) * 100

    print(f"Churn Distribution:")
    print(f"Retained (0): {churn_counts[0]} customers ({churn_percentages[0]:.1f}%)")
    print(f"Churned (1): {churn_counts[1]} customers ({churn_percentages[1]:.1f}%)")

    # Figure 1: Demographics
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Customer Churn Analysis - Part 1', fontsize=16, fontweight='bold')

    # Pie Chart
    labels = ['Retained', 'Churned']
    sizes = df['Churn'].value_counts(normalize=True) * 100
    axs[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=90)
    axs[0].set_title('Overall Churn Distribution')

    # Gender
    gender_churn = df.groupby(['Gender', 'Churn']).size().unstack().apply(lambda x: x / x.sum() * 100, axis=1)
    gender_churn[['Retained', 'Churned']] = gender_churn[[0, 1]]
    gender_churn[['Retained', 'Churned']].plot(kind='bar', stacked=False, ax=axs[1], color=['skyblue', 'salmon'])
    axs[1].set_title('Churn Rate by Gender')
    axs[1].set_ylabel('Percentage')
    axs[1].legend()

    # Age Band
    age_churn = df.groupby(['AgeBand', 'Churn']).size().unstack().apply(lambda x: x / x.sum() * 100, axis=1)
    age_churn[['Retained', 'Churned']] = age_churn[[0, 1]]
    age_churn[['Retained', 'Churned']].plot(kind='bar', stacked=False, ax=axs[2], color=['skyblue', 'salmon'])
    axs[2].set_title('Churn Rate by Age Band')
    axs[2].set_ylabel('Percentage')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # -----------------------------------------------------------------------------------------------

    # Figure 2: Behavioral Insights
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Customer Churn Analysis - Part 2', fontsize=16, fontweight='bold')

    # Activity Level
    activity_churn = df.groupby(['ActivityLevel', 'Churn']).size().unstack().apply(lambda x: x / x.sum() * 100, axis=1)
    activity_churn[['Retained', 'Churned']] = activity_churn[[0, 1]]
    activity_churn[['Retained', 'Churned']].plot(kind='bar', stacked=False, ax=axs[0], color=['skyblue', 'salmon'])
    axs[0].set_title('Churn Rate by Activity Level')
    axs[0].set_ylabel('Percentage')
    axs[0].legend()

    # Number of Products
    product_churn = df.groupby(['NumOfProducts', 'Churn']).size().unstack().apply(lambda x: x / x.sum() * 100, axis=1)
    product_churn[['Retained', 'Churned']] = product_churn[[0, 1]]
    product_churn[['Retained', 'Churned']].plot(kind='bar', stacked=False, ax=axs[1], color=['skyblue', 'salmon'])
    axs[1].set_title('Churn Rate by Number of Products')
    axs[1].set_ylabel('Percentage')
    axs[1].legend()

    # Tenure Band
    tenure_churn = df.groupby(['TenureBand', 'Churn']).size().unstack().apply(lambda x: x / x.sum() * 100, axis=1)
    tenure_churn[['Retained', 'Churned']] = tenure_churn[[0, 1]]
    tenure_churn[['Retained', 'Churned']].plot(kind='bar', stacked=False, ax=axs[2], color=['skyblue', 'salmon'])
    axs[2].set_title('Churn Rate by Tenure Band')
    axs[2].set_ylabel('Percentage')
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # -------------------------------------------------------------------------------------------------------------------------

    print("\n6.2 CORRELATION ANALYSIS:")
    print("-" * 30)

    # Select numeric columns and remove 'CustomerID' if present
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'CustomerID' in numerical_cols:
        numerical_cols.remove('CustomerID')

    # Compute correlation matrix
    corr_matrix = df[numerical_cols].corr()

    # Plot full heatmap of all numeric correlations
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .7})
    plt.title('Feature Correlation Heatmap (All Features)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


    # Filter: Keep only correlations above threshold
    threshold = 0.15
    filtered_corr = corr_matrix.mask(corr_matrix.abs() < threshold, 0)

    # Remove self-correlations for visual clarity
    np.fill_diagonal(filtered_corr.values, 0)

    # Plot filtered heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .7}, mask=(filtered_corr == 0))
    plt.title(f'Filtered Correlation Heatmap (|corr| ≥ {threshold})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


    # Show top features most correlated with 'Churn'

    # Exclude Churn self-correlation, sort by absolute value
    churn_corr = corr_matrix['Churn'].drop('Churn').abs().sort_values(ascending=False)

    # Filter features above threshold
    top_corr_features = churn_corr[churn_corr > threshold]

    print("\n Top Features Most Correlated with 'Churn':")
    print("-" * 40)
    for feature, value in top_corr_features.items():
        print(f"{feature:<25}: {value:.3f}")

    print(df.info())

    return df


def analyze_engineered_features(df):

    print("\n" + "=" * 80)
    print("SECTION 6.3: ENGINEERED FEATURES ANALYSIS")
    print("=" * 80)

    # Figure 3: Financial Health & Risk Analysis (First 2 plots)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Financial Health & Risk Analysis - Part 1', fontsize=16, fontweight='bold')

    # 1. High Value Customer Analysis
    ax1 = axs[0]
    high_value_churn = df.groupby('HighValueCustomer')['Churn'].mean() * 100
    bars1 = ax1.bar(['Regular Customer', 'High Value Customer'], high_value_churn.values,
                    color=['lightcoral', 'gold'], alpha=0.8, edgecolor='black')
    ax1.set_title('Churn Rate: High Value vs Regular Customers', fontweight='bold')
    ax1.set_ylabel('Churn Rate (%)')
    ax1.set_ylim(0, max(high_value_churn) * 1.2)

    # Add percentage labels on bars
    for bar, value in zip(bars1, high_value_churn.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add count annotations
    high_value_counts = df['HighValueCustomer'].value_counts()
    ax1.text(0, high_value_churn.iloc[0] * 0.5, f'n={high_value_counts[0]}',
             ha='center', va='center', fontweight='bold', color='white')
    ax1.text(1, high_value_churn.iloc[1] * 0.5, f'n={high_value_counts[1]}',
             ha='center', va='center', fontweight='bold', color='white')

    # 2. Customer Risk Score Distribution by Churn
    ax2 = axs[1]
    retained = df[df['Churn'] == 0]['CustomerRiskScore']
    churned = df[df['Churn'] == 1]['CustomerRiskScore']

    ax2.hist(retained, bins=15, alpha=0.7, label='Retained', color='skyblue', density=True)
    ax2.hist(churned, bins=15, alpha=0.7, label='Churned', color='salmon', density=True)
    ax2.axvline(retained.mean(), color='blue', linestyle='--', alpha=0.8,
                label=f'Retained Mean: {retained.mean():.2f}')
    ax2.axvline(churned.mean(), color='red', linestyle='--', alpha=0.8,
                label=f'Churned Mean: {churned.mean():.2f}')
    ax2.set_title('Risk Score Distribution by Churn Status', fontweight='bold')
    ax2.set_xlabel('Customer Risk Score')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Figure 4: Financial Health & Risk Analysis (Second 2 plots)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Financial Health & Risk Analysis - Part 2', fontsize=16, fontweight='bold')

    # 3. Balance to Salary Ratio Analysis
    ax3 = axs[0]
    # Filter extreme outliers for better visualization - FIX: Use .copy() to avoid warning
    df_filtered = df[df['BalanceToSalaryRatio'] <= df['BalanceToSalaryRatio'].quantile(0.95)].copy()

    # Create bins for balance to salary ratio
    df_filtered['BalanceSalaryBin'] = pd.cut(df_filtered['BalanceToSalaryRatio'],
                                             bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    # FIX: Add observed=False to suppress FutureWarning
    balance_churn = df_filtered.groupby('BalanceSalaryBin', observed=False)['Churn'].mean() * 100
    bars3 = ax3.bar(range(len(balance_churn)), balance_churn.values,
                    color='green', alpha=0.7, edgecolor='black')
    ax3.set_title('Churn Rate by Balance-to-Salary Ratio', fontweight='bold')
    ax3.set_xlabel('Balance to Salary Ratio Level')
    ax3.set_ylabel('Churn Rate (%)')
    ax3.set_xticks(range(len(balance_churn)))
    ax3.set_xticklabels(balance_churn.index, rotation=45)

    for bar, value in zip(bars3, balance_churn.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Product Usage Pattern Analysis
    ax4 = axs[1]

    # Create a comprehensive view of product patterns
    product_patterns = []
    labels = []

    # Single product users with zero balance
    single_zero = len(df[(df['IsSingleProduct'] == 1) & (df['HasZeroBalance'] == 1)])
    if single_zero > 0:
        single_zero_churn = df[(df['IsSingleProduct'] == 1) & (df['HasZeroBalance'] == 1)]['Churn'].mean() * 100
    else:
        single_zero_churn = 0
    product_patterns.append(single_zero_churn)
    labels.append(f'Single Prod.\n+ Zero Bal.\n(n={single_zero})')

    # Single product users with balance
    single_balance = len(df[(df['IsSingleProduct'] == 1) & (df['HasZeroBalance'] == 0)])
    if single_balance > 0:
        single_balance_churn = df[(df['IsSingleProduct'] == 1) & (df['HasZeroBalance'] == 0)]['Churn'].mean() * 100
    else:
        single_balance_churn = 0
    product_patterns.append(single_balance_churn)
    labels.append(f'Single Prod.\n+ Has Bal.\n(n={single_balance})')

    # Multiple products with zero balance
    multi_zero = len(df[(df['IsSingleProduct'] == 0) & (df['HasZeroBalance'] == 1)])
    if multi_zero > 0:
        multi_zero_churn = df[(df['IsSingleProduct'] == 0) & (df['HasZeroBalance'] == 1)]['Churn'].mean() * 100
    else:
        multi_zero_churn = 0
    product_patterns.append(multi_zero_churn)
    labels.append(f'Multi Prod.\n+ Zero Bal.\n(n={multi_zero})')

    # Multiple products with balance
    multi_balance = len(df[(df['IsSingleProduct'] == 0) & (df['HasZeroBalance'] == 0)])
    if multi_balance > 0:
        multi_balance_churn = df[(df['IsSingleProduct'] == 0) & (df['HasZeroBalance'] == 0)]['Churn'].mean() * 100
    else:
        multi_balance_churn = 0
    product_patterns.append(multi_balance_churn)
    labels.append(f'Multi Prod.\n+ Has Bal.\n(n={multi_balance})')

    colors = ['red', 'orange', 'yellow', 'lightgreen']
    bars4 = ax4.bar(range(len(product_patterns)), product_patterns,
                    color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('Churn Rate by Product & Balance Pattern', fontweight='bold')
    ax4.set_xlabel('Customer Pattern')
    ax4.set_ylabel('Churn Rate (%)')
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, fontsize=9)

    for bar, value in zip(bars4, product_patterns):
        if not np.isnan(value) and value > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



    # Figure 5: Advanced Feature Relationships (First 2 plots)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Advanced Feature Relationships & Insights - Part 1', fontsize=16, fontweight='bold')

    # 1. Balance Per Product vs Churn (Violin Plot)
    ax1 = axs[0]
    # Filter extreme outliers
    df_filtered_balance = df[df['BalancePerProduct'] <= df['BalancePerProduct'].quantile(0.95)]

    violin_data = [df_filtered_balance[df_filtered_balance['Churn'] == 0]['BalancePerProduct'],
                   df_filtered_balance[df_filtered_balance['Churn'] == 1]['BalancePerProduct']]

    parts = ax1.violinplot(violin_data, positions=[0, 1], showmeans=True, showmedians=True)
    ax1.set_title('Balance Per Product Distribution by Churn', fontweight='bold')
    ax1.set_xlabel('Churn Status')
    ax1.set_ylabel('Balance Per Product')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Retained', 'Churned'])

    # Color the violin plots
    colors = ['skyblue', 'salmon']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # 2. Activity Level vs Age Band Heatmap
    ax2 = axs[1]
    # FIX: Add observed=False and handle object dtype issue
    activity_age_churn = df.groupby(['ActivityLevel', 'AgeBand'], observed=False)['Churn'].mean()
    activity_age_pivot = activity_age_churn.unstack(fill_value=0)

    # FIX: Ensure numeric data type for heatmap
    activity_age_pivot = activity_age_pivot.astype(float)

    sns.heatmap(activity_age_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r',
                ax=ax2, cbar_kws={'label': 'Churn Rate'})
    ax2.set_title('Churn Rate: Activity Level vs Age Band', fontweight='bold')
    ax2.set_xlabel('Age Band')
    ax2.set_ylabel('Activity Level')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Figure 6: Advanced Feature Relationships (Second 2 plots)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Advanced Feature Relationships & Insights - Part 2', fontsize=16, fontweight='bold')

    # 3. Salary Per Product Analysis
    ax3 = axs[0]
    # Create salary per product bins - FIX: Use .copy() to avoid warning
    df_temp = df.copy()
    df_temp['SalaryPerProductBin'] = pd.cut(df_temp['SalaryPerProduct'],
                                            bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    salary_prod_churn = df_temp.groupby('SalaryPerProductBin', observed=False)['Churn'].mean() * 100
    bars5 = ax3.bar(range(len(salary_prod_churn)), salary_prod_churn.values,
                    color='purple', alpha=0.7, edgecolor='black')
    ax3.set_title('Churn Rate by Salary Per Product Level', fontweight='bold')
    ax3.set_xlabel('Salary Per Product Level')
    ax3.set_ylabel('Churn Rate (%)')
    ax3.set_xticks(range(len(salary_prod_churn)))
    ax3.set_xticklabels(salary_prod_churn.index, rotation=45)

    for bar, value in zip(bars5, salary_prod_churn.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Risk Score vs High Value Customer Matrix
    ax4 = axs[1]

    # Create risk score bins - FIX: Use .copy() to avoid warning
    df_temp['RiskBin'] = pd.cut(df_temp['CustomerRiskScore'], bins=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])

    # Create a 2x3 matrix showing combinations
    risk_value_combinations = []
    risk_labels = []

    for risk_level in ['Low Risk', 'Medium Risk', 'High Risk']:
        for value_type in [0, 1]:  # 0=Regular, 1=High Value
            subset = df_temp[(df_temp['RiskBin'] == risk_level) & (df_temp['HighValueCustomer'] == value_type)]
            if len(subset) > 0:
                churn_rate = subset['Churn'].mean() * 100
                count = len(subset)
                risk_value_combinations.append(churn_rate)
                value_label = 'High Value' if value_type == 1 else 'Regular'
                risk_labels.append(f'{risk_level}\n{value_label}\n(n={count})')
            else:
                risk_value_combinations.append(0)
                value_label = 'High Value' if value_type == 1 else 'Regular'
                risk_labels.append(f'{risk_level}\n{value_label}\n(n=0)')

    # Create color gradient based on churn rate
    colors = plt.cm.RdYlGn_r([rate / 100 for rate in risk_value_combinations])
    bars6 = ax4.bar(range(len(risk_value_combinations)), risk_value_combinations,
                    color=colors, alpha=0.8, edgecolor='black')
    ax4.set_title('Churn Rate: Risk Score vs Customer Value Matrix', fontweight='bold')
    ax4.set_xlabel('Risk & Value Combination')
    ax4.set_ylabel('Churn Rate (%)')
    ax4.set_xticks(range(len(risk_labels)))
    ax4.set_xticklabels(risk_labels, fontsize=8, rotation=45)

    for bar, value in zip(bars6, risk_value_combinations):
        if value > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    # Print key insights
    print("\n6.4 KEY INSIGHTS FROM ENGINEERED FEATURES:")
    print("-" * 50)

    print(f"• High Value Customer Impact:")
    high_val_insight = df.groupby('HighValueCustomer')['Churn'].mean()
    print(f"  - Regular customers churn rate: {high_val_insight[0]:.1%}")
    print(f"  - High value customers churn rate: {high_val_insight[1]:.1%}")

    print(f"\n• Risk Score Analysis:")
    print(f"  - Average risk score (retained): {retained.mean():.3f}")
    print(f"  - Average risk score (churned): {churned.mean():.3f}")
    print(f"  - Risk score difference: {churned.mean() - retained.mean():.3f}")

    print(f"\n• Product & Balance Pattern (Highest Risk):")
    if len(product_patterns) > 0 and not np.isnan(product_patterns[0]) and product_patterns[0] > 0:
        print(f"  - Single product + Zero balance: {product_patterns[0]:.1f}% churn rate")

    print(f"\n• Financial Health Indicators:")
    if len(balance_churn) > 0:
        max_churn_bin = balance_churn.idxmax()
        print(f"  - Highest risk balance-salary ratio: {max_churn_bin} ({balance_churn[max_churn_bin]:.1f}%)")

    print(f"\nEngineered features analysis completed")

    return df