import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime


def compare_models_and_insights(results, df=None):
    # Load dataset if not provided
    if df is None:
        df = pd.read_csv("customer_features_enriched.csv")

    print("\n" + "=" * 80)
    print("CONFUSION MATRICES FOR MODELS")
    print("=" * 80)

    # Set up the figure with improved aesthetics
    plt.style.use('default')  # Reset to default style
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance - Confusion Matrix Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # Define a modern color palette
    confusion_colors = ['#f8f9fa', '#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5', '#2196f3', '#1e88e5',
                        '#1976d2']

    # Logistic Regression Confusion Matrix
    ax1 = axes[0]
    cm1 = results['logistic regression']['confusion_matrix']

    # Create custom annotations with percentages
    total1 = cm1.sum()
    cm1_percent = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]

    # Custom annotation text
    annot1 = np.array([[f'{cm1[i, j]}\n({cm1_percent[i, j]:.1%})' for j in range(cm1.shape[1])]
                       for i in range(cm1.shape[0])])

    sns.heatmap(cm1, annot=annot1, fmt='', cmap='Blues',
                ax=ax1, cbar=True, square=True, linewidths=2, linecolor='white',
                cbar_kws={'shrink': 0.8, 'aspect': 20})
    ax1.set_title('Logistic Regression', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='semibold')
    ax1.set_ylabel('Actual', fontsize=12, fontweight='semibold')
    ax1.set_xticklabels(['Retained', 'Churned'], fontsize=11)
    ax1.set_yticklabels(['Retained', 'Churned'], fontsize=11, rotation=0)

    # Random Forest Confusion Matrix
    ax2 = axes[1]
    cm2 = results['random forest (default)']['confusion_matrix']

    # Create custom annotations with percentages
    total2 = cm2.sum()
    cm2_percent = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]

    # Custom annotation text
    annot2 = np.array([[f'{cm2[i, j]}\n({cm2_percent[i, j]:.1%})' for j in range(cm2.shape[1])]
                       for i in range(cm2.shape[0])])

    sns.heatmap(cm2, annot=annot2, fmt='', cmap='Greens',
                ax=ax2, cbar=True, square=True, linewidths=2, linecolor='white',
                cbar_kws={'shrink': 0.8, 'aspect': 20})
    ax2.set_title('Random Forest (Default)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='semibold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='semibold')
    ax2.set_xticklabels(['Retained', 'Churned'], fontsize=11)
    ax2.set_yticklabels(['Retained', 'Churned'], fontsize=11, rotation=0)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)

    # Create the performance comparison with improved aesthetics
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())
    metric_values = {metric: [results[model][metric] for model in model_names] for metric in metrics}

    # Modern color palette with gradients
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.18

    # Create bars with enhanced styling
    bars = []
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bar = ax.bar(x + i * width, metric_values[metric], width,
                     label=metric.title(), alpha=0.85, color=color,
                     edgecolor='white', linewidth=1.5)
        bars.append(bar)

        # Add value labels on top of bars
        for j, bar_patch in enumerate(bar):
            height = bar_patch.get_height()
            ax.text(bar_patch.get_x() + bar_patch.get_width() / 2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom',
                    fontweight='semibold', fontsize=10)

    # Customize the plot
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([name.title() for name in model_names], fontsize=12, fontweight='semibold')

    # Enhance legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
              fontsize=12, title='Metrics', title_fontsize=13)

    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Set y-axis limits for better visualization
    ax.set_ylim(0, max(max(values) for values in metric_values.values()) * 1.15)

    # Add subtle background color
    ax.set_facecolor('#FAFAFA')

    # Enhance spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#CCCCCC')

    plt.tight_layout()
    plt.show()

    # Create an additional detailed comparison table visualization
    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE METRICS TABLE")
    print("=" * 80)

    # Create a more detailed comparison table as a heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for heatmap
    performance_data = []
    for model in model_names:
        row = [results[model][metric] for metric in metrics]
        performance_data.append(row)

    performance_df = pd.DataFrame(performance_data,
                                  index=[name.title() for name in model_names],
                                  columns=[metric.title() for metric in metrics])

    # Create heatmap with annotations
    sns.heatmap(performance_df, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.5, square=True, linewidths=2, linecolor='white',
                ax=ax, cbar_kws={'shrink': 0.8, 'aspect': 20},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})

    ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='semibold')
    ax.set_ylabel('Models', fontsize=12, fontweight='semibold')

    # Rotate labels for better readability
    plt.xticks(rotation=0, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.show()

    # Final Recommendation
    print("\n" + "=" * 80)
    print("SECTION 10: FINAL RECOMMENDATIONS AND INSIGHTS")
    print("=" * 80)

    # Determine best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]

    print("\n10.1 MODEL PERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"Best performing model: {best_model_name}")
    print(f"F1-Score: {best_model['f1']:.3f}")
    print(f"Accuracy: {best_model['accuracy']:.3f}")
    print(f"Precision: {best_model['precision']:.3f}")
    print(f"Recall: {best_model['recall']:.3f}")
    print("\n10.2 BUSINESS INSIGHTS:")
    print("-" * 25)

    # Age band analysis
    age_churn_rates = df.groupby('AgeBand')['Churn'].mean().sort_values(ascending=False)
    print(f"Highest churn age group: {age_churn_rates.index[0]} ({age_churn_rates.iloc[0]:.1%})")

    # Product analysis
    product_churn_rates = df.groupby('NumOfProducts')['Churn'].mean().sort_values(ascending=False)
    print(f"Highest churn product count: {product_churn_rates.index[0]} products ({product_churn_rates.iloc[0]:.1%})")

    # Activity level analysis
    activity_churn_rates = df.groupby('ActivityLevel')['Churn'].mean().sort_values(ascending=False)
    print(f"Highest churn activity level: {activity_churn_rates.index[0]} ({activity_churn_rates.iloc[0]:.1%})")

    # Feature Importance – shown only if available
    if 'feature_importance' in globals() and not feature_importance.empty:
        print("\n10.3 TOP RISK FACTORS (FROM FEATURE IMPORTANCE):")
        print("-" * 55)
        for i, row in feature_importance.head(5).iterrows():
            print(f"{i + 1}. {row['feature']}: {row['importance']:.3f}")
    else:
        print("\n10.3 TOP RISK FACTORS:")
        print("-" * 55)
        print(f"Feature importance not available for {best_model_name}.")

    if 'feature_importance' in globals() and not feature_importance.empty:
        print(f"6. Consider interventions for top risk feature: {feature_importance.iloc[0]['feature']}")

    print("\n" + "=" * 80)
    print("SAVING MODEL AND METRICS TO FILES")
    print("=" * 80)

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # Save the best model
    model_path = 'model/trained_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model['model'], f)  # Note: accessing the model from results
    print(f" Best model saved to: {model_path}")

    # Save model metrics to text file
    metrics_path = 'model/model_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("CREDIT CARD CHURN PREDICTION - MODEL EVALUATION METRICS\n")
        f.write("=" * 65 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Model comparison table
        f.write("MODEL PERFORMANCE COMPARISON:\n")
        f.write("-" * 35 + "\n")
        f.write(f"{'Model':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<10}\n")
        f.write("-" * 65 + "\n")
        for model_name, result in results.items():
            f.write(f"{model_name:<25} {result['accuracy']:<10.4f} {result['precision']:<11.4f} "
                    f"{result['recall']:<8.4f} {result['f1']:<10.4f}\n")

        f.write(f"\nBEST PERFORMING MODEL:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Model Name: {best_model_name}\n")
        f.write(f"Accuracy: {best_model['accuracy']:.4f}\n")
        f.write(f"Precision: {best_model['precision']:.4f}\n")
        f.write(f"Recall: {best_model['recall']:.4f}\n")
        f.write(f"F1-Score: {best_model['f1']:.4f}\n\n")

        f.write("CONFUSION MATRIX (Best Model):\n")
        f.write("-" * 32 + "\n")
        f.write(f"{best_model['confusion_matrix']}\n\n")

        f.write("BUSINESS INSIGHTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Dataset size: {len(df)} customers\n")
        f.write(f"Overall churn rate: {df['Churn'].mean():.1%}\n")

        # Age band insights
        age_churn_rates = df.groupby('AgeBand')['Churn'].mean().sort_values(ascending=False)
        f.write(f"Highest risk age group: {age_churn_rates.index[0]} ({age_churn_rates.iloc[0]:.1%})\n")

        # Product insights
        product_churn_rates = df.groupby('NumOfProducts')['Churn'].mean().sort_values(ascending=False)
        f.write(
            f"Highest risk product count: {product_churn_rates.index[0]} products ({product_churn_rates.iloc[0]:.1%})\n")

        # Activity level insights
        activity_churn_rates = df.groupby('ActivityLevel')['Churn'].mean().sort_values(ascending=False)
        f.write(f"Highest risk activity level: {activity_churn_rates.index[0]} ({activity_churn_rates.iloc[0]:.1%})\n")

        f.write(f"\nModel file saved to: {model_path}\n")
        f.write(f"Metrics file: {metrics_path}\n")
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Model metrics saved to: {metrics_path}")
    print(f" Files ready in model/ directory for deployment!")

    return best_model_name, best_model