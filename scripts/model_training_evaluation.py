import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("model training and evaluation")

    # initialize models
    models = {
        'logistic regression': LogisticRegression(
            solver='liblinear',
            random_state=42
        ),
        'random forest (default)': RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            max_features='sqrt',
            random_state=42
        )
    }

    # store results
    results = {}

    print("8.1 training models:")
    print("-" * 25)

    for name, model in models.items():
        print(f"\ntraining {name}...")

        # confirm datatypes of training data
        print("x_train types:\n", X_train.dtypes)
        print("y_train type:\n", y_train.dtype)

        # check for non-numeric columns
        non_numeric_cols = X_train.select_dtypes(include='object').columns
        if len(non_numeric_cols) > 0:
            print("non-numeric columns found in x_train:", list(non_numeric_cols))
            print(X_train[non_numeric_cols].head())
            raise ValueError("fix required: x_train contains non-numeric data.")

        # train the model
        model.fit(X_train, y_train)

        # predict on test data
        y_pred = model.predict(X_test)

        # handle predict_proba safely
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = np.zeros_like(y_pred)

        # calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # store results
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

        # print summary
        print(f"{name} trained successfully")
        print(f"  accuracy: {accuracy:.3f}")
        print(f"  precision: {precision:.3f}")
        print(f"  recall: {recall:.3f}")
        print(f"  f1-score: {f1:.3f}")

    print("\n8.2 detailed evaluation results:")
    print("-" * 40)

    for name, result in results.items():
        print(f"\n{name} - detailed metrics")
        print("-" * 50)
        print("classification report:")
        print(classification_report(y_test, result['y_pred']))

    return results