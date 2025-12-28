import sys
from pathlib import Path
import numpy as np

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# -------------------------------------------------
# Imports
# -------------------------------------------------
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.data.preprocess import build_preprocessor

# -------------------------------------------------
# Load data
# -------------------------------------------------
data_path = project_root / "data" / "raw" / "heart.csv"
df = pd.read_csv(data_path)

X = df.drop("target", axis=1)

# Convert multi-class target to binary (0 = no disease, 1 = disease)
y = (df["target"] > 0).astype(int)

assert y.nunique() == 2, "Target must be binary for ROC-AUC"

# -------------------------------------------------
# Models
# -------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# -------------------------------------------------
# CV + Scorer
# -------------------------------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Define scoring metrics
scoring = {
    'roc_auc': make_scorer(roc_auc_score),
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0)
}

# -------------------------------------------------
# MLflow
# -------------------------------------------------
mlflow.set_experiment("Heart-Disease-Classification")

best_model = None
best_score = 0
best_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model)
        ])

        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            error_score=np.nan
        )

        # Fit on full data for final evaluation
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        y_pred_proba = pipeline.predict_proba(X)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_pred_proba)

        # Log parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)

        # Log CV metrics (mean and std)
        for metric_name in scoring.keys():
            metric_key = f"test_{metric_name}"
            if metric_key in cv_results:
                scores = cv_results[metric_key]
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) > 0:
                    mlflow.log_metric(f"cv_{metric_name}_mean", valid_scores.mean())
                    mlflow.log_metric(f"cv_{metric_name}_std", valid_scores.std())

        # Log final metrics on full dataset
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log confusion matrix
        cm = confusion_matrix(y, y_pred)
        mlflow.log_param("tn", int(cm[0, 0]))
        mlflow.log_param("fp", int(cm[0, 1]))
        mlflow.log_param("fn", int(cm[1, 0]))
        mlflow.log_param("tp", int(cm[1, 1]))

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=X.iloc[:5].to_dict('records')
        )

        # Print results
        print(f"\n{'='*60}")
        print(f"{name} - Cross-Validation Results:")
        print(f"{'='*60}")
        for metric_name in scoring.keys():
            metric_key = f"test_{metric_name}"
            if metric_key in cv_results:
                scores = cv_results[metric_key]
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) > 0:
                    print(f"CV {metric_name.upper()}: {valid_scores.mean():.4f} (+/- {valid_scores.std()*2:.4f})")
        
        print(f"\nFinal Metrics on Full Dataset:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)

        # Track best model
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = pipeline
            best_name = name

print(f"\n{'='*60}")
print(f"Best Model: {best_name} (ROC-AUC: {best_score:.4f})")
print(f"{'='*60}")


# Save best model
import joblib
if best_model is not None:
    joblib.dump(best_model, project_root / "model.joblib")
    print(f"\nSaved best model ({best_name}) to model.joblib")
else:
    joblib.dump(pipeline, project_root / "model.joblib")
    print(f"\nSaved model to model.joblib")
