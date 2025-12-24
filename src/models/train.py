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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score

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


roc_auc_scorer = make_scorer(roc_auc_score)

# -------------------------------------------------
# MLflow
# -------------------------------------------------
mlflow.set_experiment("Heart-Disease-Classification")

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=roc_auc_scorer,
            error_score=np.nan
        )

        valid_scores = scores[~np.isnan(scores)]

        mlflow.log_param("model_name", name)
        mlflow.log_param("cv_folds", 5)

        if len(valid_scores) > 0:
            mlflow.log_metric("roc_auc_mean", valid_scores.mean())
            mlflow.log_metric("roc_auc_std", valid_scores.std())
        else:
            mlflow.log_param("roc_auc_status", "invalid_folds")

        pipeline.fit(X, y)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        print(f"{name} | CV ROC-AUC scores: {valid_scores}")


import joblib

joblib.dump(pipeline, "model.joblib")
