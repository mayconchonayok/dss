import time
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, brier_score_loss, log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

RANDOM_STATE = 42
BINARY_LABELS = ["Not home win", "Home win"]


def make_pipeline(model, use_scaler=True, use_smote=True, smote_k_neighbors=3):
    steps = []
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k_neighbors)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _cv_splits(y, max_splits=5):
    counts = pd.Series(y).value_counts()
    if counts.empty:
        return 2
    return max(2, min(max_splits, int(counts.min())))


def cv_object(y, max_splits=5):
    return StratifiedKFold(n_splits=_cv_splits(y, max_splits), shuffle=True, random_state=RANDOM_STATE)


def binary_model_plan():
    """Keep the original comparison idea: 4 algorithms x 3 variants = 12 models, but for a binary target."""
    return [
        {
            "model_id": "LR-1", "algorithm": "Logistic Regression", "version": "Baseline",
            "selection_method": "Default setting", "main_hyperparameter": "C=1.0",
            "model": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs"),
            "pipeline": {"use_scaler": True, "use_smote": True}, "param_grid": None,
        },
        {
            "model_id": "LR-2", "algorithm": "Logistic Regression", "version": "Stronger regularization",
            "selection_method": "Manual tuning", "main_hyperparameter": "C=0.1",
            "model": LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", solver="lbfgs"),
            "pipeline": {"use_scaler": True, "use_smote": True}, "param_grid": None,
        },
        {
            "model_id": "LR-3", "algorithm": "Logistic Regression", "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV", "main_hyperparameter": "C, solver",
            "model": LogisticRegression(max_iter=1200, class_weight="balanced"),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": {"model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10], "model__solver": ["lbfgs", "liblinear"]},
        },
        {
            "model_id": "SVM-1", "algorithm": "SVM", "version": "Linear baseline",
            "selection_method": "Manual setting", "main_hyperparameter": "kernel=linear, C=1.0",
            "model": SVC(C=1.0, kernel="linear", class_weight="balanced", probability=True, max_iter=5000, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": True, "use_smote": True}, "param_grid": None,
        },
        {
            "model_id": "SVM-2", "algorithm": "SVM", "version": "RBF manual",
            "selection_method": "Manual kernel change", "main_hyperparameter": "kernel=rbf, gamma=scale",
            "model": SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, max_iter=5000, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": True, "use_smote": True}, "param_grid": None,
        },
        {
            "model_id": "SVM-3", "algorithm": "SVM", "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV", "main_hyperparameter": "C, gamma, kernel",
            "model": SVC(class_weight="balanced", probability=True, max_iter=5000, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": {
                "model__C": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1, 0.5],
                "model__kernel": ["rbf", "poly", "sigmoid"],
            },
        },
        {
            "model_id": "DT-1", "algorithm": "Decision Tree", "version": "Shallow tree",
            "selection_method": "Manual setting", "main_hyperparameter": "max_depth=3",
            "model": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True}, "param_grid": None,
        },
        {
            "model_id": "DT-2", "algorithm": "Decision Tree", "version": "Deeper tree",
            "selection_method": "Manual setting", "main_hyperparameter": "max_depth=6",
            "model": DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True}, "param_grid": None,
        },
        {
            "model_id": "DT-3", "algorithm": "Decision Tree", "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV", "main_hyperparameter": "max_depth, min_samples_leaf",
            "model": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": {"model__max_depth": [2, 3, 4, 5, 7, 10, None], "model__min_samples_leaf": [1, 2, 5, 10], "model__criterion": ["gini", "entropy"]},
        },
        {
            "model_id": "BST-1", "algorithm": "GradientBoosting", "version": "Baseline",
            "selection_method": "Basic setting", "main_hyperparameter": "learning_rate=0.1, n_estimators=120",
            "model": GradientBoostingClassifier(learning_rate=0.1, n_estimators=120, max_depth=3, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True}, "param_grid": None,
        },
        {
            "model_id": "BST-2", "algorithm": "GradientBoosting", "version": "Conservative learning",
            "selection_method": "Manual tuning", "main_hyperparameter": "learning_rate=0.05, n_estimators=180",
            "model": GradientBoostingClassifier(learning_rate=0.05, n_estimators=180, max_depth=3, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True}, "param_grid": None,
        },
        {
            "model_id": "BST-3", "algorithm": "GradientBoosting", "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV", "main_hyperparameter": "learning_rate, max_depth, n_estimators",
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": {"model__n_estimators": [80, 120, 180, 250], "model__learning_rate": [0.01, 0.03, 0.05, 0.1], "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.9, 1.0]},
        },
        # Random Forest benchmark: robust ensemble model for tabular data
        # and useful for feature-importance based interpretation.
        {
            "model_id": "RF-1", "algorithm": "RandomForest", "version": "Baseline",
            "selection_method": "Ensemble baseline", "main_hyperparameter": "n_estimators=300, max_depth=10, min_samples_leaf=2",
            "model": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1),
            "pipeline": {"use_scaler": False, "use_smote": False}, "param_grid": None,
        },
        {
            "model_id": "RF-2", "algorithm": "RandomForest", "version": "Regularized forest",
            "selection_method": "Manual tuning", "main_hyperparameter": "max_depth=6, min_samples_leaf=3",
            "model": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1),
            "pipeline": {"use_scaler": False, "use_smote": False}, "param_grid": None,
        },
        {
            "model_id": "RF-3", "algorithm": "RandomForest", "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV", "main_hyperparameter": "n_estimators, max_depth, min_samples_leaf",
            "model": RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1),
            "pipeline": {"use_scaler": False, "use_smote": False},
            "param_grid": {"model__n_estimators": [150, 250, 400], "model__max_depth": [3, 5, 7, None], "model__min_samples_leaf": [1, 2, 4, 8], "model__max_features": ["sqrt", "log2", None]},
        },
    ]


def _safe_probability(estimator, X):
    if hasattr(estimator, "predict_proba"):
        try:
            p = estimator.predict_proba(X)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1]
        except Exception:
            pass
    if hasattr(estimator, "decision_function"):
        try:
            s = estimator.decision_function(X)
            return 1.0 / (1.0 + np.exp(-s))
        except Exception:
            pass
    return None


def binary_metrics(estimator, X_test, y_test):
    start = time.perf_counter()
    pred = estimator.predict(X_test)
    pred_time = time.perf_counter() - start
    y_true = np.asarray(y_test).astype(int)
    y_pred = np.asarray(pred).astype(int)
    prob = _safe_probability(estimator, X_test)

    hard_mse = mean_squared_error(y_true, y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "hard_label_mae": mean_absolute_error(y_true, y_pred),
        "hard_label_mse": hard_mse,
        "hard_label_rmse": float(np.sqrt(hard_mse)),
        "mae": np.nan, "mse": np.nan, "rmse": np.nan, "mape": np.nan,
        "roc_auc": np.nan, "brier_score": np.nan, "log_loss": np.nan,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], target_names=BINARY_LABELS, zero_division=0),
        "prediction_time_sec": pred_time,
        "y_pred": y_pred.tolist(),
        "probability_1": None if prob is None else np.asarray(prob).astype(float).tolist(),
    }
    if prob is not None and len(np.unique(y_true)) == 2:
        prob = np.clip(np.asarray(prob).astype(float), 1e-6, 1 - 1e-6)
        try:
            out["roc_auc"] = roc_auc_score(y_true, prob)
            out["brier_score"] = brier_score_loss(y_true, prob)
            out["log_loss"] = log_loss(y_true, np.column_stack([1 - prob, prob]), labels=[0, 1])
            # Regression-style diagnostics on predicted probability vs binary outcome.
            # These are most interpretable for Logistic Regression; app only displays them for LR.
            prob_mse = mean_squared_error(y_true, prob)
            out["mae"] = mean_absolute_error(y_true, prob)
            out["mse"] = prob_mse
            out["rmse"] = float(np.sqrt(prob_mse))
            out["mape"] = float(np.mean(np.abs(y_true - prob) / np.maximum(np.abs(y_true), 1)))
        except Exception:
            pass
    return out

def temporal_split(data, features, target="home_win_binary", train_until_year=2018, test_year=2022):
    df = data.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["year"] = df["match_date"].dt.year
    df = df.dropna(subset=[target])
    train_df = df[df["year"] <= int(train_until_year)].copy()
    test_df = df[df["year"] == int(test_year)].copy()
    if train_df.empty or test_df.empty or train_df[target].nunique() < 2 or test_df[target].nunique() < 2:
        X = df[features].fillna(0)
        y = df[target].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        return X_train, X_test, y_train, y_test, {"mode": "Fallback stratified random split", "train_size": len(X_train), "test_size": len(X_test), "reason": "Không đủ dữ liệu temporal split."}
    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)
    y_train = train_df[target].astype(int)
    y_test = test_df[target].astype(int)
    return X_train, X_test, y_train, y_test, {
        "mode": "Temporal split", "train_until_year": train_until_year, "test_year": test_year,
        "train_size": len(X_train), "test_size": len(X_test),
        "train_years": f"{int(train_df['year'].min())}-{int(train_df['year'].max())}", "test_years": str(test_year),
        "train_label_counts": y_train.value_counts().sort_index().to_dict(), "test_label_counts": y_test.value_counts().sort_index().to_dict(),
    }


def train_eval_metrics(estimator, X_train, y_train):
    pred = estimator.predict(X_train)
    return {
        "train_accuracy": accuracy_score(y_train, pred),
        "train_f1_macro": f1_score(y_train, pred, average="macro", zero_division=0),
    }


def best_cv_train_score(cv_results):
    if cv_results is None or cv_results.empty or "mean_train_score" not in cv_results.columns:
        return np.nan
    if "rank_test_score" in cv_results.columns:
        top = cv_results.sort_values("rank_test_score").iloc[0]
    else:
        top = cv_results.iloc[0]
    return float(top["mean_train_score"])


def fit_one(config, X_train, y_train, X_test, y_test, n_iter=10, scoring="f1_macro"):
    counts = pd.Series(y_train).value_counts()
    min_count = int(counts.min()) if not counts.empty else 0
    pipe_settings = dict(config["pipeline"])
    if pipe_settings.get("use_smote", False):
        if min_count < 2:
            pipe_settings["use_smote"] = False
        else:
            pipe_settings["smote_k_neighbors"] = max(1, min(3, min_count - 1))
    pipe = make_pipeline(config["model"], **pipe_settings)
    start = time.perf_counter()
    if config.get("param_grid"):
        n_candidates = 1
        for values in config["param_grid"].values():
            n_candidates *= len(values)
        search = RandomizedSearchCV(
            pipe, config["param_grid"], n_iter=min(n_iter, n_candidates), scoring=scoring,
            cv=cv_object(y_train), n_jobs=1, random_state=RANDOM_STATE, refit=True,
            return_train_score=True, error_score=0,
        )
        search.fit(X_train, y_train)
        estimator = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_
        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    else:
        estimator = pipe.fit(X_train, y_train)
        best_params = {"fixed_setting": config["main_hyperparameter"]}
        best_cv_score = np.nan
        cv_results = pd.DataFrame()
    train_time = time.perf_counter() - start
    m = binary_metrics(estimator, X_test, y_test)
    train_metrics = train_eval_metrics(estimator, X_train, y_train)
    best_cv_train = best_cv_train_score(cv_results)
    row = {
        "model_id": config["model_id"], "algorithm": config["algorithm"], "version": config["version"],
        "selection_method": config["selection_method"], "main_hyperparameter": config["main_hyperparameter"],
        "best_cv_score": best_cv_score,
        "best_cv_train_score": best_cv_train,
        "cv_train_gap": best_cv_train - best_cv_score if pd.notna(best_cv_train) and pd.notna(best_cv_score) else np.nan,
        "train_accuracy": train_metrics["train_accuracy"],
        "train_f1_macro": train_metrics["train_f1_macro"],
        "train_test_f1_gap": train_metrics["train_f1_macro"] - m["f1_macro"],
        "accuracy": m["accuracy"], "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
        "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"],
        "roc_auc": m["roc_auc"], "brier_score": m["brier_score"], "log_loss": m["log_loss"],
        "hard_label_mae": m["hard_label_mae"], "hard_label_mse": m["hard_label_mse"], "hard_label_rmse": m["hard_label_rmse"],
        "mae": m["mae"] if config["algorithm"] == "Logistic Regression" else np.nan,
        "mse": m["mse"] if config["algorithm"] == "Logistic Regression" else np.nan,
        "rmse": m["rmse"] if config["algorithm"] == "Logistic Regression" else np.nan,
        "mape": m["mape"] if config["algorithm"] == "Logistic Regression" else np.nan,
        "train_time_sec": train_time, "prediction_time_sec": m["prediction_time_sec"], "best_params": best_params,
    }
    detail = {"estimator": estimator, "confusion_matrix": m["confusion_matrix"], "classification_report": m["classification_report"], "cv_results": cv_results, "y_pred": m["y_pred"], "probability_1": m["probability_1"]}
    return row, detail


def train_all(data, features, target="home_win_binary", n_iter=10, scoring="f1_macro", train_until_year=2018, test_year=2022):
    X_train, X_test, y_train, y_test, split_info = temporal_split(data, features, target, train_until_year, test_year)
    rows, details = [], {}
    best_score, best_model_id, best_estimator = -np.inf, None, None
    for cfg in binary_model_plan():
        row, detail = fit_one(cfg, X_train, y_train, X_test, y_test, n_iter=n_iter, scoring=scoring)
        rows.append(row)
        details[row["model_id"]] = detail
        score = row.get(scoring, row["f1_macro"]) if scoring in row else row["f1_macro"]
        if row["f1_macro"] > best_score:
            best_score = row["f1_macro"]
            best_model_id = row["model_id"]
            best_estimator = detail["estimator"]
    res = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return res, details, split_info, best_model_id, best_estimator, (X_train, X_test, y_train, y_test)
