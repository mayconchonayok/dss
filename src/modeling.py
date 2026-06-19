import time
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# XGBoost can break on some Mac machines because of native libomp issues.
# Keep the app stable for classroom demo by defaulting to scikit-learn GradientBoosting.
try:
    from xgboost import XGBClassifier  # optional, not used by default
except Exception:
    XGBClassifier = None
HAS_XGBOOST = False

RANDOM_STATE = 42
LABEL_NAMES = ["Home win", "Draw", "Away win"]


def make_pipeline(model, use_scaler=True, use_smote=True, smote_k_neighbors=3):
    steps = []
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k_neighbors)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def cv_object(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def search_model(pipe, param_grid, n_iter, scoring="f1_macro", cv_splits=5):
    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_object(cv_splits),
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
        return_train_score=True,
        error_score=0,
    )


def boosting_configs():
    if HAS_XGBOOST:
        return [
            {
                "model_id": "BST-1",
                "algorithm": "XGBoost",
                "version": "Baseline",
                "selection_method": "Basic setting",
                "helper_method": "None",
                "main_hyperparameter": "learning_rate=0.1",
                "stop_condition": "Stops after n_estimators boosting rounds",
                "model": XGBClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    objective="multi:softprob",
                    num_class=3,
                    random_state=RANDOM_STATE,
                ),
                "pipeline": {"use_scaler": False, "use_smote": True},
                "param_grid": None,
                "note": "Boosting baseline for tabular data.",
            },
            {
                "model_id": "BST-2",
                "algorithm": "XGBoost",
                "version": "Conservative learning",
                "selection_method": "Manual tuning",
                "helper_method": "Manual comparison",
                "main_hyperparameter": "learning_rate=0.05",
                "stop_condition": "Stops after n_estimators boosting rounds",
                "model": XGBClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    objective="multi:softprob",
                    num_class=3,
                    random_state=RANDOM_STATE,
                ),
                "pipeline": {"use_scaler": False, "use_smote": True},
                "param_grid": None,
                "note": "Smaller learning rate with more boosting rounds.",
            },
            {
                "model_id": "BST-3",
                "algorithm": "XGBoost",
                "version": "CV search",
                "selection_method": "RandomizedSearchCV + 5-fold CV",
                "helper_method": "Random search",
                "main_hyperparameter": "learning_rate, max_depth",
                "stop_condition": "Stops after selected n_estimators boosting rounds",
                "model": XGBClassifier(eval_metric="mlogloss", objective="multi:softprob", num_class=3, random_state=RANDOM_STATE),
                "pipeline": {"use_scaler": False, "use_smote": True},
                "param_grid": {
                    "model__n_estimators": [100, 150, 250, 350],
                    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "model__max_depth": [2, 3, 4, 5],
                    "model__subsample": [0.7, 0.9, 1.0],
                    "model__colsample_bytree": [0.7, 0.9, 1.0],
                },
                "note": "Boosting hyperparameters are selected by cross-validation.",
            },
        ]
    return [
        {
            "model_id": "BST-1",
            "algorithm": "GradientBoosting",
            "version": "Baseline",
            "selection_method": "Basic setting",
            "helper_method": "None",
            "main_hyperparameter": "learning_rate=0.1",
            "stop_condition": "Stops after selected boosting stages",
            "model": GradientBoostingClassifier(learning_rate=0.1, n_estimators=120, max_depth=3, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": None,
            "note": "Fallback boosting model from scikit-learn when XGBoost is unavailable.",
        },
        {
            "model_id": "BST-2",
            "algorithm": "GradientBoosting",
            "version": "Conservative learning",
            "selection_method": "Manual tuning",
            "helper_method": "Manual comparison",
            "main_hyperparameter": "learning_rate=0.05",
            "stop_condition": "Stops after selected boosting stages",
            "model": GradientBoostingClassifier(learning_rate=0.05, n_estimators=180, max_depth=3, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": None,
            "note": "Lower learning rate with more iterations.",
        },
        {
            "model_id": "BST-3",
            "algorithm": "GradientBoosting",
            "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV",
            "helper_method": "Random search",
            "main_hyperparameter": "learning_rate, max_depth",
            "stop_condition": "Stops after selected boosting stages",
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": {
                "model__n_estimators": [80, 120, 180, 250],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.9, 1.0],
            },
            "note": "Boosting hyperparameters are selected by cross-validation.",
        },
    ]


def model_plan():
    base = [
        {
            "model_id": "LR-1",
            "algorithm": "Logistic Regression",
            "version": "Baseline",
            "selection_method": "Default setting",
            "helper_method": "None",
            "main_hyperparameter": "C=1.0",
            "stop_condition": "Converges or reaches max_iter=500",
            "model": LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs"),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": None,
            "note": "Linear baseline for multiclass classification.",
        },
        {
            "model_id": "LR-2",
            "algorithm": "Logistic Regression",
            "version": "Manual regularization",
            "selection_method": "Manual tuning",
            "helper_method": "Manual comparison",
            "main_hyperparameter": "C=0.1",
            "stop_condition": "Converges or reaches max_iter=500",
            "model": LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", solver="lbfgs"),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": None,
            "note": "Stronger regularization than the baseline.",
        },
        {
            "model_id": "LR-3",
            "algorithm": "Logistic Regression",
            "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV",
            "helper_method": "Random search",
            "main_hyperparameter": "C",
            "stop_condition": "Converges or reaches max_iter=1000",
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                "model__solver": ["lbfgs"],
            },
            "note": "C is selected by cross-validation.",
        },
        {
            "model_id": "SVM-1",
            "algorithm": "SVM",
            "version": "Linear baseline",
            "selection_method": "Default linear kernel",
            "helper_method": "None",
            "main_hyperparameter": "kernel='linear'",
            "stop_condition": "Converges or reaches max_iter",
            "model": SVC(C=1.0, kernel="linear", class_weight="balanced", probability=True, max_iter=3000, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": None,
            "note": "Tests a linear decision boundary.",
        },
        {
            "model_id": "SVM-2",
            "algorithm": "SVM",
            "version": "RBF manual",
            "selection_method": "Manual kernel change",
            "helper_method": "Manual comparison",
            "main_hyperparameter": "kernel='rbf', gamma='scale'",
            "stop_condition": "Converges or reaches max_iter",
            "model": SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, max_iter=3000, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": None,
            "note": "Allows a non-linear decision boundary.",
        },
        {
            "model_id": "SVM-3",
            "algorithm": "SVM",
            "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV",
            "helper_method": "Random search",
            "main_hyperparameter": "C, gamma",
            "stop_condition": "Converges or reaches max_iter",
            "model": SVC(kernel="rbf", class_weight="balanced", probability=True, max_iter=3000, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": {
                "model__C": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1, 0.5],
                "model__kernel": ["rbf", "poly", "sigmoid"],
            },
            "note": "C and gamma are selected by cross-validation.",
        },
        {
            "model_id": "DT-1",
            "algorithm": "Decision Tree",
            "version": "Shallow tree",
            "selection_method": "Manual setting",
            "helper_method": "Manual comparison",
            "main_hyperparameter": "max_depth=3",
            "stop_condition": "Stops at max_depth or when no valid split remains",
            "model": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": None,
            "note": "Simple tree, lower variance.",
        },
        {
            "model_id": "DT-2",
            "algorithm": "Decision Tree",
            "version": "Deeper tree",
            "selection_method": "Manual setting",
            "helper_method": "Manual comparison",
            "main_hyperparameter": "max_depth=7, min_samples_leaf=2",
            "stop_condition": "Stops at max_depth or when no valid split remains",
            "model": DecisionTreeClassifier(max_depth=7, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": None,
            "note": "Deeper tree with a minimum leaf size to reduce overfitting risk.",
        },
        {
            "model_id": "DT-3",
            "algorithm": "Decision Tree",
            "version": "CV search",
            "selection_method": "RandomizedSearchCV + 5-fold CV",
            "helper_method": "Random search",
            "main_hyperparameter": "max_depth",
            "stop_condition": "Stops at selected max_depth or when no valid split remains",
            "model": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": {
                "model__max_depth": [2, 3, 4, 5, 7, 10, None],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__criterion": ["gini", "entropy"],
            },
            "note": "Tree depth is selected by cross-validation.",
        },
    ]
    return base + boosting_configs()


def classification_metrics(estimator, X_test, y_test):
    start = time.perf_counter()
    pred = estimator.predict(X_test)
    predict_time = time.perf_counter() - start
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            zero_division=0,
        ),
        "prediction_time_sec": predict_time,
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


def fit_one(config, X_train, y_train, X_test, y_test, n_iter=15, scoring="f1_macro"):
    class_counts = pd.Series(y_train).value_counts()
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0
    pipe_settings = dict(config["pipeline"])
    if pipe_settings.get("use_smote", False):
        if min_class_count < 2:
            pipe_settings["use_smote"] = False
        else:
            pipe_settings["smote_k_neighbors"] = max(1, min(3, min_class_count - 1))
    cv_splits = max(2, min(5, min_class_count))
    pipe = make_pipeline(config["model"], **pipe_settings)
    start = time.perf_counter()
    if config["param_grid"]:
        n_candidates = 1
        for values in config["param_grid"].values():
            n_candidates *= len(values)
        n_iter_eff = min(n_iter, n_candidates)
        fitted = search_model(pipe, config["param_grid"], n_iter=n_iter_eff, scoring=scoring, cv_splits=cv_splits)
        fitted.fit(X_train, y_train)
        estimator = fitted.best_estimator_
        best_params = fitted.best_params_
        best_cv_score = fitted.best_score_
        cv_results = pd.DataFrame(fitted.cv_results_).sort_values("rank_test_score")
    else:
        estimator = pipe.fit(X_train, y_train)
        best_params = {"fixed_setting": config["main_hyperparameter"]}
        best_cv_score = np.nan
        cv_results = pd.DataFrame()
    train_time = time.perf_counter() - start
    metrics = classification_metrics(estimator, X_test, y_test)
    train_metrics = train_eval_metrics(estimator, X_train, y_train)
    best_cv_train = best_cv_train_score(cv_results)
    row = {
        "model_id": config["model_id"],
        "algorithm": config["algorithm"],
        "version": config["version"],
        "selection_method": config["selection_method"],
        "helper_method": config["helper_method"],
        "main_hyperparameter": config["main_hyperparameter"],
        "stop_condition": config["stop_condition"],
        "best_cv_score": best_cv_score,
        "best_cv_train_score": best_cv_train,
        "cv_train_gap": best_cv_train - best_cv_score if pd.notna(best_cv_train) and pd.notna(best_cv_score) else np.nan,
        "train_accuracy": train_metrics["train_accuracy"],
        "train_f1_macro": train_metrics["train_f1_macro"],
        "train_test_f1_gap": train_metrics["train_f1_macro"] - metrics["f1_macro"],
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "train_time_sec": train_time,
        "prediction_time_sec": metrics["prediction_time_sec"],
        "best_params": best_params,
        "note": config["note"],
    }
    detail = {
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "cv_results": cv_results,
        "estimator": estimator,
    }
    return row, detail


def temporal_split(data, features, target="target", train_until_year=2018, test_year=2022):
    df = data.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["year"] = df["match_date"].dt.year
    train_df = df[df["year"] <= int(train_until_year)].copy()
    test_df = df[df["year"] == int(test_year)].copy()

    if train_df.empty or test_df.empty or train_df[target].nunique() < 2 or test_df[target].nunique() < 2:
        X = df[features].fillna(0)
        y = df[target].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        split_info = {
            "mode": "Fallback stratified random split",
            "reason": "Không đủ dữ liệu cho cách chia theo năm đã chọn.",
            "train_until_year": train_until_year,
            "test_year": test_year,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_years": "random",
            "test_years": "random",
        }
        return X_train, X_test, y_train, y_test, split_info

    X_train = train_df[features].fillna(0)
    y_train = train_df[target].astype(int)
    X_test = test_df[features].fillna(0)
    y_test = test_df[target].astype(int)
    split_info = {
        "mode": "Temporal split",
        "reason": "Train theo các trận trước hoặc bằng năm train_until_year, test trên test_year.",
        "train_until_year": train_until_year,
        "test_year": test_year,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_years": f"{int(train_df['year'].min())}-{int(train_df['year'].max())}",
        "test_years": str(test_year),
        "train_label_counts": y_train.value_counts().sort_index().to_dict(),
        "test_label_counts": y_test.value_counts().sort_index().to_dict(),
    }
    return X_train, X_test, y_train, y_test, split_info


def train_all(data, features, target="target", n_iter=15, scoring="f1_macro", train_until_year=2018, test_year=2022):
    X_train, X_test, y_train, y_test, split_info = temporal_split(
        data, features, target=target, train_until_year=train_until_year, test_year=test_year
    )

    rows = []
    details = {}
    best_estimator = None
    best_name = None
    best_score = -np.inf

    for config in model_plan():
        row, detail = fit_one(config, X_train, y_train, X_test, y_test, n_iter=n_iter, scoring=scoring)
        rows.append(row)
        details[row["model_id"]] = detail
        if row["f1_macro"] > best_score:
            best_score = row["f1_macro"]
            best_name = f'{row["model_id"]} - {row["algorithm"]}'
            best_estimator = detail["estimator"]

    result = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return result, best_name, best_estimator, details, split_info



def binary_classification_metrics(estimator, X_test, y_test):
    start = time.perf_counter()
    pred = estimator.predict(X_test)
    predict_time = time.perf_counter() - start
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=[0, 1],
            target_names=["Favorite not win", "Favorite win"],
            zero_division=0,
        ),
        "prediction_time_sec": predict_time,
    }


def binary_model_plan():
    return [
        {
            "model_id": "B-LR",
            "algorithm": "Logistic Regression",
            "model": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs"),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": None,
            "note": "Binary baseline."
        },
        {
            "model_id": "B-SVM-RBF",
            "algorithm": "SVM",
            "model": SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, max_iter=3000, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": None,
            "note": "Binary SVM with RBF kernel."
        },
        {
            "model_id": "B-SVM-CV",
            "algorithm": "SVM",
            "model": SVC(class_weight="balanced", probability=True, max_iter=3000, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": True, "use_smote": True},
            "param_grid": {
                "model__C": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1, 0.5],
                "model__kernel": ["rbf", "poly", "sigmoid"],
            },
            "note": "Binary SVM with expanded RandomizedSearchCV."
        },
        {
            "model_id": "B-DT",
            "algorithm": "Decision Tree",
            "model": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": None,
            "note": "Binary decision tree."
        },
        {
            "model_id": "B-GB",
            "algorithm": "GradientBoosting",
            "model": GradientBoostingClassifier(learning_rate=0.05, n_estimators=180, max_depth=3, random_state=RANDOM_STATE),
            "pipeline": {"use_scaler": False, "use_smote": True},
            "param_grid": None,
            "note": "Binary boosting fallback."
        },
    ]


def train_binary_favorite(data, features, train_until_year=2018, test_year=2022, n_iter=20, scoring="f1_macro"):
    df = data.dropna(subset=["favorite_win_binary"]).copy()
    X_train, X_test, y_train, y_test, split_info = temporal_split(
        df, features, target="favorite_win_binary", train_until_year=train_until_year, test_year=test_year
    )
    rows = []
    details = {}
    best_score = -np.inf
    best_estimator = None
    for cfg in binary_model_plan():
        class_counts = pd.Series(y_train).value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) else 0
        pipe_settings = dict(cfg["pipeline"])
        if pipe_settings.get("use_smote", False):
            if min_class_count < 2:
                pipe_settings["use_smote"] = False
            else:
                pipe_settings["smote_k_neighbors"] = max(1, min(3, min_class_count - 1))
        cv_splits = max(2, min(5, min_class_count))
        pipe = make_pipeline(cfg["model"], **pipe_settings)
        start = time.perf_counter()
        if cfg["param_grid"]:
            n_candidates = 1
            for values in cfg["param_grid"].values():
                n_candidates *= len(values)
            fitted = search_model(pipe, cfg["param_grid"], n_iter=min(n_iter, n_candidates), scoring=scoring, cv_splits=cv_splits)
            fitted.fit(X_train, y_train)
            estimator = fitted.best_estimator_
            best_params = fitted.best_params_
            best_cv_score = fitted.best_score_
            cv_results = pd.DataFrame(fitted.cv_results_).sort_values("rank_test_score")
        else:
            estimator = pipe.fit(X_train, y_train)
            best_params = {"fixed_setting": cfg["note"]}
            best_cv_score = np.nan
            cv_results = pd.DataFrame()
        train_time = time.perf_counter() - start
        m = binary_classification_metrics(estimator, X_test, y_test)
        row = {
            "model_id": cfg["model_id"],
            "algorithm": cfg["algorithm"],
            "best_cv_score": best_cv_score,
            "accuracy": m["accuracy"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "train_time_sec": train_time,
            "prediction_time_sec": m["prediction_time_sec"],
            "best_params": best_params,
            "note": cfg["note"],
        }
        rows.append(row)
        details[row["model_id"]] = {"confusion_matrix": m["confusion_matrix"], "classification_report": m["classification_report"], "cv_results": cv_results, "estimator": estimator}
        if row["f1_macro"] > best_score:
            best_score = row["f1_macro"]
            best_estimator = estimator
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True), details, split_info, best_estimator


def custom_config(algorithm, mode, params):
    """Build one configurable model for the single-algorithm demo page."""
    use_search = mode == "search"
    if algorithm == "Logistic Regression":
        if use_search:
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            grid = {
                "model__C": params.get("C_values", [0.01, 0.1, 1, 10]),
                "model__solver": params.get("solver_values", ["lbfgs"]),
            }
            main_hp = "C, solver"
            version = "RandomizedSearchCV"
            selection = "RandomizedSearchCV + 5-fold CV"
            helper = "Random search"
        else:
            C = float(params.get("C", 1.0))
            solver = params.get("solver", "lbfgs")
            model = LogisticRegression(C=C, solver=solver, max_iter=1000, class_weight="balanced")
            grid = None
            main_hp = f"C={C}, solver={solver}"
            version = "Manual setting"
            selection = "Manual hyperparameter setting"
            helper = "Manual comparison"
        return {
            "model_id": "CUSTOM-LR",
            "algorithm": "Logistic Regression",
            "version": version,
            "selection_method": selection,
            "helper_method": helper,
            "main_hyperparameter": main_hp,
            "stop_condition": "Converges or reaches max_iter=1000",
            "model": model,
            "pipeline": {"use_scaler": True, "use_smote": params.get("use_smote", True)},
            "param_grid": grid,
            "note": "Single-run Logistic Regression demo.",
        }

    if algorithm == "SVM":
        if use_search:
            model = SVC(kernel="rbf", class_weight="balanced", probability=True, max_iter=3000, random_state=RANDOM_STATE)
            grid = {
                "model__C": params.get("C_values", [0.1, 0.5, 1, 2, 5, 10, 20, 50]),
                "model__gamma": params.get("gamma_values", ["scale", "auto", 0.001, 0.01, 0.05, 0.1, 0.5]),
                "model__kernel": params.get("kernel_values", ["rbf", "poly", "sigmoid"]),
            }
            main_hp = "C, gamma, kernel"
            version = "RandomizedSearchCV"
            selection = "RandomizedSearchCV + 5-fold CV"
            helper = "Random search"
        else:
            C = float(params.get("C", 1.0))
            kernel = params.get("kernel", "rbf")
            gamma = params.get("gamma", "scale")
            model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight="balanced", probability=True, max_iter=3000, random_state=RANDOM_STATE)
            grid = None
            main_hp = f"C={C}, kernel={kernel}, gamma={gamma}"
            version = "Manual setting"
            selection = "Manual hyperparameter setting"
            helper = "Manual comparison"
        return {
            "model_id": "CUSTOM-SVM",
            "algorithm": "SVM",
            "version": version,
            "selection_method": selection,
            "helper_method": helper,
            "main_hyperparameter": main_hp,
            "stop_condition": "Converges or reaches max_iter",
            "model": model,
            "pipeline": {"use_scaler": True, "use_smote": params.get("use_smote", True)},
            "param_grid": grid,
            "note": "Single-run SVM demo.",
        }

    if algorithm == "Decision Tree":
        if use_search:
            model = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
            grid = {
                "model__max_depth": params.get("max_depth_values", [2, 3, 5, 7, 10, None]),
                "model__min_samples_leaf": params.get("min_samples_leaf_values", [1, 2, 5, 10]),
                "model__criterion": params.get("criterion_values", ["gini", "entropy"]),
            }
            main_hp = "max_depth, min_samples_leaf, criterion"
            version = "RandomizedSearchCV"
            selection = "RandomizedSearchCV + 5-fold CV"
            helper = "Random search"
        else:
            max_depth = params.get("max_depth", 5)
            max_depth = None if max_depth == "None" else int(max_depth)
            min_samples_leaf = int(params.get("min_samples_leaf", 1))
            criterion = params.get("criterion", "gini")
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, criterion=criterion, class_weight="balanced", random_state=RANDOM_STATE)
            grid = None
            main_hp = f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, criterion={criterion}"
            version = "Manual setting"
            selection = "Manual hyperparameter setting"
            helper = "Manual comparison"
        return {
            "model_id": "CUSTOM-DT",
            "algorithm": "Decision Tree",
            "version": version,
            "selection_method": selection,
            "helper_method": helper,
            "main_hyperparameter": main_hp,
            "stop_condition": "Stops at max_depth or when no valid split remains",
            "model": model,
            "pipeline": {"use_scaler": False, "use_smote": params.get("use_smote", True)},
            "param_grid": grid,
            "note": "Single-run Decision Tree demo.",
        }

    if algorithm == "Boosting":
        actual = "XGBoost" if HAS_XGBOOST else "GradientBoosting"
        if HAS_XGBOOST:
            if use_search:
                model = XGBClassifier(eval_metric="mlogloss", objective="multi:softprob", num_class=3, random_state=RANDOM_STATE)
                grid = {
                    "model__n_estimators": params.get("n_estimators_values", [100, 150, 250, 350]),
                    "model__learning_rate": params.get("learning_rate_values", [0.01, 0.05, 0.1]),
                    "model__max_depth": params.get("max_depth_values", [2, 3, 4, 5]),
                    "model__subsample": params.get("subsample_values", [0.7, 0.9, 1.0]),
                }
                main_hp = "learning_rate, max_depth, n_estimators"
                version = "RandomizedSearchCV"
                selection = "RandomizedSearchCV + 5-fold CV"
                helper = "Random search"
            else:
                lr = float(params.get("learning_rate", 0.1))
                max_depth = int(params.get("max_depth", 3))
                n_estimators = int(params.get("n_estimators", 150))
                model = XGBClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", objective="multi:softprob", num_class=3, random_state=RANDOM_STATE)
                grid = None
                main_hp = f"learning_rate={lr}, max_depth={max_depth}, n_estimators={n_estimators}"
                version = "Manual setting"
                selection = "Manual hyperparameter setting"
                helper = "Manual comparison"
        else:
            if use_search:
                model = GradientBoostingClassifier(random_state=RANDOM_STATE)
                grid = {
                    "model__n_estimators": params.get("n_estimators_values", [100, 150, 250, 350]),
                    "model__learning_rate": params.get("learning_rate_values", [0.01, 0.05, 0.1]),
                    "model__max_leaf_nodes": params.get("max_leaf_nodes_values", [15, 31, 63]),
                }
                main_hp = "learning_rate, max_leaf_nodes, n_estimators"
                version = "RandomizedSearchCV"
                selection = "RandomizedSearchCV + 5-fold CV"
                helper = "Random search"
            else:
                lr = float(params.get("learning_rate", 0.1))
                n_estimators = int(params.get("n_estimators", 120))
                max_depth = int(params.get("max_depth", 3))
                model = GradientBoostingClassifier(learning_rate=lr, n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE)
                grid = None
                main_hp = f"learning_rate={lr}, n_estimators={n_estimators}, max_depth={max_depth}"
                version = "Manual setting"
                selection = "Manual hyperparameter setting"
                helper = "Manual comparison"
        return {
            "model_id": "CUSTOM-BST",
            "algorithm": actual,
            "version": version,
            "selection_method": selection,
            "helper_method": helper,
            "main_hyperparameter": main_hp,
            "stop_condition": "Stops at selected boosting rounds / iterations",
            "model": model,
            "pipeline": {"use_scaler": False, "use_smote": params.get("use_smote", True)},
            "param_grid": grid,
            "note": "Single-run Boosting demo.",
        }

    raise ValueError(f"Unsupported algorithm: {algorithm}")
