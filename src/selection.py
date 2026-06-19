import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

FORBIDDEN_PATTERNS = [
    "score", "goals", "goal_", "winning_team", "losing_team", "result", "target", "penalt", "aet",
]
SAFE_GOAL_FEATURE_EXCEPTIONS = [
    "goal_diff", "goals_for", "goals_against", "goal_difference", "gd", "qualifying_gf", "qualifying_ga",
]

COMPACT_FEATURES = [
    "elo_diff", "abs_elo_diff", "home_is_elo_favorite", "fifa_rank_advantage",
    "diff_recent_comp_points_pm_last10", "diff_recent_comp_goal_diff_last10", "diff_recent_comp_win_rate_last10",
    "diff_recent_comp_goals_for_last10", "diff_recent_comp_goals_against_last10",
    "diff_recent_intl_points_pm_last10", "diff_recent_intl_goal_diff_last10", "diff_recent_intl_win_rate_last10",
    "diff_hist_points_pm", "diff_hist_goal_diff_pm", "diff_hist_win_rate", "diff_pedigree_score",
    "diff_hist_titles", "host_advantage", "is_knockout", "same_confederation",
    "standing_best_position_advantage", "standing_avg_position_advantage",
]


def numeric_candidate_features(df, model_columns):
    cols = []
    for c in model_columns:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def leakage_audit(features):
    rows = []
    for f in features:
        low = f.lower()
        suspicious = False
        reason = ""
        for pat in FORBIDDEN_PATTERNS:
            if pat in low:
                # allow pre-match and historical goal features that are intentionally built before match.
                if any(exc in low for exc in SAFE_GOAL_FEATURE_EXCEPTIONS) and not any(x in low for x in ["home_team_score", "away_team_score", "home_goals", "away_goals"]):
                    continue
                suspicious = True
                reason = f"Tên biến chứa '{pat}', cần kiểm tra có phải thông tin sau trận không."
                break
        rows.append({"feature": f, "suspicious": suspicious, "reason": reason})
    return pd.DataFrame(rows)


def f_pvalue_table(X, y):
    X2 = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    try:
        F, p = f_classif(X2, y)
    except Exception:
        F = np.zeros(X2.shape[1])
        p = np.ones(X2.shape[1])
    tab = pd.DataFrame({"feature": X2.columns, "f_score": F, "p_value": p})
    tab["f_score"] = tab["f_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tab["p_value"] = tab["p_value"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    tab["minus_log10_p"] = -np.log10(np.maximum(tab["p_value"], 1e-300))
    return tab.sort_values(["p_value", "f_score"], ascending=[True, False]).reset_index(drop=True)


def l1_selected_features(X, y, C=0.3):
    X2 = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])
    try:
        pipe.fit(X2, y)
        coef = np.abs(pipe.named_steps["lr"].coef_).ravel()
        out = pd.DataFrame({"feature": X2.columns, "abs_coef_l1": coef})
        out = out.sort_values("abs_coef_l1", ascending=False).reset_index(drop=True)
        return out[out["abs_coef_l1"] > 1e-8]
    except Exception:
        return pd.DataFrame(columns=["feature", "abs_coef_l1"])


def rfecv_selected_features(X, y, max_start_features=35):
    X2 = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if X2.shape[1] > max_start_features:
        ptab = f_pvalue_table(X2, y).head(max_start_features)
        X2 = X2[ptab["feature"].tolist()]
    min_count = pd.Series(y).value_counts().min()
    cv = StratifiedKFold(n_splits=max(2, min(5, int(min_count))), shuffle=True, random_state=RANDOM_STATE)
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])
    try:
        # RFECV cannot directly read coef_ from a Pipeline in some sklearn versions.
        X_scaled = StandardScaler().fit_transform(X2)
        lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
        selector = RFECV(lr, step=1, cv=cv, scoring="f1_macro", min_features_to_select=min(5, X2.shape[1]))
        selector.fit(X_scaled, y)
        out = pd.DataFrame({"feature": X2.columns, "rfecv_selected": selector.support_, "rfecv_rank": selector.ranking_})
        return out.sort_values(["rfecv_selected", "rfecv_rank"], ascending=[False, True]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "rfecv_selected", "rfecv_rank"])


def auto_select_features(X_train, y_train, candidate_features, p_threshold=0.20, max_features=24, use_rfecv=False):
    X = X_train[candidate_features].copy()
    ptab = f_pvalue_table(X, y_train)
    p_selected = ptab[ptab["p_value"] <= p_threshold]["feature"].tolist()
    if not p_selected:
        p_selected = ptab.head(min(15, len(ptab)))["feature"].tolist()
    l1tab = l1_selected_features(X, y_train)
    l1_selected = l1tab["feature"].tolist()
    rfetab = pd.DataFrame(columns=["feature", "rfecv_selected", "rfecv_rank"])
    rfe_selected = []
    if use_rfecv:
        rfetab = rfecv_selected_features(X, y_train)
        if not rfetab.empty:
            rfe_selected = rfetab[rfetab["rfecv_selected"] == True]["feature"].tolist()
    compact = [f for f in COMPACT_FEATURES if f in candidate_features]

    votes = {}
    sources = {}
    for source, feats in [("compact", compact), ("p_value", p_selected), ("l1", l1_selected), ("rfecv", rfe_selected)]:
        for f in feats:
            votes[f] = votes.get(f, 0) + 1
            sources.setdefault(f, []).append(source)
    vote_df = pd.DataFrame([{"feature": f, "votes": v, "sources": ", ".join(sources.get(f, []))} for f, v in votes.items()])
    if vote_df.empty:
        selected = ptab.head(min(max_features, len(ptab)))["feature"].tolist()
        vote_df = pd.DataFrame({"feature": selected, "votes": 1, "sources": "fallback_p_value"})
    else:
        vote_df = vote_df.merge(ptab[["feature", "p_value", "f_score"]], on="feature", how="left")
        vote_df = vote_df.sort_values(["votes", "p_value", "f_score"], ascending=[False, True, False]).reset_index(drop=True)
        selected = vote_df.head(max_features)["feature"].tolist()
    return selected, ptab, l1tab, rfetab, vote_df


def permutation_table(estimator, X_test, y_test, scoring="f1_macro", n_repeats=8):
    try:
        r = permutation_importance(estimator, X_test.fillna(0), y_test, scoring=scoring, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1)
        return pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
