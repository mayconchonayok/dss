import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import final_dataset, model_columns
from src.binary_modeling import train_all, temporal_split
from src.selection import (
    numeric_candidate_features, leakage_audit, auto_select_features,
    COMPACT_FEATURES, f_pvalue_table, l1_selected_features, permutation_table,
)

st.set_page_config(page_title="World Cup Match Outcome Prediction System", layout="wide")
st.title("Hệ thống dự đoán kết quả trận đấu World Cup")
st.caption("Ứng dụng phân loại nhị phân, chọn biến an toàn theo thời gian, tối ưu siêu tham số bằng cross-validation và đánh giá trên năm kiểm tra.")


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def text_to_bytes(text):
    return text.encode("utf-8")


@st.cache_data(show_spinner=False)
def load_bundled_data():
    base = os.path.join(os.path.dirname(__file__), "data")
    old_dir = os.path.join(base, "old")
    enh_dir = os.path.join(base, "enhanced")
    add_dir = os.path.join(base, "additional")

    old_matches = pd.read_csv(os.path.join(old_dir, "matches.csv"))
    old_team_appearances = pd.read_csv(os.path.join(old_dir, "team_appearances.csv"))
    old_teams = pd.read_csv(os.path.join(old_dir, "teams_rows.csv"))

    enhanced_matches = pd.read_csv(os.path.join(enh_dir, "wc_matches_historical.csv"))
    enhanced_appearances = pd.read_csv(os.path.join(enh_dir, "wc_team_appearances.csv"))
    international_matches = pd.read_csv(os.path.join(add_dir, "international_matches.csv")) if os.path.exists(os.path.join(add_dir, "international_matches.csv")) else pd.DataFrame()
    world_cup_host = pd.read_csv(os.path.join(add_dir, "world_cup_host.csv")) if os.path.exists(os.path.join(add_dir, "world_cup_host.csv")) else pd.DataFrame()
    tournament_standings = pd.read_csv(os.path.join(add_dir, "tournament_standings.csv")) if os.path.exists(os.path.join(add_dir, "tournament_standings.csv")) else pd.DataFrame()

    data = final_dataset(
        old_matches, old_team_appearances, old_teams,
        enhanced_matches=enhanced_matches,
        enhanced_appearances=enhanced_appearances,
        international_matches=international_matches,
        world_cup_host=world_cup_host,
        tournament_standings=tournament_standings,
    )
    data["home_win_binary"] = (data["target"].astype(int) == 0).astype(int)
    if "favorite_win_binary" not in data.columns or data["favorite_win_binary"].isna().all():
        fav_home = data.get("home_is_elo_favorite", pd.Series(1, index=data.index)).fillna(1).astype(int)
        home_win = (data["target"].astype(int) == 0)
        away_win = (data["target"].astype(int) == 2)
        data["favorite_win_binary"] = np.where(fav_home.eq(1), home_win, away_win).astype(int)
    data["not_draw_binary"] = (data["target"].astype(int) != 1).astype(int)
    raw = {
        "old_matches": old_matches,
        "old_team_appearances": old_team_appearances,
        "old_teams": old_teams,
        "enhanced_matches": enhanced_matches,
        "enhanced_appearances": enhanced_appearances,
        "international_matches": international_matches,
        "world_cup_host": world_cup_host,
        "tournament_standings": tournament_standings,
    }
    return data, raw


def label_dist(df, target):
    counts = df[target].value_counts().sort_index()
    return pd.DataFrame({"label": counts.index, "count": counts.values, "ratio": (counts / counts.sum()).round(4).values})


def render_cm(cm):
    return pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])


def split_message(info):
    if info.get("mode") == "Temporal split":
        st.success(f"Temporal split: train {info.get('train_years')} -> test {info.get('test_years')} | train={info.get('train_size')}, test={info.get('test_size')}")
    else:
        st.warning(f"{info.get('mode')}: {info.get('reason')} | train={info.get('train_size')}, test={info.get('test_size')}")


def select_features(mode, candidate_features, X_train, y_train, p_threshold, max_features, use_rfecv):
    if mode == "Compact recommended":
        selected = [f for f in COMPACT_FEATURES if f in candidate_features][:max_features]
        ptab = f_pvalue_table(X_train[selected], y_train) if selected else pd.DataFrame()
        return selected, ptab, pd.DataFrame(), pd.DataFrame(), pd.DataFrame({"feature": selected, "votes": 1, "sources": "compact"})
    if mode == "P-value only":
        ptab = f_pvalue_table(X_train[candidate_features], y_train)
        selected = ptab[ptab["p_value"] <= p_threshold].head(max_features)["feature"].tolist()
        if not selected:
            selected = ptab.head(max_features)["feature"].tolist()
        return selected, ptab, pd.DataFrame(), pd.DataFrame(), pd.DataFrame({"feature": selected, "votes": 1, "sources": "p_value"})
    if mode == "L1 Logistic only":
        ptab = f_pvalue_table(X_train[candidate_features], y_train)
        l1tab = l1_selected_features(X_train[candidate_features], y_train)
        selected = l1tab.head(max_features)["feature"].tolist() if not l1tab.empty else ptab.head(max_features)["feature"].tolist()
        return selected, ptab, l1tab, pd.DataFrame(), pd.DataFrame({"feature": selected, "votes": 1, "sources": "l1"})
    if mode == "All safe features":
        ptab = f_pvalue_table(X_train[candidate_features], y_train)
        return candidate_features, ptab, pd.DataFrame(), pd.DataFrame(), pd.DataFrame({"feature": candidate_features, "votes": 1, "sources": "all_safe"})
    selected, ptab, l1tab, rfetab, vote_df = auto_select_features(
        X_train, y_train, candidate_features,
        p_threshold=p_threshold, max_features=max_features, use_rfecv=use_rfecv,
    )
    return selected, ptab, l1tab, rfetab, vote_df


def actual_label_name(target_col, value):
    maps = {
        "home_win_binary": {0: "Not home win", 1: "Home win"},
        "favorite_win_binary": {0: "Favorite not win", 1: "Favorite win"},
        "not_draw_binary": {0: "Draw", 1: "Not draw"},
    }
    return maps.get(target_col, {0: "0", 1: "1"}).get(int(value), str(value))


try:
    data, raw = load_bundled_data()
except Exception as e:
    st.error("Không load được dữ liệu đóng gói.")
    st.exception(e)
    st.stop()

st.sidebar.header("Cấu hình")
st.sidebar.markdown("Cấu hình mục tiêu nhị phân, khoảng thời gian train/test và phương pháp chọn biến.")
target_choice = st.sidebar.selectbox(
    "Target nhị phân",
    ["home_win_binary", "favorite_win_binary", "not_draw_binary"],
    format_func=lambda x: {
        "home_win_binary": "1 = Home win, 0 = Not home win",
        "favorite_win_binary": "1 = ELO favorite wins, 0 = upset/draw",
        "not_draw_binary": "1 = Có đội thắng, 0 = Draw",
    }[x],
)
train_until = st.sidebar.number_input("Train đến năm", min_value=1930, max_value=2022, value=2018, step=4)
test_year = st.sidebar.number_input("Test năm", min_value=1930, max_value=2022, value=2022, step=4)
feature_mode = st.sidebar.selectbox("Chế độ chọn biến", ["Auto statistical selection", "Compact recommended", "P-value only", "L1 Logistic only", "All safe features"])
p_threshold = st.sidebar.slider("Ngưỡng p-value", 0.01, 0.50, 0.20, 0.01)
max_features = st.sidebar.slider("Số biến tối đa", 5, 50, 22, 1)
use_rfecv = st.sidebar.checkbox("Dùng RFECV trong Auto", value=False, help="Có thể chạy chậm nhưng chọn biến bằng CV.")
n_iter = st.sidebar.slider("Số tổ hợp RandomizedSearchCV", 3, 40, 10, 1)
scoring = st.sidebar.selectbox("Metric để CV chọn tham số", ["f1_macro", "accuracy", "roc_auc"], index=0)

candidate_features = numeric_candidate_features(data, model_columns())
aud = leakage_audit(candidate_features)
blocked = aud[aud["suspicious"]]["feature"].tolist()
candidate_features = [f for f in candidate_features if f not in blocked]

X_train_all, X_test_all, y_train, y_test, split_info = temporal_split(data, candidate_features, target_choice, train_until, test_year)
selected_features, ptab, l1tab, rfetab, vote_df = select_features(
    feature_mode, candidate_features, X_train_all, y_train, p_threshold, max_features, use_rfecv
)
selected_features = [f for f in selected_features if f in candidate_features]
if not selected_features:
    selected_features = candidate_features[:min(12, len(candidate_features))]

st.session_state["selected_features"] = selected_features
st.session_state["target_choice"] = target_choice

(tab_data, tab_select, tab_train, tab_test, tab_diag) = st.tabs([
    "1. Dữ liệu & mục tiêu",
    "2. Chọn biến",
    "3. Huấn luyện mô hình",
    "4. Đối chiếu năm test",
    "5. Chẩn đoán mô hình",
])

with tab_data:
    st.subheader("Tổng quan dữ liệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số trận sau xử lý", len(data))
    c2.metric("Candidate features an toàn", len(candidate_features))
    c3.metric("Selected features", len(selected_features))
    c4.metric("Test year", int(test_year))
    split_message(split_info)

    st.write("Phân bố target đang dùng")
    st.dataframe(label_dist(data, target_choice), use_container_width=True)

    st.write("Dữ liệu nguồn đã đóng gói")
    source_summary = pd.DataFrame([
        {"table": k, "rows": len(v), "cols": len(v.columns)} for k, v in raw.items()
    ])
    st.dataframe(source_summary, use_container_width=True)

    preview_cols = ["match_date", "home_team_name", "away_team_name", "home_team_score", "away_team_score", "target_name", target_choice] + selected_features[:12]
    st.write("Preview dataset sau feature engineering")
    st.dataframe(data[[c for c in preview_cols if c in data.columns]].head(40), use_container_width=True)

    st.markdown("""
**Ý nghĩa thiết kế:** ứng dụng tạo đặc trưng theo thời gian, chỉ dùng thông tin có trước trận; bổ sung nhóm mô hình Random Forest để so sánh với các mô hình phân loại khác; đồng thời hiển thị `log_loss`, `ROC AUC`, `Brier score` và bảng đối chiếu dự đoán trên năm kiểm tra.  
Các chỉ số MAE/MSE/RMSE/MAPE chỉ hiển thị riêng cho Logistic Regression theo hướng sai số xác suất, vì bài chính là phân loại 0/1 chứ không phải hồi quy số bàn.
""")

with tab_select:
    st.subheader("Tự động chọn biến có ý nghĩa")
    st.write("Feature mode:", feature_mode)
    st.write("Selected features")
    st.dataframe(pd.DataFrame({"feature": selected_features}), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("P-value / F-score ranking")
        st.dataframe(ptab.head(40), use_container_width=True)
    with col2:
        st.write("Voting sources")
        st.dataframe(vote_df.head(40), use_container_width=True)

    if not l1tab.empty:
        st.write("L1 Logistic selected features")
        st.dataframe(l1tab.head(40), use_container_width=True)
    if not rfetab.empty:
        st.write("RFECV selected features")
        st.dataframe(rfetab.head(40), use_container_width=True)

    st.download_button("Tải selected_features.csv", df_to_csv_bytes(pd.DataFrame({"feature": selected_features})), "selected_features.csv", "text/csv")
    st.download_button("Tải pvalue_ranking.csv", df_to_csv_bytes(ptab), "pvalue_ranking.csv", "text/csv")

with tab_train:
    st.subheader("Huấn luyện mô hình")
    st.info("Ứng dụng so sánh nhiều nhóm mô hình phân loại: Logistic Regression, SVM, Decision Tree, Gradient Boosting và Random Forest.")
    st.write("Target:", target_choice)
    st.write("Số biến:", len(selected_features))
    run = st.button("Chạy mô hình", type="primary")
    if run:
        with st.spinner("Đang train mô hình..."):
            result, details, split_info2, best_id, best_estimator, split_pack = train_all(
                data, selected_features, target=target_choice, n_iter=n_iter, scoring=scoring,
                train_until_year=train_until, test_year=test_year,
            )
        st.session_state["result"] = result
        st.session_state["details"] = details
        st.session_state["best_id"] = best_id
        st.session_state["best_estimator"] = best_estimator
        st.session_state["split_pack"] = split_pack
        st.session_state["split_info2"] = split_info2
        joblib.dump({"model": best_estimator, "features": selected_features, "target": target_choice}, os.path.join(os.path.dirname(__file__), "best_binary_model.joblib"))

    if "result" in st.session_state:
        result = st.session_state["result"]
        split_message(st.session_state.get("split_info2", split_info))
        base_cols = [
            "model_id", "algorithm", "version", "selection_method",
            "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
            "roc_auc", "brier_score", "log_loss",
            "train_f1_macro", "train_test_f1_gap", "best_cv_score", "best_cv_train_score", "cv_train_gap",
            "train_time_sec", "prediction_time_sec", "best_params",
        ]
        st.write("Bảng kết quả chính")
        st.dataframe(result[[c for c in base_cols if c in result.columns]], use_container_width=True)

        overfit_cols = ["model_id", "algorithm", "version", "train_f1_macro", "f1_macro", "train_test_f1_gap", "best_cv_score", "best_cv_train_score", "cv_train_gap"]
        st.write("Kiểm tra khoảng cách train/CV/test để nhận diện overfitting")
        st.caption("train_test_f1_gap = F1 train - F1 test. cv_train_gap = mean train score - mean CV score của cấu hình được CV chọn. Gap càng lớn thì nguy cơ overfit càng cao.")
        st.dataframe(result[[c for c in overfit_cols if c in result.columns]], use_container_width=True)

        st.bar_chart(result.set_index("model_id")[[c for c in ["accuracy", "f1_macro", "roc_auc"] if c in result.columns]])
        best = result.iloc[0]
        st.success(f"Model tốt nhất theo F1 macro: {best['model_id']} - {best['algorithm']} | F1={best['f1_macro']:.4f}, Acc={best['accuracy']:.4f}")

        lr_diag = result[result["algorithm"].eq("Logistic Regression")][["model_id", "mae", "mse", "rmse", "mape", "brier_score", "log_loss"]]
        st.write("Chỉ số kiểu hồi quy/xác suất chỉ cho Logistic Regression")
        st.caption("MAE/MSE/RMSE/MAPE ở đây đo sai số giữa xác suất dự đoán lớp 1 và nhãn 0/1 thật. Với bài toán phân loại, F1/ROC AUC/confusion matrix vẫn là chỉ số chính.")
        st.dataframe(lr_diag, use_container_width=True)

        st.download_button("Tải model_results.csv", df_to_csv_bytes(result), "model_results.csv", "text/csv")
    else:
        st.info("Bấm nút chạy để train mô hình.")

with tab_test:
    st.subheader(f"Đối chiếu kết quả từ các mô hình với dữ liệu năm test {int(test_year)}")
    if "result" not in st.session_state:
        st.info("Cần train model trước ở tab 3.")
    else:
        result = st.session_state["result"]
        model_id = st.selectbox("Chọn model để đối chiếu", result["model_id"].tolist(), index=0)
        detail = st.session_state["details"][model_id]
        X_train, X_test, y_train2, y_test2 = st.session_state["split_pack"]
        meta_cols = ["match_date", "home_team_name", "away_team_name", "home_team_score", "away_team_score", "target_name", target_choice]
        meta = data.loc[X_test.index, [c for c in meta_cols if c in data.columns]].copy()
        preds = np.array(detail.get("y_pred", []), dtype=int)
        probs = detail.get("probability_1")
        if len(preds) != len(meta):
            preds = detail["estimator"].predict(X_test[selected_features])
        meta["actual_label"] = [actual_label_name(target_choice, v) for v in y_test2]
        meta["pred_label"] = [actual_label_name(target_choice, v) for v in preds]
        meta["correct"] = (np.asarray(y_test2).astype(int) == preds).astype(int)
        if probs is not None and len(probs) == len(meta):
            meta["prob_class_1"] = np.round(probs, 4)
        meta = meta.sort_values("match_date")
        st.metric("Accuracy trên bảng đang xem", f"{meta['correct'].mean():.2%}")
        st.dataframe(meta, use_container_width=True)
        st.download_button("Tải test_year_predictions.csv", df_to_csv_bytes(meta), "test_year_predictions.csv", "text/csv")

        wrong = meta[meta["correct"] == 0]
        st.write("Các trận dự đoán sai")
        st.dataframe(wrong, use_container_width=True)

with tab_diag:
    st.subheader("Chẩn đoán mô hình, kiểm tra leakage và độ quan trọng biến")
    st.write("Leakage audit trên candidate features")
    st.dataframe(leakage_audit(candidate_features + ["home_team_score", "away_team_score", "winning_team", "goals.csv"]), use_container_width=True)

    if "result" not in st.session_state:
        st.info("Cần train model trước để xem confusion matrix và permutation importance.")
    else:
        result = st.session_state["result"]
        model_id = st.selectbox("Chọn model", result["model_id"].tolist(), index=0, key="diag_model")
        detail = st.session_state["details"][model_id]
        st.write("Confusion matrix")
        cm_df = render_cm(detail["confusion_matrix"])
        st.dataframe(cm_df, use_container_width=True)
        st.text("Classification report")
        st.text(detail["classification_report"])

        diag_cols = ["model_id", "algorithm", "version", "train_f1_macro", "f1_macro", "train_test_f1_gap", "best_cv_score", "best_cv_train_score", "cv_train_gap"]
        st.write("Overfitting diagnostics của model đang chọn")
        st.dataframe(result[result["model_id"].eq(model_id)][[c for c in diag_cols if c in result.columns]], use_container_width=True)

        X_train, X_test, y_train2, y_test2 = st.session_state["split_pack"]
        perm = permutation_table(detail["estimator"], X_test[selected_features], y_test2, scoring="f1_macro")
        st.write("Permutation importance")
        st.dataframe(perm.head(30), use_container_width=True)
        if not perm.empty:
            st.bar_chart(perm.head(15).set_index("feature")[["importance_mean"]])
        st.download_button("Tải confusion_matrix.csv", df_to_csv_bytes(cm_df.reset_index(names="true_label")), "confusion_matrix.csv", "text/csv")
        st.download_button("Tải classification_report.txt", text_to_bytes(detail["classification_report"]), "classification_report.txt", "text/plain")
        st.download_button("Tải permutation_importance.csv", df_to_csv_bytes(perm), "permutation_importance.csv", "text/csv")
