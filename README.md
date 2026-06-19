# Hệ thống dự đoán kết quả trận đấu World Cup

Ứng dụng Streamlit hỗ trợ dự đoán kết quả trận đấu World Cup theo bài toán phân loại nhị phân 0/1, có chọn biến tự động, tối ưu siêu tham số và đối chiếu kết quả trên năm kiểm tra.

## Điểm chính

- App tự đọc toàn bộ dữ liệu CSV đã đóng gói trong thư mục `data/`, không cần Supabase và không cần upload thủ công.
- Bài toán chính chuyển sang nhị phân:
  - `home_win_binary`: `1 = Home win`, `0 = Not home win`.
  - Có lựa chọn phụ: `favorite_win_binary`, `not_draw_binary`.
- Feature engineering theo nguyên tắc thời gian để giảm data leakage:
  - recent form: chỉ lấy trận quốc tế có `Date < match_date`.
  - historical/standing: chỉ dùng dữ liệu `wc_year < current_year`.
  - không dùng `goals.csv`, tỉ số thật, đội thắng hoặc thông tin sau trận làm feature.
- Tự chọn biến bằng:
  - p-value/F-score ranking;
  - L1 Logistic Regression;
  - RFECV tùy chọn;
  - voting giữa compact set, p-value, L1 và RFECV.
- So sánh nhiều nhóm mô hình phân loại: Logistic Regression, SVM, Decision Tree, Gradient Boosting và Random Forest.
- Có trang đối chiếu kết quả từ các mô hình với dữ liệu của năm kiểm tra: thực tế vs dự đoán.

## Dữ liệu dùng trong app

- `old/matches.csv`
- `old/team_appearances.csv`
- `old/teams_rows.csv`
- `enhanced/wc_matches_historical.csv`
- `enhanced/wc_team_appearances.csv`
- `additional/international_matches.csv`
- `additional/world_cup_host.csv`
- `additional/tournament_standings.csv`

## Các nhóm biến chính

- Sức mạnh trước trận: ELO, ELO difference, favorite flag.
- Lịch sử World Cup: win rate, points per match, goal difference per match, số lần tham dự, titles, pedigree score.
- Phong độ quốc tế gần đây: last10 win rate, points per match, goal difference, goals for/against.
- Phong độ trận chính thức: competitive form sau khi loại Friendly.
- Bối cảnh trận đấu: knockout, host advantage, same confederation.
- Thành tích lịch sử: best position, average position, top4/top8/champion counts trước năm dự đoán.

## Chỉ số đánh giá

Các model phân loại in:

- Accuracy
- Precision macro
- Recall macro
- F1 macro
- F1 weighted
- ROC AUC
- Brier score
- Log loss
- Confusion matrix
- Classification report
- Training time
- Prediction time

Các chỉ số MAE/MSE/RMSE/MAPE chỉ hiển thị riêng cho Logistic Regression, theo hướng sai số xác suất giữa xác suất dự đoán lớp 1 và nhãn 0/1 thật. Với bài toán phân loại, F1 macro, ROC AUC và confusion matrix vẫn là chỉ số chính.

## Cách chạy

```bash
pip install -r requirements.txt
streamlit run app.py
```
