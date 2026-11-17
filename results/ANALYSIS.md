# Phân tích: vì sao mô hình chưa vượt Baseline

Sinh bởi `python models/analysis.py`. **Mọi lựa chọn (feature, siêu tham số, ngưỡng) chỉ dựa trên Train/CV theo ngày có purge; Test chỉ dùng để báo cáo một lần.**

## 1. Baseline (đã sửa)
Lớp đa số lấy từ **Train** (Giảm/Đứng). Trên Test: Accuracy 57.3%, Balanced Acc 50%, Precision/Recall (Tăng) 0%, **F1-macro 36.4%** (trước đây gán nhầm 0.5).

## 2. Kết quả trên Test (đánh giá một lần)
| | Accuracy | Balanced Acc | F1-macro | ROC-AUC |
|---|---|---|---|---|
| Baseline | 0.573 | 0.500 | 0.364 | – |
| SVM (RBF), 3 feature, C=0.1, ngưỡng 0.5 | 0.524 | 0.514 | 0.514 | 0.521 |
| LogReg 12 feature (mô hình triển khai) | 0.511 | 0.498 | 0.498 | 0.514 |

- **Accuracy chưa mô hình nào vượt 57.3%**; nhưng accuracy của baseline cao chỉ vì lệch lớp (Test có 57% phiên Giảm/Đứng). Về F1-macro, mọi mô hình đều cao hơn baseline (0.36) vì baseline không bắt được lớp Tăng.
- ROC-AUC Test ≈ 0.52, khoảng tin cậy 95% (bootstrap theo ngày) **[0.467; 0.583] chứa 0.5** → chưa chứng minh được khả năng dự đoán hơn đoán ngẫu nhiên.

## 3. Nguyên nhân
1. **Tín hiệu rất yếu**: AUC đơn biến của từng feature trong CV chỉ 0.47–0.54 (`analysis_feature_contribution.csv`). Hướng giá một phiên gần như ngẫu nhiên (thị trường hiệu quả ở mức yếu).
2. **Dư feature gây nhiễu**: 12 feature tương quan mạnh (ma_ratio, close_vs_ma20, ret5, rsi14…). CV AUC giảm từ 0.522 (3 feature) xuống 0.503 (12 feature) – thêm feature chỉ thêm overfit (`analysis_feature_selection.csv`).
3. **Dữ liệu nhỏ**: Train 3.810 mẫu (≈381 ngày × 10 mã) nhưng các mã cùng ngày tương quan mạnh nên số mẫu độc lập thực tế thấp hơn nhiều.
4. **Biến động giữa các fold**: độ lệch chuẩn AUC giữa các fold ≈ 0.04–0.06, lớn hơn mức chênh giữa các mô hình → khác biệt giữa mô hình chủ yếu là nhiễu.
5. **Đổi chế độ thị trường**: Test (từ 29/06/2026) có tỷ lệ Tăng 42.7% so với 47.3% ở Train.

## 4. Các bước đã thực hiện (chỉ trên Train/CV)
| Bước | File kết quả | Kết luận |
|---|---|---|
| Đóng góp feature (permutation + AUC đơn biến) | `analysis_feature_contribution.csv/.png` | `ret_lag1`, `rsi14`, `return`, `ret5` đóng góp dương rất nhỏ; 8/12 feature có đóng góp ≤ 0 |
| Chọn đặc trưng top-k | `analysis_feature_selection.csv` | k=3 cho CV AUC cao nhất |
| Tối ưu siêu tham số (ROC-AUC, CV) | `analysis_hyperparameters.csv` | SVM C=0.1 tốt nhất (CV AUC 0.544) |
| Điều chỉnh ngưỡng (OOF F1-macro) | `analysis_threshold.csv/.png`, `models/model_meta.json` | SVM: F1 OOF 0.496→0.524 (ngưỡng 0.48) nhưng Test không cải thiện (0.514→0.506); LogReg triển khai giữ 0.50 |
| Tổng quát hóa | `analysis_final_test.csv`, `analysis_per_stock.csv`, `analysis_summary.json` | Accuracy Test theo mã dao động 0.42–0.58 |

## 5. Kết luận trung thực
Sau khi tối ưu đúng quy trình, mô hình vẫn **chưa chứng minh vượt baseline** về Accuracy và AUC chưa khác 0.5 có ý nghĩa thống kê. Đây là kết quả hợp lý cho bài toán dự đoán hướng giá 1 phiên từ chỉ báo kỹ thuật; báo cáo nên trình bày như vậy thay vì tinh chỉnh theo Test. Hướng cải thiện: thêm dữ liệu/ thời gian dài hơn, đặc trưng thị trường chung (VN-Index), ngành, tin tức; hoặc đổi bài toán (dự đoán nhiều phiên, biên độ lớn).
