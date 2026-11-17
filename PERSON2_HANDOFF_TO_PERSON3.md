# BIÊN BẢN BÀN GIAO — NGƯỜI 2 → NGƯỜI 3

## Nguồn dữ liệu
`visualizations/10_cleaned_data.csv` (file sạch do Người 1 bàn giao trong gói mới nhất).

## Hợp đồng dữ liệu
- X: 12 cột feature được khai báo trong `data/processed/feature_schema.json`.
- y: `label` (0/1).
- Metadata: `StockCode`, `Date`, `target_date` — chỉ dùng truy vết/thứ tự thời gian, không đưa vào X.
- Train/Test: `clf_train.csv`, `clf_test.csv`.
- Bản chuẩn hóa: `clf_train_scaled.csv`, `clf_test_scaled.csv`; scaler: `standard_scaler.joblib`.

## Quy tắc nhãn
`label=1` nếu Close phiên giao dịch tiếp theo của cùng mã > Close hiện tại; ngược lại `label=0`.

## Kết quả lần chạy hiện tại
- Số dòng dữ liệu sạch đầu vào: 4.980
- Số dòng sau tạo feature/label: 4.780
- Số mã cổ phiếu: 10
- Train: 3.810
- Test: 960
- Ngày cuối Train: 2026-06-25
- Ngày đầu Test: 2026-06-29
- Số ngày xuất hiện ở cả Train và Test: 0
- Phân phối label Train: 0 = 2.009; 1 = 1.801
- Phân phối label Test: 0 = 550; 1 = 410

## Các kiểm tra đã chạy
- Các feature được tính riêng theo StockCode.
- Không còn target Regression trong file bàn giao.
- Ngày Train/Test không giao nhau.
- Tất cả target_date của Train nằm trước ngày bắt đầu Test.
- Scaler fit trên Train; Test chỉ transform.
- Giá trị trung bình các feature đã scale trên Train xấp xỉ 0.

## Yêu cầu đối với Người 3
Không tự tạo lại feature/label hoặc chia tập. Đọc đúng hai file `clf_train.csv`, `clf_test.csv`. CV phải group theo ngày giao dịch; trong từng fold, scaler cần fit trong Pipeline trên fold-train để không rò rỉ vào fold-validation. Logistic Regression là mô hình chính theo phân công nhóm.

Lưu ý dữ liệu xử lý ngoại lệ: nếu có `DailyReturn_Winsorized`/`Volume_Winsorized` trong file Người 1, feature return/volume dùng cột điều chỉnh; nhãn vẫn dùng Close thực tế. Không tự winsorize lại trong bước modeling.
