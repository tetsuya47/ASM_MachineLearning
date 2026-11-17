# ASM_MachineLearning — Pipeline Người 2: Feature Engineering & Classification

## 1. Phạm vi trách nhiệm
Thư mục này thực hiện phần việc của **Người thứ 2**: nhận dữ liệu sạch từ Người 1, tạo đặc trưng, tạo nhãn Classification, chia Train/Test theo thời gian, chuẩn hóa và bàn giao dữ liệu cho Người 3.

Pipeline chính thức là **Binary Classification**. Không còn sử dụng `target_next_return` làm biến mục tiêu Regression.

## 2. Dữ liệu đầu vào
Đọc dữ liệu sạch do Người 1 bàn giao tại:

`visualizations/10_cleaned_data.csv`

Các cột tối thiểu: `StockCode`, `Date`, `High`, `Low`, `Close`, `Volume`. Các cột bổ sung như `DailyReturn`, `LogVolume`, `Volume_Winsorized` được giữ nguyên trong dữ liệu nguồn nhưng không dùng trực tiếp để tránh phụ thuộc cách xử lý ngoài pipeline.

## 3. Feature Engineering
Các đặc trưng được tính độc lập theo từng mã cổ phiếu, sau khi sắp xếp ngày giao dịch. Không dùng giá trị tuyệt đối làm đầu vào chính; ưu tiên tỷ lệ/biến động để có thể kết hợp 10 mã có mức giá khác nhau.

| Feature | Ý nghĩa |
|---|---|
| `return` | Tỷ suất sinh lời phiên hiện tại: Close(t)/Close(t-1)-1 |
| `ret_lag1` | Return của phiên trước |
| `ret_lag2` | Return của hai phiên trước |
| `ret5` | Tỷ suất sinh lời 5 phiên |
| `ma_ratio` | MA5 / MA20 |
| `close_vs_ma20` | Close / MA20 - 1 |
| `range_pct` | (High - Low) / Close |
| `close_pos` | Vị trí giá đóng cửa trong biên độ High-Low |
| `volume_change` | Tỷ lệ thay đổi Volume so với phiên trước |
| `vol_ratio` | Volume / trung bình Volume 20 phiên |
| `vol10` | Độ lệch chuẩn Return trong 10 phiên |
| `rsi14` | RSI 14 phiên |

Nếu có `DailyReturn_Winsorized` và `Volume_Winsorized` do Người 1 bàn giao, pipeline dùng hai cột điều chỉnh này cho nhóm đặc trưng return/volume (DailyReturn_Winsorized được đổi từ % sang tỷ lệ). Nhãn vẫn được xác định từ Close thực tế của phiên kế tiếp, không dùng giá trị winsorized. Nếu các cột điều chỉnh không có, pipeline tính từ Close/Volume gốc.

Các rolling/lag được group theo `StockCode`, chỉ sử dụng phiên hiện tại và quá khứ. Những dòng chưa đủ lịch sử để tính feature hoặc không có phiên tiếp theo để tạo nhãn sẽ bị loại.

## 4. Nhãn Classification
- `label = 1` nếu Close của phiên giao dịch kế tiếp **cùng mã cổ phiếu** lớn hơn Close hiện tại.
- `label = 0` nếu Close phiên kế tiếp nhỏ hơn hoặc bằng Close hiện tại.
- `target_date` là ngày tạo nhãn, chỉ để kiểm tra thời gian; không phải feature.
- Không bàn giao `target_next_return` hay target số thực Regression.

## 5. Chia Train/Test và chống leakage
- Chia theo các ngày giao dịch duy nhất trên toàn bộ 10 mã, không shuffle.
- Tất cả mã cổ phiếu của cùng một ngày thuộc cùng một tập.
- Purge các mẫu Train nếu `target_date` của mẫu đó chạm sang giai đoạn Test.
- `StandardScaler.fit()` chỉ trên feature của Train; Test chỉ gọi `transform()`.
- Không đưa `StockCode`, `Date`, `target_date` vào X.

## 6. Chạy pipeline
Mở Terminal tại thư mục `ASM_MachineLearning`:

```powershell
python -m pip install -r requirements.txt
python classification_features.py
```

Hoặc:

```powershell
python scripts/run_feature_engineering.py
```

## 7. File bàn giao cho Người 3
Tất cả được lưu trong `data/processed/`:

| File | Nội dung | Cách dùng |
|---|---|---|
| `clf_train.csv` | 12 feature thô + `label` + metadata | Dùng làm đầu vào CV; scaler được fit bên trong từng fold của Pipeline |
| `clf_test.csv` | Cùng schema với Train | Dùng đánh giá cuối cùng; không tự tạo lại feature/label |
| `clf_train_scaled.csv` | Feature Train đã StandardScaler | Dùng nếu cần tập Train đã chuẩn hóa sẵn |
| `clf_test_scaled.csv` | Feature Test được transform bằng scaler Train | Dùng cùng scaler của Train |
| `standard_scaler.joblib` | Scaler đã fit trên toàn bộ Train | Dùng cho quy trình không có CV hoặc triển khai |
| `classification_features.csv` | Toàn bộ feature + label trước chia tập | Kiểm tra/đối soát |
| `feature_schema.json` | Danh sách feature, định nghĩa label, quy tắc split | Hợp đồng dữ liệu |
| `quality_checks.json` | Kiểm tra overlap, leakage biên, phân phối label | Nghiệm thu |

### Lưu ý dành cho Người 3
1. Dùng đúng `clf_train.csv` và `clf_test.csv`; không tự tạo lại feature, label hoặc chia Train/Test.
2. `X` gồm đúng 12 cột trong `feature_schema.json` mục `feature_columns`; `y` là `label`.
3. Với TimeSeriesSplit/CV, cần chia theo **ngày giao dịch**, không chia theo dòng vì 10 mã có cùng ngày. Trong mỗi fold, fit scaler trên phần train của fold rồi transform validation bằng `Pipeline`.
4. Khi đánh giá cuối cùng, fit Pipeline trên toàn bộ `clf_train.csv` và đánh giá đúng một lần trên `clf_test.csv`.
5. Không dùng `Date`, `target_date`, `StockCode` làm feature.
6. Các file `*_scaled.csv` được tạo để đáp ứng bước chuẩn hóa của Người 2; tránh dùng chúng làm đầu vào CV nếu scaler đã fit trên toàn bộ Train, vì thống kê scaler sẽ nhìn thấy các fold validation.

## 8. Kết quả chạy với file mới nhất
Kết quả kiểm tra hiện tại được ghi trong `quality_checks.json`. Đây là lần chạy trên `visualizations/10_cleaned_data.csv` của gói mới nhất bạn gửi.

## 9. Môi trường Python 3.12 (thống nhất)

Toàn bộ project chỉ hỗ trợ **Python 3.12.x**. Nguồn cấu hình phiên bản là `.python-version`, `pyproject.toml` (`requires-python = "==3.12.*"`) và `requirements.txt` (danh sách dependency ghim phiên bản, nguồn cài đặt duy nhất). Mỗi máy chỉ tạo một môi trường cục bộ tại `.venv/` bằng script tương ứng:

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
macOS / Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
``pyproject.toml` khai báo `requires-python = 3.12.*`; `models/clf_common.py` cảnh báo nếu chạy sai phiên bản. `playwright` chỉ cần khi chạy crawler (`crawler/`); sau khi cài, chạy thêm `playwright install chromium`.

## 10. Người 3 — Mô hình Classification
Đọc đúng `clf_train.csv` / `clf_test.csv` theo hợp đồng dữ liệu, **không tạo lại feature/label**.

```powershell
python models/logistic_regression.py   # Logistic Regression (mô hình chính): Accuracy + Confusion Matrix
python models/train_models.py          # so sánh 6 thuật toán + backtest minh họa
```

Cấu trúc (không lặp code):
- `classification_features.py`: code của Người 2, giữ nguyên; `models/clf_common.py` chỉ import `FEATURES`, `LABEL`, `load_clean_data` từ đây.
- `models/clf_common.py`: nạp dữ liệu bàn giao, kiểm tra hợp đồng (không trùng ngày, không leakage `target_date`), CV theo ngày giao dịch có purge, hàm đánh giá dùng chung.
- `models/logistic_regression.py`, `models/train_models.py`: chỉ chứa phần huấn luyện riêng; `StandardScaler` nằm trong `Pipeline` nên fit riêng trên fold-train.
- `next_return` (chỉ dùng cho backtest) được tính từ Close trong `visualizations/10_cleaned_data.csv`, không phải feature hay nhãn.

Kết quả (tập Test 960 mẫu). Baseline = luôn đoán lớp đa số **của Train** (Giảm/Đứng): Accuracy 0.573, Balanced Acc 0.500, Precision/Recall (Tăng) 0, **F1-macro 0.364** (đã sửa, trước đây gán nhầm 0.5; giờ tính bằng `baseline_metrics()`).

| Mô hình | Accuracy Test | Recall (Tăng) | F1-macro |
|---|---|---|---|
| Baseline | 0.573 | 0.000 | 0.364 |
| Logistic Regression (mặc định) | 0.550 | 0.241 | 0.490 |
| Logistic Regression (balanced) | 0.511 | 0.405 | 0.498 |

Chưa mô hình nào vượt Baseline về Accuracy và ROC-AUC Test ≈ 0.52 (CI95 chứa 0.5). Phân tích nguyên nhân, đóng góp feature, chọn đặc trưng, tối ưu siêu tham số, chọn ngưỡng (đều trên Train/CV) nằm trong `results/ANALYSIS.md`.

## 11. Phân tích, dự đoán, kiểm thử, dashboard
```bash
python models/analysis.py                      # Phân tích Train/CV -> results/analysis_*.csv|png|json, ANALYSIS.md
python models/predict.py --input data/raw/all_stocks_20241113_20261113.csv --output results/predictions.csv
python models/predict.py --input du_lieu_moi.csv --all-dates      # dự đoán mọi phiên
python -m pytest                               # 22 kiểm thử tự động (tests/)
streamlit run app/dashboard.py                 # Dashboard
```
- `models/predict.py`: nhận OHLCV mới (`StockCode, Date, High, Low, Close, Volume`; mỗi mã ≥ 20 phiên) → tính 12 feature bằng đúng code Người 2 (`build_classification_data(..., for_inference=True)`) → nạp `logreg_model.joblib` → P(Tăng) → Tăng/Giảm-Đứng theo ngưỡng trong `models/model_meta.json` → CSV.
- `tests/`: làm sạch dữ liệu, feature (12 cột, không NaN, nhãn 0/1 khớp Close kế tiếp, tái tạo được), data leakage (không trùng ngày Train/Test, purge CV, feature không nhìn tương lai), baseline, dự đoán/nạp `.joblib`.
- `app/dashboard.py`: chọn mã, nến + MA20, RSI, MACD, dự đoán phiên kế tiếp + xác suất, Confusion Matrix, Accuracy/Precision/Recall/F1, so sánh mô hình, đóng góp feature, tải CSV dự đoán, tải dữ liệu mới.
- Lưu ý: `predict.py` dùng Close/Volume gốc, còn lúc huấn luyện feature return/volume dùng cột winsorized của Người 1 (chỉ khác ở điểm ngoại lai).

### Thứ tự chạy lại toàn bộ (Python 3.12)
```bash
./setup_env.sh            # hoặc .\setup_env.ps1 trên Windows
python scripts/run_feature_engineering.py
python models/logistic_regression.py
python models/train_models.py
python models/analysis.py
python -m pytest
```


## 12. Quy tắc Git — không commit môi trường

- Tuyệt đối không commit `.venv/`, `venv/`, `env/`, `ENV/`, `.conda/` hoặc thư mục môi trường Python khác.
- Không commit `.env`, token, API key, cookie, credential hoặc cấu hình chứa bí mật.
- Commit các tệp tái tạo môi trường: `.python-version`, `pyproject.toml`, `requirements.txt`, `setup_env.ps1`, `setup_env.sh`.
- Ưu tiên sửa và tái sử dụng hàm/module đang có; không tạo module trùng chức năng.
- Trước mỗi commit, chạy:
  ```bash
  git status --short
  git check-ignore -v .venv/
  git diff --check
  ```
- Nếu thư mục môi trường đã từng được Git theo dõi, `.gitignore` không tự bỏ theo dõi. Gỡ khỏi index nhưng giữ file trên máy bằng:
  ```bash
  git rm -r --cached .venv venv env ENV
  ```
