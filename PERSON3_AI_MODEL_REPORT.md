# BÁO CÁO NGƯỜI 3

## Phát triển, huấn luyện và đánh giá mô hình Machine Learning

**Dự án:** ASM_MachineLearning\
**Phân công:** Người 3 -- Machine Learning Model Development &
Evaluation\
**Phiên bản báo cáo:** 1.0\
**Phạm vi:** Huấn luyện mô hình phân loại, đánh giá, phân tích, dự đoán
và dashboard

------------------------------------------------------------------------

## 1. Tổng quan nhiệm vụ Người 3

Người 3 tiếp nhận bộ dữ liệu đã được Người 2 chuẩn bị thông qua quy
trình Feature Engineering và chia tập Train/Test. Nhiệm vụ chính là sử
dụng dữ liệu bàn giao để xây dựng, huấn luyện, so sánh và đánh giá các
mô hình Machine Learning phục vụ bài toán phân loại xu hướng biến động
giá cổ phiếu.

Bài toán được triển khai dưới dạng **Binary Classification**:

-   `label = 1`: Giá đóng cửa của phiên giao dịch kế tiếp cao hơn giá
    đóng cửa phiên hiện tại (Tăng).
-   `label = 0`: Giá đóng cửa của phiên kế tiếp bằng hoặc thấp hơn phiên
    hiện tại (Giảm/Đứng).

Người 3 không tự tạo lại feature, label hoặc chia lại Train/Test. Các
bước huấn luyện và đánh giá phải tuân thủ hợp đồng dữ liệu do Người 2
bàn giao.

### 1.1. Mục tiêu

1.  Huấn luyện mô hình Logistic Regression làm mô hình chính.
2.  So sánh nhiều thuật toán phân loại để đánh giá khả năng dự đoán.
3.  Thiết kế quy trình Cross-Validation theo ngày giao dịch, hạn chế
    Data Leakage.
4.  Đánh giá mô hình trên tập Test độc lập.
5.  Phân tích đóng góp đặc trưng và các phương án cải thiện mô hình.
6.  Xây dựng chức năng dự đoán trên dữ liệu OHLCV mới.
7.  Tích hợp kết quả vào dashboard trực quan.
8.  Kiểm thử các thành phần quan trọng của pipeline.

------------------------------------------------------------------------

## 2. Dữ liệu đầu vào từ Người 2

Người 3 sử dụng các tệp bàn giao trong thư mục `data/processed/`.

  -----------------------------------------------------------------------
  Tệp                                 Mục đích sử dụng
  ----------------------------------- -----------------------------------
  `clf_train.csv`                     Dữ liệu Train gồm 12 feature,
                                      `label` và metadata; đầu vào chính
                                      cho Cross-Validation

  `clf_test.csv`                      Dữ liệu Test cùng schema; chỉ dùng
                                      cho đánh giá cuối cùng

  `clf_train_scaled.csv`              Tập Train đã chuẩn hóa sẵn, phục vụ
                                      quy trình phù hợp

  `clf_test_scaled.csv`               Tập Test được transform bằng scaler
                                      fit trên Train

  `standard_scaler.joblib`            Scaler đã fit trên toàn bộ tập
                                      Train

  `classification_features.csv`       Toàn bộ feature và label trước khi
                                      chia tập, dùng đối soát

  `feature_schema.json`               Hợp đồng dữ liệu, danh sách feature
                                      và quy tắc tạo label/chia tập

  `quality_checks.json`               Kết quả kiểm tra chất lượng,
                                      overlap và leakage
  -----------------------------------------------------------------------

### 2.1. Quy tắc sử dụng dữ liệu

-   `X` chỉ gồm đúng 12 cột được khai báo trong `feature_schema.json`
    tại `feature_columns`.
-   `y` là cột `label`.
-   Không sử dụng `Date`, `target_date` hoặc `StockCode` làm feature.
-   Không tự tạo lại feature hoặc label.
-   Không chia ngẫu nhiên các dòng Train/Test.
-   Tập Test được giữ riêng để đánh giá cuối cùng, không dùng để lựa
    chọn mô hình hoặc tối ưu siêu tham số.

### 2.2. Lưu ý về chuẩn hóa

Trong Cross-Validation, scaler phải được đặt bên trong `Pipeline`. Với
mỗi fold, scaler chỉ được fit trên phần train của fold và sau đó
transform phần validation tương ứng.

Không nên đưa trực tiếp các tệp `*_scaled.csv` vào Cross-Validation nếu
scaler của chúng đã được fit trên toàn bộ tập Train, vì cách làm đó có
thể khiến thông tin thống kê của fold validation lọt vào quá trình chuẩn
hóa.

------------------------------------------------------------------------

## 3. Quy trình huấn luyện mô hình AI

Quy trình tổng thể của Người 3:

1.  Đọc dữ liệu `clf_train.csv` và `clf_test.csv`.
2.  Kiểm tra schema, feature, label và metadata.
3.  Tách `X` và `y` theo hợp đồng dữ liệu.
4.  Tạo Cross-Validation theo ngày giao dịch.
5.  Xây dựng Pipeline gồm chuẩn hóa và mô hình.
6.  Huấn luyện, đánh giá và so sánh các thuật toán trên Train/CV.
7.  Chọn cấu hình mô hình dựa trên kết quả Cross-Validation.
8.  Fit mô hình cuối cùng trên toàn bộ Train.
9.  Đánh giá một lần trên Test.
10. Lưu mô hình, kết quả và biểu đồ phục vụ phân tích/triển khai.

### 3.1. Kiểm soát Data Leakage

Dữ liệu có nhiều mã cổ phiếu cùng ngày giao dịch. Vì vậy,
Cross-Validation phải chia theo **ngày giao dịch**, không chia theo từng
dòng độc lập.

Các nguyên tắc:

-   Các mã cổ phiếu thuộc cùng một ngày phải nằm trong cùng một fold.
-   Scaler chỉ fit trên phần train của fold.
-   Dùng Pipeline để tránh fit scaler trước khi chia fold.
-   Có cơ chế purge để loại các mẫu Train có `target_date` chồng lấn với
    giai đoạn validation.
-   Không dùng thông tin tương lai để tạo feature.
-   Tập Test chỉ được sử dụng ở bước đánh giá cuối cùng.

------------------------------------------------------------------------

## 4. Các mô hình Machine Learning

Dự án triển khai quy trình so sánh **6 thuật toán phân loại**. Trong đó,
Logistic Regression được chọn làm mô hình chính để phân tích và xây dựng
chức năng dự đoán.

Các thuật toán được liệt kê trong kết quả so sánh của dự án gồm:

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   Support Vector Machine (SVM -- RBF)
-   K-Nearest Neighbors (KNN)
-   Naive Bayes

### 4.1. Logistic Regression

Logistic Regression là mô hình chính được triển khai riêng trong
`models/logistic_regression.py`.

Mô hình ước lượng xác suất thuộc lớp Tăng dựa trên các đặc trưng đầu
vào. Kết quả xác suất được chuyển thành nhãn dự đoán theo ngưỡng phân
loại.

Dự án có đánh giá hai cấu hình:

-   Logistic Regression mặc định.
-   Logistic Regression với `class_weight="balanced"` nhằm xử lý sự mất
    cân bằng giữa các lớp.

### 4.2. So sánh thuật toán

Tệp `models/train_models.py` thực hiện huấn luyện và so sánh các thuật
toán. Kết quả được lưu trong thư mục `results/`, trong đó có
`model_comparison.csv`.

Việc so sánh nên dựa trên nhiều chỉ số, không chỉ Accuracy, đặc biệt là
Recall của lớp Tăng, F1-macro và ROC-AUC.

------------------------------------------------------------------------

## 5. Kết quả đánh giá mô hình

Theo số liệu kết quả được cung cấp trong tài liệu bàn giao:

-   Số mẫu Test: **960**.
-   Baseline Accuracy: **0.573 (57,3%)**.
-   Logistic Regression mặc định: **0.550 (55,0%)**.
-   Logistic Regression balanced: **0.511 (51,1%)**.
-   ROC-AUC Test xấp xỉ **0.52**; khoảng tin cậy 95% chứa mức 0.5.

### 5.1. Bảng kết quả

  Mô hình                            Accuracy Test   Recall (Tăng)   F1-macro
  -------------------------------- --------------- --------------- ----------
  Baseline                                   0.573           0.000      0.364
  Logistic Regression (mặc định)             0.550           0.241      0.490
  Logistic Regression (balanced)             0.511           0.405      0.498

**Lưu ý:** Bảng trên chỉ ghi các số liệu đã được cung cấp rõ ràng. Để bổ
sung kết quả chi tiết của cả 6 thuật toán, cần đối chiếu trực tiếp tệp
`results/model_comparison.csv` của phiên bản dự án tương ứng.

### 5.2. Nhận xét

-   Logistic Regression mặc định đạt Accuracy 55,0%, thấp hơn Baseline
    57,3%.
-   Logistic Regression balanced có Recall lớp Tăng cao hơn (40,5%),
    nhưng Accuracy thấp hơn (51,1%).
-   F1-macro của cấu hình balanced là 0,498, cao hơn cấu hình mặc định
    0,490.
-   ROC-AUC xấp xỉ 0,52 cho thấy khả năng phân biệt hai lớp còn hạn chế.
-   Chưa có mô hình nào được báo cáo là vượt Baseline về Accuracy.

### 5.3. Ý nghĩa kết quả

Kết quả cho thấy bài toán dự đoán hướng biến động giá cổ phiếu trong dữ
liệu và cấu hình hiện tại còn khó. Việc tăng Recall lớp Tăng có thể đi
kèm với giảm Accuracy, do đó cần xác định mục tiêu ứng dụng trước khi
lựa chọn ngưỡng dự đoán.

Không nên kết luận mô hình có khả năng dự đoán hiệu quả hoặc sẵn sàng sử
dụng trong giao dịch thực tế chỉ dựa trên kết quả hiện tại.

------------------------------------------------------------------------

## 6. Phân tích đặc trưng và tối ưu mô hình

Người 3 thực hiện các phân tích bổ sung trên Train/Cross-Validation nhằm
tìm hiểu nguyên nhân hiệu suất còn hạn chế và đánh giá khả năng cải
thiện.

Các nội dung được lưu trong `results/ANALYSIS.md` và các tệp
`analysis_*.csv`, `analysis_*.png`, `analysis_*.json`.

Phạm vi phân tích gồm:

-   Đóng góp của các feature.
-   Lựa chọn đặc trưng.
-   Tối ưu siêu tham số.
-   Phân tích ngưỡng phân loại.
-   Phân tích kết quả theo từng mã cổ phiếu.
-   Đánh giá chiến lược minh họa bằng backtest.

**Nguyên tắc:** Các quyết định lựa chọn feature, siêu tham số và ngưỡng
phải được thực hiện trên Train/CV. Không dùng tập Test để tối ưu rồi báo
cáo lại kết quả trên chính tập Test đó.

------------------------------------------------------------------------

## 7. Chức năng dự đoán và Dashboard

### 7.1. Chức năng dự đoán

Tệp `models/predict.py` cung cấp chức năng dự đoán trên dữ liệu OHLCV
mới.

Đầu vào dự kiến gồm:

-   `StockCode`
-   `Date`
-   `High`
-   `Low`
-   `Close`
-   `Volume`

Theo quy trình được mô tả trong tài liệu dự án, mỗi mã cần tối thiểu 20
phiên dữ liệu để tạo đủ feature.

Quy trình dự đoán:

1.  Đọc dữ liệu OHLCV.
2.  Tạo 12 feature bằng lại đúng hàm Feature Engineering đã được Người 2
    bàn giao.
3.  Nạp mô hình Logistic Regression đã lưu.
4.  Tính xác suất thuộc lớp Tăng.
5.  Áp dụng ngưỡng trong `models/model_meta.json`.
6.  Xuất kết quả dự đoán ra CSV.

Các lệnh được mô tả trong tài liệu:

``` powershell
python models/predict.py --input data/raw/all_stocks_20241113_20261113.csv --output results/predictions.csv

python models/predict.py --input du_lieu_moi.csv --all-dates
```

### 7.2. Dashboard

Ứng dụng dashboard được đặt tại `app/dashboard.py`, chạy bằng Streamlit:

``` powershell
streamlit run app/dashboard.py
```

Các chức năng được mô tả:

-   Chọn mã cổ phiếu.
-   Hiển thị biểu đồ nến và đường MA20.
-   Hiển thị RSI và MACD.
-   Hiển thị dự đoán phiên kế tiếp và xác suất.
-   Hiển thị Confusion Matrix.
-   Hiển thị Accuracy, Precision, Recall và F1.
-   So sánh kết quả các mô hình.
-   Hiển thị đóng góp feature.
-   Tải CSV dự đoán và dữ liệu mới.

Các chức năng trên cần được kiểm tra trên phiên bản chạy thực tế trước
khi xác nhận trong nghiệm thu hoặc trình diễn.

------------------------------------------------------------------------

## 8. Kiểm thử và nghiệm thu

Thư mục `tests/` chứa các kiểm thử tự động liên quan đến dữ liệu,
feature, leakage và dự đoán.

Các nội dung được mô tả trong tài liệu dự án:

-   Kiểm tra phân loại.
-   Kiểm tra làm sạch dữ liệu.
-   Kiểm tra Feature Engineering.
-   Kiểm tra Data Leakage.
-   Kiểm tra dự đoán và nạp mô hình đã lưu.

Chạy kiểm thử bằng:

``` powershell
python -m pytest
```

Tài liệu bàn giao ghi nhận **22 kiểm thử tự động**. Cần chạy lại trên
môi trường và phiên bản mã nguồn đang nghiệm thu để xác nhận số test
thực tế đạt.

------------------------------------------------------------------------

## 9. Cấu trúc file liên quan đến Người 3

``` text
ASM_MachineLearning/
├── app/
│   └── dashboard.py
├── data/
│   └── processed/
│       ├── clf_train.csv
│       ├── clf_test.csv
│       ├── clf_train_scaled.csv
│       ├── clf_test_scaled.csv
│       ├── standard_scaler.joblib
│       ├── classification_features.csv
│       ├── feature_schema.json
│       └── quality_checks.json
├── models/
│   ├── clf_common.py
│   ├── logistic_regression.py
│   ├── train_models.py
│   ├── analysis.py
│   └── predict.py
├── results/
│   ├── ANALYSIS.md
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── backtest.png
│   └── predictions.csv
└── tests/
    ├── test_classification.py
    ├── test_data_cleaning.py
    ├── test_data_leakage.py
    ├── test_feature_engineering.py
    └── test_model_prediction.py
```

**Phân định trách nhiệm:** `data/processed/` là dữ liệu đầu vào được
Người 2 bàn giao; phần huấn luyện, phân tích, dự đoán, kiểm thử mô hình
và dashboard là phạm vi công việc của Người 3 theo mô tả dự án.

------------------------------------------------------------------------

## 10. Hạn chế và hướng phát triển

### 10.1. Hạn chế hiện tại

1.  Hiệu suất Accuracy chưa vượt Baseline.
2.  ROC-AUC xấp xỉ 0,52, cho thấy khả năng phân biệt hai lớp còn yếu.
3.  Recall lớp Tăng còn thấp ở cấu hình Logistic Regression mặc định.
4.  Kết quả có thể khác nhau giữa các mã cổ phiếu và giai đoạn thị
    trường.
5.  Backtest minh họa không đồng nghĩa với hiệu quả giao dịch thực tế;
    cần tính đến phí, trượt giá và các điều kiện thực thi nếu phát triển
    theo hướng này.

### 10.2. Hướng phát triển

-   Mở rộng dữ liệu và kiểm tra độ ổn định theo nhiều giai đoạn thị
    trường.
-   Đánh giá theo từng mã cổ phiếu và từng trạng thái thị trường.
-   Tiếp tục tối ưu mô hình trên Train/CV, không sử dụng Test để điều
    chỉnh.
-   Nghiên cứu thêm feature có cơ sở tài chính và kiểm tra khả năng tổng
    quát hóa.
-   Đánh giá hiệu quả bằng nhiều chỉ số và chiến lược backtest có tính
    phí giao dịch.
-   Theo dõi hiệu suất mô hình khi dữ liệu mới được cập nhật.

------------------------------------------------------------------------

## 11. Kết luận

Người 3 đã tiếp nhận dữ liệu Feature Engineering từ Người 2 để triển
khai bài toán Binary Classification cho dự án phân tích cổ phiếu. Phạm
vi công việc bao gồm xây dựng quy trình huấn luyện, so sánh thuật toán,
đánh giá mô hình, phân tích đặc trưng, phát triển chức năng dự đoán,
dashboard và kiểm thử.

Kết quả hiện tại cho thấy Logistic Regression và các cấu hình được báo
cáo chưa vượt Baseline về Accuracy. Đây là một kết quả đánh giá quan
trọng, phản ánh đúng giới hạn của mô hình ở phiên bản hiện tại và tạo cơ
sở cho các hướng cải thiện tiếp theo.

**Kết luận chung:** Dự án đã có quy trình Machine Learning và các thành
phần phục vụ phân tích/dự đoán theo thiết kế được mô tả. Tuy nhiên, hiệu
quả dự đoán cần tiếp tục được cải thiện và kiểm chứng trước khi xem xét
ứng dụng vào quyết định đầu tư thực tế.

------------------------------------------------------------------------

## Phụ lục. Lệnh chạy tham khảo

Các lệnh dưới đây được trích theo cấu trúc và hướng dẫn trong tài liệu
dự án. Chạy trong môi trường Python 3.12 đã cài dependencies:

``` powershell
python models/logistic_regression.py
python models/train_models.py
python models/analysis.py
python -m pytest
streamlit run app/dashboard.py
```

Dự đoán dữ liệu mới:

``` powershell
python models/predict.py --input du_lieu_moi.csv --all-dates
```

**Ghi chú xác minh:** Trước khi nộp báo cáo, cần đối chiếu số liệu với
`results/model_comparison.csv`, `results/ANALYSIS.md`, kiểm thử mới nhất
và dashboard trên đúng commit được dùng để nghiệm thu. Báo cáo này không
khẳng định các lệnh hoặc giao diện đã được chạy thành công trong phiên
làm việc hiện tại.
