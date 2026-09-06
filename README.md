# Shinsa AI: Hệ thống Chấm điểm Tín dụng Thay thế cho Khách hàng "Thin-file"

**Tác giả:** Võ Hoàng Long
**Bộ dữ liệu:** Home Credit Default Risk (`application_train.csv`, `installments_payments.csv`)
**Thuật toán chính:** LightGBM + Optuna Hyperparameter Tuning
**Kỹ thuật giải thích mô hình:** SHAP (SHapley Additive exPlanations)

---

## Mục lục

1. [Tóm tắt điều hành](#1-tóm-tắt-điều-hành)
2. [Bài toán kinh doanh & Mục tiêu nghiên cứu](#2-bài-toán-kinh-doanh--mục-tiêu-nghiên-cứu)
3. [Dữ liệu](#3-dữ-liệu)
4. [Phân tích khám phá dữ liệu (EDA) & Xử lý dữ liệu](#4-phân-tích-khám-phá-dữ-liệu-eda--xử-lý-dữ-liệu)
5. [Feature Engineering](#5-feature-engineering)
6. [Mô hình hóa](#6-mô-hình-hóa)
7. [Kiểm định độ ổn định (Cross-Validation)](#7-kiểm-định-độ-ổn-định-cross-validation)
8. [Tối ưu hóa siêu tham số (Optuna)](#8-tối-ưu-hóa-siêu-tham-số-optuna)
9. [Explainable AI (SHAP)](#9-explainable-ai-shap)
10. [Chiến lược Cân chỉnh Ngưỡng (Threshold Tuning)](#10-chiến-lược-cân-chỉnh-ngưỡng-threshold-tuning)
11. [Kết quả trên tập Test độc lập](#11-kết-quả-trên-tập-test-độc-lập)
12. [Hạn chế & Hướng phát triển](#12-hạn-chế--hướng-phát-triển)
13. [Kết luận](#13-kết-luận)

---

## 1. Tóm tắt điều hành

Dự án **Shinsa AI** xây dựng một hệ thống chấm điểm rủi ro tín dụng tự động, nhắm đến nhóm khách hàng **"hồ sơ mỏng" (Thin-file)** — những cá nhân chưa từng có lịch sử vay trả góp trong hệ thống, chiếm khoảng **5,16%** tổng số hồ sơ. Đây là nhóm khách hàng bị mô hình chấm điểm truyền thống (CIC/FICO) bỏ sót do thiếu dữ liệu lịch sử tín dụng.

Kết quả chính đạt được:

| Hạng mục | Kết quả |
|---|---|
| Mô hình baseline (XGBoost) | ROC-AUC = **0.7638** trên tập Test |
| Mô hình cải tiến (LightGBM + Optuna) | ROC-AUC trung bình 5-Fold CV = **0.7606** (±0.0049) |
| ROC-AUC trên tập Validation (sau tuning) | **0.7669** |
| ROC-AUC trên tập Test độc lập (đánh giá cuối) | **0.7657** |
| Đặc trưng tự tạo lọt Top 15 quan trọng nhất | `MAX_DAYS_LATE`, `TOTAL_INSTALLMENTS` |
| Ngưỡng quyết định | 2 chiến lược kinh doanh: F1-optimal (0.65) và Cost-optimal (0.55) |

Mô hình đạt độ phân tách rủi ro ổn định (~0.76 AUC) qua nhiều lần kiểm định chéo, đồng thời chứng minh được rằng các đặc trưng hành vi thanh toán tự tổng hợp (`MAX_DAYS_LATE`) có thể **thay thế một phần vai trò của điểm tín dụng ngoại kiểm (EXT_SOURCE)** khi các điểm số này bị khuyết — đúng với giả thuyết kinh doanh ban đầu của dự án.

---

## 2. Bài toán kinh doanh & Mục tiêu nghiên cứu

### 2.1. Vấn đề

Các mô hình chấm điểm tín dụng truyền thống phụ thuộc rất lớn vào lịch sử giao dịch tín dụng (điểm CIC/FICO), tạo ra một **vòng lặp luẩn quẩn** cho nhóm khách hàng "hồ sơ mỏng":

> Không có lịch sử tín dụng → Không được duyệt vay → Tiếp tục không có lịch sử tín dụng.

Shinsa AI được xây dựng để phá vỡ vòng lặp này bằng cách khai thác **Dữ liệu thay thế (Alternative Data)** — cụ thể là hành vi thanh toán các khoản trả góp nhỏ lẻ — nhằm tự động hóa và minh bạch hóa quyết định phê duyệt rủi ro.

### 2.2. Đối tượng nghiên cứu

Nhóm khách hàng chưa từng có lịch sử vay trả góp tại hệ thống (**~5,16%** tập dữ liệu), thường là người trẻ mới đi làm hoặc sinh viên.

### 2.3. Mục tiêu nghiên cứu

- Xây dựng mô hình phân loại rủi ro vỡ nợ (Default Risk Classifier) có độ phân tách cao (ROC-AUC > 0.75).
- Lượng hóa thói quen thanh toán và sinh hoạt thành các chỉ số rủi ro tín dụng.
- Chuyển đổi xác suất dự đoán của mô hình thành các chiến lược xét duyệt kinh doanh thực tế (không dùng ngưỡng 0.5 cứng nhắc).

---

## 3. Dữ liệu

| Bảng dữ liệu | Số dòng | Số cột | Vai trò |
|---|---|---|---|
| `application_train.csv` | 307.511 | 122 | Hồ sơ nhân khẩu học, tài chính của khách hàng (bảng chính) |
| `installments_payments.csv` | 13.605.401 | 8 | Lịch sử thanh toán từng kỳ trả góp trong quá khứ (bảng hành vi) |
| **Bộ dữ liệu sau khi ghép** | **307.511** | **126** | Merge theo `SK_ID_CURR` (left join, giữ toàn bộ khách hàng) |

Biến mục tiêu `TARGET`: 1 = khách hàng gặp khó khăn thanh toán/vỡ nợ, 0 = khách hàng trả nợ tốt.

---

## 4. Phân tích khám phá dữ liệu (EDA) & Xử lý dữ liệu

### 4.1. Xử lý giá trị ngoại lai ở `DAYS_EMPLOYED`

Cột `DAYS_EMPLOYED` chứa mã lỗi đặc biệt `365243` — được hệ thống dùng để đánh dấu nhóm "thất nghiệp/hưu trí". Số dòng bị ảnh hưởng: **55.374 dòng (18,01%)** tổng dữ liệu.

Cách xử lý:
- Tạo cờ nhị phân `DAYS_EMPLOYED_ANOM` để lưu lại thông tin bất thường này (tránh mất tín hiệu).
- Thay giá trị lỗi `365243` bằng `NaN` để mô hình không học sai lệch theo con số vô nghĩa này.

### 4.2. Phân tích dữ liệu khuyết (Missing Values) & khám phá nhóm "Thin-file"

Các đặc trưng hành vi trích xuất từ bảng `installments_payments` (`TOTAL_INSTALLMENTS`, `MAX_DAYS_LATE`, `MEAN_DAYS_LATE`, `TOTAL_SHORTAGE`) ghi nhận tỷ lệ thiếu đồng loạt ở mức **~5,16%**:

| Cột | % thiếu |
|---|---|
| `TOTAL_INSTALLMENTS` | 5,160% |
| `MAX_DAYS_LATE` | 5,163% |
| `TOTAL_SHORTAGE` | 5,160% |

**Nhận xét:** Tập 5,16% này chính là nhóm đối tượng mục tiêu của bài toán — khách hàng "Thin-file" chưa từng có lịch sử giao dịch trả góp. Đây không phải là dữ liệu bị lỗi mà là tín hiệu nghiệp vụ quan trọng.

**Xử lý:** Impute các cột này bằng giá trị `0`, nhằm giữ lại toàn bộ tệp khách hàng tiềm năng và buộc mô hình đánh giá rủi ro của họ dựa trên đặc trưng nhân khẩu học thay vì hành vi quá khứ không tồn tại.

### 4.3. Mất cân bằng của biến mục tiêu (Target Imbalance)

Lớp thiểu số (nhãn `1` — rủi ro/không duyệt) chỉ chiếm khoảng **8%** so với lớp đa số (nhãn `0` — an toàn).

**Xử lý:**
- Dùng tham số `scale_pos_weight` (XGBoost/LightGBM) để phạt nặng hơn khi mô hình dự đoán sai lớp thiểu số.
- Loại bỏ **Accuracy** khỏi tiêu chí đánh giá chính (dễ gây ảo tưởng do mất cân bằng lớp), thay bằng **ROC-AUC**, **F1-Score** và **Recall**.

### 4.4. Tương quan giữa hành vi thanh toán và rủi ro vỡ nợ

Biểu đồ boxplot đối chiếu `MAX_DAYS_LATE` và `TARGET` cho thấy ranh giới phân loại khá rõ ràng: nhóm rủi ro vỡ nợ có trung vị và độ phân tán số ngày trễ hạn cao hơn hẳn nhóm trả nợ tốt.

> **Kết luận EDA:** Tính kỷ luật và thói quen thanh toán các khoản nhỏ là tín hiệu dự báo rủi ro hiệu quả, có khả năng thay thế một phần điểm tín dụng truyền thống (CIC/FICO) đối với nhóm chưa có lịch sử.

---

## 5. Feature Engineering

### 5.1. Đặc trưng hành vi cơ bản (từ `installments_payments`)

Với mỗi bản ghi thanh toán, tính:

- `DAYS_LATE` = Ngày thực trả − Ngày đến hạn (âm: trả sớm | dương: trả trễ)
- `AMT_SHORTAGE` = Tiền cam kết đóng − Tiền thực đóng

Sau đó nhóm theo khách hàng (`SK_ID_CURR`) để tạo:

| Đặc trưng | Ý nghĩa |
|---|---|
| `TOTAL_INSTALLMENTS` | Tổng số lần đã thanh toán |
| `MAX_DAYS_LATE` | Số ngày trễ hạn lớn nhất từng ghi nhận |
| `MEAN_DAYS_LATE` | Số ngày trễ hạn trung bình |
| `TOTAL_SHORTAGE` | Tổng số tiền hụt qua các kỳ |

### 5.2. Đặc trưng chuỗi thời gian nâng cao — `LATE_TREND`

Nhằm nắm bắt **xu hướng thay đổi** trong kỷ luật tài chính (thay vì chỉ số tĩnh trung bình), nhóm nghiên cứu bổ sung:

- `LATEST_DAYS_LATE`: số ngày trễ của lần thanh toán **gần nhất** (sắp xếp theo `DAYS_INSTALMENT`).
- `LATE_TREND` = `LATEST_DAYS_LATE` − `MEAN_DAYS_LATE`

Nếu `LATE_TREND > 0`: khách hàng có xu hướng trễ hẹn **ngày càng nhiều hơn** so với trung bình lịch sử — một tín hiệu cảnh báo sớm mà chỉ số tĩnh không nắm bắt được.

Sau bước này, bộ đặc trưng cuối cùng đưa vào mô hình LightGBM có **127 cột** (16 cột dạng categorical).

---

## 6. Mô hình hóa

### 6.1. Mô hình Baseline — XGBoost

Cấu hình: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `scale_pos_weight` = tỷ lệ mất cân bằng lớp, `tree_method='hist'`.

Chia dữ liệu: 80% train (246.008 dòng, 233 features sau one-hot encoding) / 20% test (61.503 dòng).

**Kết quả trên tập Test:**

| Chỉ số | Giá trị |
|---|---|
| ROC-AUC | **0.7638** |
| Precision (lớp 1 – rủi ro) | 0.17 |
| Recall (lớp 1 – rủi ro) | 0.68 |
| F1-score (lớp 1) | 0.27 |
| Accuracy | 0.71 |

Baseline đã đạt Recall khá cao (bắt được 68% khách hàng xấu) nhưng đánh đổi bằng Precision thấp — cho thấy cần một chiến lược ngưỡng quyết định linh hoạt hơn thay vì mặc định 0.5.

### 6.2. Cải tiến mô hình

Ba cải tiến được triển khai tuần tự:

1. **Bổ sung đặc trưng chuỗi thời gian** `LATE_TREND` (mục 5.2).
2. **Chuyển từ XGBoost sang LightGBM** — tận dụng thuật toán tối ưu phân nhánh dạng Histogram, xử lý trực tiếp biến categorical mà không cần one-hot encoding, giúp huấn luyện nhanh hơn và giữ được thông tin gốc của biến phân loại.
3. **Threshold Tuning theo mục tiêu kinh doanh** thay vì mặc định 0.5 (trình bày ở mục 10).

---

## 7. Kiểm định độ ổn định (Cross-Validation)

Để tránh đánh giá mô hình chỉ dựa trên một lần chia dữ liệu may rủi, nhóm tách riêng 15% dữ liệu làm **tập Test hoàn toàn độc lập**, phần còn lại (85%) dùng **Stratified 5-Fold Cross-Validation**:

| Fold | ROC-AUC |
|---|---|
| Fold 1 | 0.7613 |
| Fold 2 | 0.7537 |
| Fold 3 | 0.7660 |
| Fold 4 | 0.7564 |
| Fold 5 | 0.7658 |
| **Trung bình** | **0.7606 (± 0.0049)** |

Độ lệch chuẩn rất nhỏ (0,0049) cho thấy mô hình **ổn định**, không phụ thuộc vào cách chia dữ liệu cụ thể.

### 7.1. Kiểm tra độ ổn định của Feature Importance

Nhóm nghiên cứu tính thêm chỉ số `cv_ratio` = độ lệch chuẩn / giá trị trung bình của mức độ quan trọng (feature importance) qua 5 fold — một `cv_ratio` cao cho thấy đặc trưng đó **không ổn định**, dễ là nhiễu (overfitting theo đặc thù riêng của từng fold) chứ không phải tín hiệu thật.

**Top đặc trưng quan trọng nhất (trung bình qua 5 fold, theo split-count importance của LightGBM):**

| Hạng | Đặc trưng | Mean Importance | cv_ratio |
|---|---|---|---|
| 1 | `ORGANIZATION_TYPE` | 1344,2 | 0,013 |
| 2 | `EXT_SOURCE_3` | 590,4 | 0,041 |
| 3 | `EXT_SOURCE_2` | 432,8 | 0,051 |
| 4 | `EXT_SOURCE_1` | 400,4 | 0,029 |
| 5 | `AMT_CREDIT` | 329,0 | 0,025 |
| 6 | `DAYS_BIRTH` | 285,6 | 0,084 |
| 7 | `DAYS_EMPLOYED` | 263,0 | 0,096 |
| 8 | `OCCUPATION_TYPE` | 254,6 | 0,062 |
| 9 | `AMT_ANNUITY` | 249,8 | 0,050 |
| 10 | `AMT_GOODS_PRICE` | 240,4 | 0,121 |
| 11 | **`TOTAL_INSTALLMENTS`** | 234,2 | 0,068 |
| 12 | `DAYS_LAST_PHONE_CHANGE` | 207,2 | 0,079 |
| 13 | `DAYS_ID_PUBLISH` | 202,8 | 0,056 |
| 14 | **`MAX_DAYS_LATE`** | 182,0 | 0,106 |
| 15 | `DAYS_REGISTRATION` | 174,2 | 0,089 |

Hai đặc trưng hành vi tự xây dựng (`TOTAL_INSTALLMENTS`, `MAX_DAYS_LATE`) đều lọt **Top 15**, với `cv_ratio` ở mức chấp nhận được — xác nhận đây là tín hiệu thật, ổn định qua các fold, chứ không phải nhiễu ngẫu nhiên.

Nhóm đặc trưng **kém ổn định nhất** (cv_ratio cao, importance rất thấp) chủ yếu là các cờ tài liệu hiếm gặp: `FLAG_DOCUMENT_17`, `FLAG_DOCUMENT_21`, `FLAG_CONT_MOBILE`, `FLAG_DOCUMENT_9`, `FLAG_DOCUMENT_20`... — đây là các ứng viên nên cân nhắc loại bỏ trong các lần huấn luyện tiếp theo để giảm nhiễu.

---

## 8. Tối ưu hóa siêu tham số (Optuna)

Thay vì chọn thủ công, nhóm dùng **Optuna** để tìm kiếm không gian siêu tham số tối ưu hóa AUC trung bình qua Stratified 5-Fold CV, với **30 trials**:

| Siêu tham số | Không gian tìm kiếm | Giá trị tối ưu |
|---|---|---|
| `n_estimators` | 100 – 500 | **339** |
| `max_depth` | 3 – 8 | **3** |
| `learning_rate` | 0.01 – 0.15 (log-scale) | **0,094** |
| `num_leaves` | 15 – 63 | **20** |
| `min_child_samples` | 10 – 100 | **81** |

**Best CV AUC sau tuning: 0,7626**

Sau khi huấn luyện lại mô hình LightGBM với bộ tham số tối ưu (trên tập train tách từ phần dữ liệu 85%, val riêng để chọn ngưỡng): **AUC trên tập Validation = 0,7669**.

Điều đáng chú ý: `max_depth` tối ưu chỉ là 3 (cây nông), cho thấy mô hình có xu hướng chống overfitting tốt hơn khi giữ độ phức tạp cây thấp và bù lại bằng số lượng cây (339) — phù hợp với đặc thù dữ liệu tín dụng nhiều nhiễu (noisy).

---

## 9. Explainable AI (SHAP)

Áp dụng `shap.TreeExplainer` trên mô hình baseline để giải thích đóng góp của từng biến vào quyết định duyệt/từ chối.

### Ba phát hiện chính từ SHAP Summary Plot:

1. **Dữ liệu ngoại kiểm (`EXT_SOURCE_1/2/3`):** Đây là các điểm số chuẩn hóa từ bên thứ ba. Giá trị thấp (chấm xanh) kéo mạnh về phía tăng rủi ro vỡ nợ; giá trị cao (chấm đỏ) tụ về phía an toàn — quy luật rất rõ ràng và nhất quán.

2. **Alternative Data (`MAX_DAYS_LATE`):** Việc đặc trưng tự tổng hợp này vươn lên **Top 5** biến ảnh hưởng mạnh nhất trong SHAP là minh chứng trực tiếp cho giả thuyết kinh doanh ban đầu. Các chấm màu đỏ (trễ nhiều ngày) tương quan thuận với xác suất vỡ nợ. Điều này xác nhận: đối với nhóm khách hàng "Thin-file" (thường khuyết cả 3 điểm `EXT_SOURCE`), mô hình đã **tự động chuyển trọng tâm** sang đánh giá kỷ luật thanh toán các khoản nhỏ lẻ để ra quyết định — đúng như mục tiêu thiết kế của Shinsa AI.

3. **Hồ sơ nhân khẩu học (`DAYS_BIRTH`):** Khách hàng trẻ tuổi (do dữ liệu gốc lưu tuổi dưới dạng số âm nên chấm xanh = trẻ) có xu hướng rủi ro cao hơn — khớp với đối tượng cốt lõi của bài toán là sinh viên/người trẻ mới đi làm.

---

## 10. Chiến lược Cân chỉnh Ngưỡng (Threshold Tuning)

Hệ thống **không sử dụng ngưỡng 0,5 mặc định**, mà xây dựng ngưỡng dựa trên mục tiêu kinh doanh, được lựa chọn trên tập **Validation** (không chạm vào Test) để tránh rò rỉ dữ liệu.

### 10.1. Ngưỡng tối ưu theo F1-Score

Khảo sát các ngưỡng từ 0,10 đến 0,90:

| Chỉ số (trên Validation) | Giá trị |
|---|---|
| Ngưỡng tối ưu (F1) | **0,65** |
| F1-Score cực đại | 0,3101 |
| Precision | 0,2391 |
| Recall | 0,4408 |

### 10.2. Ngưỡng tối ưu theo Chi phí kỳ vọng (Cost-based)

F1-Score coi Precision và Recall quan trọng ngang nhau, nhưng trong bài toán tín dụng, **bỏ lọt 1 khách hàng xấu (False Negative)** thường tốn kém hơn nhiều so với **từ chối oan 1 khách hàng tốt (False Positive)**, vì FN có thể dẫn đến mất trắng cả khoản vay, còn FP chỉ là chi phí cơ hội.

Với giả định `COST_FN = 8× COST_FP` (số liệu giả định, nên thay bằng dữ liệu thực tế từ bộ phận rủi ro khi triển khai):

| Chỉ số | Giá trị |
|---|---|
| Ngưỡng tối ưu theo F1 | 0,65 |
| **Ngưỡng tối ưu theo Chi phí** | **0,55** |

→ Ngưỡng cost-based thấp hơn ngưỡng F1-based vì ưu tiên "bắt" được nhiều khách xấu hơn, chấp nhận đánh đổi bằng việc từ chối oan nhiều khách tốt hơn.

### 10.3. Phân tích độ nhạy (Sensitivity Analysis)

Vì chưa có số liệu chi phí thực tế chính xác từ đội rủi ro, nhóm khảo sát ngưỡng tối ưu ứng với nhiều tỷ lệ `COST_FN/COST_FP` khác nhau thay vì chốt cứng một con số:

| Tỷ lệ COST_FN / COST_FP | Ngưỡng tối ưu |
|---|---|
| 3× | 0,78 |
| 5× | 0,67 |
| 8× | 0,56 |
| 12× | 0,46 |
| 15× | 0,41 |

**Nhận xét:** Tỷ lệ chi phí càng cao (bỏ lọt khách xấu càng "đắt") thì ngưỡng tối ưu càng thấp — mô hình buộc phải hạ ngưỡng để bắt được nhiều khách hàng rủi ro hơn, chấp nhận từ chối oan nhiều khách hàng tốt hơn. Bảng này cho phép bộ phận kinh doanh tự chọn ngưỡng phù hợp khi có số liệu chi phí thực tế, thay vì phụ thuộc vào một giả định cố định.

---

## 11. Kết quả trên tập Test độc lập

**ROC-AUC trên tập Test (hoàn toàn độc lập, không dùng để chọn tham số hay ngưỡng): 0,7657**

### 11.1. Chiến lược "Tăng trưởng" — Ngưỡng F1-optimal (0,65)

| Chỉ số | Lớp 0 (An toàn) | Lớp 1 (Rủi ro) |
|---|---|---|
| Precision | 0,95 | 0,24 |
| Recall | 0,88 | 0,45 |
| F1-score | 0,91 | 0,32 |

Accuracy tổng thể: 0,84

**Ma trận nhầm lẫn:**

| | Dự đoán An toàn | Dự đoán Rủi ro |
|---|---|---|
| **Thực tế An toàn** | 37.221 (TN) | 5.182 (FP) |
| **Thực tế Rủi ro** | 2.051 (FN) | 1.673 (TP) |

→ Phù hợp cho giai đoạn muốn **tối ưu hóa số lượng khách hàng tốt được duyệt vay thành công**, giảm tỷ lệ từ chối oan (Precision cao hơn ở ngưỡng này so với ngưỡng cost-based).

### 11.2. Chiến lược "Phòng thủ" — Ngưỡng Cost-optimal (0,55)

| Chỉ số | Lớp 0 (An toàn) | Lớp 1 (Rủi ro) |
|---|---|---|
| Precision | 0,96 | 0,19 |
| Recall | 0,77 | 0,62 |
| F1-score | 0,86 | 0,29 |

Accuracy tổng thể: 0,76

**Ma trận nhầm lẫn:**

| | Dự đoán An toàn | Dự đoán Rủi ro |
|---|---|---|
| **Thực tế An toàn** | 32.826 (TN) | 9.577 (FP) |
| **Thực tế Rủi ro** | 1.426 (FN) | 2.298 (TP) |

→ Phù hợp cho giai đoạn kinh tế rủi ro cao, khi cần **tối đa hóa khả năng chặn nợ xấu** (Recall lớp rủi ro tăng từ 0,45 lên 0,62), chấp nhận từ chối oan nhiều khách hàng tốt hơn (FP tăng từ 5.182 lên 9.577).

### 11.3. So sánh hai chiến lược

| | F1-optimal (0,65) | Cost-optimal (0,55) |
|---|---|---|
| Khách hàng rủi ro bắt đúng (TP) | 1.673 | 2.298 (+37%) |
| Khách hàng tốt bị từ chối oan (FP) | 5.182 | 9.577 (+85%) |
| Recall (lớp rủi ro) | 0,45 | 0,62 |
| Precision (lớp rủi ro) | 0,24 | 0,19 |

Đây chính là minh chứng cho cách tiếp cận **"không có ngưỡng đúng tuyệt đối"** — lựa chọn ngưỡng cần gắn với khẩu vị rủi ro và điều kiện kinh tế thực tế của tổ chức tín dụng tại từng thời điểm.

---

## 12. Hạn chế & Hướng phát triển

### 12.1. Hạn chế

- Mô hình vẫn chịu chi phối lớn bởi 3 biến ngoại kiểm `EXT_SOURCE_1/2/3`. Khi một khách hàng bị thiếu hụt toàn bộ 3 điểm này (trường hợp phổ biến ở nhóm "Thin-file"), độ tin cậy trong quyết định của mô hình bị suy giảm.
- Precision ở lớp rủi ro còn thấp (0,19 – 0,24) — tỷ lệ từ chối oan khách hàng tốt vẫn còn cao, cần cân nhắc kỹ khi triển khai thực tế.
- Tỷ lệ chi phí `COST_FN/COST_FP = 8×` hiện tại là giả định chủ quan, chưa được kiểm chứng bằng số liệu tài chính thực tế từ bộ phận rủi ro.
- SHAP hiện được tính trên mô hình baseline (XGBoost); nên tính lại trên mô hình LightGBM cuối cùng để đảm bảo tính nhất quán giữa mô hình triển khai và mô hình được diễn giải.

### 12.2. Hướng phát triển tương lai

- Triển khai kiến trúc Deep Learning (RNN/LSTM) để học sâu hơn các chuỗi tuần tự thời gian dài của hành vi thanh toán, thay vì chỉ dừng ở đặc trưng tĩnh dạng tổng hợp (aggregate features).
- Thu thập và nhúng thêm các nguồn Alternative Data khác (hành vi nạp tiền viễn thông, dữ liệu mạng xã hội, dữ liệu ví điện tử) để củng cố sức mạnh dự đoán cho nhóm khách hàng "hồ sơ trắng" hoàn toàn (không có cả dữ liệu `installments_payments`).
- Loại bỏ nhóm đặc trưng có `cv_ratio` cao (các cờ `FLAG_DOCUMENT_*` hiếm gặp) để giảm nhiễu và tăng tính diễn giải của mô hình.
- Thay thế giả định `COST_FN/COST_FP = 8×` bằng số liệu chi phí thực tế (giá trị khoản vay trung bình, tỷ lệ thu hồi nợ xấu...) để chọn ngưỡng vận hành chính xác hơn.
- Tính toán lại SHAP trên mô hình LightGBM đã tối ưu (sau Optuna) để đảm bảo bộ giải thích khớp với mô hình triển khai thực tế.

---

## 13. Kết luận

Dự án Shinsa AI đã hoàn thành và đáp ứng được các mục tiêu nghiệp vụ đề ra ban đầu:

- Xây dựng thành công mô hình phân loại rủi ro vỡ nợ với **ROC-AUC ổn định ~0,76** trên cả cross-validation (0,7606) và tập test độc lập (0,7657).
- Chứng minh được đặc trưng hành vi tự tổng hợp (`MAX_DAYS_LATE`, `TOTAL_INSTALLMENTS`) là tín hiệu dự báo rủi ro **ổn định và có ý nghĩa**, lọt Top 15 biến quan trọng nhất và Top 5 theo SHAP — có khả năng thay thế một phần vai trò của điểm ngoại kiểm khi các điểm này bị khuyết, đúng như giả thuyết kinh doanh ban đầu.
- Cung cấp hai chiến lược ngưỡng quyết định rõ ràng (Tăng trưởng và Phòng thủ) kèm phân tích độ nhạy theo tỷ lệ chi phí, giúp bộ phận kinh doanh linh hoạt lựa chọn theo khẩu vị rủi ro và bối cảnh kinh tế thực tế, thay vì phụ thuộc vào một ngưỡng 0,5 cứng nhắc.
- Ứng dụng SHAP để minh bạch hóa quyết định của mô hình, đáp ứng yêu cầu giải trình (explainability) — yếu tố then chốt trong lĩnh vực chấm điểm tín dụng.

Những hạn chế đã nêu (sự phụ thuộc vào `EXT_SOURCE`, Precision còn thấp, giả định chi phí chưa được kiểm chứng) là cơ sở rõ ràng cho các hướng cải tiến tiếp theo của dự án.
