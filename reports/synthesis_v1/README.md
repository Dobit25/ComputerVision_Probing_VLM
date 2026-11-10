# Báo Cáo Tổng Hợp Chẩn Đoán Cơ Chế, Độ Nhạy Prompt & Phân Mảnh BPE Tiếng Việt (`synthesis_v1`)

Tài liệu này được sinh tự động và tất định bởi `python -m vlm_probe.synthesis` từ các artifact trong `results/`.

---

## 1. P1.2 — Trạng Thái Xuất Bảng Chấm Mù Độc Lập (Blinded Review Sheets)

| Bộ dữ liệu | Số nguồn ảnh | File bảng chấm mù (CSV) | SHA-256 bảng chấm mù |
| --- | ---: | --- | --- |
| **Confirmation (`100` nguồn ngang)** | `100` | `data/blinded_review_v1/blinded_sheet_confirmation_100.csv` | `df1e19c0c36b6e44...` |
| **Exploratory (`200` nguồn 4 hướng)** | `200` | `data/blinded_review_v1/blinded_sheet_exploratory_200.csv` | `4594d65b97f956c4...` |

---

## 2. P2.1–P2.3 — Phân Rã Prior-to-Signal Ratio (`PSR`) & Khắc Phục Zero-Shot Cắt Vùng Không Gian

### 2.1. Phân rã Prior Bias $\lvert B \rvert$ so với Spatial Signal $\lvert \Delta_{\text{spatial}} \rvert$ trên các cặp đối xứng ngang

| Benchmark & Điều kiện | Số cặp | $\mathbb{E}[\lvert B \rvert]$ (Prior Bias) | $\mathbb{E}[\lvert \Delta_{\text{spatial}} \rvert]$ (Spatial Signal) | Tỷ số $\text{PSR} = \frac{\mathbb{E}[\lvert B \rvert]}{\mathbb{E}[\lvert \Delta \rvert]}$ | Tỷ lệ $\lvert B \rvert > \lvert \Delta \rvert$ (Collapse 1 nhãn) | 2×2 Assignment Acc | Both Images Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Synthetic Depth (`T0` thuận, ngang)** | `360` | `0.00314` | `0.00028` | **`11.32x`** | `98.61%` | `45.00%` | **`1.11%`** |
| **COCO Exploratory (`T2` thuận, ngang)** | `100` | `0.00495` | `0.00044` | **`11.18x`** | `93.00%` | `54.00%` | **`4.00%`** |
| **COCO Confirmation (`T2` thuận, ngang)** | `100` | `0.00546` | `0.00033` | **`16.69x`** | `95.00%` | `53.00%` | **`4.00%`** |

> **Kết luận cơ chế (`RQ2`):** Trên cả ảnh hình học tổng hợp lẫn ảnh tự nhiên COCO, độ lớn tuyệt đối của thiên kiến bất biến vị trí $\mathbb{E}[\lvert B \rvert]$ lớn gấp **`11.2`–`16.7` lần** tín hiệu đổi chỗ không gian $\mathbb{E}[\lvert \Delta_{\text{spatial}} \rvert]$. Do $\lvert B \rvert > \lvert \Delta_{\text{spatial}} \rvert$ trên hơn `93%–98,6%` số cặp, hai ảnh đối xứng bị đẩy về cùng một nhãn dự đoán, giải thích chính xác tại sao `both_images_correct` chỉ đạt `1,11%–4,00%` dù 2×2 assignment đạt `45%–54%`.

### 2.2. Kết quả Thực nghiệm GPU & Giới hạn hình học của Half-Canvas Zero-Shot Mitigation (`P2.2/P2.3`)

| Tập dữ liệu | Số nguồn ảnh | Tỷ lệ 2 vật thể nằm khác phía đường chia tâm | Giới hạn hình học lý tưởng | Kết quả CLIP `ViT-B/32` Holistic (Baseline) | Kết quả CLIP `ViT-B/32` Half-Canvas (`full_spatial_mitigation_v1`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Synthetic Depth (`1.440` ảnh)** | `1440` | **`100.00%`** | **`100.00%`** | `25.49%` (`0.00%` 4-relation) | **`100.00%`** (`100%` both-correct & 4-relation consistent) |
| **COCO Exploratory (`200` nguồn)** | `200` | `86.50%` | `92.50%` (`13.50%` cùng nửa) | `28.50%` | Giới hạn trần `92.50%` |
| **COCO Confirmation (`100` nguồn ngang)** | `100` | `93.00%` | `99.00%` (2-way, `7.00%` cùng nửa) | `49.00%` (`4.00%` both correct) | Giới hạn trần `93.00%`–`99.00%` |

---

## 3. P3.1–P3.4 — Độ Nhạy Prompt & Hiệu Quả Ensembling (Offline + Live GPU)

### 3.1. Thực nghiệm GPU Đa Mẫu Prompt trên CIFAR-100 (`results/full_prompt_cifar100_v1` — `10.000` ảnh)

| Mẫu Prompt / Ensemble | Công thức câu | Top-1 Accuracy [95% CI] | $\Delta$ vs `T0_baseline` [95% CI] |
| --- | --- | ---: | ---: |
| **`T0_baseline`** | `a photo of a {class}.` | **`64.48%`** `[63.64%, 65.31%]` | `0.00` pp |
| **`T1_minimal`** | `{class}.` | `60.39%` `[59.58%, 61.23%]` | `-4.09` pp `[-4.78, -3.43]` |
| **`T2_clear`** | `a clear photo of a {class}.` | `62.88%` `[62.01%, 63.73%]` | `-1.60` pp `[-2.10, -1.14]` |
| **`T3_centered`** | `a centered photo of a {class}.` | `64.26%` `[63.40%, 65.12%]` | `-0.22` pp `[-0.74, +0.28]` |
| **`T4_domain`** | `a low-resolution photo of a {class}.` | `64.02%` `[63.18%, 64.84%]` | `-0.46` pp `[-0.96, +0.01]` |
| **`T5_bright`** | `a bright photo of a {class}.` | `63.04%` `[62.19%, 63.89%]` | `-1.44` pp `[-1.96, -0.90]` |
| **`T6_cropped`** | `a cropped photo of a {class}.` | `64.30%` `[63.46%, 65.12%]` | `-0.18` pp `[-0.67, +0.29]` |
| **Logit Average Ensemble (`7` mẫu)** | Trung bình cosine similarity | **`64.95%`** `[64.13%, 65.78%]` | **`+0.47` pp** `[+0.09, +0.83]` |
| **Probability Average Ensemble (`7` mẫu)** | Trung bình xác suất Softmax ($\tau=100$) | **`65.00%`** `[64.19%, 65.82%]` | **`+0.52` pp** `[+0.16, +0.88]` |
| **Text-Embedding Ensemble (`7` mẫu)** | Chuẩn hóa L2 trung bình vector nhúng văn bản | **`64.97%`** `[64.14%, 65.78%]` | **`+0.49` pp** `[+0.10, +0.89]` |

### 3.2. Tác vụ Đếm (Counting — `5` mẫu × `1.080` cảnh) & Phủ định (Negation — `4` mẫu × `720` cảnh, `full_prompt_probing_v1`)

| Bài toán | Số templates | Best Single | Worst Single | Mean ± SD | Spread (Max − Min) | Text-Embedding Ensemble | Probability Ensemble |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Counting (`1.080` cảnh)** | `5` | `32.87%` (`C0_baseline`) | `20.83%` (`C2_there_are`) | `26.11% ± 4.57%` | **`12.04` pp** | `28.52%` | `28.06%` |
| **Negation (`720` cảnh, `4` mẫu)** | `4` | `91.39%` (`N0_with_without`) | `62.78%` (`N2_showing`) | `75.07% ± 13.50%` | **`28.61` pp** | `84.17%` | `84.44%` |
| **Synthetic Spatial (`T0..T2` thuận, 4-way)** | `3` | `27.36%` | `24.72%` | `25.67% ± 1.47%` | **`2.64` pp** | `23.68%` (Logit) | `23.96%` |
| **CIFAR-100 Semantic Valid (`baseline/cat/neutral`)** | `3` | `64.48%` | `62.92%` | `63.83% ± 0.81%` | **`1.56` pp** | `64.83%` (Logit) | `64.94%` |

### 3.3. Phân tích cơ chế cân bằng thiên kiến Present vs Absent trên Negation (`with_without` + `contains_does_not`)

| Điều kiện / Ensemble | Accuracy tổng (`720` cảnh) | Present (`360` cảnh) | Absent (`360` cảnh) | Chênh lệch `Present - Absent` | Group Consistency (`180` nhóm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`with_without` (Single tốt nhất)** | `91.39%` | `87.50%` | `95.28%` | `-7.78` pp | `82.78%` |
| **`contains_does_not` (Single thứ hai)** | `80.83%` | `80.83%` | `80.83%` | `+0.00` pp | `61.67%` |
| **Logit Average Ensemble** | **`86.81%`** | `84.72%` | `88.89%` | `-4.17` pp | **`73.61%`** |
| **Probability Average Ensemble** | **`86.81%`** | `84.72%` | `88.89%` | `-4.17` pp | **`73.61%`** |

- **Kiểm định Ensemble trên COCO Confirmation 2×2 Assignment (`100` cặp):** `T0` đạt `52.00%`, `T1` đạt `55.00%`, `T2` đạt `53.00%`; khi gộp **Logit Ensemble (`T0+T1+T2`)** đạt `54.00%` (CI 95% `[44.00%, 63.00%]`) và `both_images_correct` vẫn chỉ đạt **`1.00%`**. Điều này chứng minh **ensembling nhiều mẫu câu không thể giải cứu được hạn chế kiến trúc bag-of-words trên quan hệ không gian (`RQ3`)**.

---

## 4. P4.1 & P4.5 — Thực Nghiệm GPU Chéo Ngôn Ngữ Tiếng Việt & Chẩn Đoán Phân Mảnh BPE (`full_vietnamese_cifar100_v1`)

### 4.1. Kết quả Top-1 Accuracy & Độ tương đồng Cosine nhúng văn bản trên 10.000 ảnh CIFAR-100

| Điều kiện ngôn ngữ (`P4.1` & `P4.5`) | Mẫu câu | Top-1 Accuracy [95% CI] | $\Delta$ vs `en_baseline` | Cosine $\cos(\hat{t}_{\text{en}}, \hat{t}_{\text{cond}})$ (Mean ± SD) |
| --- | --- | ---: | ---: | ---: |
| **`en_baseline`** | `a photo of a {en_label}.` | **`64.48%`** `[63.64%, 65.31%]` | `0.00` pp | `1.000 ± 0.000` |
| **`vi_diacritics`** | `một bức ảnh chụp {vi_label}.` | **`5.10%`** `[4.88%, 5.31%]` | **`-59.38` pp** | `0.658 ± 0.057` |
| **`vi_no_diacritics`** | `mot buc anh chup {vi_no_diacritics}.` | **`4.05%`** `[3.87%, 4.25%]` | **`-60.43` pp** | `0.702 ± 0.055` |
| **`en_frame_vi_label` (`P4.5`)** | `a photo of a {vi_label}.` | **`5.44%`** `[5.17%, 5.72%]` | **`-59.04` pp** | `0.692 ± 0.078` |
| **`vi_frame_en_label` (`P4.5`)** | `một bức ảnh chụp {en_label}.` | **`55.10%`** `[54.23%, 55.96%]` | **`-9.38` pp** | `0.773 ± 0.034` |
| **`vi_en_bilingual`** | `một bức ảnh chụp {vi_label} ({en_label}).` | **`54.95%`** `[54.10%, 55.75%]` | **`-9.53` pp** | `0.761 ± 0.032` |
| **Ensemble (`en + vi + bilingual`)** | Text-Embedding Ensemble | **`60.42%`** `[59.59%, 61.28%]` | `-4.06` pp | — |

### 4.2. Chỉ số Phân mảnh BPE thực tế trên `SimpleTokenizer` của OpenAI CLIP (`49.408` vocab)

| Điều kiện ngôn ngữ | Số BPE tokens / tên lớp (Mean ± SD) | Số BPE tokens / cả câu prompt (Mean ± SD) | Token Fertility (Tokens / từ đơn) | Tỷ số phân mảnh tên lớp so với Tiếng Anh | Số câu vượt ngưỡng `77` tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`en_baseline` (Tiếng Anh)** | `1.19 ± 0.49` | `6.19 ± 0.49` | **`1.10`** | `1.00x` | `0` |
| **`vi_diacritics` (Tiếng Việt có dấu)** | **`7.05 ± 2.82`** | **`27.05 ± 2.82`** | **`3.09`** | **`6.44x`** | `0` |
| **`vi_no_diacritics` (Tiếng Việt không dấu)** | `2.88 ± 1.07` | `10.88 ± 1.07` | `1.24` | `2.62x` | `0` |
| **`en_frame_vi_label` (`P4.5`)** | `7.05 ± 2.82` | `12.05 ± 2.82` | `3.09` | `6.44x` | `0` |
| **`vi_frame_en_label` (`P4.5`)** | `1.19 ± 0.49` | `21.19 ± 0.49` | `1.10` | `1.00x` | `0` |
| **`vi_en_bilingual` (Song ngữ Việt–Anh)** | `10.24 ± 2.89` | `29.24 ± 2.89` | `3.03` | `9.28x` | `0` |

> **Phát hiện định lượng (`RQ4` & `P4.5`):**
> 1. **Phân mảnh BPE (`SimpleTokenizer` thực tế):** Một tên lớp tiếng Việt có dấu bị bẻ thành trung bình **`7.05` subword tokens** (Fertility = **`3.09` tokens/từ**), gấp **`6.44` lần** tên lớp tiếng Anh (`1.19` tokens); cả câu prompt tiếng Việt có dấu dài **`27.05` tokens** so với `6.19` tokens tiếng Anh (`4.39x`).
> 2. **Bóc tách nhân quả Khung câu vs. Tên lớp (`P4.5`):** Khi giữ khung tiếng Anh nhưng thay nhãn tiếng Việt (`en_frame_vi_label`: `"a photo of a cá heo."`), độ chính xác sụp đổ xuống **`5.44%`** (tương đương `vi_diacritics` `5.10%`). Ngược lại, khi dùng khung tiếng Việt với nhãn tiếng Anh (`vi_frame_en_label`: `"một bức ảnh chụp dolphin."`), độ chính xác phục hồi vọt lên **`55.10%`** (`+50.00` pp so với tiếng Việt thuần, tương đương song ngữ `vi_en_bilingual` `54.95%`). Điều này chứng minh **hơn 84% mức suy giảm chéo ngôn ngữ (`-50.00` pp trên tổng `-59.38` pp) xuất phát từ việc bộ mã hóa văn bản CLIP mù ngữ nghĩa danh từ tên lớp tiếng Việt bị phân mảnh BPE, trong khi khung cú pháp tiếng Việt chỉ gây nhiễu nhẹ (`-9.38` pp)**.
