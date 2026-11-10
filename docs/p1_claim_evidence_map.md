# P1: Bản đồ kết luận – bằng chứng (Claim–Evidence Map) và Tổng hợp P1.3 / P1.4

Ngày lập: **09/11/2026** (Asia/Saigon).
Bản dữ liệu máy đọc được: [docs/references/claim_evidence_map.json](references/claim_evidence_map.json).

Tài liệu này thực thi mục 5 (`P1.1`, `P1.3`, `P1.4`) của [long_term_project_plan.md](../long_term_project_plan.md):
- Mọi kết luận xuất hiện trong Abstract, Main Text và Conclusion của báo cáo phải ánh xạ trực tiếp tới `claim → metric → run → controls → uncertainty → limitation`.
- Không còn kết luận sai vì nhầm mẫu số, nhầm oracle 5-way với 100-way, nhầm độ nhạy mẫu câu với "không hiểu phủ định", hoặc suy diễn $p > 0,05$ thành "tương đương đoán ngẫu nhiên".

---

## 1. P1.1 — Bản đồ kết luận chính (`claim → metric → run → controls → uncertainty → limitation`)

| Mã Claim | Nội dung kết luận được phép viết | Metric & Giá trị chính xác | Run / Artifact nguồn | Đối chứng (Controls) | Bất định (CI 95% / p-value) | Giới hạn bắt buộc & Cách hiểu bị cấm |
|---|---|---|---|---|---|---|
| **C01 (Baseline 5 miền)** | Hiệu năng zero-shot của OpenAI CLIP ViT-B/32 biến thiên mạnh qua 5 bộ dữ liệu chuẩn (từ 42,45% trên DTD p1 đến 85,06% trên Oxford-IIIT Pet), và ngay trong bộ dữ liệu có accuracy tổng cao vẫn tồn tại các lớp có accuracy 0% (`Bombay` 0/88 trên Pet; 5 lớp 0% trên Flowers102). | - CIFAR-100: **64,4800%** (6.448/10.000)<br>- Flowers102: Top-1 **63,7665%** (3.921/6.149), Macro **64,2338%**<br>- Pet: Top-1 **85,0641%** (3.121/3.669), Macro **84,7193%**<br>- DTD p1: **42,4468%** (798/1.880)<br>- Food-101: **82,3129%** (20.784/25.250) | - `results/full_cifar100_clip_b32_v1`<br>- `results/full_flowers102_clip_b32_v2`<br>- `results/full_oxford_pet_clip_b32_v3`<br>- `results/full_dtd_clip_b32_v3`<br>- `results/full_food101_clip_b32_v3`<br>- [reports/baseline/summary.csv](../reports/baseline/summary.csv) | Cùng một checkpoint, cùng FP32/QuickGELU, một template cố định (`a photo of a {class}.`, DTD thêm `texture`). | Đánh giá toàn bộ tập test chính thức (46.948 ảnh). | **Cấm:** Gộp 5 dataset thành một accuracy trung bình theo ảnh (Food-101 sẽ chi phối) hoặc diễn giải chênh lệch giữa Pet (85,06%) và DTD (42,45%) như sự cải tiến của model. |
| **C02 (Counting Core — P1.3)** | Trên bộ 1.080 ảnh hình học (số lượng 1–6), CLIP đạt 32,87% (cao hơn mức ngẫu nhiên 16,67% nhưng MAE = 1,0157 vật thể); khó khăn đếm vẫn tồn tại ngay trên tập con nhận dạng đúng màu/hình (34,45%), và gần như không đổi giữa chế độ bán kính cố định (32,41%) và tổng diện tích cố định (33,33%). | - Toàn tập: **355/1.080 = 32,87%**<br>- MAE: **1,0157**, signed error: **−0,1954**<br>- `fixed_radius`: **175/540 = 32,41%**<br>- `fixed_total_area`: **180/540 = 33,33%**<br>- Nhóm đúng cả 6 số lượng: **0/180 = 0,00%** | - `results/full_probing_clip_b32_v1/metrics.json`<br>- `results/full_depth_historical_v1/metrics.json`<br>- [reports/probing/breakdown.csv](../reports/probing/breakdown.csv) | Control nhận dạng 9 tổ hợp màu/hình đạt **804/1.080 = 74,44%**; counting trên 804 mẫu control đúng đạt **277/804 = 34,45%**. | Group bootstrap 95% CI (180 nhóm cảnh): **`[30,65%; 35,00%]`**. | **Cấm:** Xóa 276 mẫu control sai khỏi metric chính (32,87%), hoặc phát biểu quy luật đơn điệu "số lượng càng lớn càng sai" (vì số 3 đạt 52,22%, số 5 đạt 42,22%, trong khi số 4 chỉ đạt 9,44%). |
| **C03 (Negation Core — P1.3)** | Trong cảnh hình học hai vật thể có đối chứng số lượng, CLIP phân biệt câu khẳng định/phủ định khá tốt (**86,11%**, không thiên lệch chọn câu khẳng định — tỷ lệ chọn khẳng định 48,06%), nhưng độ chính xác nhạy với mẫu câu (chênh **10,56 điểm %** giữa `with/without` và `contains/does not contain`) và chỉ **58,61%** nhóm cảnh đúng đồng thời cả hai trạng thái present/absent dưới cả hai mẫu câu. | - Chung: **1.240/1.440 = 86,11%**<br>- `with_without`: **658/720 = 91,39%**<br>- `contains_does_not`: **582/720 = 80,83%**<br>- `present`: **606/720 = 84,17%**<br>- `absent`: **634/720 = 88,06%**<br>- Nhóm đúng cả 4 phép thử: **211/360 = 58,61%** | - `results/full_probing_clip_b32_v1/metrics.json`<br>- [reports/probing/README.md](../reports/probing/README.md) | Control khẳng định nhận dạng cặp vật thể hiện diện đạt **1.406/1.440 = 97,64%**; negation khi control đúng đạt **1.222/1.406 = 86,91%**. | Group bootstrap 95% CI (360 nhóm cảnh, không coi 1.440 dòng là độc lập): **`[84,23%; 88,06%]`**. | **Cấm:** Viết "CLIP không hiểu phủ định" dựa trên kết quả của dự án này, hoặc suy rộng kết quả 86,11% trên cảnh hình học 2 vật thể sang mọi cảnh tự nhiên nhiều vật thể phức tạp. |
| **C04 (Fine-grained — P1.4)** | Khi dùng nhãn coarse thật (oracle) để thu hẹp tập ứng viên CIFAR-100 từ 100 xuống 5 lớp cùng nhóm, mô hình vẫn mắc **2.205/10.000 lỗi (accuracy 77,95%)**, tập trung ở các nhóm sinh học gần nhau (`fish` 56,60%, `aquatic_mammals` 64,00%, `trees` 66,20%); trên các tập tự nhiên không dùng oracle (Flowers102, Pet, DTD, Food-101), lỗi tập trung vào các cặp lớp gần nghĩa hoặc gần hình thái. | - CIFAR-100 oracle 5-way: **7.795/10.000 = 77,95%** (signed margin trung bình `+0,018540`)<br>- Parity 100-way: khớp **10.000/10.000** dự đoán với baseline **64,48%** | - `results/full_probing_clip_b32_v1/metrics.json`<br>- [reports/baseline/confusion_pairs.csv](../reports/baseline/confusion_pairs.csv) | Đối chiếu trực tiếp dự đoán global 100-way trên cùng vector ảnh/text. | Image bootstrap 95% CI: **`[77,16%; 78,77%]`** (chance 20%). | **Cấm:** Gọi 77,95% là "phương pháp cải thiện accuracy so với 64,48%" vì bài 5-way sử dụng nhãn nhóm lớn thật (oracle) của ảnh test. |
| **C05 (Spatial Synthetic Core & Depth)** | Trong cả bộ hình học core (720 ảnh) và bộ can thiệp mở rộng (1.440 ảnh × 6 điều kiện câu), nhận dạng hai vật thể đạt ~97% nhưng phân loại 4 quan hệ không gian chỉ đạt **29,58%** (core) và **27,36%** (depth T0 thuận), kèm thiên lệch mạnh chọn `above` (55,69%–56,18%), tỷ lệ đúng đồng thời cặp trái/phải chỉ **0,56%** và đồng thuận giữa câu thuận–đảo tương đương chỉ **16,25%**. | - Core (720 ảnh): **213/720 = 29,58%**<br>- Depth T0 thuận (1.440 ảnh): **394/1.440 = 27,36%**<br>- Cặp trái/phải cùng đúng: **2/360 = 0,56%**<br>- Cặp trên/dưới cùng đúng: **6/360 = 1,67%**<br>- Đúng cả 4 quan hệ: **0,00%**<br>- Đồng thuận câu thuận–đảo T0: **16,25%** (cùng đúng **2,99%**) | - `results/full_probing_clip_b32_v1`<br>- `results/full_depth_spatial_v1/metrics.json`<br>- [reports/probing_depth_v1/spatial_conditions.csv](../reports/probing_depth_v1/spatial_conditions.csv) | - Control nhận dạng 2 vật thể: **97,50%** (core) và **96,94%** (depth; quan hệ khi control đúng đạt **27,51%**)<br>- Đối chứng nền trống (`blank_prediction_agreement`): chỉ trùng **23,82%**. | Cluster bootstrap 95% CI theo 120 nhóm bố cục (2.000 draws, seed 1402): T0 thuận **`[26,18%; 28,61%]`**. | **Cấm:** Quy toàn bộ lỗi cho một đáp án cố định trên nền trống (vì chỉ trùng nền trống 23,82%), hoặc khẳng định lỗi nằm ở riêng vision encoder hay text encoder khi chưa can thiệp nội tại. |
| **C06 (Category Controls trên CIFAR-100)** | Thông tin category đúng mang tín hiệu ngữ nghĩa thật (tốt hơn trung bình 5 ánh xạ category sai **+8,734 điểm %**), nhưng mẫu câu `a photo of a {class}, a type of {category}.` làm giảm Top-1 accuracy so với baseline (**−1,56 điểm %**, từ 64,48% xuống 62,92%) và thấp hơn cả suffix trung tính khớp độ dài token (**−1,17 điểm %**), do điểm cosine của đối thủ mạnh nhất tăng nhiều hơn điểm của lớp đúng. | - Baseline: **64,48%**<br>- Category đúng: **62,92%** ($\Delta = -1,56$ điểm %)<br>- Neutral suffix: **64,09%** ($\Delta = -0,39$ điểm %)<br>- 5 Category sai (`shuffle_271..275`): **52,21%–55,76%** (TB **54,186%**) | - `results/full_cifar100_category_v1`<br>- `results/full_depth_category_v1/metrics.json`<br>- `results/full_depth_historical_v1/metrics.json` | - Neutral suffix khớp số token CLIP ở 80/100 lớp (lệch ±1 ở 20 lớp)<br>- 5 hoán vị derangement (không lớp nào giữ category đúng)<br>- Điểm lớp đúng tăng `+0,001628` nhưng điểm đối thủ mạnh nhất tăng `+0,002908` (margin giảm `−0,001280`). | Class-stratified paired bootstrap 95% CI (2.000 draws):<br>- Đúng − Baseline: **`[−2,14; −0,98]`**<br>- Đúng − Neutral: **`[−1,77; −0,61]`**<br>- Đúng − TB 5 Sai: **`[+8,140; +9,330]`** | **Cấm:** Nói "thêm category luôn có hại với mọi prompt" hoặc "luôn nâng điểm cùng một đối thủ cố định" (vì đối thủ mạnh nhất đổi danh tính ở 33,75% ảnh; giữ cố định đối thủ baseline thì $\Delta = -0,000111$). |
| **C07 (Spatial Natural COCO — Thăm dò & Xác nhận mới)** | Trên ảnh tự nhiên COCO `val2014`, phân loại 4 quan hệ (T2 thuận) đạt **28,00%** (28,41% trên 176 ảnh sau tự rà soát) với thiên lệch chọn `right` (62,00% — ngược với thiên lệch `above` trên hình học); trên tập xác nhận định trước gồm **100 nguồn ảnh ngang mới** (độc lập với 400 ứng viên cũ), phép ghép cặp 2×2 đạt **53/100 (53,00%)**, chưa đủ bằng chứng thống kê vượt mốc 50% ($p = 0,3086$), và chỉ **4/100 cặp** có cả ảnh gốc lẫn bản lật được dự đoán đúng riêng rẽ. | - Tập cũ 200 ảnh (4-way T2 thuận): **56/200 = 28,00%** (cặp gốc–lật ngang cùng đúng **3/100 = 3,00%**; ghép 2×2 hậu nghiệm **54/100 = 54,00%**, sau tự rà soát **52/93 = 55,91%**)<br>- Tập xác nhận mới 100 nguồn ngang (T2 thuận): Ghép 2×2 **53/100 = 53,00%**; 2-way từng ảnh **103/200 = 51,50%**; cả 2 ảnh đúng riêng rẽ **4/100 = 4,00%** | - `results/full_depth_natural_v1`<br>- `results/full_natural_confirmation_v1/metrics.json`<br>- [reports/natural_confirmation_v1/metrics.json](../reports/natural_confirmation_v1/metrics.json) | - Loại trừ toàn bộ 400 source ID cũ + sàng lọc dHash ≤ 6 + đóng băng danh sách trước suy luận<br>- Hiệu 2×2 ($D = S_{00} + S_{11} - S_{01} - S_{10}$) triệt tiêu thiên lệch cộng chỉ phụ thuộc từng ảnh hoặc từng caption. | - Tập mới 2×2 T2 thuận: CI Wilson 95% **`[43,29%; 62,49%]`**, bootstrap **`[43,00%; 63,00%]`**, exact one-sided binomial **$p = 0,3086$**.<br>- Chênh lệch Mới − Cũ: **−1,00 điểm %**, CI **`[−15,00; +12,00]`**. | **Cấm:** Kết luận "$p = 0,5$" hay "CLIP hoàn toàn không có thông tin vị trí / tương đương đoán ngẫu nhiên" từ $p = 0,3086$; cấm gọi rà soát của cùng trợ lý AI là human annotation độc lập; cấm trộn 53% (ghép cặp 2×2, mốc 50%) với 28% (phân loại 4-way, mốc 25%). |

---

## 2. P1.3 — Bảng tổng hợp chi tiết Counting và Negation từ Scores đã kiểm chứng

### 2.1. Counting (1.080 ảnh, 6 lựa chọn `one`..`six`, chance 16,67%)
Nguồn: `results/full_probing_clip_b32_v1/metrics.json` và `results/full_depth_historical_v1/metrics.json`.

- **Tổng hợp theo số lượng thật (`stratum` 1–6, mỗi mức 180 ảnh):**

| Số lượng thật | Support | Đúng | Accuracy (%) | Mean Signed Margin | Tổng số lần được mô hình chọn (`prediction_counts`) |
|---:|---:|---:|---:|---:|---:|
| **1 (`one`)** | 180 | 87 | 48,33% | −0,000258 | 189 |
| **2 (`two`)** | 180 | 47 | 26,11% | −0,003491 | 138 |
| **3 (`three`)** | 180 | 94 | **52,22%** | +0,000185 | 312 |
| **4 (`four`)** | 180 | 17 | **9,44%** | −0,005019 | 99 |
| **5 (`five`)** | 180 | 76 | 42,22% | −0,001598 | 280 |
| **6 (`six`)** | 180 | 34 | 18,89% | −0,009184 | 62 |
| **Toàn tập** | **1.080** | **355** | **32,87%** | **−0,003227** | **1.080** |

- **Ma trận nhầm lẫn Counting 6 × 6 (Hàng: Nhãn thật 1–6; Cột: Dự đoán 1–6):**
  - Thật = 1: `[87, 82, 11, 0, 0, 0]` (nhầm chủ yếu sang 2)
  - Thật = 2: `[22, 47, 102, 1, 8, 0]` (nhầm mạnh sang 3: 102/180 ảnh)
  - Thật = 3: `[20, 9, 94, 5, 48, 4]` (nhầm sang 5: 48/180 ảnh)
  - Thật = 4: `[20, 0, 69, 17, 63, 11]` (bị kéo về 3 và 5; chỉ đúng 17/180 ảnh)
  - Thật = 5: `[20, 0, 30, 41, 76, 13]`
  - Thật = 6: `[20, 0, 6, 35, 85, 34]` (nhầm mạnh sang 5: 85/180 ảnh)
- **Phân tích theo yếu tố phụ:**
  - Chế độ kích thước: `fixed_radius` **175/540 (32,41%)** vs `fixed_total_area` **180/540 (33,33%)**.
  - Theo màu sắc: `blue` **133/360 (36,94%)**, `green` **121/360 (33,61%)**, `red` **101/360 (28,06%)**.
  - Theo hình dạng: `circle` **127/360 (35,28%)**, `triangle` **126/360 (35,00%)**, `square` **102/360 (28,33%)**.
  - Sai số tuyệt đối trung bình (MAE): **1,0157 vật thể**; sai số có dấu trung bình (Mean Signed Error): **−0,1954 vật thể** (xu hướng dự đoán thấp hơn số thật ở các mức 4, 5, 6).

### 2.2. Negation (720 ảnh duy nhất, 1.440 phép thử, 2 lựa chọn, chance 50,00%)
Nguồn: `results/full_probing_clip_b32_v1/metrics.json` và `results/full_depth_historical_v1/metrics.json`.

| Phân nhóm | Số phép thử | Đúng | Accuracy (%) | Mean Signed Margin | Ghi chú diễn giải |
|---|---:|---:|---:|---:|---|
| **Toàn bộ (`all`)** | 1.440 | 1.240 | **86,11%** | +0,009436 | CI 95% theo 360 nhóm cảnh: `[84,23%; 88,06%]` |
| **Mẫu A (`with_without`)** | 720 | 658 | **91,39%** | +0,011191 | `a drawing with/without a {color} {shape}.` |
| **Mẫu B (`contains_does_not`)** | 720 | 582 | **80,83%** | +0,007680 | `a drawing that contains / does not contain...` (kém Mẫu A **10,56 điểm %**) |
| **Trạng thái Có mặt (`present`)** | 720 | 606 | **84,17%** | +0,007683 | Nhầm sang câu phủ định: 114/720 lần |
| **Trạng thái Vắng mặt (`absent`)** | 720 | 634 | **88,06%** | +0,011188 | Nhầm sang câu khẳng định: 86/720 lần |
| **Khi Control nhận dạng đúng** | 1.406 | 1.222 | **86,91%** | — | Control nhận dạng cặp vật thể đúng đạt **1.406/1.440 = 97,64%** |
| **Nhất quán toàn nhóm (360 nhóm)** | 360 nhóm | 211 nhóm | **58,61%** | — | Đúng đồng thời cả `present` + `absent` dưới cả 2 template |

---

## 3. P1.4 — Tổng hợp Fine-grained: Chẩn đoán Oracle 5-way vs Các bộ dữ liệu tự nhiên đầy đủ

### 3.1. Phân biệt rõ hai thiết kế đánh giá Fine-grained trong dự án
1. **Chẩn đoán trong nhóm có Oracle trên CIFAR-100 (`results/full_probing_clip_b32_v1`):**
   - Cung cấp nhãn nhóm lớn thật (`coarse_id` 0–19) để giới hạn tập ứng viên từ 100 xuống đúng **5 lớp cùng nhóm** (chance 20%).
   - Đạt **7.795/10.000 = 77,95%** (CI 95% `[77,16%; 78,77%]`), tức vẫn còn **2.205 lỗi nội nhóm** dù đã loại bỏ hoàn toàn 95 lớp thuộc 19 nhóm khác.
   - Ba nhóm có lỗi nội nhóm cao nhất: `fish` (**56,60%** — nhầm `ray`/`trout` sang `flatfish`), `aquatic_mammals` (**64,00%** — `beaver`, `dolphin`, `otter`, `seal`, `whale`), `trees` (**66,20%** — `maple`, `oak`, `palm`, `pine`, `willow`).
   - Ba nhóm có độ phân biệt nội nhóm cao nhất: `vehicles_1` (**92,20%**), `large_man-made_outdoor_things` (**91,20%**), `household_electrical_devices` / `large_omnivores_and_herbivores` / `vehicles_2` (**89,80%**).
2. **Đánh giá toàn tập ứng viên không có Oracle trên 5 dataset ([reports/baseline/README.md](../reports/baseline/README.md)):**
   - Mọi ảnh phải cạnh tranh trên toàn bộ số lớp của dataset (100 lớp CIFAR-100, 102 lớp Flowers102, 37 lớp Oxford-IIIT Pet, 47 lớp DTD, 101 lớp Food-101).
   - Các cặp nhầm lẫn lớn nhất đều thuộc các lớp gần nhau về hình thái thị giác hoặc chồng lấn tên gọi trong ngôn ngữ:
     - **Flowers102** (Top-1 63,77%, Macro 64,23%): `marigold → english marigold` (**47/47 = 100,00%**), `geranium → pelargonium` (**82/94 = 87,23%**), `cape flower → gaura` (**66/88 = 75,00%**), `petunia → mexican petunia` (**146/238 = 61,34%**).
     - **Oxford-IIIT Pet** (Top-1 85,06%, Macro 84,72%): `Ragdoll → Birman` (**67/100 = 67,00%**), `Bombay → Russian Blue` (**52/88 = 59,09%**, lớp `Bombay` đúng **0/88**), `Boxer → American Bulldog` (**32/99 = 32,32%**), `Persian → Birman` (**30/100 = 30,00%**).
     - **DTD partition 1** (Top-1 = Macro = 42,45%): `dotted → polka-dotted` (**36/40 = 90,00%**), `banded → striped` (**35/40 = 87,50%**), `lacelike → paisley` (**25/40 = 62,50%**), `lined → striped` (**25/40 = 62,50%**).
     - **Food-101** (Top-1 = Macro = 82,31%): `onion_rings → fried_calamari` (**70/250 = 28,00%**), `steak → filet_mignon` (**68/250 = 27,20%**) và `steak → prime_rib` (**38/250 = 15,20%**).

---

## 4. Trạng thái Gate P1

- [x] Mọi kết luận chính (`C01`–`C07`) đều có đường dẫn artifact, metric, control, khoảng tin cậy 95% và giới hạn diễn giải.
- [x] Không trộn lẫn các mẫu số hoặc số lựa chọn khác nhau (100-way, 5-way oracle, 6-way counting, 4-way spatial, 2-way negation, 2×2 pair assignment).
- [x] Sổ dữ liệu đã khóa trạng thái tiếp xúc của từng split (`docs/references/data_ledger.json`).
- [x] Quy trình kiểm nhãn độc lập bởi người (`P1.2`) được lập thành tài liệu riêng tại [docs/p1_independent_review_protocol.md](p1_independent_review_protocol.md).

