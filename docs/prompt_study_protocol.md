# Giao Thức Nghiên Cứu Độ Nhạy Prompt & Gộp Nhiều Prompt (Prompt Sensitivity & Ensembling — P3.1–P3.4)

- **Phiên bản giao thức:** `prompt_study_v1`
- **Model cố định:** `OpenAI CLIP ViT-B/32` (`QuickGELU`, `FP32`, `batch_size = 32`, `seed = 0`)
- **Mục tiêu:** Trả lời câu hỏi nghiên cứu **`RQ3`** trong [`docs/p0_requirements_and_rqs.md`](p0_requirements_and_rqs.md):
  > *Độ nhạy với cách viết prompt (prompt sensitivity) biến thiên như thế nào giữa các tác vụ nhận diện chuẩn và tác vụ suy luận cấu trúc, và phương pháp gộp nhiều prompt (text-embedding ensemble hoặc logit/probability average ensemble) cải thiện được những lỗi nào và **không** thể cứu vãn những lỗi nào?*

---

## 1. P3.1 — Bộ Template Tiền Đăng Ký Theo Từng Tác Vụ

Mọi mẫu câu (templates) dưới đây được khóa cố định trong [`configs/prompt_study_v1.json`](../configs/prompt_study_v1.json) trước khi chạy suy luận. Tuyệt đối không tìm kiếm hay lọc chọn template dựa trên tập test (`no post-hoc prompt cherry-picking`).

### 1.1. Năm bộ dữ liệu phân loại chuẩn (`cifar100`, `flowers102`, `oxford_pet`, `dtd`, `food101`)
Mỗi bộ dữ liệu được đánh giá trên **7 template đơn cố định** (`T0`–`T6`):
1. **`T0_baseline` (Chuẩn đối chiếu lịch sử):** `a photo of a {class}.`
   - *Ràng buộc bất biến:* Dự đoán và điểm cosine của `T0_baseline` phải khớp tuyệt đối (`tolerance <= 1e-6`) với 5 run baseline đã đóng băng trong `results/full_*_clip_b32_v*`.
2. **`T1_minimal` (Nhãn trần tối giản):** `{class}.`
3. **`T2_clear` (Mô tả chất lượng ảnh):** `a clear photo of a {class}.`
4. **`T3_centered` (Mô tả bố cục trung tâm):** `a centered photo of a {class}.`
5. **`T4_domain` (Ngữ cảnh đặc thù miền dữ liệu):**
   - `cifar100`: `a low-resolution photo of a {class}.`
   - `flowers102`: `a close-up photo of a {class}, a type of flower.`
   - `oxford_pet`: `a centered photo of a {class}, a domestic pet.`
   - `dtd`: `a close-up surface texture of {class}.`
   - `food101`: `a dish of {class}, a type of food.`
6. **`T5_bright` (Biến thể ánh sáng OpenAI-style):** `a bright photo of a {class}.`
7. **`T6_cropped` (Biến thể khung hình OpenAI-style):** `a cropped photo of a {class}.`

### 1.2. Ba tác vụ Probing cấu trúc (`counting`, `spatial`, `negation`)
1. **Counting (`5` templates trên 1.080 ảnh hình học, ứng viên $k = 1, \dots, 6$):**
   - `C0_baseline`: `a drawing of {word} {color} {shape_plural}.` (khớp `full_probing_clip_b32_v1`)
   - `C1_showing`: `an image showing {word} {color} {shape_plural}.`
   - `C2_there_are`: `there are {word} {color} {shape_plural} in this drawing.`
   - `C3_minimal`: `{word} {color} {shape_plural}.`
   - `C4_exactly`: `a picture with exactly {word} {color} {shape_plural}.`
2. **Spatial (`3` templates × `2` trật tự mệnh đề đã tiền đăng ký trong `probing_depth_v1`):**
   - `T0`: `a {medium} of {a} {relation} {b}.`
   - `T1`: `a {medium} where {a} is {relation} {b}.`
   - `T2`: `a {medium} showing {a} {relation} {b}.`
   - Kết hợp với trật tự thuận (`o0`: $A, B, R$) và trật tự nghịch đảo tương đương ngữ nghĩa (`o1`: $B, A, R^{-1}$).
3. **Negation (`4` cặp template khẳng định/phủ định trên 720 cảnh hình học):**
   - `N0_with_without`: `a drawing with a {target}.` vs. `a drawing without a {target}.` (khớp `full_probing_clip_b32_v1`)
   - `N1_contains`: `a drawing that contains a {target}.` vs. `a drawing that does not contain a {target}.` (khớp `full_probing_clip_b32_v1`)
   - `N2_showing`: `a drawing showing a {target}.` vs. `a drawing not showing a {target}.`
   - `N3_present_absent`: `an image where a {target} is present.` vs. `an image where a {target} is absent.`

---

## 2. P3.2 — Hai Phương Pháp Gộp Prompt (Ensembling) Tường Minh

Với một tập gồm $M$ templates tiền đăng ký $(T_1, \dots, T_M)$ và $C$ lớp ứng viên:

### 2.1. Phương pháp 1: Text-Embedding Ensemble (CLIP-style)
1. Với mỗi lớp $c = 1, \dots, C$ và mỗi template $m = 1, \dots, M$, trích xuất vector văn bản và chuẩn hóa L2:

$$
\hat{t}_{c,m} = \frac{f_{\text{text}}(T_m(c))}{\lVert f_{\text{text}}(T_m(c)) \rVert_2} \in \mathbb{R}^D
$$

2. Tính trung bình cộng các vector đã chuẩn hóa theo từng lớp:

$$
\bar{t}_c = \frac{1}{M}\sum_{m=1}^{M} \hat{t}_{c,m}
$$

3. **Chuẩn hóa L2 lần thứ hai** (bắt buộc để bảo toàn hình học tích vô hướng cosine trên mặt cầu đơn vị):

$$
\hat{\bar{t}}_c = \frac{\bar{t}_c}{\lVert \bar{t}_c \rVert_2}
$$

4. Tính điểm cosine với vector ảnh đã chuẩn hóa $\hat{v}(I)$:

$$
s_{\text{emb-ens}}(I, c) = \hat{v}(I)^\top \hat{\bar{t}}_c
$$

### 2.2. Phương pháp 2: Logit / Probability Average Ensemble
1. Tính điểm cosine độc lập cho từng template $m$: $s_m(I, c) = \hat{v}(I)^\top \hat{t}_{c,m}$.
2. **Logit Average Ensemble (Trung bình điểm cosine/logit):**

$$
s_{\text{logit-ens}}(I, c) = \frac{1}{M}\sum_{m=1}^{M} s_m(I, c)
$$

*(Lưu ý toán học: Do tính tuyến tính của tích vô hướng, $\arg\max_c s_{\text{logit-ens}}(I, c)$ chỉ khác $\arg\max_c s_{\text{emb-ens}}(I, c)$ khi độ dài chuẩn $\lVert \bar{t}_c \rVert_2$ — tức mức độ đồng thuận hướng giữa các template của lớp $c$ — khác nhau giữa các lớp $c$!)*

3. **Probability Average Ensemble (Trung bình xác suất Softmax với hệ số nhiệt độ CLIP $\tau^{-1} = 100.0$):**

$$
p_m(c ; I) = \frac{\exp(\tau^{-1} s_m(I, c))}{\sum_{c'=1}^C \exp(\tau^{-1} s_m(I, c'))}, \qquad p_{\text{prob-ens}}(c ; I) = \frac{1}{M}\sum_{m=1}^{M} p_m(c ; I)
$$

---

## 3. P3.3 & P3.4 — Hệ Thống Chỉ Số Báo Cáo & Chẩn Đoán Cơ Chế

Trên mọi bảng thực nghiệm của P3, bắt buộc báo cáo đủ 6 đại lượng:
1. **`best_single_accuracy_percent`** (kèm định danh template tốt nhất)
2. **`worst_single_accuracy_percent`** (kèm định danh template kém nhất)
3. **`mean_single_accuracy_percent`** và **`sd_single_accuracy_percent`** (Trung bình ± Độ lệch chuẩn qua $M$ templates)
4. **`spread_max_minus_min_pp`** (Biên độ dao động $\max - \min$ tính bằng điểm phần trăm)
5. **`ensemble_accuracy_percent`** (cho cả `text_embedding_ensemble`, `logit_ensemble`, `prob_ensemble`)
6. **Hiệu số ghép cặp kèm khoảng tin cậy 95% (Paired / Cluster Bootstrap 95% CI):**
   - $\Delta(\text{Ensemble} - \text{Baseline } T_0)$
   - $\Delta(\text{Ensemble} - \text{Best Single})$

### Các kiểm định cơ chế trọng tâm (P3.4):
- **Trên Negation:** Kiểm tra xem gộp `with/without` (`91,39%`) và `contains/does not contain` (`80,83%`) có thu hẹp thiên lệch khẳng định/phủ định và nâng `group_consistency` (tất cả 4 phép thử trong cùng nhóm cảnh đều đúng) vượt mức `58,61%` ban đầu hay không.
- **Trên Spatial (Synthetic & Natural):** Kiểm tra xem gộp `T0 + T1 + T2` (và gộp cả hai trật tự mệnh đề thuận/nghịch `o0 + o1`) có giúp mô hình vượt qua ngưỡng ngẫu nhiên hay vẫn bị khóa bởi position-invariant prior.
- **Trên CIFAR-100 Category Prompts:** Định lượng biên độ nhạy trên 8 điều kiện prompt của `full_depth_category_v1` và so sánh ensemble giữa các prompt đúng ngữ nghĩa (`baseline + category + neutral`).
