# Giao Thức Chẩn Đoán Cơ Chế & Khắc Phục Zero-Shot Cho Quan Hệ Không Gian (P2.1–P2.3)

- **Phiên bản giao thức:** `spatial_diagnostics_v1`
- **Model đối tượng:** `OpenAI CLIP ViT-B/32` (`QuickGELU`, `FP32`)
- **Mục tiêu:** Trả lời câu hỏi nghiên cứu **`RQ2`** trong [`docs/p0_requirements_and_rqs.md`](p0_requirements_and_rqs.md):
  > *Thất bại quan hệ không gian của CLIP bắt nguồn từ cơ chế nào (text-side directional collapse, image-side flip invariance, hay prior áp đảo), và liệu một phương pháp đánh giá có nhận thức vùng (crop/half-canvas zero-shot scoring, không huấn luyện) có khắc phục được lỗi này trên ảnh tổng hợp có kiểm soát nhưng gặp giới hạn gì trên ảnh tự nhiên?*

---

## 1. P2.1 — Phân rã toán học Ma trận Điểm 2×2: Prior Bias ($B$) vs. Spatial Signal ($\Delta_{\text{spatial}}$)

### 1.1. Động cơ lý thuyết
Trong các thí nghiệm không gian (`results/full_depth_spatial_v1`, `results/full_depth_natural_v1`, `results/full_natural_confirmation_v1`), ta quan sát thấy một hiện tượng tưởng chừng nghịch lý:
- Đối chứng nhận diện đối tượng đạt **`97,50%`** (`full_probing_clip_b32_v1`) và **`96,94%`** (`full_depth_spatial_v1`).
- Độ chính xác ghép cặp 2×2 (`assignment_accuracy`, khi khử bias cộng tính) đạt xấp xỉ **`50%–56%`** (`53,00%` trên tập xác nhận 100 nguồn).
- Thế nhưng tỷ lệ **cả hai ảnh trong cặp đối xứng cùng dự đoán đúng độc lập** (`both_images_correct`) lại sụp đổ xuống **`0,56%`** (synthetic `T0` ngang, 4-way) / **`1,11%`** (2-way) và **`4,00%`** (natural confirmation `T2` ngang)!

Tại sao `both_images_correct` lại thấp hơn rất nhiều so với $0.50^2 = 25.0\%$ của phép đoán độc lập ngẫu nhiên?

### 1.2. Định nghĩa Prior-to-Signal Decomposition
Xét một cặp ảnh đối xứng $(I_0, I_1)$ (ví dụ $I_0$: $A$ bên trái $B$; $I_1$: $A$ bên phải $B$, hoặc cặp ảnh gốc – ảnh lật ngang) và cặp câu mô tả đối xứng $(c_0, c_1)$.
Gọi điểm tương đồng cosine của CLIP là $s(I_i, c_j)$. Định nghĩa hiệu điểm trên từng ảnh:

$$
d(I_0) = s(I_0, c_0) - s(I_0, c_1), \qquad d(I_1) = s(I_1, c_0) - s(I_1, c_1)
$$

Khi đó:
- Ảnh $I_0$ dự đoán đúng ($c_0$) khi và chỉ khi $d(I_0) > 0$.
- Ảnh $I_1$ dự đoán đúng ($c_1$) khi và chỉ khi $d(I_1) < 0$.

Ta phân rã trực giao $(d(I_0), d(I_1))$ thành hai thành phần:
1. **Position-Invariant Prior Bias ($B$):**

$$
B = \frac{1}{2}\big(d(I_0) + d(I_1)\big)
$$

$B$ đo thiên kiến cố định của mô hình nghiêng về câu $c_0$ (nếu $B > 0$) hoặc câu $c_1$ (nếu $B < 0$) **bất chấp hai vật thể đã đổi chỗ cho nhau**.

2. **Half Assignment Gap / True Spatial Signal ($\Delta_{\text{spatial}}$):**

$$
\Delta_{\text{spatial}} = \frac{1}{2}\big(d(I_0) - d(I_1)\big) = \frac{1}{2}\,\text{AssignmentGap}(S)
$$

$\Delta_{\text{spatial}}$ đo độ nhạy thực sự của mô hình trước sự đảo chiều vị trí không gian giữa $I_0$ và $I_1$.

### 1.3. Điều kiện cần và đủ để phân loại đúng cả hai ảnh
Thay $d(I_0) = B + \Delta_{\text{spatial}}$ và $d(I_1) = B - \Delta_{\text{spatial}}$, điều kiện để cả hai ảnh cùng đúng ($d(I_0) > 0$ và $d(I_1) < 0$) là:

$$
\Delta_{\text{spatial}} > \lvert B \rvert \quad \Longleftrightarrow \quad \Delta_{\text{spatial}} > 0 \;\;\text{and}\;\; \text{PSR} = \frac{\lvert B \rvert}{\lvert \Delta_{\text{spatial}} \rvert} < 1
$$

trong đó **Prior-to-Signal Ratio ($\text{PSR}$)** được định nghĩa ở cấp từng cặp (hoặc tỷ số trung bình $\mathbb{E}[\lvert B \rvert] / \mathbb{E}[\lvert \Delta_{\text{spatial}} \rvert]$).
- Nếu $\Delta_{\text{spatial}} > 0$ nhưng $\lvert B \rvert \gg \Delta_{\text{spatial}}$ ($\text{PSR} \gg 1$), phép ghép cặp 2×2 (`assignment_gap > 0`) **vẫn tính là đúng** (vì $B$ bị triệt tiêu trong $d(I_0) - d(I_1)$), nhưng cả hai ảnh $I_0$ và $I_1$ đều bị kéo về cùng một nhãn do $B$ thống trị, làm `both_images_correct` sụp đổ về gần 0%!

---

## 2. P2.2 & P2.3 — Khắc Phục Zero-Shot Không Huấn Luyện: Region / Half-Canvas Spatial Scoring

### 2.1. Nguyên lý phương pháp (Lựa chọn B trong kế hoạch)
Vì CLIP ViT-B/32 nhận diện sự hiện diện của từng đối tượng đơn lẻ rất tốt (`control_accuracy = 96,94%–97,50%`) nhưng thất bại khi ghép chung hai đối tượng và giới từ không gian vào một biểu diễn toàn cục (`bag-of-words`), ta tách bài toán kiểm tra quan hệ không gian $R(A, B)$ thuộc `{"left", "right", "above", "below"}` thành hai bước **hoàn toàn zero-shot (không huấn luyện thêm tham số)**:

1. **Chia vùng ảnh (Directional Half-Canvas Cropping):**
   Từ ảnh đầu vào $I$ kích thước $W \times H$, trích xuất 4 nửa ảnh chia đôi tại tâm:
   - $I_{\text{left}}$: vùng trái $x \in [0, W/2]$
   - $I_{\text{right}}$: vùng phải $x \in [W/2, W]$
   - $I_{\text{top}}$: vùng trên $y \in [0, H/2]$
   - $I_{\text{bottom}}$: vùng dưới $y \in [H/2, H]$

2. **Định vị từng đối tượng bằng Single-Object Prompt:**
   Sử dụng chính bộ mã hóa của CLIP ViT-B/32 với hai câu mô tả đơn đối tượng:
   - $p(A)$: `"a {medium} of {a}."`
   - $p(B)$: `"a {medium} of {b}."`
   Tính điểm tương đồng cosine $s(I_{\text{region}}, p(X))$ cho $X \in (A, B)$ và $\text{region}$ thuộc `{"left", "right", "top", "bottom"}`.

3. **Điểm tương phản vị trí (Relational Contrast Scores):**
   Độ chênh lệch định vị của $A$ so với $B$ theo phương ngang và phương dọc:

$$
S_{\text{left-of}}(A, B) = \big(s(I_{\text{left}}, p(A)) - s(I_{\text{right}}, p(A))\big) - \big(s(I_{\text{left}}, p(B)) - s(I_{\text{right}}, p(B))\big)
$$

$$
S_{\text{right-of}}(A, B) = -S_{\text{left-of}}(A, B)
$$

$$
S_{\text{above}}(A, B) = \big(s(I_{\text{top}}, p(A)) - s(I_{\text{bottom}}, p(A))\big) - \big(s(I_{\text{top}}, p(B)) - s(I_{\text{bottom}}, p(B))\big)
$$

$$
S_{\text{below}}(A, B) = -S_{\text{above}}(A, B)
$$

Dự đoán 4 chiều (4-way prediction) là $\arg\max_{r \in [0,3]} S_r(A, B)$, và dự đoán 2 chiều ngang (2-way horizontal) là $\arg\max_{r \in [0,1]} S_r(A, B)$.

### 2.2. Tính chất bất biến đối xứng của Region-Aware Scoring
- Do $S_{\text{right-of}}(A, B) = -S_{\text{left-of}}(A, B)$, mọi thiên kiến cộng tính về tần suất tên lớp $p(A)$ hay $p(B)$ đều **tự động triệt tiêu** giữa hai nửa trái/phải.
- Tuy nhiên, phương pháp này phụ thuộc vào hai điều kiện vật lý:
  1. Hai đối tượng $A$ và $B$ phải nằm tách biệt qua đường phân chia tâm ảnh (hoàn toàn thỏa mãn trên ảnh hình học `probing_depth_v1`, nhưng có thể bị vi phạm một phần trên ảnh tự nhiên COCO khi cả hai vật thể cùng lệch về một bên khung hình).
  2. Mỗi nửa ảnh sau khi cắt vẫn giữ đủ ngữ cảnh hình ảnh để CLIP phân biệt $p(A)$ với $p(B)$.

### 2.3. Lệnh thực thi và Kiểm chứng
- **Phân tích cơ chế P2.1 (Offline từ artifact đã đóng băng, không cần GPU):**
  ```powershell
  python -m vlm_probe.spatial_diagnostics mechanism
  ```
- **Chạy thử nghiệm khắc phục Half-Canvas Zero-Shot (P2.2/P2.3, hỗ trợ `--smoke` hoặc full trên GPU):**
  ```powershell
  python -m vlm_probe.spatial_diagnostics mitigate --smoke --run-name spatial_mitigation_smoke
  ```
