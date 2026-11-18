# Giao Thức Đánh Giá Chéo Ngôn Ngữ Tiếng Việt & Ngữ Cảnh Văn Hóa Việt Nam (P4.1–P4.4)

- **Phiên bản giao thức:** `vietnamese_study_v1`
- **Model cố định:** `OpenAI CLIP ViT-B/32` (`QuickGELU`, `FP32`, byte-level BPE tokenizer `49.408` vocab, `context_length = 77`)
- **Mục tiêu:** Trả lời câu hỏi nghiên cứu **`RQ4`** trong [`docs/p0_requirements_and_rqs.md`](p0_requirements_and_rqs.md):
  > *Khi chuyển sang prompt tiếng Việt (có dấu, không dấu, và ghép song ngữ), hiệu năng của CLIP ViT-B/32 suy giảm ra sao, và mức suy giảm đó liên hệ thế nào với hiện tượng phân mảnh subword (BPE token fertility) cũng như độ lệch căn chỉnh không gian nhúng Anh–Việt?*

---

## 1. P4.1 — Track A: Cross-Lingual Prompting Trên Tập Ảnh Chuẩn & Probing

Để tách biệt hoàn toàn **lỗi ngôn ngữ (linguistic / tokenization bottleneck)** khỏi **sự thay đổi phân phối hình ảnh (visual domain shift)**, Track A giữ nguyên toàn bộ ảnh kiểm tra của CIFAR-100 (`10.000` ảnh) và bộ cảnh hình học tổng hợp (`counting`: `1.080` ảnh, `spatial`: `720` ảnh, `negation`: `720` ảnh), chỉ biến thiên ngôn ngữ của prompt qua **4 điều kiện tiền đăng ký**:

### 1.1. Bốn điều kiện ngôn ngữ cố định
1. **`en_baseline` (Tiếng Anh chuẩn đối chiếu):**
   - CIFAR-100: `a photo of a {en_label}.`
   - Counting: `a drawing of {en_number} {en_color} {en_shape_plural}.`
   - Spatial: `a drawing of a {en_ca} {en_sa} {en_relation} a {en_cb} {en_sb}.`
   - Negation: `a drawing with a {en_ca} {en_sa}.` vs. `a drawing without a {en_ca} {en_sa}.`
2. **`vi_diacritics` (Tiếng Việt tự nhiên có dấu đầy đủ):**
   - CIFAR-100: `một bức ảnh chụp {vi_label}.`
   - Counting: `một bức vẽ có {vi_number} {vi_shape} màu {vi_color}.`
   - Spatial: `một bức vẽ có một {vi_sa} màu {vi_ca} {vi_relation} một {vi_sb} màu {vi_cb}.`
   - Negation: `một bức vẽ có một {vi_sa} màu {vi_ca}.` vs. `một bức vẽ không có một {vi_sa} màu {vi_ca}.`
3. **`vi_no_diacritics` (Tiếng Việt lược bỏ dấu thanh và dấu phụ ASCII):**
   - Áp dụng hàm chuẩn hóa tất định `strip_vietnamese_diacritics` (chuyển `đ/Đ -> d/D` và tách bỏ toàn bộ dấu tổ hợp Unicode `Mn`).
   - CIFAR-100: `mot buc anh chup {vi_unaccented_label}.`
   - *Mục đích khoa học:* Kiểm tra giả thuyết liệu byte-level BPE của CLIP (vốn huấn luyện chủ yếu trên văn bản Latin/Anh) có bị phân mảnh nặng bởi các byte UTF-8 đa-byte của nguyên âm tiếng Việt có dấu hay không, và liệu việc bỏ dấu có làm giảm phân mảnh BPE hay làm mất nghĩa từ vựng.
4. **`vi_en_bilingual` (Ghép song ngữ / Code-Switching Việt–Anh):**
   - CIFAR-100: `một bức ảnh chụp {vi_label} ({en_label}).`
   - *Mục đích khoa học:* Tách biệt ảnh hưởng của **khung câu tiếng Việt** (`một bức ảnh chụp ...`) khỏi **từ vựng tên lớp tiếng Anh** (`{en_label}`). Nếu `vi_en_bilingual` khôi phục phần lớn độ chính xác về gần `en_baseline` trong khi `vi_diacritics` sụp đổ, điều đó chứng minh nút cổ chai chính nằm ở ánh xạ từ vựng tên lớp tiếng Việt trong bộ mã hóa văn bản, chứ không phải do các token khung câu tiếng Việt làm hỏng toàn bộ biểu diễn.

---

## 2. Chẩn Đoán Phân Mảnh BPE (BPE Tokenization Fertility) & Căn Chỉnh Nhúng

### 2.1. Chỉ số phân mảnh BPE (`vlm_probe/vietnamese_study/bpe_diagnostics.py`)
Trên toàn bộ 100 lớp CIFAR-100 và các câu probing, bộ phân tích BPE đo trực tiếp trên `SimpleTokenizer` của OpenAI CLIP:
- **`mean_subword_tokens_per_prompt`**: Số lượng BPE subword tokens trung bình (không tính `<start_of_text>` và `<end_of_text>`).
- **`mean_subword_tokens_per_label`**: Số lượng BPE subword tokens trung bình chỉ tính riêng phần tên lớp.
- **`token_fertility` (Số subword tokens trên mỗi từ đơn cách khoảng trắng)**:

$$
\text{Fertility} = \frac{N_{\text{BPE-tokens}}}{N_{\text{words}}}
$$

Trong tiếng Anh chuẩn, $\text{Fertility} \approx 1.0\text{--}1.2$. Với tiếng Việt có dấu trên byte-level BPE tiếng Anh, mỗi âm tiết có dấu thường bị bẻ thành 2–5 mảnh byte/subword.
- **`fragmentation_ratio_vs_english`**: Tỷ số số lượng subword tokens của prompt tiếng Việt chia cho prompt tiếng Anh tương ứng.
- **`truncation_count_at_77`**: Số câu vượt quá giới hạn `context_length = 77` của CLIP.

### 2.2. Độ tương đồng Cosine chéo ngôn ngữ (`Cross-Lingual Text Alignment`)
Với mỗi lớp $c = 0, \dots, 99$, tính trực tiếp tích vô hướng giữa các vector văn bản đã chuẩn hóa L2 của từng điều kiện tiếng Việt so với chuẩn tiếng Anh $\hat{t}_{\text{en}}(c)$:
- **Tiếng Việt có dấu (`vi_diacritics`):** $\cos\big(\hat{t}_{\text{en}}(c),\, \hat{t}_{\text{vi-dia}}(c)\big)$
- **Tiếng Việt không dấu (`vi_no_diacritics`):** $\cos\big(\hat{t}_{\text{en}}(c),\, \hat{t}_{\text{vi-nodia}}(c)\big)$
- **Song ngữ Việt–Anh (`vi_en_bilingual`):** $\cos\big(\hat{t}_{\text{en}}(c),\, \hat{t}_{\text{bi}}(c)\big)$

Đồng thời tính tương quan hạng Spearman / Pearson giữa độ dài phân mảnh BPE của lớp $c$ với mức sụt giảm độ chính xác $\Delta \text{Acc}_c = \text{Acc}_{\text{vi}}(c) - \text{Acc}_{\text{en}}(c)$.

---

## 3. P4.2–P4.4 — Quy Tắc Provenance Cho Bộ Thực Nghiệm Văn Hóa Việt Nam (Track B)

Khi mở rộng sang các khái niệm văn hóa Việt Nam đặc thù (ẩm thực: *phở bò, bánh mì, bánh chưng, gỏi cuốn*; trang phục/đồ vật: *áo dài, nón lá, đàn bầu*; kiến trúc/cảnh quan: *chùa Một Cột, vịnh Hạ Long, ruộng bậc thang*):
1. **Minh bạch nguồn gốc (`manifest.csv`):** Mọi ảnh phải ghi rõ `image_id`, `source_url`, `license` (`CC-BY`, `CC0`, hoặc ảnh tự chụp của nhóm), `sha256`, và người kiểm duyệt độc lập.
2. **Ba mức độ mô tả prompt:**
   - **Tên gốc tiếng Việt có dấu:** `một bức ảnh chụp nón lá.`
   - **Tên tiếng Việt không dấu / phiên âm quốc tế:** `a photo of a non la.`
   - **Mô tả giải thích bằng tiếng Anh (Descriptive English paraphrase):** `a photo of a traditional Vietnamese conical palm-leaf hat.`
   So sánh ba mức này cho phép định lượng xem CLIP nhận diện được vật thể văn hóa Việt Nam nhờ học được khái niệm thị giác (khi mô tả tiếng Anh đạt điểm cao) hay bị mù hoàn toàn cả về thị giác lẫn từ vựng địa phương.
