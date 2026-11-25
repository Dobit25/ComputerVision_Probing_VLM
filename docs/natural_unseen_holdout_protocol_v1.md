# Protocol T3 — Tập kiểm chứng ngoài mẫu chưa mở (`natural_unseen_holdout_v1` — Set C)

**Trạng thái:** Đã khóa giao thức trước khi gán nhãn và suy luận (`protocol_frozen_pending_review`, ngày 25/11/2026).  
**Phạm vi:** Một checkpoint OpenAI **CLIP ViT-B/32** (FP32, QuickGELU, không fine-tuning), báo cáo **riêng biệt hoàn toàn** với Set A (200 nguồn thăm dò) và Set B (98/100 nguồn xác nhận trước đó) — tuyệt đối không gộp chung hai tập thành một mẫu xác nhận.

---

## 1. Mục tiêu và lý do chọn nguồn COCO `train2014`

Góp ý ngày 11/11/2026 của giảng viên yêu cầu xác minh lại hiệu ứng của phương pháp chia vùng (`region-based inference`) trên **một tập ảnh tự nhiên mới chưa từng tham gia quá trình phát triển hay chọn prompt**, kèm **xác minh nhãn độc lập bởi người**.

- **Kiểm tra nguồn `val2014` / `val2017` hiện có:** Sau khi Set A (400 ứng viên trong `results/depth_natural_selection_v1/candidates.json`) và Set B (178 ứng viên trong `results/natural_confirmation_selection_v1/new_candidates.json`) đã sàng lọc hết toàn bộ 578 ảnh thỏa bộ lọc hình học ngang của `probing_depth_v1.json` trên `val2014` (và `val2017` là tập con của `val2014`), **số ảnh ngang hợp lệ còn lại trong `val2014` đúng bằng 0**.
- **Nguồn được chọn cho Set C:** `annotations/instances_train2014.json` từ bộ dữ liệu chính thức COCO 2014 (`annotations_trainval2014.zip`, giấy phép CC BY 4.0) cùng split ảnh `train2014`. Vì CLIP ViT-B/32 là checkpoint đóng băng (zero-shot, không hề huấn luyện trên nhãn COCO `train2014` hay `val2014`), việc dùng ảnh từ `train2014` không tạo rò rỉ train/test với mô hình đánh giá; giới hạn chung về khả năng trùng lặp ẩn với tập tiền huấn luyện WIT của CLIP vẫn được công khai như với `val2014`.

---

## 2. Quy tắc loại trùng tuyệt đối (Zero-Overlap & Near-Duplicate Screen)

Mọi ứng viên Set C phải thỏa đồng thời **4 điều kiện độc lập**:
1. **Disjoint `source_image_id`:** Không nằm trong tập **578** `source_image_id` đã từng xuất hiện ở Set A (`400`) hoặc Set B (`178`).
2. **Disjoint `image_sha256`:** SHA-256 file ảnh gốc khác toàn bộ 578 ảnh đã xem trước đây.
3. **Near-duplicate perceptual `dHash`:** Khoảng cách Hamming `dHash` (64-bit trên crop vuông trung tâm, kiểm tra cả bản gốc lẫn bản lật ngang) phải **`> 6`** so với toàn bộ 578 ảnh Set A/B **và** so với mọi ứng viên đã nhận trước đó trong Set C.
4. **Bộ lọc hình học giống hệt Set A & Set B:**
   - `min_box_area_fraction = 0.015`, `min_box_side_fraction = 0.06`
   - `min_separation_fraction = 0.02`, `min_orthogonal_overlap_fraction = 0.50`
   - `crop_border_fraction = 0.01`, `require_unique_category_instances = true`, `exclude_crowd = true`
   - `allowed_relations = [0, 1]` (`left` / `right`), xếp hạng tất định bằng `sha256("1124/{image_id}/{ann_a}/{ann_b}")`.

---

## 3. Quy trình chấm mù kép bởi 2 người (`RA` & `RB`) trước Inference

1. **Pool ứng viên chấm mù:** Rút **160 ứng viên sạch** cân bằng (80 `left`, 80 `right` theo thứ tự xếp hạng tất định sau lọc `dHash`), xáo trộn hiển thị ngẫu nhiên cố định với seed `"1124/set_c_blind_order"` thành các mã mù `UNSEEN_C_001` .. `UNSEEN_C_160`.
2. **Hai người chấm độc lập (`RA`, `RB`):**
   - Hai thành viên **khác nhau** (`annotator_id_A != annotator_id_B`), chấm mù hoàn toàn trên cùng bộ ảnh nguồn có box + crop 224×224 thật mà CLIP nhìn thấy.
   - **Không có bất kỳ điểm số CLIP hay inference nào được chạy trước khi `freeze.json` tồn tại.** Code từ chối chạy `run` nếu thiếu `freeze.json` hoặc `frozen_at_utc >= started_at_utc`.
3. **Tính IAA & Phân xử trước suy luận:**
   - Đo tỷ lệ đồng thuận thô (%) và **Cohen's $\kappa$** giữa `RA` và `RB` trước khi phân xử.
   - Mọi dòng bất đồng giữa `RA` và `RB` (nhận vs loại) phải đi qua bước phân xử công khai (`adjudicate`), lưu nguyên lịch sử phiếu gốc `RA`/`RB` và quyết định phân xử.
4. **Quy tắc chốt cohort cuối (`freeze`):**
   - Duyệt các nguồn được chấp nhận theo thứ tự xếp hạng `rank` gốc (ưu tiên lấy cân bằng tối đa **50 `left` + 50 `right` = 100 nguồn**; nếu một chiều thiếu nhẹ và tổng đạt $\ge$ ngưỡng tối thiểu, bù từ chiều còn lại theo `rank` đến tối đa 100 nguồn).
   - **Ngưỡng khả thi chính:** tối thiểu **80 nguồn** hợp lệ.
   - **Phương án B (Gate Ngày 3 — 18:00 ngày 27/11/2026):** Nếu tiến độ chấm người không đạt 80 nguồn trước cổng khóa dữ liệu, được phép công bố amendment hạ ngưỡng tối thiểu xuống **60 nguồn** *trước khi chạy inference*; nếu dưới 60 nguồn thì **dừng thí nghiệm Set C** mà không suy luận.

---

## 4. Giao thức 1 lượt suy luận duy nhất, Metric & Kiểm toán

Sau khi `reviews/natural_unseen_holdout_v1/freeze.json` được khóa, chạy **đúng 1 lượt duy nhất** (`smoke` kiểm tra pipeline → `full` 1 lượt) trên các biến thể `original` + `horizontal_flip` ($2N$ ảnh):

1. **Holistic T2/o0 (2-way):** Đúng template `"an image of {a} {relation} {b}."` như Set B.
2. **Native Half-Canvas (Primary):** Cắt 4 nửa trên crop 224×224 (`crop_half_canvases`) với template `"a photo of {object}."` và công thức điểm đối xứng `relational_scores_from_regions`.
3. **Padded Half-Canvas (Secondary):** Pad từng nửa về 224×224 bằng RGB `[123, 117, 104]`.
4. **Oracle COCO Bbox + Geometric Rules Baseline (CPU):**
   - *Centroid rule:* so sánh $x_{\text{center}}(A)$ và $x_{\text{center}}(B)$ trên crop 224×224.
   - *Ideal half-canvas overlap rule:* tính `_box_half_overlap` trực tiếp từ bbox trên crop 224×224 và thống kê tỷ lệ nguồn có cả 2 vật nằm trong cùng một nửa ảnh.

**Metric công bố riêng biệt cho Set C:**
- Top-1 2-way accuracy + paired source-clustered bootstrap CI 95% (2.000 draws, seed `1402`).
- `both_images_correct_percent` (cả gốc và lật đều đúng riêng rẽ).
- `2x2_assignment_accuracy` + Wilson 95% CI + kiểm định nhị thức một phía so ngưỡng 50%.
- Paired $\Delta$ (`Native Half - Holistic T2/o0`) kèm source-clustered bootstrap CI 95%.
- Kiểm toán `audit` tái tính toàn bộ từ `scores.npz` mà không cần ảnh hay model.

