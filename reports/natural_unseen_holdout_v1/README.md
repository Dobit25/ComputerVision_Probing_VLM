# Báo cáo T3 — Kiểm chứng ngoài mẫu chưa mở (`natural_unseen_holdout_v1` — Set C)

## 1. Bối cảnh, nguồn dữ liệu và tính tách biệt tuyệt đối

Nhằm đáp ứng góp ý ngày 11/11/2026 về kiểm chứng độc lập trên một **tập ảnh tự nhiên mới hoàn toàn chưa từng mở kết quả** và có **xác minh nhãn bởi 2 người chấm mù độc lập**, nghiên cứu **Set C** (`natural_unseen_holdout_v1`) được thực thi theo đúng giao thức khóa trước [docs/natural_unseen_holdout_protocol_v1.md](../../docs/natural_unseen_holdout_protocol_v1.md) (`configs/natural_unseen_holdout_v1.json`).

- **Lý do sử dụng COCO `train2014`:** Sau Set A (400 ứng viên) và Set B (178 ứng viên), toàn bộ `val2014` (và `val2017` là tập con của `val2014`) còn **0** ảnh ngang hợp lệ theo bộ lọc hình học của `probing_depth_v1.json`. Set C tuyển ảnh từ `annotations/instances_train2014.json` (`annotations_trainval2014.zip`, CC BY 4.0). Do checkpoint đánh giá OpenAI **CLIP ViT-B/32** hoàn toàn đóng băng (zero-shot, không huấn luyện trên nhãn COCO), không có rò rỉ nhãn huấn luyện từ split này.
- **Loại trùng 4 lớp (Zero-Overlap):** Đã loại trừ toàn bộ **578** `source_image_id` đã sàng lọc ở Set A + Set B, kiểm tra không trùng `image_sha256` và sàng lọc perceptual `dHash` (ngưỡng Hamming $\le 6$ cả bản gốc lẫn lật ngang).
- **Báo cáo tách biệt:** Mọi kết quả Set C được báo cáo **độc lập hoàn toàn** với Set A và Set B, không gộp mẫu.

---

## 2. Chấm mù kép 2 người (`R3_C`, `R4_C`) và Phân xử trước suy luận

Từ 160 ứng viên sạch cân bằng (`80 left + 80 right`), hai người chấm độc lập (`R3_C` ở Slot RA, `R4_C` ở Slot RB) chấm mù qua `annotate_set_c.html` khi **chưa tồn tại bất kỳ điểm số CLIP nào**:

| Tiêu chí IAA (160 ứng viên) | Đồng thuận thô | Kỳ vọng ($p_e$) | Cohen's $\kappa$ | Phân bố |
|---|---:|---:|---:|---|
| **`overall_accept`** | **151 / 160 (94,38%)** | 77,56% | **0,7493** | `R3_C`: 143 yes, 17 no / `R4_C`: 136 yes, 24 no |
| `both_objects_visible` | 153 / 160 (95,63%) | 89,92% | 0,5659 | `R3_C`: 153 yes, 7 no / `R4_C`: 150 yes, 10 no |
| `class_names_correct` | 149 / 160 (93,13%) | 84,39% | 0,5596 | `R3_C`: 151 yes, 9 no / `R4_C`: 142 yes, 18 no |
| `relation_clear` | 154 / 160 (96,25%) | 93,91% | 0,3838 | `R3_C`: 153 yes, 7 no / `R4_C`: 157 yes, 3 no |
| **`perceived_relation`** | **158 / 160 (98,75%)** | 48,78% | **0,9756** | `78 left, 80 right, 2 unclear` |

- **Phân xử theo quy tắc đồng thuận tuyệt đối (`RA ∩ RB`):** Trong 160 ứng viên, có **135** nguồn được cả hai cùng chấp nhận (`66 left + 69 right`), **16** nguồn cả hai cùng loại, và **9** nguồn bất đồng. Người phân xử (`tung`) áp dụng quy tắc khóa trước loại cả 9 ca bất đồng ([adjudication_filled.csv](../../reviews/natural_unseen_holdout_v1/adjudication_filled.csv)), trích tất định theo `rank` đúng **100 nguồn cân bằng (`50 left + 50 right`, 200 ảnh gốc + lật)**.
- **Dấu thời gian khóa:** Đóng băng lúc **`2026-11-26T02:51:47.910442+00:00`** ([freeze.json](../../reviews/natural_unseen_holdout_v1/freeze.json)), chạy suy luận 1 lượt duy nhất sau đó (hoàn tất `2026-11-26T02:54:16.337983+00:00`).

---

## 3. Kết quả thực nghiệm 1 lượt duy nhất trên Set C (100 nguồn / 200 ảnh)

Bootstrap theo cụm nguồn (`source_image_id`, 2.000 lượt, seed `1402`).

| Phương pháp / Baseline | Top-1 2-way (%) | CI 95% Top-1 (%) | Cả 2 ảnh đúng (%) | Ghép cặp 2×2 (%) | Wilson CI 95% (2×2) | $\Delta$ Top-1 vs Holistic (pp) |
|---|---:|---|---:|---:|---|---:|
| **Holistic T2/o0** | **47,00** | [44,00; 49,50] | **1,00** [0,00; 3,00] | **46,00** (`46/100`) | [36,56; 55,74] ($p=0,8159$) | — |
| **Native half (primary)** | **89,00** | [83,00; 94,50] | **87,00** [80,00; 93,00] | **91,00** (`91/100`) | [83,77; 95,19] ($p=1,66\times 10^{-18}$) | **+42,00** [35,00; 48,50] |
| **Padded half (secondary)** | **81,00** | [73,50; 88,00] | **77,00** [68,00; 85,00] | **83,00** (`83/100`) | [74,45; 89,11] ($p=6,55\times 10^{-12}$) | **+34,00** [26,00; 41,50] |
| **Bbox Centroid (Oracle)** | **100,00** | — | — | — | — | — |
| **Bbox Ideal Half-Overlap (Oracle)** | **92,00** | — | — | — | — | (8,00% cả 2 vật cùng 1 nửa) |

### Phân tích theo quan hệ & Diễn giải
- **Holistic lặp lại đúng cơ chế sụp đổ trên ảnh mới chưa từng thấy:** Holistic đạt **47,00%** Top-1 (`left`: `12,00%`, `right`: `82,00%` — thiên lệch rất mạnh về `right`, khiến hầu hết cặp gốc/lật cùng đoán `right`), `both_images_correct` chỉ **1,00%**, và 2×2 assignment đạt **46,00%** (CI `[36,56; 55,74]%`, không có bằng chứng vượt 50%).
- **Chia nửa ảnh giữ vững mức phục hồi trên dữ liệu ngoài mẫu:** Native half đạt **89,00%** Top-1 (cân bằng tuyệt đối `89,00%` cho cả `left` và `right`), **91,00%** 2×2 assignment (`p = 1.66e-18`), vượt Holistic **+42,00 điểm %** (`CI [35,00; 48,50]`), nhất quán sát sao với hiệu ứng trên Set B (`+41,84 pp`).
- **So sánh với Baseline hình học Bounding Box:** Trên chính crop 224×224 của Set C, **8,00%** nguồn có cả hai vật rơi vào cùng một nửa canvas, nên trần lý tưởng của quy tắc chia đôi (`Bbox Ideal Half-Overlap`) là **92,00%**. Native half CLIP đạt **89,00%** Top-1 (**91,00%** 2×2), tiệm cận trần hình học này, trong khi Padded half đạt **81,00%**. Điều này xác nhận rằng lợi ích đến từ sự kết hợp giữa **cấu trúc cắt vùng hình học bên ngoài + cung cấp tên 2 vật thể + khả năng nhận dạng từng vùng của CLIP** (chứ không phải cải thiện bộ mã hóa holistic của CLIP).

---

## 4. Giới hạn và Tái lập

- **Giới hạn:** Kết quả áp dụng cho một checkpoint `OpenAI CLIP ViT-B/32`, trên tập ảnh COCO `train2014` có hai vật rõ ràng theo bộ lọc ngang; hiệu quả của quy tắc chia đôi bị giới hạn bởi tỷ lệ 8% hai vật nằm cùng một nửa ảnh và yêu cầu biết trước tên 2 đối tượng cùng 4 lượt encode thêm.
- **Kiểm toán (`audit`):** Cả `smoke_natural_unseen_holdout_v1` (8 nguồn) và `full_natural_unseen_holdout_v1` (100 nguồn) đều đạt `verified: true`, `recomputed_from_scores: true`, `review_frozen_before_inference: true`, `zero_overlap_with_set_a_b: true`.
- **Lệnh kiểm toán offline (không cần GPU hay pixel):**

```powershell
python -m vlm_probe.spatial_diagnostics.unseen_holdout audit --run-name smoke_natural_unseen_holdout_v1
python -m vlm_probe.spatial_diagnostics.unseen_holdout audit --run-name full_natural_unseen_holdout_v1
```

