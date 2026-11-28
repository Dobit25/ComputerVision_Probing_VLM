# Báo cáo T1 — Đối chứng nhận dạng hai vật thể trên COCO (`natural_recognition_v1`)

## 1. Phạm vi, giao thức và trạng thái đóng băng

Nghiên cứu này thực hiện mục **T1** theo [1118.md](../../1118.md) và [giao thức đã đóng băng](../../docs/natural_recognition_protocol_v1.md) (`configs/natural_recognition_v1.json`, `"status": "frozen"`). Mục tiêu: kiểm tra trên **đúng cùng các crop COCO 224×224** mà mô hình holistic T2/o0 đã sử dụng trong `full_natural_confirmation_v1`, OpenAI **CLIP ViT-B/32** (FP32, QuickGELU, không fine-tuning) có phân biệt được caption chứa **đúng hai đối tượng** (`a photo showing {A} and {B}.`) với caption **thay sai một đối tượng bằng foil** hay không.

- **Tính hợp lệ trước suy luận (Pre-registration & Freeze):**
  - Toàn bộ 196 dòng review (98 nguồn × 2 vai trò `replace_a`, `replace_b`) đã được một reviewer người (`tuan_dat`) chấm mù trước CLIP scores qua 2 vòng và đóng băng tại [reviews/natural_recognition_v1/freeze.json](../../reviews/natural_recognition_v1/freeze.json) vào lúc **`2026-11-18T10:41:01.907282+00:00`**.
  - Suy luận GPU đầy đủ chạy sau đó vào **`2026-11-25T07:24:42.791067+00:00`** (hoàn tất `2026-11-25T07:25:05.094513+00:00`).
  - Độ phủ cohort: **98/98** nguồn được R1 chấp nhận đều có foil hợp lệ ở cả hai vai trò (`excluded_sources = []`), tạo thành **196 ảnh** (`original` + `horizontal_flip`) và **392 trial**.
- **Kiểm chứng parity với điểm holistic gốc:** Tái tạo lại 200 điểm quan hệ T2/o0 của `full_natural_confirmation_v1` từ cùng feature cache cho `max_abs_diff = 2.086e-7` ($\le$ ngưỡng cho phép `1e-6`), bảo đảm biểu diễn ảnh đầu vào đồng nhất với benchmark quan hệ trước đó.
- **Kiểm toán (`audit`):** Cả run `smoke_natural_recognition_v1` (32 trials) và `full_natural_recognition_v1` (392 trials) đều đạt `verified: true`, `recomputed_from_scores: true`, `review_frozen_before_inference: true`.

---

## 2. Kết quả chính (Primary Metric & Phân loại theo ngưỡng định trước)

Bootstrap theo cụm nguồn (`source_image_id`, 2.000 lượt, seed `1402`). Mức ngẫu nhiên (2 lựa chọn) là **50,00%**.

| Chỉ số | Giá trị |
|---|---|
| **Recognition Accuracy (Primary, 392 trials)** | **339 / 392 = 86,48%** (`86,4796%`) |
| **CI 95% (source-clustered bootstrap)** | **[81,38%; 91,07%]** |
| Exact ties / Near ties ($\|\Delta s\| \le 10^{-6}$) | 0 / 0 |
| Mean score gap ($s_{\text{true}} - s_{\text{foil}}$) | +0,028125 |
| **Phân loại theo ngưỡng đã chốt (`team_decisions`)** | **`clearly_good`** (cận dưới CI 95% = `81,38% >= 80,00%`) |

Theo quy tắc diễn giải đã khóa trước khi chạy inference (`clearly_good`: cận dưới CI 95% $\ge 80\%$; `weak`: cận trên $< 70\%$; còn lại: `unclear`), CLIP ViT-B/32 đạt mức **`clearly_good`** trong phân biệt caption chứa đúng hai vật thể so với foil thay một vật thể trên đúng các crop COCO này.

---

## 3. Phân rã chi tiết theo vai trò, biến thể ảnh, tầng foil và tính nhất quán

| Nhóm phân tích | Số đúng / Tổng | Accuracy (%) | CI 95% (%) |
|---|---:|---:|---:|
| **Theo vai trò thay thế (`by_role`)** | | | |
| `replace_a` | 163 / 196 | 83,16 | [76,02; 89,80] |
| `replace_b` | 176 / 196 | 89,80 | [83,67; 95,41] |
| **Theo biến thể ảnh (`by_variant`)** | | | |
| `original` | 170 / 196 | 86,73 | [81,63; 91,33] |
| `horizontal_flip` | 169 / 196 | 86,22 | [80,61; 91,33] |
| **Theo tầng foil (`by_foil_tier`)** | | | |
| Tầng 1 (cùng supercategory COCO) | 295 / 346 | 85,26 | [79,83; 90,30] |
| Tầng 2 (khác supercategory COCO) | 44 / 46 | 95,65 | [86,36; 100,00] |
| **Tính nhất quán mức ảnh & mức nguồn** | | | |
| Đúng cả 2 vai trò trên 1 ảnh (`both_roles_correct_per_image`) | 149 / 196 | 76,02 | [67,86; 83,67] |
| Đúng cả 4 trial trên 1 nguồn (`all_four_correct_per_source`) | 72 / 98 | 73,47 | [65,31; 81,63] |

Nhận xét từ bảng phân rã:
- Độ chính xác gần như bất biến giữa ảnh gốc (`86,73%`) và ảnh lật ngang (`86,22%`), hoàn toàn trái ngược với sự sụp đổ khi suy luận quan hệ trái/phải.
- Foil tầng 1 (cùng supercategory, khó hơn) đạt `85,26%` trên 346 trials; foil tầng 2 (chỉ dùng khi tầng 1 rỗng, chủ yếu khi thay `person`) đạt `95,65%` trên 46 trials.

---

## 4. Bảng chéo mô tả: Nhận dạng hai vật thể vs Quan hệ không gian Holistic

Bảng dưới đây đối chiếu **trên đúng 196 ảnh** (98 nguồn × 2 biến thể) giữa việc nhận dạng đúng đồng thời cả hai vai trò (`both_roles_correct`) với việc holistic T2/o0 dự đoán đúng quan hệ trái/phải (2-way) từ `after_98_holistic_predictions.csv`. Lưu ý: phân tích theo nhóm con chọn theo hành vi mô hình chỉ mang tính **mô tả (descriptive only)**, không thay thế metric chính trên toàn tập.

| Nhóm hành vi trên 196 ảnh | Quan hệ Holistic ĐÚNG | Quan hệ Holistic SAI | Tổng ảnh | Tỉ lệ quan hệ đúng (%) |
|---|---:|---:|---:|---:|
| **Nhận dạng đúng cả 2 vai trò (`both_roles_correct = True`)** | 76 | 73 | 149 | **51,01%** (`76/149`) |
| **Nhận dạng chưa đúng cả 2 vai trò (`both_roles_correct = False`)** | 25 | 22 | 47 | 53,19% (`25/47`) |
| **Toàn cohort (196 ảnh)** | **101** | **95** | **196** | **51,53%** [49,49; 53,57] |

**Ý nghĩa khoa học trực tiếp:** Ngay cả trên **149 ảnh mà CLIP nhận dạng đúng cả hai vật thể đối chiếu foil ở cả hai vai trò**, độ chính xác suy luận quan hệ không gian của mô hình holistic vẫn chỉ đạt **51,01%** (gần như giống hệt toàn cohort 51,53%). Điều này cung cấp bằng chứng định lượng rằng sự thất bại về quan hệ không gian của holistic CLIP trên Set B **không bắt nguồn chủ yếu từ việc không nhận ra hai đối tượng trong crop**, mà xuất phát từ việc không kết hợp đúng cấu trúc quan hệ không gian giữa hai đối tượng đó dưới phép so khớp toàn cục.

---

## 5. Giới hạn kết luận cần giữ

1. **Follow-up hậu nghiệm trên tập đã mở:** 98 nguồn Set B đã được mở kết quả quan hệ trước khi thiết kế T1; kết quả T1 là phân tích chẩn đoán hậu nghiệm, không phải xác nhận ngoài mẫu độc lập.
2. **Một reviewer đơn lẻ:** 196 foils được xác nhận bởi một reviewer (`tuan_dat`), chưa có inter-annotator agreement (IAA) cho riêng tập foil T1.
3. **Độ khó thiết kế control & tầng foil:** Control dùng khung trung tính `a photo showing {a} and {b}.` với 46/392 trials thuộc foil tầng 2 khác supercategory; đồng thời độ chính xác 86,48% không chứng minh CLIP xử lý được mọi ca che khuất/nhiễu ngoài bộ dữ liệu này.
4. **Phạm vi một checkpoint:** Kết quả chỉ áp dụng cho OpenAI `CLIP ViT-B/32`, không suy ra cơ chế nội tại của neuron hay khái quát hóa cho toàn bộ các VLM khác.
5. **Tác động tới bước tiếp theo (T2):** Kết quả T1 mở đường cho **T2** (so sánh với Holistic full-image padded, Oracle COCO Bbox + Geometric Rules baseline và Swapped-region sanity check) để tách phần đóng góp từ thông tin hình học/crop so với biểu diễn CLIP.

---

## 6. Artifact & Lệnh tái lập

- **Smoke run:** `results/smoke_natural_recognition_v1/` (`results/smoke_natural_recognition_v1_audit.json`)
- **Full run:** `results/full_natural_recognition_v1/` (`metrics.json`, `predictions.csv`, `image_records.csv`, `scores.npz`, `parity.json`, `manifest.json`)
- **Biên bản audit:** `results/full_natural_recognition_v1_audit.json`
- **Kiểm toán offline trên bất kỳ clone sạch nào (không cần GPU hay pixel COCO):**

```powershell
python -m vlm_probe.spatial_diagnostics.natural_recognition audit --run-name smoke_natural_recognition_v1
python -m vlm_probe.spatial_diagnostics.natural_recognition audit --run-name full_natural_recognition_v1
```

