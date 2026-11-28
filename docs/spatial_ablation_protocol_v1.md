# Giao thức T2: Spatial ablation và oracle bbox baseline (v1)

Trạng thái: **frozen** (chốt trước mọi suy luận GPU mới của T2). Config máy đọc: [`configs/spatial_ablation_v1.json`](../configs/spatial_ablation_v1.json). Thuộc WS1 phần 2, Chặng 1 trong [`1124.md`](../1124.md).

## 1. Câu hỏi

Phương pháp chia nửa ảnh (native half) đạt 93,37% so với 51,53% khi đưa cả ảnh (holistic T2/o0) trên 98 nguồn Set B. T1 đã cho thấy CLIP nhận ra hai vật 86,48% nhưng quan hệ trên các ảnh nhận đúng vẫn chỉ 51,01%. T2 tách nguồn gốc khoảng cách ~42 điểm %:

1. **Mất vật do center-crop?** Bước tiền xử lý chuẩn cắt giữa ảnh có thể cắt mất một vật. Nếu đưa cả ảnh nhưng đệm thành hình vuông (không cắt) mà độ chính xác holistic tăng mạnh, center-crop là một phần nguyên nhân.
2. **Trần hình học của quy tắc nửa ảnh:** dùng bbox COCO thật thay cho điểm CLIP để biết quy tắc gán nửa ảnh tối đa đạt bao nhiêu, và bao nhiêu ảnh có hai vật cùng nằm trong một nửa.
3. **Lợi ích có phụ thuộc đúng việc gán vùng?** Hoán vị vùng trái/phải (và trên/dưới) trên chính các điểm vùng đã lưu.

## 2. Dữ liệu (không thêm, không bớt mẫu)

| Tập | Vai trò | Nguồn / ảnh | Số lựa chọn | Lọc |
|---|---|---|---:|---|
| Set B (`confirmation`) | **Chính** | 98 nguồn / 196 ảnh (gốc + lật ngang) | 2 (trái/phải) | `accepted == True` trong `results/human_review_followup_v1/source_membership.csv` (R1; trùng đúng tập đồng thuận R1 ∩ R2) |
| Set A (`exploratory`) | Kiểm tra chéo, mô tả | 200 nguồn / 400 ảnh | 4 (trái/phải/trên/dưới) | Không lọc; chưa có review người |

Set C không dùng trong T2 (đã báo cáo riêng, không gộp).

## 3. Các điều kiện

| # | Điều kiện | Nguồn điểm | Suy luận mới? |
|---|---|---|---|
| 1 | Holistic center-crop T2/o0 | Điểm đã đóng băng của `full_natural_confirmation_v1` (B) và `full_depth_natural_v1` (A) | Không |
| 2 | **Holistic full-image padded** T2/o0 | Lật (nếu là ảnh lật) rồi đặt toàn ảnh vào giữa canvas vuông cạnh `max(w, h)`, nền RGB `[123, 117, 104]` (giống `pad_half`), rồi `adapter.preprocess` chuẩn; cùng caption và thứ tự caption với điều kiện 1 | **Có (GPU)** |
| 3a | Native half | `scores.npz` (`relational`) của `full_natural_regions_*_review_v2` | Không |
| 3b | Padded half | `scores.npz` (`padded_relational`) của cùng run | Không |
| 4a | Oracle bbox centroid | Bbox COCO của A, B chuyển sang canvas 224 của baseline (`canvas_box`), lật x với ảnh lật. Điểm `[-dx, dx, -dy, dy]` với `dx = cx(A) - cx(B)`, `dy = cy(A) - cy(B)` | Không (CPU) |
| 4b | Oracle bbox ideal half-canvas | `_box_half_overlap` trên canvas 224 rồi `relational_scores_from_regions`; báo riêng nhóm **hai vật cùng nằm trong một nửa** (tỷ lệ phủ > 0,99) | Không (CPU) |
| 5 | Swapped-region sanity | Điểm vùng native đã lưu, hoán vị cột trái↔phải và trên↔dưới trước `relational_scores_from_regions` | Không |

Điều kiện 4 dùng nhãn hộp thật nên là **trần hình học**, không phải đối thủ công bằng với CLIP. Điều kiện 5 là kiểm tra số học: với 2 lựa chọn, hoán vị đảo dấu điểm nên kết quả kỳ vọng đúng bằng 100% trừ native (khi không có hòa); nó chỉ xác nhận pipeline gán vùng, không thêm bằng chứng độc lập.

## 4. Chỉ số và thống kê

- Top-1 theo ảnh (mẫu số ảnh), tỷ lệ cả ảnh gốc lẫn ảnh lật đều đúng (mẫu số nguồn), ghép cặp 2×2 trái/phải.
- Chênh lệch ghép cặp trên cùng ảnh so với điều kiện 1, CI 95% bootstrap theo nguồn (ảnh gốc và ảnh lật luôn đi cùng nhau), **2.000 lượt, seed 1402** (`cluster_ci`).
- **So sánh chính (định trước):** Set B, Δ = (điều kiện 2) − (điều kiện 1), top-1.
- Các so sánh còn lại là phụ hoặc mô tả; CI riêng từng phép đo, không hiệu chỉnh đa kiểm định.

## 5. Quy tắc diễn giải (chốt trước khi thấy kết quả)

Gọi Δ_pad là Δ top-1 của điều kiện 2 so với điều kiện 1 trên Set B:

- Cận trên CI của Δ_pad **< 10 điểm %**: center-crop **không** phải nguyên nhân chính của khoảng cách ~42 điểm % giữa native half và holistic.
- Cận dưới CI của Δ_pad **≥ 10 điểm %**: center-crop **góp phần đáng kể**; báo tỷ lệ khoảng cách được thu hẹp Δ_pad / Δ_native.
- Còn lại: **chưa rõ**.

Kết quả ngược kỳ vọng hay CI rộng đều được giữ nguyên. Không đổi caption, ngưỡng, cách đệm hay tập mẫu sau khi thấy điểm.

## 6. Chốt chặn môi trường (parity)

Trước khi tính điều kiện 2, run tái tạo điểm holistic center-crop T2/o0 cho mọi ảnh trong run từ chính các feature vừa tính, rồi so với điểm đã đóng băng: dừng run nếu `max_abs_diff > 1e-6`. Môi trường T2: conda `lab22_env` (Python 3.10, torch 2.5.1+cu121, open_clip 3.3.0), GPU RTX 3050 4GB Laptop, FP32, thuật toán xác định (`CUBLAS_WORKSPACE_CONFIG=:4096:8`). Kiểm tra môi trường trước khi chốt giao thức (chỉ tái tạo điểm cũ, không tính điều kiện 2): 16 trial của 8 nguồn Set B, `max_abs_diff = 2,384e-7`, argmax trùng 100%.

## 7. Thực thi và audit

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m vlm_probe.spatial_diagnostics.ablation_v1 run --smoke --run-name smoke_spatial_ablation_v1
python -m vlm_probe.spatial_diagnostics.ablation_v1 audit --run-name smoke_spatial_ablation_v1
python -m vlm_probe.spatial_diagnostics.ablation_v1 run --run-name full_spatial_ablation_v1
python -m vlm_probe.spatial_diagnostics.ablation_v1 audit --run-name full_spatial_ablation_v1
```

- `run` từ chối chạy nếu config không `frozen`, hoặc nếu giao thức/config chưa commit hay có thay đổi chưa commit; manifest ghi commit và thời điểm commit của giao thức, phải sớm hơn `started_at_utc`.
- `audit` chạy offline từ `scores.npz` và các artifact đã đóng băng (không cần ảnh hay mô hình): kiểm hash, kiểm parity đã lưu, tính lại mọi chỉ số của 5 điều kiện trên Set B và Set A, đối chiếu điểm native/padded với run gốc.
- Không ghi đè bất kỳ `results/full_*` lịch sử nào.

## 8. Giới hạn đã biết

Một checkpoint; Set A và Set B đã được mở ở các thí nghiệm trước nên T2 là phân tích tiếp nối, không phải holdout mới; Set A chưa có review người; oracle bbox dùng nhãn thật nên chỉ là trần tham chiếu.
