# T2: Spatial ablation và oracle bbox baseline (28/11/2026)

Giao thức [spatial_ablation_protocol_v1.md](../../docs/spatial_ablation_protocol_v1.md) và [config](../../configs/spatial_ablation_v1.json) được commit trước khi chạy (`2708e47`, 07:51 UTC). Run: [`results/full_spatial_ablation_v1`](../../results/full_spatial_ablation_v1/), audit offline `verified: true` ([audit](../../results/full_spatial_ablation_v1_audit.json)). Một checkpoint CLIP ViT-B/32 gốc, không fine-tuning.

Môi trường: conda `lab22_env` (Python 3.10, torch 2.5.1+cu121, open_clip 3.3.0), RTX 3050 4GB Laptop, FP32, thuật toán xác định. Chốt chặn parity: điểm holistic center-crop tái tạo trên máy này lệch tối đa **2,38e-7** (Set B) và **3,28e-7** (Set A) so với điểm đã đóng băng, dưới ngưỡng 1e-6.

## So sánh chính (định trước)

Set B, 98 nguồn đã review (R1 ∩ R2), 196 ảnh, 2 lựa chọn: đưa **toàn bộ ảnh đệm vuông** thay cho center-crop cho Δ top-1 = **0,00 điểm %**, CI 95% **[-3,06; 3,06]**. Cận trên CI < 10 điểm % nên theo quy tắc §5 của giao thức: **center-crop không phải nguyên nhân chính** của khoảng cách giữa native half và holistic. Phần khoảng cách được thu hẹp: 0%.

## Set B (chính): 98 nguồn / 196 ảnh, 2 lựa chọn

Top-1 có mẫu số ảnh; "Cả hai ảnh đúng" và 2×2 có mẫu số nguồn. CI 95% bootstrap theo nguồn, 2.000 lượt, seed 1402.

| Điều kiện | Top-1 (%) [CI] | Δ so center-crop [CI] | Cả hai ảnh đúng (%) | 2×2 (%) |
|---|---|---|---:|---:|
| 1. Holistic center-crop (đóng băng) | 51,53 [49,49; 53,57] | - | 4,08 | 53,06 |
| 2. Holistic full-image padded (mới) | 51,53 [49,49; 54,08] | 0,00 [-3,06; 3,06] | 4,08 | 47,96 |
| 3a. Native half | 93,37 [88,27; 97,45] | 41,84 [36,73; 46,43] | 91,84 | 92,86 |
| 3b. Padded half | 92,86 [88,27; 96,94] | 41,33 [36,22; 45,92] | 89,80 | 90,82 |
| 4a. Oracle bbox centroid | 100,00 [100,00; 100,00] | 48,47 [46,43; 50,51] | 100,00 | 100,00 |
| 4b. Oracle bbox ideal half | 99,49 [98,47; 100,00] | 47,96 [45,41; 50,00] | 98,98 | 98,98 |
| 5. Swapped-region | 6,63 [2,55; 11,73] | -44,90 [-50,00; -39,29] | 5,10 | 7,14 |

Hai vật cùng nằm trong một nửa ảnh (theo trục của quan hệ đúng, tỷ lệ phủ > 0,99): **2/196 ảnh (1,02%)**.

## Set A (kiểm tra chéo, mô tả): 200 nguồn / 400 ảnh, 4 lựa chọn

Set A chưa có review người; 2×2 chỉ tính trên 100 nguồn trái/phải.

| Điều kiện | Top-1 (%) [CI] | Δ so center-crop [CI] | Cả hai ảnh đúng (%) | 2×2 (%) |
|---|---|---|---:|---:|
| 1. Holistic center-crop (đóng băng) | 29,00 [24,50; 33,50] | - | 10,50 | 54,00 |
| 2. Holistic full-image padded (mới) | 28,25 [23,75; 33,00] | -0,75 [-4,00; 2,25] | 9,50 | 46,00 |
| 3a. Native half | 72,25 [66,25; 78,25] | 43,25 [35,50; 50,75] | 69,50 | 93,00 |
| 3b. Padded half | 68,75 [62,75; 74,25] | 39,75 [32,25; 47,25] | 62,50 | 90,00 |
| 4a. Oracle bbox centroid | 99,50 [98,50; 100,00] | 70,50 [66,00; 75,25] | 99,50 | 100,00 |
| 4b. Oracle bbox ideal half | 92,00 [87,75; 95,50] | 63,00 [56,75; 69,00] | 91,50 | 94,00 |
| 5. Swapped-region | 1,50 [0,25; 3,25] | -27,50 [-32,25; -22,50] | 1,00 | 7,00 |

Hai vật cùng nằm trong một nửa ảnh: **20/400 ảnh (5,00%)**; trên nhóm này native half chỉ đúng 30,00%, so với 74,47% ở các ảnh còn lại (mô tả, mẫu nhỏ).

## Diễn giải

1. **Không cắt ảnh cũng không giúp.** Holistic giữ nguyên 51,53% trên Set B và 28,25% trên Set A khi toàn bộ ảnh được giữ lại. Cùng với T1 (CLIP nhận ra hai vật 86,48%, nhưng quan hệ trên các ảnh nhận đúng vẫn 51,01%), lỗi không gian của holistic không đến từ việc vật bị cắt mất hay không được nhận ra, mà từ cách điểm toàn ảnh so khớp với caption quan hệ.
2. **Native half gần trần hình học trên Set B.** Quy tắc nửa ảnh với bbox thật đạt 99,49%; native half dùng điểm CLIP đạt 93,37%. Rất ít ảnh có hai vật cùng một nửa trên Set B (1%), nên giới hạn hình học của quy tắc gần như không ảnh hưởng ở đây. Trên Set A (4 lựa chọn) trần ideal half thấp hơn (92,00%) và nhóm cùng một nửa (5%) là nơi native half sai nhiều.
3. **Lợi ích phụ thuộc đúng việc gán vùng.** Hoán vị vùng cho 6,63% trên Set B, đúng bằng 100% trừ 93,37% như kỳ vọng số học của phép đảo dấu. Đây là kiểm tra pipeline, không phải bằng chứng độc lập.
4. 2×2 của holistic đều quanh mức đoán ngẫu nhiên (53,06% center-crop, 47,96% padded trên Set B); không suy ra tương đương ngẫu nhiên từ kết quả này.

## Giới hạn

Set A và Set B đã được mở trong các thí nghiệm trước nên T2 là phân tích tiếp nối, không phải holdout mới (kiểm chứng trên tập mới là Set C, báo cáo riêng). Set A chưa có review người. Oracle bbox dùng nhãn hộp thật nên chỉ là trần tham chiếu, không phải đối thủ công bằng. Môi trường GPU khác các run gốc (RTX 4050) nhưng đã qua chốt chặn parity ≤ 1e-6. CI riêng từng phép đo, không hiệu chỉnh đa kiểm định.

## Tái lập

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m vlm_probe.spatial_diagnostics.ablation_v1 audit --run-name full_spatial_ablation_v1
```

Audit không cần ảnh hay mô hình: kiểm hash artifact và đầu vào, kiểm giao thức được commit trước `started_at_utc`, kiểm parity đã lưu, tính lại toàn bộ 7 điều kiện của cả hai tập và đối chiếu từng dự đoán. Chạy lại suy luận: lệnh `run` trong giao thức §7 (cần GPU CUDA và tự tải ảnh COCO có kiểm SHA-256).
