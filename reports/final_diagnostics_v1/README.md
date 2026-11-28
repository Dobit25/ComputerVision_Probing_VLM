# WS3: Phân tích sâu offline và bộ hình chuẩn IEEE (28/11/2026)

Thực hiện Chặng 2 (WS3) của [1124.md](../../1124.md). Mọi số liệu tính lại từ artifact đã commit, **không có suy luận mới**: [`results/final_diagnostics_v1/`](../../results/final_diagnostics_v1/) (`metrics.json`, `negation_deep.csv`, `counting_deep.csv`, `vietnamese_per_class.csv`, `manifest.json`), audit `verified: true` ([audit](../../results/final_diagnostics_v1_audit.json)). CI 95% là bootstrap theo nhóm cảnh (probe) hoặc theo lớp (Spearman), 2.000 lượt, seed 1402; CI gốc seed 0 được giữ để đối chiếu trong `metrics.json`.

Đối chiếu bắt buộc với [ma trận T0](../../docs/evidence_matrix_1117.md) (`evidence_matrix_1117_v1`) và run gốc đều khớp tuyệt đối: confusion 6×6 counting, số đúng theo điều kiện/nhãn, MAE, sai số có dấu, 804 ảnh control đúng, nhất quán 211/360, tỷ lệ chọn câu khẳng định 48,06%; độ chính xác 6 điều kiện tiếng Việt khớp `full_vietnamese_cifar100_v1`.

## 1. Negation (1.440 phép thử, 360 nhóm)

| Mẫu câu | Độ chính xác | Ảnh có vật (khẳng định) | Ảnh không có vật (phủ định) | Tỷ lệ chọn câu khẳng định |
|---|---|---|---|---|
| Tổng | 86,11 [84,31; 87,92] | 84,17 [80,56; 87,50] | 88,06 [85,28; 90,56] | 48,06 [45,56; 50,49] |
| with / without | 91,39 [89,44; 93,33] | 87,50 [83,89; 90,83] | 95,28 [93,06; 97,22] | 46,11 [43,89; 48,20] |
| contains / does not contain | 80,83 [78,33; 83,47] | 80,83 [76,67; 84,72] | 80,83 [76,39; 84,72] | 50,00 [46,81; 53,19] |

Ma trận 2×2 tổng (hàng nhãn thật, cột dự đoán): có vật → 606 đúng, 114 chọn câu phủ định; không có vật → 86 chọn câu khẳng định, 634 đúng. Chỉ 211/360 nhóm (58,61%) đúng cả 4 phép thử. Với *with/without*, mô hình hơi nghiêng về câu phủ định (chọn khẳng định 46,11%) nên ảnh không có vật dễ hơn; với *contains/does not contain* hai chiều cân bằng hoàn toàn. Kết quả vẫn trong miền hình học đơn giản, không suy ra "hiểu phủ định".

## 2. Counting (1.080 ảnh, 180 nhóm, 6 lựa chọn)

| Nhóm | n | Độ chính xác | MAE | Sai số có dấu |
|---|---:|---|---:|---:|
| Tất cả | 1.080 | 32,87 [30,65; 35,09] | 1,016 | -0,195 |
| Control đúng | 804 | 34,45 [31,61; 37,25] | 0,843 | 0,092 |
| Control sai | 276 | 28,26 [23,68; 33,72] | 1,518 | -1,033 |

Theo số vật thật 1 → 6, độ chính xác trên tất cả ảnh là 48,33 / 26,11 / 52,22 / 9,44 / 42,22 / 18,89%, trên nhóm control đúng là 45,93 / 28,83 / 61,84 / 14,53 / 34,00 / 6,00%: **không đơn điệu**, đỉnh ở 3 và đáy ở 4. Tỷ lệ trả lời "one" là 48,33% ở 1 vật, 12,22% ở 2 vật và đúng 11,11% (20/180) ở mỗi mức 3 đến 6. Hai chế độ kích thước gần nhau: bán kính cố định 32,41 [29,44; 35,56], tổng diện tích cố định 33,33 [30,19; 36,48]. Lỗi nhận dạng màu/hình làm counting kém thêm, nhưng ngay khi nhận dạng đúng, counting vẫn chỉ 34,45%.

## 3. Tiếng Việt (CIFAR-100, 10.000 ảnh, 100 lớp)

- **80/100 lớp đúng 0%** với prompt tiếng Việt có dấu. Các lớp tốt nhất: *tulip* "hoa tulip" 100%, *hamster* "chuột hamster" 95%, *couch* "ghế sofa" 85%, *kangaroo* "chuột túi kangaroo" 83%, *possum* "con thú có túi possum" 68%, *maple tree* "cây phong" 23%, *tank* "xe tăng" 16%.
- Heuristic chuỗi ký tự cố định trước khi xem kết quả (tên tiếng Anh nằm trong tên tiếng Việt không dấu, hoặc tỷ lệ giống nhau `difflib` ≥ 0,5) đánh dấu 6 lớp "gần tiếng Anh": trung bình **60,33%**, so với **1,57%** ở 94 lớp còn lại. Heuristic có sai số đã biết: "chair = cái ghế" bị đánh dấu nhầm (0%), còn "couch = ghế sofa" (chứa từ mượn *sofa*) bị bỏ sót. Đây là mô tả sơ bộ, cần người kiểm nếu đưa vào kết luận.
- Tương quan Spearman giữa số token BPE của tên lớp tiếng Việt và mức sụt so với tiếng Anh: **ρ = 0,05** [-0,14; 0,23]; bản không dấu: ρ = -0,12 [-0,33; 0,08]; tỷ số token Việt/Anh: ρ = 0,02 [-0,18; 0,21]. Số token **không** giải thích mức sụt; độ chính xác tiếng Việt chỉ tương quan yếu, âm với số token (ρ = -0,27 [-0,45; -0,06]).
- Kết hợp với điều kiện lai (khung Việt + tên Anh 55,10%; khung Anh + tên Việt 5,44%): mô hình gần như chỉ dùng được tên lớp khi tên đó chứa từ tiếng Anh hoặc từ mượn quen thuộc.

## 4. Hình (PDF vector + PNG 300 DPI, Times New Roman, dấu trừ là hyphen)

| File | Nội dung | Khổ |
|---|---|---|
| [fig_baseline_prompts](figures/fig_baseline_prompts.pdf) | 5 dataset: baseline vs embedding ensemble, Δ và CI, lớp kém nhất (ray 5/100, watercress 0/164, Bombay 0/88, lacelike 0/40, steak 106/250) | 2 cột |
| [fig_counting_negation](figures/fig_counting_negation.pdf) | (a) counting theo số vật, tất cả vs control đúng; (b) confusion 6×6; (c) negation theo nhãn thật và mẫu câu, tỷ lệ chọn câu khẳng định | 2 cột |
| [fig_spatial_overview](figures/fig_spatial_overview.pdf) | Set A, Set B (98, đã review), Set C (unseen): holistic center-crop, holistic full padded (T2), native half, padded half, oracle bbox ideal half, mức ngẫu nhiên | 2 cột |
| [fig_vietnamese_hybrid](figures/fig_vietnamese_hybrid.pdf) | (a) 6 điều kiện ngôn ngữ; (b) số token tên lớp vs mức sụt theo lớp, Spearman | 2 cột |

Ghi chú cho hình không gian: T1 trên Set B cho thấy CLIP nhận đúng hai vật 86,48%, nhưng độ chính xác quan hệ trên các ảnh nhận đúng cả hai vật vẫn chỉ 51,01%; T2 cho thấy đệm cả ảnh không cải thiện holistic (Δ = 0,00 [-3,06; 3,06]).

## 5. Tái lập

```powershell
python -m vlm_probe.final_diagnostics audit   # tính lại và đối chiếu, không ghi đè
python -m vlm_probe.final_figures             # vẽ lại 4 hình từ artifact đã audit
python -m unittest tests.test_final_diagnostics
```

`final_diagnostics run` từ chối chạy nếu `results/final_diagnostics_v1/` đã tồn tại (kết quả bất biến). `figures/figures_manifest.json` ghi hash mã nguồn và đầu vào của hình.
