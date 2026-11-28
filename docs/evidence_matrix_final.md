# Ma trận bằng chứng cuối (WS4, khóa 28/11/2026)

Truy nguyên mọi số liệu dùng cho báo cáo cuối kỳ và slide. Phạm vi: **một checkpoint CLIP ViT-B/32 gốc**, FP32, không fine-tuning. Mọi số đọc từ artifact đã commit và đã audit; không có suy luận mới khi lập tài liệu này. Bản trước: [evidence_matrix_1117.md](evidence_matrix_1117.md) (T0, giữ nguyên làm lịch sử). Phản hồi góp ý giảng viên: [scope_and_rebuttal_tracker.md](scope_and_rebuttal_tracker.md).

Quy ước: % là độ chính xác top-1 trừ khi ghi khác; CI 95% bootstrap theo nhóm cảnh, nguồn ảnh hoặc lớp (2.000 lượt, seed 1402) trừ khi ghi khác; "đã mở" nghĩa là dữ liệu đã được xem và dùng trước đó, phân tích tiếp theo là hậu nghiệm.

## 1. Phân loại zero-shot và prompt (RQ1, RQ3)

| Luận điểm | Số liệu | Mẫu số | Artifact | Kiểm | Giới hạn |
|---|---|---|---|---|---|
| Top-1 biến thiên mạnh theo miền | CIFAR-100 64,48; Flowers102 63,77; Pet 85,06; DTD p1 42,45; Food-101 82,31 | 10.000 / 6.149 / 3.669 / 1.880 / 25.250 ảnh | `reports/baseline/summary.csv`, `results/full_*_clip_b32_v*/metrics.json` | Audit run gốc | Một template; không so chênh giữa dataset như hiệu ứng model |
| Trung bình che lớp yếu | ray 5/100; watercress 0/164; Bombay 0/88; lacelike 0/40; steak 106/250; lỗi Pet 544/548 cùng loài | Theo lớp | `results/full_*_clip_b32_v*/predictions.csv`; `results/evidence_matrix_1117_v1/pet_species.csv` | `final_figures`, `evidence_1117 audit` | Mô tả hậu nghiệm |
| Ensemble embedding (7 template) không lợi đồng đều | Δ +0,49 [0,10; 0,89]; +1,69 [1,19; 2,20]; +0,55 [0,05; 1,04]; -0,53 [-1,49; 0,32]; +1,21 [1,02; 1,40] | Cùng ảnh, paired | `reports/research_review_v2/multidomain_summary.csv`; `results/full_prompt_*` | Audit review v2 | Bộ template cố định, không tìm template tốt nhất |
| Độ nhạy template CIFAR-100 | 60,39 (nhãn trần) đến 64,48 (gốc); không template mới nào vượt gốc | 10.000 ảnh | `results/full_prompt_cifar100_v1/metrics.json` | Audit | Một dataset |
| Thêm tên nhóm lớn làm giảm | 62,92 vs 64,48: Δ -1,56 [-2,14; -0,98]; so neutral -1,17 [-1,77; -0,61]; so trung bình 5 category sai +8,73 [8,14; 9,33] | 10.000 ảnh | `results/full_depth_category_v1/metrics.json` (`planned_comparisons`) | Audit | Một mẫu câu category |

## 2. Counting, negation, fine-grained (RQ2)

| Luận điểm | Số liệu | Mẫu số | Artifact | Kiểm | Giới hạn |
|---|---|---|---|---|---|
| Counting yếu | 32,87 [30,65; 35,09] (seed 0 gốc: [30,65; 35,00]); MAE 1,016 | 1.080 ảnh, 180 nhóm, 6 lựa chọn | `results/final_diagnostics_v1/metrics.json` (`counting.all`) | `final_diagnostics audit`, khớp T0 | Hình học |
| Không chỉ do nhận dạng | Control đúng 34,45 [31,61; 37,25] (n = 804); control sai 28,26 [23,68; 33,72] (n = 276) | 1.080 | `counting.control_correct`, `counting.control_wrong` | Như trên | Control 74,44% chưa hoàn hảo |
| Không đơn điệu theo số vật | 1 → 6: 48,33 / 26,11 / 52,22 / 9,44 / 42,22 / 18,89; trả lời "one" 11,11% ở mỗi mức 3-6 | 180 ảnh mỗi mức | `counting.by_true_count`; `counting_deep.csv` | Như trên | Mô tả |
| Negation tương đối tốt nhưng phụ thuộc mẫu câu | 86,11 [84,31; 87,92]; *with/without* 91,39; *contains/does not contain* 80,83 | 1.440 phép thử, 360 nhóm | `negation.all`, `negation.with_without`, `negation.contains_does_not` | Như trên, khớp T0 | Cảnh hai vật đơn giản, 2 lựa chọn |
| Theo nhãn thật và thiên lệch | Khẳng định 84,17 [80,56; 87,50]; phủ định 88,06 [85,28; 90,56]; chọn khẳng định 48,06 [45,56; 50,49] (46,11 với *with/without*, 50,00 với *contains*) | 720 mỗi nhãn | `negation.*.accuracy_present/absent`, `affirmative_prediction_rate`; `negation_deep.csv` | Như trên | Như trên |
| Nhất quán thấp | 211/360 nhóm đúng cả 4 phép thử (58,61%) | 360 nhóm | `negation.four_trial_consistency` | Như trên | Như trên |
| Fine-grained trong nhóm lớn | 77,95 [77,16; 78,77]; kém nhất fish 56,60% | 10.000 ảnh, 5 lựa chọn | `results/full_probing_clip_b32_v1/metrics.json` | Audit | **Oracle** (biết nhóm lớn), không so với 100 lớp |

## 3. Quan hệ không gian (RQ2)

| Luận điểm | Số liệu | Mẫu số | Artifact | Kiểm | Giới hạn |
|---|---|---|---|---|---|
| Hình học: nhận dạng tốt, quan hệ gần ngẫu nhiên | Core 29,58 (control 97,50); depth 27,36 [26,18; 28,61] (control 96,94); *right* 0/180 | 720 / 1.440 ảnh, 4 lựa chọn | `results/full_probing_clip_b32_v1`, `results/full_depth_spatial_v1/metrics.json` (`t0_o0_all`); CI từ evidence_matrix_1117 | Audit | Hình học |
| Chia vùng trên hình học | 100% | 1.440 ảnh | `results/full_spatial_mitigation_review_v2/metrics.json` | Audit | Cảnh đơn giản |
| COCO Set A holistic | 4 lựa chọn: 29,00 [24,50; 33,50] (400 ảnh); ảnh gốc 28,00 (chọn *right* 62%) | 200 nguồn | `results/full_spatial_ablation_v1/metrics.json` (`sets.A`); `results/full_depth_natural_v1` | `ablation_v1 audit` | Đã mở, chưa review người |
| Nhãn Set B đã kiểm | R1 giữ 98/100; R2 chấm mù: chấp nhận đồng thuận 98% (κ = 0, nghịch lý prevalence), hướng quan hệ 100% (κ = 1,0) | 100 nguồn | `results/human_review_followup_v1`, `results/human_review_iaa_set_b_v1/metrics.json` | `human_review_followup audit`, `human_review_iaa audit` | Set B đã mở |
| Holistic Set B quanh ngẫu nhiên | 51,53 [49,49; 53,57]; 2×2 52/98 = 53,06, Wilson [43,25; 62,64], p = 0,3069; cả hai ảnh đúng 4,08% | 98 nguồn / 196 ảnh | `results/human_review_iaa_set_b_v1/metrics.json` (`consensus_sensitivity_no_new_inference`) | Như trên | Không suy ra "bằng ngẫu nhiên" từ p > 0,05 |
| T1: CLIP nhận ra vật trên ảnh tự nhiên | 339/392 = 86,48 [81,38; 91,07]; thay A 83,16, thay B 89,80; quan hệ trên 149 ảnh nhận đúng cả hai vật: 76/149 = 51,01% | 98 nguồn / 196 ảnh | `results/full_natural_recognition_v1/metrics.json` | `natural_recognition audit` (`verified`, parity ≤ 1e-6) | Bảng chéo mô tả (nhóm chọn theo hành vi model) |
| T2: center-crop không phải nguyên nhân | Holistic đệm cả ảnh trừ center-crop: 0,00 [-3,06; 3,06] (Set B); -0,75 [-4,00; 2,25] (Set A) | 98 / 200 nguồn | `results/full_spatial_ablation_v1/metrics.json` (`primary_comparison`) | `ablation_v1 audit`, giao thức commit trước inference | Phân tích tiếp nối trên dữ liệu đã mở |
| Chia vùng gần trần hình học | Set B: native 93,37 [88,27; 97,45] (Δ 41,84 [36,73; 46,43]); padded 92,86; oracle bbox ideal half 99,49; centroid 100 | 98 nguồn | `sets.B.conditions` | Như trên | Oracle dùng bbox thật |
| Set A (4 lựa chọn) | Native 72,25 [66,25; 78,25]; padded 68,75; ideal half 92,00; centroid 99,50 | 200 nguồn | `sets.A.conditions` | Như trên | Chưa review người |
| Điều kiện chia vùng thất bại | Hai vật cùng một nửa: Set B 2/196 ảnh; Set A 20/400, native 30,00% vs 74,47% ở ảnh còn lại | | `sets.*.same_half` | Như trên | Mẫu nhỏ, mô tả |
| Hoán vị vùng | Set B 6,63% (bằng 100 trừ 93,37, kỳ vọng số học) | 98 nguồn | `sets.B.conditions.swapped_region` | Như trên | Kiểm tra pipeline |
| Set C chưa từng dùng: nhãn | 2 reviewer mù: chấp nhận 151/160 (94,38%, κ = 0,7493); hướng quan hệ 158/160 (98,75%, κ = 0,9756); giữ 50 trái + 50 phải | 160 ứng viên → 100 nguồn | `reviews/natural_unseen_holdout_v1/pre_adjudication_iaa.json`, `freeze.json` | Đóng băng trước inference | COCO train2014, cùng miền COCO |
| Set C: kết quả | Holistic 47,00 [44,00; 49,50], 2×2 46/100 (Wilson [36,56; 55,74]); native 89,00 [83,00; 94,50], 2×2 91/100 (Wilson [83,77; 95,19]), Δ +42,00 [35,00; 48,50]; padded 81,00; oracle ideal half 92,00, centroid 100 | 100 nguồn / 200 ảnh | `results/full_natural_unseen_holdout_v1/metrics.json` | `unseen_holdout audit` | Chạy 1 lần duy nhất |

## 4. Prompt tiếng Việt (RQ4)

| Luận điểm | Số liệu | Mẫu số | Artifact | Kiểm | Giới hạn |
|---|---|---|---|---|---|
| Tên lớp tiếng Việt làm sập độ chính xác | EN 64,48; VI có dấu 5,10; không dấu 4,05; khung EN + tên VI 5,44; khung VI + tên EN 55,10; song ngữ 54,95; ensemble EN+VI 60,42 | 10.000 ảnh, 100 lớp | `results/full_vietnamese_cifar100_v1/metrics.json`; `results/final_diagnostics_v1/metrics.json` (`vietnamese.overall_top1_percent`) | `final_diagnostics audit` (khớp tuyệt đối) | Một checkpoint tiếng Anh |
| Bản dịch đã kiểm | R1 xác nhận nghĩa 100/100 | 100 tên lớp | `reports/human_review_followup_v1/translation_review.json` | Checksum | Một reviewer |
| Gần như chỉ tên chứa từ tiếng Anh/từ mượn còn đúng | 80/100 lớp đúng 0%; 6 lớp heuristic "gần tiếng Anh" trung bình 60,33% vs 1,57% (94 lớp còn lại); tốt nhất tulip 100, hamster 95, couch 85, kangaroo 83, possum 68 | 100 lớp | `vietnamese.zero_accuracy_vi_diacritics`, `cognate_heuristic`; `vietnamese_per_class.csv` | Như trên | Heuristic chuỗi ký tự (sai: chair; sót: couch) |
| Số token không giải thích mức sụt | Token tên lớp: EN 1,19, VI 7,05, không dấu 2,88; Spearman token vs mức sụt ρ = 0,05 [-0,14; 0,23]; không dấu ρ = -0,12 [-0,33; 0,08] | 100 lớp | `vietnamese.spearman` | Như trên | Tương quan, không nhân quả |

## 5. Trạng thái kiểm tra

- Audit offline đều `verified: true`: `natural_recognition` (T1), `unseen_holdout` (Set C), `human_review_iaa` (Set B), `ablation_v1` (T2), `final_diagnostics` (WS3), cùng các audit lịch sử.
- Unit tests: 102 test trong `lab22_env` (Python 3.10): 100 đạt, 1 skip, 1 lỗi môi trường (`hashlib.file_digest` cần Python ≥ 3.11, đạt khi chạy Python 3.14).
- Hình cho báo cáo: [reports/final_diagnostics_v1/figures](../reports/final_diagnostics_v1/figures/) (PDF vector + PNG 300 DPI).

## 6. Không được viết trong báo cáo

- "CLIP không hiểu phủ định" (số liệu không ủng hộ trong thiết kế này).
- "Chia vùng là phương pháp mới" hoặc "encoder hiểu quan hệ tốt hơn khi crop" (cùng nguyên lý ReCLIP; lợi ích dựa vào cấu trúc gán vùng và tên hai vật).
- "Holistic bằng ngẫu nhiên" từ p > 0,05.
- Gộp accuracy 2 lựa chọn, 4 lựa chọn, 6 lựa chọn và oracle thành một con số.
- Khái quát cho mọi VLM.
