# Bảng phản hồi góp ý giảng viên và khóa phạm vi (WS4)

Lập ngày 28/11/2026 cho Chặng 3 của [1124.md](../1124.md). Đối chiếu 1-1 năm góp ý của thầy Toàn (email 11/11/2026) với bằng chứng trong repo. Số liệu chi tiết và đường dẫn artifact: [evidence_matrix_final.md](evidence_matrix_final.md).

## 0. Phạm vi đã chốt

- **Mục tiêu:** hoàn thiện project môn học (báo cáo cuối kỳ 6-8 trang theo định dạng hội nghị, slide 15 phút, repo tái lập). Nhóm **không** viết bài báo để công bố; các gợi ý chỉ phục vụ hướng bài báo được ghi vào "hướng phát triển".
- **Một checkpoint:** mọi kết luận giới hạn trong CLIP ViT-B/32 gốc, FP32, không fine-tuning. Không thêm model hay dataset mới trong môn học, đúng như thầy đề nghị.
- **Dữ liệu đã mở:** Set A và Set B là dữ liệu đã dùng trước đó; kết luận kiểm chứng độc lập dựa trên Set C (chưa từng dùng) và review người.

## 1. Bảng đối chiếu

| # | Góp ý của thầy | Việc đã làm | Bằng chứng | Trạng thái | Cách trình bày trong báo cáo |
|---|---|---|---|---|---|
| 1a | Bổ sung related work: ARO, Winoground, What'sUp, VSR, NegBench, ReCLIP | Đoạn related work tiếng Anh có đủ 6 công trình, thêm CLIP, CoOp, SugarCrepe, CountBench, RegionCLIP, M-CLIP, AltCLIP, SigLIP, ViCLIP-OT | [related_work_comparison_v2.md](related_work_comparison_v2.md) §1; [final_literature.bib](references/final_literature.bib) (25 mục, đã kiểm nguồn) | Xong | Dùng nguyên §1 làm Section II |
| 1b | Không trình bày region-based inference như phương pháp mới; có tiền lệ ReCLIP | Định vị chia nửa ảnh là phép đo chẩn đoán cùng nguyên lý ReCLIP | related_work_comparison_v2.md §3 | Xong | Nói rõ "follows ReCLIP", chỉ dùng như diagnostic |
| 1c | Bảng ngắn đối chiếu: replication, khác biệt, đóng góp có thể phát triển | Bảng 13 dòng, cột Positioning | related_work_comparison_v2.md §2 | Xong | Rút gọn 6-8 dòng cho báo cáo |
| 2a | Negation: tách theo đáp án khẳng định/phủ định, kiểm tra thiên lệch | Tách theo nhãn và mẫu câu, ma trận 2×2, tỷ lệ chọn câu khẳng định, nhất quán 4 phép thử | [reports/final_diagnostics_v1](../reports/final_diagnostics_v1/README.md) §1; `results/final_diagnostics_v1/negation_deep.csv` | Xong | Khẳng định 84,17%, phủ định 88,06%, chọn khẳng định 48,06%; *with/without* nghiêng nhẹ về phủ định; 86,11% cao hơn literature vì nhiệm vụ 2 lựa chọn trên cảnh hình học đơn giản |
| 2b | Counting: báo accuracy khi recognition control đúng, phân tích theo số vật | Ba nhóm all/control đúng/control sai, theo số vật 1-6, confusion 6×6 | final_diagnostics_v1 §2; `counting_deep.csv`; hình `fig_counting_negation` | Xong | Control đúng 34,45% (n = 804); không đơn điệu theo số vật |
| 3a | Xác minh nhãn quan hệ COCO | R1 duyệt Set B (giữ 98/100); R2 chấm mù độc lập: đồng thuận chấp nhận 98%, hướng quan hệ 100% (κ = 1,0) | [human_review_followup_v1](../reports/human_review_followup_v1/README.md); [human_review_iaa_set_b_v1](../reports/human_review_iaa_set_b_v1/README.md) | Xong | Báo IAA và sensitivity trên 98 nguồn đồng thuận |
| 3b | Thử trên tập ảnh mới | Set C: 100 nguồn COCO train2014 chưa dùng, 2 reviewer mù (κ = 0,75 chấp nhận, 0,98 quan hệ), chạy GPU 1 lần sau khi đóng băng | [natural_unseen_holdout_v1](../reports/natural_unseen_holdout_v1/README.md) | Xong | Holistic 47,00% vs chia vùng 89,00%, Δ +42,00 điểm |
| 3c | Baseline bounding box và quy tắc hình học | Oracle bbox centroid và ideal half-canvas trên Set A, B (T2) và Set C; thêm T1 (đối chứng nhận dạng trên ảnh tự nhiên) và holistic đệm cả ảnh | [spatial_ablation_v1](../reports/spatial_ablation_v1/README.md); [natural_recognition_v1](../reports/natural_recognition_v1/README.md) | Xong | Chia vùng 93,37% gần trần ideal half 99,49% (Set B); đệm cả ảnh Δ 0,00; CLIP nhận vật 86,48% nhưng quan hệ trên ảnh nhận đúng vẫn 51,01% |
| 4a | Literature về multilingual VLM | M-CLIP, AltCLIP, SigLIP, ViCLIP-OT trong related work | related_work_comparison_v2.md §1, §2 | Xong | Một đoạn ngắn trong Section II |
| 4b | Xác minh chất lượng bản dịch | R1 xác nhận nghĩa 100/100 tên lớp | [human_review_followup_v1](../reports/human_review_followup_v1/README.md) | Xong (một người kiểm) | Ghi rõ một reviewer, không phải hội đồng ngôn ngữ |
| 4c | Phân biệt tokenization, từ vựng, dữ liệu pretraining | Điều kiện lai, per-class, heuristic từ mượn, Spearman token vs mức sụt | final_diagnostics_v1 §3; hình `fig_vietnamese_hybrid` | Xong trong phạm vi một checkpoint | Số token không giải thích mức sụt (ρ = 0,05); chỉ tên chứa từ tiếng Anh/từ mượn còn đúng |
| 4d | Thử mô hình đa ngôn ngữ | Không làm (thầy nói không cần trong môn học; nhóm không làm bài báo) | - | Ngoài phạm vi | Hướng phát triển |
| 5 | Giới hạn kết luận trong một checkpoint | Khóa phạm vi ở §0; mọi README ghi "một checkpoint" | §0; các README kết quả | Áp dụng khi viết | Câu giới hạn ở Abstract, Discussion và slide kết luận |

## 2. Những điểm cần nói thẳng trong báo cáo và Q&A

- Các hạn chế của CLIP về counting, spatial và negation **đã được biết** trong literature; dự án không tuyên bố phát hiện mới mà đóng góp các phép đo có đối chứng trên một checkpoint.
- Chia nửa ảnh **cùng nguyên lý ReCLIP**; giá trị nằm ở các đối chứng T1, T2, oracle bbox, IAA và Set C, không ở bản thân phương pháp.
- Negation 86,11% **không mâu thuẫn** NegBench: nhiệm vụ của nhóm là chọn 1 trong 2 caption trên cảnh hai vật đơn giản; phụ thuộc mẫu câu (chênh 10,56 điểm) và nhất quán 4 phép thử chỉ 58,61% phù hợp với literature.
- Set A và Set B đã mở, nên kết quả trên đó là phân tích tiếp nối; khẳng định khái quát dựa trên Set C.
- Review bản dịch và Set B ban đầu có một reviewer (R1); Set B sau đó có R2 độc lập, Set C có hai reviewer mù.

## 3. Hướng phát triển (không làm trong môn học)

Thử mô hình đa ngôn ngữ hoặc mô hình tiếng Việt trên cùng 10.000 ảnh CIFAR-100; mở rộng counting sang ảnh tự nhiên (CountBench); nghiên cứu điều kiện chia vùng thất bại (hai vật cùng một nửa, quan hệ dọc); kiểm tra trên checkpoint khác.

## 4. Sai lệch phát hiện khi lập tài liệu

- 1124.md ghi AltCLIP thuộc EMNLP 2023; bản ghi chính thức là Findings of ACL 2023 (5 tác giả).
- Bản draft IEEE 18/11 ghi sai số trang Winoground; đúng là 5228-5238.
- Trích dẫn Oxford-IIIT Pet phải là *Cats and Dogs*, CVPR 2012.
