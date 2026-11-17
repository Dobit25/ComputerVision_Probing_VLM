# Ma trận bằng chứng T0 — 17/11/2026

Thực hiện mục **T0** của [1117.md](../1117.md). Mọi số dưới đây đọc lại từ artifact đã commit (scores/predictions/metrics), **không có inference mới**. Phần tính lại được lưu máy đọc được ở [`results/evidence_matrix_1117_v1/`](../results/evidence_matrix_1117_v1/) (metrics.json + CSV, có hash đầu vào), tạo bởi `python -m vlm_probe.evidence_1117 run` và kiểm bằng `... audit`. Phạm vi: OpenAI CLIP ViT-B/32, không fine-tuning.

Kiểm tra trước khi lập ma trận (17/11/2026, nhánh `dev` tại `b83f3dd`):

- `python -m vlm_probe.human_review_followup audit` → **Audit passed** (phiếu đóng băng, hash nguồn, metrics, trace từng ảnh/source).
- `python -m unittest discover -s tests` → xem mục 4 (kết quả theo môi trường).

## 1. Ma trận

Cột “Dữ liệu”: *đã mở* = scores/test đã được xem và dùng để ra quyết định; mọi phân tích tiếp theo trên cùng dữ liệu là hậu nghiệm.

| RQ | Luận điểm (phạm vi được phép) | Artifact / run | Mẫu số | Dữ liệu | Đối chứng | CI / seed | Giới hạn | Còn thiếu |
|---|---|---|---|---|---|---|---|---|
| Classification | Top-1 biến thiên mạnh theo miền: CIFAR-100 64,48%, Flowers 63,77%, Pet 85,06%, DTD p1 42,45%, Food 82,31% | `results/full_{cifar100_clip_b32_v1,flowers102_clip_b32_v2,oxford_pet_clip_b32_v3,dtd_clip_b32_v3,food101_clip_b32_v3}`, `reports/baseline/summary.csv` | 10.000 / 6.149 / 3.669 / 1.880 / 25.250 ảnh | Đã mở (test chính thức) | Một template cố định; per-class, confusion pairs | Không có CI cho baseline Top-1 (toàn bộ test) | Một checkpoint, một template; không trung bình gộp theo ảnh | Đủ cho luận điểm mô tả |
| Prompt / ensemble | Ensemble embedding không có lợi đồng đều: Δ +0,49 / +1,69 / +0,55 / **−0,53** / +1,21 điểm % | `reports/research_review_v2/multidomain_summary.csv`, `results/full_prompt_*_review_v2/` | Như trên | Đã mở | Cùng ảnh, paired | Paired CI; DTD [−1,49; +0,32] chứa 0 | Bộ template cố định, không phải tìm template tối ưu | Đủ |
| P1.3 Counting | 32,87% (6-way, chance 16,67%), MAE 1,0157, signed −0,1954; không đơn điệu theo số lượng (3: 52,22%, 4: 9,44%) | `results/full_probing_clip_b32_v1/{predictions.jsonl,metrics.json}`, `reports/probing/breakdown.csv` | 1.080 ảnh, 180 nhóm cảnh | Đã mở | Control màu/hình 804/1.080 = 74,44%; counting khi control đúng 277/804 = 34,45%; fixed_radius 32,41% vs fixed_total_area 33,33% | Group bootstrap [30,65; 35,00], 1.000 lượt, seed 0 | Chỉ miền hình học; không phải CountBench | Counting tự nhiên chưa có (mục 6 của 1117 — chỉ mở nếu cần). **Sửa số màu/hình trong bản đồ P1 (mục 3)** |
| P1.3 Negation | 86,11% (2-way); chênh template 10,56 điểm % (91,39% vs 80,83%); present 84,17%, absent 88,06%; nhất quán cả 4 phép thử 211/360 = 58,61%; tỷ lệ chọn khẳng định 48,06% | `results/full_probing_clip_b32_v1`, `reports/probing/README.md` | 1.440 phép thử / 720 ảnh / 360 nhóm | Đã mở | Control 1.406/1.440 = 97,64%; khi control đúng 86,91% | Group bootstrap [84,23; 88,06] | Hình học hai vật; không suy ra “không hiểu phủ định” | Negation tự nhiên chưa có; không bắt buộc |
| P1.4 Fine-grained (oracle) | Trong 5 lớp cùng superclass vẫn 2.205 lỗi (77,95%, chance 20%); kém nhất fish 56,60%, aquatic_mammals 64,00%, trees 66,20% | `results/full_probing_clip_b32_v1` | 10.000 ảnh | Đã mở | Parity 100-way khớp 10.000/10.000 với baseline | Image bootstrap [77,16; 78,77] | **Oracle 5-way**, không phải cải thiện so với 64,48% | Không dùng làm bằng chứng duy nhất → xem dòng dưới |
| P1.4 Fine-grained (tự nhiên, toàn candidate set) | Lỗi Pet gần như hoàn toàn là nhầm giống **cùng loài**: 544/548 lỗi giữ đúng cat/dog, chỉ 4 lỗi khác loài (0,73% số lỗi; cat 3/311, dog 1/237) | `results/full_oxford_pet_clip_b32_v3/predictions.csv` (37-way) → `results/evidence_matrix_1117_v1/pet_species.csv`; nhóm loài theo quy ước chính thức Oxford-IIIT (tên file viết hoa = cat), độc lập scores | 3.669 ảnh (cat 1.183 / 12 giống, dog 2.486 / 25 giống) | Đã mở | Taxonomy chính thức của dataset | Image bootstrap, seed 1402, 2.000 lượt: tỷ lệ khác loài trong số lỗi [0,18; 1,46]%; trên toàn 3.669 ảnh 0,11% [0,03; 0,22]% | Pet là taxonomy hai mức đơn giản | **Flowers/Food chưa có taxonomy độc lập scores** — cần nguồn phân nhóm (họ thực vật; nhóm món ăn) được review trước khi tính; nếu không có, giữ ở mức confusion pairs |
| Spatial hình học | Nhận dạng hai vật ~97% nhưng 4 quan hệ 29,58% (core) / 27,36% (depth T0); thiên lệch `above`; right 0/180 | `results/full_probing_clip_b32_v1`, `results/full_depth_spatial_v1/metrics.json` | 720 / 1.440 ảnh | Đã mở | Control 97,50% / 96,94%; nền trống trùng 23,82%; câu đảo tương đương (đồng thuận 16,25%) | Cluster bootstrap 120 nhóm, 2.000 lượt, seed 1402: [26,18; 28,61] | Hình học; không định vị lỗi ở encoder nào | Đủ cho miền hình học |
| Category controls (CIFAR-100) | Category đúng > TB 5 category sai +8,73 điểm %, nhưng thấp hơn baseline −1,56 và neutral −1,17 | `results/full_depth_category_v1`, `results/full_cifar100_category_v1` | 10.000 ảnh × điều kiện | Đã mở | Neutral suffix khớp token; 5 derangement | Class-stratified paired bootstrap 2.000 lượt | Một template category | Đủ |
| Spatial tự nhiên — holistic | COCO 2-way từng ảnh T2/o0 51,53%; ghép 2×2 52/98 = 53,06%, Wilson [43,25; 62,64], p ≈ 0,3069; thiên lệch mạnh theo quan hệ (18,37% vs 84,69%) | `results/human_review_followup_v1/metrics.json` (`region_comparison.after_98.holistic`) | 98 nguồn / 196 ảnh (gốc + lật) | Đã mở; **sau lọc là sensitivity hậu nghiệm** | Gốc/lật cùng nguồn; 2×2 triệt tiêu thiên lệch cộng | Bootstrap theo source, seed 1402, 2.000 lượt | Một reviewer; không phải holdout mới; không suy ra “bằng ngẫu nhiên” | **Control nhận dạng tự nhiên (foil) chưa có** → T1 |
| Spatial tự nhiên — 4-way | COCO 4-way T2 thuận: 200 ảnh gốc 28,00% (chọn `right` 124/200 = 62,00%); 200 ảnh lật 30,00%; gộp 400 ảnh 29,00% | `results/full_depth_natural_v1/metrics.json` (`t2_o0_*`) | 200 nguồn; 400 ảnh gồm gốc + lật | Đã mở | Gốc/lật | Bootstrap theo nguồn | 4-way ≠ 2-way; không so trực tiếp với dòng trên | — |
| Chia vùng | Native half 93,37% vs holistic 51,53%: +41,84 điểm %, CI [36,73; 46,43]; 2×2 92,86%; padded 92,86% | `results/human_review_followup_v1/metrics.json`, `results/full_natural_regions_*_review_v2/` | 98 nguồn / 196 ảnh | Đã mở | Padded (secondary); cùng mẫu, paired | Bootstrap theo source, seed 1402 | Dùng cấu trúc hình học + **tên hai vật** + 4 lượt encode; không phải cải thiện encoder | Ablation tách nguồn lợi ích (T2) — chỉ sau T1 |
| Cơ chế B/Δ | `spatial_diagnostics/mechanism.py` phân rã điểm 2×2 thành thiên lệch B và tín hiệu Δ | `vlm_probe/spatial_diagnostics/mechanism.py` | — | — | — | — | Phân rã đại số trên scores, **không** phải bằng chứng cơ chế nội tại | Không ưu tiên (mục 5 của 1117) |
| Tiếng Việt (CIFAR-100) | Tên lớp VI làm sập accuracy (VI có dấu 5,10%, EN khung + VI tên 5,44%), khung VI + tên EN 55,10%; ensemble 60,42% < EN 64,48% | `results/full_vietnamese_cifar100_v1/`, `results/review_hieu_v2/full_vietnamese_cifar100_v1/`; review dịch 100/100 accepted | 10.000 ảnh × 6 điều kiện | Đã mở | Bỏ dấu; hoán vị khung/tên; song ngữ | Paired | Không chứng minh BPE là nguyên nhân; không phải dataset văn hóa VN; raw text vectors run cũ không đầy đủ | Miền thứ hai (Food-101) và P5 — tùy chọn |

## 2. Kết luận nào đã đủ, thiếu phép kiểm nào

- **Đủ trong phạm vi đã nêu:** baseline 5 miền, prompt/ensemble, counting/negation/spatial hình học, category controls, holistic vs native trên 98 nguồn.
- **Gap thực sự (cần dữ liệu/protocol mới):**
  1. **T1 — control nhận dạng trên chính ảnh COCO.** Artifacts tự nhiên (`full_depth_natural_v1`, `full_natural_confirmation_v1`, `full_natural_regions_*_review_v2`, `human_review_followup_v1`) chỉ có caption quan hệ hoặc prompt tên vật đơn lẻ dùng cho quy tắc native; không có caption nhận dạng hai vật vs foil thay vật. Vì vậy chưa thể phân biệt “không nhận ra vật” với “không kết hợp quan hệ” trên ảnh tự nhiên. Cần reviewer người xác nhận foil vắng mặt trước inference — **agent không tự làm được bước này**.
  2. **P1.4 tự nhiên cho Flowers/Food:** cần taxonomy độc lập scores (được review) — chưa có.
- **Gap chỉ là tổng hợp (đã xử lý từ scores):** phân tích loài Pet ở bảng trên; confusion P1.3 ở mục 2.1.

### 2.1. Đối chiếu P1.3 từ scores

Counting (hàng = số thật, cột = dự đoán; [CSV](../results/evidence_matrix_1117_v1/counting_confusion.csv)), support 180 mỗi hàng:

| Thật \ Dự đoán | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | **87** | 82 | 11 | 0 | 0 | 0 |
| 2 | 22 | **47** | 102 | 1 | 8 | 0 |
| 3 | 20 | 9 | **94** | 5 | 48 | 4 |
| 4 | 20 | 0 | 69 | **17** | 63 | 11 |
| 5 | 20 | 0 | 30 | 41 | **76** | 13 |
| 6 | 20 | 0 | 6 | 35 | 85 | **34** |

Mỗi hàng 2–6 có đúng 20 ảnh bị đoán `one`; MAE 1,0157, signed −0,1954. Negation: present 606/720, absent 634/720, with_without 658/720, contains_does_not 582/720; nhất quán cả 4 phép thử 211/360 (từ `reports/probing/README.md`). Toàn bộ breakdown tính lại: [probing_breakdown_recomputed.csv](../results/evidence_matrix_1117_v1/probing_breakdown_recomputed.csv).
- **Không mở dataset/model mới**: không có câu hỏi nào ở trên cần checkpoint thứ hai.

## 3. Sai lệch phát hiện trong tài liệu cũ

[docs/p1_claim_evidence_map.md](p1_claim_evidence_map.md) mục 2.1 ghi counting theo màu/hình là blue 133, green 121, red 101; circle 127, triangle 126, square 102. Tính lại từ `results/full_probing_clip_b32_v1/predictions.jsonl` và đối chiếu `reports/probing/breakdown.csv` cho:

| Nhóm | Đúng / 360 |
|---|---:|
| red | 136 (37,78%) |
| green | 121 (33,61%) |
| blue | 98 (27,22%) |
| triangle | 135 (37,50%) |
| circle | 121 (33,61%) |
| square | 99 (27,50%) |

Confusion 6×6, MAE và signed error trong bản đồ P1 khớp scores. Theo quy tắc bảo toàn, **không sửa đè** tài liệu cũ; số đúng là `breakdown.csv`. Mọi trích dẫn tiếp theo dùng bảng này.

## 4. Tests

Trên máy chạy T0 (Python 3.14 hệ thống, không có torch/open_clip): 59/61 đạt; 2 lỗi là `ModuleNotFoundError` (`test_depth...neutral_selection` cần `open_clip`, `test_evaluator...partial_batch` cần `torch`), không phải lỗi logic. Trong venv CPU cài theo `requirements.txt` (torch 2.14.1, open_clip_torch 3.3.0): **61/61 tests đạt** trước khi thêm `tests/test_evidence_1117.py`; sau khi thêm: **65/65 đạt**.

## 5. Tái lập phần số mới

```powershell
python -c "import csv;R=list(csv.DictReader(open('results/full_oxford_pet_clip_b32_v3/predictions.csv')));sp=lambda r:'cat' if r['sample_id'].split('/')[-1][0].isupper() else 'dog';S={r['target_name']:sp(r) for r in R};E=[r for r in R if r['correct']=='0'];print(len(E),sum(S[r['target_name']]!=S[r['prediction_name']] for r in E))"
```

Kết quả mong đợi: `548 4`. Bản đầy đủ có CI: `python -m vlm_probe.evidence_1117 audit`.

## 6. Đề xuất phạm vi vòng tiếp theo

T1 là bước duy nhất trả lời trực tiếp câu hỏi chính (nhận dạng vs kết hợp quan hệ trên ảnh tự nhiên); cần nhóm chốt reviewer cho foil và ngân sách. T2 hoãn tới khi có kết quả T1. T3, tiếng Việt miền thứ hai, P5 và checkpoint thứ hai hoãn — không giải quyết gap lớn nhất hiện tại.
