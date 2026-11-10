# Đề Cương Chi Tiết Báo Cáo Cuối Kỳ (6–8 Trang) & Kịch Bản Trình Bày 15 Phút (P5.1–P5.3)

- **Chủ đề:** Topic 14 — *Probing Vision–Language Models: Systematic Evaluation of Compositional Failure Modes, Prompt Sensitivity, and Cross-Lingual Bottlenecks in OpenAI CLIP ViT-B/32*
- **Câu chuyện khoa học xuyên suốt (Unified Narrative):**
  1. **Nền tảng thực nghiệm có kiểm chứng (`RQ1`):** CLIP ViT-B/32 đạt hiệu năng zero-shot mạnh trên nhận diện đối tượng chuẩn (`85,06%` trên Oxford-IIIT Pet, `82,31%` trên Food-101, `64,48%` trên CIFAR-100), nhưng suy giảm nghiêm trọng trên các bài toán suy luận cấu trúc: đếm số lượng (`32,87%`), phủ định phụ thuộc mạnh vào mẫu câu (`91,39%` vs `80,83%`), và sụp đổ hoàn toàn về mức ngẫu nhiên trên quan hệ không gian (`29,58%` 4-way; `53,00%` 2×2 assignment trên tập xác nhận COCO độc lập, $p = 0.3086$).
  2. **Giải phẫu cơ chế sụp đổ không gian & giới hạn khắc phục (`RQ2`):** Phân rã ma trận điểm 2×2 chứng minh thiên kiến bất biến vị trí ($\mathbb{E}[\lvert B \rvert]$) lớn gấp `11,2x–16,7x` tín hiệu đổi chỗ không gian ($\mathbb{E}[\lvert \Delta_{\text{spatial}} \rvert]$), đẩy cả hai ảnh đối xứng về cùng một nhãn dự đoán trên `93%–98,6%` số cặp (`both_images_correct = 0,56%–4,00%`). Phương pháp chia vùng không huấn luyện (`Half-Canvas Zero-Shot Scoring`) giải quyết triệt để lỗi này trên ảnh hình học (`100%` giới hạn hình học) nhưng bị giới hạn trên ảnh tự nhiên COCO khi cả 2 vật thể nằm cùng một nửa khung hình (`7,0%–13,5%` số nguồn).
  3. **Khi nào Prompt Ensembling giúp ích và khi nào vô hiệu (`RQ3`):** Gộp nhiều prompt (Text-Embedding Ensemble & Logit/Probability Ensemble) giúp vượt qua từng template đơn lẻ trên phân loại chuẩn (`baseline + category + neutral` trên CIFAR-100 đạt `64,83%` so với `64,48%` của baseline) và giảm một nửa thiên lệch khẳng định/phủ định trên bài Negation (`-7,78` pp xuống `-4,17` pp, đạt `86,81%`), nhưng **không thể giải cứu** bài toán quan hệ không gian (`T0+T1+T2` Logit Ensemble vẫn chỉ đạt `23,68%` 4-way trên ảnh tổng hợp và `54,00%` 2×2 assignment / `1,00%` both correct trên COCO confirmation).
  4. **Nút cổ chai BPE Tokenization trên Tiếng Việt (`RQ4`):** Bộ tách từ byte-level BPE phân mảnh tên lớp tiếng Việt có dấu thành trung bình `4,27` subword tokens (gấp `3,58x` tiếng Anh `1,31` tokens; Fertility `1,86` tokens/từ so với `1,00` khi lược dấu), giải thích cơ chế suy giảm khi chuyển ngữ và vai trò của chiến lược ghép song ngữ Việt–Anh (`vi_en_bilingual`).

---

## 1. P5.1 — Cấu Trúc Báo Cáo Học Thuật 6–8 Trang (Chuẩn 2 Cột IEEE/CVPR)

| Mục báo cáo | Độ dài dự kiến | Nội dung cốt lõi & Liên kết Claim (`C01`–`C07`) | Bảng / Hình minh họa bắt buộc |
| --- | :---: | --- | --- |
| **Abstract** | `0,25` trang | Tóm tắt 4 câu hỏi nghiên cứu (`RQ1`–`RQ4`), quy mô thực nghiệm (`46.948` ảnh chuẩn + `13.240` phép thử core probing + `8.640` phép thử spatial depth + `300` nguồn COCO), và các phát hiện định lượng chính. | — |
| **1. Introduction & Research Questions** | `0,75` trang | Động cơ nghiên cứu tính hợp thành (compositionality) của VLM; phát biểu tường minh `RQ1`–`RQ4`; đóng góp về quy trình chống rò rỉ dữ liệu kiểm tra (`Data Exposure Ledger`). | **Hình 1:** Sơ đồ tổng quan kiến trúc đánh giá 4 trục (`Baselines`, `Compositional Probing`, `Prompt Ensembling`, `Vietnamese BPE`). |
| **2. Related Work** | `0,75` trang | Tổng quan ngắn gọn đặt nghiên cứu vào dòng chảy của Radford et al. (CLIP 2021), Paiss et al. (CountBench 2023), Yuksekgonul et al. (ARO 2023), Thrush et al. (Winoground 2022), và Alhamoud et al. (NegBench 2025). | **Bảng 1:** So sánh thiết kế thực nghiệm của nhóm với ARO, Winoground và CountBench (từ [`docs/p0_requirements_and_rqs.md`](p0_requirements_and_rqs.md)). |
| **3. Methodology & Reproducible Protocol** | `1,25` trang | Giao thức mô hình đơn cố định (`OpenAI CLIP ViT-B/32`, `FP32`); bộ đối chứng nhận diện đối tượng (`Control Captions`); thiết kế 2×2 Assignment khử bias cộng tính; phân rã `Prior-to-Signal Ratio (PSR)`; hai phương pháp Ensemble (`Text-Embedding` vs `Logit/Prob`). | **Bảng 2:** Tóm tắt sổ cái dữ liệu (`Data Exposure Ledger`) và đơn vị bootstrap độc lập cho từng bài toán. |
| **4. Core Baselines & Compositional Probing (`RQ1`)** | `1,50` trang | Kết quả 5 bộ dữ liệu chuẩn (`C01`), phân tích Fine-grained Oracle 5-way vs 100-way (`C04`), phân rã Counting theo $k = 1, \dots, 6$ và 2 chế độ diện tích (`C02`), phân tích Negation theo mẫu câu và tính nhất quán nhóm (`C03`). | **Bảng 3:** Bảng tổng hợp 5 Baselines & 4 Core Probes (`C01`–`C05`).<br>**Hình 2:** Ma trận nhầm lẫn Counting 6×6 & Biểu đồ độ nhạy Negation. |
| **5. Spatial Failure Mechanism, Confirmation & Zero-Shot Mitigation (`RQ2`)** | `1,25` trang | Kết quả Spatial tổng hợp (`27,36%`, blank agreement `23,82%`), COCO thăm dò (`54,00%`) và COCO xác nhận độc lập (`53,00%`, $p=0.3086$, `C05`, `C07`); kết quả phân rã `PSR` (`11,2x–16,7x`) và phương pháp `Half-Canvas Zero-Shot Scoring`. | **Bảng 4:** Phân rã Prior Bias $\lvert B \rvert$ vs Spatial Signal $\lvert \Delta_{\text{spatial}} \rvert$ và so sánh Global CLIP vs Half-Canvas Scoring.<br>**Hình 3:** Minh họa trực quan các trường hợp thành công và thất bại trên ảnh tổng hợp & COCO. |
| **6. Prompt Sensitivity, Ensembling & Vietnamese BPE Analysis (`RQ3`, `RQ4`)** | `1,25` trang | Thực nghiệm Category Prompt Controls (`64,48% -> 62,92%`, deranged `54,19%`, `C06`); tổng hợp Spread và Ensemble trên Negation, Spatial, CIFAR-100; chẩn đoán phân mảnh BPE tiếng Việt (`3,58x` subword tokens) và so sánh 4 điều kiện ngôn ngữ. | **Bảng 5:** Bảng tổng hợp Prompt Sensitivity (`Best`, `Worst`, `Mean ± SD`, `Spread`, `Ensemble`).<br>**Bảng 6:** Thống kê phân mảnh BPE Tiếng Việt (`en`, `vi_diacritics`, `vi_no_diacritics`, `vi_en_bilingual`). |
| **7. Threats to Validity, Limitations & Conclusion** | `0,50` trang | Thảo luận minh bạch 5 giới hạn phương pháp luận: (1) phạm vi một model `ViT-B/32`, (2) khoảng tin cậy rộng tại $N=100$ (`[43,29%, 62,49%]`), (3) không suy diễn $p > 0.05$ thành bằng chứng tương đương 50%, (4) giới hạn cắt tâm trên ảnh tự nhiên, (5) quy trình kiểm chứng tái lập mã nguồn. | — |

---

## 2. P5.2 — Kịch Bản Slide Trình Bày 15 Phút (12 Slides)

| Slide | Thời lượng | Tiêu đề Slide | Nội dung trọng tâm & Điểm nhấn thị giác |
| :---: | :---: | --- | --- |
| **1** | `0:45` | **Title & Overview** | Đề tài 14, thành viên nhóm, mục tiêu: Giải phẫu khả năng suy luận cấu trúc, độ nhạy prompt và tiếng Việt của CLIP ViT-B/32. |
| **2** | `1:00` | **Why Probe VLMs? (`RQ1`–`RQ4`)** | Đặt vấn đề: Độ chính xác phân loại ảnh đơn lẻ cao có đồng nghĩa với hiểu ngôn ngữ–thị giác hợp thành (Counting, Spatial, Negation) không? |
| **3** | `1:15` | **Strict Reproducible Pipeline & Data Ledger** | Kiến trúc đánh giá tất định (`FP32`, `SHA-256` manifest, tách biệt tuyệt đối tập Exploratory `200` nguồn và Confirmation `100` nguồn). |
| **4** | `1:15` | **Baseline Landscape across 5 Datasets (`C01`, `C04`)** | Từ `85,06%` (Oxford Pet) đến `42,45%` (DTD); đối chiếu CIFAR-100 Global 100-way (`64,48%`) vs Oracle 5-way (`77,95%`). |
| **5** | `1:15` | **Counting & Negation Breakdown (`C02`, `C03`)** | Counting: Nhận diện đúng màu+hình `74,44%` nhưng đếm đúng chỉ `32,87%` (thiên lệch dự đoán `3–5` tại $k=6$). Negation: Độ nhạy mẫu câu `91,39%` (`with/without`) vs `80,83%` (`contains`), Group Consistency chỉ `58,61%`. |
| **6** | `1:30` | **The Spatial Collapse: Synthetic to Natural (`C05`, `C07`)** | Nhận diện vật thể đạt `96,94%–97,50%`, nhưng 4-way Spatial đạt `27,36%–29,58%` (khớp ảnh trắng `23,82%`). Trên tập xác nhận COCO 100 nguồn mới: 2×2 Assignment đạt `53,00%` (`95% CI [43,29%, 62,49%]`, $p = 0.3086$). |
| **7** | `1:30` | **Mechanistic Diagnosis: Prior Bias vs Spatial Signal (`RQ2`)** | Công thức phân rã $d(I_0) = B + \Delta_{\text{spatial}}, d(I_1) = B - \Delta_{\text{spatial}}$. Chứng minh $\mathbb{E}[\lvert B \rvert]$ lớn gấp `11,2x–16,7x` $\mathbb{E}[\lvert \Delta_{\text{spatial}} \rvert]$, làm `both_images_correct` sụp đổ xuống `0,56%–4,00%`. |
| **8** | `1:15` | **Zero-Shot Region Mitigation & Its Natural Limit (`P2.2`)** | Phương pháp `Half-Canvas Zero-Shot Scoring`: Giải quyết 100% bài toán hình học có kiểm soát, nhưng chỉ rõ vì sao ảnh tự nhiên COCO bị giới hạn khi 2 vật thể nằm cùng một nửa khung hình (`7,0%–13,5%`). |
| **9** | `1:30` | **Prompt Sensitivity & Category Controls (`RQ3`, `C06`)** | Phép thử 8 điều kiện CIFAR-100: Thêm category thật làm giảm `-1,56` pp (`64,48% -> 62,92%`) do thu hẹp margin của lớp đúng trước đối thủ cùng nhóm, nhưng category sai làm tụt `-10,29` pp (`54,19%`). |
| **10** | `1:15` | **When Does Prompt Ensembling Work? (`P3.2–P3.4`)** | Bảng đối chiếu: Ensembling nâng phân loại chuẩn (`64,83%`) và thu hẹp khoảng cách phủ định (`86,81%`), nhưng **bất lực** trước lỗi quan hệ không gian (`54,00%` 2×2 assignment / `1,00%` both correct trên COCO confirmation). |
| **11** | `1:15` | **Vietnamese Cross-Lingual & BPE Tokenization Bottleneck (`RQ4`)** | Phân tích byte-level BPE: Tên lớp tiếng Việt có dấu bị bẻ thành `4,27` tokens (`3,58x` so với tiếng Anh `1,31` tokens); vai trò của lược dấu (`vi_no_diacritics`) và ghép song ngữ (`vi_en_bilingual`). |
| **12** | `1:00` | **Takeaways, Threats to Validity & Reproducibility** | Tổng kết 4 câu trả lời cho `RQ1`–`RQ4`, nhấn mạnh kỷ luật thống kê (không diễn giải $p > 0.05$ là bằng chứng bằng 50%) và hướng dẫn chạy kiểm chứng 1 lệnh. |

---

## 3. P5.3 — Danh Sách Lệnh Tái Lập & Kiểm Chứng Toàn Diện

Toàn bộ kết quả, bảng tổng hợp và kiểm định bất biến có thể tái lập bằng các lệnh sau:

```powershell
# 1. Chạy toàn bộ bộ kiểm thử đơn vị (bao gồm cả tests cho P1.2, P2, P3, P4)
python -m unittest discover -s tests -v

# 2. Sinh toàn bộ báo cáo tổng hợp offline (P1.2 blinded sheets, P2 PSR & geometry, P3 ensembles, P4 BPE)
python -m vlm_probe.synthesis

# 3. Kiểm tra lại toàn bộ các run lịch sử đã đóng băng (Baselines, Core Probing, Probing Depth, Confirmation)
python -m vlm_probe.verify results/full_cifar100_clip_b32_v1
python -m vlm_probe.probing.audit results/full_probing_clip_b32_v1
python -m vlm_probe.depth audit
python -m vlm_probe.depth.confirmation audit

# 4. (Tùy chọn GPU) Chạy thử nghiệm live cho P2 Mitigation, P3 Prompt Study và P4 Vietnamese Study
python -m vlm_probe.spatial_diagnostics mitigate --smoke --run-name smoke_spatial_mitigation_v1
python -m vlm_probe.prompt_study dataset --dataset cifar100 --smoke --run-name smoke_prompt_cifar100_v1
python -m vlm_probe.vietnamese_study run-cifar100 --smoke --run-name smoke_vi_cifar100_v1
```
