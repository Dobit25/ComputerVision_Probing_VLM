# Báo cáo kiểm chứng độ đồng thuận giữa hai người chấm (IAA) — Set B (`human_review_iaa_set_b_v1`)

## 1. Phạm vi và quy trình giữ mù

Nhằm đáp ứng góp ý ngày 11/11/2026 về kiểm chứng nhãn độc lập cho tập ảnh COCO xác nhận (**Set B**, 100 nguồn), một người chấm thứ hai độc lập (**R2**, khác người chấm **R1**) đã thực hiện chấm mù hoàn toàn trên chính 100 nguồn ảnh qua giao diện tương tác offline (`annotate_r2.html`) được tạo từ [gói R2 mù](../../reviews/human_review_iaa_set_b_v1/package_manifest.json).

- **Điều kiện giữ mù:** R2 không nhìn thấy quyết định của R1, không nhìn thấy lý do loại trước đó, và không nhìn thấy bất kỳ điểm số hay dự đoán nào của CLIP. Quyết định dựa trên vùng crop `224×224` thực tế mà CLIP nhận.
- **Đóng băng & Không suy luận mới:**
  - Phiếu R1 được đóng băng ngày `2026-11-16` tại [reviews/human_review_final_v1/freeze.json](../../reviews/human_review_final_v1/freeze.json) (98/100 nhận).
  - Phiếu R2 được kiểm tra schema (`validate`), chuẩn hóa và đóng băng vào lúc **`2026-11-25T16:30:34.098588+00:00`** tại [reviews/human_review_iaa_set_b_v1/freeze.json](../../reviews/human_review_iaa_set_b_v1/freeze.json) (100/100 nhận).
  - Toàn bộ phân tích IAA và độ nhạy tập đồng thuận kép (`R1 ∩ R2`) được tính lại hoàn toàn từ điểm số đã đóng băng của `full_natural_confirmation_v1` và `full_natural_regions_confirmation_review_v2` (**không chạy inference mới**).

---

## 2. Kết quả độ đồng thuận giữa R1 và R2 (100 nguồn Set B)

| Tiêu chí đánh giá | Đồng thuận thô | Kỳ vọng ngẫu nhiên ($p_e$) | Cohen's $\kappa$ | Phân bố R1 / R2 |
|---|---:|---:|---:|---|
| **`overall_accept`** | **98 / 100 (98,00%)** | 98,00% | **0,0000** | R1: 98 yes, 2 no / R2: 100 yes |
| `both_objects_visible` | 99 / 100 (99,00%) | 99,00% | 0,0000 | R1: 99 yes, 1 no / R2: 100 yes |
| `class_names_correct` | 98 / 100 (98,00%) | 98,00% | 0,0000 | R1: 98 yes, 2 no / R2: 100 yes |
| `relation_clear` | 100 / 100 (100,00%) | 100,00% | *không xác định* | R1: 100 yes / R2: 100 yes |
| **`perceived_relation`** (`left`/`right`) | **100 / 100 (100,00%)** | 52,42% | **1,0000** | R1: 61 `left`, 39 `right` / R2: 61 `left`, 39 `right` |

### Diễn giải phương pháp về nghịch lý hệ số $\kappa$ khi mất cân bằng biên (Prevalence Paradox)
Như đã cảnh báo trước trong [hướng dẫn chấm](../../docs/human_review_assignment_v2.md), Set B là tập **đã qua sàng lọc hình học chặt chẽ trước khi xuất phiếu**, nên tỷ lệ mẫu hợp lệ rất cao. Khi R2 chấp nhận toàn bộ `100/100` (`marginal_b = {"yes": 100}`), xác suất đồng thuận kỳ vọng ngẫu nhiên cho `overall_accept` trở thành $p_e = 0,98 \times 1,00 + 0,02 \times 0,00 = 0,98$, bằng đúng tỷ lệ đồng thuận thực tế $p_o = 0,98$, khiến công thức Cohen cho $\kappa = \frac{p_o - p_e}{1 - p_e} = 0,0000$ (và không xác định ở `relation_clear` khi mẫu số $1 - p_e = 0$) mặc dù hai người đồng ý **98/100** quyết định cuối và đạt **$\kappa = 1,0000$** tuyệt đối trên `perceived_relation` (`61 left / 39 right`). Do đó, báo cáo trình bày đồng thời cả tỷ lệ đồng thuận thô, biên phân bố và Cohen's $\kappa$ đúng nguyên tắc minh bạch thống kê.

---

## 3. Chi tiết các ca bất đồng và đối chiếu ca nhạy cảm

Có đúng **2/100** nguồn bất đồng ở `overall_accept` (chi tiết trong [disagreements.csv](../../results/human_review_iaa_set_b_v1/disagreements.csv)):

| Blind ID | COCO ID | Cặp vật & Quan hệ kỳ vọng | Quyết định R1 | Quyết định R2 | Các trường khác nhau |
|---|---:|---|---|---|---|
| `CONFIRMATION_100_006` | `434930` | `donut` / `chair` (`left`) | **no** (`AMBIGUOUS_LABEL`, `class_names_correct=no`) | **yes** (`left`) | `overall_accept; class_names_correct` |
| `CONFIRMATION_100_092` | `365208` | `person` / `chair` (`right`) | **no** (`OCCLUDED_OR_TINY`, `both_objects_visible=no`, `class_names_correct=no`) | **yes** (`right`) | `overall_accept; both_objects_visible; class_names_correct` |

- **Đối chiếu ca `CONFIRMATION_100_040` (`fork` / `knife`, `left`):** Ca từng gây tranh luận về việc nĩa bị cắt ở mép crop trong vòng R1 đã được **cả R1 và R2 đều độc lập đánh giá `overall_accept = yes`** (`perceived_relation = left`).

---

## 4. Phân tích độ nhạy trên tập đồng thuận kép (`R1 ∩ R2`)

Vì toàn bộ 98 nguồn được R1 chấp nhận đều được R2 chấp nhận, tập đồng thuận kép **`R1 ∩ R2` gồm đúng 98 nguồn (196 ảnh)**, trùng khít với tập `after_98` trong [báo cáo follow-up R1](../human_review_followup_v1/README.md), trong khi tập hợp **`R1 ∪ R2` gồm toàn bộ 100 nguồn (200 ảnh)**. Không có bất kỳ sự đổi chiều kết luận nào giữa hai tập:

| Tập nguồn | Phương pháp | Top-1 (%) | CI 95% Top-1 (%) | Cả 2 ảnh đúng (%) | Ghép cặp 2×2 (%) | $\Delta$ vs Holistic (pp) |
|---|---|---:|---|---:|---:|---:|
| **`R1 ∩ R2` (98 nguồn / 196 ảnh)** | Holistic T2/o0 | **51,53** (`101/196`) | [49,49; 53,57] | 4,08 | **53,06** (`52/98`, Wilson [43,25; 62,64], $p=0,3069$) | — |
| **`R1 ∩ R2` (98 nguồn / 196 ảnh)** | Native half (primary) | **93,37** (`183/196`) | [88,27; 97,45] | **91,84** | **92,86** (`91/98`, CI [87,73; 97,96]) | **+41,84** [36,73; 46,43] |
| **`R1 ∩ R2` (98 nguồn / 196 ảnh)** | Padded half (secondary) | **92,86** (`182/196`) | [88,27; 96,94] | 89,80 | **90,82** (`89/98`, CI [84,69; 95,92]) | **+41,33** [36,22; 45,92] |
| `R1 ∪ R2` (100 nguồn / 200 ảnh) | Holistic T2/o0 | 51,50 (`103/200`) | [49,50; 54,00] | 4,00 | 53,00 [43,00; 63,00] | — |
| `R1 ∪ R2` (100 nguồn / 200 ảnh) | Native half (primary) | 92,50 (`185/200`) | [87,50; 97,00] | 91,00 | 92,00 [86,00; 97,00] | +41,00 [36,00; 45,50] |

---

## 5. Giới hạn và Tái lập

- **Giới hạn:** Tập 100 nguồn Set B là tập đã được mở kết quả trước khi chia vùng, và độ đồng thuận $\kappa = 0,0000$ ở `overall_accept` phản ánh sự đồng nhất biên rất cao (`100/100` vs `98/100`) chứ không thay thế được tập kiểm chứng độc lập hoàn toàn mới (nhiệm vụ của **Set C** trong WS2).
- **Artifact & Tái lập offline (không cần GPU hay ảnh gốc):**
  - Phiếu R2 đóng băng: [reviews/human_review_iaa_set_b_v1/confirmation_100_R2.csv](../../reviews/human_review_iaa_set_b_v1/confirmation_100_R2.csv)
  - Kết quả & trace: `results/human_review_iaa_set_b_v1/` (`metrics.json`, `disagreements.csv`, `r1_r2_membership.csv`, `manifest.json`)
  - Lệnh kiểm toán:

```powershell
python -m vlm_probe.human_review_iaa audit
```

