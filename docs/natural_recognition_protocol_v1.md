# Protocol T1 — đối chứng nhận dạng trên ảnh COCO (`natural_recognition_v1`)

**Trạng thái: ĐÃ CHỐT (18/11/2026), trước khi có bất kỳ score T1 nào.** Soạn theo mục 4–6 của [1118.md](../1118.md). Nhóm đã xác nhận mọi mục **[ĐÃ CHỐT]** bên dưới sau hai vòng review foil; `configs/natural_recognition_v1.json` có `"status": "frozen"`. Mọi thay đổi sau thời điểm này cần một phiên bản config/protocol mới. Code từ chối chạy model khi config chưa `frozen` hoặc phiếu foil chưa đóng băng.

Phạm vi: một checkpoint OpenAI CLIP ViT-B/32 (`configs/clip_b32_cifar100.json`), không fine-tuning, không thêm model hay dataset.

## 1. Câu hỏi nghiên cứu

Trên cùng vùng ảnh COCO mà holistic T2/o0 đã dùng, CLIP có phân biệt được caption nêu **đúng hai đối tượng** với caption **thay sai một đối tượng** không?

- Nếu nhận dạng tốt mà quan hệ (holistic 2-way ≈ 51,5%) yếu, bằng chứng về khó khăn **kết hợp quan hệ** mạnh hơn hiện tại.
- Không mặc định kết luận “CLIP không hiểu không gian”. Kết quả có thể chỉ ra lỗi nhận dạng, dữ liệu mơ hồ hoặc hiệu ứng thiết kế.

Đây là follow-up **hậu nghiệm** trên tập đã mở (scores quan hệ đã biết), không phải xác nhận ngoài mẫu.

## 2. Cohort và đơn vị

| Mục | Giá trị |
|---|---|
| Nguồn | 98 nguồn R1 nhận (`source_membership.csv`, `accepted=True`) trong `natural_confirmation_selection_v1/selected.json` |
| Ảnh | original + `horizontal_flip` → 196 ảnh |
| Vai trò | `replace_a`, `replace_b` |
| Trial | 98 × 2 ảnh × 2 vai trò = **392 trial** |
| Đơn vị độc lập | **98 nguồn**; bootstrap resample theo nguồn, giữ mọi trial của nguồn |
| Preprocessing | Giống hệt holistic: `adapter.preprocess` của checkpoint (resize cạnh ngắn bicubic + centre crop 224). Dùng lại đúng *identity* cache đặc trưng ảnh của `full_natural_confirmation_v1` |

Nguồn thiếu foil hợp lệ ở một vai trò **không bị bỏ âm thầm**: ghi lý do, báo coverage so với 98. **[ĐÃ CHỐT]** phân tích chính chỉ gồm nguồn có foil được chấp nhận ở cả hai vai trò; nếu dưới **80** nguồn thì không chạy inference và báo T1 chưa khả thi. Sau review: 98/98 nguồn có foil được chấp nhận ở cả hai vai trò.

## 3. Caption

Khung trung tính, không có quan hệ: `a photo showing {a} and {b}.`

- Đúng: `a photo showing {A} and {B}.` (A trước, như T2/o0)
- Foil thay A: `a photo showing {F_A} and {B}.`
- Foil thay B: `a photo showing {A} and {F_B}.`

Tên lớp giữ nguyên tên COCO (ví dụ `dining table`, `tv`), mạo từ a/an theo `natural.article`, áp dụng y hệt cho caption đúng và foil. Không thêm vị trí, thuộc tính hay bối cảnh. Thứ tự hiển thị đúng/foil trong file trial xen kẽ theo hash để cân bằng (điểm cosine không phụ thuộc thứ tự; ghi lại để audit).

## 4. Chọn foil (không dùng CLIP scores)

1. Pool: 80 lớp COCO trong `instances_val2014.json` đã đóng băng (sha256 trong `natural_confirmation_selection_v1/preparation.json`).
2. Loại: A và B; **mọi lớp có annotation ở bất kỳ đâu trong ảnh nguồn** (chặt hơn chỉ trong crop); các lớp nằm trong `contested_pairs` cùng vật bị thay (gần đồng nghĩa, kiểu con, bộ phận–toàn thể). **[ĐÃ CHỐT]** danh sách trong config, giữ nguyên như khi sinh foil vòng 1.
3. Tầng 1: cùng supercategory COCO với vật bị thay (foil hợp lý, khó hơn). Tầng 2: lớp hợp lệ khác, chỉ khi tầng 1 rỗng (ví dụ `person` là supercategory một lớp).
4. Thứ tự trong tầng: `sha256("{seed}/{source_id}/{role}/{name}")` tăng dần, seed 1118. Hoàn toàn xác định, không nhìn scores.
5. `revise` → ứng viên kế tiếp trong danh sách, ở vòng review mới; tối đa 3 phiên bản mỗi nguồn/vai trò. Mọi phiên bản được giữ lại.

Giới hạn đã biết: tầng 2 có thể quá dễ; kết quả báo riêng theo tầng. Không có COCO bbox **không** chứng minh vật vắng mặt, nên bước người xác nhận là bắt buộc.

## 5. Review người (Task D)

- Reviewer **[ĐÃ CHỐT]**: `tuan_dat`, một người chấm toàn bộ (vòng 1: 196 dòng; vòng 2: 13 dòng gồm 5 foil mới và 8 phân xử). Không có inter-annotator agreement. Agent không ký thay.
- Mỗi dòng phiếu: `review_id, source_image_id, crop_sha256, role, replaced_object, kept_object, foil, foil_tier, candidate_version, reviewer_id, foil_absent, unambiguous, decision, reason, notes`.
- Reviewer xem **đúng vùng 224×224 mô hình thấy** (centre canvas), có khung A/B. Bản lật là ảnh gương của cùng vùng nên một quyết định áp dụng cho cả hai biến thể.
- `foil_absent` (yes/no/unsure): foil không xuất hiện trong vùng ảnh. `unambiguous` (yes/no/unsure): người xem không thể gọi vật bị thay bằng tên foil.
- `decision`: `accepted` (bắt buộc foil_absent=yes và unambiguous=yes), `revise` (cần lý do), `uncertain` (cần lý do; đưa ra phân xử).
- Không hiển thị scores hay dự đoán. Phiếu đã điền được đóng băng với checksum; không xuất lại phiếu trắng đè lên phiếu đã điền.

## 6. Metric

- **Primary:** recognition accuracy = tỷ lệ trial có `s(đúng) > s(foil)` trên toàn cohort phân tích; CI 95% bootstrap theo nguồn, 2.000 lượt, seed 1402.
- Hòa tuyệt đối tính là sai và báo riêng; |gap| ≤ 1e-6 báo là near tie.
- Phụ: theo vai trò, theo biến thể, theo tầng foil; `both_roles_correct` theo ảnh; `all_four_correct` theo nguồn.
- Đối chiếu mô tả với holistic T2/o0 2-way trên **cùng ảnh**: bảng chéo (nhận dạng cả hai vai trò đúng × quan hệ đúng) và tỷ lệ quan hệ đúng trong nhóm nhận dạng đúng. Đây là subgroup chọn theo hành vi model: chỉ mô tả, không thay kết quả toàn tập, không phải hiệu ứng nhân quả.
- **Không** lấy hiệu giữa accuracy nhận dạng 2-way và quan hệ (2-way hay 4-way) làm “năng lực”: caption khác nhau là hai thiết kế tác vụ.

## 7. Diễn giải

**[ĐÃ CHỐT trước khi có scores]** Ngưỡng (nhóm chọn, không từ phân tích power; mức ngẫu nhiên là 50%): nhận dạng “rõ ràng tốt” nếu cận dưới CI 95% ≥ 80%; “yếu” nếu cận trên < 70%; còn lại “chưa rõ”. Bảng kết luận dùng nguyên mục 5 của 1118.md:

| Kết quả | Kết luận được hỗ trợ |
|---|---|
| Nhận dạng tốt, quan hệ yếu | Điểm yếu kết hợp quan hệ trong thiết kế này; chưa suy ra encoder/head gây lỗi |
| Cả hai yếu | Chưa tách được nhận dạng và quan hệ |
| Cả hai tốt | Hiện tượng phụ thuộc cohort/caption/control |
| Nhận dạng tốt nhưng chủ yếu do foil tầng 2 | Control chỉ chứng minh năng lực ở độ khó đã chọn |
| CI rộng hoặc A/B trái chiều | Chưa rõ; không lấy thêm mẫu tới khi p nhỏ |
| Thiếu review foil tin cậy | Chưa đủ dữ liệu T1; hoãn inference |

Quy tắc dừng **[ĐÃ CHỐT]**: một lượt chính và tối đa một follow-up có protocol riêng; không chỉnh foil sau khi thấy scores.

Ngân sách **[ĐÃ CHỐT]**: một smoke run và một full run với checkpoint CLIP ViT-B/32 hiện có, tối đa 1 GPU-giờ, không fine-tuning.

## 8. Quy trình và lệnh

```powershell
python -m vlm_probe.spatial_diagnostics.natural_recognition prepare        # Task C: foil + gói review, không scores
python -m vlm_probe.spatial_diagnostics.natural_recognition validate --review reviews/natural_recognition_v1/round_01_filled.csv
python -m vlm_probe.spatial_diagnostics.natural_recognition freeze
python -m vlm_probe.spatial_diagnostics.natural_recognition run --smoke --run-name smoke_natural_recognition_v1
python -m vlm_probe.spatial_diagnostics.natural_recognition run --run-name full_natural_recognition_v1
python -m vlm_probe.spatial_diagnostics.natural_recognition audit --run-name full_natural_recognition_v1
```

`prepare` cần annotation COCO val2014 và 98 ảnh nguồn (không đưa vào Git; kiểm sha256 với bản đã đóng băng). `run` cần CUDA, `config.status == "frozen"` và review đã đóng băng; trước khi tính recognition, nó tái tạo điểm holistic T2/o0 từ cùng đặc trưng ảnh và yêu cầu khớp `full_natural_confirmation_v1/scores.npz`. `audit` chạy không cần pixels/checkpoint: tính lại predictions, metrics và CI từ `scores.npz`.

## 9. Lưu trữ

Kết quả: `results/{smoke,full}_natural_recognition_v1/` (config, protocol, manifest có hash input/code/checkpoint, trials, scores, predictions, pairs, metrics). Báo cáo: `reports/natural_recognition_v1/`. Review: `reviews/natural_recognition_v1/` (phiếu từng vòng, lịch sử, `freeze.json`). Kết quả âm hoặc không rõ đều được giữ và báo.
