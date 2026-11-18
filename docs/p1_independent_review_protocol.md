# P1.2: Giao thức kiểm nhãn ảnh tự nhiên độc lập bởi người (Independent Human Review Protocol)

Ngày lập: **09/11/2026** (Asia/Saigon).
Tài liệu này quy định quy trình thực hiện **P1.2** trong [long_term_project_plan.md](long_term_project_plan.md#5-p1--ho?n-thi?n-ch?t-l??ng-b?ng-ch?ng-core) nhằm kiểm tra độ nhạy của kết quả không gian trên ảnh tự nhiên đối với chất lượng nhãn, đồng thời quy định phương án dự phòng nếu không có thành viên rà soát độc lập.

---

## 1. Mục tiêu và Phạm vi dữ liệu

Trong mốc `v1.1.0-probing-depth`, việc lọc hình học từ annotation COCO `val2014` đã được kết hợp với rà soát trực quan bởi **một trợ lý AI** (ẩn điểm số CLIP từng ảnh). Để kiểm tra xem các quyết định nhận/loại hoặc sự mơ hồ của ảnh tự nhiên có làm thay đổi kết luận hay không, nhóm có thể thực hiện **kiểm nhãn độc lập bởi người (human review)** theo thứ tự ưu tiên:

1. **Ưu tiên 1 (Tập xác nhận mới — 100 nguồn ngang):**
   - Nguồn dữ liệu: `results/natural_confirmation_selection_v1/selected.json` (100 nguồn ảnh COCO `val2014` có quan hệ ngang `left`/`right`).
   - Contact sheets trực quan (sau khi chạy `python -m vlm_probe.depth.confirmation prepare-new`): `data/natural_confirmation_v1/review_sheets/new_*.png`.
2. **Ưu tiên 2 (Tập thăm dò cũ — 200 nguồn 4 quan hệ):**
   - Nguồn dữ liệu: `results/depth_natural_selection_v1/selected.json` (200 nguồn ảnh COCO `val2014` cân bằng 50 ảnh mỗi quan hệ `left`/`right`/`above`/`below`).
   - Contact sheets trực quan (sau khi chạy `python -m vlm_probe.depth.confirmation prepare-old`): `data/natural_confirmation_v1/review_sheets/old_*.png`.

> **Lưu ý phương pháp luận:** Tập 100 nguồn mới và 200 nguồn cũ **đã chạy suy luận CLIP và đã mở kết quả**. Do đó, việc người rà soát lại nhãn ở giai đoạn P1.2 là **kiểm tra chất lượng nhãn và phân tích độ nhạy (label validation & sensitivity analysis)** — không biến các tập này thành holdout chưa mở một lần nữa.

---

## 2. Nguyên tắc che thông tin (Blinding Rules)

Người review (`Reviewer 1`, `Reviewer 2`) được cung cấp:
- Hình ảnh vùng **center crop vuông** thực tế mà CLIP nhìn thấy, có vẽ hai khung chữ nhật `A` (xanh dương `#0072B2`) và `B` (cam `#E69F00`).
- Tên lớp COCO của vật thể `A` (`category_a`), vật thể `B` (`category_b`) và nhãn quan hệ hình học dự kiến (`LEFT`, `RIGHT`, `ABOVE`, `BELOW`).

Người review **TUYỆT ĐỐI KHÔNG** được xem khi đang chấm:
- Điểm cosine (`scores.npz`) hoặc nhãn dự đoán (`predictions.csv`, `pairs.csv`) của CLIP.
- Quyết định nhận/loại hoặc lý do loại trước đó của trợ lý AI (`old_review.json`, `new_review.json`).
- Quyết định của người review kia (trước bước phân xử).

---

## 3. Tiêu chí đánh giá cho từng ảnh nguồn

Với mỗi ảnh nguồn, người review chọn một trong hai trạng thái (`status`: `accepted` hoặc `rejected`) và nếu chọn `rejected` thì ghi mã lý do chuẩn (`reason_code`):

1. **`accepted`:**
   - Cả hai đối tượng `A` và `B` đều nhìn thấy rõ trong vùng center crop vuông.
   - Tên lớp `A` và `B` phù hợp với cách hiểu thông thường của vật thể trong ảnh (không phải hình in trên áo/bao bì/màn hình gây nhầm lẫn).
   - Không có vật thể thứ hai cùng loại `A` hoặc `B` xuất hiện rõ trong vùng crop làm câu mô tả `an image of a {A} {relation} a {B}.` trở nên mơ hồ.
   - Quan hệ không gian 2D giữa `A` và `B` trong bức ảnh đúng với nhãn hiển thị.
2. **`rejected` (kèm một mã lý do chính):**
   - `OCCLUDED_OR_TINY`: Một trong hai đối tượng bị che khuất nặng, quá mờ hoặc bị cắt mất phần chính trong center crop.
   - `MULTIPLE_INSTANCES`: Có thêm đối tượng cùng loại `A` hoặc `B` nhìn thấy rõ trong vùng crop khiến quan hệ không còn duy nhất.
   - `AMBIGUOUS_LABEL`: Nhãn COCO không phản ánh đúng vật thể thực tế hoặc là tranh vẽ/hình in/phản chiếu trong gương.
   - `AMBIGUOUS_RELATION`: Hai vật thể nằm chéo hoặc bao chứa nhau khiến người nhìn không đồng tình với nhãn `LEFT`/`RIGHT`/`ABOVE`/`BELOW`.

---

## 4. Bốn nhánh xử lý kết quả theo Kế hoạch (Mục 5.2)

- **Nhánh A (Đồng thuận cao, ít thay đổi so với tập đã chọn):**
  - Báo cáo tỷ lệ đồng thuận (% agreement và Cohen's $\kappa$ nếu có đủ 2 người review độc lập), giữ nguyên số liệu chính trên đủ 100 nguồn và báo thêm kết quả trên tập con được người xác nhận đồng thuận.
- **Nhánh B (Nhiều bất đồng giữa người review):**
  - Thống kê tần suất từng nguyên nhân (`MULTIPLE_INSTANCES`, `AMBIGUOUS_LABEL`, `AMBIGUOUS_RELATION`, `OCCLUDED_OR_TINY`) để bàn luận về giới hạn của việc dùng bounding box COCO làm benchmark quan hệ không gian. Không sửa artifact cũ.
- **Nhánh C (Không có đủ nhân lực thực hiện review độc lập bởi người):**
  - **Phương án mặc định an toàn:** Giữ nguyên kết quả tự rà soát của trợ lý AI (`176/200` tập cũ, `100/178` tập mới) và ghi rõ trong mục Limitations của báo cáo cuối kỳ: *"Rà soát trực quan được thực hiện bởi cùng một trợ lý AI kết hợp bộ kiểm tra hình học tự động trước suy luận; chưa có người gán nhãn độc lập hoặc hệ số đồng thuận liên người chấm (IAA)."* Tuyệt đối không chạy lại cùng AI rồi gọi đó là reviewer độc lập thứ hai.
- **Nhánh D (Nhãn sau khi người review làm thay đổi đáng kể điểm ước lượng):**
  - Công bố minh bạch cả hai con số (trên tập 100 nguồn đã đóng băng và trên tập con sau khi người lọc) và kết luận rằng hiệu năng đo được nhạy với tiêu chí lọc nhãn.

