# Hướng dẫn chấm độc lập lần 2 (R2) — Set B (100 nguồn COCO)

1. **Nguyên tắc độc lập & mù hoàn toàn:** Người chấm **R2 bắt buộc phải là thành viên khác với R1** (người đã chấm vòng trước), chưa từng xem quyết định của R1 và chưa xem bất kỳ điểm số/dự đoán nào của CLIP. Ghi mã người chấm vào cột `annotator_id` là `R2` (hoặc tên/mã riêng khác `R1`).
2. **Cách xem ảnh:** Giải nén gói `reviewer_r2_package.zip` vào thư mục riêng và mở `index.html`. Mỗi mục `CONFIRMATION_100_xxx` hiển thị:
   - Bên trái: ảnh nguồn với khung `A` (xanh dương) và `B` (cam).
   - Bên phải: **đúng vùng crop 224×224** mà CLIP nhìn thấy (phóng to 1,5 lần để dễ quan sát).
   - **Quyết định phải dựa trên vùng crop 224×224**; ảnh nguồn chỉ giúp nhận diện vật bị cắt biên.
3. **Điền phiếu `confirmation_100_R2_blank.csv` (UTF-8):**
   - Giữ nguyên toàn bộ 100 dòng và các cột nguồn (`blind_id`, `object_a`, `object_b`, `expected_relation`).
   - Điền `yes` / `no` cho `both_objects_visible`, `class_names_correct`, `relation_clear`.
   - Điền `perceived_relation` bằng đúng một trong các giá trị: `left`, `right`, `above`, `below`, `unclear` (hướng của A so với B trên mặt phẳng ảnh).
   - `overall_accept` chỉ bằng `yes` khi cả 3 câu hỏi đầu là `yes` **và** `perceived_relation == expected_relation`. Nếu `no`, bắt buộc ghi `exclusion_reason` (`OCCLUDED_OR_TINY`, `MULTIPLE_INSTANCES`, `AMBIGUOUS_LABEL`, `AMBIGUOUS_RELATION`, hoặc `OTHER`) và mô tả ngắn ở `notes`.
4. Lưu file hoàn chỉnh dạng CSV UTF-8 (ví dụ `confirmation_100_R2.csv`) và gửi lại điều phối viên để chạy `validate`, `freeze`, `run` và `audit`.
