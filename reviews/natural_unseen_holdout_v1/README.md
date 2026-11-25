# Hướng dẫn chấm mù kép (Dual Blind Review) — Set C (160 nguồn COCO mới)

1. **Mục đích & Nguyên tắc mù:** Hai thành viên độc lập (**Slot RA** và **Slot RB**) tự chấm riêng rẽ trên cùng 160 nguồn ảnh COCO mới hoàn toàn (`UNSEEN_C_001` .. `UNSEEN_C_160`).
   - Hai người chấm **không trao đổi kết quả với nhau** trước khi điều phối viên chạy phân xử (`adjudicate`).
   - Không có bất kỳ điểm số CLIP nào được sinh ra trước khi phiếu được khóa (`freeze`).
2. **Cách xem ảnh:** Giải nén `reviewer_package_set_c.zip` và mở `index.html`. Mỗi mục hiển thị ảnh nguồn có box `A` (xanh dương)/`B` (cam) bên trái và **đúng crop 224×224 của CLIP** bên phải. Phải quyết định dựa trên vùng crop bên phải.
3. **Điền đúng file của mình (`unseen_c_blank_RA.csv` hoặc `unseen_c_blank_RB.csv`):**
   - Ghi mã người chấm vào `annotator_id` (hai người phải dùng hai mã khác nhau, ví dụ tên riêng không dấu).
   - Giữ nguyên 160 dòng và 4 cột đầu (`blind_id`, `object_a`, `object_b`, `expected_relation`).
   - Điền `yes`/`no` cho `both_objects_visible`, `class_names_correct`, `relation_clear`; điền `left`/`right`/`above`/`below`/`unclear` cho `perceived_relation`.
   - `overall_accept = yes` chỉ khi cả 3 câu đầu đều `yes` và `perceived_relation == expected_relation`. Nếu `no`, ghi rõ `exclusion_reason`.
4. Lưu file CSV UTF-8 và gửi lại để tính IAA, phân xử bất đồng (nếu có) và đóng băng trước khi chạy GPU 1 lượt duy nhất.
