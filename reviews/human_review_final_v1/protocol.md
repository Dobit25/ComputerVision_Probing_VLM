# Giao thức tích hợp review người — 17/11/2026

## Nguồn và phạm vi

Người dùng xác nhận đã chốt phiếu ảnh và xác nhận đủ 100 bản dịch đúng nghĩa. Reviewer là R1; trợ lý chỉ chuẩn hóa định dạng và điền trạng thái bản dịch theo xác nhận rõ ràng của người dùng. Đây là một reviewer, có trao đổi với agent sau bản chấm ban đầu; không gọi là hai người chấm độc lập hay tính inter-annotator agreement.

Giữ nhãn `tv` theo annotation COCO, chấp nhận monitor được gán lớp này; giữ `dining table` cho đối tượng bàn đã được COCO gán nhãn, không loại chỉ vì không giống bàn ăn đời thường. Vẫn yêu cầu đúng box A/B, đủ nhận diện trong crop và quan hệ rõ. Không đổi nhãn/prompt cũ.

## Đóng băng

Lưu phiếu gốc nguyên byte, phiếu chuẩn UTF-8 BOM, checksum, các bản nháp đã có và lịch sử thảo luận tóm tắt trong `reviews/human_review_final_v1/`. Chỉ bỏ khoảng trắng đầu/cuối và cột không tên rỗng. Không bổ sung notes hay thay quyết định thay người. Phiếu ảnh hiện nhận 98, loại 006 (AMBIGUOUS_LABEL) và 092 (OCCLUDED_OR_TINY); phiếu dịch 100 accepted, tên lớp phải khớp run gốc.

Giữ nguyên lựa chọn nhận 040 của R1 dù agent đã nêu lo ngại nĩa bị crop. Dòng 092 có both_objects_visible=yes và class_names_correct=no, lý do che khuất; lưu nguyên cách ghi này, không suy diễn đó là nhãn người chắc chắn sai. Notes gốc trống vẫn được giữ nguyên. Các điểm này là giới hạn phân xử được công khai.

## Phân tích từ scores có sẵn

Không inference mới. Kiểm integrity các run nguồn trước tính toán. Chạy lại phép tính trên cả 100 và 98 source, không đổi thứ tự mẫu. Mỗi source giữ trọn original+mirror; không lọc theo scores. Kiểm bảng đầy đủ 100 source ghép đúng image ID/hash và retained mask.

- Primary: holistic T2/o0 và chia vùng native_half, hai đáp án trái/phải, cùng ảnh. Secondary: padded_half và đủ sáu điều kiện holistic cũ.
- Đo top-1 theo ảnh, cả hai ảnh đúng theo source, ghép cặp 2×2, confusion, ties và chênh lệch region−holistic.
- CI 95% bootstrap 2.000 lượt theo source, seed 1402 (config gốc); paired delta giữ cùng source. Báo Wilson/binomial như run gốc cho holistic 2×2. Đây là CI riêng từng phép đo, không hiệu chỉnh nhiều kiểm định.
- Báo mẫu số source và ảnh riêng. Kết quả trước lọc phải tái lập metrics gốc. Tạo predictions/pairs mới theo subset, ghi cả ảnh bị loại và ảnh lật của chúng.
- Kết quả sau lọc là sensitivity hậu nghiệm trên corpus đã mở, không phải holdout mới. Không suy ra model encoder được cải thiện từ phương pháp chia vùng có thêm compute/quy tắc hình học.

## Bản dịch và trạng thái

Không sửa từ điển nên chỉ liên kết phiếu accepted với hash từ điển và metrics VI đã audit; không chạy lại model, không sửa kết quả VI. Review này xác nhận nghĩa theo một người dùng, không phải hội đồng chuyên gia hoặc chứng minh mọi thuật ngữ đều không mơ hồ.

Tạo báo cáo Markdown/CSV/JSON mới và cập nhật trạng thái hiện hành. PDF và slides lịch sử không chỉnh sửa ở mốc này; chỗ còn ghi pending trong artifact cũ là trạng thái tại thời điểm tạo, được chú thích ở trang chỉ dẫn hiện hành.

Tái lập: `python -m vlm_probe.human_review_followup freeze`, sau đó `python -m vlm_probe.human_review_followup run`; kiểm lại bằng `python -m vlm_probe.human_review_followup audit`. Freeze/run không ghi đè thư mục đã tồn tại.
