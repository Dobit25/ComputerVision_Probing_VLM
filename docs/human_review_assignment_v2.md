# Giao việc cho thành viên review ảnh và bản dịch

Mục tiêu: kiểm 100 nguồn ảnh COCO và 100 tên lớp Việt. Hai đầu việc chưa được đánh dấu hoàn thành cho đến khi có phiếu do người chấm trả lại. Có thể giao cùng một người; hai người độc lập tốt hơn nếu nguồn lực cho phép.

## Người điều phối chuẩn bị

Trong conda gen, chạy `python -m vlm_probe.human_review export`. Gói local `data/human_review_v2/reviewer_package.zip` có HTML xem offline, 100 ảnh nguồn có box A/B, 100 crop thật, hai CSV trắng, hướng dẫn và nguồn/giấy phép. Ảnh không đưa vào Git. Chỉ gửi gói này cho reviewer, không gửi `coordinator_private`, reports/scores/predictions hoặc phiếu AI. Mã reviewer có thể dùng R1/R2 để hạn chế dữ liệu cá nhân.

Đề xuất giao việc: một người đọc hướng dẫn và chấm 100 ảnh (dự trù 2–4 giờ, nghỉ giữa các nhóm 20 ảnh); một người kiểm 100 nhãn EN–VI (2–3 giờ, thêm thời gian tra tên sinh học). Đây là dự trù, không phải deadline bắt buộc. Nếu hai người chấm, gửi bản trắng riêng và không trao đổi quyết định trước phân xử.

## Chấm ảnh

1. Giải nén vào thư mục riêng; mở `index.html` bằng trình duyệt. Mỗi ID có ảnh nguồn với box xanh A/cam B và crop 224×224 mà baseline CLIP nhìn thấy. Phải quyết định dựa trên crop; ảnh nguồn giúp xác định đối tượng nhưng không bù lại đối tượng đã mất khỏi crop.
2. Mở `confirmation_100_blank.csv` bằng Excel nhập UTF-8 hoặc trình soạn CSV. Giữ nguyên 100 dòng, ID, tên đối tượng và expected_relation. Điền annotator_id cho từng dòng.
3. Điền `yes/no` ở both_objects_visible, class_names_correct, relation_clear. Ghi quan hệ tự nhận thấy vào perceived_relation: left/right/above/below/unclear. Hướng là A so với B trên mặt phẳng ảnh, không phải trái/phải theo cơ thể đối tượng.
4. overall_accept=yes chỉ khi cả ba câu hỏi yes và perceived_relation khớp expected_relation. Nếu không, ghi no và exclusion_reason: OCCLUDED_OR_TINY, MULTIPLE_INSTANCES, AMBIGUOUS_LABEL, AMBIGUOUS_RELATION, hoặc OTHER kèm notes.
5. Không nhận ảnh vì muốn tăng/giảm accuracy. Nếu không xác định được đối tượng hoặc quan hệ, chọn no và mô tả lý do. Không sửa nhãn thật để khớp kỳ vọng.
6. Lưu file UTF-8 tên `confirmation_100_R1.csv`. Phiếu phải đủ 100 dòng; không xóa mẫu bị loại. Các trường chưa chấm giữ trống và chưa được tính là completed.

## Kiểm bản dịch

Mở `vietnamese_100_blank.csv`. Tên Anh và bản Việt hiện tại lấy đúng từ run đã hoàn thành. Kiểm bản Việt có biểu thị cùng lớp thị giác: loài/đồ vật cụ thể, mức độ bao quát, từ mượn và tên gần nhau. Ví dụ tên chung quá rộng so với tên loài cụ thể cần đánh dấu để bàn luận; không nhìn accuracy để chọn từ đồng nghĩa.

Điền annotator_id, status=accepted/needs_revision/uncertain. Nếu needs_revision: điền proposed_vi_label, reason và nguồn reference nếu đã tra. Nếu uncertain: ghi điều chưa rõ, không chọn đại. Giữ vi_label_current nguyên; sửa đề xuất nằm ở cột riêng. Lưu UTF-8 `vietnamese_100_R1.csv`. Không dùng auto-translate rồi ký như đã review từng mục.

## Nhận lại và phân xử

Người điều phối chạy:

```powershell
python -m vlm_probe.human_review validate path/to/confirmation_100_R1.csv --kind images
python -m vlm_probe.human_review validate path/to/vietnamese_100_R1.csv --kind translations
```

Phiếu ảnh có thể được chuyển sang công cụ `vlm_probe.depth.blinded_review score` với key local `data/human_review_v2/coordinator_private/answer_key_confirmation_100.json`. Nếu hai reviewer, báo raw agreement và κ có lưu ý mất cân bằng tỷ lệ nhận; nếu mọi nhãn giống hệt một giá trị, κ có thể không xác định và cần báo như vậy. Một reviewer không tạo inter-annotator agreement.

Giữ kết quả gốc 100 ảnh; kết quả sau lọc là sensitivity hậu nghiệm. Bất đồng được ghi trước phân xử. Bản dịch thay đổi phải thành version mới; nếu chạy lại, dùng run mới và mô tả đây là sửa theo ngữ nghĩa sau khi đã có kết quả. Không viết lại artifact VI cũ. Người điều phối cần tự xác nhận phiếu là do người chấm thật; schema validation chỉ kiểm tính đầy đủ và nhất quán, không xác thực danh tính.
