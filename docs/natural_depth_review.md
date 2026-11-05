# Rà soát ảnh tự nhiên trước suy luận

Nguồn cuối: COCO val2014, annotation gốc lấy từ ZIP chính thức bằng HTTP Range, kiểm CRC32/kích thước member và pin SHA-256. Đổi từ val2017 do chỉ có 70 ảnh đạt bộ lọc gốc; xem bổ sung A1 trong giao thức. Với val2014, 1.630 cặp có hướng thuộc 704 ảnh đạt bộ lọc; cố định 400 ảnh dự phòng (100 mỗi quan hệ), không xem điểm CLIP.

Trợ lý đã xem 25 contact sheet: 15 sheet của 240 ứng viên đầu và 10 sheet của 160 ảnh dự phòng. Mỗi ô hiển thị center crop, hai bounding box A/B, tên lớp, quan hệ và ID. Loại 142 ảnh vì đối tượng khó nhận diện, che khuất/cắt xén quá nhiều, nhãn mơ hồ, tranh/biểu tượng/màn hình/ảnh ghép, nhiều instance gây mơ hồ hoặc cảnh gần trùng. Một cảnh clock/laptop gần trùng được nhận ra trực quan và loại; kiểm tra dHash 64 bit, ngưỡng khoảng cách ≤6 trên 258 ảnh còn lại không tìm thêm cặp gần trùng. Kiểm tra dHash không bảo đảm loại hết mọi ảnh gần giống.

258 ảnh được chấp nhận trước khi chạy model; theo trái/phải/trên/dưới: 63/62/63/70. Lấy 50 ảnh đầu mỗi quan hệ theo thứ tự cố định, tổng 200 ảnh gốc. Ảnh thêm bằng horizontal flip không được tính là ảnh nguồn độc lập. Giữ cùng nhãn tên vật thể; trái/phải đổi, trên/dưới không đổi.

Rà soát này do **một trợ lý AI**, kết hợp validator hình học từ annotation, không phải hai người gán nhãn độc lập. Chưa có kiểm chứng đồng thuận giữa người đánh giá; một số ranh giới tên lớp COCO và vật thể bị che vẫn có thể gây bất định. Mọi quyết định và lý do đều mở để giảng viên/nhóm review lại. Không thay danh sách sau khi thấy kết quả CLIP.

- Quyết định đủ 400 ảnh: `results/depth_natural_selection_v1/review.json`.
- 200 ảnh được chọn: `results/depth_natural_selection_v1/selected.json`.
- Metadata, URL nguồn, giấy phép từng ảnh: `candidates.json` và `selected.json` cùng thư mục.
- Contact sheets: `data/probing_depth_v1/natural/review_sheets/` (không đưa ảnh vào Git).

Khôi phục ảnh/sheet từ danh sách có sẵn: `python -m vlm_probe.depth natural-prepare`. Lệnh kiểm hash ảnh đã rà soát và không tạo lựa chọn mới. Annotation CC BY 4.0; ảnh giữ giấy phép theo metadata COCO/Flickr, không gán chung một giấy phép cho toàn bộ ảnh.
