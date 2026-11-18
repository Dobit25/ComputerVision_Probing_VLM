# Làm rõ trước tính kết quả — 17/11/2026

Phiếu cuối đã được kiểm: dòng 092 có `both_objects_visible=no`, `class_names_correct=no`, `overall_accept=no`, lý do `OCCLUDED_OR_TINY`. Câu mô tả `both_objects_visible=yes` trong protocol ban đầu phản ánh bản nháp trước đó, không phải phiếu đóng băng. Phiếu cuối không có bất nhất này. Giữ protocol snapshot ban đầu và ghi đính chính riêng để bảo toàn lịch sử.

Ảnh lật trong metadata run cũ có tên variant `horizontal_flip`; cách gọi “mirror” trong mô tả là cùng phép biến đổi, không phải một biến thể khác. Kiểm cặp phải dùng đúng tên metadata này. Chưa có kết quả sensitivity mới khi đính chính.

Làm rõ khi đối chiếu cấu hình: bootstrap của nhánh chia vùng và bảng đối chiếu chung dùng seed **1402**; phép tái lập sáu điều kiện holistic lịch sử giữ seed **14031** từ config gốc. Cả hai dùng 2.000 lượt và đơn vị source. Cùng point estimate có thể có CI bootstrap hơi khác vì seed khác; không thay seed để làm CI trùng nhau. Wilson của holistic được báo riêng.
