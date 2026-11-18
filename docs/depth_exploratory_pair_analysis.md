# Phân tích bổ sung hậu nghiệm: ghép cặp 2×2

Động cơ xuất hiện sau khi quan sát spatial v1 mới: accuracy 4-way gần chance, phân bố đáp án lệch và tỷ lệ đúng cả hai ảnh rất thấp. Phân tích này dùng scores đã lưu, không chạy thêm model, không thay metric chính và không tuyên bố là giả thuyết chốt trước thí nghiệm.

Với hai ảnh A-left-B/A-right-B và hai caption left/right, lập ma trận S 2×2. Đo `gap = S00 + S11 − S01 − S10`: gap dương nghĩa là tổng điểm ghép đúng cao hơn ghép hoán đổi. Một thiên lệch cộng chỉ phụ thuộc caption hoặc chỉ phụ thuộc ảnh sẽ triệt tiêu trong hiệu này; không khẳng định mọi dạng bias đều được loại. Báo cả accuracy 2-way từng ảnh, tỷ lệ cả hai đúng, accuracy ghép cặp và số near-tie (|gap|≤1e-6). Khoảng tin cậy theo nhóm cảnh/ảnh gốc như giao thức chính.

Áp dụng cả trục ngang/dọc cho ảnh tổng hợp; với ảnh tự nhiên chỉ dùng các cặp gốc/lật có nhãn trái/phải, vì lật ngang không tạo phản thực trên/dưới. Đây là phép chẩn đoán được cung cấp cặp ảnh và cặp quan hệ đúng, không phải cải thiện zero-shot 4-way trong triển khai thông thường. Không so trực tiếp accuracy 2-way/assignment với 4-way như cùng một bài toán.
