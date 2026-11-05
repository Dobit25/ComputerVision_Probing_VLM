# Kiểm chứng ghép cặp 2×2 trên ảnh mới — giao thức v1

Ghi trước rà soát mới và mọi suy luận tập xác nhận. Một checkpoint OpenAI CLIP ViT-B/32, môi trường gen, cùng preprocessing/FP32/batch 32. Phát hiện thăm dò cũ là accuracy ghép cặp ngang T2 thuận 54/100. Không đổi prompt theo kết quả tập mới; không fine-tuning.

## A. Rà soát lại 200 ảnh cũ

Cùng trợ lý AI xem lại đủ 200 ảnh đã chọn, không hiển thị dự đoán/scores từng ảnh hoặc lý do chấp nhận cũ. Trợ lý đã biết kết quả tổng hợp, nên đây là tự rà soát bổ sung, **không phải đánh giá độc lập hoặc mù hoàn toàn**. Không báo inter-annotator agreement. Kiểm tra hai đối tượng, tên lớp, quan hệ 2D và vùng center crop. Loại khỏi phân tích độ nhạy nếu đối tượng quá mờ/che khuất, có nhiều đối tượng cùng tên làm câu mơ hồ, tên lớp sai/khó xác nhận, tranh/biểu tượng hoặc quan hệ không rõ. Lưu quyết định cho từng ảnh trước khi tính lại kết quả. Không sửa file lựa chọn hoặc run cũ. Báo kết quả cũ đầy đủ và kết quả trên phần được xác nhận; phần lọc lại là phân tích độ nhạy hậu nghiệm, không thay thế test cũ.

## B. Tập mới, chọn trước khi chạy model

Nguồn vẫn COCO val2014 và nguyên bộ lọc hình học của depth v1. Loại **toàn bộ 400 ID ảnh từng được xem**, kể cả ảnh đã bị loại, khỏi ứng viên mới. Chỉ chọn quan hệ ngang trái/phải để bản lật tạo phản thực thật sự. Xếp cặp theo SHA256(seed 14031, image ID, annotation A/B), lấy một cặp mỗi nguồn ảnh. Khảo sát metadata trước giao thức cho 178 ảnh mới đủ điều kiện (372 cặp có hướng); chưa tải/xem ảnh hoặc chạy CLIP trên các ảnh này tại thời điểm ghi giao thức.

Rà soát toàn bộ 178 ứng viên với cùng tiêu chí ở mục A. Sàng lọc dHash center crop 64 bit và bản mirror: khoảng cách ≤6 với bất cứ 400 ảnh cũ thì loại; trong tập mới giữ ảnh xếp trước, loại bản gần trùng xếp sau. Kiểm tra hash chỉ là sàng lọc, không bảo đảm loại hết mọi cảnh gần trùng. Rà soát trực quan cũng loại cảnh gần trùng nhìn thấy. Chọn 100 ảnh được chấp nhận đầu tiên theo rank; nếu chỉ còn 80–99 ảnh thì dùng tất cả và báo thiếu mẫu, nếu dưới 80 thì dừng trước suy luận để ghi sửa đổi giao thức. Không mở rộng/chọn lại tập sau khi xem scores. Không ép cân bằng hướng ảnh gốc: mỗi ảnh có bản lật nên cặp luôn có một trái và một phải.

Mỗi nguồn ảnh có gốc và mirror, giữ nguyên hai tên lớp. Đánh giá ba template và hai thứ tự như v1; **T2 thuận** (`an image of {A} {relation} {B}.`) là điều kiện chính. Cố định manifest lựa chọn, reviewer, timestamp, checksum và snapshot trước bất cứ inference mới nào. Smoke dùng ảnh cũ, không xem trước một phần holdout để điều chỉnh pipeline.

## C. Câu hỏi, metric và cách kết luận

Câu hỏi xác nhận: sau khi loại thành phần điểm cộng chỉ phụ thuộc ảnh/caption bằng hiệu 2×2, mô hình có ghép đúng ảnh–quan hệ cao hơn mốc 50% không?

Với hai ảnh được sắp theo nhãn trái/phải và hai caption tương ứng, tính `D=S00+S11−S01−S10` bằng float64 từ cosine FP32 đã lưu. `D>0` là đúng; D=0 tính sai theo quy tắc cũ, báo riêng near-tie |D|≤1e−6. Metric chính: tỷ lệ ghép đúng T2 thuận, CI Wilson 95% và bootstrap theo nguồn ảnh (2.000 lần, seed 14031). Kiểm định nhị thức chính một phía H0:p≤0,5, H1:p>0,5, alpha 0,05. Với tie, kiểm định này bảo thủ dưới giả định trao đổi hai cách ghép; báo số tie. Không kết luận p=0,5 hoặc “không có thông tin vị trí” chỉ vì không bác bỏ H0.

Metric phụ: hai lựa chọn trái/phải cho từng ảnh; tỷ lệ cặp mà từng ảnh đều chọn đúng; margin và near-tie; đủ sáu template/thứ tự. Các điều kiện phụ mang tính mô tả, không chọn điều kiện tốt nhất thay điều kiện chính. Bốn lựa chọn vẫn được lưu để truy vết, nhưng tập mới chỉ có nhãn ngang nên không so accuracy 4-way với 28% của bộ cũ cân bằng bốn quan hệ như cùng phân phối.

So sánh accuracy ghép cặp với tập cũ 100 nguồn ngang: hai tập độc lập về source ID, báo chênh lệch và bootstrap hai tập độc lập; không bootstrap ghép cặp giữa hai tập. Tập cũ đã được dùng để hình thành giả thuyết, vì vậy không gộp nó vào kết quả xác nhận chính. Báo cả kết quả đầy đủ cũ và phần giữ sau rà soát lại.

## D. Giới hạn và bàn giao

Tập mới chưa được xem kết quả CLIP, nhưng cùng COCO và cùng người rà soát AI; không gọi là xác nhận ngoài miền hoặc bởi annotator độc lập. Không biết mức trùng với pretraining. Ảnh mirror có thể thay đổi yếu tố khác ngoài vị trí. Giấy phép từng ảnh giữ theo metadata; ảnh không đưa vào Git. Đóng băng dữ liệu trước suy luận, lưu scores/predictions/CI, audit tính lại và cập nhật PDF tiếng Việt, bao gồm cả kết quả không ủng hộ kỳ vọng ban đầu.
