# Giao thức tích hợp và bổ sung kết quả — 11/11/2026

Điểm xuất phát: `origin/hieu@f0d6755`, tiếp nối `dev@dcc2184`. Các kết quả cũ đã được xem. Đây là nghiên cứu tiếp nối và sửa khả năng tái lập; không gọi lại COCO confirmation là holdout chưa mở cho phương pháp chia vùng.

## Phạm vi chốt cho prompt đa miền

Giữ bảy template T0–T6 trong `configs/prompt_study_v1.json` của hieu, không chọn lại theo accuracy CIFAR-100. Chạy nguyên bộ trên Flowers102 (6.149 test), Oxford-IIIT Pet (3.669), DTD test partition 1 (1.880), Food-101 (25.250). DTD giữ suffix `texture` như baseline. T4_domain của DTD không được thêm suffix hai lần: câu chốt là `a close-up surface texture of {class}.` với tên lớp gốc.

T2/T3/T4/T5/T6 thêm thông tin chất lượng, bố cục hoặc miền, do đó kết quả là sensitivity với bộ prompt này, không phải phép đo thuần túy đổi cách diễn đạt tương đương. Bộ N4/D4 trong kế hoạch là phương án khác, chưa chạy và không trộn vào ensemble T7.

Đối chiếu chính: text-embedding ensemble trọng số đều so với T0 trên từng dataset; phụ: logit mean, probability mean với scale 100 cố định và mọi single template. Report đủ năm dataset, top-1, macro, paired delta/CI, chuyển đúng–sai, support từng lớp. Không chọn một ensemble đẹp nhất rồi coi đó là primary. So với best observed single chỉ là mô tả hậu nghiệm.

Giữ checkpoint, preprocessing, FP32/QuickGELU, batch 32. T0 phải khớp 100% nhãn dự đoán baseline tương ứng; cosine lớp đúng so baseline trong tolerance 1e-6. Smoke chỉ kiểm kỹ thuật; không sửa template theo kết quả smoke. Full run mới có samples/classes/templates, toàn bộ scores, predictions, per_class, metrics, manifest và source snapshot chứa giao thức/config thực tế.

## Probing prompt thiếu scores

Không thay `full_prompt_probing_v1`. Chạy lại counting/negation thành `full_prompt_probing_review_v2` để lưu samples, caption ứng viên theo từng template, scores và predictions. Đối chiếu tất cả điểm ước lượng với run cũ; nếu khác, báo cả hai và environment, không sửa run cũ. Chưa có scores thì audit cũ phải báo `metrics_recomputed: false`.

## Chia vùng trên ảnh COCO

Hai bộ đã mở: exploratory 200 nguồn và confirmation 100 nguồn; mỗi nguồn có ảnh gốc và mirror. Tên `confirmation` chỉ định danh bộ dữ liệu của nghiên cứu trước. Phương pháp mới là phân tích tiếp nối trên bộ đã xem.

Pipeline chốt: mở ảnh RGB → center crop vuông cùng vùng nhìn baseline → flip nếu cần → chia bốn nửa trái/phải/trên/dưới → preprocessing checkpoint riêng cho mỗi nửa. Chấm `a photo of {A}.` và `a photo of {B}.`, trong đó A/B đã gồm mạo từ từ metadata; không dùng ground-truth box để tạo crop hoặc dự đoán. Box chỉ dùng đánh giá nhóm hình học và reviewer.

Đặt h=(s_left(A)-s_right(A))-(s_left(B)-s_right(B)); v tương tự top-bottom. Bốn điểm là [h,-h,v,-v]. Exploratory: primary top-1 4-way; confirmation: primary 2-way trái/phải (chỉ tập này có ground truth ngang). Phụ: 4-way, pair both correct, 2×2 assignment trên cặp ngang và margin. Tie chọn index nhỏ nhất, báo tie riêng; assignment D=0 tính sai. CI 95% bootstrap 2.000 lần theo source image, seed 1402. Delta so holistic T2 thuận trên cùng ảnh phải bootstrap ghép cặp theo source.

Lưu đủ scores nhận dạng từng đối tượng ở từng nửa (N×4×2), relational scores (N×4), ảnh/sample metadata, caption pool và predictions. Tính lại toàn bộ bằng audit mà không inference. Không gọi mô phỏng bbox cũ là trần chính xác của CLIP. Kết quả synthetic 100% chỉ áp dụng bộ cảnh đã tạo; pipeline chia vùng là phương pháp suy luận có cấu trúc riêng.

## Review người và báo cáo

Xuất gói 100 ảnh đã cố định với ảnh gốc/crop, box A/B, tên lớp và quan hệ; không scores hoặc quyết định AI. Xuất 100 nhãn EN–VI với ô review trống, không accuracy từng lớp. Không điền thay người review. Review sau suy luận là kiểm nhãn/sensitivity; bản dịch sửa theo ngữ nghĩa phải tạo version mới và study mới nếu đánh giá lại.

Báo cáo tiếng Việt tối đa 10 trang (theo chỉ dẫn mới của người dùng; yêu cầu gốc môn học vẫn cần xác nhận nếu giới hạn 6–8 trang). Mọi bảng số liệu đọc từ artifact có audit. Các run bị chặn môi trường phải ghi `blocked`, không điền số dự kiến. Báo cáo không tuyên bố BPE là nguyên nhân của 84% suy giảm, p>0,05 là tương đương ngẫu nhiên, hay ensemble không bao giờ giúp quan hệ. Không có người review thì ghi rõ chưa thực hiện.
