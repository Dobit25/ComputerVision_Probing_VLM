# Review tích hợp nhánh hieu vào dev

Phạm vi được người dùng cho phép ngày 11/11/2026: sửa diễn giải, bổ sung audit/traces, hoàn thành prompt trên bốn dataset còn lại, bộ giao việc review người, chia vùng COCO và report tiếng Việt tối đa 10 trang. Điểm xuất phát `hieu@f0d6755`, `dev@dcc2184`. Review này do trợ lý kiểm code và artifact, không phải peer review của giảng viên hay xác nhận nhãn bởi người.

## Các phát hiện đã xử lý

| Phát hiện | Xử lý | Bằng chứng |
|---|---|---|
| Kết luận BPE gây ra 84% mức suy giảm | Thay bằng mô tả khác biệt giữa khung câu/tên lớp và giới hạn nhân quả | Generator synthesis và báo cáo v2 |
| p>0,05 bị diễn giải thành hiệu năng bằng ngẫu nhiên | Report rõ CI và chưa đủ bằng chứng vượt ngưỡng; không tuyên bố tương đương | Báo cáo v2, protocol |
| Gọi ensemble vô hiệu với mọi quan hệ | Thu hẹp về bộ template và dữ liệu đã kiểm | Generator synthesis |
| Box simulation được gọi là trần/kết quả CLIP trên COCO | Phân biệt mô phỏng bbox với inference mới | Protocol, natural_regions |
| Token BPE 4,27/1,31 không khớp artifact | Dùng 7,05/1,19; tỷ số trung bình 5,92 khác trung bình tỷ số 6,44 | Đề cương P6, bảng sinh từ JSON |
| Group negation ghi 180 | Sửa thành 360 nhóm cho 720 ảnh | Metadata core và synthesis |
| Cohen κ báo 1 khi mọi nhãn đều cùng một giá trị | Báo null/undefined; raw agreement vẫn 100% | Unit test và helper review |
| Thiếu kế hoạch trên remote, bốn link sai | Đưa long_term_project_plan.md vào Git, sửa đường dẫn | Docs và commit d7fb23c |
| P5/P6 dùng không nhất quán | P5: văn hóa; P6: report/presentation | Đề cương cập nhật |
| Extension dùng nhầm config/protocol mặc định depth | Giữ run cũ; run mới có study_config/evaluation_config, đúng protocol và source ZIP | extension_artifacts.begin_extension |
| Prompt probing chỉ có metrics tổng hợp | Run mới lưu samples/templates/scores/predictions; audit run cũ báo không thể tính lại | Run review v2 và audit riêng |
| Chưa có predictions/per-class ở hai run CIFAR | Khôi phục từ scores trong thư mục review riêng | results/review_hieu_v2 |
| Thiếu audit cho extension | Kiểm checksum/snapshot; tính lại CI/argmax/parity và số mẫu đúng split | extension_artifacts |
| Preprocessing nửa ảnh có thể mất nội dung | Giữ native làm primary; padding là control phụ, chốt trước inference COCO | Commit 65f27bc, protocol |

## Giữ lịch sử và kiểm chứng

Không sửa scores, metrics hoặc manifest của các run được xuất bản trước review. Source snapshot cũ xác định mã đã chạy; git_head cũ được giữ nguyên. Sửa protocol ở working tree sau run không làm mất lịch sử vì audit đọc bản protocol đã đóng băng trong source ZIP của run đó. Input dữ liệu không được lưu trong ZIP vẫn phải khớp hash hiện tại.

Audit classification kiểm cả dự đoán và cosine lớp thật với baseline, class order/sample IDs, số mẫu full/smoke, macro/transition, metrics/CI. Khi có alignment norms/raw text vectors, kiểm thêm đồng nhất thức embedding ensemble. Run VI cũ chưa có raw text vectors nên giới hạn này vẫn được ghi rõ.

Chia vùng COCO lưu cả scores đối tượng trên bốn nửa, relational scores và baseline holistic cùng ảnh. Primary gốc và control padding đều được báo; không đổi primary theo kết quả. Control padding cũng thay nền/kích thước tương đối, không là bằng chứng cô lập cơ chế.

## Bộ bàn giao

- Báo cáo và nguồn số liệu: `reports/research_review_v2/`.
- Kết quả/audit bổ sung: `results/*_review_v2`, `results/review_hieu_v2/`.
- Trạng thái hàng đợi, tests và verifier lịch sử: `results/integration_v2/`.
- Hướng dẫn thành viên: `docs/human_review_assignment_v2.md`; phiếu trắng: `reviews/human_review_v2/`.
- Gói local có ảnh/crop: `data/human_review_v2/reviewer_package.zip`, không đưa pixels lên Git.

Chưa có phiếu hoàn thành của người; review ảnh và từ điển vẫn pending. Tạo phiếu/HTML không đồng nghĩa gán nhãn đã được người xác thực. Bộ ảnh văn hóa Việt Nam và fine-tuning chưa thực hiện trong mốc này.

Trước khi nhập dev: hàng đợi phải completed_and_audited, tests đạt, các manifest lịch sử nguyên vẹn, PDF không quá 10 trang và đã xem trực quan, kiểm clone sạch và nguồn/checksum của báo cáo. Hồ sơ kiểm bàn giao cuối cùng nằm ở `results/integration_v2/delivery_validation.json`.
