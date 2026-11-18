# Topic 14: đề bài gốc và lựa chọn triển khai

**Nguồn yêu cầu:** [Computer Vision — Final Project Topics, mục 14 và Deliverables](https://github.com/NguyenMToan/CVCourse/blob/main/project_topics.md). Tài liệu này diễn giải đề bài bằng tiếng Việt. Khi có khác biệt, đối chiếu với nguồn gốc hoặc xác nhận với giảng viên.

## 1. Yêu cầu từ đề bài

### Core task — phạm vi tối thiểu

1. Chọn **CLIP hoặc một VLM mở** và đánh giá khả năng **zero-shot classification** trên **nhiều bộ dữ liệu**. Zero-shot ở đây nghĩa là dùng mô hình đã có để phân loại bằng mô tả văn bản của các lớp, không dùng nhãn của bộ kiểm tra để huấn luyện mô hình.
2. Thiết kế đánh giá có hệ thống để tìm các tình huống mô hình sai, gồm **đếm số lượng**, **quan hệ không gian**, **phủ định** và **phân biệt các loại gần giống nhau** (fine-grained categories).

Đề bài không ấn định model cụ thể, tên dataset, số lượng ảnh, prompt template hay ngưỡng accuracy phải đạt.

### Extensions — hướng mở rộng

- Đo mức độ kết quả thay đổi khi thay cách viết prompt (prompt sensitivity).
- Kiểm tra việc gộp nhiều prompt có cải thiện kết quả hay không (prompt ensembling).
- Đánh giá prompt bằng ngôn ngữ khác tiếng Anh **hoặc** nội dung hình ảnh liên quan đến văn hóa Việt Nam.

Các mục trên là hướng mở rộng để nâng chất lượng dự án; chúng không thay thế core task. Từ “hoặc” trong đề bài không buộc nhóm phải làm cả hai hướng ngôn ngữ và văn hóa Việt Nam.

### Hướng nghiên cứu và quy mô

Đề bài gợi ý nghiên cứu hiệu năng VLM với prompt tiếng Việt và khái niệm hình ảnh gần gũi tại Việt Nam. Đây là **gợi ý về hướng đóng góp**, không phải bằng chứng rằng chưa có nghiên cứu nào trên chủ đề đó; cần khảo sát tài liệu trước khi đưa ra tuyên bố mới trong báo cáo.

Các topic của môn học được thiết kế để thực hiện trên một GPU phổ thông hoặc Colab miễn phí, với mô hình pretrained, tập dữ liệu nhỏ hay lấy mẫu, hoặc phương pháp lúc suy luận. Không yêu cầu huấn luyện mô hình lớn từ đầu. Đề bài gợi ý chia nhóm khoảng 6 người: 2 người xây benchmark, 2 người đánh giá, 1 người phân tích prompt và 1 người viết báo cáo.

### Mốc bàn giao của môn học

| Mốc | Tuần | Nội dung |
| --- | ---: | --- |
| Thành lập nhóm và chọn đề tài | 4 | Danh sách nhóm, thứ tự ưu tiên đề tài |
| Proposal | 5 | 2 trang: vấn đề, dữ liệu, phương pháp, kế hoạch đánh giá, phân công |
| Milestone report | 10 | 3 trang: kết quả baseline, vấn đề gặp phải, kế hoạch điều chỉnh |
| Trình bày cuối kỳ | 15 | 15 phút và hỏi đáp; tất cả thành viên có mặt |
| Mã nguồn và báo cáo | 15 | Repository tái lập được; báo cáo 6–8 trang theo định dạng hội nghị |

## 2. Lựa chọn triển khai của nhóm — không phải yêu cầu đề bài

| Nội dung | Trạng thái hiện tại |
| --- | --- |
| Môi trường làm việc trên máy hiện tại | Đã chọn Conda `gen`; xem hướng dẫn cài các thư viện baseline trong [README](../README.md). |
| Baseline đầu tiên | Đã thống nhất CLIP ViT-B/32 pretrained OpenAI, CIFAR-100 test split, prompt `a photo of a {class}.`, top-1 accuracy. Cấu hình và cách đo nằm trong [giao thức v1](experiment_protocol.md). |
| Phạm vi hiện hành sau điều chỉnh ngày 04/11/2026 | **Một checkpoint OpenAI CLIP ViT-B/32 × năm dataset:** CIFAR-100, Flowers102, Oxford-IIIT Pet, DTD test partition 1, Food-101. [Giao thức hiện hành](one_model_protocol.md) thay thế kế hoạch ba checkpoint trước đó. Đề bài không bắt buộc nhiều model. |
| Thứ tự công việc đã được cho phép | Hoàn tất ba dataset mới → tổng hợp baseline → counting, spatial, negation, fine-grained. Kết quả được ghi trong [báo cáo baseline](../reports/baseline/README.md) và [báo cáo probing](../reports/probing/README.md). |
| Bộ ảnh văn hóa Việt Nam | Chưa chốt danh mục, nguồn ảnh, số lượng, giấy phép và cách gán nhãn. Con số 200–500 ảnh trong kế hoạch ban đầu là ước lượng, không phải yêu cầu môn học. |
| Cách đo các failure mode | [Giao thức probing](probing_protocol.md): hình học có đối chứng cho counting/spatial/negation; CIFAR-100 trong nhóm coarse cho fine-grained. Các thiết kế này do nhóm lựa chọn, không phải benchmark bắt buộc của đề bài. |

Các quyết định trên cần được cập nhật khi nhóm thống nhất giao thức thí nghiệm. Không ghi số liệu dự đoán hoặc kết luận về model như thể đó là kết quả đã đo.
