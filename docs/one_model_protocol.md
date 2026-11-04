# Phạm vi hiện hành: một model, năm dataset và core probing

Theo điều chỉnh của người dùng ngày 04/11/2026, chỉ dùng **CLIP ViT-B/32 pretrained OpenAI**. Kế hoạch ba checkpoint trong `experiment_matrix.md/json` là tài liệu lịch sử, đã được thay thế; không tiếp tục tải/chạy LAION hoặc SigLIP.

## Classification

| Dataset | Số lớp | Test samples | Config |
| --- | ---: | ---: | --- |
| CIFAR-100 | 100 | 10.000 | `configs/clip_b32_cifar100.json` |
| Flowers102 | 102 | 6.149 | `configs/clip_b32_flowers102.json` |
| Oxford-IIIT Pet | 37 | 3.669 | `configs/clip_b32_oxford_pet.json` |
| DTD partition 1 | 47 | 1.880 | `configs/clip_b32_dtd.json` |
| Food-101 | 101 | 25.250 | `configs/clip_b32_food101.json` |

Giữ các quyết định dữ liệu/prompt của kế hoạch trước: test split gốc, tên lớp/thứ tự của torchvision, một prompt `a photo of a {class}.`, riêng DTD dùng `a photo of a {class} texture.`. Mỗi ô mới smoke 200 ảnh rồi full test; không chọn prompt theo accuracy. Model OpenAI và môi trường `gen` giữ nguyên FP32/QuickGELU, batch 32, seed 0. Protocol v3 mở rộng dataset/metric, không thêm model.

Metric chính là top-1 theo ảnh; metric phụ là mean per-class accuracy. Kết quả cũ không sửa: mean per-class được tính từ CSV khi tổng hợp. Báo cáo đủ 5 dataset, accuracy theo lớp, các cặp nhầm lẫn thường gặp và nhóm câu hỏi cho probing. Các dataset khác nhau có độ khó khác nhau, nên chênh lệch accuracy không phải hiệu quả của thay đổi model.

Pet/DTD/Food có mirror archive trên Hugging Face, revision cố định trong adapter. Chỉ chấp nhận archive khớp MD5 gốc torchvision; từng ảnh test đã giải nén và metadata được so với archive. Manifest ghi cả nguồn gốc, cấu hình tải và URL tải thực tế trong provenance archive. Không chuyển sang ảnh tái mã hóa hoặc split khác.

## Thứ tự và giới hạn kết luận

Hoàn tất ba classification mới → tổng hợp baseline → chạy giao thức trong `probing_protocol.md`. Các bài probing kiểm tra một checkpoint, không so sánh điểm yếu giữa model. Thí nghiệm hình học tự tạo được báo cáo là kiểm tra có kiểm soát; không gọi đó là benchmark ảnh tự nhiên hoặc bằng chứng về mọi tình huống thực tế. Phân tích fine-grained bổ sung ảnh tự nhiên từ CIFAR-100 và nhóm nhầm lẫn trên Flowers/Pet/Food.

Chưa thực hiện prompt ensembling, tiếng Việt/văn hóa Việt Nam hoặc fine-tuning. Đây là các extensions riêng sau core task.

## Trạng thái mốc thí nghiệm

Đã hoàn thành **5/5 classification cells** và **4/4 bài probing** theo giao thức đã chọn. Xem [baseline](../reports/baseline/README.md), [probing](../reports/probing/README.md), [diễn giải](../reports/interpretation.md). Các con số không thay thế báo cáo hội nghị/slide cuối kỳ; chúng là dữ liệu và phân tích để biên soạn các sản phẩm đó.
