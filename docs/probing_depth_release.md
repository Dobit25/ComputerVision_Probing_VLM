# Mốc nghiên cứu v1.1.0-probing-depth

Ngày đóng gói: **05/11/2026** (Asia/Saigon). Nhánh phát triển: `dev`. Mốc này tiếp nối `v1.0.0-core`, giữ một checkpoint OpenAI CLIP ViT-B/32 và không fine-tuning.

## Phạm vi và cách đọc kết quả

| Mốc | Nội dung | Tài liệu |
|---|---|---|
| Category | CIFAR-100, đủ 100 lớp ứng viên; baseline 64,48%, thêm category 62,92% | [Giao thức](cifar100_category_protocol.md), [kết quả](../reports/cifar100_category_v1/) |
| Probing chuyên sâu | Phân tích dữ liệu cũ, 1.440 ảnh hình học, tám điều kiện category, 200 ảnh tự nhiên và bản lật; phân tích ghép cặp hậu nghiệm | [Giao thức](probing_depth_protocol.md), [báo cáo](../reports/probing_depth_v1/README.md) |
| Xác nhận mới | Tự rà soát 200 ảnh cũ, giữ 176; rà soát 178 ứng viên mới, chọn 100 nguồn không trùng 400 ứng viên cũ trước suy luận | [Giao thức](natural_confirmation_protocol.md), [báo cáo mới nhất](../reports/natural_confirmation_v1/README.md) |

[PDF tiếng Việt mới nhất, 3 trang](../reports/natural_confirmation_v1/bao_cao_cap_nhat.pdf) là bản dành cho giảng viên review. Ghép cặp chính trên tập mới đạt 53/100, CI Wilson 95% [43,29%; 62,49%], p một phía = 0,3086; chỉ 4/100 cặp có cả hai ảnh được dự đoán đúng riêng rẽ. Không diễn giải kết quả như bằng chứng tương đương đoán ngẫu nhiên. Người rà soát là cùng trợ lý AI, chưa có annotator người độc lập.

## Lịch sử và tính toàn vẹn

Bốn commit lần lượt đóng gói category, probing chuyên sâu, xác nhận mới và tài liệu bàn giao. Thứ tự commit này tổ chức công việc đã thực hiện; không phải bằng chứng đăng ký trước nghiên cứu. Tập xác nhận có hồ sơ đóng băng và checksum trước suy luận; giả thuyết phát sinh từ phân tích tập cũ.

Giữ nguyên `git_head`, timestamps, scores, manifest, source ZIP và các PDF lịch sử. `git_head` cũ không nhất thiết chứa toàn bộ mã chạy vì thí nghiệm diễn ra khi working tree còn thay đổi; dùng `source_sha256` và `source_snapshot` để xác định mã đã thực thi. Các câu “chưa commit/cục bộ” trong tài liệu thí nghiệm mô tả thời điểm lập tài liệu, không phải trạng thái phát hành hiện nay.

Ảnh dataset, weights, cache và môi trường conda không đưa vào Git. Các scores, dự đoán, metadata nhãn/nguồn/quyền sử dụng, audit, quyết định rà soát và source ZIP được giữ để kiểm tra kết quả. `.gitattributes` giữ nguyên byte, tránh thay đổi LF/CRLF làm sai checksum. Tệp `results/probing_depth_release_validation.json` ghi kiểm tra khi đóng gói.

## Quay lại phiên bản

```powershell
git fetch origin --tags
git switch --detach v1.1.0-probing-depth
```

Tag cố định mốc này; `dev` tiếp tục phát triển. Dùng nhánh mới nếu muốn chỉnh sửa từ tag. Không di chuyển tag `v1.0.0-core` hoặc viết lại lịch sử run.

## Kiểm tra và chạy lại

Thiết lập môi trường theo [README](../README.md). Trên máy hiện tại dùng conda `gen`:

```powershell
conda activate gen
python -m unittest discover -s tests -v
python -m vlm_probe.category_study.audit results/full_cifar100_category_v1
python -m vlm_probe.depth audit
python -m vlm_probe.depth.confirmation audit
```

Bộ 43 unit tests đã chạy thành công từ bản xuất sạch của Git trong môi trường `gen`, không cần ảnh dataset hoặc checkpoint. Các audit đầy đủ cần dữ liệu cục bộ theo từng bài. Có thể kiểm checksum artifact và source ZIP từ repo mà chưa tải dữ liệu. Trên máy mới, khôi phục dữ liệu theo [hướng dẫn probing](../reports/probing_depth_v1/README.md) và [hướng dẫn xác nhận](../reports/natural_confirmation_v1/README.md); CIFAR-100 cần archive và dữ liệu gốc theo pipeline baseline. Không xóa hồ sơ review để tạo tập khác khi đang kiểm chứng run đã lưu.

Ví dụ chạy lại xác nhận vào thư mục mới sau khi khôi phục dữ liệu:

```powershell
python -m vlm_probe.depth.confirmation run --run-name natural_confirmation_repeat
python -m vlm_probe.depth.confirmation audit --run-name natural_confirmation_repeat
```

Không ghi đè run v1. Các lệnh `report` dựng lại tệp dẫn xuất; không cần chạy chúng chỉ để đọc PDF đã phát hành. Muốn đối chiếu checksum của bản phát hành, dùng artifact tại tag trước khi tái sinh báo cáo.

## Phần chưa thực hiện

Rà soát nhãn bởi người độc lập, prompt ensemble, prompt tiếng Việt và nội dung văn hóa Việt Nam chưa được hoàn thành trong mốc này. Các kết luận chỉ áp dụng cho checkpoint, tập ảnh và giao thức đã kiểm tra.
