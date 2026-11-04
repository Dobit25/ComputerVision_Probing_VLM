# Kết quả baseline và mở rộng hai dataset

> Mốc lịch sử hai dataset được giữ bên dưới. Phạm vi mới: **một model, năm dataset và bốn bài core probing**. Xem [báo cáo baseline hiện hành](../reports/baseline/README.md), [báo cáo probing](../reports/probing/README.md) và [giao thức](../docs/one_model_protocol.md).

Đo ngày **03/11/2026** trên máy có NVIDIA GeForce RTX 4050 Laptop GPU, môi trường Conda `gen`, Python 3.12.14, PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126 và OpenCLIP 3.3.0. Các phiên bản còn lại được lưu trong `environment.json` của từng lần chạy.

Model: CLIP ViT-B/32 pretrained OpenAI, QuickGELU, FP32, batch size 32, một prompt cố định `a photo of a {class}.`. Không huấn luyện hay chọn prompt theo kết quả test. [Giao thức đầy đủ](../docs/experiment_protocol.md).

| Lần chạy | Số ảnh | Số đúng | Top-1 accuracy | Mục đích |
| --- | ---: | ---: | ---: | --- |
| [Smoke](smoke_cifar100_clip_b32_v1/metrics.json) | 200 | 139 | 69,50% | Kiểm tra kỹ thuật trên 200 ảnh đầu của test set |
| [Full test](full_cifar100_clip_b32_v1/metrics.json) | 10.000 | 6.448 | **64,48%** | Baseline chính thức trên toàn bộ CIFAR-100 test |
| [CIFAR-100 sau refactor](full_cifar100_clip_b32_refactor/metrics.json) | 10.000 | 6.448 | **64,48%** | Đối chiếu từng dự đoán với baseline gốc |
| [Flowers102 smoke](smoke_flowers102_clip_b32_v2/metrics.json) | 200 | 107 | 53,50% | Kiểm tra adapter mới, đủ 102 lớp ứng viên |
| [Flowers102 full test](full_flowers102_clip_b32_v2/metrics.json) | 6.149 | 3.921 | **63,7665%** | Đánh giá test split gốc Oxford |

Các dòng smoke/full dùng số lượng ảnh khác nhau; chênh lệch accuracy không phải hiệu quả của một cải tiến model. Cấu hình model/prompt giữ nguyên. Hiện đã đánh giá một model trên hai dataset; các bài probing và phân tích failure mode có hệ thống của Topic 14 vẫn cần thực hiện.

## Kiểm chứng mốc mở rộng

- [Đối chiếu refactor](refactor_comparison.json): đủ **10.000 ảnh**, **0 dự đoán thay đổi**, chênh lệch mọi score lưu trong CSV là **0**. Hai `predictions.csv` giống từng byte; môi trường, config, checkpoint, preprocessing, thứ tự lớp và nguồn dữ liệu khớp.
- [Biên bản kiểm chứng chung](multi_dataset_validation.json): verifier chấp nhận cả năm run; nhãn/ID của Flowers102 được đối chiếu với `imagelabels.mat` và `setid.mat`; tên lớp khớp torchvision; 16 tests đã qua.
- 200 ảnh chung giữa smoke/full Flowers102 có cùng toàn bộ nhãn dự đoán. Chênh lệch cosine dự đoán tối đa `5.96e-08`, không làm thay đổi nhãn. Smoke dùng 200 index đầu của `tstid`, không phải mẫu ngẫu nhiên đại diện.
- Flowers102 có **20–238 ảnh test/lớp**. Accuracy theo ảnh là **63,7665%**; trung bình accuracy của 102 lớp là **64,2338%**, tính từ [per_class.csv](full_flowers102_clip_b32_v2/per_class.csv). Hai metric có trọng số khác nhau, cần ghi rõ khi báo cáo.
- [Giao thức Flowers102 và lệnh tái lập](../docs/multi_dataset_protocol.md). Nguồn tải dùng mirror với MD5 gốc; ảnh test đã giải nén được xác minh lại với archive trước mỗi lần đánh giá.

Đã lưu ba snapshot mã nguồn: [baseline ban đầu](source_snapshots/baseline_v1.zip), [sau refactor CIFAR-100](source_snapshots/refactor_cifar100.zip), [có hai dataset](source_snapshots/multi_dataset_v2.zip). Các file trong từng snapshot đã được đối chiếu với `source_sha256` của run tương ứng; hash archive nằm trong biên bản chung. Khi cần tái lập một mốc lịch sử, giải nén snapshot vào một bản sao repository riêng để giữ nguyên bản đang làm việc. Snapshot chứa mã nguồn/config/requirements, không chứa dữ liệu hay weights.

## Bằng chứng baseline CIFAR-100 ban đầu

- [Dự đoán của toàn bộ 10.000 ảnh](full_cifar100_clip_b32_v1/predictions.csv): mỗi dòng có index gốc, nhãn thật, nhãn dự đoán và cosine scores.
- [Accuracy theo lớp](full_cifar100_clip_b32_v1/per_class.csv): đủ 100 lớp, mỗi lớp 100 ảnh.
- [Confusion matrix](full_cifar100_clip_b32_v1/confusion_matrix.csv): hàng là lớp thật, cột là ID lớp dự đoán.
- [Danh sách prompt](full_cifar100_clip_b32_v1/classes.json), [config](full_cifar100_clip_b32_v1/config.json), [môi trường](full_cifar100_clip_b32_v1/environment.json), [manifest](full_cifar100_clip_b32_v1/manifest.json).
- [Biên bản kiểm tra](validation.json): checksum artifact, đếm lại accuracy, đối chiếu 10.000 nhãn thật và 100 tên lớp với tệp CIFAR-100 gốc đã kiểm tra MD5, đối chiếu hash mã nguồn, kiểm tra 200 dự đoán trùng nhau giữa smoke và full test.

200 ảnh chung giữa smoke/full CIFAR-100 ban đầu có **100% nhãn dự đoán trùng nhau**. Chênh lệch tuyệt đối lớn nhất của cosine dự đoán là khoảng `8.94e-08`; không ảnh hưởng nhãn dự đoán trong hai lần chạy này.

Full test có `evaluation_seconds = 44.4604507`, tương ứng khoảng 224,92 ảnh/giây. Thời gian này bao gồm encode text, tiền xử lý/nạp batch ảnh, suy luận và ghi CSV; không gồm tải model hoặc dữ liệu. Đỉnh bộ nhớ được PyTorch cấp phát khi đánh giá là 895.618.560 byte (khoảng 854 MiB), không phải toàn bộ bộ nhớ GPU của hệ thống. Đây là số đo trong lần chạy này, không phải cam kết tốc độ trên máy khác.

## Chạy lại

Từ gốc repository, dùng môi trường `gen` đã thiết lập:

```powershell
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_cifar100.json
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_flowers102.json
```

Chương trình tạo một thư mục kết quả mới, dùng lại trọng số và dữ liệu đã tải. Có thể kiểm tra các artifact của baseline đã lưu bằng:

```powershell
conda run --no-capture-output -n gen python -m vlm_probe.verify results/full_cifar100_clip_b32_v1
conda run --no-capture-output -n gen python -m vlm_probe.verify results/full_flowers102_clip_b32_v2
```

Mã nguồn lúc chạy còn chưa commit; manifest ghi Git HEAD và SHA-256 của từng tệp code/config để nhận diện đúng bản thực thi. Thời gian được lưu bằng UTC; ngày trình bày ở tài liệu này theo múi giờ Asia/Saigon.
