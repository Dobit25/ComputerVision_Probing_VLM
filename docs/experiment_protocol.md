# Giao thức baseline v1: CLIP trên CIFAR-100

Giao thức này được thống nhất trước khi chạy baseline. Cấu hình máy đọc được nằm tại [configs/clip_b32_cifar100.json](../configs/clip_b32_cifar100.json). Giao thức baseline đầu tiên được giữ nguyên khi refactor. Pipeline đã mở rộng sang dataset thứ hai theo [giao thức nhiều dataset](multi_dataset_protocol.md); các bài probing failure mode vẫn là công việc tiếp theo của Topic 14.

Config v1 và kết quả baseline ban đầu được giữ nguyên. Code hiện tại dùng adapter và tạo artifact schema v2, bổ sung `samples.json` cùng thông tin kích thước động. Verifier đọc được cả kết quả cũ lẫn mới; xem [kiến trúc](architecture.md).

## Model và dữ liệu

- Model: CLIP ViT-B/32, trọng số pretrained OpenAI. Dùng thư viện OpenCLIP làm implementation, bật QuickGELU để khớp checkpoint. Không cập nhật trọng số.
- Checkpoint chính thức: URL và SHA-256 cố định trong config. Pipeline xác minh hash trước khi tải trọng số vào model. Checkpoint này là TorchScript archive; việc đọc bằng `weights_only=False` chỉ diễn ra sau bước xác minh hash của tệp chính thức.
- Dataset: CIFAR-100, 100 fine classes, test split có 10.000 ảnh (100 ảnh mỗi lớp). `torchvision` kiểm tra checksum dữ liệu. Archive tải về chứa cả train và test; pipeline chỉ đánh giá test. Do nguồn Toronto tải chậm trong lần thiết lập, config dùng mirror được PaddlePaddle sử dụng tại `dataset.bj.bcebos.com`. Archive phải khớp MD5 gốc `eb9058c3a382ffc7106e4002c42a8d85`; các tệp sau giải nén cũng được kiểm tra. URL nguồn gốc và URL tải thực tế đều được lưu. Có thể đổi `dataset_download_url` về URL Toronto trong config mà không thay đổi bộ dữ liệu.
- Giữ nguyên thứ tự `dataset.classes` tương ứng ID 0–99. Chỉ thay dấu `_` trong tên lớp bằng dấu cách khi tạo prompt; lưu cả tên gốc, tên trong prompt và ID trong `classes.json`.
- Prompt duy nhất: `a photo of a {class}.` Giữ nguyên cách viết này cho cả danh từ số nhiều và từ bắt đầu bằng nguyên âm; không tối ưu prompt sau khi nhìn kết quả test.

## Xử lý và dự đoán

1. Dùng phép biến đổi đánh giá của model: resize bicubic theo cạnh ngắn, center crop 224 × 224, chuyển RGB, chuyển tensor và chuẩn hóa theo mean/std của checkpoint. Lưu mô tả phép biến đổi thực tế trong manifest.
2. Dùng tokenizer CLIP đi kèm kiến trúc (context length 77). Encode 100 prompt theo đúng thứ tự ID một lần.
3. Bật `eval()` và `inference_mode()`, chạy FP32, chuẩn hóa L2 vector ảnh và text. Mỗi vector có 512 chiều.
4. Tính cosine similarity bằng tích vô hướng giữa các vector đã chuẩn hóa. Dự đoán là ID có score cao nhất; nếu bằng nhau tuyệt đối, chọn ID nhỏ nhất. Không dùng softmax; cosine score không được diễn giải như xác suất đã hiệu chuẩn.
5. So sánh với fine label của ảnh. Top-1 accuracy = số dự đoán đúng / số ảnh được đánh giá. Accuracy theo lớp dùng số ảnh thật của lớp làm mẫu số. Confusion matrix có hàng là lớp thật, cột là lớp dự đoán.

Không dùng nhãn test để huấn luyện hoặc chọn prompt. Nếu sau này nghiên cứu tối ưu prompt, cần một tập phát triển riêng và giao thức mới. Việc checkpoint có từng nhìn thấy ảnh tương tự trong pretraining không thể được loại trừ chỉ bằng baseline này; “zero-shot” mô tả cách nhóm thực hiện đánh giá.

## Cấu hình chạy và kiểm tra

Mặc định dùng CUDA trong môi trường Conda `gen`, batch size 32, seed 0, 4 CPU threads và DataLoader không tạo tiến trình con (`num_workers=0`) để phù hợp Windows. FP32, TF32 tắt, deterministic algorithms bật, không shuffle, không bỏ batch cuối. Có thể đổi batch size hoặc dùng CPU bằng CLI; thay đổi được ghi vào config của lần chạy. Nếu CUDA không khả dụng, chương trình báo lỗi thay vì tự đổi thiết bị.

Smoke test dùng **200 ảnh đầu tiên theo thứ tự test gốc**; chỉ kiểm tra kỹ thuật, không dùng để chọn cấu hình. Lớp không xuất hiện trong smoke có support 0 và accuracy để trống. Full test đánh giá toàn bộ 10.000 ảnh theo cùng thứ tự. Mọi lần chạy dùng đủ 100 nhãn ứng viên.

Pipeline kiểm tra checksum, thứ tự nhãn, số mẫu mỗi lớp, kích thước và tính hữu hạn của vector, chuẩn hóa L2 và thứ tự ảnh trong DataLoader. Các kiểm tra không phụ thuộc ngưỡng accuracy kỳ vọng.

## Tệp kết quả và tái lập

Mỗi lần chạy tạo một thư mục mới dưới `results/`, không ghi đè lần cũ:

| Tệp | Nội dung |
| --- | --- |
| `config.json` | Cấu hình thực tế sau khi áp dụng tham số CLI |
| `classes.json` | 100 ID, tên lớp gốc, tên đã xử lý và prompt |
| `predictions.csv` | ID ảnh, index gốc, nhãn thật/dự đoán, đúng/sai, cosine dự đoán, cosine nhãn thật, khoảng cách top-1 và top-2 |
| `metrics.json` | Top-1 accuracy, số ảnh, số đúng, thời gian đánh giá và bộ nhớ GPU được PyTorch cấp phát |
| `per_class.csv` | Support, số đúng và accuracy của từng lớp |
| `confusion_matrix.csv` | Ma trận nhầm lẫn 100 × 100 với thứ tự ID tường minh |
| `environment.json` | Python, thư viện đã cài, CUDA/cuDNN, GPU, driver và các thiết lập tính toán |
| `manifest.json` | Trạng thái chạy, thời điểm UTC, lệnh, Git HEAD/trạng thái, SHA-256 code/config/trọng số/dữ liệu/đầu ra, preprocessing và kích thước embedding |

`evaluation_seconds` bao gồm encode text, tiền xử lý và nạp batch ảnh, suy luận, truyền kết quả về CPU và ghi predictions CSV; không gồm tải tài nguyên hoặc tải model. `wall_seconds` trong manifest bao gồm toàn bộ tiến trình bên trong runner. Đây không phải benchmark latency thuần GPU.

Thư mục có trạng thái `failed` hoặc `running` không được dùng làm kết quả hoàn chỉnh. Lệnh kiểm tra artifact tính lại accuracy và các bảng đếm từ CSV, đối chiếu checksum và độ đầy đủ; không cần GPU. Seed và các tùy chọn deterministic giúp tái lập trên cùng hệ thống; không cam kết score giống từng bit giữa các phiên bản thư viện hoặc phần cứng khác nhau.

Các phiên bản chính được ghim trong `requirements.txt`; snapshot trong mỗi lần chạy ghi thêm phiên bản thực tế của mọi gói trong `gen`. Snapshot có thể chứa gói phục vụ dự án khác, vì vậy không cài lại toàn bộ môi trường `gen` để chạy baseline. Nếu Git còn thay đổi chưa commit, dùng các hash mã nguồn trong manifest để nhận diện bản code đã chạy.

## Nguồn kỹ thuật

- [OpenCLIP: tải model, tokenizer, preprocessing và QuickGELU](https://github.com/mlfoundations/open_clip)
- [Torchvision CIFAR100](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html)
- [CIFAR-100: trang dữ liệu gốc](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Nguồn mirror trong mã PaddlePaddle](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/datasets/cifar.py)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
