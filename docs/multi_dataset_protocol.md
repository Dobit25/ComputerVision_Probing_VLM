# Giao thức nhiều dataset và kiểm chứng refactor

Đây là các lựa chọn triển khai của nhóm cho mốc mở rộng, không phải yêu cầu bổ sung từ đề bài. [Giao thức CIFAR-100 v1](experiment_protocol.md) vẫn giữ nguyên. [Kiến trúc và vị trí mở rộng](architecture.md) mô tả các adapter.

## Hai cấu hình

| Thuộc tính | CIFAR-100 | Oxford Flowers102 |
| --- | --- | --- |
| Config | `configs/clip_b32_cifar100.json` | `configs/clip_b32_flowers102.json` |
| Protocol | v1 | v2 |
| Split | test, fine labels | test, trường `tstid` trong `setid.mat` |
| Số lớp | 100 | 102 |
| Số ảnh full test | 10.000 | 6.149 |
| Số ảnh mỗi lớp | 100 | Không bằng nhau; ghi đầy đủ trong manifest |
| Smoke | 200 index đầu | 200 index đầu của `tstid`, giữ đúng thứ tự file |
| ID ảnh | `cifar100/test/00000` | `flowers102/test/image_XXXXX.jpg` |

Cả hai dùng CLIP ViT-B/32 pretrained OpenAI, checkpoint SHA-256 cố định, QuickGELU, FP32, CUDA, batch size 32, seed 0, một prompt `a photo of a {class}.`. Không điều chỉnh prompt theo kết quả test. Mỗi ảnh được so với **toàn bộ lớp của dataset**, kể cả smoke.

Metric chính là **top-1 accuracy theo ảnh**: tổng số dự đoán đúng chia tổng số ảnh. `per_class.csv` chứa accuracy của từng lớp. Với Flowers102, trung bình accuracy các lớp và accuracy theo ảnh là hai phép tính khác nhau vì lớp không cân bằng; phải ghi rõ metric khi so sánh công trình khác. Smoke chỉ kiểm tra kỹ thuật, không đại diện chất lượng toàn bộ dataset.

## Kiểm tra Flowers102

- Nguồn gốc: [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). Implementation: [torchvision Flowers102](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html).
- Cấu hình dùng mirror HTTPS `paddlemodels.bj.bcebos.com/flowers/` do máy không kết nối được tới máy chủ ảnh Oxford tại thời điểm triển khai. Các URL và checksum có trong [mã nguồn PaddlePaddle](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/datasets/flowers.py). Chỉ dùng mirror tệp; cách chia train/test luôn theo **Oxford/torchvision**, không dùng cách hoán đổi train/test trong loader của Paddle.
- Ba tệp bắt buộc khớp MD5 gốc trước khi sử dụng:

| Tệp | MD5 |
| --- | --- |
| `102flowers.tgz` | `52808999861908f626f3c1f4e79d11fa` |
| `imagelabels.mat` | `e0620be6f572b9609742df49c70aed4d` |
| `setid.mat` | `a5357ecc9cb78c4bef273ce3793fc85c` |

Loader xác minh ba split có lần lượt 1.020/1.020/6.149 ảnh, không trùng nhau và phủ đủ ID 1–8.189. Nhãn gốc 1–102 được chuyển về 0–101; danh sách lớp lấy từ `torchvision.datasets.Flowers102.classes` của phiên bản đã ghim. Giữ nguyên cách viết tên lớp của thư viện, kể cả dấu câu, để tránh tự sửa nhãn sau khi xem kết quả. `classes.json` lưu chính xác prompt cuối cùng; manifest ghi nguồn và hash danh sách tên lớp.

`test_index` là vị trí trong `tstid`; ID gốc nằm trong `sample_id`. Adapter đối chiếu lại thứ tự đường dẫn/nhãn của torchvision với các tệp `.mat`. Tất cả **6.149 ảnh test đã giải nén** được so từng byte qua SHA-256 với bản trong archive đã kiểm tra MD5, kể cả khi chỉ chạy smoke. Vì vậy ảnh local bị sửa sẽ làm lần chạy thất bại. Manifest lưu SHA-256 ba tệp nguồn và digest của danh sách tên ảnh/hash ảnh test. Không dùng ảnh train/val để huấn luyện hoặc chọn prompt.

Thư viện `scipy==1.18.0` đọc tệp `.mat`; đã có sẵn trong `gen` và được bổ sung vào `requirements.txt`. Không thay các gói đang cài trong môi trường.

## Lệnh chạy và kiểm tra

Từ thư mục repository:

```powershell
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_cifar100.json
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_flowers102.json
```

Thêm `--limit 200` để chạy smoke. Không có `--limit` nghĩa là chạy toàn bộ test split. Chương trình in thư mục kết quả; thay đường dẫn đó trong lệnh audit:

```powershell
conda run --no-capture-output -n gen python -m vlm_probe.verify results/full_flowers102_clip_b32_v2
```

Mỗi lần chạy mới có chín tệp: `config.json`, `classes.json`, `samples.json`, `predictions.csv`, `metrics.json`, `per_class.csv`, `confusion_matrix.csv`, `environment.json`, `manifest.json`. Số hàng/cột được xác định theo dataset. Verifier kiểm tra giới hạn smoke, số lớp/số ảnh theo giao thức, thứ tự ảnh, nhãn, score, confusion matrix, support và hash. Nó vẫn đọc được tám tệp của baseline v1.

## Kiểm chứng tổ chức lại code

```powershell
conda run --no-capture-output -n gen python -m vlm_probe.compare results/full_cifar100_clip_b32_v1 results/full_cifar100_clip_b32_refactor
```

Lệnh xác minh hai run, yêu cầu cùng config thực tế, môi trường, class mapping, prompt, checkpoint, preprocessing và provenance dữ liệu. Sau đó so ID/nhãn của từng ảnh, nhãn dự đoán, chênh lệch cosine và hash CSV. Hai run có cùng accuracy nhưng sai ở các ảnh khác nhau sẽ không vượt qua kiểm tra này. Có thể thêm `--output results/ten_moi.json` để lưu báo cáo; không ghi đè tệp đã có.

Kết quả đã lưu ở [refactor_comparison.json](../results/refactor_comparison.json): đủ 10.000 ảnh, **0 dự đoán thay đổi**, mọi score lưu trong CSV có chênh lệch **0**, và hai CSV giống từng byte. Accuracy giữ nguyên **64,48%**. Đây là bằng chứng cho lần chạy trên cùng hệ thống; không cam kết score giống từng bit trên GPU hoặc phiên bản thư viện khác.
