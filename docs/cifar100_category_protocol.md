# CIFAR-100 category v1 — một cặp prompt, chốt trước inference

Phạm vi được thu hẹp theo yêu cầu người dùng: chỉ OpenAI CLIP ViT-B/32 trên CIFAR-100. Giữ giao thức model/ảnh của baseline core, môi trường `gen`, FP32, QuickGELU, batch ảnh 32, text batch 100, seed 0. Chạy smoke 200 ảnh rồi full 10.000 ảnh test. Không chọn template theo kết quả smoke/test.

## Hai điều kiện

- Baseline: `a photo of a {class}.`
- Category: `a photo of a {class}, a type of {category}.`

Tên lớp giữ nguyên, chỉ thay `_` bằng khoảng trắng như baseline. Giữ nguyên mạo từ `a` để không đồng thời sửa cách đặt tên/grammar của baseline. Không ensemble, không thay model, không chạy các dataset khác trong phép thử này.

Ánh xạ 100 fine classes vào 20 coarse groups được suy ra từ **train** và `meta` chính thức sau kiểm tra MD5. Chỉ dùng ánh xạ lớp; không fit model, không chọn prompt bằng ảnh train. Nhãn thật của ảnh test không tham gia việc tạo prompt hoặc thu hẹp tập ứng viên: mọi ảnh luôn so với **đủ 100 lớp** trong cả hai điều kiện. Coarse group của nhãn thật chỉ dùng khi tổng hợp kết quả.

`vlm_probe/category_study/taxonomy.py` cố định 20 tên nhóm và cụm tiếng Anh đã đề xuất trong chat. Hai nhóm `vehicles_1/vehicles_2` cùng dùng `vehicle`, nên có 19 cụm category khác nhau. Không tự thay cấu trúc nhóm benchmark bằng phân loại sinh học khác. Mỗi run lưu bảng 100 lớp/nhóm/cả hai prompt trong `classes.json`.

## Scoring và tái lập

Encode hai bộ 100 prompt riêng; chuẩn hóa L2. Mỗi batch ảnh chỉ encode một lần và dùng chung vector ảnh cho hai điều kiện. Cosine và argmax trên đủ 100 lớp; exact tie chọn ID nhỏ nhất. Lưu cả hai ma trận cosine FP32 trong `scores.npz` (không pickle), nhãn/ID đầu vào, dự đoán, metrics, confusion matrices và các bảng thay đổi theo lớp/nhóm.

Baseline chạy lại phải khớp từng dự đoán với `full_cifar100_clip_b32_v1`; kiểm tra thêm cosine của lớp đúng/lớp được chọn và margin với sai số tuyệt đối tối đa 1e-6. Manifest ghi checkpoint, preprocessing, nguồn taxonomy, config, môi trường, hash artifact và source ZIP.

## Phân tích định trước

- Top-1, mean per-class accuracy cho từng điều kiện.
- Chênh lệch category − baseline theo điểm phần trăm.
- Số ảnh sai → đúng, đúng → sai, cả hai đúng, cả hai sai; số ảnh đổi dự đoán.
- Accuracy và thay đổi cho đủ 100 lớp và 20 coarse groups. Trong mỗi điều kiện, tách lỗi dự đoán sang cùng/khác coarse group.
- CI 95% của chênh lệch accuracy: paired bootstrap 1.000 lần, seed 0; resample ảnh trong từng fine class, giữ số mẫu mỗi lớp và dùng cùng index cho cả hai điều kiện. CI phản ánh lấy mẫu trên bộ test này, không bao quát biến thiên model/dataset khác.

Đây là một phép thử có thêm thông tin nhóm vào một template cố định; không chứng minh mọi cách dùng category đều hữu ích hoặc tách hoàn toàn tác động ngữ nghĩa khỏi độ dài/cấu trúc câu. Các mẫu/lớp cải thiện hoặc giảm được báo cáo đầy đủ. Kết quả không so trực tiếp với fine-grained oracle 5-way.

## Lệnh chạy

```powershell
conda activate gen
python -m vlm_probe.category_study --smoke --run-name smoke_cifar100_category_v1
python -m vlm_probe.category_study.audit results/smoke_cifar100_category_v1
python -m vlm_probe.category_study --run-name full_cifar100_category_v1
python -m vlm_probe.category_study.audit results/full_cifar100_category_v1
python -m vlm_probe.category_study.report results/full_cifar100_category_v1
```

Tên run phải mới khi chạy lại. Runner dùng adapter CIFAR-100 để chuẩn bị dữ liệu; audit cần dữ liệu CIFAR-100 cục bộ và baseline tham chiếu. Báo cáo được dựng tại `reports/cifar100_category_v1/`.
