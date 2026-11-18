# Mốc core v1.0.0-core

Phiên bản này đóng gói phạm vi **một model OpenAI CLIP ViT-B/32**, năm dataset classification và bốn bài core probing. Nhánh phát triển là `dev`; tag có chú thích `v1.0.0-core` xác định mốc cố định trước khi triển khai extensions về prompt. Tài liệu `experiment_matrix.md/json` được giữ làm lịch sử và đã được thay thế bởi `one_model_protocol.md` cùng `one_model_matrix.json`.

## Nội dung của mốc

- Pipeline, model/dataset adapters, config, verifier và tests.
- Classification trên CIFAR-100, Flowers102, Oxford-IIIT Pet, DTD partition 1 và Food-101: 46.948 ảnh test tổng cộng.
- Counting, spatial, negation và fine-grained: 13.240 mẫu đánh giá, trong đó negation dùng hai template trên 720 ảnh.
- Dự đoán, cosine scores của probing, metrics, dữ liệu mô tả mẫu, cấu hình, môi trường, checksum và các source ZIP của từng mốc thực thi.
- Báo cáo [baseline](../reports/baseline/README.md), [probing](../reports/probing/README.md) và [diễn giải](../reports/interpretation.md).

Ảnh dataset, model weights, cache và mảnh tải tạm nằm ngoài Git theo `.gitignore`. Các source ZIP chỉ chứa mã nguồn/cấu hình phục vụ tái lập. `.gitattributes` giữ nguyên byte của tệp để checkout trên Windows hoặc hệ điều hành khác không tự đổi LF/CRLF và làm sai checksum.

Các manifest cũ có thể ghi Git HEAD tại thời điểm chạy trước khi code được commit. Giữ nguyên thông tin lịch sử đó; dùng source hash và ZIP đi kèm để xác định mã thực sự đã chạy. Tag này đóng gói kết quả và bộ công cụ hiện tại, không viết lại provenance của run cũ.

## Quay lại mốc này

```powershell
git fetch origin --tags
git switch --detach v1.0.0-core
```

Nếu cần phát triển từ mốc này, tạo nhánh mới bằng `git switch -c ten-nhanh v1.0.0-core`. Nhánh `dev` sẽ tiếp tục thay đổi; tag là điểm tham chiếu cố định.

## Kiểm tra và tái lập

Dùng môi trường `gen` trên máy hiện tại, hoặc cài môi trường theo [README](../README.md).

```powershell
conda activate gen
python -m unittest discover -s tests -v
python -m vlm_probe.verify results/full_food101_clip_b32_v3
python -m vlm_probe.report baseline
```

Các verifier classification đọc artifact đã lưu và không cần dataset/model. Audit probing kiểm tra cả ảnh tổng hợp và CIFAR-100 cục bộ:

```powershell
python -m vlm_probe.probing.audit results/full_probing_clip_b32_v1
```

Trên máy mới, làm theo README để chuẩn bị dữ liệu/chạy một run probing mới trước khi audit. Tên run mới phải khác các thư mục đã có. CI và số liệu trong report được tính từ dữ liệu thực nghiệm đã lưu, không phải số liệu dự đoán.

Đây là mốc core theo giao thức hiện hành: ba bài probing hình học có đối chứng, fine-grained có nhãn coarse oracle. Prompt sensitivity/ensembling đầy đủ, prompt tiếng Việt, ảnh văn hóa Việt Nam và báo cáo hội nghị/slide cuối kỳ nằm sau mốc này.
