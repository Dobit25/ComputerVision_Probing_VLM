# Probing Vision–Language Models — Topic 14

Mốc core được đóng gói trên nhánh **`dev`**, tag **`v1.0.0-core`**. Xem [nội dung phiên bản và cách quay lại mốc này](docs/core_release.md).

Dự án môn Computer Vision nhằm đánh giá cách mô hình kết hợp ảnh và ngôn ngữ (Vision–Language Model, VLM) phân loại ảnh **zero-shot** và tìm các trường hợp mô hình thường dự đoán sai. Với zero-shot classification, mô hình dùng trọng số đã học và so sánh ảnh với mô tả bằng chữ của các lớp; nhóm không huấn luyện lại mô hình trên bộ dữ liệu kiểm tra.

## Phạm vi

- **Yêu cầu cốt lõi của đề bài:** đánh giá CLIP hoặc một VLM mở trên nhiều bộ dữ liệu; kiểm tra có hệ thống các lỗi về đếm số lượng, quan hệ không gian, phủ định và phân biệt các loại gần giống nhau.
- **Hướng mở rộng:** nghiên cứu độ nhạy với cách viết prompt, gộp nhiều prompt, và đánh giá prompt ngoài tiếng Anh hoặc nội dung hình ảnh liên quan đến văn hóa Việt Nam.
- **Sản phẩm cuối:** mã nguồn và hướng dẫn tái lập kết quả, báo cáo 6–8 trang, bài trình bày 15 phút.

Chi tiết về nguồn đề bài, yêu cầu bắt buộc và các lựa chọn nhóm đang cân nhắc nằm trong [docs/assignment.md](docs/assignment.md). Nguồn gốc là [Topic 14 trong tài liệu môn học](https://github.com/NguyenMToan/CVCourse/blob/main/project_topics.md).

## Trạng thái hiện tại

Pipeline dùng chung evaluator cho **một model CLIP ViT-B/32 (trọng số OpenAI) → năm dataset**: CIFAR-100, Flowers102, Oxford-IIIT Pet, DTD partition 1 và Food-101, chọn bằng config. Phần nạp model/dataset đã tách thành adapter; xem [kiến trúc và cách mở rộng](docs/architecture.md), [giao thức hiện hành](docs/one_model_protocol.md) và [ma trận một model](docs/one_model_matrix.json).

Kết quả đo ngày 03/11/2026: **6.448/10.000 ảnh đúng — top-1 accuracy 64,48%**. Xem [bảng kết quả và kiểm chứng](results/README.md), [metrics gốc](results/full_cifar100_clip_b32_v1/metrics.json) và [dự đoán từng ảnh](results/full_cifar100_clip_b32_v1/predictions.csv).

Dataset thứ hai đã chạy đầy đủ: **Flowers102 — 3.921/6.149 ảnh đúng, top-1 accuracy 63,7665%**. [Metrics](results/full_flowers102_clip_b32_v2/metrics.json) và [biên bản kiểm chứng hai dataset](results/multi_dataset_validation.json).

Sau refactor, cả 10.000 dự đoán và cosine scores giữ nguyên; `predictions.csv` giống từng byte. Xem [biên bản đối chiếu](results/refactor_comparison.json). Hai dataset chưa hoàn thành toàn bộ Topic 14.

Theo điều chỉnh của người dùng ngày **04/11/2026**, phạm vi core là **một model × năm dataset và bốn bài core probing**. Kế hoạch ba checkpoint/15 ô đã được thay thế; đề bài không bắt buộc nhiều model. Xem [báo cáo baseline](reports/baseline/README.md) (accuracy, toàn bộ 387 lớp, các cặp nhầm lẫn) và [báo cáo probing](reports/probing/README.md) (counting, spatial, negation, fine-grained, đối chứng và khoảng tin cậy). Đã bổ sung phép thử một cặp prompt category trên CIFAR-100 bên dưới; nghiên cứu nhiều template/ensembling và tiếng Việt/văn hóa Việt Nam vẫn là các extensions tiếp theo.

Ba ô classification mới đã hoàn tất và vượt qua verifier:

| Dataset | Số đúng / ảnh test | Top-1 |
| --- | ---: | ---: |
| Oxford-IIIT Pet | 3.121 / 3.669 | 85,0641% |
| DTD partition 1 | 798 / 1.880 | 42,4468% |
| Food-101 | 20.784 / 25.250 | 82,3129% |

Full probing đã hoàn tất và qua audit: counting **32,87%**, spatial **29,58%**, negation **86,11%**, fine-grained **77,95%** (oracle 5-way). Cả 10.000 dự đoán global CIFAR-100 khớp baseline cũ. Đọc [diễn giải kết quả và giới hạn kết luận](reports/interpretation.md) trước khi dùng các số liệu trong báo cáo môn học.

Ba bài đếm/không gian/phủ định dùng hình học tự tạo có đối chứng; fine-grained dùng CIFAR-100 với nhãn nhóm lớn thật để giới hạn năm lớp ứng viên. Đây là chẩn đoán lỗi trong nhóm, không phải cải thiện zero-shot 100 lớp. Không suy rộng kết quả một checkpoint thành so sánh nhiều VLM. [Giao thức probing](docs/probing_protocol.md) được xác định trước inference.

## Chạy đánh giá

Phép thử **CIFAR-100 category v1** so sánh `a photo of a {class}.` với `a photo of a {class}, a type of {category}.`. Ánh xạ lấy từ 20 coarse groups của tập train; mỗi ảnh vẫn chọn giữa đủ 100 lớp. Kết quả full: **64,48% → 62,92%**, chênh **−1,56 điểm phần trăm**, CI 95% ghép cặp [−2,15; −0,96]. Xem [báo cáo category](reports/cifar100_category_v1/README.md) và [giao thức cố định trước khi chạy](docs/cifar100_category_protocol.md).

```powershell
conda run --no-capture-output -n gen python -m vlm_probe.category_study --run-name category_new
conda run --no-capture-output -n gen python -m vlm_probe.category_study.audit results/category_new
conda run --no-capture-output -n gen python -m vlm_probe.category_study.report results/category_new
```

Thêm `--smoke` để chỉ kiểm tra 200 ảnh đầu; tên run phải mới. Hai bộ prompt dùng chung vector ảnh trong mỗi batch. Lưu mọi cosine score để audit lại argmax trên 100 ứng viên. Phép thử này không dùng coarse label thật của ảnh đầu vào và không phải fine-grained oracle 5-way.

Mở terminal tại thư mục repository. Với môi trường Conda `gen` đã được thiết lập trên máy hiện tại, chọn config để chạy toàn bộ test split:

```powershell
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_cifar100.json
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_flowers102.json
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_oxford_pet.json
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_dtd.json
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_food101.json
```

Lệnh tự tải checkpoint và dataset nếu chưa có, kiểm tra checksum, chạy đánh giá trên GPU và ghi kết quả vào một thư mục mới có thời điểm UTC dưới `results/`. Lần sau dùng lại các tệp đã tải. Dữ liệu nằm trong `data/`, trọng số trong `models/`, cache trong `.cache/`; các thư mục này được Git bỏ qua.

Nếu đã `conda activate gen`, có thể dùng trực tiếp:

```powershell
python -m vlm_probe --config configs/clip_b32_flowers102.json
```

Chạy thử 200 ảnh trước khi chạy đầy đủ:

```powershell
conda run --no-capture-output -n gen python -m vlm_probe --config configs/clip_b32_flowers102.json --limit 200
```

Không truyền `--config` thì mặc định dùng CIFAR-100. Flowers102 có 6.149 ảnh test và 102 lớp; adapter dùng split gốc Oxford, tên lớp của torchvision và checksum dữ liệu gốc.

Tham số tùy chọn: `--run-name my_run` để đặt tên thư mục mới, `--batch-size 16` khi cần giảm bộ nhớ GPU, `--device cpu` để chủ động dùng CPU. Tên lần chạy đã tồn tại sẽ bị từ chối nhằm tránh ghi đè. Chương trình không tự giảm batch size hoặc đổi thiết bị; cấu hình thực tế được lưu cùng kết quả.

## Chạy core probing và dựng báo cáo

Probing dùng config cố định `configs/probing_clip_b32_v1.json`, CUDA, cùng checkpoint OpenAI. Cần artifact baseline tham chiếu tại `results/full_cifar100_clip_b32_v1` (đã có trong repo). Lệnh tự chuẩn bị CIFAR-100 và ảnh hình học.

```powershell
conda activate gen
python -m vlm_probe.probing --smoke --run-name probes_smoke_new
python -m vlm_probe.probing.audit results/probes_smoke_new
python -m vlm_probe.probing --run-name probes_full_new
python -m vlm_probe.probing.audit results/probes_full_new
```

Smoke lấy 64 mẫu mỗi bài, chỉ để kiểm tra luồng. Full có 13.240 mẫu: 1.080 counting + 720 spatial + 1.440 negation + 10.000 fine-grained. Negation có hai template cho mỗi ảnh nên chỉ có 720 ảnh duy nhất. Probing lưu định nghĩa cảnh/caption/nhãn trong `samples.json`, mọi cosine score trong `predictions.jsonl`, cùng config/environment/metrics/manifest. Audit kiểm tra dữ liệu nguồn, tính lại metrics và đối chiếu 10.000 dự đoán CIFAR-100 trên toàn bộ 100 lớp với baseline cũ.

```powershell
python -m vlm_probe.report baseline
python -m vlm_probe.report probing
```

Report baseline đọc năm full run đã định danh trong `vlm_probe/report.py`; report probing mặc định đọc `full_probing_clip_b32_v1`, có thể chọn run mới bằng `--run-name probes_full_new`. Báo cáo được dựng lại dưới `reports/`, không chạy model và không ghi đè artifact thí nghiệm. Probing audit cần ảnh tổng hợp và CIFAR-100 cục bộ. Run mới lưu source ZIP có checksum tại `results/source_snapshots/`.

Pet/DTD/Food dùng archive gốc qua mirror cố định revision, kiểm tra MD5 gốc và từng ảnh test đã giải nén. Food-101 tải archive khoảng 5 GB và giải nén cả 101.000 ảnh train/test, nhưng chỉ đánh giá 25.250 ảnh test. DTD chỉ dùng partition 1, không phải trung bình 10 partition.

## Thiết lập trên máy khác

Dùng Python 3.12 trong môi trường riêng của người chạy. Trên Windows có NVIDIA GPU và driver tương thích CUDA 12.6, cài cặp PyTorch/torchvision trước, sau đó các thư viện dự án:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Máy hiện tại đã có các gói này trong `gen`. `requirements.txt` ghi các phiên bản chính; `environment.json` của mỗi lần chạy ghi toàn bộ phiên bản thực tế. Không cần cài các gói ứng dụng khác có trong môi trường `gen`. Baseline dùng NumPy và thư viện chuẩn để ghi CSV/JSON, không dùng Pandas.

## Đọc và kiểm tra kết quả

Mỗi lần chạy hoàn tất chứa `metrics.json`, `predictions.csv`, `per_class.csv`, `confusion_matrix.csv`, `classes.json`, `samples.json`, `config.json`, `environment.json` và `manifest.json`. Chỉ dùng các lần chạy có `status: completed`. `samples.json` là phần bổ sung của các run mới, ghi ID/thứ tự/nhãn đầu vào. Ý nghĩa các tệp baseline được mô tả trong [giao thức](docs/experiment_protocol.md#tệp-kết-quả-và-tái-lập).

Kiểm tra kết quả baseline đã lưu bằng lệnh dưới đây. Với lần chạy mới, thay đường dẫn cuối bằng thư mục chương trình đã in ra:

```powershell
conda run --no-capture-output -n gen python -m vlm_probe.verify results/full_cifar100_clip_b32_v1
```

Bộ kiểm tra xác minh checksum, thứ tự/độ đầy đủ của ảnh và nhãn, rồi tính lại accuracy và các bảng đếm từ dự đoán đã lưu. Không cần chạy lại model cho bước này. Tests kiểm tra metric, evaluator với kích thước khác baseline, verifier với dữ liệu không cân bằng và các trường hợp đầu vào/artifact lỗi:

```powershell
conda run --no-capture-output -n gen python -m unittest discover -s tests -v
```

## Probing chuyên sâu: quan hệ không gian và category

**Bản tích hợp nghiên cứu ngày 11/11/2026:** [báo cáo tiếng Việt tối đa 10 trang](reports/research_review_v2/bao_cao_nghien_cuu.pdf), [bảng đa miền](reports/research_review_v2/multidomain_summary.csv), [review tích hợp](docs/review_hieu_integration_v2.md). Giao thức và các hạng mục thực thi được ghi ở [integration protocol v2](docs/integration_protocol_v2.md) và `results/integration_v2/execution_status.json`. Bản này bổ sung audit từ scores, prompt đa miền và chia vùng COCO; các run lịch sử được giữ nguyên.

Để giao thành viên kiểm nhãn và bản dịch: [hướng dẫn đầy đủ](docs/human_review_assignment_v2.md), [phiếu trắng và trạng thái](reviews/human_review_v2/). `python -m vlm_probe.human_review export` tạo gói ảnh/crop offline trong `data/human_review_v2/reviewer_package.zip`; gói ảnh không đưa vào Git. Chưa có phiếu người thì không gọi dữ liệu là human-validated.

Trong `gen`, `python -m vlm_probe.integration --execute` chạy hàng đợi smoke → full → audit với run name cố định; run đã hoàn thành được audit lại và không ghi đè. `python -m vlm_probe.review_report` sinh LaTeX và bảng từ artifact đã kiểm. Biên dịch theo README trong thư mục báo cáo.

**Cập nhật mới nhất — xác nhận trên ảnh chưa dùng:** đã tự rà soát lại 200 ảnh cũ (giữ 176), rà soát 178 ứng viên mới và cố định 100 nguồn không trùng 400 ảnh từng xem trước suy luận. Ghép cặp T2 thuận đạt 53/100, CI Wilson 95% [43,29%; 62,49%], p một phía = 0,3086. [PDF cập nhật](reports/natural_confirmation_v1/bao_cao_cap_nhat.pdf), [kết quả/giới hạn/tái lập](reports/natural_confirmation_v1/README.md), [giao thức](docs/natural_confirmation_protocol.md). Chưa có annotator người độc lập; không diễn giải kết quả như bằng chứng tương đương đoán ngẫu nhiên.

Đã hoàn thành phần bổ sung với cùng checkpoint CLIP ViT-B/32: 1.440 ảnh hình học/8.640 phép thử; tám điều kiện trên 10.000 ảnh CIFAR-100; 200 ảnh COCO gốc và bản lật/2.400 phép thử. Có giao thức, scores từng mẫu, audit và báo cáo tiếng Việt.

- [Báo cáo PDF mốc probing chuyên sâu](reports/probing_depth_v1/bao_cao_probing_chuyen_sau.pdf) và [kết quả/diễn giải đầy đủ](reports/probing_depth_v1/README.md).
- [Giao thức và amendment nguồn ảnh](docs/probing_depth_protocol.md), [config](configs/probing_depth_v1.json), [quy trình rà soát ảnh tự nhiên](docs/natural_depth_review.md).
- Mã mới ở `vlm_probe/depth/`; các run cố định ở `results/full_depth_{historical,spatial,category,natural}_v1`.
- Trong môi trường `gen`: `python -m vlm_probe.depth audit` kiểm tra toàn bộ run; `python -m vlm_probe.depth report` dựng lại bảng/biểu đồ. [Lệnh chạy lại vào thư mục mới](reports/probing_depth_v1/README.md#7-mã-nguồn-kiểm-chứng-và-chạy-lại).

Phần tự nhiên đã được rà soát bằng trợ lý AI và validator hình học, chưa có người gán nhãn độc lập. Kết quả không được diễn giải như cơ chế nội tại của mọi VLM. Các phần bổ sung được đóng gói trong mốc `v1.1.0-probing-depth` trên `dev`; xem [phạm vi phiên bản và hướng dẫn kiểm tra](docs/probing_depth_release.md). Tag `v1.0.0-core` tiếp tục giữ nguyên mốc core trước đó.

## Cấu trúc thư mục

```text
configs/clip_b32_cifar100.json  # Baseline v1: 100 lớp, 10.000 ảnh test
configs/clip_b32_flowers102.json # Protocol v2: 102 lớp, 6.149 ảnh test
vlm_probe/model_adapters/      # Checkpoint, model, tokenizer, preprocessing
vlm_probe/dataset_adapters/    # Ảnh, nhãn, tên lớp, split và ID
vlm_probe/evaluate.py          # Evaluator dùng chung
vlm_probe/verify.py            # Kiểm tra artifact cũ và mới
vlm_probe/compare.py           # Đối chiếu từng dự đoán giữa hai run
tests/                        # Kiểm tra evaluator, giao thức và metrics
docs/assignment.md            # Yêu cầu môn học và lựa chọn của nhóm
docs/experiment_protocol.md   # Giao thức baseline v1
docs/multi_dataset_protocol.md # Flowers102 và kiểm chứng refactor
docs/architecture.md          # Kiến trúc và hướng dẫn thêm adapter
docs/one_model_protocol.md    # Phạm vi hiện hành: một model, năm dataset
docs/one_model_matrix.json    # Ma trận hiện hành; không phải config CLI
docs/experiment_matrix.*      # Kế hoạch lịch sử ba model, đã thay thế
docs/probing_protocol.md      # Giao thức bốn bài probing và giới hạn kết luận
vlm_probe/probing/            # Sinh cảnh, evaluator, metrics/CI và audit
vlm_probe/report.py           # Dựng CSV, PNG và Markdown từ artifact đã kiểm chứng
reports/                     # Báo cáo baseline và probing
docs/references/              # Nguồn, revision/checksum và kiểm tra khả dụng
results/                      # Kết quả nhỏ có thể đưa vào Git
requirements.txt              # Phiên bản các thư viện chính
```

Dữ liệu tải về và trọng số model được lưu cục bộ, không đưa vào Git. Các bảng kết quả nhỏ và tài liệu phân tích có thể được lưu trong repository để hỗ trợ tái lập.
