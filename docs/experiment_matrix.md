# Ma trận thí nghiệm baseline — phiên bản 1

> **Tài liệu lịch sử, đã được thay thế.** Người dùng đã thu phạm vi về **một model**. Đọc [giao thức hiện hành](one_model_protocol.md) và [ma trận một model](one_model_matrix.json). Nội dung 15 ô dưới đây được giữ để truy vết quyết định cũ, không phải công việc cần tiếp tục; `experiment_matrix.json` cũng là bản lịch sử.

**Chốt ngày 04/11/2026, múi giờ Asia/Saigon.** Đây là quyết định triển khai trong phạm vi người dùng yêu cầu: chọn model/checkpoint, dataset, prompt, split, metrics và thứ tự chạy. Các số lượng dưới đây là lựa chọn của nhóm, không phải yêu cầu cứng của môn học.

**Phạm vi đã chốt: 3 checkpoint × 5 dataset = 15 ô full test.** Hai ô đã có kết quả hợp lệ; còn 13 ô cần triển khai/chạy. Mỗi ô mới có smoke test trước khi chạy đầy đủ. Việc chốt ma trận không có nghĩa các adapter mới đã được viết hoặc các thí nghiệm mới đã chạy.

Danh sách máy đọc được: [experiment_matrix.json](experiment_matrix.json). Tệp này là **kế hoạch**, không phải config đưa trực tiếp vào `python -m vlm_probe --config`. Hai config đang chạy được vẫn nằm trong `configs/` và giữ nguyên.

## 1. Câu hỏi nghiên cứu và lý do chọn

Ma trận trả lời: với cùng tập ảnh, tập lớp ứng viên và prompt của mỗi dataset, hiệu năng của ba checkpoint khác nhau ra sao trên các miền hình ảnh? Chúng ta cần một cơ sở so sánh đủ rộng trước khi chuyển sang phân tích lỗi tổng hợp và probing.

| ID | Model/checkpoint được chọn | Vai trò trong ma trận | Implementation dự kiến |
| --- | --- | --- | --- |
| M1 | CLIP ViT-B/32, trọng số OpenAI | Mốc gốc đã kiểm chứng; giữ nguyên hai kết quả hiện có | `open_clip_torch`, QuickGELU |
| M2 | CLIP ViT-B/32, `laion2b_s34b_b79k` | Cùng họ kiến trúc B/32, checkpoint/pretraining khác | `open_clip_torch`, GELU theo config checkpoint |
| M3 | SigLIP Base, patch 16, ảnh 224 | Thêm họ model/objective khác, vẫn là dual encoder | `transformers`, `google/siglip-base-patch16-224` |

M2 được huấn luyện trên LAION-2B English; M3 dùng WebLI English theo model card. Đây là so sánh **các hệ thống pretrained**, không phải thí nghiệm chỉ thay duy nhất dữ liệu hay hàm loss: activation, tokenizer, kiến trúc và quy trình pretraining cũng khác. Không suy ra nguyên nhân chênh lệch chỉ từ bảng accuracy. [Model card M2](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K), [model card M3](https://huggingface.co/google/siglip-base-patch16-224).

Không đưa CLIP bản lớn, thêm checkpoint thứ tư hoặc VLM sinh văn bản vào ma trận v1. Ba checkpoint này giữ quy mô Base và cùng đầu vào ảnh 224 × 224; mức bộ nhớ thực tế của M2/M3 phải được xác nhận ở smoke test. Bộ ảnh Việt Nam và prompt ngoài tiếng Anh thuộc giai đoạn extensions sau ma trận này.

## 2. Khóa checkpoint và cách nạp

| ID | Định danh bất biến | Tệp trọng số | SHA-256 |
| --- | --- | --- | --- |
| M1 | URL OpenAI đang có trong `configs/clip_b32_cifar100.json` | `ViT-B-32.pt` | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| M2 | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` @ `1a25a446712ba5ee05982a381eed697ef9b435cf` | `open_clip_model.safetensors` | `ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6` |
| M3 | `google/siglip-base-patch16-224` @ `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | `model.safetensors` | `2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8` |

Revision và SHA trọng số mới được lấy từ API metadata của kho phát hành. **Chưa tải/kiểm tra byte trọng số M2/M3**; lúc triển khai phải tải đúng revision, tính lại SHA-256 rồi mới sử dụng. Config/processor/tokenizer nhỏ đã được đọc để kiểm tra lựa chọn, hash được lưu ở [model_metadata_lock.json](references/model_metadata_lock.json). Không dùng nhánh `main` làm định danh thí nghiệm.

- **M1:** giữ nguyên preprocessing, tokenizer context 77, QuickGELU và cách encode hiện tại; không sửa kết quả cũ.
- **M2:** dùng `open_clip_config.json` tại revision đã khóa; GELU, context 77 và tokenizer CLIP từ phiên bản OpenCLIP đã ghim. Không kế thừa `force_quick_gelu=True` của adapter M1. Nạp đúng tệp dành cho OpenCLIP, không nhầm với `model.safetensors` dành cho implementation Transformers trong cùng repo. Resize/crop 224, bicubic và mean/std theo cấu hình OpenCLIP/checkpoint.
- **M3:** dùng model, tokenizer và image processor cùng revision. Resize 224 × 224, mean/std 0,5 theo processor. Text padding `max_length=64`; phát hiện prompt quá dài thay vì âm thầm cắt. Giữ processor/tokenizer riêng của model; không áp preprocessing CLIP lên SigLIP. Chọn attention implementation `eager` cho mốc đầu để giảm các biến số kernel; ghi rõ trong manifest. [Hướng dẫn SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip).

Chỉ dùng vector ảnh/text cuối cùng dành cho so khớp, chuẩn hóa L2, rồi lấy lớp có cosine lớn nhất. Với SigLIP, implementation hiện có tính logits từ cosine qua `exp(logit_scale)` dương và một bias chung; phép biến đổi này giữ thứ tự lớp trong số học chính xác. Adapter mới phải kiểm tra score/ranking so với forward gốc trên smoke test trong sai số FP32, điều tra mọi khác biệt trước full test. Không diễn giải cosine hoặc sigmoid score thành xác suất đã hiệu chuẩn, không so trung bình cosine giữa các model như một thước đo chất lượng.

## 3. Dataset, split và prompt cố định

| ID | Dataset | Lớp | Số ảnh full test | Split/target | Prompt duy nhất cho mọi model trên dataset |
| --- | --- | ---: | ---: | --- | --- |
| D1 | CIFAR-100 | 100 | 10.000 | `test`, fine labels | `a photo of a {class}.` |
| D2 | Oxford Flowers102 | 102 | 6.149 | `test`, `setid.mat:tstid` | `a photo of a {class}.` |
| D3 | Oxford-IIIT Pet | 37 | 3.669 | `test`, `target_types="category"` | `a photo of a {class}.` |
| D4 | DTD | 47 | 1.880 | `test`, **partition=1**, nhãn texture chính | `a photo of a {class} texture.` |
| D5 | Food-101 | 101 | 25.250 | `test` | `a photo of a {class}.` |

D1 là ảnh vật thể nhỏ; D2/D3 bổ sung hai miền phân biệt các loại gần giống nhau; D4 kiểm tra thuộc tính texture; D5 thêm miền món ăn. Bộ này cung cấp độ đa dạng ban đầu, không đại diện cho mọi miền ảnh và chưa thay thế các bài counting/spatial/negation.

DTD dùng từ chỉ thuộc tính như `banded`, nên chốt hậu tố `texture` trước khi chạy để prompt có ngữ nghĩa phù hợp. Đây là một template cố định riêng cho dataset, dùng giống nhau ở cả ba model; không phải chọn template tốt nhất bằng kết quả test. Prompt SigLIP cũng tuân theo bảng trên để so sánh cùng câu chữ. Do đó kết quả này không được tuyên bố là tái lập bảng benchmark dùng template mặc định khác của thư viện hoặc prompt ensemble.

### Quy tắc dữ liệu và ánh xạ lớp

1. Dùng dataset loaders của torchvision **0.29.1** trong `gen`, ghi build thực tế `0.29.1+cu126`. Hash file implementation tại thời điểm chốt ở [local_implementation_lock.json](references/local_implementation_lock.json).
2. Giữ nguyên danh sách `classes` và ID từ loader; chỉ thay `_` bằng dấu cách để tạo `prompt_name`. Không dịch, đổi từ đồng nghĩa, sửa a/an, thêm mô tả riêng từng lớp hoặc thay thứ tự sau khi đã xem accuracy. Lưu bảng class ID → tên gốc → prompt cho từng run; cùng dataset phải cho bảng giống nhau giữa các model.
3. D1/D2 giữ nguyên thứ tự baseline. D3 dùng thứ tự `annotations/test.txt`, nhãn gốc trừ 1, dùng toàn ảnh RGB (không crop theo head box/mask). D4 dùng thứ tự `labels/test1.txt`, lớp sắp xếp như loader. D5 dùng thứ tự ảnh của loader từ `meta/test.json`, tên lớp sắp xếp như loader. Hash danh sách ID/nhãn để kiểm tra không đổi giữa các model.
4. D1 có 100 ảnh/lớp; D4 có 40; D5 có 250. D2/D3 phải kiểm tra support theo nhãn nguồn, không giả định cân bằng. Không bỏ ảnh lỗi rồi vẫn gọi là full test; điều tra và ghi nhận thay đổi giao thức nếu thực sự phải thay tập ảnh.
5. Smoke dùng 200 index đầu theo đúng split và đủ mọi lớp ứng viên. Smoke chỉ kiểm tra kỹ thuật; có thể không chứa đủ lớp thật. Full dùng toàn bộ split, không subsampling, không shuffle, không bỏ batch cuối.
6. Test không dùng để chọn checkpoint, prompt, ngưỡng hay ensemble. Nếu cần debug bằng dữ liệu train/val, tách khỏi kết quả đánh giá. Khi chuyển sang tối ưu prompt sau này phải định nghĩa tập phát triển riêng. Không thể loại trừ trùng lặp với dữ liệu pretraining chỉ bằng quy trình zero-shot này.

**DTD v1 chỉ đánh giá partition 1**, không gọi đó là trung bình 10 partition. Các partition dùng lại nhiều ảnh, nên không gộp chúng thành các quan sát độc lập. Việc mở rộng đủ 10 partition là thí nghiệm bổ sung có phiên bản riêng.

Nguồn: [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Oxford Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), [số lượng split Pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet), [DTD và 10 partition](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

### Nguồn tải và tính toàn vẹn

| Dataset mới | Archive | MD5 gốc theo torchvision |
| --- | --- | --- |
| Oxford-IIIT Pet | `images.tar.gz` | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| Oxford-IIIT Pet | `annotations.tar.gz` | `95a8c909bbe2e81eed6a22bccdf3f68f` |
| DTD | `dtd-r1.0.1.tar.gz` | `fff73e5086ae6bdbea199a49dfb8a4c1` |
| Food-101 | `food-101.tar.gz` | `85eeb15f3717b99a5da872d97d918f87` |

URL nằm trong JSON kế hoạch. Food-101 có HEAD HTTP 200 khi kiểm tra. Nguồn Oxford Pet/DTD gặp timeout ở kiểm tra kết nối; [nhật ký](references/dataset_source_checks.json) ghi kết quả thực tế. HEAD thành công cũng chưa chứng minh tải hoàn tất hoặc dữ liệu đúng. Khi triển khai: thử lại nguồn gốc, hoặc dùng mirror cùng archive và bắt buộc khớp MD5 trên. Ghi nguồn tải thực tế, SHA-256 archive/metadata và xác minh ảnh đã giải nén. Không tự chuyển sang bản parquet có split hoặc ảnh được mã hóa lại; nếu không lấy được archive chuẩn thì báo trở ngại dữ liệu và giữ ô ở trạng thái chưa chạy. Lỗi tải không phải lý do để báo ma trận đã hoàn tất.

## 4. Metrics và điều kiện so sánh

| Chỉ số | Định nghĩa/điều kiện | Vai trò |
| --- | --- | --- |
| Top-1 accuracy theo ảnh | Số dự đoán đúng / tổng số ảnh | Metric chính trong mọi ô |
| Mean per-class accuracy | Trung bình accuracy của các lớp có support > 0; full test phải đủ mọi lớp | Metric phụ bắt buộc, đặc biệt cho Flowers/Pet |
| Per-class support/correct/accuracy | Đủ mọi lớp; smoke thiếu lớp thì accuracy để trống | Phát hiện chênh lệch theo lớp |
| Confusion matrix | Hàng = nhãn thật, cột = dự đoán | Nguồn cho phân tích sau baseline |
| Thời gian và bộ nhớ | `evaluation_seconds`, images/s, peak CUDA allocated, thêm `wall_seconds` | Thông tin vận hành, không phải benchmark latency GPU thuần |

Không cần thêm top-5 hoặc F1 trong ma trận v1. Giữ argmax cosine và quy tắc hòa chọn ID nhỏ nhất. Không đặt ngưỡng accuracy tối thiểu để coi một run là hợp lệ: tính hợp lệ phụ thuộc dữ liệu/giao thức, không phụ thuộc kết quả cao hay thấp.

Mean per-class accuracy có thể tính lại từ `per_class.csv` của hai run đã có, ghi vào báo cáo tổng hợp và trỏ về run gốc; không sửa `metrics.json`/manifest lịch sử. Khi triển khai metric mới cho run mới phải cập nhật verifier tương ứng. Trong bảng so sánh, báo cả hai metric cho từng dataset. Nếu cần một số tổng hợp, lấy trung bình không trọng số của 5 accuracy theo dataset và ghi rõ cách tính; không gộp tất cả ảnh để Food-101 chi phối kết luận.

Ma trận chính chạy **một lần full test hợp lệ mỗi ô**, deterministic seed 0. Đây không phải thí nghiệm huấn luyện nhiều seed. Nếu báo cáo kết luận chênh lệch nhỏ giữa model, bước phân tích sau ma trận phải dùng so sánh ghép cặp trên cùng ID ảnh và ước lượng bất định; không gọi chênh lệch nhỏ là có ý nghĩa chỉ từ một bảng phần trăm.

## 5. Ma trận và thứ tự triển khai

| Model | D1 CIFAR-100 | D2 Flowers102 | D3 Oxford Pet | D4 DTD p1 | D5 Food-101 |
| --- | --- | --- | --- | --- | --- |
| M1 OpenAI CLIP B/32 | Đã chạy: 64,48% | Đã chạy: 63,7665% | Kế hoạch | Kế hoạch | Kế hoạch |
| M2 LAION CLIP B/32 | Kế hoạch | Kế hoạch | Kế hoạch | Kế hoạch | Kế hoạch |
| M3 SigLIP B/16 | Kế hoạch | Kế hoạch | Kế hoạch | Kế hoạch | Kế hoạch |

1. **Mốc A — thêm model thứ hai:** viết adapter M2 và cơ chế cấu hình đa model; smoke → full D1, rồi smoke → full D2. Đối chiếu cùng ID ảnh/nhãn/prompt, kiểm tra checkpoint/activation đúng và kiểm tra M1 không bị đổi hành vi.
2. **Mốc B — thêm model thứ ba:** viết adapter M3; kiểm tra tokenizer/processor, vector 768 chiều và parity với forward gốc; smoke → full D1 rồi D2.
3. **Mốc C — thêm D3, D4, D5 theo thứ tự:** mỗi dataset mới được kiểm tra checksum/nhãn trước, rồi lần lượt chạy smoke → full M1, M2, M3. D5 tải nặng nhất nên đặt sau khi các adapter đã ổn định.
4. **Mốc D — tổng hợp:** đủ 15 ô, verifier đều qua, báo cáo 3 × 5 có cả top-1 và mean per-class, liên kết run và provenance. Sau mốc này mới chuyển trọng tâm sang phân tích lỗi tổng hợp/probing. Kiểm tra lỗi kỹ thuật vẫn diễn ra trong từng smoke/full run.

Hai ô M1–D1/M1–D2 tái sử dụng kết quả đã kiểm chứng, không tính lần refactor như một model hay thí nghiệm độc lập mới. Chi tiết run gốc và từng ô còn lại ở JSON kế hoạch.

## 6. Ngân sách máy, thời gian và quy tắc vận hành

- Môi trường **Conda `gen`**, NVIDIA RTX 4050 Laptop, tổng VRAM báo bởi driver **6.141 MiB**. Tại lúc rà soát, ổ D còn khoảng **64,6 GiB**. Không cần dịch vụ trả phí hoặc GPU thứ hai cho phạm vi này; tính khả thi bộ nhớ M2/M3 vẫn cần smoke thực tế.
- Chạy **một model/một tiến trình đánh giá GPU tại một thời điểm**, FP32, `eval()` + inference mode, TF32 tắt, deterministic bật, seed 0, 4 CPU threads, `num_workers=0`. Không training, không TTA, không ensemble. M1/M2 batch ảnh 32; M3 khởi đầu batch ảnh 16. Candidate text bao gồm đủ lớp, có thể encode theo chunk nếu bộ nhớ yêu cầu nhưng phải ghi chunk size và kiểm tra kết quả.
- Nếu OOM trong smoke: giảm batch ảnh theo 32 → 16 → 8 → 4 → 1, giữ FP32/checkpoint/prompt/split; M3 bắt đầu từ 16. Chọn batch bằng khả năng chạy, không bằng accuracy; ghi batch thực tế trong config. Không âm thầm đổi precision hoặc dùng CPU. Thời gian giữa model chỉ mang tính mô tả khi batch/preprocessing khác nhau.
- Dự trù **25 GiB dung lượng tăng thêm**, gồm archive và giải nén, weights mới, cache, kết quả và khoảng đệm; kiểm tra lại trước khi tải. Trọng số mới khoảng **1,32 GiB** tổng cộng theo metadata; Food-101 archive khoảng **4,65 GiB**, Pet khoảng 0,76 GiB và DTD khoảng 0,6 GiB theo nguồn. Tránh lưu nhiều bản sao weights/cache, đọc ảnh theo batch; không nạp toàn bộ ảnh đã decode vào RAM. Dung lượng giải nén là ước lượng cho kế hoạch, chưa đo trên máy.
- Mỗi model đánh giá **46.948 ảnh**, cả ma trận **140.844 lượt ảnh**; trừ hai ô cũ còn **124.695 lượt ảnh full mới**, cộng tối thiểu **2.600 lượt ảnh smoke** cho 13 ô mới. Không cộng smoke vào metric full test.
- Mốc thời gian đã đo: M1 CIFAR-100 khoảng **40,77 giây**, M1 Flowers102 khoảng **71,39 giây** cho phần evaluation; đỉnh CUDA allocated dưới 0,85 GiB trong hai run này. Thời gian tải/import/kiểm tra dữ liệu nằm ngoài evaluation. Các số đó **không phải dự đoán tốc độ SigLIP**.
- Dành **2–4 giờ vận hành cho một lượt điền các ô mới sau khi adapter đã sẵn sàng**, như ngân sách kế hoạch gồm tải/kiểm tra dữ liệu và suy luận, không phải benchmark hay cam kết. Thời gian viết/debug adapter tính riêng; timeout nguồn Oxford có thể làm thời gian tải vượt ngân sách. Sau smoke đầu mỗi model/dataset, dùng throughput và thời gian chuẩn bị thực đo để cập nhật ETA, không giảm số ảnh test vì một ước lượng thời gian chưa được kiểm chứng.

## 7. Đầu ra, phiên bản và điều kiện hoàn thành

Mỗi ô full mới phải có config hiệu lực, class mapping/prompts, danh sách sample ID/nhãn, predictions CSV, metrics, per-class CSV, confusion matrix, environment và manifest/checksum. Thêm model/checkpoint revision, processor/tokenizer config, embedding dimensions, source hashes và nguồn/split dữ liệu. Metadata của cùng dataset phải đối chiếu được giữa model.

Runtime hiện tại chỉ chấp nhận protocol v1/v2 và model M1. Giai đoạn kế tiếp cần bổ sung **protocol v3 cho các ô mới**; vẫn giữ khả năng đọc/chạy v1/v2 và verifier lịch sử. Không đổi ý nghĩa `protocol_version` cũ, không ghi config chưa chạy được vào README như lệnh sẵn có. Cần tách kiểm tra QuickGELU/metadata riêng M1 khỏi verifier chung trước khi hỗ trợ M2/M3.

Giai đoạn **chốt ma trận** hoàn thành khi tài liệu này, JSON 15 ô và nguồn khóa checkpoint nhất quán, đồng thời cập nhật phạm vi repo. Giai đoạn **chạy ma trận** chỉ hoàn thành khi đủ 15 ô hợp lệ và báo cáo tổng hợp; hiện có **2/15 ô**, không có accuracy dự đoán cho 13 ô còn lại. Thay đổi model, tập ảnh, partition, prompt hoặc precision phải tạo phiên bản kế hoạch/giao thức mới, ghi lý do và không thay đổi kết quả lịch sử.
