# Kế hoạch toàn dự án Topic 14 — từ mốc hiện tại đến bàn giao

Ngày lập: **05/11/2026**, Asia/Saigon. Điểm xuất phát: `v1.1.0-probing-depth`, nhánh `dev`, commit `dcc2184`.

Đây là **đề xuất kế hoạch**, chưa phải giao thức được đóng băng cho các thí nghiệm tương lai và không phải danh sách công việc đã thực hiện. Không chạy thí nghiệm mới, đổi model hoặc fine-tuning khi lập tài liệu này. Lịch tương đối tám tuần là giả định để lập kế hoạch; cần thay bằng hạn nộp và nhân lực thực tế. Thời lượng là ước lượng giờ công của cả nhóm, không phải số giờ GPU đã đo.

## 1. Đích đến và các mức phạm vi

Mục tiêu khoa học: mô tả có bằng chứng khả năng và giới hạn của một checkpoint CLIP trong phân loại bằng ngôn ngữ, suy luận quan hệ và phản ứng với thay đổi prompt. Thành công là trả lời đúng câu hỏi bằng thiết kế hợp lệ; không phụ thuộc accuracy tăng hoặc p-value nhỏ.

Nguồn yêu cầu là [assignment](assignment.md), đối chiếu `project_topics.md` của môn học, mục 14 và Deliverables. Kế hoạch nhiều model và các nhận định “chưa ai nghiên cứu” trong tài liệu ban đầu là gợi ý lịch sử, không tự động trở thành yêu cầu hiện hành.

| Mức | Nội dung | Điều kiện đóng dự án |
|---|---|---|
| A — hoàn tất phạm vi môn học | Năm baseline, bốn bài probing, diễn giải đúng giới hạn; repo tái lập; report 6–8 trang; trình bày 15 phút + Q&A | Đối chiếu rubric/giảng viên; đủ deliverable, không cần tăng accuracy |
| B — bản mở rộng khuyến nghị | Mức A + nghiên cứu prompt/ensemble có giao thức + đánh giá tiếng Việt; củng cố độ tin cậy của nhánh không gian trong ngân sách | Báo cả kết quả thuận/nghịch/không rõ; không bắt buộc làm dataset văn hóa riêng |
| C — sau môn học hoặc hướng bài báo | Dataset văn hóa có chất lượng, kiểm chứng độc lập bổ sung, hoặc thêm một checkpoint để kiểm tra khái quát | Cần câu hỏi mới, ngân sách riêng, khảo sát tài liệu và trao đổi với giảng viên |

Extension đề bài là prompt sensitivity, ensembling, non-English prompts **hoặc** Vietnamese cultural content. Không biến mọi nhánh tùy chọn thành điều kiện bắt buộc để kết thúc. Một checkpoint vẫn là phạm vi mặc định; không mở lại ma trận nhiều model nếu chưa có quyết định mới.

## 2. Hiện trạng đã kiểm chứng

| Hạng mục | Bằng chứng đã có | Việc cần giữ rõ |
|---|---|---|
| Classification | 5 dataset, 46.948 ảnh; CIFAR-100 64,48%, Flowers102 63,77%, Pet 85,06%, DTD partition 1 42,45%, Food-101 82,31% | Độ khó và số lớp khác nhau; không dùng chênh accuracy giữa dataset như hiệu ứng model |
| Counting core | 32,87%, 6 lựa chọn; đối chứng nhận dạng 74,44% | Khó khăn nhận dạng có thể góp phần; hai chế độ diện tích không loại hết shortcut |
| Spatial core/depth | Core 29,58%; depth 27,36% ở điều kiện chính, nhận dạng đối chứng 96,94% | Ghi đúng phiên bản, mẫu số và template; không trộn các accuracy |
| Negation core | 86,11%; hai template 80,83% và 91,39% | Model làm tương đối tốt trong thiết kế này; không ép câu chuyện “không hiểu phủ định” |
| Fine-grained | CIFAR-100 oracle 5-way 77,95%; phân tích lỗi Flowers/Pet/Food | Có nhãn coarse thật; không phải cải thiện 100-way; nhóm CIFAR không đồng đều độ gần nhau |
| Category | 64,48% → 62,92%; có neutral và năm shuffle | Category đúng tốt hơn category sai không có nghĩa tốt hơn baseline |
| Ảnh tự nhiên | 200 nguồn cũ; tự rà soát giữ 176; 100 nguồn xác nhận mới không trùng 400 ứng viên cũ | Cùng AI reviewer và cùng miền COCO; chưa có người rà soát độc lập |
| Ghép cặp mới | 53/100; CI Wilson [43,29%; 62,49%]; p = 0,3086; cả hai ảnh đúng 4% | Chưa đủ bằng chứng vượt 50%; không chứng minh tương đương ngẫu nhiên |
| Phát hành | `v1.1.0-probing-depth`; 43 tests, 11 audit; kiểm checksum từ clone mới | PDF hiện tại 3 trang là báo cáo tiến độ, không thay report cuối kỳ 6–8 trang |

Xem [core](../reports/probing/README.md), [depth](../reports/probing_depth_v1/README.md), [confirmation](../reports/natural_confirmation_v1/README.md), [release](probing_depth_release.md). Không tự suy ra proposal, milestone report hoặc presentation đã được nộp chỉ vì file đã tồn tại.

## 3. Quy tắc ra quyết định áp dụng cho mọi task

Mỗi task mới có: ID, câu hỏi, dữ liệu đã xem/chưa xem, điều kiện, metric chính, mẫu số, số đơn vị độc lập, ngân sách, người phụ trách, tiêu chí hoàn thành và nhánh xử lý. Giao thức phải được commit trước inference xác nhận mới; mọi sửa đổi sau đó có amendment và lý do.

| Kết quả task | Việc phải làm | Việc không được làm |
|---|---|---|
| Lỗi kỹ thuật hoặc nhãn không hợp lệ | Dừng diễn giải, xác định phạm vi ảnh hưởng, sửa và chạy lại thành run mới | Giữ kết quả lỗi làm bằng chứng; sửa âm thầm artifact cũ |
| Kết quả ủng hộ giả thuyết | Kiểm effect size, CI, control, độ bền; xác nhận mới nếu cần tuyên bố khái quát | Đổi metric chính thành metric đẹp nhất |
| Kết quả đi ngược giả thuyết | Báo kết quả; kiểm giả thuyết thay thế bằng tối đa một vòng tiếp nối đã lập trước | Loại lớp, seed hoặc prompt làm xấu trung bình |
| Chưa rõ vì CI rộng | Nêu mức bất định; lập nghiên cứu mới với cỡ mẫu dựa trên độ chính xác/hiệu ứng cần phát hiện | Thêm từng batch và kiểm p liên tục đến khi “đạt” |
| Khác nhau theo nhóm | Báo tất cả nhóm; coi phân tích phát hiện sau khi xem là thăm dò | Chỉ công bố nhóm cải thiện |
| Không đủ dữ liệu/thời gian | Thu hẹp tuyên bố, chuyển sang phương án dự phòng hoặc dừng nhánh | Bù bằng dữ liệu không rõ nguồn hoặc kết luận mạnh hơn bằng chứng |

Đề xuất giới hạn mặc định cho mỗi nhánh: **một lượt thí nghiệm chính + tối đa một lượt tiếp nối để phân biệt cách giải thích**. Không áp dụng trần này cho việc sửa lỗi kỹ thuật bắt buộc. Nếu vẫn chưa rõ, ghi kết luận chưa đủ bằng chứng và đóng nhánh trong phạm vi môn học.

### Quy tắc thống kê và dữ liệu

- Một ảnh gốc và mọi flip/crop/caption đi cùng một nhóm; bootstrap hoặc split theo nhóm, không theo từng dòng dự đoán.
- Cùng ảnh, hai prompt: so sánh ghép cặp. Hai tập ảnh nguồn khác nhau: không ghép cặp theo vị trí dòng.
- Các test cũ đã được xem không trở lại thành holdout chỉ bằng cách đổi seed chia tập. Ghi sổ dữ liệu: chưa suy luận, phát triển, đã phân tích, xác nhận đã mở.
- Nếu chỉ mô tả ảnh hưởng của một bộ prompt cố định, có thể đánh giá trên test cũ với nhãn “phân tích tiếp nối”. Nếu chọn prompt tốt nhất, cần development và holdout chưa dùng để lựa chọn. Không tuyên bố mô hình chưa từng thấy ảnh trong pretraining.
- Nếu chọn prompt bằng nhãn development của dataset đích, ghi rõ có dùng nhãn để chọn prompt dù trọng số model vẫn frozen; không mô tả quy trình đó như hoàn toàn không dùng nhãn dataset đích.
- Một primary comparison cho mỗi câu hỏi; các phép phụ có nhãn exploratory hoặc kế hoạch hiệu chỉnh đa so sánh. Không xem 100 lớp như 100 lần xác nhận độc lập.
- Report top-1, macro theo lớp, paired delta, CI và support. Không gộp các nhiệm vụ có số lựa chọn khác nhau thành một “accuracy dự án”.
- Muốn kết luận thực tế tương đương phải định trước biên khác biệt chấp nhận được và phép kiểm phù hợp. Không lấy p > 0,05 thay cho bằng chứng tương đương.
- Cỡ mẫu chọn theo câu hỏi. Minh họa: tỷ lệ gần 50% với khoảng 100 nguồn độc lập có sai số xấp xỉ ±10 điểm %; muốn khoảng ±5 điểm % cần gần 385 nguồn theo xấp xỉ nhị thức. Đây không phải mặc định số ảnh, không phải tính power cho mọi thiết kế; nhóm phụ thuộc hoặc phân tầng có thể cần nhiều hơn.
- Các mức thay đổi “đáng quan tâm”, nếu dùng, phải được chốt trước run tương lai và ghi là tiêu chí nhóm lựa chọn, không phải chuẩn của đề bài.

## 4. P0 — chốt yêu cầu, câu hỏi và tài liệu nền

Ưu tiên bắt buộc. Ước lượng 6–10 giờ công; chưa cần GPU. Phụ thuộc: không.

| ID | Task | Đầu ra/tiêu chí hoàn thành |
|---|---|---|
| P0.1 | Lập ma trận yêu cầu → artifact → giới hạn → người phụ trách | Mỗi mục đề bài có bằng chứng hoặc việc còn thiếu; kiểm trạng thái nộp proposal/milestone |
| P0.2 | Chốt hạn nộp, người làm thực tế, giờ/tuần, phản hồi giảng viên | Lịch có ngày kết thúc, thời gian dự phòng và người chịu trách nhiệm |
| P0.3 | Lập bảng tài liệu liên quan | Mỗi công trình: task, data, metric, model, điều gì khác dự án; nguồn chính thức |
| P0.4 | Lập sổ dữ liệu đã xem và sổ quyết định | Không tái sử dụng tập đã chọn prompt như xác nhận mới |
| P0.5 | Chốt 3–4 câu hỏi chính của báo cáo | RQ baseline/failure; RQ tính nhất quán không gian; RQ prompt/category; RQ ngôn ngữ nếu làm |

Phương án A: giảng viên xác nhận hướng probing và giới hạn fine-grained phù hợp → tiếp tục. B: cần fine-grained tự nhiên mạnh hơn → ưu tiên P1.4 trên Pet/Flowers đã có trước các extension mới. C: còn rất ít thời gian → khóa phạm vi mức A và viết report; không khởi động dataset mới.

Khảo sát nguồn khởi đầu: [CLIP](https://proceedings.mlr.press/v139/radford21a.html), [ARO](https://arxiv.org/abs/2210.01936), [Winoground](https://arxiv.org/abs/2204.03162), [NegBench](https://arxiv.org/abs/2501.09425). Đây là tài liệu tham chiếu, không phải số đo của repo. Đã có preprint [ViCLIP-OT, 2026](https://arxiv.org/abs/2602.22678) về truy hồi ảnh–text tiếng Việt: không tiếp tục tuyên bố lĩnh vực tiếng Việt “chưa ai đo”. Sự tồn tại của công trình này không bắt buộc đổi model; cần phân biệt retrieval và classification/probing trong related work.

## 5. P1 — hoàn thiện chất lượng bằng chứng core

Ưu tiên bắt buộc cho rà soát và diễn giải; bổ sung thí nghiệm chỉ khi phát hiện lỗ hổng cụ thể. Ước lượng 10–18 giờ công; review nhãn bởi người thêm 12–24 giờ nếu có nhân lực. Phụ thuộc P0.

### P1.1 — bản đồ kết luận

Lập `claim → metric → run → controls → uncertainty → limitation`. Ví dụ: “thiếu nhất quán trái/phải” trỏ vào tỷ lệ cặp cùng đúng; “category giảm accuracy” trỏ vào paired delta; không suy ra cơ chế neuron từ hai thống kê đó. Kiểm số liệu PDF bằng nguồn máy đọc được, không chép lại thủ công.

Hoàn thành khi từng kết luận trong abstract/conclusion có bằng chứng và giới hạn tương ứng. Nếu không có bằng chứng: đổi câu kết luận hoặc đưa thành giả thuyết tương lai.

### P1.2 — kiểm nhãn tự nhiên độc lập, nếu có người

Ưu tiên 100 nguồn xác nhận mới; tiếp theo 200 nguồn cũ. Người review nhìn ảnh gốc, crop thật mà CLIP nhận, hai đối tượng và định nghĩa quan hệ; không nhìn scores, dự đoán hoặc quyết định AI cũ. Reviewer độc lập vẫn có thể biết mục tiêu chung, phải mô tả đúng mức độ che thông tin.

Đề xuất hai người review độc lập 100 nguồn mới, sau đó phân xử; nếu đủ người mở rộng tới 200 cũ. Lưu annotation riêng, lý do bất đồng và quyết định phân xử. Đo tỷ lệ đồng thuận; chỉ tính hệ số đồng thuận khi có đủ hai bộ nhãn độc lập và mô tả cách xử lý “không xác định”.

- A: đồng thuận cao, ít thay đổi → tăng độ tin cậy của nhãn; báo đầy đủ và subset được xác nhận.
- B: nhiều bất đồng → phân nhóm nguyên nhân: crop, nhiều vật thể cùng tên, quan hệ chéo, đối tượng in trên bao bì; sửa hướng dẫn cho tập tương lai. Kết quả cũ giữ nguyên.
- C: không có người → giữ nhãn AI với giới hạn hiện tại, tiếp tục report mức A. Không gọi một lượt chạy lại cùng AI là annotator độc lập.
- D: nhãn sau review làm đổi chiều kết quả → công bố cả hai và thừa nhận kết luận phụ thuộc chất lượng nhãn. Đây là kết quả về độ nhạy của đánh giá.

Review sau khi đã có kết quả là kiểm tra nhãn/sensitivity; không làm tập 100 nguồn trở thành holdout mới một lần nữa.

### P1.3 — counting và negation

Counting: dùng scores đã có để báo confusion 1–6, MAE, signed error, support theo số lượng, hai chế độ kích thước và control. Nếu nhận dạng cơ bản yếu, kết luận phải bao gồm cả khó khăn nhận dạng. Chỉ bổ sung renderer/phông nền/object mới nếu cần phân biệt giả thuyết; không xóa các mẫu control sai khỏi metric chính.

Negation: giữ kết quả 86,11%, phân tích theo template và present/absent, độ nhất quán cả nhóm. Accuracy cao không phải “thí nghiệm thất bại”.

- A: negation tốt ở cả hai trạng thái → kết luận checkpoint xử lý được thiết kế hình học đơn giản này.
- B: accuracy cao nhưng consistency thấp → mô tả khác biệt giữa đúng từng ảnh và đúng cả cặp.
- C: cần kiểm tra ngoài hình học → chọn một phần benchmark phủ định tự nhiên có giao thức rõ, như NegBench, chỉ sau kiểm nguồn/quyền sử dụng/schema. Không dùng “COCO không có annotation vật thể” làm bằng chứng chắc chắn vật thể vắng mặt.
- D: không đủ thời gian → công bố giới hạn miền; không bắt buộc thêm benchmark.

### P1.4 — fine-grained

Giữ nhãn “oracle 5-way” cho CIFAR. Với Pet/Flowers/Food, báo global candidate-set accuracy, macro, confusion và support từ kết quả đã có. Nhóm các lớp gần nhau theo taxonomy hoặc tiêu chí chuyên môn độc lập với scores; nếu chọn nhóm từ lỗi cũ, ghi là thăm dò.

- A: có lỗi phân biệt tinh rõ trong tập lớp đầy đủ → đủ để hỗ trợ core cùng với chẩn đoán CIFAR.
- B: chỉ có vài ví dụ thiếu support → báo toàn bộ lớp và CI/đếm; giảm độ mạnh kết luận về lớp riêng.
- C: giảng viên yêu cầu một benchmark fine-grained chuyên biệt hơn → ưu tiên chính Pet/Flowers toàn bộ lớp; chỉ thêm dataset mới nếu câu hỏi chưa thể trả lời từ dữ liệu hiện tại.

Gate P1: không còn kết luận sai vì nhầm mẫu số, oracle, điều kiện, ngôn ngữ của task hoặc dữ liệu đã xem. Gate này có thể đạt dù mọi accuracy giữ nguyên.

## 6. P2 — đóng nhánh không gian bằng chẩn đoán có mục đích

Ưu tiên cao trong mức B; không mặc định chạy thêm một tập xác nhận chỉ để tăng n. Ước lượng 12–24 giờ công; một vòng chính và tối đa một vòng tiếp nối. Phụ thuộc P1.

### P2.1 — tách nhận dạng đối tượng khỏi ghép quan hệ

Thiết kế control nhận dạng trên chính ảnh tự nhiên: caption đúng chứa hai đối tượng, foil thay một đối tượng bằng đối tượng không hiện diện được reviewer xác nhận. Cân bằng vai trò A/B, số lựa chọn, vị trí đáp án và cách tạo foil. Không tự coi “không có box” là không có đối tượng. Lưu chất lượng/độ khó foil.

Metric chính của nhánh chẩn đoán: nhận dạng đúng trên toàn tập, quan hệ đúng trên toàn tập; báo thêm quan hệ có điều kiện control đúng với mẫu số rõ ràng. Vì ảnh đã được xem, đây là thí nghiệm tiếp nối trên dữ liệu cũ, không gọi là xác nhận ngoài mẫu.

| Kết quả | Cách hiểu được hỗ trợ | B/C |
|---|---|---|
| Nhận dạng tốt, quan hệ yếu | Điểm yếu trong kết hợp quan hệ dưới thiết kế này | Kiểm độ bền theo thứ tự câu; không suy ra neuron/encoder gây lỗi |
| Cả hai yếu | Chưa tách được nhận dạng và quan hệ | Kiểm crop, kích thước vật thể, tên lớp; bổ sung control chuẩn trước tuyên bố |
| Nhận dạng và quan hệ tốt | Hiện tượng cũ phụ thuộc bộ dữ liệu/điều kiện | Báo khác biệt, tìm điều kiện phân biệt; không cố tìm lại failure |
| Control mơ hồ | Thiết kế control chưa đủ tốt | Sửa trên tập phát triển rồi cố định cho run mới; bỏ control nếu không bảo đảm nhãn |

### P2.2 — độ nhạy với preprocessing và việc lật ảnh

Nếu P2.1 chỉ ra khả năng mất đối tượng do crop, so sánh preprocessing gốc với một cách giữ toàn ảnh đã định trước. Giữ nguyên checkpoint và caption; ghi đây là thay đổi preprocessing. Nó vừa giữ thêm nội dung vừa thay tỉ lệ/kích thước, nên không chứng minh một nguyên nhân duy nhất.

Nếu còn nghi ngờ mirror làm thay đổi chữ hoặc dấu hiệu không liên quan, đánh dấu theo tiêu chí không dựa trên score; báo đầy đủ và subset. Phương án B là một tập cặp ảnh chụp độc lập/scene thật có quan hệ đổi, nhưng cần review và ngân sách riêng. Phương án C là đóng nhánh với giới hạn mirror.

### P2.3 — có cần thêm tập xác nhận?

Chỉ mở khi đã xác định một giả thuyết cụ thể và cỡ mẫu khả thi. Mốc 53/100 hiện tại không đủ để kết luận xác suất thật bằng 50%. Hai lựa chọn hợp lệ: giữ kết luận chưa đủ bằng chứng; hoặc lập một nghiên cứu mới có mục tiêu precision/power định trước.

Nếu làm: dùng nguồn chưa đưa vào lựa chọn/điều chỉnh, loại trùng ID, nội dung/cảnh, photographer/burst khi biết; COCO 2014 và 2017 không mặc nhiên độc lập vì có thể dùng lại ảnh. Chốt protocol, reviewer, metric, cỡ mẫu và phân tích trước inference. Không gộp tùy tiện tập cũ đã hình thành giả thuyết vào phép xác nhận chính.

- A: kết quả mới cùng hướng và đủ chính xác → tăng mức chắc chắn trong phạm vi dữ liệu.
- B: không rõ → dừng theo ngân sách, không lấy thêm mẫu đến khi p nhỏ.
- C: đảo chiều → báo riêng từng tập, kiểm chênh phân bố và nhãn; không trung bình để che mâu thuẫn.
- D: thiếu ảnh độc lập → không mở run xác nhận; chuyển công sức sang report/prompt.

Không tự coi accuracy nhóm được chọn sau khi xem scores là kết quả xác nhận. Attention map hoặc t-SNE chỉ là minh họa nếu bổ sung; không thay thí nghiệm đối chứng hoặc chứng minh cơ chế.

## 7. P3 — prompt sensitivity và ensemble

Extension ưu tiên đầu tiên. Ước lượng 18–30 giờ công. Phụ thuộc P0 và bộ cache được kiểm provenance; có thể làm sau khi P1 đạt gate, không cần chờ P2 hết mọi nhánh.

### P3.1 — phạm vi cố định vừa sức

Giữ một checkpoint và năm dataset. Nhóm N đổi diễn đạt gần nghĩa, nhóm D thêm category **chỉ cho CIFAR-100** như quyết định hiện hành. Không mở category chung cho Pet/Food/Flowers/DTD trong vòng đầu.

Draft bốn cấu trúc N: `a photo of a {label}.`, `a picture of a {label}.`, `an image of a {label}.`, `this is a photo of a {label}.`. N1 giữ đúng baseline từng dataset; với DTD, rendering label giữ từ `texture` nhất quán trong mọi template N. Tên lớp, mạo từ và xử lý số nhiều được chốt bằng review ngôn ngữ trước run, không sửa theo accuracy. Nếu sửa rendering khác baseline cũ, tạo điều kiện mới và giữ baseline cũ riêng.

Với CIFAR-100, tạo bốn D cùng cấu trúc N nhưng thêm `, a type of {category}` trước dấu chấm. Ánh xạ coarse/fine giữ nguyên; đủ 100 ứng viên cho mỗi ảnh. Các template là đề xuất để review, chưa phải protocol đóng băng.

Ma trận dự kiến: 4 N × 5 dataset = 20 điều kiện; 4 D × CIFAR = 4; ensemble N4 × 5 = 5; thêm D4 và All8 cho CIFAR = 2. Tổng **31 điều kiện**, bao gồm các N1 trùng baseline có thể tái dùng nếu hoàn toàn tương đương. Không gọi tất cả là 31 thí nghiệm độc lập.

### P3.2 — giao thức và triển khai

1. Ghi mục tiêu sensitivity, category information và ensemble thành ba so sánh khác nhau.
2. Đóng băng templates, rendering, ensemble và class order.
3. Cache image embeddings có checkpoint/preprocess/sample IDs/precision/source hashes. Cache không khớp phải vô hiệu hóa.
4. Unit test normalization/ensemble và parity N1; smoke trên dữ liệu phát triển/cũ.
5. Chạy ma trận cố định, lưu predictions và scores; không chỉ lưu accuracy.
6. Sinh bảng đầy đủ và bootstrap ghép cặp; CI theo lớp/ảnh phù hợp estimand.

Ensemble mỗi lớp: chuẩn hóa từng text vector → trung bình trọng số bằng nhau → chuẩn hóa lại. Không trung bình text thô hoặc tùy tiện trộn logits với embeddings. Không chọn trọng số theo test; weighted ensemble là nghiên cứu khác.

Metric: top-1, macro, paired delta so N1, range/SD giữa template, tỷ lệ ảnh đổi prediction, đúng→sai/sai→đúng, đủ per-class. SD giữa template chỉ mô tả tập template đã chọn, không là CI của accuracy và không đại diện mọi cách diễn đạt.

### P3.3 — nhánh quyết định

| Kết quả | Hành động | Điều kiện dừng |
|---|---|---|
| Ensemble tăng trên đa số dataset | Báo effect size và chi phí; có thể kiểm xác nhận mới nếu muốn tuyên bố chọn phương pháp | Không cần tìm thêm template để tăng nữa |
| Có dataset tăng, dataset giảm | Báo từng dataset, macro dataset nếu đã định trước; phân tích miền | Không chọn lại prompts riêng bằng test |
| Ensemble không hơn N1 | Kết luận không có lợi rõ trong bộ đã chọn; xét prediction flips và stability | Đây vẫn là extension hoàn thành |
| Ensemble kém một prompt đơn tốt nhất trên test | Báo prompt đó là best observed, không phải best deployable | Muốn chọn prompt phải dùng dev + holdout |
| Category đúng thua baseline nhưng hơn shuffled | Kết luận ý nghĩa category có ảnh hưởng nhưng suffix không mặc nhiên cải thiện | Đối chứng hiện có đã trả lời phần lớn; không bắt buộc sửa prompt |
| Category kết quả đảo chiều giữa cách viết | Có tương tác giữa nội dung và cách diễn đạt | Tối đa một thí nghiệm tiếp nối đã cố định, rồi đóng nhánh |
| Neutral tốt ngang/tốt hơn category | Cách kéo dài câu cũng ảnh hưởng, chưa cô lập ngữ nghĩa | Không gán mọi thay đổi cho category |

Nếu muốn đường cong số prompt K: dùng K ∈ {1,2,4} với thứ tự hoặc các tổ hợp định trước; không chọn subset tốt nhất trên test. So N4/D4 cùng K; All8 khác cả K và nội dung nên không quy mọi khác biệt cho số prompt.

Gate P3: đủ điều kiện cố định, audit pass, báo toàn bộ kết quả, phân biệt mô tả với lựa chọn phương pháp. Không có yêu cầu accuracy phải tăng.

## 8. P4 — tiếng Việt trên cùng ảnh

Extension ưu tiên thứ hai trong mức B; rẻ hơn xây dataset văn hóa mới. Ước lượng 18–30 giờ công, phần lớn là kiểm bản dịch. Bắt đầu CIFAR-100; mở Food-101 chỉ nếu có thời gian và thuật ngữ dịch đáng tin cậy. Phụ thuộc P3 về engine prompt, không phụ thuộc ensemble có tăng accuracy hay không.

### Task

| ID | Công việc | Đầu ra |
|---|---|---|
| P4.1 | Từ điển class EN–VI, giải nghĩa tên mơ hồ, tên riêng, từ mượn | File dịch có reviewer, nguồn và quyết định cho đủ lớp |
| P4.2 | Kiểm Unicode NFC, dấu, token count, truncation theo tokenizer checkpoint | Không cắt caption âm thầm; lỗi kỹ thuật xử lý trước inference |
| P4.3 | Chốt bản dịch và template cặp tương đương | Một primary comparison: EN hoàn chỉnh vs VI hoàn chỉnh |
| P4.4 | Chạy cùng ảnh và cùng tập lớp | Top-1, macro, paired delta, CI, confusion và support |
| P4.5 | Chẩn đoán nếu cần | Có thể thêm EN frame + VI label và VI frame + EN label; ghi câu trộn là control, không phải tiếng tự nhiên chuẩn |

Không thay model sang multilingual trong vòng chính: câu hỏi là checkpoint hiện tại phản ứng với tiếng Việt thế nào. Không đổi bản dịch vì một cách viết có score test cao hơn. Nếu phải chọn synonym, chọn bằng ngữ nghĩa/người review hoặc development riêng trước test.

- A: VI thấp hơn EN → kiểm dịch, tokenization và truncation; nếu không có lỗi, báo giới hạn trong checkpoint/task này.
- B: VI gần EN, CI rộng → chưa rõ; không tuyên bố tương đương chỉ từ p lớn.
- C: VI cao hơn EN → kiểm cùng tập ứng viên, tên mượn/đặc trưng từng lớp, báo đầy đủ; không suy ra CLIP giỏi tiếng Việt toàn diện.
- D: chỉ vài lớp đảo chiều → descriptive per-class; không biến các ví dụ được chọn thành khái quát.
- E: không có reviewer tiếng Việt phù hợp → dùng phạm vi thuật ngữ ít mơ hồ hoặc đánh dấu exploratory/hoãn. Không gọi bản dịch AI chưa kiểm là ground truth đã xác thực.
- F: lỗi kỹ thuật được phát hiện sau run → giữ run cũ với trạng thái không dùng kết luận, ghi sửa đổi, chạy lại vào tên mới. Đây khác với sửa prompt nhằm tăng điểm.

Gate P4: dịch có nguồn và giới hạn, tập so sánh tương đương, không rò rỉ chọn prompt, metrics tái lập. Không yêu cầu VI phải kém EN hoặc ensemble phải cứu được VI.

## 9. P5 — nội dung văn hóa Việt Nam, nhánh tùy chọn

Chỉ mở nếu còn ít nhất một vòng thu thập/review hoàn chỉnh ngoài thời gian viết report; ước lượng 30–60 giờ công. Không cần làm để thỏa hướng non-English nếu P4 đã hoàn thành. Nên chọn **một miền hẹp**, ví dụ món ăn, thay vì trộn phở, áo dài, kiến trúc và phương tiện để đạt accuracy dễ nhưng khó diễn giải.

Draft quy mô: khoảng 10 lớp, 20–30 ảnh/lớp; 200–300 ảnh là pilot, không phải benchmark đại diện văn hóa Việt. Chốt nhãn chỉ sau feasibility về nguồn và khả năng phân biệt bằng ảnh; không hứa đủ mọi món trước khi tìm nguồn.

1. Định nghĩa đối tượng, trường hợp mơ hồ, nhiều món trong ảnh và câu hỏi: phân biệt tên gọi/khái niệm trong miền đã chọn.
2. Thu ảnh tự chụp hoặc nguồn có quyền sử dụng rõ, lưu URL/license/creator theo ảnh. Nếu chỉ được cung cấp manifest thì không đưa pixels lên Git.
3. Loại trùng nội dung, cùng cảnh/burst và nguồn; phân chia theo nhóm nguồn trước mọi lựa chọn prompt. Số ảnh train/dev/test nhỏ làm CI per-class rộng, phải công bố.
4. Hai người gán nhãn nếu có; lưu bất đồng và phân xử. AI hỗ trợ chuẩn bị, không thay tên người review.
5. Cố định caption EN/VI tương đương về mức thông tin; tên Việt giữ nguyên trong EN là một điều kiện riêng, không che trong phép dịch.
6. Nếu dùng prompt có giải thích món ăn, thêm control tương ứng; không so tên ngắn VI với mô tả EN dài rồi gọi là hiệu ứng ngôn ngữ thuần túy.
7. Đánh giá, audit, viết dataset card và phân tích lỗi theo nguồn/lớp.

| Vấn đề | Phương án B | Phương án C/đóng nhánh |
|---|---|---|
| Không đủ ảnh hợp lệ | Giảm số lớp hoặc dùng dataset sẵn có sau kiểm nguồn | Giữ P4 tiếng Việt, bỏ bộ văn hóa tự xây |
| Nhãn món không phân biệt được bằng ảnh | Gộp/định nghĩa lại lớp trước đóng băng | Loại lớp khỏi bộ tương lai, không âm thầm sửa test đã chạy |
| Accuracy thấp với cả EN và VI | Kiểm chất lượng ảnh, class ambiguity và candidate set | Báo khó khăn của thiết kế, không quy ngay cho “thiếu hiểu văn hóa” |
| VI tốt hơn EN | Kiểm tương đương bản dịch/tên riêng | Giữ kết quả và giới hạn, không ép chiều mong đợi |
| Model dựa vào watermark/nền/nguồn | Phân tích theo nguồn, thêm đối chứng nếu khả thi | Thu hẹp tuyên bố; cần tập nguồn độc lập để xác nhận |
| Thiếu contrast set ngoài Việt Nam | Chỉ báo hiệu năng trên tập ảnh Việt | Không tuyên bố lợi thế/bất lợi đặc thù văn hóa so với miền khác |

Không dùng nhiều ảnh sinh AI làm thay thế âm thầm cho tập ảnh tự nhiên. Nếu nghiên cứu ảnh sinh, đó là nhánh dữ liệu khác, có nguồn và kết luận riêng.

## 10. P6 — báo cáo, trình bày và bàn giao

Bắt buộc, khởi động ngay từ P0 và hoàn thiện xuyên suốt; ước lượng 25–40 giờ công. Không đợi mọi extension xong mới viết.

### Báo cáo 6–8 trang

Draft phân bổ tám trang, điều chỉnh theo template và quy định references của giảng viên:

| Phần | Nội dung ưu tiên | Dung lượng dự kiến |
|---|---|---:|
| Abstract + introduction | Câu hỏi, phạm vi một checkpoint, đóng góp cụ thể | 1 trang |
| Related work | CLIP, compositionality, negation và tiếng Việt nếu có | 0,5 trang |
| Method/protocol | Dataset/splits, control, prompt, metrics, review, CI | 1,5 trang |
| Baseline + core | Năm dataset và bốn bài; phân biệt oracle | 1 trang |
| Depth + confirmation | Spatial consistency, category controls, kết quả mới/giới hạn | 1,5 trang |
| Extensions | Prompt/ensemble và VI đã làm; bỏ mục chưa chạy | 1 trang |
| Discussion, limitations, conclusion, references | Kết luận theo bằng chứng, không hứa cơ chế chưa kiểm | 1,5 trang |

Nếu references tính riêng hoặc template khác, phân bổ lại; không mặc định phụ lục được phép ngoài giới hạn. Bảng đủ 100 lớp, full prompt và log đặt ở repo/phụ lục được chấp nhận.

Mỗi RQ trình bày: quan sát → giả thuyết → đối chứng → kết quả → điều được hỗ trợ → điều chưa biết. Kết quả không có ý nghĩa thống kê hoặc đi ngược kỳ vọng vẫn xuất hiện trong main narrative khi liên quan câu hỏi chính.

### Slides 15 phút + Q&A

Đề xuất 10–12 slides; tập nói 12–13 phút để có dự phòng. Một slide bài toán, một slide CLIP/pipeline, một slide baseline, hai hoặc ba slide probing/depth, một slide xác nhận, một hoặc hai slide extension nếu có, một slide giới hạn/kết luận. Chuẩn bị backup slides về 53% vs 4%, CI, oracle 5-way, ảnh lật và dữ liệu đã xem.

Tất cả thành viên hiểu được ít nhất pipeline, metric và một failure case; phân vai nói theo quy định giảng viên. Demo ưu tiên chạy trên artifact đã lưu hoặc smoke nhỏ; chuẩn bị video/PDF dự phòng nếu GPU/mạng lỗi. Demo không thay số liệu thí nghiệm đã audit.

### Bàn giao tái lập

- README chỉ rõ bản report mới nhất, protocol và các lệnh reproduce theo mức: đọc artifact; audit; chạy inference mới.
- Môi trường tối thiểu/versions tương thích, checkpoint hash, nguồn dữ liệu và những bước tải cần thiết.
- Fresh clone kiểm artifact/source, tests; ít nhất một smoke inference ở môi trường sạch khi tài nguyên cho phép. Unit tests sạch hiện có không đồng nghĩa đã tái chạy inference full trên máy thứ hai.
- Release candidate khóa số liệu; kiểm PDF, bảng, nguồn trích dẫn và links.
- Commit theo câu hỏi/nghiên cứu, annotated tag mốc cuối, bản lưu dự phòng. Không force-push để sửa lịch sử nghiên cứu; correction có commit/run mới.
- Sau review nhóm, đưa phiên bản bàn giao từ `dev` sang `main` theo quy trình nhóm thống nhất; không tự merge hoặc công bố bài báo chỉ vì kế hoạch có mục này.

Gate P6: đủ report đúng định dạng, slides đúng thời lượng, repo chạy lại theo hướng dẫn, kết luận có giới hạn, trạng thái nộp được xác nhận. Đạt gate này có thể kết thúc dự án dù chưa mở P5.

## 11. Kiến trúc cần mở rộng và hạ tầng

Không refactor toàn bộ phần đã kiểm chứng nếu không có nhu cầu cụ thể.

| Phần mới | Vị trí đề xuất | Nguyên tắc |
|---|---|---|
| Giao thức tương lai | `docs/`, config trong `configs/` | Có ID phiên bản, amendment, primary metric |
| Prompt/ensemble | `vlm_probe/prompt_study/` (mới) | Tái dùng adapter và scoring hiện có; không đổi tên lớp ngầm |
| Từ điển EN–VI | `configs/` hoặc thư mục metadata riêng | Review, class IDs, Unicode, version/hash |
| Dataset văn hóa | `vlm_probe/dataset_adapters/` + manifest metadata | Trả ảnh/nhãn/classes/split; pixels vẫn trong `data/` |
| Spatial tiếp nối | `vlm_probe/depth/` hoặc module phiên bản mới | Giữ kết quả và giao thức v1 bất biến |
| Artifact | `results/<run_id>/`, `reports/<study_id>/` | Source snapshot, scores/predictions/metrics, audit |
| Kiểm nhãn | Metadata review riêng và công cụ xuất sheet | Tách nhãn/reviewer khỏi dự đoán khi review |

Giữ conda `gen`, một GPU hiện tại và FP32 theo baseline. Pilot kỹ thuật 50–100 ảnh phát triển để đo throughput, RAM/VRAM/disk; dự báo full = thời gian encode ảnh + text + scoring + I/O, thêm 20–30% dự phòng. Không suy ra cần thêm GPU chỉ từ số cấu hình prompt: ảnh có thể encode một lần nếu cache hợp lệ.

Nếu OOM: giảm batch theo config mới, kiểm parity trong tolerance định trước; không tự đổi precision/model. Nếu GPU không có: kiểm offline từ scores, tiếp tục review/report; CPU smoke hoặc GPU khác là run có environment mới, không gọi bit-identical mặc định. Nếu cache sai hash/order: bỏ cache đó, encode lại; tuyệt đối không dùng vì “accuracy trông hợp lý”.

Nếu artifact lớn lên: giữ metadata/metrics/protocol trong Git; cân nhắc kho artifact có version và checksum cho scores lớn ở các run tương lai. Không di chuyển/xóa artifact của tag cũ chỉ để giảm kích thước mà làm mất khả năng tái lập. Không cần cài thêm thư viện chỉ vì kế hoạch nhắc tới một công nghệ; chọn theo task cụ thể.

## 12. Phân công và ước lượng

Vai trò là người trong nhóm, không phải các agent tự động. Nếu ít người, gộp vai trò nhưng giữ reviewer nhãn độc lập khi có thể.

| Vai trò | Trách nhiệm | Người kiểm chéo |
|---|---|---|
| Data A | Nguồn ảnh, metadata, dedup, split | Data B |
| Data B | Annotation/review, từ điển EN–VI | Data A hoặc người thứ ba phân xử |
| Eval A | Pipeline, config, cache, inference | Eval B |
| Eval B | Metrics, audit, bootstrap, bảng | Eval A |
| Analysis | Protocol, prompt, quyết định A/B/C, related work | Người viết report + người chạy |
| Writing | Claim–evidence, LaTeX/slides, checklist bàn giao | Tất cả đọc và ký xác nhận phần mình |

Đường khuyến nghị P0/P1/P2/P3/P4/P6 khoảng **89–152 giờ công**, chưa gồm review người độc lập 12–24 giờ và P5 30–60 giờ. Cộng 20–30% dự phòng; đây là ước lượng quản lý cần cập nhật sau tuần đầu. Sáu người × 4–6 giờ/tuần cho 24–36 giờ công/tuần không có nghĩa mọi task chạy song song được; review/protocol vẫn là phụ thuộc.

## 13. Lịch tương đối và các phương án rút gọn

T0 là ngày bắt đầu thực hiện kế hoạch được nhóm chốt, không mặc định ngày lập tài liệu.

| Tuần từ T0 | Việc chính | Mốc quyết định |
|---|---|---|
| 1 | P0; sổ dữ liệu; draft report; P1 claim audit và chuẩn bị review | Chốt mức A/B, lịch và RQ |
| 2 | P1 review/diễn giải; chọn một chẩn đoán P2; chốt P3 | Gate chất lượng; không chạy prompt chưa đóng băng |
| 3 | P2 chẩn đoán; P3 smoke/parity/cache và full cố định | Có dữ liệu đủ để trả lời hoặc quyết định dừng |
| 4 | Phân tích P3; tối đa một follow-up P2/P3 nếu có lý do; P4 dịch | Khóa template tương lai, tránh test tuning |
| 5 | P4 inference/phân tích; P5 chỉ feasibility nếu có ngân sách | Quyết định mở hoặc bỏ P5 |
| 6 | Khép nhánh; P5 pilot nếu đã đủ dữ liệu/review; report hoàn chỉnh bản 1 | Đóng băng thí nghiệm chính |
| 7 | Peer review, sửa luận điểm/bảng, fresh clone/repro, slides | Release candidate |
| 8 | Chỉ sửa lỗi, tập trình bày, final PDF/tag và nộp | Hoàn thành môn học |

**Còn 2 tuần:** không mở P2 xác nhận mới hoặc P5; tập trung P0/P1/P6, review nhãn nếu có người, report và slides. Một extension nhỏ chỉ làm nếu không ảnh hưởng bàn giao.

**Còn 4 tuần:** P0/P1 + P3 prompt/ensemble + P6; P4 chỉ nếu bản dịch được review sớm. Không tự xây dataset văn hóa.

**Còn 8 tuần:** đường mức B như bảng; P5 vẫn tùy chọn, không chiếm hai tuần cuối.

**Sau môn học 4–8 tuần hoặc lâu hơn:** chỉ mở mức C khi có phản hồi giảng viên, câu hỏi rõ và nguồn dữ liệu; chất lượng công bố không bảo đảm theo thời gian bỏ ra. Ưu tiên một mở rộng có thể kiểm chứng thay vì tăng mọi trục cùng lúc.

Nếu trễ một tuần: cắt P5 trước, rồi Food-101 tiếng Việt mở rộng, rồi follow-up không gian tùy chọn; bảo vệ thời gian P6. Nếu còn ≤14 ngày tới hạn, khóa thí nghiệm mới, ngoại trừ sửa lỗi ảnh hưởng kết luận.

## 14. Sổ rủi ro và phương án B/C

| Tín hiệu | Phương án A | Phương án B | Phương án C / dừng |
|---|---|---|---|
| Giảng viên chưa đồng ý phạm vi | Gửi ma trận yêu cầu và report hiện tại | Điều chỉnh mục yếu nhất | Tạm dừng extension, hoàn thiện core |
| Kết quả trái kỳ vọng | Kiểm nhãn/code/control | Một follow-up phân biệt giả thuyết | Báo kết quả trái kỳ vọng |
| CI rộng | Xác định precision/power cần thiết | Nghiên cứu mới có fixed n | Báo chưa rõ, không tăng mẫu tùy tiện |
| Test đã bị dùng chọn prompt | Ghi là development/exploratory | Dùng holdout nguồn mới | Không tuyên bố xác nhận độc lập |
| Chỉ subgroup đẹp | Báo tất cả nhóm | Giả thuyết subgroup cho study mới | Không lựa chọn thuận lợi |
| Benchmark ngoài không tải được | Mirror chính thức, hash/revision | Dataset khác cùng câu hỏi với protocol riêng | Bỏ đối chiếu ngoài, ghi giới hạn |
| Cùng ảnh ở nhiều split/version | ID/hash/perceptual và source review | Chia lại trước inference mới | Nếu phát hiện sau, sensitivity + báo contamination |
| Dịch không tương đương | Review ngữ nghĩa | Giảm phạm vi lớp/dataset trước đóng băng | Chỉ công bố pilot exploratory |
| Quyền sử dụng ảnh không rõ | Xác minh từ nguồn | Đổi nguồn hợp lệ, cung cấp manifest | Không phân phối ảnh đó |
| GPU/runtime gián đoạn | Resume từng chunk đã hash nếu hỗ trợ | Run mới batch nhỏ, environment rõ | Tập trung artifact/report, không giả lập kết quả |
| Audit không khớp | Khoanh vùng artifact/schema/source | Sửa tool hoặc chạy lại đúng phạm vi | Không dùng số liệu chưa kiểm chứng |
| Giữa hai máy score khác nhỏ | So tolerance và prediction/margin | Ghi sai khác environment, chạy điều kiện ghép cặp cùng máy | Không khẳng định bit-identical |
| Reviewer không có | Lưu rõ cùng-AI review | Review người trên subset được chọn không theo score | Không báo IAA hoặc “human-validated” |
| Không có novelty so literature | Định vị replication/controlled case study | Tìm khoảng trống cụ thể phù hợp dữ liệu | Hoàn thành môn học, không hứa paper |
| Merge/commit làm mất artifact | Tag bất biến và backup | Khôi phục từ snapshot/checksum | Correction commit; không sửa lịch sử |
| Hạn nộp sát | Khóa số liệu, cắt optional | Report mức A và limitations | Không cắt tests/audit thiết yếu để chạy thêm |

## 15. Điều kiện kết thúc và việc đầu tiên

Được coi là hoàn thành khi: mọi yêu cầu core được ánh xạ tới bằng chứng; không còn lỗi kỹ thuật/nhãn đã biết bị che trong kết luận; extensions đã nhận làm được kết thúc hoặc ghi rõ bỏ vì sao; report 6–8 trang đúng template; slides 15 phút; repo/tag tái lập; nhóm hiểu giới hạn; xác nhận nộp đủ sản phẩm.

Không cần hoàn thành mọi hướng mức C, không cần accuracy tăng, không cần tất cả giả thuyết đúng, không cần thêm model/fine-tuning, không cần một p-value có ý nghĩa để đóng dự án.

**Thứ tự bắt đầu:** P0.1–P0.5 → P1 claim audit/review plan → giao thức P3 prompt/ensemble. P2 chỉ mở chẩn đoán có mục tiêu; P4 là extension kế tiếp; P5 phụ thuộc ngân sách. Viết report song song ngay từ P0.

Kế hoạch có thể được cập nhật theo phản hồi và dữ liệu, nhưng mỗi lần đổi cần ghi: điều gì thay đổi, bằng chứng nào dẫn tới đổi, dữ liệu nào đã được xem, tác động lên metric/scope và ngày đóng băng mới. Không sửa ngược lịch sử để khiến giả thuyết có vẻ được đặt trước kết quả.
