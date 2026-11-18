# Giao thức bổ sung chiều sâu — v1

Giao thức được viết sau khi đã xem kết quả core/category v1, trước khi chạy các điều kiện mới. Đây là nghiên cứu tiếp nối có giả thuyết hậu nghiệm, không gọi là preregistration độc lập hoặc bằng chứng cơ chế nhân quả bên trong CLIP. Cấu hình máy đọc: `configs/probing_depth_v1.json`. Mỗi run giữ bản sao giao thức/config, SHA-256, source snapshot, môi trường, dự đoán và toàn bộ cosine ứng viên.

## Phạm vi và thứ tự

Một checkpoint OpenAI CLIP ViT-B/32, cùng adapter QuickGELU, preprocessing, FP32, batch ảnh 32, seed suy luận 0, môi trường conda gen. Không huấn luyện hoặc chọn prompt theo accuracy. Năm baseline cũ giữ nguyên. Thứ tự: cố định giao thức → phân tích kết quả cũ → spatial mới → category đối chứng → ảnh tự nhiên → báo cáo.

RQ1: dự đoán quan hệ có nhất quán khi đổi vị trí, đảo cách nhắc hai vật thể, thay mẫu câu, thay thuộc tính không quyết định quan hệ?
RQ2: category đúng thay đổi cạnh tranh giữa lớp đúng và đối thủ như thế nào so với category sai và phần thêm trung tính?
RQ3: hiện tượng về không gian có lặp lại trong tập ảnh tự nhiên được chọn độc lập với dự đoán CLIP?

## 1. Phân tích cũ

Tính lại từ scores: confusion theo quan hệ, phân bố đáp án, accuracy theo màu/hình, tỷ lệ đúng cả bốn quan hệ; counting theo số lượng và chế độ kích thước; negation theo template/polarity và nhóm. Với category, xuất đủ 100 lớp: điểm lớp thật, điểm đối thủ mạnh nhất, signed margin, đối thủ cố định từ baseline, lỗi trong/ngoài coarse group và chuyển trạng thái. Những ví dụ được chọn sau phân tích được ghi rõ là minh họa hậu nghiệm.

## 2. Spatial mới

120 nhóm bố cục độc lập × 3 biến thể ngoại hình × 4 quan hệ = 1.440 ảnh. Một nhóm chia sẻ tâm cảnh, khoảng cách, bán kính, nền và cặp vật thể. Biến thể: gốc; tuần hoàn màu red→blue→green; tuần hoàn hình circle→square→triangle. Hai vật thể luôn khác màu và khác hình. Mỗi nhóm có bốn quan hệ, cặp trái/phải và trên/dưới hoán đổi chính xác hai vị trí. Seed renderer 14020; ảnh 224×224, dựng antialias giống core; mọi vật thể nằm trọn ảnh, không chồng nhau.

Mỗi ảnh có 3 template × 2 thứ tự nhắc vật thể = 6 phép thử, tổng 8.640. T0 `a drawing of {A} {r} {B}.`; T1 `{A} is {r} {B}.`; T2 `an image of {A} {r} {B}.`. Với thứ tự đảo, dùng `{B} inverse(r) {A}` và ánh xạ lại về cùng nhãn quan hệ A–B. Mỗi phép thử luôn có đủ bốn ứng viên. T0-forward là phép đo chính nối tiếp core; T2-forward là cấu trúc chung với ảnh tự nhiên. Báo cáo đủ sáu điều kiện, không chọn điều kiện tốt nhất làm kết quả chính.

Đối chứng nhận dạng dùng caption hai vật thể đúng và caption đổi màu A sang màu thứ ba. Vị trí đáp án đúng được đảo cân bằng theo ID. Phân tích điều kiện khi control đúng là bổ sung, không loại ảnh khỏi accuracy chính. Thêm canvas trống cùng màu nền: đo tần suất dự đoán ảnh thật trùng đáp án canvas trống; đây là phép chẩn đoán ảnh hưởng nền/caption, không phải chứng minh mô hình bỏ qua ảnh.

Metrics: accuracy và confusion; phân bố dự đoán; signed margin = score đúng − max score sai; tỷ lệ cùng đúng của cặp trái/phải và trên/dưới; tỷ lệ đúng cả bốn quan hệ; nhất quán thứ tự câu sau ánh xạ; nhất quán qua ba ngoại hình (báo cả cùng nhãn và cùng đúng). Cùng nhãn có thể là cùng sai. Bootstrap percentile 95%, 2.000 lần seed 1402, lấy lại mẫu theo 120 nhóm bố cục, giữ mọi ảnh/caption liên quan cùng nhóm. Những tỷ lệ đều đúng không được suy ra từ accuracy biên.

## 3. Category có đối chứng

Cùng 10.000 ảnh CIFAR-100 test, 100 lớp ứng viên, taxonomy lấy từ train như category v1. Tám điều kiện: baseline; category đúng; suffix trung tính; năm ánh xạ coarse-category bị hoán đổi với seeds 271–275. Hoán đổi 20 nhóm bằng permutation, không cho một lớp giữ lại category_text đúng, kể cả hai nhóm vehicles cùng chữ vehicle. Giữ nguyên multiset 100 cụm category; toàn bộ năm lớp của một coarse group nhận cùng cụm sai. Không dùng nhóm thật của ảnh để lọc ứng viên.

Suffix trung tính chọn từ pool cố định trong config, theo khoảng cách nhỏ nhất giữa số token CLIP của câu đầy đủ và câu category đúng; hòa chọn mục đứng trước. Việc chọn chỉ xem tokenizer, không xem ảnh, nhãn dự đoán hoặc accuracy. Lưu số token và sai khác cho đủ 100 lớp. Điều kiện này kiểm soát một phần độ dài, không tách hoàn toàn cú pháp/ngữ nghĩa. Ngữ pháp baseline được giữ nguyên để đối chiếu bit/tolerance với kết quả đã có.

Encode ảnh một lần; cache phải kiểm checkpoint, preprocessing, dataset/sample order, nguồn mã và checksum file. Encode từng bộ 100 câu cùng cách baseline. So baseline và category đúng với scores v1: nhãn khớp 100%, cosine sai khác tối đa 1e-6. Lưu đủ ma trận N×100 của tám điều kiện. Metrics: top-1/macro, paired delta và CI class-stratified bootstrap 2.000 lần, transition, score/margin và lỗi trong/ngoài nhóm. Báo từng shuffle và mean/range; không coi năm shuffle là năm mô hình độc lập. Chênh lệch chính: đúng−baseline, đúng−neutral, đúng−trung bình 5 shuffle. Per-class là mô tả, không tuyên bố 100 kiểm định độc lập có ý nghĩa.

Tập test đã được xem trước nên đây là phân tích tiếp nối. Không chọn cách viết rồi báo accuracy của cùng tập như kiểm chứng độc lập. Category sai làm giảm điểm không tự chứng minh category đúng hữu ích so với baseline.

## 4. Kiểm chứng ảnh tự nhiên

Nguồn COCO val2017, instances annotation chính thức; mục tiêu 200 ảnh gốc khác nhau (50 mỗi quan hệ). Lấy cặp khác category, mỗi category chỉ có một annotation trong ảnh; không crowd. Box diện tích ≥1,5% ảnh, mỗi cạnh ≥6% cạnh ảnh tương ứng. Hai box nằm trọn vùng center crop vuông với biên 1%; tách nhau theo trục quan hệ ≥2% cạnh crop, chồng lên nhau theo trục còn lại ≥50% độ dài box nhỏ hơn. Tiêu chí tránh quan hệ chéo mơ hồ và mất vật thể sau preprocessing. Không dùng score CLIP để chọn dữ liệu.

Xếp ứng viên bằng SHA256 của seed/ID ảnh/ID annotation; tối đa một cặp có hướng mỗi ảnh, quota quan hệ cân bằng. Tạo danh sách dự phòng cố định, ghi mọi quyết định nhận/loại. Nếu thiếu ảnh sau lọc, chỉ được điều chỉnh tiêu chí trước suy luận và ghi amendment; không lặng lẽ đổi ngưỡng theo kết quả.

Rà soát gồm (1) validator hình học thứ hai kiểm nhãn từ box/center crop, (2) trợ lý xem contact sheets có ID, hai box và caption; không xem score CLIP. Loại ảnh hai vật thể khó nhận ra, nhãn sai/mơ hồ, ảnh/crop lỗi hoặc có nhiều vật thể cùng loại khiến caption không rõ. Đây không phải hai annotator người độc lập; không báo inter-annotator agreement. Ghi reviewer, tiêu chí, image hash và lý do loại. Người nghiên cứu có thể review lại toàn bộ manifest.

Mỗi ảnh gốc thêm horizontal flip; trái/phải đảo, trên/dưới giữ nguyên. Tổng dự kiến 400 ảnh đầu vào, 2.400 phép thử (3 template × 2 thứ tự). T0 đổi drawing→photo; T1/T2 giống spatial; T2-forward là metric chính. Bootstrap theo 200 ảnh nguồn, giữ flip/caption cùng nhóm. Đo accuracy, thứ tự tương đương và cặp gốc/flip cùng đúng; không gọi flip là ảnh mới độc lập. Nhãn là vị trí 2D trong ảnh, không phải suy luận không gian 3D. COCO khác hình học cả vật thể, bố cục và nền nên chênh accuracy không cô lập một yếu tố duy nhất. Không giả định COCO chưa xuất hiện trong dữ liệu pretraining CLIP.

Annotations COCO CC BY 4.0; giấy phép ảnh theo metadata từng ảnh, lưu URL Flickr/COCO và license. Ảnh chỉ nằm trong data/ bị Git bỏ qua; chỉ đưa metadata/đường dẫn tải và kết quả vào repo. Nguồn: [COCO download](https://cocodataset.org/#download), [điều khoản](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm). Thiết kế nhất quán ảnh–text tham khảo tinh thần [Winoground, CVPR 2022](https://arxiv.org/abs/2204.03162), không phải tái lập benchmark đó.

## 5. Tiêu chí hoàn thành và giới hạn

Mỗi nhánh có smoke, full run, audit tính lại từ scores, manifest đầu vào/đầu ra và source snapshot. Tests tập trung vào đảo quan hệ, bảo toàn nhóm, nhãn renderer, permutation, neutral token matching và audit phát hiện sai dữ liệu. Báo cáo trình bày RQ→đối chứng→kết quả→giới hạn, công bố cả kết quả không ủng hộ giả thuyết. Chưa có can thiệp vào activation hoặc phân tích lớp encoder; kết luận thuộc hành vi mô hình, không phải xác định cơ chế neuron.

## Bổ sung A1 — khả thi nguồn ảnh, trước mọi suy luận ảnh tự nhiên

Lọc annotation COCO val2017 bằng tiêu chí gốc chỉ cho 154 cặp có hướng thuộc 70 ảnh (quota dự phòng trái/phải/trên/dưới: 16/19/18/17). Đã khảo sát số lượng từ metadata với vài ngưỡng lỏng hơn, không chạy CLIP hoặc chọn theo accuracy; không dùng các ngưỡng lỏng đó cho bộ cuối. Quyết định mở rộng nguồn sang **COCO val2014**, vẫn giữ nguyên tất cả ngưỡng hình học, tính duy nhất của category và mục tiêu 200 ảnh/50 mỗi quan hệ. `image_split=val2014` và member `instances_val2014.json` trong config thay thế chỉ định val2017 ở mục 4. Đây là thay đổi nguồn để đủ số lượng, không phải lựa chọn mô hình/prompt theo kết quả. Spatial và category không thay đổi thiết kế; mỗi run giữ phiên bản config/protocol tại thời điểm chạy.
