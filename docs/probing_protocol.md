# Core probing v1 — chốt trước khi chạy suy luận

Phạm vi: **một checkpoint OpenAI CLIP ViT-B/32**, frozen, FP32/QuickGELU, môi trường `gen`. Không fine-tune hoặc chọn prompt theo test accuracy. Các bài dưới đây được nhóm tự thiết kế để kiểm tra có hệ thống; **không phải bản tái lập CountBench, ARO hay NegBench**.

Các công trình [Teaching CLIP to Count to Ten](https://openaccess.thecvf.com/content/ICCV2023/papers/Paiss_Teaching_CLIP_to_Count_to_Ten_ICCV_2023_paper.pdf), [ARO/bag-of-words](https://arxiv.org/abs/2210.01936) và [NegBench](https://arxiv.org/abs/2501.09425) là cơ sở lựa chọn câu hỏi kiểm tra. Kết quả của chúng không được dùng làm số đo của dự án này.

## 1. Counting: số lượng 1–6

- **1.080 ảnh:** 6 số lượng × 3 màu (red, blue, green) × 3 hình (circle, square, triangle) × 2 chế độ kích thước × 10 bố cục.
- Canvas vuông 224 × 224, nền sáng, không chữ, không vật thể bị che hoặc cắt. Raster hóa ở độ phân giải gấp 4 rồi giảm về 224 để giảm răng cưa. Seed renderer 14; vị trí chọn từ lưới 3 × 2 với jitter nhỏ và hoán vị cố định theo bố cục.
- Hai chế độ: bán kính cố định 14 px; hoặc bán kính `32/sqrt(n)` để tổng diện tích hình cùng loại xấp xỉ cố định khi thay số lượng. Báo riêng hai chế độ nhằm kiểm tra khả năng dựa vào diện tích phủ màu thay vì đếm. Đây không loại bỏ mọi tín hiệu mật độ/vị trí.
- Sáu caption: `a drawing of one red circle.`, `a drawing of two red circles.`, ... đến six; màu/hình khớp ảnh và chỉ đổi số lượng. Top-1 giữa 6 caption; chance = 1/6. Thêm MAE số đếm, confusion 6 × 6, accuracy theo số lượng/chế độ/màu/hình.
- Control: chọn màu+hình đúng giữa 9 mô tả không nêu số lượng, ví dụ `a drawing of red circles.`. Báo accuracy control và accuracy counting trên các mẫu control đúng, với mẫu số rõ ràng. Metric chính vẫn dùng toàn bộ ảnh.

## 2. Spatial: left/right/above/below

- **720 ảnh:** 6 cặp màu có thứ tự × 6 cặp hình khác nhau có thứ tự × 4 quan hệ × 5 bố cục.
- Hai vật thể khác cả màu và hình, cùng nền/kích thước trong một nhóm; chỉ đổi vị trí để tạo bốn ảnh phản thực. Cả hai vật thể luôn nằm trọn trong ảnh, không chồng nhau.
- Ví dụ bốn caption: `a drawing of a red circle to the left of a blue square.`; thay phần quan hệ bằng `to the right of`, `above`, `below`. Giữ nguyên hai chủ thể và thứ tự ngữ pháp.
- Top-1 bốn lựa chọn (chance 25%); accuracy và signed margin theo quan hệ; tỷ lệ nhóm mà **cả bốn ảnh** đều đúng để kiểm tra sự nhất quán khi đổi vị trí.
- Control: caption nhận dạng hai vật thể, đối lập với caption thay màu của vật thể thứ nhất bằng màu thứ ba không có trong ảnh. Hoán đổi vị trí caption đúng/sai theo index để không thiên về đáp án đầu.

## 3. Negation: có/không có vật thể đích

- **720 ảnh duy nhất, 1.440 phép thử caption:** 36 cặp vật thể × 10 bố cục × 2 trạng thái present/absent; mỗi ảnh đánh giá hai cách diễn đạt đã chốt.
- Mỗi ảnh luôn có hai vật thể. Cặp present/absent giữ vị trí/hình/kích thước vật thể, chỉ đổi màu vật thể đích sang màu thứ ba; do đó tổng số lượng không tiết lộ đáp án.
- Mẫu A: `a drawing with a red circle.` / `a drawing without a red circle.`
- Mẫu B: `a drawing that contains a red circle.` / `a drawing that does not contain a red circle.`
- Positive và negative là hai caption ứng viên của cùng ảnh; nhãn đúng cân bằng present/absent. Báo accuracy chung và riêng từng template, từng polarity, tỷ lệ chọn câu khẳng định, tỷ lệ cả hai ảnh trong cặp đều đúng. Chance cho mỗi phép thử là 50%.
- Control khẳng định: chọn mô tả đúng của cặp vật thể thực sự hiện diện, không dùng phủ định. Giữ cùng ảnh nên có thể phân biệt khó khăn nhận dạng cơ bản với khó khăn trong caption phủ định. Các kết quả control đúng chỉ là phân tích điều kiện, không loại mẫu khỏi metric chính.
- Hai template dùng lại ảnh; không coi 1.440 dòng là 1.440 ảnh độc lập. Khi tính khoảng tin cậy, resample theo nhóm ảnh liên quan, không theo từng dòng caption.

## 4. Fine-grained trên ảnh tự nhiên

- Dùng **10.000 ảnh CIFAR-100 test** và ánh xạ fine/coarse chính thức trong archive đã kiểm checksum.
- Với mỗi ảnh, đánh giá giữa **5 fine classes trong coarse class thật**. Ví dụ một nhóm hoa hoặc nhóm cá có năm nhãn gần nhau về ngữ nghĩa. Dùng nguyên prompt baseline `a photo of a {class}.` và ID lớp gốc.
- Báo accuracy toàn tập và từng coarse group, confusion theo fine class, signed margin; chance 20% trong nhóm 5 lớp. Lưu cả dự đoán global 100 lớp để đối chiếu với baseline đã có.
- **Đây là phép chẩn đoán có cung cấp coarse label thật (oracle)**, không phải cách cải thiện zero-shot accuracy có thể triển khai mà không biết nhãn. Không so trực tiếp accuracy 5-way với 100-way như một cải tiến model. Các nhóm coarse của CIFAR không đồng đều về mức độ gần giống thị giác, nên kết luận phải ghi theo nhóm.
- Bổ sung quan sát ảnh tự nhiên từ accuracy theo lớp và cặp nhầm lẫn của Flowers102, Pet và Food-101 trong báo cáo baseline. Việc chọn cặp lỗi để minh họa là phân tích hậu nghiệm, không là benchmark xác nhận độc lập.

## 5. Scoring, kiểm chứng và đầu ra

Encode từng ảnh bằng preprocessing checkpoint; encode các caption duy nhất rồi cache vector text trong bộ nhớ. Chuẩn hóa L2, tính cosine, argmax; exact tie chọn index nhỏ nhất và báo tỷ lệ tie/near-tie riêng (`abs(top1-top2) <= 1e-6`). Không coi near-tie là sai tự động hoặc thay quy tắc argmax.

Mỗi mẫu lưu ID/nhóm, task/điều kiện, caption ứng viên, nhãn đúng, mọi cosine score, prediction, correctness, signed margin (`score đúng - score foil tốt nhất`), control scores/labels nếu có, đường dẫn và hash ảnh nguồn. Manifest ghi config, seed, môi trường, checkpoint, source snapshot, checksum toàn bộ output. Dataset nhân tạo có scene metadata (hình/màu/tọa độ/kích thước) để kiểm tra nhãn mà không cần suy luận model.

Smoke chạy một số mẫu đầu của từng task để kiểm tra kỹ thuật; full mới là kết quả báo cáo. Test xác nhận cân bằng lớp, số vật thể, bố cục, cặp phản thực và schema. Xem contact sheet trước khi chạy model. Verifier tính lại prediction/metric từ scores đã lưu; fine-grained phải khớp global predictions của baseline cũ trên cùng index.

Accuracy được báo kèm khoảng tin cậy bootstrap 95% theo **nhóm** (1.000 bootstrap draws, seed 0). Nhóm counting gộp các số lượng của cùng màu/hình/chế độ/bố cục; spatial gộp bốn quan hệ; negation gộp present/absent và hai template; fine-grained dùng ảnh làm đơn vị. Đây là bất định trong bộ kiểm tra đã thiết kế, không đo độ bao phủ ảnh ngoài đời.

## 6. Giới hạn kết luận

Ảnh hình học có miền khác ảnh tự nhiên; kết quả thấp có thể liên quan nhận dạng hình/màu, cách diễn đạt hoặc khả năng kết hợp khái niệm. Control giúp chẩn đoán nhưng không chứng minh nguyên nhân bên trong model. Không tuyên bố CLIP "không hiểu" một khái niệm chỉ từ vài ví dụ. Không suy rộng kết quả một checkpoint sang mọi VLM. Đây là core probing ban đầu có hệ thống; đối chiếu thêm benchmark ảnh tự nhiên là hướng củng cố nghiên cứu sau mốc này.
