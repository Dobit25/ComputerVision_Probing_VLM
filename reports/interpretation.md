# Diễn giải kết quả mốc một model — 04/11/2026

Đã hoàn thành năm full classification run (46.948 ảnh test tổng cộng) và bốn bài probing (13.240 mẫu đánh giá) bằng **OpenAI CLIP ViT-B/32**, FP32/QuickGELU, môi trường Conda `gen`. Ba ô classification mới đều được smoke test trước full run. Probing cũng qua smoke/audit trước full/audit. Không fine-tune, không thay prompt sau khi nhìn kết quả.

Nguồn số liệu: [baseline](baseline/README.md), [probing](probing/README.md), [biên bản classification](../results/one_model_classification_validation.json), [biên bản probing](../results/full_probing_clip_b32_v1_audit.json). Các bảng/biểu đồ trong hai báo cáo đầu có thể dựng lại bằng `python -m vlm_probe.report baseline` và `python -m vlm_probe.report probing`.

## 1. Baseline rộng hơn cho thấy điều gì?

| Dataset | Top-1 | Quan sát cần giữ khi đọc accuracy |
| --- | ---: | --- |
| CIFAR-100 | 64,48% | Lớp `ray` chỉ đúng 5/100; 34 ảnh bị gọi là `flatfish`. |
| Flowers102 | 63,77% | 146 ảnh `petunia` bị gọi là `mexican petunia`; cần nhìn cả support từng lớp. |
| Oxford-IIIT Pet | 85,06% | Dù accuracy tổng cao, `Bombay` đúng 0/88 và `Ragdoll` đúng 30/100. |
| DTD partition 1 | 42,45% | `dotted → polka-dotted`: 36/40; `banded → striped`: 35/40. |
| Food-101 | 82,31% | `onion_rings → fried_calamari`: 70/250; `steak → filet_mignon`: 68/250. |

Accuracy tổng thể chưa đủ để xác định model ổn trên mọi lớp. Các lớp gần nhau về hình dạng hoặc tên gọi có thể bị nhầm nhiều. Chưa thể tách riêng tác động của ảnh, độ mơ hồ của tên lớp, tập pretraining và cách diễn đạt prompt từ những bảng này. Chẳng hạn `ray` hoặc `Bombay` có thể cần mô tả rõ nghĩa hơn, nhưng đó mới là giả thuyết cho một thí nghiệm prompt riêng; kết quả hiện tại vẫn giữ tên lớp và prompt ban đầu.

DTD là miền texture, Pet là giống vật nuôi, Food là món ăn; số lớp và cấu trúc dữ liệu khác nhau. Vì vậy không coi chênh lệch 85,06% so với 42,45% là độ cải thiện model. Đây là cùng một model được đo ở các miền khác nhau.

## 2. Counting: khó khăn vẫn còn sau đối chứng nhận dạng

Đúng **355/1.080 = 32,87%**, so với đoán đều sáu số là 16,67%. Mức này cao hơn đoán ngẫu nhiên nhưng còn sai 725 mẫu. Sai số tuyệt đối trung bình là **1,0157 vật thể**.

Nhận dạng màu/hình không nêu số lượng đúng **804/1.080 = 74,44%**. Trong riêng 804 mẫu đối chứng đúng, counting đúng **277/804 = 34,45%**. Như vậy khó khăn đếm còn quan sát được khi bộ đối chứng nhận dạng thành công; đồng thời control chưa hoàn hảo nên không quy toàn bộ lỗi cho khả năng đếm.

Giữ bán kính cố định cho 32,41%; giữ tổng diện tích hình cùng loại xấp xỉ cố định cho 33,33%. Chênh lệch nhỏ này là mô tả kết quả, chưa phải kiểm định về tác động của diện tích. Accuracy theo số lượng không giảm đều: số 3 đạt 52,22%, số 4 chỉ 9,44%. Không nên diễn giải thành một quy luật đơn giản rằng model càng đếm nhiều càng sai.

## 3. Spatial: nhận dạng vật thể tốt nhưng phân biệt quan hệ kém

Đúng **213/720 = 29,58%**, so với đoán đều bốn quan hệ là 25%. Control nhận dạng hai vật thể đạt **702/720 = 97,50%**, còn spatial trên phần control đúng đạt **208/702 = 29,63%**. Khoảng cách này là bằng chứng trong bộ kiểm tra rằng nhận dạng được vật thể chưa đảm bảo chọn đúng câu mô tả quan hệ.

Các quan hệ rất không đều: above **113/180**, left **73/180**, below **27/180**, right **0/180**. Không có nhóm nào đúng cả bốn ảnh đổi vị trí. Trong confusion matrix, model thường chọn `above` ngay cả khi quan hệ thật là trái/phải/dưới. Caption, thứ tự chủ thể và ảnh đã được kiểm tra: `right` đúng là vật thể thứ nhất nằm bên phải vật thể thứ hai, không phải nhãn đảo.

Kết luận giới hạn ở bốn caption và hình học hai vật thể trong giao thức. Chưa chứng minh model không thể xử lý vị trí dưới mọi cách diễn đạt hoặc trên mọi ảnh tự nhiên.

## 4. Negation: kết quả tốt trên bài này, nhưng phụ thuộc cách diễn đạt

Đúng **1.240/1.440 = 86,11%**; chance 50%. Có 720 ảnh, mỗi ảnh dùng hai template. Present đạt 84,17%, absent đạt 88,06%; tỷ lệ chọn câu khẳng định 48,06%, không phải chỉ chọn khẳng định cho mọi ảnh.

`with/without` đạt **658/720 = 91,39%**, còn `contains/does not contain` đạt **582/720 = 80,83%** — chênh **10,56 điểm phần trăm** trên cùng tập ảnh. Đây là khác biệt quan sát được, không phải tuyên bố về mọi cách viết phủ định. Chỉ **211/360 nhóm = 58,61%** đúng đồng thời cả hai trạng thái present/absent và cả hai template.

Do đó, không được viết “CLIP không hiểu phủ định” dựa vào dự án này. Phát biểu được số liệu hỗ trợ là: checkpoint này phân biệt khá tốt sự có/không có hình–màu trong bộ ảnh đơn giản đã thiết kế, nhưng độ ổn định giữa template và cặp đối chứng chưa hoàn toàn. Các cảnh thực tế nhiều vật thể, nhiễu, che khuất hoặc cách phủ định khác chưa được kiểm tra.

## 5. Fine-grained: lỗi còn lại ngay trong nhóm lớp lớn đã biết

Đúng **7.795/10.000 = 77,95%** khi chỉ chọn giữa năm lớp thuộc **nhóm lớn thật** của ảnh. Vẫn có 2.205 lỗi trong nhóm. Nhóm fish đạt 56,60%, aquatic mammals 64,00%, trees 66,20%, small mammals 67,20%; vehicles_1 đạt 92,20%. Các coarse group không đồng đều về độ giống thị giác nên cần đọc theo nhóm.

Nhãn nhóm lớn là thông tin oracle. Vì vậy **77,95% không phải cải tiến của baseline 64,48%**: hai bài có tập ứng viên khác nhau và bài mới được cung cấp thêm thông tin thật. Khi vẫn chọn trong đủ 100 lớp bằng evaluator probing, cả 10.000 dự đoán đều khớp baseline ban đầu từng ảnh; khác biệt ở bài năm lớp đến từ giới hạn ứng viên.

## 6. Phần đã hoàn thành và phần để sau

Mốc kỹ thuật này đã có một model trên năm dataset, phân tích baseline và bốn bài core probing có giao thức, đối chứng, dự đoán, scores, metrics, CI và kiểm chứng tái lập. Các run cũ được giữ nguyên; mã nguồn thực thi của run mới được lưu trong ZIP có checksum. 24 unit tests đạt; audit full probing xác nhận 10.000 dự đoán global khớp baseline.

Chưa thực hiện bộ ảnh văn hóa Việt Nam, prompt tiếng Việt, nghiên cứu prompt sensitivity/ensembling đầy đủ, nhiều model hoặc fine-tuning. Báo cáo Markdown này là phân tích kỹ thuật của mốc thí nghiệm; báo cáo hội nghị 6–8 trang và slide cuối kỳ vẫn là sản phẩm biên soạn riêng.

Nếu muốn củng cố kết luận nghiên cứu sau mốc này, có thể thêm benchmark ảnh tự nhiên cho counting/spatial/negation và khảo sát prompt trên tập phát triển riêng. Đó là bước mở rộng phương pháp, không phải lý do sửa kết quả hoặc chọn lại prompt của các run đã báo cáo.
