# P6 — đề cương báo cáo tối đa 10 trang và trình bày 15 phút

Cập nhật 11/11/2026 theo yêu cầu mới của người dùng. Yêu cầu gốc môn học ghi 6–8 trang; cần đối chiếu khi nộp chính thức. P5 trong kế hoạch dài hạn là bộ ảnh văn hóa Việt Nam; tài liệu này thuộc **P6**.

## Luận điểm và bằng chứng

1. Baseline năm miền và bốn bài core cho biết khả năng nhận dạng và các giới hạn theo từng thiết kế. Không gộp 2-way/4-way/6-way/oracle thành một accuracy. Negation tốt ở cảnh đơn giản vẫn là kết quả có giá trị.
2. Phân rã hiệu điểm trong cặp giải thích quan hệ đại số giữa cùng nhãn, assignment và both-correct. Các thành phần B/δ mô tả scores; chưa định vị bias trong encoder hoặc chứng minh cơ chế neuron.
3. So sánh prompt cố định trên năm dataset; embedding ensemble là so sánh chính, logit/probability là phụ. Báo đủ dataset và mọi điều kiện, kể cả ensemble giảm accuracy. Best observed trên test không phải phương pháp được chọn độc lập.
4. So sánh EN/VI trên cùng ảnh, thêm control lai khung câu/tên lớp. Phân mảnh BPE, ngữ nghĩa bản dịch, mức quen thuộc từ vựng và căn chỉnh embedding cùng thay đổi; chưa quy được 84% mức giảm cho tokenizer.
5. Phương pháp chia nửa ảnh dùng prompt nhận dạng vật thể và quy tắc kết hợp vùng. Synthetic 100% không chứng minh holistic CLIP đã hiểu quan hệ. Chỉ dùng kết quả COCO thực đo sau audit; mô phỏng bbox không là accuracy CLIP hoặc trần tổng quát.

## Bố cục báo cáo

| Phần | Nội dung | Dự kiến |
|---|---|---:|
| Tóm tắt, mở đầu và related work | Câu hỏi, phạm vi, đóng góp và đối chiếu CLIP/ARO/Winoground/NegBench | 1–1,5 trang |
| Methodology | Checkpoint, dữ liệu/split, prompt, controls, bootstrap, đơn vị độc lập, nhãn | 1–1,5 trang |
| Baseline/core/category | Đủ năm dataset, bốn probe, oracle và đối chứng category | 1 trang |
| Prompt đa miền | Bảng top-1/macro, paired delta/CI, sensitivity, ensemble | 1–1,5 trang |
| Spatial | Consistency, B/δ, kết quả chia vùng có CI và delta so holistic cùng ảnh | 1–1,5 trang |
| Tiếng Việt | Điều kiện ngôn ngữ, bản dịch, BPE và giới hạn nhân quả | 1 trang |
| Giới hạn, tái lập, kết luận và references | Review người, dữ liệu đã xem, trạng thái tests/audit và phần chưa làm | 1–1,5 trang |

Nguồn hiện hành: `reports/research_review_v2/bao_cao_nghien_cuu.tex`, sinh từ `python -m vlm_probe.review_report`. Mọi bảng đọc từ artifact; không chép số cố định vào đề cương. Phần chưa chạy ghi chưa chạy. Nguồn/checksum nằm trong `sources.json`.

Thống kê tokenizer cần ghi đúng định nghĩa: trung bình token tên lớp VI có dấu là 7,05, EN là 1,19; **tỷ số hai trung bình khoảng 5,92**. Con số 6,44 là **trung bình tỷ số theo từng lớp**, không phải 7,05/1,19. Không có câu vượt context trong bộ hiện tại. Số liệu cũ 4,27/1,31 không khớp artifact mới nên đã bỏ.

## Slides và diễn tập

Đề xuất 10–12 slides: bài toán/RQ; pipeline và dữ liệu; baseline; core/fine-grained; category controls; spatial consistency; chia vùng và giới hạn; prompt đa miền; VI và control lai; limitations/tái lập; kết luận. Tập nói 12–13 phút để có dự phòng. Backup slides giải thích 53% assignment so 4% both-correct, oracle 5-way, CI và trạng thái dữ liệu đã xem.

Không dùng p>0,05 để kết luận tương đương ngẫu nhiên. Không dùng kết quả chia vùng COCO mô phỏng từ box như kết quả inference. Mẫu chấm mù đã xuất không đồng nghĩa review người đã hoàn thành.

## Chạy và kiểm tra

```powershell
python -m unittest discover -s tests -v
python -m vlm_probe.integration --execute
python -m vlm_probe.human_review export
python -m vlm_probe.review_report
tectonic --outdir reports/research_review_v2 reports/research_review_v2/bao_cao_nghien_cuu.tex
```

Lệnh integration thực thi hàng đợi theo `docs/integration_protocol_v2.md`; cần gen/CUDA đã hoạt động. Human review export chỉ tạo gói/phiếu trắng. Không có phiếu người thì report vẫn phải giữ giới hạn đó. Review tích hợp cần kiểm manifest cũ nguyên vẹn, tests, audit run mới, chất lượng PDF và đủ điều kiện hoàn thành trước khi nhập dev.
