# Review foil T1 — hướng dẫn cho người chấm

Trạng thái: **pending_review** (vòng 1 đã sinh ngày 18/11/2026, chưa có người chấm). Protocol: [natural_recognition_protocol_v1.md](../../docs/natural_recognition_protocol_v1.md) mục 4–5.

## Tệp

| Tệp | Vai trò |
|---|---|
| `candidates.json` | Danh sách foil xếp hạng cho mỗi nguồn/vai trò, lớp có annotation trong ảnh, hash ảnh và canvas. Sinh không dùng CLIP scores |
| `round_01_blank.csv` | Phiếu trắng vòng 1: 196 dòng = 98 nguồn × 2 vai trò. **Không sửa tệp này** |
| `round_01_filled.csv` | (người chấm tạo) bản đã điền, cùng thứ tự và cột |
| `freeze.json` | (sinh bởi lệnh `freeze`) checksum mọi vòng và foil cuối cùng |

Gói xem ảnh nằm ngoài Git: `data/natural_recognition_v1/review_round_01/index.html` (mở bằng trình duyệt trên máy có repo; ảnh COCO không được phân phối lại qua Git). Tạo lại bằng `python -m vlm_probe.spatial_diagnostics.natural_recognition prepare` (tải COCO, kiểm sha256).

## Cách điền

1. Sao chép `round_01_blank.csv` thành `round_01_filled.csv`. Chỉ điền các cột `reviewer_id, foil_absent, unambiguous, decision, reason, notes`; không đổi cột khác.
2. Ảnh là đúng vùng 224×224 mô hình thấy (phóng 2×), khung xanh = A, cam = B. Bản lật ngang là ảnh gương nên một quyết định dùng cho cả hai.
3. `foil_absent`: `yes` nếu foil **không** xuất hiện trong vùng ảnh; `no` nếu thấy; `unsure` nếu không chắc.
4. `unambiguous`: `yes` nếu người xem không thể gọi vật bị thay (hoặc vật khác trong ảnh) bằng tên foil.
5. `decision`: `accepted` (chỉ khi cả hai là `yes`), `revise` (foil không dùng được → vòng sau đổi foil kế tiếp, tối đa 3 phiên bản), `uncertain` (cần phân xử). `revise`/`uncertain` bắt buộc có `reason`.
6. Kiểm: `python -m vlm_probe.spatial_diagnostics.natural_recognition validate --review reviews/natural_recognition_v1/round_01_filled.csv`.
7. Nếu còn `revise`/`uncertain`: chạy lại `prepare` để sinh vòng 2. Khi mọi dòng đã giải quyết và nhóm đã chốt config (`status: frozen`): chạy `freeze`.

Không có điểm hay dự đoán của mô hình trong gói này; đừng xem kết quả T1 trước khi đóng băng. Nếu nhóm đổi quy tắc foil (ví dụ `contested_pairs`) **trước khi có ai bắt đầu chấm**, xóa `candidates.json` và `round_01_blank.csv` chưa dùng rồi chạy lại `prepare`, ghi lý do vào commit; sau khi đã có phiếu điền thì không sinh lại.
