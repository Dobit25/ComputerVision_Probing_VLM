# Review foil T1

Trạng thái: **đã chấm xong, mọi dòng đã giải quyết** (vòng 1 và 2, ngày 18/11/2026): 98 nguồn × 2 vai trò = 196 foil được chấp nhận. **Đã đóng băng** (`freeze.json`) sau khi nhóm chốt config (`status: frozen`). Protocol: [natural_recognition_protocol_v1.md](../../docs/natural_recognition_protocol_v1.md) mục 4–5.

## Tệp

| Tệp | Vai trò |
|---|---|
| `candidates.json` | Danh sách foil xếp hạng cho mỗi nguồn/vai trò, lớp có annotation trong ảnh, hash ảnh và canvas. Sinh không dùng CLIP scores |
| `round_01_blank.csv`, `round_02_blank.csv` | Phiếu trắng mỗi vòng. Giữ nguyên, là mốc để `validate` so |
| `round_01_filled.csv`, `round_02_filled.csv` | Kết quả do nhóm chấm. Cả hai hợp lệ theo `validate` |
| `freeze.json` | (sinh bởi lệnh `freeze`) checksum mọi vòng và foil cuối cùng |

Ảnh review (vùng 224×224 mô hình thấy, khung A/B) nằm ngoài Git ở `data/natural_recognition_v1/review_round_NN/img/`.

Reviewer: **`tuan_dat`**, một người chấm toàn bộ hai vòng. Không có inter-annotator agreement.

Ghi chú lịch sử: bản `round_01_filled.csv` commit đầu tiên ghi 6 `reviewer_id` theo kế hoạch chia việc (tuan_dat, duong_hieu, phong, quan, trinh, tung). Thực tế Tuấn Đạt chấm cả 196 dòng, nên `reviewer_id` được sửa thành `tuan_dat` trong một commit riêng; các cột khác giữ nguyên từng ký tự. Git giữ bản cũ.

## Vòng 1: 196 dòng

`validate` → `{'accepted': 183, 'uncertain': 8, 'revise': 5}`.

## Vòng 2: 13 dòng

`validate` → `{'accepted': 13}`.

- 5 dòng `revise` ở vòng 1 (foil có trong ảnh) được đổi sang foil kế tiếp (`_v2`) và đều được chấp nhận: 185614 knife, 455937 mouse, 277289 bed, 355325 knife, 375232 bed.
- 8 dòng `uncertain` ở vòng 1 được phân xử (`_adj`, cùng foil, xem thêm ảnh COCO gốc) và đều được chấp nhận: 364444 cell phone, 257198 cake, 448795 microwave, 436273 toaster và scissors, 490701 wine glass, 554348 suitcase, 294475 wine glass.

Phân xử do chính người chấm vòng 1 thực hiện (xem lại cùng foil, có thêm ảnh COCO gốc), không phải người thứ hai độc lập.

Tổng hợp hai vòng (`resolve`): 196/196 `accepted`, không nguồn nào bị loại vì thiếu foil.

## Bước tiếp

Review và config đã đóng băng; không sửa các tệp trong thư mục này nữa (checksum trong `freeze.json`). Tiếp theo: `run --smoke`, rồi `run`, rồi `audit` (cần máy có `open_clip` và CUDA).
