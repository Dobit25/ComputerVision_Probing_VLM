# Review foil T1

Trạng thái: **vòng 1 đã chấm xong** (18/11/2026, 6 người chấm). Còn 13 dòng `revise`/`uncertain` chưa giải quyết, nên chưa `freeze` được. Protocol: [natural_recognition_protocol_v1.md](../../docs/natural_recognition_protocol_v1.md) mục 4–5.

## Tệp

| Tệp | Vai trò |
|---|---|
| `candidates.json` | Danh sách foil xếp hạng cho mỗi nguồn/vai trò, lớp có annotation trong ảnh, hash ảnh và canvas. Sinh không dùng CLIP scores |
| `round_01_blank.csv` | Phiếu trắng vòng 1: 196 dòng = 98 nguồn × 2 vai trò. Giữ nguyên, là mốc để `validate` so |
| `round_01_filled.csv` | Kết quả vòng 1 do nhóm chấm. Hợp lệ theo `validate` |
| `freeze.json` | (sinh bởi lệnh `freeze`) checksum mọi vòng và foil cuối cùng |

Ảnh review (vùng 224×224 mô hình thấy, khung A/B) nằm ngoài Git ở `data/natural_recognition_v1/review_round_01/img/`.

## Kết quả vòng 1

`python -m vlm_probe.spatial_diagnostics.natural_recognition validate --review reviews/natural_recognition_v1/round_01_filled.csv` → `Round 1 valid: {'accepted': 183, 'uncertain': 8, 'revise': 5}`.

| `reviewer_id` | Ảnh số | Dòng |
|---|---|---|
| `tuan_dat` | 1–17 | 34 |
| `duong_hieu` | 18–34 | 34 |
| `phong` | 35–50 | 32 |
| `quan` | 51–66 | 32 |
| `trinh` | 67–82 | 32 |
| `tung` | 83–98 | 32 |

Sáu người chấm sáu khối ảnh khác nhau, mỗi dòng một người: không tính được inter-annotator agreement.

Dòng chưa giải quyết:

| Quyết định | `review_id` |
|---|---|
| `revise` (foil có trong ảnh) | NR1_185614_replace_b_v1, NR1_455937_replace_b_v1, NR1_277289_replace_a_v1, NR1_355325_replace_a_v1, NR1_375232_replace_a_v1 |
| `uncertain` (không xác định được từ crop) | NR1_364444_replace_a_v1, NR1_257198_replace_b_v1, NR1_448795_replace_a_v1, NR1_436273_replace_a_v1, NR1_436273_replace_b_v1, NR1_490701_replace_b_v1, NR1_554348_replace_b_v1, NR1_294475_replace_b_v1 |

## Bước tiếp

`freeze` từ chối khi còn dòng chưa giải quyết. Nhóm cần chọn một trong hai:

- chạy `prepare` để sinh `round_02_blank.csv` (13 dòng: `revise` lấy foil kế tiếp, `uncertain` đưa ra phân xử) rồi chấm vòng 2; hoặc
- chốt trong config quy tắc thiếu foil (hiện là `PROPOSED`) để loại các nguồn này khỏi phân tích.

Sau đó: nhóm chốt config (`status: frozen`), chạy `freeze`, rồi `run`.
