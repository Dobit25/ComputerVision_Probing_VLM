# Báo cáo tiến độ gửi giảng viên

**Bản review tích hợp mới nhất (11/11/2026):** [PDF nghiên cứu](../research_review_v2/bao_cao_nghien_cuu.pdf), [nguồn số liệu và trạng thái](../research_review_v2/README.md). Các PDF bên dưới là mốc lịch sử được giữ nguyên để đối chiếu.

**Bản mới nhất sau kiểm chứng tập ảnh mới:** [PDF cập nhật](../natural_confirmation_v1/bao_cao_cap_nhat.pdf), [kết quả và khả năng tái lập](../natural_confirmation_v1/README.md). Bao gồm tự rà soát nhãn cũ và phép ghép cặp định trước trên 100 nguồn ảnh mới.

**Mốc probing chuyên sâu trước phần xác nhận mới:** [Báo cáo probing chuyên sâu (PDF)](../probing_depth_v1/bao_cao_probing_chuyen_sau.pdf), bổ sung can thiệp không gian, đối chứng category và kiểm chứng trên ảnh tự nhiên. [Bảng kết quả, giới hạn và hướng dẫn tái lập](../probing_depth_v1/README.md). Bản bên dưới giữ nguyên để đối chiếu mốc trước phần mở rộng.

- [Bản PDF](bao_cao_tien_do.pdf): báo cáo tiếng Việt, cập nhật ngày 04/11/2026.
- [Mã nguồn LaTeX](bao_cao_tien_do.tex): chỉnh sửa nội dung và biên dịch lại.
- [Nguồn số liệu](sources.json): đường dẫn, SHA-256 của kết quả thực nghiệm được sử dụng.

Báo cáo tổng hợp năm baseline phân loại, bốn phép thăm dò và thí nghiệm bổ sung category cho CIFAR-100. Phân biệt rõ kết quả thực đo, giới hạn phương pháp và phần công việc chưa thực hiện. Mốc `v1.0.0-core` chứa phần core; thí nghiệm category là phần bổ sung cục bộ trên nhánh `dev` tại thời điểm lập báo cáo.

Ngày 05/11/2026, các bổ sung được đóng gói trong mốc `v1.1.0-probing-depth`. PDF, LaTeX và `sources.json` lịch sử giữ nguyên thông tin tại thời điểm lập; xem [tài liệu phiên bản](../../docs/probing_depth_release.md) để biết trạng thái bàn giao hiện hành.

## Biên dịch

Nguồn UTF-8, dùng XeLaTeX hoặc Tectonic; không dùng pdfLaTeX. Font chính Times New Roman, dự phòng TeX Gyre Termes. Không cần BibTeX vì tài liệu tham khảo được khai báo trực tiếp trong nguồn.

Từ thư mục repo, với Tectonic đã cài:

```powershell
tectonic --outdir reports/progress_report reports/progress_report/bao_cao_tien_do.tex
```

Bản PDF đính kèm được tạo bằng Tectonic 0.17.0, tải từ bản phát hành chính thức và kiểm tra SHA-256. Bản portable và cache nằm trong `.cache/report_tools/`, không thay đổi môi trường conda `gen`. Để dùng lại bản portable trên máy hiện tại:

```powershell
$env:TECTONIC_CACHE_DIR = Join-Path (Get-Location) '.cache/report_tools/texcache'
& .\.cache\report_tools\tectonic\tectonic.exe --outdir reports/progress_report reports/progress_report/bao_cao_tien_do.tex
```

Hoặc chạy XeLaTeX hai lần để cập nhật tham chiếu bảng/tài liệu:

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory reports/progress_report reports/progress_report/bao_cao_tien_do.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory reports/progress_report reports/progress_report/bao_cao_tien_do.tex
```

Các ví dụ lớp tốt/xấu được dùng để minh họa sau phân tích; bảng kết quả toàn bộ lớp được giữ trong các nguồn dữ liệu. Không cần chạy lại mô hình để biên dịch báo cáo.
