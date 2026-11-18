# Đối chiếu phiếu người review và đánh giá agent — 16/11/2026

## Phạm vi và tính toàn vẹn

Phiếu nguồn: `../confirmation_100_blank.csv`. SHA-256 khi nhận: `99eb663af45d272afd3ce6f8d9a4649a9e34fbb15b7e24ddd4665da7d82d70ad`.

**Cập nhật trong lúc đối chiếu:** người dùng đã sửa 005 thành right và 040 thành left. Snapshot mới nhất đã đọc là `human_latest_received.csv`, SHA-256 `37a6b9aedd34047e907785859b427093883388fcd92d7129f7590c23444eae36`. Bản mới có **85 dòng nhất quán sau bỏ khoảng trắng, 15 dòng chưa chấm, không còn hai mâu thuẫn hướng**. Các con số 83 và hai mâu thuẫn bên dưới mô tả snapshot ban đầu, không phải trạng thái cuối. Việc sửa 040 chưa giải quyết vấn đề nĩa bị cắt khỏi crop. Cả 100 ảnh nguồn đã kiểm hash và 100 crop xuất sẵn khớp từng pixel với hàm preprocessing baseline.

Không sửa phiếu nguồn, trạng thái human review, nhãn đã cố định hoặc kết quả thí nghiệm. Các đề xuất bên dưới là nhận xét của agent hiện tại để người review cân nhắc, không phải quyết định do R1 xác nhận.

Đối chiếu 100 ID với key riêng, selected.json và `reports/natural_confirmation_v1/new_review_decisions.csv`, bao gồm khớp source image ID và SHA-256 ảnh. Đã xem contact sheet crop của toàn bộ 100 ảnh; xem ảnh nguồn có box cho 17 mục chưa hoàn thiện/mâu thuẫn và các mục 031,045,075,089,095. Nhận xét thị giác vẫn có thể sai và không thay thế reviewer người.

## Kết quả kiểm phiếu

- Đủ 100 dòng, 100 ID duy nhất; object_a/object_b/expected_relation giữ nguyên.
- 85 dòng ghi overall_accept=yes, không có dòng no; 15 dòng chưa điền năm trường đánh giá chính.
- Trong 85 dòng, 83 dòng nhất quán sau khi bỏ khoảng trắng; 005 và 040 ghi perceived_relation ngược expected_relation nhưng vẫn nhận ảnh.
- 301 ô thuộc 85 dòng có khoảng trắng đầu/cuối. Validator hiện yêu cầu giá trị chính xác nên phiếu nguồn chưa hợp lệ. Bản đã bỏ khoảng trắng chỉ sửa định dạng, vẫn chưa hoàn thiện các quyết định.
- 15 dòng để trống: 000,003,006,012,013,018,024,033,034,054,068,077,087,092,096.

## So với agent trước

Agent trước đã nhận 112/178 ảnh ứng viên; từ đó bộ đã cố định chọn 100 ảnh. Cả 100 ảnh trong phiếu hiện tại đều có accepted=True trong quyết định cũ. Đây là hệ quả của bước tuyển chọn, không phải bằng chứng agent đúng 100%.

Trong `confirmation.py:freeze_review`, mẫu không nằm trong danh sách loại được ghi accepted=True và nhận cùng một câu lý do mặc định. Hồ sơ cũ không lưu riêng ba câu hỏi yes/no và perceived_relation như phiếu người hiện tại. Vì vậy chỉ đối chiếu trực tiếp được quyết định nhận/loại, không được tự suy diễn agent đã trả lời yes từng trường hoặc có lý giải riêng cho từng ảnh.

83 dòng nhất quán của người review hiện đều đồng ý nhận như agent trước. Hai dòng mâu thuẫn và 15 dòng chưa chấm không được tính là bất đồng/đồng thuận hoàn chỉnh. Không báo 100% chất lượng nhãn; không có hai reviewer người độc lập; không tính Cohen kappa có ý nghĩa từ các quyết định chỉ có một nhãn yes.

Code review trước (`depth/natural.py:contact_sheets`) tạo crop vuông liên tục từ ảnh gốc rồi resize 312, chồng thêm box và xem 16 ảnh/trang. Gói người hiện tại có crop baseline 224 thật, không chồng box. Cách trình bày và độ phân giải khác nhau có thể ảnh hưởng nhận định về vật thể nhỏ, nhưng chưa chứng minh chúng gây ra các sai sót cụ thể.

## Ghi chú cho từng mục còn vướng

Các số là ba chữ số cuối blind_id. “Nghiêng về nhận/loại” là đề xuất của agent hiện tại; người chấm cần tự quyết định và điền lại.

| ID | Quan sát và hướng xử lý đề xuất |
|---|---|
| 000 | Người ở trái lưng ghế có thanh chữ X. Ghế bị che một phần nhưng vẫn nhận dạng được; nghiêng về nhận, perceived_relation=left. Không đòi ghế phải hiện trọn vẹn. |
| 003 | Sofa ở phải màn hình nhìn nghiêng sát bên trái crop. Nghiêng về nhận nếu reviewer vẫn nhận ra màn hình; nếu chỉ thấy một mảnh không đủ nhận dạng, loại OCCLUDED_OR_TINY. |
| 006 | Bánh phủ chocolate bị tay/miệng che một phần, ghế đen phía phải. Không đủ chắc bánh là donut thay vì muffin/cupcake. Nên ghi chưa xác nhận tên lớp và cân nhắc AMBIGUOUS_LABEL; không suy ra loại bánh chỉ từ nhãn có sẵn. |
| 012 | Màn hình CRT ở phải chồng sách/tài liệu ở trái. Hướng right rõ, nhưng cần thống nhất cách hiểu nhãn tv và nhận diện đúng cuốn sách B, không chọn sách khác. |
| 013 | Laptop trái, màn hình máy tính phải. Quan hệ left rõ; vấn đề nằm ở cách dùng tv cho monitor, không ở quan hệ. |
| 018 | Ly trong tay trái ảnh, điện thoại cạnh tai phải ảnh; left rõ. Điện thoại bị tay che một phần nhưng vẫn có dấu hiệu hình dạng; nghiêng về nhận nếu reviewer tự nhận diện được. |
| 024 | Chó trái, bàn thấp giữa/phải. Quan hệ left rõ, nhưng nhãn dining table cho bàn thấp phòng khách cần xem lại. Nếu không chấp nhận theo quy tắc tên lớp đã chốt, dùng AMBIGUOUS_LABEL. |
| 033 | Laptop trái, màn hình nhìn nghiêng phải; crop cắt bớt laptop ở mép dưới nhưng còn màn hình/bàn phím. Nhận nếu đủ nhận diện hai vật và đã thống nhất tv/monitor; ngược lại ghi lý do cụ thể. |
| 034 | Laptop phải, mèo nằm trái; đầu mèo vẫn nhận ra dù bị che thân. Nghiêng về nhận, right. |
| 054 | Màn hình để bàn phải người; right rõ. Cần cùng quy tắc tv/monitor như 012/013/033. |
| 068 | Cụm lò nằm phải tủ lạnh. Cần chỉ đúng phần oven phía dưới, không nhận oven chỉ vì microwave phía trên rõ; loại nếu reviewer không đủ nhận diện oven bị đảo bếp che. |
| 077 | A là thiết bị mỏng đang đóng ở phải màn hình, có vật đen đặt trên; khó xác nhận laptop chỉ từ crop. Nghiêng về loại AMBIGUOUS_LABEL nếu reviewer không nhận ra thiết bị, đồng thời còn vấn đề tv/monitor. |
| 087 | Sink phải, toilet trái bị che một phần nhưng còn lòng bồn và đặc trưng nhận dạng. Nghiêng về nhận, right. |
| 092 | Vải/rèm che phần lớn đối tượng A, khó xác nhận person một cách độc lập dù ghế trái rõ. Nghiêng về loại OCCLUDED_OR_TINY; không kết luận chắc A không phải người. |
| 096 | Phần đồ ăn trên đĩa phải chai bên trái. Hướng right rõ nhưng pizza không thật chắc do đồ phủ và hình dạng; cân nhắc AMBIGUOUS_LABEL nếu reviewer không nhận dạng được. |
| 005 | Người thực tế ở phải chó. Nếu left là lỗi nhập, reviewer đổi perceived_relation thành right và giữ nhận. Nếu reviewer thực sự thấy left, giữ nhận xét đó và đổi overall_accept=no, OTHER, ghi rõ bất đồng. |
| 040 | Nĩa A trong ảnh nguồn ở trái dao B. Crop cắt mất phần lớn nĩa; phần còn lại sát mép không đủ chắc là nĩa. Không đơn thuần sửa right thành left để nhận ảnh. Nghiêng về loại OCCLUDED_OR_TINY; ghi rõ nguồn và crop khác nhau về mức nhìn thấy. |

## Hai mục đã nhận nên xem lại

- **075 — couch/tv:** box A khoanh ghế bành đơn ở xa, không phải chiếc giường lớn. Quan hệ trái đúng nhưng nhãn couch cần phân xử; cân nhắc AMBIGUOUS_LABEL. Không đổi đối tượng mục tiêu sang giường hoặc ghế khác để hợp thức hóa nhãn.
- **095 — keyboard/dining table:** bàn phím mờ nhưng còn thấy; box B là bàn nhỏ tròn/chân hẹp, chưa đủ chắc dining table. Cần kiểm nghĩa lớp; cân nhắc AMBIGUOUS_LABEL, không dùng việc “đã có bbox” làm bằng chứng tên lớp đúng.

## Quy tắc nên thống nhất khi hoàn thiện

1. Đánh giá trên crop baseline; ảnh nguồn chỉ xác định box A/B, không bù chi tiết mất khỏi crop. Không cần toàn bộ vật thể hiện nguyên vẹn, nhưng phải có đủ bằng chứng nhận diện.
2. Trong dữ liệu hiện tại nhiều monitor được gán tv. Phân biệt kiểm đúng taxonomy dataset và kiểm caption có nghĩa tự nhiên. Ghi rõ lựa chọn cách hiểu; dùng cùng quy tắc cho mọi ảnh, không đổi theo scores. Chưa có cơ sở từ phiếu này để khẳng định mọi monitor đều là TV theo nghĩa thông thường.
3. Nếu không xác nhận tên lớp, class_names_correct=no có thể nghĩa “không đủ bằng chứng”, không nhất thiết “chắc chắn sai”; notes cần nói rõ. relation_clear vẫn có thể yes khi vị trí hai đối tượng được chỉ định rõ.
4. Mục chưa biết để trống trong bản nháp; chỉ dùng no như quyết định cuối sau khi người review cân nhắc. Không điền agent thành R1.
5. Giữ quyết định ban đầu, ghi phân xử riêng; vì đã trao đổi đánh giá agent, phần chỉnh sửa tiếp theo là sau thảo luận, không gọi nó là vòng chấm độc lập hoàn toàn mới.
6. Khi phiếu hoàn chỉnh, lưu tên mới confirmation_100_R1.csv UTF-8, validate rồi mới tính sensitivity trên các source được nhận và cả ảnh gốc/lật tương ứng. Giữ nguyên kết quả gốc 100 nguồn, không chọn lọc dựa trên kết quả model.

## Các tệp kèm theo

- `human_original.csv`: snapshot nguyên byte phiếu đã nhận.
- `human_latest_received.csv`: snapshot nguyên byte bản người dùng sửa trong lúc đối chiếu. Bảng so sánh và bản bỏ khoảng trắng hiện theo snapshot này.
- `human_whitespace_normalized_NOT_FINAL.csv`: chỉ bỏ khoảng trắng; giữ nguyên phần trống và quyết định, chưa hợp lệ cuối.
- `human_agent_comparison.csv`: 100 dòng ghép theo source ID/hash, trạng thái và quyết định/lý do cũ của agent.
- `summary.json`: thống kê và checksum.
- `focus_*.png`, `accepted_*.png`, `detail_*.png`: ảnh ghép hỗ trợ rà soát, không phải dữ liệu suy luận mới.
