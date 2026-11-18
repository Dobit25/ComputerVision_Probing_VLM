"""Generate the current Vietnamese Markdown handoff from replayed artifacts."""
from .human_review_followup import ROOT, REVIEW, OUTPUT, REPORT
from .depth.common import read_json
from .io_utils import write_json, sha256


def main():
    metrics = read_json(OUTPUT / 'metrics.json')
    rows = []
    def number(v): return f'{v:.2f}'.replace('.', ',')
    def ci(values): return '[' + '; '.join(number(v) for v in values) + ']'
    for method, title in [('holistic','Holistic T2/o0'), ('native_half','Chia vùng native (primary)'), ('padded_half','Chia vùng padding (secondary)')]:
        for label, title_subset in [('before_100','Trước: 100 nguồn / 200 ảnh'), ('after_98','Sau: 98 nguồn / 196 ảnh')]:
            m = metrics['region_comparison'][label][method]
            rows.append(f"| {title} | {title_subset} | {m['correct_images']}/{m['n_images']} = {number(m['primary_accuracy_percent'])} | {ci(m['primary_ci95_percent'])} | {number(m['both_images_correct_percent'])} | {number(m['horizontal_assignment']['accuracy_percent'])} {ci(m['horizontal_assignment']['ci95_percent'])} |")
    deltas=[]
    for label, text in [('before_100','100 nguồn'), ('after_98','98 nguồn')]:
        for method in ['native_half','padded_half']:
            m=metrics['region_comparison'][label][method]
            deltas.append(f"| {text} | {method} | {number(m['paired_delta_pp'])} | {ci(m['paired_delta_ci95_pp'])} |")
    h=metrics['holistic_all_conditions_after']['t2_o0']
    vi=metrics['translation_review']
    vi_rows='\n'.join(f'| {k} | {number(v)} |' for k,v in vi['accuracy_percent'].items())
    text='''# Kết quả tích hợp review người — 17/11/2026

## Phạm vi và phương pháp

R1 đã hoàn thiện phiếu 100 nguồn ảnh và xác nhận nghĩa của 100 bản dịch CIFAR-100. Trợ lý chuẩn hóa định dạng, điền trạng thái dịch theo xác nhận rõ ràng của người dùng và tính lại từ scores có sẵn; **không inference mới**. Phiếu gốc, các bản nháp, phiếu chuẩn và checksum nằm ở [review đã đóng băng](../../reviews/human_review_final_v1/). Giữ nhãn `tv` và `dining table` theo annotation dataset.

Phiếu ảnh nhận **98**, loại **2**: `CONFIRMATION_100_006` (không xác nhận chắc donut, AMBIGUOUS_LABEL), `CONFIRMATION_100_092` (người bị che khuất, OCCLUDED_OR_TINY). Dòng 092 cuối cùng đã ghi both_objects_visible=no; dòng 040 được R1 giữ nhận dù đã trao đổi lo ngại nĩa bị crop. Không thay quyết định của người bằng đề xuất agent. Tên lớp VI không đổi, 100 dòng accepted/R1.

Agent trước nhận cả 100 ảnh đã chọn; nay có 98 cùng nhận và 2 bị người loại. Đây là so sánh quyết định trên bộ đã lọc, không phải accuracy gán nhãn hay inter-annotator agreement giữa hai người. Có một reviewer người và bước thảo luận với agent.

Mỗi nguồn giữ cả ảnh gốc và lật ngang: trước 200 ảnh, sau 196 ảnh. Không lọc dựa trên dự đoán model. Primary dùng hai lựa chọn trái/phải, holistic T2/o0 và native_half trên cùng ảnh; padding là secondary đã định trước. Cả hai ảnh đúng và ghép cặp 2×2 là hai metric khác nhau: metric thứ nhất yêu cầu từng dự đoán đúng; metric thứ hai chọn phép ghép ảnh–caption có tổng điểm lớn hơn.

CI 95% bootstrap 2.000 lượt theo source. Bảng đối chiếu dùng seed 1402; sáu điều kiện holistic lịch sử giữ seed 14031 và có thêm Wilson/binomial như run gốc. CI là riêng từng phép đo, không hiệu chỉnh nhiều kiểm định. Kết quả trước lọc đã tái lập metrics gốc.

## Trước và sau review

Đơn vị accuracy và CI là %. Top-1 có mẫu số ảnh; hai ảnh đúng và 2×2 có mẫu số nguồn. CI đi sau 2×2 là bootstrap theo nguồn.

| Phương pháp | Tập | Đúng / ảnh = Top-1 | CI 95% Top-1 | Cả hai ảnh đúng | 2×2 và CI 95% |
|---|---|---:|---|---:|---|
'''+ '\n'.join(rows)+'''

Chênh lệch Top-1 so với holistic được tính ghép cặp trên cùng source, đơn vị điểm phần trăm:

| Tập | Phương pháp | Δ so holistic | CI 95% Δ |
|---|---|---:|---|
'''+ '\n'.join(deltas)+f'''

Sau lọc, holistic 2×2 đúng **{h['correct']}/{h['n_sources']} = {number(h['accuracy_percent'])}%**; Wilson CI 95% **{ci(h['wilson_ci95_percent'])}**, kiểm định nhị thức một phía so 0,5 có p={h['binomial_p_greater_0_5']:.4f}. Chưa đủ bằng chứng vượt 50%; không suy ra tương đương ngẫu nhiên. Bootstrap legacy seed 14031: {ci(h['bootstrap_ci95_percent'])}.

Native Top-1 thay đổi +0,87 điểm %, holistic +0,03 điểm %; ưu thế native so holistic vẫn hiện diện sau loại hai nguồn. Đây là mô tả độ nhạy trên cùng bộ dữ liệu, không phải bằng chứng tập mới độc lập hay kiểm định hiệu ứng của việc review. Các CI/dự đoán/confusion/ties đầy đủ có trong [metrics](../../results/human_review_followup_v1/metrics.json); [bảng CSV](comparison.csv) và [mask 100 source](../../results/human_review_followup_v1/source_membership.csv) cho phép truy ngược từng mẫu.

## Gắn xác nhận bản dịch

R1 xác nhận 100 bản dịch đúng nghĩa; trợ lý điền R1/accepted theo chỉ dẫn. Đã kiểm 100 tên EN/VI khớp từ điển trong run cũ. Không sửa labels, prompts, scores hoặc metrics; không chạy lại inference. [Liên kết checksum](translation_review.json) nối phiếu review với từ điển và metrics đã có.

| Điều kiện đã chạy | Top-1 (%) giữ nguyên |
|---|---:|
'''+vi_rows+'''

Xác nhận nghĩa bởi một reviewer không loại bỏ mọi mơ hồ thuật ngữ, không chứng minh cơ chế BPE gây suy giảm hiệu năng, và không tương đương hội đồng chuyên gia ngôn ngữ.

## Giới hạn, tái lập và bàn giao

- Đây là review có thảo luận với agent, sensitivity hậu nghiệm trên COCO đã mở; không gọi là một vòng chấm độc lập hoàn toàn hoặc holdout mới.
- R1 giữ 040; công khai bất đồng về mức nhìn thấy thay vì tự loại thêm hoặc thay đổi theo scores.
- Chia vùng dùng thêm lượt encode và quy tắc hình học; không chứng minh encoder tự học quan hệ tốt hơn.
- Giữ nguyên run nguồn và kết quả gốc 100 nguồn; PDF/slides lịch sử không sửa theo phạm vi người dùng giao. Các PDF cũ ghi pending phản ánh trạng thái khi tạo; trạng thái hiện hành là trang này.
- Chạy `python -m vlm_probe.human_review_followup audit` để kiểm checksum, tái tính metric và đối chiếu từng prediction/pair. Khi tạo lại ở clone sạch, giữ thư mục review cố định; dùng `run` chỉ khi chưa có output, không gọi lại export phiếu trắng lên phiếu đã điền.

Nguồn: [giao thức](../../docs/human_review_followup_v1.md), [đính chính](../../docs/human_review_followup_v1_amendment.md), [manifest tính toán](../../results/human_review_followup_v1/manifest.json). Không có kết quả model mới ở mốc này.
'''
    (REPORT/'README.md').write_text(text,encoding='utf-8')
    sources=[OUTPUT/'manifest.json',OUTPUT/'metrics.json',OUTPUT/'source_membership.csv',REVIEW/'freeze.json',ROOT/'vlm_probe/human_review_followup_report.py']
    write_json(REPORT/'sources.json',{'input_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in sources},
                                  'derived_sha256':{p.name:sha256(p) for p in REPORT.iterdir() if p.is_file() and p.name!='sources.json'}})


if __name__=='__main__':main()
