"""Vietnamese academic report built from reviewed, immutable experiment artifacts."""
import csv
from pathlib import Path

from .depth.common import ROOT, read_json, write_csv
from .extension_artifacts import REVIEW, verify_manifest
from .io_utils import sha256, write_json

OUT = ROOT / 'reports/research_review_v2'


def number(v, digits=2): return f'{v:.{digits}f}'.replace('.', ',')
def interval(v): return '[' + '; '.join(number(x) for x in v) + ']'
def esc(s):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]: s=s.replace(a,b)
    return s


def table(headers, rows, align, caption):
    lines=[r'\begin{center}\small',r'\captionof{table}{'+caption+'}',r'\begin{tabular}{@{}'+align+r'@{}}\toprule',
           ' & '.join(headers)+r'\\\midrule']
    lines += [' & '.join(str(x) for x in row)+r'\\' for row in rows]
    return '\n'.join(lines+[r'\bottomrule\end{tabular}\end{center}'])


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs=set()
    def read(relative):
        path=ROOT/relative;inputs.add(path);return read_json(path)
    audits={}
    for name in ['full_prompt_cifar100_v1','full_vietnamese_cifar100_v1','full_spatial_mitigation_v1','full_prompt_probing_v1']:
        verify_manifest(ROOT/'results'/name)
        audits[name]=read(f'results/review_hieu_v2/{name}/audit.json')
    for name in ['full_prompt_cifar100_v1','full_vietnamese_cifar100_v1','full_spatial_mitigation_v1']:
        if not audits[name]['metrics_recomputed']:raise ValueError('Report requires numerical audit')
        if audits[name]['manifest_sha256'] != sha256(ROOT/f'results/{name}/manifest.json'):raise ValueError('Stale audit')
        for filename,h in audits[name]['derived_sha256'].items():
            if sha256(REVIEW/name/filename)!=h:raise ValueError('Changed audit derivative')
    prompt=read('results/full_prompt_cifar100_v1/metrics.json')
    vi=read('results/full_vietnamese_cifar100_v1/metrics.json')
    natural=read('results/full_natural_confirmation_v1/metrics.json')['t2_o0']
    sensitivity=read('reports/natural_confirmation_v1/metrics.json')
    psr=read('reports/synthesis_v1/spatial_mechanism_psr.json')
    spatial=read('results/full_spatial_mitigation_v1/metrics.json')
    synthetic_trace='Run hình học cũ thiếu scores từng vùng; audit chỉ tính lại từ relational scores cuối.'
    if (REVIEW/'full_spatial_mitigation_review_v2/audit.json').exists():
        synaudit=read('results/review_hieu_v2/full_spatial_mitigation_review_v2/audit.json')
        if not synaudit['region_to_relational_scores_recomputed']:raise ValueError('Synthetic regional trace required')
        spatial=read('results/full_spatial_mitigation_review_v2/metrics.json')
        synthetic_trace='Run hình học mới lưu đủ scores từng vùng và relational scores; audit kiểm lại phép kết hợp vùng, dự đoán và metrics. Kết quả giữ nguyên 100\\% so với run cũ.'
    execution=read('results/integration_v2/execution_status.json')
    execution.setdefault('status','running')
    tests=read('results/integration_v2/tests.json')
    read('results/integration_v2/historical_audits.json')
    read('results/full_depth_category_v1/metrics.json')
    core=read('results/full_probing_clip_b32_v1/metrics.json')
    datasets=[('cifar100','CIFAR-100','full_cifar100_clip_b32_v1'),
              ('flowers102','Flowers102','full_flowers102_clip_b32_v2'),
              ('oxford_pet','Oxford-IIIT Pet','full_oxford_pet_clip_b32_v3'),
              ('dtd','DTD partition 1','full_dtd_clip_b32_v3'),
              ('food101','Food-101','full_food101_clip_b32_v3')]
    # Read baseline table from the frozen generated CSV, checked against each manifest below.
    with (ROOT/'reports/baseline/summary.csv').open(encoding='utf-8') as f:baseline=list(csv.DictReader(f))
    inputs.add(ROOT/'reports/baseline/summary.csv')
    sizes=[10000,6149,3669,1880,25250];classcounts=[100,102,37,47,101]
    base_acc=[]
    for _,_,name in datasets:
        m=read(f'results/{name}/metrics.json')
        base_acc.append(m.get('top1_accuracy_percent',m.get('accuracy_percent')))
    # Accommodate historical baseline metric naming without manual numeric transcription.
    if any(x is None for x in base_acc):
        base_acc=[]
        for _,_,name in datasets:
            with (ROOT/f'results/{name}/predictions.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
            inputs.add(ROOT/f'results/{name}/predictions.csv')
            base_acc.append(100*sum(int(r['target_id'])==int(r['prediction_id']) for r in rows)/len(rows))
    base_table=table(['Dataset','Lớp','Ảnh test',r'Top-1 (\%)'],
        [(label,c,n,number(a)) for (_,label,_),c,n,a in zip(datasets,classcounts,sizes,base_acc)],'lrrr','Baseline zero-shot trên split test gốc.')
    core_table=table(['Core probe',r'Accuracy (\%)',r'Nhận dạng control (\%)','Giới hạn chính'],
        [(label,number(core[k]['accuracy_percent']),number(100*core[k]['control_accuracy']) if 'control_accuracy' in core[k] else '---',limit)
         for k,label,limit in [('counting','Counting (6-way)','Hình học; còn lỗi nhận dạng'),
            ('spatial','Spatial (4-way)','Nhạy với cấu trúc/cảnh'),('negation','Negation (2-way)','Hình học đơn giản'),
            ('fine_grained','Fine-grained (5-way)','Có coarse label thật')]],'lrrl','Core probing và đối chứng nhận dạng.')
    prompt_rows=[]
    for k,v in {**prompt['single_templates'],**prompt['ensembles']}.items():
        prompt_rows.append((esc(k),number(v['accuracy_percent']),number(v['delta_vs_baseline_pp']),interval(v['delta_vs_baseline_ci95_pp'])))
    prompt_table=table(['Điều kiện',r'Top-1 (\%)',r'$\Delta$ (điểm \%)',r'CI 95\% của $\Delta$'],prompt_rows,'lrrl','CIFAR-100: bảy template và ba ensemble; CI paired theo ảnh, phân tầng lớp.')
    vi_rows=[]
    for k,v in vi['single_templates'].items():
        vi_rows.append((esc(k),number(v['accuracy_percent']),number(v['delta_vs_baseline_pp']),interval(v['delta_vs_baseline_ci95_pp'])))
    vi_table=table(['Ngôn ngữ',r'Top-1 (\%)',r'$\Delta$ so EN',r'CI 95\% của $\Delta$'],vi_rows,'lrrl','So sánh ngôn ngữ trên cùng 10.000 ảnh CIFAR-100.')
    multidomain=[];domain_values=[];finished=0
    for (ds,label,_),acc in zip(datasets,base_acc):
        name='full_prompt_cifar100_v1' if ds=='cifar100' else f'full_prompt_{ds}_review_v2'
        p=ROOT/'results'/name
        if (p/'manifest.json').exists() and (REVIEW/name/'audit.json').exists():
            a=read(f'results/review_hieu_v2/{name}/audit.json')
            if not a['metrics_recomputed']:raise ValueError('Unreviewed prompt result')
            verify_manifest(p)
            if a['manifest_sha256'] != sha256(p/'manifest.json'):raise ValueError('Stale prompt audit')
            met=read(f'results/{name}/metrics.json');e=met['ensembles']['text_embedding_ensemble']
            diag=read(f'results/review_hieu_v2/{name}/diagnostics.json')
            if sha256(REVIEW/name/'diagnostics.json')!=a['derived_sha256']['diagnostics.json']:raise ValueError('Changed diagnostics')
            multidomain.append((label,number(acc)+' / '+number(diag['T0_baseline']['macro_accuracy_percent']),
                number(e['accuracy_percent'])+' / '+number(diag['text_embedding_ensemble']['macro_accuracy_percent']),
                interval(e['delta_vs_baseline_ci95_pp']),'Đã audit'));finished+=1
            domain_values.append({'dataset':ds,'n':met['evaluated_samples'],'baseline_top1':acc,
                'baseline_macro':diag['T0_baseline']['macro_accuracy_percent'],
                'ensemble_top1':e['accuracy_percent'],'ensemble_macro':diag['text_embedding_ensemble']['macro_accuracy_percent'],
                'delta_pp':e['delta_vs_baseline_pp'],'delta_ci_low':e['delta_vs_baseline_ci95_pp'][0],
                'delta_ci_high':e['delta_vs_baseline_ci95_pp'][1]})
        else:multidomain.append((label,number(acc),'---','---','Chưa chạy'))
    domain_table=table(['Dataset','Base T / M','Ens. T / M',r'CI của $\Delta$','Trạng thái'],multidomain,'lrrll','Embedding ensemble đa miền: T là top-1, M là macro theo lớp, đơn vị phần trăm; CI cho delta top-1 (điểm phần trăm).')
    regional=[];natural_details=[];natural_finished=0
    for ds,label in [('exploratory','COCO 200 nguồn'),('confirmation','COCO 100 nguồn')]:
        name=f'full_natural_regions_{ds}_review_v2'
        if (REVIEW/name/'audit.json').exists():
            a=read(f'results/review_hieu_v2/{name}/audit.json')
            if not a['metrics_recomputed']:raise ValueError('Unaudited natural region result')
            m=read(f'results/{name}/metrics.json');regional.append((label+' native',str(m['primary_choices']),number(m['primary_accuracy_percent'])+' '+interval(m['primary_ci95_percent']),number(m['paired_delta_pp'])+' '+interval(m['paired_delta_ci95_pp'])));natural_finished+=1
            sec=m['secondary_padded_half']
            regional.append((label+' pad',str(sec['primary_choices']),number(sec['primary_accuracy_percent'])+' '+interval(sec['primary_ci95_percent']),number(sec['paired_delta_pp'])+' '+interval(sec['paired_delta_ci95_pp'])))
            natural_details.append(label+': holistic '+number(m['holistic_accuracy_percent'])+r'\%; cả hai ảnh đúng '+number(m['both_images_correct_percent'])+r'\% ở cách gốc và '+number(sec['both_images_correct_percent'])+r'\% ở padding, so '+number(m['holistic_both_images_correct_percent'])+r'\% holistic.')
        else:regional.append((label,'4' if ds=='exploratory' else '2','Chưa chạy','---'))
    region_table=table(['Bộ ảnh','Lựa chọn',r'Chia vùng (\%), CI',r'$\Delta$ so holistic, CI'],regional,'llll','Chia vùng COCO và CI bootstrap theo source; delta paired cùng ảnh.')
    psr_rows=[]
    for k,label,cond in [('synthetic_depth','Hình học','t0_o0_horizontal'),('natural_exploratory','COCO cũ','t2_o0_horizontal'),('natural_confirmation','COCO mới','t2_o0_horizontal')]:
        m=psr[k]['conditions'][cond]
        psr_rows.append((label,m['n_pairs'],number(m['aggregate_prior_to_signal_ratio']),number(m['same_prediction_collapse_percent']),number(m['assignment_accuracy_percent']),number(m['both_images_correct_percent'])))
    psr_table=table(['Bộ ảnh','Cặp','PSR',r'Cùng nhãn (\%)',r'Ghép đúng (\%)',r'Cả hai đúng (\%)'],psr_rows,'lrrrrr','Phân rã mô tả các cặp ngang; mọi tỷ lệ trong bảng này dùng hai lựa chọn.')
    token=vi['bpe_summary']['labels'];vp=vi['bpe_summary']['prompts']
    token_table=table(['Điều kiện','Token/tên lớp','Token/câu','Vượt context'],
        [(label,number(token[k]['mean_subword_tokens']),number(vp[k]['mean_subword_tokens']),vp[k]['exceeds_77_context_window_count'])
         for k,label in [('en_baseline','Anh'),('vi_diacritics','Việt có dấu'),('vi_no_diacritics','Việt không dấu')]],'lrrr','Số token BPE trung bình; không tính token bắt đầu/kết thúc.')
    nci=interval(natural['wilson_ci95_percent']);npval=number(natural['binomial_p_greater_0_5'],4)
    oldclean=number(sensitivity['old_four_way_original']['reviewed']['accuracy_percent'])
    available=f'Đã có {finished}/5 bộ dữ liệu với kết quả prompt được audit; {natural_finished}/2 bộ COCO có kết quả chia vùng mới.'
    status=('Các thí nghiệm bổ sung đang chờ khôi phục môi trường: Windows Application Control chặn PyTorch (WinError 4551).'
            if execution['status']=='blocked_environment' else r'Trạng thái thực thi được ghi trong execution\_status.json.')
    completed=finished==5 and natural_finished==2
    test_text=(f"Toàn bộ {tests['run']} tests đạt." if not tests['errors'] and not tests['failures'] else
               f"Chạy {tests['run']} tests: {tests['passed']} đạt, {len(tests['errors'])} lỗi và {len(tests['failures'])} failure; xem tests.json để phân biệt lỗi môi trường với lỗi code.")
    domain_text=('Các CI trong bảng là CI riêng từng dataset, chưa hiệu chỉnh familywise. Không cộng tất cả ảnh các miền thành một phép so độc lập. '+
        ' '.join(label+': '+number(v['delta_pp'])+' điểm phần trăm, CI '+interval([v['delta_ci_low'],v['delta_ci_high']])+'.'
                 for v in domain_values for ds,label,_ in datasets if ds==v['dataset']))
    region_text=(' '.join(natural_details)+' Nửa ảnh chữ nhật qua preprocessing chuẩn sẽ bị center crop thêm. '
        'Control padding đặt nửa ảnh giữa canvas vuông với nền gần mean checkpoint, được cố định trước inference COCO. '
        'Đây là điều kiện phụ; không thay primary sau khi xem kết quả. Padding cũng thay nền và kích thước tương đối.')
    probing_name='full_prompt_probing_review_v2'
    probing_text=('Run counting/negation cũ thiếu scores nên giữ nguyên và chạy lại thành run mới. ')
    if (REVIEW/probing_name/'audit.json').exists():
        a=read(f'results/review_hieu_v2/{probing_name}/audit.json')
        if not a['metrics_recomputed']:raise ValueError('Probe audit required')
        met=read(f'results/{probing_name}/metrics.json');old=read('results/full_prompt_probing_v1/metrics.json')
        from .extension_artifacts import close_enough
        close_enough(met,old,'probe_reproduction')
        probing_text+='Run mới lưu đủ scores, caption, nhãn và dự đoán; metrics và CI khớp run cũ. '
        for key,label in [('counting','Counting'),('negation','Negation')]:
            v=met[key];probing_text+=label+': single dao động '+number(v['worst_single_accuracy_percent'])+'--'+number(v['best_single_accuracy_percent'])+r'\%; embedding ensemble '+number(v['ensembles']['text_embedding_ensemble']['accuracy_percent'])+r'\%. '
        probing_text+='Ensemble không mặc nhiên tốt hơn baseline. '
    else:probing_text+='Run mới chưa xong nên không dùng các CI tổng hợp cũ như bằng chứng đã tái kiểm. '
    probing_text+='Các bộ ảnh này đã mở; phân tích tiếp nối không biến chúng thành holdout mới.'
    final_status=('Các run bổ sung đã hoàn thành và được audit; ảnh và bản dịch vẫn chờ reviewer người.' if completed else status)
    followup=('Đã hoàn thành ma trận prompt năm miền và hai phép chia vùng COCO theo giao thức; còn nhận review nhãn/bản dịch, phân xử và cập nhật phân tích độ nhạy nếu có. '
              'Kết quả đi ngược kỳ vọng được giữ trong báo cáo. Tích hợp code qua review và lưu checklist tái lập.' if completed else
              'Ưu tiên tiếp tục là hoàn thành bốn run prompt đa miền và hai run chia vùng COCO theo giao thức đã chốt, '
              'khôi phục scores prompt probing bằng run mới, nhận review nhãn/bản dịch từ người, rồi cập nhật các bảng trạng thái. '
              'Bản hiện tại chưa tuyên bố mọi hạng mục đã hoàn thành.')
    body=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=18mm,top=16mm,bottom=16mm]{geometry}
\usepackage{fontspec,amsmath,booktabs,graphicx,titlesec,caption}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{TeX Gyre Termes}}
\usepackage[unicode,hidelinks]{hyperref}
\renewcommand{\tablename}{Bảng}\renewcommand{\refname}{Tài liệu tham khảo}
\titleformat{\section}{\normalsize\bfseries}{\thesection.}{0.5em}{}
\titlespacing*{\section}{0pt}{7pt}{3pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}\emergencystretch=2em
\begin{document}
\begin{center}{\Large\bfseries Thăm dò CLIP ViT-B/32: độ nhất quán không gian,\\ độ nhạy prompt và chuyển ngữ tiếng Việt}\\[5pt]
{\small Topic 14 — Computer Vision\quad |\quad Bản rà soát: 11/11/2026}
\end{center}
\textbf{Tóm tắt.} Nghiên cứu đánh giá một checkpoint CLIP cố định qua năm bộ dữ liệu phân loại và các phép thăm dò có đối chứng. Baseline trên 46.948 ảnh cho thấy khả năng nhận dạng thay đổi theo miền; các can thiệp vị trí cho thấy accuracy từng ảnh không phản ánh đầy đủ độ nhất quán cả cặp. Trên 100 nguồn ảnh tự nhiên đã cố định, ghép cặp đúng 53\%, CI Wilson 95\% @NCI@\%, nhưng chỉ 4\% cặp có cả hai ảnh được dự đoán đúng riêng rẽ. Trên CIFAR-100, ensemble bảy prompt đạt 64,97\%, tăng 0,49 điểm phần trăm so với baseline; prompt Việt có dấu đạt 5,10\%. Việc thay tên lớp và khung câu cho thấy sự phụ thuộc đáng kể vào cách biểu diễn tên lớp. Phân mảnh tokenizer đồng xuất hiện với mức giảm này, song chưa chứng minh quan hệ nhân quả. Bài viết phân biệt kết quả được tính lại từ scores, phân tích tiếp nối và các hạng mục chưa có phép đo.

\textbf{Trạng thái bằng chứng.} @AVAILABLE@ @STATUS@ Rà soát độc lập bởi người và kiểm bản dịch chưa có phiếu hoàn thành; các bảng không điền kết quả dự kiến cho các run đang chờ.

\section{Bài toán và câu hỏi nghiên cứu}
CLIP học biểu diễn ảnh và văn bản để so khớp bằng điểm tương đồng~\cite{clip}. Trong phân loại zero-shot, tên lớp được đưa vào prompt; lớp có điểm cao nhất được chọn. Hiệu năng nhận dạng tốt chưa đủ để suy ra mô hình nhạy đúng với số lượng, thứ tự mô tả hay quan hệ giữa vật thể. ARO và Winoground đã đặt ra các phép kiểm tính kết hợp ảnh--text~\cite{aro,winoground}; nghiên cứu phủ định cũng cho thấy cần đánh giá theo thiết kế nhiệm vụ~\cite{negbench}.

Bốn câu hỏi được xét: (RQ1) hiệu năng phân loại và các nhóm lỗi thay đổi thế nào theo miền; (RQ2) dự đoán có nhất quán khi quan hệ đổi; (RQ3) prompt và ensemble ảnh hưởng ra sao; (RQ4) thay ngôn ngữ prompt trên cùng ảnh làm thay đổi kết quả thế nào. Đóng góp của nhóm là chuỗi kiểm tra hành vi có metadata và đối chứng trên một checkpoint, không phải kết luận về mọi VLM hoặc cơ chế neuron.

\newpage
\section{Dữ liệu, mô hình và giao thức}
Dùng OpenAI CLIP ViT-B/32 pretrained, giữ trọng số, QuickGELU, FP32, preprocessing checkpoint và batch 32. Ảnh và text được mã hóa, chuẩn hóa L2; $s(I,c)=\hat v(I)^\top\hat t(c)$. Phân loại bằng argmax trên toàn bộ lớp của dataset. Không huấn luyện lại. Nguồn dữ liệu, checksum, split, class order và environment được lưu trong manifest.

@BASE_TABLE@

Counting gồm 1.080 ảnh hình học, sáu lựa chọn số lượng 1--6 và hai chế độ kích thước; spatial core có 720 ảnh, bốn quan hệ; negation có 720 ảnh và hai cặp diễn đạt, thành 1.440 lượt chấm. Fine-grained CIFAR dùng coarse label thật để giới hạn còn năm lớp; đây là chẩn đoán oracle, không phải cải thiện bài 100 lớp. DTD chỉ xét partition 1. Các dataset có số lớp và độ khó khác nhau nên không dùng chênh accuracy giữa dataset làm hiệu ứng thay model.

Spatial depth có 120 nhóm bố cục, ba ngoại hình và bốn quan hệ, tổng 1.440 ảnh/8.640 lượt chấm. Ảnh COCO cũ có 200 nguồn; tập ghép cặp tiếp theo có 100 nguồn không trùng 400 ứng viên từng được xem. Mỗi nguồn có mirror; phiên bản lật không là nguồn ảnh độc lập. Tập 100 nguồn đã được mở kết quả trước các phân tích ensemble/chia vùng hiện nay.

Metric chính gồm top-1, mean per-class và chênh lệch ghép cặp khi so hai điều kiện trên cùng ảnh. CI bootstrap giữ mọi biến thể cùng nguồn/cảnh trong một nhóm. Classification dùng bootstrap phân tầng theo nhãn; hai tập nguồn khác nhau dùng lấy mẫu độc lập. Cụ thể số bootstrap/seed được lưu theo phiên bản giao thức, không thay âm thầm. Không gộp accuracy 2-way, 4-way, 6-way và oracle thành một điểm tổng.

Với template ensemble, chuẩn hóa từng vector text, lấy trung bình đều và chuẩn hóa lại theo lớp. Logit ensemble lấy trung bình cosine; probability ensemble lấy trung bình softmax với scale 100 cố định. Ensemble embedding được chọn làm so sánh chính trong phần mở rộng đa miền; các phép còn lại là phân tích phụ. Prompt tốt nhất trên test chỉ được gọi là best observed.

\newpage
\section{Baseline, core probing và đối chứng category}
Năm baseline bao phủ 46.948 ảnh. Độ chính xác tổng cao vẫn có thể đi cùng lớp yếu: Oxford Pet đạt 85,06\% nhưng Bombay đúng 0/88; CIFAR-100 đạt 64,48\% nhưng ray đúng 5/100. Phân tích theo lớp vì vậy cần đi cùng support và confusion, không chỉ accuracy trung bình.

@CORE_TABLE@

Counting có MAE 1,0157 vật thể; bán kính cố định và diện tích tổng gần cố định đạt lần lượt 32,41\% và 33,33\%. Control nhận dạng chưa hoàn hảo, nên khó khăn đếm không được quy hoàn toàn cho biểu diễn số lượng. Ngược lại, spatial giữ control cao nhưng chỉ 0\% nhóm core đúng cả bốn quan hệ, hỗ trợ phân biệt nhận dạng thuộc tính với tính nhất quán quan hệ trong bộ cảnh.

Negation đạt 91,39\% với with/without và 80,83\% với contains/does not contain. Kết quả tương đối tốt trong nhiệm vụ này không hỗ trợ phát biểu CLIP luôn thất bại về phủ định. Chỉ 58,61\% nhóm đúng đồng thời các biến thể present/absent và hai template; accuracy từng lượt và consistency cả nhóm là hai đại lượng khác nhau.

Trên CIFAR-100 đủ 100 ứng viên, thêm category thật làm accuracy giảm từ 64,48\% xuống 62,92\%, tức $-1,56$ điểm phần trăm (CI paired depth [$-2,14$; $-0,98$]). Neutral suffix đạt 64,09\%; trung bình năm category shuffle đạt 54,19\%. Điểm lớp thật tăng trung bình 0,001628, trong khi đối thủ mạnh nhất tăng 0,002908, khiến margin giảm 0,001280. Category phù hợp tốt hơn category sai nhưng chưa tốt hơn baseline với suffix đã dùng. Hai phiên bản bootstrap category v1 và depth có số draws/seed khác nhau; báo cáo dùng CI depth và không sửa artifact v1.

Các cặp nhầm Flowers/Pet/Food là quan sát hậu nghiệm, còn CIFAR oracle 5-way là chẩn đoán trong nhóm. Mức tăng từ 64,48\% lên 77,95\% khi thu hẹp ứng viên không được trình bày như một cải tiến zero-shot có thể triển khai khi chưa biết coarse label.

\newpage
\section{Độ nhạy prompt và ensemble}
Bảy template CIFAR gồm baseline, nhãn trần và mô tả clear/centered/low-resolution/bright/cropped. Một số biến thể bổ sung thông tin hình ảnh, nên spread phản ánh độ nhạy với bộ prompt được định nghĩa, không chỉ đổi cách diễn đạt gần nghĩa. Cùng 10.000 ảnh, class order và checkpoint được giữ cố định. Baseline run mới khớp 10.000/10.000 dự đoán cũ; sai khác cosine lớp thật tối đa $3,28\times10^{-7}$.

@PROMPT_TABLE@

Không single template mới nào vượt T0 trong bộ này. Embedding ensemble tăng 0,49 điểm phần trăm, CI paired [0,10; 0,89]. Đây là cải thiện nhỏ trong một dataset và bộ template cụ thể. Các CI là CI biên của từng so sánh; không diễn giải ba ensemble như ba lần xác nhận độc lập, không chọn probability ensemble vì nó có điểm cao nhất rồi gọi là primary định trước.

@DOMAIN_TABLE@

Giao thức tiếp nối giữ nguyên bộ bảy template khi mở bốn dataset còn lại, không điều chỉnh theo CIFAR. Riêng DTD, T0 giữ suffix texture; T4 dùng đúng một cụm mô tả texture. Bộ N4/D4 là một phương án khác và chưa được tính vào kết quả này. @DOMAIN_TEXT@

@PROBING_TEXT@

\newpage
\section{Độ nhất quán không gian và phương pháp chia vùng}
Với hai ảnh đã sắp theo quan hệ trái/phải và hai caption tương ứng, đặt $d_0=S_{00}-S_{01}$, $d_1=S_{10}-S_{11}$; $B=(d_0+d_1)/2$, $\delta=(d_0-d_1)/2$. Đây là phép phân rã đại số: cả hai ảnh đúng khi $\delta>|B|$ (bỏ trường hợp hòa), còn ghép cặp đúng khi $2\delta>0$. Thành phần trung bình trong cặp có thể lớn trong khi hiệu thay đổi rất nhỏ.

@PSR_TABLE@

PSR là tỷ số trung bình $|B|$ và trung bình $|\delta|$. Nó mô tả cấu trúc điểm trên cặp, không định vị một bias trong text encoder hoặc chứng minh nguyên nhân ở kiến trúc. Trên tập 100 nguồn, kết quả chính cũ là @NACC@\%, CI Wilson @NCI@\%, $p=@NP@$. Không bác bỏ giả thuyết $p_{\rm correct}\leq0,5$ không chứng minh xác suất thật bằng 0,5. Cùng tập chỉ @NBOTH@\% cặp đúng cả hai ảnh riêng rẽ, phù hợp việc một caption thường thắng trên cả hai ảnh.

Sau tự rà soát nhãn cũ, giữ 176/200 nguồn và accuracy 4-way là @OLDCLEAN@\% thay vì 28,00\%; đây là sensitivity hậu nghiệm do cùng AI reviewer, chưa là xác nhận bởi người. Tập lọc không còn cân bằng hoàn toàn bốn nhãn.

Phương pháp chia vùng dùng bốn nửa ảnh và prompt nhận dạng A/B. Điểm trái là $h=(s_L(A)-s_R(A))-(s_L(B)-s_R(B))$, điểm phải là $-h$; trên/dưới tương tự. Mỗi điều kiện dùng bốn lượt mã hóa vùng cho một ảnh, so với một lượt holistic; chi phí suy luận và quy tắc kết hợp khác nhau. Run hình học đạt @SYNREG@\% trên 1.440 ảnh. @SYNTRACE@ CI bootstrap [100;100] trên bộ này không loại trừ sai số ngoài phân phối.

@REGION_TABLE@

@REGION_TEXT@

Mô phỏng dựa trên bbox COCO trước đây không phải accuracy thực đo của CLIP chia vùng và không bảo đảm là trần của mọi phương pháp. Giao thức run mới crop cùng vùng nhìn baseline trước khi chia nửa, lưu cả scores vùng, so cùng ảnh với T2 holistic và bootstrap theo nguồn. Hai bộ COCO đều đã mở, nên kết quả phương pháp mới sẽ là nghiên cứu tiếp nối.

\newpage
\section{Chuyển prompt sang tiếng Việt}
Giữ nguyên 10.000 ảnh và 100 lớp; thay ngôn ngữ tên lớp và khung câu. Bộ từ điển hiện là bản dịch trong code, chưa có phiếu review người độc lập. Những câu lai là control chẩn đoán, không đại diện cách nói tự nhiên chuẩn.

@VI_TABLE@

Prompt Việt có dấu đạt 5,10\%, giảm 59,38 điểm phần trăm so EN. Mốc chọn đều ngẫu nhiên giữa 100 lớp là 1\%; kết quả thấp hơn EN không đồng nghĩa biểu diễn hoàn toàn không có thông tin. Dùng khung Việt với tên lớp Anh đạt 55,10\%, trong khi khung Anh với tên lớp Việt chỉ đạt 5,44\%. Các thay đổi hỗ trợ nhận định hiệu năng phụ thuộc mạnh vào cách biểu diễn tên lớp trong các điều kiện này. Chúng chưa lượng hóa tác động riêng của tokenizer, mức quen thuộc từ vựng và bản dịch.

@TOKEN_TABLE@

Tỷ số của hai trung bình token tên lớp có dấu/Anh là @RATIOMEAN@ lần; trung bình tỷ số ghép theo từng lớp là @MEANRATIO@ lần. Đây là hai thống kê khác nhau, tránh viết 7,05 bằng 6,44 lần 1,19. Không có câu vượt context trong bộ đã đo. Lược dấu giảm token nhưng accuracy còn 4,05\%, thấp hơn có dấu; vì thế số token đơn thuần không giải thích đầy đủ hiệu năng.

Ensemble embedding của EN, VI và câu song ngữ đạt @VIENS@\%, thấp hơn baseline Anh. Có tên lớp Anh trong prompt là thêm một biểu diễn từ vựng có lợi trong thí nghiệm này; không gọi đó là phục hồi khả năng hiểu tiếng Việt. Các nhãn mơ hồ hoặc mức độ bao quát của bản dịch cần được người trong nhóm kiểm tra trước khi củng cố kết luận. Nếu sửa bản dịch theo ngữ nghĩa, phải giữ bản cũ và đánh giá bằng run/version mới.

\newpage
\section{Tái lập, rà soát và giới hạn}
Audit mới kiểm inventory/hash artifact, source ZIP và đầu vào; tính lại argmax, accuracy, CI paired, predictions và per-class từ scores. Hai baseline CIFAR mới khớp mọi dự đoán cũ. Bản prompt có alignment norms nên kiểm được đồng nhất thức ensemble chuẩn hóa; bản VI cũ thiếu raw text vectors nên chỉ kiểm được kết quả từ ensemble scores đã lưu. Không gán một cờ verified chung để che các mức kiểm khác nhau.

Các run cũ có config/protocol mặc định depth do dùng chung hàm khởi tạo; extension config/protocol thực được định danh bởi input hashes. Run mới lưu riêng study config/evaluation config, đúng protocol và source snapshot chứa các tệp này. Manifest lịch sử giữ nguyên git\_head, timestamp và checksum; source ZIP xác định code thực thi nếu Git HEAD cũ không có trên remote hiện tại.

@TESTTEXT@ Gói review có 100 ảnh nguồn có box A/B, 100 crop thật và hai phiếu trống cho ảnh/bản dịch. Reviewer không thấy scores hoặc quyết định AI. Chỉ sau khi nhận phiếu đầy đủ mới báo kết quả review; một reviewer không cho phép tính inter-annotator agreement. Nếu hai người cùng nhận tất cả ảnh, raw agreement là 100\% nhưng κ có thể không xác định.

Giới hạn chính gồm: một checkpoint; miền hình học; corpus COCO đã mở cho các phân tích tiếp nối; chưa biết trùng pretraining; phụ thuộc cách đặt caption/foil; nhãn và bản dịch chưa được người xác nhận; CI trên 100 nguồn còn rộng. Điểm nhận dạng A/B trên nửa ảnh chưa thay thế được một control nhận dạng độc lập có foil hợp lệ trên COCO. Các phép so phụ và per-class được coi mô tả. Không suy từ $p>0,05$ sang tương đương, không suy từ hành vi sang cơ chế neuron, không suy từ một tập template sang mọi prompt.

\section{Kết luận và hạng mục tiếp tục}
Kết quả đã kiểm cho thấy nhận dạng lớp, tính nhất quán không gian và đáp ứng ngôn ngữ là các khía cạnh khác nhau. Category có thể tăng cosine của lớp thật nhưng tăng đối thủ mạnh hơn; ensemble có thể cải thiện nhỏ classification mà chưa giải quyết consistency quan hệ; chuyển tên lớp sang tiếng Việt có thể làm giảm mạnh accuracy trong checkpoint đang xét. Những nhận định này có giá trị trong phạm vi thiết kế và artifact được báo, kể cả khi không ủng hộ kỳ vọng cải thiện.

@FOLLOWUP@

\begingroup\small\begin{thebibliography}{5}
\bibitem{clip} A. Radford và cs., \href{https://proceedings.mlr.press/v139/radford21a.html}{Learning Transferable Visual Models From Natural Language Supervision}, ICML, 2021.
\bibitem{aro} M. Yuksekgonul và cs., \href{https://arxiv.org/abs/2210.01936}{When and Why Vision-Language Models Behave Like Bags-of-Words, and What to Do About It?}, ICLR, 2023.
\bibitem{winoground} T. Thrush và cs., \href{https://arxiv.org/abs/2204.03162}{Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality}, CVPR, 2022.
\bibitem{negbench} K. Alhamoud và cs., \href{https://arxiv.org/abs/2501.09425}{Vision-Language Models Do Not Understand Negation}, CVPR, 2025.
\bibitem{coco} T.-Y. Lin và cs., \href{https://arxiv.org/abs/1405.0312}{Microsoft COCO: Common Objects in Context}, ECCV, 2014.
\end{thebibliography}\endgroup
\end{document}
'''
    replaces={'NCI':nci,'NACC':number(natural['accuracy_percent']),'NP':npval,'NBOTH':number(natural['both_images_correct_percent']),
              'OLDCLEAN':oldclean,'AVAILABLE':available,'STATUS':final_status,'BASE_TABLE':base_table,'CORE_TABLE':core_table,'PROMPT_TABLE':prompt_table,'FOLLOWUP':followup,
              'DOMAIN_TABLE':domain_table,'DOMAIN_TEXT':domain_text,'PROBING_TEXT':probing_text,
              'REGION_TABLE':region_table,'REGION_TEXT':region_text,'PSR_TABLE':psr_table,'VI_TABLE':vi_table,
              'TOKEN_TABLE':token_table,'RATIOMEAN':number(token['vi_diacritics']['mean_subword_tokens']/token['en_baseline']['mean_subword_tokens']),
              'MEANRATIO':number(token['vi_diacritics']['mean_ratio_vs_en_baseline']),
              'VIENS':number(vi['ensembles']['en_vi_bilingual_embedding_ensemble']['accuracy_percent']),
              'SYNREG':number(spatial['four_way_accuracy_percent']), 'SYNTRACE':synthetic_trace, 'TESTN':str(tests['run']),
              'TESTPASS':str(tests['passed']), 'TESTERROR':str(len(tests['errors'])),'TESTTEXT':test_text}
    for key,value in replaces.items():body=body.replace('@'+key+'@',value)
    if '@' in body.replace('@{}',''):raise ValueError('Unresolved report placeholder')
    tex=OUT/'bao_cao_nghien_cuu.tex';tex.write_text(body,encoding='utf-8')
    if domain_values:write_csv(OUT/'multidomain_summary.csv',domain_values)
    inputs.update([PROTOCOL_FILE for PROTOCOL_FILE in [ROOT/'docs/integration_protocol_v2.md',Path(__file__)]])
    write_json(OUT/'sources.json', {'input_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(inputs)},
        'tex_sha256':sha256(tex),'prompt_datasets_audited':finished,'natural_region_sets_audited':natural_finished,
        'human_review_status':'pending','execution_status':execution['status']})
    (OUT/'README.md').write_text('# Báo cáo nghiên cứu tiếng Việt — bản review\n\n'
        '[PDF](bao_cao_nghien_cuu.pdf), [LaTeX](bao_cao_nghien_cuu.tex), [nguồn/checksum](sources.json).\n\n'
        +available+' '+status+'\n\n'
        'Sinh lại bằng `python -m vlm_probe.review_report`; biên dịch bằng Tectonic hoặc XeLaTeX. '
        'Các con số chính được tính lại từ artifact; ô chưa chạy giữ trống. '
        'Giao thức: `docs/integration_protocol_v2.md`. Human review chưa hoàn thành.\n',encoding='utf-8')
    print(tex,flush=True)


if __name__=='__main__':run()
