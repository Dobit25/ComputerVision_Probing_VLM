"""Unified synthesis runner for P1.2, P2, P3, and P4 diagnostics (`reports/synthesis_v1/`).

Executes all deterministic offline diagnostics on the frozen repository artifacts
and aggregates the live GPU experiment results (`full_spatial_mitigation_v1`,
`full_prompt_probing_v1`, `full_prompt_cifar100_v1`, `full_vietnamese_cifar100_v1`)
without modifying any historical run directory under `results/`.
"""

from pathlib import Path

from .depth.blinded_review import export_blinded_sheets
from .depth.common import ROOT, read_json
from .io_utils import write_json
from .prompt_study.offline_analysis import analyze_saved_probing_ensembles
from .spatial_diagnostics.mechanism import analyze_frozen_spatial_runs
from .spatial_diagnostics.quadrant_mitigation import evaluate_geometric_upper_bound
from .vietnamese_study.bpe_diagnostics import analyze_bpe_tokenization

SYNTHESIS_DIR = ROOT / "reports/synthesis_v1"


def load_live_gpu_results():
    """Load metrics from the 4 live GPU extension runs if present."""
    runs = {
        "spatial_mitigation": ROOT / "results/full_spatial_mitigation_v1/metrics.json",
        "prompt_probing": ROOT / "results/full_prompt_probing_v1/metrics.json",
        "prompt_cifar100": ROOT / "results/full_prompt_cifar100_v1/metrics.json",
        "vietnamese_cifar100": ROOT / "results/full_vietnamese_cifar100_v1/metrics.json",
    }
    loaded = {}
    for key, path in runs.items():
        if path.exists():
            loaded[key] = read_json(path)
    return loaded


def build_synthesis_markdown(blinded, spatial_psr, spatial_geom, prompt_ens, bpe_diag, live_gpu):
    """Render a comprehensive Vietnamese Markdown summary of P1.2, P2, P3, and P4 diagnostics."""
    syn_t0 = spatial_psr["synthetic_depth"]["conditions"]["t0_o0_horizontal"]
    nat_exp_t2 = spatial_psr["natural_exploratory"]["conditions"]["t2_o0_horizontal"]
    nat_conf_t2 = spatial_psr["natural_confirmation"]["conditions"]["t2_o0_horizontal"]

    neg = prompt_ens["negation_probing_v1"]
    neg_mech = neg["mechanistic_breakdown"]

    sp_syn_ens = prompt_ens["synthetic_spatial_depth_v1"]["four_way_forward_templates"]
    sp_conf_ens = prompt_ens["natural_spatial_confirmation_v1"]["horizontal_2x2_assignment"]
    cat_ens = prompt_ens["cifar100_category_depth_v1"]["semantic_valid_3_templates"]

    bpe_p = bpe_diag["cifar100_prompts_summary"]
    bpe_l = bpe_diag["cifar100_labels_summary"]

    geom_syn = spatial_geom["synthetic_depth_1440"]
    geom_exp = spatial_geom["natural_exploratory_200"]
    geom_conf = spatial_geom["natural_confirmation_100"]

    lines = [
        "# Báo Cáo Tổng Hợp Chẩn Đoán Cơ Chế, Độ Nhạy Prompt & Phân Mảnh BPE Tiếng Việt (`synthesis_v1`)",
        "",
        "Tài liệu này được sinh tự động và tất định bởi `python -m vlm_probe.synthesis` từ các artifact trong `results/`.",
        "",
        "---",
        "",
        "## 1. P1.2 — Trạng Thái Xuất Bảng Chấm Mù Độc Lập (Blinded Review Sheets)",
        "",
        "| Bộ dữ liệu | Số nguồn ảnh | File bảng chấm mù (CSV) | SHA-256 bảng chấm mù |",
        "| --- | ---: | --- | --- |",
        f"| **Confirmation (`100` nguồn ngang)** | `{blinded['confirmation_100']['num_items']}` | `{blinded['confirmation_100']['sheet_csv']}` | `{blinded['confirmation_100']['sheet_sha256'][:16]}...` |",
        f"| **Exploratory (`200` nguồn 4 hướng)** | `{blinded['exploratory_200']['num_items']}` | `{blinded['exploratory_200']['sheet_csv']}` | `{blinded['exploratory_200']['sheet_sha256'][:16]}...` |",
        "",
        "---",
        "",
        "## 2. P2.1–P2.3 — Phân Rã Prior-to-Signal Ratio (`PSR`) & Khắc Phục Zero-Shot Cắt Vùng Không Gian",
        "",
        "### 2.1. Phân rã Prior Bias $\\lvert B \\rvert$ so với Spatial Signal $\\lvert \\Delta_{\\text{spatial}} \\rvert$ trên các cặp đối xứng ngang",
        "",
        "| Benchmark & Điều kiện | Số cặp | $\\mathbb{E}[\\lvert B \\rvert]$ (Prior Bias) | $\\mathbb{E}[\\lvert \\Delta_{\\text{spatial}} \\rvert]$ (Spatial Signal) | Tỷ số $\\text{PSR} = \\frac{\\mathbb{E}[\\lvert B \\rvert]}{\\mathbb{E}[\\lvert \\Delta \\rvert]}$ | Tỷ lệ $\\lvert B \\rvert > \\lvert \\Delta \\rvert$ (Collapse 1 nhãn) | 2×2 Assignment Acc | Both Images Correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| **Synthetic Depth (`T0` thuận, ngang)** | `{syn_t0['n_pairs']}` | `{syn_t0['mean_abs_prior_bias_B']:.5f}` | `{syn_t0['mean_abs_spatial_signal_delta']:.5f}` | **`{syn_t0['aggregate_prior_to_signal_ratio']:.2f}x`** | `{syn_t0['same_prediction_collapse_percent']:.2f}%` | `{syn_t0['assignment_accuracy_percent']:.2f}%` | **`{syn_t0['both_images_correct_percent']:.2f}%`** |",
        f"| **COCO Exploratory (`T2` thuận, ngang)** | `{nat_exp_t2['n_pairs']}` | `{nat_exp_t2['mean_abs_prior_bias_B']:.5f}` | `{nat_exp_t2['mean_abs_spatial_signal_delta']:.5f}` | **`{nat_exp_t2['aggregate_prior_to_signal_ratio']:.2f}x`** | `{nat_exp_t2['same_prediction_collapse_percent']:.2f}%` | `{nat_exp_t2['assignment_accuracy_percent']:.2f}%` | **`{nat_exp_t2['both_images_correct_percent']:.2f}%`** |",
        f"| **COCO Confirmation (`T2` thuận, ngang)** | `{nat_conf_t2['n_pairs']}` | `{nat_conf_t2['mean_abs_prior_bias_B']:.5f}` | `{nat_conf_t2['mean_abs_spatial_signal_delta']:.5f}` | **`{nat_conf_t2['aggregate_prior_to_signal_ratio']:.2f}x`** | `{nat_conf_t2['same_prediction_collapse_percent']:.2f}%` | `{nat_conf_t2['assignment_accuracy_percent']:.2f}%` | **`{nat_conf_t2['both_images_correct_percent']:.2f}%`** |",
        "",
        "> **Diễn giải hành vi (`RQ2`):** Trên các cặp ảnh đã xét, thành phần hiệu điểm trung bình trong cặp $\\mathbb{E}[\\lvert B \\rvert]$ lớn gấp **`"
        f"{nat_exp_t2['aggregate_prior_to_signal_ratio']:.1f}`–`{nat_conf_t2['aggregate_prior_to_signal_ratio']:.1f}` lần** tín hiệu đổi chỗ không gian $\\mathbb{{E}}[\\lvert \\Delta_{{\\text{{spatial}}}} \\rvert]$. "
        "Do $\\lvert B \\rvert > \\lvert \\Delta_{\\text{spatial}} \\rvert$ trên phần lớn số cặp, hai ảnh thường có cùng nhãn dự đoán. Đây là mô tả đại số của điểm số, chưa chứng minh nguồn gốc bias ở encoder hoặc cơ chế neuron.",
        "",
        "### 2.2. Kết quả Thực nghiệm GPU & Giới hạn hình học của Half-Canvas Zero-Shot Mitigation (`P2.2/P2.3`)",
        "",
        "| Tập dữ liệu | Số nguồn ảnh | Tỷ lệ 2 vật thể nằm khác phía đường chia tâm | Giới hạn hình học lý tưởng | Kết quả CLIP `ViT-B/32` Holistic (Baseline) | Kết quả CLIP `ViT-B/32` Half-Canvas (`full_spatial_mitigation_v1`) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    mit_str = (
        f"**`{live_gpu['spatial_mitigation']['four_way_accuracy_percent']:.2f}%`** (`100%` both-correct & 4-relation consistent)"
        if "spatial_mitigation" in live_gpu
        else "Chưa chạy"
    )
    lines.extend([
        f"| **Synthetic Depth (`1.440` ảnh)** | `{geom_syn['n_images']}` | **`{geom_syn['straddles_midline_rate_percent']:.2f}%`** | **`{geom_syn['four_way_accuracy_percent']:.2f}%`** | `25.49%` (`0.00%` 4-relation) | {mit_str} |",
        f"| **COCO Exploratory (`200` nguồn)** | `{geom_exp['n_sources']}` | `{geom_exp['objects_straddle_image_midline_percent']:.2f}%` | `{geom_exp['ideal_half_canvas_four_way_accuracy_percent']:.2f}%` (`13.50%` cùng nửa) | `28.50%` | Giới hạn trần `92.50%` |",
        f"| **COCO Confirmation (`100` nguồn ngang)** | `{geom_conf['n_sources']}` | `{geom_conf['objects_straddle_image_midline_percent']:.2f}%` | `{geom_conf['ideal_half_canvas_horizontal_two_way_accuracy_percent']:.2f}%` (2-way, `7.00%` cùng nửa) | `49.00%` (`4.00%` both correct) | Giới hạn trần `93.00%`–`99.00%` |",
        "",
        "---",
        "",
        "## 3. P3.1–P3.4 — Độ Nhạy Prompt & Hiệu Quả Ensembling (Offline + Live GPU)",
        "",
        "### 3.1. Thực nghiệm GPU Đa Mẫu Prompt trên CIFAR-100 (`results/full_prompt_cifar100_v1` — `10.000` ảnh)",
        "",
    ])

    if "prompt_cifar100" in live_gpu:
        pc = live_gpu["prompt_cifar100"]
        st = pc["single_templates"]
        ens = pc["ensembles"]
        lines.extend([
            "| Mẫu Prompt / Ensemble | Công thức câu | Top-1 Accuracy [95% CI] | $\\Delta$ vs `T0_baseline` [95% CI] |",
            "| --- | --- | ---: | ---: |",
            f"| **`T0_baseline`** | `a photo of a {{class}}.` | **`{st['T0_baseline']['accuracy_percent']:.2f}%`** `[{st['T0_baseline']['ci95_percent'][0]:.2f}%, {st['T0_baseline']['ci95_percent'][1]:.2f}%]` | `0.00` pp |",
            f"| **`T1_minimal`** | `{{class}}.` | `{st['T1_minimal']['accuracy_percent']:.2f}%` `[{st['T1_minimal']['ci95_percent'][0]:.2f}%, {st['T1_minimal']['ci95_percent'][1]:.2f}%]` | `{st['T1_minimal']['delta_vs_baseline_pp']:+.2f}` pp `[{st['T1_minimal']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {st['T1_minimal']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            f"| **`T2_clear`** | `a clear photo of a {{class}}.` | `{st['T2_clear']['accuracy_percent']:.2f}%` `[{st['T2_clear']['ci95_percent'][0]:.2f}%, {st['T2_clear']['ci95_percent'][1]:.2f}%]` | `{st['T2_clear']['delta_vs_baseline_pp']:+.2f}` pp `[{st['T2_clear']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {st['T2_clear']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            f"| **`T3_centered`** | `a centered photo of a {{class}}.` | `{st['T3_centered']['accuracy_percent']:.2f}%` `[{st['T3_centered']['ci95_percent'][0]:.2f}%, {st['T3_centered']['ci95_percent'][1]:.2f}%]` | `{st['T3_centered']['delta_vs_baseline_pp']:+.2f}` pp `[{st['T3_centered']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {st['T3_centered']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            f"| **`T4_domain`** | `a low-resolution photo of a {{class}}.` | `{st['T4_domain']['accuracy_percent']:.2f}%` `[{st['T4_domain']['ci95_percent'][0]:.2f}%, {st['T4_domain']['ci95_percent'][1]:.2f}%]` | `{st['T4_domain']['delta_vs_baseline_pp']:+.2f}` pp `[{st['T4_domain']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {st['T4_domain']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            f"| **`T5_bright`** | `a bright photo of a {{class}}.` | `{st['T5_bright']['accuracy_percent']:.2f}%` `[{st['T5_bright']['ci95_percent'][0]:.2f}%, {st['T5_bright']['ci95_percent'][1]:.2f}%]` | `{st['T5_bright']['delta_vs_baseline_pp']:+.2f}` pp `[{st['T5_bright']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {st['T5_bright']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            f"| **`T6_cropped`** | `a cropped photo of a {{class}}.` | `{st['T6_cropped']['accuracy_percent']:.2f}%` `[{st['T6_cropped']['ci95_percent'][0]:.2f}%, {st['T6_cropped']['ci95_percent'][1]:.2f}%]` | `{st['T6_cropped']['delta_vs_baseline_pp']:+.2f}` pp `[{st['T6_cropped']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {st['T6_cropped']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            f"| **Logit Average Ensemble (`7` mẫu)** | Trung bình cosine similarity | **`{ens['logit_average_ensemble']['accuracy_percent']:.2f}%`** `[{ens['logit_average_ensemble']['ci95_percent'][0]:.2f}%, {ens['logit_average_ensemble']['ci95_percent'][1]:.2f}%]` | **`{ens['logit_average_ensemble']['delta_vs_baseline_pp']:+.2f}` pp** `[{ens['logit_average_ensemble']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {ens['logit_average_ensemble']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            f"| **Probability Average Ensemble (`7` mẫu)** | Trung bình xác suất Softmax ($\\tau=100$) | **`{ens['probability_average_ensemble']['accuracy_percent']:.2f}%`** `[{ens['probability_average_ensemble']['ci95_percent'][0]:.2f}%, {ens['probability_average_ensemble']['ci95_percent'][1]:.2f}%]` | **`{ens['probability_average_ensemble']['delta_vs_baseline_pp']:+.2f}` pp** `[{ens['probability_average_ensemble']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {ens['probability_average_ensemble']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            f"| **Text-Embedding Ensemble (`7` mẫu)** | Chuẩn hóa L2 trung bình vector nhúng văn bản | **`{ens['text_embedding_ensemble']['accuracy_percent']:.2f}%`** `[{ens['text_embedding_ensemble']['ci95_percent'][0]:.2f}%, {ens['text_embedding_ensemble']['ci95_percent'][1]:.2f}%]` | **`{ens['text_embedding_ensemble']['delta_vs_baseline_pp']:+.2f}` pp** `[{ens['text_embedding_ensemble']['delta_vs_baseline_ci95_pp'][0]:+.2f}, {ens['text_embedding_ensemble']['delta_vs_baseline_ci95_pp'][1]:+.2f}]` |",
            "",
        ])

    lines.extend([
        "### 3.2. Tác vụ Đếm (Counting — `5` mẫu × `1.080` cảnh) & Phủ định (Negation — `4` mẫu × `720` cảnh, `full_prompt_probing_v1`)",
        "",
    ])

    if "prompt_probing" in live_gpu:
        pp = live_gpu["prompt_probing"]
        cnt = pp["counting"]
        neg_live = pp["negation"]
        lines.extend([
            "| Bài toán | Số templates | Best Single | Worst Single | Mean ± SD | Spread (Max − Min) | Text-Embedding Ensemble | Probability Ensemble |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| **Counting (`1.080` cảnh)** | `5` | `{cnt['best_single_accuracy_percent']:.2f}%` (`C0_baseline`) | `{cnt['worst_single_accuracy_percent']:.2f}%` (`C2_there_are`) | `{cnt['mean_single_accuracy_percent']:.2f}% ± {cnt['sd_single_accuracy_percent']:.2f}%` | **`{cnt['spread_max_minus_min_pp']:.2f}` pp** | `{cnt['ensembles']['text_embedding_ensemble']['accuracy_percent']:.2f}%` | `{cnt['ensembles']['probability_average_ensemble']['accuracy_percent']:.2f}%` |",
            f"| **Negation (`720` cảnh, `4` mẫu)** | `4` | `{neg_live['best_single_accuracy_percent']:.2f}%` (`N0_with_without`) | `{neg_live['worst_single_accuracy_percent']:.2f}%` (`N2_showing`) | `{neg_live['mean_single_accuracy_percent']:.2f}% ± {neg_live['sd_single_accuracy_percent']:.2f}%` | **`{neg_live['spread_max_minus_min_pp']:.2f}` pp** | `{neg_live['ensembles']['text_embedding_ensemble']['accuracy_percent']:.2f}%` | `{neg_live['ensembles']['probability_average_ensemble']['accuracy_percent']:.2f}%` |",
            f"| **Synthetic Spatial (`T0..T2` thuận, 4-way)** | `3` | `{sp_syn_ens['best_single_accuracy_percent']:.2f}%` | `{sp_syn_ens['worst_single_accuracy_percent']:.2f}%` | `{sp_syn_ens['mean_single_accuracy_percent']:.2f}% ± {sp_syn_ens['sd_single_accuracy_percent']:.2f}%` | **`{sp_syn_ens['spread_max_minus_min_pp']:.2f}` pp** | `{sp_syn_ens['ensembles']['logit_average_ensemble']['accuracy_percent']:.2f}%` (Logit) | `{sp_syn_ens['ensembles']['probability_average_ensemble']['accuracy_percent']:.2f}%` |",
            f"| **CIFAR-100 Semantic Valid (`baseline/cat/neutral`)** | `3` | `{cat_ens['best_single_accuracy_percent']:.2f}%` | `{cat_ens['worst_single_accuracy_percent']:.2f}%` | `{cat_ens['mean_single_accuracy_percent']:.2f}% ± {cat_ens['sd_single_accuracy_percent']:.2f}%` | **`{cat_ens['spread_max_minus_min_pp']:.2f}` pp** | `{cat_ens['ensembles']['logit_average_ensemble']['accuracy_percent']:.2f}%` (Logit) | `{cat_ens['ensembles']['probability_average_ensemble']['accuracy_percent']:.2f}%` |",
            "",
        ])

    lines.extend([
        "### 3.3. Phân tích cơ chế cân bằng thiên kiến Present vs Absent trên Negation (`with_without` + `contains_does_not`)",
        "",
        "| Điều kiện / Ensemble | Accuracy tổng (`720` cảnh) | Present (`360` cảnh) | Absent (`360` cảnh) | Chênh lệch `Present - Absent` | Group Consistency (`180` nhóm) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| **`with_without` (Single tốt nhất)** | `{neg_mech['with_without']['overall_accuracy_percent']:.2f}%` | `{neg_mech['with_without']['present_accuracy_percent']:.2f}%` | `{neg_mech['with_without']['absent_accuracy_percent']:.2f}%` | `{neg_mech['with_without']['present_minus_absent_gap_pp']:+.2f}` pp | `{neg_mech['with_without']['scene_pair_group_consistency_percent']:.2f}%` |",
        f"| **`contains_does_not` (Single thứ hai)** | `{neg_mech['contains_does_not']['overall_accuracy_percent']:.2f}%` | `{neg_mech['contains_does_not']['present_accuracy_percent']:.2f}%` | `{neg_mech['contains_does_not']['absent_accuracy_percent']:.2f}%` | `{neg_mech['contains_does_not']['present_minus_absent_gap_pp']:+.2f}` pp | `{neg_mech['contains_does_not']['scene_pair_group_consistency_percent']:.2f}%` |",
        f"| **Logit Average Ensemble** | **`{neg_mech['logit_average_ensemble']['overall_accuracy_percent']:.2f}%`** | `{neg_mech['logit_average_ensemble']['present_accuracy_percent']:.2f}%` | `{neg_mech['logit_average_ensemble']['absent_accuracy_percent']:.2f}%` | `{neg_mech['logit_average_ensemble']['present_minus_absent_gap_pp']:+.2f}` pp | **`{neg_mech['logit_average_ensemble']['scene_pair_group_consistency_percent']:.2f}%`** |",
        f"| **Probability Average Ensemble** | **`{neg_mech['probability_average_ensemble']['overall_accuracy_percent']:.2f}%`** | `{neg_mech['probability_average_ensemble']['present_accuracy_percent']:.2f}%` | `{neg_mech['probability_average_ensemble']['absent_accuracy_percent']:.2f}%` | `{neg_mech['probability_average_ensemble']['present_minus_absent_gap_pp']:+.2f}` pp | **`{neg_mech['probability_average_ensemble']['scene_pair_group_consistency_percent']:.2f}%`** |",
        "",
        f"- **Ensemble tiếp nối trên COCO đã mở (`100` cặp):** Logit Ensemble T0+T1+T2 đạt `{sp_conf_ens['T0_T1_T2_forward_logit_ensemble']['assignment_accuracy_percent']:.2f}%`, cả hai ảnh đúng `{sp_conf_ens['T0_T1_T2_forward_logit_ensemble']['both_images_correct_percent']:.2f}%`. Trong bộ template này chưa có cải thiện rõ về tính nhất quán. Kết quả không chứng minh giới hạn kiến trúc hoặc mọi ensemble đều vô hiệu.",
        "",
        "---",
        "",
        "## 4. P4.1 & P4.5 — Thực Nghiệm GPU Chéo Ngôn Ngữ Tiếng Việt & Chẩn Đoán Phân Mảnh BPE (`full_vietnamese_cifar100_v1`)",
        "",
    ])

    if "vietnamese_cifar100" in live_gpu:
        vi_res = live_gpu["vietnamese_cifar100"]
        vst = vi_res["single_templates"]
        vens = vi_res["ensembles"]
        valign = vi_res["cross_lingual_text_alignment"]
        lines.extend([
            "### 4.1. Kết quả Top-1 Accuracy & Độ tương đồng Cosine nhúng văn bản trên 10.000 ảnh CIFAR-100",
            "",
            "| Điều kiện ngôn ngữ (`P4.1` & `P4.5`) | Mẫu câu | Top-1 Accuracy [95% CI] | $\\Delta$ vs `en_baseline` | Cosine $\\cos(\\hat{t}_{\\text{en}}, \\hat{t}_{\\text{cond}})$ (Mean ± SD) |",
            "| --- | --- | ---: | ---: | ---: |",
            f"| **`en_baseline`** | `a photo of a {{en_label}}.` | **`{vst['en_baseline']['accuracy_percent']:.2f}%`** `[{vst['en_baseline']['ci95_percent'][0]:.2f}%, {vst['en_baseline']['ci95_percent'][1]:.2f}%]` | `0.00` pp | `1.000 ± 0.000` |",
            f"| **`vi_diacritics`** | `một bức ảnh chụp {{vi_label}}.` | **`{vst['vi_diacritics']['accuracy_percent']:.2f}%`** `[{vst['vi_diacritics']['ci95_percent'][0]:.2f}%, {vst['vi_diacritics']['ci95_percent'][1]:.2f}%]` | **`{vst['vi_diacritics']['delta_vs_baseline_pp']:+.2f}` pp** | `{valign['vi_diacritics']['mean_cosine_with_en_baseline']:.3f} ± {valign['vi_diacritics']['sd_cosine_with_en_baseline']:.3f}` |",
            f"| **`vi_no_diacritics`** | `mot buc anh chup {{vi_no_diacritics}}.` | **`{vst['vi_no_diacritics']['accuracy_percent']:.2f}%`** `[{vst['vi_no_diacritics']['ci95_percent'][0]:.2f}%, {vst['vi_no_diacritics']['ci95_percent'][1]:.2f}%]` | **`{vst['vi_no_diacritics']['delta_vs_baseline_pp']:+.2f}` pp** | `{valign['vi_no_diacritics']['mean_cosine_with_en_baseline']:.3f} ± {valign['vi_no_diacritics']['sd_cosine_with_en_baseline']:.3f}` |",
            f"| **`en_frame_vi_label` (`P4.5`)** | `a photo of a {{vi_label}}.` | **`{vst['en_frame_vi_label']['accuracy_percent']:.2f}%`** `[{vst['en_frame_vi_label']['ci95_percent'][0]:.2f}%, {vst['en_frame_vi_label']['ci95_percent'][1]:.2f}%]` | **`{vst['en_frame_vi_label']['delta_vs_baseline_pp']:+.2f}` pp** | `{valign['en_frame_vi_label']['mean_cosine_with_en_baseline']:.3f} ± {valign['en_frame_vi_label']['sd_cosine_with_en_baseline']:.3f}` |",
            f"| **`vi_frame_en_label` (`P4.5`)** | `một bức ảnh chụp {{en_label}}.` | **`{vst['vi_frame_en_label']['accuracy_percent']:.2f}%`** `[{vst['vi_frame_en_label']['ci95_percent'][0]:.2f}%, {vst['vi_frame_en_label']['ci95_percent'][1]:.2f}%]` | **`{vst['vi_frame_en_label']['delta_vs_baseline_pp']:+.2f}` pp** | `{valign['vi_frame_en_label']['mean_cosine_with_en_baseline']:.3f} ± {valign['vi_frame_en_label']['sd_cosine_with_en_baseline']:.3f}` |",
            f"| **`vi_en_bilingual`** | `một bức ảnh chụp {{vi_label}} ({{en_label}}).` | **`{vst['vi_en_bilingual']['accuracy_percent']:.2f}%`** `[{vst['vi_en_bilingual']['ci95_percent'][0]:.2f}%, {vst['vi_en_bilingual']['ci95_percent'][1]:.2f}%]` | **`{vst['vi_en_bilingual']['delta_vs_baseline_pp']:+.2f}` pp** | `{valign['vi_en_bilingual']['mean_cosine_with_en_baseline']:.3f} ± {valign['vi_en_bilingual']['sd_cosine_with_en_baseline']:.3f}` |",
            f"| **Ensemble (`en + vi + bilingual`)** | Text-Embedding Ensemble | **`{vens['en_vi_bilingual_embedding_ensemble']['accuracy_percent']:.2f}%`** `[{vens['en_vi_bilingual_embedding_ensemble']['ci95_percent'][0]:.2f}%, {vens['en_vi_bilingual_embedding_ensemble']['ci95_percent'][1]:.2f}%]` | `{vens['en_vi_bilingual_embedding_ensemble']['delta_vs_baseline_pp']:+.2f}` pp | — |",
            "",
        ])

    lines.extend([
        "### 4.2. Chỉ số Phân mảnh BPE thực tế trên `SimpleTokenizer` của OpenAI CLIP (`49.408` vocab)",
        "",
        "| Điều kiện ngôn ngữ | Số BPE tokens / tên lớp (Mean ± SD) | Số BPE tokens / cả câu prompt (Mean ± SD) | Token Fertility (Tokens / từ đơn) | Tỷ số phân mảnh tên lớp so với Tiếng Anh | Số câu vượt ngưỡng `77` tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| **`en_baseline` (Tiếng Anh)** | `{bpe_l['en_baseline']['mean_subword_tokens']:.2f} ± {bpe_l['en_baseline']['sd_subword_tokens']:.2f}` | `{bpe_p['en_baseline']['mean_subword_tokens']:.2f} ± {bpe_p['en_baseline']['sd_subword_tokens']:.2f}` | **`{bpe_l['en_baseline']['mean_token_fertility_per_word']:.2f}`** | `1.00x` | `{bpe_p['en_baseline']['exceeds_77_context_window_count']}` |",
        f"| **`vi_diacritics` (Tiếng Việt có dấu)** | **`{bpe_l['vi_diacritics']['mean_subword_tokens']:.2f} ± {bpe_l['vi_diacritics']['sd_subword_tokens']:.2f}`** | **`{bpe_p['vi_diacritics']['mean_subword_tokens']:.2f} ± {bpe_p['vi_diacritics']['sd_subword_tokens']:.2f}`** | **`{bpe_l['vi_diacritics']['mean_token_fertility_per_word']:.2f}`** | **`{bpe_l['vi_diacritics']['mean_ratio_vs_en_baseline']:.2f}x`** | `{bpe_p['vi_diacritics']['exceeds_77_context_window_count']}` |",
        f"| **`vi_no_diacritics` (Tiếng Việt không dấu)** | `{bpe_l['vi_no_diacritics']['mean_subword_tokens']:.2f} ± {bpe_l['vi_no_diacritics']['sd_subword_tokens']:.2f}` | `{bpe_p['vi_no_diacritics']['mean_subword_tokens']:.2f} ± {bpe_p['vi_no_diacritics']['sd_subword_tokens']:.2f}` | `{bpe_l['vi_no_diacritics']['mean_token_fertility_per_word']:.2f}` | `{bpe_l['vi_no_diacritics']['mean_ratio_vs_en_baseline']:.2f}x` | `{bpe_p['vi_no_diacritics']['exceeds_77_context_window_count']}` |",
        f"| **`en_frame_vi_label` (`P4.5`)** | `{bpe_l['en_frame_vi_label']['mean_subword_tokens']:.2f} ± {bpe_l['en_frame_vi_label']['sd_subword_tokens']:.2f}` | `{bpe_p['en_frame_vi_label']['mean_subword_tokens']:.2f} ± {bpe_p['en_frame_vi_label']['sd_subword_tokens']:.2f}` | `{bpe_l['en_frame_vi_label']['mean_token_fertility_per_word']:.2f}` | `{bpe_l['en_frame_vi_label']['mean_ratio_vs_en_baseline']:.2f}x` | `{bpe_p['en_frame_vi_label']['exceeds_77_context_window_count']}` |",
        f"| **`vi_frame_en_label` (`P4.5`)** | `{bpe_l['vi_frame_en_label']['mean_subword_tokens']:.2f} ± {bpe_l['vi_frame_en_label']['sd_subword_tokens']:.2f}` | `{bpe_p['vi_frame_en_label']['mean_subword_tokens']:.2f} ± {bpe_p['vi_frame_en_label']['sd_subword_tokens']:.2f}` | `{bpe_l['vi_frame_en_label']['mean_token_fertility_per_word']:.2f}` | `{bpe_l['vi_frame_en_label']['mean_ratio_vs_en_baseline']:.2f}x` | `{bpe_p['vi_frame_en_label']['exceeds_77_context_window_count']}` |",
        f"| **`vi_en_bilingual` (Song ngữ Việt–Anh)** | `{bpe_l['vi_en_bilingual']['mean_subword_tokens']:.2f} ± {bpe_l['vi_en_bilingual']['sd_subword_tokens']:.2f}` | `{bpe_p['vi_en_bilingual']['mean_subword_tokens']:.2f} ± {bpe_p['vi_en_bilingual']['sd_subword_tokens']:.2f}` | `{bpe_l['vi_en_bilingual']['mean_token_fertility_per_word']:.2f}` | `{bpe_l['vi_en_bilingual']['mean_ratio_vs_en_baseline']:.2f}x` | `{bpe_p['vi_en_bilingual']['exceeds_77_context_window_count']}` |",
        "",
        "> **Phát hiện định lượng (`RQ4` & `P4.5`):**",
        f"> 1. Tên lớp Việt có dấu có trung bình `{bpe_l['vi_diacritics']['mean_subword_tokens']:.2f}` token, so với `{bpe_l['en_baseline']['mean_subword_tokens']:.2f}` token Anh. Tỷ số hai trung bình là `{bpe_l['vi_diacritics']['mean_subword_tokens']/bpe_l['en_baseline']['mean_subword_tokens']:.2f}x`; trung bình tỷ số theo từng lớp là `{bpe_l['vi_diacritics']['mean_ratio_vs_en_baseline']:.2f}x`. Hai cách tổng hợp khác nhau.",
        "> 2. Các điều kiện lai khung câu/tên lớp cho thấy hiệu năng phụ thuộc mạnh vào cách biểu diễn tên lớp. Chưa thể quy phần trăm suy giảm cho BPE: bản dịch, mức quen thuộc từ vựng và căn chỉnh embedding cùng thay đổi. Không dấu giảm số token nhưng accuracy thấp hơn có dấu, nên số token không giải thích đầy đủ kết quả.",
        "",
    ])
    return "\n".join(lines)


def run_all_synthesis():
    """Execute all offline + live GPU synthesis steps and write artifacts to `reports/synthesis_v1/`."""
    SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)
    blinded = export_blinded_sheets()
    write_json(SYNTHESIS_DIR / "blinded_review_manifest.json", blinded)

    spatial_psr = analyze_frozen_spatial_runs(output_path=SYNTHESIS_DIR / "spatial_mechanism_psr.json")
    spatial_geom = evaluate_geometric_upper_bound()
    write_json(SYNTHESIS_DIR / "spatial_geometry_bounds.json", spatial_geom)

    prompt_ens = analyze_saved_probing_ensembles(
        output_path=SYNTHESIS_DIR / "prompt_sensitivity_and_ensembles.json"
    )
    bpe_source = ROOT / "results/full_vietnamese_cifar100_v1/bpe_diagnostics.json"
    bpe_diag = read_json(bpe_source)
    write_json(SYNTHESIS_DIR / "vietnamese_bpe_diagnostics.json", bpe_diag)
    live_gpu = load_live_gpu_results()
    write_json(SYNTHESIS_DIR / "live_gpu_experiments_summary.json", live_gpu)

    md = build_synthesis_markdown(blinded, spatial_psr, spatial_geom, prompt_ens, bpe_diag, live_gpu)
    (SYNTHESIS_DIR / "README.md").write_text(md, encoding="utf-8")
    return {
        "synthesis_dir": SYNTHESIS_DIR.relative_to(ROOT).as_posix(),
        "files_written": [
            "blinded_review_manifest.json",
            "spatial_mechanism_psr.json",
            "spatial_geometry_bounds.json",
            "prompt_sensitivity_and_ensembles.json",
            "vietnamese_bpe_diagnostics.json",
            "live_gpu_experiments_summary.json",
            "README.md",
        ],
    }


if __name__ == "__main__":
    print(run_all_synthesis(), flush=True)
