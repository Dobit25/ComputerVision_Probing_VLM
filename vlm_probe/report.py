"""Build reproducible tables/figures from audited results; never run inference."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .io_utils import sha256, write_json
from .verify import verify

RUNS = [
    ("CIFAR-100", "full_cifar100_clip_b32_v1"),
    ("Flowers102", "full_flowers102_clip_b32_v2"),
    ("Oxford-IIIT Pet", "full_oxford_pet_clip_b32_v3"),
    ("DTD partition 1", "full_dtd_clip_b32_v3"),
    ("Food-101", "full_food101_clip_b32_v3"),
]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plotting(root):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def baseline(root):
    output = root / "reports/baseline"
    output.mkdir(parents=True, exist_ok=True)
    summary, per_class, pairs, inputs, sections = [], [], [], {}, []
    for label, name in RUNS:
        run = root / "results" / name
        audit = verify(run)
        if audit["mode"] != "full_test":
            raise ValueError("Baseline report requires full test runs.")
        inputs[name] = {"manifest_sha256": sha256(run / "manifest.json"), "audit": audit}
        rows = read_csv(run / "per_class.csv")
        for row in rows:
            per_class.append({"dataset": label, "class_id": int(row["class_id"]), "class_name": row["class_name"],
                              "support": int(row["support"]), "correct": int(row["correct"]),
                              "accuracy_percent": 100 * float(row["accuracy"])})
        macro = 100 * sum(float(r["accuracy"]) for r in rows) / len(rows)
        summary.append({"dataset": label, "num_classes": len(rows), "num_samples": audit["num_samples"],
                        "correct": audit["correct"], "top1_percent": audit["top1_accuracy_percent"],
                        "mean_per_class_percent": macro, "run": name})
        predictions = read_csv(run / "predictions.csv")
        counts = Counter((r["target_name"], r["prediction_name"]) for r in predictions if r["correct"] == "0")
        support = {r["class_name"]: int(r["support"]) for r in rows}
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for (truth, pred), count in ordered:
            pairs.append({"dataset": label, "true_class": truth, "predicted_class": pred,
                          "error_count": count, "true_class_support": support[truth],
                          "percent_of_true_class": 100 * count / support[truth]})
        section = [f"## {label}", "", f"[Kết quả gốc](../../results/{name}/metrics.json) · "
                   f"[Dự đoán từng ảnh](../../results/{name}/predictions.csv) · "
                   f"[Confusion matrix đầy đủ](../../results/{name}/confusion_matrix.csv)", "",
                   "Năm lớp có accuracy thấp nhất (xếp theo accuracy, rồi tên lớp):", "",
                   "| Lớp | Đúng / số ảnh | Accuracy |", "| --- | ---: | ---: |"]
        for r in sorted(rows, key=lambda r: (float(r["accuracy"]), r["class_name"]))[:5]:
            section.append(f"| {r['class_name']} | {r['correct']} / {r['support']} | {100 * float(r['accuracy']):.2f}% |")
        section += ["", "Năm cặp nhầm có nhiều ảnh nhất; chiều mũi tên là nhãn thật → dự đoán:", "",
                    "| Cặp nhầm | Số ảnh | % trong lớp thật |", "| --- | ---: | ---: |"]
        for (truth, pred), count in ordered[:5]:
            section.append(f"| {truth} → {pred} | {count} | {100 * count / support[truth]:.2f}% |")
        sections.extend(section + [""])
    save_csv(output / "summary.csv", summary)
    save_csv(output / "per_class.csv", per_class)
    save_csv(output / "confusion_pairs.csv", pairs)
    plt = plotting(root)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = list(range(len(summary)))
    ax.bar([i - .18 for i in x], [r["top1_percent"] for r in summary], width=.36, label="Top-1 (per image)")
    ax.bar([i + .18 for i in x], [r["mean_per_class_percent"] for r in summary], width=.36, label="Mean per class")
    ax.set(xticks=x, xticklabels=[r["dataset"] for r in summary], ylim=(0, 100), ylabel="Accuracy (%)",
           title="OpenAI CLIP ViT-B/32 — official test splits")
    ax.legend()
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)
    lines = ["# Baseline một model trên năm dataset", "", "**Đã kiểm chứng tất cả 5 full test run.** "
             "Cùng OpenAI CLIP ViT-B/32, FP32, QuickGELU, không training/fine-tuning. "
             "CIFAR-100 và Flowers102 dùng kết quả đã lưu; ba dataset còn lại là các ô mới.", "",
             "| Dataset | Lớp | Ảnh test | Đúng | Top-1 | Trung bình theo lớp |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['num_classes']} | {r['num_samples']} | {r['correct']} | "
                     f"{r['top1_percent']:.4f}% | {r['mean_per_class_percent']:.4f}% |")
    lines += ["", "![Accuracy](accuracy.png)", "",
              "Top-1 là tỷ lệ đúng trên toàn bộ ảnh. Trung bình theo lớp cho mỗi lớp trọng số bằng nhau; "
              "hai metric có thể khác trên Flowers/Pet vì số ảnh mỗi lớp không đều. "
              "Không gộp năm dataset thành một accuracy chung vì Food-101 sẽ chi phối kết quả.", "",
              "Prompt cố định `a photo of a {class}.`, riêng DTD là `a photo of a {class} texture.`. "
              "DTD chỉ dùng test partition 1, không phải trung bình 10 partition. "
              "Các con số giữa dataset phản ánh cả miền ảnh, số lớp và prompt, không phải so sánh model.", "",
              "Dữ liệu chi tiết: [toàn bộ 387 lớp](per_class.csv), [mọi cặp nhầm có quan sát](confusion_pairs.csv), "
              "[bảng tổng hợp](summary.csv), [biên bản đầu vào](inputs.json).", "", *sections,
              "## Câu hỏi đưa sang probing", "",
              "- Khi giữ nguyên màu và hình, đổi số lượng 1–6 có làm model chọn đúng số không? "
              "So sánh kích thước vật thể cố định với tổng diện tích cố định.",
              "- Khi giữ nguyên hai vật thể và chỉ đổi vị trí, model phân biệt trái/phải/trên/dưới được không?",
              "- Khi chỉ thay việc có/không có vật thể được hỏi, model đổi lựa chọn giữa câu khẳng định và phủ định được không?",
              "- Trong cùng một nhóm lớn CIFAR-100, model còn nhầm bao nhiêu khi chỉ chọn giữa năm lớp gần nhau?", "",
              "Các cặp nhầm baseline gợi ý nơi cần xem sâu, chưa xác định nguyên nhân nhân quả. "
              "Ba câu hỏi đếm/vị trí/phủ định cần bộ ảnh kiểm soát riêng; accuracy classification không trực tiếp đo chúng.", "",
              "## Tái lập báo cáo", "", "```powershell", "conda activate gen", "python -m vlm_probe.report baseline", "```", "",
              "Lệnh kiểm tra artifact và tính lại bảng/biểu đồ; không chạy model. "
              "Giao thức và lệnh chạy inference: [README](../../README.md), [phạm vi hiện hành](../../docs/one_model_protocol.md).", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "inputs.json", {"runs": inputs, "report_source_sha256": sha256(Path(__file__))})
    print(f"Baseline report: {output}", flush=True)


def probes(root, name):
    from .probing.audit import verify as audit_probes
    run = root / "results" / name
    audit = audit_probes(root, run)
    if audit["mode"] != "full":
        raise ValueError("Report requires a full probe run.")
    output = root / "reports/probing"
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((run / "metrics.json").read_text())
    rows, breakdown = [], []
    for task, stats in metrics.items():
        low, high = stats["accuracy_ci95_group_bootstrap"]
        rows.append({"task": task, "num_samples": stats["num_samples"], "correct": stats["correct"],
                     "accuracy_percent": stats["accuracy_percent"], "ci95_low_percent": 100 * low,
                     "ci95_high_percent": 100 * high, "chance_percent": 100 * stats["chance_accuracy"],
                     "control_accuracy_percent": 100 * stats["control_accuracy"] if "control_accuracy" in stats else "",
                     "control_correct_samples": stats.get("control_correct_samples", ""),
                     "probe_correct_when_control_correct": round(stats["accuracy_given_control_correct"] * stats["control_correct_samples"])
                     if stats.get("accuracy_given_control_correct") is not None else "",
                     "accuracy_given_control_correct_percent": 100 * stats["accuracy_given_control_correct"]
                     if stats.get("accuracy_given_control_correct") is not None else ""})
        for field in ("condition", "stratum", "color", "shape"):
            for value, part in stats.get(f"by_{field}", {}).items():
                breakdown.append({"task": task, "group_by": field, "value": value, "num_samples": part["num_samples"],
                                  "correct": part["correct"], "accuracy_percent": part["accuracy_percent"]})
    save_csv(output / "summary.csv", rows)
    save_csv(output / "breakdown.csv", breakdown)
    plt = plotting(root)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = list(range(len(rows)))
    values = [r["accuracy_percent"] for r in rows]
    ax.bar(x, values, yerr=[[v-r["ci95_low_percent"] for v,r in zip(values,rows)],
                          [r["ci95_high_percent"]-v for v,r in zip(values,rows)]], capsize=5)
    ax.scatter(x, [r["chance_percent"] for r in rows], marker="_", s=700, c="red", label="Uniform random choice")
    ax.set(xticks=x, xticklabels=["fine-grained\n(oracle 5-way)" if r["task"] == "fine_grained" else r["task"] for r in rows], ylim=(0, 100), ylabel="Accuracy (%)",
           title="Controlled probes — group bootstrap 95% intervals")
    ax.legend()
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)
    # Deterministic illustrations: first four errors with correct recognition controls.
    # These are post-hoc examples, never an additional evaluation subset.
    import textwrap
    from PIL import Image
    samples = {s["sample_id"]: s for s in json.loads((run / "samples.json").read_text())}
    predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(3, 4, figsize=(14, 12), layout="constrained")
    example_rows = []
    for i, task in enumerate(("counting", "spatial", "negation")):
        selected = [r for r in predictions if r["task"] == task and not r["correct"] and r["control_correct"]][:4]
        for j in range(4):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(selected):
                continue
            row = selected[j]
            sample = samples[row["sample_id"]]
            truth, pred = sample["captions"][row["target"]], sample["captions"][row["prediction"]]
            with Image.open(root / sample["image_path"]) as image:
                ax.imshow(image)
            ax.set_title(task + "\n" + textwrap.fill("True: " + truth, 37) + "\n" +
                         textwrap.fill("Pred: " + pred, 37), fontsize=9)
            example_rows.append({"sample_id": row["sample_id"], "task": task, "true_caption": truth,
                                 "predicted_caption": pred, "control_correct": True})
    fig.savefig(output / "error_examples.png", dpi=140)
    plt.close(fig)
    if example_rows:
        save_csv(output / "error_examples.csv", example_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    labels = {"counting": list("123456"), "spatial": ["left", "right", "above", "below"],
              "negation": ["present", "absent"]}
    for ax, task in zip(axes, labels):
        matrix = metrics[task]["confusion"]
        ax.imshow(matrix, cmap="Blues")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=9,
                        color="white" if value > max(map(max, matrix)) / 2 else "black")
        ticks = list(range(len(matrix)))
        ax.set(xticks=ticks, yticks=ticks, xticklabels=labels[task], yticklabels=labels[task],
               xlabel="Predicted", ylabel="True", title=task)
    fig.savefig(output / "confusions.png", dpi=160)
    plt.close(fig)
    lines = ["# Core probing — OpenAI CLIP ViT-B/32", "", "Bốn bài kiểm tra đã chạy đầy đủ theo "
             "[giao thức định trước](../../docs/probing_protocol.md). "
             "Ba bài đầu dùng hình học tự tạo; fine-grained dùng ảnh CIFAR-100 tự nhiên.", "",
             "| Bài kiểm tra | Số mẫu | Đúng | Accuracy | CI 95% | Ngẫu nhiên |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['num_samples']} | {r['correct']} | {r['accuracy_percent']:.2f}% | "
                     f"{r['ci95_low_percent']:.2f}–{r['ci95_high_percent']:.2f}% | {r['chance_percent']:.2f}% |")
    lines += ["", "![Probe accuracy](accuracy.png)", "",
              "CI được bootstrap 1.000 lần, seed 0, lấy lại mẫu theo nhóm cảnh để giữ quan hệ giữa "
              "các ảnh đối chứng/caption của cùng cảnh. Fine-grained lấy lại mẫu theo ảnh. "
              "Đây là mức biến thiên trong bộ kiểm tra này, không bao quát mọi ảnh hay mọi VLM.", "",
              "## Kiểm tra nhận dạng đối chứng", "",
              "| Bài | Nhận dạng đối chứng đúng | Probe đúng trong số đối chứng đúng |",
              "| --- | ---: | ---: |"]
    for r in rows[:3]:
        conditional = (f"{r['probe_correct_when_control_correct']} / {r['control_correct_samples']} = "
                       f"{r['accuracy_given_control_correct_percent']:.2f}%") if r['control_correct_samples'] else "Không có mẫu"
        lines.append(f"| {r['task']} | {r['control_correct_samples']} / {r['num_samples']} = "
                     f"{r['control_accuracy_percent']:.2f}% | {conditional} |")
    lines += ["", "Counting dùng nhận dạng 9 tổ hợp màu/hình; spatial và negation dùng phân biệt hai mô tả "
              "vật thể không có từ quan hệ/phủ định. Kết quả đối chứng giúp kiểm tra khả năng nhận dạng "
              "cơ bản nhưng không loại hết ảnh hưởng của cách diễn đạt hoặc miền hình học.", ""]
    for task, stats in metrics.items():
        lines += [f"## {task}", "", f"Margin đúng trừ đối thủ tốt nhất trung bình: {stats['mean_signed_margin']:.6f}. "
                  f"Near-tie (≤1e-6): {stats['near_tie_count']}; exact tie: {stats['exact_tie_count']}.", "",
                  "| Điều kiện | Đúng / số mẫu | Accuracy |", "| --- | ---: | ---: |"]
        for condition, part in stats["by_condition"].items():
            lines.append(f"| {condition} | {part['correct']} / {part['num_samples']} | {part['accuracy_percent']:.2f}% |")
        if task == "counting":
            lines += ["", f"Sai số số lượng tuyệt đối trung bình: **{stats['mean_absolute_count_error']:.4f} vật thể**."]
        if task == "spatial":
            lines += ["", f"Nhóm đúng cả bốn vị trí: **{100*stats['all_rows_correct_group_rate']:.2f}%**."]
        if task == "negation":
            lines += ["", f"Tỷ lệ chọn câu khẳng định: **{100*stats['affirmative_choice_rate']:.2f}%**. "
                      f"Nhóm đúng cả present/absent và hai template: **{100*stats['all_rows_correct_group_rate']:.2f}%**.", ""]
            for label, part in stats["by_stratum"].items():
                lines.append(f"- {label}: {part['correct']}/{part['num_samples']} = {part['accuracy_percent']:.2f}%.")
        if task == "fine_grained":
            lines += ["", "Dùng **nhãn nhóm lớn thật (oracle)** để giới hạn còn 5 lớp ứng viên. "
                      "Đây là chẩn đoán lỗi trong nhóm, không phải cải thiện zero-shot 100 lớp. "
                      "Cả 10.000 dự đoán khi vẫn chọn trên toàn bộ 100 lớp khớp baseline cũ từng ảnh."]
        lines.append("")
    lines += ["## Phân bố nhầm lẫn", "", "![Confusion matrices](confusions.png)", "",
              "Hàng là nhãn thật, cột là dự đoán; các ô ghi số mẫu. Với negation, present ứng với "
              "câu khẳng định, absent ứng với câu phủ định.", "",
              "## Ví dụ lỗi", "", "![Ví dụ lỗi có đối chứng nhận dạng đúng](error_examples.png)", "",
              "Chọn bốn lỗi đầu tiên của mỗi bài có đối chứng nhận dạng đúng, theo thứ tự mẫu cố định. "
              "Đây là minh họa hậu nghiệm, không phải tập đánh giá mới hoặc ước lượng tỷ lệ lỗi. "
              "[ID và caption từng ví dụ](error_examples.csv).", "",
              "## Phạm vi diễn giải", "", "Negation có 1.440 mẫu ảnh–caption nhưng chỉ 720 ảnh duy nhất "
              "vì mỗi ảnh dùng hai template. Counting có 1.080 ảnh; spatial có 720 ảnh. "
              "Không gộp accuracy các bài có số lựa chọn khác nhau. "
              "Kết quả một checkpoint không cho phép so sánh điểm yếu giữa nhiều model. "
              "Bộ kiểm tra tự xây không phải bản chạy lại CountBench, ARO hay NegBench.", "",
              "[Chi tiết từng điều kiện](breakdown.csv) · [Tóm tắt CSV](summary.csv) · "
              f"[Mọi cosine score](../../results/{name}/predictions.jsonl) · "
              f"[Metrics và confusion matrices](../../results/{name}/metrics.json) · [Kiểm chứng](inputs.json)", "",
              "```powershell", f"python -m vlm_probe.probing.audit results/{name}",
              f"python -m vlm_probe.report probing --run-name {name}", "```", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "inputs.json", {"run": name, "manifest_sha256": sha256(run / "manifest.json"),
                                        "audit": audit, "report_source_sha256": sha256(Path(__file__))})
    print(f"Probing report: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["baseline", "probing"])
    parser.add_argument("--run-name", default="full_probing_clip_b32_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.kind == "baseline":
        baseline(root)
    else:
        from .io_utils import inside_repo
        inside_repo(root, "results/" + args.run_name)
        probes(root, args.run_name)


if __name__ == "__main__":
    main()
