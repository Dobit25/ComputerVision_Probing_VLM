"""Plot all 100 measured class accuracies, grouped by official CIFAR-100 taxonomy.

Run from any directory with the project's conda gen Python environment.
"""
from pathlib import Path
import csv
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/full_cifar100_category_v1"
OUT = Path(__file__).resolve().parent
GROUP_TITLES = [
    "Động vật có vú dưới nước", "Cá", "Hoa", "Vật chứa thực phẩm",
    "Trái cây và rau củ", "Thiết bị điện gia dụng", "Đồ nội thất", "Côn trùng",
    "Động vật ăn thịt lớn", "Công trình và vật thể nhân tạo lớn",
    "Cảnh quan tự nhiên rộng lớn", "Thú lớn ăn tạp và ăn cỏ",
    "Động vật có vú cỡ trung", "Động vật không xương sống khác",
    "Người", "Bò sát", "Động vật có vú nhỏ", "Cây", "Phương tiện — nhóm 1",
    "Phương tiện — nhóm 2",
]


def main():
    with (RUN / "per_class.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    classes = json.loads((RUN / "classes.json").read_text(encoding="utf-8"))
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    by_id = {int(row["class_id"]): row for row in rows}
    assert len(rows) == len(by_id) == len(classes) == 100
    assert all(int(row["support"]) == 100 for row in rows)
    assert sum(int(r["baseline_correct"]) for r in rows) == metrics["baseline"]["correct"]
    assert sum(int(r["category_correct"]) for r in rows) == metrics["category"]["correct"]
    baseline_total = metrics["baseline"]["top1_accuracy_percent"]
    category_total = metrics["category"]["top1_accuracy_percent"]
    delta = category_total - baseline_total
    improved = sum(float(r["delta_percentage_points"]) > 0 for r in rows)
    declined = sum(float(r["delta_percentage_points"]) < 0 for r in rows)
    unchanged = len(rows) - improved - declined

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    blue, orange = "#2766A0", "#E68A32"
    fig, axes = plt.subplots(5, 4, figsize=(24, 22), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.865, bottom=0.075,
                        hspace=0.73, wspace=0.18)
    fig.suptitle("CIFAR-100 · Độ chính xác theo từng lớp", x=0.055, y=0.98,
                 ha="left", fontsize=27, fontweight="bold", color="#182B3A")
    fig.text(0.055, 0.953,
             "OpenAI CLIP ViT-B/32 · 10.000 ảnh test · 100 ảnh/lớp · Giữ nguyên trọng số mô hình",
             fontsize=14, color="#465260")
    vi = lambda value: f"{value:.2f}".replace(".", ",")
    fig.text(0.055, 0.925,
             f"Toàn tập: {vi(baseline_total)}% → {vi(category_total)}%   |   "
             f"Chênh lệch: {vi(delta)} điểm phần trăm   |   "
             f"{improved} lớp tăng · {declined} lớp giảm · {unchanged} lớp không đổi",
             fontsize=15, fontweight="bold", color="#182B3A")
    fig.legend(handles=[Patch(facecolor=blue, label='Chỉ class: "a photo of a {class}."'),
                        Patch(facecolor=orange, label='Class + category: "a photo of a {class}, a type of {category}."')],
               loc="upper left", bbox_to_anchor=(0.05, 0.909), frameon=False,
               ncol=2, fontsize=12, columnspacing=3)

    for group, ax in enumerate(axes.flat):
        members = sorted((c for c in classes if c["coarse_id"] == group),
                         key=lambda c: c["class_id"])
        assert len(members) == 5
        records = [by_id[c["class_id"]] for c in members]
        assert all(r["coarse_name"] == c["coarse_name"] and r["class_name"] == c["class_name"]
                   for r, c in zip(records, members))
        a = np.array([float(r["baseline_accuracy_percent"]) for r in records])
        b = np.array([float(r["category_accuracy_percent"]) for r in records])
        x = np.arange(5)
        bars_a = ax.bar(x - 0.19, a, width=0.36, color=blue, zorder=3)
        bars_b = ax.bar(x + 0.19, b, width=0.36, color=orange, zorder=3)
        ax.bar_label(bars_a, labels=[f"{v:.0f}" for v in a], padding=3, fontsize=10, color=blue)
        ax.bar_label(bars_b, labels=[f"{v:.0f}" for v in b], padding=3, fontsize=10, color="#9E4C00")
        ax.set_ylim(0, 113)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#DEE4EA", linewidth=0.7, zorder=0)
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(x, [textwrap.fill(c["class_name"].replace("_", " "), width=13)
                           for c in members], fontsize=10)
        ax.set_title(f"{group+1:02d}. {GROUP_TITLES[group]}", fontsize=12,
                     loc="left", fontweight="bold", color="#182B3A", pad=21)
        category_text = " / ".join(dict.fromkeys(c["category_text"] for c in members))
        ax.text(0, 1.025, "category: " + category_text, transform=ax.transAxes,
                fontsize=9.5, color="#637181", va="bottom")
        if group % 4 == 0:
            ax.set_ylabel("Độ chính xác (%)", fontsize=11, labelpad=9)

    fig.text(0.055, 0.039,
             "Mỗi cặp cột là một lớp; số trên cột là accuracy (%). Hai điều kiện dùng cùng ảnh và luôn chọn trong đủ 100 lớp.",
             fontsize=12, color="#465260")
    fig.text(0.055, 0.022,
             "20 nhóm chỉ dùng để trình bày; không lọc ứng viên theo nhãn nhóm thật. Nguồn: results/full_cifar100_category_v1/per_class.csv",
             fontsize=11, color="#637181")
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"per_class_accuracy_comparison.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    provenance = {
        "plot": "per_class_accuracy_comparison",
        "source_files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [RUN / "per_class.csv", RUN / "classes.json", RUN / "metrics.json"]},
        "class_count": 100, "group_count": 20, "samples_per_class": 100,
        "baseline_accuracy_percent": baseline_total,
        "category_accuracy_percent": category_total,
        "delta_percentage_points": delta,
    }
    (OUT / "per_class_accuracy_comparison.sources.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Created PNG (4800 x 4400), vector PDF and source checksums.")


if __name__ == "__main__":
    main()
