"""Curated Vietnamese lexicon for CIFAR-100 and synthetic probing tasks (P4.1)."""

from collections import OrderedDict
import unicodedata

# Complete 100-class mapping aligned 1-to-1 with CIFAR-100 dataset_name keys
CIFAR100_VI_LEXICON = OrderedDict([
    ("apple", "quả táo"),
    ("aquarium_fish", "cá cảnh"),
    ("baby", "em bé"),
    ("bear", "con gấu"),
    ("beaver", "con hải ly"),
    ("bed", "cái giường"),
    ("bee", "con ong"),
    ("beetle", "bọ cánh cứng"),
    ("bicycle", "xe đạp"),
    ("bottle", "cái chai"),
    ("bowl", "cái bát"),
    ("boy", "cậu bé"),
    ("bridge", "cây cầu"),
    ("bus", "xe buýt"),
    ("butterfly", "con bướm"),
    ("camel", "con lạc đà"),
    ("can", "lon kim loại"),
    ("castle", "lâu đài"),
    ("caterpillar", "sâu bướm"),
    ("cattle", "bò gia súc"),
    ("chair", "cái ghế"),
    ("chimpanzee", "con tinh tinh"),
    ("clock", "đồng hồ"),
    ("cloud", "đám mây"),
    ("cockroach", "con gián"),
    ("couch", "ghế sofa"),
    ("crab", "con cua"),
    ("crocodile", "con cá sấu"),
    ("cup", "cái cốc"),
    ("dinosaur", "khủng long"),
    ("dolphin", "cá heo"),
    ("elephant", "con voi"),
    ("flatfish", "cá thân bẹt"),
    ("forest", "khu rừng"),
    ("fox", "con cáo"),
    ("girl", "cô bé"),
    ("hamster", "chuột hamster"),
    ("house", "ngôi nhà"),
    ("kangaroo", "chuột túi kangaroo"),
    ("keyboard", "bàn phím"),
    ("lamp", "cây đèn"),
    ("lawn_mower", "máy cắt cỏ"),
    ("leopard", "con báo hoa mai"),
    ("lion", "con sư tử"),
    ("lizard", "con thằn lằn"),
    ("lobster", "tôm hùm"),
    ("man", "người đàn ông"),
    ("maple_tree", "cây phong"),
    ("motorcycle", "xe máy"),
    ("mountain", "ngọn núi"),
    ("mouse", "con chuột"),
    ("mushroom", "cây nấm"),
    ("oak_tree", "cây sồi"),
    ("orange", "quả cam"),
    ("orchid", "hoa phong lan"),
    ("otter", "con rái cá"),
    ("palm_tree", "cây cọ"),
    ("pear", "quả lê"),
    ("pickup_truck", "xe bán tải"),
    ("pine_tree", "cây thông"),
    ("plain", "đồng bằng"),
    ("plate", "cái đĩa"),
    ("poppy", "hoa anh túc"),
    ("porcupine", "con nhím"),
    ("possum", "con thú có túi possum"),
    ("rabbit", "con thỏ"),
    ("raccoon", "con gấu trúc Mỹ"),
    ("ray", "cá đuối"),
    ("road", "con đường"),
    ("rocket", "tên lửa"),
    ("rose", "hoa hồng"),
    ("sea", "biển"),
    ("seal", "con hải cẩu"),
    ("shark", "cá mập"),
    ("shrew", "chuột chù"),
    ("skunk", "con chồn hôi"),
    ("skyscraper", "tòa nhà chọc trời"),
    ("snail", "con ốc sên"),
    ("snake", "con rắn"),
    ("spider", "con nhện"),
    ("squirrel", "con sóc"),
    ("streetcar", "tàu điện"),
    ("sunflower", "hoa hướng dương"),
    ("sweet_pepper", "ớt chuông"),
    ("table", "cái bàn"),
    ("tank", "xe tăng"),
    ("telephone", "điện thoại"),
    ("television", "ti vi"),
    ("tiger", "con hổ"),
    ("tractor", "máy kéo"),
    ("train", "tàu hỏa"),
    ("trout", "cá hồi"),
    ("tulip", "hoa tulip"),
    ("turtle", "con rùa"),
    ("wardrobe", "tủ quần áo"),
    ("whale", "cá voi"),
    ("willow_tree", "cây liễu"),
    ("wolf", "con sói"),
    ("woman", "người phụ nữ"),
    ("worm", "con giun"),
])

PROBING_VI_LEXICON = {
    "colors": {
        "red": "đỏ",
        "blue": "xanh dương",
        "green": "xanh lá",
    },
    "shapes": {
        "circle": "hình tròn",
        "square": "hình vuông",
        "triangle": "hình tam giác",
    },
    "numbers": {
        "one": "một",
        "two": "hai",
        "three": "ba",
        "four": "bốn",
        "five": "năm",
        "six": "sáu",
    },
    "relations": {
        "to the left of": "ở bên trái",
        "to the right of": "ở bên phải",
        "above": "ở phía trên",
        "below": "ở phía dưới",
    },
}


def strip_vietnamese_diacritics(text):
    """Deterministically remove all Vietnamese tone marks and vowel diacritics."""
    replaced = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", replaced)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def build_cifar100_multilingual_prompts(class_names=None):
    """Build the 6 pre-registered language & hybrid control conditions for all 100 CIFAR-100 classes."""
    names = list(class_names) if class_names is not None else list(CIFAR100_VI_LEXICON.keys())
    if len(names) != 100 or set(names) != set(CIFAR100_VI_LEXICON.keys()):
        raise ValueError("Class list does not match the official 100 CIFAR-100 classes.")

    conditions = OrderedDict([
        ("en_baseline", []),
        ("vi_diacritics", []),
        ("vi_no_diacritics", []),
        ("en_frame_vi_label", []),
        ("vi_frame_en_label", []),
        ("vi_en_bilingual", []),
    ])
    class_records = []

    for cid, ds_name in enumerate(names):
        en_label = ds_name.replace("_", " ")
        vi_label = CIFAR100_VI_LEXICON[ds_name]
        vi_no_dia = strip_vietnamese_diacritics(vi_label)

        p_en = f"a photo of a {en_label}."
        p_vi = f"một bức ảnh chụp {vi_label}."
        p_vi_nodia = f"mot buc anh chup {vi_no_dia}."
        p_en_frame_vi = f"a photo of a {vi_label}."
        p_vi_frame_en = f"một bức ảnh chụp {en_label}."
        p_bilingual = f"một bức ảnh chụp {vi_label} ({en_label})."

        conditions["en_baseline"].append(p_en)
        conditions["vi_diacritics"].append(p_vi)
        conditions["vi_no_diacritics"].append(p_vi_nodia)
        conditions["en_frame_vi_label"].append(p_en_frame_vi)
        conditions["vi_frame_en_label"].append(p_vi_frame_en)
        conditions["vi_en_bilingual"].append(p_bilingual)

        class_records.append({
            "class_id": cid,
            "dataset_name": ds_name,
            "en_label": en_label,
            "vi_label": vi_label,
            "vi_no_diacritics_label": vi_no_dia,
            "en_baseline_prompt": p_en,
            "vi_diacritics_prompt": p_vi,
            "vi_no_diacritics_prompt": p_vi_nodia,
            "en_frame_vi_label_prompt": p_en_frame_vi,
            "vi_frame_en_label_prompt": p_vi_frame_en,
            "vi_en_bilingual_prompt": p_bilingual,
        })

    return conditions, class_records


def build_probing_multilingual_captions(sample_row):
    """Build multilingual candidate captions for a synthetic probing sample (counting/spatial/negation)."""
    task = sample_row["task"]
    c_map = PROBING_VI_LEXICON["colors"]
    s_map = PROBING_VI_LEXICON["shapes"]
    n_map = PROBING_VI_LEXICON["numbers"]
    r_map = PROBING_VI_LEXICON["relations"]

    en_caps = list(sample_row["captions"])

    if task == "counting":
        color_en = sample_row["color"]
        shape_en = sample_row["shape"]
        vi_c = c_map[color_en]
        vi_s = s_map[shape_en]
        words_en = ("one", "two", "three", "four", "five", "six")
        vi_caps = [
            f"một bức vẽ có {n_map[w]} {vi_s} màu {vi_c}."
            for w in words_en
        ]
        bilingual_caps = [
            f"một bức vẽ có {n_map[w]} {vi_s} màu {vi_c} ({w} {color_en} {shape_en if i == 0 else shape_en + 's'})."
            for i, w in enumerate(words_en)
        ]
    elif task == "spatial":
        obj_a, obj_b = sample_row["objects"]
        ca, sa = obj_a["color"], obj_a["shape"]
        cb, sb = obj_b["color"], obj_b["shape"]
        vi_a = f"{s_map[sa]} màu {c_map[ca]}"
        vi_b = f"{s_map[sb]} màu {c_map[cb]}"
        en_a = f"{ca} {sa}"
        en_b = f"{cb} {sb}"
        rels_en = ("to the left of", "to the right of", "above", "below")
        vi_caps = [
            f"một bức vẽ có một {vi_a} {r_map[r]} một {vi_b}."
            for r in rels_en
        ]
        bilingual_caps = [
            f"một bức vẽ có một {vi_a} {r_map[r]} một {vi_b} ({en_a} {r} {en_b})."
            for r in rels_en
        ]
    elif task == "negation":
        ca = sample_row["target_color"]
        sa = sample_row["target_shape"]
        vi_target = f"{s_map[sa]} màu {c_map[ca]}"
        en_target = f"{ca} {sa}"
        if sample_row["condition"] == "with_without":
            vi_caps = [
                f"một bức vẽ có một {vi_target}.",
                f"một bức vẽ không có một {vi_target}.",
            ]
            bilingual_caps = [
                f"một bức vẽ có một {vi_target} (with a {en_target}).",
                f"một bức vẽ không có một {vi_target} (without a {en_target}).",
            ]
        else:
            vi_caps = [
                f"một bức vẽ chứa một {vi_target}.",
                f"một bức vẽ không chứa một {vi_target}.",
            ]
            bilingual_caps = [
                f"một bức vẽ chứa một {vi_target} (contains a {en_target}).",
                f"một bức vẽ không chứa một {vi_target} (does not contain a {en_target}).",
            ]
    else:
        raise ValueError(f"Unsupported probing task for Vietnamese translation: {task}")

    vi_nodia_caps = [strip_vietnamese_diacritics(c) for c in vi_caps]
    return OrderedDict([
        ("en_baseline", en_caps),
        ("vi_diacritics", vi_caps),
        ("vi_no_diacritics", vi_nodia_caps),
        ("vi_en_bilingual", bilingual_caps),
    ])

