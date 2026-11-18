"""Deterministic scene construction. No model scores influence this dataset."""

import hashlib
import json
import math
import random
from itertools import permutations, product

COLORS = {"red": (220, 45, 45), "blue": (40, 90, 215), "green": (35, 155, 70)}
SHAPES = ("circle", "square", "triangle")
NUMBERS = ("one", "two", "three", "four", "five", "six")
RELATIONS = ("to the left of", "to the right of", "above", "below")


def ordered_samples(rows, tasks, limit_per_task=None):
    """Persist inputs in the same task order used for inference and audit."""
    return [row for task in tasks for row in [r for r in rows if r["task"] == task][:limit_per_task]]


def rng_for(seed, key):
    value = int.from_bytes(hashlib.sha256(f"{seed}/{key}".encode()).digest()[:8], "big")
    return random.Random(value)


def obj(color, shape, x, y, radius):
    return {"color": color, "shape": shape, "x": x, "y": y, "radius": radius}


def pair_positions(rng, relation):
    dx, dy = rng.uniform(-5, 5), rng.uniform(-5, 5)
    choices = [((56 + dx, 112 + dy), (168 + dx, 112 + dy)),
               ((168 + dx, 112 + dy), (56 + dx, 112 + dy)),
               ((112 + dx, 56 + dy), (112 + dx, 168 + dy)),
               ((112 + dx, 168 + dy), (112 + dx, 56 + dy))]
    return choices[relation]


def scenes(seed=14):
    rows = []
    control_nine = [f"a drawing of {color} {shape}s." for color in COLORS for shape in SHAPES]
    for color, shape, size_mode, layout in product(COLORS, SHAPES, ("fixed_radius", "fixed_total_area"), range(10)):
        group = f"counting/{color}/{shape}/{size_mode}/{layout}"
        rng = rng_for(seed, group)
        grid = [(x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)) for y in (62, 162) for x in (42, 112, 182)]
        rng.shuffle(grid)
        for number in range(1, 7):
            radius = 14 if size_mode == "fixed_radius" else 32 / math.sqrt(number)
            objects = [obj(color, shape, x, y, radius) for x, y in grid[:number]]
            captions = [f"a drawing of {word} {color} {shape}{'' if i == 0 else 's'}." for i, word in enumerate(NUMBERS)]
            rows.append({"sample_id": f"{group}/{number}", "group_id": group, "task": "counting",
                         "condition": size_mode, "stratum": str(number), "target": number - 1,
                         "captions": captions, "objects": objects, "background": 245 if layout % 2 else 255,
                         "control_captions": control_nine,
                         "control_target": list(COLORS).index(color) * 3 + SHAPES.index(shape),
                         "color": color, "shape": shape, "layout": layout, "number": number})

    pairs = list(product(permutations(COLORS, 2), permutations(SHAPES, 2)))
    for pair_index, ((ca, cb), (sa, sb)) in enumerate(pairs):
        third_color = next(c for c in COLORS if c not in (ca, cb))
        phrase = f"a {ca} {sa} and a {cb} {sb}"
        control = [f"a drawing of {phrase}.", f"a drawing of a {third_color} {sa} and a {cb} {sb}."]
        for layout in range(5):
            group = f"spatial/{pair_index}/{layout}"
            for relation in range(4):
                rng = rng_for(seed, group)
                (ax, ay), (bx, by) = pair_positions(rng, relation)
                radius = rng.uniform(18, 22)
                reverse = (pair_index + layout + relation) % 2
                rows.append({"sample_id": f"{group}/{relation}", "group_id": group, "task": "spatial",
                             "condition": RELATIONS[relation], "stratum": RELATIONS[relation], "target": relation,
                             "captions": [f"a drawing of a {ca} {sa} {r} a {cb} {sb}." for r in RELATIONS],
                             "objects": [obj(ca, sa, ax, ay, radius), obj(cb, sb, bx, by, radius)],
                             "background": 245 if layout % 2 else 255,
                             "control_captions": control[::-1] if reverse else control, "control_target": reverse,
                             "relation": relation, "pair_index": pair_index, "layout": layout})

        for layout in range(10):
            group = f"negation/{pair_index}/{layout}"
            rng = rng_for(seed, group)
            (ax, ay), (bx, by) = pair_positions(rng, layout % 4)
            radius = rng.uniform(18, 22)
            for present in (True, False):
                actual_color = ca if present else third_color
                objects = [obj(actual_color, sa, ax, ay, radius), obj(cb, sb, bx, by, radius)]
                for template in ("with_without", "contains_does_not"):
                    captions = ([f"a drawing with a {ca} {sa}.", f"a drawing without a {ca} {sa}."]
                                if template == "with_without" else
                                [f"a drawing that contains a {ca} {sa}.", f"a drawing that does not contain a {ca} {sa}."])
                    rows.append({"sample_id": f"{group}/{int(present)}/{template}", "group_id": group,
                                 "task": "negation", "condition": template, "stratum": "present" if present else "absent",
                                 "target": 0 if present else 1, "captions": captions, "objects": objects,
                                 "background": 245 if layout % 2 else 255,
                                 "control_captions": control, "control_target": 0 if present else 1,
                                 "present": present, "target_color": ca, "target_shape": sa,
                                 "pair_index": pair_index, "layout": layout})
    validate_scenes(rows)
    return rows


def validate_scenes(rows):
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate probe IDs.")
    for row in rows:
        objects = row["objects"]
        for item in objects:
            x, y, radius = item["x"], item["y"], item["radius"]
            if not (radius <= x <= 224 - radius and radius <= y <= 224 - radius):
                raise ValueError("Object clipped by canvas.")
        for a, b in permutations(objects, 2):
            # Non-overlapping bounding boxes imply no shape overlap.
            if abs(a["x"] - b["x"]) < a["radius"] + b["radius"] and abs(a["y"] - b["y"]) < a["radius"] + b["radius"]:
                raise ValueError("Overlapping objects.")
        if row["task"] == "counting" and len(objects) != row["target"] + 1:
            raise ValueError("Count label mismatch.")
        if row["task"] == "spatial":
            a, b = objects
            relation = row["target"]
            if not (a["x"] < b["x"], a["x"] > b["x"], a["y"] < b["y"], a["y"] > b["y"])[relation]:
                raise ValueError("Spatial label mismatch.")
        if row["task"] == "negation":
            present = any(o["color"] == row["target_color"] and o["shape"] == row["target_shape"] for o in objects)
            if present != row["present"] or row["target"] != int(not present):
                raise ValueError("Negation label mismatch.")


def render(row):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (896, 896), (row["background"],) * 3)
    draw = ImageDraw.Draw(image)
    for item in row["objects"]:
        x, y, radius = (4 * item[k] for k in ("x", "y", "radius"))
        box = (x - radius, y - radius, x + radius, y + radius)
        color = COLORS[item["color"]]
        if item["shape"] == "circle":
            draw.ellipse(box, fill=color)
        elif item["shape"] == "square":
            draw.rectangle(box, fill=color)
        elif item["shape"] == "triangle":
            draw.polygon([(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)], fill=color)
        else:
            raise ValueError("Unknown shape.")
    return image.resize((224, 224), Image.Resampling.LANCZOS)


def materialize(root, seed=14):
    """Hash-based image names deduplicate images reused by caption templates."""
    from ..io_utils import sha256, write_json

    rows = scenes(seed)
    folder = root / "data/probing_v1"
    folder.mkdir(parents=True, exist_ok=True)
    for row in rows:
        scene = {"objects": row["objects"], "background": row["background"]}
        identity = hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()
        path = folder / f"{identity}.png"
        image = render(row)
        # Always render anew: source bytes cannot silently reuse a modified local PNG.
        image.save(path)
        row["image_path"] = path.relative_to(root).as_posix()
        row["image_sha256"] = sha256(path)
    write_json(folder / "scenes.json", rows)
    return rows
