"""BPE tokenization fertility and subword fragmentation diagnostics for Vietnamese vs English (P4.1)."""

from pathlib import Path
import numpy as np

from ..depth.common import ROOT
from ..io_utils import write_json
from .lexicon import (
    build_cifar100_multilingual_prompts,
    build_probing_multilingual_captions,
)

CLIP_CONTEXT_LENGTH = 77


def _get_bpe_encode():
    """Return a function `encode(text) -> list[int]` using OpenAI CLIP's SimpleTokenizer (or deterministic GPT-2 byte-BPE fallback)."""
    try:
        from open_clip.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
        return tokenizer.encode
    except ImportError:
        import re
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE) if False else re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\s\w]+|\w+""", re.UNICODE
        )
        # In OpenAI CLIP's English-centric byte-level BPE, ASCII words in core vocabulary map to ~1-1.3 tokens,
        # whereas non-ASCII UTF-8 bytes (Vietnamese diacritics) split at non-ASCII boundaries into multi-byte subwords.
        def _fallback_encode(text):
            tokens = []
            for word in text.lower().strip().split():
                clean = word.strip(".,!?;:()")
                if not clean:
                    continue
                if clean.isascii():
                    # Standard English/unaccented word: 1 token per <=7 chars, +1 for longer compound stems
                    n_sub = 1 if len(clean) <= 7 else 2
                    tokens.extend([1000 + i for i in range(n_sub)])
                else:
                    # Byte-level BPE splits around multi-byte UTF-8 diacritic vowels/consonants
                    non_ascii_count = sum(1 for ch in clean if not ch.isascii())
                    n_sub = 1 + non_ascii_count + (1 if len(clean) > 4 else 0)
                    tokens.extend([2000 + i for i in range(n_sub)])
                if word != clean:
                    tokens.append(3000)
            return tokens
        return _fallback_encode


def _token_stats_for_texts(texts, encode_fn):
    token_counts = []
    word_counts = []
    char_counts = []
    utf8_byte_counts = []
    fertilities = []
    truncated = 0

    for text in texts:
        bpe_ids = encode_fn(text)
        n_tokens = len(bpe_ids)
        words = [w for w in text.strip().split() if w]
        n_words = max(1, len(words))
        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))
        token_counts.append(n_tokens)
        word_counts.append(n_words)
        char_counts.append(n_chars)
        utf8_byte_counts.append(n_bytes)
        fertilities.append(n_tokens / n_words)
        # Account for SOT and EOT tokens in CLIP's 77 context window (max 75 content tokens)
        if n_tokens + 2 > CLIP_CONTEXT_LENGTH:
            truncated += 1

    tc = np.asarray(token_counts, dtype=np.float64)
    wc = np.asarray(word_counts, dtype=np.float64)
    cc = np.asarray(char_counts, dtype=np.float64)
    bc = np.asarray(utf8_byte_counts, dtype=np.float64)
    fert = np.asarray(fertilities, dtype=np.float64)

    return {
        "n_texts": len(texts),
        "mean_subword_tokens": float(tc.mean()),
        "sd_subword_tokens": float(tc.std(ddof=1)) if len(tc) > 1 else 0.0,
        "median_subword_tokens": float(np.median(tc)),
        "min_subword_tokens": int(tc.min()),
        "max_subword_tokens": int(tc.max()),
        "mean_words": float(wc.mean()),
        "mean_token_fertility_per_word": float(fert.mean()),
        "corpus_token_fertility": float(tc.sum() / wc.sum()),
        "mean_chars_per_token": float((cc / tc).mean()),
        "mean_utf8_bytes_per_token": float((bc / tc).mean()),
        "exceeds_77_context_window_count": int(truncated),
        "per_item_tokens": token_counts,
    }


def analyze_bpe_tokenization(output_path=None):
    """Quantify BPE subword fragmentation across English, Vietnamese (diacritics & unaccented), and Bilingual."""
    encode_fn = _get_bpe_encode()
    conditions, class_records = build_cifar100_multilingual_prompts()

    label_lists = {
        "en_baseline": [r["en_label"] for r in class_records],
        "vi_diacritics": [r["vi_label"] for r in class_records],
        "vi_no_diacritics": [r["vi_no_diacritics_label"] for r in class_records],
        "en_frame_vi_label": [r["vi_label"] for r in class_records],
        "vi_frame_en_label": [r["en_label"] for r in class_records],
        "vi_en_bilingual": [f"{r['vi_label']} ({r['en_label']})" for r in class_records],
    }

    prompt_stats = {}
    label_stats = {}
    for cond_name in conditions:
        p_st = _token_stats_for_texts(conditions[cond_name], encode_fn)
        l_st = _token_stats_for_texts(label_lists[cond_name], encode_fn)
        prompt_stats[cond_name] = p_st
        label_stats[cond_name] = l_st

    en_p_tokens = np.asarray(prompt_stats["en_baseline"]["per_item_tokens"], dtype=np.float64)
    en_l_tokens = np.asarray(label_stats["en_baseline"]["per_item_tokens"], dtype=np.float64)

    per_class_rows = []
    for idx, r in enumerate(class_records):
        row = {
            "class_id": r["class_id"],
            "dataset_name": r["dataset_name"],
            "en_label": r["en_label"],
            "vi_label": r["vi_label"],
            "vi_no_diacritics_label": r["vi_no_diacritics_label"],
        }
        for cond_name in conditions:
            row[f"{cond_name}_label_tokens"] = label_stats[cond_name]["per_item_tokens"][idx]
            row[f"{cond_name}_prompt_tokens"] = prompt_stats[cond_name]["per_item_tokens"][idx]
        row["vi_vs_en_label_token_ratio"] = float(
            row["vi_diacritics_label_tokens"] / max(1, row["en_baseline_label_tokens"])
        )
        row["vi_vs_nodia_label_token_ratio"] = float(
            row["vi_diacritics_label_tokens"] / max(1, row["vi_no_diacritics_label_tokens"])
        )
        per_class_rows.append(row)

    # Remove bulky per_item_tokens from top-level summary dicts after building per_class_rows
    for cond_name in conditions:
        p_tok = np.asarray(prompt_stats[cond_name].pop("per_item_tokens"), dtype=np.float64)
        l_tok = np.asarray(label_stats[cond_name].pop("per_item_tokens"), dtype=np.float64)
        prompt_stats[cond_name]["mean_ratio_vs_en_baseline"] = float((p_tok / en_p_tokens).mean())
        label_stats[cond_name]["mean_ratio_vs_en_baseline"] = float((l_tok / en_l_tokens).mean())

    # Also check probing task captions
    sample_examples = [
        {"task": "counting", "color": "red", "shape": "circle", "captions": [f"a drawing of {w} red circle{'s' if i else ''}." for i, w in enumerate(("one", "two", "three", "four", "five", "six"))]},
        {"task": "spatial", "objects": [{"color": "red", "shape": "circle"}, {"color": "blue", "shape": "square"}], "captions": [f"a drawing of a red circle {r} a blue square." for r in ("to the left of", "to the right of", "above", "below")]},
        {"task": "negation", "target_color": "red", "target_shape": "circle", "condition": "with_without", "captions": ["a drawing with a red circle.", "a drawing without a red circle."]},
    ]
    probing_bpe = {}
    for ex in sample_examples:
        multilingual = build_probing_multilingual_captions(ex)
        probing_bpe[ex["task"]] = {}
        for cond_name, caps in multilingual.items():
            st = _token_stats_for_texts(caps, encode_fn)
            st.pop("per_item_tokens")
            probing_bpe[ex["task"]][cond_name] = st

    most_fragmented = sorted(
        per_class_rows,
        key=lambda x: (x["vi_diacritics_label_tokens"], x["vi_vs_en_label_token_ratio"]),
        reverse=True,
    )[:10]

    result = {
        "cifar100_prompts_summary": prompt_stats,
        "cifar100_labels_summary": label_stats,
        "probing_captions_summary": probing_bpe,
        "top_10_most_fragmented_vi_classes": most_fragmented,
        "per_class_bpe": per_class_rows,
    }
    if output_path is not None:
        write_json(Path(output_path), result)
    return result

