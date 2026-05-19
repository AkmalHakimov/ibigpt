#!/usr/bin/env python3
"""
split_dataset.py
Shuffles bukhara_sft_dataset.jsonl, applies a final quality pass,
then writes train/validation/test splits and experimental subsets.
"""

import json
import re
import random
import sys
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE   = "bukhara_sft_dataset.jsonl"
SEED         = 42
TRAIN_RATIO  = 0.90
VAL_RATIO    = 0.05
# test gets the remainder

SUBSETS = {
    "train_5k.jsonl":       ("train", 5_000),
    "train_10k.jsonl":      ("train", 10_000),
    "train_30k.jsonl":      ("train", 30_000),
    "validation_1k.jsonl":  ("val",   1_000),
    "test_1k.jsonl":        ("test",  1_000),
}

REPORT_FILE = "dataset_split_report.txt"

MIN_ASST_CHARS = 12

# ─── Quality-filter patterns ─────────────────────────────────────────────────
RE_URL      = re.compile(r'https?://\S+|www\.\S+|t\.me/\S+', re.I)
RE_USERNAME = re.compile(r'@\w+')
RE_EMAIL    = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.I)
RE_PHONE    = re.compile(r'\+?\d[\d\s\-\(\)]{7,}\d')

RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    flags=re.UNICODE,
)

LOW_VALUE_EXACT = {
    'рахмат', 'rahmat', 'спасибо', 'ок', 'ok', 'да', 'нет', 'yes', 'no',
    'ха', 'хаха', 'lol', '👍', '😂', '😁', '++', '+', 'хоп', 'хорошо',
    'ладно', 'понял', 'понятно', 'salom', 'салом', 'hi', 'hello',
    'взаимно', 'rahmat)', 'рахмат)', 'ok)', 'ок)',
}


def has_pii(text: str) -> bool:
    return bool(
        RE_URL.search(text) or
        RE_USERNAME.search(text) or
        RE_EMAIL.search(text) or
        RE_PHONE.search(text)
    )


def is_mostly_emoji(text: str) -> bool:
    """True if >60% of non-space chars are emoji/symbols/punctuation."""
    clean = RE_EMOJI.sub('', text).strip()
    clean = re.sub(r'[\W_]+', '', clean, flags=re.UNICODE)
    original_alnum = re.sub(r'[\W_]+', '', text, flags=re.UNICODE)
    if not original_alnum:
        return True
    ratio = len(clean) / len(original_alnum)
    return ratio < 0.40


def is_low_value_reply(text: str) -> bool:
    t = text.strip().lower().rstrip('.,!?')
    return t in LOW_VALUE_EXACT


def quality_ok(example: dict) -> tuple[bool, str]:
    msgs = example['messages']
    user_text = next((m['content'] for m in msgs if m['role'] == 'user'), '')
    asst_text = next((m['content'] for m in msgs if m['role'] == 'assistant'), '')

    if user_text.strip() == asst_text.strip():
        return False, "identical_pair"
    if len(asst_text.strip()) < MIN_ASST_CHARS:
        return False, "asst_too_short"
    if is_mostly_emoji(asst_text):
        return False, "asst_mostly_emoji"
    if is_low_value_reply(asst_text):
        return False, "asst_low_value"
    if has_pii(user_text) or has_pii(asst_text):
        return False, "contains_pii"
    return True, "ok"


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"ERROR: '{INPUT_FILE}' not found.")
        sys.exit(1)

    print(f"Loading {INPUT_FILE} …")
    raw_examples = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_examples.append(json.loads(line))

    total_loaded = len(raw_examples)
    print(f"  Loaded: {total_loaded:,} examples")

    # ── Quality filter ────────────────────────────────────────────────────────
    print("Applying quality filter …")
    reject_counts: dict[str, int] = {}
    clean_examples = []
    for ex in raw_examples:
        ok, reason = quality_ok(ex)
        if ok:
            clean_examples.append(ex)
        else:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    total_clean = len(clean_examples)
    total_removed = total_loaded - total_clean
    print(f"  Removed {total_removed:,} low-quality examples")
    for reason, cnt in sorted(reject_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:25s}: {cnt:,}")
    print(f"  Clean examples: {total_clean:,}")

    # ── Shuffle ───────────────────────────────────────────────────────────────
    print(f"Shuffling with seed={SEED} …")
    random.seed(SEED)
    random.shuffle(clean_examples)

    # ── Split ─────────────────────────────────────────────────────────────────
    n_train = int(total_clean * TRAIN_RATIO)
    n_val   = int(total_clean * VAL_RATIO)
    n_test  = total_clean - n_train - n_val

    train_data = clean_examples[:n_train]
    val_data   = clean_examples[n_train:n_train + n_val]
    test_data  = clean_examples[n_train + n_val:]

    splits = {
        "train.jsonl":      train_data,
        "validation.jsonl": val_data,
        "test.jsonl":       test_data,
    }

    print("Writing splits …")
    for fname, data in splits.items():
        write_jsonl(Path(fname), data)
        print(f"  {fname:25s}: {len(data):,}")

    # ── Subsets ───────────────────────────────────────────────────────────────
    split_map = {"train": train_data, "val": val_data, "test": test_data}

    print("Writing subsets …")
    subset_counts: dict[str, int] = {}
    for fname, (split_name, n) in SUBSETS.items():
        source = split_map[split_name]
        subset = source[:n]
        write_jsonl(Path(fname), subset)
        subset_counts[fname] = len(subset)
        print(f"  {fname:25s}: {len(subset):,}")

    # ── Report ────────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 52,
        "DATASET SPLIT REPORT",
        "=" * 52,
        f"Input file          : {INPUT_FILE}",
        f"Random seed         : {SEED}",
        f"Min assistant chars : {MIN_ASST_CHARS}",
        "",
        "── Loading ──────────────────────────────────────",
        f"Total loaded        : {total_loaded:,}",
        "",
        "── Quality filter ───────────────────────────────",
        f"Total removed       : {total_removed:,}",
    ]
    for reason, cnt in sorted(reject_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {reason:25s}: {cnt:,}")
    report_lines += [
        f"Clean examples      : {total_clean:,}",
        "",
        "── Splits ───────────────────────────────────────",
        f"train.jsonl         : {len(train_data):,}  ({len(train_data)/total_clean*100:.1f}%)",
        f"validation.jsonl    : {len(val_data):,}   ({len(val_data)/total_clean*100:.1f}%)",
        f"test.jsonl          : {len(test_data):,}   ({len(test_data)/total_clean*100:.1f}%)",
        "",
        "── Experimental subsets ─────────────────────────",
    ]
    for fname, cnt in subset_counts.items():
        report_lines.append(f"  {fname:25s}: {cnt:,}")
    report_lines += ["", "=" * 52]

    report_text = '\n'.join(report_lines)
    Path(REPORT_FILE).write_text(report_text, encoding='utf-8')
    print()
    print(report_text)


if __name__ == '__main__':
    main()
