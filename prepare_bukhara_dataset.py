#!/usr/bin/env python3
"""
prepare_bukhara_dataset.py
Converts Telegram group export (result.json) into a supervised fine-tuning
dataset in JSONL chat format for Bukharian Uzbek/Tajik/Russian mixed style.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

# ─── Settings ────────────────────────────────────────────────────────────────
MIN_CHARS    = 8
MAX_CHARS    = 700
MAX_EXAMPLES = None          # set to int to cap output
OUTPUT_FILE  = "bukhara_sft_dataset.jsonl"
PREVIEW_FILE = "preview_bukhara_examples.txt"
INPUT_FILE   = "dataset/result.json"
PREVIEW_N    = 50

SYSTEM_PROMPT = (
    "You are a helpful assistant that responds in natural Bukharian "
    "Uzbek/Tajik mixed conversational style. Keep the tone casual, local, and natural."
)

# ─── Regex patterns ──────────────────────────────────────────────────────────
RE_PHONE    = re.compile(r'\+?\d[\d\s\-\(\)]{7,}\d')
RE_URL      = re.compile(r'https?://\S+|www\.\S+|t\.me/\S+', re.I)
RE_USERNAME = re.compile(r'@\w+')
RE_EMAIL    = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.I)

# Low-value single-token replies (exact match after stripping)
LOW_VALUE = {
    '++', '+', 'ок', 'ok', 'да', 'нет', 'yes', 'no', 'хоп', 'хоп)',
    'хорошо', 'ладно', 'понял', 'понятно', 'спасибо', 'рахмат',
    'tashaккur', 'rahmat', 'salom', 'салом', 'hi', 'hello', 'ok.',
    'ok!', 'ок.', 'ок!', 'ха', 'хаха', 'ха ха', 'lol', 'lmao',
}

# Emoji-only / symbol-only detection
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
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)


def extract_text(raw) -> str:
    """Return plain text from a Telegram message text field (str or list)."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get('text', ''))
        return ''.join(parts)
    return ''


def clean_text(text: str) -> str:
    """Remove PII and normalise whitespace; preserve dialect."""
    text = RE_URL.sub('', text)
    text = RE_USERNAME.sub('', text)
    text = RE_EMAIL.sub('', text)
    text = RE_PHONE.sub('', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_emoji_only(text: str) -> bool:
    """True if the text contains nothing but emoji/symbols/spaces."""
    stripped = RE_EMOJI.sub('', text).strip()
    # also strip punctuation
    stripped = re.sub(r'[\W_]+', '', stripped, flags=re.UNICODE)
    return len(stripped) == 0


def is_low_value(text: str) -> bool:
    """True if message is a short, near-worthless reply."""
    t = text.strip().lower()
    if t in LOW_VALUE:
        return True
    # pure digit string or very short all-punctuation
    if re.fullmatch(r'[\d\s\W]+', t):
        return True
    return False


def is_valid(text: str) -> bool:
    """Return True if the text is worth keeping."""
    if not text:
        return False
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False
    if is_emoji_only(text):
        return False
    if is_low_value(text):
        return False
    return True


def build_example(user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"ERROR: '{INPUT_FILE}' not found. "
              f"Place your Telegram export file at: {input_path.resolve()}")
        sys.exit(1)

    print(f"Loading {input_path} …")
    try:
        with open(input_path, encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed — {e}")
        print("The Telegram export file may be incomplete or corrupted.")
        print("Try re-exporting from Telegram Desktop.")
        sys.exit(1)

    messages_raw = data.get('messages', [])
    total_messages = len(messages_raw)
    print(f"  Total messages in file: {total_messages:,}")

    # ── Step 1: extract text from all 'message'-type entries ─────────────────
    id_to_text: dict[int, str] = {}   # message id → cleaned text
    msgs_with_text = 0

    for msg in messages_raw:
        if msg.get('type') != 'message':
            continue
        raw = msg.get('text', '')
        text = extract_text(raw)
        if not text.strip():
            continue
        msgs_with_text += 1
        cleaned = clean_text(text)
        if cleaned:
            id_to_text[msg['id']] = cleaned

    print(f"  Messages with text:     {msgs_with_text:,}")
    print(f"  Messages after cleaning:{len(id_to_text):,}")

    # ── Step 2: build reply pairs ────────────────────────────────────────────
    pairs_created = 0
    seen_pairs: set[tuple[str, str]] = set()
    examples = []

    for msg in messages_raw:
        if msg.get('type') != 'message':
            continue
        reply_id = msg.get('reply_to_message_id')
        if reply_id is None:
            continue

        user_text = id_to_text.get(reply_id, '')
        assistant_text = id_to_text.get(msg['id'], '')

        if not is_valid(user_text) or not is_valid(assistant_text):
            continue

        pairs_created += 1
        key = (user_text, assistant_text)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        examples.append(build_example(user_text, assistant_text))

    duplicates_removed = pairs_created - len(examples)
    print(f"  Reply-pair examples created:  {pairs_created:,}")
    print(f"  Duplicate pairs removed:      {duplicates_removed:,}")

    if not examples:
        print("\nWARNING: No examples were created.")
        print("Possible reasons:")
        print("  • Messages lack reply_to_message_id (not a reply-based group)")
        print("  • All reply pairs were filtered out by quality checks")
        print("  • The file may not be a standard Telegram chat export")
        sys.exit(0)

    if MAX_EXAMPLES is not None:
        examples = examples[:MAX_EXAMPLES]

    final_count = len(examples)
    print(f"  Final examples saved:         {final_count:,}")

    # ── Step 3: write JSONL ───────────────────────────────────────────────────
    out_path = Path(OUTPUT_FILE)
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # ── Step 4: write preview ─────────────────────────────────────────────────
    preview_path = Path(PREVIEW_FILE)
    preview_examples = examples[:PREVIEW_N]
    with open(preview_path, 'w', encoding='utf-8') as f:
        for i, ex in enumerate(preview_examples, 1):
            msgs = ex['messages']
            user_msg = next(m['content'] for m in msgs if m['role'] == 'user')
            asst_msg = next(m['content'] for m in msgs if m['role'] == 'assistant')
            f.write(f"Example {i}\n")
            f.write(f"USER: {user_msg}\n")
            f.write(f"ASSISTANT: {asst_msg}\n")
            f.write('-' * 40 + '\n')

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("STATISTICS")
    print("=" * 50)
    print(f"  Total messages in result.json : {total_messages:,}")
    print(f"  Messages with text            : {msgs_with_text:,}")
    print(f"  Messages after cleaning       : {len(id_to_text):,}")
    print(f"  Reply-pair examples created   : {pairs_created:,}")
    print(f"  Duplicate pairs removed       : {duplicates_removed:,}")
    print(f"  Final examples saved          : {final_count:,}")
    print(f"  Output file                   : {out_path.resolve()}")
    print(f"  Preview file                  : {preview_path.resolve()}")
    print("=" * 50)


if __name__ == '__main__':
    main()
