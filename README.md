# ibigpt — Bukharian Dialect LLM Fine-Tuning

QLoRA fine-tuning of **Qwen2.5-7B-Instruct** on a Bukharian Uzbek/Tajik/Russian mixed-dialect dataset, sourced from a public Telegram supergroup ("Взаимопомощь | Бухара").

---

## Project Structure

```
ibigpt/
├── dataset/
│   └── result.json                  # Raw Telegram export (~725k messages)
│
├── prepare_bukhara_dataset.py       # Step 1: clean & extract reply pairs → JSONL
├── split_dataset.py                 # Step 2: quality filter + train/val/test splits
├── train_qwen_bukhara_qlora.ipynb   # Step 3: Google Colab QLoRA training notebook
│
├── bukhara_sft_dataset.jsonl        # Full cleaned dataset (317,910 examples)
├── preview_bukhara_examples.txt     # 50 human-readable example pairs
├── dataset_split_report.txt         # Split statistics report
│
├── train.jsonl                      # 257,627 examples (90%)
├── validation.jsonl                 # 14,312 examples (5%)
├── test.jsonl                       # 14,314 examples (5%)
│
├── train_5k.jsonl                   # Experimental subset
├── train_10k.jsonl                  # Experimental subset
├── train_30k.jsonl                  # Experimental subset
├── validation_1k.jsonl              # Eval subset
└── test_1k.jsonl                    # Eval subset
```

---

## Dataset

- **Source:** Telegram group export (`result.json`) — 724,865 messages
- **Language:** Bukharian Uzbek/Tajik/Russian code-switching (Cyrillic + Latin)
- **Format:** JSONL chat format (system / user / assistant)
- **Pairs built from:** reply chains — message A (user) → reply B (assistant)

### Pipeline

```
result.json
    ↓ prepare_bukhara_dataset.py
bukhara_sft_dataset.jsonl  (317,910 pairs)
    ↓ split_dataset.py  (quality filter + shuffle seed=42)
train.jsonl / validation.jsonl / test.jsonl  (286,253 clean examples)
```

### Quality filters applied
- Removed assistant replies shorter than 12 characters
- Removed emoji-only / symbol-only messages
- Removed low-value replies (ок, да, рахмат, 👍, etc.)
- Stripped PII: phone numbers, URLs, @usernames, emails
- Removed identical user/assistant pairs
- Removed duplicates

### Dataset statistics

| Stage | Count |
|---|---|
| Raw messages | 724,865 |
| Messages with text | 693,813 |
| After cleaning | 675,900 |
| Reply pairs created | 318,065 |
| Duplicates removed | 155 |
| After quality filter | 286,253 |
| **Train (90%)** | **257,627** |
| Validation (5%) | 14,312 |
| Test (5%) | 14,314 |

### JSONL format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that responds in natural Bukharian Uzbek/Tajik mixed conversational style. Keep the tone casual, local, and natural."
    },
    { "role": "user",      "content": "Assalomu aleykum. Kadom kitob magazin yala?" },
    { "role": "assistant", "content": "Kitob olami" }
  ]
}
```

---

## Training

### Base model
`Qwen/Qwen2.5-7B-Instruct`

### Method
QLoRA — 4-bit NF4 quantization + LoRA adapters

### LoRA config
| Parameter | Value |
|---|---|
| Rank (r) | 16 → 32 (recommended) |
| Alpha | 32 → 64 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Dropout | 0.05 |

### Recommended training config

| Setting | Starter (10k) | Full (30k+) |
|---|---|---|
| Dataset | train_10k.jsonl | train_30k.jsonl |
| Epochs | 1 | 3 |
| Batch size | 8 (A100) | 8 (A100) |
| Grad accum | 2 | 2 |
| LR | 2e-4 | 2e-4 |
| Max seq len | 1024 | 1024 |
| Est. time (A100) | ~20 min | ~60–90 min |

### Running the notebook

1. Open `train_qwen_bukhara_qlora.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (A100 recommended)
3. Run cells top to bottom
4. Upload `train_10k.jsonl` + `validation_1k.jsonl` when prompted
5. Adapter saved to `./bukhara-qwen2.5-7b-lora/`

### Known Colab compatibility fixes
- Use `bitsandbytes==0.46.0` (fixes `triton.ops` error on Python 3.12)
- Use `SFTConfig` instead of passing SFT args directly to `SFTTrainer` (TRL ≥ 0.13)
- Use `processing_class=tokenizer` instead of `tokenizer=tokenizer` (TRL ≥ 0.16)
- Load JSONL with `encoding='utf-8-sig'` + Python `json` module (avoids PyArrow Cyrillic parse errors)

---

## Output

After training, the adapter is saved to `./bukhara-qwen2.5-7b-lora/`:
```
adapter_config.json
adapter_model.safetensors   (~120 MB)
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

The base model is **not** included — load it separately from `Qwen/Qwen2.5-7B-Instruct`.

### Merge adapter (for deployment)
```python
merged = model.merge_and_unload()
merged.save_pretrained("bukhara-qwen2.5-7b-merged")
```

---

## Privacy

- All phone numbers, URLs, @usernames, and emails were removed from the dataset
- Sender names and user IDs are not present in any output file
- `result.json` (raw export) is excluded from version control via `.gitignore`

---

## Requirements

```
transformers==4.47.0
peft==0.14.0
trl>=0.13.0
bitsandbytes==0.46.0
accelerate==1.2.1
datasets==3.2.0
```

---

## Next Steps

- [ ] Re-train with `train_30k.jsonl`, 3 epochs, `LORA_R=32`
- [ ] Evaluate on `test_1k.jsonl`
- [ ] Merge adapter and export to GGUF (Q4_K_M) for local inference
- [ ] Push adapter to Hugging Face Hub (private)
