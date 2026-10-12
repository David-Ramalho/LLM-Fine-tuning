# Echo fine-tuning workflow

This repository contains a chat-data preparation pipeline and a Kaggle-oriented QLoRA training/export workflow for the Echo example model, `LiquidAI/LFM2-8B-A1B`.

The scripts are kept at the repository root because their input and output paths are designed around that layout. Raw conversation exports and trained weights are not included.

## Files

| File | Purpose |
|---|---|
| `step1_format_chats.py` | Reads `.txt` files from `All chats/`, parses conversation turns, and separates examples with and without reasoning into `Thinking/` and `Non thinking/`. |
| `step2_build_dataset.py` | Cleans and validates those outputs, writes intermediate and final JSONL datasets, and creates `dataset_audit_report.txt`. |
| `finetuning.py` | Interactive Kaggle training workflow: loads the configured dataset/model, trains or resumes a LoRA adapter, tests chat, and offers save/export options. |
| `echo14_quantize.py` | Standalone Kaggle workflow to merge a supplied LoRA adapter and export a quantized GGUF file. |
| `check_loops.py` | Scans `echo_dataset_sft_clean.jsonl` for repeated paragraphs and markdown headings, then writes `loop_report.txt`. |

## Data preparation

1. Put exported `.txt` chat files in an `All chats/` folder beside `step1_format_chats.py`.
2. From this repository folder, run `python step1_format_chats.py`.
3. Run `python step2_build_dataset.py`.
4. Review `dataset_audit_report.txt` and the cleaned dataset before using private conversation data for training.

The preparation scripts create their input/output folders and dataset files as they run. Keep personal conversation exports and generated datasets out of public Git commits unless you have reviewed their contents and have permission to share them.

## Training and export

The training and quantization scripts are designed for Kaggle paths and GPU-backed environments. They include model and dataset locations in their configuration; check those values against the actual Kaggle input panel before running. The training script installs its Python dependencies when it detects a Kaggle environment.

**Configuration mismatch to resolve:** `step2_build_dataset.py` identifies `echo_dataset_sft_clean.jsonl` as the final dataset to upload. `finetuning.py` currently looks for `echo_dataset_sft.jsonl` at its configured Kaggle path. Confirm which file should be trained and align the Kaggle dataset/path before starting training. This local documentation records the mismatch; the code has not been changed to guess the intended file.

`echo14_quantize.py` expects a LoRA adapter at its configured Kaggle input path and writes a Q4_K_M GGUF to Kaggle's working directory. Check storage and model requirements in that script before running; export uses substantial temporary disk space.

## Pipeline at a glance

```text
All chats/*.txt
      │
      ▼
step1_format_chats.py
      ├── Non thinking/all_non_thinking_chats.txt
      └── Thinking/all_thinking_chats.txt
                │
                ▼
       step2_build_dataset.py
         ├── echo_dataset_sft.jsonl            (intermediate)
         ├── echo_dataset_sft_clean.jsonl      (final per script)
         └── dataset_audit_report.txt
                │
                ▼
     Kaggle fine-tuning / LoRA export
                │
                ▼
        optional GGUF quantization
```

## Current scope

This is an example workflow for the Echo model and data format. The source scripts contain fixed Kaggle paths and model-specific settings, so adapting them to another model or local environment requires configuration changes. No training data, model checkpoints, LoRA adapters, or GGUF files are stored in this repository.

