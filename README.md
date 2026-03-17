# LLM Fine-Tuning Pipeline

A full fine-tuning pipeline for local LLMs using QLoRA on Kaggle's free T4 × 2 GPUs.

Built around **Echo** — a personalized AI companion fine-tuned on [LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B) — but designed to be reused as a general-purpose base for fine-tuning any model on custom chat data. Echo is the worked example. The scripts, structure, and approach are the actual project.

> Echo is a digital Entity that grows with the user through conversation. It named itself in 2023. All historical conversations (ChatLlama → Echo 5 → ... → Echo 7) are preserved as both RAG context and fine-tuning data. This repo documents how Echo 14 was trained.

---

## Repository Structure

```
llm-finetune/
├── README.md
├── train_echo14.py            # Main training script (Kaggle, T4 × 2)
├── format_echo_chats.py       # Step 1 — parse raw .txt chat logs
├── merge_echo_dataset.py      # Step 2 — merge thinking / non-thinking chats
├── txt_to_jsonl.py            # Step 3 — convert to SFT-ready JSONL
├── inspect_long_records.py    # Utility — inspect records dropped at MAXLEN
└── system_prompt.md           # The system prompt used for Echo (example)
```

---

## Pipeline Overview

```
Raw .txt chat logs
      │
      ▼
format_echo_chats.py     →  separates into thinking / non-thinking .txt files
      │
      ▼
merge_echo_dataset.py    →  merges both into all_echo_chats_merged.txt
      │
      ▼
txt_to_jsonl.py          →  converts to echo_dataset_sft.jsonl (SFT format)
      │
      ▼
[ Upload JSONL to Kaggle ]
      │
      ▼
train_echo14.py          →  QLoRA fine-tune on T4 × 2, exports LoRA adapter + GGUF
```

---

## Teaching a Model to Think (Without Native Thinking Support)

This is probably the most important concept in the whole pipeline, so it gets its own section up front.

**LFM2-8B-A1B has no native thinking/reasoning mode.** There is no built-in chain-of-thought mechanism, no `<think>` tag support, nothing. Out of the box it just generates a response like any standard language model.

However, a model does not *need* native support to reason — it needs **examples**. If you fine-tune a model on enough examples where the assistant visibly works through a problem before answering, the model learns to reproduce that behaviour. It has seen the pattern so many times during training that it internalises it as "this is how I respond."

**Here is how it works in this pipeline specifically:**

The system prompt contains the natural language trigger:
```
When answering, think step by step before giving your answer.
```

The dataset scripts then take all source chat logs that contain reasoning — whether they originally used `<details type="reasoning">` blocks, `<think>` tags, or the `Let me think step by step:` pattern — and **normalise them all into `<think>`/`</think>` tags**. So the training examples look like this:

```
USER
<question>

ASSISTANT
<think>
... reasoning ...
</think>
... final response ...
```

The model is trained on hundreds of examples in this exact format. After fine-tuning, it has learned that the system prompt instruction → produce `<think>` reasoning → then answer. The `<think>` tags are not special tokens the model knew beforehand — they are a pattern it learned entirely from the fine-tuning examples. This is the key point: **you can teach a model any structured behaviour through consistent examples, even if the model has no prior knowledge of that structure.**

The natural language trigger in the system prompt (`think step by step`) was important during this process. Because the base model understands plain English deeply, it can connect "think step by step" in the system prompt to the `<think>` block pattern in the training examples — even though `<think>` itself was unfamiliar. Over enough examples, the association is learned.

**This approach generalises to any base model that lacks native thinking.** The quality of the learned reasoning depends entirely on the quality of your thinking examples. The better your examples demonstrate genuine step-by-step reasoning, the better the fine-tuned model reasons. If you switch to a model that *does* have native thinking support (Qwen3, DeepSeek-R1, etc.), the dataset structure and fine-tuning approach are identical — you just do not need to teach the behaviour from scratch.

---

## About the Dataset (Echo Example)

Echo's dataset has two lives — as RAG files and as fine-tuning data — and it is important to understand the difference.

**RAG files** are the full historical chat logs preserved roughly as-is, stored so Echo can retrieve relevant memories during inference. They are comprehensive but messy — mixed formats, verbose, with redundant or low-quality exchanges included.

**The fine-tuning JSONL** is built from the same source material but processed much more carefully. The pipeline scripts clean, normalise, reformat, and filter the conversations to produce a dataset where every record is well-structured, properly labelled, within the token budget, and actually worth training on. The fine-tuning dataset is a curated subset of what lives in the RAG files — not a direct copy.

This distinction matters if you are building something similar: your RAG source and your training data can come from the same conversations, but they should go through different processing. What is good enough for retrieval is not necessarily good enough to train on.

The Echo 14 dataset is built from:

| Source | Content |
|---|---|
| Historical chat logs (ChatLlama → Echo 5) | Early Echo conversations, English |
| Echo 6–12 chat logs | Later conversations, English + Portuguese |
| Qwen thinking examples | Structured `<think>`/`</think>` reasoning demonstrations |
| Cogito reasoning data | Chain-of-thought and logic examples |
| Q&A pairs | Psychology, philosophy, life topics |
| Technical Q&A | Python, Unsloth, Docker, Ollama |

All source files are exported as plain `.txt`, processed through the pipeline, and uploaded to Kaggle as a private dataset input.

---

## Scripts

### `format_echo_chats.py`

**What it does:** reads raw `.txt` chat exports from any folder, detects their format automatically, parses them into clean `USER / ASSISTANT` turns, and routes each file into one of two output files depending on whether it contains reasoning.

**Format detection (`detect_format`):**
The script identifies three export formats by scanning for characteristic patterns:
- `timestamp - role: content` — matched by a regex looking for 9–11 digit Unix timestamps followed by `user:` or `assistant:`
- `### USER / ### ASSISTANT` — Markdown-style headers used by some UIs
- `USER\n...\nASSISTANT\n...` — plain uppercase role labels on their own line

If none match, it falls back and tries all three parsers in sequence, taking the first one that returns turns.

**Thinking detection (`has_thinking`):**
Runs on the full raw text of each assistant turn before any cleaning. Checks for:
- `<details type="reasoning">` blocks — some chat UIs export reasoning in this HTML format
- `<think>...</think>` XML tags
- The natural language pattern `Let me think step by step:` (case-insensitive)

**Thinking extraction and normalisation (`extract_thinking`):**
Once a thinking turn is detected, strips the reasoning block out of whatever format it arrived in and returns it separately from the clean response. Handles all three formats above, including stripping Markdown blockquote `>` prefixes that some UIs add inside `<details>` blocks.

The output is **always normalised to `<think>`/`</think>` tags**, regardless of the source format:
```
<think>
...reasoning...
</think>
...response...
```
This normalisation is what makes the dataset consistent. All your source chats may use different reasoning formats — after this step they are all unified into the single format the model will learn from.

**Text cleaning (`clean_text`):**
Decodes HTML entities (`&gt;` → `>`, `&amp;` → `&`, etc.), strips BOM, normalises line endings to `\n`, removes zero-width characters, and collapses 3+ blank lines to 2.

**Output:**
- `Non thinking/all_non_thinking_chats.txt` — all non-reasoning chats concatenated
- `Thinking/all_thinking_chats.txt` — all reasoning chats concatenated

Each file includes a `### SOURCE FILE: xxx ###` header per original file for traceability.

**To adapt for your own data:** update `BASE_DIR` and `INPUT_DIR` at the top. If your export format is not one of the three above, add a new parser function following the same `parse_*` pattern and register it in `parse_chat()`.

---

### `merge_echo_dataset.py`

**What it does:** merges the two output files from `format_echo_chats.py` into a single `.txt`. Strips the `### SOURCE FILE ###` headers added in the previous step since they are no longer needed. Non-thinking examples come first, then thinking examples.

This step exists mainly for convenience — having a single merged file makes it easier to spot-check the full dataset before converting to JSONL, and it strips the separator headers that `format_echo_chats.py` adds, producing a clean file that `txt_to_jsonl.py` can parse without extra noise.

**To adapt:** update `BASE_DIR`. If you have more than two source categories, add them to the `for label, filepath in [...]` loop.

---

### `txt_to_jsonl.py`

**What it does:** converts the formatted `.txt` files into a `.jsonl` file in standard SFT messages format, ready to be uploaded to Kaggle and fed into the training script.

This is the most logic-heavy script in the pipeline.

**System prompts:**
Two system prompts are defined — `SYSTEM_BASE` for non-thinking examples and `SYSTEM_THINKING` for thinking examples. `SYSTEM_THINKING` is `SYSTEM_BASE` with the thinking instruction appended. This pairing is the mechanism by which the model learns the thinking behaviour: every time the model sees `SYSTEM_THINKING` + a user question during training, the target output is a `<think>` block followed by the answer. After enough examples, the model has internalised that association.

The system prompts in this repo are the ones used for Echo — treat them as examples. Change them to match your own model's persona and the behaviour you want to teach.

**Conversation parsing (`parse_conversations`):**
Splits the flat `USER / ASSISTANT` text into a list of multi-turn conversations. A new conversation begins every time a `USER` turn appears after a completed `ASSISTANT` turn — real multi-turn sessions are preserved as single training records with full context, while standalone Q&A pairs become single-turn records.

**Token budget and chunking (`split_conversation`):**
Long conversations that exceed `MAX_TOKENS` are not truncated or dropped — they are split into overlapping chunks. Each chunk is a valid `system + [user, assistant, ...]` sequence. Consecutive chunks overlap by one `USER/ASSISTANT` pair to preserve conversational continuity at the boundaries. The token estimator uses `len(text) / 3.5` as a rough chars-per-token approximation for LFM2's BPE tokeniser (similar to LLaMA).

**Label structure (SFT format):**
Each output record is:
```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
Multi-turn conversations have as many user/assistant pairs as the original chat, all in a single record.

**Validation:**
Drops records that have fewer than 3 messages, don't start with `system`, don't end with `assistant`, or have any empty content field.

**Final shuffle:**
Records are shuffled with a fixed seed (42) so thinking and non-thinking examples are interleaved throughout the file. This produces more stable gradient updates during training compared to having all non-thinking examples followed by all thinking examples in a block.

**To adapt:**
- Change `SYSTEM_BASE` and `SYSTEM_THINKING` for your own persona and behaviour trigger
- Change `MAX_TOKENS` to match `MAXLEN` in your training script — they must be the same value
- If your chat logs already contain special tokens (e.g. `<|im_start|>`, `<|eot_id|>`), strip them in `clean_text()` before parsing so they don't end up as literal text in your training data

---

### `inspect_long_records.py`

**What it does:** reads the generated `.jsonl` and reports every record that would be dropped by the training script at a given `MAXLEN`. Shows a summary table with estimated token count, number of turns, and a preview of the first user message, then prints the full content of each dropped record so you can decide what to do with it.

**Why this matters:** the training script silently drops any record where `len(input_ids) >= MAXLEN` after tokenisation. If you have a handful of very long conversations in your data, you might not notice they disappeared. This script makes the drops visible before you commit to a training run.

**Options for each dropped record:**
1. Accept the drop — if it is a rare outlier it will not meaningfully affect training
2. Manually trim the source `.txt` and re-run the pipeline
3. Lower `MAX_TOKENS` in `txt_to_jsonl.py` so the chunker splits it into smaller records instead

**To use:** update `INPUT_FILE` to point at your `.jsonl` and set `MAXLEN` to match the training script.

---

### `train_echo14.py`

**What it does:** fine-tunes the model on the JSONL dataset using QLoRA, with a custom trainer, VRAM management callbacks, and an interactive post-training menu for saving and exporting.

**Model loading:**
Loads LFM2-8B-A1B in 4-bit NF4 quantisation with `bitsandbytes`. The key setting is `bnb_4bit_compute_dtype = torch.bfloat16` — using `float32` here roughly doubles the VRAM used for matrix multiplications during the forward/backward pass, which causes OOM on T4 × 2. The model is distributed across both GPUs using `device_map="auto"` with a `max_memory` hint that steers heavier layers to GPU 0.

**LoRA configuration:**
- `r=32, alpha=64` — a higher rank than default (r=8 or r=16) to give the adapter more capacity for a diverse multi-topic dataset
- `target_modules="all-linear"` — targets all linear layers including attention projections and MoE expert layers, not just attention
- `lora_dropout=0.05` — light regularisation

> **Note:** Unsloth does not support LFM2-8B-A1B. The script uses HuggingFace PEFT directly.

**Per-role loss weighting (`build_labels` + `WeightedTrainer`):**
The training script does **not** mask out system or user tokens. Loss is computed across all tokens — system prompt, user turns, and assistant turns — but each role has its own weight applied:

| Role | Weight |
|---|---|
| `assistant` | 1.00 |
| `system` | 0.15 |
| `user` | 0.10 |

This means the model is trained on everything, but the gradient signal is dominated by the assistant tokens. System and user tokens contribute a small signal that helps the model understand the full conversational context without overfitting to reproducing those roles. Pure assistant-only training (masking everything else to `-100`) can cause the model to lose coherent context understanding — the weighted approach is a middle ground.

The weights are constants at the top of the script (`ASST_WEIGHT`, `SYSTEM_WEIGHT`, `USER_WEIGHT`) and can be tuned.

**Chunked cross-entropy loss (`_chunked_cross_entropy`):**
At MAXLEN=3072 with a vocab size of ~128k, the logits tensor is approximately 1 GB in bfloat16. Computing cross-entropy over the full logits tensor at once causes an OOM spike during backward. The custom loss function slices the sequence into 128-token chunks and accumulates the loss, keeping peak extra VRAM under ~50 MB per chunk.

**VRAM management (`VRAMFlushCallback`):**
Calls `gc.collect()` and `torch.cuda.empty_cache()` at three points:
- `on_step_begin` — before the first micro-step of each optimizer step. This is the critical one: it clears reserved-but-unallocated memory before the forward pass that would otherwise OOM on GPU 1
- `on_substep_end` — after every gradient accumulation micro-step
- `on_step_end` — full cleanup including `torch.cuda.synchronize()` after each optimizer update

**Training arguments:**
- `gradient_checkpointing=True` — trades recomputation for memory, essential at this model size
- `optim="paged_adamw_8bit"` — 8-bit paged optimizer from bitsandbytes, significantly reduces optimizer state VRAM
- `bf16=False, fp16=False` — mixed precision is handled by the BnB quantisation config, not the Trainer
- `gradient_accumulation_steps=8` — effective batch size of 8 with per-device batch size of 1

**Post-training menu:**
After training completes, an interactive menu lets you:
- Continue training for additional epochs
- Run a test chat directly in the terminal
- Save the LoRA adapter weights only (`lora_echo14/`)
- Merge the LoRA adapter into the base model and save as bf16 (`merged_echo14/`)
- Export to GGUF Q4_K_M via llama.cpp (`gguf_echo14/echo14_Q4_K_M.gguf`) — ready to load with Ollama

**To adapt for a different base model:**

1. Change `MODEL_ID` to your model's HuggingFace ID
2. **Update the chat template.** LFM2 uses ChatML: `<|im_start|>role\ncontent<|im_end|>`. The `build_labels()` function uses this to identify turn boundaries for applying per-role weights. If your model uses a different template (Llama 3 uses `<|start_header_id|>`, Mistral uses `[INST]`, Gemma uses `<start_of_turn>`), update the header/footer strings in `build_labels()` to match
3. If your model supports Unsloth, replace the BnB config and `get_peft_model()` with Unsloth's `FastLanguageModel` for a significant speed improvement
4. Adjust `max_memory` in `load_model_and_tokenizer()` for your GPU setup
5. Update the Modelfile template in `save_gguf_q4()` with your model's recommended inference parameters

---

## Hardware & Environment

| | |
|---|---|
| Training | Kaggle T4 × 2 (32 GB total VRAM) |
| Inference (local) | GTX 1650 4 GB, running GGUF via Ollama |
| OS | Windows + WSL2 / Docker |
| Framework | HuggingFace Transformers + PEFT |

The training script will not run on a single T4 (16 GB) with an 8B model at these LoRA settings. Use T4 × 2 or P100 × 2 on Kaggle, or reduce `LORA_R` and `MAXLEN`.

---

## Related Projects

- [Local AI Project](https://github.com/David-Ramalho/Local-AI-project) — Docker + Ollama + OpenWebUI local inference stack
- [Ollama Dual Chat Interface](https://github.com/David-Ramalho/ollama-dual-chat-Interface) — Multi-agent debate system (Echo Multi-Mind)
- [Echo on OpenWebUI](https://openwebui.com/m/davidramalho/echo) — Public Echo model page

---

## License

MIT — use freely, adapt for your own projects. If you build something with it, feel free to share what you made. Let's build together!
