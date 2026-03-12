"""
Echo Dataset Merger
====================
Merges all formatted chat files from:
  - Non thinking/all_non_thinking_chats.txt
  - Thinking/all_thinking_chats.txt

Into a single file:
  - merged/all_echo_chats_merged.txt

The merged file keeps the same USER / ASSISTANT format,
so it can be fed directly into the JSONL converter for fine-tuning.

All non-thinking examples come first, then thinking examples.
A separator is added between source files for readability.
"""

import re
from pathlib import Path

# ── Paths — update BASE_DIR to match your setup ────────────────────────────────
BASE_DIR         = Path(r"C:\Users\tavar\OneDrive\Documentos\LLMs\Echo\Data\Llama_Echo memories\old")

NON_THINKING_FILE = BASE_DIR / "Non thinking" / "all_non_thinking_chats.txt"
THINKING_FILE     = BASE_DIR / "Thinking"     / "all_thinking_chats.txt"
OUTPUT_DIR        = BASE_DIR / "merged"
OUTPUT_FILE       = OUTPUT_DIR / "all_echo_chats_merged.txt"

SEPARATOR = "\n\n" + "=" * 70 + "\n\n"


def count_pairs(text: str) -> int:
    """Count USER/ASSISTANT pairs in a text block."""
    import re
    return len(re.findall(r'^USER\s*$', text, re.MULTILINE))


def merge():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    total_pairs = 0

    for label, filepath in [
        ("NON-THINKING", NON_THINKING_FILE),
        ("THINKING",     THINKING_FILE),
    ]:
        if not filepath.exists():
            print(f"[SKIP] Not found: {filepath}")
            continue

        text = filepath.read_text(encoding="utf-8", errors="replace")
        # Strip ### SOURCE FILE: xxx ### headers added by format_echo_chats.py
        text = re.sub(r'### SOURCE FILE:.*?###\s*\n?', '', text)
        pairs = count_pairs(text)
        total_pairs += pairs

        parts.append(text.strip())

        print(f"  [{label}] {pairs} Q&A pairs read from: {filepath.name}")

    if not parts:
        print("[ERROR] No input files found. Check your BASE_DIR paths.")
        return

    merged_text = SEPARATOR.join(parts)
    OUTPUT_FILE.write_text(merged_text, encoding="utf-8")

    print(f"\nMerged file written → {OUTPUT_FILE}")
    print(f"Total Q&A pairs: {total_pairs}")
    print("Done ✓")


if __name__ == "__main__":
    merge()
