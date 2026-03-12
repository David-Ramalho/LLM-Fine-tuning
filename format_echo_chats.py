"""
Echo Chat Dataset Formatter
============================
Reads all .txt chat files from the "All chats" folder,
detects the format, parses turns, and writes:
  - Non-thinking chats  →  .../old/Non thinking/all_non_thinking_chats.txt
  - Thinking chats      →  .../old/Thinking/all_thinking_chats.txt

Output format
─────────────
Non-thinking:
    USER
    <message>

    ASSISTANT
    <response>

Thinking:
    USER
    <message>

    ASSISTANT
    <think>
    <reasoning>
    </think>
    <response>
"""

import os
import re
import html
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(r"C:\Users\tavar\OneDrive\Documentos\LLMs\Echo\Data\Llama_Echo memories\old")
INPUT_DIR         = BASE_DIR / "All chats"
NON_THINKING_DIR  = BASE_DIR / "Non thinking"
THINKING_DIR      = BASE_DIR / "Thinking"

OUT_NON_THINKING  = NON_THINKING_DIR / "all_non_thinking_chats.txt"
OUT_THINKING      = THINKING_DIR     / "all_thinking_chats.txt"

SEPARATOR = "\n\n" + "=" * 70 + "\n\n"


# ── thinking detection ─────────────────────────────────────────────────────────

def has_thinking(text: str) -> bool:
    if re.search(r'<details[^>]*type=["\']reasoning["\']', text):
        return True
    if re.search(r'<think>', text, re.IGNORECASE):
        return True
    if re.search(r'Let me think step by step\s*:', text, re.IGNORECASE):
        return True
    return False


def extract_thinking(text: str):
    """
    Returns (reasoning: str | None, clean_response: str).
    Tries all known thinking wrappers in order.
    """

    # ── <details type="reasoning"> ... </details> ──────────────────────────────
    m = re.search(
        r'<details[^>]*type=["\']reasoning["\'][^>]*>.*?<summary>.*?</summary>(.*?)</details>',
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        raw = m.group(1)
        lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith('> '):
                lines.append(stripped[2:])
            elif stripped == '>':
                lines.append('')
            else:
                lines.append(stripped)
        reasoning  = '\n'.join(lines).strip()
        # response is everything after </details>
        rest       = text[m.end():].strip()
        return reasoning, rest

    # ── <think> ... </think> ───────────────────────────────────────────────────
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()
        rest      = (text[:m.start()] + text[m.end():]).strip()
        return reasoning, rest

    # ── Let me think step by step: ... Answer: ────────────────────────────────
    m = re.search(
        r'Let me think step by step\s*:(.*?)(?:^Answer\s*:[ \t]*)',
        text, re.DOTALL | re.IGNORECASE | re.MULTILINE
    )
    if m:
        reasoning = m.group(1).strip()
        rest      = text[m.end():].strip()
        return reasoning, rest

    return None, text


# ── format detection ───────────────────────────────────────────────────────────

def detect_format(content: str) -> str:
    if re.search(r'^\d{9,11}\s*-\s*(user|assistant)\s*:', content,
                 re.MULTILINE | re.IGNORECASE):
        return 'timestamp'
    if re.search(r'^###\s*(USER|ASSISTANT)', content, re.MULTILINE):
        return 'markdown_header'
    if re.search(r'^(USER|ASSISTANT)\s*$', content, re.MULTILINE):
        return 'caps_newline'
    return 'unknown'


# ── parsers ────────────────────────────────────────────────────────────────────

def parse_timestamp(content: str):
    """1234567890 - user: ..."""
    pattern = r'\d{9,11}\s*-\s*(user|assistant)\s*:\s*'
    parts   = re.split(pattern, content, flags=re.IGNORECASE)
    turns   = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i].strip().lower()
        text = parts[i + 1].strip()
        if text:
            turns.append({'role': role, 'content': text})
        i += 2
    return turns


def parse_caps_newline(content: str):
    """
    USER
    <blank line(s)>
    content

    ASSISTANT
    content
    """
    # Split on a line that is exactly USER or ASSISTANT (case-insensitive)
    parts = re.split(r'\n(USER|ASSISTANT)\s*\n', content, flags=re.IGNORECASE)
    turns = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i].strip().lower()
        text = parts[i + 1].strip()
        if text:
            turns.append({'role': role, 'content': text})
        i += 2
    return turns


def parse_markdown_header(content: str):
    """### USER / ### ASSISTANT"""
    parts = re.split(r'\n?###\s*(USER|ASSISTANT)\s*\n', content, flags=re.IGNORECASE)
    turns = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i].strip().lower()
        text = parts[i + 1].strip()
        # If two consecutive USER blocks appear (context + actual message), merge
        if turns and turns[-1]['role'] == role:
            turns[-1]['content'] += '\n' + text
        elif text:
            turns.append({'role': role, 'content': text})
        i += 2
    return turns


def parse_chat(content: str):
    fmt = detect_format(content)
    if fmt == 'timestamp':
        return parse_timestamp(content)
    if fmt == 'markdown_header':
        return parse_markdown_header(content)
    if fmt == 'caps_newline':
        return parse_caps_newline(content)
    # fallback: try all
    for fn in (parse_timestamp, parse_markdown_header, parse_caps_newline):
        turns = fn(content)
        if turns:
            return turns
    return []


# ── formatting ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove invisible/control characters, normalise line endings, decode HTML entities."""
    # Decode HTML entities first (&gt; → >, &#x27; → ', &amp; → &, etc.)
    text = html.unescape(text)
    # Strip BOM
    text = text.lstrip('\ufeff')
    # Normalise line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove zero-width characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def format_turns(turns, file_has_thinking: bool) -> str:
    """Convert parsed turns list into the final string."""
    blocks = []
    for turn in turns:
        role    = turn['role']          # 'user' or 'assistant'
        content = turn['content']
        label   = 'USER' if role == 'user' else 'ASSISTANT'

        if role == 'assistant' and file_has_thinking and has_thinking(content):
            reasoning, response = extract_thinking(content)
            if reasoning:
                blocks.append(f"{label}\n<think>\n{reasoning}\n</think>\n{response}")
            else:
                blocks.append(f"{label}\n{content}")
        else:
            blocks.append(f"{label}\n{content}")

    return '\n\n'.join(blocks)


# ── main ────────────────────────────────────────────────────────────────────────

def process_file(filepath: Path):
    """
    Returns (formatted_str, is_thinking) or (None, False) on failure.
    """
    try:
        raw = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as exc:
        print(f"  [ERROR] Cannot read {filepath.name}: {exc}")
        return None, False

    content = clean_text(raw)
    turns   = parse_chat(content)

    if not turns:
        print(f"  [WARN]  Could not parse turns in: {filepath.name}")
        return None, False

    file_has_thinking = any(
        t['role'] == 'assistant' and has_thinking(t['content'])
        for t in turns
    )

    formatted = format_turns(turns, file_has_thinking)
    return formatted, file_has_thinking


def main():
    NON_THINKING_DIR.mkdir(parents=True, exist_ok=True)
    THINKING_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input folder not found:\n  {INPUT_DIR}")
        return

    txt_files = sorted(INPUT_DIR.glob('*.txt'))
    print(f"Found {len(txt_files)} .txt file(s) in: {INPUT_DIR}\n")

    non_thinking_parts = []
    thinking_parts     = []
    skipped            = []

    for fp in txt_files:
        print(f"Processing: {fp.name}")
        formatted, is_thinking = process_file(fp)

        if formatted is None:
            skipped.append(fp.name)
            continue

        header = f"### SOURCE FILE: {fp.name} ###"
        entry  = f"{header}\n\n{formatted}"

        if is_thinking:
            thinking_parts.append(entry)
            print(f"  → Thinking  ({len(thinking_parts)} so far)")
        else:
            non_thinking_parts.append(entry)
            print(f"  → Non-thinking  ({len(non_thinking_parts)} so far)")

    # ── write outputs ──────────────────────────────────────────────────────────
    print()
    if non_thinking_parts:
        OUT_NON_THINKING.write_text(
            SEPARATOR.join(non_thinking_parts),
            encoding='utf-8'
        )
        print(f"Non-thinking chats written  ({len(non_thinking_parts)} file(s)):\n  {OUT_NON_THINKING}")
    else:
        print("No non-thinking chats found.")

    if thinking_parts:
        OUT_THINKING.write_text(
            SEPARATOR.join(thinking_parts),
            encoding='utf-8'
        )
        print(f"Thinking chats written  ({len(thinking_parts)} file(s)):\n  {OUT_THINKING}")
    else:
        print("No thinking chats found.")

    if skipped:
        print(f"\nSkipped (could not parse): {skipped}")

    print("\nDone ✓")


if __name__ == '__main__':
    main()
