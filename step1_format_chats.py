#!/usr/bin/env python3
"""
Echo Dataset — Step 1: Format Raw Chats
========================================
Reads all .txt chat files from the "All chats" folder,
detects the format, parses turns, and writes:
  - Non thinking/all_non_thinking_chats.txt
  - Thinking/all_thinking_chats.txt

Classification is per USER+ASSISTANT exchange, NOT per file.
A single mixed file will have its exchanges split between both outputs.

THINKING FORMAT VARIANTS HANDLED
──────────────────────────────────
Format B — full <details> wrapper (Claude/QwQ style with <summary>):
    <details type="reasoning" done="true" duration="18">
    <summary>Thought for 18 seconds</summary>
    > reasoning line 1
    >
    > reasoning line 2
    </details>
    Response text

Format A1 — no opening <details>, inline > markers, orphan </details>:
    Thought for 68 seconds > sentence1. Let me think. > > sentence2. </details>
    Response text
    (The </details> is an export artifact — no matching opening tag.)

Format A2 — no opening <details>, newline-based > markers, no </details>:
    Thought for 91 seconds > first sentence of reasoning
    > continued reasoning
    >
    > new paragraph of reasoning
    Response text

Format C — native <think> tags:
    <think>reasoning</think>
    Response text

Format D — prompted format:
    Let me think step by step:
    ... reasoning ...
    Answer:
    Response text

All formats produce identical clean output:
    ASSISTANT
    <think>
    cleaned reasoning paragraphs
    </think>
    response text
"""

import re
import html
from pathlib import Path

# ── Paths — all relative to this script's location ────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
INPUT_DIR        = BASE_DIR / "All chats"
NON_THINKING_DIR = BASE_DIR / "Non thinking"
THINKING_DIR     = BASE_DIR / "Thinking"
OUT_NON_THINKING = NON_THINKING_DIR / "all_non_thinking_chats.txt"
OUT_THINKING     = THINKING_DIR     / "all_thinking_chats.txt"

SEPARATOR = "\n\n" + "=" * 70 + "\n\n"


# ══════════════════════════════════════════════════════════════════════════════
#  THINKING DETECTION & EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def has_thinking(text: str) -> bool:
    # <details type="reasoning"> MUST be checked first because its <summary>
    # contains "Thought for X seconds", which would prematurely trigger
    # the Format A extractor below.
    if re.search(r'<details[^>]*type=["\']reasoning["\']', text, re.IGNORECASE):
        return True
    if re.search(r'<think\s*>', text, re.IGNORECASE):
        return True
    if re.search(r'Let me think step by step\s*:', text, re.IGNORECASE):
        return True
    # Standalone "Thought for X seconds" — Format A1 / A2 (no <details> wrapper)
    if re.search(r'Thought for \d+ seconds', text, re.IGNORECASE):
        return True
    return False


def _strip_blockquote_prefix(text: str) -> str:
    """
    Strip markdown blockquote '> ' prefix from each line (line-start format).
      '> Some text'  → 'Some text'
      '>'            → '' (blank line)
      '>Some text'   → 'Some text'
      Normal lines   → unchanged
    """
    lines  = text.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('> '):
            result.append(stripped[2:])
        elif stripped == '>':
            result.append('')
        elif stripped.startswith('>') and len(stripped) > 1:
            result.append(stripped[1:].lstrip())
        else:
            result.append(line)
    return '\n'.join(result)


def _clean_blockquote_block(raw_block: str) -> str:
    """
    Clean a raw reasoning block that may use either blockquote style:

    Line-start (Format B / A2) — > prefix at start of each line:
        > sentence one
        >
        > sentence two

    Inline (Format A1) — > used as paragraph/sentence separator mid-line:
        > sentence one. More text. > > sentence two. > > sentence three.

    In the inline style ' > > ' is a paragraph break and ' > ' is a sentence
    separator. Both are normalised to newlines so the final reasoning reads as
    clean paragraphs regardless of the original spacing.

    Returns cleaned reasoning text.
    """
    # Strip line-start > prefixes (handles newline format)
    cleaned = _strip_blockquote_prefix(raw_block)
    # Convert inline paragraph separator " > > " → double newline
    cleaned = re.sub(r'\s*>\s*>\s*', '\n\n', cleaned)
    # Convert remaining inline " > " → newline
    cleaned = re.sub(r'\s+>\s+', '\n\n', cleaned)
    # Strip any surviving orphan > at the very start
    cleaned = re.sub(r'^>\s*', '', cleaned.strip())
    # Collapse excess blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def _extract_details_reasoning(text: str) -> tuple:
    """
    Extract reasoning from <details type="reasoning"> blocks (Format B).

    Handles:
      - Full block with </details>
      - Block where </details> is absent (takes everything after opening tag)
      - <summary>...</summary> always stripped
      - Both line-start and inline > styles inside the block

    Returns (reasoning: str | None, rest: str).
    """
    open_match = re.search(
        r'<details[^>]*type=["\']reasoning["\'][^>]*>',
        text, re.IGNORECASE
    )
    if not open_match:
        return None, text

    content_start = open_match.end()

    full_match = re.search(
        r'<details[^>]*type=["\']reasoning["\'][^>]*>(.*?)</details>',
        text, re.DOTALL | re.IGNORECASE
    )

    if full_match:
        raw_block = full_match.group(1)
        rest      = text[full_match.end():].strip()
    else:
        # No closing </details> — take everything after the opening tag
        raw_block = text[content_start:]
        rest      = ""

    # Strip <summary>...</summary> (e.g. "Thought for 18 seconds")
    raw_block = re.sub(r'<summary>.*?</summary>', '', raw_block,
                       flags=re.DOTALL | re.IGNORECASE).strip()

    reasoning = _clean_blockquote_block(raw_block)

    if not reasoning:
        return None, rest

    return reasoning, rest


def _extract_thought_for_seconds(text: str) -> tuple:
    """
    Extract reasoning from standalone 'Thought for X seconds' formats.

    Format A1 — inline > markers, orphan </details> closing tag:
        Thought for 68 seconds > sentence1. Let me think. > > sentence2. </details>
        Actual response here.

    Format A2 — newline-based > lines, no </details>:
        Thought for 91 seconds > First sentence.
        > More reasoning.
        >
        > New paragraph.
        Actual response here.

    The </details> in A1 is an export artifact — no opening <details> tag exists.
    It acts as a reliable end-of-reasoning marker and is stripped.

    Returns (reasoning: str | None, rest: str).
    """
    header_match = re.search(r'Thought for \d+ seconds\s*', text, re.IGNORECASE)
    if not header_match:
        return None, text

    after_header = text[header_match.end():]

    # Case A1: orphan </details> marks end of reasoning block
    details_close = re.search(r'</details>', after_header, re.IGNORECASE)
    if details_close:
        raw_block = after_header[:details_close.start()]
        rest      = after_header[details_close.end():].strip()
    else:
        # Case A2: no </details> — collect '>' lines until first non-'>' non-blank line
        lines           = after_header.splitlines()
        reasoning_lines = []
        response_lines  = []
        in_reasoning    = True

        for line in lines:
            stripped = line.strip()
            if in_reasoning and (stripped.startswith('>') or stripped == ''):
                reasoning_lines.append(line)
            elif in_reasoning and stripped:
                # First non-empty, non-'>' line — reasoning ends here
                in_reasoning = False
                response_lines.append(line)
            else:
                response_lines.append(line)

        raw_block = '\n'.join(reasoning_lines)
        rest      = '\n'.join(response_lines).strip()

    reasoning = _clean_blockquote_block(raw_block)

    if not reasoning:
        return None, text  # extraction failed — caller treats as non-thinking

    return reasoning, rest


def extract_thinking(text: str) -> tuple:
    """
    Returns (reasoning: str | None, clean_response: str).

    Extraction order — do NOT reorder (order is semantically significant):
      1. <details type="reasoning">   — checked first because its <summary>
                                        contains "Thought for X seconds", which
                                        would trigger extractor 4 if checked later.
      2. <think>...</think>           — native think-tag format
      3. Let me think step by step:   — prompted-reasoning format
      4. Thought for X seconds        — standalone blockquote (A1 / A2)
    """

    # 1. <details type="reasoning"> (Format B) ─────────────────────────────────
    if re.search(r'<details[^>]*type=["\']reasoning["\']', text, re.IGNORECASE):
        return _extract_details_reasoning(text)

    # 2. <think>...</think> (Format C) ─────────────────────────────────────────
    if re.search(r'<think\s*>', text, re.IGNORECASE):

        complete_pairs = list(re.finditer(
            r'<think\s*>(.*?)</think\s*>',
            text, re.IGNORECASE | re.DOTALL
        ))

        # Handle unclosed <think> tag
        text_no_complete = text
        for pair in reversed(complete_pairs):
            text_no_complete = (
                text_no_complete[:pair.start()] + text_no_complete[pair.end():]
            )
        orphan_open = re.search(r'<think\s*>', text_no_complete, re.IGNORECASE)

        if orphan_open:
            all_opens  = list(re.finditer(r'<think\s*>',  text, re.IGNORECASE))
            all_closes = list(re.finditer(r'</think\s*>', text, re.IGNORECASE))
            if len(all_opens) > len(all_closes):
                cut_pos = all_opens[-1].start()
                text    = text[:cut_pos].strip()
                complete_pairs = list(re.finditer(
                    r'<think\s*>(.*?)</think\s*>',
                    text, re.IGNORECASE | re.DOTALL
                ))

        if not complete_pairs:
            return None, text.strip()

        all_reasoning = [p.group(1).strip() for p in complete_pairs
                         if p.group(1).strip()]

        if not all_reasoning:
            reasoning = None
        elif len(all_reasoning) == 1:
            reasoning = all_reasoning[0]
        else:
            # Multiple blocks — join without adding --- noise
            reasoning = "\n\n".join(all_reasoning)

        clean = text
        for pair in reversed(complete_pairs):
            clean = clean[:pair.start()] + clean[pair.end():]
        clean = clean.strip()

        return reasoning, clean

    # 3. "Let me think step by step:" (Format D) ───────────────────────────────
    m = re.search(
        r'Let me think step by step\s*:(.*?)(?:^Answer\s*:[ \t]*)',
        text, re.DOTALL | re.IGNORECASE | re.MULTILINE
    )
    if m:
        reasoning = m.group(1).strip()
        rest      = text[m.end():].strip()
        return reasoning, rest

    # 4. Standalone "Thought for X seconds" (Format A1 / A2) ──────────────────
    if re.search(r'Thought for \d+ seconds', text, re.IGNORECASE):
        return _extract_thought_for_seconds(text)

    return None, text


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT DETECTION & PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_chat(content: str) -> list:
    """
    Unified turn parser that handles files mixing any combination of formats:
      - Timestamp:       1749060370 - user: ...
      - Markdown header: ### USER  /  ### ASSISTANT
      - Caps newline:    USER\\n...  /  ASSISTANT\\n...

    The old approach (detect_format → single parser) broke on mixed files
    because it picked one format and ignored turns written in the other formats.
    This unified regex splits on ALL recognised markers in a single pass so
    no turn is ever dropped regardless of which format it uses.
    """
    SPLIT_RE = re.compile(
        r'(?:'
        # Timestamp:  1749060370 - user:
        r'\d{9,11}\s*-\s*(user|assistant)\s*:\s*'
        r'|'
        # Markdown header:  ### USER  /  ### ASSISTANT
        r'###\s*(USER|ASSISTANT)\s*\n'
        r'|'
        # Caps newline (mid-file):  \nUSER\n  /  \nASSISTANT\n
        r'\n(USER|ASSISTANT)\s*\n'
        r'|'
        # Caps newline (start-of-file or after blank line):  ^USER\n
        r'(?:^|\n\n)(USER|ASSISTANT)\s*\n'
        r')',
        re.IGNORECASE | re.MULTILINE
    )

    parts = SPLIT_RE.split(content)
    turns = []
    i     = 1   # index 0 is pre-match text before any role marker

    while i + 4 < len(parts):
        # Four capture groups per match; exactly one holds the role word
        role_raw = parts[i] or parts[i+1] or parts[i+2] or parts[i+3]
        text     = parts[i+4].strip() if i+4 < len(parts) else ""
        i       += 5

        if role_raw and text:
            turns.append({'role': role_raw.strip().lower(), 'content': text})

    return turns


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.lstrip('\ufeff')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  EXCHANGE FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_exchange(user_content: str, assistant_content: str) -> tuple:
    """
    Format a single USER+ASSISTANT exchange.
    Returns (formatted_string, is_thinking).
    """
    if has_thinking(assistant_content):
        reasoning, response = extract_thinking(assistant_content)
        if reasoning:
            assistant_block = f"ASSISTANT\n<think>\n{reasoning}\n</think>\n{response}"
            return f"USER\n{user_content}\n\n{assistant_block}", True
        else:
            # has_thinking fired but extraction found nothing — treat as non-thinking
            return f"USER\n{user_content}\n\nASSISTANT\n{assistant_content}", False
    else:
        return f"USER\n{user_content}\n\nASSISTANT\n{assistant_content}", False


def turns_to_exchanges(turns: list) -> list:
    """Pair consecutive USER + ASSISTANT turns."""
    exchanges = []
    i = 0
    while i < len(turns):
        if (turns[i]['role'] == 'user'
                and i + 1 < len(turns)
                and turns[i + 1]['role'] == 'assistant'):
            exchanges.append((turns[i]['content'], turns[i + 1]['content']))
            i += 2
        else:
            i += 1
    return exchanges


# ══════════════════════════════════════════════════════════════════════════════
#  FILE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_file(filepath: Path) -> tuple:
    """
    Parse a file and return:
      (thinking_exchanges, non_thinking_exchanges, n_thinking, n_non_thinking)
    """
    try:
        raw = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as exc:
        print(f"  [ERROR] Cannot read {filepath.name}: {exc}")
        return [], [], 0, 0

    content = clean_text(raw)
    turns   = parse_chat(content)

    if not turns:
        print(f"  [WARN]  Could not parse turns in: {filepath.name}")
        return [], [], 0, 0

    exchanges        = turns_to_exchanges(turns)
    thinking_out     = []
    non_thinking_out = []

    for user_txt, asst_txt in exchanges:
        formatted, is_thinking = format_exchange(user_txt, asst_txt)
        if is_thinking:
            thinking_out.append(formatted)
        else:
            non_thinking_out.append(formatted)

    return thinking_out, non_thinking_out, len(thinking_out), len(non_thinking_out)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    NON_THINKING_DIR.mkdir(parents=True, exist_ok=True)
    THINKING_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input folder not found:\n  {INPUT_DIR}")
        print(f"  Create an 'All chats' folder next to this script and put your .txt files in it.")
        return

    txt_files = sorted(INPUT_DIR.glob('*.txt'))
    if not txt_files:
        print(f"[ERROR] No .txt files found in:\n  {INPUT_DIR}")
        return

    print(f"Echo Dataset — Step 1: Format Raw Chats")
    print("─" * 60)
    print(f"Input  : {INPUT_DIR}  ({len(txt_files)} file(s))")
    print(f"Output : {NON_THINKING_DIR.name}/  and  {THINKING_DIR.name}/")
    print()

    all_thinking     = []
    all_non_thinking = []
    total_thinking   = 0
    total_non_think  = 0
    skipped          = []

    for fp in txt_files:
        print(f"  Processing: {fp.name}")
        t_exch, nt_exch, n_t, n_nt = process_file(fp)

        if n_t == 0 and n_nt == 0:
            skipped.append(fp.name)
            continue

        # NOTE: ### SOURCE FILE ### headers are intentional.
        # step2_build_dataset.py strips them via strip_file_noise().
        header = f"### SOURCE FILE: {fp.name} ###"

        if t_exch:
            all_thinking.append(f"{header}\n\n" + '\n\n'.join(t_exch))
        if nt_exch:
            all_non_thinking.append(f"{header}\n\n" + '\n\n'.join(nt_exch))

        total_thinking  += n_t
        total_non_think += n_nt

        if n_t > 0 and n_nt > 0:
            print(f"    → Mixed: {n_t} thinking + {n_nt} non-thinking")
        elif n_t > 0:
            print(f"    → Thinking: {n_t} exchange(s)")
        else:
            print(f"    → Non-thinking: {n_nt} exchange(s)")

    print()

    if all_non_thinking:
        OUT_NON_THINKING.write_text(SEPARATOR.join(all_non_thinking), encoding='utf-8')
        print(f"  Non-thinking → {OUT_NON_THINKING.name}  ({total_non_think} exchanges)")
    else:
        print("  No non-thinking exchanges found.")

    if all_thinking:
        OUT_THINKING.write_text(SEPARATOR.join(all_thinking), encoding='utf-8')
        print(f"  Thinking     → {OUT_THINKING.name}  ({total_thinking} exchanges)")
    else:
        print("  No thinking exchanges found.")

    print(f"\n  Total: {total_thinking + total_non_think} exchanges "
          f"({total_thinking} thinking + {total_non_think} non-thinking)")

    if skipped:
        print(f"\n  Skipped (could not parse): {skipped}")

    print("\nDone  →  Run step2_build_dataset.py next")


if __name__ == '__main__':
    main()
