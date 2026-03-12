"""
inspect_long_records.py
========================
Reads echo_dataset_sft.jsonl and shows every record that would be
DROPPED by the training script at MAXLEN=2048.

Run this locally after txt_to_jsonl.py produces the JSONL.
Point INPUT_FILE at your echo_dataset_sft.jsonl.

Output:
  - Summary table of all records over the limit
  - Full content of each dropped record (so you can decide
    whether to split, trim, or just accept the loss)
"""

import json
from pathlib import Path

# ── Config — update this path ──────────────────────────────────────────────────
INPUT_FILE = Path(r"C:\Users\tavar\OneDrive\Documentos\LLMs\Echo\Data\Llama_Echo memories\old\echo_dataset_sft.jsonl")

MAXLEN         = 3072   # must match training script
CHARS_PER_TOKEN = 3.5   # same estimator as txt_to_jsonl.py

# ── Helpers ────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)

def estimate_tokens_messages(messages: list) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(m["content"])
        total += 10
    return total

def role_label(role: str) -> str:
    return {"system": "SYS", "user": "USR", "assistant": "AST"}.get(role, role.upper())

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] File not found: {INPUT_FILE}")
        print("Update INPUT_FILE at the top of this script.")
        return

    records = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["_index"] = i
                records.append(rec)

    print(f"Total records in JSONL : {len(records)}")
    print(f"MAXLEN threshold       : {MAXLEN} tokens\n")

    # ── Find the ones that would be dropped ────────────────────────────────────
    # Training script logic:
    #   1. Truncate to MAXLEN
    #   2. Drop if len(input_ids) >= MAXLEN   (i.e. it hit the limit)
    # We approximate with the token estimator.

    over_limit = []
    for rec in records:
        msgs = rec["messages"]
        est  = estimate_tokens_messages(msgs)
        if est >= MAXLEN:
            over_limit.append((rec["_index"], est, msgs))

    if not over_limit:
        print(f"✅  No records exceed {MAXLEN} tokens — nothing would be dropped.")
        return

    print(f"⚠️  {len(over_limit)} record(s) exceed {MAXLEN} estimated tokens:\n")
    print(f"  {'#':>4}  {'JSONL line':>10}  {'Est. tokens':>11}  {'Turns':>5}  Preview")
    print(f"  {'─'*4}  {'─'*10}  {'─'*11}  {'─'*5}  {'─'*40}")

    for idx, (line_no, est, msgs) in enumerate(over_limit):
        # Count non-system turns
        turns = sum(1 for m in msgs if m["role"] != "system")
        # First user message preview
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        preview = first_user[:60].replace("\n", " ")
        if len(first_user) > 60:
            preview += "…"
        print(f"  {idx+1:>4}  {line_no:>10}  {est:>11,}  {turns:>5}  {preview}")

    print(f"\n{'═'*68}")
    print(f"FULL CONTENT OF DROPPED RECORDS")
    print(f"{'═'*68}")

    for idx, (line_no, est, msgs) in enumerate(over_limit):
        print(f"\n{'─'*68}")
        print(f"  Record {idx+1}  (JSONL line {line_no})  ~{est:,} estimated tokens")
        print(f"{'─'*68}")

        for m in msgs:
            label   = role_label(m["role"])
            content = m["content"]

            if m["role"] == "system":
                # Just show first line of system prompt — it's always the same
                first_line = content.split("\n")[0]
                print(f"\n  [{label}] {first_line}  [...system prompt truncated...]")
            else:
                print(f"\n  [{label}]")
                # Word-wrap at 80 chars for readability
                for line in content.split("\n"):
                    if len(line) <= 80:
                        print(f"    {line}")
                    else:
                        words = line.split()
                        current = "    "
                        for word in words:
                            if len(current) + len(word) + 1 > 84:
                                print(current)
                                current = f"    {word} "
                            else:
                                current += word + " "
                        if current.strip():
                            print(current)

        print()

    print(f"\n{'═'*68}")
    print(f"SUMMARY")
    print(f"{'═'*68}")
    print(f"  Records over {MAXLEN} tokens : {len(over_limit)}")
    print(f"  Total records              : {len(records)}")
    print(f"  % dropped                  : {100*len(over_limit)/len(records):.1f}%")
    print(f"\nOptions for each dropped record:")
    print(f"  1. Accept the drop — if it's an outlier, training quality is unaffected")
    print(f"  2. Manually trim the conversation in your source .txt file")
    print(f"  3. Lower MAX_TOKENS in txt_to_jsonl.py so the splitter chunks it")
    print(f"     (set MAX_TOKENS = 2048 to match MAXLEN in the training script)")
    print(f"\nNote: if you set MAX_TOKENS=2048 in txt_to_jsonl.py, long conversations")
    print(f"will be automatically SPLIT into overlapping chunks instead of dropped.")


if __name__ == "__main__":
    main()
