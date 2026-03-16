import json, re
from pathlib import Path
from collections import defaultdict

DATASET = Path("echo_dataset_sft_clean.jsonl")

with open(DATASET, encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

results = []
for i, rec in enumerate(records):
    for msg in rec["messages"]:
        if msg["role"] != "assistant":
            continue
        c = msg["content"]
        if re.search(r'\n#{1,3}\s+\w', c):
            results.append(f"Record {i}: markdown header in response\n{c[:200]}\n---")
        paras = [p.strip() for p in c.split('\n\n') if len(p.strip()) > 60]
        seen = set()
        for p in paras:
            if p[:80] in seen:
                results.append(f"Record {i}: repeated paragraph\n{p[:150]}\n---")
            seen.add(p[:80])

Path("loop_report.txt").write_text('\n'.join(results) if results else "No issues found!", encoding="utf-8")
print(f"Found {len(results)} issues. See loop_report.txt")