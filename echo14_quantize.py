#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║      ECHO 14  ·  Standalone Quantize Script  ·  Kaggle T4 × 2      ║
║      Clean bfloat16 merge → GGUF Q4_K_M  ·  Uses /kaggle/tmp       ║
╚══════════════════════════════════════════════════════════════════════╝

SETUP:
  1. Add your LoRA dataset as Kaggle input.
  2. Confirm LORA_INPUT_PATH below matches the Input panel path.
  3. Set accelerator to T4 × 2.
  4. Run all — ~30-40 min total.

OUTPUT:
  /kaggle/working/echo14_Q4_K_M.gguf  (~4.8 GB) ← download this

SPACE USAGE (all heavy files in /kaggle/tmp ~90 GB):
  lora copy        ~0.2 GB
  merged bf16      ~16  GB
  f16 GGUF         ~16  GB  (deleted right after Q4 is made)
  Q4_K_M GGUF      ~4.8 GB
  llama.cpp build  ~1   GB
  Peak: ~38 GB — fits in /kaggle/tmp fine.
  /kaggle/working only gets the final 4.8 GB file.
"""

import os, sys, gc, json, shutil, subprocess, time, zipfile, glob

# ══════════════════════════════════════════════════════════════════════════════
#  !! ONLY CHANGE THIS — path shown in the Kaggle Input panel !!
# ══════════════════════════════════════════════════════════════════════════════

LORA_INPUT_PATH = "/kaggle/input/datasets/davidtramalho/echolora15/lora_echo14"

MODEL_ID = "LiquidAI/LFM2-8B-A1B"

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS — everything heavy → /kaggle/tmp, only final GGUF → /kaggle/working
# ══════════════════════════════════════════════════════════════════════════════

TMP        = "/kaggle/tmp"
LORA_DIR   = f"{TMP}/lora_echo14"
MERGED_DIR = f"{TMP}/merged_echo14"
GGUF_DIR   = f"{TMP}/gguf_echo14"
LLAMA_DIR  = f"{TMP}/llama.cpp"
F16_GGUF   = f"{GGUF_DIR}/echo14_f16.gguf"
Q4_GGUF    = f"{GGUF_DIR}/echo14_Q4_K_M.gguf"
FINAL_GGUF = "/kaggle/working/echo14_Q4_K_M.gguf"

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

DIV = "─" * 68

def section(title):
    print(f"\n{DIV}\n  {title}\n{DIV}", flush=True)

def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    ret = subprocess.call(cmd, **kwargs)
    if ret != 0:
        raise RuntimeError(f"Command failed (exit {ret}): {cmd}")

def disk_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9

def vram_report():
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            resv  = torch.cuda.memory_reserved(i)  / 1e9
            print(f"  GPU {i}: {alloc:.2f} GB alloc | {resv:.2f} GB reserved", flush=True)
    except Exception:
        pass

def disk_space_report():
    section("Disk space check")
    for path, label in [
        ("/kaggle/tmp",     "tmp  (~90 GB)"),
        ("/kaggle/working", "working (~19 GB)"),
    ]:
        try:
            stat  = os.statvfs(path)
            free  = stat.f_frsize * stat.f_bavail / 1e9
            total = stat.f_frsize * stat.f_blocks / 1e9
            print(f"  {label:25s}  free: {free:.1f} GB / {total:.1f} GB", flush=True)
        except Exception as e:
            print(f"  {label}: could not stat ({e})", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════

def step1_install():
    section("Step 1 / 8  —  Install dependencies")
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    run(pip + ["protobuf>=3.20.3,<7.0.0"])
    run(pip + [
        "--upgrade",
        "transformers>=4.55.0",
        "accelerate>=0.34.0",
        "peft>=0.11.0",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "tokenizers",        # needed for TokenizersBackend fast tokenizer
    ])
    print("  ✅  Dependencies ready.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — LOCATE / COPY LORA TO /kaggle/tmp
# ══════════════════════════════════════════════════════════════════════════════

def step2_locate_lora():
    section("Step 2 / 8  —  Locate LoRA adapter")
    os.makedirs(TMP, exist_ok=True)

    def _copy_dir(src):
        print(f"  Copying {src} → {LORA_DIR} …", flush=True)
        if os.path.exists(LORA_DIR):
            shutil.rmtree(LORA_DIR)
        shutil.copytree(src, LORA_DIR)
        print(f"  ✅  LoRA ready at: {LORA_DIR}", flush=True)
        print(f"  Contents: {sorted(os.listdir(LORA_DIR))}", flush=True)

    def _extract_zip(zip_path):
        print(f"  Extracting {zip_path} → {TMP} …", flush=True)
        if os.path.exists(LORA_DIR):
            shutil.rmtree(LORA_DIR)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(TMP)
        hits = glob.glob(f"{TMP}/**/adapter_config.json", recursive=True)
        if not hits:
            raise FileNotFoundError("Extracted zip but no adapter_config.json found inside.")
        found = os.path.dirname(hits[0])
        if os.path.realpath(found) != os.path.realpath(LORA_DIR):
            if os.path.exists(LORA_DIR):
                shutil.rmtree(LORA_DIR)
            shutil.copytree(found, LORA_DIR)
        print(f"  ✅  LoRA ready at: {LORA_DIR}", flush=True)
        print(f"  Contents: {sorted(os.listdir(LORA_DIR))}", flush=True)

    # Case A: already an extracted folder with adapter_config.json
    if os.path.isdir(LORA_INPUT_PATH):
        if os.path.exists(os.path.join(LORA_INPUT_PATH, "adapter_config.json")):
            print(f"  ✅  Found extracted LoRA at: {LORA_INPUT_PATH}", flush=True)
            _copy_dir(LORA_INPUT_PATH)
            return
        # adapter one level deeper
        hits = glob.glob(os.path.join(LORA_INPUT_PATH, "**/adapter_config.json"), recursive=True)
        if hits:
            _copy_dir(os.path.dirname(hits[0]))
            return

    # Case B: configured path is a zip
    if os.path.isfile(LORA_INPUT_PATH) and LORA_INPUT_PATH.endswith(".zip"):
        _extract_zip(LORA_INPUT_PATH)
        return

    # Case C: auto-search /kaggle/input for adapter_config.json
    print(f"  Path not found directly — scanning /kaggle/input …", flush=True)
    hits = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
    if hits:
        found = os.path.dirname(hits[0])
        print(f"  Auto-found LoRA at: {found}", flush=True)
        _copy_dir(found)
        return

    # Case D: auto-search for any zip
    zip_hits = glob.glob("/kaggle/input/**/*lora*.zip", recursive=True)
    if not zip_hits:
        zip_hits = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if zip_hits:
        _extract_zip(zip_hits[0])
        return

    raise FileNotFoundError(
        f"\n  Could not find LoRA adapter anywhere in /kaggle/input!\n"
        f"  Checked: {LORA_INPUT_PATH}\n"
        f"  Update LORA_INPUT_PATH at the top of this script."
    )

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3+4 — LOAD CLEAN BASE + MERGE LORA + SAVE
# ══════════════════════════════════════════════════════════════════════════════

def step3_merge():
    section("Step 3+4 / 8  —  Load clean bfloat16 base + merge LoRA")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if os.path.exists(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)
    os.makedirs(MERGED_DIR, exist_ok=True)

    print(f"  CUDA available : {torch.cuda.is_available()}", flush=True)
    print(f"  GPU count      : {torch.cuda.device_count()}", flush=True)
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  ({mem:.1f} GB)", flush=True)

    print(f"\n  Loading tokenizer …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Clean bfloat16 — NO BitsAndBytes — so merge produces clean tensors
    print(f"\n  Loading {MODEL_ID} in bfloat16 (no quantization) …", flush=True)
    print(f"  Splitting across both T4s (14 GiB each) …", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        max_memory        = {0: "14GiB", 1: "14GiB"},
        low_cpu_mem_usage = True,
        trust_remote_code = True,
    )
    base.config.use_cache = False
    vram_report()

    print(f"\n  Loading LoRA from {LORA_DIR} …", flush=True)
    model = PeftModel.from_pretrained(base, LORA_DIR)

    print(f"\n  Merging LoRA into base weights …", flush=True)
    merged = model.merge_and_unload()

    # Strip quantization config just in case
    if hasattr(merged, "config"):
        merged.config.__dict__.pop("quantization_config", None)
        if hasattr(merged.config, "quantization_config"):
            merged.config.quantization_config = None

    print(f"\n  Saving merged model to {MERGED_DIR} …", flush=True)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tok.save_pretrained(MERGED_DIR)

    size = disk_gb(MERGED_DIR)
    print(f"\n  ✅  Merged model saved → {MERGED_DIR}  ({size:.1f} GB)", flush=True)

    print(f"\n  Freeing VRAM …", flush=True)
    del merged, model, base
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    vram_report()

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4b — PATCH TOKENIZER CONFIG
#
#  The LFM2 tokenizer_config.json sets:
#    "tokenizer_class": "TokenizersBackend"
#  which convert_hf_to_gguf.py tries to import from transformers and fails.
#  Fix: remove that field so the converter falls back to auto-detection.
#  We also copy the chat_template.jinja from the LoRA dir if present.
# ══════════════════════════════════════════════════════════════════════════════

def step4b_patch_tokenizer():
    section("Step 4b / 8  —  Patch tokenizer_config.json for GGUF converter")

    tok_cfg_path = os.path.join(MERGED_DIR, "tokenizer_config.json")

    if not os.path.exists(tok_cfg_path):
        print(f"  tokenizer_config.json not found — skipping patch.", flush=True)
        return

    with open(tok_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    patched = False

    # Remove tokenizer_class if it is the problematic TokenizersBackend
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        print(f"  Removing 'tokenizer_class': 'TokenizersBackend' …", flush=True)
        del cfg["tokenizer_class"]
        patched = True

    # Some versions also have auto_map pointing to TokenizersBackend
    if "auto_map" in cfg:
        auto = cfg["auto_map"]
        bad_keys = [k for k, v in auto.items() if "TokenizersBackend" in str(v)]
        for k in bad_keys:
            print(f"  Removing auto_map['{k}'] = '{auto[k]}' …", flush=True)
            del auto[k]
            patched = True
        if not auto:
            del cfg["auto_map"]

    if patched:
        with open(tok_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"  ✅  tokenizer_config.json patched.", flush=True)
    else:
        print(f"  tokenizer_config.json looks fine — no patch needed.", flush=True)

    # Copy chat_template.jinja from LoRA dir if it exists and isn't in merged
    jinja_src = os.path.join(LORA_DIR, "chat_template.jinja")
    jinja_dst = os.path.join(MERGED_DIR, "chat_template.jinja")
    if os.path.exists(jinja_src) and not os.path.exists(jinja_dst):
        shutil.copy2(jinja_src, jinja_dst)
        print(f"  Copied chat_template.jinja from LoRA dir.", flush=True)

    print(f"\n  Final tokenizer_config.json keys: {sorted(cfg.keys())}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — BUILD LLAMA.CPP (CPU-only — avoids Kaggle CUDA linker bug)
# ══════════════════════════════════════════════════════════════════════════════

def step5_build_llama_cpp():
    section("Step 5 / 8  —  Clone + build llama.cpp (CPU-only)")

    quantize_bin = os.path.join(LLAMA_DIR, "build", "bin", "llama-quantize")
    if os.path.exists(quantize_bin):
        print(f"  llama.cpp already built — skipping.", flush=True)
        return quantize_bin

    if os.path.exists(LLAMA_DIR):
        shutil.rmtree(LLAMA_DIR)

    print(f"  Cloning into {LLAMA_DIR} …", flush=True)
    run(["git", "clone", "--depth=1",
         "https://github.com/ggerganov/llama.cpp",
         LLAMA_DIR])

    req = os.path.join(LLAMA_DIR, "requirements.txt")
    if os.path.exists(req):
        print(f"  Installing Python requirements …", flush=True)
        subprocess.call([sys.executable, "-m", "pip",
                         "install", "--quiet", "-r", req])

    build_dir = os.path.join(LLAMA_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    print(f"\n  CMake configure (CPU only) …", flush=True)
    run(["cmake", "..",
         "-DGGML_CUDA=OFF",
         "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir)

    print(f"\n  Building with {os.cpu_count()} cores …", flush=True)
    run(["cmake", "--build", ".",
         "--config", "Release",
         f"-j{os.cpu_count()}"],
        cwd=build_dir)

    if not os.path.exists(quantize_bin):
        raise RuntimeError(
            f"Build completed but llama-quantize not found at:\n  {quantize_bin}"
        )

    print(f"  ✅  llama.cpp built successfully.", flush=True)
    return quantize_bin

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — CONVERT MERGED MODEL → F16 GGUF
# ══════════════════════════════════════════════════════════════════════════════

def step6_convert_f16():
    section("Step 6 / 8  —  Convert merged model → f16 GGUF")

    os.makedirs(GGUF_DIR, exist_ok=True)

    convert_script = None
    for name in ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]:
        candidate = os.path.join(LLAMA_DIR, name)
        if os.path.exists(candidate):
            convert_script = candidate
            break

    if convert_script is None:
        raise FileNotFoundError(
            f"Cannot find convert_hf_to_gguf.py in {LLAMA_DIR}.\n"
            "Did step 5 complete successfully?"
        )

    print(f"  Converter : {convert_script}", flush=True)
    print(f"  Input     : {MERGED_DIR}  ({disk_gb(MERGED_DIR):.1f} GB)", flush=True)
    print(f"  Output    : {F16_GGUF}", flush=True)

    run([sys.executable, convert_script,
         MERGED_DIR,
         "--outfile", F16_GGUF,
         "--outtype", "f16"])

    if not os.path.exists(F16_GGUF):
        raise RuntimeError(
            "f16 GGUF not created — conversion failed.\n"
            "Check output above for errors."
        )

    size = os.path.getsize(F16_GGUF) / 1e9
    print(f"\n  ✅  f16 GGUF → {F16_GGUF}  ({size:.2f} GB)", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — QUANTIZE F16 → Q4_K_M
# ══════════════════════════════════════════════════════════════════════════════

def step7_quantize(quantize_bin):
    section("Step 7 / 8  —  Quantize f16 → Q4_K_M")

    print(f"  Input  : {F16_GGUF}  ({os.path.getsize(F16_GGUF)/1e9:.2f} GB)", flush=True)
    print(f"  Output : {Q4_GGUF}", flush=True)

    run([quantize_bin, F16_GGUF, Q4_GGUF, "Q4_K_M"])

    if not os.path.exists(Q4_GGUF):
        raise RuntimeError(
            "Q4_K_M GGUF not created — quantization failed.\n"
            "Check output above for errors."
        )

    size = os.path.getsize(Q4_GGUF) / 1e9
    print(f"\n  ✅  Q4_K_M GGUF → {Q4_GGUF}  ({size:.2f} GB)", flush=True)

    # Delete f16 immediately — frees ~16 GB in /kaggle/tmp
    f16_size = os.path.getsize(F16_GGUF) / 1e9
    print(f"\n  Deleting f16 intermediate ({f16_size:.1f} GB) …", flush=True)
    os.remove(F16_GGUF)
    print(f"  Space freed.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — COPY FINAL GGUF TO /kaggle/working FOR DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def step8_copy_for_download():
    section("Step 8 / 8  —  Copy GGUF to /kaggle/working for download")

    print(f"  Copying {Q4_GGUF} → {FINAL_GGUF} …", flush=True)
    shutil.copy2(Q4_GGUF, FINAL_GGUF)

    size = os.path.getsize(FINAL_GGUF) / 1e9
    print(f"\n  ✅  FINAL FILE: {FINAL_GGUF}  ({size:.2f} GB)", flush=True)

    print(f"""
  ─────────────────────────────────────────────────────────────
  📥  HOW TO DOWNLOAD:
  1. Kaggle sidebar → Output (right panel)
  2. Expand /kaggle/working/
  3. Click ⬇️  next to echo14_Q4_K_M.gguf
  ─────────────────────────────────────────────────────────────

  OLLAMA — save as Modelfile, then: ollama create echo14 -f Modelfile
  ─────────────────────────────────────────────────────────────
  FROM ./echo14_Q4_K_M.gguf
  PARAMETER num_ctx 32768
  PARAMETER temperature 0.3
  PARAMETER repeat_penalty 1.05
  SYSTEM \"\"\"
  [paste SYSTEM_THINKING from your training script for think mode,
   or SYSTEM_BASE for non-think mode]
  \"\"\"
  ─────────────────────────────────────────────────────────────
    """, flush=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()

    print(f"\n{'═'*68}")
    print(f"  ECHO 14  ·  Standalone Q4_K_M Export  ·  Kaggle T4×2")
    print(f"{'═'*68}")
    print(f"  LORA_INPUT_PATH : {LORA_INPUT_PATH}")
    print(f"  MODEL_ID        : {MODEL_ID}")
    print(f"  LORA_DIR        : {LORA_DIR}")
    print(f"  MERGED_DIR      : {MERGED_DIR}")
    print(f"  GGUF_DIR        : {GGUF_DIR}")
    print(f"  LLAMA_DIR       : {LLAMA_DIR}")
    print(f"  FINAL_GGUF      : {FINAL_GGUF}")

    disk_space_report()

    try:
        step1_install()
        step2_locate_lora()
        step3_merge()
        step4b_patch_tokenizer()      # ← fixes TokenizersBackend error
        quantize_bin = step5_build_llama_cpp()
        step6_convert_f16()
        step7_quantize(quantize_bin)
        step8_copy_for_download()

    except Exception as e:
        print(f"\n{'═'*68}")
        print(f"  ❌  FAILED: {e}")
        print(f"{'═'*68}")
        raise

    elapsed = (time.time() - total_start) / 60
    print(f"\n{'═'*68}")
    print(f"  ✅  ALL DONE in {elapsed:.1f} min")
    print(f"  Download: {FINAL_GGUF}")
    print(f"{'═'*68}\n")
