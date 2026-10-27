"""Minimal Nanbeige4.2-3B loading test on Modal (NO vLLM / Ollama / API).

Run from your Mac (after `modal setup`):
    modal run nanbeige_modal.py --stage download  # CPU: cache checkpoint in Volume
    modal run nanbeige_modal.py --stage load      # L4 GPU: load model, no generation

Or run both steps sequentially:
    modal run nanbeige_modal.py

NOTE: `modal deploy nanbeige_modal.py` only registers the functions. It does NOT
load model weights or create a network endpoint. Use `modal run` to test loading.
"""

import modal

APP_NAME = "nanbeige42-model-load-test"
MODEL_ID = "Nanbeige/Nanbeige4.2-3B"  # Instruction/agent checkpoint, not -Base
CACHE_DIR = "/model-cache/huggingface"

app = modal.App(APP_NAME)

# No custom CUDA/vLLM compilation. PyTorch installs from prebuilt wheels.
# Official Nanbeige model card specifies transformers==4.45.1.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==4.45.1",
    "accelerate==1.1.1",
    "huggingface_hub==0.26.5",
    "sentencepiece==0.2.0",
    "safetensors==0.4.5",
)

# Persist downloaded weights across test runs.
model_volume = modal.Volume.from_name("nanbeige42-model-cache", create_if_missing=True)
VOLUMES = {"/model-cache": model_volume}


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes=VOLUMES,
)
def download_model():
    """Download files only, using CPU; no model instantiation."""
    from huggingface_hub import snapshot_download

    print(f"[DOWNLOAD] Downloading {MODEL_ID} to persistent Modal Volume...", flush=True)
    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        max_workers=4,
    )
    model_volume.commit()  # Ensure next Modal container can see the checkpoint.
    print(f"[DOWNLOAD_OK] Model checkpoint cached: {local_dir}", flush=True)
    return "DOWNLOAD_OK"


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24576,
    timeout=1800,
    volumes=VOLUMES,
)
def load_model():
    """Load tokenizer and model into GPU memory; do NOT run inference."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_volume.reload()
    if not torch.cuda.is_available():
        raise RuntimeError("Modal GPU is not available (torch.cuda.is_available() is False)")

    print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[TORCH] {torch.__version__}", flush=True)

    # Fail clearly if `--stage download` hasn't successfully completed.
    snapshot_dir = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    print("[LOAD] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_dir,
        use_fast=False,
        trust_remote_code=True,
    )

    print("[LOAD] Loading model weights onto L4 GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        torch_dtype="auto",
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Verify parameters actually reside on the GPU, not merely on CPU.
    devices = sorted({str(p.device) for p in model.parameters()})
    if any(not dev.startswith("cuda:") for dev in devices):
        raise RuntimeError(f"Weights are not all on GPU: {devices}")

    param_count = sum(p.numel() for p in model.parameters())
    allocated_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"[LOAD_SUCCESS] Model: {MODEL_ID}", flush=True)
    print(f"[LOAD_SUCCESS] Parameter devices: {devices}", flush=True)
    print(f"[LOAD_SUCCESS] Parameter count: {param_count:,}", flush=True)
    print(f"[LOAD_SUCCESS] Tokenizer vocab size: {len(tokenizer):,}", flush=True)
    print(f"[LOAD_SUCCESS] CUDA memory allocated: {allocated_gb:.2f} GiB", flush=True)
    print("[NOTE] No text generation and no HTTP endpoint were started.", flush=True)

    # Function exits here. Modal can release GPU; model files remain in Volume.
    return "LOAD_SUCCESS"


@app.local_entrypoint()
def main(stage: str = "all"):
    if stage not in {"all", "download", "load"}:
        raise ValueError("--stage must be one of: all, download, load")

    if stage in {"all", "download"}:
        print(download_model.remote())
    if stage in {"all", "load"}:
        print(load_model.remote())
