"""Standalone Nanbeige4.2-3B text-generation smoke test on Modal.

Reuses the checkpoint downloaded by your existing `nanbeige_modal.py` into
Modal Volume `nanbeige42-model-cache`. Does not modify that script or
start vLLM, Ollama, or an HTTP server.

From ~/Projects/nanbeige-modal, with .venv activated:
    modal run nanbeige_generate_test.py > generate_log.txt 2>&1
    modal run nanbeige_generate_test.py \
        --prompt "What is GNU Radio? Answer in one sentence." \
        --max-new-tokens 96 > generate_log.txt 2>&1

Each `modal run` creates a temporary Modal App and releases its GPU when done.
"""

import modal

APP_NAME = "nanbeige42-generate-test"
MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
CACHE_DIR = "/model-cache/huggingface"

app = modal.App(APP_NAME)

# Same image dependencies as the previously successful download/load test.
# Uses prebuilt PyTorch wheels; does NOT compile or install vLLM/Ollama.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==4.45.1",
    "accelerate==1.1.1",
    "huggingface_hub==0.26.5",
    "sentencepiece==0.2.0",
    "safetensors==0.4.5",
)

# Exactly the same existing volume used by nanbeige_modal.py.
# Fail rather than silently creating an empty one if it is missing.
model_volume = modal.Volume.from_name(
    "nanbeige42-model-cache", create_if_missing=False
)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24576,
    timeout=1800,
    volumes={"/model-cache": model_volume},
)
def generate_text(prompt: str, max_new_tokens: int = 96):
    """Load the cached model once, generate a short reply, then exit."""
    import time

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 1 and 256 for this test")

    # Read files exclusively from the persistent Modal Volume.
    model_volume.reload()
    snapshot_dir = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available in Modal container")
    print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[MODEL] {MODEL_ID}", flush=True)
    print("[LOAD] Loading tokenizer and cached model weights...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_dir, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        torch_dtype="auto",
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("[LOAD_SUCCESS] Model is on GPU", flush=True)

    # The instruction-tuned Nanbeige model uses a chat template.
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    inputs = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda:0")

    print(f"[PROMPT] {prompt}", flush=True)
    print(f"[GENERATE] max_new_tokens={max_new_tokens}", flush=True)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=166101,  # From Nanbeige4.2's official model card
            pad_token_id=166101,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    prompt_tokens = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_tokens:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"[GENERATED_TOKENS] {generated_ids.numel()}", flush=True)
    print(f"[GENERATION_SECONDS] {elapsed:.2f}", flush=True)
    if elapsed > 0:
        print(f"[OUTPUT_TOKENS_PER_SECOND] {generated_ids.numel() / elapsed:.2f}", flush=True)
    print(f"[RESPONSE] {response}", flush=True)

    if not response:
        raise RuntimeError("Model returned no visible text")
    print("[GENERATE_SUCCESS]", flush=True)
    return "GENERATE_SUCCESS"


@app.local_entrypoint()
def main(
    prompt: str = "What is GNU Radio? Answer in one sentence.",
    max_new_tokens: int = 96,
):
    result = generate_text.remote(prompt, max_new_tokens)
    print(result)
