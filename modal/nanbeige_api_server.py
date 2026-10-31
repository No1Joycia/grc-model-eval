"""Step 4: Serve the already-cached Nanbeige4.2-3B over an HTTPS chat API.

Uses: Modal + an L4 GPU + PyTorch/Transformers + FastAPI.
Does NOT use vLLM/Ollama or modify nanbeige_modal.py.

Important scope:
  - Implements GET /health, GET /v1/models, POST /v1/chat/completions.
  - Supports non-streaming TEXT chat (system/user/assistant messages).
  - Deliberately REJECTS tool calls, tool messages, and streaming requests.
    It is NOT yet compatible with GRC-Agent's tool-calling workflow.
  - Requires a Modal Proxy Token on every request.

Run from your Mac's nanbeige-modal virtual environment:
    modal deploy modal/nanbeige_api_server.py

The deployed https://...modal.run URL is printed at the end of the deploy output.
Then authenticate using a workspace Proxy Token (wk-... + ws-...). See
https://modal.com/docs/guide/webhook-proxy-auth .
"""

import modal

APP_NAME = "nanbeige42-chat-api"
MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
CACHE_DIR = "/model-cache/huggingface"

app = modal.App(APP_NAME)

# Use the same known-good versions from the completed Download/Load/Generate tests.
# The only added dependency is FastAPI for the HTTP endpoints.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==4.45.1",
    "accelerate==1.1.1",
    "huggingface_hub==0.26.5",
    "sentencepiece==0.2.0",
    "safetensors==0.4.5",
    "fastapi==0.115.12",
)

# IMPORTANT: reuse the existing Volume; fail if you haven't downloaded weights.
model_volume = modal.Volume.from_name(
    "nanbeige42-model-cache", create_if_missing=False
)


@app.function(
    image=image,
    gpu=["L4", "A10G", "L40S"],  # fallback order if L4 capacity is unavailable
    cpu=4,
    memory=24576,
    timeout=600,
    startup_timeout=600,
    min_containers=0,      # no always-on GPU
    max_containers=1,      # prevent accidentally starting several L4s
    scaledown_window=60,  # idle GPU may scale down after ~60 s
    volumes={"/model-cache": model_volume},
)
@modal.asgi_app(requires_proxy_auth=True)
def serve():
    """Build the ASGI app once per started GPU container, with a loaded model."""
    import copy
    import threading
    import time
    import uuid
    from typing import Any, Literal, Optional

    import torch
    from fastapi import FastAPI, HTTPException
    from huggingface_hub import snapshot_download
    from pydantic import BaseModel, Field
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not provide the requested L4 GPU")

    model_volume.reload()
    snapshot_dir = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
    print(f"[STARTUP] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[STARTUP] Loading cached model: {MODEL_ID}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_dir,
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        torch_dtype="auto",
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("[SERVER_READY] Model is loaded on L4", flush=True)

    # Single-GPU, single-request-at-a-time MVP: serialize model.generate calls.
    inference_lock = threading.Lock()

    class ChatMessage(BaseModel):
        role: Literal["system", "user", "assistant"]
        content: str

    class CompletionRequest(BaseModel):
        model: str
        messages: list[ChatMessage] = Field(min_length=1)
        max_tokens: Optional[int] = Field(default=None, ge=1, le=2048)
        max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=2048)
        temperature: Optional[float] = Field(default=None, ge=0, le=2)
        top_p: Optional[float] = Field(default=None, gt=0, le=1)
        stream: bool = False
        tools: Optional[list[dict[str, Any]]] = None
        tool_choice: Optional[Any] = None
        response_format: Optional[dict[str, Any]] = None

    api = FastAPI(title="Nanbeige4.2 Modal Chat MVP")

    @api.get("/health")
    def health():
        return {"status": "ready", "model": MODEL_ID, "gpu": "L4"}

    @api.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "Nanbeige"}],
        }

    @api.post("/v1/chat/completions")
    def chat_completions(req: CompletionRequest):
        if req.model != MODEL_ID:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported in this MVP")
        if req.tools:
            raise HTTPException(
                status_code=400,
                detail="tools/tool calling is not supported yet; Step 5 requires a tool parser",
            )
        if req.tool_choice not in (None, "none"):
            raise HTTPException(status_code=400, detail="tool_choice is not supported yet")
        if req.response_format is not None:
            raise HTTPException(status_code=400, detail="response_format is not supported yet")

        max_new_tokens = req.max_completion_tokens or req.max_tokens or 128
        chat = [msg.model_dump() for msg in req.messages]
        try:
            rendered = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            inputs = tokenizer(
                rendered, add_special_tokens=False, return_tensors="pt"
            ).to("cuda:0")
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Unable to format chat messages: {exc}"
            ) from exc

        prompt_len = int(inputs["input_ids"].shape[1])
        if prompt_len > 4096:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long for this MVP: {prompt_len} tokens (limit 4096)",
            )

        config = copy.deepcopy(model.generation_config)
        temp = req.temperature
        do_sample = temp is not None and temp > 0
        config.do_sample = do_sample
        if do_sample:
            config.temperature = temp
            config.top_p = req.top_p if req.top_p is not None else 0.95
            config.top_k = 20
        else:
            # Normalize saved sampling defaults to avoid irrelevant warnings.
            config.temperature = 1.0
            config.top_p = 1.0
            config.top_k = 50

        started = time.perf_counter()
        with inference_lock, torch.inference_mode():
            output = model.generate(
                **inputs,
                generation_config=config,
                max_new_tokens=max_new_tokens,
                eos_token_id=166101,
                pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else 166101,
            )
        elapsed = time.perf_counter() - started

        generated_ids = output[0, prompt_len:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = int(generated_ids.numel())
        finish_reason = (
            "stop" if generated_ids.numel() and int(generated_ids[-1]) == 166101
            else "length" if completion_tokens >= max_new_tokens
            else "stop"
        )
        print(
            f"[REQUEST_OK] prompt={prompt_len} generated={completion_tokens} "
            f"seconds={elapsed:.2f}",
            flush=True,
        )
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_len + completion_tokens,
            },
        }

    return api
