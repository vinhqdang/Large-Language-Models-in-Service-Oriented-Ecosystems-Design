"""LLM client abstraction for deliberation agents.

Both clients implement the same duck-typed interface:
    generate(prompt: str, system: str | None = None) -> str

Real model/SDK loading happens only in the load_* factory functions below,
never at import time, and (for the local backend) never at module level —
so importing this module never pulls in torch, and never conflicts with
the sentence_transformers-before-torch import-order rule documented in
PROGRESS.md. Callers that use both retrieval (Stage 1) and the local
backend (this module) in the same process must still import
src.retrieval.embeddings (or sentence_transformers directly) before
calling load_local_hf_client.
"""
import os

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class GeminiClient:
    def __init__(self, client, model_name: str = DEFAULT_GEMINI_MODEL):
        self._client = client
        self._model_name = model_name

    def generate(self, prompt: str, system: str | None = None) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system) if system else None
        response = self._client.models.generate_content(
            model=self._model_name, contents=prompt, config=config,
        )
        return response.text


def load_gemini_client(api_key: str | None = None, model_name: str = DEFAULT_GEMINI_MODEL) -> GeminiClient:
    from google import genai

    api_key = api_key or os.environ["GEMINI_API_KEY"]
    return GeminiClient(genai.Client(api_key=api_key), model_name=model_name)


class LocalHFClient:
    def __init__(self, generator):
        self._generator = generator

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._generator(messages)


def load_local_hf_client(model_name: str = DEFAULT_LOCAL_MODEL) -> LocalHFClient:
    """Load a local instruct model directly via AutoTokenizer/AutoModelForCausalLM.

    Deliberately does NOT use transformers.pipeline("text-generation", ...):
    on this machine that high-level wrapper segfaults unpredictably (native
    crash, not a Python exception) at inconsistent points — sometimes during
    import, sometimes during model load, sometimes during generation — while
    the lower-level tokenizer/model/.generate() path used here has been
    verified reliable across repeated runs on both CPU and CUDA, including
    multiple sequential generate calls against one loaded model (the actual
    usage pattern here: agents share one loaded model across many calls).
    Root cause not identified; this is a working, verified alternative, not
    a guess. See PROGRESS.md's environment notes if this needs revisiting.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype="auto")

    def generator(messages):
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(
            input_ids, max_new_tokens=400, do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return LocalHFClient(generator)
