"""MORENA Studio: an African-language video brief turned into a Seedance 2.5 instruction.

    modal deploy serve.py

Its own app and its own URL, separate from Pay. The two share a training corpus and nothing else:
Pay is a payments assistant whose hardest job is knowing when NOT to act, Studio only ever does one
thing and its hardest job is writing a shot someone would actually film.

Studio runs mini. It scores 100% on tool choice and 95% POSTABLE on held-out briefs against nano's
87.5%, and stays in the user's language more reliably. Mini's weakness is over-calling on small
talk, which cannot bite here because this page asks for one thing.
"""
import os

import modal

APP = "morena-studio"
GGUF = os.environ.get("MORENA_STUDIO_GGUF", "mini-tools-Q4_K_M.gguf")
MODEL_DIR = "/models"
HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake")
    .pip_install("llama-cpp-python==0.3.16", "fastapi[standard]==0.115.6")
    .add_local_dir(os.path.join(HERE, "app"), "/ui")
)
vol = modal.Volume.from_name("morena-pay-models", create_if_missing=True)  # shared weights volume
app = modal.App(APP)


@app.cls(image=image, volumes={MODEL_DIR: vol},
         cpu=4, memory=4096, scaledown_window=300, timeout=600, max_containers=4)
@modal.concurrent(max_inputs=4)
class Studio:
    @modal.enter()
    def load(self):
        from llama_cpp import Llama
        self.error = None
        self.llm = None
        try:
            vol.reload()
            path = os.path.join(MODEL_DIR, GGUF)
            if not (os.path.exists(path) and os.path.getsize(path) > 10_000_000):
                raise FileNotFoundError(
                    f"{path} missing. Push it with: modal volume put morena-pay-models {GGUF}")
            # n_threads matches the 4 cores requested; llama.cpp otherwise guesses from the host,
            # which on a shared machine means oversubscribing and getting slower.
            self.llm = Llama(model_path=path, n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse, JSONResponse

        api = FastAPI(title="MORENA Studio")

        @api.get("/")
        def index():
            return FileResponse("/ui/index.html")

        @api.get("/props")
        def props():
            if self.llm is None:
                return JSONResponse({"error": self.error}, status_code=503)
            return {"model_path": GGUF, "n_ctx": 4096}

        @api.post("/completion")
        async def completion(req: Request):
            if self.llm is None:
                return JSONResponse({"error": self.error}, status_code=503)
            b = await req.json()
            prompt = b.get("prompt") or ""
            if not prompt:
                return JSONResponse({"error": "no prompt"}, status_code=400)
            # A public URL is an open text box, so the ceilings sit where the demo lives rather
            # than where the model could go.
            # Studio writes a whole shot list, so it needs more room than a single tool call.
            n = min(int(b.get("n_predict") or 420), 520)
            out = self.llm(
                prompt[-14000:],
                max_tokens=n,
                temperature=float(b.get("temperature") or 0.0),
                top_p=float(b.get("top_p") or 1.0),
                stop=b.get("stop") or ["<reserved_0>", "<reserved_5>", "<|tool_result|>"],
                echo=False,
            )
            return {"content": out["choices"][0]["text"]}

        return api
