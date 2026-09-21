"""MORENA Studio: an African-language video brief turned into a Seedance 2.5 instruction.

    modal deploy serve.py

Its own app and its own URL, separate from Pay. The two share a training corpus and nothing else:
Pay is a payments assistant whose hardest job is knowing when NOT to act, Studio only ever does one
thing and its hardest job is writing a shot someone would actually film.

Studio runs mini. It scores 100% on tool choice and 95% POSTABLE on held-out briefs against nano's
87.5%, and stays in the user's language more reliably. Mini's weakness is over-calling on small
talk, which cannot bite here because this page asks for one thing.
"""
import json
import os
import urllib.error
import urllib.request

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

# Seedance itself, through Replicate. The model writes the instruction; this renders it, so the
# page can show the clip rather than asking you to imagine it.
#
# The token lives in a Modal secret named `replicate`, holding REPLICATE_API_TOKEN. It is not in
# this repository and not in the image. If the secret is absent the app still runs and /render
# says plainly that rendering is switched off, because the instruction is the interesting part and
# should not stop working because a billing key is missing.
REPLICATE_MODEL = os.environ.get("REPLICATE_MODEL", "bytedance/seedance-2.5")
def _replicate_secret():
    try:
        import subprocess
        out = subprocess.run(["modal", "secret", "list"], capture_output=True, text=True, timeout=25)
        if out.returncode == 0 and "replicate" in out.stdout:
            return [modal.Secret.from_name("replicate")]
    except Exception:
        pass
    return []


SECRETS = _replicate_secret()
app = modal.App(APP)


@app.cls(image=image, volumes={MODEL_DIR: vol}, secrets=SECRETS,
         cpu=4, memory=4096, scaledown_window=300, timeout=900, max_containers=4)
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

        def replicate_call(method, path, body=None):
            tok = os.environ.get("REPLICATE_API_TOKEN")
            if not tok:
                return None, "no token"
            req = urllib.request.Request(
                "https://api.replicate.com/v1" + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Authorization": "Bearer " + tok,
                         "Content-Type": "application/json",
                         "Prefer": "wait=1"},
                method=method)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read()), None
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                return None, f"replicate {e.code}: {detail}"
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"

        @api.get("/render/enabled")
        def render_enabled():
            return {"enabled": bool(os.environ.get("REPLICATE_API_TOKEN")),
                    "model": REPLICATE_MODEL}

        @api.post("/render")
        async def render(req: Request):
            """Send the arguments the model produced to Seedance, unchanged.

            The whole claim is that the instruction is postable as written, so nothing here
            rewrites it. Fields Seedance does not take are dropped rather than renamed, and if
            that drops something important the fix belongs in the fine-tune, not in a shim."""
            if not os.environ.get("REPLICATE_API_TOKEN"):
                return JSONResponse(
                    {"error": "Rendering is off: no Replicate token is configured on this "
                              "deployment. The instruction above is still exactly what you would "
                              "post to the API."}, status_code=503)
            b = await req.json()
            args = b.get("arguments") or {}
            if not args.get("prompt"):
                return JSONResponse({"error": "no prompt"}, status_code=400)
            allowed = ("prompt", "duration", "resolution", "aspect_ratio", "seed",
                       "camera_fixed", "generate_audio", "image", "last_frame_image")
            payload = {k: v for k, v in args.items() if k in allowed and v is not None}
            out, err = replicate_call("POST", f"/models/{REPLICATE_MODEL}/predictions",
                                      {"input": payload})
            if err:
                return JSONResponse({"error": err}, status_code=502)
            return {"id": out.get("id"), "status": out.get("status"),
                    "output": out.get("output"), "sent": payload}

        @api.get("/render/{pid}")
        def render_status(pid: str):
            out, err = replicate_call("GET", f"/predictions/{pid}")
            if err:
                return JSONResponse({"error": err}, status_code=502)
            return {"id": out.get("id"), "status": out.get("status"),
                    "output": out.get("output"), "error": out.get("error")}

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
