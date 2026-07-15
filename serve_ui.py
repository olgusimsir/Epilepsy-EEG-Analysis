"""
NeuroScan web UI — a thin HTTP layer over the existing backend.

A single clinical tool for a neurologist: upload a patient EDF, analyze, and
generate a report. It reuses the same pipeline functions (src.predict,
src.risk_assessment, src.clinical_report) and serves the interface in design/.
It does NOT change any model, risk, report, or RAG logic.

Run:  python serve_ui.py     then open  http://localhost:8000
(Uses only the Python standard library for the server — no new dependencies.)
"""
import base64
import json
import os
import sys
import tempfile
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import load_checkpoint, predict_recording
from src.risk_assessment import assess_recording, smooth, SMOOTH_K
from src.clinical_report import build_report

WINDOW_SEC = 5
OVERLAP = 0.5     # 50% overlapping windows at inference → better boundary sensitivity
DESIGN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "design")

MODELS = {
    "single": "models/seizure_cnn.pt",
    "cross": "models/seizure_cnn2_final.pt",
}

_MODEL_CACHE = {}          # path -> (model, ckpt)
_ANALYSES = OrderedDict()  # file_key -> {"assessment":..., "filename":...} (LRU-capped)
MAX_ANALYSES = 32          # bound the cache so a long-running server can't leak memory


def get_model(kind):
    path = MODELS.get(kind, MODELS["cross"])
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = load_checkpoint(path)
    return _MODEL_CACHE[path]


def _remember(file_key, assessment, filename):
    """Cache an analysis for the follow-up report call, LRU-evicting old entries."""
    _ANALYSES[file_key] = {"assessment": assessment, "filename": filename}
    _ANALYSES.move_to_end(file_key)
    while len(_ANALYSES) > MAX_ANALYSES:
        _ANALYSES.popitem(last=False)


def run_analysis(edf_path, filename, kind, threshold):
    model, ckpt = get_model(kind)
    probs, _preds = predict_recording(model, ckpt, edf_path, threshold=threshold,
                                      overlap=OVERLAP)
    if len(probs) == 0:
        raise ValueError("recording is too short or has no usable EEG windows")
    assessment = assess_recording(probs, window_sec=WINDOW_SEC, threshold=threshold)

    file_key = f"{filename}|{kind}|{threshold}"
    _remember(file_key, assessment, filename)

    # Return the smoothed signal so the chart matches the (smoothed) episode detection.
    display = smooth(probs, SMOOTH_K)
    return {
        "probs": [round(float(p), 4) for p in display],
        "window_sec": WINDOW_SEC,
        "assessment": assessment,
        "filename": filename,
        "file_key": file_key,
    }


def warmup():
    """Load the default CNN checkpoint in the background so the first real analysis
    doesn't pay the load cost. Optionally warm the local LLM when WARMUP_LLM is set."""
    try:
        get_model("cross")
    except Exception as e:
        print(f"[warmup] model load skipped: {e}", file=sys.stderr)
    if os.environ.get("WARMUP_LLM"):
        try:
            demo = {"risk_level": "Low", "risk_score": 0, "n_windows": 1,
                    "n_abnormal_windows": 0, "pct_abnormal": 0.0, "n_episodes": 0,
                    "abnormal_seconds": 0, "peak_confidence": 0.0,
                    "mean_confidence": 0.0, "episodes": []}
            build_report(demo, "warmup.edf")
        except Exception as e:
            print(f"[warmup] LLM warmup skipped: {e}", file=sys.stderr)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter console
        pass

    def _send(self, code, body, ctype="application/json", cache=None):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def _static(self, name, ctype, cache="no-cache"):
        path = os.path.join(DESIGN_DIR, name)
        if not os.path.exists(path):
            return self._send(404, "Not found", "text/plain")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype, cache=cache)

    # ---- routes ----
    def do_GET(self):
        route = self.path.split("?")[0]
        # code assets revalidate (so edits show); the logo is cached hard (rarely changes)
        if route == "/" or route == "/app.html":
            return self._static("app.html", "text/html; charset=utf-8")
        if route == "/neuroscan.css":
            return self._static("neuroscan.css", "text/css")
        if route == "/app.js":
            return self._static("app.js", "application/javascript")
        if route == "/brainfx.js":
            return self._static("brainfx.js", "application/javascript")
        if route == "/upload.js":
            return self._static("upload.js", "application/javascript")
        if route == "/epilogo.png":
            return self._static("epilogo.png", "image/png", cache="public, max-age=31536000, immutable")
        if route == "/eeg.png":
            return self._static("eeg.png", "image/png", cache="public, max-age=31536000, immutable")
        if route == "/brain3.glb":
            return self._static("brain3.glb", "model/gltf-binary", cache="public, max-age=31536000, immutable")
        return self._send(404, "Not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad JSON"})

        if self.path == "/api/analyze":
            return self._analyze(payload)
        if self.path == "/api/report":
            return self._report(payload)
        return self._json(404, {"error": "unknown endpoint"})

    def _analyze(self, p):
        kind = p.get("model", "cross")
        threshold = float(p.get("threshold", 0.5))
        filename = p.get("filename", "uploaded.edf")
        tmp = None
        try:
            if not p.get("upload"):
                return self._json(400, {"error": "no recording uploaded"})
            if not filename.lower().endswith(".edf"):
                return self._json(400, {"error": "please upload an EDF (.edf) recording"})
            try:
                raw = base64.b64decode(p["upload"])
            except Exception:
                return self._json(400, {"error": "the uploaded file could not be read"})
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
            tmp.write(raw); tmp.close()
            edf_path = tmp.name

            result = run_analysis(edf_path, filename, kind, threshold)
            return self._json(200, result)
        except (ValueError, KeyError) as e:
            # Wrong channels / montage mismatch / too-short recording → clear 400.
            return self._json(400, {
                "error": f"could not analyze this recording: {e}. "
                         "Check that it is a scalp EEG EDF with the standard 10-20 channels."
            })
        except Exception as e:
            return self._json(500, {"error": str(e)})
        finally:
            if tmp and os.path.exists(tmp.name):
                try: os.unlink(tmp.name)
                except OSError: pass

    def _report(self, p):
        cached = _ANALYSES.get(p.get("file"))
        if not cached:
            return self._json(400, {"error": "analyze the recording first"})
        _ANALYSES.move_to_end(p.get("file"))  # keep freshly-used analyses in the LRU
        provider = p.get("provider", "foundry_local")
        model = p.get("local_model") if provider == "foundry_local" else None
        api_key = p.get("api_key") if provider == "anthropic" else None
        use_rag = bool(p.get("use_rag", True))
        try:
            text, source = build_report(
                cached["assessment"], cached["filename"],
                provider=provider, api_key=api_key, model=model, use_rag=use_rag,
            )
            return self._json(200, {"text": text, "source": source})
        except Exception as e:
            return self._json(500, {"error": str(e)})


def main():
    port = int(os.environ.get("PORT", 8000))
    print(f"NeuroScan UI  →  http://localhost:{port}")
    print("Serving the clinical interface over the existing backend (Ctrl+C to stop).")
    threading.Thread(target=warmup, daemon=True).start()  # preload the model in the background
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()