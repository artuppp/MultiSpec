#!/usr/bin/env python3
"""
MultiSpec v2 — Servidor de visualización multiespectal
Cámaras: UV (Canon via gphoto2), Térmica (TC001 USB), Visible (Webcam USB)
"""

import cv2
import numpy as np
import threading
import time
import json
import os
import signal
import sys
from flask import Flask, Response, render_template_string, jsonify, request
from pathlib import Path

app = Flask(__name__)

# ─── Configuración ─────────────────────────────────────────────────────────
_DEFAULTS = {"uv": 2, "thermal": 0, "visible": 4}
# visible_alt eliminado — solo webcam como fuente visible

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "camera_config.json")
if os.path.exists(_CONFIG_PATH):
    with open(_CONFIG_PATH) as f:
        _loaded = json.load(f)
    print(f"[config] Índices cargados: {_loaded}")
    _DEFAULTS.update(_loaded)
else:
    print("[config] Usando índices por defecto")

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

# Estado global compartido (thread-safe via lock)
state_lock = threading.Lock()

FILTERS = {
    "none":          "Sin filtro",
    "gray":          "Escala de grises",
    "invert":        "Invertir",
    "eq":            "Ecualizar histograma",
    "clahe":         "CLAHE (contraste local)",
    # ── Upscaling ──
    "upscale_lanczos":      "↑ Lanczos (suave)",
    "upscale_clahe":        "↑ Lanczos + CLAHE",
    "upscale_sharp":        "↑ Lanczos + Enfocar",
    "upscale_sr":           "↑ Super-resolución IA (EDSR)",
    "upscale_sr_inferno":   "↑ Super-resolución IA (EDSR) + Inferno",
    "upscale_sr_clahe":     "↑ Super-res IA + CLAHE",
    # ── Pseudocolor ──
    "jet":           "Pseudocolor Jet",
    "inferno":       "Pseudocolor Inferno",
    "hot":           "Pseudocolor Hot",
    "cool":          "Pseudocolor Cool",
    "rainbow":       "Pseudocolor Rainbow",
    "bone":          "Pseudocolor Bone",
    "plasma":        "Pseudocolor Plasma",
    # ── Otros ──
    "edges":         "Detección de bordes",
    "blur":          "Desenfoque suave",
    "sharpen":       "Enfocar",
}

# Mapeo filtro → colormap OpenCV
COLORMAPS = {
    "jet":     cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot":     cv2.COLORMAP_HOT,
    "cool":    cv2.COLORMAP_COOL,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "bone":    cv2.COLORMAP_BONE,
    "plasma":  cv2.COLORMAP_PLASMA,
}

# Campos que se persisten (excluimos paused/frozen/presentation que son estado de sesión)
_PERSISTENT_KEYS = {"labels", "roi", "filter", "rotate", "flip", "overlay_text"}

_STATE_PATH = os.path.join(os.path.dirname(__file__), "state.json")

_STATE_DEFAULTS = {
    "labels":      {"uv": "UV", "thermal": "Térmica", "visible": "Visible"},
    "paused":      {"uv": False, "thermal": False, "visible": False},
    "frozen":      {"uv": False, "thermal": False, "visible": False},
    "roi":         {"uv": None,  "thermal": None,  "visible": None},
    "filter":      {"uv": "none","thermal": "inferno","visible": "none"},
    "rotate":      {"uv": 0,     "thermal": 0,      "visible": 0},
    "flip":        {"uv": None,  "thermal": None,   "visible": None},
    "overlay_text": "",
    "presentation": False,
}

def _load_state():
    """Carga el estado persistido y lo fusiona con los defaults."""
    s = json.loads(json.dumps(_STATE_DEFAULTS))  # deep copy
    if os.path.exists(_STATE_PATH):
        try:
            saved = json.loads(Path(_STATE_PATH).read_text())
            for key in _PERSISTENT_KEYS:
                if key in saved:
                    if isinstance(s[key], dict) and isinstance(saved[key], dict):
                        s[key].update(saved[key])
                    else:
                        s[key] = saved[key]
            print(f"[state] Configuración cargada desde state.json")
        except Exception as e:
            print(f"[state] Error cargando state.json: {e} — usando defaults")
    return s

def _save_state():
    """Persiste los campos relevantes del estado actual a state.json."""
    try:
        with state_lock:
            to_save = {k: state[k] for k in _PERSISTENT_KEYS}
        Path(_STATE_PATH).write_text(json.dumps(to_save, indent=2))
    except Exception as e:
        print(f"[state] Error guardando state.json: {e}")

state = _load_state()


# ─── Gestor de cámaras ─────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, index, label, is_thermal=False):
        self.index        = index
        self.label        = label
        self.is_thermal   = is_thermal
        self.frame        = None
        self.frozen_frame = None
        self.lock         = threading.Lock()
        self.running      = False
        self.cap          = None
        self._make_placeholder()

    def start(self):
        self.running = True
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()

    def _capture_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"[{self.label}] Conectando a /dev/video{self.index}...")
                self.cap = cv2.VideoCapture(self.index)
                if self.cap.isOpened():
                    # TC001: forzar 256x192 YUYV para evitar formatos raw de temperatura
                    if self.is_thermal:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  256)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
                    else:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    print(f"[{self.label}] ✓ Conectada")
                else:
                    self._make_placeholder()
                    time.sleep(3)
                    continue

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                with self.lock:
                    self.frame = frame
            else:
                print(f"[{self.label}] Reconectando...")
                self.cap.release()
                self.cap = None
                self._make_placeholder()
                time.sleep(2)

    def _make_placeholder(self):
        img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype="uint8")
        img[:] = (18, 22, 30)
        cv2.putText(img, self.label,
                    (FRAME_WIDTH//2 - 55, FRAME_HEIGHT//2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 70), 2)
        cv2.putText(img, "Sin senal",
                    (FRAME_WIDTH//2 - 50, FRAME_HEIGHT//2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 60), 1)
        with self.lock:
            self.frame = img

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def freeze(self):
        f = self.get_frame()
        with self.lock:
            self.frozen_frame = f

    def get_frozen(self):
        with self.lock:
            return self.frozen_frame.copy() if self.frozen_frame is not None else None

    def reconnect(self, new_index):
        self.index = new_index
        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    @property
    def connected(self):
        with self.lock:
            return self.frame is not None


# Inicializa streams (visible = webcam, único)
streams = {
    "uv":      CameraStream(_DEFAULTS["uv"],      "UV"),
    "thermal": CameraStream(_DEFAULTS["thermal"], "Térmica", is_thermal=True),
    "visible": CameraStream(_DEFAULTS["visible"], "Visible"),
}
for s in streams.values():
    s.start()


# ─── Super-resolución con Real-ESRGAN (PyTorch) ────────────────────────────
_sr       = None
_sr_lock  = threading.Lock()

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "thermal_best.pth")
_MODEL_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"

def _get_sr():
    global _sr
    with _sr_lock:
        if _sr is not None:
            return _sr
        try:
            import torch
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from realesrgan import RealESRGANer

            # Descarga el modelo si no está en disco
            if not os.path.exists(_MODEL_PATH):
                print("[SR] Descargando thermal_best.pth...")
                import urllib.request
                urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
                print("[SR] Modelo descargado")

            use_gpu  = torch.cuda.is_available() and not os.environ.get("REALESRGAN_CPU")
            use_half = use_gpu  # fp16 solo en GPU

            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_conv=32,
                num_feat=64, act_type='prelu', upscale=4
            )
            upsampler = RealESRGANer(
                scale=4,
                model_path=_MODEL_PATH,
                model=model,
                tile=0 if use_gpu else 128,  # tiling en CPU para no agotar RAM
                tile_pad=10,
                pre_pad=0,
                half=use_half,
                device=torch.device("cuda" if use_gpu else "cpu"),
            )
            _sr = upsampler
            print(f"[SR] Real-ESRGAN listo ({'GPU CUDA' if use_gpu else 'CPU'})")
        except Exception as e:
            print(f"[SR] Error cargando Real-ESRGAN: {e} — usando Lanczos como fallback")
            _sr = None
    return _sr


def _upscale_lanczos(frame, target_w=FRAME_WIDTH, target_h=FRAME_HEIGHT):
    """Escala a resolución objetivo con interpolación Lanczos4."""
    h, w = frame.shape[:2]
    if w >= target_w and h >= target_h:
        return frame
    scale = max(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def _upscale_sr(frame):
    """Super-resolución x4 con Real-ESRGAN. Fallback a Lanczos si no disponible."""
    sr = _get_sr()
    if sr is None:
        return _upscale_lanczos(frame)
    try:
        # Real-ESRGAN espera BGR uint8 — reducimos al tamaño nativo primero
        h, w = frame.shape[:2]
        native_h, native_w = 192, 256
        if w > native_w * 1.5:
            frame = cv2.resize(frame, (native_w, native_h), interpolation=cv2.INTER_AREA)
        output, _ = sr.enhance(frame, outscale=4)
        # Escalar al tamaño de salida
        output = cv2.resize(output, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        return output
    except Exception as e:
        print(f"[SR] Error en enhance: {e}")
        return _upscale_lanczos(frame)


# ─── Pipeline de frame ─────────────────────────────────────────────────────
def apply_filter(frame, filter_name):
    """Aplica el filtro seleccionado al frame."""
    if filter_name == "none":
        return frame
    if filter_name == "gray":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if filter_name == "invert":
        return cv2.bitwise_not(frame)
    if filter_name == "eq":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq   = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    if filter_name == "clahe":
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl    = clahe.apply(gray)
        return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
    if filter_name == "upscale_lanczos":
        return _upscale_lanczos(frame)
    if filter_name == "upscale_clahe":
        frame = _upscale_lanczos(frame)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl    = clahe.apply(gray)
        return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
    if filter_name == "upscale_sharp":
        frame  = _upscale_lanczos(frame)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)
    if filter_name == "upscale_sr":
        return _upscale_sr(frame)
    if filter_name == "upscale_sr_inferno":
        frame = _upscale_sr(frame)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    if filter_name == "upscale_sr_clahe":
        frame = _upscale_sr(frame)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl    = clahe.apply(gray)
        return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
    if filter_name == "edges":
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if filter_name == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    if filter_name == "sharpen":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)
    if filter_name in COLORMAPS:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, COLORMAPS[filter_name])
    return frame


def apply_rotate(frame, angle):
    """Rota el frame 0/90/180/270 grados."""
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def apply_flip(frame, mode):
    """Voltea el frame: h=horizontal, v=vertical, hv=ambos."""
    if mode == "h":
        return cv2.flip(frame, 1)
    if mode == "v":
        return cv2.flip(frame, 0)
    if mode == "hv":
        return cv2.flip(frame, -1)
    return frame


def apply_roi_letterbox(frame, roi):
    """
    Recorta el ROI y escala directamente a FRAME_WIDTH x FRAME_HEIGHT.
    No añade barras negras — el CSS object-fit:contain ya se encarga del AR en el cliente.
    """
    src_h, src_w = frame.shape[:2]

    x1 = max(0, int(roi["x"] * src_w))
    y1 = max(0, int(roi["y"] * src_h))
    x2 = min(src_w, int((roi["x"] + roi["w"]) * src_w))
    y2 = min(src_h, int((roi["y"] + roi["h"]) * src_h))
    if x2 <= x1 or y2 <= y1:
        return frame

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)


# ─── Pipeline de frame ─────────────────────────────────────────────────────
def get_processed_frame(key):
    with state_lock:
        s = {
            "frozen":   state["frozen"][key],
            "roi":      state["roi"].get(key),
            "filter":   state["filter"].get(key, "none"),
            "rotate":   state["rotate"].get(key, 0),
            "flip":     state["flip"].get(key),
        }

    stream = streams[key]

    # Congelado
    if s["frozen"]:
        frame = stream.get_frozen() or stream.get_frame()
    else:
        frame = stream.get_frame()

    if frame is None:
        return None

    # 1. Rotar
    if s["rotate"]:
        frame = apply_rotate(frame, s["rotate"])

    # 2. Flip
    if s["flip"]:
        frame = apply_flip(frame, s["flip"])

    # 3. ROI (sobre el frame ya rotado/flipado, igual que lo ve el usuario)
    if s["roi"]:
        frame = apply_roi_letterbox(frame, s["roi"])

    # 4. Filtro / pseudocolor
    frame = apply_filter(frame, s["filter"])

    # 5. Redimensionar al tamaño de salida si hace falta
    h, w = frame.shape[:2]
    if w != FRAME_WIDTH or h != FRAME_HEIGHT:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

    return frame


# ─── Generador MJPEG ───────────────────────────────────────────────────────
def generate_stream(key):
    last_frame = None
    while True:
        with state_lock:
            paused = state["paused"].get(key, False)

        if paused and last_frame is not None:
            frame = last_frame
        else:
            frame = get_processed_frame(key)
            if frame is not None:
                last_frame = frame

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1 / 30)


# ─── API REST ──────────────────────────────────────────────────────────────
@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify(state)

@app.route("/api/pause/<key>", methods=["POST"])
def api_pause(key):
    if key not in state["paused"]:
        return jsonify({"error": "canal no válido"}), 400
    with state_lock:
        state["paused"][key] = not state["paused"][key]
        val = state["paused"][key]
    return jsonify({"paused": val})

@app.route("/api/freeze/<key>", methods=["POST"])
def api_freeze(key):
    if key not in state["frozen"]:
        return jsonify({"error": "canal no válido"}), 400
    with state_lock:
        currently = state["frozen"][key]
    if not currently:
        streams[key].freeze()
    with state_lock:
        state["frozen"][key] = not currently
        val = state["frozen"][key]
    return jsonify({"frozen": val})

@app.route("/api/invert/<key>", methods=["POST"])
def api_invert(key):
    if key not in state["filter"]:
        return jsonify({"error": "canal no válido"}), 400
    with state_lock:
        cur = state["filter"][key]
        state["filter"][key] = "none" if cur == "invert" else "invert"
        val = state["filter"][key]
    _save_state()
    return jsonify({"filter": val})

@app.route("/api/filter/<key>", methods=["POST"])
def api_filter(key):
    if key not in state["filter"]:
        return jsonify({"error": "canal no válido"}), 400
    data = request.get_json()
    f = data.get("filter", "none")
    if f not in FILTERS:
        return jsonify({"error": f"filtro no válido. Opciones: {list(FILTERS.keys())}"}), 400
    with state_lock:
        state["filter"][key] = f
    _save_state()
    return jsonify({"filter": f})

@app.route("/api/filters")
def api_filters_list():
    return jsonify(FILTERS)

@app.route("/api/rotate/<key>", methods=["POST"])
def api_rotate(key):
    if key not in state["rotate"]:
        return jsonify({"error": "canal no válido"}), 400
    data = request.get_json()
    delta = int(data.get("delta", 90))
    with state_lock:
        state["rotate"][key] = (state["rotate"][key] + delta) % 360
        val = state["rotate"][key]
    _save_state()
    return jsonify({"rotate": val})

@app.route("/api/flip/<key>", methods=["POST"])
def api_flip(key):
    if key not in state["flip"]:
        return jsonify({"error": "canal no válido"}), 400
    data = request.get_json()
    mode = data.get("mode")
    if mode not in (None, "h", "v", "hv"):
        return jsonify({"error": "mode debe ser h, v, hv o null"}), 400
    with state_lock:
        state["flip"][key] = None if state["flip"][key] == mode else mode
        val = state["flip"][key]
    _save_state()
    return jsonify({"flip": val})

@app.route("/api/roi/<key>", methods=["POST"])
def api_roi_set(key):
    if key not in state["roi"]:
        return jsonify({"error": "canal no válido"}), 400
    data = request.get_json()
    try:
        roi = {k: float(data[k]) for k in ("x","y","w","h")}
        if roi["w"] < 0.02 or roi["h"] < 0.02:
            return jsonify({"error": "ROI demasiado pequeño"}), 400
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "datos inválidos"}), 400
    with state_lock:
        state["roi"][key] = roi
    _save_state()
    return jsonify({"roi": roi})

@app.route("/api/roi/<key>", methods=["DELETE"])
def api_roi_clear(key):
    if key not in state["roi"]:
        return jsonify({"error": "canal no válido"}), 400
    with state_lock:
        state["roi"][key] = None
    _save_state()
    return jsonify({"roi": None})

@app.route("/api/label/<key>", methods=["POST"])
def api_label(key):
    data = request.get_json()
    if not data or "label" not in data:
        return jsonify({"error": "falta label"}), 400
    with state_lock:
        state["labels"][key] = data["label"][:30]
    streams[key].label = data["label"][:30]
    _save_state()
    return jsonify({"label": data["label"]})

@app.route("/api/overlay", methods=["POST"])
def api_overlay():
    data = request.get_json()
    with state_lock:
        state["overlay_text"] = data.get("text", "")[:80]
    _save_state()
    return jsonify({"overlay_text": state["overlay_text"]})

@app.route("/api/presentation", methods=["POST"])
def api_presentation():
    with state_lock:
        state["presentation"] = not state["presentation"]
        val = state["presentation"]
    return jsonify({"presentation": val})

@app.route("/status")
def status():
    result = {}
    for key, s in streams.items():
        result[key] = "ok" if s.connected else "sin señal"
    return jsonify(result)

# ─── Streams ───────────────────────────────────────────────────────────────
@app.route("/stream/uv")
def stream_uv():
    return Response(generate_stream("uv"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream/thermal")
def stream_thermal():
    return Response(generate_stream("thermal"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream/visible")
def stream_visible():
    return Response(generate_stream("visible"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ─── HTML ──────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MultiSpec</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono&display=swap');

:root {
  --bg:      #07090e;
  --surface: #0d1117;
  --surface2:#131920;
  --border:  #1c2433;
  --uv:      #a78bfa;
  --thermal: #fb923c;
  --visible: #34d399;
  --text:    #dde4f0;
  --muted:   #4e6070;
  --danger:  #f87171;
  --freeze:  #60a5fa;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Space Grotesk', sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Header ── */
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 1.5rem;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  gap: 1rem;
  transition: opacity 0.4s;
}
header.hidden { opacity: 0; pointer-events: none; height: 0; padding: 0; overflow: hidden; }

.logo {
  font-size: 1.2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  white-space: nowrap;
}
.logo-dot {
  width: 9px; height: 9px; border-radius: 50%;
  background: conic-gradient(var(--uv) 0deg 120deg, var(--thermal) 120deg 240deg, var(--visible) 240deg 360deg);
  animation: spin 8s linear infinite;
  flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

.header-controls {
  display: flex;
  gap: 0.4rem;
  align-items: center;
  flex-wrap: wrap;
}

/* ── Botones generales ── */
.btn {
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 4px 10px;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.72rem;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
  white-space: nowrap;
}
.btn:hover { color: var(--text); border-color: #334155; }
.btn.active { color: #fff; border-color: currentColor; }
.btn.active.btn-pause   { color: var(--danger);  background: rgba(248,113,113,0.1); }
.btn.active.btn-freeze  { color: var(--freeze);  background: rgba(96,165,250,0.1); }
.btn.active.btn-invert  { color: var(--thermal); background: rgba(251,146,60,0.1); }
.btn-presentation.active { color: #fbbf24; border-color: #fbbf24; background: rgba(251,191,36,0.08); }

/* ── Grid ── */
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: var(--border);
  flex: 1;
  min-height: 0;
}

.panel {
  background: var(--surface);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
}

/* Panel expandido (fullscreen de canal) */
.grid.solo .panel { display: none; }
.grid.solo .panel.expanded { display: flex; grid-column: 1 / -1; }

/* ── Panel header ── */
.panel-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.75rem;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  min-height: 36px;
}
.panel-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.panel-uv     .panel-dot { background: var(--uv); }
.panel-uv     .panel-header { border-left: 3px solid var(--uv); }
.panel-thermal .panel-dot { background: var(--thermal); }
.panel-thermal .panel-header { border-left: 3px solid var(--thermal); }
.panel-visible .panel-dot { background: var(--visible); }
.panel-visible .panel-header { border-left: 3px solid var(--visible); }

.panel-label-text {
  font-weight: 600;
  font-size: 0.8rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  cursor: pointer;
  border-bottom: 1px dashed transparent;
  transition: border-color 0.2s;
}
.panel-label-text:hover { border-color: var(--muted); }

.panel-controls { display: flex; gap: 0.3rem; margin-left: auto; align-items: center; }
.panel-btn {
  background: none;
  border: 1px solid transparent;
  color: var(--muted);
  padding: 2px 6px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.68rem;
  font-family: 'Space Mono', monospace;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
  white-space: nowrap;
}
.panel-btn:hover { color: var(--text); border-color: var(--border); }
.panel-btn.active-pause  { color: var(--danger);  border-color: var(--danger);  background: rgba(248,113,113,0.08); }
.panel-btn.active-freeze { color: var(--freeze);  border-color: var(--freeze);  background: rgba(96,165,250,0.08); }
.panel-btn.active-invert { color: var(--thermal); border-color: var(--thermal); background: rgba(251,146,60,0.08); }
.panel-btn.active-expand { color: var(--visible); border-color: var(--visible); background: rgba(52,211,153,0.08); }

/* ── Video ── */
.panel-video {
  flex: 1;
  position: relative;
  background: #000;
  min-height: 0;
}
.panel-video img {
  width: 100%; height: 100%;
  object-fit: contain;
  display: block;
}

/* Canvas ROI — superpuesto sobre el vídeo */
.roi-canvas {
  position: absolute;
  inset: 0;
  width: 100%; height: 100%;
  cursor: crosshair;
  z-index: 5;
}
.roi-canvas.idle { cursor: crosshair; }
.roi-canvas.drawing { cursor: crosshair; }

/* Botón reset ROI */
.panel-btn.active-roi {
  color: #f472b6;
  border-color: #f472b6;
  background: rgba(244,114,182,0.08);
}

/* Select filtro */
.panel-select {
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 2px 4px;
  border-radius: 4px;
  font-size: 0.65rem;
  font-family: 'Space Mono', monospace;
  cursor: pointer;
  max-width: 110px;
}
.panel-select:focus { outline: none; }
.panel-select option { background: var(--surface2); color: var(--text); }

.badge {
  position: absolute;
  top: 6px; right: 6px;
  font-size: 0.6rem;
  font-family: 'Space Mono', monospace;
  padding: 2px 5px;
  border-radius: 3px;
  background: rgba(0,0,0,0.75);
  color: var(--muted);
  pointer-events: none;
}
.badge.live { color: #4ade80; }
.badge.frozen { color: var(--freeze); }
.badge.paused { color: var(--danger); }

/* ── Overlay de texto ── */
.overlay-bar {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  background: rgba(0,0,0,0.65);
  color: #fff;
  font-size: 0.85rem;
  font-weight: 600;
  text-align: center;
  padding: 6px 12px;
  letter-spacing: 0.04em;
  pointer-events: none;
  display: none;
}
.overlay-bar.visible { display: block; }

/* ── Footer ── */
footer {
  padding: 0.4rem 1.5rem;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  gap: 1rem;
  transition: opacity 0.4s;
}
footer.hidden { opacity: 0; pointer-events: none; height: 0; padding: 0; overflow: hidden; }

.footer-status {
  font-size: 0.65rem;
  color: var(--muted);
  font-family: 'Space Mono', monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Modal edición etiqueta ── */
.modal-bg {
  display: none;
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.7);
  z-index: 100;
  align-items: center;
  justify-content: center;
}
.modal-bg.open { display: flex; }
.modal {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.5rem;
  width: 320px;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.modal h3 { font-size: 0.9rem; font-weight: 600; }
.modal input {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 10px;
  border-radius: 5px;
  font-size: 0.85rem;
  font-family: 'Space Grotesk', sans-serif;
  width: 100%;
}
.modal input:focus { outline: none; border-color: #334155; }
.modal-btns { display: flex; gap: 0.5rem; justify-content: flex-end; }

/* ── Responsive ── */
@media (max-width: 700px) {
  .grid { grid-template-columns: 1fr; }
  .header-controls { gap: 0.25rem; }
}
</style>
</head>
<body>

<header id="header">
  <div class="logo"><div class="logo-dot"></div>MultiSpec</div>
  <div class="header-controls">

    <!-- Overlay text -->
    <input id="overlay-input" type="text" placeholder="Texto overlay…"
      style="background:var(--surface2);border:1px solid var(--border);color:var(--text);
             padding:3px 8px;border-radius:5px;font-size:0.72rem;font-family:'Space Grotesk',sans-serif;width:170px;"
      oninput="setOverlay(this.value)">

    <button class="btn btn-presentation" id="btn-presentation" onclick="togglePresentation()">
      ☀ Presentación
    </button>
    <button class="btn" onclick="toggleFullscreen()">⛶ Pantalla completa</button>
  </div>
</header>

<div class="grid" id="grid">

  <!-- UV -->
  <div class="panel panel-uv" id="panel-uv">
    <div class="panel-header">
      <div class="panel-dot"></div>
      <span class="panel-label-text" id="label-uv" onclick="openLabelModal('uv')">UV</span>
      <div class="panel-controls">
        <button class="panel-btn" id="btn-pause-uv"   onclick="togglePause('uv')">⏸ Pausa</button>
        <button class="panel-btn" id="btn-freeze-uv"  onclick="toggleFreeze('uv')">❄ Congelar</button>
        <button class="panel-btn" id="btn-rotl-uv"    onclick="rotate('uv',-90)" title="Rotar -90°">↺</button>
        <button class="panel-btn" id="btn-rotr-uv"    onclick="rotate('uv',90)"  title="Rotar +90°">↻</button>
        <button class="panel-btn" id="btn-fliph-uv"   onclick="flip('uv','h')"   title="Voltear horizontal">⇄</button>
        <button class="panel-btn" id="btn-flipv-uv"   onclick="flip('uv','v')"   title="Voltear vertical">⇅</button>
        <select class="panel-select" id="sel-filter-uv" onchange="setFilter('uv',this.value)"></select>
        <button class="panel-btn" id="btn-roi-uv"     onclick="resetRoi('uv')" style="display:none">✕ ROI</button>
        <button class="panel-btn" id="btn-expand-uv"  onclick="toggleExpand('uv')">⤢</button>
      </div>
    </div>
    <div class="panel-video">
      <img id="img-uv" src="/stream/uv" alt="UV">
      <canvas class="roi-canvas" id="canvas-uv"></canvas>
      <span class="badge live" id="badge-uv">● LIVE</span>
      <div class="overlay-bar" id="overlay-uv"></div>
    </div>
  </div>

  <!-- Visible -->
  <div class="panel panel-visible" id="panel-visible">
    <div class="panel-header">
      <div class="panel-dot"></div>
      <span class="panel-label-text" id="label-visible" onclick="openLabelModal('visible')">Visible</span>
      <div class="panel-controls">
        <button class="panel-btn" id="btn-pause-visible"   onclick="togglePause('visible')">⏸ Pausa</button>
        <button class="panel-btn" id="btn-freeze-visible"  onclick="toggleFreeze('visible')">❄ Congelar</button>
        <button class="panel-btn" id="btn-rotl-visible"    onclick="rotate('visible',-90)" title="Rotar -90°">↺</button>
        <button class="panel-btn" id="btn-rotr-visible"    onclick="rotate('visible',90)"  title="Rotar +90°">↻</button>
        <button class="panel-btn" id="btn-fliph-visible"   onclick="flip('visible','h')"   title="Voltear horizontal">⇄</button>
        <button class="panel-btn" id="btn-flipv-visible"   onclick="flip('visible','v')"   title="Voltear vertical">⇅</button>
        <select class="panel-select" id="sel-filter-visible" onchange="setFilter('visible',this.value)"></select>
        <button class="panel-btn" id="btn-roi-visible"     onclick="resetRoi('visible')" style="display:none">✕ ROI</button>
        <button class="panel-btn" id="btn-expand-visible"  onclick="toggleExpand('visible')">⤢</button>
      </div>
    </div>
    <div class="panel-video">
      <img id="img-visible" src="/stream/visible" alt="Visible">
      <canvas class="roi-canvas" id="canvas-visible"></canvas>
      <span class="badge live" id="badge-visible">● LIVE</span>
      <div class="overlay-bar" id="overlay-visible"></div>
    </div>
  </div>

  <!-- Térmica -->
  <div class="panel panel-thermal" id="panel-thermal">
    <div class="panel-header">
      <div class="panel-dot"></div>
      <span class="panel-label-text" id="label-thermal" onclick="openLabelModal('thermal')">Térmica</span>
      <div class="panel-controls">
        <button class="panel-btn" id="btn-pause-thermal"   onclick="togglePause('thermal')">⏸ Pausa</button>
        <button class="panel-btn" id="btn-freeze-thermal"  onclick="toggleFreeze('thermal')">❄ Congelar</button>
        <button class="panel-btn" id="btn-rotl-thermal"    onclick="rotate('thermal',-90)" title="Rotar -90°">↺</button>
        <button class="panel-btn" id="btn-rotr-thermal"    onclick="rotate('thermal',90)"  title="Rotar +90°">↻</button>
        <button class="panel-btn" id="btn-fliph-thermal"   onclick="flip('thermal','h')"   title="Voltear horizontal">⇄</button>
        <button class="panel-btn" id="btn-flipv-thermal"   onclick="flip('thermal','v')"   title="Voltear vertical">⇅</button>
        <select class="panel-select" id="sel-filter-thermal" onchange="setFilter('thermal',this.value)"></select>
        <button class="panel-btn" id="btn-roi-thermal"     onclick="resetRoi('thermal')" style="display:none">✕ ROI</button>
        <button class="panel-btn" id="btn-expand-thermal"  onclick="toggleExpand('thermal')">⤢</button>
      </div>
    </div>
    <div class="panel-video">
      <img id="img-thermal" src="/stream/thermal" alt="Térmica">
      <canvas class="roi-canvas" id="canvas-thermal"></canvas>
      <span class="badge live" id="badge-thermal">● LIVE</span>
      <div class="overlay-bar" id="overlay-thermal"></div>
    </div>
  </div>

</div>

<footer id="footer">
  <span class="footer-status" id="footer-status">Conectando…</span>
</footer>

<!-- Modal etiqueta -->
<div class="modal-bg" id="modal-bg">
  <div class="modal">
    <h3>Editar etiqueta</h3>
    <input id="modal-input" type="text" maxlength="30" placeholder="Nombre del canal">
    <div class="modal-btns">
      <button class="btn" onclick="closeModal()">Cancelar</button>
      <button class="btn active" style="color:#fff;border-color:#334155" onclick="saveLabel()">Guardar</button>
    </div>
  </div>
</div>

<script>
// ── Estado local ──────────────────────────────────────────────
let expandedPanel = null;
let modalKey = null;
let presentationMode = false;

// ── Helpers API ──────────────────────────────────────────────
async function api(path, body = null) {
  try {
    const opts = body !== null
      ? { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body) }
      : { method: "POST" };
    const r = await fetch(path, opts);
    return await r.json();
  } catch(e) { console.error(e); }
}

// ── Pausa ─────────────────────────────────────────────────────
async function togglePause(key) {
  const d = await api(`/api/pause/${key}`);
  const btn = document.getElementById(`btn-pause-${key}`);
  const badge = document.getElementById(`badge-${key}`);
  if (d.paused) {
    btn.classList.add("active-pause");
    btn.textContent = "▶ Reanudar";
    badge.textContent = "● PAUSADO";
    badge.className = "badge paused";
  } else {
    btn.classList.remove("active-pause");
    btn.textContent = "⏸ Pausa";
    badge.className = "badge live";
    badge.textContent = "● LIVE";
  }
}

// ── Congelar ──────────────────────────────────────────────────
async function toggleFreeze(key) {
  const d = await api(`/api/freeze/${key}`);
  const btn = document.getElementById(`btn-freeze-${key}`);
  const badge = document.getElementById(`badge-${key}`);
  if (d.frozen) {
    btn.classList.add("active-freeze");
    btn.textContent = "❄ Descongelar";
    badge.textContent = "● CONGELADO";
    badge.className = "badge frozen";
  } else {
    btn.classList.remove("active-freeze");
    btn.textContent = "❄ Congelar";
    badge.className = "badge live";
    badge.textContent = "● LIVE";
  }
}

// ── Poblar selects de filtros al cargar ───────────────────────
const DEFAULT_FILTERS = {
  uv:      "none",
  thermal: "inferno",
  visible: "none",
};

async function initFilters() {
  const r = await fetch("/api/filters");
  const filters = await r.json();
  ["uv","thermal","visible"].forEach(key => {
    const sel = document.getElementById(`sel-filter-${key}`);
    Object.entries(filters).forEach(([id, label]) => {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = label;
      if (id === DEFAULT_FILTERS[key]) opt.selected = true;
      sel.appendChild(opt);
    });
  });
}
initFilters().then(restoreState);

// ── Filtro ────────────────────────────────────────────────────
async function setFilter(key, filter) {
  await api(`/api/filter/${key}`, { filter });
}

// ── Rotar ─────────────────────────────────────────────────────
const rotateState = { uv: 0, thermal: 0, visible: 0 };
async function rotate(key, delta) {
  await api(`/api/rotate/${key}`, { delta });
  rotateState[key] = (rotateState[key] + delta + 360) % 360;
  const active = rotateState[key] !== 0;
  document.getElementById(`btn-rotl-${key}`).classList.toggle("active-invert", active);
  document.getElementById(`btn-rotr-${key}`).classList.toggle("active-invert", active);
}

// ── Flip ──────────────────────────────────────────────────────
const flipState = { uv: null, thermal: null, visible: null };
async function flip(key, mode) {
  const d = await api(`/api/flip/${key}`, { mode });
  flipState[key] = d.flip;
  document.getElementById(`btn-fliph-${key}`).classList.toggle("active-invert", d.flip === "h" || d.flip === "hv");
  document.getElementById(`btn-flipv-${key}`).classList.toggle("active-invert", d.flip === "v" || d.flip === "hv");
}

// ── Restaurar estado guardado al cargar ───────────────────────
async function restoreState() {
  try {
    const r = await fetch("/api/state");
    const s = await r.json();
    const keys = ["uv", "thermal", "visible"];

    keys.forEach(key => {
      // Etiquetas
      document.getElementById(`label-${key}`).textContent = s.labels[key] || key;

      // Filtro
      const sel = document.getElementById(`sel-filter-${key}`);
      if (sel && s.filter[key]) sel.value = s.filter[key];

      // Rotar
      rotateState[key] = s.rotate[key] || 0;
      const rotActive = rotateState[key] !== 0;
      document.getElementById(`btn-rotl-${key}`).classList.toggle("active-invert", rotActive);
      document.getElementById(`btn-rotr-${key}`).classList.toggle("active-invert", rotActive);

      // Flip
      flipState[key] = s.flip[key] || null;
      const f = flipState[key];
      document.getElementById(`btn-fliph-${key}`).classList.toggle("active-invert", f === "h" || f === "hv");
      document.getElementById(`btn-flipv-${key}`).classList.toggle("active-invert", f === "v" || f === "hv");

      // ROI
      if (s.roi[key]) {
        roiState[key].rect = s.roi[key];
        redrawRoi(key);
        const btn = document.getElementById(`btn-roi-${key}`);
        btn.style.display = "";
        btn.classList.add("active-roi");
      }
    });

    // Overlay
    if (s.overlay_text) {
      document.getElementById("overlay-input").value = s.overlay_text;
      keys.forEach(key => {
        const el = document.getElementById(`overlay-${key}`);
        el.textContent = s.overlay_text;
        el.classList.add("visible");
      });
    }

    console.log("[state] UI restaurada desde estado guardado");
  } catch(e) {
    console.warn("[state] No se pudo restaurar el estado:", e);
  }
}

// ── Expandir canal ────────────────────────────────────────────
function toggleExpand(key) {
  const grid = document.getElementById("grid");
  const panel = document.getElementById(`panel-${key}`);
  const btn = document.getElementById(`btn-expand-${key}`);

  if (expandedPanel === key) {
    // Colapsar
    grid.classList.remove("solo");
    panel.classList.remove("expanded");
    btn.classList.remove("active-expand");
    btn.textContent = "⤢";
    expandedPanel = null;
  } else {
    // Expandir
    if (expandedPanel) {
      document.getElementById(`panel-${expandedPanel}`).classList.remove("expanded");
      document.getElementById(`btn-expand-${expandedPanel}`).classList.remove("active-expand");
      document.getElementById(`btn-expand-${expandedPanel}`).textContent = "⤢";
    }
    grid.classList.add("solo");
    panel.classList.add("expanded");
    btn.classList.add("active-expand");
    btn.textContent = "⤡";
    expandedPanel = key;
  }
}

// ── Overlay texto ─────────────────────────────────────────────
let overlayTimer = null;
function setOverlay(text) {
  clearTimeout(overlayTimer);
  overlayTimer = setTimeout(async () => {
    await api("/api/overlay", { text });
    ["uv","thermal","visible"].forEach(key => {
      const el = document.getElementById(`overlay-${key}`);
      if (text.trim()) {
        el.textContent = text;
        el.classList.add("visible");
      } else {
        el.classList.remove("visible");
      }
    });
  }, 300);
}

// ── Modo presentación ─────────────────────────────────────────
async function togglePresentation() {
  const d = await api("/api/presentation");
  presentationMode = d.presentation;
  const header = document.getElementById("header");
  const footer = document.getElementById("footer");
  const btn = document.getElementById("btn-presentation");

  // En modo presentación ocultamos header/footer y los controles de panel
  header.classList.toggle("hidden", presentationMode);
  footer.classList.toggle("hidden", presentationMode);
  btn.classList.toggle("active", presentationMode);
  btn.textContent = presentationMode ? "✕ Salir presentación" : "☀ Presentación";

  // Oculta controles de panel
  document.querySelectorAll(".panel-controls, .panel-label-text").forEach(el => {
    el.style.visibility = presentationMode ? "hidden" : "";
  });
}

// ── Pantalla completa ─────────────────────────────────────────
function toggleFullscreen() {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen();
  else document.exitFullscreen();
}

// ── Teclas rápidas ────────────────────────────────────────────
document.addEventListener("keydown", e => {
  if (e.key === "F1") { e.preventDefault(); toggleExpand("uv"); }
  if (e.key === "F2") { e.preventDefault(); toggleExpand("thermal"); }
  if (e.key === "F3") { e.preventDefault(); toggleExpand("visible"); }
  if (e.key === "p" || e.key === "P") togglePresentation();
  if (e.key === "f" || e.key === "F") toggleFullscreen();
  if (e.key === "Escape" && expandedPanel) toggleExpand(expandedPanel);
});

// ── Modal etiqueta ────────────────────────────────────────────
function openLabelModal(key) {
  if (presentationMode) return;
  modalKey = key;
  document.getElementById("modal-input").value = document.getElementById(`label-${key}`).textContent;
  document.getElementById("modal-bg").classList.add("open");
  document.getElementById("modal-input").focus();
}
function closeModal() {
  document.getElementById("modal-bg").classList.remove("open");
  modalKey = null;
}
async function saveLabel() {
  if (!modalKey) return;
  const text = document.getElementById("modal-input").value.trim();
  if (!text) return;
  await api(`/api/label/${modalKey}`, { label: text });
  document.getElementById(`label-${modalKey}`).textContent = text;
  closeModal();
}
document.getElementById("modal-input").addEventListener("keydown", e => {
  if (e.key === "Enter") saveLabel();
  if (e.key === "Escape") closeModal();
});
document.getElementById("modal-bg").addEventListener("click", e => {
  if (e.target === document.getElementById("modal-bg")) closeModal();
});

// ── ROI — selección con drag forzando aspect ratio ────────────
const roiState = {};

// Aspect ratio nativo de cada cámara (w/h).
// La térmica TC001 es 256x192. Las demás se leen del primer frame.
const nativeAR = { uv: 4/3, thermal: 256/192, visible: 4/3 };

// Cuando llega el primer frame de cada canal, actualizamos el AR real.
["uv","thermal","visible"].forEach(key => {
  const img = document.getElementById(`img-${key}`);
  img.addEventListener("load", () => {
    if (img.naturalWidth && img.naturalHeight) {
      nativeAR[key] = img.naturalWidth / img.naturalHeight;
    }
  }, { once: false });
});

["uv", "thermal", "visible"].forEach(key => {
  const canvas = document.getElementById(`canvas-${key}`);
  roiState[key] = { dragging: false, startX: 0, startY: 0, rect: null };

  // Sincroniza tamaño canvas con el contenedor
  function syncSize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    redrawRoi(key);
  }
  new ResizeObserver(syncSize).observe(canvas);
  syncSize();

  /**
   * Calcula el rect de la imagen REAL dentro del canvas,
   * teniendo en cuenta el letterbox de object-fit:contain.
   * Devuelve { x, y, w, h } en píxeles de canvas.
   */
  function getImageRect() {
    const cw = canvas.offsetWidth;
    const ch = canvas.offsetHeight;
    const ar = nativeAR[key];
    let iw, ih;
    if (cw / ch > ar) {
      // Barras laterales
      ih = ch;
      iw = ch * ar;
    } else {
      // Barras arriba/abajo
      iw = cw;
      ih = cw / ar;
    }
    return {
      x: (cw - iw) / 2,
      y: (ch - ih) / 2,
      w: iw,
      h: ih,
    };
  }

  /**
   * Convierte posición de ratón (clientX/Y) a fracción dentro del frame real
   * (0,0 = esquina superior izquierda del frame, 1,1 = esquina inferior derecha).
   * Devuelve null si el ratón está fuera de la imagen.
   */
  function getPos(e) {
    const bounds = canvas.getBoundingClientRect();
    const mx = (e.touches ? e.touches[0].clientX : e.clientX) - bounds.left;
    const my = (e.touches ? e.touches[0].clientY : e.clientY) - bounds.top;
    const ir = getImageRect();
    // Clamp dentro de la imagen
    const fx = Math.max(0, Math.min(1, (mx - ir.x) / ir.w));
    const fy = Math.max(0, Math.min(1, (my - ir.y) / ir.h));
    return { x: fx, y: fy };
  }

  function onStart(e) {
    if (presentationMode) return;
    e.preventDefault();
    const p = getPos(e);
    roiState[key].dragging = true;
    roiState[key].startX = p.x;
    roiState[key].startY = p.y;
    roiState[key].rect = null;
  }

  function onMove(e) {
    if (!roiState[key].dragging) return;
    e.preventDefault();
    const p  = getPos(e);
    const sx = roiState[key].startX;
    const sy = roiState[key].startY;

    // Ancho libre en fracción de frame real
    let rawW = Math.abs(p.x - sx);
    // Alto forzado por AR — pero ahora estamos en espacio del frame (AR ya es 1:1)
    // es decir, fracción de frame, y el frame ya tiene el AR nativo.
    // Por tanto alto = ancho (en fracción de frame) ya que AR está preservado.
    // Pero el frame puede no ser cuadrado: necesitamos alto en fracción de alto de frame.
    // Como el frame tiene AR = nativeAR[key] = w/h:
    //   ancho_px = rawW * frame_w
    //   alto_px  = ancho_px  (queremos el mismo número de píxeles en alto que en ancho? No)
    // Lo que queremos: la selección tenga el mismo AR que el frame completo.
    // AR frame = nativeAR[key] = fw/fh
    // rect seleccionado: rw/rh = fw/fh  →  rh = rw * fh/fw = rw / nativeAR[key]
    // Pero rw y rh son fracciones de fw y fh respectivamente, así que:
    //   rh_frac = (rawW * fw) / (nativeAR[key] * fh)  ... simplificando con AR:
    //   rh_frac = rawW  (porque rawW ya es fracción de fw, y queremos misma fracción de fh)
    //   Solo si el frame es cuadrado. En general:
    //   rh_frac = rawW * fw/fh / nativeAR[key] = rawW * nativeAR[key] / nativeAR[key] = rawW
    // Conclusión: rh_frac == rawW cuando queremos el AR completo del frame.
    let rawH = rawW;   // fracción de alto = fracción de ancho → mismo AR que el frame

    // Clamp para no salir del frame
    const x = Math.min(sx, p.x > sx ? sx + rawW : sx - rawW);
    const y = Math.min(sy, p.y > sy ? sy + rawH : sy - rawH);
    rawW = Math.min(rawW, 1 - x);
    rawH = Math.min(rawH, 1 - y);

    roiState[key].rect = { x: Math.max(0,x), y: Math.max(0,y), w: rawW, h: rawH };
    redrawRoi(key);
  }

  async function onEnd(e) {
    if (!roiState[key].dragging) return;
    roiState[key].dragging = false;
    const rect = roiState[key].rect;
    if (!rect || rect.w < 0.02 || rect.h < 0.02) {
      roiState[key].rect = null;
      redrawRoi(key);
      return;
    }
    await api(`/api/roi/${key}`, rect);
    const btn = document.getElementById(`btn-roi-${key}`);
    btn.style.display = "";
    btn.classList.add("active-roi");
  }

  canvas.addEventListener("mousedown",  onStart);
  canvas.addEventListener("mousemove",  onMove);
  canvas.addEventListener("mouseup",    onEnd);
  canvas.addEventListener("touchstart", onStart, { passive: false });
  canvas.addEventListener("touchmove",  onMove,  { passive: false });
  canvas.addEventListener("touchend",   onEnd);
});

function getImageRectForKey(key) {
  const canvas = document.getElementById(`canvas-${key}`);
  const cw = canvas.offsetWidth;
  const ch = canvas.offsetHeight;
  const ar = nativeAR[key];
  let iw, ih;
  if (cw / ch > ar) { ih = ch; iw = ch * ar; }
  else               { iw = cw; ih = cw / ar; }
  return { x: (cw - iw) / 2, y: (ch - ih) / 2, w: iw, h: ih };
}

function redrawRoi(key) {
  const canvas = document.getElementById(`canvas-${key}`);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = roiState[key].rect;
  if (!rect) return;

  const ir = getImageRectForKey(key);

  // Convertir fracción de frame → píxeles de canvas dentro del área de imagen
  const x = ir.x + rect.x * ir.w;
  const y = ir.y + rect.y * ir.h;
  const w = rect.w * ir.w;
  const h = rect.h * ir.h;

  // Oscurecer todo
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  // Limpiar zona del ROI
  ctx.clearRect(x, y, w, h);
  // Borde del ROI
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(x, y, w, h);
  // Esquinas
  ctx.setLineDash([]);
  ctx.lineWidth = 2.5;
  const cs = 10;
  [
    [x,   y,   cs,  0,  0,  cs],
    [x+w, y,  -cs,  0,  0,  cs],
    [x,   y+h, cs,  0,  0, -cs],
    [x+w, y+h,-cs,  0,  0, -cs],
  ].forEach(([px, py, dx1, dy1, dx2, dy2]) => {
    ctx.beginPath();
    ctx.moveTo(px + dx1, py + dy1);
    ctx.lineTo(px, py);
    ctx.lineTo(px + dx2, py + dy2);
    ctx.stroke();
  });
}

async function resetRoi(key) {
  await fetch(`/api/roi/${key}`, { method: "DELETE" });
  roiState[key].rect = null;
  redrawRoi(key);
  const btn = document.getElementById(`btn-roi-${key}`);
  btn.style.display = "none";
  btn.classList.remove("active-roi");
}

// ── Status polling ────────────────────────────────────────────
async function pollStatus() {
  try {
    const r = await fetch("/status");
    const d = await r.json();
    const msgs = Object.entries(d)
      .filter(([k]) => ["uv","thermal","visible"].includes(k))
      .map(([k,v]) => `${k.toUpperCase()}: ${v}`);
    document.getElementById("footer-status").textContent = msgs.join("  ·  ");
  } catch(e) {}
}
setInterval(pollStatus, 4000);
pollStatus();
</script>
</body>
</html>
"""

# ─── Arranque ──────────────────────────────────────────────────────────────
def shutdown(sig, frame):
    print("\nCerrando streams...")
    for s in streams.values():
        s.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

if __name__ == "__main__":
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    pad = max(0, 15 - len(local_ip))
    print(f"""
╔══════════════════════════════════════════╗
║         MultiSpec v2 arrancado           ║
╠══════════════════════════════════════════╣
║  Local:  http://localhost:5000           ║
║  Red:    http://{local_ip}:5000{' ' * pad}║
╠══════════════════════════════════════════╣
║  Teclas rápidas:                         ║
║   F1/F2/F3  → expandir canal            ║
║   P         → modo presentación         ║
║   F         → pantalla completa         ║
║   Esc       → colapsar canal            ║
╚══════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=5000, threaded=True)
