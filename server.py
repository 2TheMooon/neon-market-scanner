"""Flask backend for the live market-alert dashboard.

Endpoints:
  GET  /                 → dashboard.html
  GET  /api/health       → {"ok": true} liveness probe
  GET  /api/status       → pipeline state
  POST /api/analyze      → one-shot text analysis (no audio)
  POST /api/start        → start capture→transcribe→analyze pipeline
  POST /api/stop         → stop the pipeline
  GET  /api/alerts       → last N entries from output/live_log.jsonl
  GET  /api/stream       → SSE feed of new analysis results
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("server")

from flask import Flask, Response, jsonify, request, send_from_directory

# --- Optional project imports (server must boot even if one is broken) -------

try:
    from analyzer import MarketAnalyzer
    ANALYZER_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 — boot must survive any import failure
    MarketAnalyzer = None
    ANALYZER_IMPORT_ERROR = str(e)

try:
    from transcriber import Transcriber
    TRANSCRIBER_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    Transcriber = None
    TRANSCRIBER_IMPORT_ERROR = str(e)

try:
    from stream_capture import StreamCapture
    CAPTURE_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    StreamCapture = None
    CAPTURE_IMPORT_ERROR = str(e)


# --- Paths / config ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_PATH = os.path.join(OUTPUT_DIR, "live_log.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

HOST = "0.0.0.0"
PORT = 5000

# --- Global state (thread-safe) ---------------------------------------------

STATE = {
    "pipeline_running": False,
    "source": None,
    "start_time": None,  # server start time; see _SERVER_START
    "segments": 0,
    "thread": None,
}
STATE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()

_SERVER_START = time.time()
STATE["start_time"] = _SERVER_START

# SSE: one bounded queue per connected client (broadcast fan-out).
_SSE_CLIENTS: list[queue.Queue] = []
_SSE_LOCK = threading.Lock()


def sse_subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=256)
    with _SSE_LOCK:
        _SSE_CLIENTS.append(q)
    return q


def sse_unsubscribe(q: queue.Queue) -> None:
    with _SSE_LOCK:
        if q in _SSE_CLIENTS:
            _SSE_CLIENTS.remove(q)


def sse_publish(obj: dict) -> None:
    with _SSE_LOCK:
        clients = list(_SSE_CLIENTS)
    for q in clients:
        try:
            q.put_nowait(obj)
        except queue.Full:
            pass  # slow client: drop the event rather than block the pipeline


def sse_event(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# --- Log helpers ------------------------------------------------------------

def append_log(text: str, result: dict) -> dict:
    """Append one analysis entry to live_log.jsonl and broadcast it over SSE."""
    entry = {
        "ts": datetime.now().isoformat(),
        "text": text,
        "analysis": result,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    sse_publish(entry)
    return entry


def read_alerts(limit: int) -> list[dict]:
    """Last `limit` entries from live_log.jsonl (empty list if missing/broken)."""
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []

    entries: list[dict] = []
    for line in lines[-limit:] if limit > 0 else []:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


# --- Pipeline worker --------------------------------------------------------

def pipeline_worker(source: str, language: str | None, model_size: str) -> None:
    """capture.get_chunk() → transcribe → analyze → jsonl + SSE queue.

    Never raises: every failure is logged and broadcast as an SSE error event.
    """
    capture = None
    log.info("pipeline worker starting: source=%s model=%s language=%s", source, model_size, language)
    try:
        if Transcriber is None:
            raise RuntimeError(f"Transcriber unavailable: {TRANSCRIBER_IMPORT_ERROR}")
        if MarketAnalyzer is None:
            raise RuntimeError(f"MarketAnalyzer unavailable: {ANALYZER_IMPORT_ERROR}")
        if StreamCapture is None:
            raise RuntimeError(f"StreamCapture unavailable: {CAPTURE_IMPORT_ERROR}")

        try:
            transcriber = Transcriber(model_size=model_size, language=language)
            if hasattr(transcriber, "warmup"):
                transcriber.warmup()
        except Exception as e:  # noqa: BLE001 — e.g. Whisper model load failure
            raise RuntimeError(f"transcriber init failed: {e}") from e

        try:
            analyzer = MarketAnalyzer()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"analyzer init failed: {e}") from e

        try:
            capture = StreamCapture(source, chunk_seconds=10)
            capture.start()
        except Exception as e:  # noqa: BLE001 — e.g. yt-dlp / network failure
            raise RuntimeError(f"stream capture start failed: {e}") from e

        log.info("pipeline running: source=%s", source)
        while not STOP_EVENT.is_set():
            try:
                if capture is not None and hasattr(capture, "is_active") and not capture.is_active:
                    log.info("capture ended (is_active=False)")
                    break
            except Exception:  # noqa: BLE001 — capture state query failed → keep going
                pass

            try:
                chunk = capture.get_chunk(timeout=2.0) if capture is not None else None
            except Exception as e:  # noqa: BLE001 — per-chunk capture error
                log.warning("capture.get_chunk failed: %s", e)
                sse_publish({"type": "error", "error": f"capture: {e}"})
                if STOP_EVENT.wait(1.0):
                    break
                continue

            if chunk is None:
                continue

            try:
                text = transcriber.transcribe_chunk(chunk)
            except Exception as e:  # noqa: BLE001 — per-chunk transcribe error
                log.warning("transcribe_chunk failed: %s", e)
                sse_publish({"type": "error", "error": f"transcribe: {e}"})
                continue

            if not text or len(text.strip()) < 3:
                continue

            try:
                result = analyzer.analyze(text)
            except Exception as e:  # noqa: BLE001 — per-chunk analyze error
                log.warning("analyzer.analyze failed: %s", e)
                sse_publish({"type": "error", "error": f"analyze: {e}"})
                continue

            try:
                append_log(text, result)
            except Exception as e:  # noqa: BLE001 — e.g. disk write failure
                log.warning("append_log failed: %s", e)
                sse_publish({"type": "error", "error": f"log: {e}"})

            with STATE_LOCK:
                STATE["segments"] += 1

        log.info("pipeline worker loop ended")

    except Exception as e:  # noqa: BLE001 — never let the worker kill the server
        log.exception("pipeline worker crashed: %s", e)
        try:
            sse_publish({"type": "error", "error": str(e)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if capture is not None:
            try:
                capture.stop()
            except Exception as e:  # noqa: BLE001
                log.warning("capture.stop failed: %s", e)
        STOP_EVENT.set()
        with STATE_LOCK:
            STATE["pipeline_running"] = False
            STATE["thread"] = None
            STATE["source"] = None
        log.info("pipeline worker stopped")


# --- App --------------------------------------------------------------------

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with STATE_LOCK:
        running = STATE["pipeline_running"]
        source = STATE["source"]
        segments = STATE["segments"]
        start_time = STATE["start_time"]
    uptime = time.time() - start_time if start_time else 0.0
    return jsonify({
        "pipeline_running": running,
        "source": source,
        "segments_analyzed": segments,
        "uptime": uptime,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if MarketAnalyzer is None:
        return jsonify({"error": f"MarketAnalyzer unavailable: {ANALYZER_IMPORT_ERROR}"}), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "missing 'text' field"}), 400

    try:
        result = MarketAnalyzer().analyze(text)
    except Exception as e:  # noqa: BLE001 — return the failure as JSON
        return jsonify({"error": str(e)}), 500

    append_log(text, result)
    return jsonify(result)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    source = data.get("source")
    language = data.get("language") or None
    model_size = data.get("model") or "base"

    if not source:
        return jsonify({"error": "missing 'source' field"}), 400

    with STATE_LOCK:
        if STATE["pipeline_running"]:
            return jsonify({"error": "pipeline already running"}), 409

        STOP_EVENT.clear()
        STATE["pipeline_running"] = True
        STATE["source"] = source
        STATE["segments"] = 0

        thread = threading.Thread(
            target=pipeline_worker,
            args=(source, language, model_size),
            daemon=True,
        )
        STATE["thread"] = thread

    try:
        thread.start()
    except Exception as e:  # noqa: BLE001 — thread spawn failure must not 500 forever
        log.exception("failed to start pipeline thread: %s", e)
        with STATE_LOCK:
            STATE["pipeline_running"] = False
            STATE["thread"] = None
            STATE["source"] = None
        return jsonify({"error": f"failed to start pipeline: {e}"}), 500

    log.info("pipeline start requested: source=%s model=%s", source, model_size)
    return jsonify({"status": "started", "source": source})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    STOP_EVENT.set()
    with STATE_LOCK:
        STATE["pipeline_running"] = False
    log.info("pipeline stop requested")
    return jsonify({"status": "stopped"})


@app.route("/api/alerts")
def api_alerts():
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    limit = max(1, min(limit, 1000))
    return jsonify(read_alerts(limit))


@app.route("/api/stream")
def api_stream():
    def generate():
        q = sse_subscribe()
        try:
            yield sse_event({"type": "hello", "ts": datetime.now().isoformat()})
            last_ping = time.monotonic()
            while True:
                try:
                    item = q.get(timeout=0.5)
                except queue.Empty:
                    if time.monotonic() - last_ping >= 15.0:
                        yield ": ping\n\n"
                        last_ping = time.monotonic()
                    continue
                yield sse_event(item)
                if time.monotonic() - last_ping >= 15.0:
                    yield ": ping\n\n"
                    last_ping = time.monotonic()
        except GeneratorExit:
            # client disconnected — fall through to cleanup
            pass
        except Exception as e:  # noqa: BLE001 — never crash the SSE thread
            log.warning("SSE generator error: %s", e)
        finally:
            sse_unsubscribe(q)

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("=" * 60)
    print("  NEON MARKET SCANNER — dashboard server")
    print(f"  Dashboard:  http://localhost:{PORT}/")
    print(f"  API status: http://localhost:{PORT}/api/status")
    print(f"  Log:        {LOG_PATH}")
    if ANALYZER_IMPORT_ERROR:
        print(f"  [WARN] analyzer import failed: {ANALYZER_IMPORT_ERROR}")
    if TRANSCRIBER_IMPORT_ERROR:
        print(f"  [WARN] transcriber import failed: {TRANSCRIBER_IMPORT_ERROR}")
    if CAPTURE_IMPORT_ERROR:
        print(f"  [WARN] stream_capture import failed: {CAPTURE_IMPORT_ERROR}")
    print("=" * 60)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
