# NEON Market Scanner

**Real-time speech → stock & Polymarket trading signals**

Turns live political speeches into actionable market intelligence. Captures audio from YouTube / X (Twitter) / local files, transcribes with Whisper, and analyzes market impact with GLM-5.3 — identifying affected stocks, sectors, and Polymarket prediction markets in real time.

---

## Live Demo

**[https://2themooon.github.io/neon-market-scanner/](https://2themooon.github.io/neon-market-scanner/)**

Interactive dashboard with sample data. Paste a YouTube or X/Twitter link, watch the simulated live feed, explore market alerts with clickable Polymarket links.

---

## Features

- **Live transcription** — faster-whisper (local, free) converts speech to text in ~10s chunks
- **Market impact analysis** — GLM-5.3 maps statements to stocks, sectors, and Polymarket markets with confidence scores
- **Polymarket signals** — exact market questions with clickable `polymarket.com` links
- **Web dashboard** — neon-themed 3-column UI: video player, live transcript, market alerts
- **YouTube + X/Twitter** embed support
- **SSE live feed** — real-time alert streaming with auto-reconnect
- **CLI mode** — one-shot text analysis or full audio pipeline

## Quick Start

### Web Dashboard
```bash
pip install flask faster-whisper yt-dlp numpy zai-sdk
export ZAI_API_KEY=your-api-key
python server.py
# Open http://localhost:5000
```

### CLI
```bash
# Analyze text
python main.py --analyze-text "We will impose massive tariffs on China"

# YouTube stream
python main.py --source "https://youtube.com/watch?v=..." --duration 300

# Local file
python main.py --source rally.mp4
```

### Demo (no setup)
Open `index.html` in a browser or visit the GitHub Pages link above.

---

## Architecture

```
YouTube / X / file / mic
        ↓ yt-dlp + ffmpeg
Audio chunks (16kHz PCM)
        ↓ faster-whisper
Transcribed text
        ↓ GLM-5.3
Stocks / sectors / Polymarket signals
        ↓
Web dashboard (SSE) + JSON logs
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/api/status` | GET | Pipeline status |
| `/api/start` | POST | Start transcription pipeline |
| `/api/stop` | POST | Stop pipeline |
| `/api/alerts?limit=50` | GET | Recent analysis results |
| `/api/stream` | GET | SSE live feed |
| `/api/health` | GET | Health check |

## Setup

1. Get a Z.AI API key at [z.ai/manage-apikey](https://z.ai/manage-apikey/apikey-list)
2. `export ZAI_API_KEY=your-key` (or copy `.env.example` to `.env`)
3. Install: `pip install -r requirements` (see Quick Start)
4. Requires `ffmpeg` in PATH

## Built With

- **[GLM-5.3](https://z.ai)** — market impact analysis (1M context, reasoning: max)
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — local speech recognition
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — stream capture (YouTube, X, 1000+ sites)
- **Flask + SSE** — real-time web dashboard
- **Three.js demo** — neon UI theme

## License

MIT
