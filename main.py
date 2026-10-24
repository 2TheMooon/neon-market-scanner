"""
NEON MARKET SCANNER — real-time speech → market impact analysis

Pipeline:
  YouTube live / file / mic
       ↓
  StreamCapture (audio chunks, 16kHz mono)
       ↓
  Transcriber (faster-whisper → text)
       ↓
  MarketAnalyzer (GLM-5.3 → stocks / sectors / Polymarket)
       ↓
  Live terminal report + JSON log

Usage:
  python main.py --source "https://youtube.com/watch?v=..." --duration 300
  python main.py --source rally.mp4
  python main.py --source mic --duration 60
  python main.py --analyze-text "We will put tariffs on China"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from stream_capture import StreamCapture
from transcriber import Transcriber
from analyzer import MarketAnalyzer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def banner():
    print(r"""
  ╔══════════════════════════════════════════════════════╗
  ║     NEON  MARKET  SCANNER                            ║
  ║     speech → stocks / polymarket signals             ║
  ╚══════════════════════════════════════════════════════╝
    """)


def analyze_single_text(text, context=""):
    """One-shot analysis of a text snippet (no audio)."""
    banner()
    print(f"[TEXT] {text}\n")
    analyzer = MarketAnalyzer()
    result = analyzer.analyze(text, context=context)
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        if "raw" in result:
            print(result["raw"][:500])
        return result
    print(analyzer.format_report(result))
    save_result(result, text)
    return result


def run_pipeline(source, duration=None, language=None, model_size="base", chunk_seconds=10):
    """Full real-time pipeline: capture → transcribe → analyze."""
    banner()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INIT] Loading Whisper model '{model_size}' (first load may take a minute)...")
    transcriber = Transcriber(model_size=model_size, language=language)
    transcriber.warmup()
    print("[INIT] Whisper ready.")

    analyzer = MarketAnalyzer()
    print("[INIT] GLM-5.3 analyzer ready.")

    capture = StreamCapture(source, chunk_seconds=chunk_seconds)
    print(f"[INIT] Starting capture from: {source}")
    capture.start()

    start_time = time.time()
    segment_count = 0
    results = []

    print("\n[LIVE] Listening... (Ctrl+C to stop)\n")
    print("─" * 70)

    try:
        while capture.is_active:
            if duration and (time.time() - start_time) > duration:
                print(f"\n[DONE] Duration limit ({duration}s) reached.")
                break

            chunk = capture.get_chunk(timeout=2.0)
            if chunk is None:
                continue

            elapsed = time.time() - start_time
            text = transcriber.transcribe_chunk(chunk)

            if not text or len(text.strip()) < 3:
                continue

            segment_count += 1
            ts = f"[{elapsed:6.1f}s]"
            print(f"\n{ts} 🎤 {text}")

            # Analyze for market impact
            result = analyzer.analyze(text)
            results.append({"time": elapsed, "text": text, "analysis": result})

            if "error" not in result:
                print(analyzer.format_report(result))
                save_result(result, text)
            else:
                print(f"    [ANALYSIS ERROR] {result['error']}")

            print("─" * 70)

    except KeyboardInterrupt:
        print("\n\n[STOP] Interrupted by user.")
    finally:
        capture.stop()

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"SESSION SUMMARY: {segment_count} segments analyzed in {time.time()-start_time:.0f}s")
    if results:
        # Aggregate tickers mentioned
        ticker_hits = {}
        for r in results:
            for s in r.get("analysis", {}).get("stocks", []) or []:
                t = s.get("ticker", "?")
                ticker_hits.setdefault(t, []).append(s.get("direction", "?"))
        if ticker_hits:
            print("\nTOP TICKERS MENTIONED:")
            for ticker, dirs in sorted(ticker_hits.items(), key=lambda x: -len(x[1])):
                print(f"  {ticker:8s}  ×{len(dirs)}  {'/'.join(set(dirs))}")

        # Save full log
        log_path = os.path.join(OUTPUT_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[SAVED] Full log → {log_path}")

    return results


def save_result(result, text):
    """Append one analysis result to a rolling JSONL log."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "live_log.jsonl")
    entry = {
        "ts": datetime.now().isoformat(),
        "text": text,
        "analysis": result,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time speech → market impact scanner (GLM-5.3 + Whisper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", "-s", default=None,
                        help="YouTube URL, local audio file, or 'mic'")
    parser.add_argument("--duration", "-d", type=int, default=None,
                        help="Max duration in seconds (default: unlimited)")
    parser.add_argument("--language", "-l", default=None,
                        help="Force language code (en, ru) or auto")
    parser.add_argument("--model", "-m", default="base",
                        help="Whisper model size: tiny/base/small/medium/large-v3")
    parser.add_argument("--chunk", "-c", type=int, default=10,
                        help="Audio chunk size in seconds (default 10)")
    parser.add_argument("--analyze-text", "-t", default=None,
                        help="One-shot: analyze a text string instead of audio")

    args = parser.parse_args()

    # One-shot text analysis mode
    if args.analyze_text:
        analyze_single_text(args.analyze_text)
        return

    # Audio pipeline mode
    if not args.source:
        parser.error("Provide --source (YouTube URL / file / mic) or --analyze-text")

    run_pipeline(
        source=args.source,
        duration=args.duration,
        language=args.language,
        model_size=args.model,
        chunk_seconds=args.chunk,
    )


if __name__ == "__main__":
    main()
