"""Market-impact analysis for live political speech transcription.

Uses Z.AI GLM-5.3 to map transcribed speech segments to tradeable assets:
stocks, sectors, and Polymarket prediction markets.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from zai import ZaiClient

DEFAULT_API_KEY = os.environ.get("ZAI_API_KEY", "")
DEFAULT_MODEL = "glm-5.3"

SYSTEM_PROMPT = (
    "You are a sharp financial market analyst specializing in political "
    "event-driven trading. You analyze statements from politicians (especially "
    "US Presidents) and predict market impact across stocks, sectors, and "
    "Polymarket prediction markets. You always respond with STRICT JSON "
    "matching the specified schema. No prose outside the JSON. Confidence is "
    "0.0-1.0."
)

USER_PROMPT_TEMPLATE = (
    "Analyze this statement for market impact. Map policies/people/companies/countries to tradeable assets.\n"
    "\nContext: {context}\n"
    "Statement: {text}\n"
    "\n"
    "Return ONLY this JSON structure (no wrapper objects, no prose):\n"
    '{{"summary": "one line", "sentiment": "bullish|bearish|neutral", "urgency": "low|medium|high", '
    '"stocks": [{{"ticker": "AAPL", "direction": "up|down", "confidence": 0.7, "reason": "..."}}], '
    '"sectors": [{{"name": "tech", "direction": "up|down", "reason": "..."}}], '
    '"polymarket": [{{"market": "EXACT Polymarket market question as listed on polymarket.com", "direction": "up|down", "confidence": 0.6, "reason": "...", "slug": "polymarket-url-slug", "category": "politics|crypto|economics|sports|science|culture|tech"}}], '
    '"keywords": ["tariff", "china"], "trade_signal": "long X | short Y | none"}}\n\n'
    "CRITICAL for polymarket field: Use REAL Polymarket market phrasing — the exact question traders see, "
    "e.g. \"Will Trump win the 2028 presidential election?\", \"Will the US impose new tariffs on China before July 2026?\", "
    "\"Will there be a US-China trade deal in 2026?\", \"Will Bitcoin hit $150k in 2026?\". "
    "Mention the specific event, person, country, price target or date. NOT generic labels like 'trade war escalation'.\n"
    "For each polymarket entry also provide:\n"
    "- \"slug\": a plausible Polymarket URL slug (kebab-case, lowercase, no quotes) — the part after "
    "polymarket.com/ or polymarket.com/event/. Example: \"Will Trump win the 2028 presidential election?\" "
    "→ \"will-trump-win-the-2028-presidential-election\".\n"
    "- \"category\": one of politics, crypto, economics, sports, science, culture, tech."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(response_text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Handles ```json fences, stray prose, and nested braces.
    Uses brace-counting so truncated or nested JSON still resolves.
    """
    text = response_text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace-counting extraction: find first balanced {...}
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try removing trailing commas (LLM artifacts)
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    return json.loads(fixed)

    # Incomplete JSON (truncated) — try to salvage
    raise ValueError("incomplete JSON object in response")


def _enrich_polymarket_urls(result: dict[str, Any]) -> dict[str, Any]:
    """Attach a full Polymarket event URL to each polymarket entry."""
    poly = result.get("polymarket")
    if isinstance(poly, list):
        for entry in poly:
            if not isinstance(entry, dict):
                continue
            slug = entry.get("slug") or (
                str(entry.get("market", "")).lower().replace(" ", "-")[:60]
            )
            entry["url"] = "https://polymarket.com/event/" + slug
    return result


class MarketAnalyzer:
    """Analyze speech segments for market impact via GLM-5.3."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("ZAI_API_KEY") or DEFAULT_API_KEY
        self.model = model
        self.client = ZaiClient(api_key=api_key)

    def analyze(self, text: str, context: str = "") -> dict[str, Any]:
        """Analyze one speech segment. Returns a strict-JSON dict or error dict."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context or "(none)",
            text=text,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            thinking={"type": "enabled"},
            reasoning_effort="high",
            temperature=0.2,
            max_tokens=4000,
        )

        response_text = response.choices[0].message.content

        try:
            result = _extract_json(response_text)
        except (ValueError, json.JSONDecodeError) as e:
            return {"error": str(e), "raw": response_text}
        return _enrich_polymarket_urls(result)

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Analyze multiple segments in order."""
        return [self.analyze(t) for t in texts]

    def format_report(self, result: dict[str, Any]) -> str:
        """Pretty-print an analysis result for the terminal."""
        if "error" in result:
            return (
                f"❌ Analysis failed: {result['error']}\n"
                f"Raw response:\n{result.get('raw', '')}"
            )

        arrow = {"up": "🟢 ↑", "down": "🔴 ↓", "bullish": "🟢 ↑", "bearish": "🔴 ↓"}
        lines: list[str] = []

        summary = result.get("summary", "")
        sentiment = result.get("sentiment", "neutral")
        urgency = result.get("urgency", "low")
        lines.append("════════════════════════════════════════")
        lines.append("📊 MARKET IMPACT REPORT")
        lines.append("════════════════════════════════════════")
        lines.append(f"📝 {summary}")
        lines.append(
            f"Sentiment: {arrow.get(sentiment, '⚪')} {sentiment}"
            f"   |   Urgency: {urgency}"
        )

        stocks = result.get("stocks") or []
        if stocks:
            lines.append("")
            lines.append("── Stocks ──────────────────────────────")
            for s in stocks:
                conf = s.get("confidence")
                conf_str = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
                lines.append(
                    f"  {arrow.get(s.get('direction', ''), '⚪')} "
                    f"{s.get('ticker', '?')}{conf_str}: {s.get('reason', '')}"
                )

        sectors = result.get("sectors") or []
        if sectors:
            lines.append("")
            lines.append("── Sectors ─────────────────────────────")
            for s in sectors:
                lines.append(
                    f"  {arrow.get(s.get('direction', ''), '⚪')} "
                    f"{s.get('name', '?')}: {s.get('reason', '')}"
                )

        poly = result.get("polymarket") or []
        if poly:
            lines.append("")
            lines.append("── Polymarket ──────────────────────────")
            for p in poly:
                conf = p.get("confidence")
                conf_str = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
                url = p.get("url", "")
                lines.append(
                    f"  {arrow.get(p.get('direction', ''), '⚪')} "
                    f"{p.get('market', '?')}{conf_str} → {url}"
                )
                if p.get("reason"):
                    lines.append(f"      {p['reason']}")

        keywords = result.get("keywords") or []
        if keywords:
            lines.append("")
            lines.append(f"🔑 Keywords: {', '.join(str(k) for k in keywords)}")

        trade = result.get("trade_signal")
        if trade:
            lines.append(f"⚡ Trade signal: {trade}")

        lines.append("════════════════════════════════════════")
        return "\n".join(lines)


if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    result = analyzer.analyze(
        "We are going to put massive tariffs on Chinese imports "
        "to protect American steel jobs"
    )
    print(analyzer.format_report(result))
