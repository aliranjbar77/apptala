from datetime import datetime
from zoneinfo import ZoneInfo


def mini_box_open_html() -> str:
    return '<div class="mini-box">'


def mini_box_close_html() -> str:
    return "</div>"


def top_banner_html(caption: str, title: str) -> str:
    return f"""
<div class="big-live">
  <div class="caption">{caption}</div>
  <h2 style="margin: 6px 0 0 0;">{title}</h2>
</div>
"""


def live_price_html(
    live_price: float,
    price_change: float | None,
    source: str,
    final_signal: str,
    confidence: float,
    labels: dict[str, str],
) -> str:
    color = "#16c784" if price_change is not None and price_change >= 0 else "#ff4d6d"
    arrow = "UP" if price_change is not None and price_change >= 0 else "DOWN"
    chg_text = "N/A" if price_change is None else f"{price_change:+.2f}"
    pct = "N/A" if (price_change is None or live_price == 0) else f"{(price_change / live_price * 100):+.2f}%"
    tehran_now = datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d %H:%M:%S")

    return f"""
<div class="big-live" style="border-color:{color};">
  <div class="caption">{labels['main_live_box']}</div>
  <div class="kpi" style="color:{color};">{arrow} ${live_price:,.2f}</div>
  <div style="font-size:18px;color:{color};">{labels['change']}: {chg_text} ({pct})</div>
  <div class="caption">{labels['source']}: {source} | {labels['auto_refresh']}: 10s | {labels['tehran']}: {tehran_now}</div>
  <div style="margin-top:12px;font-size:16px;">{labels['final_signal']}: <b>{final_signal}</b> | {labels['confidence']}: <b>{confidence:.0%}</b></div>
</div>
"""
