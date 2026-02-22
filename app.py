import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, ADXIndicator, MACD
import requests
import json
from datetime import datetime, timedelta
import re
import time
from pathlib import Path
import os

# --- Audio Alert Function ---
def play_alert_sound(alert_type: str = "signal"):
    """Play audio alert for strong signals"""
    if alert_type == "strong_buy":
        audio_html = """
        <audio autoplay>
            <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT" type="audio/wav">
        </audio>
        """
    elif alert_type == "strong_sell":
        audio_html = """
        <audio autoplay>
            <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT" type="audio/wav">
        </audio>
        """
    else:
        return
    
    components.html(audio_html, height=0)

# --- Page Config ---
st.set_page_config(page_title="AI Trading Terminal", layout="wide")
st.markdown(
    """
    <style>
        :root {
            --bg: #0a0f1e;
            --panel: #121a2b;
            --panel-2: #172238;
            --line: #263551;
            --txt: #e6eefc;
            --muted: #95a6c6;
            --buy: #21c77a;
            --sell: #ff5a7a;
            --neutral: #8a96ad;
        }
        .stApp {
            color: var(--txt);
            background: linear-gradient(180deg, #0a0f1e 0%, #0f1727 100%);
        }
        .block-container {
            max-width: 1080px;
            padding-top: 0.8rem;
            padding-bottom: 1rem;
        }
        .app-card {
            background: rgba(18, 26, 43, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px 14px;
            margin: 6px 0;
            box-shadow: 0 3px 14px rgba(0, 0, 0, 0.18);
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 12px;
            margin: 4px 0;
        }
        [data-testid="stMetric"] {
            background: rgba(18, 26, 43, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px 12px;
        }
        [data-testid="stMetricLabel"] {
            color: #cdd9f3 !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricDelta"] {
            color: #a9b9d6 !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricDelta"] > div {
            color: inherit !important;
            opacity: 1 !important;
            text-shadow: 0 0 0 transparent !important;
        }
        [data-testid="stMetric"] * {
            filter: none !important;
        }
        h1, h2, h3, h4, h5 {
            color: #f4f8ff !important;
            letter-spacing: 0.2px;
        }
        .app-card strong,
        .app-card b,
        .app-card h4,
        .app-card h5 {
            color: #eef4ff !important;
        }
        .app-card div,
        .app-card span,
        .app-card p {
            color: #dbe7ff;
        }
        p, span, label, div {
            color: inherit;
        }
        [data-testid="stSidebar"] {
            background: #0d1424;
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }
        [data-testid="stSidebar"] h3 {
            color: #f2f7ff !important;
            font-size: 0.95rem !important;
            margin-top: 0.2rem !important;
            margin-bottom: 0.4rem !important;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p {
            color: #dce7fb !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] textarea {
            color: #f4f8ff !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] div[data-baseweb="input"] input {
            color: #f4f8ff !important;
            -webkit-text-fill-color: #f4f8ff !important;
            caret-color: #f4f8ff !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] > div,
        [data-testid="stSidebar"] .stTextInput div[data-baseweb="input"] > div {
            background: #111a2c !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.14) !important;
        }
        .focus-card {
            background: linear-gradient(135deg, rgba(255,183,77,0.14), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,183,77,0.45);
            border-radius: 10px;
            padding: 8px 10px;
            margin: 6px 0 8px 0;
        }
        .focus-title {
            color: #ffd791;
            font-weight: 700;
            font-size: 12px;
            margin-bottom: 4px;
        }
        .focus-text {
            color: #e8f0ff;
            font-size: 12px;
            line-height: 1.35;
        }
        .top-status {
            position: sticky;
            top: 0.35rem;
            z-index: 999;
            background: rgba(12, 20, 36, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 8px 12px;
            margin-bottom: 10px;
            backdrop-filter: blur(8px);
        }
        .top-status strong { color: #ffffff; font-weight: 700; }
        .top-status .k { color: #a9bcde; font-size: 12px; margin-right: 6px; }
        .top-status .v { color: #f2f6ff; font-size: 13px; margin-right: 12px; }
        .badge-buy, .badge-sell, .badge-neutral {
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 12px;
            font-weight: 700;
        }
        .badge-buy { background: rgba(33,199,122,0.18); color: #8df1be; }
        .badge-sell { background: rgba(255,90,122,0.2); color: #ffc1cf; }
        .badge-neutral { background: rgba(138,150,173,0.22); color: #dde5f6; }
        .badge-ai-external { background: rgba(78,138,255,0.24); color: #d9e7ff; }
        .badge-ai-neutral { background: rgba(138,150,173,0.22); color: #dde5f6; }
        .stButton > button {
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            padding: 12px 24px;
            box-shadow: 0 4px 20px rgba(243, 156, 18, 0.3);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 30px rgba(243, 156, 18, 0.5);
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .pulse-animation {
            animation: pulse 2s infinite;
        }
        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .slide-in {
            animation: slideIn 0.5s ease-out;
        }
        .signal-buy {
            background: linear-gradient(180deg, rgba(33, 199, 122, 0.12) 0%, rgba(18, 26, 43, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 5px solid var(--buy);
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.22);
        }
        .signal-sell {
            background: linear-gradient(180deg, rgba(255, 90, 122, 0.12) 0%, rgba(18, 26, 43, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 5px solid var(--sell);
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.22);
        }
        .signal-neutral {
            background: linear-gradient(180deg, rgba(138, 150, 173, 0.12) 0%, rgba(18, 26, 43, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 5px solid var(--neutral);
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.22);
        }
        .signal-buy h3, .signal-buy p {
            color: #dff9ec !important;
            opacity: 1 !important;
        }
        .signal-sell h3, .signal-sell p {
            color: #ffe3ea !important;
            opacity: 1 !important;
        }
        .signal-neutral h3, .signal-neutral p {
            color: #ecf2ff !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            color: #f4f8ff !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #dce7fb !important;
        }
        [data-testid="stSidebar"] [role="slider"] {
            background: #ff6a6a !important;
        }
        .hero-signal {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            padding: 10px 12px;
            margin-bottom: 8px;
            background: linear-gradient(135deg, rgba(19, 30, 52, 0.9), rgba(14, 23, 40, 0.92));
        }
        .hero-signal .h-label { color: #a9bcde; font-size: 12px; margin-right: 8px; }
        .hero-signal .h-value { color: #f3f7ff; font-size: 15px; font-weight: 700; margin-right: 14px; }
        .kpi-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 8px;
        }
        .kpi-tile {
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            padding: 8px 10px;
            background: rgba(18, 26, 43, 0.72);
        }
        .kpi-tile .k { font-size: 11px; color: #a9bcde; }
        .kpi-tile .v { font-size: 16px; font-weight: 700; color: #f6f9ff; }
        .kpi-buy { border-left: 4px solid #21c77a; }
        .kpi-sell { border-left: 4px solid #ff5a7a; }
        .kpi-neutral { border-left: 4px solid #8a96ad; }
        .method-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
        .method-card {
            background: linear-gradient(180deg, #101a2f, #0d1628);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 10px 12px;
        }
        .method-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .method-name { font-weight: 600; font-size: 14px; color: var(--txt); }
        .pill { border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 700; }
        .pill-buy { background: rgba(33,199,122,0.2); color: #79f0b4; }
        .pill-sell { background: rgba(255,90,122,0.2); color: #ff9eb0; }
        .pill-neutral { background: rgba(138,150,173,0.2); color: #c7d2e9; }
        .method-reason { font-size: 12px; color: #bcc9e3; line-height: 1.4; }
        .site-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: rgba(12, 20, 36, 0.78);
        }
        .site-header .brand { font-size: 15px; font-weight: 700; color: #f1f6ff; }
        .site-header .nav { font-size: 13px; color: #c5d3ee; }
        .right-menu {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 12px;
            background: rgba(18, 26, 43, 0.72);
            position: sticky;
            top: 70px;
            z-index: 20;
        }
        .site-footer {
            margin-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.12);
            padding-top: 10px;
            font-size: 12px;
            color: #9fb0cf;
            text-align: center;
        }
        @media (max-width: 900px) {
            .method-grid { grid-template-columns: 1fr; }
            .kpi-strip { grid-template-columns: 1fr; }
            [data-testid="stMetricValue"] { font-size: 1.28rem !important; }
            [data-testid="stMetricLabel"] { font-size: 0.88rem !important; }
            .block-container { padding-top: 0.8rem; padding-left: 0.8rem; padding-right: 0.8rem; }
        }
        @media (min-width: 1024px) {
            .block-container { padding-top: 1.05rem; }
            .app-card { padding: 14px 16px; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helpers ---
def safe_yf_download(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    """Download market data with retry to avoid transient yfinance concat errors."""
    period_candidates = [period]
    if interval == "1d":
        period_candidates.extend([p for p in ["2y", "1y", "6mo"] if p != period])
    elif interval in {"4h", "1h", "15m", "5m", "1m"}:
        period_candidates.extend([p for p in ["180d", "120d", "60d", "30d", "7d"] if p != period])

    for period_try in period_candidates:
        for attempt in range(retries):
            try:
                df = yf.download(ticker, period=period_try, interval=interval, progress=False, threads=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # yfinance can return duplicated OHLC column names; keep first occurrence.
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                if not df.empty:
                    return df
            except ValueError as e:
                if "No objects to concatenate" not in str(e):
                    # Unknown ValueError; continue retrying but do not crash app.
                    pass
            except Exception:
                # Network/API side failures should not break UI; retry first.
                pass
            if attempt < retries - 1:
                time.sleep(1 + attempt)
    return pd.DataFrame()


def get_live_price(symbol: str) -> float | None:
    """Best-effort live price: fast_info first, then 1m close fallback."""
    try:
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            last_price = fast_info.get("lastPrice") or fast_info.get("last_price")
            if last_price is not None and float(last_price) > 0:
                return float(last_price)
    except Exception:
        pass

    try:
        q = safe_yf_download(symbol, period="1d", interval="1m", retries=2)
        if not q.empty and "Close" in q.columns:
            c = q["Close"].dropna()
            if not c.empty:
                return float(c.iloc[-1])
    except Exception:
        pass
    return None

def get_fresh_quote(symbols: list[str]) -> tuple[float | None, float, str | None]:
    """Return freshest quote; prefer 1m candle (stable), fallback to fast_info."""
    best = None
    for sym in symbols:
        # Primary source: 1m candle feed (more stable for Yahoo symbols).
        q = safe_yf_download(sym, period="1d", interval="1m", retries=2)
        if not q.empty and "Close" in q.columns:
            close = pd.to_numeric(q["Close"], errors="coerce").dropna()
            if not close.empty:
                ts = pd.to_datetime(close.index[-1], errors="coerce")
                if not pd.isna(ts):
                    price = float(close.iloc[-1])
                    delta = float(close.iloc[-1] - close.iloc[-2]) if len(close) >= 2 else 0.0
                    item = (ts, price, delta, f"{sym}:1m")
                    if best is None or item[0] > best[0]:
                        best = item
                    continue

        # Fallback: fast_info when 1m is unavailable.
        try:
            ticker = yf.Ticker(sym)
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                fast_last = fast_info.get("lastPrice") or fast_info.get("last_price")
                if fast_last is not None:
                    price = float(fast_last)
                    if price > 0:
                        item = (pd.Timestamp.utcnow(), price, 0.0, f"{sym}:fast")
                        if best is None or item[0] > best[0]:
                            best = item
        except Exception:
            pass

    if best is None:
        return None, 0.0, None
    return best[1], best[2], best[3]


def get_data(symbol: str, period: str, interval: str):
    data = safe_yf_download(symbol, period=period, interval=interval)
    dxy = safe_yf_download("DX-Y.NYB", period=period, interval=interval)

    # Fix MultiIndex columns for newer pandas versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex):
        dxy.columns = dxy.columns.get_level_values(0)
    return data, dxy


def now_iran_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        # Try using ZoneInfo first (Python 3.9+)
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo("Asia/Tehran")).strftime(fmt)
        except ImportError:
            # Fallback for older Python versions
            import pytz
            return datetime.now(pytz.timezone("Asia/Tehran")).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt) + " UTC"


def get_market_context(period: str, interval: str):
    # Key intermarket drivers for gold
    symbols = {
        "dxy": "DX-Y.NYB",
        "us10y": "^TNX",
        "silver": "SI=F",
        "copper": "HG=F",
    }
    out = {}
    for key, ticker in symbols.items():
        ctx_df = safe_yf_download(ticker, period=period, interval=interval)
        if isinstance(ctx_df.columns, pd.MultiIndex):
            ctx_df.columns = ctx_df.columns.get_level_values(0)
        out[key] = ctx_df
    return out

def get_higher_timeframe(symbol: str, base_interval: str):
    mapping = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h", "4h": "1d", "1d": "1wk"}
    higher = mapping.get(base_interval, "1d")

    period = "60d" if higher in ["1m", "5m", "15m"] else "120d" if higher in ["1h", "4h"] else "5y"
    data = safe_yf_download(symbol, period=period, interval=higher)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data, higher

def calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def col_as_series(frame: pd.DataFrame, name: str) -> pd.Series:
        col = frame[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return pd.to_numeric(col, errors="coerce")

    close_s = col_as_series(df, "Close")
    open_s = col_as_series(df, "Open")
    high_s = col_as_series(df, "High")
    low_s = col_as_series(df, "Low")

    df["Body"] = (open_s - close_s).abs()
    df["Wick_Upper"] = high_s - df[["Open", "Close"]].max(axis=1).squeeze()
    df["Wick_Lower"] = df[["Open", "Close"]].min(axis=1).squeeze() - low_s

    df["Is_Doji"] = df["Body"] <= (high_s - low_s) * 0.1
    df["Is_Hammer"] = (df["Wick_Lower"] > df["Body"] * 2) & (df["Wick_Upper"] < df["Body"] * 0.5)
    return df

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).fillna(0).cumsum()


def safe_last(series: pd.Series, default: float = 0.0) -> float:
    valid = series.dropna()
    if valid.empty:
        return default
    return float(valid.iloc[-1])


def pct_change_n(series: pd.Series, n: int) -> float:
    valid = series.dropna()
    if len(valid) <= n:
        return 0.0
    prev = float(valid.iloc[-(n + 1)])
    curr = float(valid.iloc[-1])
    if prev == 0:
        return 0.0
    return (curr / prev - 1.0) * 100.0


def get_market_structure(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 80,
    volume: pd.Series | None = None,
) -> dict:
    hs = pd.to_numeric(high, errors="coerce").dropna()
    ls = pd.to_numeric(low, errors="coerce").dropna()
    cs = pd.to_numeric(close, errors="coerce").dropna()
    if hs.empty or ls.empty or cs.empty:
        return {
            "support": 0.0,
            "resistance": 0.0,
            "near_support": False,
            "near_resistance": False,
            "round_bias": "NEUTRAL",
            "signal": "NEUTRAL",
            "reason": "market structure unavailable",
            "strength": 0.0
        }
    
    lb = min(lookback, len(cs))
    window_high = hs.iloc[-lb:]
    window_low = ls.iloc[-lb:]
    current = float(cs.iloc[-1])
    support = float(window_low.quantile(0.15))  # More aggressive support level
    resistance = float(window_high.quantile(0.85))  # More aggressive resistance level
    rng = max(resistance - support, max(current * 0.002, 1e-6))
    
    # Calculate proximity percentages
    support_proximity = (current - support) / rng if rng > 0 else 0
    resistance_proximity = (resistance - current) / rng if rng > 0 else 0
    near_support = support_proximity <= 0.15
    near_resistance = resistance_proximity <= 0.15
    
    # Round-number reaction zones (gold often reacts around .00/.50 levels).
    near_00 = abs(current - round(current)) <= 0.15
    near_50 = abs(current - (np.floor(current) + 0.5)) <= 0.15
    
    # Enhanced trend confirmation
    ema20_above_ema50 = safe_last(ema20) > safe_last(ema50)
    price_above_ema20 = current > safe_last(ema20)
    price_below_ema20 = current < safe_last(ema20)
    
    round_bias = "NEUTRAL"
    signal = "NEUTRAL"
    reason = "range center"
    strength = 0.0
    
    if current > resistance:
        if price_above_ema20 and ema20_above_ema50:
            signal = "BUY"
            reason = "strong structure breakout above resistance with trend confirmation"
            strength = 0.8
        else:
            signal = "BUY"
            reason = "structure breakout above resistance"
            strength = 0.6
        round_bias = "BUY"
    elif current < support:
        if price_below_ema20 and not ema20_above_ema50:
            signal = "SELL"
            reason = "strong structure breakdown below support with trend confirmation"
            strength = 0.8
        else:
            signal = "SELL"
            reason = "structure breakdown below support"
            strength = 0.6
        round_bias = "SELL"
    elif near_support and (near_00 or near_50):
        if volume_confirmation:  # Use volume confirmation
            signal = "BUY"
            reason = "strong reaction near support zone with volume confirmation"
            strength = 0.7
        else:
            signal = "BUY"
            reason = "reaction near support zone"
            strength = 0.5
        round_bias = "BUY"
    elif near_resistance and (near_00 or near_50):
        if volume_confirmation:  # Use volume confirmation
            signal = "SELL"
            reason = "strong reaction near resistance zone with volume confirmation"
            strength = 0.7
        else:
            signal = "SELL"
            reason = "reaction near resistance zone"
            strength = 0.5
        round_bias = "SELL"
    
    # Calculate market volatility for filtering
    atr = AverageTrueRange(high, low, close).average_true_range()
    atr_series = pd.to_numeric(atr, errors="coerce").dropna()
    market_volatility = (atr_series.iloc[-1] / current) * 100 if len(atr_series) > 0 and current > 0 else 0.0
    
    # Calculate volume confirmation
    vol_sma = safe_last(volume.rolling(20).mean(), default=0.0) if volume is not None else 0.0
    vol_last = safe_last(volume, default=0.0) if volume is not None else 0.0
    rel_vol = vol_last / max(vol_sma, 1e-9) if vol_sma > 0 else 1.0
    volume_confirmation = rel_vol >= 1.2
    
    # Additional filter: avoid weak signals in choppy markets
    if market_volatility > 2.5 and strength < 0.7:
        signal = "NEUTRAL"
        reason = "structure signal filtered: choppy market"
        round_bias = "NEUTRAL"
        strength = 0.0
    
    return {
        "support": support,
        "resistance": resistance,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "round_bias": round_bias,
        "signal": signal,
        "reason": reason,
        "strength": strength,
        "volume_confirmation": volume_confirmation,
        "market_volatility": market_volatility
    }


def get_higher_tf_trend(df_higher: pd.DataFrame) -> tuple[str | None, float]:
    if df_higher.empty or "Close" not in df_higher.columns:
        return None, 0.0
    ht_close = pd.to_numeric(df_higher["Close"], errors="coerce").dropna()
    if len(ht_close) < 220:
        return None, 0.0
    ht_ema50 = EMAIndicator(ht_close, window=50).ema_indicator()
    ht_ema200 = EMAIndicator(ht_close, window=200).ema_indicator()
    if ht_ema50.dropna().empty or ht_ema200.dropna().empty:
        return None, 0.0
    slope = float(ht_ema50.iloc[-1] - ht_ema50.iloc[-4]) if len(ht_ema50.dropna()) > 5 else 0.0
    if ht_close.iloc[-1] > ht_ema200.iloc[-1] and ht_ema50.iloc[-1] > ht_ema200.iloc[-1] and slope > 0:
        return "UP", slope
    if ht_close.iloc[-1] < ht_ema200.iloc[-1] and ht_ema50.iloc[-1] < ht_ema200.iloc[-1] and slope < 0:
        return "DOWN", slope
    return "RANGE", slope


JOURNAL_PATH = Path("signal_journal.csv")


def detect_market_regime(curr_adx: float, curr_atr: float, atr_baseline: float) -> dict:
    atr_ratio = curr_atr / max(atr_baseline, 1e-9)
    if curr_adx >= 25 and atr_ratio <= 1.5:
        return {"name": "TREND", "score_mult": 1.12, "conf_bonus": 4.0}
    if atr_ratio > 1.5:
        return {"name": "HIGH_VOL", "score_mult": 0.88, "conf_bonus": -5.0}
    return {"name": "RANGE", "score_mult": 0.94, "conf_bonus": -2.0}


def compute_signal_probability(
    bias_score: float,
    confidence: float,
    method_net: float,
    method_total_weight: float,
    regime_name: str,
    backtest_win_rate: float | None = None,
    backtest_trades: int = 0,
    oos_win_rate: float | None = None,
    oos_folds: int = 0,
) -> dict:
    agreement = abs(method_net) / max(method_total_weight, 1e-9)
    base_prob = 48.0 + abs(bias_score) * 0.28 + confidence * 0.22 + agreement * 8.0
    if regime_name == "TREND":
        base_prob += 3.0
    elif regime_name == "HIGH_VOL":
        base_prob -= 4.0
    raw_win_prob = max(35.0, min(92.0, base_prob))

    observed_rates = []
    observed_weights = []
    if backtest_win_rate is not None and backtest_trades > 0:
        observed_rates.append(float(backtest_win_rate))
        observed_weights.append(min(1.0, backtest_trades / 120.0))
    if oos_win_rate is not None and oos_folds > 0:
        observed_rates.append(float(oos_win_rate))
        observed_weights.append(min(1.0, oos_folds / 4.0))

    if observed_rates:
        empirical_rate = float(np.average(observed_rates, weights=observed_weights))
        calib_weight = min(0.7, 0.25 + 0.45 * min(1.0, sum(observed_weights) / 1.6))
    else:
        empirical_rate = 50.0
        calib_weight = 0.0

    win_prob = (1.0 - calib_weight) * raw_win_prob + calib_weight * empirical_rate
    win_prob = max(38.0, min(89.0, win_prob))
    loss_prob = 100.0 - win_prob

    data_quality = "LOW"
    if calib_weight >= 0.5:
        data_quality = "HIGH"
    elif calib_weight >= 0.25:
        data_quality = "MEDIUM"

    return {
        "win_prob": win_prob,
        "loss_prob": loss_prob,
        "raw_win_prob": raw_win_prob,
        "agreement": agreement * 100.0,
        "empirical_rate": empirical_rate,
        "calibration_weight": calib_weight * 100.0,
        "data_quality": data_quality,
    }


def load_signal_journal(path: Path = JOURNAL_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_signal_journal(entry: dict, path: Path = JOURNAL_PATH) -> None:
    try:
        df = load_signal_journal(path)
        new_row = pd.DataFrame([entry])
        if not df.empty:
            same = (
                (df["asset"] == entry["asset"])
                & (df["timeframe"] == entry["timeframe"])
                & (df["signal"] == entry["signal"])
            )
            if same.any():
                last_ts = pd.to_datetime(df.loc[same, "timestamp"], errors="coerce").max()
                if pd.notna(last_ts):
                    if (pd.Timestamp.utcnow() - last_ts).total_seconds() < 300:
                        return
        out = pd.concat([df, new_row], ignore_index=True) if not df.empty else new_row
        out.tail(5000).to_csv(path, index=False)
    except Exception:
        pass


def in_high_impact_news_window(events: list, window_minutes: int = 45) -> tuple[bool, str]:
    now = pd.Timestamp.utcnow()
    for ev in events or []:
        if str(ev.get("impact", "")).lower() != "high":
            continue
        dt_str = f"{ev.get('date', '')} {ev.get('time', '')}".strip()
        ts = pd.to_datetime(dt_str, errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        diff_min = abs((ts - now).total_seconds()) / 60.0
        if diff_min <= window_minutes:
            return True, f"{ev.get('event', 'High Impact Event')} ({diff_min:.0f}m)"
    return False, ""


def ai_confirmation_signal(close: pd.Series) -> tuple[str, str, float]:
    """Lightweight local AI-style confirmation using trend + momentum."""
    valid = pd.to_numeric(close, errors="coerce").dropna()
    if len(valid) < 30:
        return "NEUTRAL", "Not enough data for AI confirmation", 0.0

    n = min(60, len(valid))
    y = valid.iloc[-n:].values.astype(float)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    projected_5bar = (slope * 5.0) / max(y[-1], 1e-9) * 100.0
    momentum_5bar = ((y[-1] / max(y[-6], 1e-9)) - 1.0) * 100.0 if n >= 6 else 0.0
    score = projected_5bar * 0.7 + momentum_5bar * 0.3

    if score > 0.20:
        return "BUY", f"AI trend+momentum score {score:.3f}", min(90.0, abs(score) * 120.0)
    if score < -0.20:
        return "SELL", f"AI trend+momentum score {score:.3f}", min(90.0, abs(score) * 120.0)
    return "NEUTRAL", f"AI trend+momentum score {score:.3f}", min(70.0, abs(score) * 100.0)


def ai_confirmation_external(
    api_key: str,
    model: str,
    close: pd.Series,
    curr_rsi: float,
    curr_macd_hist: float,
    curr_adx: float,
) -> tuple[str, str, float]:
    """Optional external AI confirmation via OpenAI-compatible Chat Completions API."""
    valid = pd.to_numeric(close, errors="coerce").dropna()
    if len(valid) < 30:
        return "NEUTRAL", "Not enough data for external AI", 0.0

    recent = valid.tail(40).tolist()
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a trading signal confirmer. Return strict JSON with keys: "
                    "signal (BUY/SELL/NEUTRAL), confidence (0-100), reason (short)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "close_40": recent,
                        "rsi": round(float(curr_rsi), 4),
                        "macd_hist": round(float(curr_macd_hist), 6),
                        "adx": round(float(curr_adx), 4),
                    }
                ),
            },
        ],
    }
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=12,
        )
        if r.status_code != 200:
            return "NEUTRAL", f"External AI HTTP {r.status_code}", 0.0
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content) if isinstance(content, str) else content
        sig = str(parsed.get("signal", "NEUTRAL")).upper().strip()
        if sig not in {"BUY", "SELL", "NEUTRAL"}:
            sig = "NEUTRAL"
        conf = float(parsed.get("confidence", 0.0))
        conf = max(0.0, min(100.0, conf))
        reason = str(parsed.get("reason", "External AI confirmation"))
        return sig, reason, conf
    except Exception as e:
        return "NEUTRAL", f"External AI error: {e}", 0.0


def resolve_ai_api_key() -> str:
    """Resolve API key from Streamlit secrets or env vars for hands-free mode."""
    try:
        key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key_from_secrets = ""
    return str(key_from_secrets or os.getenv("OPENAI_API_KEY", "")).strip()

# --- Sentiment Analysis Functions ---
def get_gold_news(api_key: str, days_back: int = 7) -> list:
    """Get gold-related news from NewsAPI"""
    keywords = ["gold", "federal reserve", "inflation", "interest rates", "geopolitical", "recession"]
    all_articles = []
    
    for keyword in keywords:
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': keyword,
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days_back)).isoformat(),
            'to': datetime.now().isoformat(),
            'pageSize': 10,
            'apiKey': api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                all_articles.extend(articles)
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            break
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title = article.get('title', '')
        if title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)
    
    return unique_articles[:20]  # Return top 20 unique articles

def analyze_sentiment_simple(text: str) -> dict:
    """Simple sentiment analysis using keyword-based approach"""
    positive_words = [
        'bullish', 'rally', 'surge', 'jump', 'rise', 'increase', 'growth', 'strong', 'positive',
        'optimistic', 'gain', 'profit', 'boom', 'expansion', 'recovery', 'upward', 'higher'
    ]
    
    negative_words = [
        'bearish', 'fall', 'drop', 'decline', 'decrease', 'loss', 'weak', 'negative',
        'pessimistic', 'crash', 'recession', 'crisis', 'risk', 'concern', 'pressure', 'downward'
    ]
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = positive_count + negative_count
    
    if total_words == 0:
        return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.5}
    
    sentiment_score = (positive_count - negative_count) / total_words
    confidence = min(0.9, total_words / 10.0)  # Confidence based on word count
    
    if sentiment_score > 0.2:
        sentiment = 'bullish'
    elif sentiment_score < -0.2:
        sentiment = 'bearish'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'score': sentiment_score,
        'confidence': confidence
    }

def calculate_overall_sentiment(articles: list) -> dict:
    """Calculate overall sentiment from multiple articles"""
    if not articles:
        return {'overall': 'neutral', 'score': 0.0, 'confidence': 0.0, 'count': 0}
    
    sentiments = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title} {description}"
        sentiment = analyze_sentiment_simple(text)
        sentiments.append(sentiment)
    
    if not sentiments:
        return {'overall': 'neutral', 'score': 0.0, 'confidence': 0.0, 'count': 0}
    
    # Calculate weighted average based on confidence
    total_weight = sum(s['confidence'] for s in sentiments)
    if total_weight == 0:
        return {'overall': 'neutral', 'score': 0.0, 'confidence': 0.0, 'count': len(sentiments)}
    
    weighted_score = sum(s['score'] * s['confidence'] for s in sentiments) / total_weight
    avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
    
    if weighted_score > 0.15:
        overall_sentiment = 'bullish'
    elif weighted_score < -0.15:
        overall_sentiment = 'bearish'
    else:
        overall_sentiment = 'neutral'
    
    return {
        'overall': overall_sentiment,
        'score': weighted_score,
        'confidence': avg_confidence,
        'count': len(sentiments),
        'articles_analyzed': len(sentiments)
    }

# --- Macro Economic Functions ---
def get_real_yields() -> dict:
    """Calculate real yields (nominal yield - inflation proxy)."""
    try:
        def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df

        tn10y = normalize_cols(safe_yf_download("^TNX", period="1mo", interval="1d"))
        if tn10y.empty or "Close" not in tn10y.columns:
            return {'nominal_yield': 0, 'inflation_rate': 0, 'real_yield': 0, 'trend': 'neutral'}

        current_yield = float(tn10y["Close"].dropna().iloc[-1])

        # CPI ticker in yfinance is unstable; use TIP (inflation-linked ETF) as proxy.
        tip = normalize_cols(safe_yf_download("TIP", period="1y", interval="1mo"))
        if not tip.empty and "Close" in tip.columns and len(tip["Close"].dropna()) >= 2:
            prev_tip = float(tip["Close"].dropna().iloc[-2])
            curr_tip = float(tip["Close"].dropna().iloc[-1])
            inflation_rate = ((curr_tip - prev_tip) / prev_tip) * 100 if prev_tip != 0 else 0.0
        else:
            # Fallback conservative default when proxy is unavailable
            inflation_rate = 2.5

        real_yield = current_yield - inflation_rate
        return {
            'nominal_yield': current_yield,
            'inflation_rate': inflation_rate,
            'real_yield': real_yield,
            'trend': 'positive' if real_yield > 0 else 'negative'
        }
    except Exception as e:
        # Keep app alive on data-source failures
        st.warning(f"Real yield data fallback: {e}")
    
    return {'nominal_yield': 0, 'inflation_rate': 0, 'real_yield': 0, 'trend': 'neutral'}

def get_fed_watch_data() -> dict:
    """Get Fed meeting probabilities and rate expectations"""
    try:
        # This would typically use a Fed data API
        # For now, we'll simulate with recent market data
        fed_funds = safe_yf_download("^FVX", period="3mo", interval="1d")  # Fed Funds Futures
        
        if not fed_funds.empty:
            current_rate = float(fed_funds["Close"].iloc[-1])
            implied_rate = current_rate * 100  # Convert to percentage
            
            # Simple probability calculation based on futures curve
            if implied_rate > 5.25:
                prob_hike = min(0.8, (implied_rate - 5.25) * 2)
                prob_cut = 0.0
            elif implied_rate < 5.0:
                prob_cut = min(0.6, (5.0 - implied_rate) * 2)
                prob_hike = 0.0
            else:
                prob_hike = prob_cut = 0.2
            
            return {
                'current_rate': implied_rate,
                'prob_hike': prob_hike,
                'prob_cut': prob_cut,
                'prob_hold': 1 - prob_hike - prob_cut
            }
    except Exception as e:
        st.error(f"Error getting Fed data: {e}")
    
    return {'current_rate': 5.25, 'prob_hike': 0.2, 'prob_cut': 0.2, 'prob_hold': 0.6}

def get_economic_calendar() -> list:
    """Get upcoming economic events"""
    # This would typically use an economic calendar API
    # For demo purposes, we'll return sample events
    today = datetime.now()
    events = [
        {
            'date': (today + timedelta(days=3)).strftime('%Y-%m-%d'),
            'time': '13:30',
            'event': 'CPI (MoM)',
            'impact': 'High',
            'forecast': '0.3%',
            'previous': '0.4%'
        },
        {
            'date': (today + timedelta(days=7)).strftime('%Y-%m-%d'),
            'time': '13:30',
            'event': 'Non-Farm Payrolls',
            'impact': 'High',
            'forecast': '180K',
            'previous': '175K'
        },
        {
            'date': (today + timedelta(days=14)).strftime('%Y-%m-%d'),
            'time': '14:00',
            'event': 'FOMC Rate Decision',
            'impact': 'High',
            'forecast': '5.25%',
            'previous': '5.25%'
        }
    ]
    return events

# --- Smart Position Sizing Functions ---
def calculate_smart_position_size(
    base_lot_size: float,
    confidence: float,
    atr: float,
    atr_baseline: float | None = None,
    sentiment_data: dict = None,
    risk_multiplier: float = 1.0,
) -> dict:
    """Calculate smart position size based on confidence, volatility, and sentiment"""
    
    # Base confidence adjustment
    confidence_factor = 0.5 + (confidence / 100.0) * 1.5  # Range: 0.5x to 2.0x
    
    # Volatility adjustment (lower volatility vs baseline -> higher position size)
    if atr_baseline is None or atr_baseline <= 0:
        atr_baseline = atr
    volatility_factor = min(2.0, max(0.5, atr_baseline / max(atr, 1e-9)))
    
    # Sentiment adjustment
    sentiment_factor = 1.0
    if sentiment_data:
        if sentiment_data['overall'] == 'bullish':
            sentiment_factor = 1.0 + (sentiment_data['confidence'] * 0.3)  # Max 1.3x
        elif sentiment_data['overall'] == 'bearish':
            sentiment_factor = 1.0 - (sentiment_data['confidence'] * 0.2)  # Min 0.8x
    
    # Calculate final position size
    sizing_factor = confidence_factor * volatility_factor * sentiment_factor * risk_multiplier
    adjusted_lot_size = base_lot_size * sizing_factor
    
    # Risk management limits
    max_position_multiplier = 3.0
    min_position_multiplier = 0.3
    
    sizing_factor = min(max_position_multiplier, max(min_position_multiplier, sizing_factor))
    adjusted_lot_size = base_lot_size * sizing_factor
    
    return {
        'base_lot_size': base_lot_size,
        'adjusted_lot_size': adjusted_lot_size,
        'confidence_factor': confidence_factor,
        'volatility_factor': volatility_factor,
        'sentiment_factor': sentiment_factor,
        'sizing_factor': sizing_factor,
        'risk_multiplier': risk_multiplier
    }

# --- Backtesting Functions ---
def run_backtest(df: pd.DataFrame, timeframe: str, initial_balance: float = 10000) -> dict:
    """Run backtest on historical data"""
    
    trades = []
    balance = initial_balance
    equity_curve = [initial_balance]
    max_balance = initial_balance
    drawdowns = []
    
    # Calculate indicators for backtest
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    
    rsi = RSIIndicator(close).rsi()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    macd = MACD(close)
    bb = BollingerBands(close)
    adx = ADXIndicator(high, low, close)
    
    for i in range(200, len(df)):  # Start from index 200 to have enough data
        current_price = close.iloc[i]
        current_rsi = rsi.iloc[i]
        current_ema50 = ema50.iloc[i]
        current_ema200 = ema200.iloc[i]
        current_macd_hist = macd.macd_diff().iloc[i]
        current_adx = adx.adx().iloc[i]
        
        # Simple signal logic for backtest
        signal = "NEUTRAL"
        long_score = 0
        short_score = 0
        
        # Trend analysis
        if current_ema50 > current_ema200:
            long_score += 2
        else:
            short_score += 2
            
        # Momentum
        if current_rsi > 55:
            long_score += 1
        elif current_rsi < 45:
            short_score += 1
            
        # MACD
        if current_macd_hist > 0:
            long_score += 1
        else:
            short_score += 1
            
        # ADX strength
        if current_adx >= 25:
            if current_price > current_ema50:
                long_score += 1
            else:
                short_score += 1
        
        # Generate signal
        if long_score >= 4:
            signal = "BUY"
        elif short_score >= 4:
            signal = "SELL"
        
        # Execute trades (simplified)
        if signal == "BUY" and len(trades) == 0:
            entry_price = current_price
            atr = bb.bollinger_hband().iloc[i] - bb.bollinger_lband().iloc[i]
            sl = entry_price - (2 * atr)
            tp = entry_price + (3 * atr)
            
            trades.append({
                'entry_date': df.index[i],
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp,
                'type': 'BUY'
            })
            
        elif signal == "SELL" and len(trades) == 0:
            entry_price = current_price
            atr = bb.bollinger_hband().iloc[i] - bb.bollinger_lband().iloc[i]
            sl = entry_price + (2 * atr)
            tp = entry_price - (3 * atr)
            
            trades.append({
                'entry_date': df.index[i],
                'entry_price': entry_price,
                'sl': sl,
                'tp': tp,
                'type': 'SELL'
            })
        
        # Check existing trades
        if trades:
            trade = trades[-1]
            if trade['type'] == 'BUY':
                if current_price <= trade['sl'] or current_price >= trade['tp']:
                    exit_price = trade['tp'] if current_price >= trade['tp'] else trade['sl']
                    pnl = (exit_price - trade['entry_price']) / trade['entry_price'] * 100
                    balance *= (1 + pnl / 100)
                    trades[-1]['exit_date'] = df.index[i]
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades.append({})  # Empty trade to prevent immediate re-entry
            else:  # SELL
                if current_price >= trade['sl'] or current_price <= trade['tp']:
                    exit_price = trade['tp'] if current_price <= trade['tp'] else trade['sl']
                    pnl = (trade['entry_price'] - exit_price) / trade['entry_price'] * 100
                    balance *= (1 + pnl / 100)
                    trades[-1]['exit_date'] = df.index[i]
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades.append({})  # Empty trade to prevent immediate re-entry
        
        # Update equity curve
        if trades and trades[-1].get('pnl') is not None:
            equity_curve.append(balance)
            max_balance = max(max_balance, balance)
            drawdown = (max_balance - balance) / max_balance * 100
            drawdowns.append(drawdown)
    
    # Clean trades (remove empty entries)
    completed_trades = [t for t in trades if t.get('pnl') is not None]
    
    # Calculate metrics
    if completed_trades:
        wins = [t for t in completed_trades if t['pnl'] > 0]
        losses = [t for t in completed_trades if t['pnl'] < 0]
        
        total_trades = len(completed_trades)
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        total_return = (balance - initial_balance) / initial_balance * 100
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] for i in range(1, len(equity_curve))]
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = avg_return / std_return * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate profit factor
        total_wins = sum(t['pnl'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
    else:
        total_trades = win_rate = avg_win = avg_loss = 0
        total_return = max_drawdown = sharpe_ratio = profit_factor = 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': (win_rate / 100.0) * avg_win + (1 - win_rate / 100.0) * avg_loss,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'final_balance': balance,
        'trades': completed_trades[:10]  # Return last 10 trades for display
    }


def run_walkforward_backtest(df: pd.DataFrame, timeframe: str, splits: int = 3, train_ratio: float = 0.7) -> dict:
    if df.empty or len(df) < 350:
        return {}
    fold_size = len(df) // splits
    fold_results = []
    for i in range(splits):
        start = i * fold_size
        end = len(df) if i == splits - 1 else (i + 1) * fold_size
        fold_df = df.iloc[start:end].copy()
        if len(fold_df) < 250:
            continue
        split_at = int(len(fold_df) * train_ratio)
        oos_df = fold_df.iloc[split_at:]
        if len(oos_df) < 120:
            continue
        res = run_backtest(oos_df, timeframe)
        fold_results.append(res)
    if not fold_results:
        return {}
    return {
        "folds": len(fold_results),
        "oos_win_rate": float(np.mean([r.get("win_rate", 0.0) for r in fold_results])),
        "oos_return": float(np.mean([r.get("total_return", 0.0) for r in fold_results])),
        "oos_drawdown": float(np.mean([r.get("max_drawdown", 0.0) for r in fold_results])),
        "oos_profit_factor": float(np.mean([r.get("profit_factor", 0.0) for r in fold_results])),
        "oos_expectancy": float(np.mean([r.get("expectancy", 0.0) for r in fold_results])),
    }

# --- Advanced Correlation Analysis Functions ---
def calculate_advanced_correlations(asset_name: str) -> dict:
    """Calculate correlations with multiple assets over different periods"""
    
    # Get gold data
    gold_df, _ = get_data(asset_name, "180d", "1d")
    if gold_df.empty:
        return {}
    
    gold_returns = gold_df["Close"].pct_change().dropna()
    
    # Assets to correlate with
    correlation_assets = {
        'sp500': '^GSPC',
        'bitcoin': 'BTC-USD', 
        'crude_oil': 'CL=F'
    }
    
    correlations = {}
    
    for asset_key, ticker in correlation_assets.items():
        try:
            asset_df = safe_yf_download(ticker, period="180d", interval="1d")
            if not asset_df.empty:
                asset_returns = asset_df["Close"].pct_change().dropna()
                
                # Align the data
                aligned_data = pd.concat([gold_returns, asset_returns], axis=1, join='inner')
                aligned_data.columns = ['gold', asset_key]
                
                # Calculate correlations for different periods
                corr_30d = aligned_data.tail(30).corr().iloc[0, 1]
                corr_90d = aligned_data.tail(90).corr().iloc[0, 1] if len(aligned_data) >= 90 else corr_30d
                corr_180d = aligned_data.corr().iloc[0, 1]
                
                # Determine trend
                if corr_30d > corr_90d > corr_180d:
                    trend = 'strengthening'
                elif corr_30d < corr_90d < corr_180d:
                    trend = 'weakening'
                else:
                    trend = 'stable'
                
                correlations[asset_key] = {
                    'corr_30d': corr_30d,
                    'corr_90d': corr_90d,
                    'corr_180d': corr_180d,
                    'trend': trend
                }
        except Exception as e:
            st.error(f"Error calculating correlation for {asset_key}: {e}")
            correlations[asset_key] = {
                'corr_30d': 0, 'corr_90d': 0, 'corr_180d': 0, 'trend': 'stable'
            }
    
    return correlations

def create_correlation_heatmap(correlations: dict) -> go.Figure:
    """Create a heatmap visualization of correlations"""
    
    if not correlations:
        return go.Figure()
    
    # Prepare data for heatmap
    assets = list(correlations.keys())
    periods = ['30d', '90d', '180d']
    
    heatmap_data = []
    for asset in assets:
        row = [correlations[asset][f'corr_{period}'] for period in periods]
        heatmap_data.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[T['correlation_30d'], T['correlation_90d'], T['correlation_180d']],
        y=[T[asset] for asset in assets],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=T['correlation_matrix'],
        xaxis_title="Time Period",
        yaxis_title="Assets",
        template="plotly_dark",
        height=400
    )
    
    return fig

# --- i18n ---
TEXT = {
    "en": {
        "settings": "Settings",
        "select_asset": "Select Asset",
        "gold": "Gold",
        "silver": "Silver",
        "timeframe": "Timeframe",
        "risk_mgmt": "Risk Management",
        "balance": "Balance ($)",
        "risk_pct": "Risk (%)",
        "atr_mult": "ATR Multiplier (SL)",
        "rr": "Risk/Reward",
        "contract_size": "Contract Size",
        "contract_units": "Contract size (units)",
        "live_update": "Live Update",
        "auto_refresh": "Auto-refresh",
        "refresh_interval": "Refresh interval (sec)",
        "live_window": "Live window (candles)",
        "title": "AI Trading Terminal",
        "price": "Price",
        "rsi": "RSI",
        "atr": "ATR",
        "dxy": "DXY",
        "corr": "Corr",
        "signal_buy": "Signal: BUY",
        "signal_sell": "Signal: SELL",
        "signal_wait": "Signal: WAIT",
        "risk": "Risk",
        "risk_amount": "Risk amount",
        "lot_size": "Lot size",
        "targets": "Targets",
        "tp": "TP",
        "sl": "SL",
        "rr_fmt": "R:R = 1:{rr}",
        "logic": "Logic Breakdown",
        "higher_tf": "Higher timeframe",
        "trend": "Trend",
        "last_update": "Last update",
        "data_fail": "Data fetch failed! Check your internet connection.",
        "lang": "Language / زبان",
        "chart_mode": "Chart Mode",
        "chart_plotly": "Internal Chart",
        "chart_tv": "TradingView Live",
        "confidence": "Confidence",
        "bias_score": "Bias score",
        "entry_zone": "Entry zone",
        "bullish_factors": "Bullish factors",
        "bearish_factors": "Bearish factors",
        "method_title": "Strategy Signals",
        "method_price_action": "Price Action",
        "method_fib": "Fibonacci",
        "method_rsi": "RSI Regime",
        "method_macd": "MACD",
        "method_bollinger": "Bollinger",
        "method_fundamental": "Fundamental",
        "method_signal_col": "Signal",
        "method_conf_col": "Confidence",
        "method_reason_col": "Reason",
        "sig_buy": "BUY",
        "sig_sell": "SELL",
        "sig_neutral": "NEUTRAL",
        "sig_strong_buy": "STRONG BUY",
        "sig_strong_sell": "STRONG SELL",
        "adx_macd": "ADX / MACD",
        "corr_dxy": "Corr(DXY)",
        "tp2": "TP2",
        "buy_factors": "Bullish factors",
        "sell_factors": "Bearish factors",
        "dxy_us10y": "DXY and US10Y returns",
        "silver_copper": "Silver and Copper returns",
        "sentiment_analysis": "Sentiment Analysis",
        "sentiment_score": "Sentiment Score",
        "sentiment_confidence": "Sentiment Confidence",
        "news_count": "News Articles Analyzed",
        "overall_sentiment": "Overall Market Sentiment",
        "bullish": "Bullish",
        "bearish": "Bearish", 
        "neutral": "Neutral",
        "news_api_key": "NewsAPI Key (Optional)",
        "enable_sentiment": "Enable Sentiment Analysis",
        "sentiment_impact": "Sentiment Impact on Signal",
        "macro_dashboard": "Macro Dashboard",
        "real_yields": "Real Yields (10Y)",
        "nominal_yield": "Nominal Yield",
        "inflation_rate": "Inflation Rate",
        "real_yield": "Real Yield",
        "fed_watch": "Fed Watch Tool",
        "current_rate": "Current Rate",
        "prob_hike": "Prob. Hike",
        "prob_cut": "Prob. Cut",
        "prob_hold": "Prob. Hold",
        "economic_calendar": "Economic Calendar",
        "upcoming_events": "Upcoming Events",
        "event": "Event",
        "impact": "Impact",
        "forecast": "Forecast",
        "previous": "Previous",
        "smart_position_sizing": "Smart Position Sizing",
        "confidence_based_sizing": "Confidence-Based Sizing",
        "base_position": "Base Position",
        "adjusted_position": "Adjusted Position",
        "sizing_factor": "Sizing Factor",
        "volatility_adjustment": "Volatility Adjustment",
        "sentiment_adjustment": "Sentiment Adjustment",
        "risk_multiplier": "Risk Multiplier",
        "backtesting": "Backtesting",
        "backtest_period": "Backtest Period",
        "total_trades": "Total Trades",
        "win_rate": "Win Rate",
        "profit_factor": "Profit Factor",
        "max_drawdown": "Max Drawdown",
        "total_return": "Total Return",
        "avg_win": "Avg Win",
        "avg_loss": "Avg Loss",
        "sharpe_ratio": "Sharpe Ratio",
        "run_backtest": "Run Backtest",
        "backtest_results": "Backtest Results",
        "correlation_matrix": "Advanced Correlation Analysis",
        "correlation_with": "Correlation with",
        "sp500": "S&P 500",
        "bitcoin": "Bitcoin",
        "crude_oil": "Crude Oil",
        "correlation_30d": "30-Day Correlation",
        "correlation_90d": "90-Day Correlation",
        "correlation_180d": "180-Day Correlation",
        "correlation_trend": "Correlation Trend",
        "strengthening": "Strengthening",
        "weakening": "Weakening",
        "stable": "Stable"
    },
    "fa": {
        "settings": "تنظیمات",
        "select_asset": "انتخاب دارایی",
        "gold": "طلا",
        "silver": "نقره",
        "timeframe": "تایم‌فریم",
        "risk_mgmt": "مدیریت ریسک",
        "balance": "موجودی ($)",
        "risk_pct": "ریسک (%)",
        "atr_mult": "ضریب ATR (حد ضرر)",
        "rr": "ریسک/سود",
        "contract_size": "اندازه قرارداد",
        "contract_units": "اندازه قرارداد (واحد)",
        "live_update": "آپدیت زنده",
        "auto_refresh": "آپدیت خودکار",
        "refresh_interval": "فاصله آپدیت (ثانیه)",
        "live_window": "پنجره زنده (کندل)",
        "title": "ترمینال معامله‌گری هوشمند",
        "price": "قیمت",
        "rsi": "RSI",
        "atr": "ATR",
        "dxy": "DXY",
        "corr": "همبستگی",
        "signal_buy": "سیگنال خرید",
        "signal_sell": "سیگنال فروش",
        "signal_wait": "سیگنال صبر",
        "risk": "ریسک",
        "risk_amount": "میزان ریسک",
        "lot_size": "اندازه لات",
        "targets": "اهداف",
        "tp": "حد سود",
        "sl": "حد ضرر",
        "rr_fmt": "ریسک/سود = 1:{rr}",
        "logic": "جزئیات منطق",
        "higher_tf": "تایم‌فریم بالاتر",
        "trend": "روند",
        "last_update": "آخرین آپدیت",
        "data_fail": "دریافت داده ناموفق بود. اتصال اینترنت را بررسی کنید.",
        "lang": "Language / زبان",
        "chart_mode": "حالت چارت",
        "chart_plotly": "چارت داخلی",
        "chart_tv": "تریدینگ‌ویو زنده",
        "confidence": "اعتماد تحلیل",
        "bias_score": "امتیاز جهت",
        "entry_zone": "محدوده ورود",
        "bullish_factors": "فاکتورهای صعودی",
        "bearish_factors": "فاکتورهای نزولی",
        "method_title": "سیگنال روش‌های تحلیلی",
        "method_price_action": "پرایس اکشن",
        "method_fib": "فیبوناچی",
        "method_rsi": "رژیم RSI",
        "method_macd": "مکدی",
        "method_bollinger": "بولینگر",
        "method_fundamental": "فاندامنتال",
        "method_signal_col": "سیگنال",
        "method_conf_col": "اعتماد",
        "method_reason_col": "دلیل",
        "sig_buy": "خرید",
        "sig_sell": "فروش",
        "sig_neutral": "خنثی",
        "sig_strong_buy": "خرید قوی",
        "sig_strong_sell": "فروش قوی",
        "adx_macd": "ای‌دی‌ایکس / مکدی",
        "corr_dxy": "همبستگی با DXY",
        "tp2": "حد سود دوم",
        "buy_factors": "فاکتورهای صعودی",
        "sell_factors": "فاکتورهای نزولی",
        "dxy_us10y": "بازده DXY و نرخ 10Y",
        "silver_copper": "بازده نقره و مس",
        "sentiment_analysis": "تحلیل سنتیمنت",
        "sentiment_score": "امتیاز سنتیمنت",
        "sentiment_confidence": "اعتماد سنتیمنت",
        "news_count": "تعداد اخبار تحلیل شده",
        "overall_sentiment": "سنتیمنت کلی بازار",
        "bullish": "صعودی",
        "bearish": "نزولی", 
        "neutral": "خنثی",
        "news_api_key": "کلید NewsAPI (اختیاری)",
        "enable_sentiment": "فعال‌سازی تحلیل سنتیمنت",
        "sentiment_impact": "تأثیر سنتیمنت بر سیگنال",
        "macro_dashboard": "داشبورد کلان اقتصادی",
        "real_yields": "بازده واقعی (۱۰Y)",
        "nominal_yield": "بازده اسمی",
        "inflation_rate": "نرخ تورم",
        "real_yield": "بازده واقعی",
        "fed_watch": "ابزار نظارت فدرال",
        "current_rate": "نرخ فعلی",
        "prob_hike": "احتمال افزایش",
        "prob_cut": "احتمال کاهش",
        "prob_hold": "احتمال ثبات",
        "economic_calendar": "تقویم اقتصادی",
        "upcoming_events": "رویدادهای پیش رو",
        "event": "رویداد",
        "impact": "تأثیر",
        "forecast": "پیش‌بینی",
        "previous": "قبلی",
        "smart_position_sizing": "پوزیشن‌سایزینگ هوشمند",
        "confidence_based_sizing": "سایزینگ بر اساس اعتماد",
        "base_position": "پوزیشن پایه",
        "adjusted_position": "پوزیشن تعدیل شده",
        "sizing_factor": "ضریب سایزینگ",
        "volatility_adjustment": "تعدیل نوسانات",
        "sentiment_adjustment": "تعدیل سنتیمنت",
        "risk_multiplier": "ضریب ریسک",
        "backtesting": "بکتست‌گیری",
        "backtest_period": "دوره بکتست",
        "total_trades": "کل معاملات",
        "win_rate": "نرخ برد",
        "profit_factor": "ضریب سود",
        "max_drawdown": "حداکثر افت",
        "total_return": "بازده کل",
        "avg_win": "میانگین سود",
        "avg_loss": "میانگین زیان",
        "sharpe_ratio": "نسبت شارپ",
        "run_backtest": "اجرای بکتست",
        "backtest_results": "نتایج بکتست",
        "correlation_matrix": "تحلیل همبستگی پیشرفته",
        "correlation_with": "همبستگی با",
        "sp500": "S&P 500",
        "bitcoin": "بیت‌کوین",
        "crude_oil": "نفت خام",
        "correlation_30d": "همبستگی ۳۰ روزه",
        "correlation_90d": "همبستگی ۹۰ روزه",
        "correlation_180d": "همبستگی ۱۸۰ روزه",
        "correlation_trend": "روند همبستگی",
        "strengthening": "تقویت",
        "weakening": "تضعیف",
        "stable": "پایدار"
    }
}

lang_choice = st.sidebar.selectbox(TEXT["en"]["lang"], ["فارسی", "English"], index=0)
lang = "fa" if lang_choice == "فارسی" else "en"
T = TEXT[lang]

# Light / dark theme switch
theme_choice = st.sidebar.selectbox("Theme / تم", ["Dark", "Light"], index=0)
is_light_theme = theme_choice == "Light"

if is_light_theme:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f3f6fb;
                --panel: #ffffff;
                --panel-2: #f8fbff;
                --line: #d4ddec;
                --txt: #1c2740;
                --muted: #5a6b8d;
                --buy: #0f9d64;
                --sell: #e53958;
                --neutral: #66758f;
            }
            .stApp {
                background: linear-gradient(180deg, #f3f6fb 0%, #eaf0f9 100%) !important;
                color: var(--txt) !important;
            }
            [data-testid="stSidebar"] {
                background: #eef3fb !important;
                border-right: 1px solid #d8e1ef !important;
            }
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] p {
                color: #344a71 !important;
                opacity: 1 !important;
            }
            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] select,
            [data-testid="stSidebar"] textarea {
                color: #112445 !important;
                background: rgba(255, 255, 255, 0.95) !important;
            }
            [data-testid="stSidebar"] .stNumberInput input,
            [data-testid="stSidebar"] .stTextInput input,
            [data-testid="stSidebar"] div[data-baseweb="input"] input {
                color: #112445 !important;
                -webkit-text-fill-color: #112445 !important;
                caret-color: #112445 !important;
                font-weight: 600 !important;
            }
            [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] > div,
            [data-testid="stSidebar"] .stTextInput div[data-baseweb="input"] > div {
                background: #ffffff !important;
                border: 1px solid #d6e0f0 !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-baseweb="input"] > div {
                color: #102344 !important;
                background: rgba(255, 255, 255, 0.97) !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"] svg {
                fill: #35527e !important;
            }
            .app-card {
                background: rgba(255, 255, 255, 0.95) !important;
                border: 1px solid #d6e0f0 !important;
                box-shadow: 0 2px 12px rgba(20, 34, 66, 0.08) !important;
            }
            [data-testid="stMetric"] {
                background: #ffffff !important;
                border: 1px solid #d6e0f0 !important;
            }
            [data-testid="stMetricLabel"] { color: #33476a !important; }
            [data-testid="stMetricValue"] { color: #0f1e39 !important; }
            [data-testid="stMetricDelta"] { color: #526788 !important; }
            h1, h2, h3, h4, h5 { color: #0f1e39 !important; }
            .app-card div, .app-card span, .app-card p { color: #2f4569 !important; }
            .top-status {
                background: rgba(255, 255, 255, 0.96) !important;
                border: 1px solid #d6e0f0 !important;
            }
            .top-status .k { color: #4f6487 !important; }
            .top-status .v { color: #102344 !important; }
            .top-status strong { color: #0f1e39 !important; }
            .hero-signal {
                background: linear-gradient(135deg, #ffffff, #f4f8ff) !important;
                border: 1px solid #d6e0f0 !important;
            }
            .hero-signal .h-label { color: #4f6487 !important; }
            .hero-signal .h-value { color: #0f1e39 !important; }
            .kpi-tile {
                background: #ffffff !important;
                border: 1px solid #d6e0f0 !important;
            }
            .kpi-tile .k { color: #4f6487 !important; }
            .kpi-tile .v { color: #0f1e39 !important; }
            .site-header { background: rgba(255,255,255,0.95) !important; border: 1px solid #d6e0f0 !important; }
            .site-header .brand { color: #0f1e39 !important; }
            .site-header .nav { color: #4f6487 !important; }
            .right-menu { background: #ffffff !important; border: 1px solid #d6e0f0 !important; }
            .site-footer { color: #4f6487 !important; border-top: 1px solid #d6e0f0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fa_text_is_broken(txt: str) -> bool:
    if not txt:
        return True
    if "?" in txt:
        return True
    # Mojibake-like patterns from broken encoding (very common in this file history).
    noisy = sum(txt.count(ch) for ch in ["?", "?", "?", "?", "?"])
    return noisy >= max(3, int(len(txt) * 0.2))


def tr(en_text: str, fa_text: str) -> str:
    if lang != "fa":
        return en_text
    return en_text if _fa_text_is_broken(fa_text) else fa_text

# --- Sidebar ---
st.sidebar.title(T["settings"])
st.sidebar.markdown(
    f"""
    <div class="focus-card">
      <div class="focus-title">{tr("Priority Controls", "تنظیمات مهم")}</div>
      <div class="focus-text">{tr("Timeframe, Risk %, Auto-refresh and Chart Mode have highest impact on signals.", "تایم‌فریم، درصد ریسک، آپدیت خودکار و حالت چارت بیشترین تاثیر را روی سیگنال دارند.")}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
asset_name = st.sidebar.selectbox(
    T["select_asset"],
    ["GC=F", "SI=F"],
    format_func=lambda x: T["gold"] if x == "GC=F" else T["silver"],
)
timeframe = st.sidebar.selectbox(T["timeframe"], ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
st.sidebar.caption(tr("Simple mode is active.", "حالت ساده فعال است."))

st.sidebar.markdown("---")
st.sidebar.subheader(T["risk_mgmt"])
acc_balance = st.sidebar.number_input(T["balance"], value=1000, key="balance_input")
risk_pct = st.sidebar.slider(T["risk_pct"], 0.5, 5.0, 2.0, key="risk_pct_input")
atr_mult = st.sidebar.slider(T["atr_mult"], 1.0, 4.0, 2.0, 0.5, key="atr_mult_input")
rr_ratio = st.sidebar.slider(T["rr"], 1.0, 5.0, 2.0, 0.5, key="rr_ratio_input")
max_dd_guard = st.sidebar.slider(tr("Max Drawdown Guard (%)", "حداکثر افت مجاز (%)"), 2.0, 30.0, 12.0, 0.5, key="max_dd_guard")
max_loss_streak_guard = st.sidebar.slider(tr("Max Loss Streak", "حداکثر زنجیره باخت"), 2, 8, 4, key="max_loss_streak_guard")

st.sidebar.markdown("---")
st.sidebar.subheader(T["contract_size"])
contract_defaults = {"GC=F": 100.0, "SI=F": 5000.0}
contract_size = st.sidebar.number_input(
    T["contract_units"],
    min_value=1.0,
    value=contract_defaults.get(asset_name, 100.0),
)

st.sidebar.markdown("---")
st.sidebar.subheader(T["live_update"])
auto_refresh = st.sidebar.checkbox(T["auto_refresh"], value=True)
refresh_sec = st.sidebar.slider(T["refresh_interval"], 2, 120, 8, 1)
live_window = st.sidebar.slider(T["live_window"], 50, 400, 150, 10)
chart_mode = st.sidebar.selectbox(T["chart_mode"], [T["chart_plotly"], T["chart_tv"]], index=0)

st.sidebar.markdown("---")
with st.sidebar.expander(T["sentiment_analysis"], expanded=False):
    enable_sentiment = st.checkbox(T["enable_sentiment"], value=False, key="enable_sentiment_compact")
    news_api_key = st.text_input(
        T["news_api_key"],
        type="password",
        help="Enter your NewsAPI key to enable sentiment analysis",
        key="news_api_key_compact",
    )
with st.sidebar.expander(T["smart_position_sizing"], expanded=False):
    enable_smart_sizing = st.checkbox(T["confidence_based_sizing"], value=True, key="enable_smart_sizing_compact")
    risk_multiplier = st.slider(T["risk_multiplier"], 0.5, 2.0, 1.0, 0.1, key="risk_multiplier_compact")
with st.sidebar.expander(T["backtesting"], expanded=False):
    enable_backtest = st.checkbox(T["run_backtest"], value=False, key="enable_backtest_compact")
    backtest_period = st.selectbox(
        T["backtest_period"],
        ["1 Month", "3 Months", "6 Months"],
        index=1,
        key="backtest_period_compact",
    )
with st.sidebar.expander("UI/UX", expanded=False):
    enable_audio_alerts = st.checkbox("Enable Audio Alerts", value=True, key="enable_audio_alerts_compact")
    enable_animations = st.checkbox("Enable Animations", value=True, key="enable_animations_compact")
with st.sidebar.expander(tr("AI Confirmation", "AI Confirmation"), expanded=False):
    enable_ai_confirmation = True
    ai_api_key = resolve_ai_api_key()
    ai_provider = tr("External API", "External API") if ai_api_key else tr("Free Local AI", "Free Local AI")
    ai_model = st.text_input(
        tr("AI Model", "AI Model"),
        value="gpt-4o-mini",
        key="ai_model",
    )
    st.caption(
        tr(
            f"Auto mode: {'External API' if ai_api_key else 'Free Local AI'} (no manual key input needed)",
            f"Auto mode: {'External API' if ai_api_key else 'Free Local AI'} (no manual key input needed)",
        )
    )
with st.sidebar.expander(tr("Signal Engine", "Signal Engine"), expanded=False):
    enforce_mtf_alignment = st.checkbox(tr("Strict higher-TF alignment", "هم‌جهتی سخت‌گیرانه تایم بالاتر"), value=True, key="enforce_mtf_alignment")
    enable_news_filter = st.checkbox(tr("High-impact news filter", "فیلتر خبرهای مهم"), value=True, key="enable_news_filter")
    news_filter_minutes = st.slider(tr("News lock window (min)", "???????? ?????? ?????? (??????????)"), 10, 90, 25, 5, key="news_filter_minutes")
    enable_quality_gate = st.checkbox(tr("Quality gate", "گیت کیفیت"), value=True, key="enable_quality_gate")
    min_adx_gate = st.slider(tr("Min ADX", "?????????? ADX"), 8, 35, 14, 1, key="min_adx_gate")
    min_agreement_gate = st.slider(tr("Min method agreement (%)", "?????????? ?????????? ????????????? (%)"), 35, 100, 50, 5, key="min_agreement_gate")
    st.caption(tr(
        "Active gates: Higher-TF alignment + News lock + Quality gate.",
        "Active gates: Higher-TF alignment + News lock + Quality gate."
    ))

st.sidebar.markdown("---")
st.sidebar.subheader(tr("Quick Actions", "اقدامات سریع"))
qa1, qa2 = st.sidebar.columns(2)
if qa1.button(tr("Reset Risk", "ریست ریسک"), width='stretch'):
    st.session_state["balance_input"] = 1000.0
    st.session_state["risk_pct_input"] = 2.0
    st.session_state["atr_mult_input"] = 2.0
    st.session_state["rr_ratio_input"] = 2.0
    st.session_state["max_dd_guard"] = 12.0
    st.session_state["max_loss_streak_guard"] = 4
    st.session_state["risk_multiplier_compact"] = 1.0
    st.session_state["enable_smart_sizing_compact"] = True
    st.rerun()
if qa2.button(tr("Reload", "بارگذاری"), width='stretch'):
    st.rerun()
snapshot_requested = st.sidebar.button(tr("Chart Snapshot (HTML)", "خروجی چارت (HTML)"), width='stretch')

# --- Main Logic ---
period_map = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "120d",
    "4h": "180d",
    "1d": "5y",
}
df, dxy = get_data(asset_name, period_map.get(timeframe, "120d"), timeframe)
market_ctx = get_market_context(period_map.get(timeframe, "120d"), timeframe)

# --- Sentiment Analysis ---
sentiment_data = None
sentiment_impact_score = 0

if enable_sentiment and news_api_key:
    with st.spinner(tr("Analyzing market sentiment...", "در حال تحلیل سنتیمنت بازار...")):
        articles = get_gold_news(news_api_key, days_back=7)
        sentiment_data = calculate_overall_sentiment(articles)
        
        # Calculate sentiment impact on signal
        if sentiment_data['overall'] == 'bullish':
            sentiment_impact_score = sentiment_data['confidence'] * 10
        elif sentiment_data['overall'] == 'bearish':
            sentiment_impact_score = -sentiment_data['confidence'] * 10
        else:
            sentiment_impact_score = 0

# --- Macro Economic Data ---
real_yields_data = get_real_yields()
fed_watch_data = get_fed_watch_data()
economic_events = get_economic_calendar()

# --- Backtesting Data ---
backtest_data = None
walkforward_data = None
if enable_backtest:
    period_map_backtest = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo"
    }
    
    with st.spinner(tr("Running backtest...", "در حال اجرای بکتست...")):
        backtest_df, _ = get_data(asset_name, period_map_backtest.get(backtest_period, "3mo"), timeframe)
        if not backtest_df.empty:
            backtest_data = run_backtest(backtest_df, timeframe)
            walkforward_data = run_walkforward_backtest(backtest_df, timeframe, splits=3, train_ratio=0.7)

# --- Advanced Correlation Data ---
correlation_data = calculate_advanced_correlations(asset_name)

if not df.empty:
    df = calculate_patterns(df)

    close = df["Close"].squeeze()
    candle_close_price = float(close.iloc[-1])
    # Keep price source on Yahoo symbols with stable live data.
    price_symbol = asset_name
    quote_candidates = [price_symbol]
    quote_price, quote_delta, quote_source = get_fresh_quote(quote_candidates)
    if quote_price is not None:
        curr_price = float(quote_price)
        price_delta_live = float(quote_delta)
    else:
        live_price = get_live_price(price_symbol)
        if live_price is None and price_symbol != asset_name:
            live_price = get_live_price(asset_name)
        curr_price = live_price if live_price is not None else candle_close_price
        price_delta_live = curr_price - candle_close_price

    # Inject latest quote into active candle so indicators/signals run on live price.
    if len(df.index) > 0 and all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        last_idx = df.index[-1]
        last_high = float(df.at[last_idx, "High"])
        last_low = float(df.at[last_idx, "Low"])
        df.at[last_idx, "Close"] = curr_price
        df.at[last_idx, "High"] = max(last_high, curr_price)
        df.at[last_idx, "Low"] = min(last_low, curr_price)
        df = calculate_patterns(df)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    rsi = RSIIndicator(close).rsi()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    macd_line = MACD(close).macd()
    macd_signal = MACD(close).macd_signal()
    macd_hist = MACD(close).macd_diff()
    adx = ADXIndicator(high, low, close).adx()
    atr = AverageTrueRange(high, low, close).average_true_range()
    bb = BollingerBands(close)
    obv = calc_obv(close, volume)

    curr_rsi = float(rsi.iloc[-1])
    curr_atr = float(atr.iloc[-1])
    curr_adx = safe_last(adx)
    curr_macd_hist = safe_last(macd_hist)

    ema50_slope = safe_last(ema50.diff(5))
    obv_slope = safe_last(obv.diff(5))

    # Higher timeframe confirmation
    df_higher, higher_tf = get_higher_timeframe(asset_name, timeframe)
    ht_trend, _ht_slope = get_higher_tf_trend(df_higher)

    # Correlation with DXY
    correlation = 0.0
    curr_dxy = 0.0
    if not dxy.empty and "Close" in dxy.columns:
        curr_dxy = float(dxy["Close"].iloc[-1])
        common_idx = df.index.intersection(dxy.index)
        if len(common_idx) > 3:
            correlation = float(df.loc[common_idx]["Close"].corr(dxy.loc[common_idx]["Close"]))

    # --- Enhanced Multi-factor Signal Engine ---
    long_pts = 0.0
    short_pts = 0.0
    bullish_reasons = []
    bearish_reasons = []

    trend_bias = "UP" if curr_price > safe_last(ema200) else "DOWN"
    market_structure = get_market_structure(high, low, close)
    
    # Calculate additional indicators for better accuracy
    bb_upper = BollingerBands(close).bollinger_hband()
    bb_lower = BollingerBands(close).bollinger_lband()
    bb_middle = BollingerBands(close).bollinger_mavg()
    bb_width = (safe_last(bb_upper) - safe_last(bb_lower)) / safe_last(bb_middle)
    price_bb_position = (curr_price - safe_last(bb_lower)) / (safe_last(bb_upper) - safe_last(bb_lower))
    
    # Enhanced trend analysis with multiple timeframe confirmation
    ema20_slope = safe_last(ema20.diff(5))
    ema200_slope = safe_last(ema200.diff(10))
    
    # 1) Enhanced Trend structure with dynamic weighting
    ema200_weight = 30  # Increased weight for primary trend
    if curr_price > safe_last(ema200):
        long_pts += ema200_weight
        bullish_reasons.append(tr("Price above EMA200 (primary trend filter)", "Price above EMA200 (primary trend filter)"))
        
        # Additional confirmation for strong trend
        if ema200_slope > 0:
            long_pts += 8
            bullish_reasons.append(tr("EMA200 trending up", "EMA200 در روند صعودی"))
    else:
        short_pts += ema200_weight
        bearish_reasons.append(tr("Price below EMA200 (primary trend filter)", "Price below EMA200 (primary trend filter)"))
        
        if ema200_slope < 0:
            short_pts += 8
            bearish_reasons.append(tr("EMA200 trending down", "EMA200 در روند نزولی"))

    # Multi-EMA alignment with stronger confirmation
    if safe_last(ema20) > safe_last(ema50) > safe_last(ema200):
        alignment_strength = min(3, (safe_last(ema20) - safe_last(ema200)) / safe_last(ema200) * 100)
        long_pts += 12 + alignment_strength * 2
        bullish_reasons.append(tr(f"EMA alignment confirmed (strength: {alignment_strength:.1f})", f"همسویی EMA تأیید شد (قدرت: {alignment_strength:.1f})"))
    elif safe_last(ema20) < safe_last(ema50) < safe_last(ema200):
        alignment_strength = min(3, (safe_last(ema200) - safe_last(ema20)) / safe_last(ema200) * 100)
        short_pts += 12 + alignment_strength * 2
        bearish_reasons.append(tr(f"EMA alignment confirmed (strength: {alignment_strength:.1f})", f"همسویی EMA تأیید شد (قدرت: {alignment_strength:.1f})"))

    # Enhanced slope analysis
    if ema50_slope > 0.001:  # More precise threshold
        long_pts += 10
        bullish_reasons.append(tr("EMA50 positive slope", "شیب مثبت EMA50"))
    elif ema50_slope < -0.001:
        short_pts += 10
        bearish_reasons.append(tr("EMA50 negative slope", "شیب منفی EMA50"))

    # 2) Enhanced Momentum with RSI divergence detection
    rsi_oversold = curr_rsi <= 30
    rsi_overbought = curr_rsi >= 70
    
    if 45 <= curr_rsi <= 55:  # Neutral zone - reduced points
        long_pts += 2
        short_pts += 2
    elif 52 <= curr_rsi <= 68:  # Optimistic bullish zone
        long_pts += 6
        bullish_reasons.append(tr(f"RSI in bullish zone ({curr_rsi:.1f})", f"RSI در منطقه صعودی ({curr_rsi:.1f})"))
    elif 32 <= curr_rsi <= 48:  # Bearish zone
        short_pts += 6
        bearish_reasons.append(tr(f"RSI in bearish zone ({curr_rsi:.1f})", f"RSI در منطقه نزولی ({curr_rsi:.1f})"))
    elif rsi_oversold and trend_bias == "UP":
        long_pts += 8  # Oversold in uptrend
        bullish_reasons.append(tr("RSI oversold in uptrend", "RSI اشباع فروش در روند صعودی"))
    elif rsi_overbought and trend_bias == "DOWN":
        short_pts += 8  # Overbought in downtrend
        bearish_reasons.append(tr("RSI overbought in downtrend", "RSI اشباع خرید در روند نزولی"))

    # Enhanced MACD analysis
    macd_indicator = MACD(close)
    macd_signal_series = macd_indicator.macd_signal()
    macd_histogram_series = macd_indicator.macd_diff()  # Use macd_diff() instead of macd_histogram()
    macd_line_series = macd_indicator.macd()
    
    # Get last values for comparison
    macd_signal = safe_last(macd_signal_series)
    macd_histogram = safe_last(macd_histogram_series)
    macd_line = safe_last(macd_line_series)
    
    if macd_histogram > 0 and macd_line > macd_signal:
        long_pts += 9
        bullish_reasons.append(tr("MACD bullish crossover", "کراس‌اور صعودی MACD"))
    elif macd_histogram < 0 and macd_line < macd_signal:
        short_pts += 9
        bearish_reasons.append(tr("MACD bearish crossover", "کراس‌اور نزولی MACD"))
    elif macd_histogram > 0:
        long_pts += 4
    elif macd_histogram < 0:
        short_pts += 4

    # 3) Enhanced Market structure with Bollinger Bands
    if market_structure["signal"] == "BUY":
        structure_bonus = 16
        # Additional confirmation if price is near lower BB
        if price_bb_position <= 0.2:
            structure_bonus += 4
            bullish_reasons.append(tr("Price near lower Bollinger Band", "قیمت نزدیک به باند پایینی بولینگر"))
        
        long_pts += structure_bonus
        bullish_reasons.append(
            tr(
                f"Structure: {market_structure['reason']} | S={market_structure['support']:.2f} R={market_structure['resistance']:.2f}",
                f"Structure: {market_structure['reason']} | S={market_structure['support']:.2f} R={market_structure['resistance']:.2f}",
            )
        )
    elif market_structure["signal"] == "SELL":
        structure_bonus = 16
        # Additional confirmation if price is near upper BB
        if price_bb_position >= 0.8:
            structure_bonus += 4
            bearish_reasons.append(tr("Price near upper Bollinger Band", "قیمت نزدیک به باند بالایی بولینگر"))
        
        short_pts += structure_bonus
        bearish_reasons.append(
            tr(
                f"Structure: {market_structure['reason']} | S={market_structure['support']:.2f} R={market_structure['resistance']:.2f}",
                f"Structure: {market_structure['reason']} | S={market_structure['support']:.2f} R={market_structure['resistance']:.2f}",
            )
        )

    # 4) Enhanced Flow / participation with volume analysis
    vol_sma = safe_last(volume.rolling(20).mean(), default=0.0)
    vol_last = safe_last(volume, default=0.0)
    rel_vol = vol_last / max(vol_sma, 1e-9) if vol_sma > 0 else 1.0
    
    # Volume spike detection
    volume_spike = rel_vol >= 1.5
    
    if volume_spike:
        if obv_slope > 0:
            long_pts += 12
            bullish_reasons.append(tr("Volume spike with positive flow", "جهش حجم با جریان مثبت"))
        elif obv_slope < 0:
            short_pts += 12
            bearish_reasons.append(tr("Volume spike with negative flow", "جهش حجم با جریان منفی"))
    elif rel_vol >= 1.15:
        if obv_slope > 0:
            long_pts += 8
        elif obv_slope < 0:
            short_pts += 8
    else:
        if obv_slope > 0:
            long_pts += 4
        elif obv_slope < 0:
            short_pts += 4

    # 5) Intermarket dependencies (gold drivers)
    dxy_ret = pct_change_n(dxy["Close"].squeeze(), 5) if not dxy.empty and "Close" in dxy.columns else 0.0
    us10y_df = market_ctx.get("us10y", pd.DataFrame())
    us10y_ret = pct_change_n(us10y_df["Close"].squeeze(), 5) if not us10y_df.empty and "Close" in us10y_df.columns else 0.0
    silver_df = market_ctx.get("silver", pd.DataFrame())
    silver_ret = pct_change_n(silver_df["Close"].squeeze(), 5) if not silver_df.empty and "Close" in silver_df.columns else 0.0
    copper_df = market_ctx.get("copper", pd.DataFrame())
    copper_ret = pct_change_n(copper_df["Close"].squeeze(), 5) if not copper_df.empty and "Close" in copper_df.columns else 0.0

    if dxy_ret < -0.08:
        long_pts += 14
        bullish_reasons.append(tr(f"DXY weakening ({dxy_ret:.2f}% / 5 bars)", f"DXY weakening ({dxy_ret:.2f}% / 5 bars)"))
    elif dxy_ret > 0.08:
        short_pts += 14
        bearish_reasons.append(tr(f"DXY strengthening ({dxy_ret:.2f}% / 5 bars)", f"DXY strengthening ({dxy_ret:.2f}% / 5 bars)"))

    if us10y_ret < -0.08:
        long_pts += 11
        bullish_reasons.append(tr(f"US10Y yield falling ({us10y_ret:.2f}% / 5 bars)", f"US10Y yield falling ({us10y_ret:.2f}% / 5 bars)"))
    elif us10y_ret > 0.08:
        short_pts += 11
        bearish_reasons.append(tr(f"US10Y yield rising ({us10y_ret:.2f}% / 5 bars)", f"US10Y yield rising ({us10y_ret:.2f}% / 5 bars)"))

    if silver_ret > 0:
        long_pts += 5
    elif silver_ret < 0:
        short_pts += 5

    if copper_ret > 0:
        long_pts += 1
    elif copper_ret < 0:
        short_pts += 1

    # 6) Higher timeframe confirmation
    if ht_trend == "UP":
        long_pts += 15
        bullish_reasons.append(tr(f"Higher TF ({higher_tf}) trend is up", f"Higher TF ({higher_tf}) trend is up"))
    elif ht_trend == "DOWN":
        short_pts += 15
        bearish_reasons.append(tr(f"Higher TF ({higher_tf}) trend is down", f"Higher TF ({higher_tf}) trend is down"))

    # 7) Sentiment Analysis (if enabled) - SEPARATE FROM MAIN SIGNAL
    # Note: Sentiment analysis now displayed separately and does NOT affect main signal
    sentiment_signal = "NEUTRAL"
    sentiment_reason = tr("No sentiment data", "داده‌ای سنتیمنت موجود نیست")
    sentiment_confidence = 0.0
    
    if sentiment_data:
        if sentiment_data['overall'] == 'bullish':
            sentiment_signal = "BUY"
            sentiment_reason = tr(f"Market sentiment bullish ({sentiment_data['confidence']:.1f})", f"سنتیمنت بازار صعودی ({sentiment_data['confidence']:.1f})")
            sentiment_confidence = sentiment_data['confidence']
        elif sentiment_data['overall'] == 'bearish':
            sentiment_signal = "SELL"
            sentiment_reason = tr(f"Market sentiment bearish ({sentiment_data['confidence']:.1f})", f"سنتیمنت بازار نزولی ({sentiment_data['confidence']:.1f})")
            sentiment_confidence = sentiment_data['confidence']

    # --- Per-method signal box (technical + fundamental) ---
    prev_high_20 = safe_last(high.shift(1).rolling(20).max(), default=curr_price)
    prev_low_20 = safe_last(low.shift(1).rolling(20).min(), default=curr_price)

    # Enhanced Price Action Analysis with better filtering
    pa_sig = "NEUTRAL"
    pa_reason = tr("[Source: price candles] Range/no clear breakout", "[Source: price candles] Range/no clear breakout")
    
    # Get volume confirmation from market structure
    volume_confirmation = market_structure.get("volume_confirmation", False)
    market_volatility = market_structure.get("market_volatility", 0.0)
    
    if curr_price > prev_high_20 and curr_price > safe_last(ema50):
        if volume_confirmation:
            pa_sig = "BUY"
            pa_reason = tr("[Source: price candles] Volume-confirmed breakout above 20-bar high", "[Source: price candles] جهش حجم تأیید شده بالاترین بالای 20 روزه")
        else:
            pa_sig = "BUY"
            pa_reason = tr("[Source: price candles] Breakout above previous 20-bar high", "[Source: price candles] شکستن بالاترین بالای 20 روزه")
    elif curr_price < prev_low_20 and curr_price < safe_last(ema50):
        if volume_confirmation:
            pa_sig = "SELL"
            pa_reason = tr("[Source: price candles] Volume-confirmed breakdown below 20-bar low", "[Source: price candles] جهش حجم تأیید شده پایین‌ترین پایین 20 روزه")
        else:
            pa_sig = "SELL"
            pa_reason = tr("[Source: price candles] Breakdown below previous 20-bar low", "[Source: price candles] شکستن پایین‌ترین پایین 20 روزه")
    elif curr_price > safe_last(ema20) and safe_last(ema20) > safe_last(ema50):
        # Trend continuation with momentum check
        if ema20_slope > 0.001 and volume_confirmation:
            pa_sig = "BUY"
            pa_reason = tr("[Source: price candles] Bullish trend continuation with volume", "[Source: price candles] ادامه روند صعودی با حجم")
        else:
            pa_sig = "BUY"
            pa_reason = tr("[Source: price candles] Trend continuation above EMA20/EMA50", "[Source: price candles] ادامه روند بالاترین EMA20/EMA50")
    elif curr_price < safe_last(ema20) and safe_last(ema20) < safe_last(ema50):
        # Trend continuation with momentum check
        if ema20_slope < -0.001 and volume_confirmation:
            pa_sig = "SELL"
            pa_reason = tr("[Source: price candles] Bearish trend continuation with volume", "[Source: price candles] ادامه روند نزولی با حجم")
        else:
            pa_sig = "SELL"
            pa_reason = tr("[Source: price candles] Trend continuation below EMA20/EMA50", "[Source: price candles] ادامه روند پایین‌تر EMA20/EMA50")
    
    # Additional filter: avoid weak signals in choppy markets
    if market_volatility > 2.5:
        if pa_sig in ["BUY", "SELL"]:
            pa_sig = "NEUTRAL"
            pa_reason = tr("[Source: price candles] Signal filtered: choppy market", "[Source: price candles] سیگنال فیلتر شد: بازار ناپایدار")

    macd_sig = "NEUTRAL"
    macd_reason = tr("[Source: MACD] Flat momentum", "[Source: MACD] Flat momentum")
    if len(macd_hist.dropna()) > 2 and len(macd_line.dropna()) > 2 and len(macd_signal.dropna()) > 2:
        # Get last non-NaN values
        last_hist = macd_hist.dropna().iloc[-1]
        last_hist_prev = macd_hist.dropna().iloc[-2] if len(macd_hist.dropna()) > 1 else 0
        last_line = macd_line.dropna().iloc[-1]
        last_signal = macd_signal.dropna().iloc[-1]
        
        if last_hist > 0 and last_hist_prev <= 0:
            macd_sig = "BUY"
            macd_reason = tr("[Source: MACD] Histogram crossed above zero", "[Source: MACD] Histogram crossed above zero")
        elif last_hist < 0 and last_hist_prev >= 0:
            macd_sig = "SELL"
            macd_reason = tr("[Source: MACD] Histogram crossed below zero", "[Source: MACD] Histogram crossed below zero")
        elif not pd.isna(last_hist) and not pd.isna(last_line) and not pd.isna(last_signal):
            if last_hist > 0 and last_line > last_signal:
                macd_sig = "BUY"
                macd_reason = tr("[Source: MACD] Positive histogram with bullish line structure", "[Source: MACD] Positive histogram with bullish line structure")
            elif last_hist < 0 and last_line < last_signal:
                macd_sig = "SELL"
                macd_reason = tr("[Source: MACD] Negative histogram with bearish line structure", "[Source: MACD] Negative histogram with bearish line structure")

    bb_mid = safe_last(bb.bollinger_mavg(), default=curr_price)
    structure_sig = str(market_structure.get("signal", "NEUTRAL"))
    structure_reason = tr(
        f"[Source: Structure] {market_structure.get('reason', 'range')} | S={market_structure.get('support', 0.0):.2f} | R={market_structure.get('resistance', 0.0):.2f}",
        f"[Source: Structure] {market_structure.get('reason', 'range')} | S={market_structure.get('support', 0.0):.2f} | R={market_structure.get('resistance', 0.0):.2f}",
    )

    fund_score = 0
    if dxy_ret < -0.08:
        fund_score += 2
    elif dxy_ret > 0.08:
        fund_score -= 2
    if us10y_ret < -0.08:
        fund_score += 2
    elif us10y_ret > 0.08:
        fund_score -= 2
    if silver_ret > 0:
        fund_score += 1
    elif silver_ret < 0:
        fund_score -= 1
    if copper_ret > 0:
        fund_score += 1
    elif copper_ret < 0:
        fund_score -= 1

    fund_sig = "NEUTRAL"
    if fund_score >= 2:
        fund_sig = "BUY"
    elif fund_score <= -2:
        fund_sig = "SELL"
    fund_reason = tr(
        f"[Source: DXY/US10Y/Silver/Copper] score={fund_score} | DXY:{dxy_ret:.2f}% | 10Y:{us10y_ret:.2f}% | Silver:{silver_ret:.2f}% | Copper:{copper_ret:.2f}%",
        f"[Source: DXY/US10Y/Silver/Copper] score={fund_score} | DXY:{dxy_ret:.2f}% | 10Y:{us10y_ret:.2f}% | Silver:{silver_ret:.2f}% | Copper:{copper_ret:.2f}%",
    )

    trend_sig = "NEUTRAL"
    trend_reason = tr("Main trend is range/unclear", "Main trend is range/unclear")
    trend_ema50 = safe_last(ema50)
    trend_ema200 = safe_last(ema200)
    if trend_ema50 > trend_ema200 and ema50_slope > 0:
        trend_sig = "BUY"
        trend_reason = tr("Main trend up (EMA50 > EMA200 + positive slope)", "Main trend up (EMA50 > EMA200 + positive slope)")
    elif trend_ema50 < trend_ema200 and ema50_slope < 0:
        trend_sig = "SELL"
        trend_reason = tr("Main trend down (EMA50 < EMA200 + negative slope)", "Main trend down (EMA50 < EMA200 + negative slope)")

    # Core technical methods (fundamental is displayed separately)
    method_signals = [
        ("price_action", T["method_price_action"], pa_sig, pa_reason),
        ("macd", T["method_macd"], macd_sig, macd_reason),
        ("market_structure", tr("Market Structure", "Market Structure"), structure_sig, structure_reason),
        ("trend_follow", tr("Trend Tracking", "Trend Tracking"), trend_sig, trend_reason),
    ]
    
    # Add strength from market structure to method signals
    method_signals_with_strength = []
    for method_code, method_name, method_sig, method_reason in method_signals:
        strength_bonus = 0.0
        if method_code == "market_structure" and market_structure.get("strength", 0.0) > 0:
            strength_bonus = market_structure["strength"] * 0.5  # Add strength bonus
        method_signals_with_strength.append((method_code, method_name, method_sig, method_reason))
    
    ai_source = tr("Free Local", "Free Local")
    ai_mode_label = tr("Free Local", "Free Local")
    ai_sig, ai_reason, ai_conf = ai_confirmation_signal(close)
    has_external_ai = bool(ai_api_key.strip()) and ai_provider == tr("External API", "External API")
    if enable_ai_confirmation and has_external_ai:
        ext_sig, ext_reason, ext_conf = ai_confirmation_external(
            api_key=ai_api_key.strip(),
            model=ai_model.strip() or "gpt-4o-mini",
            close=close,
            curr_rsi=curr_rsi,
            curr_macd_hist=curr_macd_hist,
            curr_adx=curr_adx,
        )
        if ext_conf > 0:
            ai_sig, ai_reason, ai_conf = ext_sig, ext_reason, ext_conf
            ai_source = tr("External", "External")
            ai_mode_label = tr("External", "External")
        else:
            ai_reason = tr(
                "External AI unavailable; AI branch stayed neutral",
                "External AI unavailable; local confirmation used",
            )
    method_signals.append(("ai_branch", tr("AI Branch", "AI Branch"), ai_sig, f"[{ai_source}] {ai_reason}"))

    # Enhanced Core signal calculation with improved thresholds
    method_weights = {
        "price_action": 2.5,      # Highest weight for price action
        "macd": 1.5,            # Increased for MACD importance
        "market_structure": 3.0,    # Highest weight for structure
        "trend_follow": 3.5,       # Highest weight for trend
    }
    core_methods = method_signals_with_strength
    method_count = len(core_methods)
    buy_votes = sum(1 for _, _, s, _ in core_methods if s == "BUY")
    sell_votes = sum(1 for _, _, s, _ in core_methods if s == "SELL")
    
    # Calculate weights with strength bonuses
    buy_weight = sum(method_weights.get(code, 1.0) for code, _, s, _ in core_methods if s == "BUY")
    sell_weight = sum(method_weights.get(code, 1.0) for code, _, s, _ in core_methods if s == "SELL")
    
    # Add strength bonuses from market structure
    for method_code, _, method_sig, _ in core_methods:
        if method_code == "market_structure" and market_structure.get("strength", 0.0) > 0:
            strength_bonus = market_structure["strength"] * 0.5  # Add strength bonus
            if method_sig == "BUY":
                buy_weight += strength_bonus
            elif method_sig == "SELL":
                sell_weight += strength_bonus
    
    total_core_weight = sum(method_weights.get(code, 1.0) for code, _, _, _ in core_methods)
    weight_edge = buy_weight - sell_weight
    agreement_pct = (max(buy_weight, sell_weight) / max(total_core_weight, 1e-9)) * 100.0
    
    # Calculate signal strength based on multiple factors
    signal_strength = abs(weight_edge) / total_core_weight
    consensus_strength = max(buy_votes, sell_votes) / method_count
    
    # Enhanced signal determination with stricter thresholds
    signal = "NEUTRAL"
    
    # Dynamic thresholds based on market conditions
    market_volatility = curr_atr / curr_price * 100  # ATR as percentage
    volatility_multiplier = 1.2 if market_volatility > 2.0 else 1.0  # Higher threshold in volatile markets
    
    # Very Strong signals require both weight and consensus
    if signal_strength >= (0.80 * volatility_multiplier) and consensus_strength >= 0.75:
        if buy_weight > sell_weight:
            signal = "VERY STRONG BUY"
        else:
            signal = "VERY STRONG SELL"
    
    # Strong signals with dynamic thresholds
    elif signal_strength >= (0.60 * volatility_multiplier) and consensus_strength >= 0.6:
        if buy_weight > sell_weight and sell_weight <= total_core_weight * 0.2:
            signal = "STRONG BUY"
        elif sell_weight > buy_weight and buy_weight <= total_core_weight * 0.2:
            signal = "STRONG SELL"
    
    # Regular signals with better filtering
    elif signal_strength >= (0.40 * volatility_multiplier) and consensus_strength >= 0.5:
        if buy_weight > sell_weight:
            signal = "BUY"
        elif sell_weight > buy_weight:
            signal = "SELL"
    
    # Additional confirmation using bias score with dynamic threshold
    bias_score = long_pts - short_pts
    bias_threshold = 20 * volatility_multiplier  # Dynamic bias threshold
    
    # Apply bias filter for weak signals
    if signal in ["BUY", "SELL"] and abs(bias_score) < bias_threshold:
        signal = "NEUTRAL"
        bullish_reasons.append(tr("Signal filtered: insufficient bias", "سیگنال فیلتر شد: بایاس ناکافی"))
        bearish_reasons.append(tr("Signal filtered: insufficient bias", "سیگنال فیلتر شد: بایاس ناکافی"))
    
    # Final confirmation check with stricter requirements
    if signal != "NEUTRAL":
        # Ensure minimum agreement from core methods
        min_agreement = 0.65  # 65% minimum agreement
        if consensus_strength < min_agreement:
            signal = "NEUTRAL"
            bullish_reasons.append(tr("Signal filtered: low consensus", "سیگنال فیلتر شد: اجماع پایین"))
            bearish_reasons.append(tr("Signal filtered: low consensus", "سیگنال فیلتر شد: اجماع پایین"))
        
        # Additional filter: ensure no conflicting signals
        conflicting_signals = abs(buy_weight - sell_weight) < (total_core_weight * 0.3)
        if conflicting_signals:
            signal = "NEUTRAL"
            bullish_reasons.append(tr("Signal filtered: conflicting indicators", "سیگنال فیلتر شد: اندیکاتورهای متضاد"))
            bearish_reasons.append(tr("Signal filtered: conflicting indicators", "سیگنال فیلتر شد: اندیکاتورهای متضاد"))

    # Fallback to directional trend when votes are low but one-sided.
    if signal == "NEUTRAL":
        if trend_sig == "BUY" and buy_weight > sell_weight and sell_votes == 0:
            signal = "BUY"
        elif trend_sig == "SELL" and sell_weight > buy_weight and buy_votes == 0:
            signal = "SELL"

    # Hard bias rule: no buy below EMA200 and no sell above EMA200 when trend is clear.
    if curr_price < safe_last(ema200) and "BUY" in signal:
        signal = "NEUTRAL" if ht_trend in {"DOWN", "RANGE", None} else "BUY"
        bearish_reasons.append(tr("Hard filter: buy blocked below EMA200", "Hard filter: buy blocked below EMA200"))
    if curr_price > safe_last(ema200) and "SELL" in signal and ht_trend == "UP":
        signal = "NEUTRAL"
        bullish_reasons.append(tr("Hard filter: sell blocked against EMA200 uptrend", "Hard filter: sell blocked against EMA200 uptrend"))

    # AI branch is informational only (no impact on main signal).

    gate_soft_fails = 0

    # Stronger higher-timeframe alignment gate.
    if enforce_mtf_alignment and signal != "NEUTRAL" and ht_trend in {"UP", "DOWN"}:
        if ("BUY" in signal and ht_trend == "DOWN") or ("SELL" in signal and ht_trend == "UP"):
            gate_soft_fails += 1
            signal = "NEUTRAL"
            bearish_reasons.append(tr("Higher-TF conflict: signal blocked", "Higher-TF conflict: signal blocked"))

    # Softer high-impact news gate.
    if enable_news_filter and signal != "NEUTRAL":
        news_block, news_tag = in_high_impact_news_window(economic_events, window_minutes=news_filter_minutes)
        if news_block:
            gate_soft_fails += 1
            bearish_reasons.append(tr(f"News lock active (soft): {news_tag}", f"News lock active (soft): {news_tag}"))

    # Softer quality gate.
    atr_pct = (curr_atr / max(curr_price, 1e-9)) * 100.0
    if enable_quality_gate and signal != "NEUTRAL":
        quality_ok = (curr_adx >= float(min_adx_gate)) and (agreement_pct >= float(min_agreement_gate)) and (atr_pct >= 0.03)
        if not quality_ok:
            gate_soft_fails += 1
            bearish_reasons.append(tr("Quality gate soft-fail", "Quality gate soft-fail"))

    # Neutralize if two risk gates fail together.
    if signal != "NEUTRAL" and gate_soft_fails >= 2:
        signal = "NEUTRAL"
        bearish_reasons.append(tr("Risk gates blocked signal", "Risk gates blocked signal"))

    # Confidence / bias from vote-consensus logic.
    atr_baseline_core = safe_last(atr.rolling(50).mean(), default=curr_atr)
    regime_data = detect_market_regime(curr_adx, curr_atr, atr_baseline_core)
    max_votes = max(buy_votes, sell_votes)
    unanimity_bonus = 10.0 if max_votes == method_count else 4.0 if max_votes == max(method_count - 1, 1) else 0.0
    ai_bonus = 0.0
    confidence = min(99.0, max(1.0, agreement_pct + unanimity_bonus + regime_data["conf_bonus"] + ai_bonus))

    method_net = float(weight_edge)
    method_total_weight = float(total_core_weight)
    bias_score = float(np.clip((weight_edge / max(total_core_weight, 1e-9)) * 100.0 + (fund_score * 3.0), -100.0, 100.0))

    signal_prob = compute_signal_probability(
        bias_score=bias_score,
        confidence=confidence,
        method_net=method_net,
        method_total_weight=method_total_weight,
        regime_name=regime_data["name"],
        backtest_win_rate=(backtest_data.get("win_rate") if backtest_data else None),
        backtest_trades=int(backtest_data.get("total_trades", 0)) if backtest_data else 0,
        oos_win_rate=(walkforward_data.get("oos_win_rate") if walkforward_data else None),
        oos_folds=int(walkforward_data.get("folds", 0)) if walkforward_data else 0,
    )

    # Risk-guard layer based on backtest health
    risk_guard_active = False
    risk_guard_reason = ""
    if backtest_data:
        recent_trades = backtest_data.get("trades", [])
        loss_streak = 0
        for trd in reversed(recent_trades):
            pnl = trd.get("pnl")
            if pnl is None:
                continue
            if pnl < 0:
                loss_streak += 1
            else:
                break
        if loss_streak >= max_loss_streak_guard:
            risk_guard_active = True
            risk_guard_reason = tr(
                f"Risk guard: loss streak {loss_streak}",
                f"محافظ ریسک: زنجیره باخت {loss_streak}"
            )
        if backtest_data.get("max_drawdown", 0.0) >= max_dd_guard:
            risk_guard_active = True
            risk_guard_reason = tr(
                f"Risk guard: drawdown {backtest_data.get('max_drawdown', 0.0):.1f}%",
                f"محافظ ریسک: افت سرمایه {backtest_data.get('max_drawdown', 0.0):.1f}%"
            )
    if risk_guard_active:
        signal = "NEUTRAL"
        confidence = min(confidence, 35.0)
        bearish_reasons.append(risk_guard_reason)
        signal_prob["win_prob"] = min(signal_prob["win_prob"], 50.0)
        signal_prob["loss_prob"] = 100.0 - signal_prob["win_prob"]

    # --- Risk Management ---
    is_long = signal in ["BUY", "STRONG BUY", "VERY STRONG BUY"]
    has_trade_signal = signal in ["BUY", "STRONG BUY", "VERY STRONG BUY", "SELL", "STRONG SELL", "VERY STRONG SELL"]
    atr_baseline_rm = safe_last(atr.rolling(50).mean(), default=curr_atr)
    atr_ratio = curr_atr / max(atr_baseline_rm, 1e-9)
    dynamic_atr_mult = float(np.clip(atr_mult * (0.85 + 0.5 * atr_ratio), atr_mult * 0.8, atr_mult * 1.9))
    if has_trade_signal:
        sl = curr_price - (dynamic_atr_mult * curr_atr) if is_long else curr_price + (dynamic_atr_mult * curr_atr)
        tp = curr_price + (rr_ratio * dynamic_atr_mult * curr_atr) if is_long else curr_price - (rr_ratio * dynamic_atr_mult * curr_atr)
        tp2 = curr_price + ((rr_ratio + 1.0) * dynamic_atr_mult * curr_atr) if is_long else curr_price - ((rr_ratio + 1.0) * dynamic_atr_mult * curr_atr)
        entry_low = curr_price - (0.25 * curr_atr)
        entry_high = curr_price + (0.25 * curr_atr)
    else:
        sl = 0.0
        tp = 0.0
        tp2 = 0.0
        entry_low = 0.0
        entry_high = 0.0

    risk_amt = acc_balance * (risk_pct / 100.0)
    risk_per_unit = abs(curr_price - sl) * contract_size if has_trade_signal else 0.0
    lot_size = (risk_amt / risk_per_unit) if risk_per_unit != 0 else 0

    # --- Smart Position Sizing ---
    smart_sizing_data = None
    if enable_smart_sizing:
        atr_baseline = safe_last(atr.rolling(50).mean(), default=curr_atr)
        smart_sizing_data = calculate_smart_position_size(
            base_lot_size=lot_size,
            confidence=confidence,
            atr=curr_atr,
            atr_baseline=atr_baseline,
            sentiment_data=sentiment_data,
            risk_multiplier=risk_multiplier
        )
        lot_size = smart_sizing_data['adjusted_lot_size']
        with st.sidebar.expander(tr("Smart Sizing Output", "خروجی پوزیشن‌سایزینگ هوشمند"), expanded=False):
            st.write(f"{T['base_position']}: {smart_sizing_data['base_lot_size']:.3f}")
            st.write(f"{T['adjusted_position']}: {smart_sizing_data['adjusted_lot_size']:.3f}")
            st.write(f"{T['sizing_factor']}: {smart_sizing_data['sizing_factor']:.2f}x")
            st.write(f"{T['volatility_adjustment']}: {smart_sizing_data['volatility_factor']:.2f}x")
            if sentiment_data:
                st.write(f"{T['sentiment_adjustment']}: {smart_sizing_data['sentiment_factor']:.2f}x")
            st.write(f"{T['risk_multiplier']}: {smart_sizing_data['risk_multiplier']:.2f}x")

    strength_mult = 1.8 if signal.startswith("VERY STRONG") else 1.4 if signal.startswith("STRONG") else 1.0
    expected_move_pct = (curr_atr / max(curr_price, 1e-9)) * 100.0 * strength_mult
    expected_drawdown_pct = (abs(curr_price - sl) / max(curr_price, 1e-9)) * 100.0 if has_trade_signal else 0.0

    # --- UI Layout ---
    st.markdown(
        f"""
        <div class="site-header">
            <div class="brand">◉ Golden Terminal</div>
            <div class="nav">⌂ {tr("Overview", "نمای کلی")} &nbsp;|&nbsp; ☰ {tr("Left Menu", "منوی چپ")} &nbsp;|&nbsp; ⚙ {tr("Right Menu", "منوی راست")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title(T["title"])

    signal_display = signal
    price_text = f"${curr_price:,.3f}" if chart_mode == T["chart_tv"] else f"${curr_price:,.2f}"
    price_delta_text = f"{price_delta_live:+.3f}" if chart_mode == T["chart_tv"] else f"{price_delta_live:+.2f}"
    if signal == "VERY STRONG BUY":
        signal_display = tr("VERY STRONG BUY", "VERY STRONG BUY")
    elif signal == "VERY STRONG SELL":
        signal_display = tr("VERY STRONG SELL", "VERY STRONG SELL")
    elif signal == "STRONG BUY":
        signal_display = T["sig_strong_buy"]
    elif signal == "STRONG SELL":
        signal_display = T["sig_strong_sell"]
    elif signal == "BUY":
        signal_display = T["sig_buy"]
    elif signal == "SELL":
        signal_display = T["sig_sell"]
    else:
        signal_display = T["sig_neutral"]

    append_signal_journal(
        {
            "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "asset": asset_name,
            "timeframe": timeframe,
            "signal": signal,
            "bias_score": round(float(bias_score), 3),
            "confidence": round(float(confidence), 3),
            "price": round(float(curr_price), 4),
            "win_prob": round(float(signal_prob["win_prob"]), 3),
            "regime": regime_data["name"],
        }
    )

    badge_class = "badge-neutral"
    if signal in ["BUY", "STRONG BUY", "VERY STRONG BUY"]:
        badge_class = "badge-buy"
    elif signal in ["SELL", "STRONG SELL", "VERY STRONG SELL"]:
        badge_class = "badge-sell"

    ai_badge_class = "badge-ai-neutral"
    if ai_sig in ["BUY", "SELL"] and "External" in str(ai_mode_label):
        ai_badge_class = "badge-ai-external"

    st.markdown(
        f"""
        <div class="top-status">
            <span class="k">{tr("Price", "قیمت")}:</span><span class="v"><strong>${curr_price:,.2f}</strong></span>
            <span class="k">{tr("Signal", "سیگنال")}:</span><span class="v"><span class="{badge_class}">{signal_display}</span></span>
            <span class="k">{T["confidence"]}:</span><span class="v"><strong>{confidence:.1f}%</strong></span>
            <span class="k">{tr("Win Prob", "احتمال موفقیت")}:</span><span class="v"><strong>{signal_prob['win_prob']:.1f}%</strong></span>
            <span class="k">{tr("Regime", "رژیم")}:</span><span class="v"><strong>{regime_data['name']}</strong></span>
            <span class="k">{T["last_update"]}:</span><span class="v">{now_iran_str('%H:%M:%S')} IRT</span>
            <span class="k">{tr("Quote", "Quote")}:</span><span class="v">{quote_source or "n/a"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="app-card" style="margin-top:6px; margin-bottom:8px;">
            <strong>{tr("AI Confirmation", "AI Confirmation")}</strong>:
            <span class="{ai_badge_class}" style="margin-left:8px;">{ai_sig} ({ai_mode_label})</span>
            <div style="margin-top:6px; opacity:0.9;">{ai_reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if risk_guard_active:
        st.warning(risk_guard_reason)

    st.markdown(
        f"""
        <div class="hero-signal">
            <span class="h-label">{tr("Main Signal", "سیگنال اصلی")}:</span>
            <span class="h-value">{signal_display}</span>
            <span class="h-label">{T["confidence"]}:</span>
            <span class="h-value">{confidence:.1f}%</span>
            <span class="h-label">{T["bias_score"]}:</span>
            <span class="h-value">{bias_score:.1f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    signal_kpi_class = "kpi-neutral"
    if signal in ["BUY", "STRONG BUY", "VERY STRONG BUY"]:
        signal_kpi_class = "kpi-buy"
    elif signal in ["SELL", "STRONG SELL", "VERY STRONG SELL"]:
        signal_kpi_class = "kpi-sell"
    confidence_kpi_class = "kpi-buy" if confidence >= 70 else "kpi-neutral" if confidence >= 45 else "kpi-sell"
    bias_kpi_class = "kpi-buy" if bias_score > 10 else "kpi-sell" if bias_score < -10 else "kpi-neutral"
    st.markdown(
        f"""
        <div class="kpi-strip">
            <div class="kpi-tile {signal_kpi_class}">
                <div class="k">{tr("Signal", "سیگنال")}</div>
                <div class="v">{signal_display}</div>
            </div>
            <div class="kpi-tile {confidence_kpi_class}">
                <div class="k">{T["confidence"]}</div>
                <div class="v">{confidence:.0f}%</div>
            </div>
            <div class="kpi-tile {bias_kpi_class}">
                <div class="k">{T["bias_score"]}</div>
                <div class="v">{bias_score:.1f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        tr(
            f"Calibrated win prob: {signal_prob['win_prob']:.1f}% (raw {signal_prob['raw_win_prob']:.1f}%) | Empirical ref: {signal_prob['empirical_rate']:.1f}% | Calibration weight: {signal_prob['calibration_weight']:.0f}% | Data quality: {signal_prob['data_quality']} | Expected move: {expected_move_pct:.2f}% | Expected DD: {expected_drawdown_pct:.2f}%",
            f"احتمال موفقیت کالیبره: {signal_prob['win_prob']:.1f}% (خام {signal_prob['raw_win_prob']:.1f}%) | مرجع تجربی: {signal_prob['empirical_rate']:.1f}% | وزن کالیبراسیون: {signal_prob['calibration_weight']:.0f}% | کیفیت داده: {signal_prob['data_quality']} | حرکت موردانتظار: {expected_move_pct:.2f}% | افت موردانتظار: {expected_drawdown_pct:.2f}%"
        )
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(T["price"], price_text, price_delta_text)
    c2.metric(T["rsi"], f"{curr_rsi:.2f}")
    c3.metric(T["atr"], f"{curr_atr:.2f}")
    c4.metric(T["dxy"], f"{curr_dxy:.2f}")
    conf_tag = tr("Weak", "ضعیف")
    if confidence >= 75:
        conf_tag = tr("Strong", "قوی")
    elif confidence >= 50:
        conf_tag = tr("Medium", "متوسط")
    c5.metric(T["confidence"], f"{confidence:.1f}%", conf_tag)

    # Create columns for main signal and sentiment signal
    c6, c6_sentiment, c7, c8 = st.columns([1.2, 1.0, 1, 1])
    
    with c6:
        signal_class = "pulse-animation" if enable_animations else ""
        
        # Enhanced signal display with detailed information
        if signal in ["BUY", "STRONG BUY", "VERY STRONG BUY"]:
            if signal in ["STRONG BUY", "VERY STRONG BUY"] and enable_audio_alerts:
                play_alert_sound("strong_buy")
            
            # Create detailed signal info
            signal_emoji = "🚀" if signal == "VERY STRONG BUY" else "📈" if signal == "STRONG BUY" else "⬆️"
            signal_color = "#21c77a"
            
            signal_html = f"""
            <div class='signal-buy {signal_class}'>
                <div style='font-size: 2.5em; margin-bottom: 10px;'>{signal_emoji}</div>
                <h3 style='color: {signal_color}; margin: 5px 0;'>{signal_display}</h3>
                <p style='margin: 5px 0;'><strong>{T['bias_score']}:</strong> {bias_score:.1f}</p>
                <p style='margin: 5px 0;'><strong>Signal Strength:</strong> {signal_strength:.1%}</p>
                <p style='margin: 5px 0;'><strong>Consensus:</strong> {consensus_strength:.1%}</p>
                <p style='margin: 5px 0; font-size: 0.9em; opacity: 0.8;'>{len(bullish_reasons)} confirmations</p>
            </div>
            """
            st.markdown(signal_html, unsafe_allow_html=True)
            
        elif signal in ["SELL", "STRONG SELL", "VERY STRONG SELL"]:
            if signal in ["STRONG SELL", "VERY STRONG SELL"] and enable_audio_alerts:
                play_alert_sound("strong_sell")
            
            signal_emoji = "📉" if signal == "VERY STRONG SELL" else "⬇️" if signal == "STRONG SELL" else "🔻"
            signal_color = "#ff5a7a"
            
            signal_html = f"""
            <div class='signal-sell {signal_class}'>
                <div style='font-size: 2.5em; margin-bottom: 10px;'>{signal_emoji}</div>
                <h3 style='color: {signal_color}; margin: 5px 0;'>{signal_display}</h3>
                <p style='margin: 5px 0;'><strong>{T['bias_score']}:</strong> {bias_score:.1f}</p>
                <p style='margin: 5px 0;'><strong>Signal Strength:</strong> {signal_strength:.1%}</p>
                <p style='margin: 5px 0;'><strong>Consensus:</strong> {consensus_strength:.1%}</p>
                <p style='margin: 5px 0; font-size: 0.9em; opacity: 0.8;'>{len(bearish_reasons)} confirmations</p>
            </div>
            """
            st.markdown(signal_html, unsafe_allow_html=True)
            
        else:
            signal_html = f"""
            <div class='signal-neutral'>
                <div style='font-size: 2.5em; margin-bottom: 10px;'>⏸️</div>
                <h3 style='margin: 5px 0;'>{T['signal_wait']}</h3>
                <p style='margin: 5px 0;'><strong>{T['bias_score']}:</strong> {bias_score:.1f}</p>
                <p style='margin: 5px 0;'><strong>Signal Strength:</strong> {signal_strength:.1%}</p>
                <p style='margin: 5px 0; font-size: 0.9em; opacity: 0.8;'>Waiting for confirmation</p>
            </div>
            """
            st.markdown(signal_html, unsafe_allow_html=True)

    # Sentiment Signal Box (Separate from Main Signal)
    with c6_sentiment:
        if sentiment_data:
            sentiment_class = "pulse-animation" if enable_animations else ""
            
            if sentiment_signal == "BUY":
                sentiment_emoji = "📰"
                sentiment_color = "#21c77a"
                sentiment_bg = "rgba(33, 199, 122, 0.1)"
            elif sentiment_signal == "SELL":
                sentiment_emoji = "📰"
                sentiment_color = "#ff5a7a"
                sentiment_bg = "rgba(255, 90, 122, 0.1)"
            else:
                sentiment_emoji = "📰"
                sentiment_color = "#8a96ad"
                sentiment_bg = "rgba(138, 150, 173, 0.1)"
            
            sentiment_html = f"""
            <div class='app-card' style='background: {sentiment_bg}; backdrop-filter: blur(10px); border: 1px solid {sentiment_color}20;'>
                <div style='text-align: center; padding: 15px;'>
                    <div style='font-size: 1.8em; margin-bottom: 8px;'>{sentiment_emoji}</div>
                    <h4 style='color: {sentiment_color}; margin: 5px 0; font-size: 1.1em;'>{T['sentiment_analysis']}</h4>
                    <p style='margin: 8px 0; font-weight: 600; color: {sentiment_color};'>
                        {sentiment_signal}
                    </p>
                    <p style='margin: 5px 0; font-size: 0.9em; opacity: 0.9;'>{sentiment_reason}</p>
                    <p style='margin: 5px 0; font-size: 0.85em; opacity: 0.8;'>
                        Confidence: {sentiment_confidence:.1f}%
                    </p>
                    <p style='margin: 5px 0; font-size: 0.85em; opacity: 0.7;'>
                        Based on {sentiment_data.get('articles_analyzed', 0)} news articles
                    </p>
                </div>
            </div>
            """
            st.markdown(sentiment_html, unsafe_allow_html=True)
        else:
            neutral_html = f"""
            <div class='app-card' style='background: rgba(138, 150, 173, 0.1); backdrop-filter: blur(10px);'>
                <div style='text-align: center; padding: 15px;'>
                    <div style='font-size: 1.8em; margin-bottom: 8px;'>📰</div>
                    <h4 style='color: #8a96ad; margin: 5px 0; font-size: 1.1em;'>{T['sentiment_analysis']}</h4>
                    <p style='margin: 8px 0; font-weight: 600; color: #8a96ad;'>NEUTRAL</p>
                    <p style='margin: 5px 0; font-size: 0.9em; opacity: 0.8;'>Enable sentiment analysis in sidebar</p>
                </div>
            </div>
            """
            st.markdown(neutral_html, unsafe_allow_html=True)

    with c7:
        st.subheader(T["risk"])
        st.write(f"{T['risk_amount']}: ${risk_amt:.2f}")
        if smart_sizing_data:
            st.write(f"{T['base_position']}: {smart_sizing_data['base_lot_size']:.3f}")
            st.write(f"{T['adjusted_position']}: {smart_sizing_data['adjusted_lot_size']:.3f}")
            st.write(f"{T['sizing_factor']}: {smart_sizing_data['sizing_factor']:.2f}x")
        else:
            st.write(f"{T['lot_size']}: {lot_size:.3f}")
        st.write(f"{T['adx_macd']}: {curr_adx:.1f} / {curr_macd_hist:.3f}")

    with c8:
        st.subheader(T["targets"])
        st.write(f"{T['entry_zone']}: {entry_low:,.2f} - {entry_high:,.2f}")
        st.write(f"{T['tp']}: {tp:,.2f}")
        st.write(f"{T['tp2']}: {tp2:,.2f}")
        st.write(f"{T['sl']}: {sl:,.2f}")
        st.write(T["rr_fmt"].format(rr=rr_ratio))
        st.write(f"{T['corr_dxy']}: {correlation:.2f}")
        if not has_trade_signal:
            st.caption(tr("Signal is neutral: targets are set to zero.", "Signal is neutral: targets are set to zero."))

    st.markdown("<div class='right-menu'><strong>⚙ " + tr("Right Menu", "منوی راست") + "</strong></div>", unsafe_allow_html=True)
    rm1, rm2, rm3, rm4, rm5, rm6, rm7 = st.columns(7)
    show_sentiment_block = rm1.checkbox(tr("Sentiment", "سنتیمنت"), value=False, key="show_sentiment_block")
    show_fundamental_block = rm2.checkbox(tr("Fundamental", "فاندامنتال"), value=True, key="show_fundamental_block")
    show_macro_block = rm3.checkbox(tr("Macro", "ماکرو"), value=False, key="show_macro_block")
    show_backtest_block = rm4.checkbox(tr("Backtest", "بک‌تست"), value=False, key="show_backtest_block")
    show_corr_block = rm5.checkbox(tr("Correlation", "همبستگی"), value=False, key="show_corr_block")
    show_journal_block = rm6.checkbox(tr("Journal", "ژورنال"), value=False, key="show_journal_block")
    show_logic_block = rm7.checkbox(tr("Logic", "منطق"), value=False, key="show_logic_block")

    # --- Fundamental Analysis Display ---
    if show_fundamental_block:
        fund_color = "#21c77a" if fund_sig == 'BUY' else "#ff5a7a" if fund_sig == 'SELL' else "#8a96ad"
        fund_display = T["sig_buy"] if fund_sig == 'BUY' else T["sig_sell"] if fund_sig == 'SELL' else T["sig_neutral"]

        st.markdown(f"""
        <div class='app-card' style='border-left: 5px solid {fund_color};'>
            <h4 style='margin:0; color: var(--txt);'>{T['method_fundamental']}</h4>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 10px;'>
                <div>
                    <strong>{T['method_signal_col']}:</strong>
                    <span style='color: {fund_color}; font-weight: 600;'>{fund_display}</span>
                </div>
                <div>
                    <strong>Score:</strong> {fund_score}
                </div>
                <div>
                    <strong>DXY:</strong> {dxy_ret:.2f}%
                </div>
                <div>
                    <strong>US10Y:</strong> {us10y_ret:.2f}%
                </div>
            </div>
            <div style='margin-top: 8px; font-size: 0.9em; opacity: 0.8;'>
                {fund_reason}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Macro Dashboard Display ---
    real_yield_color = "#21c77a" if real_yields_data['real_yield'] > 0 else "#ff5a7a" if real_yields_data['real_yield'] < 0 else "#8a96ad"
    if show_macro_block:
        with st.expander(tr("Macro & Fundamental", "????? ? ??????????"), expanded=False):
            st.markdown(f"<div class='app-card'><h4 style='margin:0;'>{T['macro_dashboard']}</h4></div>", unsafe_allow_html=True)
            st.markdown(f"""
    <div class='app-card' style='border-left: 5px solid {real_yield_color}; margin-bottom: 15px;'>
        <h5 style='margin:0; color: var(--txt);'>{T['real_yields']}</h5>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 10px;'>
            <div>
                <strong>{T['nominal_yield']}:</strong> {real_yields_data['nominal_yield']:.2f}%
            </div>
            <div>
                <strong>{T['inflation_rate']}:</strong> {real_yields_data['inflation_rate']:.2f}%
            </div>
            <div>
                <strong>{T['real_yield']}:</strong> 
                <span style='color: {real_yield_color}; font-weight: 600;'>{real_yields_data['real_yield']:.2f}%</span>
            </div>
        </div>
    </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(T['current_rate'], f"{fed_watch_data['current_rate']:.2f}%")
            with col2:
                st.metric(T['prob_hike'], f"{fed_watch_data['prob_hike']:.1%}")
            with col3:
                st.metric(T['prob_cut'], f"{fed_watch_data['prob_cut']:.1%}")
            with col4:
                st.metric(T['prob_hold'], f"{fed_watch_data['prob_hold']:.1%}")

            st.markdown(f"<div class='app-card'><h5 style='margin:0;'>{T['economic_calendar']}</h5></div>", unsafe_allow_html=True)
            if economic_events:
                for event in economic_events:
                    impact_color = "#ff5a7a" if event['impact'] == 'High' else "#ffa500" if event['impact'] == 'Medium' else "#8a96ad"
                    st.markdown(f"""
                <div class='app-card' style='border-left: 3px solid {impact_color}; margin-bottom: 8px; padding: 10px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong>{event['event']}</strong>
                            <span style='color: {impact_color}; font-weight: 600; margin-left: 10px;'>{event['impact']}</span>
                        </div>
                        <div style='text-align: right;'>
                            <div>{event['date']} {event['time']}</div>
                            <div style='font-size: 0.9em; opacity: 0.8;'>{T['forecast']}: {event['forecast']} | {T['previous']}: {event['previous']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- Backtesting Results Display ---
    if backtest_data and show_backtest_block:
        with st.expander(T["backtest_results"], expanded=False):
            st.markdown(f"<div class='app-card'><h4 style='margin:0;'>{T['backtest_results']}</h4></div>", unsafe_allow_html=True)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(T['total_trades'], backtest_data['total_trades'])
            with col2:
                st.metric(T['win_rate'], f"{backtest_data['win_rate']:.1f}%")
            with col3:
                st.metric(T['total_return'], f"{backtest_data['total_return']:.2f}%")
            with col4:
                st.metric(T['max_drawdown'], f"{backtest_data['max_drawdown']:.2f}%")
            
            # Additional metrics
            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric(T['profit_factor'], f"{backtest_data['profit_factor']:.2f}")
            with col6:
                st.metric(T['avg_win'], f"{backtest_data['avg_win']:.2f}%")
            with col7:
                st.metric(T['avg_loss'], f"{backtest_data['avg_loss']:.2f}%")
            st.metric(tr("Expectancy", "امید ریاضی"), f"{backtest_data.get('expectancy', 0.0):.2f}%")

            if walkforward_data:
                st.markdown(f"<div class='app-card'><h5 style='margin:0;'>{tr('Walk-Forward (OOS)', 'ارزیابی واک‌فوروارد (خارج از نمونه)')}</h5></div>", unsafe_allow_html=True)
                wf1, wf2, wf3, wf4 = st.columns(4)
                wf1.metric(tr("Folds", "فولدها"), int(walkforward_data.get("folds", 0)))
                wf2.metric(tr("OOS WinRate", "وین‌ریت OOS"), f"{walkforward_data.get('oos_win_rate', 0.0):.1f}%")
                wf3.metric(tr("OOS Return", "بازده OOS"), f"{walkforward_data.get('oos_return', 0.0):.2f}%")
                wf4.metric(tr("OOS DD", "افت OOS"), f"{walkforward_data.get('oos_drawdown', 0.0):.2f}%")
            
            # Recent trades
            if backtest_data['trades']:
                st.markdown(f"<div class='app-card'><h5 style='margin:0;'>Recent Trades</h5></div>", unsafe_allow_html=True)
                
                trades_df = pd.DataFrame(backtest_data['trades'])
                trades_df['PnL'] = trades_df['pnl'].apply(lambda x: f"{x:.2f}%")
                trades_df['PnL'] = trades_df['PnL'].apply(lambda x: f"<span style='color: #21c77a;'>{x}</span>" if float(x.replace('%', '')) > 0 else f"<span style='color: #ff5a7a;'>{x}</span>")
                
                display_df = trades_df[['entry_date', 'type', 'entry_price', 'exit_price', 'PnL']].copy()
                display_df.columns = [T['entry_date'] if col == 'entry_date' else col for col in display_df.columns]
                
                st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # --- Advanced Correlation Analysis Display ---
    if correlation_data and show_corr_block:
        with st.expander(T["correlation_matrix"], expanded=False):
            st.markdown(f"<div class='app-card'><h4 style='margin:0;'>{T['correlation_matrix']}</h4></div>", unsafe_allow_html=True)
            
            # Create and display heatmap
            heatmap_fig = create_correlation_heatmap(correlation_data)
            st.plotly_chart(heatmap_fig, width='stretch')
            
            # Display detailed correlation table
            correlation_rows = []
            for asset_key, data in correlation_data.items():
                corr_asset_name = T[asset_key]
                trend_text = T[data['trend']]
                trend_color = "#21c77a" if data['trend'] == 'strengthening' else "#ff5a7a" if data['trend'] == 'weakening' else "#8a96ad"
                
                correlation_rows.append({
                    T['correlation_with']: corr_asset_name,
                    T['correlation_30d']: f"{data['corr_30d']:.3f}",
                    T['correlation_90d']: f"{data['corr_90d']:.3f}",
                    T['correlation_180d']: f"{data['corr_180d']:.3f}",
                    T['correlation_trend']: f"<span style='color: {trend_color}; font-weight: 600;'>{trend_text}</span>"
                })
            
            correlation_df = pd.DataFrame(correlation_rows)
            st.markdown(correlation_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    journal_df = load_signal_journal()
    if not journal_df.empty and show_journal_block:
        jf = journal_df[(journal_df["asset"] == asset_name) & (journal_df["timeframe"] == timeframe)].copy()
        if not jf.empty:
            with st.expander(tr("Signal Performance Journal", "ژورنال عملکرد سیگنال"), expanded=False):
                jf["timestamp"] = pd.to_datetime(jf["timestamp"], errors="coerce")
                jf = jf.sort_values("timestamp")
                st.write(tr("Recent signal logs", "لاگ اخیر سیگنال‌ها"))
                st.dataframe(jf.tail(20), width='stretch', hide_index=True)
                if len(jf) >= 2:
                    signal_shift_rate = (jf["signal"].astype(str) != jf["signal"].astype(str).shift(1)).mean() * 100.0
                    avg_conf = float(jf["confidence"].astype(float).mean())
                    avg_prob = float(jf["win_prob"].astype(float).mean())
                    j1, j2, j3 = st.columns(3)
                    j1.metric(tr("Signal Shift Rate", "نرخ تغییر سیگنال"), f"{signal_shift_rate:.1f}%")
                    j2.metric(tr("Avg Confidence", "میانگین اعتماد"), f"{avg_conf:.1f}%")
                    j3.metric(tr("Avg Win Prob", "میانگین احتمال موفقیت"), f"{avg_prob:.1f}%")

    sig_text_map = {"BUY": T["sig_buy"], "SELL": T["sig_sell"], "NEUTRAL": T["sig_neutral"]}
    method_rows = []
    method_conf_map = {
        "price_action": min(95.0, 42.0 + abs(curr_price - safe_last(ema20)) / max(curr_atr, 1e-9) * 9.0),
        "macd": min(93.0, 40.0 + abs(curr_macd_hist) * 220.0),
        "market_structure": min(94.0, 44.0 + abs(curr_price - market_structure.get("support", curr_price)) / max(curr_atr, 1e-9) * 4.0),
        "trend_follow": min(95.0, 45.0 + abs(trend_ema50 - trend_ema200) / max(curr_atr, 1e-9) * 7.0),
        "fundamental": min(92.0, 40.0 + abs(fund_score) * 18.0),
        "ai_branch": min(95.0, max(30.0, float(ai_conf))),
    }
    for method_code, method_name, method_sig, method_reason in method_signals:
        method_conf = method_conf_map.get(method_code, 50.0)
        method_rows.append(
            {
                "Method": method_name,
                "_sig_code": method_sig,
                T["method_signal_col"]: sig_text_map.get(method_sig, method_sig),
                T["method_conf_col"]: f"{method_conf:.1f}%",
                T["method_reason_col"]: str(method_reason),
                "_conf_sort": method_conf,
            }
        )
    method_df = pd.DataFrame(method_rows)
    method_df["_sig_rank"] = method_df["_sig_code"].map({"BUY": 2, "SELL": 2, "NEUTRAL": 1}).fillna(0)
    method_df = method_df.sort_values(by=["_sig_rank", "_conf_sort"], ascending=[False, False])

    def row_style(row):
        sig = row["_sig_code"]
        if sig == "BUY":
            return ["background-color: rgba(33,199,122,0.12); color: #d9fbe9"] * len(row)
        if sig == "SELL":
            return ["background-color: rgba(255,90,122,0.12); color: #ffe1e8"] * len(row)
        return ["background-color: rgba(138,150,173,0.10); color: #d6def0"] * len(row)

    # Create display dataframe and apply style
    show_df = method_df.drop(columns=["_sig_code", "_sig_rank", "_conf_sort"])
    styled_df = show_df.style.apply(lambda row: 
        ["background-color: #0d4f2c; color: #b3e5d1; border: 1px solid #21c77a; font-weight: 600"] * len(row) 
        if method_df.loc[row.name, "_sig_code"] == "BUY" else
        ["background-color: #5a1a2e; color: #ffb3c1; border: 1px solid #ff5a7a; font-weight: 600"] * len(row) 
        if method_df.loc[row.name, "_sig_code"] == "SELL" else
        ["background-color: #2a2a3e; color: #d6def0; border: 1px solid #8a96ad; font-weight: 500"] * len(row), 
        axis=1)
    
    st.markdown(f"<div class='app-card'><h4 style='margin:0;'>{T['method_title']}</h4></div>", unsafe_allow_html=True)
    st.dataframe(styled_df, width='stretch', hide_index=True)

    # --- Chart ---
    chart_snapshot_html = None
    if chart_mode == T["chart_tv"]:
        # Match TradingView chart with selected futures instrument.
        tv_symbol_map = {
            "GC=F": "COMEX:GC1!",
            "SI=F": "COMEX:SI1!",
        }
        tv_interval_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "D",
        }
        tv_symbol = tv_symbol_map.get(asset_name, "COMEX:GC1!")
        tv_interval = tv_interval_map.get(timeframe, "60")
        tv_theme = "dark"
        tv_locale = "fa" if lang == "fa" else "en"
        tv_widget = f"""
        <div class="tradingview-widget-container">
          <div id="tradingview_chart"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
            new TradingView.widget({{
              "width": "100%",
              "height": 760,
              "symbol": "{tv_symbol}",
              "interval": "{tv_interval}",
              "timezone": "Asia/Tehran",
              "theme": "{tv_theme}",
              "style": "1",
              "locale": "{tv_locale}",
              "toolbar_bg": "#0e1117",
              "enable_publishing": false,
              "allow_symbol_change": false,
              "hide_side_toolbar": false,
              "withdateranges": true,
              "container_id": "tradingview_chart"
            }});
          </script>
        </div>
        """
        components.html(tv_widget, height=780)
        if snapshot_requested:
            st.sidebar.info(tr("Snapshot export is available in Plotly chart mode.", "خروجی Snapshot فقط در حالت چارت Plotly فعال است."))
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)

        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema50, line=dict(color="orange"), name="EMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema200, line=dict(color="cyan"), name="EMA 200"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_hband(), line=dict(color="gray", width=1), name="BB High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_lband(), line=dict(color="gray", width=1), name="BB Low"), row=1, col=1)

        # TP/SL lines
        fig.add_hline(y=tp, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=tp2, line_dash="dot", line_color="green", row=1, col=1)
        fig.add_hline(y=sl, line_dash="dash", line_color="red", row=1, col=1)

        # Entry marker
        if signal in ["BUY", "STRONG BUY", "VERY STRONG BUY", "SELL", "STRONG SELL", "VERY STRONG SELL"]:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[curr_price],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="lime" if signal in ["BUY", "STRONG BUY", "VERY STRONG BUY"] else "red",
                        symbol="triangle-up" if signal in ["BUY", "STRONG BUY", "VERY STRONG BUY"] else "triangle-down",
                    ),
                    name="Entry",
                ),
                row=1,
                col=1,
            )

        # Patterns
        hammers = df[df["Is_Hammer"]]
        if not hammers.empty:
            fig.add_trace(
                go.Scatter(x=hammers.index, y=hammers["Low"], mode="markers", marker=dict(symbol="triangle-up", size=8, color="yellow"), name="Hammer"),
                row=1,
                col=1,
            )

        dojis = df[df["Is_Doji"]]
        if not dojis.empty:
            fig.add_trace(
                go.Scatter(x=dojis.index, y=dojis["Close"], mode="markers", marker=dict(symbol="circle", size=6, color="white"), name="Doji"),
                row=1,
                col=1,
            )

        # RSI panel
        fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color="purple"), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        if len(df.index) > live_window:
            fig.update_xaxes(range=[df.index[-live_window], df.index[-1]], row=1, col=1)
            fig.update_xaxes(range=[df.index[-live_window], df.index[-1]], row=2, col=1)

        fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width='stretch')
        if snapshot_requested:
            chart_snapshot_html = fig.to_html(include_plotlyjs="cdn")
            st.sidebar.download_button(
                label=tr("Download Snapshot", "دانلود Snapshot"),
                data=chart_snapshot_html,
                file_name=f"chart_snapshot_{asset_name}_{timeframe}.html",
                mime="text/html",
                width='stretch',
            )

    if show_logic_block:
        with st.expander(T["logic"]):
            st.write(f"{T['bullish_factors']}:")
            for reason in bullish_reasons[:8]:
                st.write(f"- {reason}")
            st.write(f"{T['bearish_factors']}:")
            for reason in bearish_reasons[:8]:
                st.write(f"- {reason}")
            st.write(f"- {tr('Higher TF', 'تایم‌فریم بالاتر')}: {higher_tf} | {T['trend']}: {ht_trend}")
            st.write(f"- {tr('DXY 5-bar return', 'بازده دلار ۵ کندل')}: {dxy_ret:.2f}% | {tr('US10Y 5-bar return', 'بازده 10Y پنج کندل')}: {us10y_ret:.2f}%")
            st.write(f"- {tr('Silver 5-bar return', 'بازده نقره ۵ کندل')}: {silver_ret:.2f}% | {tr('Copper 5-bar return', 'بازده مس ۵ کندل')}: {copper_ret:.2f}%")

    st.caption(f"{T['last_update']}: {now_iran_str('%Y-%m-%d %H:%M:%S')} IRT")
    st.markdown(
        f"<div class='site-footer'>© 2026 Golden Terminal • {tr('Clean layout mode', 'چیدمان خلوت فعال')} • AI {ai_sig} ({ai_mode_label})</div>",
        unsafe_allow_html=True,
    )


    if auto_refresh:
        try:
            # Streamlit built-in auto-refresh
            st_autorefresh = getattr(st, "autorefresh", None)
            if st_autorefresh is not None:
                st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")
            else:
                import time

                time.sleep(refresh_sec)
                st.rerun()
        except Exception:
            import time

            time.sleep(refresh_sec)
            st.rerun()

else:
    st.error(T["data_fail"])

