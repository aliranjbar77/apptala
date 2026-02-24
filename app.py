from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from templates import live_price_html, mini_box_close_html, mini_box_open_html, top_banner_html


st.set_page_config(page_title="Gold Pro Terminal", layout="wide")
GOLD_SPOT_SYMBOL = "XAUUSD=X"

def get_texts(language: str) -> dict[str, str]:
    en = {
        "lang_label": "Language",
        "theme_label": "Theme",
        "dark": "Dark",
        "light": "Light",
        "live_caption": "LIVE XAU/USD TERMINAL",
        "live_title": "Professional Gold Dashboard",
        "main_live_box": "Main Live Price Box",
        "change": "Change",
        "source": "Source",
        "auto_refresh": "Auto refresh",
        "tehran": "Tehran",
        "final_signal": "Final Signal",
        "confidence": "Confidence",
        "fundamental": "Fundamental Overview",
        "smc_engine": "Smart-Money Engine (Poursamadi)",
        "classic_engine": "Classic Engine",
        "signal_monitor": "Multi-Logic Signal Monitor",
        "main_chart": "Main Chart",
        "engine_details": "Engine Details",
        "htf_trend": "HTF trend (4H)",
        "reason": "Reason",
        "hh_ll": "HH detected",
        "bos": "Bullish BOS",
        "swing": "Swing High / Swing Low",
        "price_action": "Price Action",
        "rsi": "RSI",
        "trend": "Trend (EMA)",
        "macd": "MACD",
        "bollinger": "Bollinger",
        "support_resistance": "Support/Resistance",
        "fng_unavailable": "API unavailable",
        "corr_caption": "Correlation is based on daily returns over recent 3 months.",
        "strategy_label": "Strategy",
        "style_label": "Style",
        "heatmap_title": "Fundamental Heatmap",
    }
    fa = {
        "lang_label": "زبان",
        "theme_label": "حالت نمایش",
        "dark": "تیره",
        "light": "روشن",
        "live_caption": "ترمینال زنده XAU/USD",
        "live_title": "داشبورد حرفه‌ای طلا",
        "main_live_box": "باکس اصلی قیمت لحظه‌ای",
        "change": "تغییر",
        "source": "منبع",
        "auto_refresh": "رفرش خودکار",
        "tehran": "تهران",
        "final_signal": "سیگنال نهایی",
        "confidence": "اعتماد",
        "fundamental": "نمای کلی فاندامنتال",
        "smc_engine": "موتور اسمارت‌مانی (پورصمدی)",
        "classic_engine": "موتور کلاسیک",
        "signal_monitor": "مانیتور چندمنطقی سیگنال",
        "main_chart": "نمودار اصلی",
        "engine_details": "جزئیات موتورها",
        "htf_trend": "روند تایم‌فریم بالاتر (4H)",
        "reason": "دلیل",
        "hh_ll": "تشخیص HH/LL",
        "bos": "BOS صعودی",
        "swing": "سویینگ سقف / کف",
        "price_action": "پرایس اکشن",
        "rsi": "RSI",
        "trend": "روند (EMA)",
        "macd": "MACD",
        "bollinger": "بولینگر",
        "support_resistance": "حمایت/مقاومت",
        "fng_unavailable": "API در دسترس نیست",
        "corr_caption": "همبستگی بر اساس بازدهی روزانه سه ماه اخیر محاسبه شده است.",
        "strategy_label": "استراتژی",
        "style_label": "استایل",
        "heatmap_title": "هیت‌مپ فاندامنتال",
    }
    return fa if language == "فارسی" else en


def load_css(theme: str, visual_style: str, path: str = "styles.css") -> None:
    css_path = Path(path)
    if not css_path.exists():
        st.warning(f"CSS file not found: {css_path}")
        return
    css = css_path.read_text(encoding="utf-8")
    dark_override = """
:root {
    --bg: #0f172a;
    --panel: #111827;
    --line: #2d3d58;
    --text: #e6edf7;
    --muted: #93a4bd;
    --buy: #22c55e;
    --sell: #ef4444;
    --neutral: #94a3b8;
    --app-grad-start: #1a2a44;
    --app-grad-end: #0f172a;
    --card-grad-a: rgba(20, 31, 53, 0.94);
    --card-grad-b: rgba(17, 24, 39, 0.96);
    --mini-grad-a: rgba(20, 31, 53, 0.9);
    --mini-grad-b: rgba(17, 24, 39, 0.94);
    --metric-bg: rgba(15, 23, 42, 0.4);
    --shadow: 0 10px 28px rgba(2, 8, 23, 0.34);
    --buy-bg: rgba(34, 197, 94, 0.16);
    --buy-text: #86efac;
    --buy-border: rgba(34, 197, 94, 0.45);
    --sell-bg: rgba(239, 68, 68, 0.16);
    --sell-text: #fca5a5;
    --sell-border: rgba(239, 68, 68, 0.45);
    --neutral-bg: rgba(148, 163, 184, 0.15);
    --neutral-text: #cbd5e1;
    --neutral-border: rgba(148, 163, 184, 0.45);
}
"""
    light_override = """
:root {
    --bg: #eff5ff;
    --panel: #ffffff;
    --line: #b9c9e3;
    --text: #0f172a;
    --muted: #334155;
    --buy: #15803d;
    --sell: #b91c1c;
    --neutral: #64748b;
    --app-grad-start: #cfe1ff;
    --app-grad-end: #eff5ff;
    --card-grad-a: rgba(255, 255, 255, 0.98);
    --card-grad-b: rgba(244, 249, 255, 0.98);
    --mini-grad-a: rgba(255, 255, 255, 0.98);
    --mini-grad-b: rgba(240, 246, 255, 0.98);
    --metric-bg: rgba(255, 255, 255, 0.98);
    --shadow: 0 14px 30px rgba(30, 64, 175, 0.14);
    --buy-bg: rgba(22, 163, 74, 0.12);
    --buy-text: #166534;
    --buy-border: rgba(22, 163, 74, 0.42);
    --sell-bg: rgba(220, 38, 38, 0.12);
    --sell-text: #991b1b;
    --sell-border: rgba(220, 38, 38, 0.42);
    --neutral-bg: rgba(100, 116, 139, 0.12);
    --neutral-text: #334155;
    --neutral-border: rgba(100, 116, 139, 0.38);
}
"""
    glass_override = """
.big-live, .mini-box {
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-color: color-mix(in srgb, var(--line) 55%, #60a5fa 45%) !important;
}
.big-live::before, .mini-box::before {
    opacity: 0.72 !important;
}
.metric-card {
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
"""
    minimal_override = """
.big-live::before, .mini-box::before {
    opacity: 0.42 !important;
}
"""
    active_override = light_override if theme == "Light" else dark_override
    style_override = glass_override if visual_style == "Glass" else minimal_override
    st.markdown(f"<style>{css}\n{active_override}\n{style_override}</style>", unsafe_allow_html=True)


def get_timeframe_config(timeframe: str) -> tuple[str, str, int]:
    cfg = {
        "5m": ("30d", "5m", 180),
        "15m": ("60d", "15m", 260),
        "1h": ("730d", "1h", 260),
        "4h": ("730d", "4h", 260),
        "1d": ("5y", "1d", 260),
    }
    return cfg.get(timeframe, ("730d", "1h", 260))


def get_htf_config(timeframe: str) -> tuple[str, str]:
    cfg = {
        "5m": ("60d", "15m"),
        "15m": ("730d", "1h"),
        "1h": ("730d", "4h"),
        "4h": ("5y", "1d"),
        "1d": ("10y", "1wk"),
    }
    return cfg.get(timeframe, ("730d", "4h"))


def porsamadi_reason_fa(porsamadi: dict) -> str:
    sig = str(porsamadi.get("signal", ""))
    if "BUY" in sig:
        return "شکست ساختار صعودی (BOS) با تایید HH/Order Block و هم‌راستایی تایم‌فریم بالاتر."
    if "SELL" in sig:
        return "شکست ساختار نزولی (BOS) با تایید LL/Order Block و هم‌راستایی تایم‌فریم بالاتر."
    return "در این تایم‌فریم ستاپ واضح پورصمدی تشکیل نشده است."


def as_series(values, index: pd.Index, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        out = pd.to_numeric(values, errors="coerce")
        if not out.index.equals(index):
            out = out.reindex(index)
        out.name = name
        return out
    return pd.Series(np.full(len(index), float(values)), index=index, name=name)


def safe_last(series: pd.Series, default: float = 0.0) -> float:
    s = series.dropna()
    return default if s.empty else float(s.iloc[-1])


def safe_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    for _ in range(3):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.loc[:, ~df.columns.duplicated()].copy()
        except Exception:
            pass
    return pd.DataFrame()


def resample_to_interval(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    if any(col not in df.columns for col in ["Open", "High", "Low", "Close"]):
        return pd.DataFrame()

    rule_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
    rule = rule_map.get(target_interval)
    if not rule:
        return pd.DataFrame()

    out = (
        df.resample(rule)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum" if "Volume" in df.columns else "first",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    return out


def get_analysis_data(period: str, interval: str, min_bars: int) -> tuple[pd.DataFrame, str]:
    primary_symbol = GOLD_SPOT_SYMBOL
    backup_symbol = "GC=F"

    symbols = [primary_symbol, backup_symbol]
    for sym in symbols:
        df = safe_download(sym, period=period, interval=interval)
        if not df.empty and len(df) >= min_bars:
            return df, sym

    # Soft fallback: accept lower bars if at least enough for core indicators.
    min_soft = 120
    for sym in symbols:
        df = safe_download(sym, period=period, interval=interval)
        if not df.empty and len(df) >= min_soft:
            return df, sym

    # Hard fallback for intraday gaps: build requested bars from 1m feed.
    if interval in {"5m", "15m"}:
        for sym in symbols:
            one_min = safe_download(sym, period="7d", interval="1m")
            rebuilt = resample_to_interval(one_min, interval)
            if not rebuilt.empty and len(rebuilt) >= min_soft:
                return rebuilt, sym

    return pd.DataFrame(), primary_symbol


def get_goldapi_live() -> tuple[float | None, float | None, str]:
    try:
        headers = {
            "x-access-token": st.secrets["GOLD_API_KEY"],
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    except Exception:
        return None, None, "GoldAPI key not found"

    try:
        resp = requests.get(
            "https://www.goldapi.io/api/XAU/USD",
            headers=headers,
            params={"cache": "false"},
            timeout=10,
        )
        if resp.status_code in {400, 401, 404}:
            return None, None, f"GoldAPI status={resp.status_code}"
        if resp.status_code != 200:
            return None, None, f"GoldAPI status={resp.status_code}"
        payload = resp.json()
        if "price" not in payload:
            return None, None, "GoldAPI invalid payload (missing price)"
        current_price = float(payload["price"])
        prev = float(payload.get("prev_day_price", current_price))
        return current_price, current_price - prev, "GoldAPI"
    except Exception as exc:
        return None, None, f"GoldAPI exception: {exc}"


def get_yf_live_backup() -> tuple[float | None, float | None, str]:
    intraday = safe_download("GC=F", period="1d", interval="1m")
    if intraday.empty:
        intraday = safe_download("GC=F", period="5d", interval="15m")
    if intraday.empty or "Close" not in intraday.columns:
        return None, None, "yfinance backup failed"

    close = pd.to_numeric(intraday["Close"], errors="coerce").dropna()
    if close.empty:
        return None, None, "yfinance backup empty close"

    curr = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else curr
    return curr, curr - prev, "yfinance backup (GC=F)"


def get_live_gold_price() -> tuple[float | None, float | None, str, str]:
    p, chg, src = get_goldapi_live()
    if p is not None:
        return p, chg, src, "ok"
    return None, None, "none", src


def fetch_fear_greed() -> tuple[float | None, str]:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        if r.status_code != 200:
            return None, f"status={r.status_code}"
        data = r.json().get("data", [])
        if not data:
            return None, "empty"
        value = float(data[0]["value"])
        label = str(data[0]["value_classification"])
        return value, label
    except Exception as exc:
        return None, str(exc)


def detect_pivots(high: pd.Series, low: pd.Series, span: int = 3) -> tuple[pd.Series, pd.Series]:
    # Centered window pivot detection for swing points.
    ph = high.where(high == high.rolling(window=2 * span + 1, center=True).max())
    pl = low.where(low == low.rolling(window=2 * span + 1, center=True).min())
    return ph.dropna(), pl.dropna()


def compute_htf_trend(df_htf: pd.DataFrame, live_price: float) -> str:
    if df_htf.empty or len(df_htf) < 220:
        return "NEUTRAL"

    idx = df_htf.index
    close = as_series(df_htf["Close"], idx, "htf_close")
    ema50 = as_series(EMAIndicator(close, window=50).ema_indicator(), idx, "htf_ema50")
    ema200 = as_series(EMAIndicator(close, window=200).ema_indicator(), idx, "htf_ema200")

    e50 = safe_last(ema50, live_price)
    e200 = safe_last(ema200, live_price)

    if live_price > e50 > e200:
        return "BULLISH"
    if live_price < e50 < e200:
        return "BEARISH"
    return "NEUTRAL"


def compute_porsamadi(df: pd.DataFrame, df_htf: pd.DataFrame, live_price: float) -> dict:
    high = as_series(df["High"], df.index, "high")
    low = as_series(df["Low"], df.index, "low")
    close = as_series(df["Close"], df.index, "close")

    atr = as_series(AverageTrueRange(high, low, close, window=14).average_true_range(), df.index, "atr")
    atr_last = max(0.5, safe_last(atr, 2.0))

    swing_highs, swing_lows = detect_pivots(high, low, span=3)

    recent_highs = swing_highs.tail(2)
    recent_lows = swing_lows.tail(2)

    hh = len(recent_highs) == 2 and float(recent_highs.iloc[-1]) > float(recent_highs.iloc[-2])
    ll = len(recent_lows) == 2 and float(recent_lows.iloc[-1]) < float(recent_lows.iloc[-2])

    last_swing_high = safe_last(swing_highs, safe_last(high, live_price))
    last_swing_low = safe_last(swing_lows, safe_last(low, live_price))

    bos_buffer = 0.1 * atr_last
    bullish_bos = live_price > (last_swing_high + bos_buffer)
    bearish_bos = live_price < (last_swing_low - bos_buffer)

    recent = df.tail(min(90, len(df))).copy()
    bullish_ob = None
    bearish_ob = None

    # OB = last opposite candle body/range before structural push.
    bear_candles = recent[recent["Close"] < recent["Open"]]
    bull_candles = recent[recent["Close"] > recent["Open"]]

    if not bear_candles.empty:
        c = bear_candles.iloc[-1]
        lo = float(min(c["Open"], c["Close"]))
        hi = float(max(c["Open"], c["Close"]))
        bullish_ob = (lo, hi)

    if not bull_candles.empty:
        c = bull_candles.iloc[-1]
        lo = float(min(c["Open"], c["Close"]))
        hi = float(max(c["Open"], c["Close"]))
        bearish_ob = (lo, hi)

    in_bull_ob = False
    in_bear_ob = False
    if bullish_ob:
        in_bull_ob = bullish_ob[0] - 0.15 * atr_last <= live_price <= bullish_ob[1] + 0.15 * atr_last
    if bearish_ob:
        in_bear_ob = bearish_ob[0] - 0.15 * atr_last <= live_price <= bearish_ob[1] + 0.15 * atr_last

    htf_trend = compute_htf_trend(df_htf, live_price)

    signal = "NEUTRAL"
    reason = "No clean SMC setup"

    bullish_setup = bullish_bos and (hh or in_bull_ob)
    bearish_setup = bearish_bos and (ll or in_bear_ob)

    if bullish_setup and htf_trend in {"BULLISH", "NEUTRAL"}:
        signal = "BUY"
        reason = "Bullish BOS + HH/OB with HTF alignment"
    elif bearish_setup and htf_trend in {"BEARISH", "NEUTRAL"}:
        signal = "SELL"
        reason = "Bearish BOS + LL/OB with HTF alignment"

    return {
        "signal": signal,
        "reason": reason,
        "hh": bool(hh),
        "ll": bool(ll),
        "bullish_bos": bool(bullish_bos),
        "bearish_bos": bool(bearish_bos),
        "bullish_ob": bullish_ob,
        "bearish_ob": bearish_ob,
        "last_swing_high": float(last_swing_high),
        "last_swing_low": float(last_swing_low),
        "htf_trend": htf_trend,
    }


def compute_classic_signals(df: pd.DataFrame, live_price: float) -> dict:
    idx = df.index
    close = as_series(df["Close"], idx, "close")
    high = as_series(df["High"], idx, "high")
    low = as_series(df["Low"], idx, "low")

    rsi = as_series(RSIIndicator(close, window=14).rsi(), idx, "rsi")
    ema20 = as_series(EMAIndicator(close, window=20).ema_indicator(), idx, "ema20")
    ema50 = as_series(EMAIndicator(close, window=50).ema_indicator(), idx, "ema50")
    ema200 = as_series(EMAIndicator(close, window=200).ema_indicator(), idx, "ema200")

    macd_obj = MACD(close)
    macd_line = as_series(macd_obj.macd(), idx, "macd")
    macd_signal = as_series(macd_obj.macd_signal(), idx, "macd_signal")
    macd_hist = as_series(macd_obj.macd_diff(), idx, "macd_hist")

    bb_obj = BollingerBands(close, window=20, window_dev=2)
    bb_high = as_series(bb_obj.bollinger_hband(), idx, "bb_high")
    bb_low = as_series(bb_obj.bollinger_lband(), idx, "bb_low")

    support = float(low.tail(60).quantile(0.15)) if len(low) >= 20 else float(low.min())
    resistance = float(high.tail(60).quantile(0.85)) if len(high) >= 20 else float(high.max())

    price_action = "NEUTRAL"
    if live_price > resistance:
        price_action = "BUY"
    elif live_price < support:
        price_action = "SELL"

    momentum = "NEUTRAL"
    rsi_last = safe_last(rsi, 50)
    if rsi_last <= 30:
        momentum = "BUY"
    elif rsi_last >= 70:
        momentum = "SELL"
    elif rsi_last > 55:
        momentum = "BUY"
    elif rsi_last < 45:
        momentum = "SELL"

    trend = "NEUTRAL"
    if live_price > safe_last(ema20) > safe_last(ema50) > safe_last(ema200):
        trend = "BUY"
    elif live_price < safe_last(ema20) < safe_last(ema50) < safe_last(ema200):
        trend = "SELL"
    elif live_price > safe_last(ema200):
        trend = "BUY"
    elif live_price < safe_last(ema200):
        trend = "SELL"

    macd_sig = "NEUTRAL"
    if safe_last(macd_hist) > 0 and safe_last(macd_line) > safe_last(macd_signal):
        macd_sig = "BUY"
    elif safe_last(macd_hist) < 0 and safe_last(macd_line) < safe_last(macd_signal):
        macd_sig = "SELL"

    bollinger_sig = "NEUTRAL"
    if live_price < safe_last(bb_low):
        bollinger_sig = "BUY"
    elif live_price > safe_last(bb_high):
        bollinger_sig = "SELL"

    return {
        "price_action": price_action,
        "momentum": momentum,
        "trend": trend,
        "macd": macd_sig,
        "bollinger": bollinger_sig,
        "support": support,
        "resistance": resistance,
        "rsi": rsi,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_high": bb_high,
        "bb_low": bb_low,
    }


def strength_from_ratio(ratio: float) -> str:
    if ratio >= 0.75:
        return "قوی"
    if ratio >= 0.55:
        return "متوسط"
    if ratio >= 0.4:
        return "ضعیف"
    return "خنثی"


def classic_status_from_votes(buy_votes: int, sell_votes: int) -> str:
    if buy_votes > sell_votes:
        return "سیگنال خرید کلاسیک"
    if sell_votes > buy_votes:
        return "سیگنال فروش کلاسیک"
    return "کلاسیک خنثی / انتظار"


def evaluate_classic_engine(classic: dict) -> dict:
    parts = {
        "Price Action": classic.get("price_action", "NEUTRAL"),
        "RSI": classic.get("momentum", "NEUTRAL"),
        "Trend EMA": classic.get("trend", "NEUTRAL"),
        "MACD": classic.get("macd", "NEUTRAL"),
    }
    buy_votes = sum(1 for v in parts.values() if v == "BUY")
    sell_votes = sum(1 for v in parts.values() if v == "SELL")
    dominant = max(buy_votes, sell_votes)
    ratio = dominant / 4.0
    signal = "BUY" if buy_votes > sell_votes else "SELL" if sell_votes > buy_votes else "NEUTRAL"
    return {
        "signal": signal,
        "power_percent": ratio * 100.0,
        "strength": strength_from_ratio(ratio),
        "status": classic_status_from_votes(buy_votes, sell_votes),
        "parts": parts,
    }


def evaluate_poursamadi_engine(p: dict) -> dict:
    bullish_bos = bool(p.get("bullish_bos", False))
    bearish_bos = bool(p.get("bearish_bos", False))
    htf = str(p.get("htf_trend", "NEUTRAL"))

    if bullish_bos and htf == "BULLISH":
        return {
            "signal": "BUY",
            "power_percent": 85.0,
            "strength": "قوی",
            "status": "ساختار صعودی - تایید تایم‌فریم بالاتر - منتظر بازگشت یا ادامه",
        }
    if bearish_bos and htf == "BEARISH":
        return {
            "signal": "SELL",
            "power_percent": 85.0,
            "strength": "قوی",
            "status": "ساختار نزولی - تایید تایم‌فریم بالاتر - منتظر بازگشت یا ادامه",
        }
    if bullish_bos:
        return {
            "signal": "BUY",
            "power_percent": 60.0,
            "strength": "متوسط",
            "status": "BOS صعودی دیده شد - تایم‌فریم بالاتر هنوز قطعی نیست",
        }
    if bearish_bos:
        return {
            "signal": "SELL",
            "power_percent": 60.0,
            "strength": "متوسط",
            "status": "BOS نزولی دیده شد - تایم‌فریم بالاتر هنوز قطعی نیست",
        }
    return {"signal": "NEUTRAL", "power_percent": 25.0, "strength": "خنثی", "status": "ساختار خنثی - منتظر شکست معتبر"}


def signal_class(signal: str) -> str:
    s = str(signal).upper()
    if "BUY" in s or s == "BULLISH":
        return "signal-buy"
    if "SELL" in s or s == "BEARISH":
        return "signal-sell"
    return "signal-neutral"


def signal_badge_html(signal: str) -> str:
    cls = signal_class(signal)
    return f'<span class="signal-pill {cls}">{signal}</span>'


def get_engine_subset(engine_signals: dict) -> dict:
    keys = ["Price Action", "RSI", "Trend EMA", "MACD", "Poursamadi"]
    return {k: engine_signals.get(k, "NEUTRAL") for k in keys}


def render_signal_micro_chart(engine_signals: dict, theme: str, title: str):
    engine_signals = get_engine_subset(engine_signals)
    mapping = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
    labels = list(engine_signals.keys())
    vals = [mapping.get(engine_signals[k], 0) for k in labels]
    colors = ["#16c784" if v > 0 else "#ff4d6d" if v < 0 else "#8ea4c9" for v in vals]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                text=[engine_signals[k] for k in labels],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white" if theme == "Light" else "plotly_dark",
        title=title,
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[-1.4, 1.4], tickvals=[-1, 0, 1], ticktext=["SELL", "NEUTRAL", "BUY"]),
    )
    st.plotly_chart(fig, use_container_width=True)


def heat_color(value: float) -> str:
    # value in [-1, 1]
    v = max(-1.0, min(1.0, value))
    if v >= 0:
        alpha = 0.16 + 0.32 * v
        return f"rgba(34,197,94,{alpha:.3f})"
    alpha = 0.16 + 0.32 * abs(v)
    return f"rgba(239,68,68,{alpha:.3f})"


def render_fundamental_heatmap(items: list[tuple[str, float | None]], title: str) -> None:
    parts = [f"<div class='heatmap-title'>{title}</div>", "<div class='heatmap-grid'>"]
    for name, score in items:
        val = 0.0 if score is None else max(-1.0, min(1.0, score))
        parts.append(
            f"""
<div class="heat-cell" style="background:{heat_color(val)};">
  <div class="heat-name">{name}</div>
  <div class="heat-score">{val:+.2f}</div>
</div>
"""
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_fundamental_box(labels: dict[str, str]) -> None:
    dxy = safe_download("DX-Y.NYB", period="1mo", interval="1h")
    silver = safe_download("SI=F", period="3mo", interval="1d")
    oil = safe_download("CL=F", period="3mo", interval="1d")
    gold = safe_download(GOLD_SPOT_SYMBOL, period="3mo", interval="1d")

    dxy_last = None
    if not dxy.empty and "Close" in dxy.columns:
        dxy_close = pd.to_numeric(dxy["Close"], errors="coerce").dropna()
        if not dxy_close.empty:
            dxy_last = float(dxy_close.iloc[-1])

    fg_value, fg_label = fetch_fear_greed()

    corr_silver = None
    corr_oil = None

    if not gold.empty and "Close" in gold.columns:
        gret = pd.to_numeric(gold["Close"], errors="coerce").pct_change().dropna()

        if not silver.empty and "Close" in silver.columns:
            sret = pd.to_numeric(silver["Close"], errors="coerce").pct_change().dropna()
            joint = pd.concat([gret, sret], axis=1).dropna()
            if len(joint) > 10:
                corr_silver = float(joint.corr().iloc[0, 1])

        if not oil.empty and "Close" in oil.columns:
            oret = pd.to_numeric(oil["Close"], errors="coerce").pct_change().dropna()
            joint = pd.concat([gret, oret], axis=1).dropna()
            if len(joint) > 10:
                corr_oil = float(joint.corr().iloc[0, 1])

    dxy_state = "NEUTRAL"
    if dxy_last is not None:
        dxy_state = "SELL" if dxy_last >= 103 else "BUY"

    fg_state = "NEUTRAL"
    fg_text = labels["fng_unavailable"]
    if fg_value is not None:
        fg_text = f"{fg_value:.0f} ({fg_label})"
        if fg_value >= 70:
            fg_state = "SELL"
        elif fg_value <= 30:
            fg_state = "BUY"

    silver_state = "NEUTRAL" if corr_silver is None else ("BUY" if corr_silver >= 0 else "SELL")
    oil_state = "NEUTRAL" if corr_oil is None else ("BUY" if corr_oil >= 0 else "SELL")

    st.markdown(mini_box_open_html(), unsafe_allow_html=True)
    st.markdown(f"### {labels['fundamental']}")
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">DXY</div>
  <div class="metric-value">{f"{dxy_last:.2f}" if dxy_last is not None else "N/A"}</div>
  <div>{signal_badge_html(dxy_state)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    c2.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">Fear &amp; Greed</div>
  <div class="metric-value">{fg_text}</div>
  <div>{signal_badge_html(fg_state)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    c3.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">Gold-Silver Corr</div>
  <div class="metric-value">{f"{corr_silver:+.2f}" if corr_silver is not None else "N/A"}</div>
  <div>{signal_badge_html(silver_state)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    c4.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">Gold-Oil Corr</div>
  <div class="metric-value">{f"{corr_oil:+.2f}" if corr_oil is not None else "N/A"}</div>
  <div>{signal_badge_html(oil_state)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    dxy_score = None if dxy_last is None else max(-1.0, min(1.0, (100.0 - dxy_last) / 4.0))
    fg_score = None if fg_value is None else max(-1.0, min(1.0, (50.0 - fg_value) / 30.0))
    silver_score = None if corr_silver is None else max(-1.0, min(1.0, corr_silver))
    oil_score = None if corr_oil is None else max(-1.0, min(1.0, corr_oil))
    render_fundamental_heatmap(
        [("DXY", dxy_score), ("Fear&Greed", fg_score), ("Silver Corr", silver_score), ("Oil Corr", oil_score)],
        labels["heatmap_title"],
    )

    st.caption(labels["corr_caption"])
    st.markdown(mini_box_close_html(), unsafe_allow_html=True)


def render_top_banner(labels: dict[str, str]) -> None:
    st.markdown(top_banner_html(labels["live_caption"], labels["live_title"]), unsafe_allow_html=True)


def render_live_price_box(
    live_price: float,
    price_change: float | None,
    source: str,
    labels: dict[str, str],
) -> None:
    st.markdown(
        live_price_html(live_price, price_change, source, labels),
        unsafe_allow_html=True,
    )




def render_poursamadi_box(porsamadi_eval: dict, labels: dict[str, str]) -> None:
    st.markdown(mini_box_open_html(), unsafe_allow_html=True)
    st.markdown(f"### {labels['smc_engine']}")
    st.markdown(
        f"""
<div class="metric-card porsamadi-card">
  <div class="porsamadi-row">
    <span class="metric-label">Signal ({porsamadi_eval.get('strength','خنثی')})</span>
    {signal_badge_html(porsamadi_eval.get('signal','NEUTRAL'))}
  </div>
  <div class="strength-bar engine-strength">
    <div class="strength-fill" style="width:{float(porsamadi_eval.get('power_percent', 0.0)):.1f}%;"></div>
  </div>
  <div class="porsamadi-reason">وضعیت: {porsamadi_eval.get('status','ساختار خنثی')}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(mini_box_close_html(), unsafe_allow_html=True)


def render_classic_box(classic_eval: dict, labels: dict[str, str]) -> None:
    st.markdown(mini_box_open_html(), unsafe_allow_html=True)
    st.markdown(f"### {labels['classic_engine']}")
    st.markdown(
        f"""
<div class="metric-card porsamadi-card">
  <div class="porsamadi-row">
    <span class="metric-label">Signal ({classic_eval.get('strength','خنثی')})</span>
    {signal_badge_html(classic_eval.get('signal','NEUTRAL'))}
  </div>
  <div class="strength-bar engine-strength">
    <div class="strength-fill" style="width:{float(classic_eval.get('power_percent', 0.0)):.1f}%;"></div>
  </div>
  <div class="porsamadi-reason">وضعیت: {classic_eval.get('status','کلاسیک خنثی')}</div>
  <div class="logic-grid" style="margin-top:8px;">
    <span class="logic-item">Price Action: {signal_badge_html(classic_eval.get('parts',{}).get('Price Action','NEUTRAL'))}</span>
    <span class="logic-item">RSI: {signal_badge_html(classic_eval.get('parts',{}).get('RSI','NEUTRAL'))}</span>
    <span class="logic-item">Trend EMA: {signal_badge_html(classic_eval.get('parts',{}).get('Trend EMA','NEUTRAL'))}</span>
    <span class="logic-item">MACD: {signal_badge_html(classic_eval.get('parts',{}).get('MACD','NEUTRAL'))}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(mini_box_close_html(), unsafe_allow_html=True)


def main() -> None:
    controls = st.columns([1, 1, 1, 1, 4])
    language = controls[0].selectbox("Language", ["English", "فارسی"], key="ui_lang")
    theme = controls[1].selectbox("Theme", ["Dark", "Light"], key="ui_theme")
    timeframe = controls[2].selectbox("TF", ["5m", "15m", "1h", "4h", "1d"], key="signal_timeframe")
    style = controls[3].selectbox("Style", ["Minimal Pro", "Glass"], key="ui_style")
    labels = get_texts(language)
    load_css(theme, style)

    st_autorefresh(interval=10000, key="gold_refresh_10s")

    render_top_banner(labels)

    live_price, price_change, source, status = get_live_gold_price()
    if live_price is None:
        st.error(f"Live price unavailable. Details: {status}")
        st.stop()

    period, interval, min_bars = get_timeframe_config(timeframe)
    htf_period, htf_interval = get_htf_config(timeframe)

    df, analysis_symbol = get_analysis_data(period, interval, min_bars)
    if df.empty:
        st.error(f"Historical data is not sufficient for full analysis on {timeframe}.")
        st.stop()

    df_htf = safe_download(analysis_symbol, period=htf_period, interval=htf_interval)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    if not df_htf.empty:
        for col in ["Open", "High", "Low", "Close"]:
            df_htf[col] = pd.to_numeric(df_htf[col], errors="coerce")
        df_htf = df_htf.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    else:
        # Keep analysis alive if HTF feed is temporarily unavailable.
        rebuilt_htf = resample_to_interval(df, htf_interval)
        if not rebuilt_htf.empty:
            df_htf = rebuilt_htf.copy()

    last_idx = df.index[-1]
    df.at[last_idx, "Close"] = float(live_price)
    df.at[last_idx, "High"] = max(float(df.at[last_idx, "High"]), float(live_price))
    df.at[last_idx, "Low"] = min(float(df.at[last_idx, "Low"]), float(live_price))

    classic = compute_classic_signals(df, live_price)
    porsamadi_raw = compute_porsamadi(df, df_htf, live_price)
    classic_eval = evaluate_classic_engine(classic)
    smc_eval = evaluate_poursamadi_engine(porsamadi_raw)

    render_live_price_box(live_price, price_change, source, labels)
    st.caption(f"Timeframe: {timeframe} | HTF: {htf_interval} | Style: {style} | Analysis data: {analysis_symbol}")

    c1, c2 = st.columns(2)
    with c1:
        render_classic_box(classic_eval, labels)
    with c2:
        render_poursamadi_box(smc_eval, labels)
    render_fundamental_box(labels)


if __name__ == "__main__":
    main()





