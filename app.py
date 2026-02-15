import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, MACD, ADXIndicator

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
            background: radial-gradient(1200px 500px at 10% -20%, #1a2b52 0%, var(--bg) 55%);
        }
        .app-card {
            background: linear-gradient(180deg, var(--panel-2), var(--panel));
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }
        .signal-buy { background-color: rgba(33,199,122,0.12); border-left: 5px solid var(--buy); padding: 12px; border-radius: 10px; }
        .signal-sell { background-color: rgba(255,90,122,0.12); border-left: 5px solid var(--sell); padding: 12px; border-radius: 10px; }
        .signal-neutral { background-color: rgba(138,150,173,0.15); border-left: 5px solid var(--neutral); padding: 12px; border-radius: 10px; }
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
        .method-reason { font-size: 12px; color: var(--muted); line-height: 1.4; }
        @media (max-width: 900px) {
            .method-grid { grid-template-columns: 1fr; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helpers ---
def get_data(symbol: str, period: str, interval: str):
    data = yf.download(symbol, period=period, interval=interval)
    dxy = yf.download("DX-Y.NYB", period=period, interval=interval)

    # Fix MultiIndex columns for newer pandas versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex):
        dxy.columns = dxy.columns.get_level_values(0)
    return data, dxy


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
        ctx_df = yf.download(ticker, period=period, interval=interval)
        if isinstance(ctx_df.columns, pd.MultiIndex):
            ctx_df.columns = ctx_df.columns.get_level_values(0)
        out[key] = ctx_df
    return out

def get_higher_timeframe(symbol: str, base_interval: str):
    mapping = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h", "4h": "1d", "1d": "1wk"}
    higher = mapping.get(base_interval, "1d")

    period = "60d" if higher in ["1m", "5m", "15m"] else "120d" if higher in ["1h", "4h"] else "5y"
    data = yf.download(symbol, period=period, interval=higher)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data, higher

def calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close_s = df["Close"].squeeze()
    open_s = df["Open"].squeeze()
    high_s = df["High"].squeeze()
    low_s = df["Low"].squeeze()

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
        "silver_copper": "Silver and Copper returns"
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
        "silver_copper": "بازده نقره و مس"
    }
}

lang_choice = st.sidebar.selectbox(TEXT["en"]["lang"], ["فارسی", "English"], index=0)
lang = "fa" if lang_choice == "فارسی" else "en"
T = TEXT[lang]


def tr(en_text: str, fa_text: str) -> str:
    return fa_text if lang == "fa" else en_text

# --- Sidebar ---
st.sidebar.title(T["settings"])
asset_name = st.sidebar.selectbox(
    T["select_asset"],
    ["GC=F", "SI=F"],
    format_func=lambda x: T["gold"] if x == "GC=F" else T["silver"],
)
timeframe = st.sidebar.selectbox(T["timeframe"], ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)

st.sidebar.markdown("---")
st.sidebar.subheader(T["risk_mgmt"])
acc_balance = st.sidebar.number_input(T["balance"], value=1000)
risk_pct = st.sidebar.slider(T["risk_pct"], 0.5, 5.0, 2.0)
atr_mult = st.sidebar.slider(T["atr_mult"], 1.0, 4.0, 2.0, 0.5)
rr_ratio = st.sidebar.slider(T["rr"], 1.0, 5.0, 2.0, 0.5)

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
refresh_sec = st.sidebar.slider(T["refresh_interval"], 5, 120, 30, 5)
live_window = st.sidebar.slider(T["live_window"], 50, 400, 150, 10)
chart_mode = st.sidebar.selectbox(T["chart_mode"], [T["chart_plotly"], T["chart_tv"]], index=1)

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

if not df.empty:
    df = calculate_patterns(df)

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

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

    curr_price = float(close.iloc[-1])
    curr_rsi = float(rsi.iloc[-1])
    curr_atr = float(atr.iloc[-1])
    curr_adx = safe_last(adx)
    curr_macd_hist = safe_last(macd_hist)

    ema50_slope = float(ema50.iloc[-1] - ema50.iloc[-2]) if len(ema50.dropna()) > 2 else 0.0
    obv_slope = float(obv.iloc[-1] - obv.iloc[-2]) if len(obv.dropna()) > 2 else 0.0

    # Higher timeframe confirmation
    df_higher, higher_tf = get_higher_timeframe(asset_name, timeframe)
    ht_trend = None
    if not df_higher.empty and "Close" in df_higher.columns:
        ht_close = df_higher["Close"].squeeze()
        ht_ema50 = EMAIndicator(ht_close, window=50).ema_indicator()
        if len(ht_ema50.dropna()) > 0:
            ht_trend = "UP" if ht_close.iloc[-1] > ht_ema50.iloc[-1] else "DOWN"

    # Correlation with DXY
    correlation = 0.0
    curr_dxy = 0.0
    if not dxy.empty and "Close" in dxy.columns:
        curr_dxy = float(dxy["Close"].iloc[-1])
        common_idx = df.index.intersection(dxy.index)
        if len(common_idx) > 3:
            correlation = float(df.loc[common_idx]["Close"].corr(dxy.loc[common_idx]["Close"]))

    # --- Multi-factor Signal Engine ---
    long_pts = 0.0
    short_pts = 0.0
    bullish_reasons = []
    bearish_reasons = []

    # 1) Trend structure
    if safe_last(ema20) > safe_last(ema50) > safe_last(ema200):
        long_pts += 18
        bullish_reasons.append(tr("Bull trend stack: EMA20 > EMA50 > EMA200", "انباشت روند صعودی: EMA20 > EMA50 > EMA200"))
    elif safe_last(ema20) < safe_last(ema50) < safe_last(ema200):
        short_pts += 18
        bearish_reasons.append(tr("Bear trend stack: EMA20 < EMA50 < EMA200", "انباشت روند نزولی: EMA20 < EMA50 < EMA200"))

    if ema50_slope > 0:
        long_pts += 8
        bullish_reasons.append(tr("EMA50 slope is positive", "شیب EMA50 مثبت است"))
    else:
        short_pts += 8
        bearish_reasons.append(tr("EMA50 slope is negative", "شیب EMA50 منفی است"))

    if curr_adx >= 25:
        if curr_price > safe_last(ema50):
            long_pts += 8
            bullish_reasons.append(tr(f"ADX {curr_adx:.1f}: trend strength supports upside", f"ADX {curr_adx:.1f}: قدرت روند از صعود حمایت می‌کند"))
        else:
            short_pts += 8
            bearish_reasons.append(tr(f"ADX {curr_adx:.1f}: trend strength supports downside", f"ADX {curr_adx:.1f}: قدرت روند از نزول حمایت می‌کند"))

    # 2) Momentum
    if curr_rsi > 55:
        long_pts += 8
        bullish_reasons.append(tr("RSI regime > 55", "RSI در محدوده صعودی > 55"))
    elif curr_rsi < 45:
        short_pts += 8
        bearish_reasons.append(tr("RSI regime < 45", "RSI در محدوده نزولی < 45"))

    if curr_macd_hist > 0:
        long_pts += 8
        bullish_reasons.append(tr("MACD histogram positive", "هیستوگرام MACD مثبت است"))
    else:
        short_pts += 8
        bearish_reasons.append(tr("MACD histogram negative", "هیستوگرام MACD منفی است"))

    # 3) Volatility + structure
    rolling_high = safe_last(high.rolling(20).max(), default=curr_price)
    rolling_low = safe_last(low.rolling(20).min(), default=curr_price)
    if curr_price >= rolling_high:
        long_pts += 10
        bullish_reasons.append(tr("20-bar breakout to upside", "شکست مقاومت 20 کندل به سمت بالا"))
    elif curr_price <= rolling_low:
        short_pts += 10
        bearish_reasons.append(tr("20-bar breakout to downside", "شکست حمایت 20 کندل به سمت پایین"))

    bb_mid = safe_last(bb.bollinger_mavg(), default=curr_price)
    if curr_price > bb_mid:
        long_pts += 4
        bullish_reasons.append(tr("Price above Bollinger midline", "قیمت بالاتر از خط میانی بولینگر"))
    else:
        short_pts += 4
        bearish_reasons.append(tr("Price below Bollinger midline", "قیمت پایین‌تر از خط میانی بولینگر"))

    # 4) Flow / participation
    if obv_slope > 0:
        long_pts += 6
        bullish_reasons.append(tr("OBV rising (buy-side flow)", "افزایش OBV (جریان خرید)"))
    else:
        short_pts += 6
        bearish_reasons.append(tr("OBV falling (sell-side flow)", "کاهش OBV (جریان فروش)"))

    # 5) Intermarket dependencies (gold drivers)
    dxy_ret = pct_change_n(dxy["Close"].squeeze(), 5) if not dxy.empty and "Close" in dxy.columns else 0.0
    us10y_df = market_ctx.get("us10y", pd.DataFrame())
    us10y_ret = pct_change_n(us10y_df["Close"].squeeze(), 5) if not us10y_df.empty and "Close" in us10y_df.columns else 0.0
    silver_df = market_ctx.get("silver", pd.DataFrame())
    silver_ret = pct_change_n(silver_df["Close"].squeeze(), 5) if not silver_df.empty and "Close" in silver_df.columns else 0.0
    copper_df = market_ctx.get("copper", pd.DataFrame())
    copper_ret = pct_change_n(copper_df["Close"].squeeze(), 5) if not copper_df.empty and "Close" in copper_df.columns else 0.0

    if dxy_ret < 0:
        long_pts += 8
        bullish_reasons.append(tr(f"DXY weakening ({dxy_ret:.2f}% / 5 bars)", f"تضعیف دلار ({dxy_ret:.2f}% / 5 کندل)"))
    elif dxy_ret > 0:
        short_pts += 8
        bearish_reasons.append(tr(f"DXY strengthening ({dxy_ret:.2f}% / 5 bars)", f"تقویت دلار ({dxy_ret:.2f}% / 5 کندل)"))

    if us10y_ret < 0:
        long_pts += 7
        bullish_reasons.append(tr(f"US10Y yield falling ({us10y_ret:.2f}% / 5 bars)", f"کاهش بازده 10Y ({us10y_ret:.2f}% / 5 کندل)"))
    elif us10y_ret > 0:
        short_pts += 7
        bearish_reasons.append(tr(f"US10Y yield rising ({us10y_ret:.2f}% / 5 bars)", f"افزایش بازده 10Y ({us10y_ret:.2f}% / 5 کندل)"))

    if silver_ret > 0:
        long_pts += 5
        bullish_reasons.append(tr(f"Silver confirms metals strength ({silver_ret:.2f}%)", f"نقره قدرت فلزات را تأیید می‌کند ({silver_ret:.2f}%)"))
    elif silver_ret < 0:
        short_pts += 5
        bearish_reasons.append(tr(f"Silver confirms metals weakness ({silver_ret:.2f}%)", f"نقره ضعف فلزات را تأیید می‌کند ({silver_ret:.2f}%)"))

    if copper_ret > 0:
        long_pts += 2
        bullish_reasons.append(tr("Copper risk-on support", "مس از ریسک‌پذیری حمایت می‌کند"))
    elif copper_ret < 0:
        short_pts += 2
        bearish_reasons.append(tr("Copper risk-off pressure", "مس تحت فشار ریسک‌گریزی است"))

    # 6) Higher timeframe confirmation
    if ht_trend == "UP":
        long_pts += 8
        bullish_reasons.append(tr(f"Higher TF ({higher_tf}) trend is up", f"روند تایم‌فریم بالاتر ({higher_tf}) صعودی است"))
    elif ht_trend == "DOWN":
        short_pts += 8
        bearish_reasons.append(tr(f"Higher TF ({higher_tf}) trend is down", f"روند تایم‌فریم بالاتر ({higher_tf}) نزولی است"))

    net_score = long_pts - short_pts
    bias_score = max(-100.0, min(100.0, net_score))
    confidence = min(99.0, abs(bias_score) * 0.85 + (5 if curr_adx >= 25 else 0))

    signal = "NEUTRAL"
    if bias_score >= 35:
        signal = "STRONG BUY"
    elif bias_score >= 12:
        signal = "BUY"
    elif bias_score <= -35:
        signal = "STRONG SELL"
    elif bias_score <= -12:
        signal = "SELL"

    # --- Risk Management ---
    is_long = signal in ["BUY", "STRONG BUY"]
    sl = curr_price - (atr_mult * curr_atr) if is_long else curr_price + (atr_mult * curr_atr)
    tp = curr_price + (rr_ratio * atr_mult * curr_atr) if is_long else curr_price - (rr_ratio * atr_mult * curr_atr)
    tp2 = curr_price + ((rr_ratio + 1.0) * atr_mult * curr_atr) if is_long else curr_price - ((rr_ratio + 1.0) * atr_mult * curr_atr)
    entry_low = curr_price - (0.25 * curr_atr)
    entry_high = curr_price + (0.25 * curr_atr)

    risk_amt = acc_balance * (risk_pct / 100.0)
    risk_per_unit = abs(curr_price - sl) * contract_size
    lot_size = (risk_amt / risk_per_unit) if risk_per_unit != 0 else 0

    # --- Per-method signal box (technical + fundamental) ---
    prev_high_20 = safe_last(high.shift(1).rolling(20).max(), default=curr_price)
    prev_low_20 = safe_last(low.shift(1).rolling(20).min(), default=curr_price)

    pa_sig = "NEUTRAL"
    pa_reason = tr("Range/no clear breakout", "محدوده/سیگنال واضحی وجود ندارد")
    if curr_price > prev_high_20 and curr_price > safe_last(ema50):
        pa_sig = "BUY"
        pa_reason = tr("Breakout above previous 20-bar high", "شکست مقاومت 20 کندل قبلی")
    elif curr_price < prev_low_20 and curr_price < safe_last(ema50):
        pa_sig = "SELL"
        pa_reason = tr("Breakdown below previous 20-bar low", "شکست حمایت 20 کندل قبلی")

    fib_sig = "NEUTRAL"
    fib_reason = tr("Price away from key retracement zone", "قیمت در محدوده بازگشت فیبوناچی نیست")
    swing_high = safe_last(high.rolling(120).max(), default=curr_price)
    swing_low = safe_last(low.rolling(120).min(), default=curr_price)
    fib_range = swing_high - swing_low
    if fib_range > 0:
        fib_50 = swing_high - fib_range * 0.5
        fib_618 = swing_high - fib_range * 0.618
        if safe_last(ema50) > safe_last(ema200) and fib_618 <= curr_price <= fib_50:
            fib_sig = "BUY"
            fib_reason = tr("Bull trend pullback in 0.5-0.618 zone", "پولبک روند صعودی در محدوده 0.5-0.618")
        elif safe_last(ema50) < safe_last(ema200) and fib_50 <= curr_price <= fib_618:
            fib_sig = "SELL"
            fib_reason = tr("Bear trend pullback in 0.5-0.618 zone", "پولبک روند نزولی در محدوده 0.5-0.618")

    rsi_sig = "NEUTRAL"
    rsi_reason = tr("RSI between 45 and 55", "RSI بین 45 و 55")
    if curr_rsi > 55:
        rsi_sig = "BUY"
        rsi_reason = tr("RSI bullish regime", "RSI در محدوده صعودی")
    elif curr_rsi < 45:
        rsi_sig = "SELL"
        rsi_reason = tr("RSI bearish regime", "RSI در محدوده نزولی")

    macd_sig = "NEUTRAL"
    macd_reason = tr("No fresh MACD impulse", "سیگنال جدیدی از MACD وجود ندارد")
    if len(macd_hist.dropna()) > 2:
        if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
            macd_sig = "BUY"
            macd_reason = tr("MACD histogram crossed above zero", "هیستوگرام MACD بالای صفر عبور کرد")
        elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
            macd_sig = "SELL"
            macd_reason = tr("MACD histogram crossed below zero", "هیستوگرام MACD زیر صفر عبور کرد")

    bb_sig = "NEUTRAL"
    bb_reason = tr("Price around Bollinger mid area", "قیمت در محدوده میانی بولینگر")
    bb_high = safe_last(bb.bollinger_hband(), default=curr_price)
    bb_low = safe_last(bb.bollinger_lband(), default=curr_price)
    if curr_price < bb_low and curr_rsi < 35:
        bb_sig = "BUY"
        bb_reason = tr("Lower band overshoot with low RSI", "اشباع فروش در کف باند پایین")
    elif curr_price > bb_high and curr_rsi > 65:
        bb_sig = "SELL"
        bb_reason = tr("Upper band overshoot with high RSI", "اشباع خرید در سقف باند بالا")

    fund_score = 0
    if dxy_ret < 0:
        fund_score += 1
    elif dxy_ret > 0:
        fund_score -= 1
    if us10y_ret < 0:
        fund_score += 1
    elif us10y_ret > 0:
        fund_score -= 1
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
    fund_reason = tr(f"DXY:{dxy_ret:.2f}% | 10Y:{us10y_ret:.2f}% | Silver:{silver_ret:.2f}% | Copper:{copper_ret:.2f}%", 
                       f"دلار:{dxy_ret:.2f}% | 10Y:{us10y_ret:.2f}% | نقره:{silver_ret:.2f}% | مس:{copper_ret:.2f}%")

    method_signals = [
        ("price_action", T["method_price_action"], pa_sig, pa_reason),
        ("fib", T["method_fib"], fib_sig, fib_reason),
        ("rsi", T["method_rsi"], rsi_sig, rsi_reason),
        ("macd", T["method_macd"], macd_sig, macd_reason),
        ("bollinger", T["method_bollinger"], bb_sig, bb_reason),
        ("fundamental", T["method_fundamental"], fund_sig, fund_reason),
    ]

    # --- UI Layout ---
    st.title(T["title"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(T["price"], f"${curr_price:,.2f}")
    c2.metric(T["rsi"], f"{curr_rsi:.2f}")
    c3.metric(T["atr"], f"{curr_atr:.2f}")
    c4.metric(T["dxy"], f"{curr_dxy:.2f}")
    c5.metric(T["confidence"], f"{confidence:.0f}%")

    signal_display = signal
    if signal == "STRONG BUY":
        signal_display = T["sig_strong_buy"]
    elif signal == "STRONG SELL":
        signal_display = T["sig_strong_sell"]
    elif signal == "BUY":
        signal_display = T["sig_buy"]
    elif signal == "SELL":
        signal_display = T["sig_sell"]
    else:
        signal_display = T["sig_neutral"]

    c6, c7, c8 = st.columns([1.2, 1, 1])
    with c6:
        if signal in ["BUY", "STRONG BUY"]:
            st.markdown(f"<div class='signal-buy'><h3>{signal_display}</h3><p>{T['bias_score']}: {bias_score:.1f}</p></div>", unsafe_allow_html=True)
        elif signal in ["SELL", "STRONG SELL"]:
            st.markdown(f"<div class='signal-sell'><h3>{signal_display}</h3><p>{T['bias_score']}: {bias_score:.1f}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='signal-neutral'><h3>{T['signal_wait']}</h3><p>{T['bias_score']}: {bias_score:.1f}</p></div>", unsafe_allow_html=True)

    with c7:
        st.subheader(T["risk"])
        st.write(f"{T['risk_amount']}: ${risk_amt:.2f}")
        st.write(f"{T['lot_size']}: {lot_size:.3f}")
        st.write(f"{T['entry_zone']}: {entry_low:,.2f} - {entry_high:,.2f}")
        st.write(f"{T['adx_macd']}: {curr_adx:.1f} / {curr_macd_hist:.3f}")

    with c8:
        st.subheader(T["targets"])
        st.write(f"{T['tp']}: {tp:,.2f}")
        st.write(f"{T['tp2']}: {tp2:,.2f}")
        st.write(f"{T['sl']}: {sl:,.2f}")
        st.write(T["rr_fmt"].format(rr=rr_ratio))
        st.write(f"{T['corr_dxy']}: {correlation:.2f}")

    sig_text_map = {"BUY": T["sig_buy"], "SELL": T["sig_sell"], "NEUTRAL": T["sig_neutral"]}
    method_rows = []
    method_conf_map = {
        "price_action": min(95.0, 45.0 + abs(curr_price - safe_last(ema50)) / max(curr_atr, 1e-9) * 8.0),
        "fib": min(90.0, 40.0 + (8.0 if fib_sig != "NEUTRAL" else 0.0) + abs(curr_price - (swing_high + swing_low) / 2) / max(curr_atr, 1e-9) * 3.0),
        "rsi": min(88.0, 35.0 + abs(curr_rsi - 50.0) * 1.2),
        "macd": min(90.0, 35.0 + abs(curr_macd_hist) * 180.0),
        "bollinger": min(85.0, 35.0 + abs(curr_price - bb_mid) / max(curr_atr, 1e-9) * 12.0),
        "fundamental": min(92.0, 35.0 + abs(fund_score) * 14.0),
    }
    for method_code, method_name, method_sig, method_reason in method_signals:
        method_conf = method_conf_map.get(method_code, 50.0)
        method_rows.append(
            {
                "Method": method_name,
                "_sig_code": method_sig,
                T["method_signal_col"]: sig_text_map.get(method_sig, method_sig),
                T["method_conf_col"]: f"{method_conf:.0f}%",
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

    show_df = method_df.drop(columns=["_sig_code", "_sig_rank", "_conf_sort"])
    styled_df = show_df.style.apply(row_style, axis=1)
    st.markdown(f"<div class='app-card'><h4 style='margin:0;'>{T['method_title']}</h4></div>", unsafe_allow_html=True)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # --- Chart ---
    if chart_mode == T["chart_tv"]:
        tv_symbol_map = {
            "GC=F": "OANDA:XAUUSD",
            "SI=F": "OANDA:XAGUSD",
        }
        tv_interval_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "D",
        }
        tv_symbol = tv_symbol_map.get(asset_name, "OANDA:XAUUSD")
        tv_interval = tv_interval_map.get(timeframe, "60")
        tv_theme = "dark"
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
              "timezone": "Etc/UTC",
              "theme": "{tv_theme}",
              "style": "1",
              "locale": "en",
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
        if signal in ["BUY", "STRONG BUY", "SELL", "STRONG SELL"]:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[-1]],
                    y=[curr_price],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="lime" if signal in ["BUY", "STRONG BUY"] else "red",
                        symbol="triangle-up" if signal in ["BUY", "STRONG BUY"] else "triangle-down",
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
        st.plotly_chart(fig, use_container_width=True)

    with st.expander(T["logic"]):
        st.write(f"{T['bullish_factors']}:")
        for reason in bullish_reasons[:8]:
            st.write(f"- {reason}")
        st.write(f"{T['bearish_factors']}:")
        for reason in bearish_reasons[:8]:
            st.write(f"- {reason}")
        st.write(f"- {tr('Higher TF', 'تایم‌فریم بالاتر')}: {higher_tf} | {T['trend']}: {ht_trend}")
        st.write(f"- {tr('DXY 5-bar return', 'بازده دلار 5 کندل')}: {dxy_ret:.2f}% | {tr('US10Y 5-bar return', 'بازده 10Y 5 کندل')}: {us10y_ret:.2f}%")
        st.write(f"- {tr('Silver 5-bar return', 'بازده نقره 5 کندل')}: {silver_ret:.2f}% | {tr('Copper 5-bar return', 'بازده مس 5 کندل')}: {copper_ret:.2f}%")

    st.caption(f"{T['last_update']}: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

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

