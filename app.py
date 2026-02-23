import io
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from templates import live_price_html, mini_box_close_html, mini_box_open_html, top_banner_html


st.set_page_config(page_title="Gold Pro Terminal", layout="wide")

def load_css(path: str = "styles.css") -> None:
    css_path = Path(path)
    if not css_path.exists():
        st.warning(f"CSS file not found: {css_path}")
        return
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()

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
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.loc[:, ~df.columns.duplicated()].copy()
    except Exception:
        pass
    return pd.DataFrame()


def get_goldapi_live() -> tuple[float | None, float | None, str]:
    try:
        api_key = st.secrets["GOLD_API_KEY"]
    except Exception:
        return None, None, "GoldAPI key not found"

    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.get("https://www.goldapi.io/api/XAU/USD", headers=headers, timeout=10)
        if resp.status_code != 200:
            return None, None, f"GoldAPI status={resp.status_code}"
        payload = resp.json()
        price = float(payload.get("price"))
        prev = float(payload.get("prev_day_price", price))
        return price, price - prev, "GoldAPI"
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
    return curr, curr - prev, "yfinance backup"


def get_live_gold_price() -> tuple[float | None, float | None, str, str]:
    p, chg, src = get_goldapi_live()
    if p is not None:
        return p, chg, src, "ok"

    bp, bchg, bsrc = get_yf_live_backup()
    if bp is not None:
        return bp, bchg, bsrc, src

    return None, None, "none", f"{src} | {bsrc}"


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


def aggregate_signal(classic: dict, porsamadi: dict) -> tuple[str, float, dict]:
    engines = {
        "Price Action": classic["price_action"],
        "RSI": classic["momentum"],
        "Trend EMA": classic["trend"],
        "MACD": classic["macd"],
        "Poursamadi": porsamadi["signal"],
    }

    buy_votes = sum(1 for v in engines.values() if v == "BUY")
    sell_votes = sum(1 for v in engines.values() if v == "SELL")

    if buy_votes >= 4:
        sig = "STRONG BUY"
        conf = 0.88
    elif sell_votes >= 4:
        sig = "STRONG SELL"
        conf = 0.88
    elif buy_votes > sell_votes:
        sig = "BUY"
        conf = 0.62 + 0.05 * (buy_votes - sell_votes)
    elif sell_votes > buy_votes:
        sig = "SELL"
        conf = 0.62 + 0.05 * (sell_votes - buy_votes)
    else:
        sig = "NEUTRAL"
        conf = 0.5

    return sig, min(conf, 0.95), engines


def tone_bytes(freq: int = 880, duration: float = 0.22, sample_rate: int = 22050) -> bytes:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave_data = (0.35 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    pcm = (wave_data * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def render_signal_micro_chart(engine_signals: dict):
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
        template="plotly_dark",
        title="Engine-by-Engine Minimal Signal Chart",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[-1.4, 1.4], tickvals=[-1, 0, 1], ticktext=["SELL", "NEUTRAL", "BUY"]),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_main_chart(df: pd.DataFrame, classic: dict):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("XAUUSD", "RSI", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="#16c784",
            decreasing_line_color="#ff4d6d",
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=df.index, y=classic["ema20"], name="EMA20", line=dict(color="#ffb703", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=classic["ema50"], name="EMA50", line=dict(color="#5bc0eb", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=classic["ema200"], name="EMA200", line=dict(color="#f15bb5", width=1.6)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=classic["rsi"], name="RSI", line=dict(color="#9b5de5")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4d6d", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#16c784", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=classic["macd_line"], name="MACD", line=dict(color="#80ed99")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=classic["macd_signal"], name="Signal", line=dict(color="#f28482")), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=classic["macd_hist"], name="Hist", marker_color="#8ea4c9"), row=3, col=1)

    fig.update_layout(template="plotly_dark", height=820, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_fundamental_box() -> None:
    dxy = safe_download("DX-Y.NYB", period="1mo", interval="1h")
    silver = safe_download("SI=F", period="3mo", interval="1d")
    oil = safe_download("CL=F", period="3mo", interval="1d")
    gold = safe_download("GC=F", period="3mo", interval="1d")

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

    st.markdown(mini_box_open_html(), unsafe_allow_html=True)
    st.markdown("### Fundamental Overview")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("DXY", f"{dxy_last:.2f}" if dxy_last is not None else "N/A")

    if fg_value is not None:
        c2.metric("Fear & Greed", f"{fg_value:.0f}")
        c2.caption(fg_label)
    else:
        c2.metric("Fear & Greed", "N/A")
        c2.caption("API unavailable")

    c3.metric("Gold-Silver Corr", f"{corr_silver:+.2f}" if corr_silver is not None else "N/A")
    c4.metric("Gold-Oil Corr", f"{corr_oil:+.2f}" if corr_oil is not None else "N/A")

    st.caption("Correlation is based on daily returns over recent 3 months.")
    st.markdown(mini_box_close_html(), unsafe_allow_html=True)


def render_top_banner() -> None:
    st.markdown(top_banner_html(), unsafe_allow_html=True)


def render_live_price_box(
    live_price: float,
    price_change: float | None,
    source: str,
    final_signal: str,
    confidence: float,
) -> None:
    st.markdown(
        live_price_html(live_price, price_change, source, final_signal, confidence),
        unsafe_allow_html=True,
    )


def handle_audio_alert(final_signal: str) -> None:
    if "last_audio_signal" not in st.session_state:
        st.session_state.last_audio_signal = None

    if final_signal in {"STRONG BUY", "STRONG SELL"} and final_signal != st.session_state.last_audio_signal:
        if final_signal == "STRONG BUY":
            st.audio(tone_bytes(freq=1040), format="audio/wav", autoplay=True)
        else:
            st.audio(tone_bytes(freq=380), format="audio/wav", autoplay=True)

    st.session_state.last_audio_signal = final_signal


def render_poursamadi_box(porsamadi: dict) -> None:
    st.markdown(mini_box_open_html(), unsafe_allow_html=True)
    st.markdown("### Smart-Money Engine (Poursamadi)")
    st.write(f"Signal: **{porsamadi['signal']}**")
    st.write(f"Reason: {porsamadi['reason']}")
    st.write(f"HTF trend (4H): **{porsamadi['htf_trend']}**")
    st.write(f"HH detected: `{porsamadi['hh']}` | LL detected: `{porsamadi['ll']}`")
    st.write(f"Bullish BOS: `{porsamadi['bullish_bos']}` | Bearish BOS: `{porsamadi['bearish_bos']}`")
    st.write(f"Swing High / Swing Low: {porsamadi['last_swing_high']:.2f} / {porsamadi['last_swing_low']:.2f}")
    if porsamadi["bullish_ob"]:
        st.write(f"Bullish Order Block: {porsamadi['bullish_ob'][0]:.2f} - {porsamadi['bullish_ob'][1]:.2f}")
    if porsamadi["bearish_ob"]:
        st.write(f"Bearish Order Block: {porsamadi['bearish_ob'][0]:.2f} - {porsamadi['bearish_ob'][1]:.2f}")
    st.markdown(mini_box_close_html(), unsafe_allow_html=True)


def render_classic_box(classic: dict) -> None:
    st.markdown(mini_box_open_html(), unsafe_allow_html=True)
    st.markdown("### Classic Engine")
    st.write(f"Price Action: **{classic['price_action']}**")
    st.write(f"RSI: **{classic['momentum']}**")
    st.write(f"Trend (EMA): **{classic['trend']}**")
    st.write(f"MACD: **{classic['macd']}**")
    st.write(f"Bollinger: **{classic['bollinger']}**")
    st.write(f"Support/Resistance: {classic['support']:.2f} / {classic['resistance']:.2f}")
    st.markdown(mini_box_close_html(), unsafe_allow_html=True)


def render_signal_monitor(engine_signals: dict) -> None:
    st.markdown(mini_box_open_html(), unsafe_allow_html=True)
    st.markdown("### Multi-Logic Signal Monitor")
    render_signal_micro_chart(engine_signals)
    st.markdown(mini_box_close_html(), unsafe_allow_html=True)


def render_engine_details_table(engine_signals: dict) -> None:
    d = pd.DataFrame({"Engine": list(engine_signals.keys()), "Signal": list(engine_signals.values())})
    st.dataframe(d, use_container_width=True, hide_index=True)


def main() -> None:
    st_autorefresh(interval=10000, key="gold_refresh_10s")

    render_top_banner()

    live_price, price_change, source, status = get_live_gold_price()
    if live_price is None:
        st.error(f"Live price unavailable. Details: {status}")
        st.stop()

    df = safe_download("GC=F", period="3mo", interval="1h")
    if df.empty or len(df) < 220:
        st.error("Historical data is not sufficient for full analysis.")
        st.stop()

    df_htf = safe_download("GC=F", period="6mo", interval="4h")

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    if not df_htf.empty:
        for col in ["Open", "High", "Low", "Close"]:
            df_htf[col] = pd.to_numeric(df_htf[col], errors="coerce")
        df_htf = df_htf.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    last_idx = df.index[-1]
    df.at[last_idx, "Close"] = float(live_price)
    df.at[last_idx, "High"] = max(float(df.at[last_idx, "High"]), float(live_price))
    df.at[last_idx, "Low"] = min(float(df.at[last_idx, "Low"]), float(live_price))

    classic = compute_classic_signals(df, live_price)
    porsamadi = compute_porsamadi(df, df_htf, live_price)
    final_signal, confidence, engine_signals = aggregate_signal(classic, porsamadi)

    render_live_price_box(live_price, price_change, source, final_signal, confidence)
    handle_audio_alert(final_signal)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        render_poursamadi_box(porsamadi)

    with col_b:
        render_classic_box(classic)

    render_signal_monitor(engine_signals)

    render_fundamental_box()

    tab1, tab2 = st.tabs(["Main Chart", "Engine Details"])
    with tab1:
        render_main_chart(df, classic)

    with tab2:
        render_engine_details_table(engine_signals)


if __name__ == "__main__":
    main()




