import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, ADXIndicator, MACD
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from streamlit_autorefresh import st_autorefresh

# --- Page Config ---
st.set_page_config(page_title="Live Gold Trading Terminal", layout="wide")

# --- Modern CSS ---
st.markdown("""
<style>
    :root {
        --bg: #0a0f1e;
        --panel: #121a2b;
        --panel-2: #172238;
        --line: #263551;
        --text: #f4f8ff;
        --buy: #21c77a;
        --sell: #ff5a7a;
        --neutral: #8a96ad;
    }
    
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a3244 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid var(--line);
        margin-bottom: 20px;
        text-align: center;
    }
    
    .live-price-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a3244 100%);
        padding: 30px;
        border-radius: 20px;
        border: 2px solid var(--buy);
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .signal-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a3244 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid var(--line);
        margin: 20px 0;
    }
    
    .price-display {
        font-size: 48px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 15px 0;
    }
    
    .signal-display {
        font-size: 24px;
        font-weight: bold;
        padding: 15px 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin: 10px 0;
    }
    
    .analysis-box {
        background: rgba(0,0,0,0.3);
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
    }
    
    .chart-container {
        background: var(--panel);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid var(--line);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- GoldAPI Function ---
def get_gold_price_from_api() -> tuple[float | None, float | None]:
    """Get live gold price from GoldAPI with proper error handling."""
    try:
        api_key = st.secrets.get("GOLD_API_KEY", "")
        if not api_key:
            st.warning("⚠️ GoldAPI key not found in secrets. Please add GOLD_API_KEY to your secrets.")
            return None, None
            
        headers = {
            "x-access-token": api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://www.goldapi.io/api/XAU/USD",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "price" in data:
                current_price = float(data["price"])
                previous_price = float(data.get("prev_day_price", current_price))
                price_change = current_price - previous_price
                return current_price, price_change
            else:
                st.error("❌ Invalid response format from GoldAPI")
                return None, None
        else:
            st.error(f"❌ GoldAPI error: {response.status_code}")
            return None, None
            
    except requests.exceptions.Timeout:
        st.error("❌ GoldAPI request timeout")
        return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ GoldAPI request error: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Unexpected error fetching gold price: {e}")
        return None, None

# --- Safe Data Functions ---
def safe_yf_download(ticker: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    """Download market data with retry."""
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.loc[:, ~df.columns.duplicated()].copy()
                return df
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1 + attempt)
    return pd.DataFrame()

def safe_last(series: pd.Series, default: float = 0.0) -> float:
    """Safely get last value from Series."""
    valid = series.dropna()
    if valid.empty:
        return default
    return float(valid.iloc[-1])

# --- Signal Generation ---
def generate_signals_with_live_data(
    curr_price: float, close: pd.Series, high: pd.Series, low: pd.Series,
    rsi: pd.Series, ema50: pd.Series, ema200: pd.Series, ema20: pd.Series,
    macd_line_series: pd.Series, macd_signal_series: pd.Series, macd_hist_series: pd.Series,
    market_structure: dict
) -> dict:
    """Generate trading signals based on live price data."""
    
    # Price Action Signal
    price_action_signal = "NEUTRAL"
    price_action_reason = "Price action neutral"
    
    if curr_price > market_structure["resistance"]:
        price_action_signal = "BUY"
        price_action_reason = f"Price breaks resistance at {market_structure['resistance']:.2f}"
    elif curr_price < market_structure["support"]:
        price_action_signal = "SELL"
        price_action_reason = f"Price breaks support at {market_structure['support']:.2f}"
    
    # Momentum Signal (RSI)
    momentum_signal = "NEUTRAL"
    momentum_reason = "RSI neutral"
    
    if len(rsi.dropna()) > 0:
        curr_rsi = float(rsi.iloc[-1])
        if curr_rsi <= 30:
            momentum_signal = "BUY"
            momentum_reason = f"RSI oversold at {curr_rsi:.1f}"
        elif curr_rsi >= 70:
            momentum_signal = "SELL"
            momentum_reason = f"RSI overbought at {curr_rsi:.1f}"
        elif 50 <= curr_rsi <= 60:
            momentum_signal = "BUY"
            momentum_reason = f"RSI bullish at {curr_rsi:.1f}"
        elif 40 <= curr_rsi <= 50:
            momentum_signal = "SELL"
            momentum_reason = f"RSI bearish at {curr_rsi:.1f}"
    
    # Trend Signal (EMA)
    trend_signal = "NEUTRAL"
    trend_reason = "EMA trend neutral"
    
    if len(ema20.dropna()) > 0 and len(ema50.dropna()) > 0 and len(ema200.dropna()) > 0:
        curr_ema20 = float(ema20.iloc[-1])
        curr_ema50 = float(ema50.iloc[-1])
        curr_ema200 = float(ema200.iloc[-1])
        
        if curr_price > curr_ema20 > curr_ema50 > curr_ema200:
            trend_signal = "BUY"
            trend_reason = "Strong uptrend - all EMAs aligned"
        elif curr_price < curr_ema20 < curr_ema50 < curr_ema200:
            trend_signal = "SELL"
            trend_reason = "Strong downtrend - all EMAs aligned"
        elif curr_price > curr_ema200:
            trend_signal = "BUY"
            trend_reason = "Above EMA200 - bullish bias"
        elif curr_price < curr_ema200:
            trend_signal = "SELL"
            trend_reason = "Below EMA200 - bearish bias"
    
    # MACD Signal
    macd_signal = "NEUTRAL"
    macd_reason = "MACD neutral"
    
    if len(macd_hist_series.dropna()) > 0 and len(macd_line_series.dropna()) > 0 and len(macd_signal_series.dropna()) > 0:
        curr_macd_hist = float(macd_hist_series.iloc[-1])
        curr_macd_line = float(macd_line_series.iloc[-1])
        curr_macd_signal = float(macd_signal_series.iloc[-1])
        
        if curr_macd_hist > 0 and curr_macd_line > curr_macd_signal:
            macd_signal = "BUY"
            macd_reason = "MACD bullish crossover"
        elif curr_macd_hist < 0 and curr_macd_line < curr_macd_signal:
            macd_signal = "SELL"
            macd_reason = "MACD bearish crossover"
    
    # Overall Signal (weighted combination)
    signals = [price_action_signal, momentum_signal, trend_signal, macd_signal]
    buy_votes = signals.count("BUY")
    sell_votes = signals.count("SELL")
    
    if buy_votes >= 3:
        overall_signal = "BUY"
        confidence = 0.75 + (buy_votes - 3) * 0.1
    elif sell_votes >= 3:
        overall_signal = "SELL"
        confidence = 0.75 + (sell_votes - 3) * 0.1
    elif buy_votes > sell_votes:
        overall_signal = "BUY"
        confidence = 0.55 + (buy_votes - sell_votes) * 0.1
    elif sell_votes > buy_votes:
        overall_signal = "SELL"
        confidence = 0.55 + (sell_votes - buy_votes) * 0.1
    else:
        overall_signal = "NEUTRAL"
        confidence = 0.5
    
    confidence = min(confidence, 0.95)
    
    overall_reason = f"Price: {price_action_reason} | Momentum: {momentum_reason} | Trend: {trend_reason} | MACD: {macd_reason}"
    
    return {
        "overall": overall_signal,
        "confidence": confidence,
        "price_action": price_action_signal,
        "momentum": momentum_signal,
        "trend": trend_signal,
        "macd": macd_signal,
        "overall_reason": overall_reason
    }

# --- Market Structure ---
def get_market_structure(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 50) -> dict:
    """Calculate market structure with safety checks."""
    if len(high) < lookback or len(low) < lookback or len(close) < lookback:
        return {
            "support": 0.0,
            "resistance": 0.0,
            "signal": "NEUTRAL",
            "reason": "insufficient data"
        }
    
    window_high = high.iloc[-lookback:]
    window_low = low.iloc[-lookback:]
    current = float(close.iloc[-1]) if len(close) > 0 else 0.0
    
    support = float(window_low.quantile(0.15))
    resistance = float(window_high.quantile(0.85))
    
    if current > resistance:
        return {
            "support": support,
            "resistance": resistance,
            "signal": "BUY",
            "reason": f"Price above resistance at {resistance:.2f}"
        }
    elif current < support:
        return {
            "support": support,
            "resistance": resistance,
            "signal": "SELL",
            "reason": f"Price below support at {support:.2f}"
        }
    else:
        return {
            "support": support,
            "resistance": resistance,
            "signal": "NEUTRAL",
            "reason": f"Price between support {support:.2f} and resistance {resistance:.2f}"
        }

# --- Chart Function ---
def create_chart(df: pd.DataFrame, signals: dict) -> go.Figure:
    """Create modern trading chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price', 'RSI', 'MACD')
    )
    
    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#21c77a',
            decreasing_line_color='#ff5a7a'
        ),
        row=1, col=1
    )
    
    # EMAs
    if len(df) > 50:
        ema20 = EMAIndicator(df['Close'], window=20).ema_indicator()
        ema50 = EMAIndicator(df['Close'], window=50).ema_indicator()
        ema200 = EMAIndicator(df['Close'], window=200).ema_indicator()
        
        fig.add_trace(go.Scatter(x=df.index, y=ema20, name='EMA20', line=dict(color='#ffa500', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema50, name='EMA50', line=dict(color='#00bfff', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema200, name='EMA200', line=dict(color='#ff69b4', width=2)), row=1, col=1)
    
    # RSI
    if len(df) > 14:
        rsi = RSIIndicator(df['Close']).rsi()
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#9370db')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if len(df) > 26:
        macd = MACD(df['Close'])
        fig.add_trace(go.Scatter(x=df.index, y=macd.macd(), name='MACD', line=dict(color='#32cd32')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd.macd_signal(), name='Signal', line=dict(color='#ff4500')), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=macd.macd_diff(), name='Histogram', marker_color='gray'), row=3, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=800,
        showlegend=True,
        title={
            'text': f'Gold Price Chart - Signal: {signals["overall"]} (Confidence: {signals["confidence"]:.1%})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        }
    )
    
    return fig

# --- Main App ---
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🥇 Live Gold Trading Terminal</h1>
        <p>100% Real-time Gold Price & Trading Signals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh every 10 seconds
    st_autorefresh(interval=10000, limit=None, key="price_refresh")
    
    # Get live gold price
    gold_price, price_change = get_gold_price_from_api()
    
    if gold_price is None:
        st.error("❌ Unable to fetch live gold price. Please check your GoldAPI key.")
        return
    
    # Get historical data for analysis
    df = safe_yf_download("GC=F", period="3mo", interval="1h")
    
    if df.empty:
        st.error("❌ Unable to fetch historical data for analysis.")
        return
    
    # Ensure we have enough data
    if len(df) < 50:
        st.error("❌ Not enough data for analysis. Please wait for more data.")
        return
    
    # Update last candle with live price
    last_idx = df.index[-1]
    df.at[last_idx, "Close"] = gold_price
    df.at[last_idx, "High"] = max(df.at[last_idx, "High"], gold_price)
    df.at[last_idx, "Low"] = min(df.at[last_idx, "Low"], gold_price)
    
    # Calculate indicators (keep as Series)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    
    rsi = RSIIndicator(close).rsi()
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    
    macd_line_series = MACD(close).macd()
    macd_signal_series = MACD(close).macd_signal()
    macd_hist_series = MACD(close).macd_diff()
    
    # Get market structure
    market_structure = get_market_structure(high, low, close)
    
    # Generate signals
    signals = generate_signals_with_live_data(
        gold_price, close, high, low, rsi, ema50, ema200, ema20,
        macd_line_series, macd_signal_series, macd_hist_series, market_structure
    )
    
    # Display live price
    price_color = "#21c77a" if price_change >= 0 else "#ff5a7a"
    price_arrow = "📈" if price_change >= 0 else "📉"
    
    st.markdown(f"""
    <div class="live-price-box" style="border-color: {price_color};">
        <h2 style="color: #8a96ad; margin-bottom: 20px;">🥇 LIVE GOLD PRICE (100% REAL-TIME)</h2>
        <div class="price-display" style="color: {price_color};">
            {price_arrow} ${gold_price:,.2f}
        </div>
        <div style="font-size: 18px; color: {price_color}; margin-bottom: 15px;">
            Change: {price_change:+.2f} ({(price_change/gold_price*100):+.2f}%)
        </div>
        <div style="font-size: 14px; color: #ffd700;">
            📊 Data Source: GoldAPI | 🔄 Auto-refresh: 10s
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display signals
    signal_color = "#21c77a" if signals["overall"] == "BUY" else "#ff5a7a" if signals["overall"] == "SELL" else "#8a96ad"
    signal_emoji = "🟢" if signals["overall"] == "BUY" else "🔴" if signals["overall"] == "SELL" else "🟡"
    
    st.markdown(f"""
    <div class="signal-box">
        <h3 style="color: #8a96ad; margin-bottom: 15px;">📊 TRADING SIGNALS</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <div style="font-size: 16px; color: #8a96ad; margin-bottom: 5px;">Overall Signal</div>
                <div class="signal-display" style="background: {signal_color}; color: white;">
                    {signal_emoji} {signals["overall"]}
                </div>
                <div style="font-size: 14px; color: #8a96ad; margin-top: 5px;">
                    Confidence: {signals["confidence"]:.1%}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 14px; color: #8a96ad; margin-bottom: 5px;">Individual Signals</div>
                <div style="font-size: 12px; color: #f4f8ff;">
                    Price Action: {signals["price_action"]}<br>
                    Momentum (RSI): {signals["momentum"]}<br>
                    Trend (EMA): {signals["trend"]}<br>
                    MACD: {signals["macd"]}
                </div>
            </div>
        </div>
        <div class="analysis-box">
            <div style="color: #ffd700; font-weight: bold; margin-bottom: 10px;">📈 Analysis Details:</div>
            <div style="color: #8a96ad; font-size: 14px; line-height: 1.6;">
                {signals["overall_reason"]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chart
    fig = create_chart(df, signals)
    st.plotly_chart(fig, use_container_width=True)
    
    # Market structure info
    st.markdown(f"""
    <div class="signal-box">
        <h3 style="color: #8a96ad; margin-bottom: 15px;">🏗️ MARKET STRUCTURE</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <div style="font-size: 14px; color: #8a96ad; margin-bottom: 5px;">Support Level</div>
                <div style="font-size: 18px; font-weight: bold; color: #21c77a;">
                    ${market_structure["support"]:.2f}
                </div>
            </div>
            <div>
                <div style="font-size: 14px; color: #8a96ad; margin-bottom: 5px;">Resistance Level</div>
                <div style="font-size: 18px; font-weight: bold; color: #ff5a7a;">
                    ${market_structure["resistance"]:.2f}
                </div>
            </div>
        </div>
        <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px;">
            <div style="color: #ffd700; font-weight: bold; margin-bottom: 5px;">Structure Analysis:</div>
            <div style="color: #8a96ad; font-size: 14px;">
                {market_structure["reason"]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
