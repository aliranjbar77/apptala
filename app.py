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
import requests
import json
from datetime import datetime, timedelta
import re

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
            max-width: 1200px;
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        .app-card {
            background: rgba(18, 26, 43, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 14px 16px;
            margin: 8px 0;
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
        @media (max-width: 900px) {
            .method-grid { grid-template-columns: 1fr; }
            [data-testid="stMetricValue"] { font-size: 1.28rem !important; }
            [data-testid="stMetricLabel"] { font-size: 0.88rem !important; }
            .block-container { padding-top: 0.8rem; padding-left: 0.8rem; padding-right: 0.8rem; }
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
        'count': len(sentiments)
    }

# --- Macro Economic Functions ---
def get_real_yields() -> dict:
    """Calculate real yields (nominal yield - inflation proxy)."""
    try:
        def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df

        tn10y = normalize_cols(yf.download("^TNX", period="1mo", interval="1d"))
        if tn10y.empty or "Close" not in tn10y.columns:
            return {'nominal_yield': 0, 'inflation_rate': 0, 'real_yield': 0, 'trend': 'neutral'}

        current_yield = float(tn10y["Close"].dropna().iloc[-1])

        # CPI ticker in yfinance is unstable; use TIP (inflation-linked ETF) as proxy.
        tip = normalize_cols(yf.download("TIP", period="1y", interval="1mo"))
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
        fed_funds = yf.download("^FVX", period="3mo", interval="1d")  # Fed Funds Futures
        
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
def calculate_smart_position_size(base_lot_size: float, confidence: float, atr: float, 
                                 sentiment_data: dict = None, risk_multiplier: float = 1.0) -> dict:
    """Calculate smart position size based on confidence, volatility, and sentiment"""
    
    # Base confidence adjustment
    confidence_factor = 0.5 + (confidence / 100.0) * 1.5  # Range: 0.5x to 2.0x
    
    # Volatility adjustment (lower volatility = higher position size)
    avg_atr = atr * 2  # Assume average ATR is 2x current ATR
    volatility_factor = min(2.0, max(0.5, avg_atr / max(atr, 0.1)))
    
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
        current_macd_hist = macd.macd_hist().iloc[i]
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
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'final_balance': balance,
        'trades': completed_trades[:10]  # Return last 10 trades for display
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
            asset_df = yf.download(ticker, period="180d", interval="1d")
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def tr(en_text: str, fa_text: str) -> str:
    return fa_text if lang == "fa" else en_text

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

# --- Advanced Correlation Data ---
correlation_data = calculate_advanced_correlations(asset_name)

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

    # 7) Sentiment Analysis (if enabled)
    if sentiment_data:
        if sentiment_data['overall'] == 'bullish':
            long_pts += sentiment_impact_score
            bullish_reasons.append(tr(f"Market sentiment bullish ({sentiment_data['confidence']:.1f})", f"سنتیمنت بازار صعودی ({sentiment_data['confidence']:.1f})"))
        elif sentiment_data['overall'] == 'bearish':
            short_pts += abs(sentiment_impact_score)
            bearish_reasons.append(tr(f"Market sentiment bearish ({sentiment_data['confidence']:.1f})", f"سنتیمنت بازار نزولی ({sentiment_data['confidence']:.1f})"))

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

    # --- Smart Position Sizing ---
    smart_sizing_data = None
    if enable_smart_sizing:
        smart_sizing_data = calculate_smart_position_size(
            base_lot_size=lot_size,
            confidence=confidence,
            atr=curr_atr,
            sentiment_data=sentiment_data,
            risk_multiplier=risk_multiplier
        )
        # Use the adjusted lot size
        lot_size = smart_sizing_data['adjusted_lot_size']

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

    badge_class = "badge-neutral"
    if signal in ["BUY", "STRONG BUY"]:
        badge_class = "badge-buy"
    elif signal in ["SELL", "STRONG SELL"]:
        badge_class = "badge-sell"

    st.markdown(
        f"""
        <div class="top-status">
            <span class="k">{tr("Price", "قیمت")}:</span><span class="v"><strong>${curr_price:,.2f}</strong></span>
            <span class="k">{tr("Signal", "سیگنال")}:</span><span class="v"><span class="{badge_class}">{signal_display}</span></span>
            <span class="k">{T["confidence"]}:</span><span class="v"><strong>{confidence:.0f}%</strong></span>
            <span class="k">{T["last_update"]}:</span><span class="v">{pd.Timestamp.utcnow().strftime('%H:%M:%S')} UTC</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="hero-signal">
            <span class="h-label">{tr("Main Signal", "سیگنال اصلی")}:</span>
            <span class="h-value">{signal_display}</span>
            <span class="h-label">{T["confidence"]}:</span>
            <span class="h-value">{confidence:.0f}%</span>
            <span class="h-label">{T["bias_score"]}:</span>
            <span class="h-value">{bias_score:.1f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(T["price"], f"${curr_price:,.2f}")
    c2.metric(T["rsi"], f"{curr_rsi:.2f}")
    c3.metric(T["atr"], f"{curr_atr:.2f}")
    c4.metric(T["dxy"], f"{curr_dxy:.2f}")
    conf_tag = tr("Weak", "ضعیف")
    if confidence >= 75:
        conf_tag = tr("Strong", "قوی")
    elif confidence >= 50:
        conf_tag = tr("Medium", "متوسط")
    c5.metric(T["confidence"], f"{confidence:.0f}%", conf_tag)

    c6, c7, c8 = st.columns([1.2, 1, 1])
    with c6:
        signal_class = "pulse-animation" if enable_animations else ""
        if signal in ["BUY", "STRONG BUY"]:
            if signal == "STRONG BUY" and enable_audio_alerts:
                play_alert_sound("strong_buy")
            st.markdown(f"<div class='signal-buy {signal_class}'><h3>{signal_display}</h3><p>{T['bias_score']}: {bias_score:.1f}</p></div>", unsafe_allow_html=True)
        elif signal in ["SELL", "STRONG SELL"]:
            if signal == "STRONG SELL" and enable_audio_alerts:
                play_alert_sound("strong_sell")
            st.markdown(f"<div class='signal-sell {signal_class}'><h3>{signal_display}</h3><p>{T['bias_score']}: {bias_score:.1f}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='signal-neutral'><h3>{T['signal_wait']}</h3><p>{T['bias_score']}: {bias_score:.1f}</p></div>", unsafe_allow_html=True)

    with c7:
        st.subheader(T["risk"])
        st.write(f"{T['risk_amount']}: ${risk_amt:.2f}")
        if smart_sizing_data:
            st.write(f"{T['base_position']}: {smart_sizing_data['base_lot_size']:.3f}")
            st.write(f"{T['adjusted_position']}: {smart_sizing_data['adjusted_lot_size']:.3f}")
            st.write(f"{T['sizing_factor']}: {smart_sizing_data['sizing_factor']:.2f}x")
            with st.expander(T["smart_position_sizing"]):
                st.write(f"{T['volatility_adjustment']}: {smart_sizing_data['volatility_factor']:.2f}x")
                if sentiment_data:
                    st.write(f"{T['sentiment_adjustment']}: {smart_sizing_data['sentiment_factor']:.2f}x")
                st.write(f"{T['risk_multiplier']}: {smart_sizing_data['risk_multiplier']:.2f}x")
        else:
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

    # --- Sentiment Analysis Display ---
    if sentiment_data:
        sentiment_color = "#21c77a" if sentiment_data['overall'] == 'bullish' else "#ff5a7a" if sentiment_data['overall'] == 'bearish' else "#8a96ad"
        sentiment_display = T[sentiment_data['overall']] if sentiment_data['overall'] in T else sentiment_data['overall']
        
        st.markdown(f"""
        <div class='app-card' style='border-left: 5px solid {sentiment_color};'>
            <h4 style='margin:0; color: var(--txt);'>{T['sentiment_analysis']}</h4>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-top: 10px;'>
                <div>
                    <strong>{T['overall_sentiment']}:</strong> 
                    <span style='color: {sentiment_color}; font-weight: 600;'>{sentiment_display.upper()}</span>
                </div>
                <div>
                    <strong>{T['sentiment_score']}:</strong> {sentiment_data['score']:.2f}
                </div>
                <div>
                    <strong>{T['sentiment_confidence']}:</strong> {sentiment_data['confidence']:.1f}%
                </div>
                <div>
                    <strong>{T['news_count']}:</strong> {sentiment_data['count']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Macro Dashboard Display ---
    real_yield_color = "#21c77a" if real_yields_data['real_yield'] > 0 else "#ff5a7a" if real_yields_data['real_yield'] < 0 else "#8a96ad"
    with st.expander(tr("Macro & Fundamental", "ماکرو و فاندامنتال"), expanded=False):
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
    if backtest_data:
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
    if correlation_data:
        with st.expander(T["correlation_matrix"], expanded=False):
            st.markdown(f"<div class='app-card'><h4 style='margin:0;'>{T['correlation_matrix']}</h4></div>", unsafe_allow_html=True)
            
            # Create and display heatmap
            heatmap_fig = create_correlation_heatmap(correlation_data)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Display detailed correlation table
            correlation_rows = []
            for asset_key, data in correlation_data.items():
                asset_name = T[asset_key]
                trend_text = T[data['trend']]
                trend_color = "#21c77a" if data['trend'] == 'strengthening' else "#ff5a7a" if data['trend'] == 'weakening' else "#8a96ad"
                
                correlation_rows.append({
                    T['correlation_with']: asset_name,
                    T['correlation_30d']: f"{data['corr_30d']:.3f}",
                    T['correlation_90d']: f"{data['corr_90d']:.3f}",
                    T['correlation_180d']: f"{data['corr_180d']:.3f}",
                    T['correlation_trend']: f"<span style='color: {trend_color}; font-weight: 600;'>{trend_text}</span>"
                })
            
            correlation_df = pd.DataFrame(correlation_rows)
            st.markdown(correlation_df.to_html(escape=False, index=False), unsafe_allow_html=True)

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

