==========================================
# FILE NAME: stock_ultimate_314.py
# TARGET: Python 3.14+
# DESCRIPTION: Stock AI with Native Pandas Calculations (No pandas_ta dependency)
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- CONFIGURATION ---
st.set_page_config(page_title="Stock AI: Python 3.14 Edition", layout="wide")

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- 1. DATA ENGINES ---

def get_data(symbol: str, years: int = 2) -> pd.DataFrame | None:
    """Fetches stock data using yfinance."""
    start = date.today() - timedelta(days=365*years)
    try:
        # Python 3.14 + Pandas 2.3+ handling for yfinance
        df = yf.download(symbol, start=start, progress=False)
        if df.empty: return None
        
        # Handle MultiIndex columns common in new pandas versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

def get_sector_trend(symbol: str):
    """Checks if the Sector is Bullish or Bearish."""
    sector_map = {
        "RELIANCE.NS": "^NSEI", "TCS.NS": "^CNXIT", "INFY.NS": "^CNXIT", 
        "HDFCBANK.NS": "^NSEBANK", "SBIN.NS": "^NSEBANK", "TATAMOTORS.NS": "^CNXAUTO",
        "BPCL.NS": "^CNXENERGY", "IOC.NS": "^CNXENERGY", "AAPL": "^IXIC", "TSLA": "^IXIC"
    }
    
    sector_symbol = sector_map.get(symbol, "^NSEI") 
    try:
        sec_df = yf.download(sector_symbol, period="1mo", progress=False)
        if not sec_df.empty:
            if isinstance(sec_df.columns, pd.MultiIndex):
                sec_df.columns = sec_df.columns.get_level_values(0)
            
            start = sec_df['Close'].iloc[0]
            end = sec_df['Close'].iloc[-1]
            change = ((end - start) / start) * 100
            return sector_symbol, change
    except:
        pass
    return "Market", 0.0

def calculate_max_pain(symbol: str):
    """Calculates Options Max Pain."""
    try:
        tk = yf.Ticker(symbol)
        if not tk.options: return None
        
        # Get nearest expiry
        opt = tk.option_chain(tk.options[0])
        calls = opt.calls
        puts = opt.puts
        
        strikes = calls['strike'].unique()
        pain_data = []
        
        for k in strikes:
            # Vectorized calculation for speed
            call_loss = ((calls['strike'] < k) * (k - calls['strike']) * calls['openInterest']).sum() 
            # Logic correction: Call writers lose if Price (k) > Strike. 
            # Wait, Max Pain is calculated assuming expiry price is K.
            # If Expiry = K:
            # Call Writer loses: max(0, K - Strike)
            # Put Writer loses: max(0, Strike - K)
            
            c_loss = calls.apply(lambda x: max(0, k - x['strike']) * x['openInterest'], axis=1).sum()
            p_loss = puts.apply(lambda x: max(0, x['strike'] - k) * x['openInterest'], axis=1).sum()
            pain_data.append(c_loss + p_loss)
            
        min_pain_idx = np.argmin(pain_data)
        return strikes[min_pain_idx]
    except:
        return None

def get_news_score(symbol: str) -> float:
    try:
        tk = yf.Ticker(symbol)
        news = tk.news
        if not news: return 0.0
        sia = SentimentIntensityAnalyzer()
        scores = []
        for n in news[:5]:
            if 'title' in n:
                scores.append(sia.polarity_scores(n['title'])['compound'])
        return sum(scores)/len(scores) if scores else 0.0
    except:
        return 0.0

# --- 2. MATH CORE (Native Pandas - No External Dependency) ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Technical Indicators using native Pandas (Fast & 3.14 Compatible)."""
    df['RSI_14'] = calculate_rsi(df['Close'])
    df['ATRr_14'] = calculate_atr(df)
    df['BBU_20'], df['BBL_20'] = calculate_bollinger_bands(df['Close'])
    df.dropna(inplace=True)
    return df

def ai_predict(df: pd.DataFrame):
    """Random Forest Prediction."""
    df = df.copy()
    df['Target'] = df['Close'].shift(-60) # ~3 Months
    data = df.dropna()
    
    if len(data) < 100: return 0.0, 0.0
    
    features = ['Close', 'RSI_14', 'ATRr_14', 'BBU_20', 'BBL_20']
    X = data[features]
    y = data['Target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    latest = df[features].iloc[-1].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    score = model.score(X, y)
    return pred, score

def kelly_criterion(win_prob, win_loss_ratio):
    kelly = (win_prob - ((1 - win_prob) / win_loss_ratio))
    return max(0.0, kelly * 0.5)

# --- 3. UI DASHBOARD ---

st.title("🤖 Stock AI: Python 3.14 Edition")

with st.sidebar:
    st.header("Strategy Settings")
    symbol = st.text_input("Stock Symbol", "RELIANCE.NS").upper()
    capital = st.number_input("Account Capital (₹)", value=100000, step=10000)
    st.caption("Engine: Native Pandas (Optimized for Py 3.14)")
    run = st.button("Run Full Analysis")

if run:
    with st.spinner(f"Analyzing {symbol}..."):
        
        df = get_data(symbol)
        
        if df is None:
            st.error(f"Could not find data for {symbol}.")
        else:
            # Parallel-like fetch (simplified for Streamlit)
            sector_name, sector_change = get_sector_trend(symbol)
            max_pain = calculate_max_pain(symbol)
            news_score = get_news_score(symbol)
            
            df = process_data(df)
            pred_price, model_conf = ai_predict(df)
            
            curr_price = df['Close'].iloc[-1]
            atr = df['ATRr_14'].iloc[-1]
            margin = ((pred_price - curr_price) / curr_price) * 100
            
            # Risk Logic
            win_prob = max(0.51, min(0.80, model_conf + 0.5)) 
            kelly_pct = kelly_criterion(win_prob, 2.0)
            
            # Verdict Scorecard
            score = 0
            if margin > 10: score += 1
            if sector_change > 0: score += 1
            if news_score > 0.1: score += 1
            if max_pain and curr_price < max_pain: score += 1 
            
            if score >= 3: verdict, color = "STRONG BUY 🚀", "green"
            elif score >= 1: verdict, color = "ACCUMULATE 🔵", "blue"
            else: verdict, color = "AVOID / SELL 🔴", "red"

            # --- DISPLAY ---
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"## Verdict: :{color}[{verdict}]")
            c1.caption(f"Confidence Score: {score}/4")
            c2.metric("🎯 AI Target (3M)", f"₹{pred_price:.2f}", f"{margin:.1f}%")
            
            sec_arrow = "⬆️" if sector_change > 0 else "⬇️"
            c2.caption(f"Sector: {sector_change:.1f}% {sec_arrow}")

            st.divider()
            
            k1, k2, k3 = st.columns(3)
            alloc_amt = capital * kelly_pct
            qty = int(alloc_amt / curr_price)
            
            k1.metric("Position Size", f"₹{alloc_amt:,.0f}", f"{kelly_pct*100:.1f}%")
            k2.metric("Quantity", f"{qty} Shares")
            sl_price = curr_price - (atr * 2)
            k3.metric("Volatility Stop Loss", f"₹{sl_price:.2f}")
            
            st.divider()
            
            d1, d2 = st.columns(2)
            with d1:
                st.subheader("🐋 Max Pain")
                if max_pain:
                    st.metric("Option Magnet", f"₹{max_pain}")
                    st.progress(min(1.0, curr_price/max_pain))
                else:
                    st.write("No Data")
            with d2:
                st.subheader("📰 News")
                st.write(f"Sentiment Score: {news_score:.2f}")

            # Chart
            fig = go.Figure()
            recent = df.tail(150)
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['Close'], name='Price', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['BBU_20'], name='Upper Band', line=dict(width=1, color='gray', dash='dot')))
            fig.add_trace(go.Scatter(x=recent['Date'], y=recent['BBL_20'], name='Lower Band', line=dict(width=1, color='gray', dash='dot')))
            if max_pain:
                fig.add_hline(y=max_pain, line_color="orange", annotation_text="Max Pain")
            fig.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)