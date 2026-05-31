import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Indian Stock Analyzer", layout="wide",initial_sidebar_state="expanded")
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
plt.style.use('ggplot') # Makes matplotlib charts look cleaner

# --- HELPER FUNCTIONS ---
def fetch_data(ticker, period="1y"):
    """Fetches historical stock data from Yahoo Finance."""
    # Append .NS for NSE (Indian) stocks if not provided
    if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
        ticker = ticker + '.NS' 
    
    data = yf.download(ticker, period=period, progress=False)
    

    # This checks for that and flattens it back to a standard format.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    return data

def calculate_indicators(data):
    """Calculates SMA 20, SMA 50, and RSI."""
    df = data.copy()
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI Calculation (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
    
    return df.dropna()

def calculate_risk_metrics(df):
    """Calculates Volatility, Sharpe Ratio, and Max Drawdown."""
    # Assuming 252 trading days in a year
    daily_volatility = df['Daily_Return'].std()
    annual_volatility = daily_volatility * np.sqrt(252)
    
    # Sharpe Ratio (Assuming 6% risk-free rate for India)
    risk_free_rate = 0.06 
    annual_return = df['Daily_Return'].mean() * 252
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # Max Drawdown
    cumulative_max = df['Cumulative_Return'].cummax()
    drawdown = (df['Cumulative_Return'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return annual_volatility, sharpe_ratio, max_drawdown

#  MAIN APP UI 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Single Stock Analysis", "Compare Stocks", "RSI Screener", "Nifty 50 Heatmap"])

st.sidebar.info("Tip: For Indian stocks, use symbols like RELIANCE.NS, TCS.NS, INFY.NS")

if page == "Single Stock Analysis":
    st.title("📈 Stock Price Analysis & Trading Simulation")
    
    # User Input
    col1, col2 = st.columns([1, 1])
    with col1:
        ticker = st.text_input("Enter Stock Ticker (e.g., TATAMOTORS.NS):", "RELIANCE.NS")
    with col2:
        period = st.selectbox("Select Time Period:", ["6mo", "1y", "2y", "5y"], index=1)
        
    if st.button("Analyze"):
        with st.spinner("Fetching data..."):
            df = fetch_data(ticker, period)
            
            if df.empty:
                st.error("Could not fetch data. Please check the ticker symbol.")
            else:
                df = calculate_indicators(df)
                
                # --- METRICS DASHBOARD ---
                st.subheader("1. Core Performance & Risk Metrics")
                vol, sharpe, max_dd = calculate_risk_metrics(df)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                m2.metric("Annual Volatility", f"{vol*100:.2f}%")
                m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                m4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
                
                # --- VISUALIZATIONS ---
                st.subheader("2. Visualization Dashboard")
                
                # Chart 1: Price and SMAs
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
                ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='orange')
                ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='purple')
                ax1.set_title(f"{ticker} Price History & Moving Averages")
                ax1.legend()
                st.pyplot(fig1)
                
                # Chart 2: RSI
                fig2, ax2 = plt.subplots(figsize=(10, 2))
                ax2.plot(df.index, df['RSI'], color='purple')
                ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
                ax2.axhline(30, linestyle='--', color='green', alpha=0.5)
                ax2.set_title("Relative Strength Index (RSI)")
                st.pyplot(fig2)

                # --- TRADING STRATEGY & RECOMMENDATION ---
                st.subheader("3. Strategy & Recommendation")
                
                current_rsi = df['RSI'].iloc[-1]
                current_sma20 = df['SMA_20'].iloc[-1]
                current_sma50 = df['SMA_50'].iloc[-1]
                
                st.write(f"**Current RSI:** {current_rsi:.2f}")
                st.write(f"**SMA 20:** ₹{current_sma20:.2f} | **SMA 50:** ₹{current_sma50:.2f}")
                
                # Logic for recommendation
                st.markdown("### Final Verdict")
                if current_rsi < 30:
                    st.success("🟢 **STRONG BUY SIGNAL:** The stock is currently oversold (RSI < 30). This often indicates a good buying opportunity as the asset may be undervalued.")
                elif current_sma20 > current_sma50 and current_rsi < 70:
                    st.success("🟢 **BUY SIGNAL:** The short-term trend is bullish (Golden Cross: SMA 20 > SMA 50) and it is not yet overbought.")
                elif current_rsi > 70:
                    st.error("🔴 **SELL / AVOID SIGNAL:** The stock is overbought (RSI > 70). It might be due for a price correction. Not the best time to buy.")
                elif current_sma20 < current_sma50:
                    st.warning("🟡 **HOLD / WAIT:** The stock is in a bearish trend (Death Cross: SMA 20 < SMA 50). Wait for trend reversal before buying.")
                else:
                    st.info("⚪ **NEUTRAL:** No strong signals present. Monitor the stock closely.")

elif page == "Compare Stocks":
    st.title("⚖️ Multi-Stock Comparison")
    
    tickers_input = st.text_input("Enter 2-3 Tickers separated by commas:", "TCS.NS, INFY.NS, WIPRO.NS")
    period = st.selectbox("Select Time Period:", ["6mo", "1y", "2y"], index=1)
    
    if st.button("Compare"):
        tickers = [t.strip() for t in tickers_input.split(',')]
        
        with st.spinner("Fetching data..."):
            compare_df = pd.DataFrame()
            
            for t in tickers:
                data = fetch_data(t, period)
                if not data.empty:
                    # Normalize prices to 100 for fair comparison
                    normalized = (data['Close'] / data['Close'].iloc[0]) * 100
                    compare_df[t] = normalized
            
            if not compare_df.empty:
                st.subheader("Comparative Cumulative Returns (Base 100)")
                fig, ax = plt.subplots(figsize=(10, 5))
                for col in compare_df.columns:
                    ax.plot(compare_df.index, compare_df[col], label=col)
                ax.legend()
                ax.set_ylabel("Normalized Value")
                st.pyplot(fig)
                
                st.subheader("Correlation Matrix")
                st.write("Values closer to 1 mean the stocks move together. Values closer to 0 or negative mean they move independently.")
                # Calculate daily returns for correlation
                returns_df = compare_df.pct_change().dropna()
                corr_matrix = returns_df.corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))


# --- UPDATED OVERSOLD/OVERBOUGHT SCREENER FEATURE ---
elif page == "RSI Screener":
    st.title("🎯 Market Conditions Screener")
    st.write("Scan a list of stocks to find those in **oversold (RSI < 30)** or **overbought (RSI > 70)** territory.")
    
    # 1. Added a Radio button to select the scan type
    scan_type = st.radio("Select Scan Type:", ["Oversold (RSI < 30)", "Overbought (RSI > 70)"], horizontal=True)
    
    # Default list of large-cap Indian stocks
    default_scan_list = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, SBIN.NS, BHARTIARTL.NS, ITC.NS, LT.NS, BAJFINANCE.NS, HINDUNILVR.NS, KOTAKBANK.NS, TATAMOTORS.NS, M&M.NS"
    scan_input = st.text_area("Stocks to Scan (comma-separated):", default_scan_list)
    
    if st.button("Start Scan"):
        tickers_to_scan = [t.strip() for t in scan_input.split(',')]
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers_to_scan):
            status_text.text(f"Scanning {ticker}... ({i+1}/{len(tickers_to_scan)})")
            
            try:
                data = fetch_data(ticker, period="3mo")
                
                if not data.empty and len(data) > 20: 
                    df = calculate_indicators(data)
                    latest_rsi = df['RSI'].iloc[-1]
                    latest_price = df['Close'].iloc[-1]
                    
                    # 2. Logic to handle both conditions
                    if scan_type == "Oversold (RSI < 30)":
                        if latest_rsi < 30:
                            results.append({"Ticker": ticker, "Latest Price": f"₹{latest_price:.2f}", "RSI": round(latest_rsi, 2)})
                    else: # Overbought (RSI > 70)
                        if latest_rsi > 70:
                            results.append({"Ticker": ticker, "Latest Price": f"₹{latest_price:.2f}", "RSI": round(latest_rsi, 2)})

            except Exception:
                pass  
                
            progress_bar.progress((i + 1) / len(tickers_to_scan))
            
        status_text.text("Scan Complete!")
        
        # 3. Displaying Results
        if results:
            color = "green" if "Oversold" in scan_type else "red"
            st.success(f"Found {len(results)} stock(s) in {scan_type} territory!")
            
            results_df = pd.DataFrame(results)
            
            # Dynamic styling: Green text for oversold, Red for overbought
            st.table(results_df.style.map(
                lambda x: f'color: {color}; font-weight: bold' if isinstance(x, (int, float)) else '', 
                subset=['RSI']
            ))
        else:
            st.info(f"No stocks found matching the {scan_type} criteria.")

elif page == "Nifty 50 Heatmap":
    st.title("🗺️ Nifty 50 Market Heatmap")
    st.write("For a real-time, professional-grade interactive heatmap, click the button below.")
    
    st.link_button("🔥 Open Live Nifty 50 Heatmap", "https://in.tradingview.com/heatmap/stock/#%7B%22dataSource%22%3A%22NIFTY50%22%2C%22blockColor%22%3A%22change%22%2C%22blockSize%22%3A%22market_cap_basic%22%2C%22grouping%22%3A%22sector%22%7D")
    
    st.info("This opens a specialized heat map where you can view performance by sector and market cap.")
