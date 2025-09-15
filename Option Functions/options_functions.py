import yfinance as yf
import pandas as pd
from datetime import datetime, date  # Added this import
from numpy import log, sqrt, exp
from scipy.stats import norm
from scipy.stats import norm
from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import os

from scipy.optimize import brentq


def get_options(ticker,expiry_date):
    stock = yf.Ticker(ticker)
    opt_chain = stock.option_chain(expiry_date)
    calls = opt_chain.calls
    puts = opt_chain.puts
    return calls, puts, stock


#print(calls[['lastPrice', 'bid','ask', 'strike', 'impliedVolatility']].sort_values(by='lastPrice', ascending=False).head(10))
def get_mid_price(strike, calls, puts):
    call_row = calls[calls['strike'] == strike]
    put_row = puts[puts['strike'] == strike]

    call_mid = (
        (call_row['bid'].values[0] + call_row['ask'].values[0]) / 2
        if not call_row.empty else None
    )
    put_mid = (
        (put_row['bid'].values[0] + put_row['ask'].values[0]) / 2
        if not put_row.empty else None
    )

    return {
        'call_mid_price': call_mid,
        'put_mid_price': put_mid
    }
    

#test=calls[calls['strike'] == K][['strike', 'bid','ask', 'lastPrice', 'impliedVolatility']]

#calculate implied volatility using Black-Scholes model
# great to use
#verifying IV with black-scholes model


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price=S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call_price

def implied_vol(C, S, K, T, r, initial_sigma=0.2):
    return newton(
        lambda sigma: black_scholes_call(S, K, T, r, sigma) - C,
        x0=initial_sigma
    )

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European put option.
    """
    from numpy import log, sqrt, exp
    from scipy.stats import norm

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    put_price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price



# Example: call option

from py_vollib.black_scholes import implied_volatility

def get_implied_volatility(option_price, S, K, T, r, option_type):
    """
    Calculate implied volatility using py_vollib, with intrinsic value check.

    Parameters:
    - option_price: Observed option market price
    - S: Spot price of the underlying
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free rate
    - option_type: 'c' for call, 'p' for put

    Returns:
    - Implied volatility as a float, or None if not computable
    """
    # Calculate intrinsic value
    # intrinsic_value = max(0, S - K) if option_type == 'c' else max(0, K - S)

    # if option_price < intrinsic_value:
    #     print(f"[Skipped IV] Price ({option_price}) < Intrinsic Value ({intrinsic_value})")
    #     return None

    try:
        iv = implied_volatility.implied_volatility(
            price=option_price,
            S=S,
            K=K,
            t=T,
            r=r,
            flag=option_type
        )
        return iv
    except Exception as e:
        # print(f"[IV Calculation Error] {e}")
        return None





# --- Step 1: Pull IV/HV Data from yfinance ---
def fetch_iv_hv(ticker, window=20):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y", interval="1d")
    
    hist['Daily Return'] = hist['Close'].pct_change()
    hv = hist['Daily Return'].rolling(window=window).std() * np.sqrt(252)
    latest_price = hist['Close'].iloc[-1]
    try:
        options_date = stock.options[0]  # nearest expiry
        opt_chain = stock.option_chain(options_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        iv_mean = pd.concat([calls['impliedVolatility'], puts['impliedVolatility']]).mean()
    except Exception:
        iv_mean = np.nan

    return {
        "latest_price": float(latest_price),
        "historical_volatility": float(hv.iloc[-1]),
        "latest_iv": float(iv_mean) if not np.isnan(iv_mean) else None,
        "implied_volatility": float(iv_mean),
        "hist_df": hist
        #"stock": stock
        
    }


# --- Step 2: Backtest Payoff for Iron Condor and Debit Spread ---
# Iron Condor Strategy:
	# 1.	Sell 1 out-of-the-money call (higher strike)
	# 2.	Buy 1 further out-of-the-money call (even higher strike)

	# 3.	Sell 1 out-of-the-money put (lower strike)
	# 4.	Buy 1 further out-of-the-money put (even lower strike)

# Call Debit Spread Strategy, also called Bull Call Spread:
    # 1.	Buy 1 in-the-money call (lower strike)
    # 2.	Sell 1 in-the-money call (higher strike)


def iron_condor_payoff(spot_prices, low_put, high_put, low_call, high_call, credit):
    payoff = []
    width_put = high_put - low_put
    width_call = high_call - low_call
    for S in spot_prices:
        if S < low_put:
            result = credit - width_put
        elif low_put <= S < high_put:
            result = credit - (S - low_put)
        elif high_put <= S <= low_call:
            result = credit
        elif low_call < S <= high_call:
            result = credit - (high_call - S)
        else:  # S > high_call
            result = credit - width_call
        payoff.append(result)
    return np.array(payoff)

def call_debit_spread_payoff(spot_prices, long_call, short_call, debit):
    payoff = []
    for S in spot_prices:
        value = np.maximum(0, S - long_call) - np.maximum(0, S - short_call)
        payoff.append(value - debit)
    return np.array(payoff)

# --- Step 3: Visualize Payoffs ---
def plot_payoffs(spot_prices, iron_condor, debit_spread):
    plt.figure(figsize=(10, 6))
    plt.plot(spot_prices, iron_condor, label="Iron Condor")
    plt.plot(spot_prices, debit_spread, label="Call Debit Spread")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Options Strategy Payoffs")
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Profit / Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
# Iron Condor Strategy:
	# 1.	Sell 1 out-of-the-money call (higher strike)
	# 2.	Buy 1 further out-of-the-money call (even higher strike)

	# 3.	Sell 1 out-of-the-money put (lower strike)
	# 4.	Buy 1 further out-of-the-money put (even lower strike)

# Call Debit Spread Strategy, also called Bull Call Spread:
    # 1.	Buy 1 in-the-money call (lower strike)
    # 2.	Sell 1 in-the-money call (higher strike)

# --- Example Usage ---

# ticker = "BMO"
# expiry_date = "2026-06-18"  # Define the expiry date
# expiry_dt = pd.to_datetime(expiry_date)
# lowput=85
# highput=95
# lowcall=110
# highcall=115

# calls, puts, stock = get_options(ticker, expiry_date)

# lowcall = get_mid_price(lowcall)['call_mid_price']
# highcall = get_mid_price(highcall)['call_mid_price']
# lowput = get_mid_price(lowput)['put_mid_price']
# highput = get_mid_price(highput)['put_mid_price']
# #print(f"Call Mid Price: {value['call_mid_price']:.2f}, Put Mid Price: {value['put_mid_price']:.2f}")
# print(lowcall, highcall, lowput, highput)

# #debit spread
# #debit spread credit= high call in the money - low call in the money
# d_lowstrike = 95
# d_highstrike = 105
# d_lowcall = get_mid_price(d_lowstrike)['call_mid_price']
# d_highcall = get_mid_price(d_highstrike)['call_mid_price']
# print(f"Debit Spread: Long Call {d_lowstrike} at ${d_lowcall:.2f}, Short Call {d_highstrike} at ${d_highcall:.2f}")

# if __name__ == "__main__":
#     ticker = ticker
#     data = fetch_iv_hv(ticker)
#     print(f"{ticker} IV: {data['implied_volatility']:.2%}, HV: {data['historical_volatility']:.2%}, Price: ${data['latest_price']:.2f}")

#     # Define strikes and premiums
#     iron_condor = iron_condor_payoff(
#         spot_prices=np.linspace(80, 145, 200),
#         low_put=lowput, high_put=highput,
#         low_call=lowcall, high_call=highcall,
#         credit=lowcall-highcall + lowput-highput
#     )

#     debit_spread = call_debit_spread_payoff(
#         spot_prices=np.linspace(80, 145, 200),
#         long_call=d_lowstrike, short_call=d_highstrike,
#         debit=d_lowcall-d_highcall 
#     )

#     plot_payoffs(np.linspace(80, 145, 200), iron_condor, debit_spread)


# straddle_payoff_sim.py

def straddle_payoff(spot_prices, strike_price, call_premium, put_premium):
    total_cost = call_premium + put_premium
    call_payoff = np.maximum(spot_prices - strike_price, 0)
    put_payoff = np.maximum(strike_price - spot_prices, 0)
    total_payoff = call_payoff + put_payoff - total_cost
    return total_payoff

def plot_straddle(spot_prices, payoff, strike_price):
    plt.figure(figsize=(10, 6))
    plt.plot(spot_prices, payoff, label='Straddle Payoff', color='blue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(strike_price, color='gray', linestyle='--', label='Strike Price')
    plt.title("Long Straddle Payoff Diagram")
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Profit / Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
# if __name__ == "__main__":
#     strike = 325
#     call_price = 17
#     put_price = 17
#     spot_range = np.linspace(230, 500, 200)

#     payoff = straddle_payoff(spot_range, strike, call_price, put_price)
#     plot_straddle(spot_range, payoff, strike)

#multiple rolling volatility estimators



# symbol = "NVDA"  # You can change this to any stock ticker you want
# start_date = "2020-01-01"
# end_date = "2026-01-01"
# rolling_window = 240  # 21 trading days in a month
# delta_num = 0.01  # Delta for First Exit Time Volatility

# windows = [20, 30, 40, 60, 120, 240]
#get local csv files

def get_csv_files(path, filename):
    full_path = os.path.join(path, filename)
    df=pd.read_csv(full_path)
    ticker=df['ticker'].unique().tolist()
    #price_data = df.pivot_table(index='Datetime', columns='ticker', values='Close')
    return df, ticker

# path = '/Users/jayren/Desktop/stock/Stock_Daily'
# dailydata = '/Users/jayren/Desktop/stock/stock_Daily/dailydata'
# consolidated = '/stockPrice_consolidated'


#us: us_stock_price.csv
# ca: ca_stock_price.csv
# adhoc: adhoc_stock_price.csv



# --- Step 1: Download Data ---
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close']]
    df.dropna(inplace=True)
    return df

# --- Step 2: Log Return ---
def compute_log_returns(df):
    return np.log(df['Close'] / df['Close'].shift(1))

# --- Step 3: Volatility Estimators ---
def close_to_close_vol(df, window):
    log_ret = compute_log_returns(df)
    return log_ret.rolling(window).std() * np.sqrt(252)

def parkinson_vol(df, window):
    factor = 1 / (4 * np.log(2))
    log_hl = np.log(df['High'] / df['Low']) ** 2
    return (log_hl.rolling(window).mean() * factor) ** 0.5 * np.sqrt(252)

def yang_zhang_volatility_multi_windows(df, window_list):
    """
    Calculate Yang-Zhang volatility for multiple rolling windows.
    Returns a dictionary: {window_size: volatility_series}
    """
    results = {}
    results = {}
    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']

    co = np.log(c / o)
    oc = np.log(o / c.shift(1))
    ho = np.log(h / o)
    lo = np.log(l / o)

    for window in window_list:
        sigma_o = oc.rolling(window).var()
        sigma_c = co.rolling(window).var()
        sigma_rs = (ho * (ho - co) + lo * (lo - co)).rolling(window).mean()

        k = 1.34 / (1.34 + (window + 1) / (window - 1))
        yz_vol = np.sqrt(sigma_o + k * sigma_c + (1 - k) * sigma_rs) * np.sqrt(252)

        results[window] = yz_vol
    return results


def yang_zhang_volatility(df, window):
    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    co = np.log(c / o)
    oc = np.log(o / c.shift(1))
    ho = np.log(h / o)
    lo = np.log(l / o)

    sigma_o = oc.rolling(window).var()
    sigma_c = co.rolling(window).var()
    sigma_rs = (ho * (ho - co) + lo * (lo - co)).rolling(window).mean()

    k = 1.34 / (1.34 + (window + 1) / (window - 1))
    return np.sqrt(sigma_o + k * sigma_c + (1 - k) * sigma_rs) * np.sqrt(252)

def first_exit_volatility(prices, delta):
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()  # convert single-column DataFrame to Series
    elif not isinstance(prices, pd.Series):
        prices = pd.Series(prices)

    prices = pd.to_numeric(prices, errors='coerce').dropna()
    if prices.empty:
        return np.nan

    spot = float(prices.iloc[0])
    hits = []
    t = 0

    for price in prices:
        t += 1
        price = float(price)
        if abs(np.log(price / spot)) >= delta:
            hits.append(t)
            t = 0
            spot = price

    if not hits:
        return np.nan
    mean_tau = np.mean(hits)
    return delta / np.sqrt(mean_tau) * np.sqrt(252)

def rolling_first_exit(df, window, delta):
    fe_vol = []
    closes = df['Close']
    for i in range(len(closes)):
        if i < window:
            fe_vol.append(np.nan)
        else:
            window_prices = closes.iloc[i-window+1:i+1]
            fe_vol.append(first_exit_volatility(window_prices, delta=delta))
    return pd.Series(fe_vol, index=closes.index)

# --- Step 4: Plotting ---
def plot_volatility(df, ctc, park, yz, fe):
    plt.figure(figsize=(14, 6))
    plt.plot(ctc, label="Close-to-Close Volatility", color='gray')
    plt.plot(park, label="Parkinson Volatility", color='green')
    plt.plot(yz, label="Yang-Zhang Volatility", color='blue')
    plt.plot(fe, label="First Exit Time Volatility", color='orange')
    plt.title("Rolling Volatility Estimates")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_log_returns(df):
    log_ret = compute_log_returns(df)
    plt.figure(figsize=(14, 4))
    plt.plot(log_ret, color='purple', label="Log Returns")
    plt.title("Daily Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# graphing with Plotly


import plotly.graph_objects as go

def plot_volatility_plotly(df, ctc, park, yz, fe, symbol, window):
    # Combine all volatility series into a DataFrame and drop rows with all NaNs
    # Ensure all volatility series are 1D Series
    ctc = ctc.squeeze()
    park = park.squeeze()
    yz = yz.squeeze()
    fe = fe.squeeze()
    vol_df = pd.DataFrame({
        'Close-to-Close': ctc,
        'Parkinson': park,
        'Yang-Zhang': yz,
        'First Exit': fe
    }).dropna(how='all')

    fig = go.Figure()
    for col in vol_df.columns:
        fig.add_trace(go.Scatter(x=vol_df.index, y=vol_df[col], mode='lines', name=col))

    fig.update_layout(
    title="Rolling Volatility Estimates (Plotly) " + symbol + " (" + str(window) + " days)",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    xaxis_title="Date",
    yaxis_title="Annualized Volatility",
    legend_title="Estimator",
    height=500,
    template="plotly_white"
    )

    fig.show(renderer="browser")

# --- Plot Histogram ---

# --- Log Return Statistics ---
def log_return_stats(df):
    """
    Compute and print statistics for daily log returns of the 'Close' column in df.
    """

    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    stats = {
        'Mean': log_returns.mean(),
        'Std Dev': log_returns.std(),
        'Min': log_returns.min(),
        'Max': log_returns.max(),
        'Skewness': log_returns.skew(),
        'Kurtosis': log_returns.kurtosis(),
        '25% Quantile': log_returns.quantile(0.25),
        'Median': float(log_returns.median()),
        '75% Quantile': log_returns.quantile(0.75),
        'Count': log_returns.count()
    }
    print("Statistics for Daily Log Returns:")
    for k, v in stats.items():
        try:
            print(f"{k}: {float(v):.6f}")
        except Exception:
            print(f"{k}: {v}")

# --- Plot Volatility Histogram ---
def plot_volatility_histogram_plotly(volatility_series, bins):
    """
    Plot an interactive histogram of volatility values using Plotly.
    X-axis: Volatility
    Y-axis: Relative Frequency (proportion)
    """
    import plotly.graph_objects as go
    import numpy as np

    # Drop NaN values
    vol = volatility_series.dropna()
    if vol.empty:
        print("No valid volatility values to plot.")
        return

    # Compute histogram data for relative frequency
    counts, bin_edges = np.histogram(vol, bins=bins, density=False)
    rel_freq = counts / counts.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=rel_freq,
        width=(bin_edges[1] - bin_edges[0]),
        marker_color='teal',
        opacity=0.8,
        name='Relative Frequency'
    ))

    fig.update_layout(
        title="Histogram of Volatility (Relative Frequency)",
        xaxis_title="Volatility",
        yaxis_title="Relative Frequency",
        bargap=0.05,
        template="plotly_white",
        height=500
    )

    fig.show(renderer="browser")

def plot_volatility_cone_yz(yz_vol_dict, symbol):
    cone_stats = {
        "Horizon": [],
        "Min": [],
        "25th": [],
        "Mean": [],
        "75th": [],
        "Max": [],
        "Current": []
    }

    for window, vol_df in yz_vol_dict.items():
        series = vol_df.iloc[:, 0].dropna()
        cone_stats["Horizon"].append(window)
        cone_stats["Min"].append(series.min())
        cone_stats["25th"].append(series.quantile(0.25))
        cone_stats["Mean"].append(series.mean())
        cone_stats["75th"].append(series.quantile(0.75))
        cone_stats["Max"].append(series.max())
        cone_stats["Current"].append(series.iloc[-1])

    cone_df = pd.DataFrame(cone_stats).sort_values("Horizon")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cone_df["Horizon"], y=cone_df["Min"], mode='lines+markers', name='Min'))
    fig.add_trace(go.Scatter(x=cone_df["Horizon"], y=cone_df["25th"], mode='lines+markers', name='25th Percentile'))
    fig.add_trace(go.Scatter(x=cone_df["Horizon"], y=cone_df["Mean"], mode='lines+markers', name='Mean'))
    fig.add_trace(go.Scatter(x=cone_df["Horizon"], y=cone_df["75th"], mode='lines+markers', name='75th Percentile'))
    fig.add_trace(go.Scatter(x=cone_df["Horizon"], y=cone_df["Max"], mode='lines+markers', name='Max'))
    fig.add_trace(go.Scatter(x=cone_df["Horizon"], y=cone_df["Current"], mode='lines+markers', name='Current', line=dict(dash='dot')))

    fig.update_layout(
        title=f"Volatility Cone (Yang-Zhang Estimator) {symbol}",
        xaxis_title="Rolling Window (Days)",
        yaxis_title="Annualized Volatility",
        legend_title="Statistic",
        template="plotly_white"
    )
    #return cone_df
    fig.show(renderer="browser")
# ----run--------------------------------------------------------------

# 'Close-to-Close': ctc,
# 'Parkinson': park,
# 'Yang-Zhang': yz,
# 'First Exit': fe
    

# # return price_data, ticker
# price_data, tickers = get_csv_files(
#     path=os.path.join(dailydata+consolidated), 
#     filename='us_stock_price.csv'
# )



# reading csv file which is the data by minute
#df=price_data[price_data['ticker']==symbol]

# df = fetch_data()
# ctc = close_to_close_vol(df)
# park = parkinson_vol(df)
# yz = yang_zhang_volatility(df)
# fe = rolling_first_exit(df)

# yz_vol_dict = yang_zhang_volatility_multi_windows(df, window_list=windows)


# # matplotlib plots
# plot_volatility(df, ctc, park, yz, fe)
# plot_log_returns(df)

# #plot_log_return_histogram_plotly(df)
# plot_volatility_plotly(df, ctc, park, yz, fe)

# # For close-to-close volatility
# #plot_volatility_histogram_plotly(yz, bins=200)

# log_return_stats(df)

# plot_volatility_cone_yz(yz_vol_dict)

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import date
from datetime import date

# print implied volatility vs. strike for call and put options'

def get_option_strikes_and_prices(symbol: str, expiry: date, option_type: str = "call"):
    """
    Fetches strike prices and market prices for options of a given symbol and expiry date.

    Parameters:
        symbol (str): Stock ticker symbol (e.g., "GOOG").
        expiry (str): Expiration date in format "YYYY-MM-DD".
        option_type (str): "call" or "put".

    Returns:
        Tuple[List[float], List[float]]: (list of strikes, list of market prices)
    """
    ticker = yf.Ticker(symbol)
    opt_chain = ticker.option_chain(expiry)
    options_df = opt_chain.calls if option_type.lower() == "call" else opt_chain.puts

    # Calculate mid price between bid and ask
    options_df["mid_price"] = (options_df["bid"] + options_df["ask"]) / 2

    # Drop rows with missing data
    filtered = options_df.dropna(subset=["strike", "mid_price"])

    strikes = filtered["strike"].tolist()
    market_prices = filtered["mid_price"].tolist()

    return strikes, market_prices, ticker

# Black-Scholes formula
def bs_price(option_type, S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# Implied volatility solver
def implied_vol(option_type, S, K, T, r, market_price):
    try:
        return brentq(lambda sigma: bs_price(option_type, S, K, T, r, sigma) - market_price, 1e-6, 5)
    except:
        return np.nan  # Failed to converge

# Main plotting function
def plot_iv_vs_strike(option_type, expire_date, S, K_list, r, market_prices):
    today = date.today()
    T = (expire_date - today).days / 365.0  # Time to maturity in years
    
    iv_list = []
    for K, mp in zip(K_list, market_prices):
        iv = implied_vol(option_type, S, K, T, r, mp)
        iv_list.append(iv)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=K_list, y=iv_list, mode='lines+markers',
                             name='Implied Volatility', line=dict(width=2)))
    fig.update_layout(
                    title=f'Implied Volatility vs Strike ({option_type.capitalize()}) - {symbol} (Spot: ${S:.2f})<br>Expiry: {expire_date}',
                    xaxis_title='Strike Price',
                    yaxis_title='Implied Volatility',
                    yaxis_tickformat='.0%',
                    template='plotly_white')
    fig.show(renderer="browser")


def iv_vs_strike(df, title="Implied Volatility vs. Strike"):
    """
    Plots IV vs Strike using Plotly from a DataFrame with 'strike' and 'IV' columns.
    
    Parameters:
    - df: DataFrame containing at least 'strike' and 'IV' columns
    - title: Plot title
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Strike'],
        y=df['IV'],
        mode='lines+markers',
        name='IV Curve',
        hovertemplate='Strike: %{x}<br>IV: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility',
        yaxis_tickformat=".0%",
        template='plotly_white'
    )

    fig.show(renderer="browser")

#-----run the code------------------------------------------------------------
# symbol='NVDA'  # Define the stock ticker
# r = 0.05
# expire_date = date(2026, 6, 18)  # Define the expiry date
# expire_date_str = expire_date.strftime("%Y-%m-%d")
# option_type = "call"  # Change to "call" for call options


def get_option_df(df, stock,t,r, option_type):
    """
    Creates an option DataFrame with calculated implied volatility and key pricing fields.

    Parameters:
    - df: DataFrame containing raw option chain (must include 'strike', 'bid', 'ask')
    - stock: yfinance.Ticker object
    - option_type: 'c' for call, 'p' for put

    Returns:
    - DataFrame with Strike, Mid Price, IV, Intrinsic Value, and Time Value
    """
    import pandas as pd
    s = stock.history(period='1d')['Close'].iloc[-1]

    # Create base DataFrame
    df_options = pd.DataFrame({
        'Strike': df['strike'],
        'Mid Price': (df['bid'] + df['ask']) / 2,
        'Spot Price': s,
        'lastPrice': df['lastPrice']
    })

    # Calculate Implied Volatility
    df_options['IV'] = df_options.apply(
    lambda row: get_implied_volatility(row['Mid Price'], s, row['Strike'], t, r, option_type)
    # if row['Mid Price'] >= (max(0, s - row['Strike']) if option_type == 'c' else max(0, row['Strike'] - s))
    # else None
    ,
    axis=1
)

    # Calculate Intrinsic Value and Time Value
    if option_type == 'c':
        df_options['Intrinsic Value'] = df_options.apply(lambda row: max(0, s - row['Strike']), axis=1)
    else:
        df_options['Intrinsic Value'] = df_options.apply(lambda row: max(0, row['Strike'] - s), axis=1)

    df_options['Time Value'] = df_options['Mid Price'] - df_options['Intrinsic Value']

    return df_options



def plot_stock_volatility_vs_vix(symbol, start_date, end_date, window):
    """
    Plots realized volatility of an individual stock vs VIX and their volatility spread.

    Parameters:
    - symbol (str): Ticker of the stock (e.g., 'AAPL', 'GOOG')
    - start_date (str): Start date for data download (format: 'YYYY-MM-DD')
    - end_date (str): End date for data download
    - window (int): Rolling window size for realized volatility (default: 30)

    Returns:
    - pd.DataFrame: DataFrame with realized volatility, VIX, and their spread
    """

    # Download stock and VIX data
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    vix_data = yf.download("^VIX", start=start_date, end=end_date)

    # Preprocess VIX data
    vix_data = vix_data[['Close']].rename(columns={'Close': '^VIX'}) / 100

    # Compute stock log returns and realized volatility
    stock_data['LogReturn'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    stock_data['RealizedVol'] = stock_data['LogReturn'].rolling(window).std() * np.sqrt(252)

    # Combine and align
    combined = pd.concat([stock_data['RealizedVol'], vix_data['^VIX']], axis=1).dropna()

    # Compute volatility spread
    combined['VolSpread'] = combined['^VIX'] * 100 - combined['RealizedVol'] * 100

    # Plot Realized Vol vs VIX
    plt.figure(figsize=(12, 5))
    combined[['RealizedVol', '^VIX']].multiply(100).plot(ax=plt.gca(), title=f'{symbol.upper()}: {window}-Day Realized Volatility vs VIX')
    plt.ylabel("Volatility (%)")
    plt.grid(True)
    plt.show()

    # Plot Volatility Risk Premium
    plt.figure(figsize=(12, 5))
    combined['VolSpread'].plot(title=f'{symbol.upper()}: Volatility Risk Premium (VIX - Realized Volatility)')
    plt.axhline(combined['VolSpread'].mean(), color='red', linestyle='--', label='Avg Spread')
    plt.ylabel("Spread (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return combined

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Return Black-Scholes price of a call or put option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(price, S, K, T, r, option_type='call'):
    """Return implied volatility using Brent's method; return NaN if it fails."""
    try:
        return brentq(lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - price,
                      1e-6, 5.0, maxiter=1000)
    except:
        return np.nan

def get_all_option_data(symbol, r):
    """
    Fetches all option chains for a given symbol, calculates mid-price and implied volatility.

    Parameters
    ----------
    symbol : str
        Stock ticker.
    r : float
        Risk-free rate (e.g., 0.03 for 3%).

    Returns
    -------
    all_calls : DataFrame with calculated mid_price and implied volatility.
    all_puts : Same for puts.
    """
    stock = yf.Ticker(symbol)
    expiries = stock.options
    spot_price = stock.history(period="1d")['Close'].iloc[-1]

    calls_list = []
    puts_list = []
    today = datetime.now()

    for expiry in expiries:
        try:
            opt_chain = stock.option_chain(expiry)
            calls = opt_chain.calls.copy()
            puts = opt_chain.puts.copy()

            expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
            T = (expiry_date - today).days / 365.0

            # Add expiration and time to expiry
            calls['expirationDate'] = expiry
            calls['time_to_expiry'] = T
            puts['expirationDate'] = expiry
            puts['time_to_expiry'] = T

            # Midprice and IV for calls
            calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
            calls['calculated_iv'] = calls.apply(
                lambda row: implied_volatility(
                    row['mid_price'], spot_price, row['strike'], row['time_to_expiry'], r, option_type='call'
                ), axis=1)

            # Midprice and IV for puts
            puts['mid_price'] = (puts['bid'] + puts['ask']) / 2
            puts['calculated_iv'] = puts.apply(
                lambda row: implied_volatility(
                    row['mid_price'], spot_price, row['strike'], row['time_to_expiry'], r, option_type='put'
                ), axis=1)

            calls_list.append(calls)
            puts_list.append(puts)
        except Exception as e:
            print(f"Skipping {expiry} due to error: {e}")
            continue

    all_calls = pd.concat(calls_list, ignore_index=True)
    all_puts = pd.concat(puts_list, ignore_index=True)
    return all_calls, all_puts

def plot_volatility_surface(option_df, symbol=""):
    """
    Plot implied volatility surface using the output from get_all_option_data.
    Expects columns: 'strike', 'calculated_iv', 'time_to_expiry'
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = 'browser'

    # Drop rows with missing values in required columns
    df = option_df.dropna(subset=['strike', 'calculated_iv', 'time_to_expiry'])

    # Pivot for surface plot
    df_pivot = df.pivot_table(index='strike', columns='time_to_expiry', values='calculated_iv')

    # Sort the time_to_expiry columns (x-axis) from smallest to largest
    df_pivot = df_pivot.reindex(sorted(df_pivot.columns,reverse=True),axis=1 )

    X = df_pivot.columns.values  # time to expiry (years)
    Y = df_pivot.index.values    # strike
    Z = df_pivot.values          # implied vol

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title=f'Implied Volatility Surface for {symbol.upper()}',
        scene=dict(
            xaxis_title='Time to Expiry (Years)',
            yaxis_title='Strike Price',
            zaxis_title='Implied Volatility'
        ),
        autosize=True,
        height=700,
        width=1200

    )
    fig.show()

# Example usage:
# calls_df, puts_df = get_all_option_data("GOOGL", 0.05)
# plot_volatility_surface(calls_df, symbol="GOOGL")