"""
Indian Financial Analyzer — Streamlit Web App
=============================================
Run locally : streamlit run app.py
Deploy free : streamlit.io/cloud  →  unique public URL
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for web
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import urllib.request, json, urllib.parse

# ─── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Indian Financial Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0d1117; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 10px 14px;
}
.stop-box {
    background: rgba(255,149,0,0.08);
    border: 1px solid #ff9500;
    border-radius: 10px;
    padding: 16px 18px;
    line-height: 1.9;
}
.leg-box {
    background: rgba(121,192,255,0.08);
    border: 1px solid #79c0ff;
    border-radius: 10px;
    padding: 16px 18px;
    line-height: 1.9;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────
WINDOW     = 30    # rolling window in trading days
LEG_WINDOW = 10    # swing pivot lookback/lookahead

INDIAN_COMPANIES = {
    "TCS":                "TCS.NS",
    "INFOSYS":            "INFY.NS",
    "WIPRO":              "WIPRO.NS",
    "HCL TECH":           "HCLTECH.NS",
    "TECH MAHINDRA":      "TECHM.NS",
    "MPHASIS":            "MPHASIS.NS",
    "HDFC BANK":          "HDFCBANK.NS",
    "ICICI BANK":         "ICICIBANK.NS",
    "SBI":                "SBIN.NS",
    "KOTAK BANK":         "KOTAKBANK.NS",
    "AXIS BANK":          "AXISBANK.NS",
    "BAJAJ FINANCE":      "BAJFINANCE.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "ITC":                "ITC.NS",
    "NESTLE INDIA":       "NESTLEIND.NS",
    "BRITANNIA":          "BRITANNIA.NS",
    "SUN PHARMA":         "SUNPHARMA.NS",
    "DR REDDY'S":         "DRREDDY.NS",
    "CIPLA":              "CIPLA.NS",
    "DIVI'S LAB":         "DIVISLAB.NS",
    "MARUTI SUZUKI":      "MARUTI.NS",
    "TATA MOTORS":        "TATAMOTORS.NS",
    "M&M":                "M&M.NS",
    "HERO MOTOCORP":      "HEROMOTOCO.NS",
    "BAJAJ AUTO":         "BAJAJ-AUTO.NS",
    "RELIANCE":           "RELIANCE.NS",
    "ONGC":               "ONGC.NS",
    "NTPC":               "NTPC.NS",
    "POWER GRID":         "POWERGRID.NS",
    "ADANI PORTS":        "ADANIPORTS.NS",
    "TATA STEEL":         "TATASTEEL.NS",
    "HINDALCO":           "HINDALCO.NS",
    "JSW STEEL":          "JSWSTEEL.NS",
    "BHARTI AIRTEL":      "BHARTIARTL.NS",
    "ASIAN PAINTS":       "ASIANPAINT.NS",
    "TITAN":              "TITAN.NS",
    "DMART":              "DMART.NS",
}

# ─── Helpers ──────────────────────────────────────────────────────────────
def safe_div(a, b, fallback=np.nan):
    try:
        return a / b if b and b != 0 else fallback
    except Exception:
        return fallback

def fmt(val, decimals=2, prefix="₹"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1e12: return f"{prefix}{v/1e12:.{decimals}f}T"
        if abs(v) >= 1e9:  return f"{prefix}{v/1e9:.{decimals}f}B"
        if abs(v) >= 1e6:  return f"{prefix}{v/1e6:.{decimals}f}M"
        return f"{prefix}{v:,.{decimals}f}"
    except Exception:
        return "N/A"

# ─── Yahoo Finance live search ─────────────────────────────────────────────
def search_yahoo(query: str) -> list:
    try:
        q = urllib.parse.quote(query)
        url = (f"https://query2.finance.yahoo.com/v1/finance/search"
               f"?q={q}&quotesCount=10&newsCount=0&enableFuzzyQuery=true")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        results = []
        for item in data.get("quotes", []):
            if item.get("quoteType") in ("EQUITY", "ETF", "MUTUALFUND", "INDEX"):
                results.append({
                    "symbol":   item.get("symbol", ""),
                    "name":     item.get("longname") or item.get("shortname", "—"),
                    "exchange": item.get("exchange", ""),
                    "type":     item.get("quoteType", ""),
                })
        return results
    except Exception:
        return []

# ─── Data fetching (cached for 5 min) ─────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(ticker: str, period_years: int = 3) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=365 * period_years)
    df    = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info or {}
    try:    income   = t.financials
    except: income   = pd.DataFrame()
    try:    balance  = t.balance_sheet
    except: balance  = pd.DataFrame()
    try:    cashflow = t.cashflow
    except: cashflow = pd.DataFrame()
    return {"info": info, "income": income, "balance": balance, "cashflow": cashflow}

# ─── Rolling statistics ────────────────────────────────────────────────────
def compute_rolling_stats(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["Close"].squeeze()
    df = pd.DataFrame({"Close": close})
    df["RollingMed"] = df["Close"].rolling(WINDOW).median()
    df["RollingStd"] = df["Close"].rolling(WINDOW).std()
    df["Upper2"]     = df["RollingMed"] + 2 * df["RollingStd"]
    df["Lower2"]     = df["RollingMed"] - 2 * df["RollingStd"]
    df["Alert"]      = (df["Close"] > df["Upper2"]) | (df["Close"] < df["Lower2"])
    return df

# ─── Rally leg detection — ZigZag method ──────────────────────────────────
def detect_rally_legs(df: pd.DataFrame, reversal_pct: float = 5.0) -> dict:
    """
    ZigZag-based rally leg detector.

    Instead of looking N bars ahead/behind (which produces huge multi-month
    legs on long charts), this tracks price direction and only confirms a new
    pivot when price REVERSES by at least `reversal_pct` % from the last
    confirmed peak or trough.

    reversal_pct guide:
        3 %  →  very sensitive — short 2-4 week legs  (tight stops)
        5 %  →  moderate       — 4-8 week legs         (good default)
        8 %  →  relaxed        — 2-4 month legs
       12 %  →  major swings   — multi-month legs only
    """
    close = df["Close"].squeeze()
    highs = df["High"].squeeze() if "High" in df.columns else close
    lows  = df["Low"].squeeze()  if "Low"  in df.columns else close
    dates = df.index
    n     = len(close)

    if n < 20:
        return {"swing_lows": pd.DataFrame(), "swing_highs": pd.DataFrame(),
                "legs": [], "prev_leg_low": None,
                "current_leg_low": None, "current_leg_high": None,
                "pivots": []}

    # ── State machine ────────────────────────────────────────────────────
    UP, DOWN = 1, -1
    direction    = None
    peak_idx     = 0;  peak_price   = float(highs.iloc[0])
    trough_idx   = 0;  trough_price = float(lows.iloc[0])
    pivots       = []   # confirmed pivots: (date, 'H'/'L', price)

    for i in range(1, n):
        h = float(highs.iloc[i])
        l = float(lows.iloc[i])

        if direction is None:
            # Bootstrap: find which way the market first moves meaningfully
            if h >= float(lows.iloc[0]) * (1 + reversal_pct / 100):
                direction  = UP
                peak_idx   = i;  peak_price = h
            elif l <= float(highs.iloc[0]) * (1 - reversal_pct / 100):
                direction   = DOWN
                trough_idx  = i;  trough_price = l
            continue

        if direction == UP:
            if h >= peak_price:                          # new local high
                peak_idx = i;  peak_price = h
            elif l <= peak_price * (1 - reversal_pct / 100):   # reversal down
                pivots.append((dates[peak_idx], "H", peak_price))
                direction   = DOWN
                trough_idx  = i;  trough_price = l

        else:  # DOWN
            if l <= trough_price:                         # new local low
                trough_idx = i;  trough_price = l
            elif h >= trough_price * (1 + reversal_pct / 100):  # reversal up
                pivots.append((dates[trough_idx], "L", trough_price))
                direction  = UP
                peak_idx   = i;  peak_price = h

    # Add the final in-progress pivot
    if direction == UP:
        pivots.append((dates[peak_idx],   "H", peak_price))
    elif direction == DOWN:
        pivots.append((dates[trough_idx], "L", trough_price))

    # ── Build legs (L → H pairs) ─────────────────────────────────────────
    legs = []
    for i in range(len(pivots) - 1):
        d0, t0, p0 = pivots[i]
        d1, t1, p1 = pivots[i + 1]
        if t0 == "L" and t1 == "H":
            legs.append({
                "low_date":  d0, "low_price":  p0,
                "high_date": d1, "high_price": p1,
                "gain_pct":  (p1 - p0) / p0 * 100,
            })

    # ── Swing-low / swing-high DataFrames (for chart markers) ────────────
    low_dates  = [d for d, t, _ in pivots if t == "L"]
    high_dates = [d for d, t, _ in pivots if t == "H"]
    low_prices  = [p for _, t, p in pivots if t == "L"]
    high_prices = [p for _, t, p in pivots if t == "H"]

    swing_lows  = pd.DataFrame({"Low":  low_prices},  index=low_dates)
    swing_highs = pd.DataFrame({"High": high_prices}, index=high_dates)

    # ── Stop-loss levels ─────────────────────────────────────────────────
    prev_leg_low = curr_leg_low = curr_leg_high = None

    if legs:
        prev_leg_low   = legs[-1]          # most recent completed leg low = stop reference
        last_confirmed = pivots[-1]

        if last_confirmed[1] == "L":
            # Currently IN an upleg — started at this trough
            curr_leg_low  = {"date": last_confirmed[0], "price": last_confirmed[2]}
            mask          = close.index >= last_confirmed[0]
            curr_leg_high = float(close[mask].max()) if mask.any() else last_confirmed[2]
        else:
            # Completed a leg, now pulling back — use most recent trough
            lows_list = [(d, p) for d, t, p in pivots if t == "L"]
            if lows_list:
                ld, lp        = lows_list[-1]
                curr_leg_low  = {"date": ld, "price": lp}
                mask          = close.index >= ld
                curr_leg_high = float(close[mask].max()) if mask.any() else lp

    return {
        "swing_lows":      swing_lows,
        "swing_highs":     swing_highs,
        "legs":            legs,
        "prev_leg_low":    prev_leg_low,
        "current_leg_low": curr_leg_low,
        "current_leg_high":curr_leg_high,
        "pivots":          pivots,
    }

# ─── Beneish M-Score ──────────────────────────────────────────────────────
def _row(df, *candidates):
    for c in candidates:
        if c in df.index:
            return df.loc[c]
    return pd.Series(dtype=float)

def compute_beneish(financials: dict) -> dict:
    blank = {k: np.nan for k in ["DSRI","GMI","AQI","SGI","DEPI","SGAI","LVGI","TATA","M_Score"]}
    blank["verdict"] = "Insufficient data"

    inc = financials["income"]
    bal = financials["balance"]
    cf  = financials["cashflow"]
    if inc.empty or bal.empty or len(inc.columns) < 2:
        return blank

    yr_t, yr_t1 = inc.columns[0], inc.columns[1]

    def g(df, *keys): return _row(df, *keys)

    rev_t  = g(inc,"Total Revenue","Revenue").get(yr_t,  np.nan)
    rev_t1 = g(inc,"Total Revenue","Revenue").get(yr_t1, np.nan)
    cogs_t  = g(inc,"Cost Of Revenue","Cost of Revenue").get(yr_t,  np.nan)
    cogs_t1 = g(inc,"Cost Of Revenue","Cost of Revenue").get(yr_t1, np.nan)
    sga_t   = g(inc,"Selling General Administrative","Selling General And Administrative").get(yr_t,  np.nan)
    sga_t1  = g(inc,"Selling General Administrative","Selling General And Administrative").get(yr_t1, np.nan)
    dep_t   = g(inc,"Reconciled Depreciation","Depreciation","Depreciation And Amortization").get(yr_t,  np.nan)
    dep_t1  = g(inc,"Reconciled Depreciation","Depreciation","Depreciation And Amortization").get(yr_t1, np.nan)
    ni_t    = g(inc,"Net Income","Net Income Common Stockholders").get(yr_t, np.nan)
    rec_t   = g(bal,"Receivables","Net Receivables","Accounts Receivable").get(yr_t,  np.nan)
    rec_t1  = g(bal,"Receivables","Net Receivables","Accounts Receivable").get(yr_t1, np.nan)
    ta_t    = g(bal,"Total Assets").get(yr_t,  np.nan)
    ta_t1   = g(bal,"Total Assets").get(yr_t1, np.nan)
    ca_t    = g(bal,"Current Assets","Total Current Assets").get(yr_t,  np.nan)
    ca_t1   = g(bal,"Current Assets","Total Current Assets").get(yr_t1, np.nan)
    ppe_t   = g(bal,"Net PPE","Property Plant Equipment Net","Net Property Plant And Equipment").get(yr_t,  np.nan)
    ppe_t1  = g(bal,"Net PPE","Property Plant Equipment Net","Net Property Plant And Equipment").get(yr_t1, np.nan)
    ltd_t   = g(bal,"Long Term Debt","Long-Term Debt").get(yr_t,  np.nan)
    ltd_t1  = g(bal,"Long Term Debt","Long-Term Debt").get(yr_t1, np.nan)
    cl_t    = g(bal,"Current Liabilities","Total Current Liabilities").get(yr_t,  np.nan)
    cl_t1   = g(bal,"Current Liabilities","Total Current Liabilities").get(yr_t1, np.nan)
    cfo_t   = g(cf, "Operating Cash Flow","Cash From Operations").get(yr_t, np.nan)

    dsri = safe_div(safe_div(rec_t, rev_t), safe_div(rec_t1, rev_t1))
    gmi  = safe_div(safe_div(rev_t1-cogs_t1, rev_t1), safe_div(rev_t-cogs_t, rev_t))
    nca_t  = ta_t -ca_t -ppe_t  if not any(np.isnan(x) for x in [ta_t, ca_t, ppe_t])   else np.nan
    nca_t1 = ta_t1-ca_t1-ppe_t1 if not any(np.isnan(x) for x in [ta_t1,ca_t1,ppe_t1])  else np.nan
    aqi  = safe_div(safe_div(nca_t, ta_t), safe_div(nca_t1, ta_t1))
    sgi  = safe_div(rev_t, rev_t1)
    depi = safe_div(safe_div(dep_t1, dep_t1+ppe_t1), safe_div(dep_t, dep_t+ppe_t))
    sgai = safe_div(safe_div(sga_t, rev_t), safe_div(sga_t1, rev_t1))
    lvgi = safe_div(safe_div(ltd_t+cl_t, ta_t), safe_div(ltd_t1+cl_t1, ta_t1))
    tata = safe_div(ni_t - cfo_t, ta_t)

    present = [v for v in [dsri,gmi,aqi,sgi,depi,sgai,lvgi,tata] if not np.isnan(v)]
    if len(present) >= 5:
        def n(v): return v if not np.isnan(v) else 0
        m = (-4.84 + 0.920*n(dsri) + 0.528*n(gmi) + 0.404*n(aqi) + 0.892*n(sgi)
             + 0.115*n(depi) - 0.172*n(sgai) + 4.679*n(tata) - 0.327*n(lvgi))
        verdict = ("⚠️ LIKELY MANIPULATION (M > −2.22)" if m > -2.22
                   else "✅ Low manipulation risk (M ≤ −2.22)")
    else:
        m, verdict = np.nan, "Insufficient data for reliable score"

    return {"DSRI":dsri,"GMI":gmi,"AQI":aqi,"SGI":sgi,"DEPI":depi,
            "SGAI":sgai,"LVGI":lvgi,"TATA":tata,"M_Score":m,"verdict":verdict}

# ─── Chart ────────────────────────────────────────────────────────────────
def make_chart(sd_df, prices, legs_data, company, ticker):
    alerts     = sd_df[sd_df["Alert"]]
    legs       = legs_data["legs"]
    prev_leg   = legs_data["prev_leg_low"]
    curr_leg   = legs_data["current_leg_low"]
    sw_lows    = legs_data["swing_lows"]
    sw_highs   = legs_data["swing_highs"]

    fig, ax = plt.subplots(figsize=(16, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    # Shaded rally legs
    for i, leg in enumerate(legs):
        mask = (sd_df.index >= leg["low_date"]) & (sd_df.index <= leg["high_date"])
        if mask.any():
            ax.fill_between(sd_df.index, sd_df["Close"].where(mask),
                            alpha=0.07, color="#a371f7",
                            label="Rally Leg" if i == 0 else "")

    # Price line + bands
    ax.plot(sd_df.index, sd_df["Close"],     color="#58a6ff", lw=1.3, label="Close Price", zorder=3)
    ax.plot(sd_df.index, sd_df["RollingMed"],color="#f0e68c", lw=1.0, ls="--",
            label=f"{WINDOW}-day Rolling Median", zorder=3)
    ax.fill_between(sd_df.index, sd_df["Upper2"], sd_df["Lower2"],
                    alpha=0.13, color="#3fb950", label="±2σ Band")
    ax.plot(sd_df.index, sd_df["Upper2"], color="#3fb950", lw=0.7, ls=":")
    ax.plot(sd_df.index, sd_df["Lower2"], color="#3fb950", lw=0.7, ls=":")

    # Alert dots
    if len(alerts):
        ax.scatter(alerts.index, alerts["Close"],
                   color="#f85149", s=30, zorder=5, label=f"⚠ 2σ Alert ({len(alerts)})")

    # Swing markers — ZigZag pivots
    if not sw_lows.empty and "Low" in sw_lows.columns:
        ax.scatter(sw_lows.index, sw_lows["Low"].squeeze(),
                   marker="^", color="#3fb950", s=60, zorder=6, label="Swing Low ▲")
    if not sw_highs.empty and "High" in sw_highs.columns:
        ax.scatter(sw_highs.index, sw_highs["High"].squeeze(),
                   marker="v", color="#f85149", s=60, zorder=6, label="Swing High ▼")

    # Connect pivots with a thin ZigZag line for clarity
    all_pivots = legs_data.get("pivots", [])
    if all_pivots:
        zz_dates  = [p[0] for p in all_pivots]
        zz_prices = [p[2] for p in all_pivots]
        ax.plot(zz_dates, zz_prices, color="#a371f7", lw=0.8,
                ls="-", alpha=0.5, zorder=4, label="ZigZag")

    # Stop-loss horizontal line
    if prev_leg:
        sl = prev_leg["low_price"]
        ax.axhline(sl, color="#ff9500", lw=1.6, ls="--", zorder=7,
                   label=f"Stop-Loss Zone  ₹{sl:,.2f}")
        ax.annotate(
            f"  STOP-LOSS  ₹{sl:,.2f}",
            xy=(sd_df.index[-1], sl), fontsize=8.5, color="#ff9500", va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#1a1200",
                      edgecolor="#ff9500", alpha=0.92),
        )

    # Current leg low dotted line
    if curr_leg:
        cl_p = curr_leg["price"]
        ax.axhline(cl_p, color="#79c0ff", lw=1.0, ls=":", zorder=7,
                   label=f"Current Leg Low  ₹{cl_p:,.2f}")

    ax.set_title(
        f"{company}  ({ticker})  —  Price · {WINDOW}-day σ Bands · Rally Legs",
        color="#e6edf3", fontsize=13, pad=14,
    )
    ax.set_xlabel("Date", color="#8b949e")
    ax.set_ylabel("Price (₹)", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.grid(color="#21262d", lw=0.5, ls="--")
    ax.legend(facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", fontsize=8.5, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k in ["prices","financials","sd_df","legs_data","company","ticker","reversal_pct"]:
    if k not in st.session_state:
        st.session_state[k] = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Financial Analyzer")
    st.caption("Indian & Global Stocks · Yahoo Finance")
    st.divider()

    mode = st.radio(
        "Find a company:",
        ["🏢 Indian Presets", "🔍 Search Any Company", "⌨️ Direct Ticker"],
    )
    st.divider()

    ticker_to_use  = None
    company_to_use = None

    # ── Mode 1: Preset list ──────────────────────────────────────────────
    if mode == "🏢 Indian Presets":
        company_to_use = st.selectbox("Select company:", list(INDIAN_COMPANIES.keys()))
        ticker_to_use  = INDIAN_COMPANIES[company_to_use]

    # ── Mode 2: Search ───────────────────────────────────────────────────
    elif mode == "🔍 Search Any Company":
        query = st.text_input("Company name:", placeholder="e.g. Zomato, Apple, Samsung")
        if st.button("🔍 Search", use_container_width=True):
            with st.spinner("Searching Yahoo Finance…"):
                st.session_state.search_results = search_yahoo(query)

        if st.session_state.search_results:
            options = {
                f"{r['name']}  ({r['symbol']})  [{r['exchange']}]": r
                for r in st.session_state.search_results
            }
            sel_key = st.selectbox("Results:", list(options.keys()))
            if sel_key:
                r = options[sel_key]
                ticker_to_use  = r["symbol"]
                company_to_use = r["name"] or r["symbol"]
        elif query and st.session_state.search_results == []:
            st.warning("No results found. Try different keywords.")

    # ── Mode 3: Direct ticker ────────────────────────────────────────────
    else:
        st.caption("Examples: AAPL · RELIANCE.NS · TSLA · 005930.KS")
        raw = st.text_input("Ticker:", placeholder="RELIANCE.NS").strip().upper()
        if raw:
            ticker_to_use  = raw
            company_to_use = raw

    st.divider()

    # ── Rally leg sensitivity ────────────────────────────────────────────
    st.markdown("**📐 Rally Leg Sensitivity**")
    sensitivity_label = st.select_slider(
        "Reversal threshold",
        options=["Very Short (3%)", "Short (5%)", "Medium (8%)", "Long (12%)"],
        value="Short (5%)",
        label_visibility="collapsed",
    )
    REVERSAL_MAP = {
        "Very Short (3%)": 3.0,
        "Short (5%)":      5.0,
        "Medium (8%)":     8.0,
        "Long (12%)":     12.0,
    }
    reversal_pct = REVERSAL_MAP[sensitivity_label]
    st.caption({
        "Very Short (3%)": "2-4 week legs · tightest stops",
        "Short (5%)":      "4-8 week legs · good default",
        "Medium (8%)":     "2-3 month legs · positional trades",
        "Long (12%)":      "Major swings only · long-term view",
    }[sensitivity_label])

    st.divider()

    analyze_btn = st.button(
        "▶  Analyze",
        type="primary",
        use_container_width=True,
        disabled=(ticker_to_use is None),
    )

    if analyze_btn and ticker_to_use:
        with st.spinner(f"Fetching data for {company_to_use}…"):
            try:
                prices     = fetch_price_data(ticker_to_use)
                financials = fetch_financials(ticker_to_use)
                sd_df      = compute_rolling_stats(prices)

                # Build merged df for rally leg detection (needs High/Low)
                merged = sd_df.copy()
                for col in ["High", "Low"]:
                    if col in prices.columns:
                        merged[col] = prices[col].squeeze()
                legs_data = detect_rally_legs(merged, reversal_pct=reversal_pct)

                st.session_state.prices       = prices
                st.session_state.financials   = financials
                st.session_state.sd_df        = sd_df
                st.session_state.legs_data    = legs_data
                st.session_state.company      = company_to_use
                st.session_state.ticker       = ticker_to_use
                st.session_state.reversal_pct = reversal_pct
            except Exception as e:
                st.error(f"❌ {e}")
                st.session_state.prices = None

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.prices is None:
    # ── Welcome screen ───────────────────────────────────────────────────
    st.markdown("# 📈 Indian Financial Analyzer")
    st.markdown("Pick a company from the sidebar and click **▶ Analyze** to get started.")
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.info("**📊 SD Analysis**\n\nRolling 30-day median with ±2σ bands and automatic breach alerts")
    c2.info("**📈 Rally Legs**\n\nSwing high/low detection with stop-loss reference line")
    c3.info("**🔍 Beneish M-Score**\n\nEarnings manipulation probability — 8-variable model")
    st.divider()
    st.caption("Supports any stock worldwide via Yahoo Finance — Indian (NSE/BSE), US, European, Asian markets.")

else:
    prices     = st.session_state.prices
    financials = st.session_state.financials
    sd_df      = st.session_state.sd_df
    legs_data  = st.session_state.legs_data
    company    = st.session_state.company
    ticker     = st.session_state.ticker
    info       = financials["info"]

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(f"# {company}")
    col_h1, col_h2, col_h3 = st.columns([4, 1, 1])
    with col_h1:
        st.caption(
            f"**{info.get('sector','—')}** · {info.get('industry','—')} · "
            f"{info.get('exchange','—')} · `{ticker}`"
        )
    with col_h2:
        cp = info.get("currentPrice") or info.get("regularMarketPrice")
        if cp: st.metric("Price", f"₹{cp:,.2f}")
    with col_h3:
        chg = info.get("regularMarketChangePercent")
        if chg: st.metric("Today", f"{chg:+.2f}%", delta=f"{chg:+.2f}%")

    st.divider()

    # ── Company overview ─────────────────────────────────────────────────
    with st.expander("📋 Company Overview", expanded=True):
        c = st.columns(6)
        c[0].metric("Market Cap",   fmt(info.get("marketCap")))
        c[1].metric("P/E Ratio",    f'{info.get("trailingPE","N/A"):.2f}' if isinstance(info.get("trailingPE"), float) else "N/A")
        c[2].metric("EPS (TTM)",    fmt(info.get("trailingEps"), prefix="₹"))
        c[3].metric("52W High",     fmt(info.get("fiftyTwoWeekHigh"), prefix="₹"))
        c[4].metric("52W Low",      fmt(info.get("fiftyTwoWeekLow"),  prefix="₹"))
        c[5].metric("Beta",         f'{info.get("beta","N/A"):.2f}' if isinstance(info.get("beta"), float) else "N/A")
        c2 = st.columns(6)
        c2[0].metric("Revenue",     fmt(info.get("totalRevenue")))
        c2[1].metric("Net Income",  fmt(info.get("netIncomeToCommon")))
        c2[2].metric("ROE",         f'{info.get("returnOnEquity",0)*100:.1f}%' if info.get("returnOnEquity") else "N/A")
        c2[3].metric("ROA",         f'{info.get("returnOnAssets",0)*100:.1f}%' if info.get("returnOnAssets") else "N/A")
        c2[4].metric("Div Yield",   f'{info.get("dividendYield",0)*100:.2f}%' if info.get("dividendYield") else "N/A")
        c2[5].metric("Debt/Equity", f'{info.get("debtToEquity","N/A"):.2f}' if isinstance(info.get("debtToEquity"), float) else "N/A")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📊 Chart & SD Analysis",
        "📈 Rally Legs & Stop-Loss",
        "🔍 Beneish M-Score",
    ])

    # ── TAB 1 — Chart & SD ────────────────────────────────────────────────
    with tab1:
        latest_close = float(sd_df["Close"].iloc[-1])
        latest_med   = float(sd_df["RollingMed"].iloc[-1])
        latest_std   = float(sd_df["RollingStd"].iloc[-1])
        latest_upper = float(sd_df["Upper2"].iloc[-1])
        latest_lower = float(sd_df["Lower2"].iloc[-1])
        n_alerts     = int(sd_df["Alert"].sum())
        z_score      = safe_div(latest_close - latest_med, latest_std)

        mc = st.columns(6)
        mc[0].metric("Latest Close",    f"₹{latest_close:,.2f}")
        mc[1].metric("Rolling Median",  f"₹{latest_med:,.2f}")
        mc[2].metric("Std Dev (σ)",     f"₹{latest_std:,.2f}")
        mc[3].metric("Upper Band +2σ",  f"₹{latest_upper:,.2f}")
        mc[4].metric("Lower Band −2σ",  f"₹{latest_lower:,.2f}")
        mc[5].metric("Z-Score",         f"{z_score:.3f}" if not np.isnan(z_score) else "N/A")

        st.markdown("")
        if not np.isnan(z_score) and abs(z_score) > 2:
            direction = "ABOVE upper" if z_score > 0 else "BELOW lower"
            st.error(
                f"🚨 **Alert!** Price is {direction} band · "
                f"Z-Score = {z_score:.3f} · {n_alerts} breach days in period"
            )
        else:
            st.success(
                f"✅ Price is within normal ±2σ range · "
                f"Z-Score = {z_score:.3f} · {n_alerts} breach days in period"
            )

        with st.spinner("Rendering chart…"):
            fig = make_chart(sd_df, prices, legs_data, company, ticker)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── TAB 2 — Rally Legs ────────────────────────────────────────────────
    with tab2:
        legs         = legs_data["legs"]
        prev_leg     = legs_data["prev_leg_low"]
        curr_leg     = legs_data["current_leg_low"]
        curr_high    = legs_data["current_leg_high"]
        curr_price   = float(sd_df["Close"].iloc[-1])
        used_reversal = st.session_state.get("reversal_pct", 5.0)

        if not legs:
            st.info("Not enough price history to detect rally legs. Try a longer period or a more liquid stock.")
        else:
            st.caption(f"Detected using **{used_reversal:.0f}% ZigZag reversal** · "
                       f"{len(legs)} legs found · adjust sensitivity in the sidebar and re-analyze")
            st.markdown("#### Detected Rally Legs")
            leg_rows = []
            for i, leg in enumerate(legs[-8:], 1):
                leg_rows.append({
                    "Leg #":       i,
                    "Low Date":    leg["low_date"].strftime("%d %b %Y"),
                    "Low Price":   f"₹{leg['low_price']:,.2f}",
                    "High Date":   leg["high_date"].strftime("%d %b %Y"),
                    "High Price":  f"₹{leg['high_price']:,.2f}",
                    "Gain %":      f"+{leg['gain_pct']:.1f}%",
                })
            st.dataframe(pd.DataFrame(leg_rows), hide_index=True, use_container_width=True)

            st.markdown("")
            col_sl, col_cl = st.columns(2)

            with col_sl:
                if prev_leg:
                    sl       = prev_leg["low_price"]
                    sl_dt    = prev_leg["low_date"].strftime("%d %b %Y")
                    diff     = curr_price - sl
                    diff_pct = (diff / curr_price) * 100
                    risk_lbl = ("🔴 Low buffer — price close to stop" if diff_pct < 5
                                else "🟡 Moderate buffer" if diff_pct < 10
                                else "🟢 Safe buffer")
                    st.markdown("#### 🛑 Stop-Loss Reference")
                    st.markdown(f"""
<div class="stop-box">
<b>Previous Leg Low — Hard Stop-Loss Zone</b><br><br>
📅 &nbsp;Date &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : <b>{sl_dt}</b><br>
💰 &nbsp;Stop-Loss Price : <b>₹{sl:,.2f}</b><br>
📍 &nbsp;Current Price &nbsp; : ₹{curr_price:,.2f}<br>
📉 &nbsp;Distance &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : ₹{diff:,.2f}  ({diff_pct:.1f}% below current)<br>
⚡ &nbsp;Risk Level &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : {risk_lbl}<br><br>
<small>If price closes below <b>₹{sl:,.2f}</b>, the rally leg structure
is broken. Consider this your hard exit / sell signal.</small>
</div>""", unsafe_allow_html=True)

            with col_cl:
                if curr_leg:
                    cl_p      = curr_leg["price"]
                    cl_dt     = curr_leg["date"].strftime("%d %b %Y")
                    leg_gain  = (curr_price - cl_p) / cl_p * 100 if cl_p else 0
                    peak_gain = (curr_high - cl_p) / cl_p * 100 if cl_p else 0
                    sign      = "+" if leg_gain >= 0 else ""
                    st.markdown("#### 📈 Current Leg Status")
                    st.markdown(f"""
<div class="leg-box">
<b>Current Rally Leg (In Progress)</b><br><br>
📅 &nbsp;Started &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : <b>{cl_dt}</b><br>
🟢 &nbsp;Leg Low Price &nbsp; : <b>₹{cl_p:,.2f}</b><br>
🏔️ &nbsp;Peak So Far &nbsp;&nbsp; : ₹{curr_high:,.2f}  (+{peak_gain:.1f}%)<br>
📊 &nbsp;Current Gain &nbsp;&nbsp; : <b>{sign}{leg_gain:.1f}%</b> from leg low<br><br>
<small>Consider booking partial profits as price approaches
previous swing highs or the upper <b>+2σ</b> band.</small>
</div>""", unsafe_allow_html=True)

    # ── TAB 3 — Beneish M-Score ───────────────────────────────────────────
    with tab3:
        with st.spinner("Computing Beneish M-Score…"):
            beneish = compute_beneish(financials)

        var_info = {
            "DSRI":  ("Days Sales Receivable Index",   "Receivables growing faster than sales → possible revenue inflation"),
            "GMI":   ("Gross Margin Index",             ">1 = deteriorating margins → pressure to manipulate"),
            "AQI":   ("Asset Quality Index",            ">1 = rising non-current / intangible asset ratio"),
            "SGI":   ("Sales Growth Index",             "High growth firms face greater manipulation temptation"),
            "DEPI":  ("Depreciation Index",             ">1 = slowing depreciation rate → inflates reported assets"),
            "SGAI":  ("SG&A Expense Index",             ">1 = SG&A growing faster than revenue"),
            "LVGI":  ("Leverage Index",                 ">1 = rising debt burden"),
            "TATA":  ("Total Accruals to Total Assets", "High accruals vs assets → lower earnings quality"),
        }

        rows = []
        for key, (fname, interp) in var_info.items():
            val = beneish[key]
            flag = ""
            if not np.isnan(val):
                if key == "TATA" and val > 0.031:   flag = "⚠️"
                elif key != "TATA" and val > 1.0:   flag = "⚠️"
            rows.append({
                "Variable":       key,
                "Full Name":      fname,
                "Value":          f"{val:.4f} {flag}".strip() if not np.isnan(val) else "N/A",
                "Interpretation": interp,
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.divider()

        m     = beneish["M_Score"]
        m_str = f"{m:.4f}" if not np.isnan(m) else "N/A"

        col_m1, col_m2 = st.columns([2, 3])
        with col_m1:
            if not np.isnan(m):
                st.metric("M-Score", m_str,
                          delta="⚠ Manipulation Risk" if m > -2.22 else "✅ Low Risk",
                          delta_color="inverse" if m > -2.22 else "off")
            else:
                st.metric("M-Score", "N/A")
        with col_m2:
            if not np.isnan(m) and m > -2.22:
                st.error(f"⚠️ **{beneish['verdict']}**\n\nM-Score = {m_str} exceeds the −2.22 threshold.")
            elif not np.isnan(m):
                st.success(f"✅ **{beneish['verdict']}**\n\nM-Score = {m_str} is below the −2.22 threshold.")
            else:
                st.warning("Insufficient financial statement data. Works best for large-cap companies (TCS, Infosys, Reliance, HDFC Bank).")

        st.caption(
            "ℹ️ M-Score is a probabilistic indicator. "
            "Always combine with auditor reports, cash flow analysis, and qualitative research."
        )
