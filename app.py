import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import gzip
from datetime import datetime, time as dtime
import pytz
from dataclasses import dataclass
from typing import List, Optional

st.set_page_config(page_title="Multi-Timeframe Momentum Scanner", layout="wide", page_icon="📈")

IST = pytz.timezone("Asia/Kolkata")
BASE_URL = "https://api.upstox.com/v2"

FNO_STOCKS = [
    'RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','SBIN','BHARTIARTL','KOTAKBANK',
    'LT','HINDUNILVR','AXISBANK','MARUTI','SUNPHARMA','BAJFINANCE','TATASTEEL',
    'WIPRO','TECHM','HCLTECH','POWERGRID','NTPC','ONGC','COALINDIA','JSWSTEEL',
    'HINDALCO','TATACONSUM','DRREDDY','CIPLA','BAJAJFINSV','ULTRACEMCO','TITAN',
    'NESTLEIND','DIVISLAB','ASIANPAINT','INDUSINDBK','GRASIM','ADANIENT','ADANIPORTS',
    'TATAPOWER','ITC','M&M','HEROMOTOCO','EICHERMOT','TATAMOTORS','BAJAJ-AUTO',
    'BPCL','IOC','GAIL','VEDL','HINDZINC','APOLLOHOSP','PIDILITIND','NAUKRI',
    'MCDOWELL-N','AUROPHARMA','HAVELLS','VOLTAS','CUMMINSIND','AMBUJACEM','ACC',
    'BANKBARODA','PNB','CANBK','FEDERALBNK','IDFCFIRSTB','RBLBANK','INDHOTEL',
    'CHOLAFIN','MUTHOOTFIN','SBILIFE','HDFCLIFE','ICICIGI'
]

# ─── Token ────────────────────────────────────────────────────────────────────
def get_token() -> str:
    try:
        return st.secrets["upstox"]["access_token"]
    except Exception:
        return ""

# ─── Data class ───────────────────────────────────────────────────────────────
@dataclass
class ScannerRow:
    symbol: str
    ltp: float
    chg_pct: float
    vol_today: float
    vol_avg_10d: float
    score: int
    added_at: str

# ─── Upstox API client ────────────────────────────────────────────────────────
class UpstoxClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        self.instrument_map: dict = {}

    def _get(self, url: str, params: dict = None):
        r = requests.get(url, headers=self.headers, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def load_instruments(self):
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = json.loads(gzip.decompress(r.content))
        for item in data:
            sym   = str(item.get('trading_symbol', ''))
            seg   = str(item.get('segment', ''))
            itype = str(item.get('instrument_type', ''))
            key   = str(item.get('instrument_key', ''))
            if sym in FNO_STOCKS and seg == 'NSE_EQ' and itype == 'EQ' and key:
                self.instrument_map[sym] = key
        if not self.instrument_map:
            raise RuntimeError("No FnO symbols matched in instrument file.")
        return self.instrument_map

    def _candles_to_df(self, candles: list) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame(columns=['ts','open','high','low','close','volume','oi'])
        col_names = ['ts','open','high','low','close','volume','oi'][:len(candles[0])]
        df = pd.DataFrame(candles, columns=col_names)
        df['ts'] = pd.to_datetime(df['ts'])
        for c in ['open','high','low','close','volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.sort_values('ts').reset_index(drop=True)

    def get_historical(self, instrument_key: str, interval: str, to_date: str, from_date: str) -> pd.DataFrame:
        url = f"{BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        try:
            data = self._get(url)
            return self._candles_to_df(data.get('data', {}).get('candles', []))
        except Exception:
            return pd.DataFrame()

    def get_intraday(self, instrument_key: str, interval: str = '1minute') -> pd.DataFrame:
        url = f"{BASE_URL}/historical-candle/intraday/{instrument_key}/{interval}"
        try:
            data = self._get(url)
            return self._candles_to_df(data.get('data', {}).get('candles', []))
        except Exception:
            return pd.DataFrame()

# ─── Indicators ───────────────────────────────────────────────────────────────
def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 10) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(0)

def to_ist(ts_series: pd.Series) -> pd.Series:
    if ts_series.dt.tz is None:
        return ts_series.dt.tz_localize('UTC').dt.tz_convert(IST)
    return ts_series.dt.tz_convert(IST)

def market_phase(now_ist: datetime) -> str:
    t = now_ist.time()
    if t < dtime(9, 15):   return 'Pre-Open'
    if t < dtime(9, 20):   return 'Monitoring'
    if t <= dtime(15, 30): return 'Scanning'
    return 'Closed'

# ─── Caching System ───────────────────────────────────────────────────────────
def build_historical_cache(client: UpstoxClient, symbols: List[str]):
    now = datetime.now(IST)
    today_str = now.strftime('%Y-%m-%d')
    from_daily_str = (now - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
    yesterday_str = (now - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    from_1m_str = (now - pd.Timedelta(days=6)).strftime('%Y-%m-%d')

    bar = st.progress(0, text='Fetching Historical Data (Daily Volumes & Past Candles)...')
    for i, sym in enumerate(symbols):
        key = client.instrument_map[sym]
        
        # 1. Get 10-day average volume
        df_daily = client.get_historical(key, 'day', today_str, from_daily_str)
        if not df_daily.empty:
            df_daily['ts_ist'] = to_ist(df_daily['ts'])
            df_daily = df_daily[df_daily['ts_ist'].dt.date < now.date()]
            avg_vol = df_daily['volume'].tail(10).mean() if len(df_daily) >= 10 else df_daily['volume'].mean()
            st.session_state.daily_vols[sym] = avg_vol
        else:
            st.session_state.daily_vols[sym] = 0

        # 2. Get past 5 days of 1-min data for accurate RSI & EMA seeding
        df_1m_hist = client.get_historical(key, '1minute', yesterday_str, from_1m_str)
        st.session_state.hist_1m[sym] = df_1m_hist

        bar.progress((i + 1) / len(symbols))
        time.sleep(0.01)
    bar.empty()

# ─── Per-symbol Scanner ───────────────────────────────────────────────────────
def scan_symbol(client: UpstoxClient, sym: str):
    key = client.instrument_map[sym]
    
    # 1. Fetch live intraday 1-min data
    df_live = client.get_intraday(key, '1minute')
    if df_live.empty:
        return None, None
        
    df_live['ts'] = to_ist(df_live['ts'])
    today_vol = float(df_live['volume'].sum())
    avg_vol_10d = st.session_state.daily_vols.get(sym, 0)

    # 2. Extract today's first 15m high/low
    morning_mask = (df_live['ts'].dt.time >= dtime(9, 15)) & (df_live['ts'].dt.time < dtime(9, 30))
    morning_df = df_live[morning_mask]
    high15 = morning_df['high'].max() if not morning_df.empty else np.inf
    low15 = morning_df['low'].min() if not morning_df.empty else -np.inf

    # 3. Stitch live data with cached history
    df_hist = st.session_state.hist_1m.get(sym, pd.DataFrame())
    if not df_hist.empty and df_hist['ts'].dt.tz is None:
        df_hist['ts'] = to_ist(df_hist['ts'])
        
    df_all = pd.concat([df_hist, df_live]).drop_duplicates(subset='ts').sort_values('ts')
    df_all.set_index('ts', inplace=True)

    if df_all.empty:
        return None, None

    # 4. Generate 5-min timeframe (EMA & VWAP)
    df_5m = df_all.resample('5min').agg({
        'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
    }).dropna()
    
    df_5m['ema9'] = calc_ema(df_5m['close'], 9)
    df_5m['ema21'] = calc_ema(df_5m['close'], 21)
    
    df_5m['date'] = df_5m.index.date
    df_5m['typical'] = (df_5m['high'] + df_5m['low'] + df_5m['close']) / 3
    df_5m['vwap'] = (df_5m['typical'] * df_5m['volume']).groupby(df_5m['date']).cumsum() / df_5m['volume'].groupby(df_5m['date']).cumsum()

    # 5. Generate 15-min timeframe (RSI)
    df_15m = df_all.resample('15min').agg({'close':'last'}).dropna()
    df_15m['rsi10'] = calc_rsi(df_15m['close'], 10)
    df_15m['rsi_sma14'] = df_15m['rsi10'].rolling(14).mean()

    # 6. Evaluate latest conditions
    last_5m = df_5m.iloc[-1]
    prev_5m = df_5m.iloc[-2] if len(df_5m) > 1 else last_5m
    last_15m = df_15m.iloc[-1]

    close = float(last_5m['close'])
    prev_close = float(prev_5m['close'])
    chg_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
    
    e9 = last_5m['ema9']
    e21 = last_5m['ema21']
    vwap = last_5m['vwap']
    rsi = last_15m['rsi10']
    sma = last_15m['rsi_sma14']
    ts_str = df_5m.index[-1].strftime('%H:%M')

    # Calculate Bullish Score (+1 per condition)
    bull_score = sum([
        bool(close > high15),
        bool(e9 > e21) if pd.notna(e9) and pd.notna(e21) else False,
        bool(rsi > sma) if pd.notna(rsi) and pd.notna(sma) else False,
        bool(today_vol > avg_vol_10d),
        bool(close > vwap) if pd.notna(vwap) else False
    ])

    # Calculate Bearish Score (+1 per condition)
    bear_score = sum([
        bool(close < low15),
        bool(e9 < e21) if pd.notna(e9) and pd.notna(e21) else False,
        bool(rsi < sma) if pd.notna(rsi) and pd.notna(sma) else False,
        bool(today_vol > avg_vol_10d),
        bool(close < vwap) if pd.notna(vwap) else False
    ])

    bull = ScannerRow(
        symbol=sym, ltp=close, chg_pct=chg_pct,
        vol_today=today_vol, vol_avg_10d=avg_vol_10d,
        score=bull_score, added_at=ts_str
    ) if bull_score >= 2 else None

    bear = ScannerRow(
        symbol=sym, ltp=close, chg_pct=chg_pct,
        vol_today=today_vol, vol_avg_10d=avg_vol_10d,
        score=bear_score, added_at=ts_str
    ) if bear_score >= 2 else None

    return bull, bear

# ─── DataFrame Formatting ─────────────────────────────────────────────────────
def rows_to_df(rows: List[ScannerRow]) -> pd.DataFrame:
    cols = ['Symbol', 'LTP', 'Chg %', 'Score', 'Vol Today', '10D Avg Vol', 'Time']
    if not rows:
        return pd.DataFrame(columns=cols)
        
    data = [{
        'Symbol': r.symbol,
        'LTP': f"₹{r.ltp:.2f}",
        'Chg %': f"{'+' if r.chg_pct >= 0 else ''}{r.chg_pct:.2f}%",
        'Score': f"{r.score} / 5",
        'Vol Today': f"{int(r.vol_today):,}",
        '10D Avg Vol': f"{int(r.vol_avg_10d):,}",
        'Time': r.added_at,
        '_raw_score': r.score,
        '_raw_chg': r.chg_pct
    } for r in rows]
    
    # Sort primarily by raw Score descending, secondarily by Chg % descending
    df = pd.DataFrame(data).sort_values(['_raw_score', '_raw_chg'], ascending=[False, False])
    return df.drop(columns=['_raw_score', '_raw_chg'])

# ─── Session State ────────────────────────────────────────────────────────────
for key, default in [
    ('logs', []), ('daily_vols', {}), ('hist_1m', {}),
    ('instrument_map', {}), ('bull_rows', []), ('bear_rows', []),
    ('last_scan', None), ('scan_count', 0), ('cache_built', False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 0.5rem; max-width: 97%;}
[data-testid='stMetric'] { background: #0f172a; border: 1px solid #1e293b; padding: 12px 16px; border-radius: 12px; }
[data-testid='stMetricLabel'] {font-size: 12px; color: #94a3b8;}
[data-testid='stMetricValue'] {font-size: 20px; font-weight: 700;}
.stDataFrame thead tr th { font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.04em; }
.stDataFrame tbody tr td { font-size: 13px !important; font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

# ─── Header & Auth ────────────────────────────────────────────────────────────
st.markdown("## 📈 Multi-Timeframe Momentum Scanner (Scored)")
st.caption("Scoring out of 5: 15m Breakout (+1) · 5m EMA9>21 (+1) · 15m RSI>SMA (+1) · Vol > 10D Avg (+1) · 5m VWAP (+1)")

access_token = get_token()
if not access_token or access_token == "YOUR_UPSTOX_ACCESS_TOKEN_HERE":
    st.error("🔑 Access token missing. Please add it to `.streamlit/secrets.toml`.")
    st.stop()

client = UpstoxClient(access_token)

# ─── Build Cache ──────────────────────────────────────────────────────────────
if not st.session_state.instrument_map:
    with st.spinner("Loading NSE_EQ instrument master..."):
        try:
            st.session_state.instrument_map = client.load_instruments()
        except Exception as e:
            st.error(f"Instrument load failed: {e}")
            st.stop()
client.instrument_map = st.session_state.instrument_map
symbols = [s for s in FNO_STOCKS if s in client.instrument_map]

if not st.session_state.cache_built:
    build_historical_cache(client, symbols)
    st.session_state.cache_built = True

# ─── Scanner Loop ─────────────────────────────────────────────────────────────
def do_scan():
    bull_rows, bear_rows, scan_logs = [], [], []
    bar = st.progress(0, text="Running Intraday Scan…")
    for i, sym in enumerate(symbols):
        try:
            bull, bear = scan_symbol(client, sym)
            if bull: bull_rows.append(bull)
            if bear: bear_rows.append(bear)
        except Exception as e:
            scan_logs.append(f"{sym}: {e}")
        bar.progress((i + 1) / len(symbols))
        time.sleep(0.01)
    bar.empty()
    
    now = datetime.now(IST)
    st.session_state.bull_rows = bull_rows
    st.session_state.bear_rows = bear_rows
    st.session_state.last_scan = now.strftime('%H:%M:%S')
    st.session_state.scan_count += 1
    st.session_state.logs = [f"[{now.strftime('%H:%M:%S')}] Scan #{st.session_state.scan_count} complete"] + scan_logs[:20]

do_scan()

# ─── Status Bar ───────────────────────────────────────────────────────────────
now_ist = datetime.now(IST)
phase = market_phase(now_ist)

h1, h2, h3, h4 = st.columns(4)
with h1: st.metric("Market Phase", phase)
with h2: st.metric("IST Time", now_ist.strftime('%H:%M:%S'))
with h3: st.metric("Scans Run", st.session_state.scan_count)
with h4: st.metric("Last Scan", st.session_state.last_scan or "—")

# ─── Tables ───────────────────────────────────────────────────────────────────
bull_df = rows_to_df(st.session_state.bull_rows)
bear_df = rows_to_df(st.session_state.bear_rows)

def render_scored_tables(df: pd.DataFrame, title: str, is_bullish: bool):
    st.markdown(f"### {'🟢' if is_bullish else '🔴'} {title}")
    
    df_5 = df[df['Score'] == '5 / 5'] if not df.empty else pd.DataFrame()
    df_4 = df[df['Score'] == '4 / 5'] if not df.empty else pd.DataFrame()
    
    st.markdown("#### Perfect Setups (Score: 5 / 5)")
    if df_5.empty:
        st.info("No setups currently with a perfect 5 / 5 score.")
    else:
        st.dataframe(df_5, use_container_width=True, hide_index=True)

    st.markdown("#### Strong Setups (Score: 4 / 5)")
    if df_4.empty:
        st.info("No setups currently with a 4 / 5 score.")
    else:
        st.dataframe(df_4, use_container_width=True, hide_index=True)

render_scored_tables(bull_df, "Bullish Momentum", True)
st.markdown("---")
render_scored_tables(bear_df, "Bearish Momentum", False)

with st.expander("📋 Scan Log", expanded=False):
    for line in st.session_state.logs[:30]:
        st.caption(line)

# ─── Auto Refresh ─────────────────────────────────────────────────────────────
if phase not in ('Scanning', 'Monitoring'):
    st.warning(f"Market is **{phase}**. Scanner will wait for market hours.")

placeholder = st.empty()
for remaining in range(60, 0, -1):
    placeholder.markdown(
        f"<div style='color:#64748b;font-size:13px;padding:6px 0;text-align:center;'>"
        f"🔄 Auto refreshing in <b>{remaining}s</b>"
        f"</div>",
        unsafe_allow_html=True
    )
    time.sleep(1)
placeholder.empty()
st.rerun()
