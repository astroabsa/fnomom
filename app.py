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

st.set_page_config(page_title="FnO Momentum Scanner", layout="wide", page_icon="📈")

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


# ─── Token from secrets.toml ─────────────────────────────────────────────────
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
    ref_level: float
    ema9: Optional[float]
    ema21: Optional[float]
    rsi10: Optional[float]
    rsi_sma14: Optional[float]
    f1: bool
    f2: bool
    f3: bool
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
        # Public CDN URL — NO auth header required (sending auth causes 401)
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        r = requests.get(url, timeout=60)   # plain request, no auth header
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
        for c in ['open','high','low','close']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.sort_values('ts').reset_index(drop=True)

    def get_intraday(self, instrument_key: str, interval: str = '1minute') -> pd.DataFrame:
        """Today's intraday candles via the dedicated intraday endpoint."""
        url = f"{BASE_URL}/historical-candle/intraday/{instrument_key}/{interval}"
        data = self._get(url)
        return self._candles_to_df(data.get('data', {}).get('candles', []))

    def get_historical(self, instrument_key: str, interval: str,
                       to_date: str, from_date: str) -> pd.DataFrame:
        """Historical candles for a completed past date range."""
        url = f"{BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        data = self._get(url)
        return self._candles_to_df(data.get('data', {}).get('candles', []))

    def get_candles_today(self, instrument_key: str,
                          interval: str = '1minute') -> pd.DataFrame:
        """
        Fetch today's candles.
        Tries intraday endpoint first (correct for current trading day),
        falls back to historical with today's date if intraday returns empty.
        """
        df = self.get_intraday(instrument_key, interval)
        if not df.empty:
            return df
        today = datetime.now(IST).strftime('%Y-%m-%d')
        return self.get_historical(instrument_key, interval, today, today)


# ─── Technical indicators ─────────────────────────────────────────────────────
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


# ─── Helpers ──────────────────────────────────────────────────────────────────
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

def seconds_to_next_candle(now_ist: datetime) -> int:
    return 60 - now_ist.second

def next_scan_at(now_ist: datetime) -> str:
    nxt = now_ist.replace(second=0, microsecond=0)
    nxt = nxt.replace(minute=nxt.minute + 1) if nxt.minute < 59           else nxt.replace(hour=nxt.hour + 1, minute=0)
    return nxt.strftime('%H:%M:%S')


# ─── Reference level builder ──────────────────────────────────────────────────
def build_reference_levels(client: UpstoxClient, symbols: List[str], trade_date: str):
    high15, low15, logs = {}, {}, []
    bar = st.progress(0, text='Building 15-minute reference levels...')
    for i, sym in enumerate(symbols):
        try:
            # IMPORTANT: use intraday endpoint for today's data.
            # The date-based historical endpoint does NOT return current-day candles.
            df = client.get_candles_today(client.instrument_map[sym], '1minute')
            if df.empty:
                logs.append(f'{sym}: no candle data returned')
                continue
            df = df.assign(ts_ist=to_ist(df['ts']))
            w = df[
                (df['ts_ist'].dt.time >= dtime(9, 15)) &
                (df['ts_ist'].dt.time <  dtime(9, 30))
            ]
            if w.empty:
                # fallback: use first available candles as proxy reference
                w = df.head(15)
                logs.append(f'{sym}: 15-min window empty, used first {len(w)} candles as proxy')
            high15[sym] = float(w['high'].max())
            low15[sym]  = float(w['low'].min())
        except Exception as e:
            logs.append(f'{sym}: {e}')
        bar.progress((i + 1) / len(symbols),
                     text=f'Reference levels: {i+1}/{len(symbols)}')
    bar.empty()
    return high15, low15, logs


# ─── Per-symbol scanner ───────────────────────────────────────────────────────
def scan_symbol(client: UpstoxClient, sym: str, trade_date: str,
                high15: dict, low15: dict):
    df = client.get_candles_today(client.instrument_map[sym], '1minute')
    if df.empty or len(df) < 5:
        return None, None

    close = df['close']
    df['ema9']      = calc_ema(close, 9)
    df['ema21']     = calc_ema(close, 21)
    df['rsi10']     = calc_rsi(close, 10)
    df['rsi_sma14'] = df['rsi10'].rolling(14).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_close = float(prev['close']) if pd.notna(prev['close']) else float(last['close'])
    chg_pct = ((float(last['close']) - prev_close) / prev_close * 100) if prev_close else 0

    ts_raw = pd.to_datetime(last['ts'])
    ts = ts_raw.tz_localize('UTC').tz_convert(IST) if ts_raw.tzinfo is None          else ts_raw.tz_convert(IST)
    ts_str = ts.strftime('%H:%M')

    e9, e21   = last['ema9'],     last['ema21']
    r10, rsma = last['rsi10'],    last['rsi_sma14']
    has_ema   = pd.notna(e9)  and pd.notna(e21)
    has_rsi   = pd.notna(r10) and pd.notna(rsma)

    def make_row(ref, f2, f3):
        return ScannerRow(
            symbol=sym, ltp=float(last['close']), chg_pct=float(chg_pct),
            ref_level=ref,
            ema9      = float(e9)   if pd.notna(e9)   else None,
            ema21     = float(e21)  if pd.notna(e21)  else None,
            rsi10     = float(r10)  if pd.notna(r10)  else None,
            rsi_sma14 = float(rsma) if pd.notna(rsma) else None,
            f1=True, f2=f2, f3=f3, added_at=ts_str
        )

    bull = make_row(float(high15[sym]),
                    bool(e9 > e21)   if has_ema else False,
                    bool(r10 > rsma) if has_rsi else False)            if float(last['close']) > high15.get(sym, np.inf) else None

    bear = make_row(float(low15[sym]),
                    bool(e9 < e21)   if has_ema else False,
                    bool(r10 < rsma) if has_rsi else False)            if float(last['close']) < low15.get(sym, -np.inf) else None

    return bull, bear


# ─── DataFrame builder ────────────────────────────────────────────────────────
def rows_to_df(rows: List[ScannerRow], bullish: bool) -> pd.DataFrame:
    ref_col = '15m High' if bullish else '15m Low'
    cols = ['Symbol','LTP','Chg %', ref_col,
            'EMA9','EMA21','RSI10','RSI SMA14','F1','F2','F3','Time']
    if not rows:
        return pd.DataFrame(columns=cols)
    data = [{
        'Symbol':  r.symbol,
        'LTP':     round(r.ltp, 2),
        'Chg %':   round(r.chg_pct, 2),
        ref_col:   round(r.ref_level, 2),
        'EMA9':    round(r.ema9, 2)      if r.ema9      else None,
        'EMA21':   round(r.ema21, 2)     if r.ema21     else None,
        'RSI10':   round(r.rsi10, 2)     if r.rsi10     else None,
        'RSI SMA14': round(r.rsi_sma14, 2) if r.rsi_sma14 else None,
        'F1': '✅' if r.f1 else '—',
        'F2': '✅' if r.f2 else '—',
        'F3': '✅' if r.f3 else '—',
        'Time': r.added_at,
    } for r in rows]
    df    = pd.DataFrame(data)
    score = df[['F1','F2','F3']].apply(lambda x: sum(v == '✅' for v in x), axis=1)
    return (df.assign(_s=score)
              .sort_values(['_s','Chg %'], ascending=[False, not bullish])
              .drop(columns=['_s']))


# ─── Session state init ───────────────────────────────────────────────────────
for key, default in [
    ('logs', []),
    ('high15', {}), ('low15', {}),
    ('instrument_map', {}),
    ('bull_rows', []), ('bear_rows', []),
    ('last_scan', None),
    ('scan_count', 0),
    ('auto_running', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 0.5rem; max-width: 97%;}
[data-testid='stMetric'] {
    background: #0f172a; border: 1px solid #1e293b;
    padding: 12px 16px; border-radius: 12px;
}
[data-testid='stMetricLabel'] {font-size: 12px; color: #94a3b8;}
[data-testid='stMetricValue'] {font-size: 20px; font-weight: 700;}
.stDataFrame thead tr th {
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stDataFrame tbody tr td {
    font-size: 13px !important;
    font-variant-numeric: tabular-nums;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 📈 FnO Momentum Scanner")
st.caption("Upstox 1-minute candle scanner · EMA9/21 · RSI10/SMA14 · First 15-min breakout")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    secret_token = get_token()
    if secret_token and secret_token != "YOUR_UPSTOX_ACCESS_TOKEN_HERE":
        st.success("✅ Token loaded from secrets.toml")
        access_token = secret_token
        st.text_input("Override token (optional)", type="password",
                      key="token_override",
                      help="Leave blank to use secrets.toml token")
        if st.session_state.get("token_override"):
            access_token = st.session_state.token_override
    else:
        st.warning("No token found in secrets.toml")
        access_token = st.text_input("Upstox Access Token", type="password",
                                     key="token_manual")

    st.markdown("---")
    symbol_count = st.slider("Universe size", 10, len(FNO_STOCKS),
                              min(40, len(FNO_STOCKS)), 5)
    symbols = FNO_STOCKS[:symbol_count]

    st.markdown("---")
    st.markdown("**🔄 Auto Refresh**")
    auto_on = st.toggle("Enable auto refresh",
                         value=st.session_state.auto_running)
    st.session_state.auto_running = auto_on
    refresh_interval = st.selectbox(
        "Refresh interval",
        ["Every candle close (60s)", "Every 2 minutes", "Every 5 minutes"],
        index=0
    )
    interval_secs = {
        "Every candle close (60s)": 60,
        "Every 2 minutes": 120,
        "Every 5 minutes": 300
    }[refresh_interval]

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        run_scan = st.button("▶ Scan Now",    type="primary", use_container_width=True)
    with col_b:
        rebuild  = st.button("⟳ Rebuild Ref",                 use_container_width=True)
    clear_btn    = st.button("🗑 Clear Results",               use_container_width=True)

    st.markdown("---")
    st.markdown("**📋 Filter Logic**")
    st.markdown("""
| Filter | Bullish | Bearish |
|--------|---------|---------|
| F1 | Close > 15m High | Close < 15m Low |
| F2 | EMA9 > EMA21 | EMA9 < EMA21 |
| F3 | RSI10 > RSI_SMA14 | RSI10 < RSI_SMA14 |
""")


# ─── Status bar ───────────────────────────────────────────────────────────────
now_ist    = datetime.now(IST)
phase      = market_phase(now_ist)
trade_date = now_ist.strftime('%Y-%m-%d')
secs_left  = seconds_to_next_candle(now_ist)

h1,h2,h3,h4,h5,h6 = st.columns(6)
with h1: st.metric("Phase",       phase)
with h2: st.metric("IST Time",    now_ist.strftime('%H:%M:%S'))
with h3: st.metric("Next Candle", f"{secs_left}s")
with h4: st.metric("Next Scan",   next_scan_at(now_ist) if st.session_state.auto_running else "—")
with h5: st.metric("Scans Run",   st.session_state.scan_count)
with h6: st.metric("Last Scan",   st.session_state.last_scan or "—")

if not access_token:
    st.info("🔑 Add your token to `.streamlit/secrets.toml` or enter it in the sidebar.")
    with st.expander("How to set up secrets.toml"):
        st.code('''# .streamlit/secrets.toml
[upstox]
access_token = "your_token_here"
''', language="toml")
    st.stop()


# ─── Instrument loading ───────────────────────────────────────────────────────
client = UpstoxClient(access_token)

if not st.session_state.instrument_map:
    with st.spinner("Loading instrument master from Upstox…"):
        try:
            st.session_state.instrument_map = client.load_instruments()
            st.session_state.logs.insert(
                0, f"Instruments loaded: {len(st.session_state.instrument_map)}"
            )
        except Exception as e:
            st.error(f"Instrument load failed: {e}")
            st.stop()

client.instrument_map = st.session_state.instrument_map
symbols = [s for s in symbols if s in client.instrument_map]

if clear_btn:
    st.session_state.bull_rows = []
    st.session_state.bear_rows = []
    st.session_state.logs = ["Results cleared."]


# ─── Build reference levels ───────────────────────────────────────────────────
if rebuild or (run_scan and not st.session_state.high15):
    with st.status("Building 15-minute reference levels…", expanded=True) as status_box:
        try:
            h15, l15, ref_logs = build_reference_levels(client, symbols, trade_date)
            if h15:
                st.session_state.high15 = h15
                st.session_state.low15  = l15
                st.session_state.logs   = (
                    [f"✅ Reference levels ready for {len(h15)} symbols"]
                    + ref_logs[:15]
                )
                status_box.update(
                    label=f"✅ Reference levels built for {len(h15)} symbols",
                    state="complete"
                )
            else:
                status_box.update(
                    label="⚠️ No reference levels built — see logs below",
                    state="error"
                )
                for line in ref_logs[:10]:
                    st.caption(line)
                st.stop()
        except Exception as e:
            status_box.update(label=f"Reference build failed: {e}", state="error")
            st.stop()

if not st.session_state.high15:
    st.warning("⚠️ Reference levels not built yet. Click **▶ Scan Now** to build them.")
    st.stop()


# ─── Run scan ─────────────────────────────────────────────────────────────────
def do_scan():
    bull_rows, bear_rows, scan_logs = [], [], []
    bar = st.progress(0, text="Scanning symbols…")
    for i, sym in enumerate(symbols):
        try:
            bull, bear = scan_symbol(
                client, sym, trade_date,
                st.session_state.high15,
                st.session_state.low15
            )
            if bull: bull_rows.append(bull)
            if bear: bear_rows.append(bear)
        except Exception as e:
            scan_logs.append(f"{sym}: {e}")
        bar.progress((i + 1) / len(symbols),
                     text=f"Scanning {sym}  ({i+1}/{len(symbols)})")
        time.sleep(0.02)
    bar.empty()
    now = datetime.now(IST)
    st.session_state.bull_rows   = bull_rows
    st.session_state.bear_rows   = bear_rows
    st.session_state.last_scan   = now.strftime('%H:%M:%S')
    st.session_state.scan_count += 1
    st.session_state.logs = [
        f"[{now.strftime('%H:%M:%S')}] Scan #{st.session_state.scan_count} complete",
        f"  Bullish setups : {len(bull_rows)}",
        f"  Bearish setups : {len(bear_rows)}",
    ] + scan_logs[:20]

if run_scan:
    do_scan()


# ─── Tables ───────────────────────────────────────────────────────────────────
bull_df = rows_to_df(st.session_state.bull_rows, bullish=True)
bear_df = rows_to_df(st.session_state.bear_rows, bullish=False)

def fcount(df, col):
    return int((df[col] == '✅').sum()) if not df.empty else 0

s1,s2,s3,s4,s5,s6 = st.columns(6)
with s1: st.metric("🟢 Bull F1",  fcount(bull_df, 'F1'))
with s2: st.metric("🟡 Bull F2",  fcount(bull_df, 'F2'))
with s3: st.metric("🟠 Bull F3",  fcount(bull_df, 'F3'))
with s4: st.metric("🔴 Bear F1",  fcount(bear_df, 'F1'))
with s5: st.metric("🟡 Bear F2",  fcount(bear_df, 'F2'))
with s6: st.metric("🟠 Bear F3",  fcount(bear_df, 'F3'))

def style_df(df: pd.DataFrame, bullish: bool) -> pd.DataFrame:
    """Return a plain formatted DataFrame — no pandas Styler to avoid applymap issues."""
    if df.empty:
        return df
    out = df.copy()
    ref_col = '15m High' if bullish else '15m Low'
    # Format numeric columns as strings with symbols
    for col, fmt in [('LTP','₹{:.2f}'), (ref_col,'₹{:.2f}'),
                     ('EMA9','{:.2f}'), ('EMA21','{:.2f}'),
                     ('RSI10','{:.1f}'), ('RSI SMA14','{:.1f}')]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: fmt.format(v) if pd.notna(v) and isinstance(v, (int,float)) else '—'
            )
    if 'Chg %' in out.columns:
        out['Chg %'] = out['Chg %'].apply(
            lambda v: ('+' if v >= 0 else '') + f'{v:.2f}%'
            if pd.notna(v) and isinstance(v, (int,float)) else '—'
        )
    return out

st.markdown("### 🟢 Bullish Momentum")
if bull_df.empty:
    st.info("No bullish setups yet — run a scan.")
else:
    st.dataframe(style_df(bull_df, True), use_container_width=True, hide_index=True)

st.markdown("### 🔴 Bearish Momentum")
if bear_df.empty:
    st.info("No bearish setups yet — run a scan.")
else:
    st.dataframe(style_df(bear_df, False), use_container_width=True, hide_index=True)

with st.expander("📋 Scan Log", expanded=False):
    for line in st.session_state.logs[:30]:
        st.caption(line)


# ─── Auto refresh ─────────────────────────────────────────────────────────────
if st.session_state.auto_running:
    if phase not in ('Scanning', 'Monitoring'):
        st.warning(
            f"Auto refresh is on but market is **{phase}**. "
            "Scanner will wait for market hours (9:15–15:30 IST)."
        )

    placeholder = st.empty()
    secs = interval_secs if st.session_state.last_scan else 2
    for remaining in range(secs, 0, -1):
        placeholder.markdown(
            f"<div style='color:#64748b;font-size:13px;padding:6px 0'>"
            f"🔄 Auto refresh in <b>{remaining}s</b> &nbsp;·&nbsp; "
            f"{refresh_interval} &nbsp;·&nbsp; "
            f"<span style='color:#ef4444'>toggle off in sidebar to stop</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        time.sleep(1)
    placeholder.empty()
    do_scan()
    st.rerun()
